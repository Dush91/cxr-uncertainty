#!/usr/bin/env python3
"""Agent 2 -- build the FULL reference index: train + cal + eval in one library.

Extends scripts/build_retrieval_index.py (cal-only, features straight from the
adapted arrays npz) to the whole ReXGradient fs manifest:

  * cal + eval embeddings: ``rad_feats`` from the adapted arrays npz -- EXACT
    (same vectors as the production 8,064-index; eval rows are new to the library).
  * train embeddings: the fp16 224px xrv-format memmap cache
    (``--train-cache``, opened with ``np.memmap`` -- it has no .npy header;
    PNG -> load_image_tensor(path, 224) -> (1,1,224,224) in [-1024,1024], cf.
    scripts/build_train_cache.py) through the adapted 3-member 224 ensemble
    (xrv_nih / convnextv2 / raddino with ckpt overlay). arkswin cannot run on
    train (768 PNGs not retained locally), so train annotations are 3-member.
  * Parity gate (hard failsafe): 256 rows of the fp16 CAL cache through the
    same forward path must reproduce the npz ``rad_feats`` at median cosine
    > 0.99 -- the cache went PNG->224 direct while the npz features went
    PNG->768->resize-224, and the gate proves the resampling gap is noise
    before the train pass is trusted at 113k scale. Runs whenever
    ``--cal-cache`` is present (upload it to Lightning too; it is only 1.6 GB).
  * Sidecar for ALL rows: image_id / split / pred_pathology / pred_conf /
    correct / uncertain / pred_normal (no class reaches its adapted-cal Youden
    threshold) / gt_normal (no positive among certain GT labels).

Label-free invariant preserved (as in build_retrieval_index.py): index.bin +
store.npz hold ONLY ids + vectors; all prediction/label metadata lives in
sidecar.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import hnswlib
except ImportError as e:  # pragma: no cover
    sys.exit("hnswlib is required (pip install hnswlib): %s" % e)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so this runs from scripts/ without installing the package

from build_retrieval_index import exact_topk  # noqa: E402


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def youden_from_cal(cal: dict) -> tuple[dict, list[str]]:
    """Adapted-cal Youden thresholds (same math as the app's _fit_calibration)."""
    from cxr_uncertainty.reanalyze import _youden_threshold

    probs = cal["probs"]                       # (M,N,P)
    valid = cal["valid"].astype(bool)          # (N,P)
    gt = np.asarray(cal["gt"], dtype=np.float64)
    mean_p = np.nanmean(probs, axis=1)         # (N,P) -- pooled member mean
    pats = [str(p) for p in cal["pathologies"]]
    youden: dict[str, float] = {}
    for j, pat in enumerate(pats):
        v = valid[:, j] & (gt[:, j] >= 0) & np.isfinite(mean_p[:, j])
        y = gt[v, j].astype(int)
        youden[pat] = 0.5 if (y == 1).sum() < 2 or (y == 0).sum() < 2 \
            else float(_youden_threshold(y, mean_p[v, j]))
    return youden, pats


def pred_normal_flag(mean_p: np.ndarray, valid: np.ndarray,
                     thr_row: np.ndarray) -> np.ndarray:
    """No valid class reaches its Youden threshold (model's 'no finding' verdict)."""
    fin = np.isfinite(mean_p) & valid
    reaches = fin & (mean_p >= thr_row[None, :])
    return fin.any(axis=1) & ~reaches.any(axis=1)


def gt_normal_flag(gt: np.ndarray) -> np.ndarray:
    """True 'no pathology': at least one certain label and none positive."""
    return (((gt >= 0).sum(axis=1) > 0) & ((gt == 1).sum(axis=1) == 0)).astype(bool)


def annotate_split(probs: np.ndarray, valid: np.ndarray, gt: np.ndarray,
                   ids: np.ndarray, pats: list[str], youden: dict,
                   split: str, n_members_note: str = "") -> "pd.DataFrame":
    """Sidecar rows for cal/eval, same pooled-prediction math as
    build_retrieval_index.build_sidecar (member-mean argmax + conf)."""
    import pandas as pd

    mean_p = np.nanmean(probs, axis=1)         # (N,P)
    n = len(ids)
    fin = np.isfinite(mean_p)
    masked = np.where(valid & fin, mean_p, -np.inf)
    pred_idx = masked.argmax(axis=1)
    conf = masked.max(axis=1)
    gt_lab = np.array([gt[k, pred_idx[k]] for k in range(n)], dtype=np.float64)

    thr = np.array([youden.get(p, 0.5) for p in pats], dtype=np.float64)
    rows = pd.DataFrame({
        "image_id": [str(i) for i in ids],
        "split": split,
        "pred_pathology": [pats[j] for j in pred_idx],
        "pred_conf": conf.astype(float),
        "correct": (gt_lab == 1.0),
        "uncertain": (gt_lab == -1.0),
        "pred_normal": pred_normal_flag(mean_p, valid, thr),
        "gt_normal": gt_normal_flag(gt),
        "annot_members": n_members_note,
    })
    return rows


def load_ensemble(device: str, ckpts: dict[str, str]):
    """Adapted 3-member 224 ensemble (ckpt overlay) for the train pass."""
    import torch
    from cxr_uncertainty.config import RiskConfig
    from cxr_uncertainty.models import CXREnsemble

    members = ["xrv_nih", "convnextv2", "raddino"]   # all native 224
    cfg = RiskConfig(device=device)
    cfg.img_size = 224
    ens = CXREnsemble(members=members, cfg=cfg)
    for mb in ens.members:
        path = ckpts[mb.key]
        mb.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[overlay] {mb.key} <- {path}")
    return ens, cfg


def embed_cache(cache_path: str, paths_txt: str, n_rows: int, ens, cfg,
                device: str, raddino_idx: int, pats: list[str], youden: dict,
                labels_by_id: dict, batch_size: int, dim: int) -> tuple:
    """Batched forward over a fp16 (N,1,224,224) xrv-format cache.

    Returns (L2-normed feats, ids, sidecar rows, seconds). ``n_rows`` may be
    capped by --limit (first n rows of the file, which is manifest order).
    """
    import pandas as pd
    import torch

    arr = np.memmap(cache_path, dtype=np.float16, mode="r",
                    shape=(n_rows, 1, 224, 224))
    all_paths = [ln.strip() for ln in open(paths_txt) if ln.strip()]
    assert len(all_paths) >= n_rows, f"paths sidecar has {len(all_paths)} < {n_rows} rows"
    feats = np.zeros((n_rows, dim), dtype=np.float32)
    probs3 = np.zeros((n_rows, len(pats)), dtype=np.float64)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n_rows, batch_size):
            i1 = min(i0 + batch_size, n_rows)
            x = torch.from_numpy(np.asarray(arr[i0:i1], dtype=np.float32)).to(device)
            eo = ens.forward_ensemble_full(x)
            feats[i0:i1] = eo.per_member_features[raddino_idx].detach().cpu().numpy()
            probs3[i0:i1] = eo.per_member_probs.mean(dim=0).detach().cpu().numpy()
            if (i0 // batch_size) % 25 == 0:
                dt = max(time.time() - t0, 1e-9)
                print(f"[cache] {i1}/{n_rows} rows  ({i1/dt:.0f} img/s, "
                      f"eta {dt*(n_rows-i1)/max(i1,1)/60:.0f} min)", flush=True)
    dt = time.time() - t0
    print(f"[cache] done {n_rows} rows in {dt/60:.1f} min ({n_rows/dt:.0f} img/s)")

    feats = norm(feats)
    # ids: cache row k == manifest row k (build_train_cache wrote the memmap in
    # df order; the .paths.txt sidecar is that same order).
    ids = np.array([Path(p).stem for p in all_paths[:n_rows]], dtype=object)

    pred_idx = probs3.argmax(axis=1)
    gt = np.zeros((n_rows, len(pats)), dtype=np.float64)
    for k, i in enumerate(ids):
        gt[k] = labels_by_id.get(str(i), np.full(len(pats), np.nan))
    gt_lab = np.array([gt[k, pred_idx[k]] for k in range(n_rows)], dtype=np.float64)
    thr = np.array([youden.get(p, 0.5) for p in pats], dtype=np.float64)
    rows = pd.DataFrame({
        "image_id": [str(i) for i in ids],
        "split": "train",
        "pred_pathology": [pats[j] for j in pred_idx],
        "pred_conf": probs3.max(axis=1).astype(float),
        "correct": (gt_lab == 1.0),
        "uncertain": (gt_lab == -1.0),
        "pred_normal": pred_normal_flag(probs3, np.ones_like(probs3, dtype=bool), thr),
        "gt_normal": gt_normal_flag(gt),
        "annot_members": "3-member@224 (xrv_nih, convnextv2, raddino); arkswin omitted (no 768 PNGs for train)",
    })
    print(f"[cache] forward+features: {dt/60:.1f} min for {n_rows} rows")
    return feats, ids, rows, dt


def parity_gate(args, ens, cfg, device, raddino_idx: int, arr_cal: dict,
                dim: int, batch_size: int) -> dict:
    """256 cal-cache rows vs npz rad_feats, matched by image_id, cosine."""
    import torch

    n_ids = sum(1 for _ in open(args.parity_cache_paths))
    n = min(args.parity_n, n_ids)
    arr = np.memmap(args.parity_cache, dtype=np.float16, mode="r",
                    shape=(n, 1, 224, 224))
    paths = [ln.strip() for ln in open(args.parity_cache_paths) if ln.strip()][:n]
    cache_ids = [Path(p).stem for p in paths]

    feats = np.zeros((n, dim), dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, batch_size):
            i1 = min(i0 + batch_size, n)
            x = torch.from_numpy(np.asarray(arr[i0:i1], dtype=np.float32)).to(device)
            eo = ens.forward_ensemble_full(x)
            feats[i0:i1] = eo.per_member_features[raddino_idx].detach().cpu().numpy()
    feats = norm(feats)

    npz_ids = np.array([str(i) for i in arr_cal["ids"]])
    lookup = {str(i): k for k, i in enumerate(npz_ids)}
    keep = [k for k, i in enumerate(cache_ids) if i in lookup]
    assert len(keep) >= 32, f"parity: only {len(keep)}/{n} cache ids matched npz cal ids"
    ref = np.stack([arr_cal["rad_feats"][lookup[cache_ids[k]]] for k in keep])
    ref = norm(np.asarray(ref, dtype=np.float32))
    cos = np.einsum("ij,ij->i", feats[keep], ref)
    med = float(np.median(cos))
    print(f"[parity] {len(keep)} cal rows: median cos={med:.6f} "
          f"p5={np.percentile(cos,5):.6f} min={cos.min():.6f}")
    assert med > 0.99, f"PARITY GATE FAILED: median cosine {med:.4f} <= 0.99"
    return {"parity_median_cos": med, "parity_min_cos": float(cos.min()),
            "parity_rows": len(keep)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arr-cal", default="runs/rex_phase7/eval_arrays/rex_adapted_s0/arrays_rex_adaptedcal.npz")
    ap.add_argument("--arr-eval", default="runs/rex_phase7/eval_arrays/rex_adapted_s0/arrays_rex_adaptedeval.npz")
    ap.add_argument("--manifest", default="data/rex_manifest_fs.parquet")
    ap.add_argument("--train-cache", default="data/adapted_cache_train.npy")
    ap.add_argument("--cache-path-sidecar", default=None,
                    help="paths.txt for --train-cache (default: <cache>.paths.txt)")
    ap.add_argument("--parity-cache", default="data/adapted_cache_cal.npy",
                    help="fp16 cal cache for the parity gate; '' disables")
    ap.add_argument("--parity-cache-paths", default="data/adapted_cache_cal.paths.txt")
    ap.add_argument("--parity-n", type=int, default=256)
    ap.add_argument("--ckpt-xrv", default="checkpoints/rex_adapted/xrv_nih.pt")
    ap.add_argument("--ckpt-convnextv2", default="checkpoints/rex_adapted/convnextv2_s0.pt")
    ap.add_argument("--ckpt-raddino", default="checkpoints/rex_adapted/raddino.pt")
    ap.add_argument("--out", default="runs/rex_phase7/retrieval/rex_adapted_raddino_full")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--ef-construction", type=int, default=400)
    ap.add_argument("--ef-search", type=int, default=512)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE: embed only the first N train-cache rows")
    ap.add_argument("--skip-train", action="store_true", help="cal+eval only (debug)")
    args = ap.parse_args()
    if args.cache_path_sidecar is None:
        args.cache_path_sidecar = args.train_cache[:-4] + ".paths.txt"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    cal = dict(np.load(args.arr_cal, allow_pickle=True))
    arr_eval = dict(np.load(args.arr_eval, allow_pickle=True))
    feats_cal = norm(np.asarray(cal["rad_feats"], dtype=np.float32))
    feats_eval = norm(np.asarray(arr_eval["rad_feats"], dtype=np.float32))
    pats = [str(p) for p in cal["pathologies"]]
    youden, _ = youden_from_cal(cal)
    print(f"[youden] {len(youden)} adapted-cal Youden thresholds: "
          + ", ".join(f"{k}={v:.3f}" for k, v in list(youden.items())[:5]) + ", ...")

    sc_cal = annotate_split(cal["probs"], cal["valid"].astype(bool), cal["gt"],
                            cal["ids"], pats, youden, "cal", "4-member npz (exact)")
    sc_eval = annotate_split(arr_eval["probs"], arr_eval["valid"].astype(bool), arr_eval["gt"],
                             arr_eval["ids"], pats, youden, "eval", "4-member npz (exact)")
    print(f"[sidecar] cal={len(sc_cal)} eval={len(sc_eval)} "
          f"(gt_normal: cal={int(sc_cal.gt_normal.sum())} eval={int(sc_eval.gt_normal.sum())}; "
          f"pred_normal: cal={int(sc_cal.pred_normal.sum())} eval={int(sc_eval.pred_normal.sum())})")

    feats_list = [feats_cal, feats_eval]
    ids_list = [np.array([str(i) for i in cal["ids"]], dtype=object),
                np.array([str(i) for i in arr_eval["ids"]], dtype=object)]
    sc_list = [sc_cal, sc_eval]
    diag: dict = {}

    do_train = (not args.skip_train)
    if do_train:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        manifest = pd.read_parquet(args.manifest)
        labels_by_id = {str(i): np.asarray(l, dtype=np.float64)
                        for i, l in zip(manifest["image_id"].astype(str), manifest["labels"])}
        ens, cfg = load_ensemble(device, {"xrv_nih": args.ckpt_xrv,
                                          "convnextv2": args.ckpt_convnextv2,
                                          "raddino": args.ckpt_raddino})
        raddino_idx = next(i for i, m in enumerate(ens.members) if m.key == "raddino")

        # ---- parity gate FIRST: cache path must reproduce npz rad_feats ----
        if not args.parity_cache:
            print("[parity] SKIPPED (--parity-cache '')")
            diag["parity_median_cos"] = None
        else:
            diag.update(parity_gate(args, ens, cfg, device, raddino_idx, cal,
                                    args.dim, args.batch_size))

        n_train = sum(1 for _ in open(args.cache_path_sidecar))
        size_bytes = os.path.getsize(args.train_cache)
        assert size_bytes == n_train * 1 * 224 * 224 * 2, \
            f"cache size {size_bytes} != {n_train} rows fp16 (224^2) -- wrong file/dtype?"
        if args.limit:
            print(f"[train] SMOKE --limit={args.limit}: using first {args.limit}/{n_train} cache rows")
            n_train = args.limit
        feats_tr, ids_tr, sc_tr, tr_dt = embed_cache(
            args.train_cache, args.cache_path_sidecar, n_train, ens, cfg, device,
            raddino_idx, pats, youden, labels_by_id, args.batch_size, args.dim)
        feats_list.append(feats_tr)
        ids_list.append(ids_tr)
        sc_list.append(sc_tr)
        diag["train_pass_seconds"] = tr_dt
        diag["train_n"] = int(n_train)
        del ens  # free GPU/host memory before the HNSW build
    else:
        device = "skip"

    feats = np.concatenate(feats_list, axis=0)
    ids = np.concatenate(ids_list)
    sidecar = pd.concat(sc_list, ignore_index=True)
    n, dim = feats.shape

    # ---- HNSW over the FULL library ----------------------------------------
    print(f"[index] building HNSW over {n} vectors dim={dim} M={args.M} efC={args.ef_construction}")
    t0 = time.time()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction,
                     M=args.M, random_seed=args.seed)
    index.add_items(feats, np.arange(n))
    index.set_ef(args.ef_search)
    index.save_index(str(out / "index.bin"))
    print(f"[index] built + saved in {(time.time()-t0)/60:.1f} min")

    np.savez_compressed(str(out / "store.npz"), feats=feats, ids=ids)
    sidecar.to_parquet(str(out / "sidecar.parquet"), index=False)

    # ---- recall@k: eval queries against the full library, self excluded ----
    base_eval = len(feats_cal)                      # eval block starts after cal
    n_eval = len(feats_eval)
    rng = np.random.default_rng(args.seed)
    q_sel = rng.choice(n_eval, size=min(200, n_eval), replace=False)
    recalls = []
    for qi in q_sel:
        q = feats_eval[qi]
        mask = np.ones(n, dtype=bool)
        mask[base_eval + qi] = False                # a query must not retrieve itself
        ex, _ = exact_topk(feats, q, args.k, mask)
        hn, _ = index.knn_query(q, k=args.k,
                                filter=lambda li: bool(mask[int(li)]))
        recalls.append(len(set(ex) & {int(i) for i in np.asarray(hn).ravel()}) / args.k)
    recall = float(np.mean(recalls))

    split_counts = sidecar["split"].value_counts().to_dict()
    correct_pool = int((sidecar["correct"] & (sidecar["split"] != "train")
                        & ~sidecar["uncertain"]).sum())
    starved = sidecar[sidecar.split != "train"].groupby("pred_pathology").size()
    meta = {
        "source": "rex_adapted_s0", "embed": "raddino(adapted), L2-normalized",
        "dim": int(dim), "n_reference": int(n), "space": "cosine",
        "M": args.M, "ef_construction": args.ef_construction,
        "ef_search": args.ef_search, "k": args.k, "recall_at_k": recall,
        "recall_queries": int(len(q_sel)), "seed": args.seed,
        "splits": split_counts,
        "correct_mode_pool_cal_eval": correct_pool,
        "train_annotation_members": "xrv_nih + convnextv2_s0 + raddino @224 (arkswin omitted: no 768 PNGs for train)",
        "normal_flags": "pred_normal = no class >= its adapted-cal Youden threshold; gt_normal = no positive among certain GT labels",
        "label_free": "index.bin + store.npz hold ids+feats only; sidecar.parquet separate",
        **diag,
    }
    (out / "index_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[splits] {split_counts}")
    print(f"[correct-mode pool] cal+eval correct&certain: {correct_pool} (train excluded by design)")
    print(f"[normal] gt_normal={int(sidecar.gt_normal.sum())} pred_normal={int(sidecar.pred_normal.sum())} of {n}")
    print(f"[candidates] cal+eval pred_pathology counts: {starved.sort_values(ascending=False).to_dict()}")
    print(f"[recall] HNSW recall@{args.k} = {recall:.4f} over {len(q_sel)} eval queries"
          + ("" if recall >= 0.99 else "  <-- LOW: raise ef/M"))
    print(f"[done] {out}")
    print("FULL_RETRIEVAL_BUILD_DONE")


if __name__ == "__main__":
    main()