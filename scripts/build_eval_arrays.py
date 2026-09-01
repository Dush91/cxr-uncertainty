#!/usr/bin/env python
"""Build phase-4-style feature arrays for a powered leak-free eval dataset
(ReXGradient-160K, CANDID-III, ...) from an eval-only manifest.

This is the array producer for the multi-dataset baseline comparison
(``scripts/eval_baselines.py --datasets``). It mirrors the inference path of
``scripts/eval_phase4_features.py`` (load_image_tensor + forward_ensemble_full,
batch-1) and writes the SAME npz schema that ``eval_baselines._load_arrays``
expects:

    probs (N,M,P), gt (N,P), valid (N,P), rad_logits (N,14),
    rad_feats (N,1536), energy (N,), mahalanobis (N,), ids (N,),
    member_keys (M,), pathologies (P,)

so the saved arrays can be fed straight into ``eval_baselines`` with no
re-inference. Two important differences from the OpenI/phase-4 producer:

  1. **Mahalanobis is re-fit per dataset** on this dataset's own ``cal`` split
     (not transferred from OpenI). ``MahalanobisOOD`` is an in-distribution
     density; an OpenI-fit score on a new corpus would measure cross-dataset
     shift from the OpenI centroid cloud, not within-dataset epistemic
     uncertainty. The fit class defaults to Pneumonia (for comparability with
     the OpenI production flag) and auto-falls-back to the most-balanced
     powered class if Pneumonia is under-powered (<2 pos or <2 neg in cal).

  2. **gt is float64 and preserves -1** (expert-"uncertain" labels, e.g.
     CANDID-III). The binary decision path in ``reanalyze.build_records`` is
     structurally {0,1}; ``eval_baselines`` filters -1 rows out of the binary
     metrics and consumes them separately for Baur Task 2 (uncertainty-label
     prediction). ``valid`` is a *label-availability* mask (1 = this dataset
     can label this class), independent of label certainty -- uncertain rows
     keep ``valid=1`` so Task 2 can find them.

The manifest (built per-dataset by the label-extraction helpers) needs columns:
``image_id, image_path, source, split_role in {cal,eval}, labels (P,), valid
(P,), view, patient_id``. It is an eval-only manifest (no A/B/C/D leak-free
roles) -- the leak-free invariants in ``data/manifest.parquet`` are untouched.

Usage:
    PYTHONPATH=. python scripts/build_eval_arrays.py \
        --manifest data/rex_manifest.parquet --source rex \
        --out runs/eval_arrays

    PYTHONPATH=. python scripts/build_eval_arrays.py \
        --manifest data/candid_manifest.parquet --source candidiii \
        --members xrv_nih,convnextv2,raddino,arkswin \
        --maha-class Pneumonia --out runs/eval_arrays
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure the package is importable when run as a plain script.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig  # noqa: E402
from cxr_uncertainty.feature_uq import MahalanobisOOD, energy_score, extract_member  # noqa: E402
from cxr_uncertainty.models import CXREnsemble  # noqa: E402
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402

DEFAULT_MEMBERS = "xrv_nih,convnextv2,raddino,arkswin"   # production 4-member ensemble
DEFAULT_OUT = _REPO / "runs" / "eval_arrays"


@torch.no_grad()
def forward_split(df: pd.DataFrame, ensemble, cfg, device, label: str,
                  batch_size: int = 8) -> dict:
    """Forward every image in ``df``; collect per-member probs (N,M,P), the
    RAD-DINO member's 14 logits (N,14) + 1536-d features (N,D), gt (N,P) as
    float64 (preserves -1), valid (N,P) bool, and ids. Skips images that fail
    to load (keeps arrays aligned to ids so downstream image_id joins are
    exact). Mirrors ``eval_phase4_features.forward_site``.

    Mini-batches ``batch_size`` images per ``forward_ensemble_full`` call to
    amortize the per-image launch + Python overhead -- the dominant cost at
    batch-1, especially for the 768-resolution Ark+ Swin-Large member. On a
    GPU this is ~5-15x faster than batch-1 with the same output schema. If a
    batch forward OOMs (e.g. Swin-L @ 768 batch is too large for VRAM), the
    chunk is retried image-by-image so the run degrades gracefully rather than
    crashing -- lower ``--batch-size`` to avoid the fallback."""
    member_keys = list(ensemble.member_keys)
    raddino_idx = member_keys.index("raddino") if "raddino" in member_keys else None
    probs, rad_logits, rad_feats, gts, valids, ids = [], [], [], [], [], []
    t0 = time.time()

    def _emit(eo, rows_batch):
        """Slice a batched EnsembleOutput (per_member_* are (M,N,*)) into the
        per-image lists. Equivalent to extract_member over the batch dim."""
        pmp = eo.per_member_probs.cpu().numpy().astype(np.float64)   # (M,N,P)
        n = pmp.shape[1]
        if raddino_idx is not None:
            rl = eo.per_member_logits[raddino_idx].cpu().numpy().astype(np.float64)   # (N,P)
            rf = eo.per_member_features[raddino_idx].cpu().numpy().astype(np.float64)  # (N,D)
        for k in range(n):
            probs.append(pmp[:, k, :])                 # (M,P)
            if raddino_idx is not None:
                rad_logits.append(rl[k])              # (P,)
                rad_feats.append(rf[k])               # (D,)
            r = rows_batch[k]
            gts.append(np.asarray(r.labels, dtype=np.float64))   # (P,) -1 preserved
            valids.append(np.asarray(r.valid, dtype=bool))       # (P,)
            ids.append(str(r.image_id))

    def _forward_chunk(xbuf, rows_batch):
        """Forward a stacked batch; on CUDA OOM, retry image-by-image."""
        xb = torch.cat(xbuf, dim=0)   # (N,1,H,W)
        try:
            eo = ensemble.forward_ensemble_full(xb)
            _emit(eo, rows_batch)
        except RuntimeError as e:
            if "cuda" in str(e).lower() and "memory" in str(e).lower() and torch.cuda.is_available():
                print(f"  [oom] batch of {len(xbuf)} OOM'd on {torch.cuda.get_device_name(0)}; "
                      f"falling back to batch-1 for this chunk (lower --batch-size to avoid)")
                torch.cuda.empty_cache()
                for x1, r1 in zip(xbuf, rows_batch):
                    eo1 = ensemble.forward_ensemble_full(x1)
                    _emit(eo1, [r1])
            else:
                raise

    xbuf, rows_batch = [], []
    rows = list(df.itertuples(index=False))
    for i, row in enumerate(rows):
        try:
            x = load_image_tensor(row.image_path, img_size=cfg.img_size, device=device)
        except Exception as e:  # missing/unreadable image -> skip, keep ids aligned
            print(f"  [skip] {row.image_id}: {e}")
            continue
        xbuf.append(x)
        rows_batch.append(row)
        if len(xbuf) >= batch_size:
            _forward_chunk(xbuf, rows_batch)
            xbuf.clear(); rows_batch.clear()
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            dt = time.time() - t0
            print(f"  [infer {label}] {i+1}/{len(rows)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if xbuf:                       # flush remainder
        _forward_chunk(xbuf, rows_batch)
    out = {
        "probs": np.stack(probs),        # (N,M,P)
        "gt": np.stack(gts),             # (N,P) float64
        "valid": np.stack(valids),        # (N,P) bool
        "ids": ids,
    }
    if rad_logits:
        out["rad_logits"] = np.stack(rad_logits)   # (N,14)
        out["rad_feats"] = np.stack(rad_feats)      # (N,1536)
    return out


def _pick_maha_class(gt_cal: np.ndarray, valid_cal: np.ndarray,
                     prefer: str = "Pneumonia"):
    """Pick a pathology whose cal pos/neg partition is powered (>=2 each).
    Prefers ``prefer`` for comparability with the OpenI production flag, else
    the most-balanced powered class. Returns (class_idx, class_name) or None."""
    P = gt_cal.shape[1]
    candidates = []
    for j, name in enumerate(NIH_PATHOLOGIES):
        v = valid_cal[:, j].astype(bool)
        y = gt_cal[:, j][v]
        y = y[y >= 0].astype(int)            # exclude -1 (uncertain) from the fit
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if npos >= 2 and nneg >= 2:
            candidates.append((j, name, npos, nneg, min(npos, nneg)))
    if not candidates:
        return None
    for j, name, *_ in candidates:
        if name == prefer:
            return j, name
    candidates.sort(key=lambda c: -c[4])     # most balanced
    return candidates[0][0], candidates[0][1]


def _fit_score_mahalanobis(cal: dict, eval_arr: dict, prefer: str):
    """Fit MahalanobisOOD on the cal split's RAD-DINO features using the chosen
    class's binary (>=0) labels, then score both cal and eval features.
    Returns (cal_scores, eval_scores, {class, n_per_class})."""
    j, name = _pick_maha_class(cal["gt"], cal["valid"], prefer=prefer)
    if j is None:
        raise RuntimeError(
            "MahalanobisOOD.fit: no pathology has >=2 pos AND >=2 neg in the cal "
            "split (after excluding -1). Cannot fit a class-conditional density -- "
            "need a larger cal split or a different dataset.")
    v = cal["valid"][:, j].astype(bool)
    nonneg = cal["gt"][:, j] >= 0
    mask = v & nonneg
    maha = MahalanobisOOD().fit(cal["rad_feats"][mask], cal["gt"][:, j][mask].astype(int))
    print(f"[maha] fit class='{name}' on cal: classes={maha.classes_.tolist()} "
          f"n_per_class={maha.n_per_class_}")
    cal_s = maha.score(cal["rad_feats"])
    eval_s = maha.score(eval_arr["rad_feats"])
    meta = {"class": name, "class_idx": int(j),
            "n_per_class": {int(k): v for k, v in maha.n_per_class_.items()}}
    return cal_s, eval_s, meta


def _save(arr: dict, path: Path, member_keys: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        probs=arr["probs"], gt=arr["gt"], valid=arr["valid"],
        rad_logits=arr["rad_logits"], rad_feats=arr["rad_feats"],
        energy=arr["energy"], mahalanobis=arr["mahalanobis"],
        ids=np.array(arr["ids"], dtype=object),
        member_keys=np.array(member_keys, dtype=object),
        pathologies=np.array(NIH_PATHOLOGIES, dtype=object),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="eval-only manifest parquet (split_role in {cal,eval})")
    ap.add_argument("--source", required=True,
                    help="dataset name (e.g. rex, candidiii); used in output filenames")
    ap.add_argument("--members", default=DEFAULT_MEMBERS,
                    help=f"comma-separated member keys (default: {DEFAULT_MEMBERS})")
    ap.add_argument("--maha-class", default="Pneumonia",
                    help="pathology to fit Mahalanobis on (default: Pneumonia; "
                         "auto-falls-back to the most-balanced powered class)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"base output dir (default: {DEFAULT_OUT}); writes "
                         f"<out>/<source>/arrays_<source>{{cal,eval}}.npz")
    ap.add_argument("--cal-role", default="cal",
                    help="split_role value for the cal split (default: cal; use C "
                         "to smoke-test against data/manifest.parquet)")
    ap.add_argument("--eval-role", default="eval",
                    help="split_role value for the eval split (default: eval; use D "
                         "to smoke-test against data/manifest.parquet)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split (0=all; for smoke tests)")
    ap.add_argument("--ckpt-overlay", default=None,
                    help="Phase 7: 'member_key=path'[,more] state_dicts to load into "
                         "members AFTER registry construction (e.g. adapted ckpts from "
                         "train_rex_adapted; full member state_dict() files). "
                         "Applied before inference; recorded in the arrays meta.")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="mini-batch size per forward_ensemble_full call (default 8; "
                         "amortizes launch + Python overhead -- the dominant cost at "
                         "batch-1, esp. for the 768 Ark+ member. ~5-15x faster on GPU "
                         "than batch-1. Raise to 16/32 on H100 (80GB); 8 is safe on L4 "
                         "(24GB) -- OOM auto-falls-back to batch-1 per chunk).")
    args = ap.parse_args(argv)

    MEMBERS = [m.strip() for m in args.members.split(",") if m.strip()]
    man = pd.read_parquet(args.manifest)
    required = {"image_id", "image_path", "source", "split_role", "labels", "valid"}
    missing = required - set(man.columns)
    assert not missing, f"manifest missing columns: {missing}"
    df_cal = man[man["split_role"] == args.cal_role].reset_index(drop=True)
    df_eval = man[man["split_role"] == args.eval_role].reset_index(drop=True)
    if args.limit:
        df_cal = df_cal.iloc[:args.limit].reset_index(drop=True)
        df_eval = df_eval.iloc[:args.limit].reset_index(drop=True)
    assert len(df_cal) > 0 and len(df_eval) > 0, \
        f"manifest needs non-empty cal+eval splits (got cal={len(df_cal)} eval={len(df_eval)})"
    print(f"[data] source={args.source}  cal={len(df_cal)}  eval={len(df_eval)}  "
          f"members={MEMBERS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device, use_mc_dropout=False)
    ensemble = CXREnsemble(MEMBERS, cfg=cfg)
    # Load images at the largest native_size so a 768 member (RAD-DINO / Ark+)
    # gets true resolution; forward_ensemble_full resizes per member.
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ensemble.members)
    if args.ckpt_overlay:
        overlay = {}
        for part in args.ckpt_overlay.split(","):
            k, _, v = part.partition("=")
            assert k and v, f"--ckpt-overlay entries must be key=path: {part!r}"
            assert k in ensemble.member_keys, f"overlay key {k!r} not in ensemble"
            overlay[k] = v
        print(f"[overlay] loading adapted state_dicts: {overlay}")
        for mb in ensemble.members:
            if mb.key in overlay:
                sd = torch.load(overlay[mb.key], map_location="cpu")
                mb.load_state_dict(sd)
                mb.to(device)
                print(f"[overlay] {mb.key} <- {overlay[mb.key]}")
    member_keys = list(ensemble.member_keys)
    assert "raddino" in member_keys, "feature-based UQ needs the RAD-DINO member"
    print(f"[model] device={device} img_size={cfg.img_size} members={member_keys} "
          f"batch_size={args.batch_size}")

    # --- forward both splits --------------------------------------------------
    print("\n[infer] cal split ...")
    C = forward_split(df_cal, ensemble, cfg, device, f"{args.source}cal",
                      batch_size=args.batch_size)
    print("[infer] eval split ...")
    D = forward_split(df_eval, ensemble, cfg, device, f"{args.source}eval",
                      batch_size=args.batch_size)

    # --- energy (14-class, RAD-DINO logits) ----------------------------------
    for arr in (C, D):
        arr["energy"] = energy_score(arr["rad_logits"])
    print(f"[scores] energy computed (cal range [{C['energy'].min():.2f},"
          f"{C['energy'].max():.2f}])")

    # --- Mahalanobis: re-fit on THIS dataset's cal split, score both ----------
    cal_maha, eval_maha, maha_meta = _fit_score_mahalanobis(C, D, args.maha_class)
    C["mahalanobis"] = cal_maha
    D["mahalanobis"] = eval_maha
    print(f"[maha] cal score range [{cal_maha.min():.2f},{cal_maha.max():.2f}]  "
          f"eval [{eval_maha.min():.2f},{eval_maha.max():.2f}]")

    # --- persist -------------------------------------------------------------
    out_dir = Path(args.out) / args.source
    _save(C, out_dir / f"arrays_{args.source}cal.npz", member_keys)
    _save(D, out_dir / f"arrays_{args.source}eval.npz", member_keys)
    print(f"[save] {out_dir}/arrays_{args.source}cal.npz  probs {C['probs'].shape}")
    print(f"[save] {out_dir}/arrays_{args.source}eval.npz  probs {D['probs'].shape}")

    # --- schema + power sanity printout --------------------------------------
    n_unc = int((D["gt"] < 0).sum())
    print(f"[schema] gt dtype={D['gt'].dtype}  uncertain(-1) eval rows={n_unc}")
    print(f"[schema] valid per-class eval col-sums:")
    for j, name in enumerate(NIH_PATHOLOGIES):
        print(f"   - {name:22s} valid_n={int(D['valid'][:, j].sum()):6d}  "
              f"pos={int((D['gt'][:, j][D['valid'][:, j]] == 1).sum()):4d}  "
              f"unc={int((D['gt'][:, j][D['valid'][:, j]] == -1).sum()):4d}")

    sidecar = {
        "source": args.source, "members": member_keys,
        "n_cal": int(C["probs"].shape[0]), "n_eval": int(D["probs"].shape[0]),
        "img_size": int(cfg.img_size), "device": device,
        "mahalanobis": maha_meta,
        "n_uncertain_eval_rows": n_unc,
        "files": [f"arrays_{args.source}cal.npz", f"arrays_{args.source}eval.npz"],
    }
    with open(out_dir / "build_meta.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[done] wrote arrays + build_meta.json under {out_dir}")


if __name__ == "__main__":
    main()