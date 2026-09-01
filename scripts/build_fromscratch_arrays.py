#!/usr/bin/env python
"""Build the eval_baselines npz contract for a from-scratch D-Ens (Phase 6).

Companion to ``scripts/train_fromscratch.py``. Loads the 5 seed checkpoints for
one backbone, runs all 5 over the ReXGradient cal(valid)+eval(test) splits, and
writes the SAME npz schema ``eval_baselines._load_arrays`` expects -- so the
from-scratch D-Ens drops into the unchanged multi-dataset eval with zero edits:

    probs (N,5,14) float64, gt (N,14) float64 (preserves -1), valid (N,14) bool,
    rad_logits (N,14), rad_feats (N,D), energy (N,), mahalanobis (N,),
    ids (N,), member_keys (5,), pathologies (14,)

The disagreement across the 5 sigmoid-prob outputs IS the epistemic signal:
``eval_baselines`` computes ``epistemic_std = nanstd(probs, axis=1)`` and
``mutual_info = H(pbar) - mean H(p_m)`` inline, exactly as for the pretrained
ensemble. So a from-scratch D-Ens is evaluated by the identical code path --
the scientific point of the control.

**Feature-density scores (kNN/maha/energy):** the pretrained ensemble gets these
from the RAD-DINO member. A from-scratch D-Ens has no RAD-DINO, so we designate
**seed-0** as the "feature member": its penultimate embedding -> ``rad_feats``
(kNN needs finite features or it crashes), its 14 logits -> ``energy`` +
``rad_logits``, and a ``MahalanobisOOD`` re-fit on the cal split's seed-0
features. This supplies the full baseline suite with NO edits to
``eval_baselines.py`` -- the seed-0 features stand in for RAD-DINO's [CLS]
embedding as the feature-density host.

Bypasses ``CXREnsemble`` / ``Alignment`` / the member framework entirely (those
are for the foundation-pretrained ensemble); a from-scratch control is a plain
backbone + head, loaded straight from its state_dict.

Usage:
    PYTHONPATH=. python scripts/build_fromscratch_arrays.py \
        --backbone resnet18 --seeds-dir checkpoints \
        --manifest data/rex_manifest.parquet --source rex_fs_resnet18 \
        --out runs/eval_arrays --batch-size 256

    # smoke test (CPU / tiny subset):
    PYTHONPATH=. python scripts/build_fromscratch_arrays.py --backbone resnet18 \
        --limit 64 --batch-size 16
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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# import the trainer's model + preprocess (same architecture at train/infer time)
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.feature_uq import MahalanobisOOD, energy_score  # noqa: E402
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402
from train_fromscratch import FromScratchModel, preprocess  # noqa: E402
import build_eval_arrays as _bea  # reuse _fit_score_mahalanobis / _save / _pick_maha_class  # noqa: E402

NUM_CLASSES = len(NIH_PATHOLOGIES)            # 14
DEFAULT_OUT = _REPO / "runs" / "eval_arrays"
DEFAULT_SEEDS = 5


def _load_models(backbone: str, seeds_dir: Path, nseeds: int, device: str):
    """Build nseeds eval-mode FromScratchModels and load each seed's state_dict.
    Checkpoint naming follows train_fromscratch: ``fs_<backbone>_s<seed>.pt``
    (or any ``--out`` path passed at train time -- use --seeds-dir + --pattern)."""
    models = []
    for s in range(nseeds):
        ckpt = seeds_dir / f"fs_{backbone}_s{s}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"missing seed-{s} checkpoint: {ckpt} (train it with "
                f"train_fromscratch.py --backbone {backbone} --seed {s})")
        m = FromScratchModel(backbone).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state)
        models.append(m)
    return models


@torch.no_grad()
def forward_split(df: pd.DataFrame, models, device, label: str,
                  batch_size: int, use_amp: bool) -> dict:
    """Forward every image through all nseeds models; collect probs (N,M,P),
    seed-0 logits (N,14) + features (N,D), gt (N,P) float64 (preserves -1),
    valid (N,P) bool, ids. Skips unloadable images (keeps ids aligned).

    Batches ``batch_size`` images per step -- one preprocess + one forward
    through each of the M models. AMP halves activation memory + speeds the
    small-backbone forward. On CUDA OOM the chunk retries image-by-image so the
    run degrades gracefully rather than crashing."""
    M = len(models)
    probs, rad_logits, rad_feats, gts, valids, ids = [], [], [], [], [], []
    t0 = time.time()
    rows = list(df.itertuples(index=False))

    def _emit(xb, rows_batch):
        z = preprocess(xb, device)                       # (N,3,H,W) ImageNet-norm
        per_img_probs = []
        seed0_logits = seed0_feats = None
        for mi, m in enumerate(models):
            with torch.cuda.amp.autocast(enabled=use_amp and device == "cuda"):
                logits, feats = m.forward_with_features(z)   # (N,14),(N,D)
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)  # (N,14)
            per_img_probs.append(p)
            if mi == 0:
                seed0_logits = logits.cpu().numpy().astype(np.float64)  # (N,14)
                seed0_feats = feats.cpu().numpy().astype(np.float64)   # (N,D)
        # (M,N,14) -> per-image (M,14)
        P = np.stack(per_img_probs, axis=0)             # (M,N,14)
        for k, r in enumerate(rows_batch):
            probs.append(P[:, k, :])                    # (M,14)
            rad_logits.append(seed0_logits[k])          # (14,)
            rad_feats.append(seed0_feats[k])            # (D,)
            gts.append(np.asarray(r.labels, dtype=np.float64))   # -1 preserved
            valids.append(np.asarray(r.valid, dtype=bool))
            ids.append(str(r.image_id))

    def _forward_chunk(xbuf, rows_batch):
        xb = torch.cat(xbuf, dim=0)                     # (N,1,H,W)
        try:
            _emit(xb, rows_batch)
        except RuntimeError as e:
            if ("cuda" in str(e).lower() and "memory" in str(e).lower()
                    and torch.cuda.is_available()):
                print(f"  [oom] batch of {len(xbuf)} OOM'd on "
                      f"{torch.cuda.get_device_name(0)}; retrying batch-1 "
                      f"(lower --batch-size to avoid)")
                torch.cuda.empty_cache()
                for x1, r1 in zip(xbuf, rows_batch):
                    _emit(x1, [r1])
            else:
                raise

    xbuf, rows_batch = [], []
    for i, row in enumerate(rows):
        try:
            x = load_image_tensor(row.image_path, img_size=224, device=device)
        except Exception as e:
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
    if xbuf:
        _forward_chunk(xbuf, rows_batch)
    out = {
        "probs": np.stack(probs),         # (N,M,14)
        "gt": np.stack(gts),              # (N,14) float64
        "valid": np.stack(valids),        # (N,14) bool
        "rad_logits": np.stack(rad_logits),   # (N,14)
        "rad_feats": np.stack(rad_feats),     # (N,D)
        "ids": ids,
    }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True,
                    choices=["resnet18", "vit_tiny", "convnext_tiny"],
                    help="from-scratch backbone (must match the trained checkpoints)")
    ap.add_argument("--seeds-dir", default="checkpoints",
                    help="dir holding fs_<backbone>_s<seed>.pt (default: checkpoints)")
    ap.add_argument("--nseeds", type=int, default=DEFAULT_SEEDS,
                    help=f"number of seeds in the D-Ens (default {DEFAULT_SEEDS})")
    ap.add_argument("--manifest", default="data/rex_manifest.parquet")
    ap.add_argument("--source", default=None,
                    help="dataset name in output filenames (default: rex_fs_<backbone>)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"base output dir (default {DEFAULT_OUT}); writes "
                         f"<out>/<source>/arrays_<source>{{cal,eval}}.npz")
    ap.add_argument("--cal-role", default="cal")
    ap.add_argument("--eval-role", default="eval")
    ap.add_argument("--maha-class", default="Pneumonia",
                    help="pathology to fit Mahalanobis on (default Pneumonia; "
                         "auto-falls-back to the most-balanced powered class)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true", default=None,
                    help="mixed-precision (default: on for cuda, off for cpu)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap images per split (0=all; for smoke tests)")
    args = ap.parse_args(argv)

    source = args.source or f"rex_fs_{args.backbone}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.amp if args.amp is not None else (device == "cuda"))
    seeds_dir = Path(args.seeds_dir)

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
    print(f"[data] source={source}  cal={len(df_cal)}  eval={len(df_eval)}  "
          f"backbone={args.backbone}  nseeds={args.nseeds}")

    models = _load_models(args.backbone, seeds_dir, args.nseeds, device)
    member_keys = [f"{args.backbone}_s{s}" for s in range(args.nseeds)]
    D = models[0].num_features
    print(f"[model] device={device} amp={use_amp} batch_size={args.batch_size}  "
          f"members={member_keys}  feature_dim(seed0)={D}")

    print("\n[infer] cal split ...")
    C = forward_split(df_cal, models, device, f"{source}cal",
                      args.batch_size, use_amp)
    print("[infer] eval split ...")
    E = forward_split(df_eval, models, device, f"{source}eval",
                      args.batch_size, use_amp)

    # --- energy (14-class, seed-0 logits) ------------------------------------
    for arr in (C, E):
        arr["energy"] = energy_score(arr["rad_logits"])
    print(f"[scores] energy computed (cal range [{C['energy'].min():.2f},"
          f"{C['energy'].max():.2f}])")

    # --- Mahalanobis: re-fit on THIS dataset's cal split (seed-0 features) ----
    cal_maha, eval_maha, maha_meta = _bea._fit_score_mahalanobis(C, E, args.maha_class)
    C["mahalanobis"] = cal_maha
    E["mahalanobis"] = eval_maha
    print(f"[maha] cal range [{cal_maha.min():.2f},{cal_maha.max():.2f}]  "
          f"eval [{eval_maha.min():.2f},{eval_maha.max():.2f}]")

    # --- persist -------------------------------------------------------------
    out_dir = Path(args.out) / source
    _bea._save(C, out_dir / f"arrays_{source}cal.npz", member_keys)
    _bea._save(E, out_dir / f"arrays_{source}eval.npz", member_keys)
    print(f"[save] {out_dir}/arrays_{source}cal.npz  probs {C['probs'].shape}")
    print(f"[save] {out_dir}/arrays_{source}eval.npz  probs {E['probs'].shape}")

    n_unc = int((E["gt"] < 0).sum())
    print(f"[schema] gt dtype={E['gt'].dtype}  uncertain(-1) eval rows={n_unc}")
    print(f"[schema] valid per-class eval col-sums:")
    for j, name in enumerate(NIH_PATHOLOGIES):
        print(f"   - {name:22s} valid_n={int(E['valid'][:, j].sum()):6d}  "
              f"pos={int((E['gt'][:, j][E['valid'][:, j]] == 1).sum()):4d}  "
              f"unc={int((E['gt'][:, j][E['valid'][:, j]] == -1).sum()):4d}")

    sidecar = {
        "source": source, "members": member_keys, "backbone": args.backbone,
        "nseeds": args.nseeds, "feature_member": member_keys[0],
        "n_cal": int(C["probs"].shape[0]), "n_eval": int(E["probs"].shape[0]),
        "img_size": 224, "device": device, "mahalanobis": maha_meta,
        "n_uncertain_eval_rows": n_unc,
        "files": [f"arrays_{source}cal.npz", f"arrays_{source}eval.npz"],
    }
    with open(out_dir / "build_meta.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[done] wrote arrays + build_meta.json under {out_dir}")


if __name__ == "__main__":
    main()