#!/usr/bin/env python
"""Run the 3-member arch ensemble on manifest role-B (training-distribution held-outs:
NIH 1002 + CheXpert 1000) and save probs/gt/valid/ids for the importance-weighted
calibration-set augmentation (plan §O / §E.3).

This mirrors the demo's inference path exactly (load_image_tensor + forward_ensemble_full,
batch-1) so the role-B p_bar is on the identical scale as OpenI-C/D. It saves
probs/gt/valid/ids PLUS ``rad_feats`` (N,1536) -- the frozen RAD-DINO [CLS] features
(first 768 cols; last 768 zero-pad, the Alignment.stack layout), via ``extract_member``.
The features power the class-conditional-density (representation-mismatch) confident-error
flag for the RARE pathologies (Mass/Pneumothorax): OpenI-C alone has too few positives
(Mass 4, Pneumothorax 10) to fit a 768-d Gaussian, so role-B's 114/147 positives are pooled
in. They are NOT added to the Mahalanobis fit (that flag is about image foreign-ness and
role-B is the *training* distribution; adding it would pull the in-distribution cloud
toward training and make OpenI look MORE foreign -- worse for the under-shift use case).
Keep Mahalanobis on OpenI-C (unchanged).

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/run_roleB_inference.py
"""
from __future__ import annotations

import argparse
import os
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

from cxr_uncertainty.config import RiskConfig, NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.models import CXREnsemble  # noqa: E402
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402
from cxr_uncertainty.feature_uq import extract_member  # noqa: E402

MANIFEST = _REPO / "data" / "manifest.parquet"
DEFAULT_MEMBERS = "xrv_nih,convnextv2,raddino"
DEFAULT_OUT = _REPO / "runs" / "phase4_features" / "arrays_roleB.npz"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--members", default=DEFAULT_MEMBERS,
                    help=f"comma-separated member keys (default: {DEFAULT_MEMBERS})")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"output npz path (default: {DEFAULT_OUT})")
    args = ap.parse_args()
    MEMBERS = [m.strip() for m in args.members.split(",") if m.strip()]
    OUT = Path(args.out)

    man = pd.read_parquet(MANIFEST)
    roleB = man[man["split_role"] == "B"].reset_index(drop=True)
    n = len(roleB)
    print(f"[roleB] {n} images  (sources: {dict(roleB['source'].value_counts())})")
    assert n > 0, "no role-B rows in manifest"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device)
    ens = CXREnsemble(members=MEMBERS, cfg=cfg)
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ens.members)
    print(f"[roleB] device={device} img_size={cfg.img_size} members={ens.member_keys}")

    P = len(NIH_PATHOLOGIES)
    M = len(MEMBERS)
    probs = np.full((n, M, P), np.nan, dtype=np.float64)
    gt = np.zeros((n, P), dtype=np.float64)
    valid = np.zeros((n, P), dtype=np.uint8)
    rad_feats = np.full((n, 1536), np.nan, dtype=np.float64)   # raddino [CLS] (768) + zero-pad
    ids: list[str] = []

    t0 = time.time()
    missing = 0
    for i, row in roleB.iterrows():
        path = str(row["image_path"])
        if not os.path.exists(path):
            missing += 1
            ids.append(str(row["image_id"]))
            gt[i] = np.asarray(row["labels"], dtype=np.float64)
            valid[i] = np.asarray(row["valid"], dtype=np.uint8)
            continue
        x = load_image_tensor(path, img_size=cfg.img_size, device=cfg.device)
        with torch.no_grad():
            eo = ens.forward_ensemble_full(x)
        # per_member_probs is (M, B=1, P) -> (M, P); move to host.
        pm = eo.per_member_probs[:, 0, :].detach().cpu().numpy().astype(np.float64)
        probs[i] = pm
        # RAD-DINO [CLS] features (1536-d zero-padded layout, same as arrays_openiC).
        # Only present when raddino is a voting member (it is in the 3- and 4-member
        # ensembles); skip otherwise.
        if "raddino" in MEMBERS:
            _, feat = extract_member(eo, "raddino")
            rad_feats[i] = feat
        gt[i] = np.asarray(row["labels"], dtype=np.float64)
        valid[i] = np.asarray(row["valid"], dtype=np.uint8)
        ids.append(str(row["image_id"]))
        if (i + 1) % 100 == 0 or i + 1 == n:
            dt = time.time() - t0
            print(f"[roleB] {i+1}/{n}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        del x
        if device == "cuda":
            torch.cuda.empty_cache()

    if missing:
        print(f"[roleB] WARNING: {missing} images missing on disk (rows left NaN in probs)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        probs=probs,
        gt=gt,
        valid=valid,
        rad_feats=rad_feats,
        ids=np.array(ids, dtype=object),
        member_keys=np.array(MEMBERS, dtype=object),
        pathologies=np.array(NIH_PATHOLOGIES, dtype=object),
    )
    dt = time.time() - t0
    print(f"[roleB] done in {dt:.1f}s  -> {OUT}")
    print(f"[roleB] probs {probs.shape}  finite={np.isfinite(probs).sum()}/{probs.size}")
    print(f"[roleB] rad_feats {rad_feats.shape}  finite={np.isfinite(rad_feats).sum()}/{rad_feats.size}")
    print(f"[roleB] gt positives per pathology:")
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vp = valid[:, j].astype(bool)
        pos = int((gt[:, j] * vp).sum())
        print(f"   - {pat:22s} pos={pos:4d}  valid_n={int(vp.sum())}")


if __name__ == "__main__":
    main()