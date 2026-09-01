#!/usr/bin/env python
"""Extract the ConvNeXt-V2 member's penultimate features for OpenI-C / OpenI-D /
role-B -- the SECOND feature space for a 2nd class-conditional-density
(rep-mismatch) confident-error detector (plan §R.2 follow-up: a different
representation that the model did NOT separately memorize).

WHY a 2nd feature space: the shipped RAD-DINO-[CLS] rep-mismatch detector ships
5 pathologies; the rest (Effusion/Infiltration/Nodule/Hernia/...) are
representation-limited on RAD-DINO [CLS] -- proven by the Mahalanobis-centroid
NEGATIVE (two different Gaussian scores fail on the SAME pathologies on the same
features). ConvNeXt-V2-Large is a different inductive bias (ImageNet-CNN, local
high-pass) from RAD-DINO (ViT, global low-pass); its confident-errors may separate
where RAD-DINO's do not. This script ONLY extracts the features; the fit + leak-free
eval + ship guard lives in ``scripts/fit_repmismatch_cv2.py``.

Runs the ConvNeXt-V2 member ALONE (not the full 3-member ensemble) -- 1/3 the
compute -- because only this member's features are needed; the ensemble
probs/decisions (the shared calibration) are already saved in arrays_*.npz. The
member is built via the same factory + the same LP-FT checkpoint
(``checkpoints/convnextv2_lpft.pt``) the ensemble uses, so the features match the
ensemble's convnextv2 member exactly.

Features are saved ALIGNED to each existing ``arrays_{split}.npz`` row order
(loaded from that array's ``ids`` -> manifest ``image_id`` -> ``image_path``), so
the fit script can concatenate ``cv2_feats`` directly without a join.

Run (all three splits):
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/extract_cv2_features.py
Run one split:
    env PYTHONPATH=/teamspace/studios/this_studio SPLIT=openiC python scripts/extract_cv2_features.py
Outputs: runs/phase4_features/cv2_feats_{openiC,openiD,roleB}.npz
        (cv2_feats (N,1536) float32 + ids (N,) object, aligned to arrays_{split}.npz).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import ARCH_MEMBER_REGISTRY  # noqa: E402
from cxr_uncertainty.member_factory import build_member  # noqa: E402
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402

ARR_DIR = _REPO / "runs" / "phase4_features"
MANIFEST = _REPO / "data" / "manifest.parquet"
SPLITS = (["openiC", "openiD", "roleB"]
          if not os.environ.get("SPLIT")
          else os.environ["SPLIT"].split(","))


def extract_split(split: str, member, img_size: int, device: str,
                  id_to_path: dict) -> None:
    arr_path = ARR_DIR / f"arrays_{split}.npz"
    assert arr_path.exists(), f"missing {arr_path}"
    a = np.load(arr_path, allow_pickle=True)
    ids = [str(x) for x in a["ids"]]
    n = len(ids)
    out_path = ARR_DIR / f"cv2_feats_{split}.npz"

    # query the feature dim from the member once
    D = int(member.encoder.num_features)
    feats = np.full((n, D), np.nan, dtype=np.float32)
    missing = 0
    t0 = time.time()
    for i, iid in enumerate(ids):
        path = id_to_path.get(iid)
        if path is None or not os.path.exists(path):
            missing += 1
            continue
        x = load_image_tensor(path, img_size=img_size, device=device)
        with torch.no_grad():
            mo = member.forward_batch(x)
        feats[i] = mo.features[0].detach().cpu().numpy().astype(np.float32)
        del x, mo
        if device == "cuda":
            torch.cuda.empty_cache()
        if (i + 1) % 100 == 0 or i + 1 == n:
            dt = time.time() - t0
            print(f"[cv2:{split}] {i+1}/{n}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
    if missing:
        print(f"[cv2:{split}] WARNING: {missing} ids missing on disk (rows left NaN)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, cv2_feats=feats, ids=np.array(ids, dtype=object),
             member_key=np.array("convnextv2", dtype=object), dim=D)
    dt = time.time() - t0
    fin = int(np.isfinite(feats).sum())
    print(f"[cv2:{split}] done {dt:.1f}s -> {out_path}  "
          f"feats {feats.shape} finite={fin}/{feats.size}  (D={D})")


def main() -> None:
    man = pd.read_parquet(MANIFEST)
    id_to_path = dict(zip(man["image_id"].astype(str), man["image_path"].astype(str)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ARCH_MEMBER_REGISTRY["convnextv2"]
    print(f"[cv2] building convnextv2 member (hf_id={spec.hf_id}, "
          f"ckpt={spec.probe_ckpt}) on {device}")
    member = build_member(spec, device=device)
    member.eval()
    img_size = int(getattr(member, "native_size", 224))
    print(f"[cv2] member ready; native_size={img_size}  D={int(member.encoder.num_features)}")
    for split in SPLITS:
        extract_split(split, member, img_size, device, id_to_path)


if __name__ == "__main__":
    main()