#!/usr/bin/env python
"""Extract ONE ensemble member's penultimate features for OpenI-C / OpenI-D /
role-B -- a generic, member-agnostic feature extractor for the 2nd / 3rd / ...
class-conditional-density (representation-mismatch) confident-error detectors.

Generalizes ``extract_cv2_features.py`` (which is hardcoded to the convnextv2
member) to ANY registered member via the ``MEMBER`` env var. Used so far for:
  * MEMBER=biomedclip  -> BiomedCLIP ViT-B/16 image-tower projected embedding (512-d),
    the language-contrastive (CLIP) feature space -- the 4th representation
    strategy, and the one most likely to differ from the SSL-ViT (RAD-DINO) and
    SSL-CNN (ConvNeXt-V2) spaces already tested for rep-mismatch.

WHY a 2nd+ feature space: the shipped RAD-DINO-[CLS] rep-mismatch detector ships
5 pathologies; the rest (Effusion/Infiltration/Nodule/Hernia) are
representation-limited on RAD-DINO [CLS]. ConvNeXt-V2 (a 2nd SSL pure-vision
space) was a DECISIVE NEGATIVE (ships 1, dominated). BiomedCLIP tests a
genuinely DIFFERENT *objective* (language-contrastive, not self-supervised
vision) -- its class-conditional geometry is unstudied for DDU, so it is the
highest-uncertainty / most-likely-to-surprise candidate. The fit + leak-free eval
+ ship guard lives in ``scripts/fit_repmismatch_member.py``.

Runs the member ALONE (1/M the compute of the full ensemble) -- only this
member's features are needed; the ensemble probs/decisions (the shared
calibration) are already saved in arrays_*.npz. The member is built via the same
factory + the same probe checkpoint the ensemble uses, so the features match the
ensemble's member exactly.

Features are saved ALIGNED to each existing ``arrays_{split}.npz`` row order
(loaded from that array's ``ids`` -> manifest ``image_id`` -> ``image_path``), so
the fit script can concatenate the features directly without a join.

Run (all three splits, one member):
    env PYTHONPATH=/teamspace/studios/this_studio MEMBER=biomedclip \
        python scripts/extract_member_features.py
Run one split:
    env PYTHONPATH=/teamspace/studios/this_studio MEMBER=biomedclip SPLIT=openiC \
        python scripts/extract_member_features.py
Outputs: runs/phase4_features/{MEMBER}_feats_{openiC,openiD,roleB}.npz
        (feats (N,D) float32 + ids (N,) object, aligned to arrays_{split}.npz).
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

MEMBER = os.environ.get("MEMBER", "biomedclip")
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
    out_path = ARR_DIR / f"{MEMBER}_feats_{split}.npz"

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
            print(f"[{MEMBER}:{split}] {i+1}/{n}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
    if missing:
        print(f"[{MEMBER}:{split}] WARNING: {missing} ids missing on disk "
              f"(rows left NaN)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, feats=feats, ids=np.array(ids, dtype=object),
             member_key=np.array(MEMBER, dtype=object), dim=D)
    dt = time.time() - t0
    fin = int(np.isfinite(feats).sum())
    print(f"[{MEMBER}:{split}] done {dt:.1f}s -> {out_path}  "
          f"feats {feats.shape} finite={fin}/{feats.size}  (D={D})")


def main() -> None:
    assert MEMBER in ARCH_MEMBER_REGISTRY, \
        f"MEMBER={MEMBER!r} not in ARCH_MEMBER_REGISTRY " \
        f"(have: {sorted(ARCH_MEMBER_REGISTRY)})"
    man = pd.read_parquet(MANIFEST)
    id_to_path = dict(zip(man["image_id"].astype(str), man["image_path"].astype(str)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ARCH_MEMBER_REGISTRY[MEMBER]
    print(f"[{MEMBER}] building member (hf_id={spec.hf_id}, ckpt={spec.probe_ckpt}) "
          f"on {device}")
    member = build_member(spec, device=device)
    member.eval()
    img_size = int(getattr(member, "native_size", 224))
    print(f"[{MEMBER}] member ready; native_size={img_size}  "
          f"D={int(member.encoder.num_features)}")
    for split in SPLITS:
        extract_split(split, member, img_size, device, id_to_path)


if __name__ == "__main__":
    main()