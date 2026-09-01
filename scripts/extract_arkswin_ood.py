#!/usr/bin/env python
"""Extract Ark+ (Swin-L@768) penultimate features for the OOD shift probes
(kermany, covid), aligned to the existing ``runs/phase4_features/arrays_{split}.npz``
row order.

The generic ``extract_member_features.py`` reads ``data/manifest.parquet`` for
image paths, but kermany/covid live in ``data/ood_manifest.parquet`` -- so this
small dedicated extractor mirrors the generic one but resolves paths from the
OOD manifest. Output: ``runs/phase4_features/arkswin_feats_{split}.npz``
(``feats (N,1536)`` + ``ids (N,)``), row-aligned to ``arrays_{split}.npz`` so
the under-shift Mahalanobis comparison can concatenate directly.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/extract_arkswin_ood.py
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
OOD_MANIFEST = _REPO / "data" / "ood_manifest.parquet"
SPLITS = os.environ.get("SPLITS", "kermany,covid").split(",")
MEMBER = "arkswin"


def main() -> None:
    ood = pd.read_parquet(OOD_MANIFEST)
    id_to_path = dict(zip(ood["image_id"].astype(str), ood["image_path"].astype(str)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = ARCH_MEMBER_REGISTRY[MEMBER]
    print(f"[{MEMBER}] building member (hf_id={spec.hf_id}, ckpt={spec.probe_ckpt}) on {device}")
    member = build_member(spec, device=device)
    member.eval()
    img_size = int(getattr(member, "native_size", 768))
    D = int(member.encoder.num_features)
    print(f"[{MEMBER}] ready; native_size={img_size} D={D}")
    for split in SPLITS:
        arr_path = ARR_DIR / f"arrays_{split}.npz"
        assert arr_path.exists(), f"missing {arr_path}"
        a = np.load(arr_path, allow_pickle=True)
        ids = [str(x) for x in a["ids"]]
        n = len(ids)
        out_path = ARR_DIR / f"{MEMBER}_feats_{split}.npz"
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
            print(f"[{MEMBER}:{split}] WARNING: {missing} ids missing on disk (rows left NaN)")
        np.savez(out_path, feats=feats, ids=np.array(ids, dtype=object),
                 member_key=np.array(MEMBER, dtype=object), dim=D)
        dt = time.time() - t0
        fin = int(np.isfinite(feats).sum())
        print(f"[{MEMBER}:{split}] done {dt:.1f}s -> {out_path}  "
              f"feats {feats.shape} finite={fin}/{feats.size} (D={D})")


if __name__ == "__main__":
    main()