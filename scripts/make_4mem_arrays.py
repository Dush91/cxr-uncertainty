#!/usr/bin/env python
"""Subset the 5-member arrays (runs/phase4_features_5mem) to the 4-member
Ark+ ensemble [xrv_nih, convnextv2, raddino, arkswin] (drop biomedclip, index 3)
and save to runs/phase4_features_4mem/arrays_{split}.npz.

The member-independent fields (gt, valid, ids, rad_feats, mahalanobis, energy,
rad_logits) are copied verbatim; only ``probs`` is sliced on the member axis.
This lets the rep-mismatch fit scripts (which read arrays from a path) re-fit on
the 4-member confident set without re-running inference.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/make_4mem_arrays.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SRC = _REPO / "runs" / "phase4_features_5mem"
DST = _REPO / "runs" / "phase4_features_4mem"
IDX4 = [0, 1, 2, 4]   # [xrv_nih, convnextv2, raddino, arkswin] (drop biomedclip=3)
MKEYS = ["xrv_nih", "convnextv2", "raddino", "arkswin"]

for split in ["openiC", "openiD", "kermany", "covid"]:
    z = np.load(SRC / f"arrays_{split}.npz", allow_pickle=True)
    DST.mkdir(parents=True, exist_ok=True)
    kwargs = {k: z[k] for k in z.files}
    kwargs["probs"] = z["probs"][:, IDX4, :]
    kwargs["member_keys"] = np.array(MKEYS, dtype=object)
    out = DST / f"arrays_{split}.npz"
    np.savez(out, **kwargs)
    print(f"{split}: probs {kwargs['probs'].shape}  (was {z['probs'].shape})  -> {out}")
print("done.")