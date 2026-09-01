#!/usr/bin/env python3
"""Augment a full-library retrieval sidecar with report-derived label vectors.

Adds two columns per library row for the label-agreement rerank (Agent 2):

  gt_labels -- int8 (P,)  report label: 1 positive, 0 negative, -1 uncertain
  gt_valid  -- int8 (P,)  pathology defined for that member set

Sources (joined on image_id, written in sidecar row order):
  split == "train"  -> data/rex_manifest_fs.parquet (labels + valid per image)
  split == "cal"/"eval" -> the adapted npz arrays (gt + valid keys)

The sidecar already holds only prediction-side metadata (predictions,
correctness); gt-derived columns live there already (gt_normal/uncertain), so
this keeps the label-free index/store invariant untouched.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", default="runs/rex_phase7/retrieval/"
                                        "rex_adapted_raddino_full/sidecar.parquet")
    ap.add_argument("--manifest", default="data/rex_manifest_fs.parquet")
    ap.add_argument("--arr-cal", default="runs/rex_phase7/eval_arrays/"
                                         "rex_adapted_s0/arrays_rex_adaptedcal.npz")
    ap.add_argument("--arr-eval", default="runs/rex_phase7/eval_arrays/"
                                          "rex_adapted_s0/arrays_rex_adaptedeval.npz")
    args = ap.parse_args()

    sc = pd.read_parquet(args.sidecar)
    n = len(sc)
    if "gt_labels" in sc.columns:
        sys.exit("sidecar already carries gt_labels -- nothing to do")

    label_by_id: dict[str, np.ndarray] = {}
    valid_by_id: dict[str, np.ndarray] = {}

    man = pd.read_parquet(args.manifest)
    for _, row in man.iterrows():
        label_by_id[str(row["image_id"])] = np.asarray(row["labels"], dtype=np.int8)
        valid_by_id[str(row["image_id"])] = np.asarray(row["valid"], dtype=np.int8)

    ndim = None
    for split, path in (("cal", args.arr_cal), ("eval", args.arr_eval)):
        d = np.load(path, allow_pickle=True)
        ids = np.array([str(i) for i in d["ids"]])
        gt = np.asarray(d["gt"], dtype=np.int8)
        valid = d["valid"].astype(np.int8)
        ndim = gt.shape[1]
        for k in range(len(ids)):
            label_by_id[ids[k]] = gt[k]
            valid_by_id[ids[k]] = valid[k]

    P = ndim or len(np.asarray(man.iloc[0]["labels"]))
    sc_ids = sc["image_id"].astype(str).values
    missing = [i for i in sc_ids if i not in label_by_id]
    if missing:
        sys.exit(f"{len(missing)} sidecar rows have no label source "
                 f"(e.g. {missing[0]}) -- refusing to build a partial sidecar")

    gl = np.stack([label_by_id[i] for i in sc_ids]).astype(np.int8)
    gv = np.stack([valid_by_id[i] for i in sc_ids]).astype(np.int8)
    assert gl.shape == (n, P) and gv.shape == (n, P)

    sc["gt_labels"] = list(gl)
    sc["gt_valid"] = list(gv)
    out = args.sidecar
    # keep a one-time backup next to the original artifact
    import shutil
    backup = out + ".pre_labels_backup"
    if not os.path.exists(backup):
        shutil.copy2(out, backup)
    sc.to_parquet(out)

    pos = (gl == 1).sum(axis=1)
    unc = (gl == -1).sum(axis=1)
    print(f"[sidecar] {n} rows -> {out} (P={P}); backup {backup}")
    print(f"[stats] certain-positives: median {np.median(pos):.0f}, "
          f"zero-positive rows {(pos == 0).sum()}, uncertain labels/row mean {unc.mean():.2f}")
    print("SIDECAR_AUGMENT_DONE")


if __name__ == "__main__":
    main()