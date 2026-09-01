"""Export a portable ground-truth labels file for the OpenI images in
data/openi/, so the demo app (scripts/demo_app.py) can show GT / error_type
for ANY OpenI image a user uploads -- not just the 6 hardcoded gallery
examples. Reads the leak-free manifest (source of truth); writes a flat CSV
keyed by basename (the same key predict() already uses for the gallery
lookup) so matching an uploaded file is a simple filename join.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/export_gt_labels.py
"""
import os

import numpy as np
import pandas as pd

from cxr_uncertainty.config import NIH_PATHOLOGIES

MANIFEST = "data/manifest.parquet"
OUT_CSV = "data/openi_gt_labels.csv"


def main() -> None:
    m = pd.read_parquet(MANIFEST)
    oi = m[m["source"] == "openi"].copy()

    valid = np.stack(oi["valid"].apply(list).values)
    assert (valid == valid[0]).all(), "expected a fixed valid-mask for openi"
    valid_mask = valid[0].astype(bool)   # e.g. Consolidation is False (not labeled in OpenI)

    rows = []
    for _, r in oi.iterrows():
        lab = list(r["labels"])
        row = {
            "basename": os.path.basename(r["image_path"]),
            "image_id": r["image_id"],
            "split_role": r["split_role"],
            "view": r["view"],
        }
        for i, pat in enumerate(NIH_PATHOLOGIES):
            row[pat] = int(lab[i]) if valid_mask[i] else ""   # blank = not labeled in OpenI
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(out)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
