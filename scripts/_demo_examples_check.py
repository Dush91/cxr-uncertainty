#!/usr/bin/env python
"""One-off: run the demo's exact predict() on each built-in example + an emphysema upload,
print the banner + flagged/interesting rows. Diagnoses 'it still doesn't flag'."""
import pandas as pd
import scripts.demo_app as D

D.load_state()
pd.set_option("display.width", 220)

# also pull an OpenI emphysema-positive image (the user named emphysema) from the manifest
import numpy as np
man = pd.read_parquet("data/manifest.parquet")
EMP = D.NIH_PATHOLOGIES.index("Emphysema")
emp_rows = man[(man["split_role"] == "D") & man["labels"].apply(
    lambda L: bool(np.asarray(L)[EMP]))]
emp_path = str(emp_rows.iloc[0]["image_path"]) if len(emp_rows) else None
print("\n##### BUILT-IN EXAMPLES #####")
for e in D._EXAMPLES:
    print(f"\n===== {e['image_id']} | {e['kind']} =====")
    print(f"caption: {e['caption']}")
    out = D.predict(e["image_path"])  # 7-tuple
    banner, df = out[0], out[1]
    maha_txt = out[3]
    print("BANNER:", " ".join(banner.split())[:300])
    cols = list(df.columns)
    flagged = df[(df.get("threshold_flag", False)) | (df.get("conformal_refer", False)) |
                 (df.get("repmismatch_flag", False))]
    print("FLAGGED ROWS:")
    print(flagged.to_string(index=False) if len(flagged) else "  (none)")
    print(f"MAHA: {maha_txt}")

if emp_path:
    import os
    if os.path.exists(emp_path):
        print("\n===== EMPHYSEMA (user-named) =====")
        out = D.predict(emp_path)
        banner, df, maha_txt = out[0], out[1], out[3]
        print("BANNER:", " ".join(banner.split())[:300])
        emp_row = df[df["pathology"] == "Emphysema"]
        print("EMPHYSEMA ROW:")
        print(emp_row.to_string(index=False))
        flagged = df[(df.get("threshold_flag", False)) | (df.get("conformal_refer", False)) |
                     (df.get("repmismatch_flag", False))]
        print("ANY FLAGGED ROWS:")
        print(flagged.to_string(index=False) if len(flagged) else "  (none)")
        print(f"MAHA: {maha_txt}")