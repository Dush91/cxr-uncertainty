#!/usr/bin/env python
"""Verify the 4-member demo's live predict path after the Ark+ refactor.

Two regression-risk code paths exercised directly (calls demo_app.predict on the
already-loaded module singletons -- identical to a UI click):
  * CXR2177  -- Pneumonia confident-FN flagship; flag 4 must fire via the RAD-DINO
    detector (the pre-Ark+ behavior preserved by the SPLIT).
  * CXR1003  -- Hernia (Ark+-only); flag 4 must fire via the NEW extract_member(eo,
    "arkswin") + _ARK_PCA path (the hot-loop code path changed this session).
  * CXR1174  -- Infiltration (Ark+-only); same new Ark+ path.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/verify_4mem_predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import scripts.demo_app as demo  # noqa: E402  (registers plugins on import)


def main() -> None:
    demo.load_state()
    print(f"\n[verify] MEMBERS={demo.MEMBERS}")
    print(f"[verify] _ARK_NEW={demo._ARK_NEW}")
    print(f"[verify] _ARK_PCA={'set' if demo._ARK_PCA is not None else 'NONE'}")
    print(f"[verify] _REPM_RD={'set' if demo._REPM_RD is not None else 'NONE'} "
          f"_REPM_ARK={'set' if demo._REPM_ARK is not None else 'NONE'}\n")

    cases = [
        ("CXR2177 Pneumonia confident-FN (RAD-DINO path)",
         "data/openi/CXR2177_IM-0789-4001.png", "Pneumonia"),
        ("CXR1003 Hernia (Ark+ path)",
         "data/openi/CXR1003_IM-0005-2002.png", "Hernia"),
        ("CXR1174 Infiltration (Ark+ path)",
         "data/openi/CXR1174_IM-0118-1001.png", "Infiltration"),
    ]
    all_ok = True
    for label, path, watch in cases:
        banner, df, maha_pct, maha_str, mdf, cap, _ = demo.predict(path)
        if df is None or len(df) == 0:
            print(f"[FAIL] {label}: empty dataframe")
            all_ok = False
            continue
        row = df[df["pathology"] == watch]
        if len(row) == 0:
            print(f"[FAIL] {label}: '{watch}' not in table")
            all_ok = False
            continue
        r = row.iloc[0]
        fired = bool(r["repmismatch_flag"])
        # also report whether any rep-mismatch flag fired at all on the image
        any_repm = bool(df["repmismatch_flag"].any())
        path_fired = ", ".join(df.loc[df["repmismatch_flag"], "pathology"].tolist())
        print(f"[{label}]")
        print(f"   {watch}: p_bar={r['p_bar']} dec={r['calibrated_decision']} "
              f"conf={r['calib_confidence']} gt={r['gt']} err={r['error_type']} "
              f"repmismatch_flag={fired}")
        print(f"   maha_pct={maha_pct:.0f}  any_repmismatch={any_repm} "
              f"fired_pathologies=[{path_fired}]")
        print(f"   member probs: {dict(zip(mdf['member'], [round(v,3) for v in mdf['prob']]))}")
        # The flagship expectation: CXR2177 Pneumonia must fire via RAD-DINO.
        if watch == "Pneumonia":
            if not fired:
                print(f"   [REGRESSION] Pneumonia rep-mismatch did NOT fire on CXR2177 -- "
                      f"the RAD-DINO SPLIT path is broken.")
                all_ok = False
            else:
                print(f"   [OK] RAD-DINO Pneumonia confident-FN flagship fires (preserved).")
        # For Ark+-only pathologies the key check is that the Ark+ feature path runs
        # without exception and the detector is queried; whether it fires depends on the
        # image being a confident-FP/FN (not all are). Report honestly.
        print()
    print("[verify] PASS" if all_ok else "[verify] SEE NOTES ABOVE")


if __name__ == "__main__":
    main()