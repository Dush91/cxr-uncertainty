"""Offline calibration-variant ablation: load the forward ``arrays.npz`` saved
by ``eval_phase3.py`` and run all four calibration variants in seconds (no
re-forward), to isolate which step helps/hurts the confident-error signal:

  raw       (no Beta, no TS)  — Phase-2 baseline
  ts_only   (TS on raw pooled mean, raw disagreement) — §E.2 design: fix ECE,
              preserve the raw disagreement ordering that discriminates errors
  beta_only (Beta alignment, no TS) — isolate Beta's effect on the disagreement
  beta+ts   (full plan §E.1 steps 1-5)

Key finding it tests: per-member Beta alignment fixes calibration but removes
the inter-member scale signal that discriminates confident errors; TS-only
keeps that signal while still fixing ECE (TS is monotone on the pooled mean, so
the disagreement ordering is invariant, §E.2).

Usage:
    python scripts/analyze_calibration.py --arrays runs/phase3/arrays.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr_uncertainty.calibration import build_pipeline
from cxr_uncertainty.config import RiskConfig
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds


def _drop_no_pos(df_calib, df_eval):
    pos_per = df_eval.groupby("pathology")["gt"].sum()
    unmappable = pos_per[pos_per == 0].index.tolist()
    if unmappable:
        df_eval = df_eval[~df_eval["pathology"].isin(unmappable)].reset_index(drop=True)
        df_calib = df_calib[~df_calib["pathology"].isin(unmappable)].reset_index(drop=True)
    return df_calib, df_eval


def _monotonic(sel):
    prev = -1.0
    for s in sel:
        if s["auroc_epistemic_std"] < prev - 1e-9:
            return False
        prev = s["auroc_epistemic_std"]
    return True


def run_variant(pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys, use_beta, use_ts,
                conf_pct, unc_pct, n_imgs):
    df_calib, df_eval, meta = build_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys,
        use_beta=use_beta, use_ts=use_ts)
    df_calib, df_eval = _drop_no_pos(df_calib, df_eval)
    thr = calibrate_thresholds(df_calib)
    rec, _ = build_records(df_eval, thr, conf_pct, unc_pct)
    cfg = RiskConfig()
    rep = evaluate_records(rec, cfg, n_imgs=n_imgs)
    sel = auroc_at_confidence_levels(rec)
    return rep, sel, meta


def main(argv=None):
    p = argparse.ArgumentParser(prog="analyze_calibration", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays", default="runs/phase3/arrays.npz")
    p.add_argument("--out", default="runs/phase3")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    args = p.parse_args(argv)

    z = np.load(args.arrays, allow_pickle=True)
    pC, yC, vC, idsC = z["pC"], z["yC"], z["vC"], list(z["idsC"])
    pD, yD, vD, idsD = z["pD"], z["yD"], z["vD"], list(z["idsD"])
    paths, keys = list(z["paths"]), list(z["keys"])
    print(f"[load] C {pC.shape}  D {pD.shape}  members={keys}")

    variants = [
        ("raw",       False, False),
        ("ts_only",   False, True),
        ("beta_only", True,  False),
        ("beta+ts",   True,  True),
    ]
    rows = []
    print("\n" + "=" * 92)
    print(f"{'variant':10s} {'ECE':>6s} {'AURC':>8s} {'cw':>3s} | "
          f"{'top50':>6s} {'top25':>6s} {'top15':>6s} {'top10':>6s} {'top5':>6s} | "
          f"{'monot':>5s} {'T':>5s} {'beta':>5s}")
    print("-" * 92)
    results = {}
    for name, ub, ut in variants:
        rep, sel, meta = run_variant(pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys,
                                     ub, ut, args.conf_pct, args.unc_pct, len(idsD))
        def g(pct):
            return next((s["auroc_epistemic_std"] for s in sel if s["conf_top_pct"] == pct), None)
        results[name] = {"report": rep.to_dict(), "selectivity": sel, "meta": meta}
        cw = rep.n_confident_wrong
        print(f"{name:10s} {rep.ece:6.4f} {rep.aurc:8.5f} {cw:3d} | "
              f"{g(50) or 0:6.3f} {g(25) or 0:6.3f} {g(15) or 0:6.3f} "
              f"{g(10) or 0:6.3f} {g(5) or 0:6.3f} | "
              f"{str(_monotonic(sel)):5s} {meta['temperature_T']:5.3f} "
              f"{meta['beta_slots_fit']:3d}/{meta['beta_slots_fit']+meta['beta_slots_identity']}")
    print("=" * 92)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibration_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out}/calibration_ablation.json")


if __name__ == "__main__":
    main()