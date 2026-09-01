"""Phase 3 evaluation: BCTS+ETS calibration-then-uncertainty on the 3-member
architecture-diverse ensemble, leak-free on OpenI (role-C cal / role-D eval).

This is the plan's actual Phase-2/3 measurement (§J.5 Phase 2 acceptance was
specified with ``--calibrator bcts_ets_mondrian``; Phase 2 ran the raw Youden
baseline, giving the un-calibrated 0.42 top-5% number). It forwards role C and
role D **once**, collecting **full-precision** ``(N, M, P)`` per-member prob
arrays in-process (NOT the rounded ``member_probs`` CSV JSON, so Beta
calibration near 0/1 is accurate), then runs BOTH pipelines on the identical
forward:

  * **raw (Youden)**   — disagreement on raw member probs (reproduces Phase-2).
  * **calibrated**     — per-member Beta alignment (Kull 2017) -> disagreement on
    aligned probs -> pool -> temperature scaling on the pooled aligned mean
    (monotone, §E.2). ECE should drop; whether the aligned disagreement also
    lifts the top-5% confident-error AUROC above the raw 0.42 / makes the
    selectivity curve non-decreasing is the open question this run answers.

Steps 6 (DEGRE) and 7 (Mondrian conformal) are NOT run here — the defining
metric (confident-error AUROC across selectivity) doesn't need them.

Usage:
    python scripts/eval_phase3.py --arch-ensemble xrv_nih,convnextv2,raddino \
        --out runs/phase3 --no-mc
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.calibration import calibrated_pipeline, raw_pipeline
from cxr_uncertainty.evaluate import evaluate_records
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.reanalyze import auroc_at_confidence_levels, build_records, calibrate_thresholds
from cxr_uncertainty.utils import load_image_tensor

warnings.filterwarnings("ignore")


@torch.no_grad()
def forward_collect(df, ensemble, cfg, device, label="D"):
    """Forward every image in ``df``; return (N, M, P) per-member probs + (N, P)
    gt + (N, P) valid + image_ids. Full precision (in-process, not from CSV)."""
    probs, gts, valids, ids = [], [], [], []
    t0 = time.time()
    P = len(NIH_PATHOLOGIES)
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            x = load_image_tensor(row.image_path, img_size=cfg.img_size, device=device)
        except Exception as e:
            print(f"  [skip] {row.image_id}: {e}")
            continue
        eo = ensemble.forward_ensemble_full(x)
        # per_member_probs: (M, 1, P) aligned to predictable (= NIH-14 for all 3 arch members).
        pm = eo.per_member_probs[:, 0, :].cpu().numpy().astype(np.float64)   # (M, P)
        probs.append(pm)
        gts.append(np.asarray(row.labels, dtype=np.float64))                  # (P,)
        valids.append(np.asarray(row.valid, dtype=bool))                       # (P,)
        ids.append(str(row.image_id))
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            dt = time.time() - t0
            print(f"  [infer {label}] {i+1}/{len(df)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (np.stack(probs), np.stack(gts), np.stack(valids), ids)   # (N,M,P),(N,P),(N,P),list


def evaluate_path(df_calib, df_eval, conf_pct, unc_pct, n_imgs):
    """Drop classes with no eval positives, Youden on cal, build_records on eval,
    evaluate, selectivity. Returns (report, selectivity, meta)."""
    pos_per = df_eval.groupby("pathology")["gt"].sum()
    unmappable = pos_per[pos_per == 0].index.tolist()
    if unmappable:
        df_eval = df_eval[~df_eval["pathology"].isin(unmappable)].reset_index(drop=True)
        df_calib = df_calib[~df_calib["pathology"].isin(unmappable)].reset_index(drop=True)
    thresholds = calibrate_thresholds(df_calib)
    rec, meta = build_records(df_eval, thresholds, conf_pct, unc_pct)
    cfg = RiskConfig()
    report = evaluate_records(rec, cfg, n_imgs=n_imgs)
    sel = auroc_at_confidence_levels(rec)
    meta["thresholds"] = {k: round(v, 4) for k, v in thresholds.items()}
    return report, sel, meta


def _monotonic(sel):
    prev = -1.0
    for s in sel:
        if s["auroc_epistemic_std"] < prev - 1e-9:
            return False
        prev = s["auroc_epistemic_std"]
    return True


def _print_block(title, report, sel, meta):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"records={report.n_records}  confident={report.n_confident}  "
          f"confident_wrong={report.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {report.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {report.confident_error_auroc_mi}")
    print(f"confident-error recall (risk_flag)    : {report.confident_error_recall}")
    print(f"false-flag rate (confident correct)  : {report.false_flag_rate}")
    print(f"AURC (risk-coverage)                 : {report.aurc}")
    print(f"ECE (calibration of p_bar)            : {report.ece:.4f}")
    if "temperature_T" in meta:
        print(f"temperature T (pooled mean)          : {meta['temperature_T']}")
    if "beta_slots_fit" in meta:
        print(f"Beta slots fit / identity            : {meta['beta_slots_fit']}/{meta['beta_slots_identity']}")
    print("-" * 70)
    print("selectivity curve (non-decreasing 50->5% is the defining property):")
    for s in sel:
        print(f"  top {s['conf_top_pct']:2d}%: n={s['n_confident']:5d} "
              f"wrong={s['n_confident_wrong']:4d}  "
              f"AUROC_std={s['auroc_epistemic_std']}  AUROC_mi={s['auroc_mutual_info']}")
    print("-" * 70)
    print(f"MONOTONIC non-decreasing 50->5%: {_monotonic(sel)}")
    print("=" * 70)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_phase3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.parquet")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2,raddino")
    p.add_argument("--out", default="runs/phase3")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--no-mc", action="store_true")
    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device, mc_samples=args.mc_samples,
                     use_mc_dropout=not args.no_mc)

    df = pd.read_parquet(args.manifest)
    dfC = df[df.split_role == "C"].reset_index(drop=True)
    dfD = df[df.split_role == "D"].reset_index(drop=True)
    if args.limit:
        dfC = dfC.iloc[:args.limit]
        dfD = dfD.iloc[:args.limit]
    print(f"[data] role C (cal)={len(dfC)}  role D (eval)={len(dfD)}")

    ensemble = CXREnsemble(args.arch_ensemble.split(","), cfg=cfg)
    print(f"[model] arch ensemble: {ensemble.member_keys} | "
          f"predictable={len(ensemble.predictable)} | has_dropout={ensemble.has_dropout}")

    print("[infer] forwarding role C (calibration)...")
    pC, yC, vC, idsC = forward_collect(dfC, ensemble, cfg, device, label="C")
    print("[infer] forwarding role D (eval)...")
    pD, yD, vD, idsD = forward_collect(dfD, ensemble, cfg, device, label="D")
    pathologies = list(ensemble.predictable)   # NIH-14 (all 3 arch members define all 14)
    member_keys = list(ensemble.member_keys)
    print(f"[arrays] C {pC.shape}  D {pD.shape}  (N, M, P)")

    # Persist the forward arrays so calibration variants (ts_only, beta_only,
    # beta+ts, ...) can be re-run in seconds without re-forwarding (scripts/analyze_calibration.py).
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(Path(args.out) / "arrays.npz"),
                        pC=pC, yC=yC, vC=vC, idsC=np.array(idsC, dtype=object),
                        pD=pD, yD=yD, vD=vD, idsD=np.array(idsD, dtype=object),
                        paths=np.array(pathologies, dtype=object),
                        keys=np.array(member_keys, dtype=object))
    print(f"[arrays] saved -> {Path(args.out)}/arrays.npz")

    # --- RAW (un-calibrated) baseline on the same forward -----------------
    df_cal_raw, df_eval_raw, meta_raw = raw_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, pathologies, member_keys)
    rep_raw, sel_raw, m_raw = evaluate_path(df_cal_raw, df_eval_raw,
                                           args.conf_pct, args.unc_pct, len(idsD))

    # --- CALIBRATED (BCTS + ETS, steps 1-5) on the same forward ------------
    df_cal_cal, df_eval_cal, meta_cal = calibrated_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, pathologies, member_keys)
    rep_cal, sel_cal, m_cal = evaluate_path(df_cal_cal, df_eval_cal,
                                            args.conf_pct, args.unc_pct, len(idsD))

    _print_block("RAW (Youden) — un-calibrated disagreement  [Phase-2 baseline]",
                 rep_raw, sel_raw, {**meta_raw, **m_raw})
    _print_block("CALIBRATED (BCTS + ETS) — aligned disagreement + TS on pooled mean",
                 rep_cal, sel_cal, {**meta_cal, **m_cal})

    # Side-by-side comparison summary.
    print("\n" + "=" * 70)
    print("DELTA  (calibrated - raw)")
    print("=" * 70)
    def top5(sel): return next((s["auroc_epistemic_std"] for s in sel if s["conf_top_pct"] == 5), None)
    def top10(sel): return next((s["auroc_epistemic_std"] for s in sel if s["conf_top_pct"] == 10), None)
    def top50(sel): return next((s["auroc_epistemic_std"] for s in sel if s["conf_top_pct"] == 50), None)
    print(f"  ECE            : {rep_raw.ece:.4f} -> {rep_cal.ece:.4f}  "
          f"({'improved' if rep_cal.ece < rep_raw.ece else 'WORSE'})")
    print(f"  top-50% AUROC  : {top50(sel_raw)} -> {top50(sel_cal)}")
    print(f"  top-10% AUROC  : {top10(sel_raw)} -> {top10(sel_cal)}")
    print(f"  top- 5% AUROC  : {top5(sel_raw)} -> {top5(sel_cal)}")
    print(f"  AURC           : {rep_raw.aurc} -> {rep_cal.aurc}")
    print(f"  MONOTONIC      : raw={_monotonic(sel_raw)}  cal={_monotonic(sel_cal)}")
    print("=" * 70)

    # Persist.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "evaluation_raw.json", "w") as f:
        json.dump(rep_raw.to_dict(), f, indent=2)
    with open(out_dir / "evaluation_calibrated.json", "w") as f:
        json.dump(rep_cal.to_dict(), f, indent=2)
    with open(out_dir / "phase3_meta.json", "w") as f:
        json.dump({"raw": {**meta_raw, "selectivity": sel_raw},
                   "calibrated": {**meta_cal, "selectivity": sel_cal}}, f, indent=2)
    print(f"[done] wrote {out_dir}/{{evaluation_raw,evaluation_calibrated,phase3_meta}}.json")


if __name__ == "__main__":
    main()