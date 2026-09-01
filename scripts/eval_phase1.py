"""Phase 1 evaluation: 2-member arch-ensemble on the leak-free OpenI splits.

Runs ``--arch-ensemble`` members over manifest **role C** (ensemble cal, 2k OpenI)
and **role D** (leak-free eval, ~2k OpenI), writes a schema-2 ``per_record.csv``,
and computes the headline confident-error selectivity curve with per-class Youden
thresholds fit on role C and evaluated on role D (NOT the CLI's random 50/50 split —
the manifest's patient-disjoint C/D split is the leak-free contract).

Acceptance (plan §J.5 Phase 1): the selectivity curve's ``auroc_epistemic_std`` is
monotonically non-decreasing 50→5% on role D; ``per_record.csv`` carries both
``member_probs`` and ``logits``.

Usage:
    python scripts/eval_phase1.py --arch-ensemble xrv_nih,convnextv2 --out runs/phase1
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
from cxr_uncertainty.evaluate import Records, evaluate_records, save_report
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.reanalyze import (
    auroc_at_confidence_levels, build_records, calibrate_thresholds,
)
from cxr_uncertainty.risk import assess_image
from cxr_uncertainty.uncertainty import estimate
from cxr_uncertainty.utils import load_image_tensor

warnings.filterwarnings("ignore")


def _gt_labels(labels):
    return {p: int(labels[i]) for i, p in enumerate(NIH_PATHOLOGIES)}


def _records_add(records, rep, image_id, eo, manifest_valid):
    """Mirror of cli.records_add but threads the manifest per-class validity mask
    (role-D OpenI masks Consolidation)."""
    for pr in rep.pathologies:
        mp = "{}"
        if eo is not None and pr.pathology in eo.pathologies:
            pi = eo.pathologies.index(pr.pathology)
            d = {}
            for mi, key in enumerate(eo.member_keys):
                if bool(eo.valid[mi, 0, pi].item()):
                    d[key] = round(float(eo.per_member_probs[mi, 0, pi].item()), 4)
            mp = json.dumps(d)
        idx = NIH_PATHOLOGIES.index(pr.pathology)
        records.add(image_id, pr, member_probs=mp, valid=int(manifest_valid[idx]))


@torch.no_grad()
def forward_role(df, ensemble, cfg, device, label="D"):
    """Forward every image in ``df`` and return (Records, n_images)."""
    records = Records()
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            x = load_image_tensor(row.image_path, img_size=cfg.img_size, device=device)
        except Exception as e:
            print(f"  [skip] {row.image_id}: {e}")
            continue
        eo = ensemble.forward_ensemble_full(x)
        ens = eo.per_member_probs[:, 0, :]
        mc = ensemble.forward_mc(x, cfg.mc_samples) if cfg.use_mc_dropout else None
        unc = estimate(ens, mc, pathologies=eo.pathologies)
        rep = assess_image(row.image_id, unc, cfg, gt_labels=_gt_labels(row.labels))
        _records_add(records, rep, row.image_id, eo, row.valid)
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            dt = time.time() - t0
            print(f"  [infer {label}] {i+1}/{len(df)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def records_to_df(records):
    cols = ["image_id", "pathology", "p_bar", "decision", "confidence",
            "epistemic_std", "mutual_info", "gt", "is_confident", "risk_flag",
            "error_type", "valid", "logits", "member_probs"]
    data = {c: getattr(records, c) for c in cols}
    return pd.DataFrame(data)


def main(argv=None):
    p = argparse.ArgumentParser(prog="eval_phase1", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.parquet")
    p.add_argument("--arch-ensemble", default="xrv_nih,convnextv2")
    p.add_argument("--out", default="runs/phase1")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--no-mc", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="cap images per role (smoke test); 0 = all")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(
        device=device,
        mc_samples=args.mc_samples,
        use_mc_dropout=not args.no_mc,
    )

    df = pd.read_parquet(args.manifest)
    dfC = df[df.split_role == "C"].reset_index(drop=True)
    dfD = df[df.split_role == "D"].reset_index(drop=True)
    if args.limit:
        dfC = dfC.iloc[:args.limit]
        dfD = dfD.iloc[:args.limit]
    print(f"[data] role C (cal)={len(dfC)}  role D (eval)={len(dfD)}")

    ensemble = CXREnsemble(args.arch_ensemble.split(","), cfg=cfg)
    print(f"[model] arch ensemble: {ensemble.member_keys} | predictable={len(ensemble.predictable)} "
          f"| has_dropout={ensemble.has_dropout}")

    print("[infer] forwarding role C (calibration)...")
    recC = forward_role(dfC, ensemble, cfg, device, label="C")
    print("[infer] forwarding role D (eval)...")
    recD = forward_role(dfD, ensemble, cfg, device, label="D")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_report(_empty_report(), str(out_dir), records=recD)  # writes per_record.csv (D)

    # Youden on C, evaluate on D (leak-free).
    df_calib = records_to_df(recC)
    df_eval = records_to_df(recD)
    # Drop pathologies with no ground truth in the eval split.
    pos_per = df_eval.groupby("pathology")["gt"].sum()
    unmappable = pos_per[pos_per == 0].index.tolist()
    if unmappable:
        print(f"[eval] excluding pathologies with no ground truth in role D: {unmappable}")
        df_eval = df_eval[~df_eval["pathology"].isin(unmappable)].reset_index(drop=True)
        df_calib = df_calib[~df_calib["pathology"].isin(unmappable)].reset_index(drop=True)

    thresholds = calibrate_thresholds(df_calib)
    rec, meta = build_records(df_eval, thresholds, args.conf_pct, args.unc_pct)
    report = evaluate_records(rec, cfg, n_imgs=len(dfD))
    sel = auroc_at_confidence_levels(rec)

    # Save reports.
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    with open(out_dir / "reanalyze_meta.json", "w") as f:
        json.dump({"calibration": {**meta, "cal_split": "role_C", "eval_split": "role_D"},
                   "confident_error_auroc_vs_selectivity": sel}, f, indent=2)

    _print(report, thresholds, sel, len(dfD))
    return report, sel


def _empty_report():
    """Placeholder EvalReport so save_report writes the per_record.csv; the real
    report is computed from recD below."""
    from cxr_uncertainty.evaluate import EvalReport
    return EvalReport(
        n_images=0, n_records=0, n_confident=0, n_confident_wrong=0,
        confident_error_auroc_std=None, confident_error_auroc_mi=None,
        confident_error_recall=None, false_flag_rate=None, aurc=None,
        coverage_at_lowest_risk=None, ece=0.0, per_pathology={},
        error_type_counts={"unknown": 0}, risk_config={})


def _print(r, thresholds, sel, n_imgs):
    print("\n" + "=" * 70)
    print("PHASE 1 — ARCH ENSEMBLE on role D (leak-free OpenI eval)")
    print("=" * 70)
    print(f"ensemble cal=role C  eval=role D (n_imgs={n_imgs})")
    print(f"records={r.n_records}  confident={r.n_confident}  confident_wrong={r.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {r.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {r.confident_error_auroc_mi}")
    print(f"confident-error recall (risk_flag)    : {r.confident_error_recall}")
    print(f"false-flag rate (confident correct)  : {r.false_flag_rate}")
    print(f"AURC (risk-coverage)                 : {r.aurc}")
    print(f"ECE (calibration of p_bar)           : {r.ece:.4f}")
    print("-" * 70)
    print("selectivity curve (defining property — must be non-decreasing 50→5%):")
    monotonic = True
    prev = -1.0
    for s in sel:
        a = s["auroc_epistemic_std"]
        if a < prev - 1e-9:
            monotonic = False
        prev = a
        print(f"  top {s['conf_top_pct']:2d}%: n={s['n_confident']:5d} wrong={s['n_confident_wrong']:4d}  "
              f"AUROC_std={a}  AUROC_mi={s['auroc_mutual_info']}")
    print("-" * 70)
    print(f"MONOTONIC non-decreasing 50→5%: {monotonic}")
    print("=" * 70)


if __name__ == "__main__":
    main()