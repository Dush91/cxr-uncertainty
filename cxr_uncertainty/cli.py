"""Command-line entrypoint.

Examples:
  # OpenI (labels auto-bundled with torchxrayvision; only PNGs needed on disk):
  python -m cxr_uncertainty.cli --source openi \
      --imgpath data/openi/NLMCXR_png --max-images 300 --out runs/demo

  # NIH ChestX-ray14 (user-supplied CSV + images):
  python -m cxr_uncertainty.cli --source nih \
      --csv data/nih/Data_Entry_2017.csv --imgroot data/nih/images \
      --max-images 300 --out runs/nih
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from .config import (DEFAULT_ENSEMBLE, DEFAULT_ARCH_ENSEMBLE, NIH_PATHOLOGIES,
                      RiskConfig, ENSEMBLE_REGISTRY, ARCH_MEMBER_REGISTRY)
from .data import Sample, load_nih_csv, load_openi, positive_counts
from .evaluate import EvalReport, Records, evaluate, save_report
from .interfaces import get_risk_policy
from .models import CXREnsemble
from .risk import assess_image  # default "threshold" policy (kept for direct use)
from .uncertainty import estimate
from .utils import load_image_tensor


def _build_ensemble(args, cfg: RiskConfig) -> CXREnsemble:
    if args.arch_ensemble:
        members = args.arch_ensemble.split(",")
        registry = ARCH_MEMBER_REGISTRY
        choices = list(ARCH_MEMBER_REGISTRY)
    else:
        members = args.ensemble.split(",") if args.ensemble else DEFAULT_ENSEMBLE
        registry = ENSEMBLE_REGISTRY
        choices = list(ENSEMBLE_REGISTRY)
    for m in members:
        if m not in registry:
            raise SystemExit(f"Unknown ensemble member '{m}'. Choices: {choices}")
    ens = CXREnsemble(members=members, cfg=cfg)
    # Load images at the largest native_size in the ensemble so a 768 member
    # (Ark+) receives true-resolution input; forward_ensemble_full resizes per
    # member. 224-only ensembles keep img_size=224 (byte-identical to baseline).
    cfg.img_size = max(getattr(mb, "native_size", 224) for mb in ens.members)
    return ens


def _load_samples(args) -> List[Sample]:
    if args.source == "openi":
        if not args.imgpath:
            raise SystemExit("--imgpath is required for --source openi")
        return load_openi(args.imgpath, max_images=args.max_images,
                          views=args.views.split(",") if args.views else None)
    elif args.source == "nih":
        if not (args.csv and args.imgroot):
            raise SystemExit("--csv and --imgroot are required for --source nih")
        return load_nih_csv(args.csv, args.imgroot, max_images=args.max_images)
    else:
        raise SystemExit(f"Unknown --source {args.source}")


def run(args) -> EvalReport:
    cfg = RiskConfig(
        decision_thresh=args.decision_thresh,
        conf_thresh=args.conf_thresh,
        unc_thresh=args.unc_thresh,
        mc_samples=args.mc_samples,
        use_mc_dropout=not args.no_mc,
        device="cuda" if torch.cuda.is_available() else "cpu",
        risk_policy=args.risk_policy,
        conformal_sidecar=args.conformal_sidecar or "",
    )
    # Pre-check: the live conformal-triage risk policy (plan §M.2) needs a
    # pre-fit sidecar (T + per-pathology tau_p from scripts/eval_conformal.py
    # --write-sidecar). Without it the policy cannot apply the LAC thresholds.
    conformal_active = (cfg.risk_policy == "conformal_triage")
    if conformal_active and not cfg.conformal_sidecar:
        raise SystemExit(
            "--risk-policy conformal_triage requires --conformal-sidecar "
            "<json> (build it with: python scripts/eval_conformal.py --write-sidecar)")
    # Resolve the risk policy up front so an unknown key fails fast (not mid-loop).
    risk_policy_fn = get_risk_policy(cfg.risk_policy)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(args)
    if not samples:
        raise SystemExit("No samples loaded. Check --imgpath / --csv / --imgroot.")
    print(f"[data] loaded {len(samples)} samples; positives: {positive_counts(samples)}")

    ensemble = _build_ensemble(args, cfg)
    print(f"[model] ensemble: {ensemble.member_keys} | predictable: {len(ensemble.predictable)} "
          f"| has_dropout={ensemble.has_dropout}")
    if cfg.use_mc_dropout and not ensemble.has_dropout:
        print("[model] NOTE: no Dropout layers found in arches; relying on ensemble disagreement only.")
    print(f"[risk] policy={cfg.risk_policy}"
          + (f" sidecar={cfg.conformal_sidecar}" if conformal_active else ""))

    records = Records()
    records.conformal_active = conformal_active
    reports = []
    t0 = time.time()
    for i, s in enumerate(samples):
        try:
            x = load_image_tensor(s.image_path, img_size=cfg.img_size, device=cfg.device)
        except Exception as e:
            print(f"  [skip] {s.image_id}: load failed ({e})")
            continue
        eo = ensemble.forward_ensemble_full(x)        # (M,B,P) probs/logits/valid
        ens = eo.per_member_probs[:, 0, :]           # (M,P) NaN-masked for estimate
        mc = ensemble.forward_mc(x, cfg.mc_samples) if cfg.use_mc_dropout else None
        unc = estimate(ens, mc, pathologies=eo.pathologies)
        # Dispatch through the risk-policy registry (plan §M.2). The default
        # "threshold" resolves to assess_image, so run_demo/Phase-0 stay
        # byte-identical; "conformal_triage" applies the pre-fit T + tau_p to the
        # live p_bar and sets the risk flag to the LAC refer decision.
        rep = risk_policy_fn(s.image_id, unc, cfg, gt_labels=s.labels)
        reports.append(rep)
        records_add(records, rep, s.image_id, eo)
        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            dt = time.time() - t0
            print(f"  [infer] {i+1}/{len(samples)}  ({dt:.1f}s, {dt/(i+1):.2f}s/img)  "
                  f"flagged={sum(r.any_high_risk for r in reports)}")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[eval] saving raw per-record signals (global 0.5 threshold)...")
    report = evaluate(reports, cfg, conformal_active=conformal_active)
    json_path = save_report(report, str(out_dir), records=records)
    print(f"[eval] saved raw signals {json_path}")

    # The raw global-threshold report is degenerate on uncalibrated OOD models
    # (see README). The meaningful analysis uses per-class calibrated thresholds
    # on a held-out split -> runs/reanalyze/ holds the real report + plots.
    print(f"[eval] running calibrated analysis (per-class thresholds, eval split)...")
    try:
        from . import reanalyze
        reanalyze.run(str(out_dir / "per_record.csv"), str(out_dir),
                      conf_pct=args.conf_pct, unc_pct=args.unc_pct)
    except Exception as e:
        import traceback
        print(f"[reanalyze] calibrated analysis failed: {e}")
        traceback.print_exc()
        _print_summary(report)
    return report


def records_add(records: Records, rep, image_id: str, eo=None):
    """Append one PathologyRisk per row; when an EnsembleOutput is available,
    also store the per-member probs as JSON (Part 2 schema member_probs)."""
    for pr in rep.pathologies:
        mp = "{}"
        if eo is not None and pr.pathology in eo.pathologies:
            pi = eo.pathologies.index(pr.pathology)
            d = {}
            for mi, key in enumerate(eo.member_keys):
                if bool(eo.valid[mi, 0, pi].item()):
                    d[key] = round(float(eo.per_member_probs[mi, 0, pi].item()), 4)
            mp = json.dumps(d)
        records.add(image_id, pr, member_probs=mp)


def _print_summary(r: EvalReport):
    print("\n" + "=" * 64)
    print("UNCERTAINTY / RISK-FLAGGING SUMMARY")
    print("=" * 64)
    print(f"images={r.n_images}  records={r.n_records}  "
          f"confident={r.n_confident}  confident_wrong={r.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {r.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {r.confident_error_auroc_mi}")
    print(f"confident-error recall (risk_flag)    : {r.confident_error_recall}")
    print(f"false-flag rate (confident correct)   : {r.false_flag_rate}")
    print(f"AURC (risk-coverage)                  : {r.aurc}")
    print(f"lowest-risk operating point (cov,err) : {r.coverage_at_lowest_risk}")
    print(f"ECE (calibration of p_bar)            : {r.ece:.4f}")
    print("error type counts:", r.error_type_counts)
    print("-" * 64)
    print("per-pathology (n / pos / task_auroc / conf_correct / conf_wrong / flagged):")
    for p, d in r.per_pathology.items():
        print(f"  {p:22s} {d['n']:5d} {d['positives']:4d} {str(d['task_auroc'])[:6]:>6s} "
              f"{d['confident_correct']:5d} {d['confident_wrong']:4d} {d['flagged']:4d}")
    print("=" * 64)


def main(argv=None):
    p = argparse.ArgumentParser(prog="cxr_uncertainty", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["openi", "nih"], default="openi")
    p.add_argument("--imgpath", default=None, help="OpenI PNG dir (NLMCXR_png)")
    p.add_argument("--csv", default=None, help="NIH Data_Entry_2017.csv")
    p.add_argument("--imgroot", default=None, help="NIH image root dir")
    p.add_argument("--views", default="PA,AP")
    p.add_argument("--max-images", type=int, default=300)
    p.add_argument("--ensemble", default=",".join(DEFAULT_ENSEMBLE),
                   help="comma list of legacy xrv ensemble member keys")
    p.add_argument("--arch-ensemble", default=None,
                   help="comma list of arch-primary member keys "
                        "(e.g. xrv_nih,convnextv2); overrides --ensemble")
    p.add_argument("--decision-thresh", type=float, default=0.5)
    p.add_argument("--conf-thresh", type=float, default=0.6)
    p.add_argument("--unc-thresh", type=float, default=0.15)
    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--no-mc", action="store_true", help="disable MC-Dropout")
    p.add_argument("--conf-pct", type=float, default=10.0,
                   help="calibrated analysis: %% of predictions considered confident (top by confidence)")
    p.add_argument("--unc-pct", type=float, default=50.0,
                   help="calibrated analysis: %% of confident predictions flagged high-risk")
    p.add_argument("--risk-policy", default="threshold",
                   choices=["threshold", "conformal_triage"],
                   help="risk policy: threshold [default, the legacy epistemic_std gate] | "
                        "conformal_triage [the live per-image LAC mirror of the batch §L path, "
                        "plan §M.2; needs --conformal-sidecar]")
    p.add_argument("--conformal-sidecar", default="",
                   help="path to a conformal sidecar JSON (T + per-pathology tau_p) fit on "
                        "OpenI-C by `scripts/eval_conformal.py --write-sidecar`; required for "
                        "--risk-policy conformal_triage (plan §M.2)")
    p.add_argument("--out", default="runs/demo")
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()