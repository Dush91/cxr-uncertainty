#!/usr/bin/env python
"""Ensemble-membership comparison: 3-member vs 4-member(+Ark+) vs 5-member.

Question (2026-08-04): Ark+ improves in-distribution disagreement/AURC as a
voting member (5-member: AURC 0.0014 vs 0.0030, ce_auroc_std 0.704 vs 0.631)
but degrades the under-shift flagship (disagreement below chance; Mahalanobis
fraction-caught drops via the more-confident confident set). If deployment is
in-distribution (the user's framing), is it worth adding Ark+ to voting +
re-calibrating on the larger ensemble?

This holds the calibration procedure FIXED (TS-only T on OpenI-C, Youden on
OpenI-C, Mondrian LAC τ_p on OpenI-C, Mahalanobis on RAD-DINO features) and
varies ONLY the voting ensemble, by subsetting the member axis of the existing
5-member arrays (``runs/phase4_features_5mem``) -- no inference. Three variants:

  * 3mem      = [xrv_nih, convnextv2, raddino]            (production)
  * 4mem_ark+ = [xrv_nih, convnextv2, raddino, arkswin]   (Ark+ only added)
  * 5mem      = [xrv_nih, convnextv2, raddino, biomedclip, arkswin]

Reports, per variant, on:
  * openiD (in-distribution, all-valid): AURC, confident errors, ce_auroc by
    std / mi / mahalanobis, ECE, conformal coverage (the guarantee).
  * kermany + covid (under shift, Pneumonia-only): ce_auroc by std / mahalanobis
    + conformal Pneumonia coverage (the honest break).

The §O augmented-Youden and rep-mismatch re-validation are orthogonal and
applied AFTER the ensemble is chosen; this script isolates the membership effect.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/compare_ensembles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig  # noqa: E402
from cxr_uncertainty.production import build_ts_only_mahalanobis_pipeline  # noqa: E402
from cxr_uncertainty.reanalyze import build_records, calibrate_thresholds  # noqa: E402
from cxr_uncertainty.evaluate import evaluate_records  # noqa: E402
from cxr_uncertainty.conformal import (  # noqa: E402
    apply_lac, coverage_by_label, fit_mondrian_lac,
)

PNEU = "Pneumonia"
PNEU_IDX = NIH_PATHOLOGIES.index(PNEU)
ARR = _REPO / "runs" / "phase4_features_5mem"
CONF_PCT, UNC_PCT, ALPHA = 10.0, 50.0, 0.1

# (label, member indices into the 5-member probs axis, member keys)
VARIANTS = [
    ("3mem", [0, 1, 2], ["xrv_nih", "convnextv2", "raddino"]),
    ("4mem_ark+", [0, 1, 2, 4], ["xrv_nih", "convnextv2", "raddino", "arkswin"]),
    ("5mem", [0, 1, 2, 3, 4],
     ["xrv_nih", "convnextv2", "raddino", "biomedclip", "arkswin"]),
]


def _load(site):
    z = np.load(ARR / f"arrays_{site}.npz", allow_pickle=True)
    return {
        "probs": z["probs"], "gt": z["gt"], "valid": z["valid"],
        "feats": z["rad_feats"], "ids": [str(s) for s in z["ids"]],
    }


def _run(C, site, idxs, mkeys, pneumonia_only):
    """Return per-site metrics dict for one ensemble variant."""
    Cprobs = C["probs"][:, idxs, :]
    sprobs = site["probs"][:, idxs, :]
    df_cal, df_eval, _ = build_ts_only_mahalanobis_pipeline(
        Cprobs, C["gt"], C["valid"], C["ids"], C["feats"],
        sprobs, site["gt"], site["valid"], site["ids"], site["feats"],
        list(NIH_PATHOLOGIES), mkeys, PNEU_IDX)
    if pneumonia_only:
        df_cal = df_cal[df_cal.pathology == PNEU].reset_index(drop=True)
        df_eval = df_eval[df_eval.pathology == PNEU].reset_index(drop=True)
    else:
        # drop classes with no eval positives (e.g. Consolidation on OpenI)
        pos = df_eval.groupby("pathology")["gt"].sum()
        unm = pos[pos == 0].index.tolist()
        if unm:
            df_cal = df_cal[~df_cal.pathology.isin(unm)].reset_index(drop=True)
            df_eval = df_eval[~df_eval.pathology.isin(unm)].reset_index(drop=True)
    if df_eval.empty:
        return None

    thr = calibrate_thresholds(df_cal)
    rec, _ = build_records(df_eval, thr, CONF_PCT, UNC_PCT, unc_column="mahalanobis")
    rep = evaluate_records(rec, RiskConfig(device="cpu"),
                           n_imgs=len(set(rec.image_id)))

    # conformal coverage (fit τ_p on cal, apply to eval)
    thr_p, rare, pooled_tau, _ = fit_mondrian_lac(df_cal, list(NIH_PATHOLOGIES), ALPHA)
    df_lac = apply_lac(df_eval, thr_p, rare, pooled_tau)
    cov = coverage_by_label(df_lac, thr_p, rare, pooled_tau)
    ov = cov["overall"] or {}
    return {
        "n_confident": rep.n_confident,
        "n_confident_wrong": rep.n_confident_wrong,
        "aurc": rep.aurc,
        "ece": rep.ece,
        "ce_std": rep.confident_error_auroc_std,
        "ce_mi": rep.confident_error_auroc_mi,
        "ce_maha": rep.confident_error_auroc_mahalanobis,
        "cov": ov.get("coverage"),
        "refer": ov.get("refer_rate"),
    }


def _f(x, p=4):
    return "--" if x is None else f"{x:.{p}f}"


def main():
    C = _load("openiC")
    sites = [
        ("openiD  (in-distribution, all-valid)", _load("openiD"), False),
        ("kermany (under shift, Pneumonia)",      _load("kermany"), True),
        ("covid   (under shift, Pneumonia)",      _load("covid"), True),
    ]
    for title, site, pneu in sites:
        print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
        print(f"{'variant':12s} {'n_conf':>7s} {'wrong':>6s} {'AURC':>9s} {'ECE':>7s} "
              f"{'ce_std':>8s} {'ce_mi':>8s} {'ce_maha':>9s} {'cov':>7s} {'refer':>7s}")
        for label, idxs, mkeys in VARIANTS:
            m = _run(C, site, idxs, mkeys, pneu)
            if m is None:
                print(f"{label:12s}  (empty)"); continue
            print(f"{label:12s} {m['n_confident']:7d} {m['n_confident_wrong']:6d} "
                  f"{_f(m['aurc']):>9s} {_f(m['ece'], 4):>7s} "
                  f"{_f(m['ce_std']):>8s} {_f(m['ce_mi']):>8s} {_f(m['ce_maha']):>9s} "
                  f"{_f(m['cov'], 3):>7s} {_f(m['refer'], 3):>7s}")
        print("=" * 92)
    print("\nce_* = confident-error AUROC (top-10% confident set); cov = conformal")
    print("coverage (1-alpha=0.9 in-distribution guarantee); refer = LAC refer rate.")
    print("Mahalanobis fit is member-independent (RAD-DINO feats); T/Youden/tau_p")
    print("re-fit per variant on OpenI-C. No inference -- arrays subsetting only.")


if __name__ == "__main__":
    main()