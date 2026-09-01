"""Baseline comparison for the publishable uncertainty claim.

Computes the full baseline suite from saved phase-4-style feature arrays
WITHOUT re-running inference, and now runs it across MULTIPLE leak-free eval
datasets (OpenI + the powered post-foundation sets ReXGradient-160K /
CANDID-III built by ``scripts/build_eval_arrays.py``).

Each dataset is a (cal_npz, eval_npz) pair carrying per-member probabilities
``probs (N, M, 14)`` with ``member_keys`` (the ensemble), plus RAD-DINO
features, energy, and Mahalanobis. The arrays schema (producer:
``eval_phase4_features.py`` / ``build_eval_arrays.py``) is::

    probs (N,M,P), gt (N,P), valid (N,P), rad_logits (N,14),
    rad_feats (N,1536), energy (N,), mahalanobis (N,), ids (N,),
    member_keys (M,), pathologies (P,)

Baselines compared (uncertainty_evaluation.md §2.1):
  * single-member confidence (MSP) -- each member's own confidence vs its own
    errors at matched coverage (the matched-backbone single-model baseline);
  * pooled confidence (MSP) and predictive entropy H[pbar] -- the null the
    disagreement signal must beat;
  * epistemic_std / mutual_info -- the method (ensemble disagreement);
  * TS-calibrated confidence -- the calibration-only baseline (rank-invariant,
    Corbiere et al. 2019);
  * Mahalanobis / energy / kNN -- feature-space OOD baselines.

Statistical rigor (§2.3): DeLong tests, McNemar on flags, bootstrap CI.

Baur UNSURE@MICCAI 2025 tasks:
  * Task 3 -- per-pathology correctness-prediction AUROC (``_per_pathology_...``).
  * Task 4 -- AUAC (``_auac``).
  * Task 2 -- uncertainty-label prediction (``_task2_uncertain_auroc``): AUROC
    of each uncertainty score vs the expert "uncertain" (-1) label. Only
    meaningful on datasets that ship native uncertain labels (CANDID-III);
    OpenI labels are binary so Task 2 is skipped there. -1 rows are filtered
    out of the binary decision path (which is structurally {0,1}) and consumed
    only by Task 2.

Self-check: on the OpenI dataset the script reproduces the phase-4 4mem report
(n_confident=2616, n_confident_wrong=5, std AUROC ~0.6072, Mahalanobis ~0.8698).
The EXPECTED anchors are OpenI-specific and skipped for other datasets.

Usage (single OpenI dataset, back-compat):
    PYTHONPATH=. python scripts/eval_baselines.py \
        --arrays runs/phase4_features_4mem --out runs/baselines

Usage (multi-dataset; one ``name:cal_npz:eval_npz`` triple per dataset):
    PYTHONPATH=. python scripts/eval_baselines.py --out runs/baselines \
        --datasets openi:runs/phase4_features_4mem/arrays_openiC.npz:runs/phase4_features_4mem/arrays_openiD.npz \
                   rex:runs/eval_arrays/rex/arrays_rexcal.npz:runs/eval_arrays/rex/arrays_rexeval.npz \
                   candidiii:runs/eval_arrays/candidiii/arrays_candidiiical.npz:runs/eval_arrays/candidiii/arrays_candidiiieval.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from cxr_uncertainty.calibration import apply_temperature
from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig
from cxr_uncertainty.evaluate import (_auroc, _ece, _e_aurc, _risk_coverage,
                                      _risk_coverage_curve, _trapz,
                                      evaluate_records)
from cxr_uncertainty.reanalyze import (_confidence, _knn_distances,
                                       _youden_threshold, build_records,
                                       calibrate_thresholds)
from cxr_uncertainty.statistics import (bootstrap_ci, bootstrap_paired_diff,
                                        delong_test, mcnemar_test)

SELECTIVITY = [50, 25, 15, 10, 5]
KNN_K = 5
# phase-4 4mem self-check anchors (reproduced from the saved OpenI arrays).
# OpenI-specific; only enforced for the dataset named "openi".
EXPECTED = {"n_confident": 2616, "n_confident_wrong": 5,
            "std_auroc_top10": 0.6072, "maha_auroc_top10": 0.8698}


def _entropy(p: np.ndarray) -> np.ndarray:
    """Binary predictive entropy H[p] (nats)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _load_arrays(path: Path) -> dict:
    a = np.load(path, allow_pickle=True)
    probs, gt, valid = a["probs"], a["gt"], a["valid"]
    v3 = np.broadcast_to(valid[:, None, :], probs.shape)
    m = np.where(v3, probs, np.nan)
    pbar = np.nanmean(m, axis=1)
    std = np.nanstd(m, axis=1)
    mi = _entropy(pbar) - np.nanmean(_entropy(np.clip(m, 1e-12, 1 - 1e-12)), axis=1)
    return dict(probs=probs, gt=gt, valid=valid, pbar=pbar, std=std, mi=mi,
                ids=a["ids"], maha=a["mahalanobis"], energy=a["energy"],
                rad_feats=a["rad_feats"], member_keys=a.get("member_keys"))


def _to_long(x: dict, member_keys: list) -> pd.DataFrame:
    """Long-form (image x pathology) frame with per-member prob columns."""
    n_mem = x["probs"].shape[1]
    rows = []
    for i in range(x["gt"].shape[0]):
        for p in range(14):
            if not x["valid"][i, p]:
                continue
            row = [str(x["ids"][i]), NIH_PATHOLOGIES[p], x["pbar"][i, p],
                   int(x["gt"][i, p]), x["std"][i, p], x["mi"][i, p],
                   float(x["maha"][i])]
            row += [float(x["probs"][i, mm, p]) for mm in range(n_mem)]
            rows.append(row)
    cols = (["image_id", "pathology", "p_bar", "gt", "epistemic_std",
             "mutual_info", "mahalanobis"]
            + [f"prob_{k}" for k in member_keys])
    return pd.DataFrame(rows, columns=cols)


def _member_arrays(df: pd.DataFrame, thr: dict, col: str):
    """Member m's decision / confidence / errors under its own thresholds."""
    t = df["pathology"].map(thr).fillna(0.5).values.astype(float)
    prob = df[col].values.astype(float)
    decision = (prob >= t).astype(int)
    conf = _confidence(prob, t)
    wrong = (decision != df["gt"].values.astype(int)).astype(int)
    return conf, wrong


def _selectivity_auroc(conf: np.ndarray, wrong: np.ndarray, scores: dict,
                       pcts=SELECTIVITY) -> list:
    """Confident-error AUROC of each score within the top-pct% confident set."""
    out = []
    for pct in pcts:
        if len(conf) == 0:
            continue
        cut = float(np.percentile(conf, 100 - pct))
        m = conf >= cut
        if (m & (wrong == 1)).sum() == 0 or (m & (wrong == 0)).sum() == 0:
            continue
        row = {"top_pct": pct, "n": int(m.sum()), "n_wrong": int((m & (wrong == 1)).sum())}
        for name, s in scores.items():
            mm = m & np.isfinite(s)
            if (mm & (wrong == 1)).sum() > 0 and (mm & (wrong == 0)).sum() > 0:
                row[name] = round(float(roc_auc_score(wrong[mm], s[mm])), 4)
            else:
                row[name] = None
        out.append(row)
    return out


def _aurc_eaurc(score: np.ndarray, wrong: np.ndarray):
    aurc, _ = _risk_coverage(score, wrong)
    return aurc, _e_aurc(aurc, float(wrong.mean()))


def _auac(score: np.ndarray, wrong: np.ndarray):
    """Area under the accuracy-coverage curve (Baur UNSURE Task 4).

    Higher is better (accuracy of the retained set vs coverage). Complement of
    AURC on the same risk-coverage curve: ``AUAC = trapz(1 - risk, coverage)``.
    Returns None when the curve is degenerate."""
    curve = _risk_coverage_curve(score, wrong)
    if curve is None:
        return None
    cv, rk = curve
    return float(_trapz(1.0 - np.asarray(rk), np.asarray(cv)))


def _per_pathology_correctness_auroc(records, scores: dict) -> dict:
    """Per-pathology correctness-prediction AUROC (Baur UNSURE Task 3).

    For each pathology, AUROC of each uncertainty score vs the ensemble's
    wrong/correct decision on the FULL eval set (not just the confident set) --
    the error-detection metric the CXR benchmark reports. Returns
    ``{"per_pathology": {path: {score: auroc or None}}, "macro": {score: auroc}}``
    where ``macro`` is the simple mean over pathologies with a finite AUROC."""
    a = records.arrays()
    wrong = a["wrong"]
    paths = sorted(set(records.pathology))
    per_pathology = {}
    for p in paths:
        m = np.array([q == p for q in records.pathology])
        row = {}
        for name, s in scores.items():
            mm = m & np.isfinite(s)
            if (mm & (wrong == 1)).sum() > 0 and (mm & (wrong == 0)).sum() > 0:
                row[name] = round(float(roc_auc_score(wrong[mm], s[mm])), 4)
            else:
                row[name] = None
        per_pathology[p] = {"n": int(m.sum()),
                            "n_wrong": int((m & (wrong == 1)).sum()),
                            "auroc": row}
    macro = {}
    for name in scores:
        vals = [per_pathology[p]["auroc"][name] for p in per_pathology]
        vals = [v for v in vals if v is not None]
        macro[name] = round(float(np.mean(vals)), 4) if vals else None
    return {"per_pathology": per_pathology, "macro": macro}


def _task2_uncertain_auroc(df: pd.DataFrame, thr: dict, zp: dict,
                          hybrid_w: float, D: dict) -> dict:
    """Baur UNSURE Task 2: uncertainty-label prediction.

    For each pathology, AUROC of each uncertainty score vs the expert
    "uncertain" (-1) label (positive class = uncertain), macro-averaged over
    pathologies with >=1 uncertain AND >=1 certain row. Scores are oriented
    HIGH = more uncertain (matching the binary HIGH = likely wrong convention),
    so AUROC > 0.5 means the score detects expert-marked uncertainty.

    This is a STANDALONE post-hoc metric: it consumes only the raw ``gt`` column
    (which preserves -1) and the per-row scores -- it does NOT touch the binary
    decision path (``build_records`` / ``wrong`` / ``classify_error``), which is
    structurally {0,1} and would mis-handle -1. ``df`` is the FULL eval frame
    (uncertain rows retained); the binary pipeline runs on a -1-filtered frame.
    """
    t = df["pathology"].map(thr).fillna(0.5).values.astype(float)
    pbar = df["p_bar"].values.astype(float)
    scores = {
        "epistemic_std": np.asarray(df["epistemic_std"].values, dtype=float),
        "mutual_info": np.asarray(df["mutual_info"].values, dtype=float),
        "entropy": _entropy(pbar),
        "confidence": -_confidence(pbar, t),     # HIGH = low confidence = uncertain
        "mahalanobis": np.asarray(df["mahalanobis"].values, dtype=float),
        "knn": np.asarray(df["knn_dist"].values, dtype=float),
    }
    # image-level energy -> long-form rows (stored energy: HIGH = OOD = uncertain).
    id_energy = dict(zip([str(i) for i in D["ids"]], D["energy"]))
    scores["energy"] = np.array([id_energy.get(i, np.nan) for i in df["image_id"]],
                                dtype=float)
    # hybrid disagreement + feature-density (z-params + weight from the cal fit).
    z_std = (scores["epistemic_std"] - zp["mu_std"]) / zp["sg_std"]
    z_knn = (scores["knn"] - zp["mu_knn"]) / zp["sg_knn"]
    z_maha = (scores["mahalanobis"] - zp["mu_maha"]) / zp["sg_maha"]
    scores["hybrid_std_knn"] = hybrid_w * z_std + (1 - hybrid_w) * z_knn
    scores["hybrid_std_maha"] = hybrid_w * z_std + (1 - hybrid_w) * z_maha

    gt = df["gt"].values.astype(int)
    uncertain = (gt == -1)
    paths = sorted(set(df["pathology"]))
    per_pathology = {}
    for p in paths:
        m = np.array([q == p for q in df["pathology"]])
        row = {}
        for name, s in scores.items():
            mm = m & np.isfinite(s)
            pos = int((mm & uncertain).sum())
            neg = int((mm & ~uncertain).sum())
            if pos > 0 and neg > 0:
                row[name] = round(float(roc_auc_score(uncertain[mm], s[mm])), 4)
            else:
                row[name] = None
        per_pathology[p] = {"n": int(m.sum()), "n_uncertain": int((m & uncertain).sum()),
                            "auroc": row}
    macro = {}
    for name in scores:
        vals = [per_pathology[p]["auroc"][name] for p in per_pathology]
        vals = [v for v in vals if v is not None]
        macro[name] = round(float(np.mean(vals)), 4) if vals else None
    return {"per_pathology": per_pathology, "macro": macro,
            "n_uncertain_total": int(uncertain.sum())}


def _fit_temperature(p_cal: np.ndarray, y_cal: np.ndarray,
                     lo: float = 0.1, hi: float = 10.0) -> float:
    """Global temperature on the pooled mean, fit by NLL on the cal split."""
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        p = np.clip(apply_temperature(p_cal, T), 1e-12, 1 - 1e-12)
        return float(-np.mean(y_cal * np.log(p) + (1 - y_cal) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def _fmt(v) -> str:
    return "  --" if v is None else f"{v:.4f}"


def _parse_datasets(args) -> list:
    """Build the [(name, cal_path, eval_path, do_selfcheck)] list.
    ``--datasets`` is a list of ``name:cal_npz:eval_npz`` triples; absent that,
    fall back to the single OpenI pair under ``--arrays`` (back-compat, and the
    only dataset for which the EXPECTED self-check anchors hold)."""
    out = []
    if args.datasets:
        for spec in args.datasets:
            parts = spec.split(":")
            if len(parts) != 3:
                    raise SystemExit(f"--datasets expects name:cal_npz:eval_npz, got {spec!r}")
            name, cal, ev = parts
            out.append((name, Path(cal), Path(ev), name == "openi"))
    else:
        base = Path(args.arrays)
        out.append(("openi", base / "arrays_openiC.npz",
                    base / "arrays_openiD.npz", True))
    return out


def run_dataset(name: str, C: dict, D: dict, member_keys: list, args,
                do_selfcheck: bool) -> None:
    """Run the full baseline suite for one (cal, eval) array pair and write
    ``<out>/<name>/{baseline_comparison.json, baseline_table.md, plot}``."""
    n_mem = len(member_keys)
    print(f"\n{'='*78}\n[{name}] members={member_keys}  cal={C['gt'].shape[0]}  "
          f"eval={D['gt'].shape[0]}\n{'='*78}")

    df_cal = _to_long(C, member_keys)
    df_eval = _to_long(D, member_keys)

    # kNN feature-distance baseline (Sun et al. 2022): eval -> cal features.
    # Set on the FULL eval frame before any -1 filtering so Task 2 (which runs
    # on the full frame) also has knn_dist.
    knn_img = _knn_distances(np.asarray(D["rad_feats"], dtype=np.float64),
                             np.asarray(C["rad_feats"], dtype=np.float64), k=KNN_K)
    id_to_knn = dict(zip([str(i) for i in D["ids"]], knn_img))
    df_eval["knn_dist"] = [id_to_knn.get(i, 0.0) for i in df_eval["image_id"]]
    df_eval_full = df_eval

    # --- -1 (expert-"uncertain") handling ------------------------------------
    # The binary decision path (build_records / wrong / classify_error) is
    # structurally {0,1}: with gt=-1, decision != gt is always True, so every
    # uncertain row would be scored wrong=1 and pollute the binary metrics.
    # Filter -1 out of the binary cal/eval; retain the full frame for Task 2.
    has_uncertain = bool((df_eval["gt"] < 0).any() or (df_cal["gt"] < 0).any())
    if has_uncertain:
        n_full = len(df_eval)
        df_cal = df_cal[df_cal["gt"] >= 0].reset_index(drop=True)
        df_eval = df_eval[df_eval["gt"] >= 0].reset_index(drop=True)
        print(f"[{name}] -1 uncertain labels present: binary metrics on "
              f"{len(df_eval)} certain-label eval rows; Task 2 on {n_full} full rows")

    # --- ensemble path (the method) -----------------------------------------
    thr = calibrate_thresholds(df_cal)
    rec, meta = build_records(df_eval, thr, args.conf_pct, args.unc_pct)
    rep = evaluate_records(rec, RiskConfig(device="cpu"),
                           n_imgs=len(set(rec.image_id)))
    a = rec.arrays()
    conf, wrong = a["confidence"], a["wrong"]
    entropy = _entropy(a["p_bar"])
    # image-level energy -> long-form rows (stored energy: HIGH = OOD).
    id_energy = dict(zip([str(i) for i in D["ids"]], D["energy"]))
    energy = np.array([id_energy.get(i, np.nan) for i in rec.image_id], dtype=float)

    # --- hybrid disagreement + feature-density (the strengthened method) -----
    # The confident-error cases are the ones where the members AGREE but are
    # jointly wrong (shared pretraining bias). Disagreement (epistemic_std) is
    # structurally blind to them; feature-density (kNN distance to the cal
    # manifold) catches them. The hybrid is a weighted z-sum of the two, with the
    # z-transform fit on the CAL rows only (label-free) and the weight selected on
    # the CAL split at the powered top-50% operating point -- so nothing in the
    # score is tuned against eval labels.
    def _zparams(samples: np.ndarray):
        s = np.asarray(samples, dtype=float)
        s = s[np.isfinite(s)]
        return float(np.mean(s)), float(np.std(s) + 1e-12)

    def _z(x, mu, sg):
        return (np.asarray(x, dtype=float) - mu) / sg

    mu_std, sg_std = _zparams(df_cal["epistemic_std"].values)
    mu_mi, sg_mi = _zparams(df_cal["mutual_info"].values)
    mu_maha, sg_maha = _zparams(df_cal["mahalanobis"].values)
    # cal-manifold kNN with leave-one-out (self-neighbor excluded) for the
    # z-params; the eval rows already have knn_dist against this manifold.
    nn_loo = NearestNeighbors(n_neighbors=KNN_K + 1)
    nn_loo.fit(np.asarray(C["rad_feats"], dtype=np.float64))
    d_cal, _ = nn_loo.kneighbors(np.asarray(C["rad_feats"], dtype=np.float64))
    knn_cal_img = d_cal[:, 1:].mean(axis=1)
    mu_knn, sg_knn = _zparams(knn_cal_img)
    zp = {"mu_std": mu_std, "sg_std": sg_std, "mu_mi": sg_mi, "sg_mi": sg_mi,
          "mu_maha": mu_maha, "sg_maha": sg_maha, "mu_knn": mu_knn, "sg_knn": sg_knn}

    z_std = _z(a["epistemic_std"], mu_std, sg_std)
    z_mi = _z(a["mutual_info"], mu_mi, sg_mi)
    z_maha = _z(a["mahalanobis"], mu_maha, sg_maha)
    z_knn = _z(a["knn_dist"], mu_knn, sg_knn)

    # weight selection on the CAL split (top-50% confident-error AUROC).
    cal_rec, _ = build_records(df_cal, thr, 50.0, args.unc_pct)
    ac = cal_rec.arrays()
    id_to_knn_cal = dict(zip([str(i) for i in C["ids"]], knn_cal_img))
    knn_cal_rows = np.array(
        [id_to_knn_cal.get(str(i), np.nan) for i in cal_rec.image_id], dtype=float)
    z_std_cal = _z(ac["epistemic_std"], mu_std, sg_std)
    z_knn_cal = _z(knn_cal_rows, mu_knn, sg_knn)
    conf_c, wrong_c = ac["confidence"], ac["wrong"]
    m50c = conf_c >= float(np.percentile(conf_c, 50.0))

    HYBRID_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_w, best_auc = None, -1.0
    for w in HYBRID_WEIGHTS:
        h_cal = w * z_std_cal + (1 - w) * z_knn_cal
        auc_c = _auroc(h_cal[m50c], wrong_c[m50c])
        if auc_c is not None and auc_c > best_auc:
            best_w, best_auc = w, auc_c
    hybrid_std_knn = best_w * z_std + (1 - best_w) * z_knn
    hybrid_std_maha = best_w * z_std + (1 - best_w) * z_maha
    hybrid_meta = {"weight_on_std": best_w,
                   "cal_selection_auroc_top50": round(best_auc, 4) if best_auc else None,
                   "grid_auroc_top50_cal": {str(w): round(
                       _auroc((w * z_std_cal + (1 - w) * z_knn_cal)[m50c], wrong_c[m50c]) or -1, 4)
                       for w in HYBRID_WEIGHTS}}
    print(f"[{name}] hybrid weight_on_std={best_w} (selected on cal, top-50% AUROC={best_auc:.4f})")

    # --- self-check: reproduce the phase-4 4mem report (OpenI only) ----------
    cut10 = float(np.percentile(conf, 90.0))
    m10 = conf >= cut10
    std10 = _auroc(a["epistemic_std"][m10], wrong[m10])
    mm10 = m10 & np.isfinite(a["mahalanobis"])
    maha10 = _auroc(a["mahalanobis"][mm10], wrong[mm10])
    checks = {"n_confident": rep.n_confident, "n_confident_wrong": rep.n_confident_wrong,
              "std_auroc_top10": round(std10, 4), "maha_auroc_top10": round(maha10, 4)}
    print(f"[{name}] self-check {checks}")
    if do_selfcheck:
        for k, v in EXPECTED.items():
            got = checks[k]
            ok = (got == v) if isinstance(v, int) else abs(got - v) < 0.01
            print(f"  {k:20s} expected={v}  got={got}  {'OK' if ok else 'MISMATCH'}")
    else:
        print(f"[{name}] (EXPECTED anchors are OpenI-specific; skipped for {name})")

    # --- Panel A: single-member self-detection at matched coverage ----------
    panel_a = {}
    for k in member_keys:
        thr_m = calibrate_thresholds(
            df_cal[["pathology", "gt"]].assign(p_bar=df_cal[f"prob_{k}"]))
        c_m, w_m = _member_arrays(df_eval, thr_m, f"prob_{k}")
        panel_a[k] = _selectivity_auroc(c_m, w_m, {k: c_m})
    # transpose to rows keyed by top_pct with one column per member.
    panel_a_rows = []
    by_pct = {k: {r["top_pct"]: r for r in rows} for k, rows in panel_a.items()}
    for r in panel_a[member_keys[0]]:
        row = {"top_pct": r["top_pct"], "n": r["n"], "n_wrong": r["n_wrong"]}
        for k in member_keys:
            row[k] = by_pct[k].get(r["top_pct"], {}).get(k)
        panel_a_rows.append(row)

    # --- Panel B: all scores within the ensemble confident set --------------
    scores_b = {
        "epistemic_std": a["epistemic_std"],
        "mutual_info": a["mutual_info"],
        "entropy": entropy,
        "confidence": -conf,          # lower confidence = more likely wrong
        "mahalanobis": a["mahalanobis"],
        "knn": a["knn_dist"],
        "energy": energy,             # stored energy HIGH=OOD = more likely wrong
    }
    for k in member_keys:
        thr_m = calibrate_thresholds(
            df_cal[["pathology", "gt"]].assign(p_bar=df_cal[f"prob_{k}"]))
        c_m, _ = _member_arrays(df_eval, thr_m, f"prob_{k}")
        scores_b[f"conf_{k}"] = -c_m
    scores_b["hybrid_std_knn"] = hybrid_std_knn
    scores_b["hybrid_std_maha"] = hybrid_std_maha
    panel_b = _selectivity_auroc(conf, wrong, scores_b)

    # --- E-AURC (risk-coverage) on the full eval set ------------------------
    aurc_scores = {
        "epistemic_std": a["epistemic_std"],
        "mutual_info": a["mutual_info"],
        "entropy": entropy,
        "confidence": -conf,
        "mahalanobis": a["mahalanobis"],
        "knn": a["knn_dist"],
        "energy": energy,
    }
    for k in member_keys:
        thr_m = calibrate_thresholds(
            df_cal[["pathology", "gt"]].assign(p_bar=df_cal[f"prob_{k}"]))
        c_m, _ = _member_arrays(df_eval, thr_m, f"prob_{k}")
        aurc_scores[f"conf_{k}"] = -c_m
    aurc_scores["hybrid_std_knn"] = hybrid_std_knn
    aurc_scores["hybrid_std_maha"] = hybrid_std_maha
    aurc_table = {}
    for sname, s in aurc_scores.items():
        mm = np.isfinite(s)
        aurc, e_aurc = _aurc_eaurc(s[mm], wrong[mm])
        auac = _auac(s[mm], wrong[mm])
        aurc_table[sname] = {"aurc": round(aurc, 4) if aurc is not None else None,
                            "e_aurc": round(e_aurc, 4) if e_aurc is not None else None,
                            "auac": round(auac, 4) if auac is not None else None,
                            "error_rate": round(float(wrong[mm].mean()), 4)}

    # --- per-pathology correctness AUROC (Baur UNSURE Task 3) ---------------
    # Error-detection AUROC per pathology on the FULL eval set (not just the
    # confident set), macro-averaged across the 14 pathologies -- the metric the
    # CXR benchmark reports. Scores are aligned to the Records rows.
    per_path_scores = {
        "epistemic_std": a["epistemic_std"],
        "mutual_info": a["mutual_info"],
        "entropy": entropy,
        "confidence": -conf,
        "mahalanobis": a["mahalanobis"],
        "knn": a["knn_dist"],
        "hybrid_std_knn": hybrid_std_knn,
        "hybrid_std_maha": hybrid_std_maha,
    }
    per_path = _per_pathology_correctness_auroc(rec, per_path_scores)
    print(f"[{name}] per-pathology macro correctness AUROC: "
          + ", ".join(f"{k}={v}" for k, v in per_path["macro"].items()))

    # --- Baur Task 2: uncertainty-label prediction (datasets with -1) --------
    task2 = None
    if has_uncertain:
        task2 = _task2_uncertain_auroc(df_eval_full, thr, zp, best_w, D)
        print(f"[{name}] Task 2 (uncertainty-label prediction) macro AUROC: "
              + ", ".join(f"{k}={v}" for k, v in task2["macro"].items()))

    # --- TS-calibrated confidence (calibration-only baseline) ---------------
    T = _fit_temperature(df_cal["p_bar"].values, df_cal["gt"].values)
    ts_pbar = apply_temperature(df_eval["p_bar"].values, T)
    ece_raw = _ece(df_eval["p_bar"].values, df_eval["gt"].values, n_bins=10)
    ece_ts = _ece(ts_pbar, df_eval["gt"].values, n_bins=10)
    # Rank invariance (Corbiere et al. 2019): TS is a monotone transform of the
    # pooled probability, so it cannot change the pbar ranking -> confident-error
    # AUROC of -pbar is unchanged to numerical precision. (The piecewise
    # confidence score is only approximately invariant because re-fitting the
    # Youden threshold on TS-scaled data is a data-dependent choice.)
    auroc_pbar_raw = _auroc(-df_eval["p_bar"].values[m10], wrong[m10])
    auroc_pbar_ts = _auroc(-ts_pbar[m10], wrong[m10])
    ts_facts = {"temperature": round(T, 4), "ece_raw": round(ece_raw, 4),
                "ece_ts": round(ece_ts, 4),
                "auroc_pbar_raw_top10": round(auroc_pbar_raw, 4),
                "auroc_pbar_ts_top10": round(auroc_pbar_ts, 4),
                "max_abs_auroc_diff": round(abs(auroc_pbar_raw - auroc_pbar_ts), 6)}

    # --- statistical rigor ----------------------------------------------------
    # DeLong on the FULL eval set (well-powered: ~1100 errors) and within the
    # ensemble confident set at top-50% (75 errors) and top-10% (5 errors; the
    # project's defining metric, honestly underpowered). The "best member" is
    # the member whose own confidence AUROC is highest at that level.
    def _delong_row(sa, sb, w):
        mm = np.isfinite(sa) & np.isfinite(sb)
        auc_a, auc_b, z, pv = delong_test(sa[mm], sb[mm], w[mm])
        return {"auc_a": round(auc_a, 4) if auc_a is not None else None,
                "auc_b": round(auc_b, 4) if auc_b is not None else None,
                "z": round(z, 4) if z is not None else None,
                "p": round(pv, 4) if pv is not None else None}

    def _best_member_at(mask):
        best, best_auc = None, -1.0
        for k in member_keys:
            s = scores_b[f"conf_{k}"][mask]
            w = wrong[mask]
            mm = np.isfinite(s)
            auc = _auroc(s[mm], w[mm])
            if auc is not None and auc > best_auc:
                best, best_auc = k, auc
        return best

    tests = {}
    full_mask = np.ones(len(wrong), dtype=bool)
    for level, mask in [("full", full_mask), ("conf_top50", conf >= float(np.percentile(conf, 50))),
                        ("conf_top10", m10)]:
        bm = _best_member_at(mask)
        tests[f"delong_{level}_epistemic_vs_confidence"] = _delong_row(
            a["epistemic_std"][mask], scores_b["confidence"][mask], wrong[mask])
        if bm is not None:
            tests[f"delong_{level}_epistemic_vs_best_member_{bm}"] = _delong_row(
                a["epistemic_std"][mask], scores_b[f"conf_{bm}"][mask], wrong[mask])
        tests[f"delong_{level}_epistemic_vs_mahalanobis"] = _delong_row(
            a["epistemic_std"][mask], scores_b["mahalanobis"][mask], wrong[mask])
    # bootstrap CI on the epistemic_std confident-error AUROC (top-50% / top-10%).
    for level, mask in [("conf_top50", conf >= float(np.percentile(conf, 50))),
                        ("conf_top10", m10)]:
        vals = np.column_stack([a["epistemic_std"][mask], wrong[mask]])
        lo, hi = bootstrap_ci(vals, lambda v: _auroc(v[:, 0], v[:, 1]),
                              n_boot=500, seed=args.seed)
        tests[f"bootstrap_epistemic_std_auroc_ci95_{level}"] = (
            [round(lo, 4), round(hi, 4)] if lo is not None else None)
    # McNemar on flags: epistemic_std flag vs confidence flag.
    unc_cut = float(np.percentile(a["epistemic_std"][a["is_confident"]],
                                  100.0 - args.unc_pct))
    flag_ep = a["is_confident"] & (a["epistemic_std"] >= unc_cut)
    conf_cut = float(np.percentile(conf[a["is_confident"]], args.unc_pct))
    flag_conf = a["is_confident"] & (conf <= conf_cut)
    chi2, pv = mcnemar_test(flag_ep, flag_conf, wrong)
    tests["mcnemar_epistemic_vs_confidence"] = (
        {"chi2": round(chi2, 4), "p": round(pv, 4)} if chi2 is not None else None)

    # --- hybrid significance: bootstrap paired-difference CIs ----------------
    # DeLong needs a well-conditioned variance, which the tiny confident-error
    # populations don't provide (returns None). The paired bootstrap does not:
    # it resamples rows jointly and reports the CI of the AUROC difference, so
    # "is the hybrid significantly better than baseline X" is answerable even
    # with 5-75 confident errors.
    def _safe_auroc(s, y):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=int)
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            return None
        try:
            return float(roc_auc_score(y, s))
        except ValueError:
            return None

    def _safe_eaurc(s, y):
        mm = np.isfinite(np.asarray(s, dtype=float))
        if mm.sum() == 0:
            return None
        _, e = _aurc_eaurc(np.asarray(s, dtype=float)[mm], np.asarray(y)[mm])
        return e

    tests["hybrid"] = hybrid_meta
    hybrid_comps = {"epistemic_std": a["epistemic_std"], "knn": a["knn_dist"],
                    "mahalanobis": a["mahalanobis"], "entropy": entropy,
                    "confidence": -conf}
    for level, mask in [("conf_top50", conf >= float(np.percentile(conf, 50))),
                        ("conf_top10", m10)]:
        for sname, sb in hybrid_comps.items():
            mm = mask & np.isfinite(hybrid_std_knn) & np.isfinite(sb)
            lo, hi = bootstrap_paired_diff(
                _safe_auroc, hybrid_std_knn[mm], sb[mm], wrong[mm],
                n_boot=500, seed=args.seed)
            tests[f"boot_paired_hybrid_vs_{sname}_auroc_{level}"] = (
                [round(lo, 4), round(hi, 4)] if lo is not None else None)
    # E-AURC difference on the full eval set (hybrid vs confidence, the winner
    # there): a negative CI means the hybrid is significantly WORSE at full-set
    # risk coverage -- reported honestly.
    mmf = np.isfinite(hybrid_std_knn) & np.isfinite(scores_b["confidence"])
    lo, hi = bootstrap_paired_diff(
        _safe_eaurc, hybrid_std_knn[mmf], scores_b["confidence"][mmf], wrong[mmf],
        n_boot=500, seed=args.seed)
    tests["boot_paired_hybrid_vs_confidence_eaurc_full"] = (
        [round(lo, 4), round(hi, 4)] if lo is not None else None)

    # --- outputs -------------------------------------------------------------
    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": name,
        "ensemble": {"member_keys": member_keys, "n_cal": int(C["gt"].shape[0]),
                     "n_eval": int(D["gt"].shape[0]),
                     "n_eval_certain_rows": int(len(df_eval)),
                     "n_records": rep.n_records, "n_confident": rep.n_confident,
                     "n_confident_wrong": rep.n_confident_wrong,
                     "conf_pct": args.conf_pct, "unc_pct": args.unc_pct,
                     "has_uncertain_labels": has_uncertain},
        "self_check": checks,
        "panel_a_single_member_self_detection": panel_a_rows,
        "panel_b_within_ensemble_confident_set": panel_b,
        "aurc_e_aurc_full_eval": aurc_table,
        "per_pathology_correctness_auroc": per_path,
        "ts_calibrated_confidence": ts_facts,
        "statistical_tests": tests,
    }
    if task2 is not None:
        result["task2_uncertain_label_auroc"] = task2
    with open(out / "baseline_comparison.json", "w") as f:
        json.dump(result, f, indent=2)

    # --- markdown table ------------------------------------------------------
    # Methods-as-rows: one row per method, one column per selectivity level,
    # best AUROC bolded per column, a Δ-vs-best column at the headline top-10%
    # level, and significance daggers where the paired bootstrap CI of the
    # AUROC difference vs hybrid_std_knn excludes 0.
    def _fmt_md(v) -> str:
        return "--" if v is None else f"{v:.4f}"

    def _transpose(rows, methods, levels):
        """Flip a top_pct-keyed table to methods-as-rows (one row per method)."""
        by_pct = {r["top_pct"]: r for r in rows}
        out = []
        for m in methods:
            row = {"method": m}
            for lv in levels:
                r = by_pct.get(lv)
                row[lv] = r.get(m) if r is not None else None
            out.append(row)
        return out

    def _md_methods_table(rows, levels, title, delta_level=None, sig=None):
        """Methods-as-rows markdown table. Bolds the best AUROC per column,
        appends a Δ-vs-best column at ``delta_level``, and marks cells with a
        significance dagger from the ``sig`` map {(method, level): True}."""
        best_by_level = {lv: max((r[lv] for r in rows if r[lv] is not None),
                                 default=None) for lv in levels}
        lines = [f"### {title}", "",
                 "| method | " + " | ".join(f"top-{lv}%" for lv in levels)
                 + (f" | Δ vs best (top-{delta_level}%) |" if delta_level else " |"),
                 "|:---|" + "---:|" * len(levels)
                 + ("---:|" if delta_level else "")]
        for r in rows:
            cells = []
            for lv in levels:
                v = r[lv]
                s = _fmt_md(v)
                b = best_by_level[lv]
                if v is not None and b is not None and abs(v - b) < 1e-9:
                    s = f"**{s}**"
                if sig and sig.get((r["method"], lv)):
                    s += "†"
                cells.append(s)
            if delta_level:
                v = r[delta_level]
                b = best_by_level[delta_level]
                cells.append(f"{v - b:+.4f}" if v is not None and b is not None else "--")
            lines.append(f"| {r['method']} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    levels = [r["top_pct"] for r in panel_b]
    sig = {}
    for level, lv in [("conf_top50", 50), ("conf_top10", 10)]:
        for sname in hybrid_comps:
            ci = tests.get(f"boot_paired_hybrid_vs_{sname}_auroc_{level}")
            if ci is not None and (ci[0] > 0 or ci[1] < 0):
                sig[(sname, lv)] = True

    md = [f"# Baseline comparison — {name} ({member_keys})",
          "",
          f"Data: {name}, cal={C['gt'].shape[0]} / eval={D['gt'].shape[0]} images"
          + (f" ({len(df_eval)} certain-label eval rows; {task2['n_uncertain_total']} "
             f"uncertain for Task 2)" if has_uncertain else "")
          + f", conf_pct={args.conf_pct}.",
          "",
          _md_methods_table(
              _transpose(panel_a_rows, member_keys, levels), levels,
              "Panel A — single-member self-detection at matched coverage "
              "(AUROC of each member's own confidence vs its own errors)",
              delta_level=10),
          "",
          _md_methods_table(
              _transpose(panel_b, list(scores_b.keys()), levels), levels,
              "Panel B — all scores within the ensemble confident set "
              "(AUROC vs ensemble errors; the flag's population)",
              delta_level=10, sig=sig),
          "",
          "† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn "
          "excludes 0 (n_boot=500).",
          "",
          "### E-AURC / AUAC (risk-coverage, full eval set)",
          "",
          "| score | AURC | E-AURC | AUAC | error_rate |",
          "|---|---:|---:|---:|---:|",
          *[f"| {n} | {_fmt_md(v['aurc'])} | {_fmt_md(v['e_aurc'])} | "
            f"{_fmt_md(v['auac'])} | {v['error_rate']} |"
            for n, v in aurc_table.items()],
          "",
          f"### Per-pathology correctness AUROC (macro over "
          f"{len(per_path['per_pathology'])} pathologies, full eval set)",
          "",
          "| score | macro AUROC |",
          "|---|---:|",
          *[f"| {n} | {_fmt_md(v)} |" for n, v in per_path["macro"].items()],
          ""]
    if task2 is not None:
        md += [f"### Baur Task 2 — uncertainty-label prediction (macro over "
               f"{len(task2['per_pathology'])} pathologies; positive class = expert "
               f"'uncertain' (-1); {task2['n_uncertain_total']} uncertain eval rows)",
               "",
               "| score | macro AUROC |",
               "|---|---:|",
               *[f"| {n} | {_fmt_md(v)} |" for n, v in task2["macro"].items()],
               ""]
    md += ["### TS-calibrated confidence (calibration-only baseline)",
           "",
           f"- temperature T = {ts_facts['temperature']}",
           f"- ECE raw = {ts_facts['ece_raw']}  ->  ECE TS = {ts_facts['ece_ts']}",
           f"- confident-error AUROC of -pbar raw = {ts_facts['auroc_pbar_raw_top10']}  "
           f"TS = {ts_facts['auroc_pbar_ts_top10']}  "
           f"(max abs diff {ts_facts['max_abs_auroc_diff']}; TS is rank-invariant)",
           "",
           "### Statistical tests (ensemble confident set, top-10%)",
           "",
           "| test | result |",
           "|---|---|",
           *[f"| {n} | {json.dumps(v)} |" for n, v in tests.items()],
           ""]
    (out / "baseline_table.md").write_text("\n".join(md))

    # --- plot ----------------------------------------------------------------
    try:
        from cxr_uncertainty import viz
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
        style = {"epistemic_std": (viz.BLUE, "-", "o", 2.6),
                 "mutual_info": (viz.BLUE_LIGHT, "-", "s", 1.6),
                 "entropy": (viz.GOOD, "-", "^", 1.6),
                 "confidence": (viz.CRIT, "--", "v", 1.6),
                 "mahalanobis": (viz.ORANGE, "-", "D", 1.6),
                 "knn": (viz.SERIOUS, "-", "P", 1.6),
                 "energy": (viz.WARN, "-", "X", 1.6),
                 "hybrid_std_knn": (viz.SERIOUS, "-", "*", 2.6),
                 "hybrid_std_maha": (viz.SERIOUS, "--", "*", 1.6)}
        for k in member_keys:
            style[f"conf_{k}"] = (viz.MUTED, ":", "o", 1.2)
        for ax, rows, title, names in [
                (ax1, panel_a_rows, "A: single-member self-detection", member_keys),
                (ax2, panel_b, "B: within ensemble confident set", list(style))]:
            ax.axhline(0.5, color=viz.BASE, lw=0.8, ls=":")
            for sname in names:
                c, ls, mk, lw = style.get(sname, (viz.MUTED, ":", "o", 1.2))
                xs, ys = [], []
                for r in rows:
                    v = r.get(sname)
                    if v is not None:
                        xs.append(r["top_pct"]); ys.append(v)
                if xs:
                    ax.plot(xs, ys, color=c, ls=ls, marker=mk, lw=lw,
                            label=sname, markersize=4)
            ax.set_xlabel("top % confident (selectivity)")
            ax.set_title(title)
            ax.set_xlim(55, 0)
            ax.grid(True, color=viz.GRID, lw=0.8)
        ax1.set_ylabel("confident-error AUROC")
        ax1.legend(fontsize=7, loc="lower left", ncol=1)
        ax2.legend(fontsize=7, loc="lower left", ncol=1)
        fig.suptitle(f"Baseline comparison ({name}) — confident-error AUROC vs selectivity",
                     color=viz.INK)
        fig.tight_layout()
        fig.savefig(out / "baselines_auroc_by_selectivity.png", dpi=150)
        plt.close(fig)
        print(f"[{name}] plot -> {out / 'baselines_auroc_by_selectivity.png'}")
    except Exception as e:  # plotting must never sink the run
        print(f"[{name}] plot skipped ({e})")

    # --- console -------------------------------------------------------------
    print(f"\n[{name}] Panel A — single-member self-detection (AUROC at matched coverage)")
    hdr = f"{'top%':>5s} {'n':>6s} {'wrong':>6s}"
    for k in member_keys:
        hdr += f" {k:>10s}"
    print(hdr)
    for r in panel_a_rows:
        print(f"{r['top_pct']:5d} {r['n']:6d} {r['n_wrong']:6d}"
              + "".join(f" {_fmt(r.get(k)):>10s}" for k in member_keys))
    print(f"\n[{name}] Panel B — within ensemble confident set")
    cols = ["epistemic_std", "mutual_info", "entropy", "confidence",
            "mahalanobis", "knn", "energy", "hybrid_std_knn", "hybrid_std_maha"]
    hdr = f"{'top%':>5s} {'n':>6s} {'wrong':>6s}"
    for c in cols:
        hdr += f" {c:>14s}"
    print(hdr)
    for r in panel_b:
        print(f"{r['top_pct']:5d} {r['n']:6d} {r['n_wrong']:6d}"
              + "".join(f" {_fmt(r.get(c)):>14s}" for c in cols))
    print(f"\n[{name}] E-AURC / AUAC (full eval set):")
    for n, v in aurc_table.items():
        print(f"  {n:16s} AURC={_fmt(v['aurc'])}  E-AURC={_fmt(v['e_aurc'])}  "
              f"AUAC={_fmt(v['auac'])}")
    print(f"\n[{name}] Per-pathology correctness AUROC (macro over "
          f"{len(per_path['per_pathology'])} pathologies):")
    for n, v in per_path["macro"].items():
        print(f"  {n:16s} {_fmt(v)}")
    if task2 is not None:
        print(f"\n[{name}] Baur Task 2 — uncertainty-label prediction (macro over "
              f"{len(task2['per_pathology'])} pathologies, {task2['n_uncertain_total']} "
              f"uncertain eval rows):")
        for n, v in task2["macro"].items():
            print(f"  {n:16s} {_fmt(v)}")
    print(f"\n[{name}] TS: T={ts_facts['temperature']}  ECE {ts_facts['ece_raw']} -> "
          f"{ts_facts['ece_ts']}  AUROC(-pbar) raw={ts_facts['auroc_pbar_raw_top10']} "
          f"TS={ts_facts['auroc_pbar_ts_top10']}")
    print(f"[{name}] wrote {out / 'baseline_comparison.json'} and "
          f"{out / 'baseline_table.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays", default="runs/phase4_features_4mem",
                   help="dir holding arrays_openiC/D.npz (back-compat single-OpenI "
                        "mode; ignored when --datasets is given)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="one or more name:cal_npz:eval_npz triples (e.g. "
                        "openi:.../arrays_openiC.npz:.../arrays_openiD.npz). The "
                        "dataset named 'openi' runs the EXPECTED self-check.")
    p.add_argument("--out", default="runs/baselines")
    p.add_argument("--conf-pct", type=float, default=10.0)
    p.add_argument("--unc-pct", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    datasets = _parse_datasets(args)
    print(f"[run] {len(datasets)} dataset(s): {[d[0] for d in datasets]}")
    for name, cal_path, eval_path, do_selfcheck in datasets:
        if not cal_path.exists() or not eval_path.exists():
            print(f"[{name}] SKIP -- missing arrays ({cal_path}, {eval_path})")
            continue
        C = _load_arrays(cal_path)
        D = _load_arrays(eval_path)
        member_keys = [str(k) for k in (C["member_keys"] if C["member_keys"] is not None
                                        else D["member_keys"])]
        run_dataset(name, C, D, member_keys, args, do_selfcheck)
    print("\n[done] all datasets processed.")


if __name__ == "__main__":
    main()