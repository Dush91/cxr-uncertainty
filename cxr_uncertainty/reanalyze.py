"""Recompute the evaluation from a saved per_record.csv without re-running inference.

The live run saves raw per-(image,pathology) signals: p_bar, epistemic_std,
mutual_info, gt. Because the torchxrayvision ensemble is uncalibrated and its
probability scale is compressed on out-of-distribution data, the right operating
point is NOT a global 0.5 threshold. Instead we:

  1. split images into calibration / evaluation halves (no label leakage),
  2. fit a per-class decision threshold on the calibration split (Youden's J),
  3. on the eval split, define:
       decision  = p_bar >= threshold_class
       confidence = normalized distance of p_bar from the class threshold in [0,1]
       confident  = top `conf_pct`% most-confident predictions
       risk_flag  = confident AND epistemic_std in the top `unc_pct`% (most uncertain)
  4. compute the confident-error AUROC, risk-coverage/AURC, flag recall vs
     false-flag rate, per-pathology AUROC, and plots.

Usage:
    python -m cxr_uncertainty.reanalyze runs/demo/per_record.csv --out runs/demo
    python -m cxr_uncertainty.reanalyze runs/demo/per_record.csv --conf-pct 15 --unc-pct 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import NIH_PATHOLOGIES, RiskConfig
from .evaluate import Records, evaluate_records, save_report
from .interfaces import register_calibrator, get_calibrator
from .risk import classify_error


def _youden_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximizing TPR-FPR. Falls back to 0.5 if a class is degenerate."""
    if len(np.unique(y)) < 2:
        return 0.5
    from sklearn.metrics import roc_curve
    fpr, tpr, t = roc_curve(y, s)
    # roc_curve thresholds are descending and include an extra +inf entry; pick J.
    j = int(np.argmax(tpr - fpr))
    j = min(j, len(t) - 1)
    return float(t[j])


def calibrate_thresholds(df_calib: pd.DataFrame) -> Dict[str, float]:
    thr = {}
    for name, g in df_calib.groupby("pathology"):
        thr[name] = _youden_threshold(g["gt"].values, g["p_bar"].values)
    return thr


@register_calibrator("youden")
def youden_calibrate(df_calib: pd.DataFrame, df_eval: pd.DataFrame,
                     conf_pct: float, unc_pct: float, **_):
    """Default calibrator (Part 2 Phase 0): per-class Youden thresholds on the
    calibration split, then build records on the eval split. Registered as
    ``"youden"``; the 7-step BCTS/ETS/Mondrian calibrator is added in Part 3."""
    thresholds = calibrate_thresholds(df_calib)
    rec, meta = build_records(df_eval, thresholds, conf_pct, unc_pct)
    meta["calibrator"] = "youden"
    return rec, meta


def _confidence(p_bar: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalized distance of p_bar from the per-class threshold in [0,1]:
    present -> (p_bar-t)/(1-t); absent -> (t-p_bar)/t."""
    present = p_bar >= t
    denom_p = np.where((1 - t) > 1e-6, (1 - t), 1e-6)
    denom_a = np.where(t > 1e-6, t, 1e-6)
    conf = np.where(present, (p_bar - t) / denom_p, (t - p_bar) / denom_a)
    return np.clip(conf, 0.0, 1.0)


def build_records(
    df_eval: pd.DataFrame, thresholds: Dict[str, float],
    conf_pct: float, unc_pct: float,
    unc_column: str = "epistemic_std",
    risk_flag_override: Optional[np.ndarray] = None,
) -> Tuple[Records, dict]:
    """Build ``Records`` from the eval split with per-class calibrated thresholds.

    ``unc_column`` selects which per-row uncertainty score drives the
    ``risk_flag`` (the top ``unc_pct``% most-uncertain among the confident set).
    Default ``"epistemic_std"`` preserves the original disagreement-flag behavior
    byte-for-byte; the production variant passes ``"mahalanobis"`` so the flag is
    the feature-density OOD score (Phase 4 step 6). The ``epistemic_std`` /
    ``mutual_info`` columns are ALWAYS recorded as the disagreement signal
    regardless of ``unc_column`` — only the *flag* changes. A ``mahalanobis``
    column, if present, is recorded on each row (else 0.0).

    ``risk_flag_override`` (bool array, len=df_eval; plan §L) lets a calibrator
    supply the flag semantics directly — the conformal triage calibrator passes
    the LAC ``lac_refer`` flag so the conformal-refer flag is apples-to-apples
    with the epistemic_std / Mahalanobis flags on the same confident population.
    When given, the percentile ``unc_cut`` flag is bypassed (the override IS the
    flag); default ``None`` preserves the original percentile behavior. LAC
    columns (``lac_set_size`` / ``lac_refer`` / ``lac_covered``), if present, are
    read onto each ``Records`` row and set ``conformal_active`` (Part 3 step 7)."""
    t = df_eval["pathology"].map(thresholds).fillna(0.5).values.astype(float)
    p_bar = df_eval["p_bar"].values.astype(float)
    gt = df_eval["gt"].values.astype(int)
    decision = (p_bar >= t).astype(int)
    confidence = _confidence(p_bar, t)
    ep = df_eval["epistemic_std"].values.astype(float)
    mi = df_eval["mutual_info"].values.astype(float)
    # The score that drives the risk flag (default = disagreement; production =
    # mahalanobis). Falls back to epistemic_std if the named column is absent.
    if unc_column in df_eval.columns:
        unc = df_eval[unc_column].values.astype(float)
    else:
        unc = ep
    has_maha = "mahalanobis" in df_eval.columns
    maha = df_eval["mahalanobis"].values.astype(float) if has_maha else np.zeros(len(df_eval))

    # Confident set = top conf_pct% by confidence.
    conf_cut = float(np.percentile(confidence, 100.0 - conf_pct))
    is_conf = confidence >= conf_cut
    # Among confident, flag the top unc_pct% by the chosen uncertainty score —
    # UNLESS a calibrator supplied risk_flag_override (e.g. the LAC refer flag),
    # in which case the override IS the flag and the percentile cut is bypassed.
    if risk_flag_override is not None:
        risk_flag = np.asarray(risk_flag_override, dtype=bool)
        unc_cut = None
    else:
        if is_conf.sum() > 0:
            unc_cut = float(np.percentile(unc[is_conf], 100.0 - unc_pct))
        else:
            unc_cut = float(np.percentile(unc, 100.0 - unc_pct)) if len(unc) else 0.0
        risk_flag = is_conf & (unc >= unc_cut)

    # LAC conformal columns (Part 3 step 7) — present when a conformal calibrator
    # attached them; defaults (1/False/True) keep the live per-image path aligned.
    # `conformal_active` is True only when a conformal calibrator DROVE this run
    # (it passes risk_flag_override); a re-read CSV carrying default lac columns
    # under a non-conformal calibrator (youden/ts_only) must NOT report conformal
    # metrics, so we gate on the override, not mere column presence.
    has_lac = "lac_set_size" in df_eval.columns
    lac_set = df_eval["lac_set_size"].values.astype(int) if has_lac else None
    lac_ref = df_eval["lac_refer"].values.astype(bool) if has_lac else None
    lac_cov = df_eval["lac_covered"].values.astype(bool) if has_lac else None
    # kNN feature-distance baseline (uncertainty_evaluation.md §4.3): carried
    # through from a re-read CSV; 0.0 when absent (reanalyze.run attaches real
    # values from a features sidecar after the calibrator returns).
    has_knn = "knn_dist" in df_eval.columns
    knn = df_eval["knn_dist"].values.astype(float) if has_knn else None

    rec = Records()
    rec.conformal_active = bool(has_lac and risk_flag_override is not None)
    has_valid = "valid" in df_eval.columns
    has_logits = "logits" in df_eval.columns
    has_mp = "member_probs" in df_eval.columns
    for i in range(len(df_eval)):
        iid = str(df_eval["image_id"].iloc[i])
        path = str(df_eval["pathology"].iloc[i])
        conf_i = float(confidence[i])
        is_c = bool(is_conf[i])
        et = classify_error(int(decision[i]), conf_i, int(gt[i]), conf_cut)
        # record fields
        rec.image_id.append(iid)
        rec.pathology.append(path)
        rec.p_bar.append(float(p_bar[i]))
        rec.decision.append(int(decision[i]))
        rec.confidence.append(conf_i)
        rec.epistemic_std.append(float(ep[i]))
        rec.mutual_info.append(float(mi[i]))
        rec.gt.append(int(gt[i]))
        rec.is_confident.append(is_c)
        rec.risk_flag.append(bool(risk_flag[i]))
        rec.error_type.append(et)
        # schema_version 2 passthrough (reanalyze recomputes decision/confidence;
        # valid/logits/member_probs carry through from the input per_record.csv).
        rec.valid.append(int(df_eval["valid"].iloc[i]) if has_valid else 1)
        rec.logits.append(float(df_eval["logits"].iloc[i]) if has_logits else
                           float(np.log(min(max(p_bar[i],1e-6),1-1e-6) /
                                         (1-min(max(p_bar[i],1e-6),1-1e-6)))))
        rec.member_probs.append(str(df_eval["member_probs"].iloc[i]) if has_mp else "{}")
        rec.mahalanobis.append(float(maha[i]))
        # LAC conformal (Part 3 step 7): defaults when not conformalized.
        rec.lac_set_size.append(int(lac_set[i]) if lac_set is not None else 1)
        rec.lac_refer.append(bool(lac_ref[i]) if lac_ref is not None else False)
        rec.lac_covered.append(bool(lac_cov[i]) if lac_cov is not None else True)
        rec.knn_dist.append(float(knn[i]) if knn is not None else 0.0)

    meta = {
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "conf_cut": round(conf_cut, 4),
        "unc_cut": (round(unc_cut, 4) if unc_cut is not None else None),
        "conf_pct": conf_pct,
        "unc_pct": unc_pct,
        "unc_column": unc_column,
        "uq_score": unc_column,
        "risk_flag_override": risk_flag_override is not None,
        "conformal_active": bool(has_lac and risk_flag_override is not None),
    }
    return rec, meta


def auroc_at_confidence_levels(rec: Records) -> List[dict]:
    """Confident-error AUROC of each uncertainty score at several confidence
    selectivity levels — shows the key property that uncertainty becomes more
    diagnostic of error as confidence rises. Reports ``epistemic_std`` /
    ``mutual_info`` (the disagreement baselines), ``confidence`` (the
    confidence-only baseline — the null the uncertainty scores must beat),
    ``mahalanobis`` (the production feature-density flag, Phase 4 step 6) and
    ``knn`` (feature-distance baseline) when present."""
    out = []
    a = rec.arrays()
    conf = a["confidence"]
    wrong = a["wrong"]
    ep = a["epistemic_std"]
    mi = a["mutual_info"]
    maha = a.get("mahalanobis")
    knn = a.get("knn_dist")
    has_maha = maha is not None and np.isfinite(maha).any() and not np.all(maha == 0.0)
    has_knn = knn is not None and np.isfinite(knn).any() and not np.all(knn == 0.0)
    for pct in [50, 25, 15, 10, 5]:
        if len(conf) == 0:
            continue
        cut = float(np.percentile(conf, 100 - pct))
        m = conf >= cut
        if (m & (wrong == 1)).sum() == 0 or (m & (wrong == 0)).sum() == 0:
            continue
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(wrong[m], ep[m]))
        auc_mi = float(roc_auc_score(wrong[m], mi[m]))
        # confidence-only baseline: within the confident set, does confidence
        # alone rank errors? (lower confidence = more uncertain -> -conf)
        auc_conf = float(roc_auc_score(wrong[m], -conf[m]))
        row = {"conf_top_pct": pct, "n_confident": int(m.sum()),
               "n_confident_wrong": int((m & (wrong == 1)).sum()),
               "auroc_epistemic_std": round(auc, 4),
               "auroc_mutual_info": round(auc_mi, 4),
               "auroc_confidence": round(auc_conf, 4)}
        if has_maha:
            mm = m & np.isfinite(maha)
            if (mm & (wrong == 1)).sum() > 0 and (mm & (wrong == 0)).sum() > 0:
                row["auroc_mahalanobis"] = round(
                    float(roc_auc_score(wrong[mm], maha[mm])), 4)
            else:
                row["auroc_mahalanobis"] = None
        if has_knn:
            km = m & np.isfinite(knn)
            if (km & (wrong == 1)).sum() > 0 and (km & (wrong == 0)).sum() > 0:
                row["auroc_knn"] = round(
                    float(roc_auc_score(wrong[km], knn[km])), 4)
            else:
                row["auroc_knn"] = None
        out.append(row)
    return out


def _load_features_sidecar(features_path: str):
    """Load the RAD-DINO features sidecar (plan §M.1). Returns
    ``(id_to_feat, n_feat)`` where ``id_to_feat`` maps ``str(image_id) -> ndarray``.
    The sidecar is a single npz with ``rad_feats (N_all, D)`` + ``ids (N_all,)``
    covering every CSV image (built by ``scripts/build_features_sidecar.py`` from
    the saved Phase-4-step-6 arrays)."""
    z = np.load(features_path, allow_pickle=True)
    ids = [str(s) for s in z["ids"]]
    feats = np.asarray(z["rad_feats"], dtype=np.float64)
    if feats.shape[0] != len(ids):
        raise ValueError(
            f"features sidecar {features_path}: rad_feats has {feats.shape[0]} rows "
            f"but ids has {len(ids)}")
    id_to_feat = {iid: feats[i] for i, iid in enumerate(ids)}
    return id_to_feat, feats.shape[1] if feats.ndim > 1 else 0


def _subset_features(id_to_feat, sorted_ids):
    """Stack features in the given (sorted-unique) image_id order — the exact
    order ``calibration._long_df_to_arrays`` returns, so ``features[i]`` aligns
    with ``ids[i]`` for the calibrator's by-id dict attach. Returns
    ``(features, missing)`` where ``missing`` lists ids absent from the sidecar
    (caller falls back to the no-features path if any are missing, so no id is
    silently zeroed)."""
    rows, missing = [], []
    for iid in sorted_ids:
        iid = str(iid)
        if iid in id_to_feat:
            rows.append(id_to_feat[iid])
        else:
            missing.append(iid)
    if missing:
        return None, missing
    return np.stack(rows, axis=0), []


def _knn_distances(features_query: np.ndarray, features_ref: np.ndarray,
                   k: int = 5) -> np.ndarray:
    """Mean distance to the k nearest neighbors in the reference set. Higher =
    more OOD / more uncertain (Sun et al., ICML 2022)."""
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(features_ref))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(features_ref)
    dists, _ = nn.kneighbors(features_query)
    return dists.mean(axis=1)


def _statistical_tests(rec: Records, df_eval: pd.DataFrame,
                       thresholds: Dict[str, float], conf_pct: float,
                       unc_pct: float) -> dict:
    """Bootstrap CIs + DeLong + McNemar for the key comparisons
    (uncertainty_evaluation.md §2.3). All guarded: degenerate cases return None
    rather than raising. Runs on the eval split only (no re-inference)."""
    from .evaluate import _auroc
    from .statistics import bootstrap_ci, delong_test, mcnemar_test
    a = rec.arrays()
    is_conf = a["is_confident"]
    wrong = a["wrong"]
    out: dict = {}

    # --- bootstrap CI for the confident-error AUROC (epistemic_std) ---------
    if is_conf.sum() > 0 and (is_conf & (wrong == 1)).sum() > 0 \
            and (is_conf & (wrong == 0)).sum() > 0:
        vals = np.column_stack([a["epistemic_std"][is_conf], wrong[is_conf]])
        lo, hi = bootstrap_ci(vals, lambda v: _auroc(v[:, 0], v[:, 1]),
                              n_boot=1000, seed=0)
        if lo is not None:
            out["confident_error_auroc_std_ci95"] = [round(lo, 4), round(hi, 4)]

    # --- DeLong: mahalanobis vs epistemic_std on the same confident set -----
    maha = a.get("mahalanobis")
    has_maha = maha is not None and np.isfinite(maha).any() \
        and not np.all(maha == 0.0)
    if has_maha:
        mm = is_conf & np.isfinite(maha)
        if (mm & (wrong == 1)).sum() > 0 and (mm & (wrong == 0)).sum() > 0:
            auc_m, auc_s, z, p = delong_test(maha[mm], a["epistemic_std"][mm],
                                             wrong[mm])
            out["delong_mahalanobis_vs_std"] = {
                "auc_mahalanobis": (round(auc_m, 4) if auc_m is not None else None),
                "auc_epistemic_std": (round(auc_s, 4) if auc_s is not None else None),
                "z": (round(z, 4) if z is not None else None),
                "p": (round(p, 4) if p is not None else None),
            }

    # --- McNemar: epistemic_std flag vs mahalanobis flag on the confident set
    # (rebuild records with unc_column="mahalanobis" to get the maha flag; the
    # confident population is identical since it is driven by confidence).
    if has_maha and "mahalanobis" in df_eval.columns and is_conf.sum() > 0:
        rec_maha, _ = build_records(df_eval, thresholds, conf_pct, unc_pct,
                                    unc_column="mahalanobis")
        flag_epi = np.array(rec.risk_flag, dtype=bool)
        flag_maha = np.array(rec_maha.risk_flag, dtype=bool)
        truth = np.array(rec.decision, dtype=int) != np.array(rec.gt, dtype=int)
        chi2, p = mcnemar_test(flag_epi, flag_maha, truth)
        if chi2 is not None:
            out["mcnemar_epistemic_vs_mahalanobis_flag"] = {
                "chi2": round(chi2, 4), "p": round(p, 4),
                "n_confident": int(is_conf.sum()),
            }
    return out


def run(csv_path: str, out: str, conf_pct: float, unc_pct: float, seed: int = 0,
        calibrator: str = "youden", alpha: float = 0.1,
        features_path: Optional[str] = None):
    df = pd.read_csv(csv_path)
    # schema_version>=2 CSVs carry a `logits` column (BCTS in Part 3 reloads it);
    # the youden path ignores extra columns. Old 11-col CSVs still load.
    has_logits = "logits" in df.columns
    # Drop pathologies with no ground truth in the dataset (e.g. Consolidation,
    # which has no OpenI MeSH mapping -> gt always 0 -> all present predictions
    # would be artifactual false positives).
    pos_per = df.groupby("pathology")["gt"].sum()
    unmappable = pos_per[pos_per == 0].index.tolist()
    if unmappable:
        print(f"[reanalyze] excluding pathologies with no ground truth: {unmappable}")
        df = df[~df["pathology"].isin(unmappable)].reset_index(drop=True)
    imgs = sorted(df["image_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(imgs)
    cut = len(imgs) // 2
    calib_imgs, eval_imgs = set(imgs[:cut]), set(imgs[cut:])
    df_calib = df[df["image_id"].isin(calib_imgs)]
    df_eval = df[df["image_id"].isin(eval_imgs)].reset_index(drop=True)

    calibrate = get_calibrator(calibrator)
    # Features sidecar (plan §M.1): subset the single RAD-DINO features npz to
    # the cal/eval sorted-unique image_id order so the ts_only_mahalanobis
    # calibrator joins features to rows by image_id (no silent 0.0 corruption
    # from missing ids). If any cal/eval id is missing from the sidecar, fall
    # back to the no-features path (the existing ts_only fallback fires with a
    # uq_warning) rather than silently zeroing them. Other calibrators absorb
    # the extra kwargs via **_.
    feat_kwargs = {}
    if features_path:
        # Match ``calibration._long_df_to_arrays`` EXACTLY: it sorts the RAW
        # (uncast) ``image_id`` uniques, then the calibrator casts to str when
        # building the by-id dict. Sorting str-cast ids would give lexicographic
        # order (e.g. ['1','10','2']) != numeric order (['1','2','10']) for
        # numeric ids -> features[i] would misalign with ids[i]. So sort raw,
        # cast to str only for the sidecar dict lookup.
        idsC = sorted(df_calib["image_id"].unique())
        idsD = sorted(df_eval["image_id"].unique())
        id_to_feat, _ = _load_features_sidecar(features_path)
        fC, missC = _subset_features(id_to_feat, idsC)
        fD, missD = _subset_features(id_to_feat, idsD)
        n_miss = len(missC) + len(missD)
        if n_miss:
            print(f"[reanalyze] features sidecar missing {n_miss} image_id(s) "
                  f"({len(missC)} cal, {len(missD)} eval) -> NOT passing features; "
                  f"the ts_only_mahalanobis calibrator will fall back to ts_only "
                  f"(epistemic_std flag) with a uq_warning. First missing: "
                  f"{(missC + missD)[:5]}")
        else:
            print(f"[reanalyze] features sidecar loaded: cal={fC.shape} eval={fD.shape}")
            feat_kwargs = {"features_cal": fC, "features_eval": fD,
                           "fit_label": "Pneumonia"}
    rec, meta = calibrate(df_calib, df_eval, conf_pct=conf_pct, unc_pct=unc_pct,
                          alpha=alpha, **feat_kwargs)

    # kNN feature-distance baseline (uncertainty_evaluation.md §4.3): when a
    # features sidecar was loaded, attach the mean kNN distance of each eval
    # image to the calibration-split features. Higher = more OOD. This is the
    # strongest unmeasured competitor (Woodland et al. 2024: kNN often beats
    # Mahalanobis in medical imaging), so it must be in the baseline table.
    if features_path and n_miss == 0:
        knn = _knn_distances(fD, fC, k=5)
        id_to_knn = {str(iid): float(d) for iid, d in zip(idsD, knn)}
        rec.knn_dist = [id_to_knn.get(str(iid), 0.0) for iid in rec.image_id]
        print(f"[reanalyze] kNN baseline attached (k=5, ref=cal features, "
              f"n={len(knn)})")

    cfg = RiskConfig()  # thresholds live in meta; cfg carries defaults for the report
    n_imgs = len(eval_imgs)
    report = evaluate_records(rec, cfg, n_imgs)

    out_dir = Path(out) / "reanalyze"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_report(report, str(out_dir), records=rec)
    # extra: selectivity curve + calibration meta + statistical tests
    sel = auroc_at_confidence_levels(rec)
    stats = _statistical_tests(rec, df_eval, meta["thresholds"], conf_pct, unc_pct)
    with open(out_dir / "reanalyze_meta.json", "w") as f:
        json.dump({"calibration": meta,
                   "confident_error_auroc_vs_selectivity": sel,
                   "statistical_tests": stats}, f, indent=2)

    # plots
    try:
        from . import viz
        viz.make_plots(rec, report, str(out_dir / "plots"))
    except Exception as e:
        print(f"[viz] plotting skipped ({e})")

    _print(report, meta, sel)
    return report, sel


def _print(r, meta, sel):
    print("\n" + "=" * 70)
    print("REANALYZED RESULTS (per-class calibrated thresholds, eval split)")
    print("=" * 70)
    print(f"per-class thresholds: {meta['thresholds']}")
    print(f"conf_cut={meta['conf_cut']} (top {meta['conf_pct']}% confident)  "
          f"unc_cut={meta['unc_cut']} (top {meta['unc_pct']}% uncertain among confident)")
    print(f"records={r.n_records}  confident={r.n_confident}  confident_wrong={r.n_confident_wrong}")
    print(f"confident-error AUROC (epistemic_std) : {r.confident_error_auroc_std}")
    print(f"confident-error AUROC (mutual_info)   : {r.confident_error_auroc_mi}")
    print(f"confident-error AUROC (confidence)    : {r.confidence_auroc}  [baseline]")
    print(f"confident-error recall (risk_flag)    : {r.confident_error_recall}")
    print(f"false-flag rate (confident correct)   : {r.false_flag_rate}")
    print(f"AURC (risk-coverage, epistemic_std)  : {r.aurc}")
    print(f"lowest-risk operating point (cov,err) : {r.coverage_at_lowest_risk}")
    if r.aurc_by_score:
        print("AURC / E-AURC by score (lower AURC better; E-AURC removes accuracy confound):")
        for name, d in r.aurc_by_score.items():
            print(f"  {name:14s} AURC={d['aurc']}  E-AURC={d['e_aurc']}")
    print(f"ECE (calibration of p_bar)            : {r.ece:.4f}")
    if r.ece_multi:
        print(f"ECE by bin count                     : {r.ece_multi}")
    print(f"NLL (log-loss)                        : {r.nll}")
    if r.brier_decomposition:
        bd = r.brier_decomposition
        print(f"Brier (rel - res + unc)              : {bd['brier']:.4f} "
              f"= {bd['reliability']:.4f} - {bd['resolution']:.4f} + {bd['uncertainty']:.4f}")
    if r.conformal_active and r.conformal_coverage is not None:
        print(f"conformal coverage (1-alpha={1-meta.get('alpha',0.1):.2f})   : {r.conformal_coverage:.4f}")
        print(f"conformal avg set size                : {r.conformal_size:.4f}")
    print("error type counts:", r.error_type_counts)
    print("-" * 70)
    print("confident-error AUROC vs confidence selectivity (key property):")
    for s in sel:
        line = (f"  top {s['conf_top_pct']:2d}% confident: n={s['n_confident']:5d} "
                f"wrong={s['n_confident_wrong']:4d}  AUROC_std={s['auroc_epistemic_std']}  "
                f"AUROC_mi={s['auroc_mutual_info']}  AUROC_conf={s['auroc_confidence']}")
        if "auroc_mahalanobis" in s:
            line += f"  AUROC_maha={s['auroc_mahalanobis']}"
        if "auroc_knn" in s:
            line += f"  AUROC_knn={s['auroc_knn']}"
        print(line)
    print("-" * 70)
    print("per-pathology (n / pos / task_auroc / conf_correct / conf_wrong / flagged):")
    for p, d in r.per_pathology.items():
        print(f"  {p:22s} {d['n']:5d} {d['positives']:4d} {str(d['task_auroc'])[:6]:>6s} "
              f"{d['confident_correct']:5d} {d['confident_wrong']:4d} {d['flagged']:4d}")
    print("=" * 70)


def main(argv=None):
    p = argparse.ArgumentParser(prog="cxr_uncertainty.reanalyze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="path to per_record.csv from a run")
    p.add_argument("--out", default="runs/demo", help="output dir (writes evaluation.json, plots/)")
    p.add_argument("--conf-pct", type=float, default=10.0,
                   help="%% of predictions considered 'confident' (top by confidence)")
    p.add_argument("--unc-pct", type=float, default=50.0,
                   help="%% of confident predictions flagged high-risk (top by epistemic std)")
    p.add_argument("--calibrator", default="youden",
                   help="calibrator key (youden [default] | bcts_ets_mondrian | "
                        "ts_only | ts_only_mahalanobis [production] | "
                        "conformal_triage [Part 3 step 7])")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="conformal miscoverage level (1-alpha coverage target; "
                        "conformal_triage only, ignored by other calibrators)")
    p.add_argument("--features", default="",
                   help="path to a RAD-DINO features sidecar npz (rad_feats + ids) "
                        "so --calibrator ts_only_mahalanobis runs the real Mahalanobis "
                        "flag from the CLI (plan §M.1). Built by "
                        "scripts/build_features_sidecar.py. Ignored by other calibrators.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run(args.csv, args.out, args.conf_pct, args.unc_pct, args.seed, args.calibrator,
        args.alpha, features_path=(args.features or None))


if __name__ == "__main__":
    main()