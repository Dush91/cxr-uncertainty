"""Production UQ variant: TS-only calibration + Mahalanobis confident-error flag.

Phase 4 step 6 established the production confident-error design:

  * **Calibration = TS-only** on the pooled mean (``build_pipeline(use_beta=False,
    use_ts=True)``). Per-member Beta was rejected (Phase 3): it removes the
    inter-member scale-family disagreement that carries error-discriminating
    signal. TS-only fixes ECE ~10x in-distribution while preserving the raw
    disagreement ordering (§E.2 monotonicity).
  * **Confident-error flag = Mahalanobis** on frozen RAD-DINO ``[CLS]`` features
    (``feature_uq.MahalanobisOOD``), NOT cross-member disagreement. Disagreement
    is at/below chance under distribution shift (Kermany 0.515, COVID 0.478) — two
    compounding failures: the OpenI-fit temperature shatters under prevalence
    shift, and the confident-agree-wrong core (Abe 2022) is invisible to
    disagreement. The feature-density Mahalanobis score sidesteps both and beats
    chance under shift (Kermany 0.818, COVID 0.649) AND in-distribution (0.835 vs
    0.631). See ``FINDINGS.md`` + plan §K.

This module registers that design as two named, discoverable calibrators so it is
reachable from the CLI (``reanalyze --calibrator ts_only[_mahalanobis]``) AND
exposes a full-precision in-process builder for the eval driver
(``scripts/eval_production.py``). ``youden`` stays the default calibrator
(backward-compatible; ``run_demo``/Phase-0 baselines unchanged).

Two calibrators:
  * ``ts_only``                — TS-only calibration, disagreement (epistemic_std)
    flag. CSV-feasible (rebuilds ``(N,M,P)`` from the ``member_probs`` JSON).
  * ``ts_only_mahalanobis``    — TS-only calibration, **Mahalanobis** flag. The
    feature score needs the RAD-DINO ``[CLS]`` features, which are NOT in the
    per_record CSV (a features sidecar is plan §J.1 future work). So the CLI path
    honestly falls back to ``ts_only`` with a ``meta`` warning; the driver path
    (``build_ts_only_mahalanobis_pipeline``) passes features in-process and runs
    the real production flag.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import _long_df_to_arrays, build_pipeline
from .feature_uq import MahalanobisOOD
from .reanalyze import build_records, calibrate_thresholds


# ---------------------------------------------------------------------------
# In-process builder (full precision) — the driver / headline path
# ---------------------------------------------------------------------------
def _resolve_label_idx(
    pathologies: List[str], fit_label: Optional[str], fit_label_idx: int,
) -> Tuple[int, Optional[str]]:
    """Resolve the binary-label column the Mahalanobis flag is fit on.

    Name-based (``fit_label``) is robust to the pathologies ordering: the CLI
    path's ``_long_df_to_arrays`` returns pathologies **sorted alphabetically**
    (NOT NIH order), so a positional ``fit_label_idx=6`` would index Fibrosis,
    not Pneumonia. Name-based resolution is correct regardless of ordering. The
    in-process driver passes NIH-order arrays + ``fit_label_idx=PNEUMONIA_IDX``
    and is unaffected (``fit_label=None`` keeps the positional path). Returns
    ``(lab, used_label)``; ``lab=-1`` (and a warning string) if the named label
    is absent -> caller falls back to ``ts_only``."""
    if fit_label is not None:
        if fit_label in pathologies:
            return pathologies.index(fit_label), fit_label
        return -1, None
    return int(fit_label_idx), pathologies[fit_label_idx] if 0 <= fit_label_idx < len(pathologies) else None


def build_ts_only_mahalanobis_pipeline(
    probs_cal: np.ndarray, gt_cal: np.ndarray, valid_cal: np.ndarray,
    ids_cal: List[str], feats_cal: np.ndarray,
    probs_eval: np.ndarray, gt_eval: np.ndarray, valid_eval: np.ndarray,
    ids_eval: List[str], feats_eval: np.ndarray,
    pathologies: List[str], member_keys: List[str],
    fit_label_idx: int, fit_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The production variant, full precision (no CSV rounding).

    1. TS-only calibration: ``build_pipeline(use_beta=False, use_ts=True)`` —
       temperature-scale the raw pooled mean (fit on the cal split), keep RAW
       cross-member probs for the disagreement signal that ``Records`` records.
    2. Fit ``MahalanobisOOD`` (2 class-conditional Ledoit-Wolf Gaussians) on the
       cal-split RAD-DINO ``[CLS]`` features + the binary label at
       ``fit_label_idx`` (Pneumonia pos/neg for the production flag).
    3. Score the eval-split features; attach the image-level ``mahalanobis``
       column to both long-form frames by ``image_id`` (broadcast to every
       (image, pathology) row of that image).

    Returns ``(df_calib, df_eval, meta)`` with ``meta["uq_score"]="mahalanobis"``
    and ``meta["maha_fit"]`` describing the fit. Downstream
    ``build_records(unc_column="mahalanobis")`` makes the flag the Mahalanobis
    score, and ``evaluate_records`` reports ``confident_error_auroc_mahalanobis``.
    """
    df_calib, df_eval, meta = build_pipeline(
        probs_cal, gt_cal, valid_cal, probs_eval, gt_eval, valid_eval,
        ids_cal, ids_eval, pathologies, member_keys,
        use_beta=False, use_ts=True)

    lab, used_label = _resolve_label_idx(pathologies, fit_label, fit_label_idx)
    y_cal = np.asarray(gt_cal, dtype=int)[:, lab]
    maha = MahalanobisOOD().fit(np.asarray(feats_cal, dtype=np.float64), y_cal)
    score_cal = maha.score(np.asarray(feats_cal, dtype=np.float64))
    score_eval = maha.score(np.asarray(feats_eval, dtype=np.float64))

    id_score_cal = {str(i): float(s) for i, s in zip(ids_cal, score_cal)}
    id_score_eval = {str(i): float(s) for i, s in zip(ids_eval, score_eval)}
    df_calib["mahalanobis"] = df_calib["image_id"].astype(str).map(id_score_cal).fillna(0.0)
    df_eval["mahalanobis"] = df_eval["image_id"].astype(str).map(id_score_eval).fillna(0.0)

    meta = {**meta, "uq_score": "mahalanobis",
            "maha_fit": {"label_idx": int(lab),
                         "label": used_label,
                         "n_per_class": maha.n_per_class_,
                         "n_classes": int(len(maha.mu_))}}
    return df_calib, df_eval, meta


# ---------------------------------------------------------------------------
# Registered calibrators (CSV-feasible reanalyze path)
# ---------------------------------------------------------------------------
def ts_only_calibrate(df_calib: pd.DataFrame, df_eval: pd.DataFrame,
                      conf_pct: float, unc_pct: float, **_) -> Tuple:
    """Registered ``"ts_only"``: TS-only calibration + disagreement (epistemic_std)
    flag. Rebuilds ``(N,M,P)`` from the ``member_probs`` JSON columns, runs
    ``build_pipeline(use_beta=False, use_ts=True)``, Youden thresholds, then
    ``build_records`` (default ``unc_column="epistemic_std"``). Reproduces the
    Phase-3 ts_only curve to ~1e-3 from a saved CSV."""
    pC, yC, vC, idsC, paths, keys = _long_df_to_arrays(df_calib)
    pD, yD, vD, idsD, _, _ = _long_df_to_arrays(df_eval)
    df_cal, df_eval_cal, meta = build_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys,
        use_beta=False, use_ts=True)
    thresholds = calibrate_thresholds(df_cal)
    rec, build_meta = build_records(df_eval_cal, thresholds, conf_pct, unc_pct)
    meta = {**meta, "calibrator": "ts_only",
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "conf_cut": build_meta.get("conf_cut"), "unc_cut": build_meta.get("unc_cut"),
            "conf_pct": conf_pct, "unc_pct": unc_pct,
            "uq_score": "epistemic_std"}
    return rec, meta


def ts_only_mahalanobis_calibrate(df_calib: pd.DataFrame, df_eval: pd.DataFrame,
                                  conf_pct: float, unc_pct: float,
                                  features_cal: Optional[np.ndarray] = None,
                                  features_eval: Optional[np.ndarray] = None,
                                  fit_label_idx: int = 6,
                                  fit_label: Optional[str] = "Pneumonia",
                                  cal_label_idx: Optional[int] = None,
                                  **_) -> Tuple:
    """Registered ``"ts_only_mahalanobis"``: TS-only calibration + Mahalanobis
    confident-error flag.

    With features (driver path or the CLI sidecar path, plan §M.1): attach the
    image-level Mahalanobis score (fit on ``features_cal`` + the binary label)
    to the long-form frames and run ``build_records(unc_column="mahalanobis")``
    so the flag is the feature-density OOD score. ``fit_label`` (default
    ``"Pneumonia"``) is resolved **by name** against the (alphabetically-sorted)
    ``paths`` from ``_long_df_to_arrays`` — robust to the sort-order mismatch
    between the CLI path and the NIH-order driver path (the old positional
    ``fit_label_idx=6`` indexed Fibrosis, not Pneumonia, on the CLI path).

    Without features (CLI/CSV path, no sidecar): the RAD-DINO ``[CLS]`` features
    are not in the per_record CSV, so this honestly delegates to ``ts_only`` and
    records a ``uq_warning`` in ``meta`` — the disagreement flag is used, NOT the
    production Mahalanobis flag. The features sidecar (plan §M.1) closes this gap
    on the CLI path; the driver (``scripts/eval_production.py``) is the other path
    that runs the real production flag from the saved feature arrays.
    """
    if features_cal is None or features_eval is None:
        rec, meta = ts_only_calibrate(df_calib, df_eval, conf_pct, unc_pct)
        meta["calibrator"] = "ts_only_mahalanobis"
        meta["uq_warning"] = ("no features supplied -> fell back to ts_only "
                              "(epistemic_std flag); the Mahalanobis production "
                              "flag needs RAD-DINO [CLS] features (driver path "
                              "or the --features sidecar, plan §M.1)")
        meta["uq_score"] = "epistemic_std"
        return rec, meta

    # features supplied -> the real production flag (driver or CLI sidecar path).
    pC, yC, vC, idsC, paths, keys = _long_df_to_arrays(df_calib)
    pD, yD, vD, idsD, _, _ = _long_df_to_arrays(df_eval)
    # name-based resolution (default "Pneumonia"); falls back to positional
    # fit_label_idx when fit_label is None. cal_label_idx overrides either when
    # set (back-compat for the driver's explicit index).
    if cal_label_idx is not None:
        lab = int(cal_label_idx)
        used_label = paths[lab] if 0 <= lab < len(paths) else None
    else:
        lab, used_label = _resolve_label_idx(paths, fit_label, fit_label_idx)
    if lab < 0:
        # named label absent from this CSV's pathologies -> honest fallback.
        rec, meta = ts_only_calibrate(df_calib, df_eval, conf_pct, unc_pct)
        meta["calibrator"] = "ts_only_mahalanobis"
        meta["uq_warning"] = (f"fit_label={fit_label!r} not in CSV pathologies "
                              f"{paths} -> fell back to ts_only (epistemic_std flag)")
        meta["uq_score"] = "epistemic_std"
        return rec, meta
    df_cal, df_eval_cal, meta = build_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys,
        use_beta=False, use_ts=True)
    maha = MahalanobisOOD().fit(np.asarray(features_cal, dtype=np.float64),
                                np.asarray(yC, dtype=int)[:, lab])
    id_score_cal = {str(i): float(s) for i, s in zip(idsC, maha.score(features_cal))}
    id_score_eval = {str(i): float(s) for i, s in zip(idsD, maha.score(features_eval))}
    df_cal["mahalanobis"] = df_cal["image_id"].astype(str).map(id_score_cal).fillna(0.0)
    df_eval_cal["mahalanobis"] = df_eval_cal["image_id"].astype(str).map(id_score_eval).fillna(0.0)
    thresholds = calibrate_thresholds(df_cal)
    rec, build_meta = build_records(df_eval_cal, thresholds, conf_pct, unc_pct,
                                    unc_column="mahalanobis")
    meta = {**meta, "calibrator": "ts_only_mahalanobis",
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "conf_cut": build_meta.get("conf_cut"), "unc_cut": build_meta.get("unc_cut"),
            "conf_pct": conf_pct, "unc_pct": unc_pct,
            "uq_score": "mahalanobis",
            "maha_fit": {"label_idx": int(lab), "label": used_label,
                         "n_per_class": maha.n_per_class_,
                         "n_classes": int(len(maha.mu_))}}
    return rec, meta


def _register():
    from .interfaces import register_calibrator
    register_calibrator("ts_only")(ts_only_calibrate)
    register_calibrator("ts_only_mahalanobis")(ts_only_mahalanobis_calibrate)


_register()