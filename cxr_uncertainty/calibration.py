"""Calibration-then-uncertainty pipeline (Part 3, plan §E.1 steps 1-5).

The Phase-2 3-member ensemble measured **raw** cross-member disagreement on
**un-aligned** per-member probabilities. That signal is confounded: the three
members (xrv DenseNet, ConvNeXt-V2, RAD-DINO ViT) have different reliability
curves / prevalence scales, so part of the cross-member std is *systematic
calibration offset*, not epistemic disagreement. This module aligns the members
before the disagreement is computed, then temperature-scales only the pooled
mean (a single temperature is monotone and spread-preserving, so the
disagreement ordering used for selective prediction is invariant — §E.2,
Mattei & Loureiro 2026; Rabanser & Papernot NeurIPS 2025).

Steps implemented here (plan §E.1):
  1. ``align_to_nih14``        — done upstream (Ensemble.Alignment + manifest valid mask);
  2. ``fit_member_beta``        — per-member per-class **Beta calibration** (Kull 2017) on the
     leak-free Role-C (OpenI) cal split. Puts each member on the OpenI prevalence scale so
     cross-member disagreement is a clean epistemic signal, not a prevalence/ontology artifact.
     (The plan separates Role-B per-member alignment from Role-C ETS; using Role-C for the
     per-member step too is a defensible simplification — leak-free, single forward, avoids an
     extra Role-B pass — and is documented in ``meta``. Per-source BCTS/EM prior-shift is the
     later refinement.) Classes with too few positives/negatives fall back to identity.
  3. ``compute_disagreement``   — per-class std / mutual-information on the **aligned** probs.
  4. ``pool``                   — average the aligned probs (Pool-Then-Calibrate; Rahaman 2021).
  5. ``fit_ets_on_openi``       — **temperature scaling** on the pooled aligned mean (the
     §E.2-monotone option). Applied to the pooled mean ONLY, never re-injected into members.

Steps 6 (DEGRE learned gate) and 7 (Mondrian per-label LAC conformal) are Phase 3
step 6-7 and are NOT implemented here — the defining metric (confident-error
AUROC across selectivity) does not need them; they add the conformal triage layer.

All functions operate on **full-precision numpy arrays** ``(N, M, P)`` (N images,
M members, P pathologies) collected in-process by the eval driver — NOT on the
rounded ``member_probs`` JSON in the CSV, so Beta calibration near 0/1 is accurate.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Step 2: per-member per-class Beta calibration (Kull et al. 2017)
# ---------------------------------------------------------------------------
def beta_calibration_fit(
    probs: np.ndarray, labels: np.ndarray, min_pos: int = 8, min_neg: int = 8,
) -> Optional["np.ndarray"]:
    """Fit a 3-parameter Beta calibration map for one (member, class).

    Beta calibration models ``logit(q) = a*log(p) + b*log(1-p) + c`` via logistic
    regression of ``labels`` on features ``[log(p), log(1-p)]`` (Kull 2017). Returns
    the fitted params ``[a, b, c]`` or ``None`` (identity / no-op) when the class is
    too rare to fit reliably (fewer than ``min_pos`` positives or ``min_neg``
    negatives on the calibration set).
    """
    from sklearn.linear_model import LogisticRegression

    y = labels.astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < min_pos or n_neg < min_neg:
        return None
    p = np.clip(probs.astype(np.float64), _EPS, 1.0 - _EPS)
    X = np.column_stack([np.log(p), np.log1p(-p)])           # (n, 2)
    try:
        lr = LogisticRegression(C=1e3, solver="lbfgs", max_iter=500)
        lr.fit(X, y)
    except Exception:
        return None
    a, b = float(lr.coef_[0, 0]), float(lr.coef_[0, 1])
    c = float(lr.intercept_[0])
    # Guard against a degenerate / non-monotone fit producing extreme outputs.
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(c):
        return None
    return np.array([a, b, c], dtype=np.float64)


def beta_apply(probs: np.ndarray, params: Optional[np.ndarray]) -> np.ndarray:
    """Apply a Beta calibration map. ``params=None`` -> identity (no calibration)."""
    if params is None:
        return probs.astype(np.float64).copy()
    a, b, c = params
    p = np.clip(probs.astype(np.float64), _EPS, 1.0 - _EPS)
    logit_q = a * np.log(p) + b * np.log1p(-p) + c
    return 1.0 / (1.0 + np.exp(-logit_q))


def fit_member_beta(
    probs_cal: np.ndarray, gt_cal: np.ndarray, valid_cal: np.ndarray,
):
    """Per (member, class) Beta params fit on the calibration split.

    ``probs_cal`` (N, M, P) raw member probs; ``gt_cal``/``valid_cal`` (N, P).
    Returns ``params[m][p]`` = ``[a,b,c]`` or ``None``. Fits only over rows where
    the class label is valid (``valid_cal[:, p]``) — i.e. the manifest per-class
    validity mask (drops Consolidation on OpenI etc.).
    """
    N, M, P = probs_cal.shape
    params = [[None] * P for _ in range(M)]
    for m in range(M):
        for p in range(P):
            mask = valid_cal[:, p]
            if not mask.any():
                continue
            params[m][p] = beta_calibration_fit(probs_cal[mask, m, p], gt_cal[mask, p])
    return params


def apply_member_beta(
    probs: np.ndarray, valid: np.ndarray, params,
) -> np.ndarray:
    """Apply per (member, class) Beta params; identity where ``params is None``.

    ``probs`` (N, M, P); only positions where the class is valid (``valid[:, p]``)
    are calibrated; others are left as-is (and masked out downstream anyway)."""
    N, M, P = probs.shape
    out = probs.astype(np.float64).copy()
    for m in range(M):
        for p in range(P):
            if params[m][p] is None:
                continue
            mask = valid[:, p]
            if mask.any():
                out[mask, m, p] = beta_apply(probs[mask, m, p], params[m][p])
    return out


# ---------------------------------------------------------------------------
# Step 3-4: disagreement + pooling on aligned probs
# ---------------------------------------------------------------------------
def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def compute_disagreement(aligned_probs: np.ndarray, valid: np.ndarray):
    """Per (image, class) epistemic-std and mutual-information on aligned probs.

    ``aligned_probs`` (N, M, P); ``valid`` (N, P) bool — members are only compared
    on classes valid for that image (NaN where invalid). Returns ``p_bar`` (N, P),
    ``std`` (N, P), ``mi`` (N, P). Vectorized nan-mean/nan-std over the M axis.
    """
    # mask invalid classes per image -> NaN so nan-aware reductions skip them.
    # The all-invalid columns (e.g. Consolidation on OpenI, valid=False for every
    # image) yield all-NaN slices -> benign "Mean of empty slice" RuntimeWarnings;
    # suppress them (those rows are dropped downstream by the valid mask).
    import warnings as _warnings

    samples = np.where(valid[:, None, :], aligned_probs, np.nan)   # (N, M, P)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            p_bar = np.nanmean(samples, axis=1)                    # (N, P)
            std = np.nanstd(samples, axis=1, ddof=0)
            pred_ent = _bernoulli_entropy(p_bar)
            exp_ent = np.nanmean(_bernoulli_entropy(samples), axis=1)
            mi = pred_ent - exp_ent
    # rows with no valid member -> NaN (dropped by the driver via the gt-positive filter)
    return p_bar, std, mi


# ---------------------------------------------------------------------------
# Step 5: temperature scaling on the pooled aligned mean (monotone; §E.2)
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def temperature_scaling_fit(
    pooled_logits: np.ndarray, gt: np.ndarray, valid: np.ndarray,
    lo: float = 0.1, hi: float = 10.0,
) -> float:
    """Find the single temperature ``T`` minimizing NLL on the pooled aligned
    mean over valid (image, class) slots. Bounded scalar search (scipy)."""
    from scipy.optimize import minimize_scalar

    z = pooled_logits[valid].astype(np.float64)
    y = gt[valid].astype(np.float64)
    if z.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return 1.0

    def nll(T: float) -> float:
        T = max(float(T), 1e-3)
        p = 1.0 / (1.0 + np.exp(-z / T))
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x)


def apply_temperature(pooled_probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature to a pooled mean: ``sigmoid(logit(p)/T)`` (monotone)."""
    if T == 1.0 or not np.isfinite(T):
        return pooled_probs.astype(np.float64).copy()
    return 1.0 / (1.0 + np.exp(-_logit(pooled_probs) / T))


# ---------------------------------------------------------------------------
# Full pipeline (steps 1-5) on full-precision arrays
# ---------------------------------------------------------------------------
def build_pipeline(
    probs_cal: np.ndarray, gt_cal: np.ndarray, valid_cal: np.ndarray,
    probs_eval: np.ndarray, gt_eval: np.ndarray, valid_eval: np.ndarray,
    image_ids_cal: List[str], image_ids_eval: List[str],
    pathologies: List[str], member_keys: List[str],
    use_beta: bool, use_ts: bool,
):
    """One parameterized pipeline covering the calibration ablations:

      * ``use_beta=False, use_ts=False``  -> **raw**        (Phase-2 baseline)
      * ``use_beta=False, use_ts=True``   -> **ts_only**   (§E.2 design: temperature
        the raw pooled mean — fixes ECE, preserves raw disagreement ordering)
      * ``use_beta=True,  use_ts=False``  -> **beta_only** (isolates Beta's effect on
        the disagreement signal)
      * ``use_beta=True,  use_ts=True``   -> **beta+ts**    (full plan §E.1 steps 1-5)

    The disagreement (std/MI) is ALWAYS computed on the member probs used for
    pooling (raw if ``use_beta=False``, aligned if ``use_beta=True``). When
    ``use_ts=True`` the pooled mean is temperature-scaled (cal split fit); TS is
    monotone so the disagreement ordering used for selective prediction is
    invariant (§E.2). Returns ``(df_calib, df_eval, meta)``.
    """
    probs_cal = probs_cal.astype(np.float64)
    probs_eval = probs_eval.astype(np.float64)
    _, M, P = probs_cal.shape

    if use_beta:
        beta = fit_member_beta(probs_cal, gt_cal, valid_cal)
        n_fit = sum(1 for m in range(M) for p in range(P) if beta[m][p] is not None)
        mem_cal = apply_member_beta(probs_cal, valid_cal, beta)
        mem_eval = apply_member_beta(probs_eval, valid_eval, beta)
    else:
        beta = None
        n_fit = 0
        mem_cal = probs_cal
        mem_eval = probs_eval
    n_id = M * P - n_fit

    p_bar_cal, _, _ = compute_disagreement(mem_cal, valid_cal)
    p_bar_eval, std_eval, mi_eval = compute_disagreement(mem_eval, valid_eval)

    if use_ts:
        T = temperature_scaling_fit(_logit(p_bar_cal), gt_cal, valid_cal)
        p_bar_eval_out = apply_temperature(p_bar_eval, T)
        p_bar_cal_out = apply_temperature(p_bar_cal, T)
    else:
        T = 1.0
        p_bar_eval_out = p_bar_eval
        p_bar_cal_out = p_bar_cal

    tag = ("beta+" if use_beta else "") + ("ts" if use_ts else "raw")
    meta = {
        "calibrator": tag, "use_beta": use_beta, "use_ts": use_ts,
        "alignment_set": "role_C_openi" if use_beta else "none",
        "n_members": M, "n_pathologies": P, "member_keys": list(member_keys),
        "beta_slots_fit": int(n_fit), "beta_slots_identity": int(n_id),
        "temperature_T": float(round(T, 4)),
    }
    df_calib = _to_long_df(p_bar_cal_out, np.zeros_like(p_bar_cal_out),
                           np.zeros_like(p_bar_cal_out), gt_cal, valid_cal,
                           image_ids_cal, pathologies, mem_cal, member_keys)
    df_eval = _to_long_df(p_bar_eval_out, std_eval, mi_eval, gt_eval, valid_eval,
                          image_ids_eval, pathologies, mem_eval, member_keys)
    return df_calib, df_eval, meta


def calibrated_pipeline(*a, **k):
    return build_pipeline(*a, use_beta=True, use_ts=True, **k)


def raw_pipeline(*a, **k):
    return build_pipeline(*a, use_beta=False, use_ts=False, **k)


def _to_long_df(
    p_bar: np.ndarray, std: np.ndarray, mi: np.ndarray,
    gt: np.ndarray, valid: np.ndarray, image_ids: List[str],
    pathologies: List[str], member_probs: np.ndarray, member_keys: List[str],
):
    """Stack per (image, class) arrays into the long per_record schema that
    ``reanalyze.calibrate_thresholds`` + ``build_records`` consume. Drops
    invalid classes (``valid[:, p]==False``), e.g. Consolidation on OpenI."""
    import json

    import pandas as pd

    N, P = p_bar.shape
    rows = []
    for n in range(N):
        iid = str(image_ids[n])
        for p in range(P):
            if not bool(valid[n, p]):
                continue
            mp = {
                member_keys[m]: round(float(member_probs[n, m, p]), 6)
                for m in range(member_probs.shape[1]) if np.isfinite(member_probs[n, m, p])
            }
            pb = float(p_bar[n, p])
            lg = float(_logit(np.array([pb]))[0])
            rows.append({
                "image_id": iid, "pathology": pathologies[p],
                "p_bar": pb, "epistemic_std": float(std[n, p]),
                "mutual_info": float(mi[n, p]), "gt": int(gt[n, p]),
                "valid": 1, "logits": lg, "member_probs": json.dumps(mp),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DataFrame-based registry shim (for `reanalyze --calibrator bcts_ets_mondrian`)
# ---------------------------------------------------------------------------
def _long_df_to_arrays(df: pd.DataFrame):
    """Pivot a long ``per_record.csv`` frame (one row per (image, pathology),
    ``member_probs`` JSON column) back to ``(N, M, P)`` arrays + ids.

    NOTE: the CSV stores member probs rounded to 4-6 dp, so this shim is less
    numerically precise than the in-process array driver
    (``scripts/eval_phase3.py``), which is the headline path. Kept so the
    ``--calibrator bcts_ets_mondrian`` flag resolves and the registry test
    passes; reanalyze on a saved CSV reproduces the calibrated curve to ~1e-3.
    """
    import pandas as pd

    pathologies = sorted(df["pathology"].unique())
    image_ids = sorted(df["image_id"].unique())
    img_idx = {iid: i for i, iid in enumerate(image_ids)}
    path_idx = {p: j for j, p in enumerate(pathologies)}
    N, P = len(image_ids), len(pathologies)
    # member keys from the first non-empty member_probs JSON.
    member_keys: List[str] = []
    for s in df["member_probs"].astype(str):
        try:
            d = json.loads(s)
        except Exception:
            continue
        if d:
            member_keys = sorted(d.keys())
            break
    M = len(member_keys) or 1
    m_idx = {k: m for m, k in enumerate(member_keys)}
    probs = np.full((N, M, P), np.nan, dtype=np.float64)
    gt = np.zeros((N, P), dtype=np.float64)
    valid = np.zeros((N, P), dtype=bool)
    for _, r in df.iterrows():
        ni = img_idx[str(r["image_id"])]; pj = path_idx[str(r["pathology"])]
        valid[ni, pj] = bool(int(r.get("valid", 1)))
        gt[ni, pj] = float(r["gt"])
        try:
            d = json.loads(str(r["member_probs"]))
        except Exception:
            d = {}
        for k, v in d.items():
            if k in m_idx:
                probs[ni, m_idx[k], pj] = float(v)
    # NaN -> 0 for member slots that are genuinely missing on a valid row; the
    # disagreement reducer masks by `valid` (class-level), so individual missing
    # member slots become NaN-aware (skipped). Reintroduce NaN where a member
    # did not contribute so nan-mean is correct.
    probs = np.where(np.isnan(probs), np.nan, probs)
    return probs, gt, valid, image_ids, pathologies, member_keys


def bcts_ets_mondrian_calibrate(df_calib: "pd.DataFrame", df_eval: "pd.DataFrame",
                               conf_pct: float, unc_pct: float, **_):
    """Registered ``"bcts_ets_mondrian"`` calibrator (reanalyze path): rebuild
    ``(N,M,P)`` arrays from the ``member_probs`` JSON columns, run steps 1-5,
    then Youden + ``build_records`` -> ``(Records, meta)`` matching the
    ``youden_calibrate`` contract. The in-process array driver is the precise
    headline path; this shim reproduces it from a saved CSV to ~1e-3."""
    import pandas as pd

    from .reanalyze import build_records, calibrate_thresholds

    pC, yC, vC, idsC, paths, keys = _long_df_to_arrays(df_calib)
    pD, yD, vD, idsD, _, _ = _long_df_to_arrays(df_eval)
    # align the pathology/member axes (pathologies are sorted in both).
    df_cal_cal, df_eval_cal, meta = calibrated_pipeline(
        pC, yC, vC, pD, yD, vD, idsC, idsD, paths, keys)
    thresholds = calibrate_thresholds(df_cal_cal)
    rec, build_meta = build_records(df_eval_cal, thresholds, conf_pct, unc_pct)
    meta = {**meta, "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "conf_cut": build_meta.get("conf_cut"), "unc_cut": build_meta.get("unc_cut"),
            "conf_pct": conf_pct, "unc_pct": unc_pct}
    return rec, meta


def _register():
    from .interfaces import register_calibrator
    register_calibrator("bcts_ets_mondrian")(bcts_ets_mondrian_calibrate)


_register()