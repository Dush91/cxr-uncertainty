"""Mondrian per-label LAC conformal prediction + conformal triage (Part 3 step 7).

The production confident-error flag (Mahalanobis, ``production.py`` / plan §K) is a
*heuristic* detector — it catches confidently-wrong predictions but offers no
coverage guarantee. This module adds the **distribution-free coverage layer**
(plan §E.1 step 7 / §J.4): Mondrian per-label **LAC** (Least Ambiguous set-valued
Classifier, Liu & Lindsey 2022) conformal prediction with temperature scaling
(Dabah & Tirer ICML 2025: TS+LAC is the safest CP combination), fit on the
leak-free OpenI role-C cal split and evaluated on role-D. It delivers the plan's
Phase-3 acceptance: per-label coverage within ±2% of nominal 1−α **in
distribution**. It complements (does not replace) the Mahalanobis flag — conformal
gives an in-distribution coverage guarantee; the Mahalanobis flag is the
under-shift safety net (Phase 4 showed in-distribution calibration does not
transfer, so conformal coverage is expected to break under shift — an honest
negative the driver reports, paralleling Phase 4).

**Scope (confirmed with user): batch path + driver only.** The live per-image
``--risk-policy conformal_triage`` mirror (needs pre-fit τ_p from a sidecar) is
deferred; ``cli.py`` is untouched. The coverage guarantee is only demonstrable in
the batch fit-on-cal / eval-on-test path.

**LAC (binary per-pathology).** Nonconformity score ``s(x,y) = 1 − p̂(y|x)`` (prob
of the true class). Per pathology p, threshold ``τ_p = ⌈(1−α)(n_p+1)⌉``-th order
statistic of the cal scores (split-conformal finite-sample correction). For a new
example with calibrated ``p_bar``:

  * ``include_1 = p_bar >= 1 − τ_p``; ``include_0 = p_bar <= τ_p``.
  * LAC set size = ``include_1 + include_0`` ∈ {0,1,2} (0 = empty/"refer",
    1 = auto-read single label, 2 = ambiguous/"refer").
  * **Triage rule:** ``auto = (set_size == 1)``; ``refer = (set_size != 1)``.
  * **Coverage:** ``covered = (gt==1 & include_1) | (gt==0 & include_0)``;
    per-pathology ``P(covered) >= 1−α`` (Mondrian marginal guarantee). ``τ_p≥0.5``
    → middle band is ambiguous {0,1}; ``τ_p<0.5`` → middle band is empty {} —
    both are "refer."

The conformal sits on the **TS-only calibrated** pooled mean (reuse
``calibration.build_pipeline(use_ts=True)``, the §K production calibration). Fit
TS + τ_p on OpenI-C, evaluate coverage on OpenI-D: standard split-conformal (TS is
a fixed transform learned on C; applying it to D then conformalizing with
τ_p-from-C on D is valid — C and D are patient-disjoint/exchangeable; no
transductive adaptation, so the §E.4 SCA-T exchangeability concern does not apply).

**J.6b resolved by the data:** OpenI role-C has n=2002 per pathology (each image
contributes one row per pathology), so a per-pathology marginal LAC threshold is
well-powered (n≥300 for all 13 valid pathologies; only Consolidation is n=0 /
masked → excluded). The rareness is in *positives* (Mass=4, Fibrosis=7), which
only bites for *class-conditional* (pos/neg stratum) coverage — the
marginal-per-pathology LAC is fine. So **Mondrian = per-pathology** (one τ_p per
pathology, fit on all ~2002 rows); class-conditional coverage is reported as a
diagnostic, not guaranteed. Pathologies with ``0 < n_p < min_n`` use a
**pooled-rare fallback** (one shared ``τ_rare`` from the pooled rare scores).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .calibration import _long_df_to_arrays, apply_temperature, build_pipeline
from .reanalyze import build_records, calibrate_thresholds


# ---------------------------------------------------------------------------
# LAC nonconformity score + Mondrian per-pathology threshold fit
# ---------------------------------------------------------------------------
def lac_scores(p_bar: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """LAC nonconformity ``s(x,y) = 1 − p̂(y|x)`` for binary labels.

    ``p̂(1|x) = p_bar`` so for a positive the score is ``1 − p_bar``; for a
    negative ``p̂(0|x) = 1 − p_bar`` so the score is ``p_bar``. Both in [0,1];
    *small* = the model was confident in the TRUE class (conformant)."""
    p_bar = np.asarray(p_bar, dtype=np.float64)
    gt = np.asarray(gt, dtype=int)
    return np.where(gt == 1, 1.0 - p_bar, p_bar)


def fit_mondrian_lac(
    df_cal: pd.DataFrame, pathologies: Optional[List[str]] = None,
    alpha: float = 0.1, min_n: int = 300,
) -> Tuple[dict, List[str], float, dict]:
    """Per-pathology LAC threshold ``τ_p`` from the calibration split.

    For each pathology with ``n_p >= 1`` valid cal rows, ``τ_p`` is the
    ``⌈(1−α)(n_p+1)⌉``-th order statistic of that pathology's LAC scores
    (split-conformal finite-sample correction; ``⌈·⌉`` > ``n_p`` → ``τ_p=inf``,
    the conservative "include both / always refer" case when the cal set is too
    small for the α level). Pathologies with ``n_p == 0`` are excluded (no
    threshold; e.g. Consolidation on OpenI). Pathologies with
    ``0 < n_p < min_n`` are pooled into one shared ``τ_rare`` (the same order
    statistic of the pooled rare scores) — the documented rare-label fallback
    (plan §J.6b); none are expected on OpenI but coded generally.

    Returns ``(thresholds, rare_group, pooled_tau, meta)`` where ``thresholds``
    maps well-powered pathology → ``τ_p``, ``rare_group`` lists the rare
    pathologies covered by ``pooled_tau``, and ``meta`` documents the strategy.
    """
    thresholds: dict = {}
    rare_scores: List[float] = []
    rare_group: List[str] = []
    per_path_n: dict = {}
    present = set()
    for p, g in df_cal.groupby("pathology"):
        present.add(p)
        n = int(len(g))
        per_path_n[p] = n
        if n == 0:
            continue
        sc = lac_scores(g["p_bar"].values, g["gt"].values)
        if n < min_n:
            rare_scores.extend(sc.tolist())
            rare_group.append(p)
            continue
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        if k > n:
            tau = float(np.inf)
        else:
            tau = float(np.sort(sc)[k - 1])
        thresholds[p] = tau

    # Pathologies in the canonical list but absent from the cal frame (e.g.
    # Consolidation on OpenI, dropped by the valid mask before the long-form df)
    # are excluded — no threshold, not pooled. groupby never yields them, so we
    # detect absence against the canonical `pathologies` list.
    excluded = [p for p in (pathologies or []) if p not in present]

    pooled_tau = float(np.inf)
    if rare_scores:
        sc = np.sort(np.asarray(rare_scores, dtype=np.float64))
        n = int(len(sc))
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        pooled_tau = float(np.inf) if k > n else float(sc[k - 1])

    meta = {
        "alpha": float(alpha), "min_n": int(min_n),
        "rare_label_strategy": ("pooled_rare" if rare_group else "none"),
        "rare_group": list(rare_group), "pooled_tau_rare": (
            None if not np.isfinite(pooled_tau) else round(pooled_tau, 6)),
        "excluded_no_cal": excluded, "n_per_pathology": per_path_n,
        "n_thresholds": len(thresholds),
    }
    return thresholds, rare_group, pooled_tau, meta


# ---------------------------------------------------------------------------
# Apply LAC to the eval split + coverage
# ---------------------------------------------------------------------------
def _tau_for(pathology: str, thresholds: dict, rare_group: List[str],
             pooled_tau: float) -> Optional[float]:
    if pathology in thresholds:
        return thresholds[pathology]
    if pathology in rare_group:
        return pooled_tau
    return None


def apply_lac(df_eval: pd.DataFrame, thresholds: dict, rare_group: List[str],
              pooled_tau: float) -> pd.DataFrame:
    """Attach ``lac_set_size`` (int, −1 = not conformalized), ``lac_refer``
    (bool, set_size != 1) and ``lac_covered`` (bool, true label in set) to df_eval."""
    df = df_eval.reset_index(drop=True).copy()
    p_bar = df["p_bar"].values.astype(np.float64)
    gt = df["gt"].values.astype(int)
    paths = df["pathology"].values
    set_sizes = np.full(len(df), -1, dtype=int)
    refers = np.ones(len(df), dtype=bool)      # default: refer (excluded pathologies)
    covered = np.ones(len(df), dtype=bool)     # default: vacuously covered
    for i in range(len(df)):
        tau = _tau_for(str(paths[i]), thresholds, rare_group, pooled_tau)
        if tau is None:
            continue  # excluded pathology -> defaults (refer, covered)
        if not np.isfinite(tau):
            inc1, inc0 = True, True          # inf threshold -> include both
        else:
            inc1 = bool(p_bar[i] >= (1.0 - tau))
            inc0 = bool(p_bar[i] <= tau)
        ss = int(inc1) + int(inc0)
        set_sizes[i] = ss
        refers[i] = (ss != 1)
        covered[i] = bool((gt[i] == 1 and inc1) or (gt[i] == 0 and inc0))
    df["lac_set_size"] = set_sizes
    df["lac_refer"] = refers
    df["lac_covered"] = covered
    return df


def coverage_by_label(df_eval: pd.DataFrame, thresholds: dict,
                      rare_group: List[str], pooled_tau: float) -> dict:
    """Per-pathology + overall coverage, avg set size, refer rate. Only rows that
    were conformalized (``lac_set_size >= 0``) count toward the overall."""
    per_path: dict = {}
    for p, g in df_eval.groupby("pathology"):
        tau = _tau_for(str(p), thresholds, rare_group, pooled_tau)
        if tau is None:
            continue
        g = g[g["lac_set_size"] >= 0]
        if len(g) == 0:
            continue
        per_path[p] = {
            "n": int(len(g)),
            "coverage": float(g["lac_covered"].mean()),
            "avg_set_size": float(g["lac_set_size"].mean()),
            "refer_rate": float(g["lac_refer"].mean()),
            "tau_p": (None if not np.isfinite(tau) else round(float(tau), 6)),
        }
    conf = df_eval[df_eval["lac_set_size"] >= 0]
    overall = {
        "n": int(len(conf)),
        "coverage": float(conf["lac_covered"].mean()) if len(conf) else None,
        "avg_set_size": float(conf["lac_set_size"].mean()) if len(conf) else None,
        "refer_rate": float(conf["lac_refer"].mean()) if len(conf) else None,
    }
    return {"per_pathology": per_path, "overall": overall}


# ---------------------------------------------------------------------------
# In-process builder (full precision) — the driver / headline path
# ---------------------------------------------------------------------------
def build_conformal_pipeline(
    probs_cal: np.ndarray, gt_cal: np.ndarray, valid_cal: np.ndarray,
    ids_cal: List[str],
    probs_eval: np.ndarray, gt_eval: np.ndarray, valid_eval: np.ndarray,
    ids_eval: List[str],
    pathologies: List[str], member_keys: List[str],
    alpha: float = 0.1, min_n: int = 300,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The conformal variant, full precision (no CSV rounding).

    1. TS-only calibration: ``build_pipeline(use_beta=False, use_ts=True)`` —
       temperature-scale the raw pooled mean (fit on the cal split), the §K
       production calibration. The LAC score + threshold sit on this calibrated
       ``p_bar``.
    2. ``fit_mondrian_lac`` on the calibrated cal frame → per-pathology ``τ_p``
       (pooled-rare fallback for any ``n_p < min_n``).
    3. ``apply_lac`` on the calibrated eval frame → ``lac_set_size`` /
       ``lac_refer`` / ``lac_covered`` columns.
    4. ``coverage_by_label`` on the eval frame → per-pathology + overall coverage.

    Returns ``(df_calib, df_eval, meta)`` with ``meta["uq_score"]="conformal_lac"``,
    ``meta["conformal"]`` (the LAC fit meta) and ``meta["coverage"]`` (the eval
    coverage). Downstream ``build_records(risk_flag_override=df_eval["lac_refer"])``
    makes the conformal-refer flag the risk flag, and ``evaluate_records`` reports
    the conformal coverage/size/Brier (only when ``conformal_active``).
    """
    df_calib, df_eval, meta = build_pipeline(
        probs_cal, gt_cal, valid_cal, probs_eval, gt_eval, valid_eval,
        ids_cal, ids_eval, pathologies, member_keys,
        use_beta=False, use_ts=True)

    thresholds, rare_group, pooled_tau, lac_meta = fit_mondrian_lac(
        df_calib, pathologies=list(pathologies), alpha=alpha, min_n=min_n)
    df_eval = apply_lac(df_eval, thresholds, rare_group, pooled_tau)
    df_calib = apply_lac(df_calib, thresholds, rare_group, pooled_tau)
    coverage = coverage_by_label(df_eval, thresholds, rare_group, pooled_tau)

    meta = {**meta, "uq_score": "conformal_lac", "alpha": float(alpha),
            "conformal": lac_meta, "coverage": coverage,
            # Expose the fit (T + per-pathology tau_p) so a sidecar can be written
            # for the live per-image --risk-policy conformal_triage mirror (§M.2).
            # `temperature_T` already comes from build_pipeline; tau_p / rare_group
            # / pooled_tau are the fit_mondrian_lac outputs (JSON-serializable:
            # inf -> null).
            "tau_p": {p: (None if not np.isfinite(t) else round(float(t), 6))
                      for p, t in thresholds.items()},
            "rare_group": list(rare_group),
            "pooled_tau": (None if not np.isfinite(pooled_tau)
                          else round(float(pooled_tau), 6))}
    return df_calib, df_eval, meta


# ---------------------------------------------------------------------------
# Registered calibrator (CSV-feasible reanalyze path)
# ---------------------------------------------------------------------------
def conformal_triage_calibrate(df_calib: pd.DataFrame, df_eval: pd.DataFrame,
                               conf_pct: float, unc_pct: float,
                               alpha: float = 0.1, **_) -> Tuple:
    """Registered ``"conformal_triage"``: TS-only calibration + Mondrian per-label
    LAC conformal triage. Rebuilds ``(N,M,P)`` from the ``member_probs`` JSON
    columns, runs ``build_conformal_pipeline`` (TS + τ_p fit on cal, applied to
    eval), Youden thresholds (for decision/confidence/selectivity), then
    ``build_records(risk_flag_override=lac_refer)`` so the conformal-refer flag is
    the risk flag — apples-to-apples with the epistemic_std / Mahalanobis flags on
    the same confident population. Returns ``(Records, meta)`` with the coverage
    in ``meta["coverage"]``. Reproduces the in-process driver to ~1e-3 from a
    saved CSV."""
    pC, yC, vC, idsC, paths, keys = _long_df_to_arrays(df_calib)
    pD, yD, vD, idsD, _, _ = _long_df_to_arrays(df_eval)
    df_cal, df_eval_cal, meta = build_conformal_pipeline(
        pC, yC, vC, idsC, pD, yD, vD, idsD, paths, keys, alpha=alpha)
    thresholds = calibrate_thresholds(df_cal)
    rec, build_meta = build_records(
        df_eval_cal, thresholds, conf_pct, unc_pct,
        risk_flag_override=df_eval_cal["lac_refer"].values)
    meta = {**meta, "calibrator": "conformal_triage",
            "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
            "conf_cut": build_meta.get("conf_cut"), "unc_cut": build_meta.get("unc_cut"),
            "conf_pct": conf_pct, "unc_pct": unc_pct, "alpha": float(alpha),
            "uq_score": "conformal_lac"}
    return rec, meta


def _register():
    from .interfaces import register_calibrator, register_risk_policy
    register_calibrator("conformal_triage")(conformal_triage_calibrate)
    # the live per-image mirror of the batch conformal path (plan §M.2)
    register_risk_policy("conformal_triage")(assess_conformal_triage)


# ---------------------------------------------------------------------------
# Conformal sidecar (plan §M.2) — pre-fit T + per-pathology tau_p for the live
# per-image --risk-policy conformal_triage mirror of the batch §L path.
# ---------------------------------------------------------------------------
_SIDECAR_CACHE: dict = {"path": None, "data": None}


def save_conformal_sidecar(path, meta: dict, pathologies: List[str],
                           alpha: float, member_keys: Optional[List[str]] = None):
    """Write the conformal sidecar JSON: {T, tau_p, rare_group, pooled_tau, alpha,
    pathologies, member_keys}. ``meta`` is the ``build_conformal_pipeline`` meta
    (carries ``temperature_T`` / ``tau_p`` / ``rare_group`` / ``pooled_tau``).
    The sidecar is fit on a specific ensemble (T, tau_p depend on the p_bar
    distribution); the live ``--arch-ensemble`` must match (recorded for
    traceability). inf thresholds serialize to null."""
    out = {
        "temperature_T": float(meta["temperature_T"]),
        "tau_p": dict(meta["tau_p"]),
        "rare_group": list(meta["rare_group"]),
        "pooled_tau": meta["pooled_tau"],
        "alpha": float(alpha),
        "pathologies": list(pathologies),
        "member_keys": list(member_keys) if member_keys is not None else
                       list(meta.get("member_keys", [])),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    # invalidate the cache so a re-load picks up the new file
    _SIDECAR_CACHE["path"] = None
    _SIDECAR_CACHE["data"] = None
    return str(p)


def load_conformal_sidecar(path: str) -> dict:
    """Load + cache the conformal sidecar. ``None``/``inf`` tau_p values become
    ``np.inf`` (the conservative "include both / always refer" case). Module-level
    cache so the live per-image loop reads the file once."""
    if not path:
        raise ValueError("conformal sidecar path is empty")
    if _SIDECAR_CACHE["path"] != str(path):
        with open(path) as f:
            d = json.load(f)
        tau_p = {p: (np.inf if t is None else float(t))
                 for p, t in d["tau_p"].items()}
        pooled = np.inf if d.get("pooled_tau") is None else float(d["pooled_tau"])
        d["tau_p"] = tau_p
        d["pooled_tau"] = pooled
        d["temperature_T"] = float(d["temperature_T"])
        d["rare_group"] = list(d.get("rare_group", []))
        _SIDECAR_CACHE["path"] = str(path)
        _SIDECAR_CACHE["data"] = d
    return _SIDECAR_CACHE["data"]


def _tau_for_live(pathology: str, sidecar: dict) -> Optional[float]:
    """Per-pathology tau lookup for the live policy. Mirrors ``_tau_for``: a
    pathology in ``tau_p`` -> its threshold; a rare pathology -> pooled_tau; an
    absent/excluded pathology (e.g. Consolidation, not in the OpenI-C fit) ->
    None (not conformalized)."""
    tau_p = sidecar["tau_p"]
    if pathology in tau_p:
        return tau_p[pathology]
    if pathology in sidecar["rare_group"]:
        return sidecar["pooled_tau"]
    return None


def assess_conformal_triage(image_id: str, unc, cfg, gt_labels: Optional[dict] = None):
    """Registered ``"conformal_triage"`` RISK policy (plan §M.2): the live
    per-image mirror of the batch §L conformal path.

    Applies the pre-fit TS temperature **T** (from the sidecar, fit on OpenI-C)
    to the live ``unc.p_bar`` (the raw pooled mean over members — identical to
    ``calibration.compute_disagreement``'s p_bar for valid pathologies), then
    looks up the per-pathology ``tau_p`` **by name** and computes the LAC set
    (``include_1 = p_bar >= 1 - tau_p``; ``include_0 = p_bar <= tau_p``). The
    risk flag is the LAC ``refer`` decision (set_size != 1); ``lac_set_size`` /
    ``lac_refer`` / ``lac_covered`` are populated on each ``PathologyRisk`` so
    the live per_record.csv carries the conformal columns. Excluded pathologies
    (absent from the sidecar's tau_p, e.g. Consolidation) get set_size=-1,
    refer=True, covered=True (the conservative default).

    ``decision`` / ``confidence`` / ``is_confident`` / ``error_type`` follow the
    threshold policy (so the decision/confidence selectivity analysis is
    unchanged); only the *flag* switches to the LAC refer decision.
    """
    from .risk import PathologyRisk, ImageRiskReport, classify_error

    gt_labels = gt_labels or {}
    sidecar = load_conformal_sidecar(cfg.conformal_sidecar)
    T = sidecar["temperature_T"]
    p_bar_raw = np.asarray(unc.p_bar, dtype=np.float64)
    p_bar_cal = apply_temperature(p_bar_raw, T)

    rows = []
    flagged = []
    for i, p in enumerate(unc.pathologies):
        pbar_raw = float(p_bar_raw[i])
        if np.isnan(pbar_raw):
            continue  # no model covered this class
        pbar = float(p_bar_cal[i])
        tau = _tau_for_live(str(p), sidecar)
        if tau is None:
            # excluded pathology (not in the OpenI-C fit) -> not conformalized.
            set_size, refer, covered = -1, True, True
        elif not np.isfinite(tau):
            # inf threshold -> include both -> ambiguous -> refer.
            set_size, refer, covered = 2, True, True
        else:
            inc1 = bool(pbar >= (1.0 - tau))
            inc0 = bool(pbar <= tau)
            set_size = int(inc1) + int(inc0)
            refer = (set_size != 1)
            gt = gt_labels.get(p)
            covered = bool((gt == 1 and inc1) or (gt == 0 and inc0)) if gt is not None else True
        # decision/confidence/is_confident/error_type follow the threshold policy.
        decision = int(pbar_raw >= cfg.decision_thresh)
        conf = float(unc.confidence[i])
        estd = float(unc.epistemic_std[i])
        mi = float(unc.mutual_info[i])
        is_conf = conf >= cfg.conf_thresh
        gt = gt_labels.get(p)
        err = classify_error(decision, conf, gt, cfg.conf_thresh)
        risk = bool(refer)
        if risk:
            flagged.append(p)
        rows.append(PathologyRisk(
            pathology=p, p_bar=pbar_raw, decision=decision, confidence=conf,
            epistemic_std=estd, mutual_info=mi, is_confident=is_conf,
            risk_flag=risk, gt=gt, error_type=err,
            lac_set_size=set_size, lac_refer=refer, lac_covered=covered,
        ))
    return ImageRiskReport(
        image_id=image_id, pathologies=rows,
        any_high_risk=bool(flagged), high_risk_pathologies=flagged,
    )


_register()