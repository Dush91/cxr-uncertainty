"""Agent 4 -- Improvement Suggestion Agent (population-level).

Analyses the adapted-ensemble eval-split error set and produces improvement
hypotheses in five lanes: (a) which classes need more examples, (b) image
augmentation strategies, (c) threshold calibration, (d) retrain vs fine-tune,
(e) metadata stratification (slices). Same architecture as Agent 3: a
deterministic evidence pack (dotted EV ids) -> a constrained LLM renderer
(cite-everything prompt) -> a 0-tolerance mechanical faithfulness audit that
swaps any failing rendering for the deterministic template.

Honesty stance (hard, enforced by the audit):
  - every suggestion is correlational / hypothesis-generating; causal language
    is banned;
  - Agent 4 NEVER executes anything -- action phrasing is banned;
  - every number in prose must ground to a computed EV leaf;
  - threshold candidates are refit on the CALIBRATION split only (eval effects
    are reported descriptively; the eval split is never used for selection);
  - no pixels leave the machine and no image_path / patient_id is emitted
    (privacy in code; patient ids are factorized to int codes for clustered
    bootstraps and never appear in evidence).

All share-like quantities are stored 0-100 (`err_rate_pct = 20.6`), so a
renderer writing "20.6%" grounds numerically.

stdlib + numpy + pandas + scipy + scikit-learn only (no new pip deps).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

from . import agent3 as a3
from .config import NIH_PATHOLOGIES
from .reanalyze import _youden_threshold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "runs", "app_rex_adapted")
ARR_DIR = os.path.join(REPO, "runs", "rex_phase7", "eval_arrays", "rex_adapted_s0")
ARR_EVAL = os.path.join(ARR_DIR, "arrays_rex_adaptedeval.npz")
ARR_CAL = os.path.join(ARR_DIR, "arrays_rex_adaptedcal.npz")
CAL_JSON = os.path.join(CACHE_DIR, "calibration.json")
STATS_JSON = os.path.join(CACHE_DIR, "agent3_quality_stats.json")
MANIFEST = os.path.join(REPO, "data", "rex_manifest_fs.parquet")
OUT_DIR = os.path.join(CACHE_DIR, "agent4")
QUALITY_NPZ = os.path.join(OUT_DIR, "quality_cache.npz")
EVIDENCE_JSON = os.path.join(OUT_DIR, "suggestions_evidence.json")
FAITH_JSON = os.path.join(OUT_DIR, "faithfulness_report.json")
DEFAULT_REPORT = os.path.join(REPO, "docs", "agent4_improvement.md")

SCHEMA = "agent4_suggestion_evidence_v1"
DISCLAIMER = ("Population-level, correlational improvement hypotheses derived "
              "from a fixed eval split; not causal claims, not a validated "
              "intervention plan. Nothing here is executed -- suggestions only.")
SEED = 1234
N_BOOT = 1000
MIN_SLICE_CELLS = 200
MIN_SLICE_ERR_ROWS = 30
MIN_POS_FOR_SENS = 30
NEAR_THR = 0.05

LANES = ("class_data_need", "augmentation_gap", "threshold_levers",
         "retrain_vs_ft", "stratification_slices")

# Lane (b): metric -> which extreme band is the hypothesised failure mode.
_LOW_EXTREME = ("contrast_span", "contrast_std", "laplacian_var", "tenengrad")
_HIGH_EXTREME = ("midgray_dev", "clip_frac_total", "mirror_asymmetry",
                 "centroid_offset")
QUALITY_METRICS = _LOW_EXTREME + _HIGH_EXTREME


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_inputs(arr_eval=ARR_EVAL, arr_cal=ARR_CAL, manifest=MANIFEST,
                calibration=CAL_JSON, stats=STATS_JSON):
    """Load npz + calibration + agent3 quality stats + manifest view/patient.

    patient_id is factorized to int codes here and NEVER leaves the module."""
    import pandas as pd

    ev = dict(np.load(arr_eval, allow_pickle=True))
    cal = dict(np.load(arr_cal, allow_pickle=True))
    with open(calibration) as f:
        calj = json.load(f)
    stats = a3.load_quality_stats(stats)

    man = pd.read_parquet(manifest, columns=["image_id", "image_path", "view",
                                             "patient_id", "split_role"])
    man = man[man["split_role"] == "eval"]
    by_id = man.set_index("image_id")
    ids = [str(x) for x in ev["ids"]]
    views, pat_codes = [], []
    codes: dict[str, int] = {}
    paths = {}
    for iid in ids:
        if iid in by_id.index:
            row = by_id.loc[iid]
            v = row.get("view")
            views.append("nan" if (v is None or pd.isna(v)) else str(v))
            pid = str(row.get("patient_id"))
            if pid not in codes:
                codes[pid] = len(codes)
            pat_codes.append(codes[pid])
            paths[iid] = str(row["image_path"])
        else:
            views.append("nan")
            codes["__missing__%d" % len(codes)] = len(codes)
            pat_codes.append(codes[list(codes)[-1]])
    inp = {
        "probs": np.asarray(ev["probs"], dtype=np.float64),
        "gt": np.asarray(ev["gt"], dtype=np.float64),
        "valid": np.asarray(ev["valid"], dtype=bool),
        "rad_feats": np.asarray(ev["rad_feats"], dtype=np.float64),
        "maha": np.asarray(ev["maha"] if "maha" in ev else ev["mahalanobis"],
                           dtype=np.float64),
        "ids": ids,
        "pathologies": [str(p) for p in ev["pathologies"]],
        "cal_probs": np.asarray(cal["probs"], dtype=np.float64),
        "cal_gt": np.asarray(cal["gt"], dtype=np.float64),
        "cal_valid": np.asarray(cal["valid"], dtype=bool),
        "cal_ids": [str(x) for x in cal["ids"]],
        "youden": {k: (None if v is None else float(v))
                   for k, v in calj["youden"].items()},
        "temperature": float(calj.get("temperature", 1.0)),
        "view": np.asarray(views),
        "pat_codes": np.asarray(pat_codes, dtype=np.int64),
        "paths": paths,
        "stats": stats,
        "calibration_sha": hashlib.sha1(
            open(calibration, "rb").read()).hexdigest()[:8]
        if os.path.exists(calibration) else None,
        "arr_eval": os.path.abspath(arr_eval),
    }
    return inp


# ---------------------------------------------------------------------------
# Shared math
# ---------------------------------------------------------------------------

def build_error_set(inp):
    """Single source of truth for all lanes: pooled probs, thresholds, the
    per-cell pred/fp/fn masks and the error set. Youden thresholds with the
    0.5 fallback are identical to the demo app's predict()."""
    probs = inp["probs"]
    pbar = np.nanmean(probs, axis=1)                     # (N, P)
    pats = inp["pathologies"]
    thr = np.array([0.5 if inp["youden"].get(p) is None
                    else float(inp["youden"][p]) for p in pats])
    pred = np.where(np.isfinite(pbar), pbar >= thr[None, :], False)
    gt, valid = inp["gt"], inp["valid"]
    labelled = valid & np.isfinite(gt) & (gt >= 0)
    gt_eff = np.where(labelled, gt, -1.0)
    gt_pos = labelled & (gt_eff > 0)
    gt_neg = labelled & (gt_eff == 0)
    fp = pred & gt_neg
    fn = (~pred) & gt_pos
    err = fp | fn
    return {"pbar": pbar, "thr": thr, "pred": pred, "labelled": labelled,
            "gt_eff": gt_eff, "gt_pos": gt_pos, "gt_neg": gt_neg,
            "fp": fp, "fn": fn, "err": err,
            "err_rows": np.where(err.any(axis=1))[0],
            "member_std": np.nanstd(probs, axis=1)}


def _pat_sums(mask, pat_codes, n_pat):
    return np.bincount(pat_codes, weights=np.asarray(mask, dtype=np.float64),
                       minlength=n_pat)


def _rate_ci(num_pat, den_pat, n_boot, seed):
    """Patient-cluster bootstrap CI of num/den (resample patient codes, not
    rows, so multi-image patients cannot produce anti-conservative CIs)."""
    den = float(den_pat.sum())
    if den <= 0:
        return None
    point = float(num_pat.sum()) / den
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num_pat), size=(n_boot, len(num_pat)))
    nb = num_pat[idx].sum(axis=1)
    db = den_pat[idx].sum(axis=1)
    rates = np.where(db > 0, nb / np.maximum(db, 1e-12), np.nan)
    lo, hi = np.nanpercentile(rates, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def _lift_ci(num_a, den_a, num_b, den_b, n_boot, seed):
    """Patient-cluster bootstrap CI of the ratio (num_a/den_a)/(num_b/den_b)."""
    da, db = float(den_a.sum()), float(den_b.sum())
    if da <= 0 or db <= 0:
        return None
    pa, pb = float(num_a.sum()) / da, float(num_b.sum()) / db
    if pb <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num_a), size=(n_boot, len(num_a)))
    na, da_ = num_a[idx].sum(axis=1), den_a[idx].sum(axis=1)
    nb, db_ = num_b[idx].sum(axis=1), den_b[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        lifts = np.where((da_ > 0) & (db_ > 0) & (nb > 0),
                         (na / np.maximum(da_, 1e-12)) / (nb / np.maximum(db_, 1e-12)),
                         np.nan)
    lo, hi = np.nanpercentile(lifts, [2.5, 97.5])
    return {"lift": pa / pb, "lo": float(lo), "hi": float(hi)}


def _auroc(y, s):
    """Rank-based AUROC (tie-corrected); None when a class side is empty."""
    y = np.asarray(y)
    s = np.asarray(s)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _band(value, table, clip_frac=None):
    return a3.assign_band(value, table, clip_frac=clip_frac)


def _row_bands(inp, qtab):
    """Per-row quintile bands for every quality metric, against the committed
    per-view percentile tables (agent3_quality_stats.json)."""
    by_view = ((inp.get("stats") or {}).get("by_view") or {})
    out = {}
    for m in QUALITY_METRICS:
        bands = np.full(len(inp["ids"]), "unavailable", dtype=object)
        for v in np.unique(inp["view"]):
            tab = (by_view.get(str(v)) or {}).get(m)
            sel = inp["view"] == v
            vals = np.array([qtab.get(iid, {}).get(m, np.nan) for iid in
                             np.asarray(inp["ids"])[sel]])
            for k, val in zip(np.where(sel)[0], vals):
                cf = None
                if m == "clip_frac_total" and np.isfinite(val):
                    cf = val
                bands[k] = _band(val, tab, clip_frac=cf)
        out[m] = bands
    return out


def _extreme(band_str, metric):
    if metric in _LOW_EXTREME:
        return band_str in ("very low", "low")
    if metric == "clip_frac_total":
        return band_str in ("high clipping", "very high", "high")
    return band_str in ("very high", "high")


# ---------------------------------------------------------------------------
# Lane (a) -- which classes need more examples
# ---------------------------------------------------------------------------

def lane_class_data_need(inp, es, n_boot=N_BOOT, seed=SEED):
    pats = inp["pathologies"]
    labelled, err, fp, fn, gt_eff = (es["labelled"], es["err"], es["fp"],
                                     es["fn"], es["gt_eff"])
    n_pat = int(inp["pat_codes"].max()) + 1
    sp = ((inp.get("stats") or {}).get("bias") or {}).get("split_prevalence") or {}

    def prev(role, c):
        v = (sp.get(role) or {}).get(pats[c])
        return None if v is None else float(v) * 100.0

    rows_err = err.any(axis=1)
    ambiguous_row = (inp["valid"] & (inp["gt"] == -1)).any(axis=1)
    n_rows = len(inp["ids"])
    sec: dict = {
        "rows_with_any_ambiguous_label_pct":
            round(float(ambiguous_row.mean()) * 100.0, 2),
        "zero_prevalence_classes": [p for p in pats if not labelled[:, pats.index(p)].any()],
        "note_zero_prev": ("these classes have no labelled eval cells, so "
                           "sensitivity is unmeasurable on this split; this "
                           "is itself a dataset-composition finding"),
    }
    meas = []
    per = {}
    for c, p in enumerate(pats):
        lab = labelled[:, c]
        npos, nneg = int((gt_eff[:, c] > 0).sum()), int((gt_eff[:, c] == 0).sum())
        namb = int((inp["valid"][:, c] & (gt_eff[:, c] == -1)).sum())
        cells = int(lab.sum())
        zero_prev = cells == 0
        e = {
            "npos_eval": npos, "nneg_eval": nneg,
            "n_ambiguous_eval": namb, "labelled_cells": cells,
            "prevalence_train_pct": prev("train", c),
            "prevalence_cal_pct": prev("cal", c),
            "prevalence_eval_pct": prev("eval", c),
        }
        pt, pe = e["prevalence_train_pct"], e["prevalence_eval_pct"]
        e["prev_drift_train_to_eval_pct"] = (
            round((pe - pt), 2) if (pt is not None and pe is not None) else None)
        e["fn_n"] = int(fn[:, c].sum())
        e["fp_n"] = int(fp[:, c].sum())
        e["err_rate_pct"] = round(float(err[:, c].sum()) / cells * 100.0, 2) if cells else None
        e["fn_rate_pct"] = round(e["fn_n"] / npos * 100.0, 2) if npos else None
        e["fp_rate_pct"] = round(e["fp_n"] / nneg * 100.0, 2) if nneg else None
        if npos >= MIN_POS_FOR_SENS:
            num = _pat_sums(es["gt_pos"][:, c] & es["pred"][:, c],
                            inp["pat_codes"], n_pat)
            den = _pat_sums(es["gt_pos"][:, c], inp["pat_codes"], n_pat)
            ci = _rate_ci(num, den, n_boot, seed + c)
            if ci:
                e["sensitivity_pct"] = round(ci["point"] * 100.0, 2)
                e["sensitivity_ci_lo_pct"] = round(ci["lo"] * 100.0, 2)
                e["sensitivity_ci_hi_pct"] = round(ci["hi"] * 100.0, 2)
        e["zero_prevalence_eval"] = bool(zero_prev)
        e["measurable"] = not zero_prev
        per[p] = e
        if not zero_prev:
            meas.append(p)
    # support rank + need score over measurable classes only
    by_pos = sorted(meas, key=lambda p: (per[p]["npos_eval"], p))
    for r, p in enumerate(by_pos, 1):
        per[p]["support_rank"] = r
    for p in pats:
        if p not in meas:
            per[p]["support_rank"] = None
    if meas:
        def _z(vals, x):
            ranks = sorted(range(len(vals)), key=lambda i: vals[i])
            return (ranks.index(vals.index(x)) / max(len(vals) - 1, 1)) * 2.0 - 1.0
        comps = {p: (
            _z([round(per[q]["fp_rate_pct"] or 0.0, 4) for q in meas], round(per[p]["fp_rate_pct"] or 0.0, 4)),
            _z([round(per[q]["fn_rate_pct"] or 0.0, 4) for q in meas], round(per[p]["fn_rate_pct"] or 0.0, 4)),
            _z([round(1.0 / (per[q]["npos_eval"] + 5.0), 6) for q in meas],
               round(1.0 / (per[p]["npos_eval"] + 5.0), 6)),
            _z([round(per[q]["prev_drift_train_to_eval_pct"] or 0.0, 4) for q in meas],
               round(per[p]["prev_drift_train_to_eval_pct"] or 0.0, 4)),
        ) for p in meas}
        scores = {p: round(sum(comps[p]), 4) for p in meas}
        order = sorted(meas, key=lambda p: scores[p])
        tert = {p: ("high" if scores[p] >= scores[order[2 * len(order) // 3]]
                    else "low" if scores[p] <= scores[order[len(order) // 3]]
                    else "medium") for p in meas}
        for p in meas:
            per[p]["need_score"] = scores[p]
            per[p]["need_band"] = tert[p]
    for p in pats:
        if p not in meas:
            per[p]["need_score"] = None
            per[p]["need_band"] = "unmeasurable"
    sec.update(per)
    return sec


# ---------------------------------------------------------------------------
# Lane (b) -- augmentation strategies
# ---------------------------------------------------------------------------

def _quality_cache_load(inp):
    if not os.path.exists(QUALITY_NPZ):
        return None
    try:
        z = np.load(QUALITY_NPZ, allow_pickle=True)
        if str(z["arr_mtime"]) != str(os.path.getmtime(ARR_EVAL)):
            return None
        ids = [str(x) for x in z["ids"]]
        keys = [str(k) for k in z["metric_keys"]]
        arrs = {k: np.asarray(z["m_" + k], dtype=np.float64) for k in keys}
        return {iid: {k: float(arrs[k][i]) for k in keys}
                for i, iid in enumerate(ids)}
    except Exception:                                # noqa: BLE001 -- cache
        return None


def compute_quality_table(inp, mode="auto", log=print):
    """agent3.compute_quality for all eval rows, cached to quality_cache.npz.
    mode: 'auto' (use/extend cache), 'skip' (cache only), 'never' (None)."""
    if mode == "never":
        return None
    cached = _quality_cache_load(inp)
    ids = list(inp["ids"])
    if mode == "skip":
        return cached
    todo = [iid for iid in ids
            if cached is None or iid not in cached]
    if cached is not None and not todo:
        return cached
    tab = dict(cached or {})
    paths = inp.get("paths") or {}
    for n, iid in enumerate(todo):
        p = paths.get(iid)
        if not p or not os.path.exists(p):
            continue
        try:
            tab[iid] = a3.compute_quality(p)
        except Exception as e:                       # noqa: BLE001 -- one image
            log("  [quality] skip %s: %s" % (iid, e))
    os.makedirs(OUT_DIR, exist_ok=True)
    keys = sorted({k for v in tab.values() for k in v})
    np.savez(QUALITY_NPZ,
             ids=np.asarray(list(tab.keys()), dtype=object),
             metric_keys=np.asarray(keys, dtype=object),
             arr_mtime=str(os.path.getmtime(ARR_EVAL)),
             **{"m_" + k: np.array([tab[iid].get(k, np.nan)
                                    for iid in tab], dtype=np.float64)
                for k in keys})
    return tab


def lane_augmentation_gap(inp, es, qtab, n_boot=N_BOOT, seed=SEED):
    """Measured failure modes (error vs non-error films) + the deterministic
    repo-gap statement + the trigger->candidate map."""
    sec: dict = {
        "policy": {
            "current_train_aug": "horizontal flip p=0.5 + rotation +/-10 deg "
                                 "(inline torch ops in train_rex_adapted.py, "
                                 "all four members)",
            "convnextv2_exception": "horizontal flip only "
                                    "(train_convnextv2_lpft.py)",
            "absent_from_repo": "CLAHE/contrast, blur/sharpness, vertical "
                                "flip, brightness or contrast jitter",
            "test_time_transforms": "none beyond resize/normalize (no CLAHE "
                                    "at inference)",
        }
    }
    if qtab is None:
        sec["quality"] = {"note": "quality sweep skipped; no measured signal"}
        sec["candidates"] = [{"trigger": "no measured quality signal available",
                              "metric": "none",
                              "suggestion": "no image-quality lever identified",
                              "literature_anchor": "none"}]
        return sec

    bands = _row_bands(inp, qtab)
    err_rows = es["err_rows"]
    is_err = np.zeros(len(inp["ids"]), dtype=bool)
    is_err[err_rows] = True
    lab_row = es["labelled"].any(axis=1)
    ctl = (~is_err) & lab_row
    n_pat = int(inp["pat_codes"].max()) + 1
    err_pat = _pat_sums(is_err, inp["pat_codes"], n_pat)

    # view-stratified importance weights for the non-error comparison set
    vsel = inp["view"]
    w = np.zeros(len(inp["ids"]), dtype=np.float64)
    for v in np.unique(vsel):
        e_sh = float((is_err & (vsel == v)).sum())
        c_sh = float((ctl & (vsel == v)).sum())
        w[(vsel == v) & ctl] = (e_sh / c_sh) if c_sh > 0 else 0.0

    quality = {}
    cands = []
    for m in QUALITY_METRICS:
        ext = np.array([_extreme(b, m) for b in bands[m]])
        err_share = float(ext[is_err].mean()) * 100.0 if is_err.any() else None
        num_w = _pat_sums(ext & ctl, inp["pat_codes"], n_pat)
        den_w = _pat_sums(ctl, inp["pat_codes"], n_pat)
        # weighted share: weight each ctl cell by its view-stratified weight
        wsum_pat = np.bincount(inp["pat_codes"], weights=w, minlength=n_pat)
        xwsum_pat = np.bincount(inp["pat_codes"],
                                weights=w * ext.astype(np.float64),
                                minlength=n_pat)
        den = float(w.sum())
        ctl_share = float((w * ext).sum() / den) if den > 0 else None
        ci = None
        if err_share is not None and ctl_share is not None and ctl_share > 0:
            rng = np.random.default_rng(seed + QUALITY_METRICS.index(m))
            idx = rng.integers(0, n_pat, size=(n_boot, n_pat))
            eb = err_pat[idx].sum(axis=1) / max(float(err_pat.sum()), 1.0)
            xb = xwsum_pat[idx].sum(axis=1)
            db = wsum_pat[idx].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                lifts = np.where(db > 0, eb / (xb / np.maximum(db, 1e-12)), np.nan)
            lo, hi = np.nanpercentile(lifts, [2.5, 97.5])
            point = err_share / ctl_share
            if hi < 1.0 and abs(point - 1.0) >= 0.15:
                sig = "strong"
            elif hi < 1.0:
                sig = "weak"
            else:
                sig = "none"
            ci = {"lift": point, "lo": float(lo), "hi": float(hi), "signal": sig}
        quality[m] = {
            "err_rows_n": int(is_err.sum()), "ctl_rows_n": int(ctl.sum()),
            "err_extreme_pct": None if err_share is None else round(err_share, 2),
            "ctl_extreme_pct": None if ctl_share is None else round(ctl_share, 2),
            "lift": None if ci is None else round(ci["lift"], 3),
            "ci_lo": None if ci is None else round(ci["lo"], 3),
            "ci_hi": None if ci is None else round(ci["hi"], 3),
            "signal": None if ci is None else ci["signal"],
            "direction": "low" if m in _LOW_EXTREME else "high",
        }
        if ci and ci["signal"] != "none":
            trig = ("error films land in the extreme %s band more often"
                    % ("low" if m in _LOW_EXTREME else "high"))
            if m in ("contrast_span", "contrast_std", "midgray_dev",
                     "clip_frac_total"):
                cands.append({
                    "trigger": trig, "metric": m,
                    "suggestion": "hypothesis: add contrast-window augmentation "
                                  "at train time (e.g. CLAHE clipLimit 2.0, 8x8, "
                                  "LAB L-channel) and never fixed CLAHE at test "
                                  "time",
                    "literature_anchor": "PELM 2025 (clipLimit 2.0, 8x8, LAB L); "
                                         "XM-pipeline Eur Radiol Exp 2023 "
                                         "(fixed test-time CLAHE hurt cross-domain "
                                         "AUC 0.705 vs 0.944)"})
                cands.append({
                    "trigger": "guard for the same trigger", "metric": m,
                    "suggestion": "hypothesis: never apply fixed CLAHE as a "
                                  "test-time normalization",
                    "literature_anchor": "XM-pipeline Eur Radiol Exp 2023"})
            if m in ("laplacian_var", "tenengrad"):
                cands.append({
                    "trigger": trig, "metric": m,
                    "suggestion": "hypothesis: add a blur/sharpness augmentation "
                                  "family during training",
                    "literature_anchor": "XM-pipeline Eur Radiol Exp 2023 "
                                         "(physics-based contrast/sharpness/noise "
                                         "augmentation)"})
            if m == "mirror_asymmetry":
                cands.append({
                    "trigger": trig, "metric": m,
                    "suggestion": "hypothesis: keep horizontal flip, keep "
                                  "rotation narrow (+/-5-15 deg), no vertical flip",
                    "literature_anchor": "PELM 2025 (+/-5 deg protects the "
                                         "mediastinum)"})
    if not cands:
        cands.append({"trigger": "no quality metric separates error from "
                                 "non-error films",
                      "metric": "none",
                      "suggestion": "no image-quality lever identified on this "
                                    "split",
                      "literature_anchor": "none"})
    sec["quality"] = quality
    sec["candidates"] = cands
    return sec


# ---------------------------------------------------------------------------
# Lane (c) -- threshold calibration (refit on CAL only; eval is descriptive)
# ---------------------------------------------------------------------------

def _f2(tp, fp, fn, beta=2.0):
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return ((1 + b2) * tp / den) if den > 0 else 0.0


def _refit_thresholds(cal_pbar, cal_gt, cal_valid, pats, eval_prev):
    """Candidates fit on the CALIBRATION split ONLY. Returns per class
    {kind: thr} with nulls where the cal split is degenerate."""
    out = {}
    lab_all = cal_valid & np.isfinite(cal_gt) & (cal_gt >= 0)
    for c, p in enumerate(pats):
        lab = lab_all[:, c]
        y = cal_gt[lab, c]
        s = cal_pbar[lab, c]
        cand = {"youden_refit": None, "fbeta2_refit": None, "prev_matched": None}
        if len(np.unique(y)) >= 2:
            cand["youden_refit"] = _youden_threshold(y, s)
            grid = np.unique(np.concatenate([
                np.quantile(s, np.linspace(0, 1, 101)), [0.0, 1.0]]))
            best, best_thr = -1.0, None
            for t in grid:
                pr = s >= t
                tp = int((pr & (y == 1)).sum())
                fp_ = int((pr & (y == 0)).sum())
                fn_ = int((~pr & (y == 1)).sum())
                sc = _f2(tp, fp_, fn_)
                if sc > best:
                    best, best_thr = sc, float(t)
            cand["fbeta2_refit"] = best_thr
        ev_prev = eval_prev.get(p)
        if ev_prev is not None and 0.0 < ev_prev < 1.0 and s.size:
            cand["prev_matched"] = float(np.clip(
                np.quantile(s, 1.0 - ev_prev), 1e-4, 1.0 - 1e-4))
        out[p] = cand
    return out


def lane_threshold_levers(inp, es, n_boot=N_BOOT, seed=SEED):
    pats = inp["pathologies"]
    cal_pbar = np.nanmean(inp["cal_probs"], axis=1)
    eval_prev = {}
    for c, p in enumerate(pats):
        cells = int(es["labelled"][:, c].sum())
        eval_prev[p] = (float((es["gt_pos"][:, c]).sum()) / cells) if cells else None
    cands = _refit_thresholds(cal_pbar, inp["cal_gt"], inp["cal_valid"], pats,
                              eval_prev)

    sec: dict = {"eval_used_for_selection": False,
                 "precedent_note": "prior calibration work fit thresholds for "
                                   "Mass, Pneumothorax and Fibrosis on 4, 10 "
                                   "and 7 positives respectively and found "
                                   "them noise-driven; the same small-n "
                                   "mechanism applies when npos is small"}
    cal_lab = inp["cal_valid"] & np.isfinite(inp["cal_gt"]) & (inp["cal_gt"] >= 0)
    total_err = int(es["err"].sum())
    total_near = 0
    per = {}
    for c, p in enumerate(pats):
        cur = inp["youden"].get(p)
        prod_thr = 0.5 if cur is None else float(cur)
        lab_c = cal_lab[:, c]
        y = inp["cal_gt"][lab_c, c]
        s = cal_pbar[lab_c, c]
        cal_npos = int((y == 1).sum())
        cal_nneg = int((y == 0).sum())

        def counts(t, yy, ss):
            pr = ss >= t
            tp = int((pr & (yy == 1)).sum())
            fp_ = int((pr & (yy == 0)).sum())
            fn_ = int((~pr & (yy == 1)).sum())
            return tp, fp_, fn_

        rows = {}
        if cal_npos and cal_nneg:
            for kind, t in cands[p].items():
                if t is None:
                    continue
                tp, fp_, fn_ = counts(t, y, s)
                rows[kind] = {"thr": float(t), "tp": tp, "fp": fp_, "fn": fn_}
        best_kind, best_thr, best_f2 = None, None, -1.0
        for kind in ("youden_refit", "fbeta2_refit", "prev_matched"):
            if kind in rows:
                tp, fp_, fn_ = rows[kind]["tp"], rows[kind]["fp"], rows[kind]["fn"]
                sc = _f2(tp, fp_, fn_)
                if sc > best_f2:
                    best_kind, best_thr, best_f2 = kind, rows[kind]["thr"], sc
        cur_tp, cur_fp, cur_fn = counts(prod_thr, y, s)
        # eval side (descriptive)
        pbar = es["pbar"][:, c]
        eval_fp = int(es["fp"][:, c].sum())
        eval_fn = int(es["fn"][:, c].sum())
        err_mask = es["err"][:, c]
        lab_mask = es["labelled"][:, c]
        near_err = float(np.mean(np.abs(pbar[err_mask] - prod_thr) <= NEAR_THR)) \
            if err_mask.any() else None
        near_all = float(np.mean(np.abs(pbar[lab_mask] - prod_thr) <= NEAR_THR)) \
            if lab_mask.any() else None
        if near_err is not None:
            total_near += int((np.abs(pbar[err_mask] - prod_thr) <= NEAR_THR).sum())
        ratio = (eval_fp / eval_fn) if eval_fn else None
        asym = ("fp_dominated" if ratio and ratio >= 2.0 else
                "fn_dominated" if ratio and ratio <= 0.5 else "balanced")
        e = {
            "current_youden_thr": cur,
            "production_thr": round(prod_thr, 4),
            "null_threshold_in_production": cur is None,
            "cal_npos": cal_npos, "cal_nneg": cal_nneg,
            "cal_optimal_kind": best_kind,
            "cal_optimal_thr": None if best_thr is None else round(best_thr, 4),
            "cal_fp_at_current": cur_fp, "cal_fn_at_current": cur_fn,
            "cal_fp_at_candidate": (rows[best_kind]["fp"] if best_kind else None),
            "cal_fn_at_candidate": (rows[best_kind]["fn"] if best_kind else None),
            "eval_fp_n": eval_fp, "eval_fn_n": eval_fn,
            "fp_fn_ratio": None if ratio is None else round(ratio, 2),
            "asymmetry_band": asym,
            "near_threshold_err_pct": None if near_err is None else round(near_err * 100.0, 2),
            "near_threshold_all_pct": None if near_all is None else round(near_all * 100.0, 2),
        }
        per[p] = e
    if total_err:
        sec["threshold_reachable_err_cells_pct"] = round(total_near / total_err * 100.0, 2)
    fpd = max((p for p in pats if per[p]["fp_fn_ratio"] is not None),
              key=lambda p: per[p]["fp_fn_ratio"], default=None)
    sec["biggest_fp_dominated_class"] = fpd
    sec.update(per)
    return sec


# ---------------------------------------------------------------------------
# Lane (d) -- retrain vs fine-tune vs calibrate
# ---------------------------------------------------------------------------

def lane_retrain_vs_ft(inp, es, lane_a=None, lane_c=None, lane_b=None,
                       n_boot=N_BOOT, seed=SEED):
    pats = inp["pathologies"]
    labelled, gt_eff, pbar = es["labelled"], es["gt_eff"], es["pbar"]
    n_pat = int(inp["pat_codes"].max()) + 1
    err_rows = es["err_rows"]
    is_err = np.zeros(len(inp["ids"]), dtype=bool)
    is_err[err_rows] = True
    maha = inp["maha"]
    p90 = float(np.nanpercentile(maha, 90))
    med_err = float(np.nanmedian(maha[is_err])) if is_err.any() else None
    med_all = float(np.nanmedian(maha))
    sec: dict = {
        "ood_separation_median_pct": (
            round((med_err - med_all) / med_all * 100.0, 2)
            if med_err is not None and med_all else None),
        "ood_p90_share_err_pct": round(float((maha[is_err] >= p90).mean()) * 100.0, 2)
        if is_err.any() else None,
        "ood_p90_share_all_pct": round(float((maha >= p90).mean()) * 100.0, 2),
        "cost_framing": ("cheapest lever first: threshold calibration before "
                         "fine-tuning before data collection; retraining "
                         "decisions should weigh cost against expected error "
                         "reduction (Regol et al., ICML 2025)"),
        "no_signal_note": ("neither Mahalanobis OOD mass nor measured quality "
                           "bands separate error rows from the population on "
                           "this split; no retraining trigger is present"),
    }
    qb_drift = None
    if lane_b and lane_b.get("quality"):
        try:
            qb_drift = max(abs((q["err_extreme_pct"] or 0.0) -
                               (q["ctl_extreme_pct"] or 0.0))
                           for q in lane_b["quality"].values())
        except (TypeError, ValueError):
            qb_drift = None
    sec["quality_band_drift_max_pct"] = None if qb_drift is None else round(qb_drift, 2)

    per = {}
    for c, p in enumerate(pats):
        lab = labelled[:, c]
        y = gt_eff[lab, c]
        s = pbar[lab, c]
        npos = int((y == 1).sum())
        a = _auroc(y, s) if npos >= MIN_POS_FOR_SENS else None
        ci = None
        if a is not None:
            # patient-cluster bootstrap of the AUROC
            cell_idx = np.where(lab)[0]
            rng = np.random.default_rng(seed + 1000 + c)
            codes_c = inp["pat_codes"][cell_idx]
            uniq = np.unique(codes_c)
            by_pat = [cell_idx[codes_c == u] for u in uniq]
            boots = []
            for _ in range(n_boot):
                take = rng.integers(0, len(uniq), size=len(uniq))
                sel = np.concatenate([by_pat[i] for i in take])
                v = _auroc(gt_eff[sel, c], pbar[sel, c])
                if v is not None:
                    boots.append(v)
            if boots:
                lo, hi = np.percentile(boots, [2.5, 97.5])
                ci = (float(lo), float(hi))
        if a is None:
            band, lever, rule = None, "unmeasurable", "R0"
        else:
            band = ("low" if ci[1] < 0.70 else
                    "high" if ci[0] >= 0.80 else "moderate")
            asym = (lane_c or {}).get(p, {}).get("asymmetry_band", "balanced")
            ood_gap = None
            if sec["ood_p90_share_err_pct"] is not None:
                ood_gap = sec["ood_p90_share_err_pct"] - sec["ood_p90_share_all_pct"]
            if band == "low":
                lever, rule = "more_data_retrain", "R1"
            elif band == "high" and asym in ("fp_dominated", "fn_dominated"):
                lever, rule = "threshold_calibration", "R2"
            elif (qb_drift is not None and qb_drift > 20.0) or \
                    (ood_gap is not None and ood_gap > 10.0):
                lever, rule = "fine_tune", "R3"
            else:
                lever, rule = "no_signal", "R4"
        e = {
            "auroc_eval": None if a is None else round(a, 4),
            "auroc_ci_lo": None if ci is None else round(ci[0], 4),
            "auroc_ci_hi": None if ci is None else round(ci[1], 4),
            "auroc_band": band,
            "ood_maha_median_err_rows": med_err,
            "ood_maha_median_all_rows": med_all,
            "member_std_mean_err": (float(np.nanmean(es["member_std"][is_err, c]))
                                    if is_err.any() else None),
            "member_std_mean_all": float(np.nanmean(es["member_std"])),
            "lever": lever, "rule_id": rule,
        }
        per[p] = e
    sec.update(per)
    return sec


# ---------------------------------------------------------------------------
# Lane (e) -- metadata stratification (slices)
# ---------------------------------------------------------------------------

def lane_stratification_slices(inp, es, qtab=None, k=6, n_boot=N_BOOT, seed=SEED):
    n_pat = int(inp["pat_codes"].max()) + 1
    err_rows = set(int(r) for r in es["err_rows"])
    is_err = np.zeros(len(inp["ids"]), dtype=bool)
    is_err[sorted(err_rows)] = True
    pop_rate = float(es["err"].sum()) / max(int(es["labelled"].sum()), 1)

    def slice_stats(name, mask):
        cells = int((es["labelled"] & mask[:, None]).sum())
        errc = int((es["err"] & mask[:, None]).sum())
        rows_n = int((mask & es["err"].any(axis=1)).sum())
        # cell-level patient codes: each cell of row i belongs to row i's patient
        cell_codes = np.repeat(inp["pat_codes"], len(inp["pathologies"]))
        num = _pat_sums((es["err"] & mask[:, None]).ravel(), cell_codes, n_pat)
        den = _pat_sums((es["labelled"] & mask[:, None]).ravel(), cell_codes, n_pat)
        ci = _rate_ci(num, den, n_boot, seed + sum(ord(ch) for ch in name))
        rep = cells >= MIN_SLICE_CELLS and rows_n >= MIN_SLICE_ERR_ROWS
        out = {
            "name": name, "n_images": int(mask.sum()),
            "n_patients": int(len(np.unique(inp["pat_codes"][mask]))),
            "labelled_cells": cells, "err_rows_n": rows_n,
            "err_cells": errc,
            "err_rate_pct": round(errc / cells * 100.0, 2) if cells else None,
            "pop_err_rate_pct": round(pop_rate * 100.0, 2),
            # pop_rate is 0 only when the split has no errors at all; a ratio
            # against it is undefined, not infinite.
            "ratio_vs_pop": (round((errc / cells) / pop_rate, 3)
                             if (cells and pop_rate > 0) else None),
            "ci_lo_pct": None if not ci else round(ci["lo"] * 100.0, 2),
            "ci_hi_pct": None if not ci else round(ci["hi"] * 100.0, 2),
            "reported": bool(rep),
        }
        if not rep:
            out["too_small_note"] = ("slice below the reporting floor "
                                     "(>= %d labelled cells and >= %d error "
                                     "rows); ratios here are unstable"
                                     % (MIN_SLICE_CELLS, MIN_SLICE_ERR_ROWS))
        return out

    sec: dict = {"view": {}, "quality": {}, "cluster": {}}
    for v in sorted(set(str(x) for x in inp["view"])):
        sec["view"][v] = slice_stats("view=" + v, inp["view"] == v)

    if qtab is not None:
        bands = _row_bands(inp, qtab)
        for m in QUALITY_METRICS:
            ext = np.array([_extreme(b, m) for b in bands[m]])
            if ext.sum() == 0:
                continue
            st = slice_stats("%s extreme band" % m, ext)
            st["direction"] = "low" if m in _LOW_EXTREME else "high"
            sec["quality"][m] = st

    # Domino-lite embedding clusters: fit on error rows, assign all rows.
    try:
        from sklearn.cluster import KMeans
        feats = inp["rad_feats"]
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        fnorm = feats / np.maximum(norms, 1e-12)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(fnorm[sorted(err_rows)] if err_rows else fnorm)
        cl = km.transform(fnorm).argmin(axis=1)
        for i in range(k):
            st = slice_stats("cluster=%d" % i, cl == i)
            m_mask = cl == i
            ap = float((m_mask & (inp["view"] == "AP")).sum()) / max(int(m_mask.sum()), 1)
            pa = float((m_mask & (inp["view"] == "PA")).sum()) / max(int(m_mask.sum()), 1)
            st["view_share_ap_pct"] = round(ap * 100.0, 2)
            st["view_share_pa_pct"] = round(pa * 100.0, 2)
            errs = [j for j in sorted(err_rows) if cl[j] == i]
            top, top_share = None, None
            if errs:
                cnts = {p: int(es["err"][j, c]) for j in errs
                        for c, p in enumerate(inp["pathologies"])}
                top = max(cnts, key=lambda k2: cnts[k2])
                tot = sum(cnts.values())
                top_share = cnts[top] / tot if tot else None
            st["top_class"] = top
            st["top_class_err_share_pct"] = (round(top_share * 100.0, 2)
                                             if top_share is not None else None)
            if qtab is not None:
                ids_arr = np.asarray(inp["ids"])
                low_cs = [1.0 if bands["contrast_span"][j] in ("very low", "low")
                          else 0.0 for j in np.where(m_mask)[0]]
                low_tn = [1.0 if bands["tenengrad"][j] in ("very low", "low") else 0.0
                          for j in np.where(m_mask)[0]]
                st["contrast_span_low_pct"] = round(float(np.mean(low_cs)) * 100.0, 2) \
                    if low_cs else None
                st["tenengrad_low_pct"] = round(float(np.mean(low_tn)) * 100.0, 2) \
                    if low_tn else None
            st["maha_median"] = (float(np.nanmedian(inp["maha"][m_mask]))
                                 if m_mask.any() else None)
            sec["cluster"][str(i)] = st
    except Exception as e:                           # noqa: BLE001 -- sklearn
        sec["cluster_note"] = "clustering unavailable (%s: %s)" % (type(e).__name__, e)
    return sec


# ---------------------------------------------------------------------------
# Evidence pack assembly
# ---------------------------------------------------------------------------

def build_evidence(inp, k=6, n_boot=N_BOOT, seed=SEED, quality_mode="never"):
    """Full population evidence pack (all five lanes). quality_mode:
    'never' (tests / no image reads), 'skip' (cache only), 'auto' (compute)."""
    es = build_error_set(inp)
    qtab = compute_quality_table(inp, mode=quality_mode)
    lane_a = lane_class_data_need(inp, es, n_boot, seed)
    lane_b = lane_augmentation_gap(inp, es, qtab, n_boot, seed)
    lane_c = lane_threshold_levers(inp, es)
    lane_d = lane_retrain_vs_ft(inp, es, lane_a, lane_c, lane_b, n_boot, seed)
    lane_e = lane_stratification_slices(inp, es, qtab, k, n_boot, seed)
    n_lab = int(es["labelled"].sum())
    ev = {
        "schema": SCHEMA,
        "provenance": {
            "eval_artifact_path": os.path.basename(inp["arr_eval"]),
            "n_eval": len(inp["ids"]),
            "n_err_rows": int(len(es["err_rows"])),
            "n_err_cells": int(es["err"].sum()),
            "labelled_cells": n_lab,
            "err_rate_pct": round(float(es["err"].sum()) / max(n_lab, 1) * 100.0, 2),
            "calibration_sha": inp["calibration_sha"],
            "temperature": inp["temperature"],
            "seed": seed, "n_boot": n_boot,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        },
        "class_data_need": lane_a,
        "augmentation_gap": lane_b,
        "threshold_levers": lane_c,
        "retrain_vs_ft": lane_d,
        "stratification_slices": lane_e,
    }
    return ev


# ---------------------------------------------------------------------------
# Constrained renderer (per lane) -- cite-everything prompt.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Agent 4 of a chest X-ray triage assistant. You receive a \
structured evidence pack (items labelled EV:<id>) computed by deterministic code over a \
FIXED evaluation split of a chest X-ray ensemble. Write ONE short paragraph of \
population-level improvement suggestions for the requested lane.

HARD RULES:
1. Cite the evidence id for every factual claim, like [EV:class_data_need.Atelectasis.err_rate_pct]. \
Every claim must be traceable to a cited evidence item.
2. Hypothesis language only: "is associated with", "is consistent with", "suggests", \
"hypothesis". NEVER causal language: causes, caused, because of, due to, leads to, \
results in, will fix, proves. These are correlations on one fixed split, not experiments.
3. NEVER claim an action was taken or will be taken (no "we will retrain", no "execute"). \
Agent 4 suggests; it does not run anything.
4. Only pathology names and only augmentation/threshold/retraining concepts that appear \
in the evidence may be mentioned.
5. Few positives and a high false-positive rate are DIFFERENT diagnoses; do not conflate \
data volume with threshold placement.
6. No patient identifiers, no file paths, no device names.

At most 180 words. End with exactly this line:
""" + DISCLAIMER

_LANE_INSTRUCTION = {
    "class_data_need":
        "Which classes need more examples? Weigh support rank, error rate, "
        "the FP/FN split, prevalence drift and the unmeasurable classes. Say "
        "plainly when few positives is a measurement limit rather than a "
        "data need.",
    "augmentation_gap":
        "Which augmentation strategies are suggested? Ground each candidate in "
        "a measured trigger; mention the training-policy gap and the fixed-"
        "CLAHE-at-test-time warning only if the evidence carries them.",
    "threshold_levers":
        "What threshold calibration is suggested? Report the FP/FN asymmetry, "
        "the calibration-split refit candidates and the near-threshold error "
        "mass. The eval split was not used for selection.",
    "retrain_vs_ft":
        "Retrain, fine-tune, or calibrate? Report each class's lever and rule, "
        "and state plainly when no signal is present.",
    "stratification_slices":
        "Which metadata slices carry the largest error-rate differences? Name "
        "the reported slices with their ratios; note the excluded small slices.",
}


def evidence_prompt(evidence, lane):
    leaves = {k: v for k, v in a3.flatten_leaves({lane: evidence.get(lane, {})})
              .items() if k != "provenance"}
    lines = ["EV:%s = %s" % kv for kv in sorted(leaves.items())]
    user = ("Evidence for the '%s' lane of the population improvement report:\n%s\n\n"
            "Write the one-paragraph suggestion per the rules."
            % (lane, "\n".join(lines)))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Deterministic template renderer (test oracle + audit-failure fallback).
# ---------------------------------------------------------------------------

def _g(leaves, key, default="unavailable"):
    return leaves.get(key, default)


def render_template_lane(evidence, lane):
    leaves = a3.flatten_leaves({lane: evidence.get(lane, {})})
    pats = list(NIH_PATHOLOGIES)

    if lane == "class_data_need":
        ranked = sorted((p for p in pats
                         if _g(leaves, "class_data_need.%s.support_rank" % p)
                         not in ("unavailable", "none")),
                        key=lambda p: float(_g(leaves, "class_data_need.%s.support_rank" % p)))
        top = ranked[0] if ranked else None
        p = top or pats[0]
        t = ("Support profile: %s has the fewest labelled positives (support "
             "rank %s, n=%s) [EV:class_data_need.%s.support_rank]"
             "[EV:class_data_need.%s.npos_eval]. Its eval error rate is %s%% "
             "with %s missed positives and %s false positives "
             "[EV:class_data_need.%s.err_rate_pct][EV:class_data_need.%s.fn_n]"
             "[EV:class_data_need.%s.fp_n]. "
             ) % (p, _g(leaves, "class_data_need.%s.support_rank" % p),
                  _g(leaves, "class_data_need.%s.npos_eval" % p), p, p,
                  _g(leaves, "class_data_need.%s.err_rate_pct" % p),
                  _g(leaves, "class_data_need.%s.fn_n" % p),
                  _g(leaves, "class_data_need.%s.fp_n" % p), p, p, p)
        sens = _g(leaves, "class_data_need.%s.sensitivity_pct" % p)
        if sens not in ("unavailable", "none"):
            t += ("Sensitivity there is %s%% (CI %s-%s) "
                  "[EV:class_data_need.%s.sensitivity_pct]"
                  "[EV:class_data_need.%s.sensitivity_ci_lo_pct]"
                  "[EV:class_data_need.%s.sensitivity_ci_hi_pct]. "
                  ) % (sens, _g(leaves, "class_data_need.%s.sensitivity_ci_lo_pct" % p),
                       _g(leaves, "class_data_need.%s.sensitivity_ci_hi_pct" % p), p, p, p)
        zp = evidence.get("class_data_need", {}).get("zero_prevalence_classes", [])
        if zp:
            t += ("%s have no labelled eval cells, so their sensitivity is "
                  "unmeasurable on this split "
                  "[EV:class_data_need.zero_prevalence_classes.0]."
                  ) % ", ".join(zp)
        return t + "\n" + DISCLAIMER

    if lane == "augmentation_gap":
        q = evidence.get("augmentation_gap", {}).get("quality", {})
        best, best_lift = None, None
        for m, st in q.items():
            if isinstance(st, dict) and st.get("signal") not in (None, "none") \
                    and st.get("lift") is not None:
                if best_lift is None or abs(st["lift"] - 1.0) > abs(best_lift - 1.0):
                    best, best_lift = m, st["lift"]
        t = "Training-policy gap: current augmentation is %s; absent from the repo: %s [EV:augmentation_gap.policy.current_train_aug][EV:augmentation_gap.policy.absent_from_repo]. " % (
            evidence["augmentation_gap"]["policy"]["current_train_aug"],
            evidence["augmentation_gap"]["policy"]["absent_from_repo"])
        if best:
            st = q[best]
            t += ("Measured separation: the extreme-%s-band share of error "
                  "films for %s is %s%% vs %s%% on non-error films (lift %s, "
                  "CI %s-%s) [EV:augmentation_gap.quality.%s.err_extreme_pct]"
                  "[EV:augmentation_gap.quality.%s.ctl_extreme_pct]"
                  "[EV:augmentation_gap.quality.%s.lift]"
                  "[EV:augmentation_gap.quality.%s.ci_lo]"
                  "[EV:augmentation_gap.quality.%s.ci_hi]. "
                  ) % (st.get("direction"), best, st.get("err_extreme_pct"),
                       st.get("ctl_extreme_pct"), st.get("lift"),
                       st.get("ci_lo"), st.get("ci_hi"), best, best, best,
                       best, best)
        cands = evidence.get("augmentation_gap", {}).get("candidates", [])
        if cands:
            t += ("Candidate hypothesis: %s [EV:augmentation_gap.candidates.0.suggestion]."
                  ) % cands[0]["suggestion"]
        return t + "\n" + DISCLAIMER

    if lane == "threshold_levers":
        big = evidence.get("threshold_levers", {}).get("biggest_fp_dominated_class")
        p = big or pats[0]
        t = ("FP/FN structure: %s shows %s false positives vs %s missed "
             "positives at the current operating point "
             "[EV:threshold_levers.%s.eval_fp_n][EV:threshold_levers.%s.eval_fn_n]. "
             ) % (p, _g(leaves, "threshold_levers.%s.eval_fp_n" % p),
                  _g(leaves, "threshold_levers.%s.eval_fn_n" % p), p, p)
        kind = _g(leaves, "threshold_levers.%s.cal_optimal_kind" % p)
        if kind not in ("unavailable", "none"):
            t += ("Calibration-split refit (%s) moves the threshold to %s, with "
                  "cal FP %s vs %s at the current threshold "
                  "[EV:threshold_levers.%s.cal_optimal_kind]"
                  "[EV:threshold_levers.%s.cal_optimal_thr]"
                  "[EV:threshold_levers.%s.cal_fp_at_candidate]"
                  "[EV:threshold_levers.%s.cal_fp_at_current]. "
                  ) % (kind, _g(leaves, "threshold_levers.%s.cal_optimal_thr" % p),
                       _g(leaves, "threshold_levers.%s.cal_fp_at_candidate" % p),
                       _g(leaves, "threshold_levers.%s.cal_fp_at_current" % p), p, p, p, p)
        near = _g(leaves, "threshold_levers.threshold_reachable_err_cells_pct")
        if near != "unavailable":
            t += ("Near-threshold mass: %s%% of error cells sit within the "
                  "near-threshold window "
                  "[EV:threshold_levers.threshold_reachable_err_cells_pct]." % near)
        else:
            t += "Near-threshold mass was not computed."
        return t + "\n" + DISCLAIMER

    if lane == "retrain_vs_ft":
        p = pats[0]
        a = _g(leaves, "retrain_vs_ft.%s.auroc_eval" % p)
        if a not in ("unavailable", "none"):
            t = ("Per-class levers: %s has correctness-relevant AUROC %s "
                 "(CI %s-%s), lever %s [EV:retrain_vs_ft.%s.auroc_eval]"
                 "[EV:retrain_vs_ft.%s.auroc_ci_lo][EV:retrain_vs_ft.%s.auroc_ci_hi]"
                 "[EV:retrain_vs_ft.%s.lever]. "
                 ) % (p, a, _g(leaves, "retrain_vs_ft.%s.auroc_ci_lo" % p),
                      _g(leaves, "retrain_vs_ft.%s.auroc_ci_hi" % p),
                      _g(leaves, "retrain_vs_ft.%s.lever" % p), p, p, p, p)
        else:
            t = ("Per-class levers: %s is unmeasurable here "
                 "[EV:retrain_vs_ft.%s.lever]. " % (p, p))
        ood = _g(leaves, "retrain_vs_ft.ood_separation_median_pct")
        if ood != "unavailable":
            t += ("OOD separation is weak overall: error-row median Mahalanobis "
                  "differs from the population by %s%% "
                  "[EV:retrain_vs_ft.ood_separation_median_pct]." % ood)
        else:
            t += "OOD separation was not computed."
        return t + "\n" + DISCLAIMER

    if lane == "stratification_slices":
        views = evidence.get("stratification_slices", {}).get("view", {})
        named = [v for v, st in views.items() if st.get("reported")]
        if named:
            v0 = named[0]
            st0 = views[v0]
            t = ("View stratification: %s carries an error rate of %s%% vs a "
                 "population rate of %s%% (ratio %s) "
                 "[EV:stratification_slices.view.%s.err_rate_pct]"
                 "[EV:stratification_slices.view.%s.pop_err_rate_pct]"
                 "[EV:stratification_slices.view.%s.ratio_vs_pop]. "
                 ) % (v0, st0.get("err_rate_pct"), st0.get("pop_err_rate_pct"),
                      st0.get("ratio_vs_pop"), v0, v0, v0)
            small = [v for v, st in views.items() if not st.get("reported")]
            if small:
                t += ("%s is excluded by the reporting floor "
                      "[EV:stratification_slices.view.%s.reported]. ") % (
                          ", ".join(small), small[0])
        else:
            t = "No view slice cleared the reporting floor."
        cl = evidence.get("stratification_slices", {}).get("cluster", {})
        rep = [(i, st) for i, st in sorted(cl.items()) if st.get("reported")]
        if rep:
            i, st = rep[0]
            t += (" One embedding cluster (n=%s images) carries err rate %s%%, "
                  "with %s the most frequent error class "
                  "[EV:stratification_slices.cluster.%s.n_images]"
                  "[EV:stratification_slices.cluster.%s.err_rate_pct]"
                  "[EV:stratification_slices.cluster.%s.top_class]."
                  ) % (st.get("n_images"), st.get("err_rate_pct"),
                       st.get("top_class"), i, i, i)
        return t + "\n" + DISCLAIMER
    raise ValueError("unknown lane: %s" % lane)


def render_template(evidence):
    return "\n\n".join(render_template_lane(evidence, ln) for ln in LANES)


# ---------------------------------------------------------------------------
# Faithfulness audit -- Agent 3's checks re-implemented here (agent3.py is NOT
# edited) plus four Agent-4 hard checks: causal-language ban, execution ban,
# suggestion grounding, and a generalized band-stem derivation for nested ids.
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")
_CITE_RE = re.compile(r"\[EV:([^\]\[]+)\]")
_BAND_WORDS = a3._BAND_WORDS
_CAUSAL_RE = re.compile(
    r"\bcaus\w*|because of|due to|leads to|led to|results in|resulted in|"
    r"will fix|fixes|guarantee\w*|proves|therefore the model|responsible for",
    re.IGNORECASE)
_ACTION_MODAL_RE = re.compile(r"\b(i|we) (will|shall|must|now)\b", re.IGNORECASE)
_ACTION_VERB_RE = re.compile(
    r"\b(run|execute|launch|retrain|train|delete|modify|fine-tune|fine tun|"
    r"augment|implement)\b", re.IGNORECASE)
_ACTION_EXEC_RE = re.compile(r"\bexecut(e|es|ed|ing)\b", re.IGNORECASE)
_AUG_LEXICON = ("clahe", "rotation", "flip", "blur", "jitter", "threshold",
                "fine-tun", "retrain")


def _numeric_leaves(leaves):
    return a3._numeric_leaves(leaves)


def _matches(x, y, tol_rel=0.02, tol_abs=0.011):
    return a3._matches(x, y, tol_rel, tol_abs)


def audit(text, evidence, lane=None):
    """Mechanical faithfulness audit (0-tolerance). Same hard checks as
    Agent 3 plus the causal-language and execution bans and augmentation-
    keyword grounding. The band-stem derivation is generalized to nested
    leaf ids (last component minus the _band suffix)."""
    failures, warnings = [], []
    body = text.replace(DISCLAIMER, "").strip()
    leaves = a3.flatten_leaves(evidence)
    nums = _numeric_leaves(leaves)

    cited = _CITE_RE.findall(body)
    if not cited:
        failures.append("no citations at all")
    for cid in cited:
        if cid not in leaves:
            failures.append("invalid citation [EV:%s]" % cid)

    for sent in re.split(r"(?<=[.!?])\s+", body):
        bare = _CITE_RE.sub("", sent)
        sent_nums = [float(x) for x in
                     _NUM_RE.findall(re.sub(r"(?<=\d)-(?=\d)", " ", bare))]
        sent_cites = _CITE_RE.findall(sent)
        if not sent_nums or not sent_cites:
            continue
        pool = []
        for cid in sent_cites:
            if cid in leaves:
                pool.extend(nums.get(cid, []))
        for x in sent_nums:
            if not any(_matches(x, y) for y in pool):
                failures.append("ungrounded number %.4g in sentence citing %s"
                                % (x, ",".join(sent_cites)))

    # class-name grounding (pathology lexicon, as Agent 3)
    from .config import NIH_PATHOLOGIES as _P
    ev_blob = (" ".join(leaves.values()) + " " + " ".join(leaves.keys()))
    lex = set(_P) | {"No Finding", "no-finding", "Pleural Thickening"}
    for tok in set(re.findall(r"\b[A-Z][a-zA-Z_]{2,}\b", body)):
        if tok in lex and tok.replace("_", " ") not in ev_blob.replace("_", " "):
            failures.append("pathology named but absent from evidence: %s" % tok)

    # band agreement with the generalized stem (second-to-last component is
    # NOT used; the last component minus the _band suffix generalizes both
    # flat Agent-3 ids and nested Agent-4 ids)
    for sent in re.split(r"(?<=[.!?])\s+", body):
        low = sent.lower()
        for leaf_id, val in leaves.items():
            if not leaf_id.endswith("_band") or val is None:
                continue
            parts = leaf_id.split(".")
            stem = parts[-1][: -len("_band")].replace("_", " ")
            if not stem or stem not in low:
                continue
            m = re.search(re.escape(stem) + r"\s*[=:]*\s*([a-z ]{1,20})", low)
            if not m:
                continue
            present = [bw for bw in _BAND_WORDS
                       if re.search(r"\b%s\b" % re.escape(bw), m.group(1))]
            if not present:
                continue
            said = max(present, key=len)
            if said != str(val).lower():
                failures.append("band mismatch for %s: text says '%s', "
                                "evidence '%s'" % (leaf_id, said, val))

    # causal-language ban (hard)
    m = _CAUSAL_RE.search(body)
    if m:
        failures.append("causal language: '%s'" % m.group(0))
    # execution ban (hard): a first-person commitment to act, or any
    # execution vocabulary
    for sent in re.split(r"(?<=[.!?])\s+", body):
        if _ACTION_MODAL_RE.search(sent) and _ACTION_VERB_RE.search(sent):
            failures.append("Agent 4 never executes; suggestion phrased as "
                            "an action: '%s'" % sent.strip()[:80])
    m = _ACTION_EXEC_RE.search(body)
    if m:
        failures.append("Agent 4 never executes; execution vocabulary: '%s'"
                        % m.group(0))

    # suggestion grounding: augmentation/threshold concepts named in prose
    # must appear in the evidence blob (no invented interventions)
    norm = re.sub(r"[-_]", " ", ev_blob.lower())
    for kw in _AUG_LEXICON:
        if re.search(r"\b%s" % re.escape(kw).replace(r"\-", "[- ]?"),
                     re.sub(r"[-_]", " ", body.lower())) \
                and kw.replace("-", " ") not in norm:
            failures.append("concept named but absent from evidence: %s" % kw)

    if len(body.split()) > 200:
        failures.append("too long: %d words" % len(body.split()))

    if lane is not None:
        if not any(cid.startswith(lane) for cid in cited):
            warnings.append("no citation from the lane's own section")
    return {"ok": not failures, "failures": failures, "warnings": warnings}


# ---------------------------------------------------------------------------
# Provider chain (same backends as Agent 3, template renders per lane)
# ---------------------------------------------------------------------------

PROVIDERS = {}


def register_provider(name):
    def _wrap(fn):
        PROVIDERS[name] = fn
        return fn
    return _wrap


@register_provider("ollama_local")
def ollama_local(messages, model=a3.DEFAULT_CLOUD_MODEL, **kw):
    return a3._chat(a3.LOCAL_BASE, model, messages)


@register_provider("ollama_cloud")
def ollama_cloud(messages, model=a3.DEFAULT_CLOUD_MODEL, **kw):
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY not set")
    return a3._chat(a3.CLOUD_BASE, model, messages, api_key=key)


@register_provider("template")
def template_provider(messages, model="", evidence=None, lane=None, **_):
    if evidence is None or lane is None:
        raise RuntimeError("template backend requires evidence and lane")
    return render_template_lane(evidence, lane)


def provider_chain(backend="auto"):
    if backend == "template":
        return ["template"]
    chain = ["ollama_local"]
    if os.environ.get("OLLAMA_API_KEY"):
        chain.append("ollama_cloud")
    chain.append("template")
    return chain


def render(evidence, lane, backend="auto", model=a3.DEFAULT_CLOUD_MODEL):
    """Render one lane through the provider chain with the 0-tolerance gate.
    Never raises."""
    msgs = evidence_prompt(evidence, lane)
    last_err = None
    for name in provider_chain(backend):
        if name == "template":
            text = render_template_lane(evidence, lane)
            return {"text": text, "backend": "template",
                    "audit": audit(text, evidence, lane),
                    "error": (None if last_err is None else
                              "LLM backends unavailable (%s); template "
                              "rendering shown." % last_err)}
        try:
            text = PROVIDERS[name](messages=msgs, model=model,
                                   evidence=evidence, lane=lane)
        except Exception as e:                       # noqa: BLE001 -- chain step
            last_err = "%s: %s" % (type(e).__name__, e)
            continue
        res = audit(text, evidence, lane)
        if res["ok"]:
            return {"text": text, "backend": name, "audit": res, "error": None}
        return {"text": render_template_lane(evidence, lane),
                "backend": name + "->template", "audit": res,
                "error": ("LLM rendering failed the faithfulness audit "
                          "(%d failures); template rendering shown."
                          % len(res["failures"]))}
    text = render_template_lane(evidence, lane)
    return {"text": text, "backend": "template",
            "audit": audit(text, evidence, lane),
            "error": "no backend rendered (%s)" % last_err}


_LAST_EVIDENCE = {}


def render_cached(evidence=None, backend="auto", model=a3.DEFAULT_CLOUD_MODEL):
    """UI entry point: renders all five lanes. Never raises."""
    if evidence is None:
        evidence = _LAST_EVIDENCE
    if not evidence:
        return ("*Agent 4 has no population analysis yet -- run "
                "`python scripts/agent4_batch.py --analyze --render` first.*")
    parts, faith = [], []
    for lane in LANES:
        try:
            res = render(evidence, lane, backend=backend, model=model)
        except Exception as e:                       # noqa: BLE001 -- UI safety
            parts.append("**%s**: unavailable (%s: %s)"
                         % (lane, type(e).__name__, e))
            faith.append({"lane": lane, "backend": "error",
                          "ok": False, "failures": [str(e)]})
            continue
        faith.append({"lane": lane, "backend": res["backend"],
                      "ok": bool(res["audit"]["ok"]),
                      "failures": res["audit"]["failures"]})
        head = ""
        if res.get("error"):
            head = "**Agent 4 note**: %s\n\n" % res["error"]
        elif res.get("backend") == "template":
            head = "*Template rendering (deterministic; no LLM call).*\n\n"
        if res["audit"].get("warnings"):
            head += "*Faithfulness warnings*: %s\n\n" \
                % "; ".join(res["audit"]["warnings"])
        parts.append("### %s\n\n%s%s" % (lane, head, res["text"]))
    ok = sum(1 for f in faith if f["ok"])
    tail = ("\n\n---\n*Faithfulness audit: %d/%d lanes passed the 0-tolerance "
            "gate.*" % (ok, len(faith)))
    return "\n\n".join(parts) + tail