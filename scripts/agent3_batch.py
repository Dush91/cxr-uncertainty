"""Agent 3 batch tooling: quality-threshold calibration + batch rendering.

Two modes (repo root):

  # 1) fit per-metric/per-view percentile bands + the dataset-bias block
  #    (1,000 eval images stratified by view; fixed quintiles, no hand tuning)
  python scripts/agent3_batch.py --calibrate-quality

  # 2) build evidence + render explanations for eval cases
  python scripts/agent3_batch.py --sample 50 --render --backend template
  python scripts/agent3_batch.py --sample 3  --render --backend auto \
      --case confident_error --case normal --case ood

Outputs (adapted regime) land in runs/app_rex_adapted/:
  agent3_quality_stats.json   percentile tables + bias block (committed)
  agent3_batch/<case_id>.json evidence + rendering + audit record
  agent3_batch/faithfulness_report.json
  agent3_batch/side_by_side.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np
import pandas as pd

import cxr_uncertainty.agent3 as a3
from cxr_uncertainty.config import NIH_PATHOLOGIES
from cxr_uncertainty.reanalyze import _confidence

# Same regime split as the demo app (paths-only swap).
ARR_DIR = os.path.join(REPO, "runs", "rex_phase7", "eval_arrays", "rex_adapted_s0")
ARR_EVAL = os.path.join(ARR_DIR, "arrays_rex_adaptedeval.npz")
MANIFEST = os.path.join(REPO, "data", "rex_manifest_fs.parquet")
CACHE_DIR = os.path.join(REPO, "runs", "app_rex_adapted")
STATS_JSON = os.path.join(CACHE_DIR, "agent3_quality_stats.json")
CONF_PCT, UNC_PCT = 10.0, 50.0
MAHA_CLASS = "Pneumonia"

QUALITY_METRICS = ("contrast_span", "contrast_std", "exposure_mean",
                   "exposure_median", "midgray_dev", "clip_lo_frac",
                   "clip_hi_frac", "laplacian_var", "tenengrad",
                   "mirror_asymmetry", "centroid_offset")


def _percentile_table(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return None
    ps = np.percentile(v, [20, 40, 60, 80])
    return {"percentiles": {"p20": float(ps[0]), "p40": float(ps[1]),
                            "p60": float(ps[2]), "p80": float(ps[3])},
            "median": float(np.median(v)),
            "mad": float(np.median(np.abs(v - np.median(v)))),
            "n": int(len(v))}


# ---------------------------------------------------------------------------
# --calibrate-quality
# ---------------------------------------------------------------------------
def calibrate_quality(n_total: int = 1000) -> dict:
    """Eval-split percentile tables (stratified by view; fixed quintiles) +
    the dataset-bias block. Deterministic sample (seed 1234)."""
    m = pd.read_parquet(MANIFEST)
    ev = m[m["split_role"] == "eval"].copy()
    ev = ev[ev["image_path"].apply(os.path.exists)]
    views = [v for v in ev["view"].astype(str).unique() if v and v != "nan"]
    rng = random.Random(1234)
    sampled = []
    for v in views:
        sub = ev[ev["view"].astype(str) == v]
        idx = list(sub.index)
        rng.shuffle(idx)
        take = (idx[: n_total // len(views)] if len(views) > 1
                else idx[:n_total])
        sampled.extend(take)
        print(f"[calibrate] view={v}: sampled {len(take)}/{len(sub)}")

    by_view: dict[str, dict[str, list]] = {v: {} for v in views}
    t0 = time.time()
    for i, idx in enumerate(sampled):
        row = ev.loc[idx]
        try:
            q = a3.compute_quality(str(row["image_path"]))
        except Exception as e:                          # noqa: BLE001
            print(f"[calibrate] skip {row['image_id']}: {type(e).__name__}: {e}")
            continue
        vw = str(row["view"])
        for k in QUALITY_METRICS:
            by_view[vw].setdefault(k, []).append(q.get(k, np.nan))
        if (i + 1) % 100 == 0:
            print(f"[calibrate] {i + 1}/{len(sampled)} images "
                  f"({time.time() - t0:.0f}s)")
    tables = {v: {k: t for k in QUALITY_METRICS
                  if (t := _percentile_table(by_view[v].get(k, []))) is not None}
              for v in views}

    # epistemic-std reference: all valid class cells of the eval npz
    ev_npz = dict(np.load(ARR_EVAL, allow_pickle=True))
    probs = np.asarray(ev_npz["probs"], dtype=np.float64)            # (N,M,P)
    valid = np.asarray(ev_npz["valid"], dtype=bool)
    std_all = np.nanstd(np.where(valid[:, None, :], probs, np.nan), axis=1)  # (N,P)
    std_table = _percentile_table(std_all.ravel())

    stats = {"by_view": tables, "uncertainty_std": std_table,
             "n_images": len(sampled), "views": views, "seed": 1234,
             "band_rule": "fixed quintiles p20/p40/p60/p80 of the eval split, "
             "per metric per view; clip-fraction special case: "
             "clip_lo+clip_hi > 0.10 -> 'high clipping'",
             "bias": _compute_bias(m, ev_npz)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[calibrate] wrote {STATS_JSON}")
    return stats


def _confidence_math(ev_npz, youden, conf_cut, unc_cut):
    """Replicates the app's exact two-regions math on the eval npz.

    Returns (is_confident (N,P), fires (N,P), wrong (N,P), mean_p, std).
    Definitions: confident = Youden-anchored confidence >= conf_cut
    (top-CONF_PCT of cal); fires = confident AND epistemic_std >= unc_cut
    (top-UNC_PCT within the confident set); wrong = model decision (pooled
    mean >= Youden thr) != gt>0 among finite, valid, certain-label cells.
    """
    probs = np.asarray(ev_npz["probs"], dtype=np.float64)
    valid = np.asarray(ev_npz["valid"], dtype=bool)
    gt = np.asarray(ev_npz["gt"], dtype=np.float64)
    pats = [str(p) for p in ev_npz["pathologies"]]
    thr = np.array([0.5 if youden.get(p) is None else float(youden[p])
                    for p in pats])
    mean_p = np.nanmean(probs, axis=1)
    cm = _confidence(mean_p, thr[None, :])
    std = np.stack([np.nanstd(probs[:, :, j], axis=1)
                    for j in range(len(pats))], axis=1)
    is_conf = cm >= conf_cut
    fires = is_conf & valid & (std >= unc_cut)
    wrong = fires & np.isfinite(gt) & (gt >= 0) & \
        ((mean_p >= thr[None, :]) != (gt > 0))
    return is_conf, fires, wrong, mean_p, std


def _compute_bias(manifest, ev_npz):
    """Precomputed dataset-bias block (offline; committed in the stats JSON).

    class_prevalence: positive fraction among certain, valid cal+eval labels;
    split_prevalence: the same per split; zero_prevalence_classes: no positive
    cal AND eval example; view_shares: split-level view fractions;
    confident_wrong_eval_n: eval rows where Region B fires on a wrong call
    (app-exact math via _confidence_math; definition recorded, not hand-copied).
    """
    out = {"split_prevalence": {}, "class_prevalence": {}, "view_shares": {},
           "zero_prevalence_classes": [],
           "note": "prevalences over certain, valid labels of the "
                   "ReXGradient splits (manifest fs)",
           "maha_class": MAHA_CLASS}
    for role in ("train", "cal", "eval"):
        sub = manifest[manifest["split_role"] == role]
        labs = np.stack(sub["labels"].apply(
            lambda x: np.asarray(list(x), dtype=np.float64)))
        val = np.stack(sub["valid"].values).astype(bool)
        with np.errstate(invalid="ignore"):
            prev = np.nanmean(np.where((labs >= 0) & val, labs, np.nan), axis=0)
        out["split_prevalence"][role] = {
            p: (None if not np.isfinite(x) else float(x))
            for p, x in zip(NIH_PATHOLOGIES, prev)}
    ce = manifest[manifest["split_role"].isin(["cal", "eval"])]
    labs = np.stack(ce["labels"].apply(
        lambda x: np.asarray(list(x), dtype=np.float64)))
    val = np.stack(ce["valid"].values).astype(bool)
    with np.errstate(invalid="ignore"):
        prev = np.nanmean(np.where((labs >= 0) & val, labs, np.nan), axis=0)
    out["class_prevalence"] = {p: (None if not np.isfinite(x) else float(x))
                               for p, x in zip(NIH_PATHOLOGIES, prev)}
    out["zero_prevalence_classes"] = [
        p for p, x in zip(NIH_PATHOLOGIES, prev)
        if (not np.isfinite(x)) or x == 0.0]
    out["view_shares"] = {k: float(v) for k, v in
                          ce["view"].astype(str).value_counts(normalize=True).items()}
    out["n_train"] = int((manifest["split_role"] == "train").sum())
    out["n_cal"] = int((manifest["split_role"] == "cal").sum())
    out["n_eval"] = int((manifest["split_role"] == "eval").sum())
    try:
        youden, conf_cut, unc_cut = _load_calibration()
        _, _, wrong, _, _ = _confidence_math(ev_npz, youden, conf_cut, unc_cut)
        out["confident_wrong_eval_n"] = int(wrong.any(axis=1).sum())
        out["confident_wrong_definition"] = (
            "Region B fires (Youden-anchored confidence >= conf_cut AND "
            "epistemic_std >= unc_cut within the confident set) on a row "
            "whose model decision != labeler gt")
        out["maha_class"] = MAHA_CLASS
    except Exception as e:                              # noqa: BLE001
        out["confident_wrong_eval_n"] = None
        out["confident_wrong_note"] = f"unavailable: {type(e).__name__}: {e}"
    return out


def _load_calibration():
    with open(os.path.join(CACHE_DIR, "calibration.json")) as f:
        cc = json.load(f)
    youden = {p: (None if v is None else float(v))
              for p, v in cc["youden"].items()}
    return youden, float(cc["conf_cut"]), float(cc["unc_cut"])


_MANIFEST_CACHE = None


def manifest_view(image_id):
    """View tag for one image_id from the manifest (cached read)."""
    global _MANIFEST_CACHE
    try:
        if _MANIFEST_CACHE is None:
            _MANIFEST_CACHE = pd.read_parquet(MANIFEST).set_index("image_id")
        v = _MANIFEST_CACHE.loc[image_id].get("view")
        return None if (v is None or pd.isna(v)) else str(v)
    except Exception:                                   # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# --sample N --render
# ---------------------------------------------------------------------------
def _normal_rows(ev_npz, mean_p, thr):
    """Rows where the model reaches NO valid class (its own no-finding verdict)
    AND at least one certain label exists (so 'normal' is checkable)."""
    dec = np.where(np.isfinite(mean_p), mean_p, -np.inf) >= thr[None, :]
    valid = np.asarray(ev_npz["valid"], dtype=bool)
    gt = np.asarray(ev_npz["gt"], dtype=np.float64)
    none_reach = ~(dec & valid).any(axis=1)
    certain = np.isfinite(gt).any(axis=1) & (gt >= 0).any(axis=1)
    return np.where(none_reach & certain)[0]


def run_batch(args) -> None:
    import demo_app_rex as app            # reuses the app's exact loaders/math

    app.load_state()
    stats = a3.load_quality_stats(STATS_JSON) or {}
    ev_npz = app._EVAL
    probs_all = np.asarray(ev_npz["probs"], dtype=np.float64)
    youden, conf_cut, unc_cut = app._YOUDEN, app._CONF_CUT, app._UNC_CUT
    thr = np.array([0.5 if youden.get(p) is None else float(youden[p])
                    for p in NIH_PATHOLOGIES])
    _, _, wrong, mean_p, _std = _confidence_math(
        {k: np.asarray(v) for k, v in ev_npz.items()},
        youden, conf_cut, unc_cut)

    used: set[int] = set()
    picks: list[tuple[str, int]] = []

    def pick(rows, kind):
        for k in rows:
            if int(k) not in used:
                used.add(int(k))
                picks.append((kind, int(k)))
                return

    kinds = args.case or ["confident_error", "normal", "ood"]
    rng = random.Random(1234)
    # cycle kinds until --sample cases are picked (one pick may fail when a
    # kind is exhausted; the used-set guard prevents duplicates)
    while len(picks) < args.sample:
        before = len(picks)
        for kind in kinds:
            if len(picks) >= args.sample:
                break
            if kind == "confident_error":
                rows_k = np.where(wrong.any(axis=1))[0]
            elif kind == "normal":
                rows_k = np.where(
                    ~(np.where(np.isfinite(mean_p), mean_p, -np.inf)
                      >= thr[None, :]).any(axis=1))[0]
            elif kind == "ood":
                mah = np.asarray(ev_npz["mahalanobis"], dtype=np.float64)
                rows_k = np.where(mah >= np.percentile(mah, 99))[0]
            else:                                       # random
                rows_k = list(range(len(mean_p)))
                rng.shuffle(rows_k)
            pick(rows_k, kind)
        if len(picks) == before:
            break                                       # all kinds exhausted

    os.makedirs(args.out, exist_ok=True)
    rows, n_gate_fail = [], 0
    for kind, k in picks[: args.sample]:
        image_id = str(ev_npz["ids"][k])
        path = app._PATH_BY_ID.get(image_id)
        if not path or not os.path.exists(path):
            print(f"[batch] skip {image_id}: no manifest image_path")
            continue
        probs = np.asarray(ev_npz["probs"][k], dtype=np.float64)
        p_ts = app._ts(mean_p[k])
        pred_set = app._pred_set(image_id) or ["no-finding"]
        # Region-B flag key, exactly the app's rule
        cm = _confidence(mean_p[k][None, :], thr[None, :])[0]
        stds = np.array([np.nanstd(probs[:, j])
                         for j in range(len(NIH_PATHOLOGIES))])
        flags = (cm >= conf_cut) & (stds >= unc_cut)
        flag_class = None
        if flags.any():
            order = sorted(np.where(np.isfinite(cm))[0], key=lambda j: -cm[j])
            flag_class = next((NIH_PATHOLOGIES[j] for j in order if flags[j]),
                              None)
        feat = np.asarray(ev_npz["rad_feats"][k], dtype=np.float32)
        feat_n = (feat / max(float(np.linalg.norm(feat)), 1e-12)).astype(np.float32)
        neighbors, _lane, _nallow = app._retrieval(
            feat_n, str(NIH_PATHOLOGIES[int(np.nanargmax(mean_p[k]))]),
            "plain", image_id,
            p_q=np.where(np.isfinite(mean_p[k]), mean_p[k], 0.0),
            flag_class=flag_class)
        gt = (app._GT_BY_ID.get(image_id)
              if image_id in getattr(app, "_GT_BY_ID", {}) else None)
        bias = dict(stats.get("bias", {})) if stats else {}
        zero = set(bias.get("zero_prevalence_classes", []))
        bias["predicted_in_zero_prevalence"] = [c for c in pred_set if c in zero]
        bias["query_view"] = manifest_view(image_id)
        ev = a3.build_evidence(
            path=path, view=manifest_view(image_id), probs=probs,
            pathologies=[str(p) for p in ev_npz["pathologies"]], youden=youden,
            p_ts=p_ts, pred_set=pred_set, flag_class=flag_class,
            maha_pct=app._maha_pct(float(ev_npz["mahalanobis"][k])),
            ood=app._maha_pct(float(ev_npz["mahalanobis"][k])) >= 95.0,
            neighbors=neighbors, gt=gt, include_gt=True,
            quality_stats=stats, bias=bias,
            pool_note="FULL library (train+cal+eval)")
        t0 = time.time()
        llm_err = None
        if args.backend == "template":
            text, aud, backend_used = a3.render_template(ev), None, "template"
            aud = a3.audit(text, ev)
        else:
            res = a3.render(ev, backend="auto",
                            model=getattr(args, "model", a3.DEFAULT_CLOUD_MODEL))
            text, aud, backend_used = res["text"], res["audit"], res["backend"]
            llm_err = res.get("error")
        lat = time.time() - t0
        if not aud["ok"]:                    # 0-tolerance: template + red note
            n_gate_fail += 1
            text = a3.render_template(ev)
            aud = a3.audit(text, ev)
        rec = {"kind": kind, "image_id": image_id, "backend": backend_used,
               "latency_s": round(lat, 2), "llm_error": llm_err,
               "audit": aud, "evidence": ev, "rendering": text}
        with open(os.path.join(args.out, image_id + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        rows.append({"kind": kind, "image_id": image_id,
                     "backend": backend_used, "audit_ok": bool(aud["ok"]),
                     "failures": aud["failures"], "llm_error": llm_err,
                     "rendering": text})
        print(f"[batch] {kind:16s} {image_id}: backend={backend_used} "
              f"audit_ok={aud['ok']} ({lat:.1f}s)")

    with open(os.path.join(args.out, "faithfulness_report.json"), "w") as f:
        json.dump({"n_cases": len(rows), "n_gate_failures": n_gate_fail,
                   "backend": args.backend,
                   "rows": rows}, f, indent=2)
    with open(os.path.join(args.out, "side_by_side.md"), "w") as f:
        f.write("# Agent 3 side-by-side (rendering only; evidence in "
                "<case_id>.json)\n\n")
        for r in rows:
            f.write(f"## {r['kind']} — {r['image_id']} "
                    f"(backend: {r['backend']}, audit ok: {r['audit_ok']})\n\n"
                    + r["rendering"] + "\n\n")
    print(f"[batch] {len(rows)} cases, {n_gate_fail} gate failures -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate-quality", action="store_true")
    ap.add_argument("--n", type=int, default=1000,
                    help="approx total eval images for --calibrate-quality")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--backend", choices=("auto", "template"), default="auto")
    ap.add_argument("--model", default=a3.DEFAULT_CLOUD_MODEL,
                    help="LLM model tag (default glm-4.6 via ollama cloud; "
                         "local ollama needs an installed name)")
    ap.add_argument("--case", action="append", default=[],
                    help="case kind: confident_error|normal|ood|random")
    ap.add_argument("--out", default=os.path.join(CACHE_DIR, "agent3_batch"))
    args = ap.parse_args()

    if args.calibrate_quality:
        calibrate_quality(args.n)
        return
    if args.sample <= 0 or not args.render:
        print("nothing to do: pass --calibrate-quality or --sample N --render")
        return
    run_batch(args)


if __name__ == "__main__":
    main()