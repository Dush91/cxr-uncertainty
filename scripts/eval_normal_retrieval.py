#!/usr/bin/env python3
"""Normal-retrieval quality gate: before/after normal-aware retrieval.

User request: increase retrieval accuracy for images with NO pathologies
(GT=0 everywhere). The old app behavior sent a normal query into its dominant
argmax pathology bucket; the new design adds a `normal` mode (gt_normal
candidates) and a `pred_normal` fallback for pathology mode.

Quantifies the gain offline, no app or GPU needed:

  queries  = eval-split rows with gt_normal (at least one certain label and
             none positive) -- exactly the population the user cares about.
  old      = "pathology mode, argmax bucket": candidates must share the
             query's dominant pred_pathology (member-mean argmax). Measured on
             BOTH the deployed cal-only library (the status quo ante) and the
             full 129k library (isolates library effect from filter effect).
  new      = "normal mode" on the full library: candidates = gt_normal &
             ~uncertain (any split).

  metric   = normal-precision@8: share of the top-8 neighbors that are gt_normal.

Self-match excluded via the sidecar image_id. Features are the L2-normalized
store vectors; queries come from the npz rad_feats (same vectors).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402  (14-name mapping)


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def filtered_topk(feats, q, k, allow):
    sims = feats @ q
    sims = np.where(allow, sims, -np.inf)
    return np.argsort(-sims)[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-dir", default="runs/rex_phase7/retrieval/rex_adapted_raddino_full")
    ap.add_argument("--arr-eval", default="runs/rex_phase7/eval_arrays/rex_adapted_s0/arrays_rex_adaptedeval.npz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=2000)
    args = ap.parse_args()

    import pandas as pd

    store = np.load(os.path.join(args.full_dir, "store.npz"), allow_pickle=True)
    feats = norm(np.asarray(store["feats"], dtype=np.float32))
    ids = np.array([str(i) for i in store["ids"]])
    meta = pd.read_parquet(os.path.join(args.full_dir, "sidecar.parquet")) \
             .set_index("image_id").reindex(ids).reset_index(drop=True)
    assert not meta["pred_pathology"].isna().any(), "gap: sidecar rows missing reindexed ids"

    ev = dict(np.load(args.arr_eval, allow_pickle=True))
    q_feats = norm(np.asarray(ev["rad_feats"], dtype=np.float32))
    q_ids = np.array([str(i) for i in ev["ids"]])
    q_gt = np.asarray(ev["gt"], dtype=np.float64)
    q_probs = np.asarray(ev["probs"], dtype=np.float64)
    q_valid = ev["valid"].astype(bool)
    q_mean = np.nanmean(q_probs, axis=1)
    q_pred_idx = np.where(q_valid & np.isfinite(q_mean), q_mean, -np.inf).argmax(axis=1)
    q_pats = [str(p) for p in ev["pathologies"]]

    q_gt_normal = ((q_gt >= 0).sum(axis=1) > 0) & ((q_gt == 1).sum(axis=1) == 0)
    sel = np.where(q_gt_normal)[0]
    if args.max_queries:
        sel = sel[: args.max_queries]
    print(f"[queries] {len(sel)} eval GT-normal queries "
          f"(of {int(q_gt_normal.sum())} GT-normal eval rows)")

    # --- query-side no-finding verdict (app fallback semantics) -------------
    # recomputed Youden thresholds from the eval npz is wrong; load the app's
    # calibration.json when present so pred_normal uses the SAME thresholds.
    youden = None
    cal_json = "runs/app_rex_adapted/calibration.json"
    if os.path.exists(cal_json):
        with open(cal_json) as f:
            calj = json.load(f)
        yj = calj.get("youden", calj)
        youden = np.array([0.5 if yj.get(p) is None else float(yj.get(p, 0.5))
                           for p in q_pats]) if isinstance(yj, dict) else None

    id_pos = {i: k for k, i in enumerate(ids)}
    gt_normal_col = meta["gt_normal"].fillna(False).values.astype(bool)
    uncertain_col = meta["uncertain"].fillna(False).values.astype(bool)
    pat_col = meta["pred_pathology"].fillna("").values
    split_col = meta["split"].values

    arms = [("old@cal-lib", "old", split_col == "cal"),
            ("old@full-lib", "old", np.ones(len(ids), dtype=bool)),
            ("new-normal-mode", "new", gt_normal_col & ~uncertain_col)]

    results = {}
    for tag, kind, base in arms:
        precs, q_pred_normal_flags = [], []
        for qi in sel:
            allow = base & ~uncertain_col & (pat_col != "")
            r = id_pos.get(q_ids[qi])
            if r is not None:
                allow = allow.copy()
                allow[r] = False
            if kind == "old":
                allow &= pat_col == NIH_PATHOLOGIES[int(q_pred_idx[qi])]
            q_pred_normal_flags.append(bool(
                youden is not None
                and not (np.isfinite(q_mean[qi]) & q_valid[qi]
                         & (q_mean[qi] >= youden)).any()))
            precs.append(float(gt_normal_col[filtered_topk(
                feats, q_feats[qi], args.k, allow)].mean()))
        results[tag] = {"mean": float(np.mean(precs)), "std": float(np.std(precs)),
                        "n": len(precs)}
        print(f"[{tag}] normal-precision@{args.k} = {np.mean(precs):.4f} "
              f"(+-{np.std(precs):.3f} over {len(precs)})")
    if len(q_pred_normal_flags):
        print(f"[note] {sum(q_pred_normal_flags)}/{len(q_pred_normal_flags)} of these "
              "GT-normal queries also read as pred_normal (model's own no-finding "
              "verdict) -- the pathology-mode fallback lane")

    out = os.path.join(args.full_dir, "normal_precision.json")
    with open(out, "w") as f:
        json.dump({"k": args.k, "n_queries": len(sel),
                   "queries_pred_normal_frac": float(np.mean(q_pred_normal_flags)),
                   **results}, f, indent=2)
    print(f"[done] {out}")
    print("NORMAL_PRECISION_DONE")


if __name__ == "__main__":
    main()