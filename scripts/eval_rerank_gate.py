#!/usr/bin/env python3
"""Acceptance gate for the label-agreement rerank (Agent 2).

Compares retrieval quality before/after adding the report-label rerank to the
FULL 129k library. No app involvement -- exact offline emulation of the app's
two-stage query: visual top-(K*4) oversample under the mode filter, rerank by
score = (1-lambda)*cosine + lambda*agreement(posterior, candidate labels),
truncate to top-K.

Design honesty recap: the query has NO ground truth at retrieval time -- the
label signal used is the query's pooled model posterior (report-derived labels
exist only on the candidate side). GT for queries is used here ONLY because
this is an offline evaluation.

Arms:
  old_pathology  argmax-bucket filter, pure cosine                      (status quo)
  new_pathology  same filter + rerank(=0.3)                             (candidate)
  old_plain      no filter, pure cosine
  new_plain      no filter + rerank

Queries: eval rows with >=1 certain positive report label AND >=1 class
reaching its Youden threshold (confident) -- the population pathology mode
serves.

Metrics (mean over the top-K neighbors, then over queries):
  label_jaccard@K   Jaccard(neighbor certain-positives, query certain-positives)
  cosine@K          raw visual similarity (guard against similarity collapse)
  posterior_agree@K agreement(posterior, neighbor certain-positives)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402

RERANK_MODES = ("pathology", "plain")


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard over certain-positive label sets (a,b are 0/1 int vectors)."""
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def agreement(p_q: np.ndarray, y_c: np.ndarray, valid_c: np.ndarray) -> float:
    """Fraction of the query's predicted probability mass on classes the
    candidate's report confirms (certain positives). -1 excluded both ways."""
    pos = valid_c & (y_c == 1)
    denom = float(p_q[valid_c & (y_c != 0)].sum())
    return float(p_q[pos].sum() / denom) if denom > 1e-6 else 0.0


def topk_exact(feats, q, k, allow):
    sims = np.where(allow, feats @ q, -np.inf).astype(np.float64)
    return np.argsort(-sims)[:k], sims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-dir", default="runs/rex_phase7/retrieval/"
                                         "rex_adapted_raddino_full")
    ap.add_argument("--arr-eval", default="runs/rex_phase7/eval_arrays/"
                                          "rex_adapted_s0/arrays_rex_adaptedeval.npz")
    ap.add_argument("--arr-cal", default="runs/rex_phase7/eval_arrays/"
                                         "rex_adapted_s0/arrays_rex_adaptedcal.npz")
    ap.add_argument("--calibration", default="runs/app_rex_adapted/calibration.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--oversample", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--max-queries", type=int, default=2000)
    args = ap.parse_args()

    store = np.load(os.path.join(args.full_dir, "store.npz"), allow_pickle=True)
    feats = norm(np.asarray(store["feats"], dtype=np.float32))
    ids = np.array([str(i) for i in store["ids"]])
    meta = pd_read_sidecar(os.path.join(args.full_dir, "sidecar.parquet")) \
        .set_index("image_id").reindex(ids).reset_index(drop=True)
    assert meta["gt_labels"].notna().all(), "run scripts/augment_sidecar_labels.py first"

    gl = np.stack(meta["gt_labels"].values).astype(np.int8)
    gv = np.stack(meta["gt_valid"].values).astype(bool)
    split_col = meta["split"].values
    pat_col = meta["pred_pathology"].fillna("").values
    uncertain_col = meta["uncertain"].fillna(False).values.astype(bool)
    id_pos = {i: k for k, i in enumerate(ids)}

    ev = np.load(args.arr_eval, allow_pickle=True)
    q_feats = norm(np.asarray(ev["rad_feats"], dtype=np.float32))
    q_ids = np.array([str(i) for i in ev["ids"]])
    q_probs = np.asarray(ev["probs"], dtype=np.float64)
    q_valid = ev["valid"].astype(bool)
    q_gt = np.asarray(ev["gt"], dtype=np.int8)
    mean = np.nanmean(q_probs, axis=1)
    argmax = np.where(q_valid & np.isfinite(mean), mean, -np.inf).argmax(axis=1)
    pats = [str(p) for p in ev["pathologies"]]
    assert pats == list(NIH_PATHOLOGIES), "unexpected pathology order"

    with open(args.calibration) as f:
        yj = json.load(f).get("youden", {})
    thr = np.array([0.5 if yj.get(p) is None else float(yj[p]) for p in pats])
    reaches = q_valid & np.isfinite(mean) & (mean >= thr[None, :])
    q_pos = (q_gt == 1)
    sel = np.where(reaches.any(axis=1) & q_pos.any(axis=1))[0]
    if args.max_queries:
        sel = sel[: args.max_queries]
    print(f"[queries] {len(sel)} confident eval queries with >=1 certain positive label")

    q_p_clean = np.where(q_valid, np.nan_to_num(mean), 0.0)  # (N,P) cleaned posteriors

    # per-library-row model decision vector (N,P): cal/eval rows from their
    # arrays (pooled mean vs Youden); train rows False (contrast pool = cal+eval)
    dec = np.zeros((len(ids), len(pats)), dtype=bool)
    pos_map = {iid: k for k, iid in enumerate(ids)}
    for arr_path in (args.arr_eval, args.arr_cal):
        d = np.load(arr_path, allow_pickle=True)
        pbar = np.nanmean(np.asarray(d["probs"], dtype=np.float64), axis=1)
        dvr = np.asarray(d["valid"]).astype(bool) & (pbar >= thr[None, :])
        for k, iid in enumerate([str(i) for i in d["ids"]]):
            j = pos_map.get(iid)
            if j is not None:
                dec[j] = dvr[k]

    arms = {
        "old_pathology": ("pathology", False),
        "new_pathology": ("pathology", True),
        "old_plain": ("plain", False),
        "new_plain": ("plain", True),
    }
    results = {}
    for tag, (mode, rerank) in arms.items():
        jcs, css, ags = [], [], []
        for qi in sel:
            allow = np.ones(len(ids), dtype=bool)
            allow &= pat_col != ""  # rows with metadata
            if mode == "pathology":
                allow &= (pat_col == pats[int(argmax[qi])]) & ~uncertain_col
            allow &= ids != q_ids[qi]
            ov = list(topk_exact(feats, q_feats[qi], args.oversample, allow)[0])

            if rerank:
                scores = np.array([(1 - args.lam) * (feats[i] @ q_feats[qi])
                                   + args.lam * agreement(q_p_clean[qi], gl[i], gv[i])
                                   for i in ov])
                ov = [i for _, i in sorted(zip(-scores, ov))]

            top = ov[: args.k]
            jcs.append(float(np.mean([jaccard(gl[i] == 1, q_pos[qi]) for i in top])))
            css.append(float(np.mean([feats[i] @ q_feats[qi] for i in top])))
            ags.append(float(np.mean([agreement(q_p_clean[qi], gl[i], gv[i])
                                      for i in top])))
        results[tag] = {"label_jaccard@k": float(np.mean(jcs)),
                        "cosine@k": float(np.mean(css)),
                        "posterior_agree@k": float(np.mean(ags))}
        print(f"[{tag:<13}] jaccard {results[tag]['label_jaccard@k']:.4f} | "
              f"cos {results[tag]['cosine@k']:.4f} | "
              f"agree {results[tag]['posterior_agree@k']:.4f}")

    out = os.path.join(args.full_dir, "rerank_gate.json")

    # ------------------------------------------------------------------
    # Flag-keyed gate: over eval rows where Region B FIRES (confident AND
    # top-50% epistemic std, app semantics), compare the status-quo
    # argmax-bucket neighborhood against the flag-keyed lane. The key comes
    # from model internals (conf + std) exactly as at deployment; query GT is
    # used ONLY to score neighbor quality offline (flag-class report
    # precision@8).
    # ------------------------------------------------------------------
    from cxr_uncertainty.reanalyze import _confidence  # noqa: E402
    with open(args.calibration) as f:
        cal = json.load(f)
    conf_cut = float(cal["conf_cut"])
    unc_cut = float(cal["unc_cut"])
    conf_mat = _confidence(mean, thr[None, :])          # (N,P), NaN -> NaN -> not confident
    with np.errstate(invalid="ignore"):
        is_conf = conf_mat >= conf_cut
    std_mat = np.nanstd(q_probs, axis=1)                # (N,P) member std
    disagree = is_conf & (std_mat >= unc_cut)
    fire = disagree.any(axis=1)
    # key = flagged class with the highest calibrated confidence (app rule)
    flag_conf = np.where(disagree, conf_mat, -np.inf)
    flag_idx = flag_conf.argmax(axis=1)
    sel_flag = np.where(fire)[0]
    if args.max_queries:
        sel_flag = sel_flag[: args.max_queries]
    print(f"[flag-gate] {len(sel_flag)} Region-B-firing eval queries")

    flag_arms = {
        "old_flag_pathology": ("pathology", False),   # status quo: argmax bucket
        "new_flag": ("flag", False),                  # flag pool, pure cosine
        "new_flag_rerank": ("flag", True),            # flag pool + agreement rerank
        "new_contrast": ("contrast", False),          # report-confirmed X AND model-read X
    }
    fp_res = {}
    for tag, (mode, rerank) in flag_arms.items():
        prec, css, pools, decs = [], [], [], []
        for qi in sel_flag:
            fi = int(flag_idx[qi])
            allow = np.ones(len(ids), dtype=bool)
            allow &= pat_col != ""  # rows with metadata
            if mode == "flag":
                allow &= (gl[:, fi] == 1) & gv[:, fi]
            elif mode == "contrast":
                allow &= dec[:, fi] & (gl[:, fi] == 1) & gv[:, fi]
            else:
                allow &= (pat_col == pats[int(argmax[qi])]) & ~uncertain_col
            allow &= ids != q_ids[qi]
            pools.append(int(allow.sum()))
            if allow.sum() == 0:
                prec.append(0.0)
                css.append(0.0)
                decs.append(0.0)
                continue
            ov = list(topk_exact(feats, q_feats[qi], args.oversample, allow)[0])
            if rerank:
                scores = np.array([(1 - args.lam) * (feats[i] @ q_feats[qi])
                                   + args.lam * agreement(q_p_clean[qi], gl[i], gv[i])
                                   for i in ov])
                ov = [i for _, i in sorted(zip(-scores, ov))]
            top = ov[: args.k]
            prec.append(float(np.mean([(gl[i, fi] == 1) for i in top])))
            css.append(float(np.mean([feats[i] @ q_feats[qi] for i in top])))
            decs.append(float(np.mean([dec[i, fi] for i in top])))
        fp_res[tag] = {"flag_precision@k": float(np.mean(prec)),
                       "cosine@k": float(np.mean(css)),
                       "model_detected_share@k": float(np.mean(decs)),
                       "mean_pool": float(np.mean(pools))}
        print(f"[{tag:<18}] flag-prec {fp_res[tag]['flag_precision@k']:.4f} | "
              f"cos {fp_res[tag]['cosine@k']:.4f} | "
              f"model-detect {fp_res[tag]['model_detected_share@k']:.4f} | "
              f"pool {fp_res[tag]['mean_pool']:.0f}")

    # ------------------------------------------------------------------
    # Neighbor-vote score (Agent 2 votes): over Region-B-firing eval queries,
    # vote@8 = share of the top-8 UNFILTERED nearest neighbors whose report
    # confirms the flagged class. GT-free at deployment (reference-side report
    # labels only). Offline test: does 1-vote rank the confident errors among
    # firing rows? Compared against epistemic std (max over flagged classes).
    # ------------------------------------------------------------------
    def auroc(y: np.ndarray, s: np.ndarray) -> float:
        """Rank-based AUROC with tie-corrected ranks (Mann-Whitney U)."""
        pos, neg = int(y.sum()), int((1 - y).sum())
        if pos == 0 or neg == 0:
            return float("nan")
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt)
        avg = (cum - cnt + cum + 1) / 2.0  # average rank per tie group
        r = avg[inv]
        return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    def boot_ci(y: np.ndarray, s: np.ndarray, n: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            a = auroc(y[idx], s[idx])
            if np.isfinite(a):
                vals.append(a)
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    # confident-wrong = flagged call where the model's decision (class reaches
    # its Youden threshold) contradicts the certain report label
    wrong_cls = (q_gt != -1) & (reaches.astype(int) != (q_gt == 1).astype(int))
    conf_wrong_row = (disagree & wrong_cls).any(axis=1)   # full-length, indexed by sel_flag
    qy = conf_wrong_row[sel_flag]
    n_err = int(qy.sum())
    print(f"[vote] firing rows with a confident-WRONG flagged call: {n_err}/{len(qy)}")

    votes, stds, confs = [], [], []
    for qi in sel_flag:
        fi = int(flag_idx[qi])
        allow = (pat_col != "") & (ids != q_ids[qi])
        top = list(topk_exact(feats, q_feats[qi], args.k, allow)[0])
        votes.append(float(np.mean([(gl[i, fi] == 1) for i in top])))
        stds.append(float(np.nanmax(std_mat[qi][disagree[qi]])) if disagree[qi].any()
                    else float(np.nanmax(std_mat[qi])))
        confs.append(float(np.max(conf_mat[qi][disagree[qi]])) if disagree[qi].any()
                     else float(np.nanmax(conf_mat[qi])))
    votes, stds, confs = np.array(votes), np.array(stds), np.array(confs)

    y = qy.astype(int)
    vote_res = {}
    if 0 < n_err < len(qy):
        a_vote = auroc(y, 1 - votes)
        a_std = auroc(y, -stds)
        a_conf = auroc(y, 1 - confs)
        ci_vote = boot_ci(y, 1 - votes)
        vote_res = {"auroc_1_minus_vote": a_vote, "auroc_1_minus_vote_ci95": ci_vote,
                    "auroc_neg_max_std_flagged": a_std, "auroc_1_minus_conf_flagged": a_conf,
                    "mean_vote_confident_correct": float(votes[~qy].mean()),
                    "mean_vote_confident_wrong": float(votes[qy].mean()),
                    "n_firing": int(len(qy)), "n_confident_wrong": n_err}
        print(f"[vote] AUROC(1-vote) = {a_vote:.3f} (CI95 {ci_vote[0]:.3f}-{ci_vote[1]:.3f})"
              f" | AUROC(-max-std) = {a_std:.3f} | AUROC(1-conf) = {a_conf:.3f}")
        print(f"[vote] mean vote: confident-correct {votes[~qy].mean():.3f} vs "
              f"confident-wrong {votes[qy].mean():.3f}")
        print("[vote] POWER CAVEAT: only {n} confident-error rows among {m} firing "
              "queries -- CI is wide by construction".format(n=n_err, m=len(qy)))
    else:
        print("[vote] degenerate error count -- AUROC undefined")
    fp_res["neighbor_vote"] = vote_res

    with open(out, "w") as f:
        json.dump({"k": args.k, "lambda": args.lam, "n_queries": len(sel),
                   "n_flag_queries": len(sel_flag), **results,
                   "flag_gate": fp_res}, f, indent=2)
    print("[done] rerank_gate.json (updated with flag_gate + neighbor_vote)")
    print("RERANK_GATE_DONE")


def pd_read_sidecar(path):
    import pandas as pd
    return pd.read_parquet(path)


if __name__ == "__main__":
    main()