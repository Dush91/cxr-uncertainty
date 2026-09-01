#!/usr/bin/env python3
"""Agent 2 -- similar-case retrieval: query the indexed case library.

Two Agent-2 query modes over the HNSW index built by ``build_retrieval_index.py``:

  --mode correct     "Show me similar images where the model was correct."
                     Restricts matches to reference cases whose pooled
                     ensemble prediction matched their label (correct=True,
                     uncertain rows excluded).
  --mode plain       Unfiltered nearest neighbours.
  --mode flag        Keyed on Agent 1: candidates whose REPORT confirms the
                     class passed via --flag-class (the Region-B-flagged
                     class -- model internals; no query GT at deployment).
  --mode contrast    The other half of the flag review: candidates whose
                     report confirms class X AND whose model read is
                     X-positive (the model DID call it there). X =
                     --flag-class, else the query's argmax. cal+eval pool
                     (per-class decisions come from the arrays).

Design rules honoured here:
  * Never returns the query image itself (self-exclusion by id).
  * The index/store carry ids + vectors only; predictions/correctness come
    from sidecar.parquet joined at query time. Ground truth is never printed
    and never stored in the index.
  * Similarity is cosine (both feats L2-normalized -> dot product).
  * hnswlib's knn_query filter applies the metadata predicate *during graph
    traversal* (inline filtering, the VLDB-2025-recommended pattern) -- this
    avoids the classic post-filter recall collapse when the filter is
    selective. --search exact brute-forces over the filtered candidates and
    is available as the ground-truth validation path.

Usage:
  python scripts/query_retrieval.py --index-dir runs/retrieval/rex_raddino \
      --arr-dir runs/eval_arrays --source rex --id <eval_image_id> \
      --mode correct --k 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import hnswlib
except ImportError as e:  # pragma: no cover
    sys.exit("hnswlib is required (pip install hnswlib): %s" % e)
try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    sys.exit("pandas is required: %s" % e)

# reuse the pooled-prediction definition from the builder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_Q = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_Q)  # so cxr_uncertainty resolves without installation
from build_retrieval_index import load_split, pooled_prediction  # noqa: E402


def get_query_feat(arr_dir: str, source: str, split: str, image_id: str,
                   embed: str = "raddino") -> np.ndarray:
    """L2-normalized embedding for a query image from the eval/cal arrays."""
    d = load_split(arr_dir, source, split)
    ids = np.array([str(i) for i in d["ids"]])
    hit = np.where(ids == image_id)[0]
    if len(hit) == 0:
        sys.exit(f"image_id {image_id!r} not found in {split} arrays for {source}")
    f = np.asarray(d["rad_feats"][hit[0]], dtype=np.float32)
    return f / max(np.linalg.norm(f), 1e-12)


def get_query_pred(arr_dir: str, source: str, split: str, image_id: str):
    """(pred_pathology, conf) for the query image under the pooled-mean rule."""
    d = load_split(arr_dir, source, split)
    ids = np.array([str(i) for i in d["ids"]])
    hit = np.where(ids == image_id)[0]
    if len(hit) == 0:
        sys.exit(f"image_id {image_id!r} not found in {split} arrays")
    k = int(hit[0])
    pathologies = [str(p) for p in d["pathologies"]]
    pred_idx, conf, _ = pooled_prediction(d["probs"][k:k + 1], d["valid"][k:k + 1].astype(bool))
    return pathologies[int(pred_idx[0])], float(conf[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--arr-dir", default="runs/eval_arrays")
    ap.add_argument("--source", default="rex")
    ap.add_argument("--embed", default="raddino")
    ap.add_argument("--id", required=True, help="query image_id (from the eval split, unless --split cal)")
    ap.add_argument("--split", default="eval", choices=["eval", "cal"])
    ap.add_argument("--mode", default="correct",
                    choices=["correct", "normal", "flag", "contrast", "plain"])
    ap.add_argument("--flag-class", default=None,
                    help="pathology key for --mode flag (the Region-B-flagged "
                         "class); candidates = report-confirmed positives")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--search", default="hnsw", choices=["hnsw", "exact"])
    ap.add_argument("--ef", type=int, default=256, help="hnswlib ef-search (oversampled)")
    ap.add_argument("--lam", type=float, default=0.3,
                    help="rerank weight on report-label agreement "
                         "(0 disables; applied to plain/flag/contrast modes only)")
    ap.add_argument("--calibration", default="runs/app_rex_adapted/calibration.json",
                    help="Youden thresholds JSON (contrast-lane model-read filter)")
    args = ap.parse_args()

    store = np.load(os.path.join(args.index_dir, "store.npz"), allow_pickle=True)
    feats, ids = store["feats"], np.array([str(i) for i in store["ids"]])
    sidecar = pd.read_parquet(os.path.join(args.index_dir, "sidecar.parquet"))

    q = get_query_feat(args.arr_dir, args.source, args.split, args.id, args.embed)
    q_pred, q_conf = get_query_pred(args.arr_dir, args.source, args.split, args.id)
    if q.shape[0] != feats.shape[1]:
        sys.exit(f"dim mismatch: query features {q.shape[0]}-d vs index {feats.shape[1]}-d "
                 f"(pass --source/--embed matching the built index)")

    allow = np.ones(len(ids), dtype=bool)
    allow &= np.asarray(pd.Index(ids).isin(sidecar["image_id"]))  # only rows with metadata
    meta = sidecar.set_index("image_id").reindex(ids)
    split_col = meta["split"].values if "split" in meta \
        else np.full(len(ids), "cal", dtype=object)
    gt_normal_col = meta["gt_normal"].fillna(False).values if "gt_normal" in meta \
        else np.zeros(len(ids), dtype=bool)
    youden: dict = {}
    if os.path.exists(args.calibration):
        with open(args.calibration) as f:
            youden = json.load(f).get("youden", {})
    q_d = load_split(args.arr_dir, args.source, args.split)
    q_pats = [str(p) for p in q_d["pathologies"]]
    idx0 = int(np.where(np.array([str(i) for i in q_d["ids"]]) == args.id)[0][0])
    q_mean_row = np.nanmean(np.asarray(q_d["probs"][idx0], dtype=np.float64), axis=0)
    q_valid_row = q_d["valid"][idx0].astype(bool)
    thr_row = np.array([0.5 if youden.get(p) is None else float(youden.get(p, 0.5))
                        for p in q_pats])
    q_pred_normal = bool(q_valid_row.sum() > 0 and not
                         (np.isfinite(q_mean_row) & q_valid_row
                          & (q_mean_row >= thr_row)).any())

    lane = ""
    gl = gv = None
    if "gt_labels" in sidecar.columns:
        gl = np.stack(meta["gt_labels"].values).astype(np.int8)
        gv = np.stack(meta["gt_valid"].values).astype(bool)
    if args.mode == "flag":
        if gl is None:
            sys.exit("flag mode needs the label-augmented sidecar "
                     "(run scripts/augment_sidecar_labels.py)")
        if not args.flag_class:
            sys.exit("--mode flag requires --flag-class <pathology>")
        if args.flag_class not in q_pats:
            sys.exit(f"unknown flag class {args.flag_class!r}; "
                     f"choose from {q_pats}")
        fi = q_pats.index(args.flag_class)
        allow &= (gl[:, fi] == 1) & gv[:, fi]
        lane = (f"Region-B flag on {args.flag_class} -> report-confirmed "
                f"{args.flag_class} references (no query GT used)")
    elif args.mode == "contrast":
        if gl is None:
            sys.exit("contrast mode needs the label-augmented sidecar "
                     "(run scripts/augment_sidecar_labels.py)")
        key = args.flag_class or q_pred
        if key not in q_pats:
            sys.exit(f"unknown contrast class {key!r}; choose from {q_pats}")
        xi = q_pats.index(key)
        # per-library-row model decision vector: cal/eval rows from the arrays
        # (pooled mean vs Youden threshold); train rows False (no stored probs)
        pos_map = {iid: k for k, iid in enumerate(ids)}
        thr_arr = np.array([0.5 if youden.get(p) is None else float(youden[p])
                            for p in q_pats])
        dec = np.zeros((len(ids), len(q_pats)), dtype=bool)
        for spl in ("eval", "cal"):
            d = load_split(args.arr_dir, args.source, spl)
            d_ids = [str(i) for i in d["ids"]]
            pbar = np.nanmean(np.asarray(d["probs"], dtype=np.float64), axis=1)
            dv = np.asarray(d["valid"]).astype(bool) & (pbar >= thr_arr[None, :])
            for k2, iid in enumerate(d_ids):
                j = pos_map.get(iid)
                if j is not None:
                    dec[j] = dv[k2]
        allow &= dec[:, xi] & (gl[:, xi] == 1) & gv[:, xi]
        lane = (f"contrast: model read {key} AND report confirms {key} "
                "(cal+eval pool -- the model DID call it there)")
    elif args.mode == "correct":
        allow &= meta["correct"].fillna(False).values & ~meta["uncertain"].fillna(False).values \
            & (split_col != "train")   # train excluded: adapted members memorized these
        lane = "pool = cal+eval only (train excluded: memorization)"
    elif args.mode == "normal":
        allow &= gt_normal_col & ~meta["uncertain"].fillna(False).values
        lane = "GT-normal references"

    allow &= ids != args.id  # never return the query itself
    n_allow = int(allow.sum())
    if n_allow == 0:
        sys.exit("filter leaves zero candidate references -- widen the query")

    if args.search == "exact":
        sims = feats @ q
        sims = np.where(allow, sims, -np.inf)
        order = np.argsort(-sims)[:args.k * 4]
        neighbours = [(int(i), float(sims[i])) for i in order]
    else:
        index_path = os.path.join(args.index_dir, "index.bin")
        if not os.path.exists(index_path):
            sys.exit(f"missing {index_path}; run build_retrieval_index.py first")
        index = hnswlib.Index(space="cosine", dim=feats.shape[1])
        index.load_index(index_path)
        index.set_ef(max(args.ef, args.k * 10))
        labels, _ = index.knn_query(q, k=min(args.k * 4, index.get_current_count()),
                                    filter=lambda li: bool(allow[int(li)]))
        seen = []
        for li in labels[0]:
            li = int(li)
            if allow[li] and ids[li] != args.id:
                seen.append((li, float(feats[li] @ q)))
        neighbours = seen

    # second-stage rerank: blend visual cosine with report-label agreement
    # (query side = model posterior only; report labels are reference-side)
    do_rerank = args.mode in ("plain", "flag", "contrast") \
        and args.lam is not None and gl is not None
    if do_rerank:
        p_q = np.where(q_valid_row, q_mean_row, 0.0)

        def _agree(li: int) -> float:
            y, v = gl[li], gv[li]
            denom = float(p_q[v & (y != 0)].sum())
            return float(p_q[v & (y == 1)].sum() / denom) if denom > 1e-6 else 0.0
        neighbours.sort(key=lambda lc: -((1.0 - args.lam) * lc[1]
                                         + args.lam * _agree(lc[0])))
        lane = (lane + " | " if lane else "") \
            + f"rerank (lambda={args.lam:.1f}): + report-label agreement"
    neighbours = neighbours[:args.k]

    print(f"query        : {args.id}  (split={args.split})")
    print(f"prediction   : {q_pred}  conf={q_conf:.3f}"
          + ("  [reads as no-finding]" if q_pred_normal else ""))
    print(f"mode={args.mode}  search={args.search}  candidates_allowed={n_allow}/{len(ids)}"
          + (f"  lane: {lane}" if lane else ""))
    print("-" * 84)
    print(f"{'rank':>4}  {'image_id':<28} {'sim':>6}  {'pred_pathology':<24} {'conf':>6}  {'spl':<5} correct")
    for r, (li, s) in enumerate(neighbours, 1):
        row = meta.iloc[li]
        spl = (str(row["split"]) if "split" in meta.columns else "-")
        print(f"{r:>4}  {ids[li]:<28} {s:>6.3f}  {row['pred_pathology']:<24} "
              f"{row['pred_conf']:>6.3f}  {spl:<5} "
              f"{'Y' if row['correct'] else ('?' if row['uncertain'] else 'N')}")
    print("-" * 84)
    print(f"top-{len(neighbours)} similar reference cases returned")


if __name__ == "__main__":
    main()