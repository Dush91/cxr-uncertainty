#!/usr/bin/env python3
"""Agent 2 -- similar-case retrieval: build the HNSW reference index.

Indexes the **cal split** (patient-disjoint from eval queries) as the case
library. Embeddings come from the arrays produced by ``build_eval_arrays.py``
(default: the RAD-DINO member's ``rad_feats``) -- no new inference.

Label-free invariant: the vector store (index + feats + ids) contains ONLY
ids + vectors. Predictions / correctness metadata are derived from the npz
probs into a SEPARATE sidecar parquet, joined at query time
(``query_retrieval.py``). Ground truth never enters the index.

Recall validation: for a sample of held-out eval queries we compare HNSW
top-k against exact cosine top-k and report recall@k (VLDB 2025 filtered-
search guidance: at our scale exact is also sub-ms; HNSW is the scalable
methodology, validated here so the same code path is trustworthy at 113K+).

Usage:
  python scripts/build_retrieval_index.py --arr-dir runs/eval_arrays \
      --source rex --embed raddino --out runs/retrieval/rex_raddino
"""
from __future__ import annotations

import argparse
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

# Feature-member table: which npz key + which member supplies the embedding.
#   raddino    -> 1536-d RAD-DINO features (pretrained arrays, default)
#   resnet18/vit_tiny/convnext_tiny -> from-scratch seed-0 rad_feats (ablation)
EMBEDS = {
    "raddino": "rad_feats",
    "resnet18": "rad_feats",
    "vit_tiny": "rad_feats",
    "convnext_tiny": "rad_feats",
}


def arr_path(arr_dir: str, source: str, split: str) -> str:
    return os.path.join(arr_dir, source, f"arrays_{source}{split}.npz")


def load_split(arr_dir: str, source: str, split: str) -> dict:
    p = arr_path(arr_dir, source, split)
    if not os.path.exists(p):
        sys.exit(f"missing arrays: {p}")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def pooled_prediction(probs: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled-mean prediction per image over the valid pathologies.

    probs (N,M,P), valid (N,P) bool -> (pred_idx (N,) int, conf (N,) float,
    pred_prob (N,P) float = member-mean prob). Mirrors eval_baselines' pooled
    mean so the correct-flag matches the UQ evaluation's notion of a correct
    case. -1/uncertain labels never enter here (they only mark a reference
    row's correctness as unknown in the sidecar).
    """
    mean_p = probs.mean(axis=1)                       # (N,P)
    masked = np.where(valid, mean_p, -np.inf)
    pred_idx = masked.argmax(axis=1)                  # (N,)
    conf = masked.max(axis=1)
    return pred_idx, conf, mean_p


def build_sidecar(cal: dict) -> "pd.DataFrame":
    """Prediction/correctness metadata for the reference library (cal split)."""
    probs, valid, gt, ids = cal["probs"], cal["valid"].astype(bool), cal["gt"], cal["ids"]
    pathologies = [str(p) for p in cal["pathologies"]]
    pred_idx, conf, _ = pooled_prediction(probs, valid)
    n = len(ids)
    rows = {
        "image_id": [str(i) for i in ids],
        "pred_pathology": [pathologies[j] for j in pred_idx],
        "pred_conf": conf.astype(float),
        "gt_label": np.array([gt[k, pred_idx[k]] for k in range(n)], dtype=float),
    }
    rows["correct"] = rows["gt_label"] == 1.0
    rows["uncertain"] = rows["gt_label"] == -1.0
    rows.pop("gt_label")
    return pd.DataFrame(rows)


def exact_topk(feats: np.ndarray, q: np.ndarray, k: int, mask: np.ndarray | None = None):
    """Exact cosine top-k (feats are L2-normalized). mask: bool (N,) of allowed."""
    sims = feats @ q
    if mask is not None:
        sims = np.where(mask, sims, -np.inf)
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arr-dir", default="runs/eval_arrays")
    ap.add_argument("--source", default="rex")
    ap.add_argument("--embed", default="raddino", choices=sorted(EMBEDS))
    ap.add_argument("--out", default=None, help="default runs/retrieval/<source>_<embed>")
    ap.add_argument("--M", type=int, default=32, help="HNSW graph degree")
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall-queries", type=int, default=200,
                    help="number of eval-split queries used to validate HNSW recall@k vs exact")
    ap.add_argument("--k", type=int, default=10, help="k for recall validation")
    args = ap.parse_args()

    out = args.out or os.path.join("runs", "retrieval", f"{args.source}_{args.embed}")
    os.makedirs(out, exist_ok=True)

    cal = load_split(args.arr_dir, args.source, "cal")
    eval_ = load_split(args.arr_dir, args.source, "eval")
    feats_cal = np.asarray(cal["rad_feats"], dtype=np.float32)
    feats_eval = np.asarray(eval_["rad_feats"], dtype=np.float32)
    ids_cal = np.array([str(i) for i in cal["ids"]])
    ids_eval = np.array([str(i) for i in eval_["ids"]])

    # L2-normalize (hnswlib 'cosine' space re-normalizes internally; we store
    # normalized feats so exact-mode and reported similarities are plain dots).
    def norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)

    feats_cal, feats_eval = norm(feats_cal), norm(feats_eval)

    # --- HNSW index over the cal reference library ---------------------------
    n, dim = feats_cal.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=args.ef_construction, M=args.M,
                     random_seed=args.seed)
    index.add_items(feats_cal, np.arange(n))
    index.set_ef(max(64, args.k * 10))
    index.save_index(os.path.join(out, "index.bin"))

    # --- label-free store + sidecar (separate file, joined at query time) ----
    np.savez_compressed(os.path.join(out, "store.npz"),
                        feats=feats_cal, ids=ids_cal,
                        member_keys=np.array(cal["member_keys"], dtype=object)
                        if "member_keys" in cal else np.array([], dtype=object))
    sidecar_path = os.path.join(out, "sidecar.parquet")
    build_sidecar(cal).to_parquet(sidecar_path, index=False)

    # --- recall validation: eval queries, exact vs HNSW ----------------------
    rng = np.random.default_rng(args.seed)
    q_idx = rng.choice(len(feats_eval), size=min(args.recall_queries, len(feats_eval)),
                       replace=False)
    recalls = []
    for qi in q_idx:
        q = feats_eval[qi]
        ex, _ = exact_topk(feats_cal, q, args.k)
        hn, _ = index.knn_query(q, k=args.k)
        recalls.append(len(set(ex) & {int(i) for i in np.asarray(hn).ravel()}) / args.k)
    recall = float(np.mean(recalls)) if len(recalls) else float("nan")

    meta = {
        "source": args.source, "embed": args.embed, "feature_key": EMBEDS[args.embed],
        "dim": int(dim), "n_reference": int(n), "M": args.M,
        "ef_construction": args.ef_construction, "space": "cosine",
        "recall_at_k": recall, "k": args.k, "n_recall_queries": int(len(q_idx)),
        "reference_split": "cal", "query_split": "eval",
        "label_free": "store holds ids+feats only; sidecar.parquet separate",
    }
    import json
    with open(os.path.join(out, "index_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[index] {n} vectors, dim={dim}, space=cosine, M={args.M}, efC={args.ef_construction}")
    print(f"[sidecar] {sidecar_path}  (n={len(cal['ids'])})")
    ns = build_sidecar(cal)
    print(f"[sidecar] correct={int(ns['correct'].sum())}  incorrect={int((~ns['correct']).sum() - ns['uncertain'].sum())}  uncertain={int(ns['uncertain'].sum())} of {len(ns)}")
    print(f"[recall] HNSW recall@{args.k} = {recall:.4f} over {len(q_idx)} eval queries"
          f"{'' if recall >= 0.99 else '  <-- LOW: raise ef or M'}")
    print(f"[done] {out}/index.bin saved via hnswlib; RETRIEVAL_BUILD_DONE")


if __name__ == "__main__":
    main()