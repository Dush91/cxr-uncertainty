"""OOD-detection evaluation from the saved phase-4 feature arrays (no re-inference).

The phase-4 run saved per-site arrays (``runs/phase4_features/arrays_*.npz``)
with RAD-DINO logits, 768-d [CLS] features, energy, and Mahalanobis scores for
OpenI-C (fit set), OpenI-D (in-distribution eval), Kermany and COVID (under
shift). This script turns those into the OOD-detection headline numbers:

  * AUROC        -- does the score separate ID (OpenI-D) from OOD (Kermany/COVID)?
  * FPR@95TPR    -- false-positive rate on ID at 95% true-positive rate on OOD.

Scores compared (uncertainty_evaluation.md §2.1 baseline stack):
  * mahalanobis  -- class-conditional Gaussian density on RAD-DINO features
                    (Lee et al. 2018), fit on OpenI-C.
  * energy       -- 14-class energy from RAD-DINO logits (Liu et al. 2020);
                    stored as ``-logsumexp`` so higher = more OOD.
  * knn          -- mean distance to the k nearest OpenI-C features (Sun et al.
                    2022); the strongest unmeasured competitor (Woodland et al.
                    2024: kNN often beats Mahalanobis in medical imaging).

Usage:
    python scripts/eval_ood_detection.py \
        --arrays runs/phase4_features --out runs/ood_detection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cxr_uncertainty.statistics import ood_detection_metrics

SITES = ["kermany", "covid"]
SCORES = ["mahalanobis", "energy", "knn"]
K = 5


def _knn_distances(query: np.ndarray, ref: np.ndarray, k: int = K) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(ref))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(ref)
    dists, _ = nn.kneighbors(query)
    return dists.mean(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays", default="runs/phase4_features",
                   help="dir holding arrays_openiC/D/kermany/covid.npz")
    p.add_argument("--out", default="runs/ood_detection")
    args = p.parse_args()

    base = Path(args.arrays)
    C = np.load(base / "arrays_openiC.npz", allow_pickle=True)
    D = np.load(base / "arrays_openiD.npz", allow_pickle=True)
    ref_feats = np.asarray(C["rad_feats"], dtype=np.float64)

    # kNN distances: ID = OpenI-D to OpenI-C; OOD = Kermany/COVID to OpenI-C.
    id_knn = _knn_distances(np.asarray(D["rad_feats"], dtype=np.float64), ref_feats)
    id_scores = {
        "mahalanobis": np.asarray(D["mahalanobis"], dtype=float),
        "energy": np.asarray(D["energy"], dtype=float),   # stored HIGH = OOD
        "knn": id_knn,
    }

    results = {}
    for site in SITES:
        arr = np.load(base / f"arrays_{site}.npz", allow_pickle=True)
        ood_scores = {
            "mahalanobis": np.asarray(arr["mahalanobis"], dtype=float),
            "energy": np.asarray(arr["energy"], dtype=float),
            "knn": _knn_distances(np.asarray(arr["rad_feats"], dtype=np.float64),
                                  ref_feats),
        }
        results[site] = {}
        for name in SCORES:
            m = ood_detection_metrics(id_scores[name], ood_scores[name])
            results[site][name] = {
                "auroc": round(m["auroc"], 4),
                "fpr_at_95tpr": (round(m["fpr_at_95tpr"], 4)
                                 if m["fpr_at_95tpr"] is not None else None),
            }
            print(f"[ood] {site:8s} {name:12s} AUROC={m['auroc']:.4f}  "
                  f"FPR@95TPR={m['fpr_at_95tpr']:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ood_detection.json", "w") as f:
        json.dump({"id_site": "openiD", "ood_sites": SITES,
                   "scores": SCORES, "knn_k": K, "results": results},
                  f, indent=2)
    print(f"[ood] wrote {out / 'ood_detection.json'}")


if __name__ == "__main__":
    main()
