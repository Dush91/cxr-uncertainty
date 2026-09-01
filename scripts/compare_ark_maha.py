#!/usr/bin/env python
"""Under-shift Mahalanobis confident-error comparison: RAD-DINO [CLS] vs Ark+
(Swin-L@768) as the OOD feature space for the production confident-error flag.

Prereq: ``scripts/extract_arkswin_ood.py`` produced
``runs/phase4_features/arkswin_feats_{kermany,covid}.npz`` (OpenI-C/D already
exist from the rep-mismatch work).

Protocol (mirrors ``scripts/eval_production.py`` exactly so the RAD-DINO number
reproduces production ~0.818/0.649):
  * Fit MahalanobisOOD (2 class-conditional LedoitWolf Gaussians) on OpenI-C
    features with Pneumonia gt -- the production fit label.
  * Score each shift site (kermany, covid) -- one Mahalanobis per image.
  * Confident set = top 10% by ``_confidence(p_bar_T, youden)`` over the
    Pneumonia-only valid rows (production uses pneumonia_only for shift sites).
  * AUROC of Mahalanobis vs error (decision != gt) on the confident set.
  * In-distribution OpenI-D sanity (all-valid, top-10%): RAD-DINO must reproduce
    ~0.835.

Compares: RAD-DINO (768-d of the 1536 zero-padded layout), Ark+ raw-1536, Ark+
PCA-256 (fit on OpenI-C, the rep-mismatch form). In-distribution showed raw-1536
ties RAD-DINO and PCA-256 degrades the global OOD score; this checks whether
under shift Ark+ raw-1536 BEATS RAD-DINO (a flagship upgrade) or merely ties.

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/compare_ark_maha.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.calibration import apply_temperature  # noqa: E402
from cxr_uncertainty.feature_uq import MahalanobisOOD  # noqa: E402

NIH = list(NIH_PATHOLOGIES)
PNEU = NIH.index("Pneumonia")
CONF_PCT = 10.0
ARR = _REPO / "runs" / "phase4_features"
SIDECAR = _REPO / "runs" / "augmented" / "conformal_sidecar_aug.json"


def _confidence(p, t):
    present = p >= t
    dp = np.where((1 - t) > 1e-6, (1 - t), 1e-6)
    da = np.where(t > 1e-6, t, 1e-6)
    return np.clip(np.where(present, (p - t) / dp, (t - p) / da), 0, 1)


def _load(split):
    a = np.load(ARR / f"arrays_{split}.npz", allow_pickle=True)
    k = np.load(ARR / f"arkswin_feats_{split}.npz", allow_pickle=True)
    # ark feats are row-aligned to arrays_{split}.npz by construction (the
    # extractor writes them in arrays id order), so use k["feats"] directly.
    # (Building a {id: row} dict via k["feats"][j] through the lazy npz copies the
    # full array per row -> ~24 GB bloat; direct indexing avoids it.)
    assert list(a["ids"]) == list(k["ids"]), f"{split}: ark feats not aligned to arrays"
    ark = np.asarray(k["feats"], dtype=np.float64)
    # subset to rows where BOTH feature spaces are finite (covid train images were
    # partly evicted after the RAD-DINO extraction, so Ark+ has NaNs there).
    rd_fin = np.isfinite(a["rad_feats"]).all(axis=1)
    ark_fin = np.isfinite(ark).all(axis=1)
    keep = rd_fin & ark_fin
    if (~keep).any():
        print(f"  [{split}] both-finite subset {int(keep.sum())}/{len(keep)} "
              f"(rad_fin={int(rd_fin.sum())}, ark_fin={int(ark_fin.sum())})")
    sub = {}
    for key in a.files:
        v = a[key]
        if hasattr(v, "shape") and v.shape and v.shape[0] == keep.shape[0]:
            v = v[keep]
        sub[key] = v
    return sub, ark[keep], keep


def _conf_rows(probs, gt, valid, T, youden, pneumonia_only):
    pT = apply_temperature(np.nanmean(probs, axis=1), T)
    pat_idx, pbar, g, img = [], [], [], []
    js = [PNEU] if pneumonia_only else range(len(NIH))
    for j in js:
        m = np.where(valid[:, j] > 0)[0]
        for i in m:
            pat_idx.append(j); pbar.append(float(pT[i, j]))
            g.append(int(gt[i, j])); img.append(i)
    pat_idx = np.array(pat_idx); pbar = np.array(pbar); g = np.array(g); img = np.array(img)
    thr = np.array([youden.get(NIH[j], 0.5) for j in pat_idx])
    dec = (pbar >= thr).astype(int)
    conf = _confidence(pbar, thr)
    conf_cut = float(np.percentile(conf, 100 - CONF_PCT))
    is_conf = conf >= conf_cut
    err = (dec != g).astype(int)
    return is_conf, err, img


def _auroc(maha_img, is_conf, err, img):
    sc = maha_img[img]; m = is_conf
    try:
        return roc_auc_score(err[m], sc[m]), int(m.sum()), int((m & (err == 1)).sum())
    except ValueError:
        return float("nan"), int(m.sum()), int((m & (err == 1)).sum())


def main() -> None:
    sc = json.load(open(SIDECAR))
    T = float(sc["temperature_T"]); youden = sc.get("youden") or {}
    ac, ark_c, _ = _load("openiC")
    gtc = ac["gt"].astype(float)
    rdc = ac["rad_feats"]

    # three feature spaces, each fit on OpenI-C with Pneumonia gt
    fit_rd = MahalanobisOOD().fit(rdc, gtc[:, PNEU].astype(int))
    fit_ark_raw = MahalanobisOOD().fit(ark_c, gtc[:, PNEU].astype(int))
    pca = PCA(256, random_state=0).fit(ark_c)
    fit_ark_pca = MahalanobisOOD().fit(pca.transform(ark_c), gtc[:, PNEU].astype(int))
    print(f"[fit] RAD-DINO classes={fit_rd.n_per_class_}  Ark+ raw classes={fit_ark_raw.n_per_class_}  "
          f"Ark+ PCA256 var={pca.explained_variance_ratio_.sum():.3f}\n")

    # in-distribution sanity (all-valid, top-10%) -- RAD-DINO must ~reproduce 0.835
    ad, ark_d, _ = _load("openiD")
    is_c, err_c, img_c = _conf_rows(ad["probs"], ad["gt"].astype(float),
                                    ad["valid"].astype(float), T, youden, pneumonia_only=False)
    print("== in-distribution OpenI-D (all valid, top-10% confident) ==")
    for name, s in [("RAD-DINO", fit_rd.score(ad["rad_feats"])),
                    ("Ark+ raw1536", fit_ark_raw.score(ark_d)),
                    ("Ark+ PCA256", fit_ark_pca.score(pca.transform(ark_d)))]:
        au, nc, ne = _auroc(s, is_c, err_c, img_c)
        print(f"  {name:14s} AUROC={au:.4f}  (n_conf={nc}, n_err={ne})")
    print("  [reference: production RAD-DINO in-dist = 0.8354]\n")
    del ad, ark_d, is_c, err_c, img_c
    import gc; gc.collect()

    print("== under shift (Pneumonia-only, top-10% confident) ==")
    print(f"{'site':10s} {'feature':12s} {'AUROC':>7s} {'n_conf':>7s} {'n_err':>6s}")
    for site in ["kermany", "covid"]:
        a, ark, _ = _load(site)
        is_c, err_c, img_c = _conf_rows(a["probs"], a["gt"].astype(float),
                                        a["valid"].astype(float), T, youden, pneumonia_only=True)
        rd = a["rad_feats"]
        for name, s in [("RAD-DINO", fit_rd.score(rd)),
                        ("Ark+ raw1536", fit_ark_raw.score(ark)),
                        ("Ark+ PCA256", fit_ark_pca.score(pca.transform(ark)))]:
            au, nc, ne = _auroc(s, is_c, err_c, img_c)
            print(f"  {site:10s} {name:12s} {au:>7.4f} {nc:>7d} {ne:>6d}")
            del s; gc.collect()
        print()
        del a, ark, rd, is_c, err_c, img_c; gc.collect()
    print("[reference: production RAD-DINO under shift = kermany 0.818, covid 0.649]")


if __name__ == "__main__":
    main()