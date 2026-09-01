#!/usr/bin/env python
"""2nd class-conditional-density (representation-mismatch) confident-error flag on
the ConvNeXt-V2 member's features -- the SECOND feature space (plan §R.2
follow-up). A different inductive bias (ImageNet-CNN, local high-pass) from
RAD-DINO (ViT, global low-pass); the hypothesis is that pathologies RAD-DINO [CLS]
could NOT separate (Effusion/Infiltration/Nodule/Hernia -- representation-limited,
proven by the §R.2 Mahalanobis-centroid NEGATIVE) may separate under ConvNeXt-V2's
features. The two detectors are COMPLEMENTARY: RAD-DINO ships 5; ConvNeXt-V2 may
ship a different/disjoint set, and combining (flag if EITHER fires) broadens
coverage without raising the per-detector false-flag rate.

PROTOCOL: identical to ``fit_repmismatch.py`` (the RAD-DINO detector) so the two
are comparable -- same fit-data selection (OpenI-C only for non-rare; OpenI-C+
role-B for the genuinely rare Mass/Pneumothorax/Fibrosis), same 2-fold CV
threshold, same AUROC ship guard, same score function (log-likelihood mismatch --
the proven-better score; the Mahalanobis-centroid score was strictly dominated).
ONLY the feature space differs: ConvNeXt-V2-Large's 1536-d penultimate embedding
(``feat_dim=1536``), loaded from ``runs/phase4_features/cv2_feats_{split}.npz``
(aligned by row order to ``arrays_{split}.npz``).

Leak-free: the convnextv2 encoder was fine-tuned on NIH/CheXpert (role-A) only --
it NEVER saw OpenI -- so OpenI is a leak-free OOD eval for it (same as RAD-DINO).
role-B is the training held-out (NIH+CheXpert), used only for the rare pathologies'
Gaussian count (the same leak-free rationale as the RAD-DINO detector). Decisions
and confidence come from the SAME 3-member ensemble pooled p_bar already saved in
``arrays_*.npz`` -- unchanged. OpenI-D never touches fit/threshold; it only
MEASURES.

Ship (honest Pareto): ConvNeXt-V2 mismatch AUROC beats the model-confidence
baseline AND > 0.6 AND >= 5 confident errors on OpenI-D. Reported alongside the
RAD-DINO detector's results so the COMPLEMENT (pathologies one ships but not the
other) is visible.

Prereq: scripts/extract_cv2_features.py produced cv2_feats_{openiC,openiD,roleB}.npz.
Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/fit_repmismatch_cv2.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.calibration import apply_temperature  # noqa: E402
from cxr_uncertainty.repmismatch import RepMismatch  # noqa: E402

# --- paths -----------------------------------------------------------------
ARR_C = _REPO / "runs" / "phase4_features" / "arrays_openiC.npz"
ARR_D = _REPO / "runs" / "phase4_features" / "arrays_openiD.npz"
ARR_B = _REPO / "runs" / "phase4_features" / "arrays_roleB.npz"
CV2_C = _REPO / "runs" / "phase4_features" / "cv2_feats_openiC.npz"
CV2_D = _REPO / "runs" / "phase4_features" / "cv2_feats_openiD.npz"
CV2_B = _REPO / "runs" / "phase4_features" / "cv2_feats_roleB.npz"
SIDECAR = _REPO / "runs" / "augmented" / "conformal_sidecar_aug.json"
OUT_DIR = _REPO / os.environ.get("REPMCV2_OUT", "runs/repmismatch_cv2")
ARTIFACT = OUT_DIR / "repmismatch.npz"
REPORT = OUT_DIR / "repmismatch_cv2_report.json"

CONF_THRESH = 0.5
FFR = float(os.environ.get("REPMCV2_FFR", "0.20"))  # match production operating point
FEAT_DIM = int(os.environ.get("REPMCV2_PCA", "256"))  # PCA-reduced ConvNeXt-V2 dims
RAW_DIM = 1536             # ConvNeXt-V2-Large penultimate embedding (full, no pad)
MIN_POS = 20
MIN_NEG = 20
MIN_POS_FOLD = 10
MIN_NEG_FOLD = 10
MIN_ERR_TO_SHIP = 5
SHIP_AUROC_MIN = 0.60
SPLIT_SEED = 0


def _pbar_scaled(probs: np.ndarray, T: float) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    pbar = np.nanmean(p, axis=1) if p.ndim == 3 else p
    return apply_temperature(pbar, T)


def _fit(feats, gt, valid, min_pos, min_neg):
    return RepMismatch().fit(np.concatenate(feats), np.concatenate(gt),
                             np.concatenate(valid), list(NIH_PATHOLOGIES),
                             min_pos=min_pos, min_neg=min_neg, feat_dim=FEAT_DIM)


def _load_feats(split: str) -> np.ndarray:
    p = _REPO / "runs" / "phase4_features" / f"cv2_feats_{split}.npz"
    d = np.load(p, allow_pickle=True)
    f = np.asarray(d["cv2_feats"], dtype=np.float64)
    assert np.isfinite(f).all(), f"NaN in {p} (extraction incomplete?)"
    return f


def main() -> None:
    for p in (CV2_C, CV2_D, CV2_B):
        assert p.exists(), f"missing {p} -- run scripts/extract_cv2_features.py first"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIDECAR) as f:
        sc = json.load(f)
    T = float(sc["temperature_T"])
    youden = {p: sc.get("youden", {}).get(p) for p in NIH_PATHOLOGIES}

    c = np.load(ARR_C, allow_pickle=True)
    d = np.load(ARR_D, allow_pickle=True)
    b = np.load(ARR_B, allow_pickle=True)
    pc = _pbar_scaled(c["probs"], T)
    pd_ = _pbar_scaled(d["probs"], T)
    gt_c = np.asarray(c["gt"], dtype=np.float64)
    gt_d = np.asarray(d["gt"], dtype=np.float64)
    gt_b = np.asarray(b["gt"], dtype=np.float64)
    vc = np.asarray(c["valid"], dtype=np.float64)
    vd = np.asarray(d["valid"], dtype=np.float64)
    vb = np.asarray(b["valid"], dtype=np.float64)
    Fc = _load_feats("openiC")
    Fd = _load_feats("openiD")
    Fb = _load_feats("roleB")
    # alignment sanity: cv2_feats rows must match arrays rows (same id order)
    assert Fc.shape[0] == gt_c.shape[0] and Fd.shape[0] == gt_d.shape[0] and \
        Fb.shape[0] == gt_b.shape[0], "cv2_feats / arrays row mismatch"
    assert Fc.shape[1] == RAW_DIM, f"expected {RAW_DIM}-d raw cv2 feats, got {Fc.shape[1]}"
    # PCA-reduce ConvNeXt-V2's 1536-d embedding to FEAT_DIM components, fit leak-free on
    # OpenI-C (the calibration set) and applied to role-B / OpenI-D. A 1536-d Gaussian on
    # ~1-2k samples is p>>n-territory (heavy shrinkage); the top-256 PCA subspace is
    # well-conditioned (n>>p), fast to fit, and standard practice for DDU-style density on
    # high-d CNN features. The subspace is learned on C only -> leak-free (a fixed transform,
    # like temperature scaling). When FEAT_DIM == RAW_DIM, PCA is skipped (raw full-width).
    if FEAT_DIM < RAW_DIM:
        from sklearn.decomposition import PCA
        t0 = time.time()
        pca = PCA(n_components=FEAT_DIM, random_state=SPLIT_SEED).fit(Fc)
        Fc = pca.transform(Fc).astype(np.float64)
        Fd = pca.transform(Fd).astype(np.float64)
        Fb = pca.transform(Fb).astype(np.float64)
        var = float(pca.explained_variance_ratio_.sum())
        print(f"[cv2-repm] PCA {RAW_DIM}->{FEAT_DIM} on OpenI-C "
              f"({time.time()-t0:.1f}s, retained var={var:.3f})")
    std_d = np.nanstd(np.asarray(d["probs"], dtype=np.float64), axis=1)
    P = len(NIH_PATHOLOGIES)
    print(f"[cv2-repm] feat_dim={FEAT_DIM} FFR={FFR}  C={Fc.shape} D={Fd.shape} B={Fb.shape}")

    def dec_conf(p):
        dec = np.zeros_like(p, dtype=int)
        cc = np.zeros_like(p, dtype=np.float64)
        for j, pat in enumerate(NIH_PATHOLOGIES):
            t = youden[pat]
            if t is None:
                dec[:, j] = -1
                cc[:, j] = np.nan
                continue
            dj = (p[:, j] >= t).astype(int)
            dec[:, j] = dj
            cc[:, j] = np.where(dj == 1, p[:, j], 1.0 - p[:, j])
        return dec, cc

    dec_c, cc_c = dec_conf(pc)
    dec_d, cc_d = dec_conf(pd_)

    openiC_pos = np.array([int((gt_c[:, j] * vc[:, j]).sum()) for j in range(P)])
    openiC_neg = np.array([int(((vc[:, j] > 0) & (gt_c[:, j] == 0)).sum()) for j in range(P)])
    use_roleB = {pat: bool(openiC_pos[j] < MIN_POS or openiC_neg[j] < MIN_NEG)
                 for j, pat in enumerate(NIH_PATHOLOGIES)}
    print(f"[cv2-repm] role-B used for (OpenI-C too rare): "
          f"{[p for p, u in use_roleB.items() if u]}")

    rep_oc = _fit([Fc], [gt_c], [vc], MIN_POS, MIN_NEG)
    rep_rb = _fit([Fc, Fb], [gt_c, gt_b], [vc, vb], MIN_POS, MIN_NEG)

    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(Fc.shape[0])
    half = Fc.shape[0] // 2
    idxA, idxB = idx[:half], idx[half:]
    rep_oc_A = _fit([Fc[idxA]], [gt_c[idxA]], [vc[idxA]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_oc_B = _fit([Fc[idxB]], [gt_c[idxB]], [vc[idxB]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_rb_A = _fit([Fc[idxA], Fb], [gt_c[idxA], gt_b], [vc[idxA], vb], MIN_POS, MIN_NEG)
    rep_rb_B = _fit([Fc[idxB], Fb], [gt_c[idxB], gt_b], [vc[idxB], vb], MIN_POS, MIN_NEG)

    def _fold_thr(score_rep, score_idx, dec_idx, cc_idx):
        out = {}
        for j, pat in enumerate(NIH_PATHOLOGIES):
            if not score_rep.enabled.get(pat, False):
                out[pat] = None
                continue
            mm = score_rep.mismatch(Fc[score_idx], dec_idx[:, j], pat)
            conf = cc_idx[:, j] >= CONF_THRESH
            mm_c = mm[conf & np.isfinite(mm)]
            out[pat] = (float(np.quantile(mm_c, 1.0 - FFR)) if mm_c.size >= 20 else None)
        return out

    thr_oc_A = _fold_thr(rep_oc_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_oc_B = _fold_thr(rep_oc_B, idxA, dec_c[idxA], cc_c[idxA])
    thr_rb_A = _fold_thr(rep_rb_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_rb_B = _fold_thr(rep_rb_B, idxA, dec_c[idxA], cc_c[idxA])

    rep_final = RepMismatch()
    rep_final.pathologies = list(NIH_PATHOLOGIES)
    rep_final.enabled = {p: False for p in NIH_PATHOLOGIES}
    rep_final.flag_threshold = {p: np.inf for p in NIH_PATHOLOGIES}
    rep_final._feat_dim = FEAT_DIM
    rep_final.meta = {"min_pos": MIN_POS, "min_neg": MIN_NEG, "conf_thresh": CONF_THRESH,
                      "ffr": FFR, "split_seed": SPLIT_SEED, "feat_dim": FEAT_DIM,
                      "feature_space": "convnextv2"}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        src = rep_rb if use_roleB[pat] else rep_oc
        if not src.enabled[pat]:
            continue
        rep_final.enabled[pat] = True
        rep_final.mu_pos[pat] = src.mu_pos[pat]
        rep_final.mu_neg[pat] = src.mu_neg[pat]
        rep_final.inv_cov_pos[pat] = src.inv_cov_pos[pat]
        rep_final.inv_cov_neg[pat] = src.inv_cov_neg[pat]
        rep_final.logdet_pos[pat] = src.logdet_pos[pat]
        rep_final.logdet_neg[pat] = src.logdet_neg[pat]
        rep_final.n_pos[pat] = src.n_pos[pat]
        rep_final.n_neg[pat] = src.n_neg[pat]
        ta = (thr_rb_A if use_roleB[pat] else thr_oc_A)[pat]
        tb = (thr_rb_B if use_roleB[pat] else thr_oc_B)[pat]
        if ta is not None and tb is not None:
            rep_final.flag_threshold[pat] = 0.5 * (ta + tb)
        elif ta is not None:
            rep_final.flag_threshold[pat] = ta
        elif tb is not None:
            rep_final.flag_threshold[pat] = tb
        else:
            rep_final.flag_threshold[pat] = np.inf
    fit_enabled = {p: rep_final.enabled[p] for p in NIH_PATHOLOGIES}
    print(f"[cv2-repm] fit-enabled: {[p for p, e in fit_enabled.items() if e]}")

    # also load the RAD-DINO shipped set for the complement comparison
    rd_rep = RepMismatch.load(_REPO / "runs" / "repmismatch" / "repmismatch.npz")
    rd_shipped = {p for p in NIH_PATHOLOGIES if rd_rep.enabled.get(p, False)}

    report = {"temperature_T": T, "conf_thresh": CONF_THRESH, "ffr": FFR,
              "feature_space": "convnextv2", "feat_dim": FEAT_DIM, "raw_dim": RAW_DIM,
              "pca": (FEAT_DIM < RAW_DIM), "split_seed": SPLIT_SEED,
              "n_openiC": int(Fc.shape[0]), "n_openiD": int(Fd.shape[0]),
              "n_roleB": int(Fb.shape[0]), "use_roleB": use_roleB,
              "raddino_shipped": sorted(rd_shipped), "per_pathology": {}, "shipped": []}
    shipped = []
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vj = vd[:, j] > 0
        info = {"fit_enabled": fit_enabled[pat], "use_roleB": use_roleB[pat],
                "n_pos_fit": rep_final.n_pos.get(pat, 0), "n_neg_fit": rep_final.n_neg.get(pat, 0),
                "flag_threshold": None, "shipped": False, "raddino_shipped": pat in rd_shipped,
                "auroc_mismatch_cv2": None, "auroc_confidence": None, "auroc_disagreement": None,
                "n_confident": 0, "n_confident_errors": 0, "flag_recall_at_ffr": None,
                "false_flag_rate": None}
        if not fit_enabled[pat]:
            report["per_pathology"][pat] = info
            continue
        thr = rep_final.flag_threshold[pat]
        info["flag_threshold"] = (None if not np.isfinite(thr) else float(thr))
        conf = (cc_d[:, j] >= CONF_THRESH) & vj
        decj = dec_d[conf, j]
        gtj = gt_d[conf, j].astype(int)
        err = (decj != gtj).astype(int)
        info["n_confident"] = int(conf.sum())
        info["n_confident_errors"] = int(err.sum())
        if err.sum() < MIN_ERR_TO_SHIP:
            report["per_pathology"][pat] = info
            rep_final.enabled[pat] = False
            continue
        mm = rep_final.mismatch(Fd[conf], decj, pat)
        conf_score = 1.0 - cc_d[conf, j]
        std_score = std_d[conf, j]
        try:
            a_mm = float(roc_auc_score(err, mm))
            a_cf = float(roc_auc_score(err, conf_score))
            a_st = float(roc_auc_score(err, std_score))
        except ValueError:
            a_mm = a_cf = a_st = float("nan")
        info["auroc_mismatch_cv2"] = a_mm
        info["auroc_confidence"] = a_cf
        info["auroc_disagreement"] = a_st
        if np.isfinite(thr):
            flagged = mm > thr
            info["flag_recall_at_ffr"] = float(err[flagged].sum() / max(err.sum(), 1))
            info["false_flag_rate"] = float((~err.astype(bool) & flagged).sum()
                                            / max((~err.astype(bool)).sum(), 1))
        ship = (np.isfinite(a_mm) and a_mm > a_cf and a_mm > SHIP_AUROC_MIN
                and err.sum() >= MIN_ERR_TO_SHIP)
        info["shipped"] = bool(ship)
        if ship:
            shipped.append(pat)
        else:
            rep_final.enabled[pat] = False
        report["per_pathology"][pat] = info

    report["shipped"] = shipped
    # write the report FIRST (the primary deliverable) so a save hiccup can't lose it
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    rep_final.save(ARTIFACT)

    print(f"\n[cv2-repm] measurement on OpenI-D (confident, calib_conf>={CONF_THRESH}):")
    print(f"{'pathology':22s} {'rb':3s} {'#err':>4s} {'a_cv2':>6s} {'a_cf':>6s} "
          f"{'a_std':>6s} {'rec':>6s} {'ffr':>6s} {'rd?':>4s} {'ship':>5s}")

    def _f(v):
        return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  -  "

    for pat in NIH_PATHOLOGIES:
        r = report["per_pathology"][pat]
        rb = "B" if r["use_roleB"] else "."
        rd = "Y" if r["raddino_shipped"] else "."
        ship = "YES" if r["shipped"] else "."
        print(f"{pat:22s} {rb:3s} {r['n_confident_errors']:4d} "
              f"{_f(r['auroc_mismatch_cv2']):>6s} {_f(r['auroc_confidence']):>6s} "
              f"{_f(r['auroc_disagreement']):>6s} {_f(r['flag_recall_at_ffr']):>6s} "
              f"{_f(r['false_flag_rate']):>6s} {rd:>4s} {ship:>5s}")
    print(f"\n[cv2-repm] SHIPPED ({len(shipped)}): {shipped}")
    print(f"[cv2-repm] RAD-DINO shipped: {sorted(rd_shipped)}")
    only_cv2 = sorted(set(shipped) - rd_shipped)
    only_rd = sorted(rd_shipped - set(shipped))
    both = sorted(set(shipped) & rd_shipped)
    print(f"[cv2-repm] complement -> both: {both}  only-ConvNeXt-V2: {only_cv2}  "
          f"only-RAD-DINO: {only_rd}")
    print(f"[cv2-repm] artifact -> {ARTIFACT}")
    print(f"[cv2-repm] report   -> {REPORT}")


if __name__ == "__main__":
    main()