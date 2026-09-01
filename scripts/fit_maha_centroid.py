#!/usr/bin/env python
"""Leak-free prototype of the Mahalanobis-distance-to-decision-centroid confident-error
score (the "cleaner complement" to the log-likelihood mismatch flag, plan §Q out-of-scope).

WHY: the shipped rep-mismatch score is ``ll(opposite) - ll(decision)`` -- a difference of
two Gaussian log-likelihoods whose ABSOLUTE scale is dominated by the ``logdet`` of each
cluster's covariance. The ordering works (it ranks confident-errors above confident-correct)
and the threshold calibrates the false-flag rate, but the score is hard to interpret and the
logdet dominance can mask the signal on some pathologies. The DDU-style alternative
(Mukhoti CVPR 2023) is a pure quadratic distance: for a CONFIDENT decision, the Mahalanobis
distance to the cluster the model *called* this class -- ``mu_pos``/``inv_cov_pos`` if
``dec==1`` else ``mu_neg``/``inv_cov_neg``. HIGH distance = "the representation is far from
what you confidently called" = confident-error suspicion. No logdet term; the absolute
magnitude is a distance (comparable across pathologies).

PROTOCOL: identical to ``fit_repmismatch.py`` (so the comparison is apples-to-apples on the
SAME fit data, SAME 2-fold CV threshold, SAME ship guard). Only the SCORE function differs:
``decision_distance`` instead of ``mismatch``. Per-pathology fit data is OpenI-C only when
it has enough positives/negatives, OpenI-C+role-B for the genuinely rare (Mass/Pneumothorax/
Fibrosis). Threshold = (1-ffr) quantile of distance among CONFIDENT predictions on the
held-out fold half (out-of-sample, 2-fold averaged). Ship iff distance AUROC beats the
model-confidence baseline AND > 0.6 AND >= 5 confident errors on leak-free OpenI-D.

GOAL: see whether the cleaner distance score lifts the WEAK / NOT-SHIPPED pathologies
(Effusion, Infiltration, Nodule, Hernia, Mass, Pleural_Thickening, Emphysema, Fibrosis)
above the confidence baseline -- the representation lever that more threshold data could
not move (those are representation-limited under the mismatch score; a different score on
the same features may or may not separate them better).

Run:
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/fit_maha_centroid.py
Outputs: runs/repmismatch_maha/{repmismatch.npz, repmismatch.meta.json,
        repmismatch_maha_report.json}  (a standalone distance-score artifact, separate from
        the production mismatch artifact; the demo does NOT load it -- this is a prototype).
"""
from __future__ import annotations

import json
import os
import sys
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
SIDECAR = _REPO / "runs" / "augmented" / "conformal_sidecar_aug.json"
OUT_DIR = _REPO / os.environ.get("MAHA_OUT", "runs/repmismatch_maha")
ARTIFACT = OUT_DIR / "repmismatch.npz"
REPORT = OUT_DIR / "repmismatch_maha_report.json"

CONF_THRESH = 0.5
FFR = float(os.environ.get("MAHA_FFR", "0.10"))  # same default as production mismatch
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
                             min_pos=min_pos, min_neg=min_neg)


def _score(rep: RepMismatch, feats, dec_idx, pat):
    """The prototype score: Mahalanobis distance to the decision-cluster centroid."""
    return rep.decision_distance(feats, dec_idx, pat)


def _fold_thr(score_rep, score_idx, dec_idx, cc_idx):
    """Threshold = (1-ffr) quantile of distance among confident on the held-out half."""
    out = {}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        if not score_rep.enabled.get(pat, False):
            out[pat] = None
            continue
        sc = _score(score_rep, Fc[score_idx], dec_idx[:, j], pat)
        conf = cc_idx[:, j] >= CONF_THRESH
        sc_c = sc[conf & np.isfinite(sc)]
        out[pat] = (float(np.quantile(sc_c, 1.0 - FFR)) if sc_c.size >= 20 else None)
    return out


def main() -> None:
    assert ARR_B.exists(), f"missing {ARR_B} -- run scripts/run_roleB_inference.py first"
    assert "rad_feats" in np.load(ARR_B, allow_pickle=True), \
        "arrays_roleB.npz has no rad_feats -- re-run run_roleB_inference.py"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIDECAR) as f:
        sc = json.load(f)
    T = float(sc["temperature_T"])
    youden = {p: sc.get("youden", {}).get(p) for p in NIH_PATHOLOGIES}

    global Fc  # the fold-threshold closure uses Fc (the OpenI-C features)
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
    Fc = np.asarray(c["rad_feats"], dtype=np.float64)
    Fd = np.asarray(d["rad_feats"], dtype=np.float64)
    Fb = np.asarray(b["rad_feats"], dtype=np.float64)
    std_d = np.nanstd(np.asarray(d["probs"], dtype=np.float64), axis=1)
    P = len(NIH_PATHOLOGIES)
    print(f"[maha-centroid] T={T:.4f}  C={Fc.shape[0]} D={Fd.shape[0]} B={Fb.shape[0]}  FFR={FFR}")

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
    print(f"[maha-centroid] role-B used for (OpenI-C too rare): "
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

    thr_oc_A = _fold_thr(rep_oc_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_oc_B = _fold_thr(rep_oc_B, idxA, dec_c[idxA], cc_c[idxA])
    thr_rb_A = _fold_thr(rep_rb_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_rb_B = _fold_thr(rep_rb_B, idxA, dec_c[idxA], cc_c[idxA])

    # also score with the log-likelihood mismatch (the shipped score) for direct comparison
    def _fold_thr_mm(score_rep, score_idx, dec_idx, cc_idx):
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

    mm_oc_A = _fold_thr_mm(rep_oc_A, idxB, dec_c[idxB], cc_c[idxB])
    mm_oc_B = _fold_thr_mm(rep_oc_B, idxA, dec_c[idxA], cc_c[idxA])
    mm_rb_A = _fold_thr_mm(rep_rb_A, idxB, dec_c[idxB], cc_c[idxB])
    mm_rb_B = _fold_thr_mm(rep_rb_B, idxA, dec_c[idxA], cc_c[idxA])

    rep_final = RepMismatch()
    rep_final.pathologies = list(NIH_PATHOLOGIES)
    rep_final.enabled = {p: False for p in NIH_PATHOLOGIES}
    rep_final.flag_threshold = {p: np.inf for p in NIH_PATHOLOGIES}
    rep_final.meta = {"min_pos": MIN_POS, "min_neg": MIN_NEG, "conf_thresh": CONF_THRESH,
                      "ffr": FFR, "split_seed": SPLIT_SEED, "score": "decision_distance"}
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
    print(f"[maha-centroid] fit-enabled: {[p for p, e in fit_enabled.items() if e]}")

    report = {"temperature_T": T, "conf_thresh": CONF_THRESH, "ffr": FFR,
              "score": "decision_distance", "min_pos": MIN_POS, "min_neg": MIN_NEG,
              "split_seed": SPLIT_SEED, "n_openiC": int(Fc.shape[0]),
              "n_openiD": int(Fd.shape[0]), "n_roleB": int(Fb.shape[0]),
              "use_roleB": use_roleB, "per_pathology": {}, "shipped": []}
    shipped = []
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vj = vd[:, j] > 0
        info = {"fit_enabled": fit_enabled[pat], "use_roleB": use_roleB[pat],
                "n_pos_fit": rep_final.n_pos.get(pat, 0), "n_neg_fit": rep_final.n_neg.get(pat, 0),
                "flag_threshold": None, "shipped": False, "auroc_distance": None,
                "auroc_mismatch": None, "auroc_confidence": None, "auroc_disagreement": None,
                "n_confident": 0, "n_confident_errors": 0, "flag_recall_at_ffr": None,
                "false_flag_rate": None, "mm_recall_at_ffr": None, "mm_false_flag_rate": None}
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
        dist = _score(rep_final, Fd[conf], decj, pat)
        mm = rep_final.mismatch(Fd[conf], decj, pat)
        conf_score = 1.0 - cc_d[conf, j]
        std_score = std_d[conf, j]
        try:
            a_dist = float(roc_auc_score(err, dist))
            a_mm = float(roc_auc_score(err, mm))
            a_cf = float(roc_auc_score(err, conf_score))
            a_st = float(roc_auc_score(err, std_score))
        except ValueError:
            a_dist = a_mm = a_cf = a_st = float("nan")
        info["auroc_distance"] = a_dist
        info["auroc_mismatch"] = a_mm
        info["auroc_confidence"] = a_cf
        info["auroc_disagreement"] = a_st
        if np.isfinite(thr):
            flagged = dist > thr
            info["flag_recall_at_ffr"] = float(err[flagged].sum() / max(err.sum(), 1))
            info["false_flag_rate"] = float((~err.astype(bool) & flagged).sum()
                                            / max((~err.astype(bool)).sum(), 1))
            # mismatch at its own threshold (for a side-by-side rec/ffr read)
            src_rep = rep_rb if use_roleB[pat] else rep_oc
            ta = (mm_rb_A if use_roleB[pat] else mm_oc_A)[pat]
            tb = (mm_rb_B if use_roleB[pat] else mm_oc_B)[pat]
            mmt = (0.5 * (ta + tb) if (ta is not None and tb is not None)
                   else (ta or tb or float("inf")))
            if np.isfinite(mmt):
                mmf = mm > mmt
                info["mm_recall_at_ffr"] = float(err[mmf].sum() / max(err.sum(), 1))
                info["mm_false_flag_rate"] = float((~err.astype(bool) & mmf).sum()
                                                   / max((~err.astype(bool)).sum(), 1))
        # ship on the DISTANCE score (the prototype), same guard as mismatch
        ship = (np.isfinite(a_dist) and a_dist > a_cf and a_dist > SHIP_AUROC_MIN
                and err.sum() >= MIN_ERR_TO_SHIP)
        info["shipped"] = bool(ship)
        if ship:
            shipped.append(pat)
        else:
            rep_final.enabled[pat] = False
        report["per_pathology"][pat] = info

    report["shipped"] = shipped
    rep_final.save(ARTIFACT)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[maha-centroid] measurement on OpenI-D (confident, calib_conf>={CONF_THRESH}):")
    print(f"{'pathology':22s} {'rb':3s} {'#err':>4s} {'a_dist':>7s} {'a_mm':>6s} "
          f"{'a_cf':>6s} {'a_std':>6s} {'rec':>6s} {'ffr':>6s} {'mmrec':>6s} {'ship':>5s}")

    def _f(v):
        return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  -  "

    for pat in NIH_PATHOLOGIES:
        r = report["per_pathology"][pat]
        rb = "B" if r["use_roleB"] else "."
        ship = "YES" if r["shipped"] else "."
        print(f"{pat:22s} {rb:3s} {r['n_confident_errors']:4d} "
              f"{_f(r['auroc_distance']):>7s} {_f(r['auroc_mismatch']):>6s} "
              f"{_f(r['auroc_confidence']):>6s} {_f(r['auroc_disagreement']):>6s} "
              f"{_f(r['flag_recall_at_ffr']):>6s} {_f(r['false_flag_rate']):>6s} "
              f"{_f(r['mm_recall_at_ffr']):>6s} {ship:>5s}")
    print(f"\n[maha-centroid] SHIPPED ({len(shipped)}): {shipped}")
    # highlight where distance beats mismatch AND beats confidence (the prototype's win)
    print("[maha-centroid] distance beats BOTH mismatch AND confidence on:")
    for pat in NIH_PATHOLOGIES:
        r = report["per_pathology"][pat]
        if (r["auroc_distance"] and r["auroc_mismatch"] and r["auroc_confidence"]
                and np.isfinite(r["auroc_distance"])
                and r["auroc_distance"] > r["auroc_mismatch"]
                and r["auroc_distance"] > r["auroc_confidence"]):
            print(f"    {pat:22s} a_dist={r['auroc_distance']:.3f} "
                  f"> a_mm={r['auroc_mismatch']:.3f} > a_cf={r['auroc_confidence']:.3f}")
    print(f"[maha-centroid] artifact -> {ARTIFACT}")
    print(f"[maha-centroid] report   -> {REPORT}")


if __name__ == "__main__":
    main()