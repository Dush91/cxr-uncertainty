#!/usr/bin/env python
"""Direct test of the user's proposal: pool the in-distribution held-out set (role-B = NIH +
CheXpert training held-outs) into BOTH calibrations of the rep-mismatch flag -- the Gaussian FIT
and the threshold -- for ALL pathologies (not just the rare). Measure leak-free on OpenI-D.

Compares three configs (all eval on OpenI-D, which never touches fit/threshold):
  A. CURRENT  : fit on OpenI-C (non-rare) / OpenI-C+role-B (rare); threshold via 2-fold CV on OpenI-C.
  B. POOLED-FIT: fit on OpenI-C+role-B for ALL; threshold via 2-fold CV on OpenI-C (isolates the
                FIT effect of adding role-B -- keeps the threshold on the deployment-distribution
                exchangeable set, so any change is purely the fit).
  C. POOLED-BOTH: fit on OpenI-C+role-B for ALL; threshold via 2-fold CV on the POOLED OpenI-C+role-B
                (the full proposal -- threshold also on the training distribution). This breaks
                exchangeability with OpenI-D; we report the MEASURED false-flag rate on OpenI-D to
                show the guarantee drifts.

Run: env PYTHONPATH=/teamspace/studios/this_studio python scripts/_analyze_roleB_pooled.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from cxr_uncertainty.config import NIH_PATHOLOGIES
from cxr_uncertainty.calibration import apply_temperature
from cxr_uncertainty.repmismatch import RepMismatch

ARR_C = _REPO / "runs" / "phase4_features" / "arrays_openiC.npz"
ARR_D = _REPO / "runs" / "phase4_features" / "arrays_openiD.npz"
ARR_B = _REPO / "runs" / "phase4_features" / "arrays_roleB.npz"
SIDECAR = _REPO / "runs" / "augmented" / "conformal_sidecar_aug.json"
CONF_THRESH = 0.5
MIN_POS, MIN_NEG = 20, 20
MIN_POS_FOLD, MIN_NEG_FOLD = 10, 10
FFR = 0.10
SEED = 0


def _pbar(probs, T):
    p = np.asarray(probs, float)
    return apply_temperature(np.nanmean(p, axis=1) if p.ndim == 3 else p, T)


def fit(feats, gt, valid, mp, mn):
    return RepMismatch().fit(np.concatenate(feats), np.concatenate(gt),
                             np.concatenate(valid), list(NIH_PATHOLOGIES), min_pos=mp, min_neg=mn)


def main():
    sc = json.load(open(SIDECAR)); T = float(sc["temperature_T"])
    youden = {p: sc.get("youden", {}).get(p) for p in NIH_PATHOLOGIES}
    P = len(NIH_PATHOLOGIES)
    c, d, b = (np.load(ARR_C, allow_pickle=True), np.load(ARR_D, allow_pickle=True),
               np.load(ARR_B, allow_pickle=True))
    pc, pd_ = _pbar(c["probs"], T), _pbar(d["probs"], T)
    gt_c, gt_d, gt_b = (np.asarray(c["gt"], float), np.asarray(d["gt"], float), np.asarray(b["gt"], float))
    vc, vb = np.asarray(c["valid"], float), np.asarray(b["valid"], float)
    Fc, Fd, Fb = (np.asarray(c["rad_feats"], float), np.asarray(d["rad_feats"], float),
                  np.asarray(b["rad_feats"], float))

    def dec_conf(p):
        dec = np.zeros_like(p, int); cc = np.zeros_like(p, float)
        for j, pat in enumerate(NIH_PATHOLOGIES):
            t = youden[pat]
            if t is None:
                dec[:, j] = -1; cc[:, j] = np.nan; continue
            dj = (p[:, j] >= t).astype(int); dec[:, j] = dj
            cc[:, j] = np.where(dj == 1, p[:, j], 1.0 - p[:, j])
        return dec, cc
    dec_c, cc_c = dec_conf(pc); dec_d, cc_d = dec_conf(pd_)
    openiC_pos = np.array([int((gt_c[:, j] * vc[:, j]).sum()) for j in range(P)])
    openiC_neg = np.array([int(((vc[:, j] > 0) & (gt_c[:, j] == 0)).sum()) for j in range(P)])
    rare = {pat: bool(openiC_pos[j] < MIN_POS or openiC_neg[j] < MIN_NEG) for j, pat in enumerate(NIH_PATHOLOGIES)}

    # ---- fits -------------------------------------------------------------
    rep_oc = fit([Fc], [gt_c], [vc], MIN_POS, MIN_NEG)                       # OpenI-C only
    rep_rb = fit([Fc, Fb], [gt_c, gt_b], [vc, vb], MIN_POS, MIN_NEG)         # OpenI-C + role-B (ALL)
    rng = np.random.RandomState(SEED)
    idxC = rng.permutation(Fc.shape[0]); halfC = Fc.shape[0] // 2
    iA, iB = idxC[:halfC], idxC[halfC:]
    rep_oc_A = fit([Fc[iA]], [gt_c[iA]], [vc[iA]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_oc_B = fit([Fc[iB]], [gt_c[iB]], [vc[iB]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_rb_A = fit([Fc[iA], Fb], [gt_c[iA], gt_b], [vc[iA], vb], MIN_POS, MIN_NEG)
    rep_rb_B = fit([Fc[iB], Fb], [gt_c[iB], gt_b], [vc[iB], vb], MIN_POS, MIN_NEG)
    # pooled-set folds for config C (split the concatenated OpenI-C+role-B)
    Fp = np.concatenate([Fc, Fb]); gtp = np.concatenate([gt_c, gt_b]); vp = np.concatenate([vc, vb])
    idxP = rng.permutation(Fp.shape[0]); halfP = Fp.shape[0] // 2
    jA, jB = idxP[:halfP], idxP[halfP:]
    rep_pool_A = fit([Fp[jA]], [gtp[jA]], [vp[jA]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_pool_B = fit([Fp[jB]], [gtp[jB]], [vp[jB]], MIN_POS_FOLD, MIN_NEG_FOLD)

    def fold_thr(score_rep, score_idx, dec_idx, cc_idx, feats):
        out = {}
        for j, pat in enumerate(NIH_PATHOLOGIES):
            if not score_rep.enabled.get(pat, False):
                out[pat] = None; continue
            mm = score_rep.mismatch(feats[score_idx], dec_idx[:, j], pat)
            conf = cc_idx[:, j] >= CONF_THRESH
            mmc = mm[conf & np.isfinite(mm)]
            out[pat] = float(np.quantile(mmc, 1 - FFR)) if mmc.size >= 20 else None
        return out

    dec_p, cc_p = dec_conf(_pbar(np.concatenate([c["probs"], b["probs"]], axis=0)
                                 if c["probs"].ndim == 3 else np.concatenate([c["probs"], b["probs"]]), T))
    # thresholds
    thr_oc_A = fold_thr(rep_oc_A, iB, dec_c[iB], cc_c[iB], Fc)
    thr_oc_B = fold_thr(rep_oc_B, iA, dec_c[iA], cc_c[iA], Fc)
    thr_rb_A = fold_thr(rep_rb_A, iB, dec_c[iB], cc_c[iB], Fc)
    thr_rb_B = fold_thr(rep_rb_B, iA, dec_c[iA], cc_c[iA], Fc)
    thr_pool_A = fold_thr(rep_pool_A, jB, dec_p[jB], cc_p[jB], Fp)
    thr_pool_B = fold_thr(rep_pool_B, jA, dec_p[jA], cc_p[jA], Fp)

    def thr_of(a, b, use):
        return 0.5 * (a + b) if (a is not None and b is not None) else (a or b or np.inf)

    def _f(v): return f"{v:.3f}" if np.isfinite(v) else "  -  "

    print("=" * 96)
    print("Does pooling the in-distribution held-out set (role-B) into BOTH calibrations help")
    print("detect confidently-wrong predictions?  (leak-free eval on OpenI-D, FFR=0.10)")
    print("=" * 96)
    hdr = (f"{'pathology':20s} {'rare':>4s} | {'--A current--':^22s} | {'--B pooled-fit--':^22s} | "
           f"{'--C pooled-both--':^22s}")
    print(hdr)
    print(f"{'':20s} {'':>4s} | {'a_mm':>6s} {'a_cf':>6s} {'rec':>5s} {'ffr':>5s} | "
          f"{'a_mm':>6s} {'a_cf':>6s} {'rec':>5s} {'ffr':>5s} | "
          f"{'a_mm':>6s} {'a_cf':>6s} {'rec':>5s} {'ffr':>5s}")
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vj = (np.asarray(d["valid"], float)[:, j] > 0)
        conf = (cc_d[:, j] >= CONF_THRESH) & vj
        decj = dec_d[conf, j]; gtj = gt_d[conf, j].astype(int)
        err = (decj != gtj).astype(int)
        nerr = int(err.sum())
        # A: current (rep_oc for non-rare, rep_rb for rare); threshold oc folds (non-rare) / rb folds (rare)
        repA = rep_rb if rare[pat] else rep_oc
        thrA = thr_of((thr_rb_A[pat] if rare[pat] else thr_oc_A[pat]),
                      (thr_rb_B[pat] if rare[pat] else thr_oc_B[pat]), pat)
        # B: pooled fit (rep_rb) for all; threshold on OpenI-C (rb folds)
        repB = rep_rb
        thrB = thr_of(thr_rb_A[pat], thr_rb_B[pat], pat)
        # C: pooled fit + pooled threshold
        repC = rep_rb
        thrC = thr_of(thr_pool_A[pat], thr_pool_B[pat], pat)

        def meas(rep, thr):
            if not rep.enabled.get(pat, False) or nerr < 5:
                return ("-", "-", "-", "-")
            mm = rep.mismatch(Fd[conf], decj, pat)
            a_mm = roc_auc_score(err, mm) if 0 < nerr < len(err) else float("nan")
            a_cf = roc_auc_score(err, 1 - cc_d[conf, j]) if 0 < nerr < len(err) else float("nan")
            if np.isfinite(thr):
                flag = mm > thr
                rec = err[flag].sum() / max(nerr, 1)
                ffr = (~err.astype(bool) & flag).sum() / max((~err.astype(bool)).sum(), 1)
            else:
                rec, ffr = float("nan"), float("nan")
            return (_f(a_mm), _f(a_cf), (f"{rec:.2f}" if np.isfinite(rec) else "  -  "),
                    (f"{ffr:.2f}" if np.isfinite(ffr) else "  -  "))
        aA = meas(repA, thrA); aB = meas(repB, thrB); aC = meas(repC, thrC)
        rtag = "B" if rare[pat] else "."
        print(f"{pat:20s} {rtag:>4s} | {aA[0]:>6s} {aA[1]:>6s} {aA[2]:>5s} {aA[3]:>5s} | "
              f"{aB[0]:>6s} {aB[1]:>6s} {aB[2]:>5s} {aB[3]:>5s} | "
              f"{aC[0]:>6s} {aC[1]:>6s} {aC[2]:>5s} {aC[3]:>5s}")
    print()
    print("A = current production.  B = fit on OpenI-C+role-B for ALL, threshold on OpenI-C.")
    print("C = fit + threshold both on OpenI-C+role-B (the full proposal).")
    print("If B's a_mm < A's a_mm for the shipped, pooling the fit HURTS (contamination).")
    print("If C's ffr drifts far above 0.10, pooling the threshold breaks the OpenI-D false-flag")
    print("guarantee (exchangeability lost -- role-B is a different hospital than OpenI).")


if __name__ == "__main__":
    main()