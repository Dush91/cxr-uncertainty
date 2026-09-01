#!/usr/bin/env python
"""Validate the leak-free per-pathology fit-set selection criterion: does 2-fold CV on OpenI-C
(held-out half mismatch AUROC, rep_oc-fold vs rep_rb-fold) agree with the OpenI-D direction
(A OpenI-C-only vs B OpenI-C+role-B)? If yes, we can SELECT the fit set leak-free (pick role-B
where OpenI-C held-out CV says it helps) and expect OpenI-D to follow -- the refinement that
turns the current 'rare-only' heuristic into a data-driven per-pathology rule.

Reuses the fold Gaussians already fit in _analyze_roleB_pooled by re-fitting them here (cheap
relative to the deployed fits; only the fold Gaussians, MIN_POS_FOLD=10). No OpenI-D used.
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
SIDECAR = _REPO / "runs" / "augmented" / "conformal_sidecar_aug.json"
CONF = 0.5
MPF, MNF = 10, 10


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
    c = np.load(ARR_C, allow_pickle=True)
    b = np.load(_REPO / "runs" / "phase4_features" / "arrays_roleB.npz", allow_pickle=True)
    pc = _pbar(c["probs"], T)
    gt_c = np.asarray(c["gt"], float); gt_b = np.asarray(b["gt"], float)
    vc = np.asarray(c["valid"], float); vb = np.asarray(b["valid"], float)
    Fc = np.asarray(c["rad_feats"], float); Fb = np.asarray(b["rad_feats"], float)

    def dec_conf(p):
        dec = np.zeros_like(p, int); cc = np.zeros_like(p, float)
        for j, pat in enumerate(NIH_PATHOLOGIES):
            t = youden[pat]
            if t is None:
                dec[:, j] = -1; cc[:, j] = np.nan; continue
            dj = (p[:, j] >= t).astype(int); dec[:, j] = dj
            cc[:, j] = np.where(dj == 1, p[:, j], 1.0 - p[:, j])
        return dec, cc
    dec_c, cc_c = dec_conf(pc)

    rng = np.random.RandomState(0)
    idx = rng.permutation(Fc.shape[0]); half = Fc.shape[0] // 2
    iA, iB = idx[:half], idx[half:]
    rep_oc_A = fit([Fc[iA]], [gt_c[iA]], [vc[iA]], MPF, MNF)
    rep_oc_B = fit([Fc[iB]], [gt_c[iB]], [vc[iB]], MPF, MNF)
    rep_rb_A = fit([Fc[iA], Fb], [gt_c[iA], gt_b], [vc[iA], vb], 20, 20)
    rep_rb_B = fit([Fc[iB], Fb], [gt_c[iB], gt_b], [vc[iB], vb], 20, 20)

    def held_auroc(rep_score, score_idx, fit_half):
        # rep_score was fit on fit_half (iA or iB); score the OTHER half out-of-sample.
        out = {}
        for j, pat in enumerate(NIH_PATHOLOGIES):
            if not rep_score.enabled.get(pat, False):
                out[pat] = None; continue
            vj = vc[score_idx, j] > 0
            conf = (cc_c[score_idx, j] >= CONF) & vj
            decj = dec_c[score_idx][conf, j]; gtj = gt_c[score_idx][conf, j].astype(int)
            err = (decj != gtj).astype(int)
            if 0 < err.sum() < len(err):
                mm = rep_score.mismatch(Fc[score_idx][conf], decj, pat)
                out[pat] = float(roc_auc_score(err, mm))
            else:
                out[pat] = None
        return out

    # fold A Gaussian (fit on iA) scores iB; fold B Gaussian (fit on iB) scores iA
    aoc = held_auroc(rep_oc_A, iB, iA); boc = held_auroc(rep_oc_B, iA, iB)
    arb = held_auroc(rep_rb_A, iB, iA); brb = held_auroc(rep_rb_B, iA, iB)

    def avg(x, y):
        vals = [v for v in (x, y) if v is not None]
        return float(np.mean(vals)) if vals else None

    # OpenI-D direction from the pooled analysis (hardcode the measured A vs B a_mm for the verdict)
    D_A = {"Atelectasis": 0.652, "Cardiomegaly": 0.841, "Effusion": 0.716, "Infiltration": 0.664,
           "Nodule": 0.531, "Pneumonia": 0.812, "Pneumothorax": 0.832, "Edema": 0.873, "Hernia": 0.806}
    D_B = {"Atelectasis": 0.653, "Cardiomegaly": 0.865, "Effusion": 0.789, "Infiltration": 0.571,
           "Nodule": 0.522, "Pneumonia": 0.782, "Pneumothorax": 0.832, "Edema": 0.849, "Hernia": 0.845}

    print(f"{'pathology':20s} | {'OpenI-C held-out CV':^24s} | {'OpenI-D (eval)':^18s} | CV picks | agrees?")
    print(f"{'':20s} | {'oc-fit':>7s} {'rb-fit':>7s} {'dCV':>6s} | {'A':>6s} {'B':>6s} {'dD':>5s} | role-B? |")
    for pat in NIH_PATHOLOGIES:
        cv_oc = avg(aoc.get(pat), boc.get(pat))
        cv_rb = avg(arb.get(pat), brb.get(pat))
        if cv_oc is None or cv_rb is None:
            print(f"{pat:20s} | (not measurable on OpenI-C CV)"); continue
        d_cv = cv_rb - cv_oc
        if pat in D_A:
            d_d = D_B[pat] - D_A[pat]
            pick_rb = d_cv > 0
            d_pick = (d_d > 0) == pick_rb
            print(f"{pat:20s} | {cv_oc:7.3f} {cv_rb:7.3f} {d_cv:+6.3f} | "
                  f"{D_A[pat]:6.3f} {D_B[pat]:6.3f} {d_d:+5.3f} | "
                  f"{'roleB' if pick_rb else 'OpenIC':>6s} | {'YES' if d_pick else 'NO'}")
        else:
            print(f"{pat:20s} | {cv_oc:7.3f} {cv_rb:7.3f} {d_cv:+6.3f} | (rare / not in OpenI-D table)")


if __name__ == "__main__":
    main()