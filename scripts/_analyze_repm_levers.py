#!/usr/bin/env python
"""Lightweight diagnostic (no fold refit, no OOM): for the NOT-shipped, OpenI-C-fit pathologies,
does the representation-mismatch SIGNAL exist? Two questions the user's "add more data" idea
turns on:

(A) role-B diagnostic: the deployed Gaussian was fit on OpenI-C only, so role-B (NIH+CheXpert,
    ~1000+ positives for common pathologies) is OUT-OF-SAMPLE with far more confident errors to
    measure. If mismatch AUROC > confidence AUROC > 0.6 on role-B, the signal EXISTS -- a bigger
    leak-free eval set could measure/ship it (the user's instinct, but it needs EVAL data not
    threshold-cal data). If ~chance / below baseline even with thousands of points, the RAD-DINO
    representation genuinely doesn't separate the classes -> no data lever helps.

(B) FFR illustration (in-sample re-quantile on OpenI-C, clearly optimistic): shows raising FFR
    trades false-flag for recall on the 5 SHIPPED pathologies -- the lever that actually moves
    recall, vs threshold-data-size which doesn't.
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
FFRS = [0.10, 0.15, 0.20, 0.30]


def _pbar(probs, T):
    p = np.asarray(probs, dtype=np.float64)
    return apply_temperature(np.nanmean(p, axis=1) if p.ndim == 3 else p, T)


def main():
    sc = json.load(open(SIDECAR)); T = float(sc["temperature_T"])
    youden = {p: sc.get("youden", {}).get(p) for p in NIH_PATHOLOGIES}
    P = len(NIH_PATHOLOGIES)
    c, d, b = (np.load(ARR_C, allow_pickle=True), np.load(ARR_D, allow_pickle=True),
               np.load(ARR_B, allow_pickle=True))
    pc, pd_, pb_ = _pbar(c["probs"], T), _pbar(d["probs"], T), _pbar(b["probs"], T)
    gt_c = np.asarray(c["gt"], float); gt_d = np.asarray(d["gt"], float); gt_b = np.asarray(b["gt"], float)
    vc = np.asarray(c["valid"], float); vb = np.asarray(b["valid"], float)
    Fc = np.asarray(c["rad_feats"], float); Fd = np.asarray(d["rad_feats"], float); Fb = np.asarray(b["rad_feats"], float)

    def dec_conf(p):
        dec = np.zeros_like(p, int); cc = np.zeros_like(p, float)
        for j, pat in enumerate(NIH_PATHOLOGIES):
            t = youden[pat]
            if t is None:
                dec[:, j] = -1; cc[:, j] = np.nan; continue
            dj = (p[:, j] >= t).astype(int); dec[:, j] = dj
            cc[:, j] = np.where(dj == 1, p[:, j], 1.0 - p[:, j])
        return dec, cc
    dec_c, cc_c = dec_conf(pc); dec_d, cc_d = dec_conf(pd_); dec_b, cc_b = dec_conf(pb_)
    openiC_pos = np.array([int((gt_c[:, j] * vc[:, j]).sum()) for j in range(P)])
    use_roleB = {pat: bool(openiC_pos[j] < MIN_POS or (np.sum((vc[:, j] > 0) & (gt_c[:, j] == 0)) < MIN_NEG))
                 for j, pat in enumerate(NIH_PATHOLOGIES)}
    rep_oc = RepMismatch().fit(Fc, gt_c, vc, list(NIH_PATHOLOGIES), MIN_POS, MIN_NEG)  # deployed (non-rare)
    rep_rb = RepMismatch().fit(np.concatenate([Fc, Fb]), np.concatenate([gt_c, gt_b]),
                               np.concatenate([vc, vb]), list(NIH_PATHOLOGIES), MIN_POS, MIN_NEG)

    def _f(v): return f"{v:.3f}" if np.isfinite(v) else "  -  "

    print("=" * 78)
    print("(A) role-B diagnostic: does the mismatch SIGNAL exist for the NOT-shipped?")
    print("    role-B is out-of-sample for the OpenI-C-fit Gaussian; far more confident errors.")
    print("=" * 78)
    print(f"  {'pathology':22s} | {'OpenI-D':^20s} | {'role-B (more data)':^24s} | verdict")
    print(f"  {'':22s} | {'#err':>5s} {'a_mm':>6s} {'a_cf':>6s} | {'#err':>6s} {'a_mm':>7s} {'a_cf':>7s} |")
    for j, pat in enumerate(NIH_PATHOLOGIES):
        rep = rep_rb if use_roleB[pat] else rep_oc
        if not rep.enabled.get(pat, False):
            print(f"  {pat:22s} | (not fit-enabled)"); continue
        # OpenI-D
        vj = (np.asarray(d["valid"], float)[:, j] > 0); cD = (cc_d[:, j] >= CONF_THRESH) & vj
        eD = (dec_d[cD, j] != gt_d[cD, j].astype(int)).astype(int)
        mmD = rep.mismatch(Fd[cD], dec_d[cD, j], pat)
        aD = roc_auc_score(eD, mmD) if 0 < eD.sum() < len(eD) else float("nan")
        aDcf = roc_auc_score(eD, 1 - cc_d[cD, j]) if 0 < eD.sum() < len(eD) else float("nan")
        # role-B (out-of-sample for non-rare; in-sample-ish for rare but still informative count)
        vjb = (vb[:, j] > 0); cB = (cc_b[:, j] >= CONF_THRESH) & vjb
        eB = (dec_b[cB, j] != gt_b[cB, j].astype(int)).astype(int)
        mmB = rep.mismatch(Fb[cB], dec_b[cB, j], pat)
        aB = roc_auc_score(eB, mmB) if 0 < eB.sum() < len(eB) else float("nan")
        aBcf = roc_auc_score(eB, 1 - cc_b[cB, j]) if 0 < eB.sum() < len(eB) else float("nan")
        sig = np.isfinite(aB) and aB > aBcf and aB > 0.6
        verdict = "SIGNAL EXISTS (more eval data could measure/ship)" if sig else "weak/representation-limited"
        tag = " [rare, role-B-fit]" if use_roleB[pat] else ""
        print(f"  {pat:22s} | {int(eD.sum()):5d} {_f(aD):>6s} {_f(aDcf):>6s} | "
              f"{int(eB.sum()):6d} {_f(aB):>7s} {_f(aBcf):>7s} | {verdict}{tag}")

    print("\n" + "=" * 78)
    print("(B) FFR illustration on the 5 SHIPPED (in-sample re-quantile on OpenI-C; OPTIMISTIC")
    print("    but shows the lever). More threshold DATA would not change this -- FFR does.")
    print("=" * 78)
    shipped = ["Atelectasis", "Cardiomegaly", "Pneumonia", "Pneumothorax", "Edema"]
    print(f"  {'pathology':22s} " + " ".join(("rec@"+str(int(f*100))+"%").rjust(8) for f in FFRS)
          + "  " + " ".join(("ffr@"+str(int(f*100))+"%").rjust(8) for f in FFRS))
    for pat in shipped:
        j = NIH_PATHOLOGIES.index(pat); rep = rep_rb if use_roleB[pat] else rep_oc
        if not rep.enabled.get(pat, False):
            print(f"  {pat:22s} (not fit-enabled)"); continue
        # in-sample OpenI-C confident mismatch for the threshold (optimistic proxy)
        vjc = (vc[:, j] > 0); cC = (cc_c[:, j] >= CONF_THRESH) & vjc
        mmC = rep.mismatch(Fc[cC], dec_c[cC, j], pat)
        # OpenI-D eval
        vj = (np.asarray(d["valid"], float)[:, j] > 0); cD = (cc_d[:, j] >= CONF_THRESH) & vj
        eD = (dec_d[cD, j] != gt_d[cD, j].astype(int)).astype(int)
        mmD = rep.mismatch(Fd[cD], dec_d[cD, j], pat)
        recs, ffrs = [], []
        for ffr in FFRS:
            thr = np.quantile(mmC, 1 - ffr) if mmC.size >= 20 else np.inf
            flag = mmD > thr
            recs.append(eD[flag].sum() / max(eD.sum(), 1))
            ffrs.append((~eD.astype(bool) & flag).sum() / max((~eD.astype(bool)).sum(), 1))
        print(f"  {pat:22s} " + " ".join(f"{r:8.2f}" for r in recs) + "  " + " ".join(f"{f:8.2f}" for f in ffrs))
    print("  >> Raising FFR (lower threshold) catches more confident errors at more false-flags.")
    print("     This is a knob on the OPERATING POINT, not a data-size fix; more calibration")
    print("     data would give ~the same thresholds (already ~hundreds of confident points).")


if __name__ == "__main__":
    main()