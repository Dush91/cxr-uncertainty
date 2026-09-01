#!/usr/bin/env python
"""Fit + leak-free evaluation of the class-conditional-density (representation-mismatch)
confident-error flag (plan §P; module ``cxr_uncertainty.repmismatch.py``).

This is the 4th flag -- the ONE signal that targets confident in-distribution FP/FN,
where the three current flags are blind (disagreement at chance, Mahalanobis models
foreign-ness, conformal auto-reads confident single labels).

Method: per pathology, fit two Ledoit-Wolf Gaussians (G_pos / G_neg) on the frozen
RAD-DINO [CLS] features (first 768 cols of the 1536-d zero-padded layout). For a
confident model decision, the mismatch score = log-likelihood under the OPPOSITE
cluster minus under the decision cluster (high = the representation favours the
opposite of the confident call -> confident-error suspicion).

Per-pathology fit data (avoids training-distribution contamination):
  * Genuinely RARE on OpenI-C (< MIN_POS positives OR < MIN_NEG negatives over the
    FULL OpenI-C set -- Mass 4, Pneumothorax 10, Fibrosis 7): fit on OpenI-C + role-B
    (role-B supplies the positive count a 768-d Gaussian needs; OpenI-C alone cannot
    fit). This is where role-B earns its keep (Pneumothorax: OpenI-C-only was
    unmeasurable; OpenI-C+role-B -> AUROC ~0.92).
  * Otherwise (enough OpenI-C positives): fit on OpenI-C ONLY -- the pure in-distribution
    representation. Adding role-B here would contaminate the Gaussian with training-
    distribution representation and WEAKEN the signal (measured: Pneumonia 0.81 OpenI-C
    only vs 0.70 with role-B pooled; Edema 0.87 vs 0.82). So role-B is per-pathology,
    only for the rare.

Leak-free threshold (2-fold CV on OpenI-C, no half-split data loss):
  * Deployed Gaussian: fit on the FULL chosen data (OpenI-C, or OpenI-C+role-B if rare).
  * Threshold: 2-fold CV -- fit a fold Gaussian on OpenI-C-half-A (+role-B if rare),
    score the HELD-OUT half-B (out-of-sample for that fold's Gaussian), threshold_A =
    (1-ffr) quantile of mismatch among CONFIDENT half-B predictions; symmetric fold
    -> average. The deployed Gaussian never sees OpenI-D; OpenI-D only MEASURES.
  * Fold fits use a lower MIN_POS_FOLD (they only estimate the threshold quantile, not
    the deployed Gaussian, so a noisier fold Gaussian is acceptable -- this keeps
    borderline pathologies like Pneumonia/Edema fit-enabled in each half).

Shipping (honest Pareto guard): a pathology ships ONLY if, on OpenI-D, the mismatch
AUROC (vs confident-error) BEATS the model-confidence baseline AND is > 0.6, with
>= 5 confident errors. Weak pathologies are NOT shipped; the artifact stores winners.

Run (after scripts/run_roleB_inference.py produced arrays_roleB.npz WITH rad_feats):
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/fit_repmismatch.py
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
# All env-overridable so the same script re-fits on a different ensemble's
# confident set (e.g. the 4-member Ark+ ensemble) by pointing at that ensemble's
# arrays + sidecar. The FEATURES (rad_feats inside the arrays) are
# ensemble-independent; only the probs (confident set) + T/youden (sidecar) change.
ARR_C = Path(os.environ.get("REPM_ARR_C",
          "runs/phase4_features/arrays_openiC.npz"))
ARR_D = Path(os.environ.get("REPM_ARR_D",
          "runs/phase4_features/arrays_openiD.npz"))
ARR_B = Path(os.environ.get("REPM_ARR_B",
          "runs/phase4_features/arrays_roleB.npz"))
SIDECAR = Path(os.environ.get("REPM_SIDECAR",
             "runs/augmented/conformal_sidecar_aug.json"))
if not ARR_C.is_absolute():
    ARR_C = _REPO / ARR_C; ARR_D = _REPO / ARR_D; ARR_B = _REPO / ARR_B
    SIDECAR = _REPO / SIDECAR
OUT_DIR = _REPO / os.environ.get("REPM_OUT", "runs/repmismatch")
ARTIFACT = OUT_DIR / "repmismatch.npz"
REPORT = OUT_DIR / "repmismatch_report.json"

CONF_THRESH = 0.5     # confident gate (model leans to its decision)
FFR = float(os.environ.get("REPM_FFR", "0.10"))   # target false-flag rate -> (1-ffr) quantile
MIN_POS = 20          # min positives to fit the DEPLOYED 768-d Gaussian
MIN_NEG = 20
MIN_POS_FOLD = 10     # min positives for the FOLD (threshold-only) Gaussian -- lower
MIN_NEG_FOLD = 10     # because a fold only estimates a score quantile, not the deploy
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


def main() -> None:
    assert ARR_B.exists(), f"missing {ARR_B} -- run scripts/run_roleB_inference.py first"
    assert "rad_feats" in np.load(ARR_B, allow_pickle=True), \
        "arrays_roleB.npz has no rad_feats -- re-run run_roleB_inference.py"
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
    Fc = np.asarray(c["rad_feats"], dtype=np.float64)
    Fd = np.asarray(d["rad_feats"], dtype=np.float64)
    Fb = np.asarray(b["rad_feats"], dtype=np.float64)
    std_d = np.nanstd(np.asarray(d["probs"], dtype=np.float64), axis=1)
    P = len(NIH_PATHOLOGIES)
    print(f"[repm] T={T:.4f}  C={Fc.shape[0]} D={Fd.shape[0]} B={Fb.shape[0]}")

    # --- decisions / confidence on the shared temp-scaled scale ----------------
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

    # --- per-pathology: is OpenI-C (full) too rare to fit alone? ---------------
    openiC_pos = np.array([int((gt_c[:, j] * vc[:, j]).sum()) for j in range(P)])
    openiC_neg = np.array([int(((vc[:, j] > 0) & (gt_c[:, j] == 0)).sum()) for j in range(P)])
    use_roleB = {pat: bool(openiC_pos[j] < MIN_POS or openiC_neg[j] < MIN_NEG)
                 for j, pat in enumerate(NIH_PATHOLOGIES)}
    print(f"[repm] role-B used for (OpenI-C too rare): "
          f"{[p for p,u in use_roleB.items() if u]}")

    # --- deployed Gaussians (full chosen data) --------------------------------
    rep_oc = _fit([Fc], [gt_c], [vc], MIN_POS, MIN_NEG)             # OpenI-C only
    rep_rb = _fit([Fc, Fb], [gt_c, gt_b], [vc, vb], MIN_POS, MIN_NEG)  # OpenI-C + role-B

    # --- 2-fold CV threshold (out-of-sample) ----------------------------------
    rng = np.random.RandomState(SPLIT_SEED)
    idx = rng.permutation(Fc.shape[0])
    half = Fc.shape[0] // 2
    idxA, idxB = idx[:half], idx[half:]
    # fold Gaussians: A fit on half-A, B fit on half-B (role-B folded in for rare only,
    # but a single fit covers all pathologies; we pick the right rep per pathology).
    rep_oc_A = _fit([Fc[idxA]], [gt_c[idxA]], [vc[idxA]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_oc_B = _fit([Fc[idxB]], [gt_c[idxB]], [vc[idxB]], MIN_POS_FOLD, MIN_NEG_FOLD)
    rep_rb_A = _fit([Fc[idxA], Fb], [gt_c[idxA], gt_b], [vc[idxA], vb], MIN_POS, MIN_NEG)
    rep_rb_B = _fit([Fc[idxB], Fb], [gt_c[idxB], gt_b], [vc[idxB], vb], MIN_POS, MIN_NEG)

    def _fold_thr(score_rep, score_idx, dec_idx, cc_idx):
        """Threshold = (1-ffr) quantile of mismatch among confident on the held-out half."""
        mismatch = {}
        for j, pat in enumerate(NIH_PATHOLOGIES):
            if not score_rep.enabled.get(pat, False):
                mismatch[pat] = None
                continue
            mm = score_rep.mismatch(Fc[score_idx], dec_idx[:, j], pat)
            conf = cc_idx[:, j] >= CONF_THRESH
            mm_c = mm[conf & np.isfinite(mm)]
            mismatch[pat] = (float(np.quantile(mm_c, 1.0 - FFR))
                              if mm_c.size >= 20 else None)
        return mismatch

    # fold A Gaussian (fit on half-A) scores half-B; fold B Gaussian scores half-A
    thr_oc_A = _fold_thr(rep_oc_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_oc_B = _fold_thr(rep_oc_B, idxA, dec_c[idxA], cc_c[idxA])
    thr_rb_A = _fold_thr(rep_rb_A, idxB, dec_c[idxB], cc_c[idxB])
    thr_rb_B = _fold_thr(rep_rb_B, idxA, dec_c[idxA], cc_c[idxA])

    # --- assemble the deployed artifact ---------------------------------------
    rep_final = RepMismatch()
    rep_final.pathologies = list(NIH_PATHOLOGIES)
    rep_final.enabled = {p: False for p in NIH_PATHOLOGIES}
    rep_final.flag_threshold = {p: np.inf for p in NIH_PATHOLOGIES}
    rep_final.meta = {"min_pos": MIN_POS, "min_neg": MIN_NEG, "conf_thresh": CONF_THRESH,
                      "ffr": FFR, "split_seed": SPLIT_SEED}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        src = rep_rb if use_roleB[pat] else rep_oc
        if not src.enabled[pat]:
            continue    # even with role-B too rare (shouldn't happen) -> disabled
        rep_final.enabled[pat] = True
        rep_final.mu_pos[pat] = src.mu_pos[pat]
        rep_final.mu_neg[pat] = src.mu_neg[pat]
        rep_final.inv_cov_pos[pat] = src.inv_cov_pos[pat]
        rep_final.inv_cov_neg[pat] = src.inv_cov_neg[pat]
        rep_final.logdet_pos[pat] = src.logdet_pos[pat]
        rep_final.logdet_neg[pat] = src.logdet_neg[pat]
        rep_final.n_pos[pat] = src.n_pos[pat]
        rep_final.n_neg[pat] = src.n_neg[pat]
        # average the two folds' out-of-sample thresholds
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
    print(f"[repm] fit-enabled: {[p for p,e in fit_enabled.items() if e]}")

    # --- measure on OpenI-D (leak-free eval) ----------------------------------
    report = {"temperature_T": T, "conf_thresh": CONF_THRESH, "ffr": FFR,
              "min_pos": MIN_POS, "min_neg": MIN_NEG, "split_seed": SPLIT_SEED,
              "n_openiC": int(Fc.shape[0]), "n_openiD": int(Fd.shape[0]),
              "n_roleB": int(Fb.shape[0]), "use_roleB": use_roleB,
              "per_pathology": {}, "shipped": []}
    shipped = []
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vj = vd[:, j] > 0
        info = {"fit_enabled": fit_enabled[pat], "use_roleB": use_roleB[pat],
                "n_pos_fit": rep_final.n_pos.get(pat, 0),
                "n_neg_fit": rep_final.n_neg.get(pat, 0), "flag_threshold": None,
                "shipped": False, "auroc_mismatch": None, "auroc_confidence": None,
                "auroc_disagreement": None, "n_confident": 0, "n_confident_errors": 0,
                "flag_recall_at_ffr": None, "false_flag_rate": None}
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
            rep_final.enabled[pat] = False    # too few errors to trust -> don't ship
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
        info["auroc_mismatch"] = a_mm
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
            rep_final.enabled[pat] = False    # honest: don't ship weak pathologies
        report["per_pathology"][pat] = info

    report["shipped"] = shipped
    rep_final.save(ARTIFACT)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    # --- report -------------------------------------------------------------
    print(f"\n[repm] measurement on OpenI-D (confident predictions, calib_conf>={CONF_THRESH}):")
    print(f"{'pathology':22s} {'rb':3s} {'#conf':>6s} {'#err':>5s} "
          f"{'a_mm':>6s} {'a_cf':>6s} {'a_std':>6s} {'rec':>6s} {'ffr':>6s} {'ship':>5s}")
    for pat in NIH_PATHOLOGIES:
        r = report["per_pathology"][pat]

        def _f(v):
            return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  -  "
        rb = "B" if r["use_roleB"] else "."
        ship = "YES" if r["shipped"] else "."
        print(f"{pat:22s} {rb:3s} {r['n_confident']:6d} {r['n_confident_errors']:5d} "
              f"{_f(r['auroc_mismatch']):>6s} {_f(r['auroc_confidence']):>6s} "
              f"{_f(r['auroc_disagreement']):>6s} {_f(r['flag_recall_at_ffr']):>6s} "
              f"{_f(r['false_flag_rate']):>6s} {ship:>5s}")
    print(f"\n[repm] SHIPPED ({len(shipped)}): {shipped}")
    print(f"[repm] artifact -> {ARTIFACT}")
    print(f"[repm] report   -> {REPORT}")


if __name__ == "__main__":
    main()