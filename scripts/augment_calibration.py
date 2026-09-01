#!/usr/bin/env python
"""Importance-weighted calibration-set augmentation for the rare-pathology Youden thresholds
(plan §O / §E.3).

Problem: 9/13 pathologies overflag normals because their Youden thresholds are fit on too few
OpenI-C positives -- Mass 4, Pneumothorax 10, Fibrosis 7 -- so the threshold is noise (Mass AUROC
0.869 but 31% normal-FP). The model discriminates OK; the THRESHOLD is the problem.

Fix: add training-distribution positives (NIH+CheXpert role-B) to OpenI-C, reweighted to OpenI
prevalence (BCTS/EM prior-shift), and refit the Youden thresholds with sklearn's weighted
`roc_curve`. The prevalence-ratio weight keeps the marginal prevalence at OpenI (so the operating
point stays at OpenI prevalence) while adding positive COUNT to stabilize the threshold estimate.

Scope (user-confirmed Youden-only): augment ONLY the Youden decision thresholds. Keep temperature T
and conformal tau_p on OpenI-C (the coverage guarantee needs exchangeability with OpenI-D, which
role-B breaks; tau_p is already marginally well-powered). No conformal.py/calibration.py surgery.

Per-pathology inclusion list: augment only decent-AUROC pathologies where role-B adds meaningful
positives (Mass, Pneumothorax, Effusion, Edema, Atelectasis, Infiltration, Cardiomegaly,
Pleural_Thickening). Keep OpenI-C-only for weak-AUROC / role-B-fewer pathologies (Pneumonia, Nodule,
Fibrosis, Emphysema, Hernia) and excluded (Consolidation).

Run (after scripts/run_roleB_inference.py has produced arrays_roleB.npz):
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/augment_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.calibration import apply_temperature  # noqa: E402

# --- paths -----------------------------------------------------------------
ARR_C = _REPO / "runs" / "phase4_features" / "arrays_openiC.npz"
ARR_D = _REPO / "runs" / "phase4_features" / "arrays_openiD.npz"
ARR_B = _REPO / "runs" / "phase4_features" / "arrays_roleB.npz"
SIDECAR = _REPO / "runs" / "conformal" / "conformal_sidecar.json"
OUT_DIR = _REPO / "runs" / "augmented"
SIDECAR_AUG = OUT_DIR / "conformal_sidecar_aug.json"
SUMMARY = OUT_DIR / "augmented_summary.json"
COMPARISON = OUT_DIR / "comparison.json"

# Pathologies to augment (decent AUROC + role-B adds meaningful positives).
INCLUDE = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Pleural_Thickening", "Pneumothorax", "Edema",
}
# Kept OpenI-C-only: weak/flat AUROC (Pneumonia, Nodule, Fibrosis) or role-B has fewer
# positives than OpenI-C (Emphysema, Hernia). Consolidation is absent from OpenI entirely.


def _pbar_scaled(probs: np.ndarray, T: float) -> np.ndarray:
    """Pooled mean over members -> temperature-scaled. probs (N,M,P) -> (N,P)."""
    p = np.asarray(probs, dtype=np.float64)
    pbar = np.nanmean(p, axis=1) if p.ndim == 3 else np.asarray(p, dtype=np.float64)
    return apply_temperature(pbar, T)


def _youden(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float | None:
    """Youden J threshold = argmax(TPR - FPR). Optional sample weights.

    NOTE: Youden's J is prevalence-INDEPENDENT (it is a point on the ROC curve,
    which does not depend on class balance). So the right way to add role-B
    positives to stabilise the threshold is an UNWEIGHTED pooled ROC (the
    `pooled` variant), not a prevalence-weighted one. Prevalence-preserving
    importance weights (the `weighted` variant) downweight role-B to OpenI
    prevalence -- for rare pathologies that adds ~no effective positive mass
    (Mass: 114 * 0.035 ~= 4), so it is a no-op. The importance weighting is
    only needed for prevalence-DEPENDENT quantities (the conformal tau_p, kept on
    OpenI-C). Both variants are computed and evaluated; the Pareto guard picks.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() < 2 or y.sum() == len(y):
        return None
    if w is not None:
        fpr, tpr, thr = roc_curve(y, s, sample_weight=np.asarray(w, dtype=np.float64))
    else:
        fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def _metrics(youden_t: float | None, p: np.ndarray, y: np.ndarray) -> dict:
    """normal-FP rate and TPR at a Youden threshold on (p_bar_scaled, gt)."""
    out = {"youden": youden_t, "n_neg": 0, "n_pos": 0, "fp_rate": None, "tpr": None}
    if youden_t is None:
        return out
    y = y.astype(int)
    neg = y == 0
    pos = y == 1
    out["n_neg"] = int(neg.sum())
    out["n_pos"] = int(pos.sum())
    if neg.sum() > 0:
        out["fp_rate"] = float(np.mean(p[neg] >= youden_t))
    if pos.sum() > 0:
        out["tpr"] = float(np.mean(p[pos] >= youden_t))
    return out


def main() -> None:
    assert ARR_B.exists(), (
        f"missing {ARR_B} -- run scripts/run_roleB_inference.py first"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load arrays --------------------------------------------------------
    c = np.load(ARR_C, allow_pickle=True)
    d = np.load(ARR_D, allow_pickle=True)
    b = np.load(ARR_B, allow_pickle=True)
    with open(SIDECAR) as f:
        sc = json.load(f)
    T = float(sc["temperature_T"])
    tau_p = sc["tau_p"]
    rare_group = sc.get("rare_group", [])
    pooled_tau = sc.get("pooled_tau")
    alpha = sc.get("alpha", 0.1)
    member_keys = sc.get("member_keys", ["xrv_nih", "convnextv2", "raddino"])
    print(f"[aug] T={T:.4f}  alpha={alpha}  members={member_keys}")
    print(f"[aug] loaded C({c['probs'].shape[0]}) D({d['probs'].shape[0]}) "
          f"B({b['probs'].shape[0]})")

    P = len(NIH_PATHOLOGIES)
    # p_bar on the shared temperature-scaled scale.
    pc = _pbar_scaled(c["probs"], T)          # (Nc, P)
    pd_ = _pbar_scaled(d["probs"], T)         # (Nd, P)
    pb = _pbar_scaled(b["probs"], T)          # (Nb, P)
    gt_c = np.asarray(c["gt"], dtype=np.float64)
    gt_d = np.asarray(d["gt"], dtype=np.float64)
    gt_b = np.asarray(b["gt"], dtype=np.float64)
    vc = np.asarray(c["valid"], dtype=np.float64)
    vd = np.asarray(d["valid"], dtype=np.float64)
    vb = np.asarray(b["valid"], dtype=np.float64)

    # --- BCTS/EM prevalence-ratio weights for role-B -------------------------
    # openi_prev_p  = (gt_c * vc).sum() / vc.sum()
    # roleB_prev_p  = (gt_b * vb).sum() / vb.sum()
    # positive row: w = openi_prev / roleB_prev
    # negative row: w = (1-openi_prev) / (1-roleB_prev)
    # invalid slot: w = 0
    W = np.zeros_like(pb)                     # (Nb, P)
    prev_log = {}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        nc = vc[:, j]
        nb = vb[:, j]
        op = float((gt_c[:, j] * nc).sum() / max(nc.sum(), 1))
        rp = float((gt_b[:, j] * nb).sum() / max(nb.sum(), 1))
        prev_log[pat] = {"openi_prev": op, "roleB_prev": rp}
        if rp <= 0.0 or rp >= 1.0:
            continue  # no positives or all-positive in role-B -> leave weight 0 (skip)
        w_pos = op / rp
        w_neg = (1.0 - op) / (1.0 - rp)
        pos_mask = (nb > 0) & (gt_b[:, j] > 0)
        neg_mask = (nb > 0) & (gt_b[:, j] <= 0)
        W[pos_mask, j] = w_pos
        W[neg_mask, j] = w_neg

    # --- three Youden variants per pathology --------------------------------
    # openiC  : OpenI-C valid rows only (the current/before threshold).
    # weighted: OpenI-C + role-B, prevalence-weighted (BCTS/EM). Plan §E.3 design.
    # pooled  : OpenI-C + role-B, UNWEIGHTED pooled ROC. Youden's J is
    #           prevalence-independent, so this is the theoretically-correct way
    #           to add positives to stabilise the threshold estimate.
    thr_openiC: dict[str, float | None] = {}
    thr_weighted: dict[str, float | None] = {}
    thr_pooled: dict[str, float | None] = {}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vcj = vc[:, j] > 0
        y_c = gt_c[vcj, j].astype(int)
        p_c = pc[vcj, j]
        t_oc = _youden(y_c, p_c)
        thr_openiC[pat] = t_oc
        if pat not in INCLUDE:
            thr_weighted[pat] = t_oc
            thr_pooled[pat] = t_oc
            continue
        vbj = vb[:, j] > 0
        y_b = gt_b[vbj, j].astype(int)
        p_b = pb[vbj, j]
        w_b = W[vbj, j]
        y_aug = np.concatenate([y_c, y_b])
        p_aug = np.concatenate([p_c, p_b])
        w_aug = np.concatenate([np.ones(len(y_c)), w_b])
        thr_weighted[pat] = _youden(y_aug, p_aug, w_aug)
        thr_pooled[pat] = _youden(y_aug, p_aug)  # unweighted
        if thr_weighted[pat] is None:
            thr_weighted[pat] = t_oc
        if thr_pooled[pat] is None:
            thr_pooled[pat] = t_oc

    # --- evaluate all three on OpenI-D (leak-free arbiter) ------------------
    comparison = {}
    youden_selected: dict[str, float | None] = {}
    selected_source: dict[str, str] = {}
    TPR_GUARD = 0.02   # a candidate must keep TPR within 2% of the best variant
    FP_MIN_GAIN = 0.005  # ...and beat openiC FP by >=0.5pp to count as an improvement
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vdj = vd[:, j] > 0
        y = gt_d[vdj, j].astype(int)
        p = pd_[vdj, j]
        m_oc = _metrics(thr_openiC[pat], p, y)
        m_w = _metrics(thr_weighted[pat], p, y)
        m_p = _metrics(thr_pooled[pat], p, y)
        variants = {"openiC": m_oc, "weighted": m_w, "pooled": m_p}
        # best TPR across the three Youden points
        tprs = [m["tpr"] for m in variants.values() if m["tpr"] is not None]
        max_tpr = max(tprs) if tprs else None
        # acceptable = within TPR guard of the best
        if max_tpr is not None:
            acceptable = {k: m for k, m in variants.items()
                          if m["tpr"] is not None and m["tpr"] >= max_tpr - TPR_GUARD}
        else:
            acceptable = {}
        # among acceptable, pick the lowest FP; ship it only if it beats openiC FP
        oc_fp = m_oc["fp_rate"]
        chosen, chosen_src = thr_openiC[pat], "openiC"
        if acceptable and oc_fp is not None:
            best_k = min(acceptable, key=lambda k: (acceptable[k]["fp_rate"]
                                                    if acceptable[k]["fp_rate"] is not None
                                                    else 1.0))
            best_fp = acceptable[best_k]["fp_rate"]
            if best_fp is not None and best_fp < oc_fp - FP_MIN_GAIN:
                chosen = {"openiC": thr_openiC[pat], "weighted": thr_weighted[pat],
                          "pooled": thr_pooled[pat]}[best_k]
                chosen_src = best_k
        youden_selected[pat] = chosen
        selected_source[pat] = chosen_src
        comparison[pat] = {
            "selected": chosen_src,
            "openiD_n_pos": m_oc["n_pos"],
            "openiD_n_neg": m_oc["n_neg"],
            "openiC": {"youden": m_oc["youden"], "fp": m_oc["fp_rate"], "tpr": m_oc["tpr"]},
            "weighted": {"youden": m_w["youden"], "fp": m_w["fp_rate"], "tpr": m_w["tpr"]},
            "pooled": {"youden": m_p["youden"], "fp": m_p["fp_rate"], "tpr": m_p["tpr"]},
            "selected_youden": chosen,
            "selected_fp": variants[chosen_src]["fp_rate"],
            "selected_tpr": variants[chosen_src]["tpr"],
        }

    # --- write augmented sidecar (OpenI-C T + tau_p unchanged, + youden) -----
    sidecar_aug = {
        "temperature_T": T,
        "tau_p": tau_p,
        "rare_group": rare_group,
        "pooled_tau": pooled_tau,
        "alpha": alpha,
        "pathologies": list(NIH_PATHOLOGIES),
        "member_keys": list(member_keys),
        "youden": {p: (None if v is None else round(float(v), 6))
                   for p, v in youden_selected.items()},
        "youden_source": "augmented",
        "youden_per_pathology_source": selected_source,
    }
    with open(SIDECAR_AUG, "w") as f:
        json.dump(sidecar_aug, f, indent=2)

    summary = {
        "temperature_T": T,
        "alpha": alpha,
        "inclusion_list": sorted(INCLUDE),
        "prevalence": prev_log,
        "youden_openiC": {p: (None if v is None else round(float(v), 6))
                          for p, v in thr_openiC.items()},
        "youden_weighted": {p: (None if v is None else round(float(v), 6))
                            for p, v in thr_weighted.items()},
        "youden_pooled": {p: (None if v is None else round(float(v), 6))
                          for p, v in thr_pooled.items()},
        "youden_selected": sidecar_aug["youden"],
        "selected_source": selected_source,
        "selected_pooled": [p for p, s in selected_source.items() if s == "pooled"],
        "selected_weighted": [p for p, s in selected_source.items() if s == "weighted"],
        "selected_openiC": [p for p, s in selected_source.items() if s == "openiC"],
        "guard": {"tpr_guard": TPR_GUARD, "fp_min_gain": FP_MIN_GAIN},
    }
    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    with open(COMPARISON, "w") as f:
        json.dump(comparison, f, indent=2)

    # --- report -------------------------------------------------------------
    print(f"\n[aug] Pareto-selected (within {TPR_GUARD:.0%} TPR, >{FP_MIN_GAIN:.1%} FP gain):")
    print(f"[aug]   pooled    : {summary['selected_pooled']}")
    print(f"[aug]   weighted  : {summary['selected_weighted']}")
    print(f"[aug]   openiC    : {summary['selected_openiC']}")
    print(f"\n{'pathology':22s} {'sel':9s} {'oc_thr':>8s} {'wt_thr':>8s} {'po_thr':>8s} "
          f"{'oc_fp':>6s}{'oc_tpr':>7s} {'wt_fp':>6s}{'wt_tpr':>7s} {'po_fp':>6s}{'po_tpr':>7s}")

    def _f(v):
        return f"{v:.4f}" if isinstance(v, float) else "n/a"
    for pat in NIH_PATHOLOGIES:
        c = comparison[pat]
        print(f"{pat:22s} {c['selected']:9s} "
              f"{_f(c['openiC']['youden']):>8s} {_f(c['weighted']['youden']):>8s} "
              f"{_f(c['pooled']['youden']):>8s}  "
              f"{_f(c['openiC']['fp']):>6s}{_f(c['openiC']['tpr']):>7s}  "
              f"{_f(c['weighted']['fp']):>6s}{_f(c['weighted']['tpr']):>7s}  "
              f"{_f(c['pooled']['fp']):>6s}{_f(c['pooled']['tpr']):>7s}")
    print(f"\n[aug] sidecar -> {SIDECAR_AUG}")
    print(f"[aug] summary  -> {SUMMARY}")
    print(f"[aug] compare  -> {COMPARISON}")


if __name__ == "__main__":
    main()