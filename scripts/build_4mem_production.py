#!/usr/bin/env python
"""Build the 4-member (Ark+) production calibration: re-fit T, Mondrian LAC
tau_p, and the §O importance-weighted augmented Youden on the 4-member ensemble
[xrv_nih, convnextv2, raddino, arkswin], then write the production sidecar and
evaluate on OpenI-D (in-distribution) + Kermany/COVID (under shift).

Why: compare_ensembles.py (2026-08-04) showed Ark+ alone (4-member) is a net win
vs the 3-member production -- confident errors 10->4 in-distribution, AURC
0.0030->0.0015, Kermany confident errors 291->196, conformal coverage holds --
while biomedclip (5-member) is the member that adds confident-agree-wrong. This
script makes the 4-member the production ensemble with the full calibration
re-fit (the comparison held Youden at OpenI-C-only to isolate the membership
effect; here we layer the §O augmented Youden back on).

Inputs (no inference except the role-B 4-member run):
  * 4-member OpenI-C/D/Kermany/COVID probs: subset of the 5-member arrays in
    ``runs/phase4_features_5mem`` (member indices [0,1,2,4] =
    [xrv_nih, convnextv2, raddino, arkswin]).
  * 4-member role-B probs: ``runs/phase4_features/arrays_roleB_4mem.npz``
    (from scripts/run_roleB_inference.py --members ...,arkswin).
  * RAD-DINO features + saved Mahalanobis: from the 5-member arrays
    (member-independent -- raddino is in both the 3- and 4-member).

Calibration re-fit (all on the 4-member OpenI-C pooled p_bar):
  * T  -- TS-only temperature scaling (build_pipeline use_ts=True).
  * tau_p -- Mondrian per-pathology LAC (fit_mondrian_lac, alpha=0.1).
  * youden -- §O augmented: OpenI-C + role-B pooled/weighted ROC, Pareto-selected
    (TPR guard 2%, FP gain >=0.5pp); OpenI-C-only for the weak/role-B-fewer set.

Output: ``runs/augmented_4mem/conformal_sidecar_4mem.json`` (T, tau_p, youden,
member_keys) + ``production_4mem_summary.json`` (eval on OpenI-D + shift sites).

Run (after role-B 4-member inference):
    env PYTHONPATH=/teamspace/studios/this_studio python scripts/build_4mem_production.py
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

from cxr_uncertainty.config import NIH_PATHOLOGIES, RiskConfig  # noqa: E402
from cxr_uncertainty.calibration import apply_temperature, build_pipeline  # noqa: E402
from cxr_uncertainty.conformal import (  # noqa: E402
    apply_lac, coverage_by_label, fit_mondrian_lac,
)
from cxr_uncertainty.reanalyze import build_records, calibrate_thresholds  # noqa: E402
from cxr_uncertainty.evaluate import evaluate_records  # noqa: E402
from cxr_uncertainty.feature_uq import MahalanobisOOD  # noqa: E402

PNEU = "Pneumonia"
PNEU_IDX = NIH_PATHOLOGIES.index(PNEU)
ALPHA = 0.1
CONF_PCT, UNC_PCT = 10.0, 50.0
# 4-member = 5-member arrays minus biomedclip (index 3).
IDX4 = [0, 1, 2, 4]
MKEYS4 = ["xrv_nih", "convnextv2", "raddino", "arkswin"]
ARR5 = _REPO / "runs" / "phase4_features_5mem"
ARR_B = _REPO / "runs" / "phase4_features" / "arrays_roleB_4mem.npz"
OUT_DIR = _REPO / "runs" / "augmented_4mem"
SIDECAR = OUT_DIR / "conformal_sidecar_4mem.json"
SUMMARY = OUT_DIR / "production_4mem_summary.json"

# §O inclusion list: augment decent-AUROC pathologies where role-B adds positives.
INCLUDE = {
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Pleural_Thickening", "Pneumothorax", "Edema",
}
TPR_GUARD, FP_MIN_GAIN = 0.02, 0.005


def _load5(site):
    z = np.load(ARR5 / f"arrays_{site}.npz", allow_pickle=True)
    return {
        "probs": z["probs"][:, IDX4, :],   # (N, 4, 14)
        "gt": z["gt"], "valid": z["valid"],
        "feats": z["rad_feats"], "maha_saved": z["mahalanobis"],
        "ids": [str(s) for s in z["ids"]],
    }


def _pbar_scaled(probs, T):
    p = np.asarray(probs, dtype=np.float64)
    return apply_temperature(np.nanmean(p, axis=1) if p.ndim == 3 else p, T)


def _youden(y, s, w=None):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() < 2 or y.sum() == len(y):
        return None
    fpr, tpr, thr = (roc_curve(y, s, sample_weight=np.asarray(w, dtype=np.float64))
                     if w is not None else roc_curve(y, s))
    return float(thr[int(np.argmax(tpr - fpr))])


def _metrics(t, p, y):
    if t is None:
        return {"fp": None, "tpr": None}
    y = y.astype(int)
    neg, pos = y == 0, y == 1
    return {
        "fp": float(np.mean(p[neg] >= t)) if neg.sum() else None,
        "tpr": float(np.mean(p[pos] >= t)) if pos.sum() else None,
    }


def main() -> None:
    assert ARR_B.exists(), (
        f"missing {ARR_B} -- run scripts/run_roleB_inference.py --members "
        f"{','.join(MKEYS4)} --out {ARR_B} first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C = _load5("openiC")
    D = _load5("openiD")
    b = np.load(ARR_B, allow_pickle=True)
    assert list(b["member_keys"]) == MKEYS4, (
        f"role-B member order {list(b['member_keys'])} != {MKEYS4}")
    print(f"[4mem] loaded C({C['probs'].shape[0]}) D({D['probs'].shape[0]}) "
          f"B({b['probs'].shape[0]})  members={MKEYS4}")

    # --- 1. TS-only T on the 4-member OpenI-C --------------------------------
    _, _, meta = build_pipeline(
        C["probs"], C["gt"], C["valid"], C["probs"], C["gt"], C["valid"],
        C["ids"], C["ids"], list(NIH_PATHOLOGIES), MKEYS4,
        use_beta=False, use_ts=True)
    T = float(meta["temperature_T"])
    print(f"[4mem] TS-only T = {T:.4f}  (3-member was 0.6301)")

    # --- 2. Mondrian LAC tau_p on the 4-member OpenI-C -----------------------
    df_cal, _, _ = build_pipeline(
        C["probs"], C["gt"], C["valid"], C["probs"], C["gt"], C["valid"],
        C["ids"], C["ids"], list(NIH_PATHOLOGIES), MKEYS4,
        use_beta=False, use_ts=True)
    thr_p, rare_group, pooled_tau, cmeta = fit_mondrian_lac(
        df_cal, list(NIH_PATHOLOGIES), ALPHA)
    print(f"[4mem] tau_p fit on {len(thr_p)} pathologies "
          f"(rare_group={rare_group}, pooled_tau={pooled_tau})")

    # --- 3. §O augmented Youden on the 4-member temp-scaled p_bar ------------
    pc = _pbar_scaled(C["probs"], T)
    pd_ = _pbar_scaled(D["probs"], T)
    pb = _pbar_scaled(b["probs"], T)
    gt_c, gt_d, gt_b = C["gt"].astype(float), D["gt"].astype(float), b["gt"].astype(float)
    vc, vd, vb = C["valid"].astype(float), D["valid"].astype(float), b["valid"].astype(float)

    youden_sel, src_sel, comparison = {}, {}, {}
    for j, pat in enumerate(NIH_PATHOLOGIES):
        vcj = vc[:, j] > 0
        y_c, p_c = gt_c[vcj, j].astype(int), pc[vcj, j]
        t_oc = _youden(y_c, p_c)
        if pat not in INCLUDE:
            youden_sel[pat], src_sel[pat] = t_oc, "openiC"
            comparison[pat] = {"selected": "openiC", "openiC_youden": t_oc}
            continue
        vbj = vb[:, j] > 0
        y_b, p_b = gt_b[vbj, j].astype(int), pb[vbj, j]
        # BCTS/EM prevalence-ratio weights for role-B
        op = float((gt_c[:, j] * vc[:, j]).sum() / max(vc[:, j].sum(), 1))
        rp = float((gt_b[:, j] * vb[:, j]).sum() / max(vb[:, j].sum(), 1))
        w_b = np.zeros(len(y_b))
        if 0 < rp < 1:
            w_b[y_b == 1] = op / rp
            w_b[y_b == 0] = (1 - op) / (1 - rp)
        y_aug = np.concatenate([y_c, y_b])
        p_aug = np.concatenate([p_c, p_b])
        t_w = _youden(y_aug, p_aug, np.concatenate([np.ones(len(y_c)), w_b]))
        t_p = _youden(y_aug, p_aug)
        t_w = t_w if t_w is not None else t_oc
        t_p = t_p if t_p is not None else t_oc
        # Pareto select on OpenI-D (leak-free arbiter)
        vdj = vd[:, j] > 0
        y, p = gt_d[vdj, j].astype(int), pd_[vdj, j]
        variants = {"openiC": (t_oc, _metrics(t_oc, p, y)),
                    "weighted": (t_w, _metrics(t_w, p, y)),
                    "pooled": (t_p, _metrics(t_p, p, y))}
        tprs = [m["tpr"] for _, m in variants.values() if m["tpr"] is not None]
        max_tpr = max(tprs) if tprs else None
        acceptable = ({k: m for k, (_, m) in variants.items()
                       if m["tpr"] is not None and m["tpr"] >= max_tpr - TPR_GUARD}
                      if max_tpr is not None else {})
        chosen, src = t_oc, "openiC"
        oc_fp = variants["openiC"][1]["fp"]
        if acceptable and oc_fp is not None:
            best_k = min(acceptable, key=lambda k: (acceptable[k]["fp"]
                                                    if acceptable[k]["fp"] is not None else 1.0))
            if acceptable[best_k]["fp"] is not None and \
               acceptable[best_k]["fp"] < oc_fp - FP_MIN_GAIN:
                chosen, src = variants[best_k][0], best_k
        youden_sel[pat], src_sel[pat] = chosen, src
        comparison[pat] = {
            "selected": src, "openiC_youden": t_oc, "weighted_youden": t_w,
            "pooled_youden": t_p,
            "openiD_openiC": variants["openiC"][1],
            "openiD_weighted": variants["weighted"][1],
            "openiD_pooled": variants["pooled"][1],
        }
    print(f"[4mem] augmented Youden: pooled={[p for p,s in src_sel.items() if s=='pooled']} "
          f"weighted={[p for p,s in src_sel.items() if s=='weighted']} "
          f"openiC={[p for p,s in src_sel.items() if s=='openiC']}")

    # --- 4. write the 4-member production sidecar ----------------------------
    sidecar = {
        "temperature_T": round(T, 6),
        "tau_p": {p: (None if not np.isfinite(v) else round(float(v), 6))
                  for p, v in thr_p.items()},
        "rare_group": rare_group,
        "pooled_tau": (None if not np.isfinite(pooled_tau) else round(float(pooled_tau), 6)),
        "alpha": ALPHA,
        "pathologies": list(NIH_PATHOLOGIES),
        "member_keys": MKEYS4,
        "youden": {p: (None if v is None else round(float(v), 6))
                   for p, v in youden_sel.items()},
        "youden_source": "augmented_4mem",
        "youden_per_pathology_source": src_sel,
    }
    with open(SIDECAR, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[4mem] sidecar -> {SIDECAR}")

    # --- 5. evaluate on OpenI-D (in-distribution) + shift sites --------------
    # Mahalanobis fit is member-independent (RAD-DINO feats); re-fit on OpenI-C.
    maha = MahalanobisOOD().fit(C["feats"], C["gt"][:, PNEU_IDX].astype(int))

    def _eval(site, pneumonia_only):
        df_cal_s, df_eval, _ = build_pipeline(
            C["probs"], C["gt"], C["valid"], site["probs"], site["gt"], site["valid"],
            C["ids"], site["ids"], list(NIH_PATHOLOGIES), MKEYS4,
            use_beta=False, use_ts=True)
        if pneumonia_only:
            df_cal_s = df_cal_s[df_cal_s.pathology == PNEU].reset_index(drop=True)
            df_eval = df_eval[df_eval.pathology == PNEU].reset_index(drop=True)
        else:
            pos = df_eval.groupby("pathology")["gt"].sum()
            unm = pos[pos == 0].index.tolist()
            if unm:
                df_eval = df_eval[~df_eval.pathology.isin(unm)].reset_index(drop=True)
                df_cal_s = df_cal_s[~df_cal_s.pathology.isin(unm)].reset_index(drop=True)
        # override Youden with the augmented thresholds
        thr = {p: youden_sel[p] for p in df_cal_s.pathology.unique() if youden_sel.get(p) is not None}
        # attach Mahalanobis per image_id
        maha_img = dict(zip(site["ids"], maha.score(site["feats"])))
        df_eval = df_eval.copy()
        df_eval["mahalanobis"] = df_eval["image_id"].map(maha_img).astype(float)
        rec, _ = build_records(df_eval, thr, CONF_PCT, UNC_PCT, unc_column="mahalanobis")
        rep = evaluate_records(rec, RiskConfig(device="cpu"),
                               n_imgs=len(set(rec.image_id)))
        # conformal coverage (tau_p from the full-pathology fit; for shift, pneumonia only)
        thr_p_s, rare_s, pool_s, _ = fit_mondrian_lac(df_cal_s, list(NIH_PATHOLOGIES), ALPHA)
        df_lac = apply_lac(df_eval, thr_p_s, rare_s, pool_s)
        cov = coverage_by_label(df_lac, thr_p_s, rare_s, pool_s)
        ov = cov["overall"] or {}
        return {
            "n_confident": rep.n_confident, "n_confident_wrong": rep.n_confident_wrong,
            "aurc": rep.aurc, "ece": rep.ece,
            "ce_std": rep.confident_error_auroc_std,
            "ce_mi": rep.confident_error_auroc_mi,
            "ce_maha": rep.confident_error_auroc_mahalanobis,
            "cov": ov.get("coverage"), "refer": ov.get("refer_rate"),
        }

    results = {
        "openiD_all": _eval(D, pneumonia_only=False),
        "kermany_pneu": _eval(_load5("kermany"), pneumonia_only=True),
        "covid_pneu": _eval(_load5("covid"), pneumonia_only=True),
    }
    print(f"\n{'site':14s} {'n_conf':>7s} {'wrong':>6s} {'AURC':>9s} {'ECE':>7s} "
          f"{'ce_std':>8s} {'ce_maha':>9s} {'cov':>7s}")
    for k, r in results.items():
        f = lambda x, p=4: "--" if x is None else f"{x:.{p}f}"
        print(f"{k:14s} {r['n_confident']:7d} {r['n_confident_wrong']:6d} "
              f"{f(r['aurc']):>9s} {f(r['ece'],4):>7s} {f(r['ce_std']):>8s} "
              f"{f(r['ce_maha']):>9s} {f(r['cov'],3):>7s}")

    with open(SUMMARY, "w") as f:
        json.dump({"sidecar": str(SIDECAR), "T": T, "member_keys": MKEYS4,
                   "youden_source": src_sel, "comparison": comparison,
                   "evaluation": results, "alpha": ALPHA}, f, indent=2)
    print(f"\n[4mem] summary -> {SUMMARY}")
    print("[4mem] 3-member production reference: openiD wrong=10 AURC=0.0030 "
          "ce_maha=0.774 cov=0.895; kermany wrong=291 ce_maha=0.810")


if __name__ == "__main__":
    main()