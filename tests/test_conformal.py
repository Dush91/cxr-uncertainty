"""Conformal triage tests (plan §L). Verifies the Mondrian per-label LAC core:
the distribution-free coverage guarantee holds on exchangeable held-out data;
the rare-label pooled fallback fires for under-powered pathologies; the
registered ``conformal_triage`` calibrator resolves; and the driver reproduces
the in-distribution guarantee / under-shift break on the saved Phase-4 arrays.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from cxr_uncertainty.conformal import (
    apply_lac, build_conformal_pipeline, coverage_by_label, fit_mondrian_lac,
    lac_scores, save_conformal_sidecar, load_conformal_sidecar,
    assess_conformal_triage,
)


def _make_cal_df(p_bar, gt, pathology="Pneumonia"):
    import pandas as pd
    return pd.DataFrame({
        "image_id": [f"img{i}" for i in range(len(p_bar))],
        "pathology": [pathology] * len(p_bar),
        "p_bar": p_bar.astype(float),
        "gt": gt.astype(int),
        "valid": [1] * len(p_bar),
    })


def test_lac_scores():
    # gt=1 -> 1-p_bar ; gt=0 -> p_bar
    s = lac_scores(np.array([0.2, 0.8, 0.5]), np.array([0, 1, 0]))
    assert np.allclose(s, [0.2, 0.2, 0.5])


def test_mondrian_lac_coverage_guarantee():
    # Exchangeable synthetic data: well-calibrated-ish probs -> held-out coverage
    # >= 1-alpha per pathology (the core split-conformal guarantee).
    rng = np.random.default_rng(0)
    alpha = 0.1
    n = 2000
    n_pos = 400
    gt = np.zeros(n, dtype=int)
    gt[:n_pos] = 1
    # probs correlate with gt (AUROC ~0.8) so sets are non-trivial.
    p = np.where(gt == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
    p = np.clip(p, 0.0, 1.0)
    df_cal = _make_cal_df(p[:n // 2], gt[:n // 2])
    df_eval = _make_cal_df(p[n // 2:], gt[n // 2:])
    thresholds, rare, pooled_tau, meta = fit_mondrian_lac(df_cal, alpha=alpha)
    assert "Pneumonia" in thresholds
    assert rare == []
    df_eval = apply_lac(df_eval, thresholds, rare, pooled_tau)
    cov = coverage_by_label(df_eval, thresholds, rare, pooled_tau)
    cov_p = cov["per_pathology"]["Pneumonia"]["coverage"]
    # finite-sample split-conformal: P(covered) >= 1-alpha (with small slack).
    assert cov_p >= 1 - alpha - 0.02, f"coverage {cov_p} < {1 - alpha - 0.02}"


def test_mondrian_lac_rare_fallback():
    # A pathology with 0 < n < min_n is pooled into the rare group (one shared tau).
    rng = np.random.default_rng(1)
    alpha = 0.1
    big = _make_cal_df(rng.uniform(0, 1, 600), (rng.uniform(0, 1, 600) > 0.7).astype(int),
                       pathology="Pneumonia")
    rare = _make_cal_df(rng.uniform(0, 1, 20), (rng.uniform(0, 1, 20) > 0.5).astype(int),
                        pathology="Hernia")
    import pandas as pd
    df_cal = pd.concat([big, rare], ignore_index=True)
    thresholds, rare_group, pooled_tau, meta = fit_mondrian_lac(df_cal, alpha=alpha, min_n=300)
    assert "Pneumonia" in thresholds
    assert rare_group == ["Hernia"]
    assert meta["rare_label_strategy"] == "pooled_rare"
    assert "Hernia" not in thresholds


def test_mondrian_lac_excludes_empty_pathology():
    # A pathology absent from the cal frame (e.g. Consolidation on OpenI, dropped
    # by the valid mask) is excluded against the canonical list — no threshold,
    # not pooled into the rare group.
    rng = np.random.default_rng(3)
    p = np.clip(rng.uniform(0, 1, 600), 0, 1)
    gt = (rng.uniform(0, 1, 600) < 0.2).astype(int)
    df = _make_cal_df(p, gt, pathology="Pneumonia")
    thresholds, rare, pooled_tau, meta = fit_mondrian_lac(
        df, pathologies=["Pneumonia", "Consolidation"], alpha=0.1, min_n=300)
    assert "Pneumonia" in thresholds
    assert "Consolidation" not in thresholds
    assert rare == []
    assert "Consolidation" in meta["excluded_no_cal"]


def test_apply_lac_set_size_regimes():
    # tau >= 0.5 -> middle band ambiguous {0,1} (set_size 2); tau < 0.5 -> middle
    # band empty {} (set_size 0); edges auto-read (set_size 1).
    import pandas as pd
    df = pd.DataFrame({
        "image_id": ["a", "b", "c", "d"],
        "pathology": ["P", "P", "P", "P"],
        "p_bar": [0.1, 0.5, 0.9, 0.5],
        "gt": [0, 0, 1, 1], "valid": [1, 1, 1, 1],
    })
    # tau = 0.6 (>= 0.5): p_bar=0.1 -> {0} (size1); 0.5 -> {0,1} (size2, refer);
    # 0.9 -> {1} (size1); second 0.5 -> {0,1} (size2).
    d1 = apply_lac(df, {"P": 0.6}, [], np.inf)
    assert list(d1["lac_set_size"]) == [1, 2, 1, 2]
    assert list(d1["lac_refer"]) == [False, True, False, True]
    # covered: gt0&p_bar<=0.6 -> covered; gt1&p_bar>=0.4 -> covered.
    assert list(d1["lac_covered"]) == [True, True, True, True]
    # tau = 0.3 (< 0.5): p_bar=0.5 -> neither (size0, refer, NOT covered).
    d2 = apply_lac(df, {"P": 0.3}, [], np.inf)
    assert list(d2["lac_set_size"]) == [1, 0, 1, 0]
    assert list(d2["lac_refer"]) == [False, True, False, True]
    # p_bar=0.5, gt=0: include_0 = 0.5<=0.3 False -> not covered.
    assert list(d2["lac_covered"]) == [True, False, True, False]


def test_conformal_triage_calibrate_runs():
    # The registered calibrator produces a Records with conformal_active=True and
    # a populated coverage on a synthetic long-form frame.
    import pandas as pd
    from cxr_uncertainty.interfaces import get_calibrator
    rng = np.random.default_rng(2)
    rows = []
    for iid in range(400):
        gt = int(rng.random() < 0.2)
        p = float(np.clip(rng.beta(5, 2, 1)[0] if gt else rng.beta(2, 5, 1)[0], 0, 1))
        mp = f'{{"xrv_nih":{p:.4f},"convnextv2":{min(1,p+0.05):.4f},"raddino":{max(0,p-0.05):.4f}}}'
        rows.append({"image_id": f"img{iid}", "pathology": "Pneumonia",
                     "p_bar": p, "epistemic_std": 0.1, "mutual_info": 0.01,
                     "gt": gt, "valid": 1, "logits": 0.0, "member_probs": mp})
    df = pd.DataFrame(rows)
    cal = get_calibrator("conformal_triage")
    rec, meta = cal(df.iloc[:200], df.iloc[200:], conf_pct=50, unc_pct=50, alpha=0.1)
    assert rec.conformal_active is True
    assert meta["uq_score"] == "conformal_lac"
    assert "coverage" in meta
    assert len(rec.lac_set_size) == len(rec.image_id)


def test_eval_conformal_on_saved_arrays(tmp_path):
    # Smoke: the driver runs from the saved Phase-4-step-6 arrays (no inference)
    # and OpenI-D coverage >= 1-alpha (the guarantee) while under-shift coverage
    # breaks (< 1-alpha). Parallels the production driver test.
    from pathlib import Path
    arrays = Path("runs/phase4_features")
    if not (arrays / "arrays_openiC.npz").exists():
        pytest.skip("Phase-4-step-6 arrays not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_conformal", "scripts/eval_conformal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    member_keys = ["xrv_nih", "convnextv2", "raddino"]
    pathologies = list(mod.NIH_PATHOLOGIES)
    C = mod._load(arrays, "openiC")
    alpha = 0.1
    # in-distribution (all valid classes): per-pathology coverage >= 1-alpha.
    site = mod._load(arrays, "openiD")
    rep, sel, flags, meta = mod.eval_site(C, site, pathologies, member_keys,
                                          10.0, 50.0, alpha, pneumonia_only=False)
    assert rep.conformal_active is True
    assert rep.conformal_coverage is not None
    # the OVERALL in-distribution coverage should meet the target (±2% slack).
    assert rep.conformal_coverage >= 1 - alpha - 0.02, (
        f"OpenI-D coverage {rep.conformal_coverage} < {1 - alpha - 0.02}")
    # under shift: coverage breaks below the target on at least one OOD site
    # (the honest negative — in-distribution calibration does not transfer, Phase 4).
    ood_covs = []
    for site_name in ("kermany", "covid"):
        s = mod._load(arrays, site_name)
        r, _, _, _ = mod.eval_site(C, s, pathologies, member_keys,
                                   10.0, 50.0, alpha, pneumonia_only=True)
        if r.conformal_coverage is not None:
            ood_covs.append(r.conformal_coverage)
    assert ood_covs, "no OOD coverage computed"
    assert min(ood_covs) < 1 - alpha, (
        f"expected under-shift coverage break (< {1 - alpha}); got {ood_covs}")


def _synthetic_uncertainty(p_bars, pathologies):
    """Build a minimal UncertaintyResult for the live conformal policy tests."""
    from cxr_uncertainty.uncertainty import UncertaintyResult
    n = len(p_bars)
    return UncertaintyResult(
        pathologies=list(pathologies), p_bar=np.array(p_bars, dtype=float),
        confidence=np.abs(np.array(p_bars, dtype=float) - 0.5) * 2,
        predictive_entropy=np.zeros(n), expected_entropy=np.zeros(n),
        mutual_info=np.zeros(n), epistemic_std=np.zeros(n),
        n_samples=np.array([3] * n, dtype=float))


def test_conformal_sidecar_roundtrip(tmp_path):
    # Plan §M.2: fit on synthetic data -> save sidecar -> load -> apply in the
    # live assess_conformal_triage policy -> lac_* fields set and `refer` fires
    # on a middle-band p_bar. Mirrors the batch apply_lac semantics.
    from cxr_uncertainty.config import RiskConfig
    rng = np.random.default_rng(0)
    # cal frame where Pneumonia has a tau_p of ~0.6 (>=0.5 -> middle band ambiguous).
    p = np.clip(np.where(rng.random(600) < 0.2,
                         rng.beta(5, 2, 600), rng.beta(2, 5, 600)), 0, 1)
    gt = (p > 0.5).astype(int)
    df_cal = _make_cal_df(p, gt, pathology="Pneumonia")
    thresholds, rare, pooled_tau, lac_meta = fit_mondrian_lac(df_cal, alpha=0.1)
    assert "Pneumonia" in thresholds
    meta = {"temperature_T": 1.0, "tau_p": {k: (None if not np.isfinite(v) else v)
                                            for k, v in thresholds.items()},
            "rare_group": list(rare), "pooled_tau": (None if not np.isfinite(pooled_tau)
                                                    else pooled_tau),
            "member_keys": ["xrv_nih", "convnextv2", "raddino"]}
    sp = tmp_path / "sidecar.json"
    save_conformal_sidecar(sp, meta, ["Pneumonia"], 0.1,
                           member_keys=["xrv_nih", "convnextv2", "raddino"])
    loaded = load_conformal_sidecar(str(sp))
    assert loaded["temperature_T"] == 1.0
    assert "Pneumonia" in loaded["tau_p"]
    # a p_bar in the middle band (between 1-tau and tau) is a "refer" regardless
    # of the tau regime: tau>=0.5 -> ambiguous {0,1} (set_size 2); tau<0.5 ->
    # empty {} (set_size 0). Both refer; the load-bearing claim is that the live
    # policy fires a refer + risk_flag on a middle-band p_bar.
    tau = loaded["tau_p"]["Pneumonia"]
    mid = 0.5 if (1 - tau) <= 0.5 <= tau else (1 - tau + tau) / 2
    unc = _synthetic_uncertainty([mid], ["Pneumonia"])
    cfg = RiskConfig(conformal_sidecar=str(sp), risk_policy="conformal_triage")
    rep = assess_conformal_triage("img1", unc, cfg, gt_labels={"Pneumonia": 0})
    pr = rep.pathologies[0]
    assert pr.lac_set_size in (0, 2), (tau, mid, pr.lac_set_size)
    assert pr.lac_refer is True
    assert pr.risk_flag is True
    assert rep.any_high_risk is True
    # an auto-read p_bar (well below 1-tau AND below tau) -> set_size 1, no refer.
    unc2 = _synthetic_uncertainty([0.01], ["Pneumonia"])
    rep2 = assess_conformal_triage("img2", unc2, cfg, gt_labels={"Pneumonia": 0})
    pr2 = rep2.pathologies[0]
    assert pr2.lac_set_size == 1
    assert pr2.lac_refer is False
    assert pr2.risk_flag is False


def test_assess_conformal_triage_excluded_pathology(tmp_path):
    # Plan §M.2: a pathology absent from the sidecar's tau_p (e.g. Consolidation,
    # not in the OpenI-C fit) is not conformalized -> set_size=-1, refer=True,
    # covered=True (the conservative default), and it counts toward any_high_risk.
    from cxr_uncertainty.config import RiskConfig
    import json
    sidecar = {"temperature_T": 1.0, "tau_p": {"Pneumonia": 0.6},
               "rare_group": [], "pooled_tau": None, "alpha": 0.1,
               "pathologies": ["Pneumonia", "Consolidation"],
               "member_keys": ["xrv_nih", "convnextv2", "raddino"]}
    sp = tmp_path / "sidecar.json"
    json.dump(sidecar, open(sp, "w"))
    unc = _synthetic_uncertainty([0.5, 0.5], ["Pneumonia", "Consolidation"])
    cfg = RiskConfig(conformal_sidecar=str(sp), risk_policy="conformal_triage")
    rep = assess_conformal_triage("img1", unc, cfg,
                                  gt_labels={"Pneumonia": 0, "Consolidation": 0})
    pr_cons = [pr for pr in rep.pathologies if pr.pathology == "Consolidation"][0]
    assert pr_cons.lac_set_size == -1
    assert pr_cons.lac_refer is True
    assert pr_cons.lac_covered is True
    assert pr_cons.risk_flag is True  # excluded -> refer -> flagged
    assert rep.any_high_risk is True