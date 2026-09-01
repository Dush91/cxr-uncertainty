"""Registry + factory tests (plan §J.6). Verifies the pluggable seams resolve:
``build_member`` dispatches on ``spec.arch``; ``_resolve_spec`` finds both arch
and legacy registries; the UQ/Calibrator/RiskPolicy registries resolve their
defaults (so the by-name CLI contract is preserved); and the legacy
``DEFAULT_ENSEMBLE`` (xrv-only) still builds end-to-end.
"""
from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")

from cxr_uncertainty.config import (
    ARCH_MEMBER_REGISTRY, DEFAULT_ARCH_ENSEMBLE, DEFAULT_ENSEMBLE, ENSEMBLE_REGISTRY,
    RiskConfig,
)
from cxr_uncertainty.interfaces import (
    get_calibrator, get_risk_policy, get_uq_estimator,
    register_calibrator, register_uq_estimator, register_risk_policy,
)
from cxr_uncertainty.member_factory import _resolve_spec, build_member


def test_resolve_spec_both_registries():
    assert _resolve_spec("xrv_nih").arch == "xrv_densenet"
    assert _resolve_spec("convnextv2").arch == "convnextv2"
    assert _resolve_spec("nih").weights == "densenet121-res224-nih"  # legacy
    with pytest.raises(KeyError):
        _resolve_spec("does_not_exist")


def test_build_member_dispatch_xrv():
    m = build_member(ENSEMBLE_REGISTRY["nih"], device="cpu")
    assert m.key == "nih"
    assert m.arch_family == "xrv_densenet"


def test_uq_registry_default_resolves():
    # ensemble_mc is registered by uncertainty.py at import.
    fn = get_uq_estimator("ensemble_mc")
    assert callable(fn)
    with pytest.raises(KeyError):
        get_uq_estimator("nope")


def test_calibrator_registry_default_resolves():
    # youden is registered by reanalyze.py at import.
    assert callable(get_calibrator("youden"))
    with pytest.raises(KeyError):
        get_calibrator("nope")


def test_production_calibrators_resolve():
    # ts_only + ts_only_mahalanobis are registered by production.py at import
    # (plan §K). youden stays the default; these are opt-in via --calibrator.
    import cxr_uncertainty.production  # noqa: F401  (registers on import)
    assert callable(get_calibrator("ts_only"))
    assert callable(get_calibrator("ts_only_mahalanobis"))


def test_conformal_calibrator_resolves():
    # conformal_triage is registered by conformal.py at import (plan §L).
    import cxr_uncertainty.conformal  # noqa: F401  (registers on import)
    assert callable(get_calibrator("conformal_triage"))


def test_conformal_triage_risk_policy_resolves():
    # the live per-image --risk-policy conformal_triage mirror (plan §M.2) is
    # registered as a RISK policy alongside the calibrator of the same name.
    import cxr_uncertainty.conformal  # noqa: F401  (registers on import)
    fn = get_risk_policy("conformal_triage")
    assert callable(fn)
    # default "threshold" still resolves (run_demo / Phase-0 path unchanged).
    assert callable(get_risk_policy("threshold"))


def test_ts_only_mahalanobis_uses_features_from_cli(tmp_path):
    # Plan §M.1: reanalyze.run(features_path=...) subsets a RAD-DINO features
    # sidecar to the cal/eval sorted-unique image_id order so the
    # ts_only_mahalanobis calibrator runs the real Mahalanobis flag from the CLI
    # (no fallback to ts_only / no uq_warning). Synthetic: 4 images x 2 pathologies
    # + a sidecar covering all 4 ids.
    import json
    import numpy as np
    import pandas as pd
    from cxr_uncertainty.reanalyze import run

    mp = '{"xrv_nih":0.4,"convnextv2":0.6,"raddino":0.5}'
    rows = []
    # 8 images; Pneumonia gt alternates 0/1 (both classes present); Atelectasis
    # gt all 0 (dropped by reanalyze as unmappable) — only Pneumonia is kept, so
    # the cal half (4 images) always has >=2 of some class for the Mahalanobis fit.
    pneum_gt = [0, 1, 0, 1, 0, 1, 0, 1]
    for k in range(8):
        iid = f"img{k}"
        for path, pb, gt in [("Atelectasis", 0.3, 0),
                             ("Pneumonia", 0.7 if pneum_gt[k] else 0.2, pneum_gt[k])]:
            rows.append({"image_id": iid, "pathology": path, "p_bar": pb,
                         "epistemic_std": 0.1, "mutual_info": 0.01, "gt": gt,
                         "valid": 1, "logits": -0.8, "member_probs": mp})
    df = pd.DataFrame(rows)
    csv = tmp_path / "per_record.csv"
    df.to_csv(csv, index=False)

    # sidecar: small D is fine (MahalanobisOOD just needs >=1 dim + >=2 samples
    # in a class). ids cover all 8 CSV images; feats separate the two classes so
    # the eval scores are non-zero.
    rng = np.random.default_rng(0)
    ids = np.array([f"img{k}" for k in range(8)], dtype=object)
    feats = np.zeros((8, 4), dtype=np.float64)
    for k in range(8):
        feats[k] = rng.normal(loc=(3.0 if pneum_gt[k] else -3.0), size=4)
    sidecar = tmp_path / "features.npz"
    np.savez(sidecar, rad_feats=feats, ids=ids)

    out = tmp_path / "out"
    report, sel = run(str(csv), str(out), conf_pct=50, unc_pct=50, seed=0,
                      calibrator="ts_only_mahalanobis",
                      features_path=str(sidecar))
    meta = json.load(open(out / "reanalyze" / "reanalyze_meta.json"))["calibration"]
    # Load-bearing: the CLI path used the real Mahalanobis flag (not the ts_only
    # fallback). uq_score==mahalanobis + no uq_warning + a maha_fit on Pneumonia
    # (name-resolved, not the latent Fibrosis-at-idx-6 bug) is the §M.1 closure.
    assert meta["uq_score"] == "mahalanobis", meta
    assert "uq_warning" not in meta, meta
    assert "maha_fit" in meta and meta["maha_fit"]["label"] == "Pneumonia", meta
    # the reanalyze per_record.csv carries a real (non-zero) Mahalanobis column
    # from the sidecar (column index 14 in the schema_version-2 header).
    import csv as _csv
    with open(out / "reanalyze" / "per_record.csv") as f:
        rd = list(_csv.reader(f))
    header = rd[0]
    mi = header.index("mahalanobis")
    maha_vals = [float(r[mi]) for r in rd[1:]]
    assert len(maha_vals) > 0
    assert max(maha_vals) > 0.0, maha_vals[:4]  # sidecar scores are non-zero


def test_ts_only_mahalanobis_falls_back_without_features():
    # CLI/CSV path: no RAD-DINO features available -> the calibrator honestly
    # delegates to ts_only (epistemic_std flag) and records a uq_warning (plan §K).
    import numpy as np
    import pandas as pd
    import cxr_uncertainty.production as prod
    # tiny synthetic long-form frame with a member_probs JSON column.
    mp = '{"xrv_nih":0.4,"convnextv2":0.6,"raddino":0.5}'
    df = pd.DataFrame({
        "image_id": ["img0", "img0", "img1", "img1"],
        "pathology": ["Atelectasis", "Pneumonia", "Atelectasis", "Pneumonia"],
        "p_bar": [0.3, 0.7, 0.2, 0.8],
        "epistemic_std": [0.1, 0.2, 0.05, 0.3],
        "mutual_info": [0.01, 0.02, 0.005, 0.03],
        "gt": [0, 1, 0, 1], "valid": [1, 1, 1, 1],
        "logits": [-0.8, 0.9, -1.0, 1.2], "member_probs": [mp, mp, mp, mp],
    })
    rec, meta = prod.ts_only_mahalanobis_calibrate(df, df, conf_pct=50, unc_pct=50)
    assert meta["uq_score"] == "epistemic_std"
    assert "uq_warning" in meta
    # Records carry a mahalanobis column (0.0 fallback) so arrays() stays aligned.
    assert len(rec.mahalanobis) == len(rec.image_id)


def test_eval_production_reproduces_step6(tmp_path):
    # CPU smoke: the production driver runs from the saved Phase-4-step-6 arrays
    # (no inference) and the Mahalanobis confident-error AUROC beats the
    # disagreement baseline on the OOD sites (plan §K acceptance).
    import json
    from pathlib import Path
    import numpy as np
    arrays = Path("runs/phase4_features")
    if not (arrays / "arrays_kermany.npz").exists():
        pytest.skip("Phase-4-step-6 arrays not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_production", "scripts/eval_production.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = {}
    member_keys = ["xrv_nih", "convnextv2", "raddino"]
    pathologies = list(mod.NIH_PATHOLOGIES)
    C = mod._load(arrays, "openiC")
    for key, site_name in [("kermany", "kermany"), ("covid", "covid")]:
        site = mod._load(arrays, site_name)
        rep, sel, meta = mod.eval_site(C, site, pathologies, member_keys,
                                       10.0, 50.0, pneumonia_only=True)
        summary[key] = (rep, sel)
    # Mahalanobis > epistemic_std on both OOD sites (reproduces step 6).
    for key in ("kermany", "covid"):
        rep, _ = summary[key]
        assert rep.confident_error_auroc_mahalanobis is not None
        assert rep.confident_error_auroc_mahalanobis > rep.confident_error_auroc_std
    # Kermany reproduces the published ~0.82; COVID ~0.65.
    assert summary["kermany"][0].confident_error_auroc_mahalanobis > 0.75
    assert summary["covid"][0].confident_error_auroc_mahalanobis > 0.60


def test_risk_policy_registry_default_resolves():
    # threshold is registered by risk.py at import.
    assert callable(get_risk_policy("threshold"))
    with pytest.raises(KeyError):
        get_risk_policy("nope")


def test_register_helpers():
    @register_uq_estimator("dummy_uq")
    def f(*a, **k): return None
    assert get_uq_estimator("dummy_uq") is f

    @register_calibrator("dummy_cal")
    def g(*a, **k): return None
    assert get_calibrator("dummy_cal") is g

    @register_risk_policy("dummy_risk")
    def h(*a, **k): return None
    assert get_risk_policy("dummy_risk") is h


def test_legacy_default_ensemble_builds():
    """The pre-refactor DEFAULT_ENSEMBLE (xrv-only) must still build an ensemble
    of Members (behavior-preserving refactor)."""
    from cxr_uncertainty.models import CXREnsemble
    ens = CXREnsemble(DEFAULT_ENSEMBLE, cfg=RiskConfig(device="cpu"))
    assert ens.member_keys == DEFAULT_ENSEMBLE
    assert all(m.arch_family == "xrv_densenet" for m in ens.members)
    assert len(ens.predictable) == 14


def test_arch_default_ensemble_keys_valid():
    for k in DEFAULT_ARCH_ENSEMBLE:
        assert k in ARCH_MEMBER_REGISTRY
    assert "convnextv2" in DEFAULT_ARCH_ENSEMBLE


def test_phase3_member_specs_registered():
    # M4 Ark+ (Swin) + M5 BiomedCLIP (CLIP) are registered with the right arch /
    # input-kind / frozen contract (plan §C, Phase-3 M=5 knee). Cheap: checks the
    # registry specs only — does NOT build the encoders (avoids HF/download).
    bm = ARCH_MEMBER_REGISTRY["biomedclip"]
    assert bm.arch == "biomedclip" and bm.input_kind == "clip" and bm.frozen
    assert bm.hf_id == "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    assert bm.probe_ckpt == "checkpoints/biomedclip_probe.pt"
    # the factory knows how to dispatch both new arches (without building).
    from cxr_uncertainty.member_factory import build_member  # noqa: F401
    import cxr_uncertainty.member_factory as mf
    # build_member dispatches on spec.arch; assert the arch strings are handled
    # by checking the source references the new member modules.
    import inspect
    src = inspect.getsource(build_member)
    assert "biomedclip" in src and "arkswin" in src
    ark = ARCH_MEMBER_REGISTRY["arkswin"]
    assert ark.arch == "arkswin" and ark.frozen and ark.input_kind == "imagenet"
    assert ark.probe_ckpt == "checkpoints/arkswin_probe.pt"
    # local user-uploaded Swin-L checkpoint (not HF) — present after the Ark+ upload.
    assert ark.ckpt_path == "checkpoints/Ark6_swinLarge768_ep50.pth.tar"