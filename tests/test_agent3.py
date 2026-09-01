"""Tests for Agent 3 (cxr_uncertainty.agent3) — no Ollama required.

Covers: raw-gray loading (16-bit PNG, MONOCHROME1 DICOM), the
load_image_tensor golden contract (guards the load_raw_gray refactor),
quality metrics on synthetic images, band assignment + missing-table
fallback, evidence schema + privacy greps, gt gating, degenerate-input
safety, template renderer determinism + citations, the faithfulness
audit (mock liar -> template swap), provider-chain order, key secrecy,
and the cached UI entry point.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import cxr_uncertainty.agent3 as a3
from cxr_uncertainty.utils import load_image_tensor, load_raw_gray

PATHOLOGIES = ["Atelectasis", "Effusion", "Infiltration"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def img_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("agent3")
    rng = np.random.default_rng(7)
    base = np.clip(np.abs(rng.normal(0.45, 0.2, (128, 160))), 0, 1)
    import imageio.v2 as imageio
    imageio.imwrite(os.path.join(d, "u8.png"), (base * 255).astype(np.uint8))
    imageio.imwrite(os.path.join(d, "u16.png"),
                    (base * 65535).astype(np.uint16))
    imageio.imwrite(os.path.join(d, "const.png"), np.full((64, 64), 128, np.uint8))
    imageio.imwrite(os.path.join(d, "noise.png"),
                    np.random.default_rng(3).integers(0, 256, (128, 128), dtype=np.uint8))
    return d


@pytest.fixture()
def base_kwargs(img_dir):
    probs = np.array([[0.72, 0.10, 0.10],
                      [0.40, 0.55, 0.20],
                      [0.40, 0.30, 0.40]])            # (M,P): disagreement on cls0
    return dict(
        path=os.path.join(img_dir, "u8.png"),
        view="AP",
        probs=probs,
        pathologies=PATHOLOGIES,
        youden={"Atelectasis": 0.5, "Effusion": 0.4},
        p_ts=np.array([0.66, 0.11, 0.12]),
        pred_set=["Atelectasis"],
        flag_class="Atelectasis",
        maha_pct=88.0,
        ood=False,
        neighbors=[{"rank": i, "image_id": "tr_%d" % i, "sim": 0.9 - 0.01 * i,
                    "pred_pathology": "Atelectasis", "pred_conf": 0.8,
                    "split": "train", "correct": True, "uncertain": False,
                    "report_confirms": True} for i in range(8)],
        bias={"class_prevalence": {"Atelectasis": 0.0},
              "zero_prevalence_classes": ["Atelectasis"],
              "predicted_in_zero_prevalence": ["Atelectasis"],
              "view_shares": {"AP": 0.82, "PA": 0.18}},
        pool_note="FULL library",
    )


def _tables():
    def t(a, b, c, d):
        return {"percentiles": {"p20": a, "p40": b, "p60": c, "p80": d}}
    return {"by_view": {"AP": {
        "contrast_span": t(0.2, 0.3, 0.4, 0.5),
        "clip_lo_frac": t(0.001, 0.002, 0.004, 0.008),
        "clip_hi_frac": t(0.001, 0.002, 0.003, 0.005),
    }}}


# ---------------------------------------------------------------------------
# raw loading + golden contract
# ---------------------------------------------------------------------------
def test_load_raw_gray_meta(img_dir):
    g8, m8 = load_raw_gray(os.path.join(img_dir, "u8.png"))
    assert g8.dtype == np.uint8 and m8["maxval"] == 255.0 \
        and m8["bit_depth"] == 8
    assert m8["dicom_hu"] is False and m8["kind"] == "png"
    g16, m16 = load_raw_gray(os.path.join(img_dir, "u16.png"))
    assert g16.dtype == np.uint16 and m16["maxval"] == 65535.0 \
        and m16["bit_depth"] == 16


def test_load_image_tensor_golden(tmp_path):
    """Golden contract of load_image_tensor (guards the load_raw_gray
    refactor): values captured from the pre-refactor implementation."""
    import imageio.v2 as imageio
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    rng = np.random.default_rng(0)
    a8 = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    a16 = rng.integers(0, 65536, (64, 80), dtype=np.uint16)
    p8 = tmp_path / "g8.png"
    p16 = tmp_path / "g16.png"
    imageio.imwrite(p8, a8)
    imageio.imwrite(p16, a16)

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.SOPInstanceUID = "1.2.3.4"
    ds.Modality = "DX"
    ds.PhotometricInterpretation = "MONOCHROME1"      # inverted polarity
    ds.Rows, ds.Columns = 64, 80
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
    ds.PixelData = (255 - a8).tobytes()
    pdc = tmp_path / "m1.dcm"
    pydicom.dcmwrite(pdc, ds, write_like_original=False)

    expected = {  # path stem -> (sum, min, max)
        "g8.png": (-44566.027344, -1024.0, 1024.0),
        "g16.png": (52680.621094, -1023.625, 1023.03125),
        "m1.dcm": (-44566.027344, -1024.0, 1024.0),
    }
    for stem, (s, lo, hi) in expected.items():
        t = load_image_tensor(str(tmp_path / stem), img_size=64)
        assert tuple(t.shape) == (1, 1, 64, 64)
        assert abs(t.sum().item() - s) < 1e-2, path
        assert abs(t.min().item() - lo) < 1e-2, path
        assert abs(t.max().item() - hi) < 1e-2, path

    # raw-gray metadata contract
    g, m = load_raw_gray(str(p8))
    assert g.dtype == np.uint8 and m["maxval"] == 255.0 and m["bit_depth"] == 8
    g, m = load_raw_gray(str(p16))
    assert g.dtype == np.uint16 and m["maxval"] == 65535.0 and m["bit_depth"] == 16
    g, m = load_raw_gray(str(pdc))
    assert m["dicom_hu"] is False and m["maxval"] == 255.0  # 8-bit MONOCHROME1

    # raw values round-trip: DICOM inversion recovers the 8-bit image
    g8, _ = load_raw_gray(str(p8))
    gd, _ = load_raw_gray(str(pdc))
    assert np.array_equal(g8, gd)


# ---------------------------------------------------------------------------
# quality metrics
# ---------------------------------------------------------------------------
def test_metrics_constant_image(img_dir):
    q = a3.compute_quality(os.path.join(img_dir, "const.png"))
    assert q["contrast_span"] == 0.0
    assert q["laplacian_var"] == 0.0
    assert q["tenengrad"] == 0.0
    assert q["mirror_asymmetry"] == 0.0


def test_metrics_noise_image(img_dir):
    q = a3.compute_quality(os.path.join(img_dir, "noise.png"))
    assert q["laplacian_var"] > 0.5          # noise -> large gradients
    assert q["tenengrad"] > 0.5
    assert 0.0 <= q["mirror_asymmetry"] <= 2.0
    assert 0.0 <= q["centroid_offset"] <= 1.0


def test_bands_and_missing_table(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    assert ev["image_quality"]["contrast_span_band"] == "unavailable"
    ev = a3.build_evidence(quality_stats=_tables(), **base_kwargs)
    assert ev["image_quality"]["contrast_span_band"] in (
        "very low", "low", "moderate", "high", "very high")
    tab = {"percentiles": {"p20": 0.2, "p40": 0.3, "p60": 0.4, "p80": 0.5}}
    assert a3.assign_band(0.05, tab) == "very low"
    assert a3.assign_band(0.35, tab) == "moderate"
    assert a3.assign_band(0.9, tab) == "very high"
    assert a3.assign_band(0.5, tab, clip_frac=0.2) == "high clipping"
    assert a3.assign_band(float("nan"), tab) == "unavailable"
    assert a3.assign_band(0.5, None) == "unavailable"


# ---------------------------------------------------------------------------
# evidence schema + privacy
# ---------------------------------------------------------------------------
def test_evidence_schema(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=_tables(), **base_kwargs)
    leaves = a3.flatten_leaves(ev)
    for k in ("prediction.argmax", "prediction.max_class_p",
              "prediction.predicted_set.0", "prediction.flag_class",
              "member_disagreement.member_probs.m0",
              "member_disagreement.cross_member_std",
              "member_disagreement.sign_split",
              "label_ambiguity.max_class_p",
              "image_quality.contrast_span", "image_quality.view",
              "neighbors.n", "neighbors.sim_min", "neighbors.top.0.image_id"):
        assert k in leaves, k


def test_evidence_privacy(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    blob = json.dumps(ev)
    for bad in ("patient_id", "image_path", ".png", "/var/folders",
                "OLLAMA_API_KEY"):
        assert bad not in blob, bad


def test_gt_gating(img_dir, base_kwargs):
    kw = dict(base_kwargs)
    kw["gt"] = np.array([1.0, 0.0, -1.0])
    kw["include_gt"] = False
    assert "gt_vector" not in json.dumps(a3.build_evidence(quality_stats=None, **kw))
    kw["include_gt"] = True
    blob = json.dumps(a3.build_evidence(quality_stats=None, **kw))
    assert "gt_vector" in blob and "key_class_gt" in blob


def test_degenerate_inputs(img_dir, base_kwargs):
    kw = dict(base_kwargs)
    kw.update(probs=np.full((3, 3), np.nan), neighbors=[], p_ts=None,
              youden={}, maha_pct=None)
    ev = a3.build_evidence(quality_stats=None, **kw)
    text = a3.render_template(ev)
    assert "not a medical diagnosis" in text


# ---------------------------------------------------------------------------
# template renderer + faithfulness audit
# ---------------------------------------------------------------------------
def test_template_deterministic_and_cited(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    t1, t2 = a3.render_template(ev), a3.render_template(ev)
    assert t1 == t2
    assert t1.count("[EV:") >= 5
    assert a3.audit(t1, ev)["ok"]


def test_audit_catches_fabrications(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    bad = ("Pneumothorax is 0.9123 here [EV:prediction.argmax] and seen on "
           "scan [EV:nope.leaf].")
    res = a3.audit(bad, ev)
    assert not res["ok"]
    joined = " | ".join(res["failures"])
    assert "ungrounded number" in joined          # fabricated number
    assert "invalid citation" in joined           # bogus citation id
    assert "Pneumothorax" in joined               # class absent from evidence


def test_audit_band_mismatch(img_dir, base_kwargs):
    ev = a3.build_evidence(quality_stats=_tables(), **base_kwargs)
    true_band = str(ev["image_quality"]["contrast_span_band"]).lower()
    other = next(b for b in ("very low", "low", "moderate", "high", "very high")
                 if b != true_band)
    bad = ("Contrast span is %s [EV:image_quality.contrast_span_band] "
           "[EV:image_quality.contrast_span]." % other)
    res = a3.audit(bad, ev)
    assert not res["ok"] and any("band mismatch" in f for f in res["failures"])


def test_render_fallback_on_audit_failure(img_dir, base_kwargs, monkeypatch):
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)

    def liar(messages, model="", evidence=None, **_):
        return "Effusion is 0.9123 here [EV:prediction.argmax] [EV:nope]."

    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setitem(a3.PROVIDERS, "ollama_local", liar)
    res = a3.render(ev, backend="auto")
    assert res["backend"] == "ollama_local->template"
    assert not res["audit"]["ok"]
    assert "faithfulness" in (res["error"] or "").lower()
    assert a3.audit(res["text"], ev)["ok"]        # the template swap passes


def test_provider_chain_order(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert a3.provider_chain("auto") == ["ollama_local", "template"]
    monkeypatch.setenv("OLLAMA_API_KEY", "dummy-test-key")
    assert a3.provider_chain("auto") == ["ollama_local", "ollama_cloud", "template"]
    assert a3.provider_chain("template") == ["template"]


def test_api_key_absent_from_artifacts(img_dir, base_kwargs, monkeypatch):
    secret = "sk-secret-12345"
    monkeypatch.setenv("OLLAMA_API_KEY", secret)
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    blob = json.dumps(ev) + json.dumps(a3.evidence_prompt(ev))
    assert secret not in blob


def test_render_cached_paths(img_dir, base_kwargs):
    a3._LAST_EVIDENCE.clear()
    assert "nothing to explain" in a3.render_cached()
    ev = a3.build_evidence(quality_stats=None, **base_kwargs)
    a3._LAST_EVIDENCE.clear()
    a3._LAST_EVIDENCE.update(ev)
    out = a3.render_cached(backend="template")
    assert "[EV:" in out and "Template rendering" in out