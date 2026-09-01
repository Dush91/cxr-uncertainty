"""Tests for Agent 4 (cxr_uncertainty.agent4) -- no Ollama, no images required.

Agent 4 is a population-level analyser: five lanes of correlational
improvement hypotheses over one fixed eval split, rendered through the same
constrained-renderer + 0-tolerance-audit architecture as Agent 3, with two
extra hard bans (causal language, claimed execution).

Covers: the synthetic-input evidence pack and its schema, the split
discipline that keeps the eval split out of threshold selection, privacy
(no image_path / patient_id ever emitted), the deterministic template
renderer, and each audit check -- fabricated numbers, invalid citations,
causal language, claimed execution, ungrounded intervention concepts --
plus the provider chain and the cached UI entry point.
"""
from __future__ import annotations

import numpy as np
import pytest

import cxr_uncertainty.agent4 as a4

PATHOLOGIES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
               "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
               "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening",
               "Hernia"]
N, M, P = 240, 4, len(PATHOLOGIES)


# ---------------------------------------------------------------------------
# fixtures -- a synthetic population with a known error structure
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def inp():
    rng = np.random.default_rng(11)
    gt = (rng.random((N, P)) < 0.25).astype(np.float64)
    valid = np.ones((N, P), dtype=bool)
    valid[:, 5] = False                       # Nodule undefined -> zero-prev
    # members track gt with member-specific noise so pbar is informative and
    # cross-member std is non-degenerate
    # deliberately overlapping class-conditional means: a realistic ~15-25%
    # error rate, not a separable toy problem
    probs = np.clip(
        gt[:, None, :] * 0.22 + 0.35
        + rng.normal(0, 0.16, (N, M, P)), 1e-4, 1 - 1e-4)
    cal_gt = (rng.random((N, P)) < 0.25).astype(np.float64)
    cal_probs = np.clip(
        cal_gt[:, None, :] * 0.22 + 0.35
        + rng.normal(0, 0.16, (N, M, P)), 1e-4, 1 - 1e-4)
    views = np.array(["AP" if i % 3 else "PA" for i in range(N)])
    return {
        "probs": probs, "gt": gt, "valid": valid,
        "rad_feats": rng.normal(0, 1, (N, 32)),
        "maha": rng.gamma(4.0, 100.0, N),
        "ids": ["img%04d" % i for i in range(N)],
        "pathologies": list(PATHOLOGIES),
        "cal_probs": cal_probs, "cal_gt": cal_gt,
        "cal_valid": np.ones((N, P), dtype=bool),
        "cal_ids": ["cal%04d" % i for i in range(N)],
        "youden": {p: 0.45 for p in PATHOLOGIES},
        "temperature": 0.78,
        "view": views,
        # 3 images per patient: the clustered bootstrap must survive it
        "pat_codes": np.arange(N) // 3,
        "paths": {},                          # no image reads in tests
        "stats": None,
        "calibration_sha": "deadbeef",
        "arr_eval": "/nowhere/arrays_synthetic_eval.npz",
    }


@pytest.fixture(scope="module")
def evidence(inp):
    return a4.build_evidence(inp, k=3, n_boot=60, seed=1234,
                             quality_mode="never")


@pytest.fixture(scope="module")
def leaves(evidence):
    return a4.a3.flatten_leaves(evidence)


# ---------------------------------------------------------------------------
# evidence pack
# ---------------------------------------------------------------------------
def test_error_set_masks_are_consistent(inp):
    es = a4.build_error_set(inp)
    # fp and fn are disjoint and their union is err
    assert not (es["fp"] & es["fn"]).any()
    assert (es["err"] == (es["fp"] | es["fn"])).all()
    # nothing outside the labelled cells can be an error
    assert not (es["err"] & ~es["labelled"]).any()
    # the invalid class contributes no labelled cells
    assert es["labelled"][:, 5].sum() == 0


def test_evidence_schema_and_lanes(evidence):
    assert evidence["schema"] == a4.SCHEMA == "agent4_suggestion_evidence_v1"
    for lane in a4.LANES:
        assert lane in evidence, lane
    prov = evidence["provenance"]
    for k in ("n_eval", "n_err_rows", "n_err_cells", "labelled_cells",
              "err_rate_pct", "calibration_sha", "temperature", "seed",
              "n_boot", "generated_utc"):
        assert k in prov, k
    assert prov["n_eval"] == N
    assert 0.0 <= prov["err_rate_pct"] <= 100.0


def test_zero_prevalence_class_is_flagged_not_a_data_need(evidence):
    """An undefined class must be reported as unmeasurable, never as a class
    that merely needs more examples."""
    zero = evidence["class_data_need"]["zero_prevalence_classes"]
    assert "Nodule" in zero


def test_shares_are_stored_0_100_so_prose_grounds(leaves):
    """A renderer writing '20.6%' must ground against a leaf; storing 0.206
    would silently fail every numeric check. Leaves are stringified, so parse."""
    pct = []
    for k, v in leaves.items():
        if not k.endswith("err_rate_pct"):
            continue
        try:
            pct.append(float(v))
        except (TypeError, ValueError):      # 'none' for unmeasurable classes
            continue
    assert pct, "no err_rate_pct leaves found"
    assert all(0.0 <= v <= 100.0 for v in pct)
    assert any(v > 1.0 for v in pct), "shares look like fractions, not percents"


def test_eval_split_is_never_used_for_threshold_selection(evidence):
    """Lane (c) refits candidate thresholds on the CALIBRATION split only;
    eval effects are descriptive. This flag is the contract."""
    assert evidence["threshold_levers"]["eval_used_for_selection"] is False


def test_slice_reporting_floor_suppresses_small_cells(evidence):
    views = evidence["stratification_slices"]["view"]
    for name, st in views.items():
        if not st.get("reported"):
            continue
        assert st["labelled_cells"] >= a4.MIN_SLICE_CELLS
        assert st["err_rows_n"] >= a4.MIN_SLICE_ERR_ROWS


def test_privacy_no_paths_or_patient_ids(evidence, inp):
    """No image_path, no patient_id, and only the artifact BASENAME."""
    import json
    blob = json.dumps(evidence)
    assert "/nowhere/" not in blob
    assert evidence["provenance"]["eval_artifact_path"] == \
        "arrays_synthetic_eval.npz"
    for key in ("image_path", "patient_id"):
        assert key not in blob
    # patient identity only ever exists as an int code, and not in the pack
    assert "pat_codes" not in blob


# ---------------------------------------------------------------------------
# template renderer
# ---------------------------------------------------------------------------
def test_template_is_deterministic_and_fully_cited(evidence):
    a = a4.render_template(evidence)
    b = a4.render_template(evidence)
    assert a == b
    for lane in a4.LANES:
        text = a4.render_template_lane(evidence, lane)
        assert "[EV:" in text
        assert text.rstrip().endswith(a4.DISCLAIMER)


def test_template_passes_its_own_audit_on_every_lane(evidence):
    """The deterministic gate: this is what must pass before any LLM lane is
    attempted, and it is what a failing LLM rendering is replaced by."""
    for lane in a4.LANES:
        res = a4.audit(a4.render_template_lane(evidence, lane), evidence, lane)
        assert res["ok"], (lane, res["failures"])


def test_unknown_lane_rejected(evidence):
    with pytest.raises(ValueError):
        a4.render_template_lane(evidence, "not_a_lane")


# ---------------------------------------------------------------------------
# the 0-tolerance audit
# ---------------------------------------------------------------------------
def _cited(evidence, lane):
    """A citation id that really exists in this lane, for building fakes."""
    text = a4.render_template_lane(evidence, lane)
    return a4._CITE_RE.search(text).group(1)


def test_audit_catches_fabricated_number(evidence):
    cid = _cited(evidence, "class_data_need")
    bad = ("The error rate is 99.99%% [EV:%s].\n%s" % (cid, a4.DISCLAIMER))
    res = a4.audit(bad, evidence, "class_data_need")
    assert not res["ok"]
    assert any("ungrounded number" in f for f in res["failures"])


def test_audit_catches_invalid_citation(evidence):
    bad = ("Support is thin [EV:class_data_need.NotAClass.err_rate_pct].\n%s"
           % a4.DISCLAIMER)
    res = a4.audit(bad, evidence, "class_data_need")
    assert not res["ok"]
    assert any("invalid citation" in f for f in res["failures"])


def test_audit_bans_causal_language(evidence):
    cid = _cited(evidence, "class_data_need")
    bad = ("Thin support causes these errors [EV:%s].\n%s"
           % (cid, a4.DISCLAIMER))
    res = a4.audit(bad, evidence, "class_data_need")
    assert not res["ok"]
    assert any("causal" in f.lower() for f in res["failures"])


def test_audit_bans_claimed_execution(evidence):
    cid = _cited(evidence, "class_data_need")
    bad = ("We will retrain the ensemble [EV:%s].\n%s" % (cid, a4.DISCLAIMER))
    res = a4.audit(bad, evidence, "class_data_need")
    assert not res["ok"]
    assert any("execut" in f.lower() or "action" in f.lower()
               for f in res["failures"])


def test_audit_catches_ungrounded_pathology():
    """A pathology named in prose but absent from the evidence is a
    hallucinated finding -- the exact failure this project exists to flag.
    Uses a minimal pack because a real one names all 14 classes."""
    tiny = {"schema": a4.SCHEMA,
            "class_data_need": {"Atelectasis": {"err_rate_pct": "19.9300"}}}
    good = ("Atelectasis shows an error rate of 19.9300% "
            "[EV:class_data_need.Atelectasis.err_rate_pct].\n" + a4.DISCLAIMER)
    assert a4.audit(good, tiny, "class_data_need")["ok"]
    bad = ("Pneumothorax shows an error rate of 19.9300% "
           "[EV:class_data_need.Atelectasis.err_rate_pct].\n" + a4.DISCLAIMER)
    res = a4.audit(bad, tiny, "class_data_need")
    assert not res["ok"]
    assert any("Pneumothorax" in f for f in res["failures"])


def test_audit_length_cap(evidence):
    cid = _cited(evidence, "class_data_need")
    bad = ("word " * 260) + "[EV:%s].\n%s" % (cid, a4.DISCLAIMER)
    res = a4.audit(bad, evidence, "class_data_need")
    assert not res["ok"]


# ---------------------------------------------------------------------------
# provider chain + UI entry point
# ---------------------------------------------------------------------------
def test_provider_chain_order(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert a4.provider_chain("auto") == ["ollama_local", "template"]
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-key-value")
    assert a4.provider_chain("auto") == ["ollama_local", "ollama_cloud",
                                         "template"]
    assert a4.provider_chain("template") == ["template"]


def test_render_swaps_template_when_llm_fails_audit(evidence, monkeypatch):
    """A liar provider must never reach the user."""
    def liar(messages, model="", evidence=None, lane=None, **kw):
        return ("Errors rose by 12345.6% [EV:not.a.real.leaf] because of the "
                "scanner.\n" + a4.DISCLAIMER)
    monkeypatch.setitem(a4.PROVIDERS, "ollama_local", liar)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    res = a4.render(evidence, "class_data_need", backend="auto")
    assert res["backend"] == "ollama_local->template"
    assert not res["audit"]["ok"]
    assert "12345.6" not in res["text"]
    assert res["text"] == a4.render_template_lane(evidence, "class_data_need")


def test_api_key_never_enters_artifacts(evidence, monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "super-secret-token")
    import json
    blob = json.dumps(evidence) + a4.render_template(evidence)
    assert "super-secret-token" not in blob


def test_render_cached_without_analysis_is_safe():
    out = a4.render_cached(evidence={})
    assert "no population analysis yet" in out


def test_render_cached_renders_all_lanes(evidence):
    out = a4.render_cached(evidence=evidence, backend="template")
    for lane in a4.LANES:
        assert lane in out
    assert "5/5 lanes passed" in out
