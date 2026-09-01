"""Leak-free invariants for the CXR uncertainty build (Part 1).

Two kinds of leaks are guarded:
  1. *Data leak* — OpenI is the leak-free eval/cal target (roles C/D). No OpenI
     image or patient may appear in roles A/B (train / per-member cal), and C/D
     must be patient-disjoint (split-conformal exchangeability). Within each
     training source, A and B must be patient-disjoint.
  2. *Weight leak* — the torchxrayvision `densenet121-res224-all` weights were
     trained on a superset containing OpenI, so they must never be in any
     ensemble used to evaluate on OpenI.

Run: ``pytest tests/test_leak_free.py``. Skips gracefully if no manifest yet.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

MANIFEST = Path("data/manifest.parquet")

# NIH-14 index helper (mirrors config.NIH_PATHOLOGIES order)
from cxr_uncertainty.config import NIH_PATHOLOGIES, DEFAULT_ENSEMBLE, ENSEMBLE_REGISTRY

_NIH_IDX = {p: i for i, p in enumerate(NIH_PATHOLOGIES)}


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        pytest.skip(f"no manifest at {MANIFEST} (run data_curation.build_manifest first)")
    return pd.read_parquet(MANIFEST)


# --- weight-leak invariants (no manifest needed) ---------------------------
def test_default_ensemble_excludes_all_weights():
    """The `-all` xrv weights fold OpenI -> never safe for OpenI eval."""
    assert "all" not in DEFAULT_ENSEMBLE, \
        "DEFAULT_ENSEMBLE must exclude `-all` (OpenI train/test leak)"
    assert ENSEMBLE_REGISTRY["all"].weights == "densenet121-res224-all"


def test_no_openi_eval_ensemble_uses_all():
    """Any ensemble key set that includes OpenI eval must not include `all`.
    Today the only OpenI-eval ensemble is DEFAULT_ENSEMBLE; guard it."""
    assert "all" not in DEFAULT_ENSEMBLE


# --- data-leak invariants (need the manifest) ------------------------------
def test_manifest_leak_free():
    m = _load_manifest()
    # OpenI images never in roles A/B
    bad = m.query("source == 'openi' and split_role in ['A','B']")
    assert len(bad) == 0, f"OpenI images in train roles: {len(bad)}"

    # C and D patient-disjoint within OpenI
    oi = m[m.source == "openi"]
    c_pids = set(oi[oi.split_role == "C"].patient_id)
    d_pids = set(oi[oi.split_role == "D"].patient_id)
    assert not (c_pids & d_pids), "OpenI C/D patient overlap"

    # A and B patient-disjoint within each training source
    for src in ["nih", "chexpert", "padchest"]:
        s = m[m.source == src]
        if s.empty:
            continue
        a_pids = set(s[s.split_role == "A"].patient_id)
        b_pids = set(s[s.split_role == "B"].patient_id)
        assert not (a_pids & b_pids), f"{src} A/B patient overlap"

    # OpenI patient ids must not collide with any train patient id (structural
    # guard; OpenI ids are 'CXRNN', train ids numeric, so this should hold)
    train = m[m.split_role.isin(["A", "B"])]
    overlap = set(oi.patient_id) & set(train.patient_id)
    assert not overlap, f"OpenI<->train patient_id overlap: {sorted(overlap)[:5]}"


def test_manifest_roles_nonempty():
    m = _load_manifest()
    for role in ["A", "C", "D"]:
        assert (m.split_role == role).any(), f"role {role} empty"
    # B may be empty only if no training source had enough images for a 2nd split


def test_ontology_validity_masks():
    m = _load_manifest()
    # NIH rows: all 14 valid
    nih = m[m.source == "nih"]
    if not nih.empty:
        v = nih.iloc[0]["valid"]
        assert all(x == 1 for x in v), "NIH should label all 14 classes"
    # CheXpert rows: Emphysema/Fibrosis/Hernia/Nodule NOT labeled
    chex = m[m.source == "chexpert"]
    if not chex.empty:
        v = chex.iloc[0]["valid"]
        for cls in ["Nodule", "Emphysema", "Fibrosis", "Hernia"]:
            assert v[_NIH_IDX[cls]] == 0, f"CheXpert should not label {cls}"
        for cls in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                   "Effusion", "Pneumonia", "Pneumothorax"]:
            assert v[_NIH_IDX[cls]] == 1, f"CheXpert should label {cls}"
    # OpenI rows: Consolidation NOT labeled, the rest labeled
    oi = m[m.source == "openi"]
    if not oi.empty:
        v = oi.iloc[0]["valid"]
        assert v[_NIH_IDX["Consolidation"]] == 0, "OpenI has no Consolidation map"
        assert v[_NIH_IDX["Atelectasis"]] == 1


def test_manifest_labels_are_multihot_14():
    m = _load_manifest()
    for _, r in m.head(50).iterrows():
        assert len(r["labels"]) == 14 and len(r["valid"]) == 14
        assert all(lb in (0, 1) for lb in r["labels"])
        assert all(v in (0, 1) for v in r["valid"])