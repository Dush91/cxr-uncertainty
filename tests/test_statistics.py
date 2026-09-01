"""Tests for the statistical-rigor layer (uncertainty_evaluation.md §2.3) and
the always-on proper-scoring metrics added to evaluate.py.

These are pure numpy/sklearn/scipy — no torch, no torchxrayvision — so they run
in any environment (unlike the member tests, which need model weights).
"""
import numpy as np
import pytest

from cxr_uncertainty.evaluate import (_auroc, _brier_decompose, _ece, _e_aurc,
                                      _nll, _reliability_diagram)
from cxr_uncertainty.statistics import (bootstrap_ci, delong_test,
                                        fpr_at_95tpr, mcnemar_test,
                                        ood_detection_metrics)


def _synthetic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    gt = (rng.uniform(0, 1, n) < p).astype(int)
    return p, gt


# --- proper scoring rules / calibration surface ------------------------------
def test_nll_is_positive_and_finite():
    p, gt = _synthetic()
    nll = _nll(p, gt)
    assert nll is not None and np.isfinite(nll) and nll > 0


def test_brier_decomposition_identity():
    """Brier = reliability - resolution + uncertainty (Murphy 1973), up to the
    within-bin variance that binning discards (exact only in the fine-bin
    limit)."""
    p, gt = _synthetic()
    d = _brier_decompose(p, gt)
    assert d is not None
    assert abs(d["brier"] - (d["reliability"] - d["resolution"]
                             + d["uncertainty"])) < 0.01
    # a perfectly calibrated predictor has ~0 reliability
    assert d["reliability"] < 0.01


def test_ece_multi_bins_and_reliability():
    p, gt = _synthetic()
    for nb in [5, 10, 20]:
        assert 0 <= _ece(p, gt, nb) <= 1
    rel = _reliability_diagram(p, gt, n_bins=10)
    assert rel is not None and len(rel["bin_center"]) == len(rel["accuracy"])
    assert len(rel["bin_center"]) == len(rel["count"])


def test_e_aurc_floor():
    """E-AURC = AURC - oracle floor; floor is positive for 0<r<1."""
    assert _e_aurc(0.3, 0.2) == pytest.approx(0.3 - (0.2 + 0.8 * np.log(0.8)))
    assert _e_aurc(None, 0.2) is None
    assert _e_aurc(0.3, 0.0) is None   # degenerate error rate


# --- bootstrap -----------------------------------------------------------------
def test_bootstrap_ci_covers_point_estimate():
    p, gt = _synthetic()
    vals = np.column_stack([p, gt])
    lo, hi = bootstrap_ci(vals, lambda v: _auroc(v[:, 0], v[:, 1]),
                          n_boot=200, seed=1)
    point = _auroc(p, gt)
    assert lo is not None and hi is not None
    assert lo <= point <= hi


# --- DeLong -------------------------------------------------------------------
def test_delong_identical_scores_give_p_1():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 400)
    lab = (rng.uniform(0, 1, 400) < 0.5).astype(int)
    auc_a, auc_b, z, p = delong_test(s, s, lab)
    assert auc_a == pytest.approx(auc_b)
    assert p is not None and p > 0.9


def test_delong_separated_scores_give_small_p():
    rng = np.random.default_rng(1)
    lab = (rng.uniform(0, 1, 500) < 0.5).astype(int)
    good = rng.normal(0, 1, 500) + 2.0 * lab   # strong discriminator
    bad = rng.normal(0, 1, 500)                # chance
    auc_a, auc_b, z, p = delong_test(good, bad, lab)
    assert auc_a > 0.8 and auc_b == pytest.approx(0.5, abs=0.05)
    assert p is not None and p < 0.01


def test_delong_degenerate_class_returns_none():
    s = np.ones(10)
    lab = np.ones(10)  # no negatives
    assert delong_test(s, s, lab) == (None, None, None, None)


# --- McNemar ------------------------------------------------------------------
def test_mcnemar_detects_difference():
    rng = np.random.default_rng(2)
    truth = (rng.uniform(0, 1, 300) < 0.5).astype(bool)
    flag_a = truth.copy()
    flag_b = truth.copy()
    flag_b[::10] = ~flag_b[::10]   # introduce discordance
    chi2, p = mcnemar_test(flag_a, flag_b, truth)
    assert chi2 is not None and p is not None
    assert p < 0.05


def test_mcnemar_identical_flags_degenerate():
    truth = np.array([True, False, True, False])
    assert mcnemar_test(truth, truth, truth) == (None, None)


# --- OOD detection -------------------------------------------------------------
def test_ood_detection_separates_id_from_ood():
    rng = np.random.default_rng(3)
    id_s = rng.normal(0, 1, 500)
    ood_s = rng.normal(3, 1, 500)
    m = ood_detection_metrics(id_s, ood_s)
    assert m["auroc"] > 0.95
    assert m["fpr_at_95tpr"] is not None and m["fpr_at_95tpr"] < 0.1


def test_fpr_at_95tpr_chance_scores():
    rng = np.random.default_rng(4)
    s = rng.normal(0, 1, 1000)
    lab = np.concatenate([np.zeros(500), np.ones(500)])
    assert fpr_at_95tpr(s, lab) == pytest.approx(0.95, abs=0.05)
