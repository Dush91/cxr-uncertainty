"""Statistical rigor for the uncertainty evaluation.

The point estimates in ``evaluate.py`` (confident-error AUROC, AURC, ECE, ...)
answer "how good is this score?"; this module answers "is the difference real?"
-- bootstrap CIs, DeLong's test for two correlated AUROCs, McNemar's test for
paired binary flags, and the OOD-detection metrics (AUROC + FPR@95TPR).

All functions are pure numpy/sklearn/scipy (no torch), so they run on a saved
``per_record.csv`` or the phase-4 feature arrays without re-running inference.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def bootstrap_ci(values: np.ndarray, stat_fn: Callable[[np.ndarray], Optional[float]],
                n_boot: int = 1000, seed: int = 0,
                ci: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for a statistic computed on ``values``.

    ``values`` is (n_examples, n_features) or (n_examples,); each resample draws
    rows with replacement and applies ``stat_fn``. ``stat_fn`` may return None
    (e.g. AUROC when a resample has no positives/negatives) -- those draws are
    skipped. Returns (lo, hi) percentile interval, or (None, None) if fewer than
    ~10 valid resamples.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = stat_fn(values[idx])
        if v is not None and np.isfinite(v):
            stats.append(float(v))
    if len(stats) < 10:
        return None, None
    lo = (1 - ci) / 2
    hi = 1 - lo
    return (float(np.percentile(stats, lo * 100)),
            float(np.percentile(stats, hi * 100)))


def bootstrap_paired_diff(stat_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
                          score_a: np.ndarray, score_b: np.ndarray, label: np.ndarray,
                          n_boot: int = 1000, seed: int = 0,
                          ci: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for ``stat(score_a) - stat(score_b)`` on the SAME rows.

    The two scores are paired per row (e.g. two uncertainty scores on the same
    confident-error population), so each resample draws rows with replacement and
    evaluates the statistic on both scores jointly. ``stat_fn(score, label)`` must
    return a float or None (e.g. AUROC with no positives in a draw); None draws are
    skipped. Returns (lo, hi) of the *difference*, or (None, None) if fewer than
    ~10 valid resamples. This is the significance machinery the DeLong test cannot
    provide when the confident-error population is tiny: it does not assume a
    Gaussian variance, and it works at any error count.
    """
    rng = np.random.default_rng(seed)
    n = len(label)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        va = stat_fn(score_a[idx], label[idx])
        vb = stat_fn(score_b[idx], label[idx])
        if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
            continue
        diffs.append(float(va - vb))
    if len(diffs) < 10:
        return None, None
    lo = (1 - ci) / 2
    hi = 1 - lo
    return (float(np.percentile(diffs, lo * 100)),
            float(np.percentile(diffs, hi * 100)))


# ---------------------------------------------------------------------------
# DeLong's test for two correlated AUROCs (DeLong, DeLong & Clarke-Pearson 1988)
# ---------------------------------------------------------------------------
def _structural_components(score: np.ndarray, label: np.ndarray):
    """DeLong structural components.

    Returns ``(V10, V01, n1, n0)`` where, for positive example i,
    ``V10[i] = (1/n0) * (# negatives with score < s_i + 0.5 * # ties)`` and for
    negative example j, ``V01[j] = (1/n1) * (# positives with score > s_j +
    0.5 * # ties)``. AUC = mean(V10) = mean(V01). ``None`` if a class is missing.
    """
    pos = label == 1
    neg = label == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return None, None, n1, n0
    pos_s = np.sort(score[pos])
    neg_s = np.sort(score[neg])
    # V10 for positives: count negatives strictly below + half the ties.
    lt = np.searchsorted(neg_s, score[pos], side="left")
    rt = np.searchsorted(neg_s, score[pos], side="right")
    V10 = (lt + 0.5 * (rt - lt)) / n0
    # V01 for negatives: count positives strictly above + half the ties.
    gt = n1 - np.searchsorted(pos_s, score[neg], side="right")
    eq = (np.searchsorted(pos_s, score[neg], side="right")
          - np.searchsorted(pos_s, score[neg], side="left"))
    V01 = (gt + 0.5 * eq) / n1
    return V10, V01, n1, n0


def delong_test(score_a: np.ndarray, score_b: np.ndarray, label: np.ndarray):
    """DeLong's test comparing two AUROCs computed on the SAME examples.

    Returns ``(auc_a, auc_b, z, p)`` (two-sided p). ``(None, None, None, None)``
    if a class is missing or the variance is degenerate (n1 or n0 < 2).
    """
    V10a, V01a, n1, n0 = _structural_components(score_a, label)
    V10b, V01b, _, _ = _structural_components(score_b, label)
    if V10a is None or V10b is None:
        return None, None, None, None
    auc_a = float(V10a.mean())
    auc_b = float(V10b.mean())
    if n1 < 2 or n0 < 2:
        return auc_a, auc_b, None, None
    S10 = float(np.cov(V10a, V10b)[0, 1])
    S01 = float(np.cov(V01a, V01b)[0, 1])
    var = S10 / n1 + S01 / n0
    if var <= 0:
        return auc_a, auc_b, None, None
    z = (auc_a - auc_b) / np.sqrt(var)
    from scipy.stats import norm
    p = float(2 * (1 - norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p


# ---------------------------------------------------------------------------
# McNemar's test for paired binary flags (Dietterich 1998)
# ---------------------------------------------------------------------------
def mcnemar_test(flag_a: np.ndarray, flag_b: np.ndarray, truth: np.ndarray):
    """McNemar's test comparing two binary flag rules against the same truth.

    ``truth`` is the ground-truth label each flag is trying to predict (e.g. the
    "prediction is wrong" indicator). Returns ``(chi2, p)`` with continuity
    correction, or ``(None, None)`` when the discordant count is zero.
    """
    err_a = np.asarray(flag_a, dtype=bool) != np.asarray(truth, dtype=bool)
    err_b = np.asarray(flag_b, dtype=bool) != np.asarray(truth, dtype=bool)
    b = int((err_a & ~err_b).sum())   # A wrong, B right
    c = int((~err_a & err_b).sum())   # A right, B wrong
    n = b + c
    if n == 0:
        return None, None
    chi2 = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = float(1 - chi2_dist.cdf(chi2, 1))
    return float(chi2), p


# ---------------------------------------------------------------------------
# OOD detection (labels: 1 = OOD, 0 = ID; higher score = more OOD)
# ---------------------------------------------------------------------------
def fpr_at_95tpr(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """FPR at 95% TPR for OOD detection (the headline number in OOD papers)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return None
    return float(fpr[idx[0]])


def ood_detection_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """AUROC + FPR@95TPR for a score where HIGHER = more OOD.

    ``id_scores`` are the in-distribution scores, ``ood_scores`` the
    out-of-distribution scores. For energy (where lower = more OOD), negate the
    inputs before calling.
    """
    scores = np.concatenate([np.asarray(id_scores, dtype=float),
                             np.asarray(ood_scores, dtype=float)])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(labels, scores))
    return {"auroc": auroc, "fpr_at_95tpr": fpr_at_95tpr(scores, labels)}
