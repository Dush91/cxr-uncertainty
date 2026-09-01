"""Uncertainty estimation.

Given a set of posterior probability samples per (image, pathology) -- coming
from the multi-model ensemble and (optionally) MC-Dropout -- decompose
uncertainty into the standard Kendall-Gal quantities:

  * predictive entropy  H[p_bar]            (total uncertainty)
  * expected entropy    mean_i H[p_i]       (aleatoric / data uncertainty)
  * mutual information  H[p_bar]-mean H[p_i](epistemic / model uncertainty)
  * sample std          std(p_i)            (spread of the posterior)

For binary per-class probabilities p in [0,1], entropy is the Bernoulli entropy
H(p) = -p log p -(1-p) log(1-p). The "samples" for a class are the probability
outputs of every ensemble member that defines that class plus every MC-Dropout
draw -- a single unified posterior sample set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .interfaces import register_uq_estimator

_EPS = 1e-7


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@dataclass
class UncertaintyResult:
    pathologies: List[str]
    # Per-pathology 1-D arrays, length = len(pathologies):
    p_bar: np.ndarray              # posterior-mean probability (the prediction)
    confidence: np.ndarray         # |p_bar - 0.5| * 2  in [0,1]
    predictive_entropy: np.ndarray # total uncertainty  (nats)
    expected_entropy: np.ndarray   # aleatoric uncertainty (nats)
    mutual_info: np.ndarray        # epistemic uncertainty (nats)
    epistemic_std: np.ndarray      # std of posterior samples (probability units)
    n_samples: np.ndarray          # how many samples backed each class

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in [
            "p_bar", "confidence", "predictive_entropy", "expected_entropy",
            "mutual_info", "epistemic_std", "n_samples"]}


def estimate(
    ensemble_probs: torch.Tensor,      # (M, P) with NaN for undefined classes
    mc_probs: Optional[torch.Tensor] = None,  # (K, P) with NaN, or None
    pathologies: Optional[List[str]] = None,
) -> UncertaintyResult:
    """Compute uncertainty decomposition for one image.

    ensemble_probs: (num_models, P) probabilities, NaN where a model does not
        define that class.
    mc_probs: (K, P) MC-Dropout draws (pooled across members), or None.

    Registered as the ``"ensemble_mc"`` UQ estimator (Part 2 Phase 0); the
    numerical core (below) is reused on aligned/calibrated per-member probs in
    Part 3 step 3.
    """
    return _estimate_impl(ensemble_probs, mc_probs, pathologies)


@register_uq_estimator("ensemble_mc")
def _ensemble_mc_estimator(ensemble_probs, mc_probs=None, pathologies=None, **_):
    """Registry shim -> estimate(). Default UQ estimator."""
    return _estimate_impl(ensemble_probs, mc_probs, pathologies)


def _estimate_impl(ensemble_probs, mc_probs, pathologies):
    ens = ensemble_probs.cpu().numpy().astype(np.float64)
    P = ens.shape[1]
    pathologies = pathologies or [f"p{i}" for i in range(P)]

    mc = mc_probs.cpu().numpy().astype(np.float64) if mc_probs is not None else None
    if mc is not None and mc.size == 0:
        mc = None

    p_bar = np.zeros(P, dtype=np.float64)
    pred_ent = np.zeros(P, dtype=np.float64)
    exp_ent = np.zeros(P, dtype=np.float64)
    mi = np.zeros(P, dtype=np.float64)
    std = np.zeros(P, dtype=np.float64)
    n_samples = np.zeros(P, dtype=np.int64)

    for pi in range(P):
        samples = ens[:, pi]
        samples = samples[~np.isnan(samples)]
        if mc is not None:
            mc_s = mc[:, pi]
            mc_s = mc_s[~np.isnan(mc_s)]
            samples = np.concatenate([samples, mc_s])
        if samples.size == 0:
            p_bar[pi] = np.nan
            continue
        n_samples[pi] = samples.size
        pbar = float(np.mean(samples))
        p_bar[pi] = pbar
        pred_ent[pi] = float(_bernoulli_entropy(np.array([pbar]))[0])
        ent_samples = _bernoulli_entropy(samples)
        exp_ent[pi] = float(np.mean(ent_samples))
        mi[pi] = float(pred_ent[pi] - exp_ent[pi])
        std[pi] = float(np.std(samples))

    confidence = np.abs(p_bar - 0.5) * 2.0
    confidence = np.where(np.isnan(p_bar), np.nan, confidence)

    return UncertaintyResult(
        pathologies=pathologies,
        p_bar=p_bar.astype(np.float32),
        confidence=confidence.astype(np.float32),
        predictive_entropy=pred_ent.astype(np.float32),
        expected_entropy=exp_ent.astype(np.float32),
        mutual_info=mi.astype(np.float32),
        epistemic_std=std.astype(np.float32),
        n_samples=n_samples,
    )