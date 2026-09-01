"""Feature-based OOD / epistemic UQ (Part 3 step 6; plan §D, §E.6, §H.1).

Phase 4 showed cross-member disagreement (``epistemic_std`` / MI) is at/below
chance under distribution shift (Kermany Pneumonia AUROC 0.515, COVID 0.478):
two compounding failures --

  1. The OpenI-fit temperature calibration **shatters** under prevalence/scanner
     shift (ECE 0.72-0.74): a single in-distribution T encodes OpenI prevalence
     and does not transfer (§E.3 / Cohen 2020 label shift).
  2. The **confident-agree-wrong core** (Abe NeurIPS 2022) is provably invisible
     to disagreement: under shift all archs confidently agree on the wrong
     answer, so cross-member std is uncorrelated with error.

This module implements two FEATURE-based OOD scores that do NOT rely on
cross-member disagreement and instead target inputs the *representation* was
not trained on (a feature-density view, immune to the prevalence-scale
calibration collapse that broke the disagreement flag):

  * :func:`energy_score` -- 14-class energy (Liu NeurIPS 2020):
        E(x) = -logsumexp(z_14)
    Binary energy degenerates to ``-log(1+exp(z))`` ~ confidence, so we use ALL
    14 NIH logits from the RAD-DINO linear head (a proper multi-class OOD score).
    HIGH energy = the input lies far from the logit simplex the head was trained
    to produce.

  * :class:`MahalanobisOOD` -- K class-conditional Gaussians (Lee NeurIPS 2018)
    on the RAD-DINO 768-d [CLS] features, fit on the in-distribution OpenI
    role-C with Ledoit-Wolf shrinkage. OOD score = min squared Mahalanobis
    distance to the nearest class centroid:
        d(x) = min_k (f - mu_k)^T Sigma_k^{-1} (f - mu_k)     (HIGH = OOD)
    Frozen SSL ViT [CLS] features are a stable host (Kumar 2022; RAD-DINO is
    frozen by design) and a feature-density score sidesteps the prevalence
    calibration collapse entirely.

Both are evaluated as confident-error detectors alongside ``epistemic_std``:
does any feature-based score beat chance (0.5) under shift where disagreement is
at chance? All scores are oriented so HIGH = more OOD / more likely wrong (so
AUROC > 0.5 = diagnostic of error).
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-6


# ---------------------------------------------------------------------------
# 14-class energy score (Liu et al. NeurIPS 2020)
# ---------------------------------------------------------------------------
def energy_score(logits: np.ndarray) -> np.ndarray:
    """14-class energy score. ``logits`` (N, K) raw logits -> (N,) energy.

    ``E(x) = -logsumexp(z)``. HIGH = OOD (far from the trained logit simplex).
    For K=1 this degenerates to ``-softplus(z)`` ~ confidence, which is why we
    use all 14 NIH logits (K=14) -- a proper multi-class OOD score rather than a
    confidence re-label. NaN logits (member does not define that class) are
    replaced with the per-row min so they do not dominate the logsumexp.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    # NaN slots (a member that does not define a class) -> row min, so an
    # undefined class never inflates the energy. For RAD-DINO all 14 are valid.
    rowmin = np.nanmin(z, axis=1, keepdims=True)
    z = np.where(np.isnan(z), np.broadcast_to(rowmin, z.shape), z)
    m = z.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1))
    return -lse


# ---------------------------------------------------------------------------
# Class-conditional Mahalanobis OOD (Lee et al. NeurIPS 2018)
# ---------------------------------------------------------------------------
class MahalanobisOOD:
    """Class-conditional Gaussian Mahalanobis OOD score.

    Fit K class-conditional Gaussians on in-distribution features with
    Ledoit-Wolf shrinkage; ``score(x) = min_k (f-mu_k)^T Sigma_k^{-1}
    (f-mu_k)``. HIGH = OOD. For the Pneumonia task we fit 2 classes (Pneumonia
    pos / neg) on OpenI role-C RAD-DINO [CLS] features, but any label vector
    works (one Gaussian per unique label value with >=2 samples).
    """

    def __init__(self):
        self.classes_: np.ndarray = None
        self.mu_: np.ndarray = None        # (K, D) class means
        self.inv_sigma_: list = []         # per-class (D, D) inverse covariance
        self.n_per_class_: dict = {}

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisOOD":
        from sklearn.covariance import LedoitWolf

        F = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels).astype(int).ravel()
        assert F.shape[0] == y.shape[0], "features/labels length mismatch"
        self.classes_ = np.unique(y)
        mus, invs, counts = [], [], []
        for c in self.classes_:
            mask = y == c
            Xc = F[mask]
            self.n_per_class_[int(c)] = int(mask.sum())
            if mask.sum() < 2:
                # too few to estimate a covariance -> drop this class; the
                # min-distance score then uses the remaining class(es).
                continue
            mu = Xc.mean(axis=0)
            lw = LedoitWolf().fit(Xc - mu)
            inv = np.linalg.pinv(lw.covariance_)
            mus.append(mu); invs.append(inv); counts.append(int(mask.sum()))
        if not mus:
            raise ValueError("MahalanobisOOD.fit: no class had >=2 samples")
        self.mu_ = np.stack(mus)
        self.inv_sigma_ = invs
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance to the nearest class centroid (HIGH=OOD)."""
        F = np.asarray(features, dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        N = F.shape[0]
        best = np.full(N, np.inf, dtype=np.float64)
        for mu, inv in zip(self.mu_, self.inv_sigma_):
            d = F - mu                                   # (N, D)
            # Two-step matmul avoids the (N, D, D) intermediate that the
            # einsum "nd,dk,nk->n" can materialize (~37 GB for N=2k, D=1536).
            # Mathematically identical: diag(d @ inv @ d.T) = (d @ inv) * d row-sum.
            dinv = d @ inv                               # (N, D)
            q = np.einsum("nd,nd->n", dinv, d)           # per-row quadratic form
            best = np.minimum(best, q)
        return best


# ---------------------------------------------------------------------------
# Driver helper: pull a single member's logits + features from an EnsembleOutput
# ---------------------------------------------------------------------------
def extract_member(ensemble_output, member_key: str):
    """Return (logits_14 (B,14), features (B,D)) for one member from an
    ``EnsembleOutput`` (per_member_logits / per_member_features are (M,B,*))."""
    keys = list(ensemble_output.member_keys)
    if member_key not in keys:
        raise KeyError(f"member '{member_key}' not in ensemble {keys}")
    mi = keys.index(member_key)
    logits = ensemble_output.per_member_logits[mi, 0, :].cpu().numpy()   # (P,)
    feats = ensemble_output.per_member_features[mi, 0, :].cpu().numpy()   # (D,)
    return logits.astype(np.float64), feats.astype(np.float64)