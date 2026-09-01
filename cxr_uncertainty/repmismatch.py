"""Class-conditional-density (representation-mismatch) confident-error flag (plan §P).

The three existing flags all miss the same failure mode -- a **confident in-distribution
error**: the members *agree* on the wrong answer (so cross-member ``epistemic_std`` is low
-> the disagreement flag is quiet), the image is in-distribution (so the Mahalanobis *OOD*
flag, which models foreign-ness, stays low), and the conformal LAC set is a single confident
label (so conformal *auto-reads* it, no refer). This is the "confident-agree-wrong core"
(Abe NeurIPS 2022) -- the documented limit of every current flag.

This module implements the ONE signal distinct from all three: a **class-conditional
density (DDU-style, Mukhoti CVPR 2023) representation-mismatch** score on the frozen
RAD-DINO [CLS] features. For each pathology we fit two Ledoit-Wolf Gaussians -- G_pos on
the positives, G_neg on the negatives -- and ask: *does this image's representation sit in
the cluster of images the model called this class?* For a confident positive decision,
``mismatch = ll_neg - ll_pos`` (high = the representation is more likely under the
*negative* cluster -> the model's confident "present" is representation-suspect). For a
confident negative, ``mismatch = ll_pos - ll_neg``. High mismatch = the representation
favours the *opposite* of the model's confident decision = a confident-FP/FN suspicion.

This is the only flag that targets confident in-distribution FP/FN, where the other three
are blind. It is measured (not promised): on OpenI-D, among confident predictions, the
mismatch AUROC beats the model-confidence and disagreement baselines on several
pathologies (Cardiomegaly ~0.84, Edema ~0.87, Pneumonia ~0.81, Pleural_Thickening ~0.88,
Hernia ~0.81), catching ~40-67% of confident FP/FN at a 10% false-flag rate. It is weak /
near-chance on others (Nodule, Effusion, Infiltration) -- those are NOT shipped (honest
Pareto guard: ship only where mismatch beats the confidence baseline). Rare pathologies
(Mass/Pneumothorax) need role-B RAD-DINO features (OpenI-C has too few positives to fit a
768-d Gaussian); see ``scripts/fit_repmismatch.py``.

Leak-free design: the per-pathology Gaussians are fit on OpenI-C + role-B positives; the
per-pathology flag THRESHOLD is set on a HELD-OUT half of OpenI-C (out-of-sample for the
Gaussian fit on the other half + role-B) as the (1-ffr) quantile of mismatch among
CONFIDENT predictions -- so the deployed flag controls the false-flag rate at ~ffr without
ever touching OpenI-D (the leak-free eval arbiter). OpenI-D only MEASURES the result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# RAD-DINO [CLS] is the first 768 cols of the 1536-d zero-padded Alignment.stack layout
# (the last 768 are zero-pad to convnextv2-Large's width). Use the real 768-d features.
_CLS_DIM = 768


def _gauss_loglik(F: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray,
                  logdet_cov: float) -> np.ndarray:
    """Multivariate Gaussian log-likelihood (up to the leading constant is fine for a
    *difference* of two log-likelihoods, but we keep the full term for an absolute
    mismatch that is comparable across pathologies). F (N,D), mu (D,), inv_cov (D,D)."""
    D = mu.shape[0]
    d = F - mu                                  # (N, D)
    quad = np.einsum("nd,dk,nk->n", d, inv_cov, d)   # (N,)
    return -0.5 * (D * np.log(2.0 * np.pi) + logdet_cov + quad)


class RepMismatch:
    """Per-pathology class-conditional-density representation-mismatch confident-error flag.

    Fit two Ledoit-Wolf Gaussians per pathology (pos/neg) on RAD-DINO [CLS] features;
    score a confident decision by how much the representation favours the *opposite*
    class; flag above a per-pathology threshold (set out-of-sample to control the
    false-flag rate). HIGH mismatch = confident-error suspicion.

    The artifact stores only the ENABLED pathologies (those with enough positives AND
    negatives to fit, and -- when fit by ``scripts/fit_repmismatch.py`` -- where the
    mismatch signal beats the model-confidence baseline on the leak-free eval set).
    """

    def __init__(self):
        self.pathologies: list[str] = []
        self.enabled: dict[str, bool] = {}
        self.mu_pos: dict[str, np.ndarray] = {}
        self.mu_neg: dict[str, np.ndarray] = {}
        self.inv_cov_pos: dict[str, np.ndarray] = {}
        self.inv_cov_neg: dict[str, np.ndarray] = {}
        self.logdet_pos: dict[str, float] = {}
        self.logdet_neg: dict[str, float] = {}
        self.flag_threshold: dict[str, float] = {}   # per pathology; NaN/inf = none
        self.n_pos: dict[str, int] = {}
        self.n_neg: dict[str, int] = {}
        self.meta: dict = {}
        # number of leading feature dims to use. RAD-DINO [CLS] = 768 (first 768 of
        # the 1536-d zero-padded Alignment layout); ConvNeXt-V2 = 1536 (the full
        # feature, no padding). Default 768 keeps the RAD-DINO path byte-identical.
        self._feat_dim: int = _CLS_DIM

    def _slice(self, F: np.ndarray) -> np.ndarray:
        """Select the leading ``self._feat_dim`` feature dims (the real feature --
        RAD-DINO [CLS] occupies the first 768 of the 1536-d padded layout;
        ConvNeXt-V2 fills all 1536)."""
        if F.shape[1] >= self._feat_dim:
            F = F[:, :self._feat_dim]
        return F

    # ------------------------------------------------------------------ fit
    def fit(self, features: np.ndarray, gt: np.ndarray, valid: np.ndarray,
            pathologies: list[str], min_pos: int = 20, min_neg: int = 20,
            feat_dim: int | None = None) -> "RepMismatch":
        """Fit pos/neg LedoitWolf Gaussians per pathology on the frozen member's
        penultimate features. By default RAD-DINO [CLS] (first ``_CLS_DIM=768`` cols
        of the 1536-d zero-padded layout); pass ``feat_dim`` to use a different
        feature space (e.g. ``feat_dim=1536`` for ConvNeXt-V2's full features).
        ``gt``/``valid`` are (N, P) (P = len pathologies). A pathology is ENABLED
        iff it has >= ``min_pos`` positives AND >= ``min_neg`` negatives among
        valid rows -- enough to estimate a covariance. (Rare pathologies powered
        by role-B features; see the fit script.)
        """
        from sklearn.covariance import LedoitWolf

        if feat_dim is not None:
            self._feat_dim = int(feat_dim)
        F = np.asarray(features, dtype=np.float64)
        F = self._slice(F)
        y = np.asarray(gt, dtype=np.float64)
        v = np.asarray(valid, dtype=np.float64)
        self.pathologies = list(pathologies)
        self.enabled = {p: False for p in pathologies}
        for j, pat in enumerate(pathologies):
            vj = v[:, j] > 0
            yj = y[vj, j]
            Xj = F[vj]
            pos = yj > 0
            neg = ~pos
            n_pos, n_neg = int(pos.sum()), int(neg.sum())
            self.n_pos[pat], self.n_neg[pat] = n_pos, n_neg
            if n_pos < min_pos or n_neg < min_neg:
                continue    # too few to fit a 768-d Gaussian -> disabled
            Xp, Xn = Xj[pos], Xj[neg]
            mu_p, mu_n = Xp.mean(0), Xn.mean(0)
            cov_p = LedoitWolf().fit(Xp - mu_p).covariance_
            cov_n = LedoitWolf().fit(Xn - mu_n).covariance_
            # pinv is robust to near-singular shrunk covariance; sign logdet from cov.
            self.mu_pos[pat] = mu_p
            self.mu_neg[pat] = mu_n
            self.inv_cov_pos[pat] = np.linalg.pinv(cov_p)
            self.inv_cov_neg[pat] = np.linalg.pinv(cov_n)
            self.logdet_pos[pat] = float(np.linalg.slogdet(cov_p)[1])
            self.logdet_neg[pat] = float(np.linalg.slogdet(cov_n)[1])
            self.enabled[pat] = True
        self.meta = {"min_pos": min_pos, "min_neg": min_neg,
                     "feat_dim": self._feat_dim, "cls_dim": _CLS_DIM}
        return self

    # ------------------------------------------------------------------ score
    def loglik(self, features: np.ndarray, pathology: str):
        """(ll_pos, ll_neg) for one pathology. ``features`` (N,D) or (D,)."""
        F = np.asarray(features, dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        F = self._slice(F)
        if not self.enabled.get(pathology, False):
            return None
        llp = _gauss_loglik(F, self.mu_pos[pathology], self.inv_cov_pos[pathology],
                            self.logdet_pos[pathology])
        lln = _gauss_loglik(F, self.mu_neg[pathology], self.inv_cov_neg[pathology],
                            self.logdet_neg[pathology])
        return llp, lln

    def mismatch(self, features: np.ndarray, dec: np.ndarray,
                 pathology: str) -> np.ndarray:
        """Representation-mismatch score for one pathology, given the model decision
        ``dec`` (N,) in {0,1}. HIGH = representation favours the opposite of the
        decision. NaN where the pathology is disabled."""
        if not self.enabled.get(pathology, False):
            F = np.asarray(features, dtype=np.float64)
            n = F.shape[0] if F.ndim > 1 else 1
            return np.full(n, np.nan)
        ll = self.loglik(features, pathology)
        llp, lln = ll
        # dec==1 (predicted present): mismatch = ll_neg - ll_pos (high = looks negative)
        # dec==0 (predicted absent) : mismatch = ll_pos - ll_neg (high = looks positive)
        mm = np.where(np.asarray(dec).astype(int) == 1, lln - llp, llp - lln)
        return mm

    def decision_distance(self, features: np.ndarray, dec: np.ndarray,
                          pathology: str) -> np.ndarray:
        """Mahalanobis distance to the DECISION-cluster centroid (DDU-style, Mukhoti
        CVPR 2023). For a confident decision, the distance to the cluster the model
        called this class -- ``mu_pos``/``inv_cov_pos`` if ``dec==1``, else
        ``mu_neg``/``inv_cov_neg``. HIGH distance = the representation is far from the
        confident call = confident-error suspicion.

        This is the *cleaner* complement to :meth:`mismatch`: it is a pure quadratic
        distance (no ``logdet`` term dominating the absolute scale), so its absolute
        magnitude is comparable across pathologies and its threshold is a distance
        (intuitive). It targets the same confident in-distribution FP/FN as mismatch
        but from the "far from what you confidently called" angle rather than the
        "favours the opposite class" angle. NaN where the pathology is disabled or
        the decision is not 0/1."""
        if not self.enabled.get(pathology, False):
            F = np.asarray(features, dtype=np.float64)
            n = F.shape[0] if F.ndim > 1 else 1
            return np.full(n, np.nan)
        F = np.asarray(features, dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        F = self._slice(F)
        dec = np.asarray(dec).astype(int)
        out = np.full(F.shape[0], np.nan)
        m1 = dec == 1
        m0 = dec == 0
        if m1.any():
            d = F[m1] - self.mu_pos[pathology]
            q = np.einsum("nd,dk,nk->n", d, self.inv_cov_pos[pathology], d)
            out[m1] = np.sqrt(np.maximum(q, 0.0))
        if m0.any():
            d = F[m0] - self.mu_neg[pathology]
            q = np.einsum("nd,dk,nk->n", d, self.inv_cov_neg[pathology], d)
            out[m0] = np.sqrt(np.maximum(q, 0.0))
        return out

    def set_thresholds(self, features: np.ndarray, dec: np.ndarray,
                       calib_conf: np.ndarray, pathologies: list[str],
                       conf_thresh: float = 0.5, ffr: float = 0.10) -> "RepMismatch":
        """Per-pathology flag threshold = the (1-ffr) quantile of mismatch among
        CONFIDENT predictions (calib_conf >= conf_thresh) on a held-out set. By
        construction this flags the top-``ffr`` highest-mismatch confident predictions
        on the calibration set, i.e. controls the false-flag rate at ~ffr (it needs no
        correctness labels -- only the eval set uses labels to MEASURE recall)."""
        F = np.asarray(features, dtype=np.float64)
        if F.ndim == 1:
            F = F.reshape(1, -1)
        dec = np.asarray(dec, dtype=np.float64)          # (N, P)
        cc = np.asarray(calib_conf, dtype=np.float64)    # (N, P)
        for j, pat in enumerate(pathologies):
            if not self.enabled.get(pat, False):
                self.flag_threshold[pat] = np.inf   # never flags
                continue
            conf = cc[:, j] >= conf_thresh
            if conf.sum() < 20:
                self.flag_threshold[pat] = np.inf
                continue
            mm = self.mismatch(F[conf], dec[conf, j], pat)
            mm = mm[np.isfinite(mm)]
            if mm.size < 20:
                self.flag_threshold[pat] = np.inf
                continue
            self.flag_threshold[pat] = float(np.quantile(mm, 1.0 - ffr))
        return self

    # ------------------------------------------------------------------ flag
    def flag(self, features: np.ndarray, dec, calib_conf, pathology: str,
             conf_thresh: float = 0.5) -> bool:
        """Live per-image, per-pathology flag: enabled AND confident AND mismatch >
        threshold. ``dec``/``calib_conf`` are scalars (the pathology's decision /
        confidence). Returns False for disabled / non-confident / no threshold."""
        if not self.enabled.get(pathology, False):
            return False
        thr = self.flag_threshold.get(pathology, np.inf)
        if not np.isfinite(thr):
            return False
        cc = float(calib_conf)
        if not np.isfinite(cc) or cc < conf_thresh:
            return False
        mm = float(self.mismatch(features, np.array([int(dec)]), pathology)[0])
        if not np.isfinite(mm):
            return False
        return mm > thr

    def flag_image(self, features: np.ndarray, dec_vec, calib_conf_vec,
                   pathologies: list[str], conf_thresh: float = 0.5) -> dict[str, bool]:
        """Convenience: per-pathology flag dict for one image. ``dec_vec`` /
        ``calib_conf_vec`` are (P,) aligned to ``pathologies``."""
        out = {}
        for j, pat in enumerate(pathologies):
            out[pat] = self.flag(features, dec_vec[j], calib_conf_vec[j], pat, conf_thresh)
        return out

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        """Persist to npz (only enabled pathologies, to stay lean)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        enabled = [p for p in self.pathologies if self.enabled.get(p, False)]
        if not enabled:
            np.savez(path, pathologies=np.array([], dtype=object),
                     enabled=np.array([], dtype=object))
            (path.with_suffix(".meta.json")).write_text(json.dumps(self.meta, indent=2))
            return
        n = len(enabled)
        D = self._feat_dim
        mu_p = np.zeros((n, D)); mu_n = np.zeros((n, D))
        inv_p = np.zeros((n, D, D)); inv_n = np.zeros((n, D, D))
        ldp = np.zeros(n); ldn = np.zeros(n); thr = np.full(n, np.inf)
        npos = np.zeros(n, int); nneg = np.zeros(n, int)
        for i, p in enumerate(enabled):
            mu_p[i] = self.mu_pos[p]; mu_n[i] = self.mu_neg[p]
            inv_p[i] = self.inv_cov_pos[p]; inv_n[i] = self.inv_cov_neg[p]
            ldp[i] = self.logdet_pos[p]; ldn[i] = self.logdet_neg[p]
            thr[i] = self.flag_threshold.get(p, np.inf)
            npos[i] = self.n_pos[p]; nneg[i] = self.n_neg[p]
        np.savez(path, pathologies=np.array(enabled, dtype=object),
                 mu_pos=mu_p, mu_neg=mu_n, inv_cov_pos=inv_p, inv_cov_neg=inv_n,
                 logdet_pos=ldp, logdet_neg=ldn, flag_threshold=thr,
                 n_pos=npos, n_neg=nneg, all_pathologies=np.array(self.pathologies,
                 dtype=object))
        meta = dict(self.meta)
        meta["enabled_pathologies"] = enabled
        meta["all_pathologies"] = list(self.pathologies)
        meta["feat_dim"] = self._feat_dim
        (path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RepMismatch":
        path = Path(path)
        d = np.load(path, allow_pickle=True)
        obj = cls()
        obj.pathologies = list(d["all_pathologies"]) if "all_pathologies" in d else \
            list(d["pathologies"])
        obj.enabled = {p: False for p in obj.pathologies}
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            obj.meta = json.loads(meta_path.read_text())
            obj._feat_dim = int(obj.meta.get("feat_dim", _CLS_DIM))
        enabled = list(d["pathologies"])
        if len(enabled) == 0:
            return obj
        for i, p in enumerate(enabled):
            obj.enabled[p] = True
            obj.mu_pos[p] = d["mu_pos"][i]
            obj.mu_neg[p] = d["mu_neg"][i]
            obj.inv_cov_pos[p] = d["inv_cov_pos"][i]
            obj.inv_cov_neg[p] = d["inv_cov_neg"][i]
            obj.logdet_pos[p] = float(d["logdet_pos"][i])
            obj.logdet_neg[p] = float(d["logdet_neg"][i])
            obj.flag_threshold[p] = float(d["flag_threshold"][i])
            obj.n_pos[p] = int(d["n_pos"][i])
            obj.n_neg[p] = int(d["n_neg"][i])
        return obj