"""Evaluation: does epistemic uncertainty catch confident errors?

We collect one record per (image, predictable pathology) and measure:

  * confident_error_auroc : among CONFIDENT predictions, AUROC of the epistemic
    signal (ensemble std / mutual info) for separating *wrong* from *correct*.
    This is the core claim -- if this is high, the risk flag works.
  * risk_coverage / AURC  : abstain on the most epistemically-uncertain
    confident predictions; error rate among retained vs coverage.
  * confident_error_recall: fraction of confident_fp + confident_fn that the
    risk_flag catches, and the false-flag rate among confident_correct.
  * standard metrics      : per-pathology AUROC of p_bar for the task itself,
    sensitivity/specificity at the decision threshold, and ECE (calibration).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import NIH_PATHOLOGIES, RiskConfig
from .risk import ImageRiskReport, PathologyRisk


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration. numpy>=2 renamed ``np.trapz`` -> ``np.trapezoid``;
    this shim keeps the code working on both (the build host has numpy 1.26)."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


@dataclass
class Records:
    image_id: List[str] = field(default_factory=list)
    pathology: List[str] = field(default_factory=list)
    p_bar: List[float] = field(default_factory=list)
    decision: List[int] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    epistemic_std: List[float] = field(default_factory=list)
    mutual_info: List[float] = field(default_factory=list)
    gt: List[int] = field(default_factory=list)
    is_confident: List[bool] = field(default_factory=list)
    risk_flag: List[bool] = field(default_factory=list)
    error_type: List[str] = field(default_factory=list)
    # --- schema_version 2 additions (Part 2/3): logits/valid/member_probs.
    # `logits` = inverse-sigmoid of p_bar (BCTS reloads this in Part 3);
    # `valid` = per-example per-class validity mask (1 = trustworthy label);
    # `member_probs` = JSON {member_key: prob} (sparse; members defining the class).
    valid: List[int] = field(default_factory=list)
    logits: List[float] = field(default_factory=list)
    member_probs: List[str] = field(default_factory=list)
    # --- production UQ (Phase 4 step 6): the Mahalanobis feature-density OOD score
    # on frozen RAD-DINO [CLS] features. The production confident-error flag uses
    # this (reanalyze.build_records unc_column="mahalanobis"); 0.0 when absent.
    mahalanobis: List[float] = field(default_factory=list)
    # --- conformal triage (Part 3 step 7): Mondrian per-label LAC set-valued
    # output. `lac_set_size` in {0,1,2} (-1 = pathology not conformalized);
    # `lac_refer` = (set_size != 1) -> defer to human; `lac_covered` = true label
    # in the LAC set (the distribution-free coverage event). `conformal_active`
    # gates whether evaluate_records reports the conformal metrics (else None ->
    # backward-compatible with non-conformal calibrators).
    lac_set_size: List[int] = field(default_factory=list)
    lac_refer: List[bool] = field(default_factory=list)
    lac_covered: List[bool] = field(default_factory=list)
    conformal_active: bool = False
    # --- kNN feature-distance baseline (uncertainty_evaluation.md §4.3): mean
    # distance to the k nearest neighbors in a reference feature set (e.g. the
    # calibration split's RAD-DINO [CLS] features). Higher = more OOD / more
    # uncertain. Populated by reanalyze when a features sidecar is supplied
    # (--features); 0.0 when absent (like mahalanobis).
    knn_dist: List[float] = field(default_factory=list)

    def add(self, iid: str, pr: PathologyRisk, member_probs: str = "{}",
            valid: int = 1):
        if pr.gt is None:
            return
        self.image_id.append(iid)
        self.pathology.append(pr.pathology)
        self.p_bar.append(pr.p_bar)
        self.decision.append(pr.decision)
        self.confidence.append(pr.confidence)
        self.epistemic_std.append(pr.epistemic_std)
        self.mutual_info.append(pr.mutual_info)
        self.gt.append(pr.gt)
        self.is_confident.append(pr.is_confident)
        self.risk_flag.append(pr.risk_flag)
        self.error_type.append(pr.error_type or "unknown")
        self.valid.append(valid)
        # inverse-sigmoid of p_bar (clipped) -> the pooled-mean logit BCTS needs.
        p = min(max(pr.p_bar, 1e-6), 1 - 1e-6)
        self.logits.append(float(np.log(p / (1.0 - p))))
        self.member_probs.append(member_probs)
        # Mahalanobis is populated only by the production driver (build_records
        # from feature arrays); the live per-image path has no feature score ->
        # 0.0, which evaluate_records treats as "absent" (auc_maha=None).
        self.mahalanobis.append(0.0)
        # LAC conformal: read from the PathologyRisk so the live conformal-triage
        # risk policy (plan §M.2) populates the lac_* columns on the live
        # per_record.csv. The ``threshold`` policy leaves PathologyRisk at its
        # defaults (1/False/True) -> this path stays byte-identical for run_demo.
        self.lac_set_size.append(int(pr.lac_set_size))
        self.lac_refer.append(bool(pr.lac_refer))
        self.lac_covered.append(bool(pr.lac_covered))
        # kNN feature-distance baseline: the live per-image path has no feature
        # sidecar -> 0.0 (treated as "absent" by evaluate_records, like maha).
        self.knn_dist.append(0.0)

    def arrays(self) -> Dict[str, np.ndarray]:
        return {
            "p_bar": np.array(self.p_bar, dtype=float),
            "decision": np.array(self.decision, dtype=int),
            "confidence": np.array(self.confidence, dtype=float),
            "epistemic_std": np.array(self.epistemic_std, dtype=float),
            "mutual_info": np.array(self.mutual_info, dtype=float),
            "gt": np.array(self.gt, dtype=int),
            "is_confident": np.array(self.is_confident, dtype=bool),
            "risk_flag": np.array(self.risk_flag, dtype=bool),
            "wrong": (np.array(self.decision, dtype=int) != np.array(self.gt, dtype=int)).astype(int),
            "mahalanobis": np.array(self.mahalanobis, dtype=float),
            "lac_set_size": np.array(self.lac_set_size, dtype=int),
            "lac_refer": np.array(self.lac_refer, dtype=bool),
            "lac_covered": np.array(self.lac_covered, dtype=bool),
            "knn_dist": np.array(self.knn_dist, dtype=float),
        }


# ---------------------------------------------------------------------------
def _auroc(score: np.ndarray, label: np.ndarray) -> Optional[float]:
    """ROC-AUC via Mann-Whitney U. Returns None if a class is missing."""
    pos = score[label == 1]
    neg = score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # handle ties by average rank
    uniq, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        avg = (ranks.reshape(-1)[np.argsort(order)])  # not needed; recompute below
        s = np.sort(score)
        # average ranks for ties
        rank_sum = np.zeros(len(score))
        for u, c in zip(uniq, cnt):
            mask = score == u
            r = ranks[mask].mean()
            rank_sum[mask] = r
        ranks_pos = rank_sum[label == 1].sum()
    else:
        ranks_pos = ranks[label == 1].sum()
    n1, n0 = len(pos), len(neg)
    u = ranks_pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def _ece(probs: np.ndarray, gt: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary probabilities."""
    if len(probs) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            m = (probs >= lo) & (probs <= hi)
        else:
            m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        acc = gt[m].mean()
        conf = probs[m].mean()
        ece += (m.sum() / N) * abs(acc - conf)
    return float(ece)


def _nll(probs: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """Negative log-likelihood (log-loss) for binary probabilities."""
    if len(probs) == 0:
        return None
    p = np.clip(probs, 1e-12, 1 - 1e-12)
    return float(-np.mean(gt * np.log(p) + (1 - gt) * np.log(1 - p)))


def _brier_decompose(probs: np.ndarray, gt: np.ndarray, n_bins: int = 10) -> Optional[dict]:
    """Brier score with the Murphy (1973) reliability - resolution + uncertainty
    decomposition. A lower Brier does not by itself mean better calibration --
    the decomposition separates the calibration term (reliability) from the
    sharpness terms (resolution, uncertainty)."""
    if len(probs) == 0:
        return None
    brier = float(np.mean((probs - gt) ** 2))
    base = float(gt.mean())
    bins = np.linspace(0, 1, n_bins + 1)
    reliability = resolution = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            m = (probs >= lo) & (probs <= hi)
        else:
            m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        w = m.sum() / N
        reliability += w * (gt[m].mean() - probs[m].mean()) ** 2
        resolution += w * (gt[m].mean() - base) ** 2
    uncertainty = base * (1 - base)
    return {"brier": brier, "reliability": reliability,
            "resolution": resolution, "uncertainty": uncertainty}


def _reliability_diagram(probs: np.ndarray, gt: np.ndarray,
                         n_bins: int = 10) -> Optional[dict]:
    """Bin centers / accuracy / mean confidence / counts for a reliability
    diagram (Guo et al. 2017)."""
    if len(probs) == 0:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            m = (probs >= lo) & (probs <= hi)
        else:
            m = (probs > lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(float(gt[m].mean()))
        confs.append(float(probs[m].mean()))
        counts.append(int(m.sum()))
    return {"bin_center": centers, "accuracy": accs,
            "confidence": confs, "count": counts}


def _e_aurc(aurc: Optional[float], error_rate: float) -> Optional[float]:
    """Excess-AURC: AURC minus the oracle floor AURC* ~= r + (1-r)ln(1-r)
    (Geifman, Uziel & El-Yaniv, ICLR 2019). Isolates ranking quality from
    accuracy, so AURC is comparable across models of different error rates."""
    if aurc is None or error_rate <= 0 or error_rate >= 1:
        return None
    floor = error_rate + (1 - error_rate) * np.log(1 - error_rate)
    return float(aurc - floor)


# ---------------------------------------------------------------------------
@dataclass
class EvalReport:
    n_images: int
    n_records: int
    n_confident: int
    n_confident_wrong: int
    confident_error_auroc_std: Optional[float]
    confident_error_auroc_mi: Optional[float]
    confident_error_auroc_mahalanobis: Optional[float]  # production flag (Phase 4 step 6)
    confident_error_recall: Optional[float]   # of confident errors, frac flagged
    false_flag_rate: Optional[float]          # of confident correct, frac flagged
    aurc: Optional[float]                     # area under risk-coverage curve
    coverage_at_lowest_risk: Optional[Tuple[float, float]]  # (coverage, error)
    ece: float
    per_pathology: Dict[str, dict]
    error_type_counts: Dict[str, int]
    risk_config: dict
    # --- conformal triage (Part 3 step 7): distribution-free coverage layer.
    # Populated only when `Records.conformal_active` (a conformal calibrator
    # attached LAC columns); else None -> backward-compatible with non-conformal
    # calibrators (youden/ts_only/...). `conformal_coverage` = P(true label in
    # LAC set) over conformalized rows; `conformal_size` = avg LAC set size;
    # `per_pathology_conformal` = per-pathology coverage/size/refer.
    conformal_active: bool = False
    conformal_coverage: Optional[float] = None
    conformal_size: Optional[float] = None
    per_pathology_conformal: Optional[Dict[str, dict]] = None
    # --- proper scoring rules + calibration surface (uncertainty_evaluation.md
    # §1.1/§1.2): always-on. `brier` = mean (p_bar - gt)^2; `brier_decomposition`
    # = Murphy reliability/resolution/uncertainty; `nll` = log-loss; `ece_multi`
    # = ECE at several bin counts (equal-width binning is biased -- report a
    # range); `reliability` = bin centers/acc/conf/count for the diagram.
    brier: Optional[float] = None
    brier_decomposition: Optional[dict] = None
    nll: Optional[float] = None
    ece_multi: Optional[dict] = None
    reliability: Optional[dict] = None
    # --- discrimination baselines (§2.1): `confidence_auroc` = confident-error
    # AUROC of the confidence-only baseline (1 - confidence) -- the null that
    # uncertainty must beat; `aurc_by_score` = AURC + E-AURC per score
    # (epistemic_std / mutual_info / confidence / mahalanobis / knn).
    confidence_auroc: Optional[float] = None
    aurc_by_score: Optional[Dict[str, dict]] = None

    def to_dict(self) -> dict:
        d = {
            "schema_version": 3,
            "n_images": self.n_images, "n_records": self.n_records,
            "n_confident": self.n_confident,
            "n_confident_wrong": self.n_confident_wrong,
            "confident_error_auroc_std": self.confident_error_auroc_std,
            "confident_error_auroc_mi": self.confident_error_auroc_mi,
            "confident_error_auroc_mahalanobis": self.confident_error_auroc_mahalanobis,
            "confident_error_recall": self.confident_error_recall,
            "false_flag_rate": self.false_flag_rate,
            "aurc": self.aurc,
            "coverage_at_lowest_risk": self.coverage_at_lowest_risk,
            "ece": self.ece,
            "per_pathology": self.per_pathology,
            "error_type_counts": self.error_type_counts,
            "risk_config": self.risk_config,
            "conformal_active": self.conformal_active,
            "conformal_coverage": self.conformal_coverage,
            "conformal_size": self.conformal_size,
            "per_pathology_conformal": self.per_pathology_conformal,
            "brier": self.brier,
            "brier_decomposition": self.brier_decomposition,
            "nll": self.nll,
            "ece_multi": self.ece_multi,
            "reliability": self.reliability,
            "confidence_auroc": self.confidence_auroc,
            "aurc_by_score": self.aurc_by_score,
        }
        return d


def evaluate(reports: List[ImageRiskReport], cfg: RiskConfig,
             conformal_active: bool = False) -> EvalReport:
    rec = Records()
    rec.conformal_active = conformal_active
    for r in reports:
        for pr in r.pathologies:
            rec.add(r.image_id, pr)
    n_imgs = len({i for i in rec.image_id})
    return evaluate_records(rec, cfg, n_imgs)


def evaluate_records(rec: Records, cfg: RiskConfig, n_imgs: int) -> EvalReport:
    """Compute the EvalReport directly from a Records object (no ImageRiskReports
    required). Used by the live pipeline and by reanalyze (which rebuilds
    Records from a saved per_record.csv with per-class calibrated thresholds)."""
    a = rec.arrays()
    if len(rec.image_id) == 0:
        return EvalReport(
            n_images=0, n_records=0, n_confident=0, n_confident_wrong=0,
            confident_error_auroc_std=None, confident_error_auroc_mi=None,
            confident_error_auroc_mahalanobis=None,
            confident_error_recall=None, false_flag_rate=None,
            aurc=None, coverage_at_lowest_risk=None, ece=0.0,
            per_pathology={}, error_type_counts={"unknown": 0},
            risk_config={"decision_thresh": cfg.decision_thresh,
                         "conf_thresh": cfg.conf_thresh,
                         "unc_thresh": cfg.unc_thresh,
                         "mc_samples": cfg.mc_samples,
                         "use_mc_dropout": cfg.use_mc_dropout},
            conformal_active=False, conformal_coverage=None, conformal_size=None,
            per_pathology_conformal=None, brier=None,
            brier_decomposition=None, nll=None, ece_multi=None,
            reliability=None, confidence_auroc=None, aurc_by_score=None,
        )

    is_conf = a["is_confident"]
    wrong = a["wrong"]
    conf_wrong = is_conf & (wrong == 1)
    conf_correct = is_conf & (wrong == 0)

    # Core: among confident predictions, can epistemic signal detect errors?
    if is_conf.sum() > 0 and conf_wrong.sum() > 0 and conf_correct.sum() > 0:
        auc_std = _auroc(a["epistemic_std"][is_conf], wrong[is_conf])
        auc_mi = _auroc(a["mutual_info"][is_conf], wrong[is_conf])
        # Confidence-only baseline (uncertainty_evaluation.md §2.1): how well
        # does confidence alone rank errors within the confident set? The
        # uncertainty scores must beat this to justify their existence.
        auc_conf = _auroc(-a["confidence"][is_conf], wrong[is_conf])
        # Production flag (Phase 4 step 6): Mahalanobis feature-density OOD score
        # on frozen RAD-DINO [CLS] features. Only defined when the score column is
        # populated (production variant); 0.0/absent -> None (backward-compatible).
        maha = a.get("mahalanobis")
        if maha is not None and np.isfinite(maha).any() and not np.all(maha == 0.0):
            mm = is_conf & np.isfinite(maha)
            if (mm & (wrong == 1)).sum() > 0 and (mm & (wrong == 0)).sum() > 0:
                auc_maha = _auroc(maha[mm], wrong[mm])
            else:
                auc_maha = None
        else:
            auc_maha = None
        # recall of risk_flag over confident errors, false-flag over correct
        ce_recall = float(a["risk_flag"][conf_wrong].mean()) \
            if conf_wrong.sum() > 0 else None
        ff_rate = float(a["risk_flag"][conf_correct].mean()) \
            if conf_correct.sum() > 0 else None
        aurc, cov = _risk_coverage(a["epistemic_std"][is_conf], wrong[is_conf])
        # AURC + E-AURC for every available score (E-AURC removes the accuracy
        # confound so scores are comparable across models -- §1.3/§2.2).
        err_rate = float(wrong[is_conf].mean())
        aurc_by_score = {}
        for name, (cv, rk) in risk_coverage_curves(a, is_conf).items():
            a_ = _trapz(rk, cv)
            aurc_by_score[name] = {
                "aurc": round(a_, 4),
                "e_aurc": (round(_e_aurc(a_, err_rate), 4)
                           if _e_aurc(a_, err_rate) is not None else None),
                "error_rate": round(err_rate, 4),
            }
    else:
        auc_std = auc_mi = auc_maha = auc_conf = ce_recall = ff_rate = None
        aurc, cov = None, None
        aurc_by_score = None

    # --- proper scoring rules + calibration surface (always-on) --------------
    ece = _ece(a["p_bar"], a["gt"]) if len(a["p_bar"]) else 0.0
    nll = _nll(a["p_bar"], a["gt"])
    brier = float(np.mean((a["p_bar"] - a["gt"]) ** 2)) if len(a["p_bar"]) else None
    brier_dec = _brier_decompose(a["p_bar"], a["gt"])
    ece_multi = {nb: round(_ece(a["p_bar"], a["gt"], nb), 4)
                 for nb in [5, 10, 15, 20]} if len(a["p_bar"]) else None
    reliability = _reliability_diagram(a["p_bar"], a["gt"])

    # Per-pathology task AUROC + counts.
    per_path: Dict[str, dict] = {}
    pathos = np.array(rec.pathology)
    for p in NIH_PATHOLOGIES:
        m = pathos == p
        if m.sum() == 0:
            continue
        gt_p = a["gt"][m]
        if gt_p.sum() == 0 or gt_p.sum() == m.sum():
            task_auc = None
        else:
            task_auc = _auroc(a["p_bar"][m], gt_p)
        cw = (is_conf[m] & (wrong[m] == 1)).sum()
        cc = (is_conf[m] & (wrong[m] == 0)).sum()
        flagged = a["risk_flag"][m].sum()
        per_path[p] = {
            "n": int(m.sum()), "positives": int(gt_p.sum()),
            "task_auroc": task_auc,
            "confident_correct": int(cc), "confident_wrong": int(cw),
            "flagged": int(flagged),
        }

    err_counts = {k: 0 for k in ["confident_correct", "tentative_correct",
                                 "confident_fp", "tentative_fp",
                                 "confident_fn", "tentative_fn", "unknown"]}
    for e in rec.error_type:
        err_counts[e] = err_counts.get(e, 0) + 1

    # --- conformal triage metrics (Part 3 step 7): only when a conformal
    # calibrator attached LAC columns (conformal_active). Else all None ->
    # backward-compatible with youden/ts_only/... Per-pathology coverage/size/
    # refer over conformalized rows (lac_set_size >= 0); overall coverage/size
    # over the same; Brier = mean (p_bar - gt)^2.
    conf_active = bool(getattr(rec, "conformal_active", False))
    if conf_active:
        lac_ss = a["lac_set_size"]
        lac_cov = a["lac_covered"]
        conf_mask = lac_ss >= 0
        conf_cov = (float(lac_cov[conf_mask].mean())
                    if conf_mask.any() else None)
        conf_size = (float(lac_ss[conf_mask].mean())
                     if conf_mask.any() else None)
        per_path_conf: Optional[Dict[str, dict]] = {}
        for p in NIH_PATHOLOGIES:
            mp = pathos == p
            if mp.sum() == 0:
                continue
            mpc = mp & conf_mask
            if mpc.sum() == 0:
                continue
            per_path_conf[p] = {
                "n": int(mpc.sum()),
                "coverage": float(lac_cov[mpc].mean()),
                "avg_set_size": float(lac_ss[mpc].mean()),
                "refer_rate": float(a["lac_refer"][mpc].mean()),
            }
    else:
        conf_cov = conf_size = per_path_conf = None

    return EvalReport(
        n_images=n_imgs, n_records=len(rec.image_id),
        n_confident=int(is_conf.sum()),
        n_confident_wrong=int(conf_wrong.sum()),
        confident_error_auroc_std=auc_std,
        confident_error_auroc_mi=auc_mi,
        confident_error_auroc_mahalanobis=auc_maha,
        confident_error_recall=ce_recall,
        false_flag_rate=ff_rate,
        aurc=aurc,
        coverage_at_lowest_risk=cov,
        ece=ece,
        per_pathology=per_path,
        error_type_counts=err_counts,
        risk_config={
            "decision_thresh": cfg.decision_thresh,
            "conf_thresh": cfg.conf_thresh,
            "unc_thresh": cfg.unc_thresh,
            "mc_samples": cfg.mc_samples,
            "use_mc_dropout": cfg.use_mc_dropout,
        },
        conformal_active=conf_active,
        conformal_coverage=conf_cov,
        conformal_size=conf_size,
        per_pathology_conformal=per_path_conf,
        brier=brier,
        brier_decomposition=brier_dec,
        nll=nll,
        ece_multi=ece_multi,
        reliability=reliability,
        confidence_auroc=auc_conf,
        aurc_by_score=aurc_by_score,
    )


def rec_arrays_subset(a: Dict[str, np.ndarray], mask: np.ndarray,
                      key: str) -> np.ndarray:
    return a[key][mask]


def _risk_coverage_curve(score: np.ndarray, wrong: np.ndarray):
    """Risk-coverage curve: abstain on the highest-``score`` examples first.
    Returns ``(coverages, risks)`` sorted by coverage, or None if degenerate
    (empty, or no wrong / all wrong).

    Vectorized via suffix means (cumulative sum from the sorted tail): O(n log n)
    instead of the original O(n^2) Python loop -- critical because the bootstrap
    paired-difference CIs call this ~1000x."""
    if len(score) == 0 or wrong.sum() == 0 or wrong.sum() == len(wrong):
        return None
    n = len(wrong)
    order = np.argsort(-score)     # most uncertain first -> abstain from front
    w_sorted = wrong[order].astype(np.float64)
    total = w_sorted.sum()
    csum = np.concatenate(([0.0], np.cumsum(w_sorted)))   # csum[k] = sum of first k
    k = np.arange(0, n)            # abstain on the k most uncertain
    kept = n - k
    coverages = kept / n           # 1.0 .. 1/n
    risks = (total - csum[:n]) / kept   # mean wrong among the kept tail
    # ascending coverage (1/n .. 1.0) so the trapezoid integration is well-ordered.
    return coverages[::-1], risks[::-1]


def _risk_coverage(score: np.ndarray, wrong: np.ndarray) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """Selective-prediction risk-coverage by abstaining on highest-``score``.

    ``score`` is any uncertainty signal (higher = abstain first): epistemic_std,
    mutual_info, mahalanobis, knn_dist, or ``-confidence``. Returns
    (AURC, (coverage, error_rate) at the lowest-risk operating point).
    """
    curve = _risk_coverage_curve(score, wrong)
    if curve is None:
        return None, None
    cv, rk = curve
    aurc = _trapz(rk, cv)
    idx = int(np.argmin(rk))
    return aurc, (float(cv[idx]), float(rk[idx]))


def risk_coverage_curves(a: Dict[str, np.ndarray], is_conf: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Risk-coverage curves for every available uncertainty score on the
    confident population. Returns ``{score_name: (coverages, risks)}``. The
    ``confidence`` entry uses ``-confidence`` (abstain on the least confident),
    the confidence-only baseline. ``mahalanobis`` / ``knn`` are included only
    when their columns are populated (non-zero / finite)."""
    wrong = a["wrong"]
    wc = wrong[is_conf]
    scores = {
        "epistemic_std": a["epistemic_std"][is_conf],
        "mutual_info": a["mutual_info"][is_conf],
        "confidence": -a["confidence"][is_conf],
    }
    maha = a.get("mahalanobis")
    if maha is not None and np.isfinite(maha).any() and not np.all(maha == 0.0):
        scores["mahalanobis"] = maha[is_conf]
    knn = a.get("knn_dist")
    if knn is not None and np.isfinite(knn).any() and not np.all(knn == 0.0):
        scores["knn"] = knn[is_conf]
    out = {}
    for name, s in scores.items():
        curve = _risk_coverage_curve(s, wc)
        if curve is not None:
            out[name] = curve
    return out


def save_report(report: EvalReport, out_dir: str, records: Optional[Records] = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "evaluation.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    if records is not None:
        import csv
        with open(out / "per_record.csv", "w", newline="") as f:
            w = csv.writer(f)
            # schema_version 2: keep the legacy 11 cols first (old CSVs still
            # load), then append valid/logits/member_probs, then mahalanobis
            # (production UQ, Phase 4 step 6; 0.0 when absent), then the LAC
            # conformal columns (Part 3 step 7; defaults 1/0/1 when not
            # conformalized so the row count stays aligned).
            w.writerow(["image_id", "pathology", "p_bar", "decision", "confidence",
                        "epistemic_std", "mutual_info", "gt", "is_confident",
                        "risk_flag", "error_type", "valid", "logits",
                        "member_probs", "mahalanobis",
                        "lac_set_size", "lac_refer", "lac_covered",
                        "knn_dist"])
            for i in range(len(records.image_id)):
                w.writerow([records.image_id[i], records.pathology[i],
                            f"{records.p_bar[i]:.4f}", records.decision[i],
                            f"{records.confidence[i]:.4f}", f"{records.epistemic_std[i]:.4f}",
                            f"{records.mutual_info[i]:.4f}", records.gt[i],
                            int(records.is_confident[i]), int(records.risk_flag[i]),
                            records.error_type[i], records.valid[i],
                            f"{records.logits[i]:.4f}", records.member_probs[i],
                            f"{records.mahalanobis[i]:.4f}",
                            int(records.lac_set_size[i]), int(records.lac_refer[i]),
                            int(records.lac_covered[i]),
                            f"{records.knn_dist[i]:.4f}"])
    return out / "evaluation.json"