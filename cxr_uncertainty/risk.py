"""Risk flagging.

A prediction is flagged HIGH RISK when it is *confident yet epistemically
uncertain* -- exactly the regime where confident wrong detections and confident
misses live. Softmax confidence alone cannot separate confident errors from
confident correct calls; epistemic uncertainty (ensemble/MC disagreement) can.

Per (image, pathology) we produce:
  * decision      : predicted present (p_bar >= decision_thresh) / absent
  * confidence    : |p_bar - 0.5| * 2
  * epistemic     : mutual_info (nats) and epistemic_std (prob units)
  * is_confident  : confidence >= conf_thresh
  * risk_flag     : is_confident AND epistemic >= unc_thresh
  * error_type    : vs ground truth -> "correct" | "confident_fp" | "confident_fn"
                    | "tentative_correct" | "tentative_fp" | "tentative_fn"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import RiskConfig
from .interfaces import register_risk_policy
from .uncertainty import UncertaintyResult


@dataclass
class PathologyRisk:
    pathology: str
    p_bar: float
    decision: int           # 1 predicted present, 0 absent
    confidence: float       # [0,1]
    epistemic_std: float
    mutual_info: float
    is_confident: bool
    risk_flag: bool         # high-risk (confident & epistemic-uncertain)
    gt: Optional[int]       # ground-truth label if known
    error_type: Optional[str]
    # LAC conformal triage fields (plan §M.2 / §L). Defaults keep the ``threshold``
    # policy byte-identical (assess_image leaves them at defaults; the conformal
    # policy populates them). lac_set_size: 0=empty/refer, 1=auto-read single
    # label, 2=ambiguous/refer, -1=not conformalized (excluded pathology).
    lac_set_size: int = 1
    lac_refer: bool = False
    lac_covered: bool = True


@dataclass
class ImageRiskReport:
    image_id: str
    pathologies: List[PathologyRisk]
    any_high_risk: bool
    high_risk_pathologies: List[str]

    def to_summary(self) -> dict:
        return {
            "image_id": self.image_id,
            "any_high_risk": self.any_high_risk,
            "high_risk_pathologies": self.high_risk_pathologies,
            "n_flagged": len(self.high_risk_pathologies),
        }


def classify_error(decision: int, confidence: float, gt: Optional[int],
                   conf_thresh: float) -> str:
    """Label a single (image,pathology) prediction against ground truth."""
    if gt is None:
        return "unknown"
    correct = (decision == gt)
    confident = confidence >= conf_thresh
    if correct:
        return "confident_correct" if confident else "tentative_correct"
    # wrong
    if decision == 1 and gt == 0:
        return "confident_fp" if confident else "tentative_fp"
    if decision == 0 and gt == 1:
        return "confident_fn" if confident else "tentative_fn"
    return "tentative_wrong"


@register_risk_policy("threshold")
def assess_image(
    image_id: str,
    unc: UncertaintyResult,
    cfg: RiskConfig,
    gt_labels: Optional[dict] = None,
) -> ImageRiskReport:
    """Build the per-image risk report. gt_labels maps pathology->0/1.

    Registered as the ``"threshold"`` risk policy (default): flag = is_confident
    AND epistemic_std >= unc_thresh. The conformal-triage policy is added in
    Part 3 phase 3.
    """
    gt_labels = gt_labels or {}
    rows = []
    flagged = []
    for i, p in enumerate(unc.pathologies):
        pbar = float(unc.p_bar[i])
        if np.isnan(pbar):
            continue  # no model covered this class
        decision = int(pbar >= cfg.decision_thresh)
        conf = float(unc.confidence[i])
        estd = float(unc.epistemic_std[i])
        mi = float(unc.mutual_info[i])
        is_conf = conf >= cfg.conf_thresh
        # Epistemic gate: use mutual_info (nats) OR std; we gate on std in
        # probability units for interpretability, with a separate MI report.
        epistemic_high = estd >= cfg.unc_thresh
        risk = is_conf and epistemic_high
        gt = gt_labels.get(p)
        err = classify_error(decision, conf, gt, cfg.conf_thresh)
        if risk:
            flagged.append(p)
        rows.append(PathologyRisk(
            pathology=p, p_bar=pbar, decision=decision, confidence=conf,
            epistemic_std=estd, mutual_info=mi, is_confident=is_conf,
            risk_flag=risk, gt=gt, error_type=err,
        ))
    return ImageRiskReport(
        image_id=image_id, pathologies=rows,
        any_high_risk=bool(flagged), high_risk_pathologies=flagged,
    )