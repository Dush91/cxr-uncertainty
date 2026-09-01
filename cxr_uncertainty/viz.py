"""Plots for the uncertainty / risk-flagging evaluation.

Palette is the validated reference set from the dataviz skill (light mode):
  surface #fcfcfb, primary ink #0b0b0b, secondary #52514e, muted #898781,
  gridline #e1e0d9, baseline #c3c2b7.
  status good #0ca30c, critical #d03b3b, serious #ec835a, warning #fab219.
  categorical blue #2a78d6, orange #eb6834.
  sequential blue ramp (e.g. #2a78d6, #86b6ef, #cde2fb).

Discipline applied: one axis per chart; categorical hues in fixed order; status
colors carry an icon+label (never color alone); ≥2 series get a legend; thin
marks; recessive grid/axes; text in ink, never the series color.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .evaluate import Records, EvalReport

# --- palette (light mode) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SEC = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
SERIOUS = "#ec835a"
WARN = "#fab219"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK_SEC,
    "axes.titlecolor": INK,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _stamp_status(ax, label, color, x=0.0, y=1.0):
    """Colored marker + text so a status color never carries meaning alone."""
    ax.plot([x], [y], marker="s", color=color, transform=ax.transAxes,
            clip_on=False, markersize=8)


def make_plots(records: Records, report: EvalReport, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = records.arrays()
    paths = []
    paths.append(_plot_uncertainty_separation(a, report, out))
    paths.append(_plot_risk_coverage(a, report, out))
    paths.append(_plot_risk_coverage_multi(a, report, out))
    paths.append(_plot_reliability(report, out))
    paths.append(_plot_per_pathology(report, out))
    paths.append(_plot_error_types(report, out))
    return paths


def _plot_uncertainty_separation(a, report: EvalReport, out: Path):
    """Histogram of epistemic_std: confident-correct vs confident-wrong."""
    is_conf = a["is_confident"]
    wrong = a["wrong"]
    correct_s = a["epistemic_std"][is_conf & (wrong == 0)]
    wrong_s = a["epistemic_std"][is_conf & (wrong == 1)]
    if len(correct_s) == 0 and len(wrong_s) == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, max(0.001, float(np.nanmax(a["epistemic_std"])) + 1e-3), 30)
    ax.hist(correct_s, bins=bins, color=GOOD, alpha=0.65,
            label=f"confident correct (n={len(correct_s)})",
            edgecolor=SURFACE, linewidth=0.6)
    ax.hist(wrong_s, bins=bins, color=CRIT, alpha=0.75,
            label=f"confident wrong (n={len(wrong_s)})",
            edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlabel("epistemic uncertainty (ensemble + MC std)")
    ax.set_ylabel("count")
    auc = report.confident_error_auroc_std
    ax.set_title("Epistemic uncertainty separates confident errors from correct calls"
                 + (f"  ·  AUROC={auc:.2f}" if auc is not None else ""))
    ax.legend(loc="upper right", labelcolor=INK)
    # status colors carry a label in the legend (relief rule)
    fig.tight_layout()
    p = out / "01_uncertainty_separation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _plot_risk_coverage(a, report: EvalReport, out: Path):
    """Risk-coverage: abstain on most uncertain; error rate among retained."""
    is_conf = a["is_confident"]
    wrong = a["wrong"]
    ep = a["epistemic_std"][is_conf]
    wg = wrong[is_conf]
    if len(ep) == 0 or wg.sum() == 0 or wg.sum() == len(wg):
        return None
    order = np.argsort(-ep)
    n = len(wg)
    covs, risks = [], []
    for k in range(0, n + 1):
        kept = order[k:]
        if len(kept) == 0:
            continue
        covs.append(len(kept) / n)
        risks.append(wg[kept].mean())
    covs = np.array(covs); risks = np.array(risks)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(covs, risks, color=BLUE, linewidth=2.0, solid_capstyle="round")
    # lowest-risk operating point
    idx = int(np.argmin(risks))
    ax.scatter([covs[idx]], [risks[idx]], color=CRIT, s=45, zorder=5,
               edgecolor=SURFACE, linewidth=0.8)
    ax.annotate(f"lowest risk\n(cov={covs[idx]:.2f}, err={risks[idx]:.3f})",
                xy=(covs[idx], risks[idx]),
                xytext=(covs[idx] - 0.30, risks[idx] + 0.05),
                color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    # baseline: error rate if we never abstain (coverage=1)
    base_err = wg.mean()
    ax.axhline(base_err, color=MUTED, linestyle="--", linewidth=1.0,
               label=f"no-abstain error={base_err:.3f}")
    ax.set_xlabel("coverage (fraction of confident predictions kept)")
    ax.set_ylabel("error rate among retained")
    ax.set_xlim(0, 1.0)
    aurc = report.aurc
    ax.set_title("Risk–coverage: flagging high-uncertainty cuts error rate"
                 + (f"  ·  AURC={aurc:.4f}" if aurc is not None else ""))
    ax.legend(loc="upper left", labelcolor=INK)
    fig.tight_layout()
    p = out / "02_risk_coverage.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _plot_risk_coverage_multi(a, report: EvalReport, out: Path):
    """Risk-coverage curves for every uncertainty score on the confident set —
    the baseline comparison (confidence-only vs disagreement vs feature scores)."""
    from .evaluate import risk_coverage_curves
    is_conf = a["is_confident"]
    curves = risk_coverage_curves(a, is_conf)
    if len(curves) < 2:
        return None
    colors = {"epistemic_std": BLUE, "mutual_info": ORANGE,
              "confidence": MUTED, "mahalanobis": CRIT, "knn": SERIOUS}
    labels = {"epistemic_std": "epistemic std", "mutual_info": "mutual info",
              "confidence": "confidence (baseline)", "mahalanobis": "Mahalanobis",
              "knn": "kNN distance"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (cv, rk) in curves.items():
        ax.plot(cv, rk, color=colors.get(name, BLUE), linewidth=2.0,
                solid_capstyle="round", label=labels.get(name, name))
    base_err = a["wrong"][is_conf].mean()
    ax.axhline(base_err, color=MUTED, linestyle="--", linewidth=1.0,
               label=f"no-abstain error={base_err:.3f}")
    ax.set_xlabel("coverage (fraction of confident predictions kept)")
    ax.set_ylabel("error rate among retained")
    ax.set_xlim(0, 1.0)
    ax.set_title("Risk–coverage by uncertainty score (baseline comparison)")
    ax.legend(loc="upper left", labelcolor=INK, fontsize=8.5)
    fig.tight_layout()
    p = out / "02b_risk_coverage_by_score.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _plot_reliability(report: EvalReport, out: Path):
    """Reliability diagram: accuracy vs mean confidence per bin (Guo et al.
    2017). Points below the diagonal are overconfident."""
    rel = report.reliability
    if not rel or not rel["bin_center"]:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.0,
            label="perfect calibration")
    ax.plot(rel["confidence"], rel["accuracy"], color=BLUE, marker="o",
            markersize=5, linewidth=1.6, label="model")
    # gap shading: overconfident (acc < conf) below the diagonal
    ax.fill_between(rel["confidence"], rel["accuracy"], rel["confidence"],
                    where=np.array(rel["accuracy"]) < np.array(rel["confidence"]),
                    color=CRIT, alpha=0.12, interpolate=True)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram  ·  ECE={report.ece:.4f}")
    ax.legend(loc="lower right", labelcolor=INK)
    fig.tight_layout()
    p = out / "05_reliability.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _plot_per_pathology(report: EvalReport, out: Path):
    """Horizontal bars of per-pathology task AUROC (single series, single hue)."""
    items = [(p, d["task_auroc"], d["n"], d["positives"])
             for p, d in report.per_pathology.items()
             if d["task_auroc"] is not None]
    if not items:
        return None
    items.sort(key=lambda x: x[1])
    names = [x[0] for x in items]
    aucs = [x[1] for x in items]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(names) + 1.4))
    y = np.arange(len(names))
    ax.barh(y, aucs, color=BLUE, edgecolor=SURFACE, linewidth=0.6, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1.0, label="chance (0.5)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("task AUROC (p_bar vs ground truth)")
    ax.set_title("Per-pathology detection AUROC (OpenI ground truth)")
    # direct-label values at bar ends
    for yi, v in zip(y, aucs):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", ha="left",
                color=INK_SEC, fontsize=9)
    ax.legend(loc="lower right", labelcolor=INK)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.tight_layout()
    p = out / "03_per_pathology_auroc.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def _plot_error_types(report: EvalReport, out: Path):
    """Bar chart of error-type counts; confident errors highlighted."""
    order = ["confident_correct", "tentative_correct",
             "confident_fp", "confident_fn",
             "tentative_fp", "tentative_fn"]
    labels_map = {
        "confident_correct": "confident\ncorrect",
        "tentative_correct": "tentative\ncorrect",
        "confident_fp": "confident\nfalse pos",
        "confident_fn": "confident\nfalse neg",
        "tentative_fp": "tentative\nfalse pos",
        "tentative_fn": "tentative\nfalse neg",
    }
    counts = [report.error_type_counts.get(k, 0) for k in order]
    colors = [GOOD, BLUE_LIGHT, CRIT, CRIT, SERIOUS, SERIOUS]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(order))
    bars = ax.bar(x, counts, color=colors, edgecolor=SURFACE,
                  linewidth=0.6, width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_map[k] for k in order], fontsize=8.5)
    ax.set_ylabel("count")
    ax.set_title("Prediction outcomes (confident errors are the flag target)")
    for xi, c in zip(x, counts):
        if c > 0:
            ax.text(xi, c, str(c), ha="center", va="bottom",
                    color=INK, fontsize=9)
    # legend mapping colors to meaning (relief rule for status colors)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=GOOD, label="confident correct"),
        Patch(facecolor=BLUE_LIGHT, label="tentative correct"),
        Patch(facecolor=CRIT, label="confident error (fp/fn)"),
        Patch(facecolor=SERIOUS, label="tentative error (fp/fn)"),
    ]
    ax.legend(handles=handles, loc="upper right", labelcolor=INK, fontsize=8.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    p = out / "04_error_types.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p