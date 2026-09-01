"""Uncertainty & high-risk flagging system for chest X-ray (CXR) classification.

The system runs publicly available pretrained CXR models, estimates epistemic
uncertainty (model ensemble disagreement + MC-Dropout), and flags *confident
errors* -- cases where the model is highly confident yet wrong (confident false
positives) or confidently misses a real pathology (confident false negatives).

See README.md for the full description and usage.
"""

__version__ = "0.1.0"

# Importing the registering modules populates the UQ/Calibrator/RiskPolicy
# registries (interfaces.py) via their @register_* decorators, so any
# `import cxr_uncertainty.*` resolves the default keys (ensemble_mc / youden /
# threshold) without callers having to import every submodule by hand.
try:  # guarded so partial environments (no torchxrayvision) still import the package
    from . import uncertainty as _uncertainty  # noqa: F401  -> register "ensemble_mc"
    from . import risk as _risk                  # noqa: F401  -> register "threshold"
    from . import reanalyze as _reanalyze        # noqa: F401  -> register "youden"
    from . import calibration as _calibration    # noqa: F401  -> register "bcts_ets_mondrian"
    from . import production as _production      # noqa: F401  -> register "ts_only", "ts_only_mahalanobis"
    from . import conformal as _conformal        # noqa: F401  -> register "conformal_triage"
except Exception:  # pragma: no cover
    pass