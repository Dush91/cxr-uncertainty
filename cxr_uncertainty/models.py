"""Model ensemble wrapper (Part 2 Phase 0 refactor).

Holds a list of pluggable ``Member`` objects (built by ``member_factory``) and
aligns their outputs to the shared NIH-14 target via ``Alignment``. This
replaces the legacy hard-coded xrv-DenseNet loop while preserving the public
surface used by the CLI: ``member_keys``, ``predictable``, ``has_dropout``,
``forward_ensemble`` (returns an ``(M, P)`` NaN-masked prob tensor) and
``forward_mc`` (returns a ``(K, P)`` NaN-masked tensor or None), so
``uncertainty.estimate`` works unchanged.

Numerics note: ``MemberOutput.logits`` are the TRUE raw logits (BCTS needs them).
``MemberOutput.probs`` is each member's pooled-prob value used for the
cross-member disagreement signal — for xrv members this is the legacy
``sigmoid(model(x))`` double-sigmoid+op_norm value (preserved because its
compression gives the disagreement signal good OOD dynamic range; the
plain-``sigmoid(logits)`` alternative regresses the selectivity curve); for
non-xrv members it is plain ``sigmoid(logits)``. BCTS in Part 3 works on
``logits`` (true) via a separate path, so the xrv double-sigmoid does not
corrupt calibration. See interfaces.MemberOutput. Phase 0 verified the
500-image OpenI selectivity curve matches the pre-refactor baseline exactly.
"""
from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import NIH_PATHOLOGIES, RiskConfig, DEFAULT_ENSEMBLE, ENSEMBLE_REGISTRY
from .interfaces import Member, Alignment, EnsembleOutput
from .member_factory import build_ensemble


def _to_native(x: torch.Tensor, native_size: int) -> torch.Tensor:
    """Resize the shared xrv tensor (B,1,H,W) to a member's native_size. A no-op
    when already at that size (so 224-only ensembles are byte-identical). Bilinear
    on the raw [-1024,1024] grayscale so each member's preprocess then
    normalizes from the correct resolution (Ark+ gets true 768, 224 members get
    768->224 = a downsample of the high-res original, ~their native 224)."""
    if x.shape[-1] == native_size and x.shape[-2] == native_size:
        return x
    return F.interpolate(x, size=(native_size, native_size),
                         mode="bilinear", align_corners=False)


class CXREnsemble:
    """A multi-architecture ensemble of ``Member`` objects."""

    def __init__(
        self,
        members: Optional[List[str]] = None,
        cfg: Optional[RiskConfig] = None,
        target_pathologies: Optional[List[str]] = None,
    ):
        self.cfg = cfg or RiskConfig()
        self.member_keys = list(members) if members is not None else list(DEFAULT_ENSEMBLE)
        # Validate keys against both the arch registry and the legacy xrv registry.
        for k in self.member_keys:
            if k not in ENSEMBLE_REGISTRY and k not in __import__(
                "cxr_uncertainty.config", fromlist=["ARCH_MEMBER_REGISTRY"]
            ).ARCH_MEMBER_REGISTRY:
                raise SystemExit(f"Unknown ensemble member '{k}'.")
        self.target_pathologies = target_pathologies or NIH_PATHOLOGIES
        self.members: List[Member] = build_ensemble(
            self.member_keys, cfg=self.cfg, target_pathologies=self.target_pathologies,
        )
        self.alignment = Alignment(self.members, self.target_pathologies)
        self.predictable = self.alignment.predictable

    @property
    def has_dropout(self) -> bool:
        return any(m.has_dropout for m in self.members)

    def _enable_dropout(self, enable: bool):
        for m in self.members:
            m.enable_dropout(enable)

    @torch.no_grad()
    def forward_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-member probabilities aligned to ``self.predictable``.

        Shape: ``(M, P)`` with NaN where a member does not define that class
        (same NaN-sentinel semantics as the legacy implementation, so
        ``uncertainty.estimate`` works unchanged). Handles batch ``B`` (legacy
        was batch-1 only).
        """
        outputs = [m.forward_batch(_to_native(x, m.native_size)) for m in self.members]
        probs, logits, feats, valid = self.alignment.stack(outputs)
        # (M,B,P) -> return the (M,P) view for B==1 (the cli/estimate path), but
        # keep the full batch when B>1 by returning (M,B,P) — estimate uses the
        # last dim as P and iterates rows, so we return (M,P) for B==1 to match
        # the legacy contract exactly.
        if probs.shape[1] == 1:
            return probs[:, 0, :]            # (M, P) NaN-masked
        return probs                         # (M, B, P)

    @torch.no_grad()
    def forward_ensemble_full(self, x: torch.Tensor) -> EnsembleOutput:
        """Full batched output (logits + features + valid) for the new
        Records/CSV schema (Part 3 calibration). Each member receives the shared
        tensor resized to its ``native_size`` (Ark+ 768 vs 224 members)."""
        outputs = [m.forward_batch(_to_native(x, m.native_size)) for m in self.members]
        probs, logits, feats, valid = self.alignment.stack(outputs)
        return EnsembleOutput(
            member_keys=self.member_keys, pathologies=self.predictable,
            per_member_probs=probs, per_member_logits=logits,
            per_member_features=feats, valid=valid,
        )

    @torch.no_grad()
    def forward_mc(self, x: torch.Tensor, samples: Optional[int] = None) -> Optional[torch.Tensor]:
        """MC-Dropout forward pooled across members. Returns
        ``(samples * num_members_with_dropout, P)`` probabilities, or None if no
        member has Dropout layers (in which case MC adds no signal)."""
        if not self.has_dropout:
            return None
        n = samples or self.cfg.mc_samples
        P = len(self.predictable)
        rows = []
        for mi, m in enumerate(self.members):
            has_d = m.has_dropout
            n_iter = n if has_d else 1
            m.enable_dropout(has_d)
            try:
                for _ in range(n_iter):
                    o = m.forward_batch(_to_native(x, m.native_size))
                    # pull the aligned (P,) row for member mi (B==1)
                    row = np.full(P, np.nan, dtype=np.float32)
                    for pi, tp in enumerate(self.predictable):
                        for (mmi, si) in self.alignment.slots[tp]:
                            if mmi == mi:
                                row[pi] = float(o.probs[0, si].item())
                                break
                    rows.append(row)
            finally:
                m.enable_dropout(False)
                m.eval()
        if not rows:
            return None
        return torch.from_numpy(np.asarray(rows))