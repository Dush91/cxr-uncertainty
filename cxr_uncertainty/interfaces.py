"""Pluggable seams for the CXR uncertainty ensemble (Part 2, Phase 0).

This module defines the contracts the rest of the package registers against, so
members / uncertainty estimators / calibrators / risk policies are no longer
called by name but resolved through registries. The existing module-level
functions (``uncertainty.estimate``, ``reanalyze.calibrate_thresholds``,
``risk.assess_image``) stay as thin default-dispatch shims over these
registries, so the CLI contract is preserved byte-for-byte.

The single most important change vs. the legacy code: a ``Member`` exposes
``MemberOutput`` carrying **logits AND features AND a validity mask**, not just
a NaN-encoded sigmoid tensor. The ``valid`` bool mask replaces the NaN-as-
sentinel encoding that forced the old batch-1 forward. Per-pathology name
alignment (the old ``models._align``) moves to the ``Alignment`` helper.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Member output / ensemble output
# ---------------------------------------------------------------------------
@dataclass
class MemberOutput:
    """Output of a single member's forward pass over a batch.

    ``logits`` are the TRUE raw logits (pre-sigmoid, pre-op-norm) — Beta/BCTS
    calibration in Part 3 operates on these. ``probs`` is the value the ensemble
    pools for the cross-member disagreement signal. For xrv members this is the
    **legacy operating-point-normalized + outer-sigmoid prob** (``sigmoid(model(x))``
    = ``sigmoid(op_norm(sigmoid(logits)))``), deliberately preserved because the
    compression gives the disagreement signal good dynamic range on OOD data —
    the confident-error AUROC selectivity curve (the defining metric) regresses
    badly under plain ``sigmoid(logits)``. Note: ``probs != sigmoid(logits)`` by
    design for xrv; BCTS in Part 3 works on ``logits`` via a separate path so the
    double-sigmoid does not corrupt calibration. ``features`` is the penultimate
    embedding (DEGRE gate / OOD). ``valid`` is the per-class mask: True where this
    member actually defines that class (replaces the old NaN sentinel).
    """
    logits: torch.Tensor    # (B, P_member) raw logits
    probs: torch.Tensor      # (B, P_member) pooled prob (legacy-preserving for xrv)
    features: torch.Tensor   # (B, D) penultimate embedding
    valid: torch.Tensor      # (B, P_member) bool
    member_key: str = ""

    def to(self, device):
        self.logits = self.logits.to(device)
        self.probs = self.probs.to(device)
        self.features = self.features.to(device)
        self.valid = self.valid.to(device)
        return self


@dataclass
class EnsembleOutput:
    """Per-member tensors stacked for the whole ensemble over a batch."""
    member_keys: List[str]
    pathologies: List[str]                       # NIH-14 target list (predictable)
    per_member_probs: torch.Tensor               # (M, B, P) aligned + NaN-masked
    per_member_logits: torch.Tensor              # (M, B, P) aligned + NaN-masked
    per_member_features: torch.Tensor           # (M, B, D)
    valid: torch.Tensor                          # (M, B, P) bool


# ---------------------------------------------------------------------------
# Member protocol
# ---------------------------------------------------------------------------
class Member(nn.Module, ABC):
    """A single ensemble member. Subclasses implement ``forward_batch``.

    ``target_pathologies`` is the member's raw output slot names (length P_member;
    '' for untrained slots in xrv). The Ensemble aligns these to the shared
    NIH-14 target via ``Alignment``. ``input_kind`` selects preprocessing.
    """
    key: str
    arch_family: str
    target_pathologies: List[str]
    input_kind: str          # "xrv" | "imagenet" | "raddino" | "clip"
    needs_logits: bool = True
    # Native input resolution (px). The shared loader fetches at
    # ``max(member native_size)``; ``CXREnsemble.forward_ensemble_full`` resizes
    # the shared tensor to each member's ``native_size`` before its forward, so a
    # 768 member (Ark+) and 224 members coexist in one ensemble without the 224
    # members changing behaviour (224->224 is a no-op). Default 224 = all members
    # except Ark+ today.
    native_size: int = 224

    @abstractmethod
    def forward_batch(self, x: torch.Tensor) -> MemberOutput:
        """Run the member on a batch. ``x`` is the xrv-format reference tensor
        (B,1,H,W) in [-1024,1024] already resized to this member's
        ``native_size`` by ``forward_ensemble_full``; each member re-normalizes
        to its own input_kind (see ``preprocess``)."""
        ...

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the shared xrv-format tensor to this member's input kind.
        Default (xrv): pass through unchanged. Subclasses override."""
        return x

    @property
    def has_dropout(self) -> bool:
        return any(isinstance(m, nn.Dropout) for m in self.modules())

    def enable_dropout(self, enable: bool) -> bool:
        """Toggle dropout layers. Returns whether any dropout was toggled."""
        n = 0
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(enable)
                n += 1
        return n > 0


# ---------------------------------------------------------------------------
# Alignment: member slot -> shared NIH-14 target
# ---------------------------------------------------------------------------
class Alignment:
    """Per shared-target pathology, the (member_idx, member_slot_idx) pairs that
    define it. Equivalent to the legacy ``CXREnsemble._align`` but operating on
    explicit slot names + the ``valid`` mask."""

    def __init__(self, members: List[Member], target_pathologies: List[str]):
        self.members = members
        self.target = target_pathologies
        self.slots: Dict[str, List[Tuple[int, int]]] = {}
        for tp in target_pathologies:
            pairs = []
            for mi, m in enumerate(members):
                for si, name in enumerate(m.target_pathologies):
                    if name == tp and name != "":
                        pairs.append((mi, si))
                        break
            self.slots[tp] = pairs
        self.predictable = [p for p in target_pathologies if self.slots[p]]

    def stack(self, outputs: List[MemberOutput]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (M,B,P) probs / logits / features / valid aligned to
        ``self.predictable``. Classes a member does not define are NaN in
        probs/logits and False in valid (preserving the legacy NaN semantics so
        ``uncertainty.estimate`` works unchanged)."""
        P = len(self.predictable)
        M = len(outputs)
        B = outputs[0].probs.shape[0]
        device = outputs[0].probs.device
        probs = torch.full((M, B, P), float("nan"), dtype=torch.float32, device=device)
        logits = torch.full((M, B, P), float("nan"), dtype=torch.float32, device=device)
        valid = torch.zeros((M, B, P), dtype=torch.bool, device=device)
        D = max(o.features.shape[1] for o in outputs)
        feats = torch.zeros((M, B, D), dtype=torch.float32, device=device)
        for mi, o in enumerate(outputs):
            for pi, tp in enumerate(self.predictable):
                for (mmi, si) in self.slots[tp]:
                    if mmi == mi:
                        probs[mi, :, pi] = o.probs[:, si]
                        logits[mi, :, pi] = o.logits[:, si]
                        valid[mi, :, pi] = o.valid[:, si] & True
                        break
            feats[mi, :, : o.features.shape[1]] = o.features
        return probs, logits, feats, valid


# ---------------------------------------------------------------------------
# Registries (UQ estimator / calibrator / risk policy)
# ---------------------------------------------------------------------------
UQ_ESTIMATORS: Dict[str, Callable] = {}
CALIBRATORS: Dict[str, Callable] = {}
RISK_POLICIES: Dict[str, Callable] = {}


def register_uq_estimator(key: str):
    def deco(fn):
        UQ_ESTIMATORS[key] = fn
        return fn
    return deco


def register_calibrator(key: str):
    def deco(fn):
        CALIBRATORS[key] = fn
        return fn
    return deco


def register_risk_policy(key: str):
    def deco(fn):
        RISK_POLICIES[key] = fn
        return fn
    return deco


def get_uq_estimator(key: str) -> Callable:
    if key not in UQ_ESTIMATORS:
        raise KeyError(f"unknown UQ estimator '{key}'; choices: {list(UQ_ESTIMATORS)}")
    return UQ_ESTIMATORS[key]


def get_calibrator(key: str) -> Callable:
    if key not in CALIBRATORS:
        raise KeyError(f"unknown calibrator '{key}'; choices: {list(CALIBRATORS)}")
    return CALIBRATORS[key]


def get_risk_policy(key: str) -> Callable:
    if key not in RISK_POLICIES:
        raise KeyError(f"unknown risk policy '{key}'; choices: {list(RISK_POLICIES)}")
    return RISK_POLICIES[key]