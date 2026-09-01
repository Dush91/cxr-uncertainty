"""Member / ensemble factory (Part 2 Phase 0).

Dispatches a ``ModelSpec`` to the right ``Member`` subclass. Phase 0 only
implements the xrv DenseNet member (M3); ConvNeXt-V2 (M2) and the frozen
foundation members (M1 RAD-DINO) are added in Phase 1/2.
"""
from __future__ import annotations

from typing import List, Optional

from .config import ModelSpec, ARCH_MEMBER_REGISTRY, ENSEMBLE_REGISTRY, NIH_PATHOLOGIES, RiskConfig
from .interfaces import Member


def build_member(spec: ModelSpec, device: str = "cpu") -> Member:
    if spec.arch == "xrv_densenet":
        from .members.xrv_densenet import XrvDenseNetMember
        return XrvDenseNetMember(spec.key, spec.weights, device=device,
                                 prob_mode=spec.prob_mode)
    if spec.arch == "convnextv2":
        from .members.convnext_v2 import ConvNextV2Member
        return ConvNextV2Member(spec.key, hf_id=spec.hf_id, probe_ckpt=spec.probe_ckpt,
                                device=device)
    if spec.arch == "raddino":
        from .members.raddino import RadDinoMember
        return RadDinoMember(spec.key, hf_id=spec.hf_id, probe_ckpt=spec.probe_ckpt,
                             device=device)
    if spec.arch == "biomedclip":
        from .members.biomedclip import BiomedClipMember
        return BiomedClipMember(spec.key, hf_id=spec.hf_id, probe_ckpt=spec.probe_ckpt,
                                device=device)
    if spec.arch == "arkswin":
        from .members.ark_swin import ArkSwinMember
        return ArkSwinMember(spec.key, hf_id=spec.hf_id, probe_ckpt=spec.probe_ckpt,
                             ckpt_path=spec.ckpt_path, device=device)
    raise KeyError(f"unknown arch '{spec.arch}' for member '{spec.key}'")


def _resolve_spec(key: str) -> ModelSpec:
    if key in ARCH_MEMBER_REGISTRY:
        return ARCH_MEMBER_REGISTRY[key]
    if key in ENSEMBLE_REGISTRY:
        return ENSEMBLE_REGISTRY[key]
    raise KeyError(f"unknown member key '{key}'")


def build_ensemble(
    member_keys: List[str],
    cfg: Optional[RiskConfig] = None,
    target_pathologies: Optional[List[str]] = None,
) -> List[Member]:
    cfg = cfg or RiskConfig()
    target_pathologies = target_pathologies or NIH_PATHOLOGIES
    device = cfg.device
    members: List[Member] = []
    for k in member_keys:
        spec = _resolve_spec(k)
        members.append(build_member(spec, device=device))
    return members