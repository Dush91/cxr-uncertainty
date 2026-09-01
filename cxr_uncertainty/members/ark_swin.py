"""M4: Ark+6 (ASU + Mayo, Swin-Large) as a ``Member`` (Part 2 Phase 3, 4th member).

Ark+ is a CXR foundation model: a **Swin Transformer** trained with **cyclic
supervised pretraining** that accrues knowledge from many differently-labeled
public datasets *without label consolidation* (Ma et al., Nature 2025). This
checkpoint is **Ark+6** — Swin-**Large**, pretrained on ~704K CXRs (CheXpert,
ChestX-ray14, RSNA, VinDr-CXR, Shenzhen, MIMIC-CXR), at **768x768** input. It is
the **3rd representation family** in the ensemble (supervised multi-dataset
Swin), distinct from RAD-DINO (vision-SSL ViT), ConvNeXt-V2 (ImageNet CNN), xrv
DenseNet (supervised CNN) and BiomedCLIP (language-contrastive CLIP) — the M=5
diversity goal (plan §C/§F.3).

**Leak status:** OpenI is leak-free (not in the Ark+ corpus). The OOD probes are
leak-free *even with Ark+ in* (Kermany/COVID/Montgomery/VinDr-PCXR are in no
Ark+ corpus) — so we keep Ark+ AND get multi-site leak-free OOD reporting. Only
CheXpert/NIH/MIMIC/VinDr-CXR/Shenzhen/RSNA are leaked (in the Ark+ corpus), but
those are role-A/B training/cal sources, not the eval target (plan §I.2).

**Checkpoint format (inspected 2026-08-03 from ``Ark6_swinLarge768_ep50.pth.tar``):**
  * a training dict ``{state_dict, epoch, lossMIN, optimizer, scheduler}``; the
    state_dict keys are prefixed ``module.`` (DDP wrapper -> stripped) and the
    encoder is a **Swin-L** (patch 4, embed 192/384/768/1536, depths [2,2,18,2],
    window 12, heads [6,12,24,48]) -> ``timm``'s ``swin_large_patch4_window12_384``
    architecture, run at ``img_size=768`` (Swin is window-based and handles
    dynamic input size; 768 is divisible by 384 = window*2^(stages-1), so the
    stage token maps stay divisible by window 12).
  * **Key remap:** Ark+ names each patch-merge ``layers.{i}.downsample`` (after
    stage i); timm names it ``layers.{i+1}.downsample`` (before stage i+1). The
    weights are identical shapes — only the key path is off by one. We remap
    ``layers.{i}.downsample.*`` -> ``layers.{i+1}.downsample.*`` on load; the
    stage blocks / patch_embed / final norm align directly. The trained 14-class
    ``head`` (14,1536) and the ``relative_position_index``/``attn_mask`` buffers
    are dropped (num_classes=0; timm recomputes the buffers dynamically).
  * Feature dim **1536** (Swin-L final stage); 195.2M params.

**Input kind = ``imagenet``** (ImageNet normalization — Ark+ is cyclically
fine-tuned from an ImageNet-pretrained Swin init). ``native_size = 768`` — the
ensemble's shared loader is set to ``max(member native_size)`` so Ark+ receives
true 768 input; ``forward_ensemble_full`` resizes the shared tensor to each
member's ``native_size`` before its forward (so 224 members still get 224).
Frozen + linear probe (nn.Linear(1536,14)); ``probs = sigmoid(logits)`` (plain).
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import torch

from .base import LinearProbeMember

# ImageNet normalization (Ark+ is cyclically fine-tuned from an ImageNet-pretrained
# Swin-L init; the repo uses standard ImageNet stats on 3-channel CXR).
_ARK_MEAN = (0.485, 0.456, 0.406)
_ARK_STD = (0.229, 0.224, 0.225)

# Ark+ ships a trained 14-class head on the 1536-d features; we drop it and
# learn our own NIH-14 linear probe on the frozen encoder (the Ark+ head was
# trained on the cyclic multi-dataset labels, not our role-A union; a fresh
# probe on role A matches the other members' calibration target).
_ARK_TIMM_NAME = "swin_large_patch4_window12_384"
_ARK_NATIVE = 768


def _remap_ark_keys(state_dict):
    """Strip the ``module.`` DDP prefix and shift the downsample key path by one
    stage to match timm's Swin naming (see module docstring)."""
    sd = {k[len("module."):]: v for k, v in state_dict.items()
          if k.startswith("module.")}
    def shift(k):
        mt = re.match(r"^layers\.(\d+)\.downsample\.(.*)$", k)
        return f"layers.{int(mt.group(1)) + 1}.downsample.{mt.group(2)}" if mt else k
    return {shift(k): v for k, v in sd.items()}


class ArkSwinMember(LinearProbeMember):
    arch_family = "arkswin"
    input_kind = "imagenet"
    native_size = _ARK_NATIVE

    def __init__(self, key: str, hf_id: str = "", probe_ckpt: str = "",
                 ckpt_path: str = "", device: str = "cpu"):
        """Args:
            hf_id: unused for Ark+ (kept for interface symmetry); the encoder
                weights are a local file -> pass its path via ``ckpt_path``.
            ckpt_path: path to the uploaded Ark+6 encoder checkpoint
                (``Ark6_swinLarge768_ep50.pth.tar``).
        """
        super().__init__()
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Ark+ checkpoint not found: {ckpt_path!r}. The user uploads the "
                "Ark+6 weights locally (not on HF) -> set ModelSpec.ckpt_path.")
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install timm") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.encoder = timm.create_model(
                _ARK_TIMM_NAME, pretrained=False, num_classes=0,
                img_size=_ARK_NATIVE)
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        self.encoder.load_state_dict(_remap_ark_keys(sd), strict=False)
        self.encoder.eval()
        # timm Swin exposes .num_features = 1536; ensure it is set for _setup.
        if not getattr(self.encoder, "num_features", None):
            self.encoder.num_features = 1536
        self._mean = torch.tensor(_ARK_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_ARK_STD).view(1, 3, 1, 1)
        self._setup(key, device=device, load_ckpt=probe_ckpt)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """xrv (B,1,H,W) in ~[-1024,1024] -> ImageNet-norm (B,3,768,768).

        ``forward_ensemble_full`` has already resized the shared tensor to this
        member's ``native_size`` (768), so ``x`` arrives at 768 here. The
        [-1024,1024]->[0,1] + 3-channel + ImageNet-norm path mirrors ConvNeXt-V2.
        """
        z = (x.float() + 1024.0) / 2048.0
        z = z.clamp(0.0, 1.0)
        z = z.repeat(1, 3, 1, 1)              # grayscale -> 3-channel
        z = (z - self._mean) / self._std
        return z

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        """Swin-L pooled embedding -> (B, 1536)."""
        return self.encoder(z)