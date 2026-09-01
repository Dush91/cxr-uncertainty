"""M2: ConvNeXt-V2-Large (ImageNet FCMAE) as a ``Member`` (Part 2 Phase 1).

The only GPU-trained member: the ImageNet-pretrained encoder never saw any CXR,
so fine-tuning it on the NIH/CheXpert union (role A) is a genuine per-dataset
specialization — a real new loss basin, not the same-basin shallow diversity of
fine-tuning a CXR-foundation checkpoint (§H.1). Training is LP-FT (linear probe
then unfreeze fine-tune) in ``scripts/train_convnextv2_lpft.py``; this module is
the inference wrapper that loads the resulting checkpoint.

Input kind = ``imagenet``: the shared pipeline hands us the xrv-format tensor
(B,1,224,224) in ~[-1024,1024]; we map it to ImageNet-normalized 3-channel
[0,1]→(x-mean)/std. ``probs = sigmoid(logits)`` (plain — the xrv double-sigmoid
is not applicable to a non-xrv model).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch

from .base import LinearProbeMember

# ImageNet normalization (ConvNeXt-V2 fcmae_ft_in1k was fine-tuned on ImageNet-1k).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvNextV2Member(LinearProbeMember):
    arch_family = "convnext_v2"
    input_kind = "imagenet"

    def __init__(self, key: str, hf_id: str = "timm/convnextv2_large.fcmae_ft_in1k",
                 probe_ckpt: str = "", device: str = "cpu"):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install timm") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.encoder = timm.create_model(hf_id, pretrained=True, num_classes=0)
        self.encoder.eval()
        self._mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self._setup(key, device=device, load_ckpt=probe_ckpt)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """xrv (B,1,224,224) in ~[-1024,1024] -> ImageNet-norm (B,3,224,224)."""
        # [-1024,1024] -> [0,1] (clip; xrv normalize already clips but be safe).
        z = (x.float() + 1024.0) / 2048.0
        z = z.clamp(0.0, 1.0)
        z = z.repeat(1, 3, 1, 1)            # grayscale -> 3-channel
        z = (z - self._mean) / self._std
        return z