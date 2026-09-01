"""M1: RAD-DINO (microsoft/rad-dino) as a ``Member`` (Part 2 Phase 2).

RAD-DINO is a DINOv2 ViT-B/14 (86.6M params) self-supervised-pretrained on 882K
CXRs (MIMIC/CheXpert/NIH/PadChest/BRAX). The encoder is **frozen** + a linear
probe head — fine-tuning would distort the SSL features (~7% OOD, Kumar 2022;
RAD-DINO card: FT "typically not necessary") and a single frozen ViT member in
a multi-arch ensemble preserves architecture diversity (the "frozen kills
diversity" rule only bites when MULTIPLE members share one frozen encoder — we
have one each, §C/§H.1). This member is the architecture-diversity lever that
decorrelates confident errors the 2-CNN Phase-1 ensemble couldn't.

Input kind = ``raddino``: xrv-format (B,1,224,224)~[-1024,1024] -> RAD-DINO's
BitImageProcessor normalization (mean/std = 0.5307/0.2583, grayscale-replicated
to 3-channel), kept at 224 (DINOv2 takes variable resolution; 224 aligns with
the rest of the ensemble and is fast). Features = the [CLS] token
(``last_hidden_state[:,0]``, 768-d) — the canonical DINOv2 representation for
linear probing. ``probs = sigmoid(logits)`` (plain, non-xrv).
"""
from __future__ import annotations

import warnings

import torch

from .base import LinearProbeMember

# RAD-DINO BitImageProcessor normalization (CXR-tuned, grayscale-ish).
_RADDINO_MEAN = (0.5307, 0.5307, 0.5307)
_RADDINO_STD = (0.2583, 0.2583, 0.2583)


class RadDinoMember(LinearProbeMember):
    arch_family = "raddino_vit"
    input_kind = "raddino"

    def __init__(self, key: str, hf_id: str = "microsoft/rad-dino",
                 probe_ckpt: str = "", device: str = "cpu"):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install transformers") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.encoder = AutoModel.from_pretrained(hf_id)
        self.encoder.eval()
        # HF Dinov2Model has no .num_features; expose 768 (the CLS/hidden size)
        # so the shared LinearProbeMember._setup builds the right head dim.
        D = self.encoder.config.hidden_size
        self.encoder.num_features = D
        self._mean = torch.tensor(_RADDINO_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_RADDINO_STD).view(1, 3, 1, 1)
        self._setup(key, device=device, load_ckpt=probe_ckpt)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """xrv (B,1,224,224) in ~[-1024,1024] -> RAD-DINO-norm (B,3,224,224)."""
        z = (x.float() + 1024.0) / 2048.0
        z = z.clamp(0.0, 1.0)
        z = z.repeat(1, 3, 1, 1)
        z = (z - self._mean) / self._std
        return z

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        """[CLS] token of the DINOv2 last hidden state -> (B, 768)."""
        out = self.encoder(z)
        return out.last_hidden_state[:, 0]