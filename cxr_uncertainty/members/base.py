"""Shared base for the frozen-encoder + linear-probe members (Part 2, §J.2).

Holds an ``encoder`` (returns a (B, D) pooled embedding) + an ``nn.Linear(D, P)``
head, frozen by default, and implements ``forward_batch`` returning a
``MemberOutput``. Subclasses (ConvNeXt-V2, RAD-DINO, ...) only override
``preprocess`` (input-kind normalization) and the encoder construction; the
LP-FT member (ConvNeXt-V2) additionally unfreezes the encoder for the FT phase
inside its training script (this module stays frozen by default for inference).

Non-xrv members use plain ``sigmoid(logits)`` for ``probs`` (the legacy
double-sigmoid + op_norm is an xrv-only property — see ``xrv_densenet.py``).
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from ..config import NIH_PATHOLOGIES
from ..interfaces import Member, MemberOutput


class LinearProbeMember(Member):
    """Encoder + linear head, frozen by default. ``target_pathologies`` is the
    full NIH-14 list (every class is a valid output slot, so ``valid`` is all-True
    for every position). Subclasses set ``arch_family`` / ``input_kind`` / build
    ``self.encoder`` + ``self.head`` then call ``super().__init__``-free setup
    via :meth:`_setup`."""

    arch_family: str = "probe"
    input_kind: str = "imagenet"
    native_size: int = 224

    def _setup(self, key: str, device: str = "cpu", load_ckpt: str = ""):
        """Call after building ``self.encoder`` (with ``.num_features``) and
        before freezing. Loads a probe/LP-FT checkpoint if present."""
        self.key = key
        D = self.encoder.num_features
        self.head = nn.Linear(D, len(NIH_PATHOLOGIES)).to(device)
        self.target_pathologies: List[str] = list(NIH_PATHOLOGIES)
        # valid = all True (every NIH-14 slot is a real output of the probe head).
        self._slot_valid = torch.ones(len(NIH_PATHOLOGIES), dtype=torch.bool,
                                      device=device)
        self._device = device
        self.encoder.to(device)
        if load_ckpt and Path(load_ckpt).exists():
            self.load_state_dict(torch.load(load_ckpt, map_location=device))
        self.freeze_encoder(True)

    # ---- training hooks (used by scripts/train_*.py) ------------------------
    def freeze_encoder(self, freeze: bool):
        requires = not freeze
        for p in self.encoder.parameters():
            p.requires_grad_(requires)
        self.encoder.train(not freeze) if False else None  # keep eval mode
        self.encoder.eval() if freeze else self.encoder.train()

    def trainable_parameters(self):
        """Params that should receive gradients (head always; encoder only when
        unfrozen)."""
        out = list(self.head.parameters())
        for p in self.encoder.parameters():
            if p.requires_grad:
                out.append(p)
        return out

    # ---- inference ----------------------------------------------------------
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses implement input-kind normalization")

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        """Embedding (B, D) used by the head. Default = encoder forward (ConvNeXt
        returns a (B,D) tensor directly). Foundation models that return a
        transformer output (RAD-DINO ``BaseModelOutput``) override this to pull
        the CLS token / pooled embedding."""
        return self.encoder(z)

    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor) -> MemberOutput:
        z = self.preprocess(x).to(self._device)
        feats = self.forward_features(z)        # (B, D)
        logits = self.head(feats)                # (B, 14) raw logits
        probs = torch.sigmoid(logits)            # plain sigmoid (non-xrv member)
        B = logits.shape[0]
        valid = self._slot_valid.unsqueeze(0).expand(B, -1).clone()
        return MemberOutput(
            logits=logits.float(), probs=probs.float(),
            features=feats.float(), valid=valid, member_key=self.key,
        )

    def enable_dropout(self, enable: bool) -> bool:
        # ConvNeXt-V2 / ViT probes have stochastic depth / dropout in the encoder;
        # toggling them gives MC-Dropout variance when unfrozen-train mode is on.
        n = 0
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train(enable)
                n += 1
        return n > 0