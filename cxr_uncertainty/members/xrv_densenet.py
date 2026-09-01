"""M3: torchxrayvision DenseNet121 as a ``Member`` (Part 2 Phase 0).

Wraps the existing xrv DenseNet (the legacy free ensemble member, no training).
Exposes TRUE raw logits by splitting the xrv forward into ``features2 ->
classifier`` — bypassing xrv's conditional ``sigmoid`` + ``op_norm``. ``logits``
is the raw ``classifier(features2)`` output (Beta/BCTS needs these).

``prob_mode`` selects what ``MemberOutput.probs`` (the pooled value the ensemble
disagrees on) contains:

  * ``"legacy_double"`` (default): ``sigmoid(model(x))`` = the legacy
    ``sigmoid(op_norm(sigmoid(logits)))`` value. Preserved because in a
    **pure-xrv** ensemble the shared [0.5, 0.73] compression band makes the
    cross-member std a good confident-error signal (matches the pre-refactor
    0.94 selectivity curve). DO NOT use in a mixed arch ensemble: the
    [0.5,0.73]-pinned member vs a full-[0,1] non-xrv member makes std ≈
    |other − 0.6| → large when the other member is confidently *right* → the
    selectivity signal INVERTS (AUROC ≪ 0.5).
  * ``"plain"``: ``sigmoid(logits)`` (plain). Required for the architecture-
    primary ensemble so xrv and non-xrv members share one scale and the std
    measures genuine cross-arch disagreement. (In a pure-xrv ensemble this
    regresses the selectivity curve on OOD data — see plan §J.1 — hence the
    mode switch instead of a global change.)
"""
from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn

from ..interfaces import Member, MemberOutput


class XrvDenseNetMember(Member):
    key: str
    arch_family: str = "xrv_densenet"
    input_kind: str = "xrv"
    native_size: int = 224

    def __init__(self, key: str, weights: str, device: str = "cpu",
                 prob_mode: str = "legacy_double"):
        super().__init__()
        self.key = key
        self.prob_mode = prob_mode
        try:
            import torchxrayvision as xrv
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install torchxrayvision") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = xrv.models.DenseNet(weights=weights)
        self.model.eval()
        self._device = device
        self.model.to(device)
        # 18-slot pathologies; '' for untrained classes.
        self.target_pathologies: List[str] = list(self.model.pathologies)
        self._slot_valid = torch.tensor(
            [name != "" for name in self.target_pathologies], dtype=torch.bool,
            device=device,
        )

    # xrv-format tensor in [-1024,1024] is already this member's native input.
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor) -> MemberOutput:
        x = self.preprocess(x).to(self._device)
        # True raw logits (pre-sigmoid, pre-op_norm) via the features2->classifier
        # split — Part 3 BCTS/Beta calibration operates on these.
        features = self.model.features2(x)          # (B, 1024) penultimate
        logits = self.model.classifier(features)    # (B, 18) true raw logits
        # See the module docstring for the prob_mode rationale. logits (true raw)
        # is always exposed for BCTS regardless of the probs mode.
        if self.prob_mode == "plain":
            probs = torch.sigmoid(logits)
        else:  # "legacy_double": sigmoid(op_norm(sigmoid(logits)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_out = self.model(x)           # op_norm(sigmoid(logits))
            probs = torch.sigmoid(model_out)
        B = logits.shape[0]
        valid = self._slot_valid.unsqueeze(0).expand(B, -1).clone()
        return MemberOutput(
            logits=logits.float(), probs=probs.float(),
            features=features.float(), valid=valid, member_key=self.key,
        )

    def enable_dropout(self, enable: bool) -> bool:
        # DenseNet121 has no nn.Dropout layers -> MC adds nothing; keep semantics.
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train(enable)
        return self.has_dropout