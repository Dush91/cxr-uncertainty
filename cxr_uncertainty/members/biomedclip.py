"""M5: BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) as a
``Member`` (Part 2 Phase 3, optional 5th member).

BiomedCLIP is a CLIP-style vision-language model: a ViT-B/16 image tower
(86.2M) + PubMedBERT text tower, contrastive-pretrained on **PMC-15M** (15M
PubMed Central figure-caption pairs — general biomedical literature, NOT
MIMIC/CheXpert/NIH/PadChest/OpenI CXR datasets). It is therefore the **4th
representation strategy** in the ensemble (language-contrastive CLIP), distinct
from RAD-DINO (vision-SSL ViT), ConvNeXt-V2 (ImageNet CNN), xrv DenseNet
(supervised CNN) and Ark+ (cyclic supervised Swin) — the M=5 diversity goal
(plan §C/§F.3).

**Leak status:** clean — PMC-15M contains none of the CXR datasets, so this
member is leak-free for OpenI (eval) and all OOD probes, same standing as the
other members.

**Known caveat (plan §G):** BiomedCLIP/CheXzero are reported *inconsistent on
CXR* externally (the pretraining corpus is general biomedical figures, not
chest X-rays). It is carried here as a **diversity member**, not for peak
accuracy — expect a lower per-member macroAUC than RAD-DINO/ConvNeXt. The
linear-probe head on role A is what adapts it to the 14-finding task. If it is
too weak, the plan's fallback is the Hybrid ConvNeXtV2-ViT as the 5th member.

Input kind = ``clip``: xrv-format (B,1,224,224)~[-1024,1024] -> CLIP-norm
3-channel (the OpenAI CLIP normalization mean/std, NOT ImageNet — verified from
``open_clip``'s BiomedCLIP preprocess). Features = the image tower's projected
embedding (**512-d**, the CLIP image projection; the ViT-B/16 hidden size is
768 but the projection head maps it to 512 for text alignment). Frozen + linear
probe (nn.Linear(512,14)); ``probs = sigmoid(logits)`` (plain, non-xrv).
"""
from __future__ import annotations

import warnings

import torch

from .base import LinearProbeMember

# CLIP normalization (OpenAI CLIP standard; verified from open_clip's
# BiomedCLIP preprocess transform: mean=[0.48145466, 0.4578275, 0.40821073],
# std=[0.26862954, 0.26130258, 0.27577711]).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BiomedClipMember(LinearProbeMember):
    arch_family = "biomedclip"
    input_kind = "clip"

    def __init__(self, key: str,
                 hf_id: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                 probe_ckpt: str = "", device: str = "cpu"):
        super().__init__()
        try:
            from open_clip import create_model_from_pretrained
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install open_clip_torch") from e
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # open_clip needs the ``hf-hub:`` prefix to load a HF-hub model
            # (without it, ``create_model`` looks up the name in its built-in
            # registry and fails for a HF repo id).
            model, _ = create_model_from_pretrained(
                hf_id if hf_id.startswith("hf-hub:") else f"hf-hub:{hf_id}")
        self.encoder = model.visual          # TimmModel ViT-B/16, frozen
        self.encoder.eval()
        # open_clip's TimmModel exposes the projected image-embedding dim via
        # ``.num_features`` on some versions but not all; set it explicitly to
        # the 512-d projection output (verified by a forward probe) so the
        # shared LinearProbeMember._setup builds Linear(512,14).
        if not hasattr(self.encoder, "num_features") or self.encoder.num_features is None:
            with torch.no_grad():
                d = int(self.encoder(torch.zeros(1, 3, 224, 224)).shape[-1])
            self.encoder.num_features = d
        self._mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)
        self._setup(key, device=device, load_ckpt=probe_ckpt)
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.eval()

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """xrv (B,1,224,224) in ~[-1024,1024] -> CLIP-norm (B,3,224,224)."""
        z = (x.float() + 1024.0) / 2048.0
        z = z.clamp(0.0, 1.0)
        z = z.repeat(1, 3, 1, 1)              # grayscale -> 3-channel
        z = (z - self._mean) / self._std
        return z

    def forward_features(self, z: torch.Tensor) -> torch.Tensor:
        """Projected image embedding -> (B, 512)."""
        return self.encoder(z)