"""Central configuration: pathology names, thresholds, model registry.

All knobs that a user may want to tweak live here so the rest of the package
stays declarative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# ---------------------------------------------------------------------------
# Pathology label space
# ---------------------------------------------------------------------------
# The 14 NIH ChestX-ray14 classes. torchxrayvision v1.5 DenseNet models expose
# an 18-slot output; the first 14 (below) are the NIH-14 set, and the naming
# already matches NIH ("Pleural_Thickening" uses an underscore). The trailing 4
# xrv classes (Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum)
# are not part of NIH-14 ground truth, so we restrict evaluation to these 14.
NIH_PATHOLOGIES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

# NIH "Finding Labels" uses identical names (with "No Finding" for negatives).
NO_FINDING = "No Finding"


# ---------------------------------------------------------------------------
# Pretrained model registry (torchxrayvision) -- the "free ensemble".
# ---------------------------------------------------------------------------
# torchxrayvision v1.5 weight tags. Each model outputs 18 slots; slots for
# classes a model was *not* trained on are empty strings (''). The ensemble
# loader aligns by pathology name and only averages over members that actually
# define a given class, so mixing partial-coverage models is safe.
@dataclass
class ModelSpec:
    key: str            # short id used in outputs
    weights: str = ""   # xrv weights="..." argument (xrv_densenet arch)
    arch: str = "xrv_densenet"   # member arch: xrv_densenet|convnextv2|raddino|arkswin|biomedclip
    hf_id: str = ""     # HF hub id for foundation/CLIP members
    probe_ckpt: str = "" # path to a trained linear-probe / LP-FT head checkpoint
    frozen: bool = True
    input_kind: str = "xrv"   # xrv (-1024..1024) | imagenet (0..1 3-ch) | raddino | clip
    # Local encoder checkpoint path for members NOT on the HF hub (Ark+ ships its
    # own weights; hf_id="" and ckpt_path points at the uploaded file). HF members
    # ignore this.
    ckpt_path: str = ""
    # xrv members only: what MemberOutput.probs holds. "legacy_double" (default)
    # preserves the pure-xrv baseline selectivity curve; "plain" is required in the
    # arch-primary ensemble so xrv + non-xrv members share one prob scale (else the
    # cross-member std INVERTS the confident-error signal). See xrv_densenet.py.
    prob_mode: str = "legacy_double"


ENSEMBLE_REGISTRY: Dict[str, ModelSpec] = {
    "all": ModelSpec("all", "densenet121-res224-all"),
    "nih": ModelSpec("nih", "densenet121-res224-nih"),
    "padchest": ModelSpec("padchest", "densenet121-res224-pc"),
    "chexpert": ModelSpec("chexpert", "densenet121-res224-chex"),
    "mimic": ModelSpec("mimic", "densenet121-res224-mimic_ch"),
    "rsna": ModelSpec("rsna", "densenet121-res224-rsna"),
}

# Leak-free default ensemble for OpenI evaluation. The "all" weights were
# trained on a superset that *includes OpenI*, so we exclude it to avoid
# train/test leakage on the OpenI eval set. `nih`/`padchest` cover all of
# NIH-14; `chexpert`/`mimic` cover the Chex subset and add diversity there.
# (For evaluating on NIH ChestX-ray14 instead, you may add "all" back.)
DEFAULT_ENSEMBLE: List[str] = ["nih", "padchest", "chexpert", "mimic"]

# Architecture-primary ensemble keys (Part 2). Phase 1 baseline is the 2-member
# xrv_nih + convnextv2 (supervised-CNN + ImageNet-CNN, architecture-diverse,
# no credentials). RAD-DINO/Ark+/BiomedCLIP are added in later phases. The
# "all"/per-dataset xrv keys above remain available for the legacy path.
ARCH_MEMBER_REGISTRY: Dict[str, ModelSpec] = {
    "xrv_nih": ModelSpec("xrv_nih", "densenet121-res224-nih", prob_mode="plain"),
    "xrv_padchest": ModelSpec("xrv_padchest", "densenet121-res224-pc", prob_mode="plain"),
    "convnextv2": ModelSpec(
        "convnextv2", arch="convnextv2",
        hf_id="timm/convnextv2_large.fcmae_ft_in1k",
        probe_ckpt="checkpoints/convnextv2_lpft.pt",
        input_kind="imagenet", frozen=False),
    "raddino": ModelSpec(
        "raddino", arch="raddino", hf_id="microsoft/rad-dino",
        probe_ckpt="checkpoints/raddino_probe.pt",
        input_kind="raddino", frozen=True),
    # M5 BiomedCLIP — language-contrastive CLIP (4th representation strategy),
    # frozen + linear probe. PMC-15M corpus -> leak-free for OpenI + all OOD probes.
    "biomedclip": ModelSpec(
        "biomedclip", arch="biomedclip",
        hf_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        probe_ckpt="checkpoints/biomedclip_probe.pt",
        input_kind="clip", frozen=True),
    # M4 Ark+6 — cyclic supervised Swin-L @ 768 (3rd representation family),
    # frozen + linear probe. OpenI + OOD probes leak-free; weights uploaded
    # locally (NOT on HF) -> hf_id="" + ckpt_path -> the 1.5GB training ckpt.
    # native_size=768 (see members/ark_swin.py); the ensemble loader uses
    # max(member native_size) so Ark+ gets true 768 input.
    "arkswin": ModelSpec(
        "arkswin", arch="arkswin", hf_id="",
        probe_ckpt="checkpoints/arkswin_probe.pt",
        input_kind="imagenet", frozen=True,
        ckpt_path="checkpoints/Ark6_swinLarge768_ep50.pth.tar"),
}
DEFAULT_ARCH_ENSEMBLE: List[str] = ["xrv_nih", "convnextv2"]


# ---------------------------------------------------------------------------
# Uncertainty / risk thresholds
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    # Binary decision threshold on the (ensemble-mean) probability per pathology.
    decision_thresh: float = 0.5

    # A prediction is "confident" if its confidence c = |p_bar - 0.5| * 2 >= this.
    # c in [0,1]; 1.0 means p_bar is 0 or 1. Default 0.6 -> p_bar <= 0.2 or >= 0.8.
    conf_thresh: float = 0.6

    # Epistemic-uncertainty threshold for flagging. Units = ensemble std of a
    # probability (in [0,1]); tuned empirically from the eval split.
    unc_thresh: float = 0.15

    # MC-Dropout settings (best-effort: only adds variance if the arch has
    # Dropout layers; BatchNorm is kept in eval mode).
    mc_samples: int = 10

    # Image size expected by the models.
    img_size: int = 224

    # Device: forced to cpu on this host (no GPU).
    device: str = "cpu"

    # If True, also compute MC-Dropout epistemic signal; otherwise rely on the
    # multi-model ensemble alone.
    use_mc_dropout: bool = True

    # Risk policy key (plan §M.2): ``"threshold"`` (default, the legacy
    # epistemic_std gate in risk.assess_image) or ``"conformal_triage"`` (the live
    # per-image ConformalTriageRiskPolicy mirror of the batch §L path — applies a
    # pre-fit TS temperature + per-pathology tau_p from the conformal sidecar to
    # the live p_bar and sets the risk flag to the LAC refer decision). Default
    # ``"threshold"`` keeps run_demo / Phase-0 byte-identical.
    risk_policy: str = "threshold"

    # Path to a conformal sidecar JSON (plan §M.2) carrying {T, tau_p, rare_group,
    # pooled_tau, alpha, pathologies, member_keys} fit on OpenI-C by
    # ``scripts/eval_conformal.py --write-sidecar``. Required when
    # ``risk_policy == "conformal_triage"``; ignored otherwise.
    conformal_sidecar: str = ""