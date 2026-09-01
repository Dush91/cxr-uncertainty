"""Member interface tests (plan §J.6). Verifies each ``Member`` exposes a
``MemberOutput`` with finite logits, probs in [0,1], a (B,D) feature tensor, and
a ``valid`` mask matching the declared target pathologies; and that the batch
dim is preserved (fixes the legacy batch-1 bug at models.py:104/129).

xrv members: ``probs`` is the legacy double-sigmoid+op_norm value (NOT
sigmoid(logits)) — see ``interfaces.MemberOutput`` — so we assert ``probs in
[0,1]`` rather than ``probs==sigmoid(logits)``. ConvNeXt-V2 (non-xrv) uses plain
sigmoid and IS asserted ``probs==sigmoid(logits)``; that test runs only once the
LP-FT checkpoint exists (post-training), to avoid an 800MB timm download in CI.
"""
from __future__ import annotations

import os
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")

from cxr_uncertainty.config import NIH_PATHOLOGIES, ARCH_MEMBER_REGISTRY
from cxr_uncertainty.interfaces import Alignment, MemberOutput
from cxr_uncertainty.member_factory import build_member


def _xrv_nih():
    return build_member(ARCH_MEMBER_REGISTRY["xrv_nih"], device="cpu")


def test_xrv_member_batch_and_shapes():
    """Feed (4,1,224,224) -> outputs preserve B=4 over the member's NATIVE 18
    slots; valid is True for the 14 non-empty NIH slots and False for the 4
    trailing empty ones. (Alignment reduces these to the 14 predictable.)"""
    m = _xrv_nih()
    x = torch.randn(4, 1, 224, 224)
    o = m.forward_batch(x)
    assert isinstance(o, MemberOutput)
    assert o.logits.shape == (4, 18)
    assert o.probs.shape == (4, 18)
    assert o.features.shape[0] == 4 and o.features.shape[1] > 0  # (B, D=1024)
    assert o.valid.shape == (4, 18)
    assert bool(torch.isfinite(o.logits).all())
    assert bool((o.probs >= 0).all() and (o.probs <= 1).all())
    # 14 non-empty NIH slots valid, 4 trailing empty slots invalid.
    assert bool(o.valid[:, :14].all())
    assert bool(~o.valid[:, 14:].any())


def test_xrv_member_target_pathologies_length():
    m = _xrv_nih()
    # xrv DenseNet exposes 18 slots; '' for untrained. valid marks non-empty.
    assert len(m.target_pathologies) == 18
    assert sum(1 for n in m.target_pathologies if n) == 14


def test_xrv_member_probs_in_unit_range_legacy():
    """xrv probs are the legacy double-sigmoid value; must be a valid prob."""
    m = _xrv_nih()
    x = torch.randn(2, 1, 224, 224)
    o = m.forward_batch(x)
    assert bool((o.probs >= 0).all() and (o.probs <= 1).all())
    # logits are the TRUE raw pre-sigmoid values (BCTS needs them).
    assert bool(torch.isfinite(o.logits).all())


def test_alignment_stacks_to_MBP():
    """Alignment.stack over a single-member list yields (M,B,P) with valid masks."""
    m = _xrv_nih()
    al = Alignment([m], NIH_PATHOLOGIES)
    assert len(al.predictable) == 14
    o = m.forward_batch(torch.randn(3, 1, 224, 224))
    probs, logits, feats, valid = al.stack([o])
    assert probs.shape == (1, 3, 14)
    assert logits.shape == (1, 3, 14)
    assert feats.shape[0] == 1 and feats.shape[1] == 3
    assert valid.shape == (1, 3, 14)
    assert bool(valid.all())
    # no NaN where a member defines the class (it defines all 14).
    assert bool(torch.isfinite(probs).all())


def test_member_has_dropout_semantics():
    """DenseNet has no Dropout -> has_dropout False and enable_dropout no-op."""
    m = _xrv_nih()
    assert m.has_dropout is False
    assert m.enable_dropout(True) is False


@pytest.mark.skipif(
    not os.path.exists("checkpoints/convnextv2_lpft.pt"),
    reason="convnextv2 LP-FT checkpoint not trained yet (run scripts/train_convnextv2_lpft.py)",
)
def test_convnextv2_member_interface():
    """Non-xrv member: probs == sigmoid(logits); batch dim preserved; features."""
    m = build_member(ARCH_MEMBER_REGISTRY["convnextv2"], device="cpu")
    x = torch.randn(4, 1, 224, 224) * 200  # xrv-range-ish input
    o = m.forward_batch(x)
    assert o.logits.shape == (4, 14)
    assert o.features.shape == (4, m.encoder.num_features)
    assert bool(torch.isfinite(o.logits).all())
    # non-xrv member uses plain sigmoid -> invariant holds.
    assert bool(torch.allclose(o.probs, torch.sigmoid(o.logits), atol=1e-4))
    assert bool(o.valid.all())


@pytest.mark.skipif(
    not os.path.exists("checkpoints/biomedclip_probe.pt"),
    reason="biomedclip probe not trained yet (run: python scripts/train_convnextv2_lpft.py "
           "--member biomedclip --lp-epochs 5 --ft-epochs 0 --out checkpoints/biomedclip_probe.pt)",
)
def test_biomedclip_member_interface():
    """M5 BiomedCLIP: frozen CLIP image tower (512-d) + linear probe. Gated on
    the probe checkpoint to avoid a 748MB HF download in CI (the cached encoder
    is reused once training has run). Verifies the 4th representation strategy
    slots into the shared Member interface: plain sigmoid, all-14 valid, 512-d
    features, input_kind clip, encoder frozen + head trainable."""
    spec = ARCH_MEMBER_REGISTRY["biomedclip"]
    assert spec.arch == "biomedclip" and spec.input_kind == "clip" and spec.frozen
    m = build_member(spec, device="cpu")
    assert m.arch_family == "biomedclip"
    assert m.encoder.num_features == 512
    assert m.head.in_features == 512 and m.head.out_features == 14
    # encoder frozen, head trainable (frozen-probe contract).
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert any(p.requires_grad for p in m.head.parameters())
    x = torch.rand(4, 1, 224, 224) * 2048 - 1024  # xrv-range input
    o = m.forward_batch(x)
    assert o.logits.shape == (4, 14)
    assert o.features.shape == (4, 512)
    assert o.valid.shape == (4, 14) and bool(o.valid.all())
    assert bool(torch.isfinite(o.logits).all())
    assert bool(torch.allclose(o.probs, torch.sigmoid(o.logits), atol=1e-4))
    assert o.member_key == "biomedclip"


@pytest.mark.skipif(
    not os.path.exists("checkpoints/Ark6_swinLarge768_ep50.pth.tar"),
    reason="Ark+ Swin-L checkpoint not present (user-uploaded "
           "checkpoints/Ark6_swinLarge768_ep50.pth.tar)",
)
@pytest.mark.skipif(
    not os.path.exists("checkpoints/arkswin_probe.pt"),
    reason="arkswin probe not trained yet (run: python scripts/train_convnextv2_lpft.py "
           "--member arkswin --lp-epochs 5 --ft-epochs 0 --out checkpoints/arkswin_probe.pt)",
)
def test_arkswin_member_interface():
    """M4 Ark+ Swin-L: the 3rd representation family (cyclic supervised multi-dataset
    Swin), frozen + linear probe. Verifies the high-resolution (native_size=768, 1536-d
    features) member slots into the shared Member interface AND that forward_batch
    accepts the member's native 768 input: plain sigmoid, all-14 valid, 1536-d features,
    input_kind imagenet, native_size 768, encoder frozen + head trainable. (The
    ensemble's forward_ensemble_full resizes the shared tensor to 768 for this member;
    here we feed 768 directly to exercise the member's own path.)"""
    spec = ARCH_MEMBER_REGISTRY["arkswin"]
    assert spec.arch == "arkswin" and spec.input_kind == "imagenet" and spec.frozen
    assert spec.ckpt_path.endswith("Ark6_swinLarge768_ep50.pth.tar")
    m = build_member(spec, device="cpu")
    assert m.arch_family == "arkswin"
    assert m.native_size == 768
    assert m.encoder.num_features == 1536
    assert m.head.in_features == 1536 and m.head.out_features == 14
    # encoder frozen, head trainable (frozen-probe contract).
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert any(p.requires_grad for p in m.head.parameters())
    # feed the member's native 768 input (as forward_ensemble_full would after resize).
    x = torch.rand(2, 1, 768, 768) * 2048 - 1024  # xrv-range input at native res
    o = m.forward_batch(x)
    assert o.logits.shape == (2, 14)
    assert o.features.shape == (2, 1536)
    assert o.valid.shape == (2, 14) and bool(o.valid.all())
    assert bool(torch.isfinite(o.logits).all())
    assert bool(torch.allclose(o.probs, torch.sigmoid(o.logits), atol=1e-4))
    assert o.member_key == "arkswin"