#!/usr/bin/env python3
"""Phase 6 follow-up: build architecture-DIVERSE from-scratch ensembles by
recombining per-member probabilities already saved in the per-backbone npz.

NO new training, NO new inference -- the three from-scratch backbones
(resnet18 / vit_tiny / convnext_tiny) were each trained 5-seed and their
per-member probs live in ``runs/eval_arrays/rex_fs_<bb>/arrays_rex_fs_<bb>{cal,eval}.npz``
with IDENTICAL image ordering (verified by ``ids`` alignment). This script
stacks seed-0 probs across backbones into a cross-arch ensemble, plus a
matched-M=3 same-arch control, to isolate the *diversity-type* axis from the
*training-regime* axis (see FINDINGS.md Phase 6 / docs §4.6).

Why: Phase 5 (arch-diverse, pretrained) deflates (disagreement ≡ confidence);
Phase 6 (same-arch, from-scratch) does not (disagreement > confidence). Those
two confound regime x diversity. The cross-arch from-scratch cell isolates
whether the driver is member weakness (regime) or architecture diversity.

Outputs (same schema as ``build_fromscratch_arrays.py`` so ``eval_baselines.py``
runs unchanged):
  ``<out>/<source>{cal,eval}.npz`` with probs (N,M,14), gt, valid, rad_feats,
  rad_logits, energy, mahalanobis, ids, member_keys, pathologies. The feature
  member is seed-0 of the chosen feature backbone (rad_feats/energy/maha reused
  verbatim from that backbone's seed-0 npz).

Usage:
  PYTHONPATH=. python scripts/build_crossarch_arrays.py
"""
from __future__ import annotations
import argparse
import os
import numpy as np


def load_bb(arr_dir, bb, split):
    p = os.path.join(arr_dir, f"rex_fs_{bb}", f"arrays_rex_fs_{bb}{split}.npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def build(arr_dir, out_dir, source, members, feat_bb, split):
    """members: list of (bb, seed_idx). feat_bb: backbone whose seed-0 npz
    supplies rad_feats/rad_logits/energy/mahalanobis/gt/valid/ids."""
    feat = load_bb(arr_dir, feat_bb, split)
    probs = []
    keys = []
    for bb, si in members:
        d = load_bb(arr_dir, bb, split)
        assert np.array_equal(d["ids"], feat["ids"]), f"id mismatch {bb} vs {feat_bb} ({split})"
        probs.append(d["probs"][:, si, :])
        keys.append(f"{bb}_s{si}")
    probs = np.stack(probs, axis=1).astype(np.float64)  # (N, M, 14)
    os.makedirs(os.path.join(out_dir, source), exist_ok=True)
    out = os.path.join(out_dir, source, f"arrays_{source}{split}.npz")
    np.savez_compressed(
        out,
        probs=probs,
        gt=feat["gt"],
        valid=feat["valid"],
        rad_feats=feat["rad_feats"],
        rad_logits=feat["rad_logits"],
        energy=feat["energy"],
        mahalanobis=feat["mahalanobis"],
        ids=feat["ids"],
        member_keys=np.array(keys),
        pathologies=feat["pathologies"],
    )
    print(f"[{source}] {split}: probs={probs.shape} members={keys} feat={feat_bb}_s0 -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arr-dir", default="runs/eval_arrays")
    ap.add_argument("--out", default="runs/eval_arrays")
    args = ap.parse_args()

    # M=3 cross-arch from-scratch: one seed-0 per backbone (feature=convnext_tiny_s0, 768-d).
    cross = [("resnet18", 0), ("vit_tiny", 0), ("convnext_tiny", 0)]
    for split in ("cal", "eval"):
        build(args.arr_dir, args.out, "rex_fs_crossarch", cross, "convnext_tiny", split)

    # M=3 same-arch control (resnet18 seeds 0,1,2) -- holds regime + M constant, varies diversity.
    same = [("resnet18", 0), ("resnet18", 1), ("resnet18", 2)]
    for split in ("cal", "eval"):
        build(args.arr_dir, args.out, "rex_fs_resnet18_m3", same, "resnet18", split)

    # M=2 cross-arch ROBUSTNESS variant: pure CNN (ResNet-18) vs pure transformer
    # (ViT-Tiny) -- the two endpoints of the inductive-bias axis. Motivation: the
    # M=3 crossarch above is 2 CNNs (ResNet + ConvNeXt) + 1 ViT, i.e. CNN-heavy on
    # the inductive-bias axis (ConvNeXt shares ResNet's conv bias; it is a CNN with
    # ViT-inspired design, no self-attention). This pair is the cleanest CNN<->TX
    # contrast. Feature member = resnet18_s0 (512-d; in the ensemble).
    cross_rv = [("resnet18", 0), ("vit_tiny", 0)]
    for split in ("cal", "eval"):
        build(args.arr_dir, args.out, "rex_fs_crossarch_rv", cross_rv, "resnet18", split)

    # M=2 same-arch control (resnet18 seeds 0,1) -- matched M=2 so member-count is
    # not re-confounded when comparing against the M=2 cross-arch pair above.
    same_m2 = [("resnet18", 0), ("resnet18", 1)]
    for split in ("cal", "eval"):
        build(args.arr_dir, args.out, "rex_fs_resnet18_m2", same_m2, "resnet18", split)

    print("CROSSARCH_BUILD_DONE")


if __name__ == "__main__":
    main()