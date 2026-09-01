#!/usr/bin/env python
"""Train a from-scratch backbone + 14-class head on a manifest train split
(Baur D-Ens control -- Phase 6).

The Phase-5 deflation (epistemic_std == confidence at power) was measured on the
**architecture-diverse, foundation-pretrained** 4-member ensemble. Baur et al.
(UNSURE@MICCAI 2025) report that a **Deep Ensemble** of *from-scratch* trainings
of one backbone yields a disagreement signal that carries independent
confident-error information. This script trains ONE such seed: a random-init
backbone (ResNet-18 / ViT-Tiny / ConvNeXt-Tiny -- Baur's anchors) end-to-end on
ReXGradient-train with masked BCE, early-stopped on the ReX-valid macro-AUROC.
Run it 5x with different ``--seed`` to build a D-Ens; the disagreement across
the 5 (nanstd / mutual-info of their sigmoid probs) is the epistemic signal
``scripts/build_fromscratch_arrays.py`` then feeds to the unchanged
``eval_baselines.py``.

This is deliberately self-contained: it does NOT touch the pretrained-member
framework (``Member`` / ``Alignment`` / registries) -- those are for the
foundation-model ensemble. A from-scratch control is a plain backbone + head,
trained directly.

Design (matches Baur / the literature-standard from-scratch regime):

  * Random-init end-to-end (no LP/FT split, no frozen encoder), single phase.
  * 224px, ImageNet-norm 3-channel input (xrv [-1024,1024] -> [0,1] -> 3ch ->
    ImageNet mean/std; same mapping as ``members/convnext_v2.py`` so the
    from-scratch models live in the same input space as the pretrained members).
  * Masked BCE with the per-(example,class) validity mask (the 4 NIH-only
    classes have valid=0 -- no CheXpert source -- so they contribute no loss).
  * **u_policy="ones"**: uncertain (-1) train labels collapse to 1 (positive),
    the Baur / CheXpert literature default for training. Test labels keep -1
    (Task 2) -- that is handled in the manifest, not here.
  * AMP + bs256 + AdamW + cosine; early-stop on val macro-AUROC with patience.
  * Aggressive DataLoader workers (persistent_workers + pin_memory) to keep the
    GPU fed -- data loading (xrv imread + center-crop + resize) is the dominant
    cost on 110k images, not the small-backbone forward/backward.
  * Saves the full ``state_dict`` (backbone + head); the producer loads it back
    into the identical architecture for inference + feature extraction.

Usage (one seed; loop --seed 0..4 for a D-Ens):

    PYTHONPATH=. python scripts/train_fromscratch.py \
        --backbone resnet18 --seed 0 \
        --manifest data/rex_manifest.parquet --train-role train --val-role cal \
        --epochs 20 --bs 256 --lr 1e-3 --out checkpoints/fs_resnet18_s0.pt

    # smoke test (CPU / tiny subset, 1 epoch):
    PYTHONPATH=. python scripts/train_fromscratch.py --backbone resnet18 \
        --limit 64 --epochs 1 --bs 16
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402

warnings.filterwarnings("ignore")

NUM_CLASSES = len(NIH_PATHOLOGIES)          # 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# timm model names (random init via pretrained=False). We use the BARE
# architecture names (no pretrained-cfg tag) so resolution is version-robust
# across timm releases -- a specific tag (e.g. ``..._in22k``) may not exist in
# a given timm version, but the bare arch name always does. pretrained=False so
# no weights are downloaded -- we train from scratch. vit_tiny -> 192-d,
# convnext_tiny -> 768-d.
_TIMM_NAMES = {
    "vit_tiny": "vit_tiny_patch16_224",
    "convnext_tiny": "convnext_tiny",
}
# default LR per backbone (heavier backbones -> smaller LR).
_DEFAULT_LR = {"resnet18": 1e-3, "vit_tiny": 1e-3, "convnext_tiny": 3e-4}


# ---------------------------------------------------------------------------
# Model: random-init backbone + 14-class head, with a features() hook for the
# producer's feature-member extraction (penultimate embedding).
# ---------------------------------------------------------------------------
class FromScratchModel(nn.Module):
    """Wraps a backbone so ``forward`` -> logits (training) and
    ``forward_with_features`` -> (logits, penultimate embedding) (inference)."""

    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.backbone_name = backbone
        self.is_timm = backbone in _TIMM_NAMES
        if self.is_timm:
            import timm
            name = _TIMM_NAMES[backbone]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.net = timm.create_model(name, pretrained=False,
                                             num_classes=num_classes)
            self.num_features = int(self.net.num_features)
        elif backbone == "resnet18":
            import torchvision
            self.net = torchvision.models.resnet18(weights=None)
            self.num_features = int(self.net.fc.in_features)   # 512
            self.net.fc = nn.Linear(self.num_features, num_classes)
        else:
            raise ValueError(f"unknown backbone '{backbone}'; "
                             f"choose one of {list(_DEFAULT_LR)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits only (training path -- no feature overhead)."""
        return self.net(x)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate embedding (B, D). For timm: forward_features ->
        forward_head(pre_logits=True). For resnet18: the manual conv->...->
        avgpool->flatten path (identical to net.forward up to the fc)."""
        if self.is_timm:
            f = self.net.forward_features(x)
            return self.net.forward_head(f, pre_logits=True)
        n = self.net
        x = n.conv1(x); x = n.bn1(x); x = n.relu(x); x = n.maxpool(x)
        x = n.layer1(x); x = n.layer2(x); x = n.layer3(x); x = n.layer4(x)
        x = n.avgpool(x); x = torch.flatten(x, 1)
        return x

    def forward_with_features(self, x: torch.Tensor):
        """(logits, features) in one forward pass (inference producer).
        Computes the backbone features exactly once -- timm's forward_features
        is the expensive part, so we reuse it for both the pre-logits embedding
        and the class head (avoids a 2x backbone forward)."""
        if self.is_timm:
            f = self.net.forward_features(x)
            feats = self.net.forward_head(f, pre_logits=True)
            logits = self.net.forward_head(f)
            return logits, feats
        feats = self._features(x)
        logits = self.net.fc(feats)
        return logits, feats


def preprocess(x: torch.Tensor, device: str) -> torch.Tensor:
    """xrv (B,1,H,W) in ~[-1024,1024] -> ImageNet-norm (B,3,H,W). Same mapping as
    ``members/convnext_v2.py`` so from-scratch models share the pretrained
    members' input space."""
    z = (x.float() + 1024.0) / 2048.0
    z = z.clamp(0.0, 1.0)
    z = z.repeat(1, 3, 1, 1)                                  # grayscale -> 3ch
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (z - mean) / std


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ManifestImageDataset(Dataset):
    """Loads manifest images as the xrv-format (1,1,224,224)~[-1024,1024] tensor
    (preprocess is applied in the loop). Returns (image, labels[14], valid[14]).

    ``u_policy_ones`` collapses uncertain (-1) labels to 1 (positive) -- the
    training convention; leave False for a val/eval loader that must preserve
    -1 (though the val macro-AUROC here only uses the binary {0,1} part)."""

    def __init__(self, df: pd.DataFrame, img_size: int = 224, augment: bool = False,
                 u_policy_ones: bool = False, cache_path=None):
        self.paths = df["image_path"].tolist()
        labels = np.stack(df["labels"].values).astype(np.float32)
        if u_policy_ones:                          # -1 -> 1 (training)
            labels = np.where(labels == -1, 1.0, labels)
        self.labels = labels
        self.valid = np.stack(df["valid"].values).astype(np.float32)
        self.img_size = img_size
        self.augment = augment
        # Optional pre-decoded memmap (float32 (N,1,H,W) in [-1024,1024]); row i
        # must align with df row i. Built by scripts/build_train_cache.py in the
        # same df order (split_role filter + reset_index). Eliminates the
        # per-epoch 16-bit PNG decode+resize that left the GPU idle.
        self.cache = None
        if cache_path:
            self.cache = np.memmap(cache_path, dtype=np.float32, mode="r",
                                   shape=(len(self.paths), 1, img_size, img_size))
            sidecar = (cache_path[:-4] if cache_path.endswith(".npy") else cache_path) + ".paths.txt"
            if os.path.exists(sidecar):
                with open(sidecar) as f:
                    cached_paths = [ln.rstrip("\n") for ln in f if ln.strip()]
                assert cached_paths == self.paths, (
                    f"cache sidecar {sidecar} order != manifest "
                    f"({len(cached_paths)} vs {len(self.paths)} paths)")
                print(f"[cache] using {cache_path} ({len(self.paths)} imgs, aligned)")
            else:
                print(f"[cache] WARNING: no sidecar {sidecar}; skipping alignment check")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.cache is not None:
            x = torch.from_numpy(self.cache[i].copy())      # (1,H,W) float32 [-1024,1024]
        else:
            x = load_image_tensor(self.paths[i], self.img_size, device="cpu")  # (1,1,H,W)
            x = x.squeeze(0)                                # (1,H,W)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])               # horizontal flip
            if torch.rand(1).item() < 0.5:                  # small rotation
                ang = float(torch.empty(1).uniform_(-10.0, 10.0).item())
                from torchvision.transforms import functional as TF
                x = TF.rotate(x, ang, interpolation=TF.InterpolationMode.BILINEAR,
                              fill=[-1024.0])
        return (x, torch.from_numpy(self.labels[i]), torch.from_numpy(self.valid[i]))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)         # (B,1,H,W)
    labels = torch.stack([b[1] for b in batch], dim=0)       # (B,14)
    valid = torch.stack([b[2] for b in batch], dim=0)        # (B,14)
    return imgs, labels, valid


# ---------------------------------------------------------------------------
# Loss + metric
# ---------------------------------------------------------------------------
def masked_bce(logits, labels, valid):
    """BCE-with-logits zeroed on invalid (example,class) slots; mean over valid."""
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = loss * valid
    denom = valid.sum().clamp(min=1.0)
    return loss.sum() / denom


@torch.no_grad()
def macro_auroc(model, loader, device, max_n=None):
    """Macro-AUROC over the 14 classes (skip classes with no pos/neg among valid
    examples). Uses the model's logits -> sigmoid."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    probs, gts, valids = [], [], []
    n = 0
    for imgs, labels, valid in loader:
        imgs = imgs.to(device)
        z = preprocess(imgs, device)
        logits = model(z)
        probs.append(torch.sigmoid(logits).cpu())
        gts.append(labels); valids.append(valid)
        n += imgs.shape[0]
        if max_n and n >= max_n:
            break
    P = torch.cat(probs).numpy(); Y = torch.cat(gts).numpy(); V = torch.cat(valids).numpy()
    aucs = []
    for c in range(Y.shape[1]):
        m = V[:, c] == 1                       # label-available rows
        y = Y[m, c]
        p = P[m, c]
        nonneg = y >= 0                         # exclude -1 (uncertain) from the AUC
        y = y[nonneg].astype(int)
        p = p[nonneg]
        if y.sum() == 0 or y.sum() == len(y):  # no pos or no neg -> undefined
            continue
        aucs.append(roc_auc_score(y, p))
    return (float(np.mean(aucs)) if aucs else 0.0), len(aucs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, loader, val_loader, device, epochs, lr, weight_decay,
          patience, use_amp):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best, best_state, best_ep = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for imgs, labels, valid in loader:
            imgs, labels, valid = imgs.to(device), labels.to(device), valid.to(device)
            opt.zero_grad(set_to_none=True)
            z = preprocess(imgs, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(z)
                loss = masked_bce(logits, labels, valid)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        auc, nc = macro_auroc(model, val_loader, device)
        dt = time.time() - t0
        print(f"  epoch {ep+1}/{epochs}  train_loss={tot/max(nb,1):.4f}  "
              f"val_macroAUC={auc:.4f} (n_classes={nc})  lr={opt.param_groups[0]['lr']:.2e}  "
              f"{dt:.0f}s")
        if auc > best:
            best, best_ep = auc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif patience and (ep - best_ep) >= patience:
            print(f"  [early-stop] no val improvement for {patience} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    p = argparse.ArgumentParser(prog="train_fromscratch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=list(_DEFAULT_LR),
                   help="from-scratch backbone (resnet18 | vit_tiny | convnext_tiny)")
    p.add_argument("--seed", type=int, required=True, help="random seed (D-Ens member index)")
    p.add_argument("--manifest", default="data/rex_manifest.parquet")
    p.add_argument("--train-role", default="train", help="split_role value for train rows")
    p.add_argument("--val-role", default="cal", help="split_role value for val/early-stop rows")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=None, help=f"default per backbone {_DEFAULT_LR}")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5, help="early-stop patience (0=off)")
    p.add_argument("--limit", type=int, default=0, help="cap train+val images (smoke test); 0=all")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true", default=None,
                   help="mixed-precision (default: on for cuda, off for cpu)")
    p.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/fs_<backbone>_s<seed>.pt)")
    p.add_argument("--cache-train", default=None,
                   help="memmap .npy of pre-decoded train images (float32 (N,1,224,224)); skip PNG decode")
    p.add_argument("--cache-val", default=None,
                   help="memmap .npy of pre-decoded val images (same format)")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.amp if args.amp is not None else (device == "cuda"))
    lr = args.lr if args.lr is not None else _DEFAULT_LR[args.backbone]
    out = args.out or str(_REPO / "checkpoints" / f"fs_{args.backbone}_s{args.seed}.pt")
    print(f"[train] backbone={args.backbone} seed={args.seed} device={device} amp={use_amp} lr={lr}")

    df = pd.read_parquet(args.manifest)
    train_df = df[df.split_role == args.train_role].reset_index(drop=True)
    val_df = df[df.split_role == args.val_role].reset_index(drop=True)
    assert len(train_df) > 0, f"no train rows (split_role=='{args.train_role}') in {args.manifest}"
    assert len(val_df) > 0, f"no val rows (split_role=='{args.val_role}') in {args.manifest}"
    if args.limit:
        train_df = train_df.iloc[:args.limit].reset_index(drop=True)
        val_df = val_df.iloc[:args.limit].reset_index(drop=True)
    print(f"[data] train({args.train_role})={len(train_df)}  val({args.val_role})={len(val_df)}")

    model = FromScratchModel(args.backbone).to(device)
    print(f"[model] {args.backbone}  features={model.num_features}  "
          f"head=Linear({model.num_features},{NUM_CLASSES})  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    nw = min(16, max(2, (os.cpu_count() or 4) - 1))
    train_ds = ManifestImageDataset(train_df, augment=True, u_policy_ones=True,
                                     cache_path=args.cache_train)
    val_ds = ManifestImageDataset(val_df, augment=False, u_policy_ones=False,
                                  cache_path=args.cache_val)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=collate, num_workers=nw, pin_memory=(device == "cuda"),
                              drop_last=True, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                            collate_fn=collate, num_workers=nw, pin_memory=(device == "cuda"),
                            persistent_workers=(nw > 0))

    best = train(model, train_loader, val_loader, device, args.epochs, lr,
                 args.wd, args.patience, use_amp)
    final_auc, nc = macro_auroc(model, val_loader, device)
    print(f"\n[done] best-val macroAUC={best:.4f}  final-val macroAUC={final_auc:.4f} (n_classes={nc})")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.save(model.state_dict(), out)
    print(f"[done] saved -> {out}")


if __name__ == "__main__":
    main()