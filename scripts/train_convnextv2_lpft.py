"""Train M2 ConvNeXt-V2-Large via LP-FT (Part 2 Phase 1, the only GPU-trained member).

The ImageNet-FCMAE encoder never saw any CXR, so fine-tuning it on the NIH +
CheXpert union (manifest role A) is genuine per-dataset specialization — a real
new loss basin (§H.1). Recipe (plan §J.2):

  1. Linear probe: 3-5 epochs, encoder frozen, AdamW head-only lr=1e-3.
  2. Fine-tune: 5-10 epochs, encoder unfrozen, AdamW lr=1e-5 cosine schedule.
  3. Loss = BCE-with-logits with a per-example, per-class **validity mask**
     (zero the loss for class c on a source that doesn't define it — e.g. CheXpert
     has no Nodule/Emphysema/Fibrosis/Hernia). Normalized by the number of valid
     (example, class) entries in the batch.
  4. Early-stop / best-model on Role-B **macro-AUROC** (patient-disjoint from A).
  5. Save the full state_dict (encoder + head) to ``checkpoints/convnextv2_lpft.pt``
     so :class:`cxr_uncertainty.members.convnext_v2.ConvNextV2Member` loads it at
     inference (exact same preprocessing path — the member's own ``preprocess``).

Usage:
    python scripts/train_convnextv2_lpft.py [--lp-epochs 3] [--ft-epochs 6] [--bs 48]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from cxr_uncertainty.config import NIH_PATHOLOGIES, ARCH_MEMBER_REGISTRY
from cxr_uncertainty.member_factory import build_member
from cxr_uncertainty.utils import load_image_tensor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ManifestImageDataset(Dataset):
    """Loads role-A/B images as the xrv-format (1,1,224,224)~[-1024,1024] tensor
    the member's ``preprocess`` expects. Returns (image, labels[14], valid[14])."""

    def __init__(self, df: pd.DataFrame, img_size: int = 224, augment: bool = False):
        self.paths = df["image_path"].tolist()
        self.labels = np.stack(df["labels"].values).astype(np.float32)
        self.valid = np.stack(df["valid"].values).astype(np.float32)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = load_image_tensor(self.paths[i], self.img_size, device="cpu")  # (1,1,H,W)
        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])     # random horizontal flip (CXR is ~symmetric)
        return (
            x.squeeze(0),                                    # (1,224,224)
            torch.from_numpy(self.labels[i]),
            torch.from_numpy(self.valid[i]),
        )


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)         # (B,1,224,224)
    labels = torch.stack([b[1] for b in batch], dim=0)        # (B,14)
    valid = torch.stack([b[2] for b in batch], dim=0)         # (B,14)
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
def macro_auroc(member, loader, device, max_n=None):
    """Macro-AUROC over the 14 classes (skip classes with no positives or no
    negatives among valid examples in this loader)."""
    member.eval()
    probs, gts, valids = [], [], []
    n = 0
    for imgs, labels, valid in loader:
        imgs = imgs.to(device)
        z = member.preprocess(imgs)
        feats = member.forward_features(z)
        logits = member.head(feats)
        p = torch.sigmoid(logits).cpu()
        probs.append(p); gts.append(labels); valids.append(valid)
        n += imgs.shape[0]
        if max_n and n >= max_n:
            break
    P = torch.cat(probs).numpy(); Y = torch.cat(gts).numpy(); V = torch.cat(valids).numpy()
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(Y.shape[1]):
        m = V[:, c] == 1
        y = Y[m, c]
        if y.sum() == 0 or y.sum() == m.sum():
            continue
        aucs.append(roc_auc_score(y, P[m, c]))
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_phase(member, loader, val_loader, device, epochs, lr, freeze_encoder,
                weight_decay, cos=False, label="LP"):
    opt = torch.optim.AdamW(
        [p for p in member.trainable_parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cos else None
    member.freeze_encoder(freeze_encoder)
    best, best_state = -1.0, None
    for ep in range(epochs):
        member.encoder.train(not freeze_encoder)
        member.head.train()
        tot, nb = 0.0, 0
        for imgs, labels, valid in loader:
            imgs, labels, valid = imgs.to(device), labels.to(device), valid.to(device)
            opt.zero_grad(set_to_none=True)
            z = member.preprocess(imgs)
            if freeze_encoder:
                with torch.no_grad():           # frozen probe: don't retain encoder graph
                    feats = member.forward_features(z).detach()
            else:
                feats = member.forward_features(z)
            logits = member.head(feats)
            loss = masked_bce(logits, labels, valid)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1
        if sched:
            sched.step()
        auc, nc = macro_auroc(member, val_loader, device)
        print(f"  [{label}] epoch {ep+1}/{epochs}  train_loss={tot/max(nb,1):.4f}  "
              f"val_macroAUC={auc:.4f} (n_classes={nc})  lr={opt.param_groups[0]['lr']:.2e}")
        if auc > best:
            best = auc
            best_state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
    if best_state is not None:
        member.load_state_dict(best_state)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(prog="train_convnextv2_lpft", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/manifest.parquet")
    p.add_argument("--member", default="convnextv2",
                   help="ARCH_MEMBER_REGISTRY key to train (convnextv2 | raddino)")
    p.add_argument("--out", default="checkpoints/convnextv2_lpft.pt")
    p.add_argument("--lp-epochs", type=int, default=3)
    p.add_argument("--ft-epochs", type=int, default=6)
    p.add_argument("--bs", type=int, default=48)
    p.add_argument("--lp-lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--val-subset", type=int, default=2000,
                   help="cap #val images per AUROC eval (speed); 0 = all")
    p.add_argument("--train-subset", type=int, default=0,
                   help="cap #train images (smoke test); 0 = all role-A")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    df = pd.read_parquet(args.manifest)
    train_df = df[df.split_role == "A"].reset_index(drop=True)
    val_df = df[df.split_role == "B"].reset_index(drop=True)
    if args.train_subset:
        train_df = train_df.iloc[:args.train_subset]
    print(f"[data] train(A)={len(train_df)}  val(B)={len(val_df)}")
    print(f"[data] train source counts: {train_df.source.value_counts().to_dict()}")

    spec = ARCH_MEMBER_REGISTRY[args.member]
    member = build_member(spec, device=device)        # encoder pretrained, head random
    native = getattr(member, "native_size", 224)
    print(f"[model] {args.member}  features={member.encoder.num_features}  "
          f"frozen={spec.frozen}  head=Linear({member.encoder.num_features},14)  "
          f"native_size={native}")

    # Load images at the member's native_size (768 for Ark+; 224 otherwise). The
    # member's preprocess assumes its native resolution; the train path calls
    # preprocess + forward_features directly (not forward_batch), so the dataset
    # must produce the member's native size.
    train_ds = ManifestImageDataset(train_df, augment=True, img_size=native)
    val_ds = ManifestImageDataset(val_df, augment=False, img_size=native)
    nw = min(8, max(2, torch.get_num_threads() // 2))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              collate_fn=collate, num_workers=nw, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                           collate_fn=collate, num_workers=nw, pin_memory=True)
    val_eval = val_loader
    if args.val_subset and len(val_df) > args.val_subset:
        val_eval = DataLoader(ManifestImageDataset(val_df.iloc[:args.val_subset],
                                                   img_size=native),
                              batch_size=args.bs, collate_fn=collate, num_workers=nw)

    if args.lp_epochs > 0:
        print("\n=== Phase 1: Linear probe (encoder frozen) ===")
        train_phase(member, train_loader, val_eval, device,
                    args.lp_epochs, args.lp_lr, freeze_encoder=True,
                    weight_decay=args.wd, cos=False, label="LP")

    if args.ft_epochs > 0:
        print("\n=== Phase 2: Fine-tune (encoder unfrozen, cosine) ===")
        train_phase(member, train_loader, val_eval, device,
                    args.ft_epochs, args.ft_lr, freeze_encoder=False,
                    weight_decay=args.wd, cos=True, label="FT")

    final_auc, nc = macro_auroc(member, val_eval, device)
    print(f"\n[done] best-val macroAUC={final_auc:.4f} (n_classes={nc})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    member.freeze_encoder(True)                       # inference: frozen, eval
    member.eval()
    torch.save(member.state_dict(), args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()