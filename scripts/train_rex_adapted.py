#!/usr/bin/env python
"""Phase 7: adapt the PRETRAINED ensemble members to ReXGradient-train (LP-FT).

The production-realistic arm: ReX is OOD for every pretrained member, so this
arm answers "what does the hospital's UQ story look like once the ensemble is
adapted to the local dataset?" -- the twin of the published zero-adaptation
regime, under the IDENTICAL protocol as the Phase-6 from-scratch control:

  * data    : ``data/rex_manifest_fs.parquet`` (patient-disjoint train 112,967 /
              cal 8,064 / eval 8,082 -- built for the fs control; no new labeling).
  * loss    : masked BCE on the 14 NIH classes; ``u_policy="ones"`` -- train -1
              labels collapse to 1 (Baur/CheXpert default); cal keeps -1 (only
              certain labels enter the val metric).
  * model   : early-stop / best-epoch on **cal certain-label macro-AUROC**.
  * augment : hflip 50% + rotate [-10,10]deg (same as the fs protocol).
  * cache   : pre-decoded 224px memmaps (float16) from build_train_cache.py for
              the 224-px members; ``arkswin`` runs native 768 on PNG paths.

Per-member adaptation (Route A -- adapt all four pretrained foundations):
    xrv_nih    LP (classifier only) then FT (end-to-end, low LR)
    convnextv2 LP-FT per train_convnextv2_lpft.py, warm start = published ckpt;
               --seed reproduces s0/s1/s2 for seed-variance robustness
    raddino    LP then SHORT low-LR encoder FT (ViT-B; tiny LR)
    arkswin    linear probe only @768 (frozen encoder; budget guard)

Usage (Lightning GPU):  bash -lc 'PYTHONPATH=. python scripts/train_rex_adapted.py \
                            --member convnextv2 --seed 0'
Smoke (CPU):            PYTHONPATH=. python scripts/train_rex_adapted.py \
                            --member convnextv2 --limit 200 --lp-epochs 1 --ft-epochs 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr_uncertainty.config import NIH_PATHOLOGIES, ARCH_MEMBER_REGISTRY  # noqa: E402
from cxr_uncertainty.member_factory import build_member                    # noqa: E402
from cxr_uncertainty.utils import load_image_tensor                        # noqa: E402

warnings.filterwarnings("ignore")
PATS = [str(p) for p in NIH_PATHOLOGIES]
DEFAULT_MANIFEST = "data/rex_manifest_fs.parquet"
OUT_DIR = "checkpoints/rex_adapted"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ManifestImageDataset(Dataset):
    """Manifest rows -> (1,H,W) xrv-format [-1024,1024] + labels/valid, with the
    fs protocol's augment block. Optionally backed by a pre-decoded float16/32
    memmap at 224 px (row i must align with df row i -- sidecar-asserted)."""

    def __init__(self, df: pd.DataFrame, img_size: int = 224, augment: bool = False,
                 u_policy_ones: bool = False, cache_path: str | None = None,
                 cache_dtype: str = "float16", limit: int = 0):
        if limit:
            df = df.iloc[:limit].reset_index(drop=True)
        self.paths = df["image_path"].tolist()
        labels = np.stack(df["labels"].values).astype(np.float32)
        if u_policy_ones:                          # -1 -> 1 (training convention)
            labels = np.where(labels == -1, 1.0, labels)
        self.labels = labels
        self.valid = np.stack(df["valid"].values).astype(np.float32)
        self.img_size = img_size
        self.augment = augment
        self.cache = None
        if cache_path:
            dt = np.dtype(cache_dtype)
            self.cache = np.memmap(cache_path, dtype=dt, mode="r",
                                   shape=(len(self.paths), 1, img_size, img_size))
            sidecar = (cache_path[:-4] if cache_path.endswith(".npy") else cache_path) \
                + ".paths.txt"
            with open(sidecar) as f:
                cached_paths = [ln.rstrip("\n") for ln in f if ln.strip()]
            assert cached_paths[:len(self.paths)] == self.paths, (
                f"cache sidecar {sidecar} order != manifest subset "
                f"({len(cached_paths)} vs {len(self.paths)} paths) -- manifest "
                "order changed? rebuild the cache for this subset")
            print(f"[cache] using {cache_path} dtype={cache_dtype} "
                  f"({len(self.paths)} imgs, sidecar-aligned)")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.cache is not None:
            x = torch.from_numpy(self.cache[i].astype(np.float32))  # (1,H,W)
        else:
            x = load_image_tensor(self.paths[i], self.img_size, device="cpu").squeeze(0)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(1).item() < 0.5:
                ang = float(torch.empty(1).uniform_(-10.0, 10.0).item())
                x = TF.rotate(x, ang, interpolation=TF.InterpolationMode.BILINEAR,
                              fill=[-1024.0])
        return (x, torch.from_numpy(self.labels[i]), torch.from_numpy(self.valid[i]))


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


# ---------------------------------------------------------------------------
# Member plumbing
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def nih_logits(member, z: torch.Tensor):
    """(B,14) raw logits in NIH order for either member type."""
    if hasattr(member, "model"):                    # XrvDenseNetMember
        feats = member.model.features2(z)           # (B,1024) penultimate
        log18 = member.model.classifier(feats)      # (B,18)
        cols = [list(member.target_pathologies).index(p) for p in PATS]
        return log18[:, cols]
    feats = member.forward_features(z)              # (B,D)
    return member.head(feats)


def reset_head(member) -> None:
    """Fresh random head for the LP phase (warm-started encoder). xrv keeps its
    pretrained NIH classifier head -- a legitimate warm start."""
    if hasattr(member, "head"):
        member.head.reset_parameters()
        print("[init] head re-initialized (LP starts from a random head)")


@torch.no_grad()
def macro_auroc_nih(member, loader, device, amp=True):
    """Certain-label macro-AUROC on the 14 NIH classes (cal -1 rows dropped,
    mirroring eval_baselines' certain-label convention)."""
    member.eval()
    ps, ys, vs = [], [], []
    use_amp = amp and device.startswith("cuda")
    for imgs, labels, valid in loader:
        imgs = imgs.to(device)
        with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
            z = member.preprocess(imgs)
            log14 = nih_logits(member, z)
        ps.append(torch.sigmoid(log14.float()).cpu())
        ys.append(labels)
        vs.append(valid)
    P = torch.cat(ps).numpy()
    Y, V = torch.cat(ys).numpy(), torch.cat(vs).numpy()
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(P.shape[1]):
        m = (V[:, c] == 1) & (Y[:, c] >= 0)
        y = Y[m, c]
        if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, P[m, c]))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (0.0, 0)


def masked_bce(logits, labels, valid):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = loss * valid
    return loss.sum() / valid.sum().clamp(min=1.0)

# ---------------------------------------------------------------------------
# One training phase (LP or FT), common to every member
# ---------------------------------------------------------------------------
def train_phase(member, loader, val_loader, device, epochs, lr, kind, label,
                freeze_encoder, amp=True, wd=1e-4):
    """AdamW (+ cosine schedule when unfreezing) over one LP-FT stage. Best state
    is captured on cal certain-label macro-AUROC -- the fs protocol's early-stop
    signal -- and restored before returning."""
    if freeze_encoder:
        if kind == "xrv":
            # NB: xrv DenseNet.features2 is a *method*; the encoder trunk module
            # is model.features -- classifier is the head.
            for p in member.model.features.parameters():
                p.requires_grad_(False)
        else:
            for p in member.encoder.parameters():
                p.requires_grad_(False)
    elif kind == "xrv":
        # an earlier LP phase may have frozen features -- re-enable everything
        for p in member.model.features.parameters():
            p.requires_grad_(True)
        for p in member.model.classifier.parameters():
            p.requires_grad_(True)
    else:
        for p in member.encoder.parameters():   # undo the LP-phase freeze
            p.requires_grad_(True)
    if kind == "xrv":
        params = [p for p in member.model.parameters() if p.requires_grad]
    else:
        params = list(member.trainable_parameters())
    trainable = sum(p.numel() for p in params)
    print(f"[{label}] trainable params: {trainable:,} "
          f"({'frozen encoder' if freeze_encoder else 'unfrozen'})")

    member.to(device)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
             if not freeze_encoder else None)
    scaler = torch.amp.GradScaler(enabled=amp and device.startswith("cuda"))

    state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
    best = macro_auroc_nih(member, val_loader, device, amp)[0]
    print(f"[{label}] epoch -1 (pre)  val_macro_auroc={best:.4f}")

    for ep in range(epochs):
        member.train()
        if kind == "xrv":
            member.model.train(not freeze_encoder)
        running, seen = 0.0, 0
        t0 = time.time()
        for imgs, labels, valid in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=amp and device.startswith("cuda")):
                z = member.preprocess(imgs)
                if kind == "xrv":
                    log14 = nih_logits(member, z)
                elif freeze_encoder:
                    with torch.no_grad():
                        feats = member.forward_features(z)
                    log14 = member.head(feats)
                else:
                    feats = member.forward_features(z)
                    log14 = member.head(feats)
                loss = masked_bce(log14, labels, valid)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * imgs.size(0)
            seen += imgs.size(0)
        if sched is not None:
            sched.step()
        val, n_cls = macro_auroc_nih(member, val_loader, device, amp)
        if val > best:
            best = val
            state = {k: v.detach().cpu().clone() for k, v in member.state_dict().items()}
        print(f"[{label}] epoch {ep}  loss={running/max(seen,1):.5f}  "
              f"val_macro_auroc={val:.4f} (n_cls={n_cls})  best={best:.4f}  "
              f"({time.time()-t0:.0f}s)")
        if not np.isfinite(running):
            raise RuntimeError(f"[{label}] non-finite loss at epoch {ep}")

    member.load_state_dict(state)
    print(f"[{label}] best val_macro_auroc={best:.4f} restored")
    return best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--member", required=True,
                    choices=["xrv_nih", "convnextv2", "raddino", "arkswin"])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lp-epochs", type=int, default=None,
                    help="default: 2 (xrv) / 3 (convnextv2) / 2 (raddino) / 3 (arkswin)")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="default: 3 (xrv) / 2 (convnextv2) / 1 (raddino) / 0 (arkswin)")
    ap.add_argument("--bs", type=int, default=0, help="0 = 64 (224px) / 16 (768px)")
    ap.add_argument("--lp-lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--train-cache", default=None,
                    help="224px memmap for 224px members (unused by arkswin)")
    ap.add_argument("--cal-cache", default=None,
                    help="224px memmap for the cal/val loader (unused by arkswin)")
    ap.add_argument("--cache-dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--train-subset", type=int, default=32000,
                    help="cap on train rows for arkswin's 768 PNG loader (seeded sample)")
    ap.add_argument("--png768-root", default="data/rex_adapted_png768",
                    help="verbatim-PNG subset root (paths stored without the "
                         "data/rexgradient/deid_png/ prefix)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train+cal rows (CPU smoke); 0 = full")

    args = ap.parse_args(argv)
    set_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ux224 = args.member != "arkswin"

    df = pd.read_parquet(args.manifest)
    dtr = df[df.split_role == "train"].reset_index(drop=True)
    dca = df[df.split_role == "cal"].reset_index(drop=True)
    from pathlib import Path as _P
    if args.member == "arkswin":
        # cache_from_parts stores subset PNGs under --png768-root WITHOUT the
        # data/rexgradient/deid_png prefix; remap manifest paths onto it.
        dtr = dtr.copy()
        dtr["image_path"] = dtr["image_path"].map(
            lambda p: str(_P(args.png768_root) / p.split("deid_png/", 1)[1]))
    if args.member == "arkswin" and args.train_subset and len(dtr) > args.train_subset:
        dtr = dtr.sample(n=args.train_subset, random_state=args.seed).reset_index(drop=True)
        print(f"[data] arkswin train subset: {len(dtr)} rows (seed={args.seed})")
    if args.member == "arkswin":
        # existence check AFTER the 32k subsample -- the subset root only holds
        # the sampled rows, not all 112,967 train rows
        miss = sum(1 for p in dtr["image_path"] if not _P(p).exists())
        assert miss == 0, f"{miss}/{len(dtr)} png768-subset files missing"
        print(f"[data] arkswin png768 root: {args.png768_root} ({len(dtr)} rows, all present)")
    if args.limit:
        dtr, dca = dtr.iloc[:args.limit], dca.iloc[: min(args.limit, len(dca))]

    train_ds = ManifestImageDataset(
        dtr, img_size=224 if ux224 else 768, augment=True, u_policy_ones=True,
        cache_path=args.train_cache if ux224 else None,
        cache_dtype=args.cache_dtype, limit=0)
    cal_ds = ManifestImageDataset(
        dca, img_size=224 if ux224 else 768, augment=False, u_policy_ones=False,
        cache_path=args.cal_cache if ux224 else None,
        cache_dtype=args.cache_dtype, limit=0)
    bs = args.bs or (16 if args.member == "arkswin" else 64)
    dl_kw = dict(batch_size=bs, num_workers=args.workers, collate_fn=collate,
                 pin_memory=device.startswith("cuda"), persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_kw)
    cal_loader = DataLoader(cal_ds, shuffle=False, **dl_kw)

    spec = ARCH_MEMBER_REGISTRY[args.member]
    member = build_member(spec, device=device)
    kind = "xrv" if args.member == "xrv_nih" else "probe"
    if kind == "probe":
        reset_head(member)                      # LP starts from a random head

    # per-member phase plan (Route A table in the docstring)
    if args.member == "xrv_nih":
        lp_ep = args.lp_epochs if args.lp_epochs is not None else 2
        ft_ep = args.ft_epochs if args.ft_epochs is not None else 3
    elif args.member == "convnextv2":
        lp_ep = args.lp_epochs if args.lp_epochs is not None else 3
        ft_ep = args.ft_epochs if args.ft_epochs is not None else 2
    elif args.member == "raddino":
        lp_ep = args.lp_epochs if args.lp_epochs is not None else 2
        ft_ep = args.ft_epochs if args.ft_epochs is not None else 1
    else:                                       # arkswin: frozen encoder, LP only
        lp_ep = args.lp_epochs if args.lp_epochs is not None else 3
        ft_ep = 0 if args.ft_epochs is None else args.ft_epochs

    name = args.member if args.member != "convnextv2" else f"convnextv2_s{args.seed}"
    log = {"member": args.member, "name": name, "seed": args.seed,
           "train_rows": len(train_ds), "cal_rows": len(cal_ds),
           "protocol": "masked_bce u_policy=ones, augment hflip+rot10, "
                       "early-stop on cal certain-label macro-AUROC"}
    history = []

    # ---- LP phase (head / xrv classifier only; frozen encoder) ----
    best = train_phase(member, train_loader, cal_loader, device,
                       epochs=lp_ep, lr=args.lp_lr, kind=kind, label=f"{name}/LP",
                       freeze_encoder=True, amp=args.amp, wd=args.wd)
    history.append({"phase": "LP", "best_cal_macro_auroc": best})

    # ---- FT phase (encoder unfrozen, low LR) except arkswin ----
    if ft_ep > 0:
        best = train_phase(member, train_loader, cal_loader, device,
                           epochs=ft_ep, lr=args.ft_lr, kind=kind, label=f"{name}/FT",
                           freeze_encoder=False, amp=args.amp, wd=args.wd)
        history.append({"phase": "FT", "best_cal_macro_auroc": best})

    ckpt = out_dir / f"{name}.pt"
    torch.save(member.state_dict(), ckpt)
    log["phases"] = history
    log["final_cal_macro_auroc"] = history[-1]["best_cal_macro_auroc"]
    (out_dir / f"{name}.trainlog.json").write_text(json.dumps(log, indent=2))
    print(f"[done] saved {ckpt}  final_cal_macro_auroc="
          f"{log['final_cal_macro_auroc']:.4f}")


if __name__ == "__main__":
    main()
