#!/usr/bin/env python
"""1-image ensemble smoke test: build the 4-member production ensemble, forward
one ReXGradient PNG, on CPU (or GPU if available). Validates that every member
loads + the forward produces the expected tensor shapes on the Lightning
cloudspace env *before* GPU activation -- so a member build failure (timm model,
RAD-DINO under transformers 5.x, xrv weight download, checkpoint path) surfaces
now, not after the 16k-image inference run.
"""
import sys
import glob
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cxr_uncertainty.config import RiskConfig
from cxr_uncertainty.models import CXREnsemble
from cxr_uncertainty.utils import load_image_tensor
from cxr_uncertainty.feature_uq import extract_member

MEMBERS = ["xrv_nih", "convnextv2", "raddino", "arkswin"]

def main():
    pngs = sorted(glob.glob("data/rexgradient/deid_png/**/*.png", recursive=True))
    assert pngs, "no PNGs found under data/rexgradient/deid_png"
    path = pngs[0]
    print(f"[smoke] image={Path(path).name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RiskConfig(device=device, use_mc_dropout=False)
    print(f"[smoke] building ensemble (device={device}) ...")
    ens = CXREnsemble(MEMBERS, cfg=cfg)
    cfg.img_size = max(getattr(m, "native_size", 224) for m in ens.members)
    print(f"[smoke] device={device} img_size={cfg.img_size} members={list(ens.member_keys)}")
    for mb in ens.members:
        print(f"  - member {getattr(mb, 'key', '?'):12s} native={getattr(mb, 'native_size', '?')}")
    print(f"[smoke] ensemble built with {len(ens.members)} members")
    x = load_image_tensor(path, img_size=cfg.img_size, device=device)
    eo = ens.forward_ensemble_full(x)
    print(f"[smoke] per_member_probs shape={tuple(eo.per_member_probs.shape)}")
    lg, ft = extract_member(eo, "raddino")
    print(f"[smoke] raddino logits {tuple(lg.shape)} feats {tuple(ft.shape)}")
    assert eo.per_member_probs.shape[0] == len(MEMBERS), "member count mismatch"
    assert lg.shape[0] == 14, f"raddino logits expected 14, got {lg.shape}"
    assert ft.shape[0] == 1536, f"raddino feats expected 1536, got {ft.shape}"
    print("[smoke] OK")

if __name__ == "__main__":
    main()