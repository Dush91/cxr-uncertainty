"""Inspect an uploaded Ark+ checkpoint to finalize ``members/ark_swin.py``.

Ark+ (github.com/jlianglab/Ark) ships weights in its own format (not a HF hub
id), so once the user uploads the file we need to determine:
  * file format (raw state_dict / {'state_dict': ...} / torch.save of a module /
    safetensors / .pth);
  * the parameter-name prefix + shapes -> infer the architecture (Swin-B vs
    Swin-L, patch/window config) and the pooled feature dim D;
  * whether there is a separate config.json / model card next to it describing
    input normalization (ImageNet vs CLIP vs custom) and resolution.

Usage:
    python scripts/inspect_ark_checkpoint.py --ckpt <path-or-dir> [--max-keys 40]

Prints a structured report. Run this, then fill the ``# <<INSPECT>>`` sections in
``cxr_uncertainty/members/ark_swin.py`` and set ``ARCH_MEMBER_REGISTRY["arkswin"]
.ckpt_path`` accordingly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _summarize_state_dict(sd, max_keys=40):
    keys = list(sd.keys())
    print(f"  #keys: {len(keys)}")
    # group by top-level prefix
    prefixes = {}
    for k in keys:
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print(f"  top-level prefixes: {prefixes}")
    # show first max_keys (name + shape + dtype)
    print(f"  first {min(max_keys, len(keys))} entries:")
    for k in keys[:max_keys]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        dtype = v.dtype if hasattr(v, "dtype") else ""
        print(f"    {k:60s} {shape}  {dtype}")
    # guess the pooled feature dim: look for a head/classifier weight or a final
    # norm + a 1-D bias that reveals the hidden dim.
    candidates = {}
    for k, v in sd.items():
        if hasattr(v, "shape"):
            if any(s in k.lower() for s in ("head", "classifier", "fc", "proj", "embeddings")):
                candidates[k] = tuple(v.shape)
    print(f"  head/proj-like tensors (feature-dim hints): {candidates}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="inspect_ark_checkpoint", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to the checkpoint file OR a dir containing it")
    p.add_argument("--max-keys", type=int, default=40)
    args = p.parse_args(argv)

    root = Path(args.ckpt)
    print(f"=== Ark+ checkpoint inspection: {root} ===")
    if root.is_dir():
        files = sorted(root.rglob("*"))
        print(f"  dir contents:")
        for f in files:
            if f.is_file():
                print(f"    {f.relative_to(root)}  ({f.stat().st_size/1e6:.1f} MB)")
        # pick the largest .pt/.pth/.bin/.safetensors as the likely weights
        cands = [f for f in files if f.suffix.lower() in (".pt", ".pth", ".bin", ".safetensors", ".ckpt")]
        if not cands:
            print("  no checkpoint file found in dir"); return
        ckpt = max(cands, key=lambda f: f.stat().st_size)
        print(f"  -> inspecting largest checkpoint: {ckpt.relative_to(root)} ({ckpt.stat().st_size/1e6:.1f} MB)")
        # also look for a config.json
        cfgs = [f for f in files if f.name.lower() in ("config.json", "model_config.json", "args.json", "model.yaml")]
        for c in cfgs:
            print(f"  config file: {c.relative_to(root)}")
            try:
                print("    " + json.dumps(json.load(open(c)), indent=2)[:2000])
            except Exception as e:
                print(f"    (could not parse: {e})")
        root = ckpt

    print(f"  file: {root}  ({root.stat().st_size/1e6:.1f} MB)")
    if root.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(root))
        print("  format: safetensors")
        _summarize_state_dict(sd, args.max_keys); return
    obj = torch.load(str(root), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        # common wrappers
        for wrap in ("state_dict", "model", "model_state", "encoder", "teacher", "ema"):
            if wrap in obj and isinstance(obj[wrap], dict):
                print(f"  format: dict with key '{wrap}' (other top keys: {[k for k in obj if k!=wrap]})")
                _summarize_state_dict(obj[wrap], args.max_keys); return
        # bare state_dict
        sample = next(iter(obj.values()))
        if hasattr(sample, "shape"):
            print("  format: bare state_dict (dict of tensors)")
            _summarize_state_dict(obj, args.max_keys); return
        print(f"  format: dict (non-tensor values). top-level keys: {list(obj.keys())[:20]}")
        print(f"  sample value types: { {k: type(v).__name__ for k,v in list(obj.items())[:5]} }")
        # could be a config dict
        try:
            print("  (as JSON): " + json.dumps({k: (str(v) if not isinstance(v,(int,float,str,bool,list,dict)) else v) for k,v in obj.items()}, indent=2)[:2000])
        except Exception as e:
            print(f"  (could not json-serialize: {e})")
    elif isinstance(obj, torch.nn.Module):
        print("  format: a torch.nn.Module (torch.save(model))")
        print(f"  module class: {type(obj).__name__}")
        _summarize_state_dict(obj.state_dict(), args.max_keys)
    else:
        print(f"  format: {type(obj).__name__} (unexpected)")


if __name__ == "__main__":
    main()