#!/usr/bin/env python
"""Pre-decode manifest images to a float32 memmap so from-scratch training reads
pre-decoded 224px tensors instead of re-decoding 16-bit PNGs every epoch.

Caches *exactly* what ``train_fromscratch.ManifestImageDataset.__getitem__``
computes *before* augmentation::

    load_image_tensor(path, 224, "cpu").squeeze(0)  -> (1, 224, 224) float32 in [-1024, 1024]

Augmentation (hflip / rotate) still runs per-epoch in the dataset; only the
expensive 16-bit PNG decode + resize is eliminated. On the H100 this took
training from ~196s/epoch (CPU-decode-bound, GPU idle at 0%) to a GPU-bound
regime.

The memmap is written in the SAME df order the dataset will use
(``df[df.split_role == role].reset_index(drop=True)``), so cache row ``i``
aligns with dataset row ``i``. A sidecar ``.paths.txt`` records the ordered
image_path list; the dataset asserts equality at load time so a stale or
re-ordered cache is caught immediately rather than silently corrupting labels.

Usage (run on the studio that holds the PNGs)::

    python scripts/build_train_cache.py --manifest data/rex_manifest_fs.parquet \\
        --role train --out data/fs_cache_train.npy --workers 16
    python scripts/build_train_cache.py --role cal \\
        --out data/fs_cache_val.npy --workers 8

The val/cal cache is small (~1.5 GB for 8k imgs); the train cache is ~22 GB for
113k imgs (float32 224x224). Both fit in the OS page cache after the first read.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cxr_uncertainty.utils import load_image_tensor  # noqa: E402


# Per-process memmap handle (opened once in the pool initializer so each forked
# worker reuses one read-write mapping rather than re-opening per task).
_ARR = None
_SHAPE = None
_DTYPE = np.float32


def _init_worker(out: str, shape: tuple, dtype: str):
    global _ARR, _SHAPE, _DTYPE
    _DTYPE = np.dtype(dtype)
    _ARR = np.memmap(out, dtype=_DTYPE, mode="r+", shape=shape)
    _SHAPE = shape


def _work(item):
    """Decode one image and store it at its row index in the shared memmap."""
    idx, path, img_size = item
    t = load_image_tensor(path, img_size, device="cpu")   # (1,1,H,W) float32 [-1024,1024]
    _ARR[idx] = t.squeeze(0).numpy().astype(_DTYPE)       # (1,H,W)
    return idx


def main(argv=None):
    p = argparse.ArgumentParser(prog="build_train_cache", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/rex_manifest_fs.parquet")
    p.add_argument("--role", required=True, help="split_role to cache (train | cal | eval)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out", required=True, help="output memmap .npy path (float32 (N,1,H,W))")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="memmap dtype; float16 halves the file (values in [-1024,1024] "
                        "keep ~3 significant digits, plenty for training)")
    p.add_argument("--sidecar", default=None,
                   help="ordered image_path list .txt (default: <out>.paths.txt)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="cap rows (smoke); 0=all")
    args = p.parse_args(argv)

    sidecar = args.sidecar or (args.out[:-4] if args.out.endswith(".npy") else args.out) + ".paths.txt"

    df = pd.read_parquet(args.manifest)
    sub = df[df.split_role == args.role].reset_index(drop=True)
    assert len(sub) > 0, f"no rows for split_role=='{args.role}' in {args.manifest}"
    paths = sub["image_path"].tolist()
    if args.limit:
        paths = paths[: args.limit]
    n = len(paths)
    H = W = args.img_size
    shape = (n, 1, H, W)
    dt = np.dtype(args.dtype)
    print(f"[cache] role={args.role} n={n} img_size={args.img_size} workers={args.workers} "
          f"out={args.out} ({n*1*H*W*dt.itemsize/1e9:.2f} GB {args.dtype})")

    # sanity: the manifest must point at real files
    sample = paths[: min(200, n)]
    miss = sum(1 for q in sample if not Path(q).exists())
    assert miss == 0, f"{miss}/{len(sample)} sample paths missing -- manifest points at absent files"

    # allocate the memmap file (creates / truncates), then close so workers open r+
    arr = np.memmap(args.out, dtype=dt, mode="w+", shape=shape)
    arr.flush()
    del arr

    items = [(i, paths[i], args.img_size) for i in range(n)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.out, shape, args.dtype)) as ex:
        futs = [ex.submit(_work, it) for it in items]
        for _ in tqdm(as_completed(futs), total=n, desc="decode", unit="img"):
            pass

    # sidecar: ordered paths (one per line) for the dataset's alignment assert
    with open(sidecar, "w") as f:
        f.write("\n".join(paths) + "\n")

    # verify a slice is finite + in range
    chk = np.memmap(args.out, dtype=dt, mode="r", shape=shape)
    slab = chk[: min(1000, n)].astype(np.float32)
    print(f"[cache] wrote {args.out} shape={shape} dtype={args.dtype} "
          f"size={os.path.getsize(args.out)/1e9:.2f}GB")
    print(f"[cache] sidecar -> {sidecar} ({n} paths)")
    print(f"[cache] verify[0:1000] finite={bool(np.isfinite(slab).all())} "
          f"range=[{float(slab.min()):.1f},{float(slab.max()):.1f}]")
    print("CACHE_DONE")


if __name__ == "__main__":
    main()