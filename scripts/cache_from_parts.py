#!/usr/bin/env python
"""Build the 224px fp16 train cache by STREAMING the deid_png zstd parts.

The Mac has ~69GB free -- not enough to store the ~67GB of train PNGs next to
the 155GB of parts. This skips the PNG stage entirely: a single pass over
``cat deid_png.part* | zstd -dc`` (one concatenated tar stream, 10 zstd frames)
decodes each target-role image straight into the destination memmap. The
bytes-based decode is bit-identical to ``cxr_uncertainty.utils.load_image_tensor``
for ReXGradient PNGs (verified in-session on cal-split images).

A seed-sampled *subset* of the ROLE's images is additionally copied verbatim
(raw PNG bytes) into ``--png768-root`` preserving its ``deid_png/...`` relative
path -- that is the arkswin native-768 linear-probe training subset, whose
loader needs real PNG files.

Usage:

    PYTHONPATH=. python scripts/cache_from_parts.py \
        --manifest data/rex_manifest_fs.parquet --role train \
        --parts 'data/rexgradient/deid_png.part*' \
        --out data/adapted_cache_train.npy --workers 8 \
        --png768-subset 32000 --png768-root data/rex_adapted_png768
"""
from __future__ import annotations

import argparse
import io
import subprocess
import tarfile
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_JOB_DECODE = None


def _decoder():
    """Worker-side decode: PNG bytes -> (1,224,224) float32 in [-1024,1024],
    mirroring cxr_uncertainty.utils.load_image_tensor's PNG branch exactly."""
    import imageio.v2 as imageio
    import torchvision
    import torchxrayvision as xrv

    t = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])

    def decode(buf: bytes) -> np.ndarray:
        img2d = imageio.imread(io.BytesIO(buf))
        img2d = img2d if img2d.ndim == 2 else img2d.mean(axis=2)
        if np.issubdtype(img2d.dtype, np.integer):
            maxval = float(np.iinfo(img2d.dtype).max)
        else:
            maxval = 65535.0 if float(img2d.max()) > 255 else 255.0
        img = img2d[None, :, :].astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = xrv.datasets.normalize(img, maxval)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = np.asarray(t(img)).astype(np.float32)
        return img

    return decode


def decode_job(buf: bytes) -> np.ndarray:
    global _JOB_DECODE
    if _JOB_DECODE is None:
        _JOB_DECODE = _decoder()
    return _JOB_DECODE(buf)


def norm_key(name: str) -> str | None:
    """tar member / manifest path -> the part after 'deid_png/'."""
    i = name.find("deid_png/")
    if i < 0:
        return None
    return name[i + len("deid_png/"):].rstrip("/") or None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default="data/rex_manifest_fs.parquet")
    ap.add_argument("--role", default="train")
    ap.add_argument("--parts", default="data/rexgradient/deid_png.part*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--png768-subset", type=int, default=0,
                    help="also copy N seeded-split verbatim PNGs (arkswin @768)")
    ap.add_argument("--png768-seed", type=int, default=0)
    ap.add_argument("--png768-root", default="data/rex_adapted_png768")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke); 0=all")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="abort the tar stream after N decoded rows (smoke)")
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    sub = df[df.split_role == args.role].reset_index(drop=True)
    paths = sub["image_path"].tolist()
    if args.limit:
        paths = paths[: args.limit]
    n = len(paths)
    H = W = args.img_size
    dt = np.dtype(args.dtype)
    print(f"[parts] role={args.role} n={n} out={args.out} dtype={args.dtype} "
          f"({n*H*W*dt.itemsize/1e9:.2f} GB)", flush=True)

    row_of = {}                       # norm_key -> cache row
    for i, p in enumerate(paths):
        k = norm_key(p)
        assert k, f"manifest path has no deid_png/ prefix: {p}"
        assert k not in row_of, f"duplicate path in manifest: {p}"
        row_of[k] = i

    subset = {}
    root = Path(args.png768_root)
    if args.png768_subset:
        if args.limit:                # smoke: first N rows, no resampling
            for i in range(min(args.limit, args.png768_subset)):
                subset[norm_key(paths[i])] = i
        else:
            rng = np.random.RandomState(args.png768_seed)
            for i in sorted(rng.choice(n, size=min(args.png768_subset, n),
                                       replace=False).tolist()):
                subset[norm_key(paths[i])] = i
        print(f"[parts] png768 subset: {len(subset)} imgs -> {root}", flush=True)

    arr_mm = np.memmap(args.out, dtype=dt, mode="w+", shape=(n, 1, H, W))

    parts = sorted(str(p) for p in Path().glob(args.parts))
    assert parts, f"no parts match {args.parts}"
    total_gb = sum(p.stat().st_size for p in map(Path, parts)) / 1e9
    print(f"[parts] streaming {len(parts)} parts ({total_gb:.0f} GB) ...", flush=True)
    # NB: only part00 starts a zstd frame -- parts 01+ are raw split shards of
    # the SAME frame, so they must be cat-fed into one zstd stream (as in
    # scripts/rex_extract_totmp.sh), not handed to zstd as separate args.
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    proc = subprocess.Popen(["zstd", "-dc"], stdin=cat.stdout,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cat.stdout.close()

    from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
    from multiprocessing import get_context
    ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"))
    futs = {}                         # future -> (row_idx, key)
    written = copied = seen = 0
    missing_keys = set()
    t0 = time.time()

    def drain(block: bool):
        nonlocal written
        if block:
            done, _ = wait(list(futs), return_when=FIRST_COMPLETED,
                           timeout=10 if not futs else None)
        else:
            done = {f for f in futs if f.done()}
        for f in done:
            idx, k = futs.pop(f)
            a = f.result()
            if a is None:
                missing_keys.add(k)
                continue
            arr_mm[idx, 0] = a.astype(dt)[0]
            written += 1

    tf = tarfile.open(fileobj=proc.stdout, mode="r|")
    try:
        for m in tf:
            if not m.isfile():
                continue
            k = norm_key(m.name)
            if k is None or (row_of.get(k) is None and k not in subset):
                continue
            seen += 1
            # read the member bytes ONCE (stream mode: a second extractfile() of
            # the same member returns nothing) -- subset rows are also cache rows
            buf = tf.extractfile(m).read()
            if k in subset:
                dest = root / k
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(buf)
                copied += 1
            if row_of.get(k) is not None:
                futs[ex.submit(decode_job, buf)] = (row_of[k], k)
                while len(futs) > 3 * args.workers:
                    drain(block=True)
                if written and written % 2000 == 0:
                    el = time.time() - t0
                    print(f"[progress] decoded={written}/{n} copied={copied} "
                          f"rate={written/el:.1f}/s eta={(n-written)*el/written/60:.0f} min",
                          flush=True)
                if args.stop_after and written >= args.stop_after:
                    print(f"[smoke] stop-after {args.stop_after} reached", flush=True)
                    break
        while futs:
            drain(block=True)
        tf.close()
    except tarfile.ReadError as e:
        print(f"[diag] ReadError: {e!r} | written={written} copied={copied} "
              f"seen={seen} pending={len(futs)} tar_offset={tf.offset} "
              f"zstd_rc={proc.returncode}", flush=True)
        raise
    finally:
        proc.terminate()
    print(f"[parts] stream done: members seen={seen} "
          f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    fails = len(missing_keys)
    assert written + fails == n, f"{written} decoded + {fails} failed != {n}"
    arr_mm.flush()
    with open((args.out[:-4] if args.out.endswith(".npy") else args.out)
              + ".paths.txt", "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"[parts] sidecar -> {args.out[:-4]}.paths.txt")
    if args.png768_subset:
        idx_list = sorted(subset.values())
        (root / "png768_subset.paths.txt").write_text(
            "\n".join(paths[i] for i in idx_list) + "\n")
        print(f"[parts] png768 subset sidecar -> {root}/png768_subset.paths.txt "
              f"({copied} files)")
    if fails:
        print(f"[parts] WARN {fails} decode failures; first 5: "
              f"{sorted(missing_keys)[:5]}")
    chk = np.memmap(args.out, dtype=dt, mode="r", shape=(n, 1, H, W))
    slab = chk[: min(1000, n)].astype(np.float32)
    print(f"[cache] wrote {args.out} shape={(n,1,H,W)} dtype={args.dtype} "
          f"size={Path(args.out).stat().st_size/1e9:.2f}GB finite="
          f"{bool(np.isfinite(slab).all())} range=[{float(slab.min()):.1f},"
          f"{float(slab.max()):.1f}]")
    print("PARTS_CACHE_DONE", f"decoded={written}/{n} subset={copied}")


if __name__ == "__main__":
    main()