"""Download leak-free OOD probes for Phase 4 and build a per-site manifest.

Two ungated HuggingFace mirrors (no Kaggle / no credentials needed):
  * Kermany pediatric pneumonia  -- ``hf-vision/chest-xray-pneumonia`` (parquet;
    5856 imgs, NORMAL/PNEUMONIA). Pediatric + scanner shift; the encoders saw
    only adult NIH/CheXpert/MIMIC/PadChest/BRAX -> leak-free.
  * COVID-19 radiography         -- ``hawking32/covid_chestXray`` (imagefolder;
    6432 imgs, COVID19/NORMAL/PNEUMONIA). COVID lesions no encoder ever saw ->
    strongest shift; leak-free.

Both reduce to the NIH **Pneumonia** class: gt(Pneumonia)=1 for PNEUMONIA/COVID19,
0 for NORMAL; all other NIH-14 classes are unlabeled on these sites -> valid mask
keeps ONLY Pneumonia (so the OOD confident-error AUROC is a clean single-class
Pneumonia measurement, directly comparable to the in-distribution Pneumonia
number). Calibration/thresholds are fit on OpenI role-C (in-distribution) and
applied here -- these sets are pure eval under shift.

Writes ``data/ood_manifest.parquet`` (image_id, image_path, source, site,
split, labels[14], valid[14]) + the images under ``data/ood/``.
"""
from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cxr_uncertainty.config import NIH_PATHOLOGIES

warnings.filterwarnings("ignore")
PNEUMONIA_IDX = NIH_PATHOLOGIES.index("Pneumonia")


def _empty_labels():
    z = np.zeros(len(NIH_PATHOLOGIES), dtype=np.float32)
    v = np.zeros(len(NIH_PATHOLOGIES), dtype=bool)
    v[PNEUMONIA_IDX] = True          # only Pneumonia is reliably labeled on OOD sites
    return z, v


def download_kermany(out: Path):
    """hf-vision/chest-xray-pneumonia (parquet). 0=NORMAL, 1=PNEUMONIA."""
    from datasets import load_dataset

    dest = out / "kermany"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    ds = load_dataset("hf-vision/chest-xray-pneumonia")
    for split in ds:
        feats = ds[split].features["label"].names
        for i, ex in enumerate(ds[split]):
            cls = feats[ex["label"]]                               # NORMAL | PNEUMONIA
            sub = dest / cls
            sub.mkdir(parents=True, exist_ok=True)
            fp = sub / f"{split}_{i:05d}.png"
            if not fp.exists():                                    # resumable: skip decoded
                ex["image"].convert("L").save(fp)
            z, v = _empty_labels()
            if ex["label"] == 1:                                   # PNEUMONIA
                z[PNEUMONIA_IDX] = 1.0
            rows.append({
                "image_id": f"kermany_{split}_{i:05d}",
                "image_path": str(fp), "source": "kermany", "site": "kermany",
                "split": split, "labels": z, "valid": v,
            })
    return rows


def download_covid(out: Path):
    """hawking32/covid_chestXray (imagefolder) via snapshot_download -- images
    stay as files (train|test / COVID19|NORMAL|PNEUMONIA / *.jpg)."""
    from huggingface_hub import snapshot_download

    # snapshot_download is resumable (already-fetched blobs are skipped). HF
    # rate-limits unauthenticated dataset downloads (HTTP 429) when concurrency
    # is high, so use few workers and retry with backoff until it converges.
    import time as _time

    repo = None
    for attempt in range(8):
        try:
            repo = snapshot_download("hawking32/covid_chestXray", repo_type="dataset",
                                     max_workers=2, retry_failed=True)
            break
        except Exception as e:                                   # ConnectionError / 429
            wait = 20 * (attempt + 1)
            print(f"[covid] snapshot attempt {attempt+1} failed ({type(e).__name__}); "
                  f"retrying in {wait}s ...")
            _time.sleep(wait)
    if repo is None:
        raise RuntimeError("covid snapshot_download failed after retries")
    repo = Path(repo)
    rows = []
    n = 0
    for split_dir in sorted(p for p in repo.iterdir() if p.is_dir() and p.name in ("train", "test")):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = cls_dir.name                       # COVID19 | NORMAL | PNEUMONIA
            for fp in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
                z, v = _empty_labels()
                if cls in ("PNEUMONIA", "COVID19"):  # radiographic opacity -> Pneumonia=1
                    z[PNEUMONIA_IDX] = 1.0
                rows.append({
                    "image_id": f"covid_{split_dir.name}_{cls}_{n:05d}",
                    "image_path": str(fp), "source": "covid", "site": "covid",
                    "split": split_dir.name, "labels": z, "valid": v,
                })
                n += 1
    return rows


def _kermany_rows_from_disk(out: Path):
    """Rebuild kermany manifest rows from the already-saved class folders
    (data/ood/kermany/{NORMAL,PNEUMONIA}/<split>_<i>.png) -- no HF access, no decode."""
    dest = out / "kermany"
    rows = []
    for cls_dir in sorted(p for p in dest.iterdir() if p.is_dir()):
        cls = cls_dir.name                          # NORMAL | PNEUMONIA
        for fp in sorted(cls_dir.glob("*.png")):
            stem = fp.stem                          # <split>_<i>
            split = stem.rsplit("_", 1)[0]
            z, v = _empty_labels()
            if cls == "PNEUMONIA":
                z[PNEUMONIA_IDX] = 1.0
            rows.append({
                "image_id": f"kermany_{stem}", "image_path": str(fp),
                "source": "kermany", "site": "kermany", "split": split,
                "labels": z, "valid": v,
            })
    return rows


def _covid_rows_from_snapshot(out: Path):
    """Rebuild covid rows from the HF snapshot dir (train|test / cls / *.jpg)."""
    import glob
    snaps = glob.glob(str(Path.home() / ".cache/huggingface/hub"
                          / "datasets--hawking32--covid_chestXray/snapshots/*"))
    if not snaps:
        return []
    repo = Path(snaps[0])
    rows = []
    n = 0
    for split_dir in sorted(p for p in repo.iterdir() if p.is_dir() and p.name in ("train", "test")):
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = cls_dir.name
            for fp in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
                z, v = _empty_labels()
                if cls in ("PNEUMONIA", "COVID19"):
                    z[PNEUMONIA_IDX] = 1.0
                rows.append({
                    "image_id": f"covid_{split_dir.name}_{cls}_{n:05d}",
                    "image_path": str(fp), "source": "covid", "site": "covid",
                    "split": split_dir.name, "labels": z, "valid": v,
                })
                n += 1
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(prog="download_ood", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/ood")
    p.add_argument("--manifest", default="data/ood_manifest.parquet")
    p.add_argument("--only", choices=["kermany", "covid", "both"], default="both")
    p.add_argument("--from-disk", action="store_true",
                   help="build the manifest by scanning on-disk images only (no HF "
                        "dataset loading / no decode); use after images are saved.")
    args = p.parse_args(argv)

    out = Path(args.out)
    rows = []
    if args.from_disk:
        if args.only in ("kermany", "both"):
            rows += _kermany_rows_from_disk(out)
            print(f"[kermany from-disk] {sum(1 for r in rows if r['source']=='kermany')} images")
        if args.only in ("covid", "both"):
            rows += _covid_rows_from_snapshot(out)
            print(f"[covid from-disk] {sum(1 for r in rows if r['source']=='covid')} images")
    else:
        if args.only in ("kermany", "both"):
            print("[kermany] downloading + saving images ...")
            rows += download_kermany(out)
            print(f"[kermany] {sum(1 for r in rows if r['source']=='kermany')} images")
        if args.only in ("covid", "both"):
            print("[covid] snapshot_download (imagefolder) ...")
            rows += download_covid(out)
            print(f"[covid] {sum(1 for r in rows if r['source']=='covid')} images")

    df = pd.DataFrame(rows)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.manifest, index=False)
    print(f"[done] manifest {len(df)} rows -> {args.manifest}")
    print("site counts:", df["site"].value_counts().to_dict())
    print("Pneumonia positives per site:",
          {s: int(df[df.site == s]["labels"].apply(lambda z: int(z[PNEUMONIA_IDX])).sum())
           for s in df.site.unique()})


if __name__ == "__main__":
    main()