"""Dataset download & curation (Part 1 of the CXR uncertainty build).

Produces a leak-aware, patient-level-split, NIH-14-aligned manifest from the
no-credential sources:

    NIH ChestX-ray14   (HF shrinusn77/nih-chest-xray, parquet)  -- roles A/B
    CheXpert           (HF danjacobellis/chexpert, parquet)       -- roles A/B
    OpenI / IU-Xray    (local data/openi via xrv Openi_Dataset)  -- roles C/D
    PadChest           (optional local path; user-provided)     -- roles A/B

MIMIC/BRAX/VinDr/CANDID-PTX are intentionally excluded (PhysioNet credentialed
=> CITI human-subjects training required; out of scope).

Design notes (see .claude/plans/cxr-part1-datasets.md):
  * Leak-free: OpenI is the only full-13/14-finding leak-free eval target; it is
    never used in role A/B. xrv ``-all`` weights are excluded elsewhere.
  * Patient-level splits (mandatory): no patient appears in two roles.
  * NIH-14 canonical ontology with a per-example validity mask (Consolidation=0
    on OpenI; Emphysema/Fibrosis/Hernia/Nodule=0 on CheXpert; etc.).
  * patient_id capture: NIH ``Patient ID``; CheXpert ``patient\\d+`` from Path;
    OpenI ``patientid`` (xrv CSV). The earlier data.py loaders ignored patient_id.
"""
from __future__ import annotations

import os
import re
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import NIH_PATHOLOGIES, NO_FINDING

NIH14 = NIH_PATHOLOGIES
_NIH_IDX = {p: i for i, p in enumerate(NIH14)}

# ---------------------------------------------------------------------------
# Ontology maps (CheXpert-14 -> NIH-14). PadChest is largely identity.
# Clean (~8): Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion,
#   Pneumonia, Pneumothorax, Pleural_Thickening.
# Approximate (~2): Lung Opacity -> Infiltration, Lung Lesion -> Mass.
# NIH-only (no CheXpert analog): Emphysema, Fibrosis, Hernia, Nodule.
# CheXpert-only (no NIH-14 analog, dropped): Enlarged Cardiomediastinum,
#   Fracture, Support Devices, No Finding.
# ---------------------------------------------------------------------------
CHEXPERT_CLASSES: List[str] = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
CHEXPERT_TO_NIH: Dict[str, Optional[str]] = {
    "No Finding": None,
    "Enlarged Cardiomediastinum": None,
    "Cardiomegaly": "Cardiomegaly",
    "Lung Opacity": "Infiltration",      # approximate
    "Lung Lesion": "Mass",                # approximate (Mass/Nodule -> Mass)
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
    "Pleural Other": "Pleural_Thickening",  # approximate
    "Fracture": None,
    "Support Devices": None,
}
# CheXpert class_label codes (verified from danjacobellis/chexpert README).
CHEX_UNLABELED, CHEX_UNCERTAIN, CHEX_ABSENT, CHEX_PRESENT = 0, 1, 2, 3

# Per-source validity: which NIH-14 classes a source can label. 1 = source
# provides a (possibly approximate) label for that class.
VALID_OPENI = [0] * 14
for p in NIH14:
    if p != "Consolidation":          # OpenI MeSH has no Consolidation map
        VALID_OPENI[_NIH_IDX[p]] = 1
VALID_NIH = [1] * 14                  # NIH-14 is the canonical source
VALID_CHEXPERT = [0] * 14
for chex, nih in CHEXPERT_TO_NIH.items():
    if nih is not None:
        VALID_CHEXPERT[_NIH_IDX[nih]] = 1
# PadChest validity is set from its labels at load time (default: identity).

# OpenI MeSH -> NIH (re-exported shape; the actual mapping uses xrv labels).
OPENI_TO_NIH: Dict[str, str] = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion", "Infiltration": "Infiltration", "Mass": "Mass",
    "Nodule": "Nodule", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
    "Edema": "Edema", "Emphysema": "Emphysema", "Fibrosis": "Fibrosis",
    "Pleural_Thickening": "Pleural_Thickening", "Hernia": "Hernia",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _multi_hot(findings: List[str]) -> List[int]:
    y = [0] * 14
    for f in findings:
        i = _NIH_IDX.get(f)
        if i is not None:
            y[i] = 1
    return y


def _save_image_bytes(b: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


def _streaming_image_dataset(repo: str, split: str = "train"):
    """Return a streaming HF dataset whose image column is NOT decoded (raw
    bytes dict), avoiding PIL decode + the datasets/PIL teardown crash. Uses
    ``cast_column`` (not ``features=``) so the rest of the schema is preserved."""
    from datasets import load_dataset, Image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = load_dataset(repo, split=split, streaming=True)
        ds = ds.cast_column("image", Image(decode=False))
    return ds


# ---------------------------------------------------------------------------
# Source: NIH ChestX-ray14 (HF shrinusn77/nih-chest-xray)
# ---------------------------------------------------------------------------
def download_nih(out_dir: Path, n: int = 12000, seed: int = 0) -> pd.DataFrame:
    """Stream shrinusn77/nih-chest-xray, save the first n images, return a
    metadata frame (Patient ID, View Position, NIH-14 multi-hot labels)."""
    img_dir = out_dir / "nih"
    rows: List[dict] = []
    ds = _streaming_image_dataset("shrinusn77/nih-chest-xray", split="train")
    for row in ds:
        if len(rows) >= n:
            break
        img = row["image"]                       # {"bytes":..., "path": "..."}
        fname = img.get("path") or f"nih_{len(rows):08d}.png"
        if not fname.endswith(".png"):
            fname = os.path.splitext(fname)[0] + ".png"
        ipath = img_dir / fname
        if not ipath.exists() and img.get("bytes"):
            _save_image_bytes(img["bytes"], ipath)
        labels = row.get("label", []) or []
        findings = [str(x) for x in labels if str(x) != NO_FINDING]
        rows.append({
            "image_id": fname,
            "image_path": str(ipath),
            "source": "nih",
            "patient_id": str(int(row["Patient ID"])),
            "view": str(row.get("View Position", "Unknown")),
            "labels": _multi_hot(findings),
            "valid": list(VALID_NIH),
            "encoder_input_kind": "xrv",
        })
    print(f"[nih] collected {len(rows)} images, "
          f"{pd.Series([r['patient_id'] for r in rows]).nunique()} patients")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source: CheXpert (HF danjacobellis/chexpert)
# ---------------------------------------------------------------------------
def download_chexpert(out_dir: Path, n: int = 8000, seed: int = 0,
                      u_policy: str = "ones") -> pd.DataFrame:
    """Stream danjacobellis/chexpert, save the first n images, return a
    metadata frame. u_policy in {ones, zeros}: uncertain (code 1) -> 1/0."""
    assert u_policy in ("ones", "zeros")
    img_dir = out_dir / "chexpert"
    rows: List[dict] = []
    ds = _streaming_image_dataset("danjacobellis/chexpert", split="train")
    pid_re = re.compile(r"patient(\d+)")
    for row in ds:
        if len(rows) >= n:
            break
        img = row["image"]
        # NOTE: image.path is just the basename (e.g. "view1_frontal.jpg") which
        # is NOT unique across patients. Build a unique filename from the full
        # CheXpert Path: ".../train/patient00001/study1/view1_frontal.jpg" ->
        # "patient00001_study1_view1_frontal.jpg".
        full_path = row.get("Path", "") or img.get("path") or ""
        ext = ".jpg" if str(full_path).lower().endswith(".jpg") else ".png"
        tail = full_path.split("/train/")[-1] if "/train/" in full_path else \
              os.path.basename(full_path)
        fname = tail.replace("/", "_") if "/" in tail else tail
        if not fname:
            fname = f"chex_{len(rows):08d}{ext}"
        if os.path.splitext(fname)[1] != ext:
            fname = os.path.splitext(fname)[0] + ext
        ipath = img_dir / fname
        if not ipath.exists() and img.get("bytes"):
            _save_image_bytes(img["bytes"], ipath)
        m = pid_re.search(full_path)
        patient_id = m.group(1) if m else f"unk_{len(rows)}"
        # map CheXpert class codes -> NIH-14 multi-hot
        y = [0] * 14
        for chex_cls in CHEXPERT_CLASSES:
            if chex_cls not in row:
                continue
            code = int(row[chex_cls])
            nih = CHEXPERT_TO_NIH.get(chex_cls)
            if nih is None:
                continue
            if code == CHEX_PRESENT:
                y[_NIH_IDX[nih]] = 1
            elif code == CHEX_UNCERTAIN and u_policy == "ones":
                y[_NIH_IDX[nih]] = 1
            # absent / unlabeled -> 0
        frontal = int(row.get("Frontal/Lateral", 0)) == 0
        view = "Frontal" if frontal else "Lateral"
        rows.append({
            "image_id": fname,
            "image_path": str(ipath),
            "source": "chexpert",
            "patient_id": patient_id,
            "view": view,
            "labels": y,
            "valid": list(VALID_CHEXPERT),
            "encoder_input_kind": "imagenet",
        })
    print(f"[chexpert] collected {len(rows)} images, "
          f"{pd.Series([r['patient_id'] for r in rows]).nunique()} patients")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source: OpenI (local data/openi, labels via xrv Openi_Dataset)
# ---------------------------------------------------------------------------
def build_openi_manifest(imgpath: Path, views: Optional[List[str]] = None) -> pd.DataFrame:
    """Build the OpenI manifest from on-disk PNGs using xrv's labeled csv.
    Frontal (PA/AP) only by default (~4k labeled images). patient_id from the
    xrv ``patientid`` column (== the CXR\\d+ study id)."""
    import torchxrayvision as xrv
    views = views or ["PA", "AP"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = xrv.datasets.Openi_Dataset(
            imgpath=str(imgpath), views=views, unique_patients=False, seed=0)
    df = d.csv.reset_index(drop=True)
    labels_mat = np.asarray(d.labels)        # (N, 18) aligned to d.pathologies
    col = {str(name): i for i, name in enumerate(d.pathologies)}
    rows: List[dict] = []
    for idx in range(len(df)):
        r = df.iloc[idx]
        img_id = str(r["imageid"])
        ipath = imgpath / (img_id + ".png")
        if not ipath.exists():
            continue
        y = [0] * 14
        for openi_key, nih_key in OPENI_TO_NIH.items():
            if openi_key in col and labels_mat[idx, col[openi_key]] > 0:
                y[_NIH_IDX[nih_key]] = 1
        rows.append({
            "image_id": img_id,
            "image_path": str(ipath),
            "source": "openi",
            "patient_id": str(r.get("patientid", img_id.split("_")[0])),
            "view": str(r.get("view", "Unknown")),
            "labels": y,
            "valid": list(VALID_OPENI),
            "encoder_input_kind": "xrv",
        })
    print(f"[openi] {len(rows)} labeled {views} images, "
          f"{pd.Series([rr['patient_id'] for rr in rows]).nunique()} patients")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Source: PadChest (optional, user-provided local subset)
# ---------------------------------------------------------------------------
def load_padchest_local(pad_dir: Path, labels_csv: Optional[Path] = None) -> pd.DataFrame:
    """Load a user-provided PadChest subset.

    Expects images in ``pad_dir`` and a labels csv with at least an image-id
    column and pathology columns. If ``labels_csv`` is None, looks for
    pad_dir/PADCHEST_chest_x_ray_images_labels.csv or pad_dir/labels.csv.
    Labels are mapped to NIH-14 by name (PadChest uses NIH names directly for
    the 14 classes). Edit PADCHEST_TO_NIH below if your subset uses different
    column names. Returns an empty frame if no labels csv is found.
    """
    if labels_csv is None:
        for cand in ("PADCHEST_chest_x_ray_images_labels.csv", "labels.csv"):
            if (pad_dir / cand).exists():
                labels_csv = pad_dir / cand
                break
    if labels_csv is None or not Path(labels_csv).exists():
        print(f"[padchest] no labels csv found under {pad_dir}; skipping PadChest")
        return pd.DataFrame()
    df = pd.read_csv(labels_csv)
    # Heuristic: use the first column as image id; map known NIH-named columns.
    id_col = df.columns[0]
    imgid_col = "ImageID" if "ImageID" in df.columns else id_col
    PADCHEST_TO_NIH = {p: p for p in NIH14}   # PadChest uses NIH names directly
    rows: List[dict] = []
    valid = [0] * 14
    for p in NIH14:
        if p in df.columns:
            valid[_NIH_IDX[p]] = 1
    for _, r in df.iterrows():
        img_id = str(r[imgid_col])
        ipath = pad_dir / (img_id + ".png")
        if not ipath.exists():
            ipath2 = pad_dir / (img_id + ".jpg")
            ipath = ipath2 if ipath2.exists() else ipath
        if not ipath.exists():
            continue
        y = [0] * 14
        for p in NIH14:
            if p in df.columns and pd.notna(r[p]) and str(r[p]).strip() in ("1", "1.0"):
                y[_NIH_IDX[p]] = 1
        rows.append({
            "image_id": img_id,
            "image_path": str(ipath),
            "source": "padchest",
            "patient_id": str(r.get("PatientID", r.get("patient_id", img_id))),
            "view": str(r.get("Projection", r.get("ViewPosition", "Unknown"))),
            "labels": y,
            "valid": list(valid),
            "encoder_input_kind": "xrv",
        })
    print(f"[padchest] {len(rows)} images, "
          f"{pd.Series([rr['patient_id'] for rr in rows]).nunique()} patients")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Patient-level role splits
# ---------------------------------------------------------------------------
def _patient_split_within(df: pd.DataFrame, role_a_size: int,
                          role_b_size: int, seed: int = 0) -> pd.DataFrame:
    """Patient-disjoint A/B split within a single source (NIH/CheXpert/PadChest).
    Assigns whole patients to A or B so no patient is in both."""
    rng = np.random.default_rng(seed)
    patients = sorted(df["patient_id"].unique())
    rng.shuffle(patients)
    # accumulate patients into A until role_a_size images, then B
    a_pids, b_pids, a_cnt, b_cnt = set(), set(), 0, 0
    for pid in patients:
        n_imgs = int((df["patient_id"] == pid).sum())
        if a_cnt < role_a_size:
            a_pids.add(pid); a_cnt += n_imgs
        elif b_cnt < role_b_size:
            b_pids.add(pid); b_cnt += n_imgs
        else:
            break
    role = np.where(df["patient_id"].isin(a_pids), "A",
                   np.where(df["patient_id"].isin(b_pids), "B", "unused"))
    df = df.copy()
    df["split_role"] = role
    return df[df["split_role"] != "unused"].reset_index(drop=True)


def _openi_split(df: pd.DataFrame, cal_size: int, seed: int = 0) -> pd.DataFrame:
    """Patient-level C (cal) / D (eval) split of OpenI, prevalence-stratified
    by total positive count per patient (balanced alternating assignment)."""
    rng = np.random.default_rng(seed)
    g = df.groupby("patient_id")["labels"].apply(
        lambda s: int(np.sum([sum(x) for x in s])))
    pids = list(g.index)
    # shuffle then sort by positive count, then alternate C/D to balance
    order = list(pids)
    rng.shuffle(order)
    order.sort(key=lambda p: g[p])
    c_pids, d_pids = set(), set()
    c_cnt, d_cnt = 0, 0
    for pid in order:                      # alternate, but cap C at cal_size
        n_imgs = int((df["patient_id"] == pid).sum())
        if c_cnt < cal_size and (c_cnt <= d_cnt or d_cnt >= len(df) - cal_size):
            c_pids.add(pid); c_cnt += n_imgs
        else:
            d_pids.add(pid); d_cnt += n_imgs
    role = np.where(df["patient_id"].isin(c_pids), "C", "D")
    df = df.copy()
    df["split_role"] = role
    return df.reset_index(drop=True)


def make_splits(nih: pd.DataFrame, chex: pd.DataFrame, openi: pd.DataFrame,
                padchest: Optional[pd.DataFrame] = None,
                nih_a: int = 10000, nih_b: int = 1000,
                chex_a: int = 6000, chex_b: int = 1000,
                openi_cal: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Patient-level role assignment across all sources -> one manifest df."""
    parts: List[pd.DataFrame] = []
    if not nih.empty:
        parts.append(_patient_split_within(nih, nih_a, nih_b, seed=seed))
    if not chex.empty:
        parts.append(_patient_split_within(chex, chex_a, chex_b, seed=seed + 1))
    if padchest is not None and not padchest.empty:
        parts.append(_patient_split_within(padchest, 4000, 1000, seed=seed + 2))
    if not openi.empty:
        parts.append(_openi_split(openi, openi_cal, seed=seed + 3))
    manifest = pd.concat(parts, ignore_index=True)
    return manifest


# ---------------------------------------------------------------------------
# Leak-free assertions
# ---------------------------------------------------------------------------
def assert_leak_free(manifest: pd.DataFrame) -> None:
    """Hard checks: no OpenI patient in roles A/B; no OpenI image in train; the
    xrv -all weights are not used for OpenI eval (checked in config, asserted
    here as a documented invariant)."""
    openi_pids = set(manifest.loc[manifest.source == "openi", "patient_id"])
    train = manifest[manifest.split_role.isin(["A", "B"])]
    leaked = train[train.source != "openi"].copy()
    # OpenI images must never be in A/B
    assert not manifest.query("source == 'openi' and split_role in ['A','B']").any().any(), \
        "OpenI images leaked into train/role A or B"
    # OpenI patients must not coincide with any train patient (cross-source by
    # patient_id string; OpenI patient ids are 'CXRNN' which won't match NIH/
    # CheXpert numeric ids, so this is a structural guard).
    train_pids = set(train["patient_id"])
    overlap = openi_pids & train_pids
    assert not overlap, f"patient_id overlap OpenI<->train: {sorted(overlap)[:5]}"
    # Roles C and D must be patient-disjoint within OpenI
    oi = manifest[manifest.source == "openi"]
    if not oi.empty:
        c_pids = set(oi[oi.split_role == "C"]["patient_id"])
        d_pids = set(oi[oi.split_role == "D"]["patient_id"])
        assert not (c_pids & d_pids), "OpenI C/D patient overlap"
    # Roles A and B must be patient-disjoint within each train source
    for src in ["nih", "chexpert", "padchest"]:
        s = manifest[manifest.source == src]
        if s.empty:
            continue
        a_pids = set(s[s.split_role == "A"]["patient_id"])
        b_pids = set(s[s.split_role == "B"]["patient_id"])
        assert not (a_pids & b_pids), f"{src} A/B patient overlap"
    print("[leak-free] all assertions passed")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
MANIFEST_COLS = ["image_id", "image_path", "source", "patient_id", "split_role",
                 "view", "labels", "valid", "encoder_input_kind"]


def build_manifest(out_dir: Path = Path("data"),
                   n_nih: int = 12000, n_chex: int = 8000,
                   openi_imgpath: Path = Path("data/openi"),
                   padchest_path: Optional[Path] = None,
                   u_policy: str = "ones", seed: int = 0) -> pd.DataFrame:
    """Download/collect sources, map to NIH-14, split patient-level, write
    data/manifest.parquet, run leak-free assertions, return the manifest."""
    out_dir = Path(out_dir)
    raw = out_dir / "raw"
    nih = download_nih(raw, n=n_nih, seed=seed)
    chex = download_chexpert(raw, n=n_chex, seed=seed, u_policy=u_policy)
    openi = build_openi_manifest(Path(openi_imgpath))
    padchest = (load_padchest_local(Path(padchest_path))
                if padchest_path else pd.DataFrame())
    manifest = make_splits(nih, chex, openi, padchest, seed=seed)
    manifest = manifest[MANIFEST_COLS]
    assert_leak_free(manifest)
    out = out_dir / "manifest.parquet"
    manifest.to_parquet(out, index=False)
    _print_summary(manifest, out)
    return manifest


def _print_summary(manifest: pd.DataFrame, out: Path) -> None:
    print(f"\n=== manifest written: {out} ({len(manifest)} rows) ===")
    by = manifest.groupby(["source", "split_role"]).size()
    print(by.to_string())
    # prevalence per class on the leak-free eval set (role D)
    d = manifest[manifest.split_role == "D"]
    if not d.empty:
        y = np.array(d["labels"].tolist())
        v = np.array(d["valid"].tolist())
        pos = y.sum(axis=0)
        print("Role D (OpenI eval) positives per class:")
        for i, p in enumerate(NIH14):
            print(f"  {p:20s} valid={int(v[:,i].sum()):4d} pos={int(pos[i]):4d}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Download/curate CXR datasets -> manifest")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-nih", type=int, default=12000)
    ap.add_argument("--n-chex", type=int, default=8000)
    ap.add_argument("--openi-imgpath", default="data/openi")
    ap.add_argument("--padchest-path", default=None)
    ap.add_argument("--u-policy", choices=["ones", "zeros"], default="ones")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    build_manifest(Path(args.out), args.n_nih, args.n_chex,
                   Path(args.openi_imgpath),
                   Path(args.padchest_path) if args.padchest_path else None,
                   args.u_policy, args.seed)


if __name__ == "__main__":
    main()