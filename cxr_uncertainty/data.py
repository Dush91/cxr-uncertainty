"""Dataset loading.

Primary source: OpenI / Indiana University CXR collection (publicly hosted, no
credentials). torchxrayvision bundles the report XML + metadata (so labels are
free); only the PNG images must be present on disk. OpenI MeSH labels overlap 13
of the 14 NIH pathologies (all except Consolidation).

A secondary generic CSV loader is provided for NIH ChestX-ray14 (the user places
Data_Entry_2017.csv + images locally).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import NIH_PATHOLOGIES, NO_FINDING

# OpenI MeSH term -> NIH-14 pathology. Terms without an NIH-14 analog are dropped.
OPENI_TO_NIH: Dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Effusion": "Effusion",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Pleural_Thickening": "Pleural_Thickening",
    "Hernia": "Hernia",
    # Consolidation has no clean OpenI MeSH map -> intentionally absent.
}


@dataclass
class Sample:
    image_id: str
    image_path: str
    labels: Dict[str, int]   # pathology -> 0/1 over NIH_PATHOLOGIES
    view: str


def load_openi(
    imgpath: str,
    max_images: Optional[int] = None,
    views: Optional[List[str]] = None,
    seed: int = 0,
) -> List[Sample]:
    """Load OpenI samples with NIH-14-aligned multi-hot labels.

    imgpath: directory containing the extracted NLMCXR_png/*.png images.
    Uses Openi_Dataset's precomputed, synonym-aware ``labels`` matrix (aligned
    to its renamed ``pathologies`` list) rather than re-parsing MeSH strings.
    """
    import torchxrayvision as xrv
    views = views or ["PA", "AP"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xrv.datasets.Openi_Dataset(
            imgpath=imgpath,
            views=views,
            unique_patients=False,
            seed=seed,
        )
    df = ds.csv.reset_index(drop=True)
    labels_mat = np.asarray(ds.labels)           # (N, 18) aligned to ds.pathologies
    pathos = list(ds.pathologies)                # renamed (Lung Opacity, Lung Lesion, ...)
    # Build OpenI-name -> column index for the classes we care about.
    col = {name: i for i, name in enumerate(pathos)}

    samples: List[Sample] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        img_id = str(row["imageid"])
        img_path = os.path.join(imgpath, img_id + ".png")
        if not os.path.exists(img_path):
            continue
        labels: Dict[str, int] = {p: 0 for p in NIH_PATHOLOGIES}
        y = labels_mat[idx]
        for openi_key, nih_key in OPENI_TO_NIH.items():
            if openi_key in col and y[col[openi_key]] > 0:
                labels[nih_key] = 1
        view = str(row.get("view", "Unknown"))
        samples.append(Sample(image_id=img_id, image_path=img_path,
                              labels=labels, view=view))
        if max_images is not None and len(samples) >= max_images:
            break
    return samples


def load_nih_csv(
    csv_path: str,
    imgroot: str,
    max_images: Optional[int] = None,
) -> List[Sample]:
    """Load NIH ChestX-ray14 from Data_Entry_2017.csv + an image root.

    imgroot may contain the 12 images_00N subfolders or a flat set of PNGs; we
    glob for each Image Index.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Build a quick lookup of available files.
    available: Dict[str, str] = {}
    for root, _, files in os.walk(imgroot):
        for f in files:
            if f.lower().endswith(".png"):
                available[f] = os.path.join(root, f)
    samples: List[Sample] = []
    for _, row in df.iterrows():
        fname = str(row["Image Index"])
        finding = str(row["Finding Labels"])
        labels = {p: 0 for p in NIH_PATHOLOGIES}
        if finding != NO_FINDING:
            for tok in finding.split("|"):
                tok = tok.strip()
                if tok in labels:
                    labels[tok] = 1
        path = available.get(fname)
        if path is None:
            continue
        samples.append(Sample(image_id=fname, image_path=path,
                              labels=labels, view=str(row.get("View Position", "Unknown"))))
        if max_images is not None and len(samples) >= max_images:
            break
    return samples


def positive_counts(samples: List[Sample]) -> Dict[str, int]:
    """Count positive ground-truth cases per pathology (sanity check)."""
    counts = {p: 0 for p in NIH_PATHOLOGIES}
    for s in samples:
        for p, v in s.labels.items():
            if v:
                counts[p] += 1
    return counts