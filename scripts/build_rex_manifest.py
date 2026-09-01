#!/usr/bin/env python
"""Build the ReXGradient-160K eval-only manifest (``data/rex_manifest.parquet``)
from the CheXpert-labeled reports produced by ``scripts/label_rexgradient.py``.

Joins per-study labels (from the labeler) to per-image rows and applies the
project's existing **CheXpert-14 -> NIH-14 ontology map**
(``data_curation.CHEXPERT_TO_NIH``). The ensemble predicts NIH-14; ReXGradient's
labels come from the CheXpert labeler (CheXpert-14). We **reuse the same map**
the project already uses for its CheXpert A/B role data, so ReXGradient is
treated identically to the CheXpert already ingested -- no ReXGradient-specific
override. The map yields **10 comparable NIH classes**: 7 exact 1:1
(Atelectasis, Cardiomegaly, Consolidation, Edema, Pneumonia, Pneumothorax,
Pleural Effusion -> Effusion) + 3 approximate (Lung Opacity -> Infiltration --
well justified, CheXpert introduced Lung Opacity as the Infiltration/opacity
umbrella; Lung Lesion -> Mass; Pleural Other -> Pleural_Thickening -- the one
genuinely soft link). The 4 NIH-only classes (Nodule, Emphysema, Fibrosis,
Hernia) have no CheXpert source -> ``valid=0``.

``valid`` is a **label-availability** mask (1 = this dataset can label this NIH
class via the map), not a model-output mask and not a certainty mask. The 10
mapped classes get ``valid=1``; the 4 NIH-only classes get ``valid=0``.
``eval_baselines._to_long`` skips rows where ``valid`` is False, so the 4
NIH-only classes are cleanly excluded from every metric. Crucially, **uncertain
(-1) rows keep ``valid=1``** -- certainty is orthogonal to availability, and
Task 2 needs to find them.

Labels are stored as a 14-vector in NIH-14 order with values ``{1, 0, -1}``
(-1 preserved from the labeler; blank -> 0). ``build_eval_arrays`` reads
``row.labels`` as ``float64`` so -1 survives into the arrays for Baur Task 2.

Output schema matches ``build_openi_manifest`` (``data_curation.py:231``) +
``data/ood_manifest.parquet`` precedent: an **eval-only** manifest by default
(no A/B/C/D leak-free role coupling -- those invariants in
``data/manifest.parquet`` are untouched). ``split_role`` is ``cal`` for the
native ``valid`` split and ``eval`` for the native ``test`` split (ReXGradient
public splits; the private ReXrank test set is unavailable). The two are
internal cal/eval used to fit per-class Youden thresholds / TS temperature /
hybrid weights in-distribution per dataset, mirroring Baur's within-MIMIC
protocol. ``--split-train`` optionally adds ``split_role="train"`` rows from the
native ``train`` split for the Phase-6 from-scratch D-Ens control (those rows
are never consumed by the eval path -- only by ``train_fromscratch.py``).

**Image/view source of truth = ``view_position.json``.** The ReXGradient
metadata CSV is *per-study* (one row per study: id, AccessionNumber,
StudyInstanceUid, PatientSex, PatientAge, StudyDate, StudyDescription,
Indication, Comparison, Findings, Impression) and carries NO per-image columns.
The per-image image-path + view come from ``<split>_metadata_view_position.json``:
a dict keyed by composite ``id``, each value holding ``StudyInstanceUid``,
``PatientID``, and the parallel lists ``ImagePath`` (per-image relpaths under
``../deid_png/<PatientID>/<AccessionNumber>/studies/<StudyUID>/series/<SeriesUID>/instances/<SOPUID>.png``)
and ``ImageViewPosition`` (per-image view code). We iterate that JSON, join
per-study labels by ``StudyInstanceUid`` (so ``label_rexgradient.py`` must key
its labels on StudyInstanceUid -- its ``--study-key-col`` default), and emit one
manifest row per kept image. No tree walk over the extracted image tree.

**Frontal-only by default.** The ensemble is trained on frontal CXRs; lateral /
oblique are OOD. Default ``--views AP,PA,POSTERO_ANTERIOR`` keeps the ~47% of
images that are frontal (~8k valid + ~8k test), far more power than OpenI's 2k.
Pass ``--views ''`` to keep all views (incl. lateral) if desired.

Usage:

    PYTHONPATH=. python scripts/build_rex_manifest.py \
        --labels runs/rex_labeling/rex_labels.parquet \
        --metadata-dir data/rexgradient/metadata \
        --img-root data/rexgradient \
        --out data/rex_manifest.parquet

    # then build arrays + run baselines:
    PYTHONPATH=. python scripts/build_eval_arrays.py \
        --manifest data/rex_manifest.parquet --source rex --out runs/eval_arrays
    PYTHONPATH=. python scripts/eval_baselines.py \
        --datasets rex:runs/eval_arrays/rex/arrays_rexcal.npz:runs/eval_arrays/rex/arrays_rexeval.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cxr_uncertainty.config import NIH_PATHOLOGIES  # noqa: E402
from cxr_uncertainty.data_curation import (  # noqa: E402
    CHEXPERT_TO_NIH, VALID_CHEXPERT,
)

NIH14: List[str] = list(NIH_PATHOLOGIES)
_NIH_IDX = {p: i for i, p in enumerate(NIH14)}

# ---------------------------------------------------------------------------
# CheXpert-14 -> NIH-14 ontology map. We REUSE the project's existing
# ``data_curation.CHEXPERT_TO_NIH`` map and its derived ``VALID_CHEXPERT``
# label-availability mask rather than a ReXGradient-specific override, for
# consistency: the project already maps CheXpert labels this exact way for the
# CheXpert A/B role data, so ReXGradient (also CheXpert-labeled) must be
# treated identically -- a stricter ReXGradient-only map would be an
# unjustified inconsistency. This yields 10 comparable NIH classes: 7 exact 1:1
# (Atelectasis, Cardiomegaly, Consolidation, Edema, Pneumonia, Pneumothorax,
# Pleural Effusion->Effusion) + 3 approximate (Lung Opacity->Infiltration,
# Lung Lesion->Mass, Pleural Other->Pleural_Thickening). The 4 NIH-only classes
# (Nodule, Emphysema, Fibrosis, Hernia) get valid=0.
#
# Caveat on the 3 approximate links: Lung Opacity->Infiltration is well
# justified (CheXpert introduced Lung Opacity as the Infiltration/opacity
# umbrella per the CheXpert paper) and Lung Lesion->Mass is a reasonable
# judgment; only Pleural Other->Pleural_Thickening is genuinely soft. Reusing
# the production map keeps ReXGradient consistent with how the project already
# handles CheXpert labels elsewhere.
# ---------------------------------------------------------------------------
# CHEXPERT_TO_NIH and VALID_CHEXPERT are imported from cxr_uncertainty.data_curation.
_NIH_ONLY_REX = [p for p in NIH14 if VALID_CHEXPERT[_NIH_IDX[p]] == 0]
assert _NIH_ONLY_REX == ["Nodule", "Emphysema", "Fibrosis", "Hernia"], _NIH_ONLY_REX
assert sum(VALID_CHEXPERT) == 10, f"expected 10 comparable classes, got {sum(VALID_CHEXPERT)}"

# Frontal view codes (the ensemble is trained on frontal CXRs; lateral/oblique
# are OOD). Derived from the valid-split ImageViewPosition distribution:
#   AP 5765, PA 2286, POSTERO_ANTERIOR 13  (frontal, ~47% of images)
#   LATERAL 6046, LL 198, LAO/RAO/LAT/RPO ~14, UNKNOWN 2153, None 380, N/A 152.
# Keeping AP+PA+POSTERO_ANTERIOR ~= 8k valid + 8k test images -- far more power
# than OpenI's 2k. Override with --views (pass '' to keep ALL incl. lateral).
FRONTAL_VIEWS = ("AP", "PA", "POSTERO_ANTERIOR")

CHEX_POSITIVE, CHEX_NEGATIVE, CHEX_UNCERTAIN = 1, 0, -1


def _chex_labels_to_nih(chex_vec: np.ndarray) -> np.ndarray:
    """Map a 14-vector in CheXpert-CATEGORIES order to a 14-vector in NIH-14
    order, values {1,0,-1} preserved. Unmapped CheXpert classes are dropped;
    NIH-only classes are 0 (and valid=0 so they're never scored)."""
    y = np.zeros(14, dtype=np.int8)
    for j, chex_cls in enumerate(_CHEX_CATEGORIES_ORDER):
        nih = CHEXPERT_TO_NIH.get(chex_cls)
        if nih is None:
            continue
        y[_NIH_IDX[nih]] = int(chex_vec[j])
    return y


# CheXpert labeler CATEGORIES order (must match label_rexgradient.CHEXPERT_CATEGORIES;
# duplicated here so this script does not depend on the labeler script's presence).
_CHEX_CATEGORIES_ORDER = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]


def _load_labels(paths) -> pd.DataFrame:
    """Load one or more rex_labels.parquet (from label_rexgradient) -> a single
    frame indexed by study_key with the 14 CATEGORIES columns. ``paths`` may be a
    single Path or an iterable of Paths (the train labels come from a separate
    parallel labeler run than the valid+test labels, so the from-scratch control
    merges them here). Duplicate study keys are de-duped (first wins)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["study_key"] = df["study_key"].astype(str)
        assert set(_CHEX_CATEGORIES_ORDER).issubset(df.columns), \
            f"labels parquet {p} missing CATEGORIES; columns: {list(df.columns)}"
        frames.append(df)
        print(f"[labels] loaded {len(df)} rows from {p}")
    lab = pd.concat(frames, ignore_index=True)
    lab = lab.drop_duplicates(subset="study_key", keep="first")
    return lab.set_index("study_key")


def _load_viewpos(split: str, metadata_dir: Path) -> dict:
    """Load ``<split>_metadata_view_position.json`` -- per-study records (dict
    keyed by the composite ``id``). Each value carries StudyInstanceUid,
    PatientID, and the parallel lists ``ImagePath`` (per-image relpaths under
    ``../deid_png/...``) and ``ImageViewPosition`` (per-image view code). This
    is the manifest source of truth: it gives every image path + view per study,
    so we never walk the extracted tree."""
    p = metadata_dir / f"{split}_metadata_view_position.json"
    assert p.exists(), f"view_position JSON not found: {p}"
    return json.loads(p.read_text())


def _resolve_image_path(img_root: Path, rel: str) -> Path:
    """ReXGradient ``ImagePath`` is relative to the metadata/ dir, e.g.
    ``../deid_png/<PatientID>/<AccessionNumber>/studies/<StudyUID>/series/<SeriesUID>/instances/<SOPUID>.png``.
    Strip the leading ``../`` and join under ``img_root`` (so it resolves to
    ``<img_root>/deid_png/...`` after ``cat deid_png.part* | tar -xf`` extracts
    under ``<img_root>/deid_png/``)."""
    rel = str(rel)
    if rel.startswith("../"):
        rel = rel[3:]
    elif rel.startswith("./"):
        rel = rel[2:]
    return img_root / rel


def build_manifest(
    labels_paths, metadata_dir: Path, img_root: Path,
    split_cal: str, split_eval: str, views: List[str],
    split_train: str = None,
) -> pd.DataFrame:
    """One manifest row per (kept) image. Per-study labels (keyed by
    StudyInstanceUid) are joined onto each image of the study via
    ``view_position.json``'s ImagePath/ImageViewPosition lists.

    ``labels_paths`` is a single path or list of rex_labels.parquet files
    (merged; the train labels come from a separate parallel labeler run than the
    valid+test labels). ``split_train`` (default None) optionally also emits
    ``split_role="train"`` rows from the native ReXGradient ``train`` split --
    used by the Phase-6 from-scratch D-Ens control (``train_fromscratch.py``
    reads role=="train"). The merged labels must cover the train-split keys."""
    lab = _load_labels(labels_paths)              # index = study_key (StudyInstanceUid)
    rows: List[dict] = []
    split_map = {split_cal: "cal", split_eval: "eval"}
    if split_train:
        split_map[split_train] = "train"
    for split, role in split_map.items():
        vp = _load_viewpos(split, metadata_dir)
        n_studies = len(vp)
        n_no_labels = n_kept = n_skipped_view = 0
        for key, rec in vp.items():
            suid = str(rec.get("StudyInstanceUid", "")).strip()
            if not suid or suid not in lab.index:
                n_no_labels += 1
                continue
            pid = str(rec.get("PatientID", key))
            chex_row = lab.loc[suid]
            chex_vec = np.asarray(
                [int(chex_row[c]) for c in _CHEX_CATEGORIES_ORDER], dtype=np.int8)
            y = _chex_labels_to_nih(chex_vec)
            paths = rec.get("ImagePath") or []
            iviews = rec.get("ImageViewPosition") or []
            if len(iviews) < len(paths):     # pad if view list shorter than path list
                iviews = list(iviews) + [None] * (len(paths) - len(iviews))
            for ip, view in zip(paths, iviews):
                view = str(view) if view is not None else "UNKNOWN"
                if views and view not in views:
                    n_skipped_view += 1
                    continue
                ipath = _resolve_image_path(img_root, ip)
                rows.append({
                    "image_id": ipath.stem,          # instance (SOP) UID
                    "image_path": str(ipath),
                    "source": "rex",
                    "split_role": role,
                    "patient_id": pid,
                    "view": view,
                    "labels": y.astype(int).tolist(),    # NIH-14, {1,0,-1}
                    "valid": list(VALID_CHEXPERT),       # label-availability mask
                    "encoder_input_kind": "xrv",
                })
                n_kept += 1
        msg = f"[{split}] {role}: {n_studies} studies, {n_no_labels} w/o labels (dropped), {n_kept} images kept"
        if views:
            msg += f", {n_skipped_view} dropped by view filter {list(views)}"
        print(msg)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default="runs/rex_labeling/rex_labels.parquet",
                    help="rex_labels.parquet from scripts/label_rexgradient.py "
                         "(keyed by StudyInstanceUid). Comma-separated list of "
                         "paths is supported -- e.g. valid+test labels + the "
                         "train labels from the parallel labeler "
                         "(runs/rex_labeling_train/rex_labels.parquet) for the "
                         "Phase-6 from-scratch control.")
    ap.add_argument("--metadata-dir", default="data/rexgradient/metadata",
                    help="dir with {split}_metadata_view_position.json")
    ap.add_argument("--img-root", default="data/rexgradient",
                    help="image root; ImagePath '../deid_png/...' resolves to "
                         "<img-root>/deid_png/... (default: data/rexgradient)")
    ap.add_argument("--split-cal", default="valid",
                    help="ReXGradient split to use as cal (default: valid)")
    ap.add_argument("--split-eval", default="test",
                    help="ReXGradient split to use as eval (default: test)")
    ap.add_argument("--split-train", default=None,
                    help="optional ReXGradient split to ALSO emit as "
                         "split_role='train' (default: None; set 'train' for the "
                         "Phase-6 from-scratch D-Ens control so train_fromscratch.py "
                         "has role=='train' rows). The labels parquet must cover it.")
    ap.add_argument("--views", default=",".join(FRONTAL_VIEWS),
                    help="comma-separated view codes to keep (default: frontal "
                         "'%s'; pass '' to keep ALL incl. lateral/oblique)"
                         % ",".join(FRONTAL_VIEWS))
    ap.add_argument("--out", default="data/rex_manifest.parquet")
    args = ap.parse_args(argv)

    views = [v.strip() for v in args.views.split(",") if v.strip()]

    print(f"[map] CheXpert-14 -> NIH-14: {sum(VALID_CHEXPERT)} comparable classes "
          f"(reusing data_curation.CHEXPERT_TO_NIH)")
    for chex, nih in CHEXPERT_TO_NIH.items():
        if nih:
            print(f"       {chex:24s} -> {nih}")
    print(f"[map] NIH-only (valid=0): {_NIH_ONLY_REX}")
    print(f"[view] {'keeping frontal only: ' + str(views) if views else 'keeping ALL views (incl. lateral)'}")

    label_paths = [Path(p.strip()) for p in args.labels.split(",") if p.strip()]
    df = build_manifest(label_paths, Path(args.metadata_dir),
                        Path(args.img_root), args.split_cal, args.split_eval, views,
                        split_train=args.split_train)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[done] {len(df)} image rows -> {out}")
    if len(df):
        for role in ("cal", "eval", "train"):
            sub = df[df["split_role"] == role]
            if len(sub):
                labs = np.asarray(sub["labels"].tolist(), dtype=np.int8)  # (n,14)
                print(f"[{role}] {len(sub)} images")
                for j, p in enumerate(NIH14):
                    v = np.asarray(sub["valid"].tolist())[:, j].astype(bool)
                    if not v.any():
                        continue
                    col = labs[:, j][v]
                    print(f"    {p:18s} valid={int(v.sum()):5d}  "
                          f"pos={int((col==1).sum()):5d}  "
                          f"unc={int((col==-1).sum()):5d}")
    print(f"[schema] columns: {list(df.columns)}")


if __name__ == "__main__":
    main()