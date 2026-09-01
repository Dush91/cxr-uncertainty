#!/usr/bin/env python
"""Label ReXGradient-160K free-text reports with the official CheXpert
rule-based labeler (via Docker) -> 14-class labels with -1 (uncertain) preserved.

ReXGradient-160K ships **only free-text reports** (Indication / Comparison /
Findings / Impression), NOT pathology labels -- so the confident-error eval
needs per-pathology ground truth produced by a labeler. We use the canonical
**CheXpert rule-based labeler** (``stanfordmlgroup/chexpert-labeler``): the same
labeler that produced the CheXpert/MIMIC labels Baur et al. and the rest of the
field use, which keeps our ReXGradient labels methodologically comparable to
the literature (the deciding factor for choosing the rule-based labeler over
the learned CheXbert alternative). The labeler's ``-1`` (uncertain) output is
**preserved** here so ``eval_baselines`` can compute labeler-derived Baur Task 2
(uncertainty-label prediction) on ReXGradient as a secondary signal.

Labeler I/O (verified from the repo source, ``label.py`` / ``constants.py`` /
``stages/aggregate.py``):

  * INPUT  -- a headerless, single-column CSV; one report per row, quoted if it
    contains commas/newlines.
    ``python label.py --reports_path IN.csv --output_path OUT.csv --verbose``.
  * OUTPUT -- CSV with columns ``["Reports"] + CATEGORIES`` (14), cell values
    ``{1=positive, 0=negative, -1=uncertain, NaN=blank/unmentioned}``.
  * CATEGORIES (labeler output order) = ``["No Finding",
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]``.

This driver:

  1. Reads the ReXGradient per-split metadata CSV(s), extracts per-study report
     text (Findings + Impression by default -- the diagnostically informative
     sections), and writes the labeler's headerless input CSV plus a sidecar
     ``<split>_index.parquet`` (study_key -> row order) so the labeler output
     rows rejoin to studies exactly (row order is preserved by ``label.py``).
  2. Runs the labeler in its Docker image (isolates the NegBio / bllipparser /
     GENIA toolchain from this CPU-only Py3.12 host -- the labeler does not run
     natively on modern Python).
  3. Parses the labeled output into a normalized ``rex_labels.parquet``: one
     row per study with ``study_key``, ``split``, and the 14 CATEGORIES columns
     with values ``{1, 0, -1}`` (blank ``NaN`` -> ``0``, i.e. unmentioned is
     treated as negative; uncertain ``-1`` is kept). This parquet is the join
     input to ``scripts/build_rex_manifest.py``.

We label only the splits we need for eval -- default ``valid,test`` (20k reports
total). The 140k ``train`` split is not labeled (not needed for evaluation and
would be ~hours of bllipparser parsing).

ReXGradient metadata-schema unknowns (gated download not yet landed): the exact
metadata-CSV column names for the report sections, the study key, and whether
reports live in the metadata CSV or a separate file. These are exposed as CLI
args with documented defaults so they are filled at runtime once access is
granted -- no code edit required.

Usage (defaults assume the ReXGradient README's section names + StudyInstanceUid
as the study key -- StudyInstanceUid is what ``build_rex_manifest.py`` joins on
from ``view_position.json``; override once the real CSV headers are known):

    PYTHONPATH=. python scripts/label_rexgradient.py \
        --metadata-dir data/rexgradient/metadata \
        --out runs/rex_labeling

    # if the report sections live in a separate reports parquet keyed by study:
    PYTHONPATH=. python scripts/label_rexgradient.py \
        --metadata-dir data/rexgradient/metadata \
        --reports-parquet data/rexgradient/reports.parquet \
        --out runs/rex_labeling
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Labeler output column order (verified from constants/constants.py). The
# labeler writes ["Reports"] + CATEGORIES in this exact order; we read by
# name so order does not matter, but keeping it for the schema printout.
CHEXPERT_CATEGORIES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion",
    "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]
# Labeler value encoding (verified from constants/constants.py / aggregate.py):
#   1 = POSITIVE, 0 = NEGATIVE, -1 = UNCERTAIN, NaN = blank/unmentioned.
CHEX_POSITIVE, CHEX_NEGATIVE, CHEX_UNCERTAIN = 1, 0, -1

# Default column-name guesses (ReXGradient README mentions Indication /
# Comparison / Findings / Impression report sections). The study key MUST match
# the key ``build_rex_manifest.py`` joins on -- that script uses StudyInstanceUid
# from ``view_position.json`` as the join key, so default to it here (NOT
# AccessionNumber, which is not unique across patients in view_position.json's
# composite id). Override via CLI if the real CSV headers differ.
DEFAULT_STUDY_KEY_COL = "StudyInstanceUid"
DEFAULT_FINDINGS_COL = "Findings"
DEFAULT_IMPRESSION_COL = "Impression"


def _report_text(row: pd.Series, findings_col: str, impression_col: str) -> str:
    """Concatenate the Findings + Impression sections into one report string.
    The CheXpert labeler (NegBio) was built for CheXpert-style Findings/Impression
    reports; feeding both sections gives it the diagnostically informative text
    while omitting Indication/Comparison (reason-for-study / priors), which carry
    less finding signal. NaN sections -> empty."""
    parts = []
    for c in (findings_col, impression_col):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts) if parts else ""


def _load_split_reports(
    split: str, metadata_dir: Path, reports_parquet: Path | None,
    study_key_col: str, findings_col: str, impression_col: str,
) -> pd.DataFrame:
    """Return a frame with columns [study_key, report] for one split, in a
    stable row order. Reports come from a separate --reports-parquet if given,
    else from the per-split metadata CSV. study_key is coerced to str."""
    if reports_parquet is not None:
        rp = pd.read_parquet(reports_parquet)
        assert study_key_col in rp.columns, \
            f"--reports-parquet missing study-key col '{study_key_col}'"
        assert findings_col in rp.columns, \
            f"--reports-parquet missing findings col '{findings_col}'"
        df = rp[[study_key_col, findings_col, impression_col]].copy() \
            if impression_col in rp.columns else \
            rp[[study_key_col, findings_col]].copy()
    else:
        csv = metadata_dir / f"{split}_metadata.csv"
        assert csv.exists(), f"metadata CSV not found: {csv}"
        df = pd.read_csv(csv)
    assert study_key_col in df.columns, \
        f"split '{split}' missing study-key col '{study_key_col}'; " \
        f"available: {list(df.columns)}"
    df[study_key_col] = df[study_key_col].astype(str)
    # De-duplicate to one report per study (ReXGradient is study-centric; a study
    # with multiple images shares one report -> one label set propagated to its
    # images by build_rex_manifest). Keep first occurrence in file order.
    df = df.drop_duplicates(subset=[study_key_col], keep="first").reset_index(drop=True)
    df["report"] = df.apply(lambda r: _report_text(r, findings_col, impression_col), axis=1)
    return df[[study_key_col, "report"]].rename(columns={study_key_col: "study_key"})


def _write_labeler_input(df: pd.DataFrame, path: Path) -> None:
    """Write the headerless single-column CSV the labeler expects. Reports are
    csv-quoted automatically by pandas (handles commas / embedded newlines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["report"]].to_csv(path, index=False, header=False, quoting=1)  # QUOTE_ALL


def _run_labeler_docker(in_csv: Path, out_csv: Path, image: str) -> None:
    """Run ``label.py`` inside the chexpert-labeler Docker image, mounting the
    parent workdir to /data so input/output are visible to both host and container.
    """
    workdir = in_csv.resolve().parent
    rel_in = in_csv.name
    rel_out = out_csv.name
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{workdir}:/data",
        image,
        "python", "label.py",
        "--reports_path", f"/data/{rel_in}",
        "--output_path", f"/data/{rel_out}",
        "--verbose",
    ]
    print(f"[docker] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_labeler_local(in_csv: Path, out_csv: Path, labeler_dir: Path) -> None:
    """Run ``label.py`` from a local checkout (for users who set up the conda
    env instead of Docker). ``labeler_dir`` is the chexpert-labeler repo root."""
    cmd = [
        sys.executable, str(Path(labeler_dir) / "label.py"),
        "--reports_path", str(in_csv),
        "--output_path", str(out_csv),
        "--verbose",
    ]
    print(f"[local] {' '.join(cmd)}")
    env = dict(os.environ)
    # NegBio must be on PYTHONPATH per the labeler README.
    nb = env.get("NEGBIO_PATH")
    if nb:
        env["PYTHONPATH"] = nb + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env, cwd=str(labeler_dir))


def _parse_labeler_output(out_csv: Path, df_in: pd.DataFrame, split: str) -> pd.DataFrame:
    """Parse the labeled CSV into a normalized frame: one row per study with
    study_key + 14 CATEGORIES columns, values {1,0,-1} (blank NaN -> 0). Row
    order is preserved by label.py, so output row i rejoins to df_in row i."""
    lab = pd.read_csv(out_csv)
    # The labeler writes a leading "Reports" column + the 14 CATEGORIES.
    cat_cols = [c for c in CHEXPERT_CATEGORIES if c in lab.columns]
    missing = set(CHEXPERT_CATEGORIES) - set(cat_cols)
    assert not missing, f"labeler output missing categories: {missing}; " \
                       f"columns were: {list(lab.columns)}"
    assert len(lab) == len(df_in), \
        f"labeler output row count ({len(lab)}) != input ({len(df_in)}); " \
        f"row-order join would be misaligned -- aborting"
    out = pd.DataFrame({"study_key": df_in["study_key"].values, "split": split})
    for c in CHEXPERT_CATEGORIES:
        v = lab[c].values.astype(float)             # {1,0,-1,NaN}
        v = np.where(np.isnan(v), CHEX_NEGATIVE, v)  # blank -> 0 (negative)
        out[c] = v.astype(np.int8)                   # {1,0,-1}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata-dir", default="data/rexgradient/metadata",
                    help="dir with {train,valid,test}_metadata.csv (default: "
                         "data/rexgradient/metadata)")
    ap.add_argument("--reports-parquet", default=None,
                    help="separate reports parquet keyed by study (if reports "
                         "are NOT columns in the metadata CSV); uses the same "
                         "study/findings/impression col names")
    ap.add_argument("--splits", default="valid,test",
                    help="comma-separated ReXGradient splits to label "
                         "(default: valid,test -- the cal/eval splits we need; "
                         "skip the 140k train split)")
    ap.add_argument("--study-key-col", default=DEFAULT_STUDY_KEY_COL,
                    help=f"metadata/reports study-key column (default: "
                         f"{DEFAULT_STUDY_KEY_COL})")
    ap.add_argument("--findings-col", default=DEFAULT_FINDINGS_COL,
                    help=f"report Findings column (default: {DEFAULT_FINDINGS_COL})")
    ap.add_argument("--impression-col", default=DEFAULT_IMPRESSION_COL,
                    help=f"report Impression column (default: "
                         f"{DEFAULT_IMPRESSION_COL}); ignored if absent")
    ap.add_argument("--out", default="runs/rex_labeling",
                    help="workdir for labeler I/O + output parquet "
                         "(default: runs/rex_labeling)")
    ap.add_argument("--docker-image", default="chexpert-labeler:latest",
                    help="labeler Docker image (build with `docker build -t "
                         "chexpert-labeler:latest .` from the chexpert-labeler repo)")
    ap.add_argument("--no-docker", action="store_true",
                    help="run label.py from a local checkout (conda-env path) "
                         "instead of Docker; requires --labeler-dir")
    ap.add_argument("--labeler-dir", default=None,
                    help="path to a local chexpert-labeler checkout (for --no-docker)")
    ap.add_argument("--parse-only", action="store_true",
                    help="skip report extraction + labeler run; only re-parse an "
                         "existing labeled CSV (use after a manual labeler run)")
    ap.add_argument("--input-only", action="store_true",
                    help="extract reports + write the labeler input CSV and the "
                         "rex_index_<split>.parquet sidecar, but SKIP the labeler "
                         "run. Used by the Phase-6 parallel train-labeler wrapper: "
                         "shard the input CSV, run N labelers in parallel, concat "
                         "the labeled shards in order, then re-run with --parse-only. "
                         "Mutually exclusive with --parse-only.")
    args = ap.parse_args(argv)
    assert not (args.input_only and args.parse_only), \
        "--input-only and --parse-only are mutually exclusive"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    metadata_dir = Path(args.metadata_dir)
    reports_parquet = Path(args.reports_parquet) if args.reports_parquet else None

    all_labels = []
    for split in splits:
        print(f"\n=== split: {split} ===")
        in_csv = out_dir / f"rex_reports_{split}.csv"
        out_csv = out_dir / f"rex_reports_{split}_labeled.csv"
        idx_path = out_dir / f"rex_index_{split}.parquet"

        if args.parse_only:
            assert out_csv.exists(), f"--parse-only but {out_csv} not found"
            df_in = pd.read_parquet(idx_path) if idx_path.exists() else None
            if df_in is None:
                # fall back: re-extract just to get the study_key order
                df_in = _load_split_reports(split, metadata_dir, reports_parquet,
                                            args.study_key_col, args.findings_col,
                                            args.impression_col)
        else:
            df_in = _load_split_reports(
                split, metadata_dir, reports_parquet,
                args.study_key_col, args.findings_col, args.impression_col)
            # sidecar index preserves row order -> study_key for the join
            df_in[["study_key"]].to_parquet(idx_path)
            n_empty = int((df_in["report"].str.len() == 0).sum())
            print(f"[extract] {split}: {len(df_in)} unique studies, "
                  f"{n_empty} empty reports -> {in_csv.name}")
            _write_labeler_input(df_in, in_csv)

            if args.input_only:
                # Phase-6 parallel path: stop here. The wrapper shards the input
                # CSV, runs N labelers in parallel, concats the labeled shards in
                # order -> out_csv, then re-runs this script with --parse-only.
                print(f"[input-only] wrote {in_csv.name} + {idx_path.name}; "
                      f"skipping labeler run (shard + run in parallel, then "
                      f"--parse-only)")
                continue

            if args.no_docker:
                assert args.labeler_dir, "--no-docker requires --labeler-dir"
                _run_labeler_local(in_csv, out_csv, Path(args.labeler_dir))
            else:
                _run_labeler_docker(in_csv, out_csv, args.docker_image)

        lab = _parse_labeler_output(out_csv, df_in, split)
        all_labels.append(lab)
        # quick per-class prevalence printout
        pos = {c: int((lab[c] == CHEX_POSITIVE).sum()) for c in CHEXPERT_CATEGORIES}
        unc = {c: int((lab[c] == CHEX_UNCERTAIN).sum()) for c in CHEXPERT_CATEGORIES}
        print(f"[parse] {split}: {len(lab)} labeled studies")
        print(f"        top pos: {sorted(pos.items(), key=lambda kv: -kv[1])[:5]}")
        print(f"        top unc: {sorted(unc.items(), key=lambda kv: -kv[1])[:5]}")

    if args.input_only:
        print(f"\n[done] --input-only: wrote labeler input CSV(s) + index "
              f"sidecar(s) under {out_dir}; no rex_labels.parquet (run the "
              f"parallel labelers, then re-run with --parse-only to produce it)")
        return

    labels = pd.concat(all_labels, ignore_index=True)
    out_par = out_dir / "rex_labels.parquet"
    labels.to_parquet(out_par, index=False)
    print(f"\n[done] wrote {len(labels)} study-labels -> {out_par}")
    print(f"[schema] columns: {list(labels.columns)}")
    # manifest of what produced this, for reproducibility
    with open(out_dir / "label_run_meta.json", "w") as f:
        json.dump({
            "splits": splits, "labeler": "chexpert-labeler (rule-based, NegBio)",
            "docker_image": None if args.no_docker else args.docker_image,
            "study_key_col": args.study_key_col,
            "findings_col": args.findings_col, "impression_col": args.impression_col,
            "reports_parquet": str(reports_parquet) if reports_parquet else None,
            "value_encoding": {"positive": 1, "negative": 0, "uncertain": -1,
                                "blank->negative": 0},
            "n_studies": int(len(labels)),
        }, f, indent=2)
    print(f"[done] wrote label_run_meta.json")


if __name__ == "__main__":
    main()