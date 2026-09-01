# Publishing this repository

This repository is published as the code artifact accompanying an MSc
dissertation. It is **code and results summaries only**. The working tree from
which it was produced is roughly 206 GB, of which about 192 GB is licensed
patient data and 8.8 GB is model checkpoints; none of that is published, and
the exclusions are a condition of the data-use agreements rather than a
convenience.

## What is excluded, and why

| Excluded | Reason |
|---|---|
| `data/` (~192 GB) | Chest radiographs from ReXGradient-160K, NIH ChestX-ray14, CheXpert, Open-i, Kermany and the COVID-19 Radiography Database. Each is obtainable from its own source under its own licence. ReXGradient in particular is distributed under a data-use agreement that forbids redistribution. |
| `runs/rex_labeling*/` | Contains **free-text radiology reports** and the labels derived from them. Report text is patient-derived content and is not republishable under any of the agreements. |
| `runs/**/*.npz`, `*.npy`, `*.parquet`, `*.csv` | Per-image derived arrays, manifests and label tables. These carry image identifiers and per-patient structure. |
| `runs/**/per_record.csv` | One row per image-pathology pair, keyed by image id. |
| `runs/**/index.bin`, `store.npz` | The retrieval vector store: 129,113 learned embeddings of patient radiographs. Embeddings are patient-derived representations and are treated as such. |
| `checkpoints/` (~8.8 GB) | Model weights, several of which were trained on licensed patient data and are therefore derived works of it. The pretrained encoders are separately obtainable from their original publishers. |
| `runs/app_rex_adapted/agent3_batch*/` | Agent 3's per-case faithfulness records. Each file is **named by** a ReXGradient DICOM instance UID and contains that image's measured quality metrics and a generated paragraph about it. The citable aggregate — case counts, failure counts, audit-failure kinds — is published instead as `runs/app_rex_adapted/agent3_gate_summary.json`, written by `scripts/redact_agent3_batch.py`, which carries no identifier. |
| `chexpert-labeler/`, `_extern/` | Third-party vendored tools with their own licences; obtain from upstream. |
| `.claude/`, `.cursor/`, `CLAUDE.md` | Private working material: session transcripts, plans and editor configuration. |
| University-issued `.docx` / `.pdf` | The dissertation template, the write-up guidance and the project brief belong to the School and the supervisor and are not ours to redistribute. |

## What is published

The `cxr_uncertainty` package, all experiment and analysis scripts, the test
suite, the documentation in `docs/`, the consolidated research log
`FINDINGS.md`, the dissertation sources in `dissertation/`, and the small JSON
and Markdown result summaries that carry no patient-derived content —
`baseline_comparison.json`, `baseline_table.md`, `evaluation*.json`,
`conformal_summary.json`, Agent 4's evidence pack and faithfulness record,
Agent 3's redacted gate summary, and the generated plots.

Agent 4's evidence pack is publishable **by construction**: both
modules are prohibited in code from emitting an image path or a patient
identifier, patient identity exists inside Agent 4 only as an anonymous integer
code used for clustered bootstrapping and never leaves the module, and the case
reference in Agent 3's pack is a truncated hash. Tests assert each of these
properties (`tests/test_agent3.py`, `tests/test_agent4.py`).

## Pre-publication check

Before pushing, confirm that no staged file carries patient-derived content:

```bash
# 1. dataset image identifiers (ReXGradient DICOM instance UIDs)
git ls-files -z | xargs -0 grep -lE '1\.2\.826\.0\.1\.[0-9.]{20,}'
# 2. per-image keys carrying values (matched as JSON/CSV keys, so that a file
#    which merely documents the invariant does not trip the check)
git ls-files -z | xargs -0 grep -lE '"(image_path|patient_id)" *:' | grep -vE '\.py$'
# 3. data formats that should never be staged
git ls-files | grep -E '\.(npz|npy|parquet|pt|pth|bin|csv|dcm)$'
# 4. anything under runs/ that is not an aggregate summary or a plot
git ls-files runs | grep -vE '\.(json|png|md)$'
# 5. credentials
git ls-files -z | xargs -0 grep -lniE 'sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}'
```

All five should return nothing. A denylist for `runs/` was tried first and
missed a renamed parquet backup carrying 129,113 rows keyed by DICOM UIDs;
`runs/` is therefore an **allowlist** (json, md, png only) with the aggregate
formats that still carry per-image identity re-excluded on top.

## Reproducing the results

Obtain the corpora from their own sources, follow the setup in `README.md`, and
use the commands in `dissertation/appendix.md` §A.2. Every result in Chapters 4
and 5 of the dissertation is recomputed from saved arrays without re-running
inference, so regenerating the arrays is the only expensive step.
