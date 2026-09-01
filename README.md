# Uncertainty-Aware Error Analysis for Chest X-ray Classification

A four-agent system that runs pretrained chest X-ray (CXR) classifiers, estimates
uncertainty, flags **confidently wrong** predictions, retrieves comparable prior
cases, explains itself, and proposes population-level improvements — with every
language-model claim mechanically checked against computed evidence.

This is the code artifact for an MSc dissertation (University of Surrey). The
dissertation sources are in [`dissertation/`](dissertation/); the consolidated
research log is [`FINDINGS.md`](FINDINGS.md).

> **No patient data or model weights are published here.** See
> [`PUBLISHING.md`](PUBLISHING.md) for what is excluded and why.

## The headline finding

Ensemble disagreement is widely assumed to flag confident errors that confidence
alone cannot. Measured at power on a leak-free 8,082-image split, it does not:
disagreement scores 0.5626 against pooled confidence's 0.5630 (DeLong *p* = 0.99).
A feature-density score that reached AUROC 0.87 on a 2,012-image split — five
confident errors — falls to 0.53 on a split with 34.

Building the missing experimental cells located the cause. A same-architecture
from-scratch ensemble *does* gain from disagreement (+0.05 to +0.14), but an
**architecture-diverse** from-scratch ensemble reproduces the deflation exactly
(0.5585 against 0.5626) despite the opposite training regime. At the operating
point that matters, the driver is architectural diversity, not training regime.
Adapting the pretrained ensemble to the deployment corpus restores the effect
(0.679 against 0.542, *p* = 0.006).

The resulting usage rule: **pooled confidence decides whether to abstain;
disagreement decides which confident prediction is wrong** — and the latter only
where members share an architecture or the ensemble is adapted in domain.

## The four agents

| Agent | Question | Where |
|---|---|---|
| 1 — error detection | should I trust this prediction? | `cxr_uncertainty/` (`uncertainty`, `calibration`, `conformal`, `feature_uq`, `repmismatch`, `risk`, `evaluate`) |
| 2 — similar-case retrieval | show me cases like this one | `scripts/build_retrieval_index*.py`, `scripts/query_retrieval.py`, `docs/agent2_retrieval.md` |
| 3 — error explanation | why might this be wrong? | `cxr_uncertainty/agent3.py`, `docs/agent3_explanation.md` |
| 4 — improvement suggestion | what should we change? | `cxr_uncertainty/agent4.py`, `scripts/agent4_batch.py`, `docs/agent4_improvement.md` |

Agents 3 and 4 are **constrained renderers**: deterministic code computes an
evidence pack whose every leaf is addressable (`EV:<section>.<key>`), the language
model must cite an identifier for every factual claim, and a zero-tolerance
mechanical audit checks citations, numbers, pathology names and quality bands
before anything reaches a reader. A failing rendering is replaced by a
deterministic template. The model never sees pixels.

There is **no autonomous planner**: the agents are composed by direct function
calls with typed hand-offs, so every edge is unit-tested.

## Install and run

CPU-only, Python 3.12. No build system; tests are pytest.

```bash
pip install -r requirements.txt

# Agent 1 — inference + evaluation (OpenI; images in data/openi/NLMCXR_png)
python -m cxr_uncertainty.cli --source openi --imgpath data/openi/NLMCXR_png \
    --arch-ensemble xrv_nih,convnextv2,raddino,arkswin --max-images 300 \
    --out runs/demo

# recompute calibrated metrics + plots from a saved run, no inference
python -m cxr_uncertainty.reanalyze runs/demo/per_record.csv --out runs/demo \
    --conf-pct 10 --unc-pct 50

# Agent 4 — population evidence pack, then the audited rendering
python scripts/agent4_batch.py --analyze --quality auto
python scripts/agent4_batch.py --render --backend template

# the deployed four-agent application
python scripts/demo_app_rex.py --regime adapted        # http://0.0.0.0:7862

python -m pytest tests/
```

Per-phase evaluation lives as one-off scripts under `scripts/` (for example
`scripts/eval_baselines.py`, `scripts/eval_conformal.py --write-sidecar`,
`scripts/train_fromscratch.py`).

## Architecture

**Registry seam** (`cxr_uncertainty/interfaces.py`). Uncertainty estimators,
calibrators and risk policies resolve by key through `UQ_ESTIMATORS`,
`CALIBRATORS` and `RISK_POLICIES`, populated by decorators. Five calibrators and
two risk policies were added over the project without touching the inference
path, which is what makes the calibration ablation in the dissertation an
ablation rather than a comparison of separate runs.

**Members and the ensemble** (`config.py`, `member_factory.py`, `models.py`,
`members/`). Two registries: a legacy per-corpus torchxrayvision DenseNet set,
and the architecture-primary set used for all reported results — `xrv_nih`,
`convnextv2` (ConvNeXt-V2-L, LP-FT), `raddino` (RAD-DINO ViT-B/14),
`arkswin` (Ark+ Swin-L @768), `biomedclip`. One shared tensor is loaded at the
ensemble's largest native size and resized per member; each member returns true
logits, probabilities, penultimate features and a per-class validity mask, and
`Alignment.stack` maps them onto the shared NIH-14 ontology.

**Invariants worth knowing before changing anything.**

- The torchxrayvision `all` weights were trained on a superset containing OpenI.
  They must never appear in an ensemble evaluating OpenI; `tests/test_leak_free.py`
  fails the build if they do.
- All members of one ensemble must share a probability scale. Mixing the legacy
  double-sigmoid convention with plain sigmoid *inverts* the cross-member
  disagreement signal, which is why the architecture-primary `xrv_nih` uses
  `prob_mode="plain"`.
- MC-dropout is a no-op for the DenseNet members (no dropout layers). The code
  detects this; `--mc-samples` only matters for architectures that have them.
- Production calibration is **TS-only** on the pooled mean. Per-member Beta was
  measured and rejected: it improves ECE and destroys the disagreement signal.
- `--risk-policy conformal_triage` requires a pre-fit sidecar from
  `scripts/eval_conformal.py --write-sidecar`.
- The retrieval vector store holds ids and vectors only — ground truth never
  enters the index (asserted in `tests/test_retrieval.py`).

## Data

Every corpus is obtained from its own source under its own licence; none is
redistributed here. NIH ChestX-ray14, CheXpert, Open-i and ReXGradient-160K are
the classification corpora; Kermany paediatric pneumonia and the COVID-19
Radiography Database are the distribution-shift probes. Splits are
**patient-disjoint** — one patient contributes up to 65 images, so an image-level
split would leak. `data/manifest.parquet` assigns roles A (train) / B (per-member
calibration) / C (ensemble calibration) / D (leak-free evaluation).

ReXGradient is used under a data-use agreement forbidding persistence on
third-party infrastructure: GPU work uploaded a verified cache, pulled the
checkpoints and arrays back, and deleted all patient data from the host,
verified by scan, after every phase.

## Documentation

| Document | Contents |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | the consolidated research log, phase by phase |
| [`docs/uncertainty_evaluation.md`](docs/uncertainty_evaluation.md) | evaluation methodology and the literature it follows |
| [`docs/agent2_retrieval.md`](docs/agent2_retrieval.md) | retrieval design, modes and validation gates |
| [`docs/agent3_explanation.md`](docs/agent3_explanation.md) | evidence pack, prompt contract and faithfulness audit |
| [`docs/agent4_improvement.md`](docs/agent4_improvement.md) | Agent 4's generated report over the evaluation split |
| [`dissertation/`](dissertation/) | the dissertation sources, figures and build script |
| [`PUBLISHING.md`](PUBLISHING.md) | what is excluded from this repository, and why |
