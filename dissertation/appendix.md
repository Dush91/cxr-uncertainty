# Appendix

## A.1 Source code

The complete source code for the system described in this dissertation — the `cxr_uncertainty` package, the experiment and analysis scripts, the test suite, the four agents, the deployment application and the sources of this document — is published at:

`https://github.com/<USERNAME>/cxr-uncertainty`

Per the guidance, program listings are not reproduced here. The repository excludes all patient data, model checkpoints and derived arrays; `PUBLISHING.md` in the repository records what was excluded and why. In particular, no image from any corpus, no radiology report text and no patient identifier is published, in accordance with the data-use agreements described in Section 9.5.

## A.2 Reproducing the results

Every number in Chapters 4 and 5 is recomputed from saved arrays and requires no GPU and no inference pass. With the arrays in place, the principal commands are:

```
# Agent 1 -- inference and evaluation over a corpus
python -m cxr_uncertainty.cli --source openi --imgpath data/openi/NLMCXR_png \
    --arch-ensemble xrv_nih,convnextv2,raddino,arkswin --out runs/demo

# recompute calibrated metrics and plots from a saved run, no inference
python -m cxr_uncertainty.reanalyze runs/demo/per_record.csv --out runs/demo \
    --conf-pct 10 --unc-pct 50

# the Chapter 5 baseline battery, from saved arrays
python scripts/eval_baselines.py --out runs/baselines --conf-pct 10 \
    --datasets rex:runs/eval_arrays/rex/arrays_rexcal.npz:runs/eval_arrays/rex/arrays_rexeval.npz

# conformal triage (Section 4.8)
python scripts/eval_conformal.py --arrays-dir runs/phase4_features \
    --out runs/conformal --alpha 0.1 --write-sidecar

# Agent 4 (Chapter 8): population evidence pack, then the audited rendering
python scripts/agent4_batch.py --analyze --quality auto
python scripts/agent4_batch.py --render --backend template

# the deployed four-agent application (Section 9.1)
python scripts/demo_app_rex.py --regime adapted

# the test suite (Section 9.3)
python -m pytest tests/
```

## A.3 Artifact inventory

| Artifact | Location in the repository |
|---|---|
| Consolidated research log | `FINDINGS.md` |
| Evaluation methodology and literature notes | `docs/uncertainty_evaluation.md` |
| Agent 2 design and validation | `docs/agent2_retrieval.md` |
| Agent 3 design and validation | `docs/agent3_explanation.md` |
| Agent 4 generated report | `docs/agent4_improvement.md` |
| Chapter 5 baseline comparisons | `runs/baselines/*/baseline_comparison.json` |
| Adapted-ensemble results | `runs/rex_phase7/baselines/rex_adapted_s{0,1,2}/` |
| Conformal coverage results | `runs/conformal/conformal_summary.json` |
| Out-of-distribution detection results | `runs/ood_detection/ood_detection.json` |
| Agent 3 faithfulness gate summary | `runs/app_rex_adapted/agent3_gate_summary.json` (the per-case records are withheld; see `PUBLISHING.md`) |
| Agent 4 evidence pack and audit record | `runs/app_rex_adapted/agent4/` |
| Dissertation sources and figure generator | `dissertation/` |

## A.4 Nomenclature

| Term | Meaning |
|---|---|
| Confident stratum | The top *k*% of image-pathology records ranked by calibrated confidence |
| Confident-error AUROC | AUROC of an uncertainty score for separating wrong from correct predictions, computed within the confident stratum |
| Epistemic std | Cross-member standard deviation of the per-member probabilities |
| PU / AU / EU | Predictive, aleatoric and epistemic uncertainty in the Kendall-Gal decomposition |
| AURC / E-AURC | Area under the risk-coverage curve, and its excess over the oracle |
| AUAC | Area under the accuracy-coverage curve under progressive abstention |
| LAC | Least Ambiguous set-valued Classifier, the conformal score used here |
| Region A / Region B | The abstention decision and the confident-wrong triage decision (Section 5.7) |
| D-Ens | Deep ensemble: several independent random-initialisation trainings of one architecture |
