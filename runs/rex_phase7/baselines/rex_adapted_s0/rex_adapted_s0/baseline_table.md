# Baseline comparison — rex_adapted_s0 (['xrv_nih', 'convnextv2', 'raddino', 'arkswin'])

Data: rex_adapted_s0, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| xrv_nih | 0.2209 | 0.3204 | 0.3632 | 0.3548 | 0.3463 | -0.1472 |
| convnextv2 | 0.2567 | 0.3614 | 0.4429 | 0.4835 | 0.4040 | -0.0185 |
| raddino | **0.2765** | 0.3460 | **0.4499** | **0.5020** | 0.4013 | +0.0000 |
| arkswin | 0.2735 | **0.4034** | 0.4357 | 0.4322 | **0.4057** | -0.0698 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.7917 | 0.7088 | 0.7092 | 0.6789 | 0.5755 | -0.0764 |
| mutual_info | 0.7428 | 0.6496 | 0.6529 | 0.6275 | 0.5513 | -0.1278 |
| entropy | **0.8307**† | **0.7738** | **0.7716** | **0.7553**† | **0.6513** | +0.0000 |
| confidence | 0.7165† | 0.6212 | 0.5952 | 0.5417† | 0.4081 | -0.2136 |
| mahalanobis | 0.5721† | 0.5072 | 0.5782 | 0.5460 | 0.3682 | -0.2093 |
| knn | 0.5861† | 0.5239 | 0.5606 | 0.4950† | 0.3382 | -0.2603 |
| energy | 0.3634 | 0.4775 | 0.4104 | 0.3911 | 0.4993 | -0.3642 |
| conf_xrv_nih | 0.6157 | 0.5652 | 0.5594 | 0.5581 | 0.5699 | -0.1972 |
| conf_convnextv2 | 0.6573 | 0.5864 | 0.5545 | 0.5126 | 0.4110 | -0.2427 |
| conf_raddino | 0.6790 | 0.6158 | 0.5828 | 0.5277 | 0.4110 | -0.2276 |
| conf_arkswin | 0.6726 | 0.6051 | 0.5328 | 0.4868 | 0.3635 | -0.2685 |
| hybrid_std_knn | 0.7917 | 0.7088 | 0.7092 | 0.6789 | 0.5755 | -0.0764 |
| hybrid_std_maha | 0.7917 | 0.7088 | 0.7092 | 0.6789 | 0.5755 | -0.0764 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0589 | 0.0414 | 0.9411 | 0.1812 |
| mutual_info | 0.0761 | 0.0586 | 0.9239 | 0.1812 |
| entropy | 0.0510 | 0.0335 | 0.9490 | 0.1812 |
| confidence | 0.0347 | 0.0172 | 0.9653 | 0.1812 |
| mahalanobis | 0.1349 | 0.1174 | 0.8651 | 0.1812 |
| knn | 0.1241 | 0.1066 | 0.8758 | 0.1812 |
| energy | 0.3173 | 0.2998 | 0.6827 | 0.1812 |
| conf_xrv_nih | 0.0698 | 0.0522 | 0.9302 | 0.1812 |
| conf_convnextv2 | 0.0503 | 0.0328 | 0.9497 | 0.1812 |
| conf_raddino | 0.0435 | 0.0260 | 0.9565 | 0.1812 |
| conf_arkswin | 0.0460 | 0.0284 | 0.9540 | 0.1812 |
| hybrid_std_knn | 0.0589 | 0.0414 | 0.9411 | 0.1812 |
| hybrid_std_maha | 0.0589 | 0.0414 | 0.9411 | 0.1812 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8714 |
| mutual_info | 0.7908 |
| entropy | 0.9296 |
| confidence | 0.9206 |
| mahalanobis | 0.6173 |
| knn | 0.6474 |
| hybrid_std_knn | 0.8714 |
| hybrid_std_maha | 0.8714 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.7133 |
| mutual_info | 0.6611 |
| entropy | 0.7637 |
| confidence | 0.7074 |
| mahalanobis | 0.5756 |
| knn | 0.6081 |
| energy | 0.2926 |
| hybrid_std_knn | 0.7133 |
| hybrid_std_maha | 0.7133 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.7843
- ECE raw = 0.0246  ->  ECE TS = 0.0034
- confident-error AUROC of -pbar raw = 0.2447  TS = 0.2447  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.8091, "auc_b": 0.9211, "z": -305.3719, "p": 0.0} |
| delong_full_epistemic_vs_best_member_raddino | {"auc_a": 0.8091, "auc_b": 0.8766, "z": -295.9817, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.8091, "auc_b": 0.6116, "z": 277.0021, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.7917, "auc_b": 0.7165, "z": 8.819, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_raddino | {"auc_a": 0.7917, "auc_b": 0.679, "z": 21.0788, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.7917, "auc_b": 0.5721, "z": 39.6951, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6789, "auc_b": 0.5417, "z": 2.7467, "p": 0.006} |
| delong_conf_top10_epistemic_vs_best_member_xrv_nih | {"auc_a": 0.6789, "auc_b": 0.5581, "z": 2.5289, "p": 0.0114} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6789, "auc_b": 0.546, "z": 4.3984, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7698, 0.8111] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5343, 0.7958] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0083, "p": 0.9273} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8131, "grid_auroc_top50_cal": {"0.0": 0.5817, "0.25": 0.6444, "0.5": 0.7108, "0.75": 0.7748, "1.0": 0.8131}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1732, 0.2347] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.1882, 0.248] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0475, -0.0315] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0555, 0.0933] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [0.0445, 0.3343] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.024, 0.277] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1525, -0.0138] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.038, 0.2315] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0232, 0.0254] |
