# Baseline comparison — rex_fs_crossarch_rv (['resnet18_s0', 'vit_tiny_s0'])

Data: rex_fs_crossarch_rv, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| resnet18_s0 | **0.2542** | **0.3517** | **0.4075** | **0.4459** | **0.5649** | +0.0000 |
| vit_tiny_s0 | 0.2474 | 0.3242 | 0.3619 | 0.4086 | 0.4194 | -0.0373 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.7605 | 0.6602 | 0.5865 | 0.5266 | 0.4598 | -0.0864 |
| mutual_info | 0.6603 | 0.5770 | 0.5199 | 0.4859 | 0.4154 | -0.1271 |
| entropy | **0.8445**† | **0.7603** | **0.6778** | **0.6130** | 0.6572 | +0.0000 |
| confidence | 0.7494 | 0.6382 | 0.5730 | 0.5348 | 0.5813 | -0.0782 |
| mahalanobis | 0.5271† | 0.4873 | 0.4828 | 0.4685 | 0.5090 | -0.1445 |
| knn | 0.5902† | 0.4899 | 0.4877 | 0.4984 | 0.5211 | -0.1146 |
| energy | 0.3218 | 0.5030 | 0.5839 | 0.5729 | 0.5161 | -0.0401 |
| conf_resnet18_s0 | 0.7063 | 0.6272 | 0.5929 | 0.5634 | **0.6730** | -0.0496 |
| conf_vit_tiny_s0 | 0.6520 | 0.5614 | 0.5271 | 0.5400 | 0.4852 | -0.0730 |
| hybrid_std_knn | 0.7605 | 0.6602 | 0.5865 | 0.5266 | 0.4598 | -0.0864 |
| hybrid_std_maha | 0.7605 | 0.6602 | 0.5865 | 0.5266 | 0.4598 | -0.0864 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.1214 | 0.0940 | 0.8786 | 0.2247 |
| mutual_info | 0.1675 | 0.1401 | 0.8325 | 0.2247 |
| entropy | 0.0844 | 0.0570 | 0.9156 | 0.2247 |
| confidence | 0.0490 | 0.0216 | 0.9510 | 0.2247 |
| mahalanobis | 0.2148 | 0.1875 | 0.7851 | 0.2247 |
| knn | 0.1448 | 0.1174 | 0.8552 | 0.2247 |
| energy | 0.3611 | 0.3337 | 0.6389 | 0.2247 |
| conf_resnet18_s0 | 0.0638 | 0.0364 | 0.9362 | 0.2247 |
| conf_vit_tiny_s0 | 0.0722 | 0.0448 | 0.9278 | 0.2247 |
| hybrid_std_knn | 0.1214 | 0.0940 | 0.8786 | 0.2247 |
| hybrid_std_maha | 0.1214 | 0.0940 | 0.8786 | 0.2247 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.7884 |
| mutual_info | 0.6710 |
| entropy | 0.9305 |
| confidence | 0.9126 |
| mahalanobis | 0.5343 |
| knn | 0.6480 |
| hybrid_std_knn | 0.7884 |
| hybrid_std_maha | 0.7884 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6754 |
| mutual_info | 0.6136 |
| entropy | 0.7375 |
| confidence | 0.6853 |
| mahalanobis | 0.5201 |
| knn | 0.5983 |
| energy | 0.3024 |
| hybrid_std_knn | 0.6754 |
| hybrid_std_maha | 0.6754 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8306
- ECE raw = 0.0254  ->  ECE TS = 0.0115
- confident-error AUROC of -pbar raw = 0.387  TS = 0.387  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7183, "auc_b": 0.9197, "z": null, "p": null} |
| delong_full_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.7183, "auc_b": 0.8607, "z": null, "p": null} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7183, "auc_b": 0.5354, "z": 325.0988, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.7605, "auc_b": 0.7494, "z": 2.0253, "p": 0.0428} |
| delong_conf_top50_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.7605, "auc_b": 0.7063, "z": 33.0043, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.7605, "auc_b": 0.5271, "z": 83.797, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.5266, "auc_b": 0.5348, "z": -0.2629, "p": 0.7926} |
| delong_conf_top10_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.5266, "auc_b": 0.5634, "z": null, "p": null} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.5266, "auc_b": 0.4685, "z": 4.1505, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7411, 0.7773] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.4149, 0.6402] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0003, "p": 0.986} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.7632, "grid_auroc_top50_cal": {"0.0": 0.6018, "0.25": 0.6444, "0.5": 0.688, "0.75": 0.7349, "1.0": 0.7632}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1463, 0.1937] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.206, 0.2607] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0973, -0.0708] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [-0.0075, 0.0276] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.1199, 0.1852] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.096, 0.2091] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1851, 0.0145] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.1286, 0.1136] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0701, 0.0749] |
