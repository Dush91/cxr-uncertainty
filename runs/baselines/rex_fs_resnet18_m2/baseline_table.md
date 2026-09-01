# Baseline comparison — rex_fs_resnet18_m2 (['resnet18_s0', 'resnet18_s1'])

Data: rex_fs_resnet18_m2, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| resnet18_s0 | **0.2542** | 0.3517 | 0.4075 | 0.4459 | **0.5649** | -0.0289 |
| resnet18_s1 | 0.2539 | **0.3775** | **0.4426** | **0.4748** | 0.5001 | +0.0000 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.7646 | 0.6491 | 0.6099 | **0.6159** | **0.6571** | +0.0000 |
| mutual_info | 0.6572 | 0.5828 | 0.5850 | 0.5843 | 0.6348 | -0.0316 |
| entropy | **0.8479**† | **0.7189** | **0.6320** | 0.6094 | 0.6388 | -0.0065 |
| confidence | 0.7444† | 0.6339 | 0.5515 | 0.4773† | 0.5334 | -0.1386 |
| mahalanobis | 0.5265† | 0.5215 | 0.5593 | 0.5599 | 0.4803 | -0.0560 |
| knn | 0.5955† | 0.5141 | 0.5319 | 0.5489 | 0.5330 | -0.0670 |
| energy | 0.3084 | 0.4779 | 0.5045 | 0.5128 | 0.5294 | -0.1031 |
| conf_resnet18_s0 | 0.6937 | 0.5999 | 0.5155 | 0.5122 | 0.5033 | -0.1037 |
| conf_resnet18_s1 | 0.6834 | 0.5677 | 0.4695 | 0.4059 | 0.4322 | -0.2100 |
| hybrid_std_knn | 0.7646 | 0.6491 | 0.6099 | **0.6159** | **0.6571** | +0.0000 |
| hybrid_std_maha | 0.7646 | 0.6491 | 0.6099 | **0.6159** | **0.6571** | +0.0000 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.1006 | 0.0748 | 0.8994 | 0.2184 |
| mutual_info | 0.1475 | 0.1217 | 0.8525 | 0.2184 |
| entropy | 0.0696 | 0.0438 | 0.9303 | 0.2184 |
| confidence | 0.0468 | 0.0210 | 0.9532 | 0.2184 |
| mahalanobis | 0.2140 | 0.1882 | 0.7860 | 0.2184 |
| knn | 0.1370 | 0.1112 | 0.8630 | 0.2184 |
| energy | 0.3489 | 0.3231 | 0.6511 | 0.2184 |
| conf_resnet18_s0 | 0.0567 | 0.0309 | 0.9433 | 0.2184 |
| conf_resnet18_s1 | 0.0601 | 0.0343 | 0.9399 | 0.2184 |
| hybrid_std_knn | 0.1006 | 0.0748 | 0.8994 | 0.2184 |
| hybrid_std_maha | 0.1006 | 0.0748 | 0.8994 | 0.2184 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8104 |
| mutual_info | 0.6927 |
| entropy | 0.9350 |
| confidence | 0.9122 |
| mahalanobis | 0.5266 |
| knn | 0.6487 |
| hybrid_std_knn | 0.8104 |
| hybrid_std_maha | 0.8104 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6714 |
| mutual_info | 0.6023 |
| entropy | 0.7401 |
| confidence | 0.6826 |
| mahalanobis | 0.5201 |
| knn | 0.5983 |
| energy | 0.3024 |
| hybrid_std_knn | 0.6714 |
| hybrid_std_maha | 0.6714 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8646
- ECE raw = 0.0272  ->  ECE TS = 0.0167
- confident-error AUROC of -pbar raw = 0.3731  TS = 0.3731  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7554, "auc_b": 0.9192, "z": -484.2937, "p": 0.0} |
| delong_full_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.7554, "auc_b": 0.8759, "z": null, "p": null} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7554, "auc_b": 0.5268, "z": 440.8376, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.7646, "auc_b": 0.7444, "z": 3.5943, "p": 0.0003} |
| delong_conf_top50_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.7646, "auc_b": 0.6937, "z": 24.3327, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.7646, "auc_b": 0.5265, "z": 66.7449, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6159, "auc_b": 0.4773, "z": 4.4165, "p": 0.0} |
| delong_conf_top10_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.6159, "auc_b": 0.5122, "z": 4.3238, "p": 0.0} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6159, "auc_b": 0.5599, "z": 1.7569, "p": 0.0789} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7474, 0.7812] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5304, 0.7029] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0311, "p": 0.8601} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.775, "grid_auroc_top50_cal": {"0.0": 0.609, "0.25": 0.651, "0.5": 0.6973, "0.75": 0.7477, "1.0": 0.775}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.146, 0.1934] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.2129, 0.2648] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.097, -0.0695] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0017, 0.0394] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.0327, 0.1618] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0559, 0.1622] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.0575, 0.0723] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.0195, 0.263] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.052, 0.0558] |
