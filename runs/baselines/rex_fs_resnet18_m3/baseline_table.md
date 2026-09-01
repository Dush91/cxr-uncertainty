# Baseline comparison — rex_fs_resnet18_m3 (['resnet18_s0', 'resnet18_s1', 'resnet18_s2'])

Data: rex_fs_resnet18_m3, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| resnet18_s0 | 0.2542 | 0.3517 | 0.4075 | 0.4459 | 0.5649 | -0.0289 |
| resnet18_s1 | 0.2539 | 0.3775 | **0.4426** | **0.4748** | 0.5001 | +0.0000 |
| resnet18_s2 | **0.2597** | **0.3942** | 0.3926 | 0.3892 | **0.5858** | -0.0856 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.7923 | 0.6578 | 0.6456 | **0.6453** | 0.5953 | +0.0000 |
| mutual_info | 0.7115 | 0.6004 | 0.6177 | 0.6230 | 0.5571 | -0.0223 |
| entropy | **0.8367**† | **0.7098** | **0.6595** | 0.6434 | **0.6500** | -0.0019 |
| confidence | 0.7476† | 0.6225 | 0.5341 | 0.5455† | 0.5763 | -0.0998 |
| mahalanobis | 0.5278† | 0.5398 | 0.5721 | 0.5358 | 0.4046 | -0.1095 |
| knn | 0.5884† | 0.5169 | 0.5503 | 0.5223 | 0.4654 | -0.1230 |
| energy | 0.3292 | 0.5093 | 0.5329 | 0.5315 | 0.5564 | -0.1138 |
| conf_resnet18_s0 | 0.6706 | 0.5720 | 0.4923 | 0.5332 | 0.4861 | -0.1121 |
| conf_resnet18_s1 | 0.6639 | 0.5322 | 0.4422 | 0.4133 | 0.4231 | -0.2320 |
| conf_resnet18_s2 | 0.7046 | 0.5899 | 0.4932 | 0.5137 | 0.5616 | -0.1316 |
| hybrid_std_knn | 0.7923 | 0.6578 | 0.6456 | **0.6453** | 0.5953 | +0.0000 |
| hybrid_std_maha | 0.7923 | 0.6578 | 0.6456 | **0.6453** | 0.5953 | +0.0000 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0684 | 0.0464 | 0.9316 | 0.2023 |
| mutual_info | 0.0981 | 0.0761 | 0.9019 | 0.2023 |
| entropy | 0.0569 | 0.0348 | 0.9431 | 0.2023 |
| confidence | 0.0416 | 0.0196 | 0.9583 | 0.2023 |
| mahalanobis | 0.1998 | 0.1778 | 0.8002 | 0.2023 |
| knn | 0.1265 | 0.1045 | 0.8734 | 0.2023 |
| energy | 0.3300 | 0.3079 | 0.6700 | 0.2023 |
| conf_resnet18_s0 | 0.0547 | 0.0326 | 0.9453 | 0.2023 |
| conf_resnet18_s1 | 0.0588 | 0.0368 | 0.9412 | 0.2023 |
| conf_resnet18_s2 | 0.0507 | 0.0287 | 0.9493 | 0.2023 |
| hybrid_std_knn | 0.0684 | 0.0464 | 0.9316 | 0.2023 |
| hybrid_std_maha | 0.0684 | 0.0464 | 0.9316 | 0.2023 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8719 |
| mutual_info | 0.7758 |
| entropy | 0.9344 |
| confidence | 0.9125 |
| mahalanobis | 0.5258 |
| knn | 0.6501 |
| hybrid_std_knn | 0.8719 |
| hybrid_std_maha | 0.8719 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6964 |
| mutual_info | 0.6352 |
| entropy | 0.7407 |
| confidence | 0.6874 |
| mahalanobis | 0.5201 |
| knn | 0.5983 |
| energy | 0.3024 |
| hybrid_std_knn | 0.6964 |
| hybrid_std_maha | 0.6964 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8754
- ECE raw = 0.0241  ->  ECE TS = 0.0142
- confident-error AUROC of -pbar raw = 0.3403  TS = 0.3403  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.8117, "auc_b": 0.9186, "z": -220.9253, "p": 0.0} |
| delong_full_epistemic_vs_best_member_resnet18_s2 | {"auc_a": 0.8117, "auc_b": 0.8753, "z": -190.7583, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.8117, "auc_b": 0.5248, "z": 541.3251, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.7923, "auc_b": 0.7476, "z": 6.6002, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_resnet18_s2 | {"auc_a": 0.7923, "auc_b": 0.7046, "z": 18.2399, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.7923, "auc_b": 0.5278, "z": 70.8726, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6453, "auc_b": 0.5455, "z": 2.5158, "p": 0.0119} |
| delong_conf_top10_epistemic_vs_best_member_resnet18_s0 | {"auc_a": 0.6453, "auc_b": 0.5332, "z": 3.6686, "p": 0.0002} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6453, "auc_b": 0.5358, "z": 6.8671, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7723, 0.8126] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5434, 0.7433] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0085, "p": 0.9265} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8141, "grid_auroc_top50_cal": {"0.0": 0.5898, "0.25": 0.6386, "0.5": 0.6973, "0.75": 0.7663, "1.0": 0.8141}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [-0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1797, 0.2291] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.237, 0.2939] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0538, -0.0357] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0276, 0.0617] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.0087, 0.2601] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0156, 0.2414] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.0608, 0.06] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.0042, 0.1904] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0256, 0.0279] |
