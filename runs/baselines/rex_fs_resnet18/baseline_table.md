# Baseline comparison — rex_fs_resnet18 (['resnet18_s0', 'resnet18_s1', 'resnet18_s2', 'resnet18_s3', 'resnet18_s4'])

Data: rex_fs_resnet18, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| resnet18_s0 | 0.2542 | 0.3517 | 0.4075 | 0.4459 | 0.5649 | -0.0289 |
| resnet18_s1 | 0.2539 | 0.3775 | 0.4426 | **0.4748** | 0.5001 | +0.0000 |
| resnet18_s2 | 0.2597 | **0.3942** | 0.3926 | 0.3892 | **0.5858** | -0.0856 |
| resnet18_s3 | 0.2549 | 0.3890 | **0.4895** | 0.4572 | 0.5258 | -0.0176 |
| resnet18_s4 | **0.2625** | 0.3543 | 0.3862 | 0.3554 | 0.4521 | -0.1194 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.8129 | 0.6903 | 0.6446 | 0.6360 | 0.6490 | -0.0227 |
| mutual_info | 0.7626 | 0.6531 | 0.6175 | 0.6024 | 0.6256 | -0.0563 |
| entropy | **0.8393**† | **0.7205** | **0.6655** | **0.6587** | **0.6711** | +0.0000 |
| confidence | 0.7423† | 0.6120 | 0.5390 | 0.5585 | 0.6007 | -0.1002 |
| mahalanobis | 0.5385† | 0.5019 | 0.5486 | 0.5266 | 0.4595 | -0.1321 |
| knn | 0.5921† | 0.4821 | 0.5123 | 0.5070 | 0.5178 | -0.1517 |
| energy | 0.3293 | 0.5440 | 0.5303 | 0.5780 | 0.6209 | -0.0807 |
| conf_resnet18_s0 | 0.6814 | 0.5926 | 0.5531 | 0.5951 | 0.4638 | -0.0636 |
| conf_resnet18_s1 | 0.6683 | 0.5429 | 0.4579 | 0.4441 | 0.4589 | -0.2146 |
| conf_resnet18_s2 | 0.7103 | 0.5865 | 0.5367 | 0.5296 | 0.4950 | -0.1291 |
| conf_resnet18_s3 | 0.7211 | 0.5797 | 0.5158 | 0.5117 | 0.5026 | -0.1470 |
| conf_resnet18_s4 | 0.7019 | 0.5973 | 0.5632 | 0.6169 | 0.5478 | -0.0418 |
| hybrid_std_knn | 0.8129 | 0.6903 | 0.6446 | 0.6360 | 0.6490 | -0.0227 |
| hybrid_std_maha | 0.8129 | 0.6903 | 0.6446 | 0.6360 | 0.6490 | -0.0227 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0767 | 0.0510 | 0.9232 | 0.2182 |
| mutual_info | 0.0929 | 0.0672 | 0.9070 | 0.2182 |
| entropy | 0.0700 | 0.0442 | 0.9300 | 0.2182 |
| confidence | 0.0454 | 0.0197 | 0.9546 | 0.2182 |
| mahalanobis | 0.2155 | 0.1898 | 0.7845 | 0.2182 |
| knn | 0.1400 | 0.1143 | 0.8599 | 0.2182 |
| energy | 0.3515 | 0.3257 | 0.6485 | 0.2182 |
| conf_resnet18_s0 | 0.0568 | 0.0310 | 0.9432 | 0.2182 |
| conf_resnet18_s1 | 0.0617 | 0.0360 | 0.9383 | 0.2182 |
| conf_resnet18_s2 | 0.0547 | 0.0289 | 0.9453 | 0.2182 |
| conf_resnet18_s3 | 0.0564 | 0.0306 | 0.9436 | 0.2182 |
| conf_resnet18_s4 | 0.0536 | 0.0278 | 0.9464 | 0.2182 |
| hybrid_std_knn | 0.0767 | 0.0510 | 0.9232 | 0.2182 |
| hybrid_std_maha | 0.0767 | 0.0510 | 0.9232 | 0.2182 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8928 |
| mutual_info | 0.8225 |
| entropy | 0.9331 |
| confidence | 0.9147 |
| mahalanobis | 0.5229 |
| knn | 0.6464 |
| hybrid_std_knn | 0.8928 |
| hybrid_std_maha | 0.8928 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.7082 |
| mutual_info | 0.6613 |
| entropy | 0.7436 |
| confidence | 0.6873 |
| mahalanobis | 0.5201 |
| knn | 0.5983 |
| energy | 0.3024 |
| hybrid_std_knn | 0.7082 |
| hybrid_std_maha | 0.7082 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8618
- ECE raw = 0.0227  ->  ECE TS = 0.0117
- confident-error AUROC of -pbar raw = 0.3277  TS = 0.3277  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7993, "auc_b": 0.9244, "z": -358.4399, "p": 0.0} |
| delong_full_epistemic_vs_best_member_resnet18_s4 | {"auc_a": 0.7993, "auc_b": 0.8887, "z": -176.2555, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7993, "auc_b": 0.5237, "z": 595.04, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.8129, "auc_b": 0.7423, "z": 10.3845, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_resnet18_s3 | {"auc_a": 0.8129, "auc_b": 0.7211, "z": 14.9559, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.8129, "auc_b": 0.5385, "z": 74.7357, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.636, "auc_b": 0.5585, "z": 2.6919, "p": 0.0071} |
| delong_conf_top10_epistemic_vs_best_member_resnet18_s4 | {"auc_a": 0.636, "auc_b": 0.6169, "z": 0.5995, "p": 0.5489} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.636, "auc_b": 0.5266, "z": 12.0321, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7952, 0.8287] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5395, 0.727] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0034, "p": 0.9532} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8383, "grid_auroc_top50_cal": {"0.0": 0.5907, "0.25": 0.6458, "0.5": 0.7124, "0.75": 0.7875, "1.0": 0.8383}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1993, 0.2427] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.252, 0.3005] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0319, -0.0203] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0562, 0.0851] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.0158, 0.269] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0328, 0.2334] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.0812, 0.032] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.0271, 0.1765] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0302, 0.0325] |
