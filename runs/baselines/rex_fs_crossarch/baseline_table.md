# Baseline comparison — rex_fs_crossarch (['resnet18_s0', 'vit_tiny_s0', 'convnext_tiny_s0'])

Data: rex_fs_crossarch, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| resnet18_s0 | **0.2542** | 0.3517 | 0.4075 | 0.4459 | **0.5649** | -0.0039 |
| vit_tiny_s0 | 0.2474 | 0.3242 | 0.3619 | 0.4086 | 0.4194 | -0.0412 |
| convnext_tiny_s0 | 0.2514 | **0.3687** | **0.4187** | **0.4498** | 0.4718 | +0.0000 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.8048 | 0.7055 | 0.5926 | 0.5585 | 0.5011 | -0.0767 |
| mutual_info | 0.7283 | 0.6451 | 0.5468 | 0.5164 | 0.4703 | -0.1188 |
| entropy | **0.8512**† | **0.7471** | **0.6357** | **0.6352** | 0.5721 | +0.0000 |
| confidence | 0.7642† | 0.6722 | 0.5707 | 0.6093 | 0.5525 | -0.0259 |
| mahalanobis | 0.4217† | 0.4542 | 0.4971 | 0.4851 | 0.5187 | -0.1501 |
| knn | 0.5079† | 0.4925 | 0.5213 | 0.5058 | 0.5076 | -0.1294 |
| energy | 0.2874 | 0.4680 | 0.5295 | 0.5266 | 0.5305 | -0.1086 |
| conf_resnet18_s0 | 0.7145 | 0.6150 | 0.5436 | 0.5357 | **0.5821** | -0.0995 |
| conf_vit_tiny_s0 | 0.6571 | 0.5717 | 0.5212 | 0.5530 | 0.4324 | -0.0822 |
| conf_convnext_tiny_s0 | 0.7146 | 0.6067 | 0.5320 | 0.5254 | 0.4583 | -0.1098 |
| hybrid_std_knn | 0.8048 | 0.7055 | 0.5926 | 0.5585 | 0.5011 | -0.0767 |
| hybrid_std_maha | 0.8048 | 0.7055 | 0.5926 | 0.5585 | 0.5011 | -0.0767 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0991 | 0.0708 | 0.9009 | 0.2282 |
| mutual_info | 0.1332 | 0.1050 | 0.8667 | 0.2282 |
| entropy | 0.0833 | 0.0550 | 0.9167 | 0.2282 |
| confidence | 0.0503 | 0.0220 | 0.9497 | 0.2282 |
| mahalanobis | 0.2485 | 0.2202 | 0.7515 | 0.2282 |
| knn | 0.1927 | 0.1644 | 0.8073 | 0.2282 |
| energy | 0.3737 | 0.3454 | 0.6263 | 0.2282 |
| conf_resnet18_s0 | 0.0649 | 0.0367 | 0.9350 | 0.2282 |
| conf_vit_tiny_s0 | 0.0744 | 0.0461 | 0.9256 | 0.2282 |
| conf_convnext_tiny_s0 | 0.0622 | 0.0339 | 0.9378 | 0.2282 |
| hybrid_std_knn | 0.0991 | 0.0708 | 0.9009 | 0.2282 |
| hybrid_std_maha | 0.0991 | 0.0708 | 0.9009 | 0.2282 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8486 |
| mutual_info | 0.7454 |
| entropy | 0.9301 |
| confidence | 0.9131 |
| mahalanobis | 0.4512 |
| knn | 0.5576 |
| hybrid_std_knn | 0.8486 |
| hybrid_std_maha | 0.8486 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6919 |
| mutual_info | 0.6376 |
| entropy | 0.7372 |
| confidence | 0.6813 |
| mahalanobis | 0.4290 |
| knn | 0.5054 |
| energy | 0.3055 |
| hybrid_std_knn | 0.6919 |
| hybrid_std_maha | 0.6919 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8242
- ECE raw = 0.0253  ->  ECE TS = 0.0103
- confident-error AUROC of -pbar raw = 0.3648  TS = 0.3648  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.757, "auc_b": 0.9182, "z": null, "p": null} |
| delong_full_epistemic_vs_best_member_convnext_tiny_s0 | {"auc_a": 0.757, "auc_b": 0.8707, "z": null, "p": null} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.757, "auc_b": 0.4554, "z": 1139.7244, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.8048, "auc_b": 0.7642, "z": 7.2547, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_convnext_tiny_s0 | {"auc_a": 0.8048, "auc_b": 0.7146, "z": 18.8453, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.8048, "auc_b": 0.4217, "z": null, "p": null} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.5585, "auc_b": 0.6093, "z": -1.2217, "p": 0.2218} |
| delong_conf_top10_epistemic_vs_best_member_vit_tiny_s0 | {"auc_a": 0.5585, "auc_b": 0.553, "z": 0.124, "p": 0.9013} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.5585, "auc_b": 0.4851, "z": 2.2921, "p": 0.0219} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7875, 0.8215] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.4494, 0.6686] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0004, "p": 0.9848} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8191, "grid_auroc_top50_cal": {"0.0": 0.4986, "0.25": 0.5805, "0.5": 0.6702, "0.75": 0.7549, "1.0": 0.8191}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [-0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.2695, 0.3204] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.3541, 0.4083] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0552, -0.0386] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0245, 0.0544] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [-0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.0556, 0.1641] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0454, 0.2022] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1585, 0.0027] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.1546, 0.0492] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0471, 0.0504] |
