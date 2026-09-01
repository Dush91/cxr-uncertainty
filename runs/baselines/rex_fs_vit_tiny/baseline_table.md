# Baseline comparison — rex_fs_vit_tiny (['vit_tiny_s0', 'vit_tiny_s1', 'vit_tiny_s2', 'vit_tiny_s3', 'vit_tiny_s4'])

Data: rex_fs_vit_tiny, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| vit_tiny_s0 | 0.2474 | 0.3242 | **0.3619** | 0.4086 | 0.4194 | -0.0171 |
| vit_tiny_s1 | **0.2509** | **0.3529** | 0.3536 | 0.4161 | 0.4669 | -0.0096 |
| vit_tiny_s2 | 0.2438 | 0.3454 | 0.3321 | **0.4257** | 0.3736 | +0.0000 |
| vit_tiny_s3 | 0.2366 | 0.3298 | 0.3286 | 0.3375 | **0.4772** | -0.0882 |
| vit_tiny_s4 | 0.2448 | 0.2969 | 0.3548 | 0.3675 | 0.3005 | -0.0582 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.8541 | 0.7510 | 0.6788 | 0.7190 | 0.6745 | -0.0420 |
| mutual_info | 0.8051 | 0.7039 | 0.6187 | 0.6685 | 0.6249 | -0.0925 |
| entropy | **0.8767**† | **0.7775** | **0.7263** | **0.7610** | **0.7052** | +0.0000 |
| confidence | 0.7612† | 0.6739 | 0.6023 | 0.6736 | 0.5280 | -0.0874 |
| mahalanobis | 0.3692† | 0.4942 | 0.5092 | 0.5132† | 0.4817 | -0.2478 |
| knn | 0.4883† | 0.5099 | 0.5079 | 0.5474† | 0.4924 | -0.2136 |
| energy | 0.2651 | 0.4923 | 0.5158 | 0.4993 | 0.4470 | -0.2617 |
| conf_vit_tiny_s0 | 0.7084 | 0.6379 | 0.5480 | 0.5564 | 0.4742 | -0.2046 |
| conf_vit_tiny_s1 | 0.7106 | 0.6144 | 0.5350 | 0.5651 | 0.4095 | -0.1959 |
| conf_vit_tiny_s2 | 0.6905 | 0.6030 | 0.5652 | 0.5835 | 0.4888 | -0.1775 |
| conf_vit_tiny_s3 | 0.7150 | 0.6461 | 0.6259 | 0.6724 | 0.5459 | -0.0886 |
| conf_vit_tiny_s4 | 0.6692 | 0.6082 | 0.5516 | 0.5579 | 0.6104 | -0.2031 |
| hybrid_std_knn | 0.8541 | 0.7510 | 0.6788 | 0.7190 | 0.6745 | -0.0420 |
| hybrid_std_maha | 0.8541 | 0.7510 | 0.6788 | 0.7190 | 0.6745 | -0.0420 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0972 | 0.0631 | 0.9028 | 0.2497 |
| mutual_info | 0.1217 | 0.0875 | 0.8783 | 0.2497 |
| entropy | 0.0840 | 0.0499 | 0.9160 | 0.2497 |
| confidence | 0.0619 | 0.0278 | 0.9380 | 0.2497 |
| mahalanobis | 0.3049 | 0.2708 | 0.6950 | 0.2497 |
| knn | 0.2241 | 0.1899 | 0.7759 | 0.2497 |
| energy | 0.4114 | 0.3772 | 0.5886 | 0.2497 |
| conf_vit_tiny_s0 | 0.0762 | 0.0420 | 0.9238 | 0.2497 |
| conf_vit_tiny_s1 | 0.0752 | 0.0411 | 0.9248 | 0.2497 |
| conf_vit_tiny_s2 | 0.0799 | 0.0457 | 0.9201 | 0.2497 |
| conf_vit_tiny_s3 | 0.0754 | 0.0412 | 0.9246 | 0.2497 |
| conf_vit_tiny_s4 | 0.0845 | 0.0503 | 0.9155 | 0.2497 |
| hybrid_std_knn | 0.0972 | 0.0631 | 0.9028 | 0.2497 |
| hybrid_std_maha | 0.0972 | 0.0631 | 0.9028 | 0.2497 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8634 |
| mutual_info | 0.7771 |
| entropy | 0.9252 |
| confidence | 0.9049 |
| mahalanobis | 0.4017 |
| knn | 0.5447 |
| hybrid_std_knn | 0.8634 |
| hybrid_std_maha | 0.8634 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6861 |
| mutual_info | 0.6410 |
| entropy | 0.7200 |
| confidence | 0.6695 |
| mahalanobis | 0.4012 |
| knn | 0.4879 |
| energy | 0.3170 |
| hybrid_std_knn | 0.6861 |
| hybrid_std_maha | 0.6861 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8302
- ECE raw = 0.0247  ->  ECE TS = 0.0088
- confident-error AUROC of -pbar raw = 0.239  TS = 0.239  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7853, "auc_b": 0.9059, "z": -1177.9537, "p": 0.0} |
| delong_full_epistemic_vs_best_member_vit_tiny_s1 | {"auc_a": 0.7853, "auc_b": 0.8538, "z": null, "p": null} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7853, "auc_b": 0.4057, "z": null, "p": null} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.8541, "auc_b": 0.7612, "z": 21.5533, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_vit_tiny_s3 | {"auc_a": 0.8541, "auc_b": 0.715, "z": 38.0445, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.8541, "auc_b": 0.3692, "z": null, "p": null} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.719, "auc_b": 0.6736, "z": 1.8784, "p": 0.0603} |
| delong_conf_top10_epistemic_vs_best_member_vit_tiny_s3 | {"auc_a": 0.719, "auc_b": 0.6724, "z": 2.0252, "p": 0.0428} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.719, "auc_b": 0.5132, "z": 11.9664, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.843, 0.8648] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.6381, 0.794] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0004, "p": 0.9843} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8531, "grid_auroc_top50_cal": {"0.0": 0.4842, "0.25": 0.5976, "0.5": 0.72, "0.75": 0.8071, "1.0": 0.8531}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [-0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.3469, 0.3832] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.4638, 0.5043] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0272, -0.0182] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0821, 0.1038] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [0.0619, 0.2726] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [0.0962, 0.3197] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.0982, 0.0154] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.0594, 0.1447] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0339, 0.0365] |
