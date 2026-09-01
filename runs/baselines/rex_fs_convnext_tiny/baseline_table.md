# Baseline comparison — rex_fs_convnext_tiny (['convnext_tiny_s0', 'convnext_tiny_s1', 'convnext_tiny_s2', 'convnext_tiny_s3', 'convnext_tiny_s4'])

Data: rex_fs_convnext_tiny, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| convnext_tiny_s0 | 0.2514 | 0.3687 | 0.4187 | 0.4498 | 0.4718 | -0.0103 |
| convnext_tiny_s1 | **0.2626** | 0.3553 | **0.4549** | **0.4601** | 0.3586 | +0.0000 |
| convnext_tiny_s2 | 0.2544 | **0.3706** | 0.4077 | 0.4119 | **0.5229** | -0.0482 |
| convnext_tiny_s3 | 0.2476 | 0.3155 | 0.4118 | 0.4573 | 0.4849 | -0.0028 |
| convnext_tiny_s4 | 0.2513 | 0.3247 | 0.3814 | 0.4489 | 0.4294 | -0.0112 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.8478 | 0.7473 | 0.6741 | 0.6688 | 0.6275 | -0.0467 |
| mutual_info | 0.8020 | 0.7062 | 0.6284 | 0.6188 | 0.6136 | -0.0967 |
| entropy | **0.8668**† | **0.7740** | **0.7105** | **0.7155** | **0.6585** | +0.0000 |
| confidence | 0.7527† | 0.6712 | 0.5441 | 0.5385† | 0.4996 | -0.1770 |
| mahalanobis | 0.4166† | 0.4658 | 0.4949 | 0.5236† | 0.5727 | -0.1919 |
| knn | 0.5077† | 0.4983 | 0.5056 | 0.5229† | 0.5749 | -0.1926 |
| energy | 0.2776 | 0.4696 | 0.5103 | 0.5423 | 0.4576 | -0.1732 |
| conf_convnext_tiny_s0 | 0.7423 | 0.6402 | 0.5496 | 0.5617 | 0.5619 | -0.1538 |
| conf_convnext_tiny_s1 | 0.7243 | 0.6299 | 0.4925 | 0.5046 | 0.4785 | -0.2109 |
| conf_convnext_tiny_s2 | 0.7188 | 0.6196 | 0.5363 | 0.5489 | 0.5121 | -0.1666 |
| conf_convnext_tiny_s3 | 0.7445 | 0.6803 | 0.6297 | 0.6443 | 0.5426 | -0.0712 |
| conf_convnext_tiny_s4 | 0.7307 | 0.6750 | 0.6191 | 0.6363 | 0.6036 | -0.0792 |
| hybrid_std_knn | 0.8478 | 0.7473 | 0.6741 | 0.6688 | 0.6275 | -0.0467 |
| hybrid_std_maha | 0.8478 | 0.7473 | 0.6741 | 0.6688 | 0.6275 | -0.0467 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.1006 | 0.0677 | 0.8994 | 0.2452 |
| mutual_info | 0.1229 | 0.0901 | 0.8771 | 0.2452 |
| entropy | 0.0886 | 0.0557 | 0.9114 | 0.2452 |
| confidence | 0.0575 | 0.0246 | 0.9425 | 0.2452 |
| mahalanobis | 0.2683 | 0.2354 | 0.7317 | 0.2452 |
| knn | 0.2100 | 0.1771 | 0.7900 | 0.2452 |
| energy | 0.4015 | 0.3686 | 0.5985 | 0.2452 |
| conf_convnext_tiny_s0 | 0.0638 | 0.0310 | 0.9362 | 0.2452 |
| conf_convnext_tiny_s1 | 0.0641 | 0.0313 | 0.9359 | 0.2452 |
| conf_convnext_tiny_s2 | 0.0672 | 0.0344 | 0.9328 | 0.2452 |
| conf_convnext_tiny_s3 | 0.0661 | 0.0332 | 0.9339 | 0.2452 |
| conf_convnext_tiny_s4 | 0.0650 | 0.0322 | 0.9350 | 0.2452 |
| hybrid_std_knn | 0.1006 | 0.0677 | 0.8994 | 0.2452 |
| hybrid_std_maha | 0.1006 | 0.0677 | 0.8994 | 0.2452 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8818 |
| mutual_info | 0.8066 |
| entropy | 0.9300 |
| confidence | 0.9079 |
| mahalanobis | 0.4475 |
| knn | 0.5541 |
| hybrid_std_knn | 0.8818 |
| hybrid_std_maha | 0.8818 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6947 |
| mutual_info | 0.6513 |
| entropy | 0.7304 |
| confidence | 0.6748 |
| mahalanobis | 0.4290 |
| knn | 0.5054 |
| energy | 0.3055 |
| hybrid_std_knn | 0.6947 |
| hybrid_std_maha | 0.6947 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.8462
- ECE raw = 0.0245  ->  ECE TS = 0.0125
- confident-error AUROC of -pbar raw = 0.2841  TS = 0.2841  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.771, "auc_b": 0.9157, "z": null, "p": null} |
| delong_full_epistemic_vs_best_member_convnext_tiny_s0 | {"auc_a": 0.771, "auc_b": 0.8894, "z": -396.3551, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.771, "auc_b": 0.4514, "z": 600.3857, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.8478, "auc_b": 0.7527, "z": 18.7284, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_convnext_tiny_s3 | {"auc_a": 0.8478, "auc_b": 0.7445, "z": 22.2283, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.8478, "auc_b": 0.4166, "z": null, "p": null} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6688, "auc_b": 0.5385, "z": 4.2043, "p": 0.0} |
| delong_conf_top10_epistemic_vs_best_member_convnext_tiny_s3 | {"auc_a": 0.6688, "auc_b": 0.6443, "z": 0.8242, "p": 0.4098} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6688, "auc_b": 0.5236, "z": null, "p": null} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.8344, 0.8587] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5801, 0.7523] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0101, "p": 0.9199} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8483, "grid_auroc_top50_cal": {"0.0": 0.5083, "0.25": 0.6048, "0.5": 0.7105, "0.75": 0.8002, "1.0": 0.8483}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [-0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.3205, 0.3619] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.4096, 0.4517] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0245, -0.0143] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0817, 0.1066] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [0.0169, 0.2828] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [0.0178, 0.2834] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1107, 0.0079] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.0453, 0.2091] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0416, 0.0445] |
