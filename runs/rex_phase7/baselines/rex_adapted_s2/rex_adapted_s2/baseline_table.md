# Baseline comparison — rex_adapted_s2 (['xrv_nih', 'convnextv2', 'raddino', 'arkswin'])

Data: rex_adapted_s2, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| xrv_nih | 0.2209 | 0.3204 | 0.3632 | 0.3548 | 0.3463 | -0.1472 |
| convnextv2 | **0.2817** | **0.4076** | 0.4473 | 0.4373 | **0.4942** | -0.0647 |
| raddino | 0.2765 | 0.3460 | **0.4499** | **0.5020** | 0.4013 | +0.0000 |
| arkswin | 0.2735 | 0.4034 | 0.4357 | 0.4322 | 0.4057 | -0.0698 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.7763 | 0.7067 | 0.6978 | 0.6927 | 0.7087 | -0.0516 |
| mutual_info | 0.7225 | 0.6504 | 0.6466 | 0.6438 | 0.6615 | -0.1005 |
| entropy | **0.8210**† | **0.7663** | **0.7475** | **0.7443** | **0.7588** | +0.0000 |
| confidence | 0.7124† | 0.6257 | 0.5839 | 0.5331 | 0.4021 | -0.2112 |
| mahalanobis | 0.5635† | 0.5267 | 0.5605 | 0.5348† | 0.4497 | -0.2095 |
| knn | 0.5806† | 0.5597 | 0.5277 | 0.5052† | 0.4311 | -0.2391 |
| energy | 0.3657 | 0.4902 | 0.4573 | 0.4249 | 0.4479 | -0.3194 |
| conf_xrv_nih | 0.6001 | 0.5432 | 0.5200 | 0.5216 | 0.5567 | -0.2227 |
| conf_convnextv2 | 0.6703 | 0.5841 | 0.5209 | 0.4943 | 0.5956 | -0.2500 |
| conf_raddino | 0.6725 | 0.5907 | 0.5451 | 0.4825 | 0.4231 | -0.2618 |
| conf_arkswin | 0.6719 | 0.5887 | 0.5348 | 0.5010 | 0.4277 | -0.2433 |
| hybrid_std_knn | 0.7763 | 0.7067 | 0.6978 | 0.6927 | 0.7087 | -0.0516 |
| hybrid_std_maha | 0.7763 | 0.7067 | 0.6978 | 0.6927 | 0.7087 | -0.0516 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0517 | 0.0351 | 0.9483 | 0.1763 |
| mutual_info | 0.0691 | 0.0525 | 0.9309 | 0.1763 |
| entropy | 0.0434 | 0.0269 | 0.9565 | 0.1763 |
| confidence | 0.0337 | 0.0171 | 0.9663 | 0.1763 |
| mahalanobis | 0.1311 | 0.1145 | 0.8689 | 0.1763 |
| knn | 0.1233 | 0.1068 | 0.8766 | 0.1763 |
| energy | 0.3066 | 0.2900 | 0.6934 | 0.1763 |
| conf_xrv_nih | 0.0683 | 0.0517 | 0.9317 | 0.1763 |
| conf_convnextv2 | 0.0462 | 0.0296 | 0.9538 | 0.1763 |
| conf_raddino | 0.0432 | 0.0267 | 0.9568 | 0.1763 |
| conf_arkswin | 0.0459 | 0.0294 | 0.9541 | 0.1763 |
| hybrid_std_knn | 0.0517 | 0.0351 | 0.9483 | 0.1763 |
| hybrid_std_maha | 0.0517 | 0.0351 | 0.9483 | 0.1763 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8727 |
| mutual_info | 0.7921 |
| entropy | 0.9287 |
| confidence | 0.9206 |
| mahalanobis | 0.6165 |
| knn | 0.6444 |
| hybrid_std_knn | 0.8727 |
| hybrid_std_maha | 0.8727 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.7136 |
| mutual_info | 0.6625 |
| entropy | 0.7629 |
| confidence | 0.7082 |
| mahalanobis | 0.5756 |
| knn | 0.6081 |
| energy | 0.2926 |
| hybrid_std_knn | 0.7136 |
| hybrid_std_maha | 0.7136 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.7853
- ECE raw = 0.0244  ->  ECE TS = 0.0038
- confident-error AUROC of -pbar raw = 0.2557  TS = 0.2557  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.8299, "auc_b": 0.9188, "z": -184.1843, "p": 0.0} |
| delong_full_epistemic_vs_best_member_raddino | {"auc_a": 0.8299, "auc_b": 0.8692, "z": -159.5392, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.8299, "auc_b": 0.6114, "z": 304.9688, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.7763, "auc_b": 0.7124, "z": 7.4428, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_raddino | {"auc_a": 0.7763, "auc_b": 0.6725, "z": 23.6852, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.7763, "auc_b": 0.5635, "z": 34.3214, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6927, "auc_b": 0.5331, "z": 4.0902, "p": 0.0} |
| delong_conf_top10_epistemic_vs_best_member_xrv_nih | {"auc_a": 0.6927, "auc_b": 0.5216, "z": 3.5436, "p": 0.0004} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6927, "auc_b": 0.5348, "z": 4.294, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7561, 0.8003] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5786, 0.8012] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0267, "p": 0.8703} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8106, "grid_auroc_top50_cal": {"0.0": 0.5674, "0.25": 0.6312, "0.5": 0.7004, "0.75": 0.7682, "1.0": 0.8106}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1669, 0.2257] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.1838, 0.2455] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0534, -0.0349] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0416, 0.0829] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [0.0007, 0.3469] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [0.0147, 0.2994] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1283, 0.0207] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.0039, 0.2987] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0171, 0.019] |
