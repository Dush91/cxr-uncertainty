# Baseline comparison — rex_adapted_s1 (['xrv_nih', 'convnextv2', 'raddino', 'arkswin'])

Data: rex_adapted_s1, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| xrv_nih | 0.2209 | 0.3204 | 0.3632 | 0.3548 | 0.3463 | -0.1472 |
| convnextv2 | 0.2689 | 0.3705 | 0.4216 | 0.4778 | **0.5167** | -0.0242 |
| raddino | **0.2765** | 0.3460 | **0.4499** | **0.5020** | 0.4013 | +0.0000 |
| arkswin | 0.2735 | **0.4034** | 0.4357 | 0.4322 | 0.4057 | -0.0698 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.8074 | 0.7058 | 0.6833 | 0.6832 | 0.6580 | -0.0501 |
| mutual_info | 0.7562 | 0.6512 | 0.6323 | 0.6417 | 0.6308 | -0.0916 |
| entropy | **0.8461**† | **0.7717** | **0.7397** | **0.7333** | **0.6679** | +0.0000 |
| confidence | 0.7324† | 0.6234 | 0.5663 | 0.5540† | 0.4303 | -0.1793 |
| mahalanobis | 0.5868† | 0.4960 | 0.5443 | 0.5401† | 0.4620 | -0.1932 |
| knn | 0.6089† | 0.5157 | 0.5404 | 0.4821† | 0.4443 | -0.2512 |
| energy | 0.3255 | 0.4720 | 0.4175 | 0.4118 | 0.4165 | -0.3215 |
| conf_xrv_nih | 0.6396 | 0.5759 | 0.5432 | 0.5505 | 0.5660 | -0.1828 |
| conf_convnextv2 | 0.7029 | 0.6012 | 0.5471 | 0.5268 | 0.4876 | -0.2065 |
| conf_raddino | 0.7104 | 0.6172 | 0.5722 | 0.5438 | 0.4801 | -0.1895 |
| conf_arkswin | 0.7080 | 0.6007 | 0.5327 | 0.5076 | 0.4558 | -0.2257 |
| hybrid_std_knn | 0.8074 | 0.7058 | 0.6833 | 0.6832 | 0.6580 | -0.0501 |
| hybrid_std_maha | 0.8074 | 0.7058 | 0.6833 | 0.6832 | 0.6580 | -0.0501 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0844 | 0.0612 | 0.9156 | 0.2073 |
| mutual_info | 0.1031 | 0.0800 | 0.8969 | 0.2073 |
| entropy | 0.0756 | 0.0525 | 0.9244 | 0.2073 |
| confidence | 0.0417 | 0.0186 | 0.9582 | 0.2073 |
| mahalanobis | 0.1561 | 0.1329 | 0.8439 | 0.2073 |
| knn | 0.1432 | 0.1200 | 0.8568 | 0.2073 |
| energy | 0.3559 | 0.3328 | 0.6441 | 0.2073 |
| conf_xrv_nih | 0.0791 | 0.0559 | 0.9209 | 0.2073 |
| conf_convnextv2 | 0.0514 | 0.0282 | 0.9486 | 0.2073 |
| conf_raddino | 0.0507 | 0.0275 | 0.9493 | 0.2073 |
| conf_arkswin | 0.0528 | 0.0297 | 0.9472 | 0.2073 |
| hybrid_std_knn | 0.0844 | 0.0612 | 0.9156 | 0.2073 |
| hybrid_std_maha | 0.0844 | 0.0612 | 0.9156 | 0.2073 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8688 |
| mutual_info | 0.7834 |
| entropy | 0.9309 |
| confidence | 0.9201 |
| mahalanobis | 0.6194 |
| knn | 0.6515 |
| hybrid_std_knn | 0.8688 |
| hybrid_std_maha | 0.8688 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.7129 |
| mutual_info | 0.6600 |
| entropy | 0.7642 |
| confidence | 0.6990 |
| mahalanobis | 0.5756 |
| knn | 0.6081 |
| energy | 0.2926 |
| hybrid_std_knn | 0.7129 |
| hybrid_std_maha | 0.7129 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.778
- ECE raw = 0.0263  ->  ECE TS = 0.0046
- confident-error AUROC of -pbar raw = 0.2667  TS = 0.2667  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7638, "auc_b": 0.9248, "z": null, "p": null} |
| delong_full_epistemic_vs_best_member_raddino | {"auc_a": 0.7638, "auc_b": 0.8851, "z": -301.6097, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7638, "auc_b": 0.6106, "z": 203.5435, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.8074, "auc_b": 0.7324, "z": 10.1108, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_raddino | {"auc_a": 0.8074, "auc_b": 0.7104, "z": 19.3761, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.8074, "auc_b": 0.5868, "z": 43.1958, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6832, "auc_b": 0.554, "z": 2.8385, "p": 0.0045} |
| delong_conf_top10_epistemic_vs_best_member_xrv_nih | {"auc_a": 0.6832, "auc_b": 0.5505, "z": 3.2474, "p": 0.0012} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6832, "auc_b": 0.5401, "z": 4.5444, "p": 0.0} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.7899, 0.8255] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.5852, 0.7788] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0432, "p": 0.8354} |
| hybrid | {"weight_on_std": 1.0, "cal_selection_auroc_top50": 0.8266, "grid_auroc_top50_cal": {"0.0": 0.5917, "0.25": 0.667, "0.5": 0.7352, "0.75": 0.7947, "1.0": 0.8266}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [0.1699, 0.2234] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.1955, 0.2466] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0461, -0.0314] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [0.0579, 0.0913] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [0.0, 0.0] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [0.0403, 0.3617] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [0.0277, 0.2659] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.1061, 0.0114] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.0356, 0.2279] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0412, 0.0441] |
