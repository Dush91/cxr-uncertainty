# Baseline comparison — rex (['xrv_nih', 'convnextv2', 'raddino', 'arkswin'])

Data: rex, cal=8064 / eval=8082 images (78311 certain-label eval rows; 2509 uncertain for Task 2), conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| xrv_nih | **0.3300** | **0.4445** | **0.4326** | 0.4391 | 0.4221 | -0.0241 |
| convnextv2 | 0.2350 | 0.3156 | 0.3453 | 0.3510 | 0.2974 | -0.1122 |
| raddino | 0.3255 | 0.4270 | 0.4032 | **0.4632** | **0.5040** | +0.0000 |
| arkswin | 0.2822 | 0.3836 | 0.4195 | 0.4602 | 0.3937 | -0.0030 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.5808† | 0.5042 | 0.5416 | 0.5626 | 0.5348 | -0.0451 |
| mutual_info | 0.5283 | 0.4813 | 0.5193 | 0.5513 | 0.5495 | -0.0564 |
| entropy | 0.6324† | 0.5282 | 0.5452 | 0.5473 | 0.4628 | -0.0604 |
| confidence | 0.6775† | 0.6021 | 0.6095 | 0.5630 | 0.4496 | -0.0447 |
| mahalanobis | 0.5446† | 0.5070 | 0.5418 | 0.5288 | 0.4796 | -0.0789 |
| knn | 0.5941 | 0.5771 | 0.6007 | 0.5942 | 0.5283 | -0.0135 |
| energy | 0.4045 | 0.4247 | 0.4292 | 0.4509 | 0.5293 | -0.1568 |
| conf_xrv_nih | 0.6007 | 0.5293 | 0.4748 | 0.4573 | 0.4824 | -0.1504 |
| conf_convnextv2 | 0.5596 | 0.5740 | 0.5881 | **0.6077** | **0.5890** | +0.0000 |
| conf_raddino | 0.5788 | 0.5288 | 0.5014 | 0.4686 | 0.4267 | -0.1391 |
| conf_arkswin | **0.7004** | **0.6540** | **0.6354** | 0.5808 | 0.5493 | -0.0269 |
| hybrid_std_knn | 0.6127 | 0.5453 | 0.5832 | 0.5979 | 0.5346 | -0.0098 |
| hybrid_std_maha | 0.5820 | 0.4915 | 0.5295 | 0.5368 | 0.4827 | -0.0709 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.0907 | 0.0649 | 0.9093 | 0.2183 |
| mutual_info | 0.1290 | 0.1032 | 0.8710 | 0.2183 |
| entropy | 0.0740 | 0.0483 | 0.9259 | 0.2183 |
| confidence | 0.0448 | 0.0190 | 0.9552 | 0.2183 |
| mahalanobis | 0.1531 | 0.1274 | 0.8468 | 0.2183 |
| knn | 0.1575 | 0.1317 | 0.8425 | 0.2183 |
| energy | 0.3552 | 0.3295 | 0.6447 | 0.2183 |
| conf_xrv_nih | 0.1204 | 0.0946 | 0.8796 | 0.2183 |
| conf_convnextv2 | 0.1332 | 0.1075 | 0.8668 | 0.2183 |
| conf_raddino | 0.0998 | 0.0740 | 0.9002 | 0.2183 |
| conf_arkswin | 0.0839 | 0.0581 | 0.9161 | 0.2183 |
| hybrid_std_knn | 0.0893 | 0.0635 | 0.9107 | 0.2183 |
| hybrid_std_maha | 0.0880 | 0.0622 | 0.9120 | 0.2183 |

### Per-pathology correctness AUROC (macro over 10 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.8391 |
| mutual_info | 0.7470 |
| entropy | 0.9074 |
| confidence | 0.9175 |
| mahalanobis | 0.6439 |
| knn | 0.6547 |
| hybrid_std_knn | 0.8372 |
| hybrid_std_maha | 0.8302 |

### Baur Task 2 — uncertainty-label prediction (macro over 10 pathologies; positive class = expert 'uncertain' (-1); 2509 uncertain eval rows)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.6353 |
| mutual_info | 0.5752 |
| entropy | 0.6983 |
| confidence | 0.6567 |
| mahalanobis | 0.5896 |
| knn | 0.6188 |
| energy | 0.3346 |
| hybrid_std_knn | 0.6510 |
| hybrid_std_maha | 0.6405 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.9662
- ECE raw = 0.0219  ->  ECE TS = 0.0207
- confident-error AUROC of -pbar raw = 0.4527  TS = 0.4527  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7745, "auc_b": 0.9273, "z": -240.3735, "p": 0.0} |
| delong_full_epistemic_vs_best_member_arkswin | {"auc_a": 0.7745, "auc_b": 0.7876, "z": null, "p": null} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7745, "auc_b": 0.6373, "z": 223.5233, "p": 0.0} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.5808, "auc_b": 0.6775, "z": -12.6855, "p": 0.0} |
| delong_conf_top50_epistemic_vs_best_member_arkswin | {"auc_a": 0.5808, "auc_b": 0.7004, "z": null, "p": null} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.5808, "auc_b": 0.5446, "z": 9.7665, "p": 0.0} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.5626, "auc_b": 0.563, "z": -0.0131, "p": 0.9895} |
| delong_conf_top10_epistemic_vs_best_member_convnextv2 | {"auc_a": 0.5626, "auc_b": 0.6077, "z": -3.3986, "p": 0.0007} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.5626, "auc_b": 0.5288, "z": 1.1029, "p": 0.2701} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.5571, 0.6056] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.4631, 0.6597] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0004, "p": 0.9845} |
| hybrid | {"weight_on_std": 0.75, "cal_selection_auroc_top50": 0.5891, "grid_auroc_top50_cal": {"0.0": 0.552, "0.25": 0.565, "0.5": 0.5772, "0.75": 0.5891, "1.0": 0.573}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0181, 0.0446] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [-0.0049, 0.0409] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [0.0431, 0.0921] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.037, -0.0028] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [-0.0865, -0.0439] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [-0.0359, 0.1209] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.0454, 0.0493] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0064, 0.1406] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.0457, 0.1383] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [-0.0881, 0.1486] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0428, 0.0464] |
