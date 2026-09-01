# Baseline comparison — openi (['xrv_nih', 'convnextv2', 'raddino', 'arkswin'])

Data: openi, cal=2002 / eval=2012 images, conf_pct=10.0.

### Panel A — single-member self-detection at matched coverage (AUROC of each member's own confidence vs its own errors)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| xrv_nih | 0.2695 | **0.4379** | 0.3736 | 0.4960 | 0.3847 | -0.0345 |
| convnextv2 | 0.1296 | 0.2836 | 0.4030 | 0.4283 | **0.6137** | -0.1022 |
| raddino | 0.1840 | 0.2914 | **0.5032** | **0.5305** | 0.4860 | +0.0000 |
| arkswin | **0.3059** | 0.3601 | 0.3697 | 0.4023 | -- | -0.1282 |

### Panel B — all scores within the ensemble confident set (AUROC vs ensemble errors; the flag's population)

| method | top-50% | top-25% | top-15% | top-10% | top-5% | Δ vs best (top-10%) |
|:---|---:|---:|---:|---:|---:|---:|
| epistemic_std | 0.6193† | 0.5741 | 0.6319 | 0.6072 | 0.4971 | -0.2987 |
| mutual_info | 0.5277 | 0.4909 | 0.5901 | 0.5691 | 0.4077 | -0.3368 |
| entropy | 0.7190 | 0.6794 | 0.6812 | 0.6510 | 0.6199 | -0.2549 |
| confidence | 0.6642 | 0.5931 | 0.5157 | 0.5058† | 0.5086 | -0.4001 |
| mahalanobis | 0.6798 | 0.7075 | 0.7888 | 0.8698 | 0.9382 | -0.0361 |
| knn | 0.6805 | 0.7348 | 0.7518 | 0.8763 | **0.9683** | -0.0296 |
| energy | 0.4261 | 0.4751 | 0.4772 | 0.5422 | 0.5481 | -0.3637 |
| conf_xrv_nih | 0.5981 | 0.5939 | 0.5772 | 0.6019 | 0.7673 | -0.3040 |
| conf_convnextv2 | 0.4953 | 0.3879 | 0.4108 | 0.3186 | 0.2782 | -0.5873 |
| conf_raddino | 0.5491 | 0.4444 | 0.3335 | 0.2499 | 0.4156 | -0.6560 |
| conf_arkswin | **0.7846** | **0.7821** | 0.7269 | 0.7200 | 0.7236 | -0.1859 |
| hybrid_std_knn | 0.7139 | 0.7345 | 0.8096 | **0.9059** | 0.9418 | +0.0000 |
| hybrid_std_maha | 0.7230 | 0.7228 | **0.8190** | 0.8904 | 0.9282 | -0.0155 |

† paired bootstrap 95% CI of the AUROC difference vs hybrid_std_knn excludes 0 (n_boot=500).

### E-AURC / AUAC (risk-coverage, full eval set)

| score | AURC | E-AURC | AUAC | error_rate |
|---|---:|---:|---:|---:|
| epistemic_std | 0.1264 | 0.0824 | 0.8736 | 0.2816 |
| mutual_info | 0.1440 | 0.1000 | 0.8560 | 0.2816 |
| entropy | 0.1297 | 0.0857 | 0.8703 | 0.2816 |
| confidence | 0.0575 | 0.0135 | 0.9424 | 0.2816 |
| mahalanobis | 0.2678 | 0.2238 | 0.7321 | 0.2816 |
| knn | 0.3555 | 0.3115 | 0.6444 | 0.2816 |
| energy | 0.4810 | 0.4370 | 0.5190 | 0.2816 |
| conf_xrv_nih | 0.1687 | 0.1247 | 0.8313 | 0.2816 |
| conf_convnextv2 | 0.1777 | 0.1337 | 0.8223 | 0.2816 |
| conf_raddino | 0.1224 | 0.0784 | 0.8775 | 0.2816 |
| conf_arkswin | 0.1279 | 0.0839 | 0.8721 | 0.2816 |
| hybrid_std_knn | 0.1451 | 0.1011 | 0.8549 | 0.2816 |
| hybrid_std_maha | 0.1434 | 0.0994 | 0.8566 | 0.2816 |

### Per-pathology correctness AUROC (macro over 13 pathologies, full eval set)

| score | macro AUROC |
|---|---:|
| epistemic_std | 0.9080 |
| mutual_info | 0.8423 |
| entropy | 0.9674 |
| confidence | 0.9435 |
| mahalanobis | 0.5751 |
| knn | 0.4406 |
| hybrid_std_knn | 0.8785 |
| hybrid_std_maha | 0.8533 |

### TS-calibrated confidence (calibration-only baseline)

- temperature T = 0.6335
- ECE raw = 0.0484  ->  ECE TS = 0.0047
- confident-error AUROC of -pbar raw = 0.349  TS = 0.349  (max abs diff 0.0; TS is rank-invariant)

### Statistical tests (ensemble confident set, top-10%)

| test | result |
|---|---|
| delong_full_epistemic_vs_confidence | {"auc_a": 0.7822, "auc_b": 0.9543, "z": -617.5873, "p": 0.0} |
| delong_full_epistemic_vs_best_member_arkswin | {"auc_a": 0.7822, "auc_b": 0.7725, "z": 18.8411, "p": 0.0} |
| delong_full_epistemic_vs_mahalanobis | {"auc_a": 0.7822, "auc_b": 0.545, "z": null, "p": null} |
| delong_conf_top50_epistemic_vs_confidence | {"auc_a": 0.6193, "auc_b": 0.6642, "z": -1.9058, "p": 0.0567} |
| delong_conf_top50_epistemic_vs_best_member_arkswin | {"auc_a": 0.6193, "auc_b": 0.7846, "z": -17.7448, "p": 0.0} |
| delong_conf_top50_epistemic_vs_mahalanobis | {"auc_a": 0.6193, "auc_b": 0.6798, "z": null, "p": null} |
| delong_conf_top10_epistemic_vs_confidence | {"auc_a": 0.6072, "auc_b": 0.5058, "z": 0.622, "p": 0.5339} |
| delong_conf_top10_epistemic_vs_best_member_arkswin | {"auc_a": 0.6072, "auc_b": 0.72, "z": -2.3914, "p": 0.0168} |
| delong_conf_top10_epistemic_vs_mahalanobis | {"auc_a": 0.6072, "auc_b": 0.8698, "z": null, "p": null} |
| bootstrap_epistemic_std_auroc_ci95_conf_top50 | [0.5583, 0.6758] |
| bootstrap_epistemic_std_auroc_ci95_conf_top10 | [0.2152, 0.9845] |
| mcnemar_epistemic_vs_confidence | {"chi2": 0.0013, "p": 0.9714} |
| hybrid | {"weight_on_std": 0.75, "cal_selection_auroc_top50": 0.7342, "grid_auroc_top50_cal": {"0.0": 0.6785, "0.25": 0.7013, "0.5": 0.7282, "0.75": 0.7342, "1.0": 0.6572}} |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top50 | [0.0502, 0.1403] |
| boot_paired_hybrid_vs_knn_auroc_conf_top50 | [-0.024, 0.0918] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top50 | [-0.0323, 0.102] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top50 | [-0.0493, 0.0432] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top50 | [-0.0083, 0.1113] |
| boot_paired_hybrid_vs_epistemic_std_auroc_conf_top10 | [-0.1092, 0.6494] |
| boot_paired_hybrid_vs_knn_auroc_conf_top10 | [-0.105, 0.1991] |
| boot_paired_hybrid_vs_mahalanobis_auroc_conf_top10 | [-0.0452, 0.1606] |
| boot_paired_hybrid_vs_entropy_auroc_conf_top10 | [-0.117, 0.5792] |
| boot_paired_hybrid_vs_confidence_auroc_conf_top10 | [0.0078, 0.7405] |
| boot_paired_hybrid_vs_confidence_eaurc_full | [0.0835, 0.0917] |
