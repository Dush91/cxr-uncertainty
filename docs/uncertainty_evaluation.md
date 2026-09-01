# Evaluating & Benchmarking CXR Uncertainty: Methods, Baselines, and the Literature

Reference document for the CXR uncertainty / confident-error flagging project.
Covers (1) how to evaluate an uncertainty estimate rigorously, (2) how to compare
against baselines to establish an edge, and (3) how recent (2019–2026) papers
benchmark uncertainty in chest X-ray imaging. Written 2026-08-14; literature
surveyed via live web search.

> **Caveat on numbers:** several reported figures below come from search
> abstracts/secondary summaries and are marked *"as reported."* Verify any
> number against the primary source before citing it in a thesis/paper.

---

## 0. Framing principle

The field's governing idea (from probabilistic forecasting): **maximize
sharpness subject to calibration** (Gneiting & Raftery 2007,
[JASA](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)).
Modern UQ evaluation adds a third axis — **discrimination** (does the uncertainty
score *rank* errors above corrects?). No single number is sufficient: each
standard metric is known to be gameable in a specific way (Section 4 of the
methodology survey). A rigorous evaluation reports all three axes plus the
task-specific ones (OOD detection, set-valued prediction).

---

## 1. The three axes of UQ evaluation

### 1.1 Calibration — "are the probabilities honest?"

- **ECE** (Expected Calibration Error): bin [0,1] into M bins, weight each bin by
  its size, sum |acc − conf| (Guo et al., ICML 2017,
  [PMLR](https://proceedings.mlr.press/v70/guo17a.html); origin Naeini et al.,
  AAAI 2015). **MCE** = worst-case bin.
- **Classwise-ECE / ACE** (Nixon et al., CVPR-W 2019,
  [arXiv:1904.01685](https://doi.org/10.48550/arxiv.1904.01685)): average
  per-class binning; ACE uses adaptive (equal-mass) bins.
- **Reliability diagrams** (Guo et al. 2017); modern tuning-free version: CORP
  diagram (Dimitriadis et al., *The Triptych*,
  [arXiv:2301.10803](https://arxiv.org/pdf/2301.10803)).
- **Known pitfalls** (this is where rigor lives):
  - Equal-width bins are statistically biased; equal-mass bins lower bias
    (Roelofs et al., AISTATS 2022,
    [PMLR](https://proceedings.mlr.press/v151/roelofs22a.html)).
  - The histogram ECE estimator is an inconsistent *lower bound* on true
    miscalibration (Vaicenavicius et al., AISTATS 2019,
    [PMLR](https://proceedings.mlr.press/v89/vaicenavicius19a.html)).
  - **ECE is gameable**: minimized by constant/uniform predictions (Ashukha et
    al. 2020, [arXiv:2002.06470](https://doi.org/10.48550/arxiv.2002.06470));
    formalized as a "truthfulness gap" (Haghtalab et al., NeurIPS 2024,
    [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/d4cbcae8cfc8aa3ae897a1296e4e0cac-Paper-Conference.pdf)).
  - Naive bootstrap CIs for ECE undercover near zero; use debiased/analytic
    intervals (Sun et al., [arXiv:2408.08998](https://doi.org/10.48550/arxiv.2408.08998)).
  - **Statistical tests** for "is A better calibrated than B": T-Cal (Lee et al.,
    JMLR 2023, [JMLR](https://jmlr.org/papers/volume24/22-0320/22-0320.pdf)) or
    consistency resampling (Vaicenavicius et al. 2019).

### 1.2 Sharpness — proper scoring rules

Proper scoring rules evaluate calibration *and* sharpness jointly and cannot be
gamed by hedging (Gneiting & Raftery 2007).

- **Brier score** with its canonical **reliability − resolution + uncertainty**
  decomposition (Murphy 1973; Bröcker 2009,
  [DOI](https://doi.org/10.1002/qj.456)). Caveat: a lower Brier does not
  necessarily mean better calibration — it conflates all three terms.
- **NLL / log-loss**: strictly proper, harshly penalizes assigning ~0
  probability to the true class. Report **CLL** (NLL at the temperature that
  optimizes validation NLL) so "model + calibration" is one system (Ashukha et
  al. 2020).
- **CRPS / energy score**: primarily regression/forecasting; for unordered
  multiclass labels the standard strictly-proper scores over the simplex (log,
  Brier, spherical) play its role.

### 1.3 Discrimination / ranking — "does the score separate wrong from correct?"

- **Score-vs-correctness AUROC / AUPR** (failure prediction): treat "prediction
  is wrong" as positive, the uncertainty score as discriminator; report
  AUPR-Error and AUPR-Success, plus FPR@95TPR (Corbière et al., NeurIPS 2019,
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf)).
  **Critical caveat:** AUROC/AUPR on correct-vs-wrong are **not comparable
  across models with different accuracy** (Ding et al., CVPR-W 2020,
  [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Ding_Revisiting_the_Evaluation_of_Uncertainty_Estimation_and_Its_Application_to_CVPRW_2020_paper.pdf)).
- **Risk–coverage curves and AURC**: sort by descending confidence; risk(κ) is
  the error rate in the top-κ prefix; AURC = ∫ risk dκ (lower better). AURC
  evaluates prediction *and* uncertainty jointly.
- **Excess-AURC (E-AURC)**: subtract the oracle floor
  `AURC* ≈ r̂ + (1−r̂)ln(1−r̂)` (r̂ = full-coverage error rate) — isolates
  *ranking quality* from accuracy (Geifman, Uziel & El-Yaniv, ICLR 2019,
  [arXiv:1805.08206](https://ar5iv.labs.arxiv.org/html/1805.08206)). This is the
  de-facto standard for comparing uncertainty estimators for selective prediction.
- **Newer guards** against accuracy confounds: NAURC (Cattelan & Silva, UAI
  2024, [arXiv:2305.15508](https://arxiv.org/html/2305.15508)); AUGRC (Traub et
  al., NeurIPS 2024, [arXiv:2407.01032](https://doi.org/10.48550/arxiv.2407.01032)).
- **Working-point metrics**: risk@coverage (e.g., risk at 80% coverage),
  coverage@risk, SAC (max coverage under a selective-accuracy constraint)
  (Geifman & El-Yaniv, NeurIPS 2017,
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)).

### 1.4 OOD detection

- **AUROC** (ID scores higher than OOD), **AUPR-in / AUPR-out** (name the
  positive class explicitly), **FPR@95TPR** (the headline number in most OOD
  papers). Conventions set by **OpenOOD**
  ([arXiv:2306.09301](https://arxiv.org/html/2306.09301v2)).
- **"Detect correct vs. detect OOD" are distinct**: a score can rank OOD low but
  rank ID misclassifications high (or vice versa) — report both
  misclassification-detection (AURC / correctness AUROC) and OOD-detection
  metrics.

### 1.5 Set-valued / conformal prediction

- Conformal gives a **marginal** guarantee P(Y ∈ C(X)) ≥ 1−α; exact conditional
  coverage is distribution-free-impossible (Vovk 2012; Lei & Wasserman 2014).
  The evaluation question is how close a method comes (Angelopoulos & Bates,
  [arXiv:2107.07511](https://doi.org/10.48550/arxiv.2107.07511)).
- **Metrics**: marginal coverage (a Binomial quantity — CI via Beta
  distribution), **average set size / efficiency** (smaller at same coverage =
  sharper), **size-stratified coverage violation (SSCV)** and class-conditional
  coverage gaps (Angelopoulos et al., ICLR 2021,
  [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)).
- Methods to compare: APS (Romano et al., NeurIPS 2020,
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf))
  vs RAPS vs top-k.

---

## 2. Comparing against baselines — the fair-comparison protocol

### 2.1 The standard baseline set

A rigorous paper compares against, at minimum:

1. **Maximum softmax / sigmoid confidence (MSP)** — the foundational baseline
   (Hendrycks & Gimpel, ICLR 2017, [arXiv:1610.02136](https://arxiv.org/abs/1610.02136)).
2. **Temperature-scaled confidence** — the calibration-only baseline (Guo et al.
   2017). Key fact: TS is a monotone transform — it fixes calibration but
   **cannot change the confidence ranking**, so it cannot improve AUROC/AURC
   (Corbière et al. 2019). It is a calibration baseline, not a ranking
   competitor.
3. **MC-Dropout** — predictive mean over T stochastic passes, entropy/BALD as
   score (Gal & Ghahramani, ICML 2016,
   [PMLR](https://proceedings.mlr.press/v48/gal16.html)).
4. **Deep ensembles** — the de-facto strongest general baseline (Lakshminarayanan
   et al., NIPS 2017,
   [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)).
5. **Feature-space / density baselines for OOD**: Mahalanobis (Lee et al.,
   NeurIPS 2018, [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)), energy
   (Liu et al., NeurIPS 2020,
   [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)),
   **k-NN distance** (Sun et al., ICML 2022,
   [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)), trust score (Jiang et
   al., NeurIPS 2018, [arXiv:1805.11783](https://arxiv.org/abs/1805.11783)),
   SNGP (Liu et al., NeurIPS 2020,
   [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/file/543e83748234f7cbab21aa0ade66565f-Paper.pdf)).

### 2.2 Protocol points that make comparisons fair

- **Same backbone and compute budget.** Methods that alter training (ensembles,
  MC-dropout, energy fine-tuning) must be compared against single-model baselines
  with the same total compute (forward-pass-matched) — the DEE normalization of
  Ashukha et al. 2020.
- **Same training / calibration / evaluation splits; no tuning on the test set.**
  Calibration hyperparameters (temperature, ensemble size, MC-dropout T,
  conformity score) are chosen on a held-out calibration split; the test set is
  touched exactly once.
- **Matched confident populations in selective prediction.** Compare curves at
  *matched coverage* (equal retained fraction), not matched thresholds. When
  classifiers differ in accuracy, raw AURC and AUROC/AUPR are invalid (Ding et
  al. 2020) — use E-AURC / NAURC / AUGRC, or report full risk–coverage curves at
  matched coverage plus working-point metrics at several coverages (70/80/90%).
- **Same uncertainty estimator definition** across methods (entropy vs
  max-softmax vs variance; temperature before or after ensembling — Ovadia et al.
  show this matters).
- **Multiple seeds** (≥5) and report mean ± std, not single runs (Gundersen et
  al., ACM REP 2023,
  [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3589806.3600044)).
- **Multi-dataset / multi-shift reporting** — the benchmarks that made the field
  quantitative: Uncertainty Baselines (Nado et al. 2021,
  [arXiv:2106.04015](https://arxiv.org/pdf/2106.04015)) and Ovadia et al.
  (NeurIPS 2019, [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)).

### 2.3 Statistical rigor — establishing that a difference is real

- **Bootstrap CIs** for AUROC / AURC / E-AURC / NLL / Brier — resample the
  *paired* score/label pairs to preserve correlation; percentile or BCa. (Caveat:
  naive bootstrap undercovers for ECE near zero — use debiased intervals.)
- **DeLong test** for correlated AUROC (DeLong et al., Biometrics 1988,
  [DOI](https://doi.org/10.2307/2531595)) — the right test for "Mahalanobis vs
  disagreement" on the same examples.
- **McNemar's test** for paired binary flag decisions (Dietterich, Neural
  Computation 1998, [PubMed](https://pubmed.ncbi.nlm.nih.gov/9744903/)).
- **Paired multi-seed protocol** for small improvements: BCa bootstrap on
  per-seed deltas + sign-flip permutation (Du,
  [arXiv:2511.19794](https://arxiv.org/html/2511.19794)); clustered standard
  errors (Miller, [arXiv:2411.00640](https://arxiv.org/html/2411.00640)).
- **Multiple-testing control** across pathologies / selectivity levels / OOD
  sets (Bonferroni or FDR). Overlapping 95% CIs do *not* imply "no difference"
  (Longjohn et al., [arXiv:2501.04234](https://doi.org/10.48550/arxiv.2501.04234)).
- **Seed variability**: >25 seeds may be needed at α=10⁻³; non-parametric tests
  (Brunner–Munzel) should replace t-tests across seeds (Gundersen et al. 2023).

### 2.4 Common failure modes / critiques

1. **ECE is trivially gameable** (Ashukha et al. 2020; Haghtalab et al. 2024) —
   never let ECE be the only calibration evidence; pair with proper scores and
   hypothesis tests.
2. **Calibration collapses under distribution shift** (Ovadia et al. 2019):
   post-hoc calibration does not transfer under shift (Tomani et al., CVPR 2021,
   [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Tomani_Post-Hoc_Uncertainty_Calibration_for_Domain_Drift_Scenarios_CVPR_2021_paper.html)).
   A UQ paper must report calibration *and* selective/OOD metrics under shift.
3. **Well-calibrated ≠ useful for selective prediction** (Fisch et al.,
   [arXiv:2208.12084](https://arxiv.org/html/2208.12084)): the constant predictor
   E[Y] is calibrated but useless; TS fixes calibration but leaves ranking
   untouched.
4. **Confidence alone misses "confidently wrong" errors** — scores built on the
   model's own output cannot flag confident errors inside the training support.
   Entropy alone is insufficient for safe selective prediction (Phillips et al.,
   [arXiv:2603.21172](https://arxiv.org/html/2603.21172)).
5. **Ensembles are not automatically calibrated** — "temperature scaling is a
   must even for ensembles" (Ashukha et al. 2020). Comparing an uncalibrated
   single model against a calibrated ensemble conflates method quality with
   calibration protocol.
6. **Cross-model metric invalidity** — AUROC/AUPR (correct-vs-wrong) and raw
   AURC are not comparable across models of different accuracy (Ding et al. 2020;
   Galil et al. ICLR 2023,
   [OpenReview](https://openreview.net/forum?id=p66AzKi6Xim)).
7. **Benchmark surface area** — most UQ methods fail to maintain nominal coverage
   under shift (Kompa et al., Entropy 2021,
   [MDPI](https://www.mdpi.com/1099-4300/23/12/1608)); single-dataset conclusions
   do not generalize.

---

## 3. How recent CXR papers benchmark uncertainty

### 3.1 The anchor benchmark

**Baur, Samek & Ma — *Benchmarking Uncertainty and its Disentanglement in
multi-label Chest X-Ray Classification*, UNSURE@MICCAI 2025**
([arXiv:2508.04457](https://arxiv.org/abs/2508.04457)).

- **Setup:** 13 UQ methods (deep/shallow ensembles, SWAG, MC-Dropout, HET-XL,
  Het-NN, EDL, DDU, loss/correctness prediction, temperature scaling, masked /
  Gumbel attention) × ResNet-18 and ViT-Tiny × MIMIC-CXR-JPG, 14 pathologies,
  CheXpert-style {0,1,−1} labels. 5 seeds, 5 forward passes.
- **Metrics:** OOD-detection AUROC, uncertainty-label-prediction AUROC,
  correctness-prediction AUROC, AUAC (area under accuracy-coverage), ECE, MCE.
  Notably does **not** report Brier or NLL.
- **Key findings:**
  - **Deep and shallow ensembles win consistently** across tasks and both
    architectures; SWAG is excellent for OOD detection and best calibration on
    ResNet (MCE ~0.12).
  - **EDL is consistently ill-calibrated** (ECE ≈ 0.4).
  - ViTs calibrate better than ResNets.
  - The information-theoretic **epistemic/aleatoric disentanglement mostly
    fails** — EU and AU scores stay significantly correlated.
  - **No conformal methods are included** — there is no published head-to-head of
    conformal vs feature-based flags on a large CXR dataset.

### 3.2 Metric-surface / evaluation-practice papers

- **Mosquera et al.** (CHIL 2022 / *European Radiology* 2024,
  [arXiv:2112.12843](https://arxiv.org/abs/2112.12843)): audit of 68 CXR papers —
  **~90% report no calibration metric**; standard Brier is dominated by the
  majority class (a dummy all-zero model on a 1%-positive set gets Brier 0.01);
  proposes **Balanced Brier = Brier⁺ + Brier⁻** and AUC-PR for imbalanced CXR.
  The citation for why imbalanced OpenI results need stratified/balanced metrics.
- **Anthony & Kamnitsas** (Oxford, [arXiv:2309.01488](https://arxiv.org/abs/2309.01488)):
  Mahalanobis OOD in CXR (CheXpert); **no single optimal layer** — last-hidden
  layer often suboptimal; multi-branch Mahalanobis wins on pacemaker/sex OOD.
- **OpenMIBOOD** (CVPR 2025,
  [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Gutbrod_OpenMIBOOD_Open_Medical_Imaging_Benchmarks_for_Out-Of-Distribution_Detection_CVPR_2025_paper.pdf)):
  24 post-hoc OOD methods across medical imaging — **energy is near-bottom on
  near-OOD (~48.8% AUROC); feature-based (MDS, ViM, kNN) dominate**; ImageNet-tuned
  OOD methods transfer poorly to medicine.
- **Woodland et al.** (MELBA 2024,
  [MELBA](https://www.melba-journal.org/papers/2024:020.html)): **kNN on reduced
  features beats Mahalanobis** in medical segmentation (AUROC 69%→96%) because
  medical features are non-Gaussian. Directly relevant: kNN is the strongest
  unmeasured competitor for the RAD-DINO `[CLS]` Mahalanobis flag.
- **Wollek et al.** (Med Phys 2023, [DOI](https://doi.org/10.1002/mp.16790)):
  without OOD detection a CheXnet classifier discarded **zero** OOD images
  (AUC 0.5); in-distribution voting reached ~99.9% OOD AUC on knee/hand/bone
  X-rays vs Mahalanobis 98.2%, MaxLogit 72.6%, MaxEnergy 72.4% — feature /
  outlier-exposure methods dominate logit-based scores for CXR OOD.

### 3.3 Method papers (CXR)

- **Ghesu et al.** (MICCAI 2019; *MedIA* 2021,
  [DOI](https://doi.org/10.1016/j.media.2020.101855)): evidential (subjective
  logic) uncertainty; uncertainty-driven rejection improved ROC-AUC ~8% to 0.91
  at <25% rejection; uncertainty correlated with multi-radiologist label
  agreement. The early template for selective prediction in CXR.
- **Asgharnezhad et al.** (*Sci Rep* 2022,
  [Nature](https://www.nature.com/articles/s41598-022-05052-x)): introduces the
  **uncertainty confusion matrix** (True/False Certainty, UAcc/USen/USpe) —
  reused in several later CXR UQ papers.
- **Laksara & Thayasivam** (2025, [arXiv:2511.18839](https://arxiv.org/abs/2511.18839)):
  on NIH ChestX-ray14, **MC-Dropout failed catastrophically** (ECE 0.7588); a
  9-member deep ensemble reached AUROC 0.8559, ECE 0.0728, NLL 0.1916; aleatoric
  uncertainty (~0.31) dominates epistemic (~0.02) — consistent with label-noise-
  heavy CXR data.
- **Yang et al.** (CVPR-W 2019,
  [CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Yang_Learn_To_Be_Uncertain_Leveraging_Uncertain_Labels_In_Chest_X-rays_CVPRW_2019_paper.pdf)):
  treating CheXpert "uncertain" labels as a third class raises predictive
  variance for ambiguous cases; cross-dataset validation raises predictive
  entropy.
- **Arora et al.** (Bioengineering 2023,
  [MDPI](https://www.mdpi.com/2306-5354/10/8/946)): explicit "equivocal" class +
  label smoothing; MCE 0.150 with AUROC 0.82 internal / 0.84 external.

### 3.4 Selective prediction / rejection in CXR

- ***Knowing When Not to Decide*** (Research Square 2025,
  [DOI](https://doi.org/10.21203/rs.3.rs-6644332/v1)): entropy-based rejection
  beats decision-boundary-based on 4 pathologies × PadChest/NIH/MIMIC; average
  AUC 0.83 vs 0.79 baseline; performance drops under domain shift.
- **SelectiveCXR** ([GitHub](https://github.com/seancpc/SelectiveCXR)): three
  cross-backbone VLMs, inter-model disagreement + Mondrian conformal, three-way
  triage on MIMIC-CXR; honest negative that strict α=0.05 is unreachable in
  multi-label CXR.

### 3.5 Conformal prediction in CXR / medical imaging

- **Conformal Triage** (Angelopoulos, Bates et al., medRxiv 2024,
  [DOI](https://doi.org/10.1101/2024.02.09.24302543)): the canonical
  "refer-to-human with guarantees" design — Learn-Then-Test risk control with
  guaranteed PPV/NPV (≥95% each at 4.9% abstention) on head CT; false positives
  cut from 45% to 5% at 14% abstention. The model for the "conformal =
  in-distribution coverage layer" framing.
- ***Pitfalls of Conformal Predictions for Medical Image Classification***
  (Mehrtens et al., DKFZ, [arXiv:2506.18162](https://arxiv.org/abs/2506.18162)):
  coverage **holds in-distribution but is violated under input and label
  shift**; recalibration on ~1,000 shifted points restores it; per-class
  coverage gaps (size-1 sets over-covered); the authors argue CP "should not be
  used for selective classification." **This paper predicts exactly the
  conformal-under-shift collapse observed in this project** (Kermany 0.27, COVID
  0.25) and is why class-conditional coverage gaps must be reported, not just
  marginal coverage.
- **Risk-sensitive CP for catheter placement** (Long et al., 2025,
  [arXiv:2505.22496](https://arxiv.org/abs/2505.22496)): class-specific α
  (α=0.01 critical, α=0.1 standard); critical-condition coverage 99.29%, zero
  high-risk mispredictions. A natural extension for Pneumonia.
- **CONRep** (J. Imaging Informatics in Medicine 2026,
  [Springer](https://link.springer.com/article/10.1007/s10278-026-02179-5)):
  conformal report drafting; uncertain outputs flagged for mandatory radiologist
  review.
- **Fair/group-conditional CP**: Group-APS/GRAPS (Lu et al., AAAI 2022,
  [AAAI](https://doi.org/10.1609/aaai.v36i11.21459)); Mondrian CP for
  intracranial hemorrhage (Gamble et al., Mayo Clinic,
  [arXiv:2401.08058](https://arxiv.org/html/2401.08058)); conformal risk control
  for nodule detection (Hulsman et al., PMLR 2025,
  [arXiv:2412.20167](https://doi.org/10.48550/arxiv.2412.20167)).
- **Calibration-set sizing**: ~2,429 points for 90% coverage within ±0.01 at 90%
  tolerance (Marques, [arXiv:2303.02770](https://doi.org/10.48550/arxiv.2303.02770));
  contra view for scarce medical data (Kladny et al., Artif. Intell. Med. 2026,
  [DOI](https://doi.org/10.1016/j.artmed.2026.103462)).

### 3.6 Distribution shift — the stress test the field now demands

- **Ovadia et al.** (NeurIPS 2019): calibration collapses under shift severity;
  ensembles degrade most gracefully. The paper the under-shift framing hangs off.
- **Zech et al.** (*PLOS Med* 2018,
  [PLOS](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683)):
  cross-hospital calibration slope varied 0.047–10.4; internal AUC 0.931 →
  0.815 external.
- **MIDL 2020** cross-domain X-ray
  ([arXiv:2002.02497](https://arxiv.org/pdf/2002.02497)): failure is mostly
  **label/concept shift**, not covariate shift; **calibration operating points
  differ per dataset and require per-dataset recalibration**.
- **CheXphoto** (ML4H 2020,
  [PMLR](https://proceedings.mlr.press/v136/phillips20a.html)): the standard
  *image-quality* shift protocol (photos of CXRs + synthetic transformations) — a
  cheap third OOD axis beyond Kermany/COVID.
- **The Subgroup Imperative** (Radiology: AI 2023,
  [DOI](https://doi.org/10.1148/ryai.220270)): n=197,540 external test showing
  subgroup gaps by age/sex/ancestry/setting/pathology.

### 3.7 Clinical / human-AI collaboration (for the significance section)

- **Farzaneh et al.** (npj Digit Med 2023,
  [Nature](https://www.nature.com/articles/s41746-023-00797-9)): physician-aided
  AI with deferral beat both alone (0.869 vs 0.808/0.847) on ARDS; AI handled up
  to 79% of cases.
- ***On the Limits of Selective AI Prediction*** (2025,
  [arXiv:2508.07617](https://arxiv.org/abs/2508.07617)): the counterpoint — when
  the AI abstains, clinicians **underdiagnose (+18%) and undertreat (+35%)**.
  Deferral is not behaviorally neutral; must be addressed if clinical value is
  claimed.
- **Patel et al.** (npj Digit Med 2019,
  [Nature](https://www.nature.com/articles/s41746-019-0189-7)): low-confidence
  band routed to humans; augmented 92% vs AI-alone 82% vs human-alone 84%.
- **Care to Explain?** (Radiology 2024,
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/39560483/)): how confidence is
  *communicated* to radiologists changes trust calibration.

---

## 4. Where this project sits — and what to add

### 4.1 The edge, in one sentence

The project pairs a **distribution-free coverage guarantee** (Mondrian LAC
conformal, in-distribution) with a **feature-density error detector that beats
chance under shift** (Mahalanobis on frozen RAD-DINO `[CLS]` features), on
leak-free OpenI + two OOD probes — precisely the gap the UNSURE 2025 benchmark
leaves open (it excludes conformal methods entirely). The layered claim —
*disagreement fails under shift, conformal guarantees coverage in-distribution
but collapses under shift, Mahalanobis is the only detector that beats chance
under shift* — is a clean, defensible narrative. **The baseline comparison
(§4.4) sharpens the in-distribution half of that narrative**: no single member
can self-detect its own confident errors (≈chance AUROC), while a
disagreement+density hybrid flags them at 0.91 AUROC @ top-10% selectivity —
and pooled confidence, not the hybrid, is the abstention-policy score. **(The
0.91 is the OpenI small-n number; it deflates at power — §4.5 — and the
disagreement-vs-confidence result is diversity-type-dependent at the defining
top-10% metric and regime-dependent at broader selectivity — §4.6 + the §4.6
cross-backbone follow-up: same-arch from-scratch disagreement beats confidence
3/3 backbones, but architecture-diverse ensembles deflate at top-10%
*regardless of regime* — cross-arch from-scratch reproduces the frozen-foundation
deflation; regime dominates only at top-50%. **(And the deflation is not a
property of architecture-diverse ensembles at all: LP-FT-adapting the same four
foundation members on ReX-train moves the tail std-vs-conf result from a tie
(p=0.99) to a clear std advantage inside the from-scratch band (0.679–0.693,
p≤0.006) — §4.6.2. The failure is a data-shift artifact of off-domain
deployment, not an ensemble property.)**

### 4.2 Current state vs. the publishable standard

Status as of 2026-08-16 (items 1–6 of §4.3 implemented):

| Metric | Status | Where |
|---|---|---|
| Confident-error AUROC (std / MI / Mahalanobis) | ✅ | `evaluate.py` + `reanalyze.py` selectivity curve |
| Confidence-only baseline AUROC | ✅ | `evaluate.py` (`confidence_auroc`) + selectivity curve |
| AURC + **E-AURC** for every score | ✅ | `evaluate.py` (`aurc_by_score`: std / MI / confidence / Mahalanobis / kNN) |
| ECE multi-bin (5/10/15/20) + reliability diagram | ✅ | `evaluate.py` (`ece_multi`, `reliability`) + `viz.py` `05_reliability.png` |
| Brier with Murphy decomposition | ✅ | `evaluate.py` (`brier_decomposition`) |
| NLL / log-loss | ✅ | `evaluate.py` (`nll`) |
| OOD-detection AUROC + FPR@95TPR | ✅ | `scripts/eval_ood_detection.py` (from saved phase-4 arrays) |
| kNN-distance baseline | ✅ | `reanalyze.py --features` (kNN on the cal-split features) |
| Bootstrap CI / DeLong / McNemar | ✅ | `statistics.py` + `reanalyze_meta.json` `statistical_tests` |
| **Single-member confidence baselines** (all 4 members, two views) | ✅ | `scripts/eval_baselines.py` (§4.4, Panel A) |
| **Predictive entropy baseline** | ✅ | `scripts/eval_baselines.py` (§4.4, Panel B) |
| **TS-calibrated confidence** (rank-invariance demo) | ✅ | `scripts/eval_baselines.py` (§4.4) |
| **Hybrid disagreement+feature-density score** (cal-split weight) | ✅ | `scripts/eval_baselines.py` (§4.4) |
| **Bootstrap paired-difference CIs** (survive tiny error counts) | ✅ | `statistics.py` `bootstrap_paired_diff` (§4.4) |
| Conformal coverage | ✅ | Beta CI + class-conditional coverage gap still to add (Mehrtens pitfall) |
| Per-pathology ECE | ⚠️ | Not yet; add if the calibration section needs it |

### 4.3 Prioritized recommendations

Implemented (2026-08-16):

1. **NLL + Brier (with decomposition)** — always-on in `evaluate.py`.
2. **Confidence-only and kNN-distance baselines** — confidence is always-on
   (`confidence_auroc`, `auroc_confidence`); kNN attaches when `reanalyze.py
   --features <sidecar>` is passed (kNN of each eval image to the cal-split
   features). kNN on `features_raddino.npz` is the strongest competitor not
   previously measured.
3. **E-AURC** — `aurc_by_score` reports AURC + E-AURC for every score.
4. **OOD-detection AUROC + FPR@95TPR** — `scripts/eval_ood_detection.py` reads
   the saved `runs/phase4_features/arrays_*.npz` (OpenI-D vs Kermany/COVID) for
   Mahalanobis, energy, and kNN.
5. **Statistical rigor** — `cxr_uncertainty/statistics.py` (bootstrap CI,
   DeLong, McNemar, FPR@95TPR); `reanalyze_meta.json` gains a
   `statistical_tests` block. Wide top-5% CIs are reported honestly.
6. **Reliability diagram + multi-bin ECE** — `viz.py` `05_reliability.png` +
   `ece_multi` in the report.

Still open (incl. one documented omission):

7. **MC-Dropout** — deliberately omitted. The 4-member ensemble is
   DenseNet-family (xrv_nih) + ViT/convnext (no Dropout at inference), so
   MC-Dropout is a no-op for the members we have; a proper MC-Dropout baseline
   would need a fresh MC forward run of an architecture with Dropout
   (CPU-expensive, defers). Documented rationale, not an oversight.

8. Frame the paper against the **Baur et al. metric surface** and the
   **Mehrtens et al. pitfall list** explicitly — the project passes on
   under-shift, which most CXR UQ papers never attempt.
9. Conformal coverage **Beta CI** + **class-conditional coverage gap** (the
   Mehrtens pitfall) — report per-pathology coverage, not just marginal.
10. Per-pathology ECE if the calibration section needs it.

### 4.4 Baseline comparison results (2026-08-16)

Computed by `scripts/eval_baselines.py` entirely from the saved phase-4 arrays
(`runs/phase4_features_4mem/arrays_openiC/D.npz`, 4 members) — **no re-inference**.
A self-check reproduces the phase-4 pipeline to 4 decimals (`n_confident=2616`,
`n_confident_wrong=5`, std AUROC@10% = 0.6072, Mahalanobis@10% = 0.8698) before
any baseline numbers are emitted. Tables: `runs/baselines/baseline_table.md`,
machine-readable `runs/baselines/baseline_comparison.json`, selectivity plot
`runs/baselines/baselines_auroc_by_selectivity.png`.

**The three facts that survive scrutiny** (all on leak-free OpenI-D eval, cal =
2002 imgs / eval = 2012 imgs):

1. **Single-member confidence cannot self-detect its own errors.** Panel A —
   each member's own confidence AUROC against its *own* errors within its own
   top-X% confident set — is at or below chance everywhere: best is 0.53
   (raddino) at top-10%, and all four members are ≈0.13–0.31 at top-50%.
   There is no "pick the most confident single member" shortcut; a member
   cannot see that it is wrong about itself.

2. **A hybrid disagreement + feature-density score is the best confident-error
   detector at tight selectivity.** Weight `w` on std (rest on kNN distance) is
   selected on the **cal** split at top-50% (grid over {0, .25, .5, .75, 1},
   picks `w=0.75`, cal AUROC 0.7342) and then evaluated out-of-sample on eval:

   | score | AUROC @ top-10% | @ top-50% |
   |---|---|---|
   | **hybrid_std_knn** | **0.9059** | 0.7139 |
   | kNN distance | 0.8763 | 0.6805 |
   | Mahalanobis | 0.8698 | 0.6798 |
   | predictive entropy | 0.6510 | **0.7190** |
   | pooled confidence | 0.5058 | 0.6642 |
   | epistemic_std (the method alone) | 0.6072 | 0.6193 |

   Within the ensemble confident set, hybrid beats every single baseline at
   top-10% and every score except entropy at top-50%. The bootstrap paired
   difference CI at the **powered** level (top-50%, 75 confident errors) is
   **[0.050, 0.140]** vs the pure disagreement score — excludes 0, so the
   hybrid is significantly better than the method's own signal. The top-10%
   cell (5 errors) is honestly underpowered: every CI crosses 0 there.

3. **Confidence still owns risk-coverage (E-AURC); the hybrid does not.**
   On the full eval set, pooled-confidence E-AURC = **0.0135** vs
   hybrid 0.1011 and epistemic_std 0.0824; the paired E-AURC CI
   [0.084, 0.092] is entirely positive, i.e. the hybrid is *significantly
   worse* than confidence as a coverage score. The hybrid is a
   confident-error *detector*, not an abstention-policy score — report the
   two metrics separately, never conflate them.

**Secondary facts.** TS calibration (T=0.6335) cuts ECE 0.0484 → 0.0047 while
leaving the confident-error AUROC of -p̄ numerically unchanged (0.349 → 0.349):
calibration cannot fix ranking (Corbière et al.), so TS is a calibration
baseline only. Predictive entropy is the strongest single *free* score (0.719
at top-50%, above the 0.7846 best-single-member confidence within the ensemble
set) — cheap, and should always be in the table. DeLong stays usable at
top-50% (epistemic vs best-member arkswin: p=0.0, in the member's favor) but
returns `None` for the feature scores at top-10% (degenerate variance with 5
positives) — that is precisely why the bootstrap paired-difference CI is the
significance machinery that matters here.

**CXR-benchmark metrics (Baur UNSURE Task 3/4).** Per-pathology correctness
AUROC on the full eval set, macro-averaged over the 13 pathologies with valid
predictions (Consolidation is all-NaN in the eval arrays and dropped): entropy
0.9674, pooled confidence 0.9435, epistemic_std 0.9080, hybrid_std_knn 0.8785 —
all above the 0.70–0.75 band Baur et al. report for correctness prediction on
MIMIC-CXR. AUAC (area under accuracy-coverage, the complement of AURC on the
same risk-coverage curve) ranks the scores identically to E-AURC: confidence
0.9424 best, then epistemic_std 0.8736, entropy 0.8703, hybrid_std_knn 0.8549 —
the hybrid is a confident-error *detector*, not an abstention score, on both
metrics. Full per-pathology rows are in `baseline_comparison.json`.

**Publishable reframing.** The one-sentence claim in §4.1 must be updated: the
edge is *not* "ensemble disagreement alone is the best flag". It is: **single
members cannot self-detect errors (≈chance), and a disagreement+density hybrid
detects confident errors at 0.91 AUROC @ top-10% while pooled confidence
remains the abstention-policy score** — a separation of the detection and
coverage questions that most CXR UQ papers do not draw. OOD stays secondary
(Mahalanobis the only detector above chance under shift).

> **Update 2026-08-18 (see §4.5):** the confident-error edge reported here
> (hybrid 0.91 @ top-10%, mahalanobis 0.87) rests on **5 confident errors** and
> **does not survive power**. ReXGradient-160K (34 confident errors) gives
> hybrid 0.598, mahalanobis 0.53 (near chance), `epistemic_std` ≡ `confidence`.
> The OOD/shift mahalanobis result (Kermany/COVID, powered) stands. §4.4 is
> retained as the OpenI underpowered reference; §4.5 is the powered result that
> supersedes its confident-error reframing.

### 4.5 Powered leak-free results — ReXGradient-160K (2026-08-18)

ReXGradient-160K (2025; postdates every member's pretraining → leak-free by
recency) is a powered, in-distribution, frontal-only re-test of §4.4: **16,146
images, 34 confident errors at top-10% (6.8× OpenI's 5)**, with per-dataset
in-distribution calibration (Youden/TS/hybrid fit on ReX-cal, applied to
ReX-eval — the Baur-within-MIMIC protocol). Labels via the CheXpert rule-based
labeler on ReX free-text reports → NIH-14 (`CHEXPERT_TO_NIH`, 10 comparable
classes; −1 preserved for Task 2). Same 4-member ensemble, same
`eval_baselines.py`. Tables: `runs/baselines/rex/baseline_table.md`; JSON +
selectivity plot alongside. Full pipeline detail in `FINDINGS.md` Phase 5.

**The §4.4 confident-error edge does not survive power.** Panel B, top-10%
confident-error AUROC:

| score | OpenI (5 wrong) | ReX (34 wrong) |
|---|---:|---:|
| hybrid_std_knn (the §4.4 winner) | 0.9059 | 0.5979 |
| kNN distance | 0.8763 | 0.5942 |
| mahalanobis | 0.8698 | **0.5288** |
| predictive entropy | 0.6510 | 0.5473 |
| pooled confidence | 0.5058 | 0.5630 |
| epistemic_std (disagreement) | 0.6072 | 0.5626 |

- The hybrid 0.91 → 0.598; mahalanobis 0.87 → 0.53 (near chance). The §4.4
  "disagreement+density hybrid detects confident errors at 0.91" is **not
  supported at power** — it was two noise signals compounding on 5 errors.
- `epistemic_std` ≡ `confidence` (0.5626 vs 0.5630; DeLong p=0.99; McNemar
  p=0.98 — flags disagree on ~0 cases). A single member's confidence
  (conf_convnextv2 0.6077) beats the ensemble's disagreement (p=0.0007).
- Every hybrid-vs-baseline paired-bootstrap CI at top-10% includes 0 (34
  errors still too few for the small differences); at top-50% the hybrid is
  significantly *worse* than confidence (CI [−0.0865, −0.0439]).

**OOD detector stands; the in-distribution flag does not.** The Phase-4
mahalanobis *OOD* result (Kermany 0.818, COVID 0.649 on 244/313 errors) is
powered and unchanged — mahalanobis is a distribution-shift detector. Power
refutes only its *in-distribution confident-error* role (0.87 → 0.53).

**Baur Task 2/3/4 (ReX):**

| task | top scores | verdict |
|---|---|---|
| Task 3 — per-pathology correctness (macro, 10 classes) | confidence 0.9175, epistemic_std 0.8391, hybrid_std_knn 0.8372 | ensemble+feature UQ strong (~0.84) — **its real, distinct value** |
| Task 2 — uncertainty-label prediction (macro, 2509 −1 rows) | entropy 0.6983, hybrid_std_knn 0.6510, epistemic_std 0.6353 | tentative — ReX −1 are labeler-inferred, not expert; no expert-−1 dataset in scope (CANDID-III removed — NZ govt ban) |
| Task 4 — AUAC | confidence 0.9552, entropy 0.9259, epistemic_std 0.9093 | pooled confidence wins selective abstention, not disagreement |

*Task-4 protocol (Baur realization):* AUAC = ∫(1−risk)·d(coverage) over the
**certain-label eval records** (ReX 78,311; −1/uncertain rows excluded, used only
by Task 2). We abstain on the highest-uncertainty record first
(`argsort(-score)`, scores oriented HIGH = uncertain; `confidence` enters as
`-conf`) and track accuracy of the retained tail — exactly Baur's "iteratively
remove the most-uncertain, track accuracy," but at **record-level continuous
resolution** (all 78,311 points) rather than Baur's 5% discrete steps (same
quantity, finer grid). **Abstention unit = the per-pathology record**
(image × valid class), not a whole image; if Baur's unit is per-image/study
(MIMIC), that granularity difference folds into the existing ReX-vs-MIMIC
dataset/label confound — AUAC *magnitudes* aren't a clean head-to-head, but
the within-regime result (which score wins AUAC) is.

E-AURC (full coverage): confidence 0.0190 best; hybrid vs confidence CI
[0.0428, 0.0464] → hybrid significantly worse at full coverage. Calibration:
ECE 0.0219 → 0.0207 (TS, T=0.9662); rank-invariant, so TS cannot have caused
the deflation. **Energy fails on every ReX task** (AUAC 0.645, Task 2 0.335,
confident-error 0.451) — matches Baur + OpenMIBOOD. **kNN > mahalanobis**
(0.5942 vs 0.5288 confident-error; 0.6547 vs 0.6439 Task 3) — matches Woodland
2024 (non-Gaussian medical features).

**Reconciliation with Baur — and the contribution.** Baur's D-Ens (5
independent full trainings) and S-Ens (1 shared backbone + 5 diverse heads)
are *ensemble methods*, each yielding three scores: **PU = H(p̄)** (predictive
entropy of the pooled prediction = our `entropy` score; note our `confidence`
is max p̄, a different scalar, *not* PU), **AU** = mean member entropy (aleatoric;
computed internally as the second term of MI, not surfaced standalone), **EU =
PU − AU** = mutual information (= our `mutual_info`; our preferred
`epistemic_std` is a different estimator of the same epistemic concept — we
report both). Baur report **only the best of {PU,AU,EU} per method per task**,
so "ensembles win" hides *which score* carried it (on Task 1/OOD it was EU; on
Tasks 3/4 unspecified). Our result: the **pooled** score (PU = entropy / the
confidence family) wins the flagging/coverage tasks, not the **disagreement**
(EU) — we disentangle what Baur's max-over-scores reporting obscures. Baur's
AUAC band 0.82–0.84 (ResNet-18/ViT-Tiny, MIMIC, from-scratch) vs our 0.91–0.96
is a different regime (frozen foundation ensemble) and not a clean head-to-head.

**Updated publishable claim (supersedes §4.4's reframing; itself qualified by
the §4.6 cross-backbone follow-up for the defining metric).** Ensemble
disagreement does not improve confident-error flagging over pooled confidence
at power **in the architecture-diverse frozen-foundation deployment regime**;
the §4.4 0.91 was a 5-error artifact (mahalanobis 0.87 → 0.53 the centerpiece).
The ensemble's distinct, powered value is per-pathology correctness (Task 3,
~0.84) and — tentatively, pending expert −1 labels — uncertainty-label
prediction (Task 2). The value of disagreement is **diversity-type-dependent at
the defining top-10% metric and regime-dependent at broader selectivity**:
Baur's from-scratch *same-architecture* regime is where it wins at top-10%;
architecture-diverse ensembles deflate at top-10% *regardless of regime*
(cross-arch from-scratch reproduces the frozen-foundation deflation); regime
dominates only at top-50%. The decisive control — a from-scratch 5-seed ensemble
on ReX — **is done (§4.6)**, with the cross-backbone follow-up resolving the
regime×diversity confound; it turns "our method fails" into "our method's value
depends on architecture diversity (at high selectivity) and pretraining regime
(at broad selectivity)."

---

### 4.6 From-scratch D-Ens control — the regime test (2026-08-23)

*Experiment story (two-role split):* the **deployment system** — Agent 1
(UQ + confident-error flagging) and Agent 2 (similar-case retrieval,
docs/agent2_retrieval.md) — runs on the **pretrained members, individually
and cross-arch**. The from-scratch D-Ens here are the **UQ benchmark
control**: members initialized at random with **no prior exposure to any
data** (Baur's Deep-Ensemble regime), used to test whether §4.5's deflation
is a property of foundation deployment or of ensembled disagreement
generally — not a candidate deployment model.

§4.5's negative rests on a **frozen, architecture-diverse, foundation-pretrained**
ensemble. Baur et al. (UNSURE@MICCAI 2025) find disagreement *does* carry signal
in the **from-scratch, same-architecture** regime (5 independent random-init
trainings = D-Ens). §4.6 is the missing control: a 5-seed from-scratch D-Ens
for each of Baur's three anchors — ResNet-18, ViT-Tiny, ConvNeXt-Tiny — trained
on ReXGradient-train (112,967 frontal; CheXpert-rule labels, −1→1 for training,
−1 kept for Task 2), early-stopped on ReX-valid, evaluated on ReX-test (8,082)
with the **same splits, labels, and `eval_baselines.py`** as §4.5. seed-0 is
the feature member so the full baseline suite (kNN/maha/energy/hybrid) runs
unchanged. 15 models, H100, pre-decode memmap cache; pulled to local + patient
data deleted per DUA. Tables: `runs/baselines/rex_fs_<bb>/baseline_table.md`.

**The defining metric (same-architecture): disagreement beats confidence in from-scratch.** Confident-error AUROC, top-10% (within
the ensemble's confident set — the flag's population). NB: these are
same-architecture (cross-seed) ensembles; the cross-backbone follow-up below
shows the top-10% result is diversity-type-dependent, not regime-dependent —
architecture-diverse from-scratch *reproduces* the §4.5 deflation.

| score | §4.5 pretrained | resnet18 FS | vit_tiny FS | convnext_tiny FS |
|---|---:|---:|---:|---:|
| epistemic_std (disagreement) | 0.5626 | 0.6360 | 0.7190 | 0.6688 |
| confidence (pooled p_bar) | 0.5630 | 0.5585 | 0.6736 | 0.5385 |
| entropy (PU = H(p_bar)) | 0.5473 | 0.6587 | 0.7610 | 0.7155 |
| mahalanobis | 0.5288 | 0.5266 | 0.5132 | 0.5236 |
| Δ(std − conf) | −0.0004 | +0.0775 | +0.0454 | +0.1303 |
| DeLong p (std vs conf, top-10%) | 0.99 | 0.0071 | 0.0603 | 0.0 |
| DeLong p (std vs conf, top-50%) | — | 0.0 | 0.0 | 0.0 |

- **From-scratch, same-architecture: disagreement beats confidence — 3/3 backbones.** Gap +0.045
  to +0.130 AUROC; top-50% DeLong p=0.0 for all three, top-10% p=0.0071 / 0.0603
  / 0.0. **Frozen-foundation: the same test gave p=0.99 (≡).** This reproduces
  Baur's finding on ReX. **But the cross-backbone follow-up below qualifies the
  attribution**: at the defining top-10% metric the driver is diversity-*type*
  (architecture-diverse from-scratch *also* deflates, 0.5585 ≡ confidence), not
  regime; regime dominates only at broader top-50% selectivity. "Disagreement
  doesn't help" is scoped to *architecture-diverse deployment at high
  selectivity*.
- **Mechanism = member weakness *plus* same-architecture disagreement.** Panel A
  (each member's own confidence vs its own errors) is at/below chance for all
  three from-scratch backbones (best member top-10% AUROC 0.47 / 0.43 / 0.52) —
  a single from-scratch member cannot self-detect its errors; D-Ens value is
  *cross-member* disagreement. Frozen-foundation members are strong individuals
  that *agree* → disagreement is a rescaled confidence. The two regimes are two
  points on the member-strength axis. **But member weakness alone is
  incomplete** — the cross-backbone follow-up shows cross-arch from-scratch
  members are equally weak yet *deflate* at top-10%; the top-10% signal needs
  *same-architecture* (cross-seed) disagreement, which tracks data-ambiguity.
- **Predictive entropy (PU) is the best single score in from-scratch** (0.66 /
  0.76 / 0.72) — above both std and confidence, because PU = H(p_bar) folds in
  both. In §4.5 pretrained, entropy (0.547) was *below* confidence — its
  ranking flips with the regime, tracking where the signal lives.

**Cross-regime invariants (robust to the regime):**

- **Mahalanobis is near-chance in-distribution in both regimes** (0.53 / 0.51 /
  0.52 FS vs 0.53 pretrained) — a distribution-shift detector, not an
  in-distribution confident-error flagger. The §4.5 conclusion is universal,
  not foundation-specific. (The *OOD* mahalanobis result — Kermany 0.818,
  COVID 0.649 — is a different question and stands.)
- **Pooled confidence wins full-coverage abstention in both regimes.** AUAC:
  confidence 0.955 / 0.938 / 0.943 FS vs epistemic_std 0.923 / 0.903 / 0.899
  (§4.5: confidence 0.955 vs std 0.909); E-AURC confidence lowest; hybrid-vs-
  confidence E-AURC CI excludes 0 on the worse side for all three ([0.030,0.033]
  / [0.034,0.037] / [0.042,0.044]). Disagreement's edge is specifically *within*
  the confident set: confidence is the right signal for *whether to abstain*
  (monotone in error probability), but once a confident prediction is
  committed, confidence is saturated and cannot discriminate among confident
  predictions — and that is where disagreement (from-scratch) takes over.
- **Density is regime-dependent.** kNN is the best density in foundation
  (0.594, above confidence) on frozen RAD-DINO `[CLS]` features but **adds
  nothing from-scratch**: the hybrid selector picks `weight_on_std=1.0` for all
  three, so `hybrid_std_knn ≡ epistemic_std`. maha/kNN/energy are all near/below
  chance from-scratch. Energy fails everywhere (Baur + OpenMIBOOD, both
  regimes).

**Baur Task 2/3/4 (from-scratch, ReX-test):**

| task | resnet18 | vit_tiny | convnext_tiny | verdict |
|---|---|---|---|---|
| Task 3 — per-pathology correctness (macro, 10) | std 0.893, entropy 0.933, conf 0.915 | std 0.863, entropy 0.925, conf 0.905 | std 0.882, entropy 0.930, conf 0.908 | ensemble scores strong (~0.88–0.93) — regime-**robust** |
| Task 2 — uncertainty-label (2509 −1 rows) | std 0.708, entropy 0.744, conf 0.687 | std 0.686, entropy 0.720, conf 0.670 | std 0.695, entropy 0.730, conf 0.675 | disagreement > confidence — tentative (ReX −1 labeler-inferred) |
| Task 4 — AUAC | conf 0.955, entropy 0.930, std 0.923 | conf 0.938, entropy 0.916, std 0.903 | conf 0.943, entropy 0.911, std 0.899 | pooled confidence wins abstention (both regimes) |

Calibration is not the explanation: TS T = 0.86 / 0.83 / 0.85, ECE 0.023→0.012
/ 0.025→0.009 / 0.025→0.013, rank-invariant (AUROC(−p_bar) raw = TS for all
three) — TS cannot have moved the confident-error ranking; std>conf is a regime
effect.

*Two regions, two metrics — why Task 4 (AUAC) and the confident-error metric
point in opposite directions on the same ensemble.* The pair "confidence wins
AUAC (0.955 vs std 0.923, full set)" and "disagreement wins confident-error
AUROC (std 0.636 vs conf ≈0.48 = chance, top-10% only)" is not a
contradiction: the two metrics sample different regions of the distribution,
and both outcomes are consequences of the same low-dynamic-range (squeeze)
property of from-scratch models.

1. **Different sampling frames.** Task 4 integrates (1−risk) over the *entire*
   coverage range — its area is dominated by the bulk of the 78,311
   certain-label records. The confident-error metric conditions on the
   top-10% most-confident sliver by design. A signal that only wins in the
   tail can lose the integral while still winning where confident-error
   triage actually happens; and the tail contributes ≤~10% of the coverage
   axis, so even a large tail win barely moves AUAC.
2. **Why confidence dominates the bulk.** Pooled confidence is a
   first-order statistic — an average of 5 members' predictions, with direct
   variance reduction from ensembling. Epistemic std is a second-order
   statistic estimated from only 5 samples — noisier. Over most of the
   distribution "how sure is the ensemble" is simply the better estimator of
   "is it right" (conf 0.955 > entropy 0.930 > std 0.923).
3. **Why disagreement wins the tail.** In the top-10% band the pooled mean
   has *saturated*: by the space-squeeze (low dynamic range of from-scratch
   backbones — verified independently in the Agent-2 retrieval feature
   spaces, docs/agent2_retrieval.md §2.1: NN cosine ≈ 0.97 vs a 0.66
   random-pair background, z ≈ 1.7σ vs RAD-DINO's z ≈ 7.7σ) all
   confidently-predicted records have similar p̄, so within the band
   confidence's ranking is ~chance (0.48) at separating the wrong ones.
   Member disagreement still varies across the band and correlates with
   error (0.636). The squeeze is what *creates* a tail regime where
   disagreement holds information that confidence has exhausted.
4. **Same low-dynamic-range story in both spaces.** Feature space: random
   pairs sit at 0.66–0.71 cosine (everything crowds into a narrow cone) so a
   0.97 neighbor is weak evidence (~1.7σ) while RAD-DINO's 0.58 is strong
   (~7.7σ). Probability space: squeezed confidences (pooled-median ≈
   0.16–0.32, little saturation) leave residual member-to-member variation
   that correlates with error — which is exactly why Phase-6's headline is
   "disagreement beats confidence" only in the from-scratch regime: weak
   members compress everything toward the middle, and the interesting signal
   lives in the residuals relative to that narrow background.
5. **Baur-compatible.** Baur report best-of-{PU, AU, EU} per task; the
   Task-4-style full-coverage winner (PU/confidence family) and the Task-2/3
   error-detection winner (EU/disagreement) are different regions of the same
   risk-coverage trade-off, not rival Olympic tables. Per-regime usage rule
   implied by (1)–(3): full-coverage selective abstention → pooled
   confidence; triage of confidently-wrong cases within the confident
   stratum → disagreement.

**Reconciliation with Baur.** Our D-Ens is now Baur's exact object (5 full
trainings of one backbone) on ReX for all three anchors. Baur report only the
**best of {PU, AU, EU} per method per task**, hiding *which score* carried it;
we disentangle: in from-scratch the **disagreement (EU)** carries confident-
error flagging (above confidence/PU), while **PU (= entropy / the confidence
family)** carries selective abstention (AUAC) in both regimes. Our from-scratch D-Ens AUAC
(0.90–0.95) is above Baur's 0.82–0.84 (ResNet-18/ViT-Tiny, MIMIC) — a
dataset/label confound (ReX CheXpert-rule vs MIMIC expert labels, different base
error rate), not a regime statement; the contribution is the within-regime
structural comparison (disagreement vs confidence), clean on ReX. The §4.5
residual confound (architecture-diverse frozen vs same-arch trained —
"different objects") is resolved by the cross-backbone follow-up below: the
*same* disagreement score is signal for same-architecture (cross-seed) and
null for architecture-diverse, *regardless of regime* at the defining top-10%
metric — so the two findings are not contradictory but reflect two diversity
types (data-ambiguity vs inductive-bias-difference) at the defining metric,
and two regimes (from-scratch vs foundation) at broader selectivity.

**Updated publishable claim (supersedes §4.5's "honest negative"; itself
superseded for the defining metric by the cross-backbone follow-up below).**
The value of ensemble disagreement for confident-error flagging is
**diversity-type-dependent at high selectivity and regime-dependent at broad
selectivity**: from-scratch same-architecture (Baur's regime; weak diverse
members that disagree where they err) it beats pooled confidence (3/3
backbones, +0.045–0.130 AUROC, top-50% p=0.0); architecture-diverse ensembles
deflate at top-10% *regardless of regime* (cross-arch from-scratch reproduces
the frozen-foundation deflation, 0.5585 vs 0.5626); frozen-foundation
deployment it is ≡ confidence (p=0.99) and pooled confidence suffices.
Per-pathology correctness (Task 3, ~0.88–0.93) and OOD detection (mahalanobis,
Kermany 0.818) are regime-**robust**; confident-error flagging and the
kNN-vs-maha density ranking are diversity-** and regime-**dependent**. The
§4.5 negative is thereby scoped to "architecture-diverse foundation
deployment at high selectivity," not "the method fails" — the from-scratch
control is what gives the negative its meaning (as §4.5 itself flagged).

**Caveats.** (1) ReX −1 are labeler-inferred, not expert — Task 2 stays
tentative; an expert-−1 dataset would be the real test, but none is in scope
(CANDID-III, the planned one, is removed from scope — the NZ government has
banned use of that dataset). (2) From-scratch models are
Baur-protocol-cheap (224 px, ~25 epochs, early-stop, not tuned to SOTA) —
comparable, not strongest. (3) ReX-vs-MIMIC dataset confound on AUAC magnitude
remains; the structural disagreement-vs-confidence comparison does not. (4)
5-seed single-backbone D-Ens is Baur's object — matched; S-Ens (shared backbone
+ diverse heads) is not run here.

#### 4.6.1 Cross-backbone from-scratch — the regime×diversity resolution (2026-08-23)

The §4.5-vs-§4.6 comparison above confounds **two axes**: training regime
(pretrained vs from-scratch) *and* diversity type (architecture-diverse vs
same-architecture). §4.5 = arch-diverse + pretrained (deflates); §4.6 =
same-arch + from-scratch (signal). The missing cell — an **architecture-diverse
from-scratch** ensemble — isolates the two. It is free to build: the three
backbones' per-member probs already live in the per-backbone npz with identical
image ordering (verified), so a cross-arch ensemble is just stacking seed-0
probs across backbones — **no new training, no new inference**
(`scripts/build_crossarch_arrays.py`). Built two ensembles: `rex_fs_crossarch`
(M=3: {resnet18_s0, vit_tiny_s0, convnext_tiny_s0}, arch-diverse from-scratch)
and `rex_fs_resnet18_m3` (M=3: {resnet18 s0,s1,s2}, same-arch from-scratch at
matched M=3 to remove the member-count confound). Baselines:
`runs/baselines/rex_fs_crossarch/`, `runs/baselines/rex_fs_resnet18_m3/`.
**M=2 robustness variants** (same script, free recombine): `rex_fs_crossarch_rv`
= {resnet18_s0, vit_tiny_s0} — pure-CNN-vs-pure-transformer endpoints (ConvNeXt
is a CNN with ViT-inspired design, no self-attention, so the M=3 crossarch is
2 CNNs + 1 ViT, CNN-heavy on the inductive-bias axis; this pair is the clean
CNN↔TX contrast) — plus matched `rex_fs_resnet18_m2` = {resnet18 s0,s1} same-arch
control so member-count stays unconfounded at M=2.

**The defining top-10% metric is diversity-type-dependent, not regime-dependent.** Confident-error AUROC, std (disagreement) vs confidence:

| ensemble | diversity | regime | M | std t10 | conf t10 | Δ₁₀ | p₁₀ | std t50 | conf t50 | p₅₀ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| §4.5 | arch-diverse | pretrained | 4 | 0.5626 | 0.5630 | −0.000 | 0.99 | 0.5808 | 0.6775 | 0.0 (std worse) |
| **crossarch** | **arch-diverse** | **from-scratch** | 3 | **0.5585** | 0.6093 | **−0.051** | 0.22 | 0.8048 | 0.7642 | 0.0 (std better) |
| resnet18_m3 | same-arch | from-scratch | 3 | 0.6453 | 0.5455 | +0.100 | 0.012 | 0.7923 | 0.7476 | 0.0 |
| §4.6 (×3) | same-arch | from-scratch | 5 | 0.636–0.719 | 0.54–0.67 | +0.05–0.13 | ≤0.06 | 0.81–0.85 | 0.74–0.76 | 0.0 |
| **crossarch_rv** | **arch-diverse (CNN↔TX)** | **from-scratch** | 2 | **0.5266** | 0.5348 | **−0.008** | 0.79 | 0.7605 | 0.7494 | 0.043 |
| resnet18_m2 | same-arch | from-scratch | 2 | 0.6159 | 0.4773 | +0.139 | 0.0 | 0.7646 | 0.7444 | 0.0003 |

- **At fixed regime (from-scratch), diversity varied** — the decisive top-10%
  axis: arch-diverse `crossarch` deflates (std 0.5585 **below** confidence,
  p=0.22); same-arch `resnet18_m3` wins (std 0.6453 > conf, p=0.012). The M=3
  control shows member-count isn't the cause (M=3 same-arch still wins).
- **At fixed diversity (arch-diverse), regime varied** — regime does **not**
  rescue the top-10% deflation: pretrained std 0.5626 ≈ from-scratch `crossarch`
  std 0.5585 (both ~0.56, both ≤ confidence). Switching pretrained→from-scratch
  moved the arch-diverse disagreement by **−0.004**; switching arch-diverse→same-
  arch (at fixed from-scratch) moved it by **+0.08 to +0.16**. → architecture
  diversity, not foundation pretraining, drives the top-10% deflation. **This
  revises the §4.6 "regime-dependent" attribution above for the defining
  metric.**
- **M=2 robustness (rules out a ConvNeXt/"hybrid" artifact and member-count).**
  The M=3 `crossarch` is 2 CNNs + 1 ViT (ConvNeXt shares ResNet's conv inductive
  bias; it is a CNN with ViT-inspired design, no self-attention). The pure-CNN-
  vs-pure-transformer pair `crossarch_rv` {ResNet-18, ViT-Tiny} at M=2 **deflates
  even harder** at top-10% (std 0.5266 ≤ conf 0.5348, p=0.79) than the M=3
  crossarch (0.5585, p=0.22) — the cleanest inductive-bias-difference contrast
  deflates most. The matched M=2 same-arch control `resnet18_m2` **wins** (std
  0.6159 vs conf 0.4773, p=0.0, Δ=+0.139 — larger than the M=3 control's +0.100).
  So at M=2 the diversity-type gap is +0.089 (crossarch_rv→resnet18_m2), almost
  identical to the M=3 gap +0.087 — the top-10% diversity effect is ~+0.09 AUROC
  **independent of M**, and is not an artifact of ConvNeXt or of having 2 CNNs.
  Top-50% replicates the regime effect at M=2 (both from-scratch cells std > conf:
  resnet18_m2 p=0.0003; crossarch_rv p=0.043, weaker — the broad-selectivity
  regime effect shrinks with fewer members but keeps its sign).

**Selectivity splits the two axes (the nuance):**

- **top-10% (defining, high selectivity): diversity-type dominates.** arch-
  diverse deflates regardless of regime; same-arch from-scratch wins.
- **top-50% (broader selectivity): regime dominates.** all three from-scratch
  cells (arch-diverse *and* same-arch) have std > confidence (p=0.0); the
  pretrained arch-diverse cell has std < confidence (p=0.0). So from-scratch
  (weak, high-variance members) gives disagreement that beats confidence at
  broad selectivity regardless of diversity; pretrained (strong, low-variance)
  does not.

**Mechanism (revised — "member weakness" alone is incomplete).** Cross-arch
from-scratch members are weak too, yet cross-arch deflates at top-10%. The
cleaner story: **same-arch cross-seed disagreement = where the training data is
ambiguous/noisy** (shared inductive bias, different init) → aligns with errors
at all selectivity; **cross-arch disagreement = where inductive biases differ**
→ aligns with errors at broad selectivity but decouples from the *rarest*
confident errors at high selectivity. Member weakness explains the top-50%
regime effect (weak members disagree more → signal); diversity-type explains
the top-10% effect. Note also the diversity tradeoff: cross-arch *improves* the
pooled mean (crossarch confidence 0.609 > same-arch 0.546 at top-10%) while
eroding the disagreement flag — a better ensemble mean, a worse disagreement
flag.

**Reconciliation with Baur (tighter).** Baur's D-Ens is **same-architecture**
(5 seeds) — exactly the diversity type where disagreement wins at the defining
metric. Our §4.5 is **architecture-diverse** — the type where it deflates. So
the §4.5 vs Baur difference is a diversity-type difference (at top-10%), not
(primarily) a foundation-pretraining difference; Baur's protocol is the
favorable one. The regime effect is real but lives at broader selectivity
(top-50%), not at the defining top-10% flag.

**Corrected publishable claim (supersedes the §4.6 "regime-dependent" claim for
the defining metric).** Ensemble disagreement's value for confident-error
flagging is **diversity-type-dependent at high selectivity** (the defining
top-10% metric: architecture-diverse deflates — cross-arch from-scratch
reproduces the §4.5 pretrained deflation, 0.5585 vs 0.5626, despite the opposite
regime; same-arch from-scratch wins, 0.636–0.719) **and regime-dependent at
broader selectivity** (top-50%: from-scratch beats confidence regardless of
diversity; pretrained does not). The §4.5 deflation at the defining metric is
primarily an **architecture-diversity** effect, not a foundation-pretraining
effect. Same-arch disagreement is a data-ambiguity signal (error-aligned);
cross-arch disagreement is an inductive-bias-difference signal (error-aligned
at broad selectivity, decoupled at high selectivity). Baur's D-Ens is
same-architecture — the favorable diversity type — which is why his
"disagreement wins" reproduces only in the same-arch cell here.

#### 4.6.2 ReX-adapted pretrained ensemble — adaptation re-emerges the std edge (2026-08-31)

§4.6.1 left one competing explanation open: the frozen pretrained members are
**OOD on ReX**, so their tail deflation could be a *data-shift* artifact rather
than an architecture-diversity property. The Phase-7 adaptation arm closes
that gap in the deployment-realistic direction: LP-FT each foundation member on
patient-disjoint ReXGradient-train (112,967; masked BCE NIH-14, u_policy="ones",
early-stop on cal certain-label macro-AUROC — identical protocol to §4.6), then
re-run the **identical** `eval_baselines` battery on cal 8,064 / eval 8,082.
Adaptation per Route A: convnextv2 LP-FT ×3 seeds, xrv_nih LP-FT (NIH head kept
as warm start), raddino LP + one short low-LR FT (protecting retrieval features),
arkswin linear-probe-only @768. Per-seed artifacts:
`runs/rex_phase7/{eval_arrays,baselines}/rex_adapted_s{0,1,2}`; checkpoints
`checkpoints/rex_adapted/`. Patient data deleted from the GPU host per DUA.

**Pre-registered headline — top-10%-confident std-vs-conf AUROC:**

| regime | tail std AUROC | tail conf AUROC | DeLong p |
|---|---|---|---|
| un-adapted pretrained (§4.5) | 0.5626 | 0.5630 | 0.99 |
| from-scratch control (§4.6) | 0.636–0.719 | 0.559–0.674 | — |
| **adapted, seed 0** | **0.6789** | **0.5417** | **0.006** |
| adapted, seeds 1 / 2 | 0.6832 / 0.6927 | 0.5540 / 0.5331 | 0.0045 / <0.0001 |

All three seeds land **inside the from-scratch band** while confidence itself
*weakens* in the tail after adaptation (0.533–0.554). The architecture-diversity
deflation at the defining metric is therefore a **data-shift artifact**: remove
the OOD mismatch (by in-domain adaptation) and cross-arch disagreement becomes
error-aligned exactly as the same-arch from-scratch control predicted. Within
the tail, epistemic std beats every per-member confidence (best 0.558,
DeLong p=0.011) and Mahalanobis (0.546, p≈0.004); predictive **entropy** is the
strongest tail signal overall (0.7553 @ top-10) — predictive and epistemic
uncertainty carry complementary tail information.

**Two-regime structure survives adaptation.** Full-coverage AUAC still favors
pooled confidence (adapted: 0.9653 vs std 0.9411, full-set DeLong p≈0) — but the
margin narrows sharply against the un-adapted run (0.9552 vs 0.9093), because
adaptation lifted std's ranking power everywhere (E-AURC 0.0649 → 0.0414) while
confidence barely moved (0.019 → 0.0172). The §4.1 layered claim now holds
*conditioned on being in-domain*: cross-arch disagreement fails at the defining
metric **when the ensemble is deployed off-domain**; after the adaptation a
hospital would perform, it recovers to from-scratch-control levels. Note the
adaptation probe alone reaches 0.8668 cal macro-AUROC on raddino's frozen
self-supervised features (vs 0.8854 after FT) — ReX transfer is nearly total
without touching the encoder, which is why the adapted ensemble is a realistic
deployment state rather than a heavily-retrained one. Methodologically: this is
the **deployment-lens control** for any "disagreement fails" claim — the same
UQ machinery evaluated before and after routine fine-tuning can flip from tie to
significant advantage (p=0.99 → p<0.001), so the OOD state of the deployed
ensemble is itself a first-class variable in the evaluation design.

---

## 5. Key references (methodology)

- Guo, Pleiss, Sun & Weinberger — *On Calibration of Modern Neural Networks*,
  ICML 2017. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
- Naeini, Cooper & Hauskrecht — *Obtaining Well Calibrated Probabilities Using
  Bayesian Binning*, AAAI 2015. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9602)
- Nixon, Dusenberry & Jerfel — *Measuring Calibration in Deep Learning*, CVPR-W
  2019. [arXiv:1904.01685](https://doi.org/10.48550/arxiv.1904.01685)
- Roelofs et al. — *Mitigating Bias in Calibration Error Estimation*, AISTATS
  2022. [PMLR](https://proceedings.mlr.press/v151/roelofs22a.html)
- Vaicenavicius et al. — *Evaluating Model Calibration in Classification*,
  AISTATS 2019. [PMLR](https://proceedings.mlr.press/v89/vaicenavicius19a.html)
- Lee et al. — *T-Cal: An Optimal Test for the Calibration of Predictive
  Models*, JMLR 2023. [JMLR](https://jmlr.org/papers/volume24/22-0320/22-0320.pdf)
- Gneiting & Raftery — *Strictly Proper Scoring Rules, Prediction, and
  Estimation*, JASA 2007.
  [PDF](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)
- Geifman, Uziel & El-Yaniv — *Bias-Reduced Uncertainty Estimation for Deep
  Neural Classifiers*, ICLR 2019 (E-AURC). [arXiv:1805.08206](https://ar5iv.labs.arxiv.org/html/1805.08206)
- Ding et al. — *Revisiting the Evaluation of Uncertainty Estimation*, CVPR-W
  2020. [CVF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w1/Ding_Revisiting_the_Evaluation_of_Uncertainty_Estimation_and_Its_Application_to_CVPRW_2020_paper.pdf)
- Corbière et al. — *Addressing Failure Prediction by Learning Model
  Confidence*, NeurIPS 2019.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/757f843a169cc678064d9530d12a1881-Paper.pdf)
- Gal & Ghahramani — *Dropout as a Bayesian Approximation*, ICML 2016.
  [PMLR](https://proceedings.mlr.press/v48/gal16.html)
- Lakshminarayanan et al. — *Simple and Scalable Predictive Uncertainty
  Estimation using Deep Ensembles*, NIPS 2017.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)
- Lee et al. — *A Simple Unified Framework for Detecting OOD Samples and
  Adversarial Attacks* (Mahalanobis), NeurIPS 2018. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- Sun et al. — *OOD Detection with Deep Nearest Neighbors*, ICML 2022.
  [arXiv:2204.06507](https://arxiv.org/abs/2204.06507)
- Ovadia et al. — *Can You Trust Your Model's Uncertainty?*, NeurIPS 2019.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/file/8558cb408c1d76621371888657d2eb1d-Paper.pdf)
- Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction*.
  [arXiv:2107.07511](https://doi.org/10.48550/arxiv.2107.07511)
- Romano, Sesia & Candès — *Classification with Valid and Adaptive Coverage*,
  NeurIPS 2020.
  [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)
- DeLong et al. — *Comparing the Areas under Two or More Correlated ROC Curves*,
  Biometrics 1988. [DOI](https://doi.org/10.2307/2531595)
- Dietterich — *Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms*, Neural Computation 1998.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/9744903/)
- Ashukha et al. — *Pitfalls of In-Domain Uncertainty Estimation and Ensembling
  in Deep Learning*, 2020. [arXiv:2002.06470](https://doi.org/10.48550/arxiv.2002.06470)
