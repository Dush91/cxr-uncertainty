# CXR Epistemic-Uncertainty / Confident-Error Flagging — Findings

A leak-free, multi-architecture chest-X-ray ensemble that flags **confidently-wrong**
predictions. This document consolidates the end-to-end research arc and results
(Parts 1–3, Phases 0–4 + step 6) into one readable place. Full research/theory
lives in `.claude/plans/refactored-wobbling-orbit.md` (§A–§J); the per-session
working log with every decision's "why" is in
`.claude/projects/-teamspace-studios-this-studio/memory/cxr-uncertainty-project.md`.

---

## TL;DR (the decisive results)

1. **Cross-member disagreement fails under distribution shift.** The TS-only
   disagreement flag — the production in-distribution design — is **at/below
   chance** on powered, leak-free OOD probes: Pneumonia confident-error AUROC
   **0.515 (Kermany)** and **0.478 (COVID)** vs 0.5 chance. Two compounding
   failures, both predicted by theory: (a) the in-distribution temperature
   calibration shatters under prevalence/scanner shift (ECE 0.055 → 0.72–0.74);
   (b) the *confident-agree-wrong* core (Abe NeurIPS 2022) is invisible to
   disagreement — under shift all architectures confidently agree on the wrong
   answer.
2. **Feature-based Mahalanobis on RAD-DINO `[CLS]` features beats chance under
   shift where disagreement cannot.** A 2-class Gaussian (Ledoit-Wolf) on the
   frozen ViT member's 768-d `[CLS]` features, fit on in-distribution OpenI,
   gives Pneumonia confident-error AUROC **0.818 (Kermany)** and **0.649 (COVID)**
   — and dominates in-distribution too (0.835 vs 0.631 for disagreement). A
   feature-*density* score sidesteps the prevalence-calibration collapse and
   catches the confident-agree-wrong core. **This is the production
   distribution-shift / OOD detector.** (Caveat added by Phase 5: the
   *in-distribution* "0.835 vs 0.631" line above was on 8 confident errors and
   **deflates at power** — see result 3. The *OOD* numbers, 0.818/0.649 on
   244/313 errors, are powered and stand.)
3. **At usable power the confident-error flagging thesis is not supported.**
   ReXGradient-160K (leak-free by recency, 34 confident errors at top-10% vs
   OpenI's 5) shows `epistemic_std` ≡ `confidence` (DeLong p=0.99; flags
   disagree on ~0 cases) and **mahalanobis collapses 0.87 → 0.53** (near
   chance) — the OpenI in-distribution advantage was a small-n artifact. A
   single member's confidence beats the ensemble (conf_convnextv2 0.608 vs
   0.563, p=0.0007). The ensemble's real, distinct value is **per-pathology
   correctness (Task 3, ~0.84)**, not the defining confident-error flag. The
   density with legs is **kNN** (not maha); **energy fails** on every task —
   both match the literature. See Phase 5 for the full Baur Task 2/3/4 tables
   and the reconciliation (Baur's "ensembles win" reports max-over-scores and
   hides that the pooled score PU, not the disagreement EU, carries it).
4. **The Phase-5 deflation is diversity-type-dependent at the defining metric
   — resolved by the from-scratch control + a cross-backbone test (Phase 6).**
   A 5-seed from-scratch D-Ens (Baur's protocol) on the *same* ReX splits for
   ResNet-18, ViT-Tiny, ConvNeXt-Tiny shows disagreement *beats* confidence for
   confident-error flagging — gap +0.045 to +0.130 AUROC, top-50% DeLong p=0.0
   for all three — the opposite of the frozen-foundation p=0.99. **But a cross-
   backbone from-scratch ensemble** ({resnet18_s0, vit_tiny_s0,
   convnext_tiny_s0}, recombined from existing probs — no new training)
   **reproduces the Phase-5 deflation** (std 0.5585 ≈ pretrained 0.5626, both
   ≤ confidence at top-10%), despite the opposite regime. So at the defining
   top-10% metric the driver is **architecture diversity, not pretraining
   regime**: same-arch disagreement (data-ambiguity signal, error-aligned) wins;
   cross-arch disagreement (inductive-bias-difference signal) deflates — and
   cross-arch *improves* the pooled mean while eroding the disagreement flag
   (the diversity tradeoff). Regime still matters, but at **broader selectivity**
   (top-50%): from-scratch disagreement beats confidence regardless of diversity
   (weak members disagree more); pretrained does not. So "disagreement doesn't
   help" is scoped to *architecture-diverse ensembles at high selectivity*, not
   "the method fails" — and Baur's D-Ens is same-architecture, the favorable
   type. Per-pathology correctness (Task 3 ~0.85–0.93) and OOD (maha Kermany
   0.818) are regime- and diversity-**robust**; confident-error flagging and the
   kNN-vs-maha density ranking are diversity-** and regime-dependent** (kNN
   helps on frozen RAD-DINO features, adds nothing from-scratch where `hybrid ≡
   epistemic_std`). Pooled confidence wins full-coverage abstention (AUAC) in
   *both* regimes — disagreement's edge is specifically *within* the confident
   set, where confidence is saturated. See Phase 6 + the cross-backbone
   follow-up.
5. **Adapting the pretrained ensemble to the deployment dataset re-emerges the
   std edge at the defining metric (Phase 7, the production-realistic arm).**
   LP-FT of all four pretrained members on ReX-train (112,967 pts, identical
   protocol to the Phase-6 control) moves the top-10% confident-error AUROC
   from std 0.5626 ≈ conf 0.5630 (p=0.99, tied) to **std 0.679 vs conf 0.542
   (DeLong p=0.006**; s1 0.683/0.554 p=0.0045; s2 0.693/0.533 p<0.0001) —
   inside the from-scratch same-arch band (0.636–0.719). The Phase-5 deflation
   was therefore a **data-shift artifact, not an ensemble property**; the
   mechanism question is closed in the favorable direction and seed variance is
   negligible (band 0.679–0.693 across convnextv2 s0/s1/s2). Full-coverage
   abstention still favors confidence (AUAC 0.9653 vs 0.9411; unchanged
   two-regime story), but adaptation lifted std's ranking power everywhere
   (E-AURC 0.0649→0.0414) while confidence improved less (0.019→0.0172).
   Predictive entropy is the strongest tail signal in the adapted arm (0.7553
   @ top-10). Both regimes are live in the demo app
   (`--regime {unadapted,adapted}`).

---

## System

- **Ensemble (3 members, architecture-primary, leak-free for OpenI):**
  - **M1 RAD-DINO** — DINOv2 ViT-B/14 (86.6M), frozen + linear probe (768-d `[CLS]`).
  - **M2 ConvNeXt-V2-L** — ImageNet FCMAE, LP-FT (the only GPU-trained member).
  - **M3 xrv DenseNet121** — `xrv_nih` per-dataset weights (supervised CNN).
  Three distinct representation strategies (SSL ViT / ImageNet CNN / supervised
  CNN); the `all` weights are excluded (OpenI leak). Member count M=3 today; M=5
  is the power-law knee — **M4 Ark+ Swin-Large + M5 BiomedCLIP now built
  (2026-08-03), see Phase-3 M=5 knee below** → 5 representation strategies
  (SSL-ViT / ImageNet-CNN / supervised-CNN / supervised-multi-dataset-Swin / CLIP).
- **Data (leak-aware, patient-level splits):** `data/manifest.parquet` (22,018
  rows) — role A train 16,002 (NIH 10k + CheXpert 6k), role B per-member cal 2,002,
  role C ensemble cal 2,002 OpenI, role D leak-free eval 2,012 OpenI. Canonical
  ontology = NIH-14. OOD probes in `data/ood_manifest.parquet` (12,288 rows):
  Kermany pediatric pneumonia (5,856) + COVID-19 radiography (6,432), both
  ungated + leak-free, both reduced to the NIH Pneumonia class.
- **Production calibration:** **TS-only** — temperature-scale the raw pooled
  mean (fit on OpenI role-C), keep **raw** cross-member probs for the
  disagreement signal. Per-member Beta was **rejected** (it removes the
  inter-member scale-family disagreement that carries error-discriminating
  signal; §E.2). TS-only fixes ECE ~10× while preserving the top-end
  disagreement ordering.

---

## Research arc & key numbers

### Part 1 — Datasets (DONE)
Leak-free curation + NIH-14 mapping + validity mask + patient-level role A/B/C/D
splits; 6 leak-free assertions pass. OpenI labeled frontal subset = 4,014
(patient-disjoint 2k cal / 2k eval). *(Correction vs plan: role D = 2,012, not
~5.4k — OpenI's labeled frontal subset is only 4,014.)*

### Phase 0 — Refactor seams (DONE, behavior-preserving)
`MemberOutput[logits/probs/features/valid]`, `Member`/`Alignment`/registries,
batched `forward_ensemble_full`. Legacy xrv double-sigmoid **preserved** (plain
`sigmoid(logits)` regressed the defining selectivity curve). 500-img OpenI
selectivity curve matches the pre-refactor baseline exactly.

### Phase 1 — 2-member {xrv_nih, convnextv2} (DONE)
Found + fixed a **prob-mode scale mismatch**: xrv legacy double-sigmoid pins
probs to [0.5,0.73] while ConvNeXt spans [0,1] → cross-member std flags
confident-*correct* as uncertain (AUROC ≪ 0.5). Fix: arch-registry xrv keys use
`plain` sigmoid. After fix on role-D (2,012 imgs, no-MC): confidence-error AUROC
**0.71**, ECE **0.034**, recall 0.72, AURC 0.004. **But selectivity is
decreasing** (0.84→0.64) — two same-family CNNs correlate on confident errors
(the architecture-diversity thesis predicts this; needs the ViT member). MC-dropout
tested as a rescue and **rejected** (top-5% 0.43 < random — dropout noise doesn't
separate rare confident-wrong).

### Phase 2 — +RAD-DINO, 3-member (DONE)
RAD-DINO frozen-probe macroAUC **0.7986** (vs ConvNeXt 0.7053). 3-member vs 2-member
on role-D: confident_wrong top-10 **18→8 (−56%)**, top-50 errors **1,071→387 (−64%)**,
**AURC 0.00403→0.00279 (−31%)**, top-50 AUROC 0.839→0.889, recall 0.722→0.750. Top-10
confident-error AUROC 0.713→0.639 and top-5 0.639→0.423 (worse) — but top-5% has only
**2 errors** (statistically ill-posed) and the survivors are the irreducible
confident-agree-wrong core. Architecture diversity cut errors ~50–64% and improved
AURC, but the literal top-5% gate is under-powered in-distribution.

### Phase 3 — Calibration ablation (DONE; the calibration-vs-error tension)
4 variants on the identical 3-member forward (2,012 role-D, `runs/phase3/calibration_ablation.json`):

| variant | ECE | AURC | cw | top50 | top25 | top15 | top10 | top5 | T | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.0554 | 0.00279 | 8 | 0.889 | 0.695 | 0.698 | 0.638 | 0.423 | 1.000 | 0/42 |
| **ts_only** | **0.0054** | 0.00284 | 8 | 0.695 | 0.694 | 0.700 | 0.631 | 0.410 | 0.630 | 0/42 |
| beta_only | 0.0104 | 0.00411 | 9 | 0.790 | 0.588 | 0.517 | 0.454 | 0.292 | 1.000 | 33/42 |
| beta+ts | 0.0017 | 0.00415 | 9 | 0.726 | 0.586 | 0.515 | 0.450 | 0.289 | 0.893 | 33/42 |

**Conclusion: per-member Beta is rejected.** Beta (not TS) is the culprit — it
removes the inter-member scale-family disagreement that carries the
error-discriminating signal. **TS-only is the production design**: ECE 10×
better while the top-end disagreement ordering is preserved (top-5 0.423→0.410,
top-10 0.638→0.631, AURC preserved). The weak/non-monotonic top-5 in-distribution
is **not** a calibration problem — it's (a) under-powering (2–4 errors at top-5%)
and (b) the irreducible confident-agree-wrong core. Fixing it needs powered OOD
probes (Phase 4) and/or feature-based UQ (step 6), not more calibration.

### Phase 4 — OOD shift probes (DONE; decisive negative)
TS-only calibration fit on OpenI role-C, evaluated on leak-free OOD (Pneumonia
class only, hundreds of confident errors):

| site | n | prev | cw | AUROC (disagree) | AUROC (MI) | ECE | AURC | monotone |
|---|---|---|---|---|---|---|---|---|
| kermany | 5,856 | 0.730 | 244 | **0.515** | 0.418 | 0.720 | 0.419 | False |
| covid | 6,432 | 0.754 | 313 | **0.478** | 0.402 | 0.743 | 0.533 | False |
| *(ID OpenI ref)* | — | — | — | *0.45* | — | *0.0017* | *0.004* | — |

The disagreement flag is **at/below chance under shift**. The in-distribution
weakness was not just under-powering — with hundreds of confident errors the
flag is genuinely uninformative. MI is worse than std (below chance; Xia &
Bouganis raw-MI-unreliable-for-OOD confirmed).

### Phase 4 step 6 — Feature-based UQ (DONE; decisive positive)
Re-forwarded OpenI-C/D + Kermany + COVID collecting RAD-DINO 14 logits + 768-d
`[CLS]` features. Fit Mahalanobis (2 class-conditional Gaussians, Ledoit-Wolf)
on OpenI-C Pneumonia pos/neg (38 pos / 1,964 neg). TS-only calibration path
intact (ECE/AURC reproduce Phase 4). Confident-error AUROC by score, top-10%
confidence:

| site | n | prev | cw | epistemic_std | MI | energy14 | **mahalanobis** |
|---|---|---|---|---|---|---|---|
| openiD (ID, all classes) | 2,012 | 0.02 | 8 | 0.631 | 0.619 | 0.473 | **0.835** |
| kermany (OOD) | 5,856 | 0.73 | 244 | 0.515 (chance) | 0.418 | 0.253 | **0.818** |
| covid (OOD) | 6,432 | 0.75 | 313 | 0.478 (chance) | 0.402 | 0.344 | **0.649** |

Mahalanobis holds across selectivity: Kermany 0.779→0.818→**0.851** (50→10→5%,
monotone increasing — the defining property); COVID 0.703→0.649→0.601 (all above
chance). **14-class energy fails** (anti-diagnostic, 0.25/0.34 < 0.5): logit-
magnitude from the linear head is not the right OOD axis under shift; the
feature-space density is.

### Phase 5 — Powered leak-free eval (ReXGradient-160K) (DONE 2026-08-18; honest deflation)

The OpenI confident-error numbers above rest on **8 confident errors** in
OpenI-D (top-10% = 5 wrong). ReXGradient-160K (2025, postdates every member's
pretraining → leak-free by recency) supplies a powered, in-distribution,
frontal-only eval: **16,146 images → 34 confident errors at top-10% (6.8×
OpenI)**, with per-dataset in-distribution calibration (Youden/TS/hybrid fit
on ReX-cal, applied to ReX-eval — methodologically what Baur does within
MIMIC). Labels: CheXpert rule-based labeler on ReX free-text reports → NIH-14
via `CHEXPERT_TO_NIH` (10 comparable classes; −1 uncertain preserved for Baur
Task 2). 4-member production ensemble (xrv_nih, convnextv2, raddino, arkswin),
batched H100 inference (~21 min). Full pipeline + results on the Mac at
`runs/baselines/rex/`, `runs/eval_arrays/rex/`, `data/rex_manifest.parquet`,
`runs/rex_labeling/`; **all patient data deleted from the Lightning host** per
DUA non-persistence. Executed on Lightning AI Studios (volc kept off-limits for
other projects).

**The defining metric deflates — and that is the point of the pivot:**

| confident-error AUROC, top-10% | OpenI (5 wrong) | ReX (34 wrong) |
|---|---:|---:|
| epistemic_std (cross-member disagreement) | 0.6072 | 0.5626 |
| mahalanobis (RAD-DINO [CLS] density) | 0.8698 | **0.5288** |
| confidence (pooled p_bar) | — | 0.5630 |

- **epistemic_std ≡ confidence at power**: 0.5626 vs 0.5630 (DeLong p=0.99;
  McNemar on the flags p=0.98 — the two flags disagree on ~0 cases). Cross-
  member disagreement adds nothing over the pooled mean for flagging confident
  errors once the benchmark is powered.
- **The mahalanobis collapse (0.87 → 0.53) is the small-n cautionary tale.**
  OpenI's 0.87 was noise on 5 errors, not a feature-density signal; at 34
  errors it is near chance. **Caveat: the *OOD* mahalanobis result (Kermany
  0.818, COVID 0.649, 244/313 errors) is powered and stands** — what fails at
  power is the *in-distribution confident-error* flag, not OOD/shift detection.
  Mahalanobis is a shift detector; it is not an in-distribution confident-error
  flagger at power.
- **A single member beats the ensemble**: conf_convnextv2 0.6077 >
  epistemic_std 0.5626 (DeLong p=0.0007). The hybrid does not rescue it — every
  hybrid-vs-baseline paired-bootstrap CI at top-10% includes 0; at top-50%
  hybrid_std_knn is significantly *worse* than confidence (CI [−0.0865,
  −0.0439]); on full-coverage E-AURC, hybrid vs confidence CI [0.0428, 0.0464]
  → hybrid significantly worse.

**Where the ensemble + feature UQ still wins (the reframe):**

| Baur task | best ensemble-family score | verdict |
|---|---|---|
| Task 3 — per-pathology correctness (macro over 10) | epistemic_std 0.8391, hybrid_std_knn 0.8372 | **positive** (strong) |
| Task 2 — uncertainty-label prediction (macro, 2509 −1 rows) | entropy 0.6983, hybrid_std_knn 0.6510 | tentative (ReX −1 are labeler-inferred, not expert) |
| Task 4 — AUAC (accuracy-coverage) | confidence 0.9552, epistemic_std 0.9093 | pooled confidence wins, not disagreement |
| Defining — confident-error flagging (top-10%) | conf_convnextv2 0.6077, knn 0.5942 | single-member confidence / kNN; ensemble disagreement ≈ confidence |

*Task-4 protocol (Baur realization):* AUAC = ∫(1−risk)·d(coverage) over the
**certain-label eval records** (ReX 78,311; −1/uncertain rows excluded, used only
by Task 2). Abstain on highest-uncertainty record first (`argsort(-score)`,
scores HIGH = uncertain; `confidence` enters as `-conf`), track accuracy of the
retained tail — Baur's "iteratively remove the most-uncertain, track accuracy"
at **record-level continuous resolution** (all 78,311 points) vs Baur's 5% steps
(same quantity, finer grid). **Abstention unit = per-pathology record**
(image × valid class), not a whole image; if Baur's is per-image/study (MIMIC),
that granularity difference folds into the existing ReX-vs-MIMIC confound — AUAC
*magnitudes* aren't a clean head-to-head, the within-regime result (which score
wins) is.

So the OpenI "ensemble + density wins" conclusion was **right about the wrong
task**: it holds for *per-pathology correctness* (Task 3, ~0.84), not for
*confident-error flagging* (the defining metric), where pooled confidence and a
single good member's confidence win. The density with legs is **kNN**
(hybrid_std_knn 0.598 top-10%, numerically above confidence but **not**
significant at 34 errors), not mahalanobis — matching Woodland 2024 (kNN >
maha on non-Gaussian medical features). **Energy fails on every ReX task**
(AUAC 0.645, Task 2 0.335, confident-error 0.451) — matching Baur + OpenMIBOOD.

**Calibration (not the explanation):** ECE 0.0219 → 0.0207 (TS); already
well-calibrated, and TS is rank-invariant (AUROC(−p_bar) raw = TS = 0.4527) so
it cannot move the confident-error ranking. The deflation is a power effect,
not a calibration artifact. TS T=0.9662.

**Reconciling with Baur (UNSURE@MICCAI 2025, arXiv:2508.04457):** Baur's
D-Ens (Deep Ensemble, 5 independent full trainings) and S-Ens (Shallow
Ensemble, 1 shared backbone + 5 diverse heads) are *ensemble methods*, each
yielding three scores — **PU = H(p̄)** (predictive entropy of the pooled
prediction; = our `entropy` score on `p_bar` — note our `confidence` is max p̄, a
different scalar, *not* PU), **AU** = mean member entropy (aleatoric; we compute
it internally as the second term of MI but don't surface it as a standalone
score), **EU = PU−AU** = mutual information (= our `mutual_info`; our preferred
`epistemic_std` is a different estimator of the same epistemic concept — we
report both). Baur report **only the best of {PU,AU,EU} per method per task**,
so "ensembles win" hides *which score* carried it. On Task 1 (OOD) EU was the
ensemble's strong score; on Tasks 3/4 the winning score is unspecified. Our
result: the **pooled** score (PU = entropy / the confidence family) wins the
flagging/coverage tasks, not the **disagreement** (EU) the project was built
around — i.e. we disentangle what Baur's max-over-scores reporting obscures.
Baur's AUAC band is 0.82–0.84 (ResNet-18/ViT-Tiny, MIMIC, from-scratch); our
0.91–0.96 is a different regime (frozen foundation ensemble) and **not a clean
head-to-head** — different dataset, model class, and training regime. The residual confound vs Baur: architecture-diverse
frozen ensemble vs same-arch trained ensemble (a different object; "ensembles
win" and "disagreement doesn't help flagging" are about different ensemble
kinds and are not contradictory).

**Publishable shape:** an honest-negative-with-reinterpretation paper. Claim:
ensemble disagreement does not improve confident-error flagging over pooled
confidence at power; small-benchmark advantages are artifacts (maha case
study). Contributions: the power demonstration itself; per-pathology
correctness as where ensemble UQ actually helps; reproduction of the
kNN > maha / energy-fails ranking; a deployment-time (frozen, no-retrain) UQ
benchmark — a gap Baur leaves open. **Key pending control (DONE — Phase 6
below)**: a from-scratch ResNet-18 (+ ViT-Tiny, + ConvNeXt-Tiny) 5-seed
ensemble on ReX, to test whether the disagreement-vs-confidence result is
foundation-specific (Baur's from-scratch regime is where disagreement wins) —
this is what gives the negative its meaning. **Phase 6 resolves it, and the
answer is diversity-type-dependent (at the defining top-10% metric), not
foundation-specific**: cross-arch from-scratch *reproduces* the Phase-5
deflation (confident-error AUROC 0.5585 vs pretrained 0.5626, both ≡ confidence),
while same-arch from-scratch *wins* (0.636–0.719 > confidence, 3/3 backbones,
top-50% p=0.0). The driver at the defining metric is architecture diversity, not
training regime — see the Phase 6 cross-backbone follow-up below. (At broader
top-50% selectivity the regime does dominate: from-scratch beats confidence
regardless of diversity; pretrained does not.) Task 2 (uncertainty-label
prediction) stays **tentative on ReX's labeler-inferred −1 labels** — a
dataset with native *expert* −1 labels would be the real test, but none is in
scope (CANDID-III was the planned one; it is removed from scope — NZ
government has banned use of that dataset).

---

### Phase 6 — From-scratch D-Ens control (Baur regime) (DONE 2026-08-23; the defining metric is diversity-type-dependent — see the cross-backbone follow-up below)

*Experiment story (the two-role split):* the **deployment system** — Agent 1
(UQ + confident-error flagging) and Agent 2 (similar-case retrieval,
docs/agent2_retrieval.md) — runs on the **pretrained members, individually
and cross-arch** (xrv/convnextv2/raddino/arkswin). The from-scratch
ResNet-18/ViT-Tiny/ConvNeXt-Tiny D-Ens below are **not** a deployment model:
they are the **UQ benchmark control** — members initialized at random and
trained with **no prior exposure to any data** (no pretraining of any kind),
the literature-standard Deep-Ensemble regime (Baur et al.) — whose purpose is
to test whether the Phase-5 deflation is a property of foundation deployment
or of ensembled disagreement generally. Agent 2 mirrors the control with
model-matched retrieval indexes for the same reason.

Phase 5's negative — `epistemic_std` ≡ `confidence` for confident-error flagging
at power — was obtained on a **frozen, architecture-diverse, foundation-pretrained**
ensemble. Baur et al. (UNSURE@MICCAI 2025) find disagreement *does* carry signal in
the **from-scratch, same-architecture** regime (5 independent random-init
trainings = D-Ens). Phase 6 is the missing control: train a 5-seed from-scratch
D-Ens for each of Baur's three anchor backbones — **ResNet-18, ViT-Tiny,
ConvNeXt-Tiny** — on ReXGradient-train (112,967 frontal images, CheXpert-rule
labels, `u_policy="ones"` −1→1 for training; test keeps −1 for Baur Task 2),
early-stopped on ReX-valid, evaluated on ReX-test (8,082) with the **same
splits, labels, and `eval_baselines.py`** as Phase 5. seed-0 is the feature
member (`rad_feats`/`energy`/`MahalanobisOOD`) so the full baseline suite runs
with zero edits. 15 models trained on Lightning H100 via a pre-decode float32
memmap cache (the per-epoch 16-bit-PNG decode was CPU-bound at 196 s/epoch with
the GPU idle; the cache drops it to ~22 s/epoch, 9× — see
[[lightning-studio-environment]]); all 15 checkpoints + 6 array sets pulled to
the Mac and **all patient data deleted from the host per DUA**. Arrays:
`runs/eval_arrays/rex_fs_<bb>/arrays_rex_fs_<bb>{cal,eval}.npz` (8064/8082,
5,14); baselines: `runs/baselines/rex_fs_<bb>/`.

**The defining metric (same-architecture): disagreement beats confidence in from-scratch.** Confident-error AUROC, top-10% (Panel B —
within the ensemble's confident set, the flag's population). NB: these are
same-architecture (cross-seed) ensembles; the cross-backbone follow-up below
shows the top-10% flip is diversity-type-dependent, not regime-dependent —
architecture-diverse from-scratch *reproduces* the Phase-5 deflation.

| score | Phase 5 pretrained | resnet18 FS | vit_tiny FS | convnext_tiny FS |
|---|---:|---:|---:|---:|
| epistemic_std (disagreement) | 0.5626 | 0.6360 | 0.7190 | 0.6688 |
| confidence (pooled p_bar) | 0.5630 | 0.5585 | 0.6736 | 0.5385 |
| entropy (PU = H(p_bar)) | 0.5473 | 0.6587 | 0.7610 | 0.7155 |
| mahalanobis | 0.5288 | 0.5266 | 0.5132 | 0.5236 |
| kNN | 0.5942 | 0.5070 | 0.5474 | 0.5229 |
| **Δ(std − conf)** | **−0.0004** | **+0.0775** | **+0.0454** | **+0.1303** |
| DeLong p (std vs conf, top-10%) | 0.99 | 0.0071 | 0.0603 | 0.0 (z=4.20) |
| DeLong p (std vs conf, top-50%) | — | 0.0 | 0.0 | 0.0 |
| confident errors @ top-10% | 34 | 28 | 34 | 34 |

- **In the from-scratch, same-architecture regime, disagreement beats confidence for confident-
  error flagging — all three backbones.** Gap +0.045 to +0.130 AUROC; at top-50%
  all three DeLong p=0.0, at top-10% p=0.0071 / 0.0603 / 0.0. **In the frozen-
  foundation regime the same test gave p=0.99 (≡).** This reproduces Baur's
  finding on ReX with our pipeline. **But the cross-backbone follow-up below
  qualifies the attribution**: at the defining top-10% metric the driver is
  diversity-*type* (architecture-diverse from-scratch *also* deflates, 0.5585 ≡
  confidence), not regime; regime dominates only at broader top-50% selectivity.
  The negative was "disagreement does not help in *architecture-diverse
  deployment*," not "disagreement does not help."
- **The mechanism is member weakness *plus* same-architecture disagreement.**
  Panel A (each member's *own* confidence vs its *own* errors) is at/below chance
  for all three from-scratch backbones (best-member top-10% AUROC 0.47 / 0.43 /
  0.52 — i.e. a single from-scratch member cannot tell its own errors from its
  correct calls). D-Ens value comes from **cross-member** disagreement, not
  single-member confidence: the five weak members disagree precisely where they
  err. The frozen-foundation members are strong *individuals* that *agree* →
  disagreement is a rescaled confidence and adds nothing. This is the "functions
  disagree where they err" regime (Baur) vs the "strong models agree" regime
  (Phase 5). **But member weakness alone is incomplete** — the cross-backbone
  follow-up shows cross-arch from-scratch members are equally weak yet *deflate*
  at top-10%; the top-10% signal needs *same-architecture* (cross-seed)
  disagreement, which tracks data-ambiguity, not just weakness.
- **Predictive entropy (PU) is the best single score in from-scratch**
  (0.66 / 0.76 / 0.72 top-10%) — above both epistemic_std and confidence, because
  PU = H(p_bar) folds in *both* the pooled confidence and the disagreement. In
  Phase 5 pretrained, entropy (0.547) was *below* confidence — its ranking
  flips with the regime, tracking where the signal lives.

**Cross-regime invariants (robust, both regimes):**

- **Mahalanobis is near-chance in-distribution in *both* regimes**
  (0.53 / 0.51 / 0.52 FS vs 0.53 pretrained). It is a distribution-shift
  detector, not an in-distribution confident-error flagger — the Phase-5
  conclusion holds universally, not just for foundation features. (The *OOD*
  mahalanobis result, Kermany 0.818 / COVID 0.649, is a different question and
  stands.)
- **Pooled confidence wins full-coverage selective abstention in *both*
  regimes.** AUAC: confidence 0.955 / 0.938 / 0.943 FS vs epistemic_std 0.923 /
  0.903 / 0.899 (and Phase-5 confidence 0.955 vs std 0.909). E-AURC: confidence
  lowest (0.020 / 0.028 / 0.025). The hybrid-vs-confidence E-AURC CI excludes 0
  on the *worse* side for all three ([0.030,0.033] / [0.034,0.037] /
  [0.042,0.044]) — hybrid is significantly worse at full coverage. So
  disagreement's edge is **specifically *within* the confident set**: confidence
  is the right signal for *whether to abstain* (monotone in error probability),
  but once you have committed to a confident prediction, confidence is
  saturated and cannot discriminate among confident predictions — and that is
  where disagreement (in from-scratch) takes over.
- **Density is regime-dependent.** kNN is the best density in foundation
  (0.594, above confidence) on frozen RAD-DINO `[CLS]` features, but **adds
  nothing in from-scratch**: the hybrid selector picks `weight_on_std=1.0` for
  all three backbones, so `hybrid_std_knn ≡ epistemic_std`. From-scratch seed-0
  features are a weaker density than frozen foundation features. maha/kNN/energy
  are all near or below chance in from-scratch. Energy fails everywhere (matches
  Baur + OpenMIBOOD, both regimes).

**Baur Task 2/3/4 (from-scratch, ReX-test):**

| task | resnet18 | vit_tiny | convnext_tiny | verdict |
|---|---|---|---|---|
| Task 3 — per-pathology correctness (macro, 10) | std 0.893, entropy 0.933, conf 0.915 | std 0.863, entropy 0.925, conf 0.905 | std 0.882, entropy 0.930, conf 0.908 | ensemble scores strong (~0.88–0.93) — regime-robust |
| Task 2 — uncertainty-label (2509 −1 rows) | std 0.708, entropy 0.744, conf 0.687 | std 0.686, entropy 0.720, conf 0.670 | std 0.695, entropy 0.730, conf 0.675 | disagreement > confidence (predicts −1) — tentative (ReX −1 are labeler-inferred) |
| Task 4 — AUAC | conf 0.955, entropy 0.930, std 0.923 | conf 0.938, entropy 0.916, std 0.903 | conf 0.943, entropy 0.911, std 0.899 | pooled confidence wins abstention (both regimes) |
| Defining — confident-error flag (top-10%) | std 0.636 > conf 0.559 | std 0.719 > conf 0.674 | std 0.669 > conf 0.539 | **disagreement > confidence (from-scratch only)** |

Calibration (not the explanation): TS T = 0.86 / 0.83 / 0.85, ECE 0.023 →
0.012 / 0.025 → 0.009 / 0.025 → 0.013; rank-invariant (AUROC(−p_bar) raw = TS
for all three), so TS cannot have moved the confident-error ranking — the
std>conf result is a regime effect, not a calibration artifact.

*Why Task 4 (AUAC) and the defining metric disagree — two regions, one
squeeze.* The table shows confidence winning the full-set AUAC (0.955 vs std
0.923) while disagreement wins the top-10% confident-error AUROC (0.636 vs
conf ≈0.48 = chance). Not a contradiction — different sampling frames:
AUAC's area is dominated by the bulk of 78k records where pooled confidence
(a first-order average of 5 members) is simply the better error estimator,
while std (second-order from only 5 samples) is noisier. The confident-error
metric lives in the top-10% sliver, where the pooled mean has *saturated* —
the same from-scratch low-dynamic-range squeeze seen directly in the Agent-2
retrieval feature spaces (random-pair cosine 0.66–0.71, NN 0.97 = only ~1.7σ
evidence vs RAD-DINO's ~7.7σ) — so within the band confidence ranks ~chance
while disagreement (still varying, still error-correlated) holds the
remaining information. Usage rule: full-coverage abstention → pooled
confidence; triage of confidently-wrong within the confident stratum →
disagreement. Baur-compatible: their best-of-per-task reporting spans the
same two regions. (Full derivation: docs/uncertainty_evaluation.md §4.6,
"Two regions, two metrics".)

**Reconciliation with Baur — and what this settles.** Baur's D-Ens is exactly
this object (5 independent full trainings of one backbone); we now match it on
ReX for ResNet-18, ViT-Tiny, and ConvNeXt-Tiny. Baur report only the **best of
{PU, AU, EU} per method per task**, so "ensembles win" hides *which score*
carried it; we disentangle: in the from-scratch regime the **disagreement
(EU)** carries confident-error flagging (above confidence/PU), while **PU
(= entropy / the confidence family)** carries selective abstention (AUAC) in
both regimes — the same
split Baur's max-over-scores reporting obscures. Our from-scratch D-Ens AUAC
(0.90–0.95) is higher than Baur's 0.82–0.84 (ResNet-18/ViT-Tiny, MIMIC), but
that is a dataset/label confound (ReX CheXpert-rule labels vs MIMIC expert
labels, different base error rate), not a regime statement — the contribution
is the **within-regime structural comparison** (disagreement vs confidence),
which is clean on ReX. The Phase-5 residual confound (architecture-diverse
frozen ensemble vs same-arch trained ensemble — "different objects") is now
resolved by the cross-backbone follow-up below: the *same* disagreement score is
signal for same-architecture (cross-seed) and null for architecture-diverse,
*regardless of regime* at the defining top-10% metric — so the two findings are
not contradictory but reflect two diversity types (data-ambiguity vs
inductive-bias-difference) at the defining metric, and two regimes
(from-scratch vs foundation) at broader selectivity.

**Updated publishable claim (supersedes Phase 5's "honest negative"; itself
qualified by the cross-backbone follow-up below for the defining metric).** The
value of ensemble disagreement for confident-error flagging is
**regime-dependent at broad selectivity and diversity-type-dependent at the
defining top-10% metric**: in the from-scratch, same-architecture regime
(Baur's; weak diverse members that disagree where they err) it beats pooled
confidence (3/3 backbones, +0.045 to +0.130 AUROC, top-50% p=0.0); in the
frozen-foundation deployment regime it is ≡ confidence (p=0.99) and pooled
confidence suffices. Per-pathology correctness (Task 3,
~0.88–0.93) and OOD detection (mahalanobis, Kermany 0.818) are regime-**robust**;
confident-error flagging and the kNN-vs-maha density ranking are regime-
**dependent**. The Phase-5 negative is thereby scoped to "architecture-diverse
foundation deployment," not generalized to "the method fails" — the from-scratch
control is what gives the negative its meaning (as Phase 5 itself flagged).

**Caveats.** (1) ReX −1 are labeler-inferred, not expert — Task 2 stays
tentative; an expert-−1 dataset would be the real test, but none is in scope
(CANDID-III, the planned one, is removed from scope — NZ government has banned
its use). (2) From-scratch models
are trained Baur-protocol-cheap (224 px, ~25 epochs, early-stop, no tuning to
SOTA) — intentionally comparable, not strongest. (3) The ReX-vs-MIMIC dataset
confound means AUAC magnitudes are not a clean head-to-head; the structural
disagreement-vs-confidence comparison is. (4) 5-seed single-backbone D-Ens is
Baur's exact object — now matched; S-Ens (shared backbone + diverse heads) is
not run here.

#### Phase 6 follow-up — cross-backbone from-scratch (the regime×diversity resolution, 2026-08-23)

The Phase-5-vs-Phase-6 comparison above confounds **two axes**: training regime
(pretrained vs from-scratch) *and* diversity type (architecture-diverse vs
same-architecture). Phase 5 = arch-diverse + pretrained (deflates); Phase 6 =
same-arch + from-scratch (signal). The missing cell — an **architecture-diverse
from-scratch** ensemble — isolates the two. It is free to build: the three
backbones' per-member probs already live in the per-backbone npz with identical
image ordering (verified), so a cross-arch ensemble is just stacking seed-0
probs across backbones — **no new training, no new inference** (`scripts/build_crossarch_arrays.py`).
Built two ensembles: `rex_fs_crossarch` (M=3: {resnet18_s0, vit_tiny_s0,
convnext_tiny_s0}, arch-diverse from-scratch) and `rex_fs_resnet18_m3` (M=3:
{resnet18 s0,s1,s2}, same-arch from-scratch at matched M=3 to remove the
member-count confound). Baselines: `runs/baselines/rex_fs_crossarch/`,
`runs/baselines/rex_fs_resnet18_m3/`. **M=2 robustness variants** (same script,
free recombine): `rex_fs_crossarch_rv` = {resnet18_s0, vit_tiny_s0} — the
pure-CNN-vs-pure-transformer endpoints (ConvNeXt is a CNN with ViT-inspired
design, no self-attention, so the M=3 crossarch is 2 CNNs + 1 ViT, CNN-heavy on
the inductive-bias axis; this pair is the clean CNN↔TX contrast) — and a matched
`rex_fs_resnet18_m2` = {resnet18 s0,s1} same-arch control so member-count stays
unconfounded at M=2. Baselines: `runs/baselines/rex_fs_crossarch_rv/`,
`runs/baselines/rex_fs_resnet18_m2/`.

**The defining top-10% metric is diversity-type-dependent, not regime-dependent.** Confident-error AUROC, std (disagreement) vs confidence:

| ensemble | diversity | regime | M | std t10 | conf t10 | Δ₁₀ | p₁₀ | std t50 | conf t50 | p₅₀ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 5 | arch-diverse | pretrained | 4 | 0.5626 | 0.5630 | −0.000 | 0.99 | 0.5808 | 0.6775 | 0.0 (std worse) |
| **crossarch** | **arch-diverse** | **from-scratch** | 3 | **0.5585** | 0.6093 | **−0.051** | 0.22 | 0.8048 | 0.7642 | 0.0 (std better) |
| resnet18_m3 | same-arch | from-scratch | 3 | 0.6453 | 0.5455 | +0.100 | 0.012 | 0.7923 | 0.7476 | 0.0 |
| Phase 6 (×3) | same-arch | from-scratch | 5 | 0.636–0.719 | 0.54–0.67 | +0.05–0.13 | ≤0.06 | 0.81–0.85 | 0.74–0.76 | 0.0 |
| **crossarch_rv** | **arch-diverse (CNN↔TX)** | **from-scratch** | 2 | **0.5266** | 0.5348 | **−0.008** | 0.79 | 0.7605 | 0.7494 | 0.043 |
| resnet18_m2 | same-arch | from-scratch | 2 | 0.6159 | 0.4773 | +0.139 | 0.0 | 0.7646 | 0.7444 | 0.0003 |

- **At fixed regime (from-scratch), diversity varied** — the decisive top-10%
  axis: arch-diverse `crossarch` deflates (std 0.5585 **below** confidence, p=0.22);
  same-arch `resnet18_m3` wins (std 0.6453 > conf, p=0.012). The M=3 control
  shows member-count isn't the cause (M=3 same-arch still wins).
- **At fixed diversity (arch-diverse), regime varied** — regime does **not**
  rescue the top-10% deflation: pretrained std 0.5626 ≈ from-scratch `crossarch`
  std 0.5585 (both ~0.56, both ≤ confidence). Switching pretrained→from-scratch
  moved the arch-diverse disagreement by **−0.004**; switching arch-diverse→same-
  arch (at fixed from-scratch) moved it by **+0.08 to +0.16**. → architecture
  diversity, not foundation pretraining, drives the top-10% deflation. **This
  revises the Phase-6 "regime-dependent" attribution above for the defining
  metric.**
- **M=2 robustness (rules out a ConvNeXt/"hybrid" artifact and member-count).**
  The M=3 `crossarch` is 2 CNNs + 1 ViT (ConvNeXt shares ResNet's conv inductive
  bias). The pure-CNN-vs-pure-transformer pair `crossarch_rv` {ResNet-18, ViT-Tiny}
  at M=2 **deflates even harder** at top-10% (std 0.5266 ≤ conf 0.5348, p=0.79)
  than the M=3 crossarch (0.5585, p=0.22) — the cleanest inductive-bias-difference
  contrast deflates most. The matched M=2 same-arch control `resnet18_m2` **wins**
  (std 0.6159 vs conf 0.4773, p=0.0, Δ=+0.139 — larger than the M=3 control's
  +0.100). So at M=2 the diversity-type gap is +0.089 (crossarch_rv→resnet18_m2),
  almost identical to the M=3 gap +0.087 — the top-10% diversity effect is
  ~+0.09 AUROC **independent of M**, and is not an artifact of ConvNeXt or of
  having 2 CNNs. Top-50% replicates the regime effect at M=2 (both from-scratch
  cells std > conf: resnet18_m2 p=0.0003; crossarch_rv p=0.043, weaker — the
  broad-selectivity regime effect shrinks with fewer members but keeps its sign).

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
metric. Our Phase 5 is **architecture-diverse** — the type where it deflates.
So the Phase-5 vs Baur difference is a diversity-type difference (at top-10%),
not (primarily) a foundation-pretraining difference; Baur's protocol is the
favorable one. The regime effect is real but lives at broader selectivity
(top-50%), not at the defining top-10% flag.

**Corrected publishable claim (supersedes the Phase-6 "regime-dependent"
claim for the defining metric).** Ensemble disagreement's value for confident-
error flagging is **diversity-type-dependent at high selectivity** (the defining
top-10% metric: architecture-diverse deflates — cross-arch from-scratch
reproduces the Phase-5 pretrained deflation, 0.5585 vs 0.5626, despite the
opposite regime; same-arch from-scratch wins, 0.636–0.719) **and regime-
dependent at broader selectivity** (top-50%: from-scratch beats confidence
regardless of diversity; pretrained does not). The Phase-5 deflation at the
defining metric is primarily an **architecture-diversity** effect, not a
foundation-pretraining effect. Same-arch disagreement is a data-ambiguity signal
(error-aligned); cross-arch disagreement is an inductive-bias-difference signal
(error-aligned only at broad selectivity). Per-pathology correctness (Task 3,
~0.85–0.93) and OOD detection (mahalanobis, Kermany 0.818) remain regime- and
diversity-**robust**; confident-error flagging and the kNN-vs-maha density
ranking are diversity-** and regime-dependent**. The remaining untested cell is
same-arch *pretrained* (would need multiple frozen pretrained inits of one
backbone, which we lack); the two straddled axes both point to diversity-type
at top-10 and regime at top-50.

### Phase 7 — ReX-adapted pretrained arm (DONE 2026-08-31; the production-realistic arm — adaptation re-emerges the std edge)

**Question (pre-registered)**: every pretrained member is OOD on ReX — a hospital
would LP-FT them on ReX-train before trusting any UQ story. Does epistemic std
re-emerge above confidence in the confident stratum once the ensemble is
adapted (fs control predicts yes: 0.636–0.719), or stay tied (un-adapted
pretrained: 0.5626 vs 0.5630, DeLong p=0.99)?

**Setup — identical protocol to Phase 6 (fs control)**: patient-disjoint
ReX-train 112,967 / cal 8,064 / eval 8,082 (`data/rex_manifest_fs.parquet`,
no new labeling); masked BCE on NIH-14 with valid mask, u_policy="ones",
hflip+rot10 augment, early-stop on cal certain-label macro-AUROC; fp16 224px
caches; H100 (Lightning) ~6.5 GPU-h total. Members adapted per Route A:
convnextv2 LP-FT ×3 seeds, xrv_nih LP-FT, raddino LP + 1 short low-LR FT,
arkswin linear-probe-only @768 (frozen encoder). Identical eval battery
(`eval_baselines`) as the un-adapted regime.

**Per-member adaptation deltas (cal certain-label macro-AUROC; "pre-LP" = first
val on ReX-cal — a true zero-shot for xrv_nih, a random-head baseline for the
probe members whose heads were initialized empty)**:

| member | pre-LP (ReX-cal) | adapted | gain |
|---|---|---|---|
| convnextv2 s0/s1/s2 | ~0.51–0.53 (random head) | 0.8692 / 0.8696 / 0.8688 | +~0.35 |
| xrv_nih | 0.7290 (NIH head zero-shot) | 0.7996 | +0.071 |
| raddino (frozen DINOv2 enc) | 0.5109 (random head) | 0.8854 | +0.375 (LP alone 0.8668) |
| arkswin LP @768 (frozen enc) | 0.5344 (random head) | 0.8871 | +0.353 |

raddino's **linear probe alone reaches 0.8668** — its self-supervised radiology
features transfer to ReX almost fully without touching the encoder. Seed
replication: final convnextv2 band 0.8688–0.8696 (spread 0.0008) — one seed
suffices for this protocol.

**Headline (pre-registered), tail = top-10% confident (7,832 rows, 23 wrong)**:

| regime | tail std AUROC | tail conf AUROC | DeLong p |
|---|---|---|---|
| un-adapted pretrained | 0.5626 | 0.5630 | 0.99 |
| from-scratch control | 0.636–0.719 | 0.559–0.674 | — |
| **adapted s0** | **0.6789** | **0.5417** | **0.006** |
| adapted s1 / s2 | 0.6832 / 0.6927 | 0.5540 / 0.5331 | 0.0045 / <0.0001 |

The hypothesis is **confirmed**: in-domain adaptation re-emerges the epistemic-std
edge, and it lands **inside the fs-control band** (0.679–0.693 across seeds) while
confidence itself *drops* in the tail (0.554–0.533). The mechanism claim from
Phase 5/6 survives its last competing explanation — the Phase-5 deflation was a
data-shift artifact, not an ensemble property: remove the OOD mismatch and
cross-arch disagreement becomes error-aligned exactly as the from-scratch
control predicted. Selectivity ladder (std): 0.7917 @ top-50 → 0.6789 @ top-10.
Within the tail, epistemic std beats every per-member confidence (best 0.558,
DeLong p=0.011 vs xrv) and mahalanobis (0.546, p≈0.004).

**Full coverage (AUAC/E-AURC, adapted s0)**: confidence still wins the full set
(0.9653 vs std 0.9411; DeLong full p≈0 against std) — the two-regime story is
unchanged. But the margin narrows vs the un-adapted run (there 0.9552 vs
0.9093): adaptation improved std's ranking power everywhere (E-AURC 0.0649→0.0414)
while improving confidence less (0.019→0.0172). Best single full-set score is
now per-member raddino confidence 0.9565. Predictive **entropy** is the best
tail signal in the adapted arm (0.7553 @ top-10) — predictive uncertainty and
epistemic disagreement carry complementary tail information; the deployed hybrid
(std-based) keeps the calibrated-confidence gate unchanged.

**Agent-2 space**: adaptation contracts the Rad-DINO space toward the case
library (eval→cal top-1 cosine median 0.520→0.614; top-8 mean 0.472→0.568) —
adapted queries sit closer to their nearest ReX-cal references, consistent with
the calibrated-space reading of "in-domain"; retrieval-side recall@k=1.0 and the
production index carry over unchanged (`runs/rex_phase7/retrieval/rex_adapted_raddino`).

**Deployment story now has both lenses**: `demo_app_rex.py --regime adapted
(default and deployed: s0 headline, live-forward rides the adapted ckpts via
ckpt-overlay, port 7862) | unadapted (comparison lens: pass --regime unadapted,
port 7861)`.
TS fits agree exactly across app and battery (T=0.7843; ECE 0.0246→0.0034).
pytest: 53 passed, 1 pre-existing env-gated failure (biomedclip).

**Agent-2 full deployment library (2026-08-31)**: retrieval library expanded from
the 8,064-row cal split to the full 129,113-case train+cal+eval ReXGradient set —
`runs/rex_phase7/retrieval/rex_adapted_raddino_full` (index.bin 829 MB, HNSW
cosine M=32 efC=400; **recall@10 = 1.0000** over 200 eval queries; cal+eval
vectors are the exact npz `rad_feats`, train rows embedded from the fp16 224-px
xrv-format cache through the adapted raddino encoder — parity gate median cos
**0.9983**, p5 0.9925 vs the npz route). Sidecar gains `split`,
`gt_normal`, `pred_normal` columns. Consequences:
- **Pools de-starved**: pathology-mode candidates go from 0 for 10 of 14
  pathologies (starved pools for 6 more) to e.g. Infiltration 10,009,
  Cardiomegaly 3,762, Atelectasis 1,465, Pneumothorax 66 over the full set.
- **`correct` mode excludes train rows by design** (pool = **5,540** cal+eval
  correct&certain): the adapted members were trained on train labels, so
  train-split "correct" flags are memorization-contaminated — caveat surfaced
  in-app. Train-row annotations are **3-member** (xrv_nih+convnextv2+raddino,
  all 224-px natives); arkswin cannot run without 768-px PNGs — disclosed in
  the retrieval caption.
- **Normal-aware retrieval** (accuracy for no-pathology queries): new `normal`
  mode (candidate pool = GT-normal & certain, 66,306 rows) + pathology-mode
  fallback when the query itself reads as no-finding under the Youden rule
  (48,322 `pred_normal` rows; 2/3 of eval GT-normal queries read this way).
  Full-scale acceptance gate over 2,000 eval GT-normal queries,
  **normal-precision@8**: old behavior on the cal library **0.7396**; old
  behavior on the full library **0.6740** (naive library expansion would have
  *hurt* normal queries — the argmax bucket floods with pathology-dominant
  neighbors); new normal lane **1.0000**. (normal_precision.json in the dist.)
- **Label-agreement rerank (pathology + plain modes)**: second-stage re-scoring
  of the visual top-32 — score = 0.7·cosine + 0.3·agreement(posterior,
  candidate report-labels; −1 excluded). Query side = model posterior only
  (queries carry no GT at retrieval time); report labels are reference-side
  metadata (`gt_labels`/`gt_valid` sidecar columns, local join from manifest +
  npz). Gate over 500 confident eval queries: label-Jaccard@8 0.355→**0.411**
  (pathology) and 0.301→**0.392** (plain) at a ~0.02 cosine cost; **off** in
  correct/normal modes and fallback lanes (their pools are exact predicates).
  (rerank_gate.json in the dist; `docs/agent2_retrieval.md` §2.5.)
- **Flag-keyed retrieval lane (2026-09-01, `flag` mode)** — the Agent-1→Agent-2
  coupling made explicit. Motivation: on the demo's confident-ERROR case Region B
  flagged **Cardiomegaly** (the confident false negative) but pathology mode keyed
  on the spurious Hernia argmax (p=0.394) and returned wrong-Hernia neighbors.
  New lane: key = the Region-B-flagged class with the highest calibrated
  confidence (model internals only — no query GT at deployment); candidates =
  library rows whose report confirms that class (`gt_labels[:,X]==1`, −1
  excluded), cosine + agreement rerank; graceful fallback to pathology semantics
  when Region B doesn't fire. Gate over 2,000 Region-B-firing eval queries
  (flag-class report-precision@8): argmax bucket **0.0142** → flag-keyed
  **1.0000** (definitional; pool ~14.5k, no starvation) at cosine 0.63→0.49 —
  the expected cost of keying on a flagged pathology over raw visual proximity.
  Demo neighborhood flips to report-confirmed-Cardiomegaly references
  (`report_confirms` column in the UI). CLI `--mode flag --flag-class <p>`;
  `docs/agent2_retrieval.md` §2.6.
- **Auto mode + contrast lane + neighbor-vote null (2026-09-01)** — three
  Agent-2 story upgrades. (1) App radio defaults to **`auto`**: resolves to the
  flag lane when Region B fires, `correct` otherwise, with the resolution shown
  in the caption (`mode **auto -> flag**`) — the Agent-1→Agent-2 coupling now
  runs with no user action. (2) **`contrast` lane** ("the model DID call it
  there"): candidates = library rows where the model itself reads X-positive
  (pooled mean ≥ Youden, cal+eval arrays) AND the report confirms X; X = the
  flagged class when Region B fires, else the argmax. Gate (same 2,000 firing
  queries): flag-prec@8 **1.0000** (definitional both ways — model-detect@8
  1.0000 vs 0.0018 status quo / 0.019 flag lane), cosine@8 0.33, pool ~1.5k.
  Flag+contrast give the reviewer both sides of a Region-B flag. (3)
  **Neighbor-vote score = clean null**: vote@8 (share of top-8 unfiltered
  neighbors whose report confirms the flagged class) does NOT rank confident
  errors among firing rows — AUROC(1−vote) **0.453** (CI95 0.336–0.527, only
  14 confident-error rows; baselines equally flat: −max-std 0.505, 1−conf
  0.431); direction reversed (wrong films' mean vote 0.062 vs correct 0.012 —
  neighbors' reports confirm the flagged class MORE when the model is wrong).
  Within the already-flagged population, residual std/conf/neighbor-agreement
  variation carries no extra error signal; the vote stays a display element
  (`report_confirms`), not a score. **`pathology` mode retired** (app radio +
  CLI): the weakest lane, whose argmax-bucket failure mode is exactly what the
  flag lane fixes (1.4% flag-precision, 0.2% model-detected neighbors); its
  no-finding fallback lives on in `normal`, and the rerank gate keeps the
  old-pathology arms as historical baselines. Final mode set: **auto** (default;
  flag when Region B fires, correct otherwise), correct, normal, flag,
  contrast, plain. `docs/agent2_retrieval.md` §2.6–2.7.
- Build ran on Lightning H100 (train pass 6.5 min @ 291 img/s + 1.4 min HNSW);
  11.3 GB train cache upload sha256-verified end-to-end; all patient data
  scrubbed post-extraction per DUA and verified empty by repo-wide find-scan.
  Old 8,064-case index kept on disk as the published prior artifact.

**Task-2 caveat carries over**: training u_policy="ones" fixes −1→1 on train
labels; the uncertain-label row remains labeler-inferred and is reported as
`task2_uncertain_label_auroc` (unchanged machinery, adapted members).

**Reproducibility**: `checkpoints/rex_adapted/*.pt` + `.trainlog.json`
(convnextv2_s{0,1,2}, xrv_nih, raddino, arkswin);
`runs/rex_phase7/{eval_arrays,baselines}/rex_adapted_s{0,1,2}`,
`runs/rex_phase7/retrieval/rex_adapted_raddino`, logs in `runs/rex_lightning_logs`.
All Lightning patient data deleted post-extraction per DUA. The injected
checkpoints ride on the stock members via `build_eval_arrays --ckpt-overlay`
(no registry zoo weights touched; leak-free tests untouched).

---

## Conclusions

1. **Production distribution-shift / OOD detector = Mahalanobis on frozen
   RAD-DINO `[CLS]` features**, not cross-member disagreement and not energy. It
   is the only score tested that beats chance under distribution shift (Kermany
   0.818, COVID 0.649 — powered, 244/313 errors). **Note (Phase 5): the earlier
   "also wins in-distribution (0.835)" claim was on 8 confident errors and does
   not survive power** — at ReX's 34 errors in-distribution mahalanobis is near
   chance (0.53). Mahalanobis is a shift detector, not an in-distribution
   confident-error flagger; see conclusion 6.
2. **Calibration = TS-only** on the pooled mean (per-member Beta rejected). This
   fixes ECE in-distribution but does **not** transfer under shift (prevalence
   shift) — so calibrated probabilities under shift are unreliable; the
   Mahalanobis flag is what carries the safety guarantee there.
3. **Architecture diversity earned its keep** — adding the ViT cut
   confident errors ~50–64% and improved AURC in-distribution — but
   **disagreement-based UQ has a hard ceiling** under shift (the
   confident-agree-wrong core). Feature-based UQ on the frozen ViT member is the
   way past that ceiling.
4. **Conformal gives the in-distribution coverage guarantee; Mahalanobis gives
   the under-shift error detection — they are complementary, not redundant
   (step 7).** Mondrian per-label LAC on the TS-only calibrated pooled mean
   delivers per-pathology coverage within ±2% of 1−α in-distribution (OpenI-D
   0.884–0.913 at α=0.1) — the distribution-free guarantee the heuristic
   Mahalanobis flag cannot offer. But under shift the guarantee **breaks**
   (Kermany 0.27, COVID 0.25) and the LAC set stays size-1 (auto-read) with
   refer-rate ~0.03, so conformal confidently auto-reads *wrong* predictions
   without referring — the conformal-refer flag recall is 0.0 under shift where
   Mahalanobis recall is 0.79/0.65. So: **conformal = in-distribution coverage
   layer; Mahalanobis = under-shift safety net.** A defensible system carries
   both.
5. **Honest framing (plan §H.4):** regulators (NICE/FDA/MHRA) evaluate
   discrimination, not uncertainty. The contribution is a **safety + workflow**
   feature (fewer confident errors, auditable deferral) for an
   otherwise-discrimination-evaluated product, aligned with the abstention-based
   triage direction RCR/NICE/FDA are leaning toward (LungIMPACT RCT showed naive
   worklist prioritisation fails).
6. **At power, confident-error flagging is dominated by pooled confidence, not
   ensemble disagreement or feature density (Phase 5).** On ReXGradient-160K
   (leak-free, 34 confident errors vs OpenI's 5): `epistemic_std` ≡ `confidence`
   (DeLong p=0.99; McNemar p=0.98); mahalanobis 0.87→0.53 (the OpenI
   in-distribution win was an artifact of 8 errors); a single member
   (convnextv2) beats the ensemble (p=0.0007). The ensemble's distinct value is
   **per-pathology correctness (Task 3, ~0.84)**, not the defining flag. kNN is
   the density with legs (not maha); energy fails everywhere — both reproduce
   the literature. The honest publishable claim: ensemble disagreement does not
   improve confident-error flagging over pooled confidence at power **in the
   architecture-diverse foundation-deployment regime**; the value of disagreement
   is **diversity-type-dependent at the defining top-10% metric** (architecture-
   diverse deflates regardless of regime — cross-arch from-scratch reproduces the
   Phase-5 pretrained deflation, 0.5585 vs 0.5626; same-arch from-scratch wins,
   0.636–0.719) **and regime-dependent at broader top-50% selectivity**
   (from-scratch beats confidence regardless of diversity; pretrained does not) —
   **settled by Phase 6 + the cross-backbone follow-up**: "disagreement doesn't
   help" is scoped to architecture-diverse deployment, not "the method fails."
   Pooled confidence wins full-coverage abstention (AUAC) in *both* regimes.

---

## Reproducibility — what's saved

- **Run artifacts:** `runs/phase{1,2,3}/evaluation*.json`,
  `runs/phase3/calibration_ablation.json` + `arrays.npz`,
  `runs/phase4*/phase4_summary.json` + `arrays_<site>.npz`,
  `runs/phase4_features/phase4_features_summary.json` + `arrays_<site>.npz`
  (with `rad_logits`/`rad_feats`/`energy`/`mahalanobis`).
- **Checkpoints:** `checkpoints/convnextv2_lpft.pt`, `checkpoints/raddino_probe.pt`.
- **Manifests:** `data/manifest.parquet`, `data/ood_manifest.parquet`.
- **Scripts:** `scripts/eval_phase{1,2,3,4,4_features}.py`,
  `scripts/analyze_calibration.py`, `scripts/download_ood.py`,
  `scripts/train_convnextv2_lpft.py`, `scripts/eval_production.py`,
  `scripts/eval_conformal.py`, `scripts/build_features_sidecar.py` (§M.1).
- **Run with:** `env PYTHONPATH=. python scripts/eval_phase4_features.py
  --arch-ensemble xrv_nih,convnextv2,raddino --out runs/phase4_features`.
- **Conformal (step 7):** `runs/conformal/{conformal_summary.json,
  evaluation_<site>.json, conformal_sidecar.json}`; `env PYTHONPATH=. python
  scripts/eval_conformal.py --arrays-dir runs/phase4_features --out
  runs/conformal --alpha 0.1 --write-sidecar`.
- **Features sidecar (§M.1):** `runs/phase4_features/features_raddino.npz`
  (`rad_feats (16302,1536)` + `ids`); `env PYTHONPATH=. python
  scripts/build_features_sidecar.py`. CLI Mahalanobis flag: `env PYTHONPATH=.
  python -m cxr_uncertainty.reanalyze runs/production/per_record.csv
  --calibrator ts_only_mahalanobis --features
  runs/phase4_features/features_raddino.npz --out runs/production_reanalyze`
- **Agent 2 — similar-case retrieval (separate doc):** design, SOTA basis
  (HNSW / inline-filtered search / foundation-model embeddings), usage, and
  validation live in `docs/agent2_retrieval.md`. Deployment index is the
  **full 129,113-case library** `runs/rex_phase7/retrieval/rex_adapted_raddino_full`
  (correct-mode = cal+eval 5,540; normal lane + no-finding fallback;
  normal-precision@8 0.6740→1.0000; see Phase 7 section above); the original
  cal-library index `runs/rex_phase7/retrieval/rex_adapted_raddino` (8,064,
  recall@10 = 1.0000 vs exact, 2026-08-29) is kept as the published prior
  artifact.
  (writes per_record.csv via `scripts/eval_production.py --write-records`).
- **ReXGradient powered eval (Phase 5):** `runs/baselines/rex/`
  (`baseline_comparison.json` + `baseline_table.md` +
  `baselines_auroc_by_selectivity.png`), `runs/eval_arrays/rex/`
  (`arrays_rexcal.npz` (8064,4,14) + `arrays_rexeval.npz` (8082,4,14) +
  `build_meta.json`), `data/rex_manifest.parquet` (16,146 frontal rows),
  `runs/rex_labeling/` (`rex_labels.parquet` 20k study-labels w/ −1,
  `rex_index_{valid,test}.parquet`, `label_run_meta.json`), `runs/build_rex.log`,
  `runs/eval_baselines_rex.log`. Scripts: `scripts/build_eval_arrays.py`
  (`--batch-size 8`, OOM→batch-1 fallback), `scripts/build_rex_manifest.py`,
  `scripts/label_rexgradient.py`, `scripts/eval_baselines.py`
  (`--datasets name:cal_npz:eval_npz`). Re-run baselines without re-inference:
  `env PYTHONPATH=. python scripts/eval_baselines.py --datasets
  rex:runs/eval_arrays/rex/arrays_rexcal.npz:runs/eval_arrays/rex/arrays_rexeval.npz
  --out runs/baselines`. Build arrays: `env PYTHONPATH=. python
  scripts/build_eval_arrays.py --manifest data/rex_manifest.parquet --source rex
  --out runs/eval_arrays --batch-size 8`. **Executed on Lightning AI Studios
  (H100); all patient data deleted from the host per DUA non-persistence.**
- **From-scratch D-Ens control (Phase 6):** `runs/baselines/rex_fs_<bb>/`
  (`baseline_comparison.json` + `baseline_table.md` +
  `baselines_auroc_by_selectivity.png` for `<bb>` in {resnet18, vit_tiny,
  convnext_tiny}), `runs/eval_arrays/rex_fs_<bb>/` (`arrays_rex_fs_<bb>cal.npz`
  (8064,5,14) + `arrays_rex_fs_<bb>eval.npz` (8082,5,14) + `build_meta.json`),
  `checkpoints/fs_<bb>_s<seed>.pt` (15, 851 MB — pulled to Mac; patient-data-
  derived, deleted from host), `data/rex_manifest_fs.parquet` (112,967 train +
  8,064 cal + 8,082 eval), `runs/rex_labeling_train_q/rex_labels.parquet`
  (train labels), `runs/fs_phase6/` (`train_fs_all.log`, `build_arrays.log`,
  `cache_build.log`). Scripts: `scripts/train_fromscratch.py` (AMP, bs256,
  224 px, early-stop; `--cache-train/--cache-val` memmap reader),
  `scripts/build_train_cache.py` (pre-decode float32 memmap; the CPU-decode-bound
  fix, 196 s→22 s/epoch), `scripts/build_fromscratch_arrays.py` (5-member
  arrays, seed-0 = feature member), `scripts/train_fs_all.sh` (15-model driver,
  resumable). Re-run baselines without re-inference:
  `env PYTHONPATH=. python scripts/eval_baselines.py --out runs/baselines
  --datasets rex_fs_resnet18:runs/eval_arrays/rex_fs_resnet18/arrays_rex_fs_resnet18cal.npz:runs/eval_arrays/rex_fs_resnet18/arrays_rex_fs_resnet18eval.npz
  rex_fs_vit_tiny:runs/eval_arrays/rex_fs_vit_tiny/arrays_rex_fs_vit_tinycal.npz:runs/eval_arrays/rex_fs_vit_tiny/arrays_rex_fs_vit_tinyeval.npz
  rex_fs_convnext_tiny:runs/eval_arrays/rex_fs_convnext_tiny/arrays_rex_fs_convnext_tinycal.npz:runs/eval_arrays/rex_fs_convnext_tiny/arrays_rex_fs_convnext_tinyeval.npz
  --conf-pct 10`. **Executed on Lightning AI Studios (H100); all patient data
  (PNGs, cache, train manifest, labels, checkpoints) pulled to the Mac and
  deleted from the host per DUA non-persistence.**

---

## Next steps (per plan)

1. ~~Register a **`ts_only + mahalanobis`** production UQ variant (TS on pooled
   mean for calibrated probabilities + Mahalanobis `[CLS]` distance as the
   confident-error flag), replacing the disagreement-based flag.~~ **DONE
   (2026-08-02, plan §K).** Registered as two opt-in calibrators
   (`ts_only`, `ts_only_mahalanobis`) in `cxr_uncertainty/production.py`;
   `youden` stays the default (backward-compatible). Mahalanobis confident-error
   AUROC is a first-class metric (`EvalReport.confident_error_auroc_mahalanobis`
   + `auroc_mahalanobis` in the selectivity curve). Driver
   `scripts/eval_production.py` runs it from the saved feature arrays (no
   inference) and reproduces step 6 exactly: Kermany **0.818**, COVID **0.649**,
   OpenI-D **0.835** vs disagreement 0.515/0.478/0.631 (re-fit Mahalanobis matches
   the saved scores to 0.0). CLI path now runs the real Mahalanobis flag too
   (plan §M.1, 2026-08-02): `reanalyze --calibrator ts_only_mahalanobis --features
   runs/phase4_features/features_raddino.npz` subsets a single RAD-DINO features
   sidecar (built by `scripts/build_features_sidecar.py` from the saved arrays,
   16,302 ids) to the cal/eval sorted-unique `image_id` order and joins by id →
   `uq_score=="mahalanobis"`, no `uq_warning`, `maha_fit` on Pneumonia. This also
   fixed a latent sort-order bug: the CLI path's `_long_df_to_arrays` returns
   pathologies sorted alphabetically, so the old positional `fit_label_idx=6`
   indexed Fibrosis, not Pneumonia — name-based `fit_label="Pneumonia"` resolution
   is now used.
2. ~~**Step 7** — Mondrian per-label LAC conformal for distribution-free coverage
   guarantees (ConformalTriageRiskPolicy).~~ **DONE (2026-08-02, plan §L).**
   Mondrian per-label **LAC** (Least Ambiguous set-valued Classifier) conformal
   on the TS-only calibrated pooled mean, fit on OpenI role-C (one `τ_p` per
   pathology; pooled-rare fallback for `n_p<300`, none triggered — n=2002/pathology).
   Registered as `"conformal_triage"` (opt-in via `--calibrator`; `youden` stays
   default). First-class metrics: `EvalReport.conformal_coverage` /
   `conformal_size` / `per_pathology_conformal` / `brier`. Driver
   `scripts/eval_conformal.py` runs it from the saved arrays (no inference).
   **Result — the in-distribution guarantee holds, and it breaks under shift
   exactly as predicted:** OpenI-D per-pathology coverage **0.884–0.913** (all
   13 valid pathologies within ±2% of the 0.90 target; overall 0.896) — the
   distribution-free Mondrian marginal guarantee. Under shift it **collapses**
   (Kermany **0.269**, COVID **0.245**) — in-distribution calibration does not
   transfer (paralleling Phase 4), AND the LAC set stays size-1 (auto-read) with
   refer-rate ~0.03, so conformal confidently auto-reads *wrong* predictions
   under shift without referring. **This is the complementarity the plan
   predicted:** conformal is an in-distribution *coverage* mechanism; the
   Mahalanobis flag (Kermany 0.818, COVID 0.649) is the under-shift *error
   detector* — the conformal-refer flag recall is 0.0 under shift (it doesn't
   fire) where Mahalanobis recall is 0.79/0.65. The two layers are complementary,
   not redundant. **Live per-image mirror now wired (plan §M.2, 2026-08-02):**
   `cli --risk-policy conformal_triage --conformal-sidecar
   runs/conformal/conformal_sidecar.json` applies the pre-fit TS temperature
   (T=0.6301) + per-pathology τ_p (13 thresholds, Consolidation excluded) to the
   live `p_bar` and sets the risk flag to the LAC `refer` decision; the live
   per_record.csv carries the `lac_*` columns. The sidecar is written by
   `scripts/eval_conformal.py --write-sidecar` (reuses the exact OpenI-C fit);
   `cli.py` dispatches through `get_risk_policy(cfg.risk_policy)`, so the default
   `"threshold"` path stays byte-identical (`get_risk_policy("threshold") is
   assess_image`).
3. **M4 Ark+ (Swin) + M5 BiomedCLIP (CLIP) — M=5 power-law knee — DONE
   (2026-08-03, plan §C/§F.3/§J.5 Phase 3).** Built the two extra representation
   families behind the shared `Member` interface (opt-in via `--arch-ensemble`;
   `DEFAULT_ARCH_ENSEMBLE` unchanged → baselines byte-identical):
   - **M4 Ark+** `members/ark_swin.py` — user-uploaded
     `checkpoints/Ark6_swinLarge768_ep50.pth.tar` (Swin-Large, 1536-d,
     `native_size=768`, 195.2M params; cyclic supervised multi-dataset CXR
     pretrain CheXpert/CXR14/RSNA/VinDr/Shenzhen/MIMIC — NOT OpenI → leak-free).
     Frozen + linear probe, `input_kind="imagenet"`. Loads via
     `timm swin_large_patch4_window12_384(img_size=768)` + key remap. Probe
     best-val macroAUC **0.8584** (the strongest member).
   - **M5 BiomedCLIP** `members/biomedclip.py` — `microsoft/BiomedCLIP-
     PubMedBERT_256-vit_base_patch16_224` CLIP image tower (512-d,
     `input_kind="clip"`), frozen + linear probe. Probe macroAUC **0.7610**.
   - **Per-member native-size handling** (load once at `max(native_size)=768`,
     `forward_ensemble_full` resizes the shared tensor to each member's native
     via bilinear `F.interpolate`). **Proven correct:** per-image Mahalanobis
     Spearman ρ = **0.9977** between 3-member@224 and 5-member@768→224 on
     identical covid images — the 768→224 downsample for the 224-native members
     does NOT degrade RAD-DINO's Mahalanobis.
   - 5-member ensemble order: `xrv_nih, convnextv2, raddino, biomedclip, arkswin`
     (probs stacked → 3-member subset = `probs[:, :3, :]`). Arrays re-collected
     FULL (no `--limit` cap — the prior capped run's first-N OOD subset destroyed
     the Mahalanobis signal) at `runs/phase4_features_5mem/`. `Alignment.stack`
     zero-pads features to D_max=1536 so RAD-DINO's 768-d features (first 768
     cols) keep their effective Mahalanobis space; adding Ark+/BiomedCLIP does
     NOT change RAD-DINO's Mahalanobis.

   **Phase-3 acceptance (plan §J.5), 5-member vs 3-member-subset on matched
   768-regime arrays (`scripts/eval_phase3_5member.py`) + production eval
   (`scripts/eval_production.py`):**
   - **(a) AURC 5≤3 — PASSES in-distribution:** OpenI-D all 0.0014 (5mem) vs
     0.0030 (3mem), a 53% reduction. Under heavy shift AURC is slightly worse
     with 5 (kermany 0.565 vs 0.530; covid 0.637 vs 0.576) — but AURC under
     ~70%-prevalence shift is dominated by the base error rate, not the
     uncertainty ordering; the M=5 power-law knee is an in-distribution
     selective-prediction claim and it holds there.
   - **(b) top-5% disagreement AUROC — DROPS everywhere (the honest negative):**
     openiD_all 0.534→0.299, kermany 0.453→0.410, covid 0.423→0.383. This is the
     **confident-agree-wrong** limit (Abe NeurIPS 2022): the two new strong,
     correlated members (Ark+, BiomedCLIP) AGREE on confidently-wrong cases,
     diluting cross-member std. Expected — and exactly why the production flag
     is Mahalanobis, not disagreement. Adding members improves risk-coverage
     (AURC) but cannot improve a disagreement-based confident-error detector
     beyond the irreducible agree-wrong core.
   - **(c) Mahalanobis beats chance under shift — PASSES.** Production eval on
     full 5-member arrays (top-10% confidence): kermany **0.773** (n=5856,
     full), covid **0.521** (n=2024 — see data-loss caveat), OpenI-D
     **0.757** (in-distribution, also beats disagreement 0.704). Disagreement
     (std/MI) is at/below chance on both OOD sites (0.399/0.379, 0.295/0.300).
     The Mahalanobis flag is member-count-agnostic (RAD-DINO features only) and
     remains the only detector that beats chance under shift.

   **Clean under-shift comparison = Kermany (full 5856, no data loss):** 5mem
   Mahalanobis 0.773 vs 3mem 0.818 — both strongly beat chance; 5 members
   slightly reduce the top-10% Mahalanobis AUROC because the stronger 5-member
   p_bar selects a different (larger, harder-to-separate) confident set, not
   because the per-image scores changed (ρ=0.9977).

   **Covid data-loss caveat (environmental, NOT a code bug):** 4408/6432 covid
   *train* images were evicted from the HF cache
   (`datasets--hawking32--covid_chestXray/.../train/PNEUMONIA` empty, `train/
   NORMAL` lost 652) AFTER the 3-member run, BEFORE the 5-member re-collection.
   The 5-member covid array = 2024 survivors (1288 test + 736 train), so the
   covid 5-vs-3 comparison is confounded (3mem full-6432 Mahalanobis 0.649 vs
   5mem 2024 0.521). On the SAME 2024 images, 3mem Mahalanobis is 0.366 (below
   chance) and 5mem 0.521 (beats chance) — but that subset is test-heavy and
   harder for 3mem than the full 6432, so no clean covid member-count claim is
   possible. **Lesson:** both `--limit` caps AND HF cache eviction destroy the
   OOD Mahalanobis signal by selecting unrepresentative subsets; pin/verify OOD
   data on disk before re-collection. Outputs: `runs/phase4_features_5mem/`,
   `runs/production_5mem/`, `runs/phase3_5mem/phase3_5member_summary.json`,
   `checkpoints/{arkswin_probe.pt, biomedclip_probe.pt}`. Tests: 23 pass
   (`test_member_interface.test_arkswin_member_interface` +
   `test_registry.test_phase3_member_specs_registered` updated). Remaining
   future: thread RAD-DINO features through the live CLI path (live
   Mahalanobis); class-conditional conformal; CRC; re-download covid train for a
   full 5-vs-3 covid comparison.
4. Optional: a Mahalanobis variant fit per-pathology (not just Pneumonia pos/neg)
   once multi-class OOD probes are available.
4. **Interactive demo app (DONE 2026-08-03, plan §N).** `scripts/demo_app.py` — a
   Gradio Blocks app (gradio 6.22) that demos uncertainty detection live per-image,
   surfacing all three flags (disagreement / conformal triage / Mahalanobis OOD)
   for a single CXR with a 6-image built-in gallery (4 OpenI in-distribution +
   kermany/covid OOD). Single new file, no package edits — a read-only wrapper
   over the live `cli.py` inference path with the fixed 3-member ensemble
   (xrv_nih,convnextv2,raddino) that matches both sidecars. Run:
   `pip install "gradio>=4.0" && env PYTHONPATH=/teamspace/studios/this_studio
   python scripts/demo_app.py` → http://0.0.0.0:7860. **Key correctness point:**
   the Mahalanobis percentile reference must be the out-of-sample in-distribution
   scores (`arrays_openiD.npz` `mahalanobis`, median 983 / 9c5th 1859), not the
   in-sample OpenI-C scores (median 667, biased low — training points sit closer
   to their own centroid). In-sample reference makes every out-of-sample image
   look OOD (all 97-100%), wrecking the contrast; OpenI-D reference gives clean
   in-distribution 57-90% vs OOD kermany/covid 99.4-99.9% (threshold 95%). Verified
   end-to-end via the live Gradio API: OOD kermany → HIGH-RISK banner, Mahalanobis
   99.85th pct, member bar shows the 3 members AGREE (low std) = the
   confident-agree-wrong core where the disagreement flag stays quiet — the
   headline that only Mahalanobis beats chance under shift.
5. ~~**From-scratch Baur-comparable control (NEXT; the regime test that gives
   Phase 5's negative its meaning).**~~ **DONE 2026-08-23 (Phase 6).** Trained a
   5-seed from-scratch D-Ens on ReX for **all three** Baur anchors — ResNet-18,
   ViT-Tiny, *and* ConvNeXt-Tiny — on the full ReX-train frontal set (112,967
   imgs; CheXpert-rule train labels via the queued shard labeler), same ReX
   splits/labels/`eval_baselines.py` as Phase 5. **Result: yes — disagreement
   wins in the from-scratch regime.** Confident-error AUROC top-10%:
   `epistemic_std` 0.636 / 0.719 / 0.669 vs `confidence` 0.559 / 0.674 / 0.539
   (gap +0.045–0.130; top-50% DeLong p=0.0 all three; top-10% p=0.0071 / 0.0603
   / 0.0) — the opposite of the frozen-foundation p=0.99. Mechanism: from-
   scratch members are at/below-chance self-detectors (Panel A ~0.43–0.52)
   whose cross-member disagreement is the error signal. Pooled confidence still
   wins full-coverage abstention (AUAC 0.94–0.96) in both regimes; mahalanobis
   near-chance in-distribution in both; kNN helps in foundation (RAD-DINO
   features) but not from-scratch (`hybrid ≡ epistemic_std`). See Phase 6 above
   for the full tables, the Baur reconciliation, and the regime-robust vs
   regime-dependent split. **Cross-backbone follow-up (also DONE):** a
   cross-arch from-scratch ensemble reproduces the Phase-5 deflation at the
   defining top-10% metric (std 0.5585 ≈ pretrained 0.5626), so the top-10%
   driver is **architecture diversity, not regime**; regime dominates only at
   broader top-50% selectivity. The Phase-5 negative is thereby scoped to
   "architecture-diverse deployment at high selectivity," not "the method
   fails." Task 2 (uncertainty-label) stays tentative on ReX's labeler-inferred
   −1 labels; CANDID-III (the planned expert-−1 test) is removed from scope —
   the NZ government has banned use of that dataset.
6. ~~**ReX-adapted pretrained arm (the production-realistic lens; NEXT).**~~
   **DONE 2026-08-31 (Phase 7).** LP-FT'd all four pretrained members on
   ReX-train under the identical Phase-6 protocol (Route A table: convnextv2
   LP-FT ×3 seeds, xrv_nih LP-FT, raddino LP + 1 short FT, arkswin LP-only
   @768), rebuilt the full eval battery + retrieval index under adapted
   weights, and ran the pre-registered comparison. **Result: the std edge
   re-emerges** — top-10% confident-error AUROC std 0.6789 vs conf 0.5417
   (DeLong p=0.006; s1/s2 p=0.0045/<0.0001) vs the un-adapted tie (p=0.99) and
   inside the Phase-6 from-scratch band (0.636–0.719). Deployment keeps both
   regimes in `demo_app_rex.py --regime {unadapted,adapted}`. See Phase 7
   above.
7. **Agent 3 — Error Explanation Agent (DONE 2026-09-01).** The third agent
   answers the reviewer's remaining question — *why might this prediction be
   wrong?* — as a **constrained renderer over structured evidence**, not a
   free-text VLM (SOTA anchors MAIRA-2/ChestX-Reasoner/GREEN/CREST supply the
   *constraints*; the pixel-reading adjudicator stays proposed-not-built and
   the DUA forbids pixels leaving the machine anyway). Deterministic code
   computes every signal into an evidence pack with dotted ids
   (`EV:<section>.<key>`): prediction, member disagreement (static
   inductive-bias note, never image interpretation), label ambiguity
   (near-threshold classes), measured image quality (contrast/sharpness/
   clipping/positional asymmetry — explicitly *not* rotation — with
   eval-split, view-stratified quintile bands from a committed 1,010-image
   calibration), dataset bias (zero-prevalence classes, predicted∩zero-prev,
   confident-wrong count), and the Agent-2 neighbor block. An LLM (ollama
   local → ollama cloud glm-4.6 if `OLLAMA_API_KEY`, → template) synthesizes
   3 paragraphs under a cite-everything prompt; a mechanical 0-tolerance
   faithfulness audit (citation validity, numeric grounding, NIH-lexicon
   class grounding, local band agreement) replaces any failing rendering
   with the deterministic template + red note. Gate proven before any LLM:
   50-case template batch → 0 failures; adversarial mock rendering fails with
   exactly the 3 expected audit failures. Privacy in code: no image_path/
   patient_id ever emitted, gt gated behind `include_gt` (app = False), API
   key env-only. App: 8th output + "Explain this prediction" button +
   backend radio on 7862. `docs/agent3_explanation.md`;
   `pytest tests/test_agent3.py tests/test_retrieval.py` → 20 passed.
8. **Agent 4 — Improvement Suggestion Agent (DONE 2026-09-01).** The fourth
   agent answers the reviewer's last question — *what should be done about
   these errors?* — at **population level** over the adapted-ensemble eval
   split, with the same honesty architecture as Agent 3: deterministic
   evidence pack (dotted `EV:` ids) → per-lane LLM rendering → 0-tolerance
   mechanical audit → template fallback. Five lanes, all computed from
   `arrays_rex_adaptedeval.npz` + cal-only assets: **(a)** class data need
   (need_score = rank-z of fp_rate + fn_rate + 1/(npos+5) + prev_drift over
   the 10 measurable classes; top need band = Pleural_Thickening,
   Pneumothorax, Consolidation, Mass; Nodule/Emphysema/Fibrosis/Hernia carry
   zero labelled eval cells → *unmeasurable*, a dataset-composition finding);
   **(b)** augmentation gap (measured `compute_quality` over 5,014 error rows
   + view-stratified matched controls ≈ 3.7 min, cached npz: quality metrics
   barely separate errors — contrast_span err-band share 18.43% vs pop
   18.12% — so the deterministic verdict is "no quality lever identified";
   repo-policy gap leaves recorded: hflip 0.5 + rot ±10° present, no
   CLAHE/blur/vflip/jitter); **(c)** threshold levers (candidates refit on
   the CAL split only — youden/fbeta2/prev-matched — with
   `eval_used_for_selection=False`; near-threshold mass 48.54%; all 10
   measurable classes FP-dominated, ratios 2.2–61.9×);
   **(d)** retrain vs fine-tune vs calibrate (correctness AUROC +
   patient-cluster CI per class → 9× threshold_calibration, 4× unmeasurable,
   1× fine_tune (Pleural_Thickening, AUROC CI 0.785–0.871 + quality drift);
   OOD separation weak: err-row Mahalanobis median 6.27% off population);
   **(e)** metadata slices (view: AP 20.58% vs PA 11.72% err rate, ratio
   1.135; Domino-lite KMeans k=6 on L2-normalized RAD-DINO features of error
   rows: best cluster 8.78%, worst 37.29% err rate — 4.25× spread the
   per-class view ignores). Every CIs patient-cluster bootstrap (37.7% of
   eval patients have ≥2 images; seed 1234, n_boot 1000). Hard bans enforced
   by audit regexes: causal language, execution claims, invented
   interventions; thresholds refit on cal only; patient_id factorized to
   int codes and image_path never emitted. Gate proven before any LLM:
   template batch → **0 failures across 5 lanes**; pack reproduces every
   design-phase ground truth (5,014 err rows / 14,192 err cells / 18.12%;
   AP 20.58 vs PA 11.72; clusters 8.78/17.88/37.29/22.98/13.72/9.01%).
   `docs/agent4_improvement.md` (§6 cross-lane priority table is
   deterministic-only: 6 low/medium-cost threshold + 1 medium fine-tune
   hypotheses); app: independent "Suggest improvements (Agent 4)" button +
   output on 7862, lazy singleton preferring the committed evidence pack;
   `pytest tests/test_agent4.py` → 21 passed (agent3 suite unchanged, 17).
   LLM pass (`--render`, local glm-5.3-flash:cloud) exercised the gate for
   real: 2 lanes timed out → template; the 3 renderings that came back all
   **failed the audit** (16 catches total — comma-batched citations like
   `[EV:a, EV:b]` rejected, ungrounded numerals, and — the best catch — the
   LLM quoted 34.78% for the POSTERO_ANTERIOR slice that the reporting floor
   had excluded, exactly the instability the floor exists to prevent) → all
   swapped to template. Final committed report is therefore all-deterministic
   prose with per-lane red notes documenting the swap; the audit trail of
   what the LLM tried to claim is preserved in
   `runs/app_rex_adapted/agent4/faithfulness_report.json`.
