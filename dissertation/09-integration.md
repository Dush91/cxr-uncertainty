# Integration, Deployment and Professional Practice

This chapter covers what ties the four agents together and the practice around them. Section 9.1 describes the deployed application. Section 9.2 explains the orchestration choice — specifically, why there is no autonomous planner — and shows the two-region decision rule from Section 5.7 as implemented. Section 9.3 covers testing, Section 9.4 reproducibility, and Section 9.5 data governance. Section 9.6 covers code publication. Section 9.7 evaluates the system against the four outcomes the brief specified.

## The deployed application

The system is delivered as a Gradio [@abid2019gradio] application. A reviewer uploads a radiograph, or selects one of the built-in examples, and receives seven linked outputs: a risk banner; a per-pathology table giving the calibrated probability, the Youden threshold, the decision, the calibrated confidence, the cross-member standard deviation, the abstention flag, the disagreement flag and the error type; a per-member probability bar chart with a caption; a gallery of retrieved comparable cases with their metadata table and a caption naming the retrieval lane that was selected; and, on request, Agent 3's explanation. A separate control renders Agent 4's population-level analysis.

![The deployed application. A single upload drives all four agents; the retrieval caption names the lane the system selected and why.](figures/fig-ch9-app.png){#fig:ch9-app}

The application carries a `--regime` switch selecting between the un-adapted and ReX-adapted ensembles of Chapter 5, which is a paths-only swap. Keeping both live is deliberate: the comparison in Section 5.6 is the project's most consequential result, and a reviewer can see both sides of it on the same image rather than take the tables on trust.

Three presentation decisions reflect findings rather than taste. The disagreement flag carries an in-interface caveat that on the un-adapted pretrained ensemble it is statistically equivalent to confidence, so it should be read as second-order triage only — the interface states the limitation from Section 5.2 rather than implying a capability the measurement does not support. The Mahalanobis percentile is presented as an image-level shift watchdog and explicitly not as an error ranker, per Section 5.2. And the retrieval caption discloses that the correct-cases lane excludes training rows because their correctness annotations are memorisation-contaminated (Section 6.4). An interface that overstates its own signal is a safety problem in this domain, so each of these is surfaced rather than buried in documentation.

## Orchestration: explicit composition, not an autonomous planner

There is no agent framework in this system. No LangGraph, no tool-calling loop, no planner deciding which agent to invoke. The composition is direct Python function calls, with structured hand-offs between them.

This was a considered decision and is defensible on three grounds. First, **auditability**: every inter-agent edge is a typed data structure, so every edge can be unit-tested, and the tests in Section 9.3 do exactly that. A planner that decides at run time which agent to call cannot be tested in the same way, because its behaviour is a property of a language model's output rather than of the code. Second, **there is no planning problem to solve**: the pipeline order is fixed by the data dependencies — the flag must exist before retrieval can be keyed on it, and both must exist before an explanation can reference them — so a planner would add a failure mode and a latency cost while choosing among a single valid ordering. Third, **determinism under audit**: the guarantee in Chapters 7 and 8 is that no unverified generated text reaches the reader, which requires knowing statically what is generated and when.

The couplings are the interesting part and are explicit. Agent 1 to Agent 2 passes `flag_class`, the Region-B-flagged class with the highest calibrated confidence, computed from model internals with no access to the query's label; the application's default `auto` mode resolves to the flag lane when Region B fires and the correct-cases lane otherwise, and prints the resolution it chose. Agents 1 and 2 to Agent 3 pass the structured evidence pack, whose `prediction`, `member_disagreement` and `neighbors` sections are precisely their outputs. Each per-image request clears the cached evidence at entry, so stale evidence from a previous case cannot leak into a new explanation.

The two-region rule of Section 5.7 is implemented directly. **Region A** sets an abstention flag when the calibrated confidence falls below the calibration split's top-decile cut; this is the whether-to-abstain decision, and pooled confidence is the right score for it. **Region B** sets a disagreement flag when a record is in the confident set and its cross-member standard deviation is above the median within that set; this is the which-confident-call-is-wrong decision. The two flags appear as separate columns because they answer separate questions, and merging them into a single "risk" number would discard the distinction the research established.

## Testing

The suite comprises 91 tests across eight files, all of which run on CPU in about thirty seconds without a GPU, a language model or the patient corpus. At the last full run, 90 passed and one failed: a BiomedCLIP import that requires an optional dependency absent from this environment, which is a pre-existing environment gate rather than a regression.

Coverage is organised around the properties that could silently invalidate results rather than around line coverage.

- **`test_leak_free.py`** (6) asserts the leakage controls of Section 3.4: that the default ensemble excludes the contaminated weights, that no OpenI evaluation may include them, that the manifest's splits are patient-disjoint, and that the validity masks match each source's actual label coverage.
- **`test_statistics.py`** (12) asserts the statistical machinery, including the Murphy decomposition identity for the Brier score, that DeLong returns *p* = 1 for identical scores and degrades gracefully on a degenerate class, and that E-AURC is non-negative.
- **`test_member_interface.py`** (8) asserts the member contract, including that the torchxrayvision probability convention remains a valid probability, that `Alignment.stack` produces correct validity masks, that the DenseNet correctly reports having no dropout, and that the 768-pixel member is fed at its native size.
- **`test_registry.py`** (15) asserts the registry seam, including that the default risk policy is the same object as before the refactor and that `ts_only_mahalanobis` degrades honestly when features are unavailable rather than silently producing a different score.
- **`test_conformal.py`** (9) asserts the Mondrian coverage guarantee, the rare-group fallback, set-size regimes and sidecar round-tripping.
- **`test_retrieval.py`** (4) asserts the label-free store invariant, HNSW recall against exact search, and query self-exclusion.
- **`test_agent3.py`** (16) and **`test_agent4.py`** (21) assert the evidence schemas, the privacy properties, template determinism, each audit check individually, and that a mock liar provider is swapped for the template.

Two tests are worth singling out. `test_agent3.py` carries a **golden-value contract** for the image loader, capturing its output before a refactor so that a change in preprocessing cannot silently invalidate every saved array. And `test_agent4.py`'s split-discipline test asserts `eval_used_for_selection is False`, turning a methodological commitment into a build-time check. Writing the Agent 4 suite also surfaced a genuine latent defect: an unguarded division by the population error rate in the stratification lane, which crashes on a split containing no errors. It was found because the synthetic fixture happened to produce a separable problem, and it was fixed.

## Reproducibility

Every number in Chapters 4 and 5 is recomputed from saved arrays rather than from a re-run of inference. The evaluation arrays, per-run comparison JSON files, calibration sidecars, checkpoints and manifests are all retained, and the analysis scripts accept an arrays directory so that the whole battery re-runs on CPU in minutes. This is what makes the data-deletion policy of Section 9.5 possible without losing the ability to check the work, and it is what made the cross-backbone experiment of Section 5.5 free: the decisive cell of the design was built by recombining predictions that already existed.

The dissertation itself is reproducible on the same terms. The figures are generated by a script that reads only saved JSON and npz, and the document is built from Markdown sources by a script that renders into the school's template.

## Data governance

ReXGradient is distributed under a data-use agreement forbidding persistence on third-party infrastructure, and the GPU work was rented. The working pattern was therefore fixed in advance: upload the pre-decoded cache with end-to-end SHA-256 verification, train or infer, pull checkpoints and derived arrays back to the local machine, then delete every image, cache, manifest and label file from the host and verify by a repository-wide scan that nothing remains. This was done after each phase, not once at the end.

Two design consequences follow and are worth noting because they are not merely administrative. Agent 3's refusal to send pixels to a language model is partly a governance consequence and partly a design principle (Section 7.1), and the two agree. And the reproducibility discipline above is what makes deletion tolerable: because the arrays are sufficient to recompute every result, the images do not need to be retained.

## Code publication

The repository as it stands is 206 GB, of which about 192 GB is licensed patient data and 8.8 GB is model checkpoints. Publishing it requires a deliberate exclusion pass rather than a default `git init`.

The published repository contains the package, the scripts, the tests, the documentation, the dissertation sources and the small JSON and Markdown result summaries that carry no patient-derived content. It excludes all corpora, all checkpoints, all arrays and manifests, and in particular the labelling outputs, which contain free-text radiology reports. A pre-publication scan checks staged content for image paths, patient identifiers and report text. The exclusions and their reasons are recorded in a `PUBLISHING.md` file, so that a reader can see what is missing and why rather than inferring it.

## Evaluation against the brief

The brief specified four deliverables.

**A working agentic pipeline** covering error detection, uncertainty analysis, similarity retrieval, automated explanation and automated improvement suggestion. Delivered, with all five capabilities implemented and measured. The uncertainty analysis substantially exceeds the brief's scope, which specified Monte Carlo dropout and temperature scaling: MC-dropout was implemented, measured, found to be a no-op for the chosen architectures and rejected with evidence (Section 3.3); temperature scaling was implemented and ablated against three alternatives (Section 4.4); and conformal prediction, feature-space density and representation-mismatch scores were added.

**A lightweight clinician-facing interface** showing image, prediction, similar cases, explanations and suggestions. Delivered as described in Section 9.1, running on a CPU laptop as the brief required.

**A structured error analysis report** generated by a language model, covering major error types, dataset bias patterns and improvement recommendations. Delivered as Agent 4's report, with the significant qualification that the shipped renderings are the audited ones — which, in the live trial of Section 8.6, meant mostly the deterministic templates. That is the correct outcome under the design and is reported as such.

**Evaluation.** The brief asked for qualitative retrieval assessment, a simulated clinician rating of explanation usefulness, and optionally a measured error-rate reduction. Retrieval was evaluated quantitatively rather than qualitatively, which exceeds the ask (Chapter 6). The clinician rating was **not** conducted; explanation *faithfulness* was measured mechanically instead, which is a different and narrower property, and Section 10.3 lists the reader study as future work. The error-rate reduction was not attempted, because Agent 4's own analysis found no retraining trigger and identified threshold placement as the reachable lever — acting on that and re-measuring would require a second evaluation corpus to avoid the split-discipline violation of Section 8.3.

Beyond the brief, the work delivered a powered leak-free benchmark, a fifteen-model from-scratch control study, the regime-and-diversity decomposition of Chapter 5, and the adaptation experiment — none of which the brief anticipated, and which constitute the research contribution.
