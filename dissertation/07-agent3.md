# Agent 3: Evidence-Grounded Error Explanation

Agent 1 flags a prediction and Agent 2 supplies comparable cases. Agent 3 answers the remaining question — *why might this prediction be wrong?* — in a short paragraph a clinician can read. This chapter sets out the design stance in Section 7.1, the evidence pack in Section 7.2, the prompt contract in Section 7.3, the mechanical audit that is the component's real contribution in Section 7.4, and the validation and privacy controls in Sections 7.5 and 7.6.

## Design stance: a constrained renderer, not a vision-language model

The state of the art in radiology explanation reads pixels. MAIRA-2 [@bannur2024maira] generates grounded reports in which findings are tied to image regions; ChestX-Reasoner [@fan2025chestx] adds process supervision so intermediate reasoning steps can be checked. This work deliberately does not take that approach, for two reasons.

The first is auditability, and it is the substantive one. A free-form narrative about what is visible in a radiograph cannot be mechanically checked against what is actually in the radiograph. A hallucinated finding — a fluent, confident, unsupported claim [@ji2023hallucination] — is precisely the failure mode this entire project exists to detect in a classifier. A system that detects confident unsupported claims by a convolutional network while emitting confident unsupported claims of its own in natural language is self-defeating, and no amount of prompt engineering converts an unverifiable claim into a verifiable one.

The second is governance. ReXGradient's data-use agreement forbids images leaving the machine, so a pixel-reading cloud model is excluded by construction. Where a governance constraint and a design principle point the same way, the design principle is the one worth recording.

Agent 3 is therefore a **constrained renderer over structured evidence**:

```
deterministic code computes every signal
        -> an evidence pack, every leaf addressable as EV:<section>.<key>
        -> LLM synthesis under a cite-everything prompt        (optional)
        -> a mechanical 0-tolerance faithfulness audit
        -> any failure is replaced by the deterministic template rendering
```

The language model never sees pixels; every image-level statement it can make originates in a computed metric. It never sees anything outside the evidence pack. This sits deliberately *below* the capability ceiling of a grounded vision-language model, and buys in exchange a property those systems cannot offer: a provable correspondence between every claim and a specific computed quantity. The SOTA systems supply the *constraints* here — the "every claim tied to evidence" contract from MAIRA-2, the verifiable-intermediate-steps idea from ChestX-Reasoner, and the error taxonomy from GREEN [@ostmeier2024green] — rather than the models. Rudin's argument [@rudin2019] applies: rather than claim post-hoc interpretability of the classifier, the *uncertainty layer's* reasoning is made fully auditable, which is a weaker goal and an achievable one.

## The evidence pack

Deterministic code assembles six sections, each leaf addressable by a dotted identifier such as `EV:prediction.flag_class`.

- **`prediction`** — the argmax class, raw and temperature-scaled pooled probabilities, the predicted set, the Region-B flagged class with its probability and Youden threshold, the Mahalanobis percentile and out-of-distribution indicator, and the member count.
- **`member_disagreement`** — the per-member probabilities, the pooled mean and threshold, the cross-member standard deviation, and the split of members above and below threshold. This section carries a fixed note that cross-architecture disagreement is a difference of inductive biases and is *never* an interpretation of the image — a constraint that follows directly from Chapter 5's mechanism finding and which prevents the renderer from narrating disagreement as though the members had seen different things.
- **`label_ambiguity`** — classes within 0.10 of their decision threshold, the maximum class probability, and flagged classes that carry no label.
- **`image_quality`** — measured quantities only: contrast span and standard deviation, exposure statistics, mid-grey deviation, clipping fractions, Laplacian variance, Tenengrad focus, mirror positional asymmetry and centroid offset. Each is placed in a quintile band derived from a committed calibration over 1,010 evaluation images stratified by projection, so that "low contrast" means low *relative to this corpus and this projection*, not relative to an intuition.
- **`dataset_bias`** — per-class prevalence across splits, zero-prevalence classes, the intersection of predicted classes with zero-prevalence classes, projection shares, and the confident-wrong count.
- **`neighbors`** — Agent 2's output: neighbour count, similarity range, split composition, model-correct and uncertain counts, the report-confirmation vote and the top-five identifiers.

One naming decision deserves comment. The brief listed "unusual rotation" as an error characteristic to summarise. Rotation is not what the code measures; it measures **mirror positional asymmetry**, which is a different quantity that rotation is only one cause of. The metric is named for what it measures, and the prompt contains an explicit rule forbidding the renderer from describing it as rotation. Naming a proxy after the thing it proxies is how an unfalsifiable claim gets into a report.

## The prompt contract

The system prompt imposes seven hard rules: cite an `EV:` identifier for every factual claim; never invent an image finding, and name only pathologies present in the evidence; do not describe mirror asymmetry as rotation; treat neighbours as retrieval results, not ground truth; make dataset-bias claims only from the bias section and only with its numbers, phrased as "in this dataset"; use uncertainty-preserving language ("consistent with", never "the diagnosis is"); and emit no identifiers, paths or device names.

The output contract is three short paragraphs of at most 180 words — why the case was flagged, what the neighbours do and do not support, and exactly one category drawn from a closed set of five: image-quality artefact, label ambiguity, dataset bias, genuine model error, or no error signal present — followed by a fixed non-diagnostic disclaimer.

The user message is the flattened evidence pack, one `EV:id = value` line per leaf, sorted. There is no chain of thought and no tool use: the evidence pack *is* the reasoning, computed deterministically before the model is invoked.

## The faithfulness audit

The audit is the component's actual contribution. It runs in under a millisecond, is purely mechanical, and has **zero tolerance**: any failure means the language model's rendering is discarded and replaced by the deterministic template, with a visible note.

Four checks run.

- **Citation validity.** Every `[EV:<id>]` must resolve to a real leaf of the pack. This catches invented identifiers and malformed citations.
- **Numeric grounding.** Every number appearing in a sentence must match a number reachable from that sentence's own cited leaves, within a 2% relative or 0.011 absolute tolerance. This is the check that catches a fabricated statistic wearing a valid citation, which is the most dangerous failure because it looks correct.
- **Class grounding.** Any pathology named from the NIH lexicon must appear in the evidence. This catches a hallucinated finding directly.
- **Band agreement.** A qualitative band phrase must match the evidence's own band for that metric, matched longest-phrase-first so that "low" inside "very low" is not misread as a mismatch.

A word-count cap is enforced, and an advisory coverage warning fires when fewer than three of the four content axes are addressed; the warning does not gate.

The design consequence is that the *worst case* is bounded. When the language model behaves, the reader gets fluent prose with every claim traceable. When it does not, the reader gets a slightly stilted but fully grounded deterministic paragraph and an explicit note that the model's output was rejected. There is no configuration in which unverified generated text about a patient's radiograph reaches a reader.

## Validation

The gate was proven before any language model was involved: a batch of 50 evaluation cases, spanning confident errors, no-finding cases and out-of-distribution cases, rendered by the deterministic template alone, produced **zero audit failures** across the whole batch. That establishes that the audit does not simply reject everything.

Two adversarial tests establish the converse. A deliberately corrupted rendering — fabricated number, invented citation, hallucinated pathology — fails with exactly the three expected checks and no others, so the audit is specific as well as sensitive. And a mock "liar" provider injected into the chain produces the backend annotation `ollama_local->template`, confirming that a failing rendering is swapped rather than shown.

The realistic setting is measured in Chapter 8, where the same architecture was exercised against a live language model over five lanes; the honest summary is that the audit rejected both of the renderings the model produced within the time limit. That is reported there rather than here because Agent 4's run is the one with a complete record.

An empirical limitation should be stated. Explanation *usefulness* — the brief's "simulated clinician rating" — was not measured. What is measured is faithfulness: whether every claim is traceable to a computed quantity. These are different properties, and a perfectly faithful explanation can still be unhelpful. A reader study is the correct instrument and is listed in Section 10.3 as future work.

## Privacy, enforced in code

Three controls are implemented in the module rather than in documentation, and each is asserted by a test. The evidence pack cannot emit an image path or a patient identifier; the case reference is a truncated hash. Ground truth is gated behind an explicit flag which the deployed application sets to false, so the reviewer's explanation is derived from model internals only, exactly as it would be on an unlabelled clinical case. And any API key is read from the environment and never enters the evidence pack, the prompt or any saved artifact — a test asserts that a planted key value appears in none of them.
