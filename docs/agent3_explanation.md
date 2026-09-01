# Agent 3 — Error Explanation Agent (LLM, evidence-grounded)

*Companion to `FINDINGS.md`, `docs/uncertainty_evaluation.md` (Agent 1) and
`docs/agent2_retrieval.md` (Agent 2). Status: **implemented and validated on
ReX** (2026-09-01). The LLM lane is optional — every path degrades to the
deterministic template renderer.*

## 1. Purpose and scope

Agent 1 answers *"should I trust this prediction?"* (UQ + confident-error
flagging); Agent 2 answers *"show me cases like this one"* (retrieval). Agent 3
answers the reviewer's remaining question: **"WHY might this prediction be
wrong?"** — in a short, clinician-readable paragraph. It summarizes error
characteristics (low contrast, unusual rotation → measured as positional
asymmetry, ambiguous anatomy, dataset-bias indications), compares the retrieved
neighbors to the error case, and produces a 3-paragraph explanation.

## 2. Design stance: constrained renderer, not a free-text VLM

SOTA CXR explanation systems (MAIRA-2's grounded report generation,
ChestX-Reasoner's process-supervised reasoning) **read pixels**. Two reasons
that stance is deliberately not taken here:

1. **Auditability.** A free-form VLM narrative cannot be mechanically checked
   against what is actually in the image; hallucinated findings are the exact
   failure mode this project exists to flag.
2. **Privacy/DUA.** ReXGradient is a Harvard non-commercial dataset: images
   must not leave the machine, so any pixel-reading cloud model is off the
   table by construction.

So Agent 3 is a **constrained renderer over structured evidence**:

```
deterministic evidence pack (every leaf has an id EV:<section>.<key>)
        →  LLM synthesis under a cite-everything prompt   (optional lane)
        →  mechanical faithfulness audit  (0-tolerance gate)
        →  audit failure ⇒ deterministic template rendering + red note
```

The LLM never sees pixels (image-level claims come from **computed metrics
only**) and never sees anything not in the evidence pack. This is deliberately
*below* the MAIRA-2/ChestX-Reasoner capability ceiling: capability is traded
for a provable claim-evidence correspondence. The pixel-reading VLM
adjudicator remains **proposed-not-built** (inherited from
`docs/agent2_retrieval.md` §6).

**SOTA anchors** (what each contributes): MAIRA-2 (arXiv:2406.04449) — grounded
report generation, the "every claim tied to evidence" contract; ChestX-Reasoner
(arXiv:2504.20930) — process supervision / reasoning-trace evaluation, the
ancestor of our audit gate; GREEN (EMNLP 2024) — error taxonomy
(hallucination/omission/location/severity/comparison) for LLM report critique;
CREST (2025) — risk-weighted generation. Agent 3 borrows the *constraints*, not
the models.

## 3. Evidence pack (`agent3.build_evidence` → `schema="agent3_evidence_v1"`)

Every leaf is addressable as `EV:<section>.<key>` (lists flatten to
`EV:<section>.<list>.<i>`). Sections:

| Section | Carries |
|---|---|
| `prediction.*` | argmax + pooled p (raw + TS), predicted set, Region-B `flag_class`, key-class p / Youden threshold, Mahalanobis percentile + OOD flag, member count |
| `member_disagreement.*` | per-member probs, pooled p, Youden threshold, cross-member std, sign split ("N above / N below threshold") + a static interpretation note (cross-arch disagreement = inductive-bias difference, **never** an image interpretation) |
| `label_ambiguity.*` | near-threshold classes (\|p−thr\| ≤ 0.10), max-class p, unlabeled-flagged classes; `gt_vector` only when `include_gt=True` (offline eval) |
| `image_quality.*` | measured metrics from `utils.load_raw_gray` resized to long side 1024: contrast span/std, exposure mean/median/mid-gray deviation, clip fractions, Laplacian variance, Tenengrad, mirror **positional asymmetry** (explicitly *not* a rotation estimate), centroid offset, aspect, bit depth; each with an eval-split quintile **band**; manifest view (AP/PA) + view note |
| `dataset_bias.*` | per-class prevalence (train/cal/eval), zero-prevalence classes, predicted∩zero-prevalence, view shares, query view, confident-wrong eval count |
| `neighbors.*` | n, similarity range, split counts (train-heavy = memorization-adjacent), model-correct / model-uncertain counts, report-confirm vote, top-5 with ids, pool note |

**Privacy in code:** the builder cannot emit `image_path` or `patient_id`;
`case_ref` is a sha1 prefix; ground truth is gated behind `include_gt=True`
(offine batch only — the app passes `include_gt=False`); the API key is read
from the `OLLAMA_API_KEY` environment variable only and never enters the
evidence, prompt, or any saved artifact (enforced by tests).

## 4. Image-quality metrics and bands

`agent3.compute_quality(path)` reuses the exact raw loader from
`utils.load_raw_gray` (16-bit PNG → /65535, DICOM MONOCHROME1 inversion + HU
window — identical preprocessing to the models). Metrics are computed on the
[0,1] image at long side 1024 for comparability across native sizes.

Bands come from **`--calibrate-quality`**: ~1,000 eval-split images stratified
by view (seed 1234) → per-metric / per-view quintile tables
(p20/p40/p60/p80, fixed — no hand tuning), committed as
`runs/app_rex_adapted/agent3_quality_stats.json`. Special case:
`clip_lo + clip_hi > 0.10` → "high clipping" overrides the band. Missing
table → band `"unavailable"` (demo still works). AP and PA tables differ, as
they must (e.g. mirror-asymmetry p80 0.348 AP vs 0.272 PA).

## 5. LLM backend chain

```
ollama_local (127.0.0.1:11434)
  → ollama_cloud (https://ollama.com/api/chat, Bearer $OLLAMA_API_KEY,
                  default model glm-4.6 — strong general models on Ollama
                  cloud; a small medical VLM's vision edge is unused here
                  since the LLM sees only text evidence)
  → template (deterministic rendering, always succeeds)
```

stdlib `urllib.request` only (no new dependencies); temperature 0.2,
seed 1234, 60 s timeout, single-shot — the evidence pack *is* the chain of
thought. `OLLAMA_API_KEY` is environment-only. A medgemma-class local medical
VLM was considered and dropped: it is not on Ollama cloud, and the vision
capability would be wasted on a text-only evidence pack.

The system prompt's hard rules: cite `[EV:<id>]` on every factual claim; only
the 14 NIH pathology names present in evidence; asymmetry is **not** rotation;
neighbors are **retrieval, not truth**; bias claims only from `bias.*` with the
actual numbers ("in this dataset"); uncertainty-preserving language
("consistent with", never "the diagnosis is"); no identifiers. Output: exactly
3 paragraphs ≤180 words (why flagged / what neighbors support / category from
[image-quality artifact | label ambiguity | dataset bias: unseen class |
genuine model error | no error signal present]) + the fixed non-diagnostic
disclaimer.

## 6. Faithfulness audit (the gate)

`agent3.audit(text, evidence)` — mechanical, <1 ms, 0-tolerance:

- **citation validity** — every `[EV:<id>]` must be a flattened evidence leaf;
- **numeric grounding** — every number in a sentence must match a number from
  that sentence's cited leaves (2% relative tolerance; ranges like "0.83-0.90"
  are hyphen-split; numbers embedded in string leaves such as "2 above / 1
  below" count; citation ids are stripped first so `…classes.0` is not a
  claimed number);
- **class grounding** — any NIH-lexicon pathology named must appear in the
  evidence (invented findings fail);
- **band agreement** — a band phrase must match the evidence band *locally*
  (immediately after the metric name), longest phrase first ("low" inside
  "very low" is not a mismatch);
- **length cap** 200 words; **coverage** (touching ≥3 of 4 axes) is advisory.

`render()` walks the provider chain: an LLM rendering that fails the audit is
**never shown** — replaced by the template rendering + a red note naming the
failure. The gate is proven before any LLM: a 50-case template batch
(`scripts/agent3_batch.py --sample 50 --render --backend template`) ends with
**0 gate failures**. An adversarial mock rendering (invented number + bogus
citation + absent pathology) fails with exactly the 3 expected audit failures
(tests).

## 7. Files and usage

| File | Role |
|---|---|
| `cxr_uncertainty/agent3.py` | metrics, evidence builder, provider chain, prompt, template renderer, audit, `render_cached()` |
| `scripts/agent3_batch.py` | `--calibrate-quality` (band tables + bias block), `--sample N --render --backend {auto,template}` (batch evidence + render + `faithfulness_report.json` + `side_by_side.md`) |
| `scripts/demo_app_rex.py` | 8th UI output + **Explain this prediction** button + backend radio (auto/template); `predict()` caches the per-case evidence pack |
| `tests/test_agent3.py` | 17 tests, no Ollama required (mock provider) |
| `runs/app_rex_adapted/agent3_quality_stats.json` | committed band tables + precomputed bias block |

```bash
python scripts/agent3_batch.py --calibrate-quality          # re-fit bands
python scripts/agent3_batch.py --sample 50 --render --backend template
python scripts/agent3_batch.py --sample 3 --render --backend auto \
    --case confident_error --case normal --case ood
```

App: run a prediction → **Explain this prediction (Agent 3)** renders the
cached evidence pack with the selected backend; a failed LLM render shows the
template + red note, never the unfaithful text.

## 8. Validation summary (2026-09-01)

| Check | Result |
|---|---|
| Template batch gate | 50 cases, **0 gate failures** (`faithfulness_report.json`) |
| Adversarial rendering | exactly 3 expected audit failures (number / citation / class) |
| LLM fallback | mock liar provider → backend `ollama_local->template`, template swap passes audit |
| Tests | `pytest tests/test_agent3.py tests/test_retrieval.py` → 20 passed |
| Privacy greps on artifacts | no `patient_id` / `image_path` / `.png` path / API key |
| Calibration tables | 1,010 eval images, view-stratified, AP/PA bands differ, sharpness non-degenerate |

## 9. Limitations

1. **Text-only adjudication.** The LLM explains the *model's* behavior from
   measured signals; it cannot see the image. A pixel-reading adjudicator
   (ChestX-Reasoner-style) remains proposed-not-built.
2. **Neighbor comparison is retrieval-not-truth** — neighbors support the
   "is this read plausible" question only; their reports are reference-side.
3. **Band dynamics.** Quality bands are eval-split quintiles; a deployment
   distribution shift would re-band the population.
4. **English-language evidence.** The prompt and evidence are English-only;
   the disclaimer is fixed.