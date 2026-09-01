# Agent 2 — Similar-Case Retrieval

*Companion to `FINDINGS.md` (research log) and `docs/uncertainty_evaluation.md`
(Agent 1 UQ evaluation). This document covers the retrieval subsystem only:
its purpose, design, SOTA methodology basis, implementation, usage, and
validation. Status: **implemented and validated on ReX** (2026-08-29).*

## 1. Purpose and scope

Agent 1 (UQ + confident-error flagging) answers *"should I trust this
prediction?"*. Agent 2 answers the clinician's natural follow-up: *"show me
cases like this one"* — so that a flagged or uncertain prediction can be
reviewed against comparable, already-predicted cases. It is a standard
content-based image retrieval (CBIR) layer over the same pipeline, with two
first-class queries:

| Query | Mode | Filter semantics |
|---|---|---|
| *"Show me similar images where the model was correct."* | `--mode correct` | reference rows whose pooled ensemble prediction matched their label (`correct=True`); labeler-uncertain rows excluded |
| *"Region B flagged class X — show references whose reports confirm X."* (§2.6) | `--mode flag` | reference rows whose **report** confirms the flagged class X (key = Agent-1's Region-B flag, model internals; no query GT) |
| *"Where has the model itself called X — and been right?"* (§2.7) | `--mode contrast` | reference rows where the **model** reads X-positive (pooled mean ≥ Youden) **and** the report confirms X; cal+eval pool |
| app default (§2.7) | `auto` | resolves to `flag` when Region B fires, `correct` otherwise — no user action |
| *(retired 2026-09-01)* | ~~`--mode pathology`~~ | argmax-bucket filter — weakest lane (see §2.5 gate and the §2.6 motivation case: on a confident error it keyed on the spurious argmax); superseded by `flag`/`contrast` (evidence) + `correct` (trust) + `normal` (no-finding) |

**Deployment story (framing for the whole system):** pretrained CXR models
(xrv/convnextv2/raddino — individually and cross-arch) back **both** Agent 1
(UQ + confident-error flagging) **and** Agent 2 (this retrieval layer). The
from-scratch ResNet-18/ViT-Tiny/ConvNeXt-Tiny D-Ens are *not* a deployment
model: they exist as the leak-free UQ benchmark control (Baur et al.
D-Ens protocol) — members trained from random init with no exposure to any
prior data — to test whether the Phase-5 disagreement finding generalizes to
the literature-standard regime. Agent 2 mirrors that control with
model-matched indexes (`rex_fs_*`), but production retrieval runs on the
pretrained embedding.

**Label-free invariant (hard constraint):** the vector store holds **ids +
vectors only**. Predicted pathology, correctness, and confidence are derived
from the ensemble's own probabilities into a *separate* sidecar parquet and
joined at query time. Ground truth never enters the index and is never
printed by the query CLI. (Labels are used only upstream, to *derive* the
`correct` attribute of reference cases — library curation, not retrieval
ranking.)

## 2. Design decisions and their basis

### 2.1 Embedding: model-consistency first, RAD-DINO as reference

**Two axes must not be conflated:**

1. **Whose correctness? (hard requirement).** The `correct` /
   `pred_pathology` filter must refer to the **deployed Agent-1 model**.
   The sidecar is derived from the same npz that supplies the features, so
   building the index from `runs/eval_arrays/rex` yields pretrained-ensemble
   correctness, while `rex_fs_resnet18` / `rex_fs_vit_tiny` yield
   from-scratch D-Ens correctness (sidecar correct/incorrect/uncertain =
   2,577/5,104/383 vs 2,343/5,352/369 of 8,064). **When Agent 1's production
   path is the from-scratch D-Ens, build the index from the matching
   `rex_fs_*` arrays** — otherwise "similar images where the model was
   correct" silently answers for a different model.
2. **Whose space for similarity? (empirical).** All three are built and
   validated (recall@10 = 1.0000 vs exact in each):

   | Index | Source features | dim | random-pair cos | NN cos | spread | behavior |
   |---|---|---|---|---|---|---|
   | `rex_raddino` | RAD-DINO (pretrained member) | 1536 | 0.081 | 0.582 | **0.502** | well-spread anatomical space; neighbors = visually similar Effusion cases |
   | `rex_fs_resnet18` | ResNet-18 seed-0 (deployed candidate) | 512 | 0.663 | 0.969 | 0.307 | **collapsed**: random pairs already 0.66; one demo neighbor was a *Cardiomegaly* case (space less pathology-aligned) |
   | `rex_fs_vit_tiny` | ViT-Tiny seed-0 | 192 | 0.708 | 0.963 | 0.255 | most collapsed |

   The from-scratch spaces' compression (NN ≈ 0.97 vs 0.58) is the **same
   phenomenon as Phase-6's UQ finding** — supervised from-scratch backbones
   squeeze their penultimate space — and it shows up here as weaker visual
   discrimination despite identical retrieval recall.

**Default rule (v2.0 of this doc, set by the deployment story):** the
production path for **both Agents 1 and 2 is the pretrained ensemble**
(xrv NIH / convnextv2 / raddino [arkswin] — individually and cross-arch);
the from-scratch ResNet-18/ViT-Tiny/ConvNeXt-Tiny D-Ens exist as the **UQ
benchmark control** (Baur MalPIS/UNSURE-2025 protocol — models with *no prior
exposure to any training data*), not as the deployment model. Accordingly the
production index is `rex_raddino` (pretrained features + pretrained-ensemble
sidecar). The `rex_fs_*` indexes exist as the control-lens counterpart: when
Agent 1 is evaluated in the from-scratch control regime, Agent 2 must be
rebuilt from the *matching* arrays so the correctness filter answers for the
same model — the sidecar-consistency rule of §2.1 point 1. If the deployment
model ever changes, the index must be rebuilt from that model's arrays (the
query CLI rejects feature/sidecar mismatches by dimension).

### 2.2 Reference/query discipline

- **Reference library = cal split** (8,064 ReX-valid images, patient-disjoint
  from eval). Queries = eval split. This mirrors deployment (a library of
  past cases) and keeps queries structurally unseen by the library.
- Self-exclusion by `image_id` at query time — the query is never returned.
- Cosine similarity on L2-normalized embeddings (the hnswlib `cosine` space
  re-normalizes internally; stored feats are normalized so exact mode and
  reported similarities are plain dot products).

### 2.3 Index: HNSW (SOTA graph index)

`hnswlib`, M=32, ef_construction=200, ef_search=max(256, 10k), cosine space.

- 2025 ANN-benchmark consensus: graph-based indexes (HNSW, DiskANN) are
  unequivocally superior when maximum recall/latency is required; partition
  methods (IVF/ScaNN) win only high-throughput/moderate-recall lanes; DiskANN
  pays off only at ≥10⁵–10⁶ vectors. **HNSW is the correct SOTA pick for a
  RAM-resident, max-recall medical library.**
- **Scale honesty:** at 8K × 1536-d (~50 MB) exact search is also sub-ms;
  HNSW here is the SOTA-comparable methodology and a *validated scalable code
  path* for the full ReXGradient-train library (~113K images). `index_meta.json`
  records measured recall so the claim stays empirical.
- **turbovec** (TurboQuant, ICLR 2026 — training-free 2–4-bit PQ, SIMD
  search, id-allowlist filtered kernels): the credible *compression lane*
  (16× memory reduction) if the library scales to millions; at 16K vectors
  quantization buys nothing. Not defaulted; benchmark optional.

### 2.4 Filtered search correctness

Both Agent-2 queries are **metadata filters**, the classic failure mode of
vector search (VLDB 2025 survey: post-filtering silently loses results when
selective; naive pre-filtering on HNSW causes graph disconnection). We use
**hnswlib's traversal-time `filter=` predicate** (inline filtering — the
recommended pattern), with `--search exact` brute-force over filtered
candidates as the ground-truth validation path. Filter selectivity is always
reported (`candidates_allowed=N/N_total`).

### 2.5 Label-agreement rerank (built 2026-08-31)

`pathology` and `plain` modes apply a second-stage rerank after the visual
top-(K·4) oversample:

```
score = (1 − λ)·cosine + λ·agreement(p_q, y_c),   λ = 0.3
agreement(p_q, y_c) = Σ_class p_q[c]·1[y_c=1] ⟋ Σ_class p_q[c]·1[y_c≠0]
```

`p_q` is the **query's own pooled ensemble posterior** (no ground truth exists
for a query at retrieval time — the model does all query-side work); `y_c`
is the candidate's report-derived CheXpert-14 label with −1 excluded both
ways (report labels live only on the reference side: sidecar `gt_labels` /
`gt_valid`, added by `scripts/augment_sidecar_labels.py` from the manifest
train labels + the adapted cal/eval npz `gt` — same label-free
store invariant as before). Motivation: single-argmax filtering throws away
13 of 14 predicted dimensions, and argmax is unstable on soft queries
(observed conf 0.12–0.24 in smoke queries).

Acceptance gate (`scripts/eval_rerank_gate.py`; 500 confident eval queries
with ≥1 certain positive label; exact emulation of the two-stage query):

| arm | label-Jaccard@8 | cosine@8 | posterior-agree@8 |
|---|---|---|---|
| old pathology | 0.3551 | 0.6085 | 0.735 |
| **new pathology** | **0.4113** | 0.5914 | 0.987 |
| old plain | 0.3013 | 0.6341 | 0.636 |
| **new plain** | **0.3922** | 0.6141 | 0.977 |

Neighbors carry +16% (pathology) / +30% (plain) more report-label agreement
at a ~0.02 cosine cost — no similarity collapse. Rerank is **off** in
`correct`/`normal` modes (their GT-flag pools are already exact predicates;
blending them would re-bias toward pathology-populated rows) and in the
no-finding fallback lanes. CLI: `--lam` (0 disables); the app fixes λ=0.3.

*(2026-09-01: the `pathology` mode itself was **retired** after the flag/contrast
lanes landed — it was the weakest lane, and its failure mode is exactly what
§2.6 documents: on the demo's confident-error film it keyed on the spurious
Hernia argmax and returned wrong-pathology references. The gate table above is
kept as the historical before/after evidence; the mode was removed from the app
radio and the CLI. Its one unique behavior — the no-finding fallback to normal
references — lives on in the dedicated `normal` mode.)*

### 2.6 Flag-keyed lane (built 2026-09-01)

**The Agent-1 → Agent-2 coupling made explicit.** Motivating case (app demo):
a confident-ERROR film where Region B fires on **Cardiomegaly** (the confident
false negative) while the pooled argmax lands on Hernia at p=0.394 — `pathology`
mode keyed on that *spurious* argmax and returned 8 wrong-Hernia references:
Agent 1 flagged the right class and Agent 2 retrieved for the wrong one.

The `flag` mode keys retrieval on **Agent 1's Region-B flag** instead:

```
Region B fires on class X   (confident AND top-50% epistemic std — model
                             internals only: pooled posterior + member std)
    → candidates = library rows whose REPORT confirms X   (gt_labels[:, X]==1,
                             −1 excluded; report labels are reference-side)
    → ranked by visual cosine (+ agreement rerank on the remaining mass)
```

Deployment-computable with **no query GT anywhere**: the key comes from the
model's own flags (the flagged class with the highest calibrated confidence
when several fire); the candidate filter reads reference-side report labels
from the sidecar. If Region B doesn't fire (or the label sidecar is absent)
the lane degrades gracefully to unfiltered nearest neighbors with an explicit
`flag fallback -> plain` lane note.

Acceptance gate (`scripts/eval_rerank_gate.py`, flag arm; 2,000 Region-B-firing
eval queries; flag-class report-precision@8):

| arm | flag-precision@8 | cosine@8 | model-detect@8 | mean pool |
|---|---|---|---|---|
| status quo (argmax pathology bucket) | **0.0142** | 0.6285 | 0.0018 | 28,954 |
| flag-keyed (report-confirmed pool) | **1.0000** | 0.4908 | 0.0185 | 14,527 |
| flag-keyed + rerank | **1.0000** | 0.4808 | 0.0228 | 14,527 |
| contrast (report-confirmed X **and** model-read X) | **1.0000** | 0.3285 | **1.0000** | 1,494 |

The argmax bucket returns report-confirmed-flagged-class neighbors only **1.4%**
of the time; the flag lane is definitionally 100% with no pool starvation
(~14.5k candidates mean). The cosine drop (0.63 → 0.49) is the expected cost of
keying on a specific pathology rather than raw visual proximity — precisely the
trade the lane exists to make. The `model-detect@8` column makes the lanes'
characters legible: status-quo neighbors are 0.2% model-detected on the flagged
class, flag-lane 1.9–2.3%, contrast 100% by construction. On the demo's
confident-ERROR case the neighborhood flips from spurious-argmax cases to
report-confirmed-Cardiomegaly references (each row annotated
`report_confirms = Cardiomegaly` in the UI). CLI:
`--mode flag --flag-class <pathology>`.

**Neighbor-vote score: display lane, not a ranking signal (null result,
2026-09-01).** The natural "Agent 2 votes" extension — vote@8 = share of the
top-8 *unfiltered* neighbors whose report confirms the flagged class, then
`AUROC(1−vote)` for ranking confident-wrong rows among Region-B-firing eval
queries — was measured in the same gate and **does not work**: AUROC(1−vote)
= **0.453** (bootstrap CI95 0.336–0.527), with the epistemic baselines equally
flat among firing rows (AUROC(−max-std) = 0.505, AUROC(1−conf) = 0.431). The
direction is in fact *reversed*: confident-wrong flagged films have a *higher*
mean vote than confident-correct ones (0.062 vs 0.012) — when the model is
confidently wrong on class X, visually similar films' reports tend to confirm
X *more* often, so low vote does not indicate error. Two honest readings: (a)
the score is severely underpowered (14 confident-error rows among 2,000
firing queries — CI spans 0.34–0.53), and (b) more substantively, *within*
the already-flagged (confident + top-std) population the residual variation in
std, confidence, and neighbor agreement carries essentially no additional
error signal. The vote therefore stays where it earns its keep: as the
`report_confirms` evidence display in the flag/contrast lanes, not as a
gate or score.

**Neighbor display in flag mode (app).** Because the flagged pool is full of
multilabel films, a single argmax per neighbor misrepresents both sides of the
comparison. In flag mode the UI therefore shows, per neighbor: the full
**predicted set** (every class above its Youden threshold, dominant first —
the model's actual read) in `pred_pathology`, and the full **report-positive
set** (flagged class first) in `report_confirms`. cal/eval predicted sets come
from the local npz; train predicted sets are computed lazily with the same
3-member 224 forward that annotated the sidecar (xrv_nih + convnextv2 +
raddino over the local fp16 train cache), memoized per row (~1 s first time
per train neighbor, free thereafter). Other modes keep the single dominant
argmax display.

### 2.7 Auto mode and the contrast lane (built 2026-09-01)

**`auto` (app default).** The radio's default resolves Agent-1's verdict into
the right Agent-2 lane with no user action:

```
eff_mode = "flag"     if Region B fires on class X   → evidence: reports that confirm X
          "correct"   otherwise                      → trust:    similar correct cases
```

The caption states the resolution (`mode **auto -> flag**` / `**auto ->
correct**`) so the coupling stays visible. Manual modes remain for targeted
review.

**`contrast` — the other half of the flag review.** The `flag` lane shows
"the model should have called X here" (report-confirmed-X references). The
`contrast` lane shows the complementary evidence: *"the model DID call X
somewhere — here is what that looks like."* Candidates = library rows where
**the model itself reads X-positive** (pooled mean ≥ X's Youden threshold,
per-class decision from the stored arrays) **AND the report confirms X**
(`dec[:, X] & gt_labels[:, X]==1 & gt_valid[:, X]`). X = the flagged class
when Region B fires, else the query's argmax; ranked by cosine (+ agreement
rerank). Pool = cal+eval only — train rows have no stored per-class probs
(their decisions would require a fresh 3-member forward per candidate, and
adapted members memorized train labels anyway). Contrast and flag together
give the reviewer both sides of a Region-B flag: what the finding looks like
where it was missed, and what the model's positive read looks like when it
agrees with the report.

## 3. Implementation

| File | Role |
|---|---|
| `scripts/build_retrieval_index.py` | loads cal arrays → L2-normalize → HNSW build → **label-free store** (`store.npz`) + prediction sidecar (`sidecar.parquet`) → **recall validation** (N eval queries, HNSW vs exact) → `index_meta.json` |
| `scripts/augment_sidecar_labels.py` | joins report-derived label vectors (`gt_labels`/`gt_valid`) into a full-library sidecar for the label-agreement rerank |
| `scripts/eval_rerank_gate.py` | before/after acceptance gate for the label-agreement rerank (label-Jaccard / cosine / posterior-agreement @K over confident eval queries) + the flag-keyed arm (flag-class report-precision@8 over Region-B-firing queries) |
| `scripts/query_retrieval.py` | loads index + sidecar, embeds the query from the eval/cal arrays, applies the mode filter inline, reranks plain/flag/contrast by report-label agreement, returns top-k with sim / pred-pathology / confidence / correct flag |
| `tests/test_retrieval.py` | smoke tests on synthetic arrays: label-free store contents, no-gt leakage, self-exclusion, k-respect, correct-mode filter semantics, sidecar correctness math, recall ≥ 0.95 |

Sidecar schema: `image_id, pred_pathology, pred_conf, correct, uncertain`
where `correct` = *pooled-mean argmax prediction equals the reference label*
(same pooled-mean rule as `eval_baselines.py`; uncertain = label −1,
excluded from the correct filter). Prediction metadata is derived from the
ensemble's own probabilities — this is Agent 1's output reused, not ground
truth in the retrieval path.

## 4. Usage

```bash
# Build (default: RAD-DINO feats, cal library, recall-validated)
python scripts/build_retrieval_index.py --arr-dir runs/eval_arrays \
    --source rex --embed raddino --out runs/retrieval/rex_raddino

# "Show me similar images where the model was correct."
python scripts/query_retrieval.py --index-dir runs/retrieval/rex_raddino \
    --id <eval_image_id> --mode correct --k 8

# Agent-1-coupled: "Region B flagged class X -- show references whose
# reports confirm X." (X = the flagged class; pass it explicitly here,
# the app derives it from Agent-1's flags.)
python scripts/query_retrieval.py --index-dir runs/retrieval/rex_raddino \
    --id <eval_image_id> --mode flag --flag-class Cardiomegaly --k 8

# The other half of the flag review: references the model read positive
# on X AND whose report confirms X.
python scripts/query_retrieval.py --index-dir runs/retrieval/rex_raddino \
    --id <eval_image_id> --mode contrast --flag-class Cardiomegaly --k 8

# Ground-truth validation path (exact cosine over filtered candidates)
python scripts/query_retrieval.py ... --search exact
```

`--id` may be any image in the eval split by default (`--split cal` to query
from the library itself; self-exclusion still enforced).

## 5. Validation on ReX (2026-08-29)

| Check | Result |
|---|---|
| Library size / dim | 8,064 reference vectors (3 indexes: `rex_raddino` 1536-d, `rex_fs_resnet18` 512-d, `rex_fs_vit_tiny` 192-d; cosine) |
| HNSW recall@10 vs exact | **1.0000** in all three indexes (200 eval queries each) |
| Model-consistency | each `rex_fs_*` index carries its own sidecar — correct = 2,577 (resnet18) / 2,384 (vit_tiny) / 2,343 (pretrained) of 8,064; `--source`/`--embed` mismatch rejected at query time |
| Space spread (200 random pairs vs NN) | raddino 0.081→0.582 (spread 0.50); fs_resnet18 0.663→0.969 (0.31); fs_vit_tiny 0.708→0.963 (0.26) — from-scratch spaces visibly collapsed, consistent with Phase-6 |
| `--mode correct` demo | correct-Effusion query (conf 0.795) → raddino: top-5 all correct Effusion cases sims 0.35–0.57; fs_resnet18: top-5 correct, sims 0.97+ (incl. one correct-Cardiomegaly case) |
| Confidently-wrong query | wrong-Effusion prediction (conf 0.668) → top-8 = its most-similar *correctly-predicted* Effusion cases (the Agent-1→Agent-2 contrast set for review) |
| hnsw vs exact agreement | identical top-8 on both demo queries |
| Tests | `pytest tests/test_retrieval.py` → 4/4 pass |

Outputs under `runs/retrieval/{rex_raddino, rex_fs_resnet18, rex_fs_vit_tiny}/`
each: `index.bin`, `store.npz`, `sidecar.parquet`, `index_meta.json` (records
recall, index params, and the label-free guarantee).

## 6. Limitations and next steps

1. **Embedding ablation pending** — BiomedCLIP feature pass + head-to-head
   retrieval quality (P@1 / neighbor agreement) vs RAD-DINO. From-scratch
   seed-0 spaces available as appearance baselines.
2. **Deployment GT-freeness & retrieval-quality evaluation** — the deployed
   path needs no ground truth on the query side: the query contributes an
   embedding + Agent-1's prediction; labels touch only offline library
   curation. This doc therefore makes no clinical-relevance claim about
   neighbor quality — rankings are validated against exact search, not
   against expert relevance. If such a claim is ever wanted, the available
   protocol would follow RadIR-style report-mined relevance; **a VLM path
   (e.g. ChestX-Reasoner) can serve this GT-free as well**: generate a
   query-side report at deployment time and compare findings against
   neighbor cases.
3. **Correct-flag granularity** — correctness is defined at the *predicted*
   pathology only (the case the clinician is looking at). A per-pathology
   filter variant ("correct *for pathology X*") is a small extension of the
   sidecar.
4. **turbovec / quantization** — deliberate no-op at the full-library scale
   too (129K vectors, RAM-resident, sub-ms queries); revisit only at ~10⁶+.
5. **Image thumbnails for review** — the CLI returns ids; a follow-on viewer
   can join `image_path` from `data/rex_manifest.parquet` to render cases.
6. **VLM integration path (proposed, not built)** — at deployment the query
   has no report. A reasoning VLM (ChestX-Reasoner-7B, open-sourced,
   process-supervised from radiology reports) could (a) generate a query-side
   report enabling findings-level retrieval filters beyond the argmax
   pathology, and (b) act as an independent adjudicator of Agent 1's
   confident-wrong flags, reading image + prediction + retrieved correct
   cases (CBIR-RAG pattern). Scope decision for the report; any VLM output
   stays decision support inside a clinician review loop, never automated.
   **Agent 3 (built 2026-09-01) inherits this framing one level down**: it is
   the no-pixel, text-only realization — it explains flags from the structured
   evidence pack (including the `neighbors.*` block this system supplies) under
   a cite-everything constraint + mechanical audit, and the pixel-reading VLM
   adjudicator remains proposed-not-built (see `docs/agent3_explanation.md`).

## References

*(RadIR and ChestX-Reasoner are referenced as optional evaluation /
deployment-extension protocols only — neither is part of the built system.)*

- Malkov & Yashunin, *Efficient and robust ANN search using HNSW graphs* (TPAMI) — the HNSW method hnswlib implements.
- Caminal et al., *Filtered Vector Search: State-of-the-art and Research Opportunities*, VLDB 2025 — pre/post/inline filtering semantics; basis for our inline-filter design.
- Denner et al., *Leveraging foundation models for content-based image retrieval in radiology*, Comput. Biol. Med. 2025 — foundation-model CXR CBIR benchmark; BiomedCLIP/RAD-DINO selection rationale.
- TurboQuant (arXiv:2504.19874, ICLR 2026) / turbovec — quantized-search lane, future scale.
- RadIR, MICCAI 2025 — report-mined relevance-supervised CBIR evaluation; referenced only as the optional protocol behind next-step 2, not part of the deployed path.
- ChestX-Reasoner (arXiv:2504.20930, Commun. Med. 2026) — reasoning VLM candidate for the deployment path (next-step 6).