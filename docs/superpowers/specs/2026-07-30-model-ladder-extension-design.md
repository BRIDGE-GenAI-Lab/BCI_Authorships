# Design spec: Cross-family model-ladder and classical n-gram extension

Date: 2026-07-30
Status: locked for implementation
Project: `study_bci_llm_authorship`
Extends: `docs/superpowers/specs/2026-07-26-bci-llm-authorship-design.md`

## Why

The original design named "modern open models (Llama, Qwen, Gemma)" for the prior ladder, but only
the Qwen dynasty (plus GPT-2) was ever scored. The submission-ready manuscript's strongest secondary
claim — "model capability did not change the picture... phantom agreement stayed between 7.2% and
8.6%... unrelated to model size" — is currently demonstrated within a single model family. A reviewer
can reasonably ask whether that flatness is a property of language models in general or an artefact
of the Qwen lineage specifically. This extension adds cross-family neural priors and a small classical
n-gram sub-ladder to close that gap, completing the original design's intent rather than expanding
scope.

## What this is not

This is not a re-derivation of the co-primary outcomes. Qwen2.5-32B remains the primary prior for the
headline neural contribution fraction and phantom agreement point estimates, and the existing `ngram5`
(order-5, add-k, Brown corpus) remains the reference for co-primary contrasts. The new priors feed only
the existing "model capability tier" secondary/robustness analysis, broadening its claim from
"capability" to "capability and architecture family."

## New neural priors

All scored as base (non-instruct) checkpoints, consistent with every existing ladder entry, on the HMS
O2 cluster (`gpu_sun`, reusing the `idrift_env` and `/n/scratch/users/a/alg7274/authorship/` payload
convention established for the Qwen cluster tier).

| Name | Family | Parameters | Architecture | Comparable existing rung |
|---|---|---|---|---|
| `meta-llama/Llama-3.1-8B` | Llama | 8B | dense | Qwen2.5-14B |
| `meta-llama/Llama-3.1-70B` | Llama | 70B | dense | new largest rung |
| `google/gemma-2-9b` | Gemma | 9B | dense | Qwen2.5-14B |
| `google/gemma-2-27b` | Gemma | 27B | dense | Qwen2.5-32B |
| `mistralai/Mistral-7B-v0.3` | Mistral | 7B | dense | Qwen2.5-14B |
| `mistralai/Mixtral-8x7B-v0.1` | Mistral | 47B total / 13B active | sparse MoE | Qwen2.5-32B, plus a within-family dense-vs-MoE contrast against Mistral-7B |
| `deepseek-ai/DeepSeek-V2-Lite` | DeepSeek | 16B total / 2.4B active | sparse MoE | distinct training lineage, medium rung |

**Llama-3.3 substitution decision.** Meta released Llama-3.3 only as a 70B Instruct checkpoint; no
non-instruct base weights exist at that size. Every existing ladder entry (Qwen2.5, GPT-2) is a base
checkpoint, so mixing in an instruct-tuned model would confound "raw next-character prior strength"
with chat-alignment tuning. `Llama-3.1-70B` (base) is used instead. If this is ever revisited, treat
any instruct-only substitution as a labeled, explicit deviation — never a silent swap.

## New classical priors

Local, small compute, no O2 required. Existing `ngram5` (order-5, add-k smoothing, Brown corpus,
hand-rolled backoff in `authorship/priors.py::CharNgramPrior`) is untouched and remains the reference.

- **`ngram5_kn`** — order-5, interpolated Kneser-Ney smoothing, same Brown corpus. Uses
  `nltk.lm.models.KneserNeyInterpolated` (already-vetted library implementation, not hand-rolled) over
  a character-level vocabulary. Isolates the smoothing effect against the add-k reference.
- **`ngram5_wiki_kn`** — order-5, interpolated Kneser-Ney, fit on a WikiText-103 slice. Isolates the
  corpus effect (size + recency) against `ngram5_kn`, holding smoothing constant.
  - **Corpus size cap.** The full WikiText-103 corpus is ~103M tokens (~500M+ characters); a
    character-level order-5 count table over the full corpus is not memory-tractable in the existing
    dict-of-arrays design. The corpus is capped at a disclosed **20M characters** (still ~20x the
    Brown corpus and reflects contemporary web/encyclopedic text vs. Brown's 1961 balanced written
    corpus). This cap is stated explicitly in Methods, not silently applied.

Both new n-gram priors report through the same `prior(context) -> np.ndarray` interface as
`CharNgramPrior`, so no changes are required to `build_attribution.py` or downstream consumers beyond
registering the two new `LADDER` entries.

## Gating and compatibility checks (before trusting any new shard)

- Llama and Gemma are gated HF repos requiring license acceptance and an HF token on the O2 environment
  — a human step, not automatable; flagged explicitly rather than worked around.
- Before scoring the full 951 unique contexts for each new architecture, validate
  `TransformerPrior._exact_batched` against `TransformerPrior.exact_prior` on one context, mirroring the
  existing gpt2 O2-vs-local cross-validation already reported in eMethods S1.3. This catches
  architecture-specific issues early (e.g., Gemma-2's logit soft-capping, Mixtral/DeepSeek MoE routing)
  rather than after a full ladder run.
- Confirm every new checkpoint name resolves to a base (non-instruct/non-chat) repo before scoring.

## Compute plan (O2, `gpu_sun`, 4x L40S 48G per node)

- **Tier A (single GPU, fast — by analogy to the existing 14B run at 2m51s):** Llama-3.1-8B,
  Gemma-2-9b, Mistral-7B-v0.3, DeepSeek-V2-Lite.
- **Tier B (multi-GPU, slower — by analogy to the existing 32B/27B runs at 6-11 min):** Gemma-2-27b
  (comparable to the existing 32B run), Mixtral-8x7B (~94GB fp16, needs 2 GPUs), Llama-3.1-70B
  (~140GB fp16, needs a full 4-GPU node).
- Extend the existing `score_priors_o2.py` array job and `scripts/ingest_o2_priors.py` validation
  (context set / alphabet / normalization) with the new specs. No new scripts; same contract.

## Downstream re-analysis

1. `scripts/build_priors.py`: extend `LADDER` with the 9 new specs (2 local n-gram, 7 cluster).
2. `scripts/ingest_o2_priors.py`: ingest the 7 new O2 shards once scored.
3. `scripts/run_stats.py`: add a `family` covariate/contrast to the capability-tier moderator model
   (Qwen / Llama / Gemma / Mistral / DeepSeek / classical); re-run the Spearman scale check **within
   each family separately**, not only pooled — this is the exact check that would have caught the
   original truncated-ladder artefact sooner, and guards against a new family reintroducing it.
4. `scripts/make_figures.py`: redesign the ladder figure to group/color ~18 priors by family rather
   than a single flat axis; legibility is a first-class requirement of this step, not an afterthought.
5. `scripts/assemble_manuscript.py` + manuscript prose: update every hardcoded ladder count/range
   ("eight priors", "a character 5-gram to a 32.5-billion-parameter model", eTable 9); add one
   discussion sentence on cross-family stability; add 5 new references (Llama 3, Gemma 2, Mistral 7B,
   Mixtral, DeepSeek-V2), verified via the `ama-citation-zotero` skill before shipping.
6. Word/reference budget check against npj Digital Medicine limits (current packet: body 3,976 words,
   refs ≤60, 18 used) — new citations plus discussion additions must stay within cap; trim elsewhere
   in Discussion if needed rather than requesting a new cap.
7. Re-run the full test suite (47 tests) and add tests for the `nltk.lm`-backed n-gram wrapper and any
   architecture-specific quirks surfaced during the compatibility checks above.

## Threats and mitigations

| Threat | Mitigation |
|---|---|
| Gated HF repos block O2 download | Human step: accept licenses + HF token before running; flag explicitly, never silently substitute a different checkpoint |
| MoE/70B models don't fit the existing single-context validation assumption | Validate one context on O2 before the full run; document any `device_map` spread across GPUs |
| WikiText-103 full corpus is impractically large for the character-level count table | Fixed, disclosed 20M-character cap; documented as a stated corpus-size decision, not silent truncation |
| Ladder growth (9 to 18 priors) breaks existing figure legibility | Group-by-family redesign is a first-class step (item 4 above), not squeezed into the existing figure |
| Re-analysis silently shifts already-reported co-primary point estimates | Primary prior (Qwen2.5-32B) and n-gram reference (`ngram5`) are untouched; verify headline NCF/phantom-agreement CIs are byte-identical before vs. after this extension |
| New family reintroduces a truncation-style spurious trend within itself | Per-family Spearman rho check (item 3 above), not just a pooled check |

## Scope guardrails

- No change to the co-primary definitions, the fusion rule, the decoder, or the primary prior.
- No change to the headline NCF / phantom-agreement point estimates — this extends the robustness
  ladder only.
- Local git commits only; nothing pushed or submitted, per the standing convention for this study.
- Every new reference verified by DOI against Crossref before it ships.
