# Design spec: Additive architecture-inclusive prior ladder

Date: 2026-08-01
Status: ready for review
Project: `study_bci_llm_authorship`
Extends: `docs/superpowers/specs/2026-07-30-model-ladder-extension-design.md`

## Objective

Strengthen the authorship-BCI prior ladder by adding scientifically justified causal-language-model
architectures identified through a documented online search. The extension is strictly additive:
every existing prior remains in the ladder, including the null prior, classical n-grams, Qwen primary
prior, and all previously scored neural families. No model is removed, replaced, or down-selected on
the basis of observed results.

The expanded ladder is a robustness analysis. Qwen/Qwen2.5-32B remains the locked primary neural prior,
and the existing primary estimates must remain unchanged after the extension.

## Reporting standard and scope

The study is not a clinical trial or a prospective prediction-model validation study. TRIPOD-LLM,
MI-CLAIM-GEN, Nature Medicine transparency guidance, and NEJM statistical-reporting guidance are used
as strict reporting and reproducibility standards where applicable. They do not justify deleting an
existing model; they require that model provenance, versioning, analysis roles, exclusions, missing
outputs, and computational details be visible and auditable.

## Prespecified architecture set

The current 24-prior ladder is preserved exactly. The following families are additive primary
architecture candidates:

| Family | Representative checkpoint rule | Scientific rationale |
|---|---|---|
| RWKV | Official public RWKV-7/6 base checkpoint selected before scoring | Attention-free recurrent/linear-attention sequence modeling |
| RecurrentGemma/Griffin | `google/recurrentgemma-2b` base checkpoint | Gated linear recurrences combined with local sliding-window attention |
| xLSTM | `NX-AI/xLSTM-7b` base checkpoint | Recurrent memory cells with modern LSTM-style scaling |
| StripedHyena | `togethercomputer/StripedHyena-Hessian-7B` base checkpoint | Hybrid attention plus gated Hyena convolutions |

The following families are screened as conditional candidates. They may be added only if their exact
checkpoint, tokenizer, provenance, base-model status, and causal likelihood interface pass the O2
preflight before full scoring:

| Family | Reason for conditional status |
|---|---|
| RetNet | Distinct Microsoft retentive architecture, but the available public checkpoint has incomplete model-card/provenance metadata |
| Gated DeltaNet | Distinct gated delta-rule linear-attention architecture, but the accessible checkpoint currently has incomplete metadata |

No conditional candidate is silently substituted with another checkpoint. If it fails preflight, it
remains in the screening ledger as excluded/failed and does not enter the inferential ladder.

## Online search and screening protocol

The search cutoff is 2026-08-01. The search record must preserve search queries, dates, source URLs,
paper identifiers, model-card URLs, repository URLs, checkpoint IDs, and the decision for every
candidate family found.

Search sources are limited to primary or authoritative materials: the architecture paper or technical
report, the original organization’s repository or documentation, and the checkpoint/model card. A
candidate is eligible for the primary ladder only when it satisfies all of the following:

1. It is a genuinely distinct causal sequence-model family, not merely a new tokenizer, quantization,
   parameter count, instruction-tuning recipe, or attention implementation.
2. It has an English/general-purpose pretrained base checkpoint suitable for raw next-token
   likelihood scoring.
3. The weights and tokenizer are publicly accessible under a documented license or access agreement.
4. The architecture and checkpoint provenance are documented well enough for independent reproduction.
5. It exposes a reproducible causal-language-model forward pass with finite next-token probabilities.
6. It can complete the O2 smoke test and full-context coverage gate without missing model-context rows.

The screening ledger must record the exact inclusion/exclusion reason, independent of performance. The
following candidates are screened but excluded from the primary architecture ladder:

- HyenaDNA and Evo2: autoregressive but trained on DNA rather than natural-language token sequences.
- Test-time-training models: inference changes the model weights/procedure and is not comparable to
  fixed-model likelihood scoring.
- Zamba, Jamba, and Falcon-Mamba: hybrid or Mamba-overlap candidates that do not add a sufficiently
  independent family to this primary expansion.
- BitNet, MoE, GQA, RoPE, and LongNet variants: important within-Transformer design changes, but not
  independent sequence-model families for the present architecture question.
- Instruction-tuned, chat, code-only, multilingual-only, inaccessible, or non-reproducible checkpoints
  when a suitable general base model is unavailable.

## Frozen model manifest

Before any new O2 scoring, create a versioned manifest containing, for every current and new model:

- prior name and family;
- checkpoint ID and immutable revision/commit;
- base versus instruction/chat status;
- parameter count and architecture description;
- tokenizer ID and revision;
- license/access status;
- source paper, official documentation, and model-card URLs;
- ladder role: primary, existing secondary, additive primary-family, conditional, or excluded;
- screening rationale;
- manifest hash.

The manifest is frozen before any new model result is read. The existing 24 entries are copied into the
manifest unchanged.

## O2-only execution plan

All model loading, GPU scoring, CPU aggregation, bootstrap, and postprocessing must run on HMS O2,
partition `gpu_sun`. Local CPU/GPU execution is prohibited for model scoring or heavy statistical
processing.

For each additive or conditional candidate:

1. Load the exact frozen checkpoint and tokenizer on O2.
2. Run the 31-context smoke test, checking model load, tokenizer behavior, finite probabilities,
   normalized 36-way priors, and a deterministic repeat.
3. Run the complete 951-context score using the same context set, scoring function, precision policy,
   and segmentation rules as the existing ladder.
4. Require 951/951 valid context rows. Do not impute or silently drop failed contexts.
5. Record SLURM job IDs, array element, node/GPU, software environment, model revision, warnings,
   elapsed time, exit code, and output digest.
6. Run ingestion, attribution, participant-cluster bootstrap, and statistics on O2 CPU resources.

A failed smoke or full run may be retried once with the identical manifest and configuration. Repeated
failure means exclusion with the complete log retained. No local fallback or result-driven substitution
is permitted.

## Statistical plan

- Preserve the current Qwen primary analysis and headline estimates.
- Add new models as secondary robustness rows and architecture-family sensitivity analyses.
- Report point estimates and 95% confidence intervals for every successfully completed model.
- Do not use model-specific P values to decide whether a model belongs in the ladder.
- If formal comparisons across the new candidates are presented, use a prespecified Holm correction and
  label them secondary/exploratory.
- Preserve the participant-cluster bootstrap design and fixed bootstrap seed.
- Report successful and failed model-context counts explicitly.
- Use scientifically meaningful precision and avoid excess decimal places.
- Add leave-one-family-out and parameter-size-matched sensitivity analyses when feasible.
- Interpret consistency across architectures as internal robustness, not independent external validation
  or evidence of clinical deployment performance.

## Manuscript and supplement changes

Add a Methods subsection titled “Architecture search and prior-model inclusion.” It must state that the
extension was additive, describe the search cutoff and evidence sources, define inclusion/exclusion
criteria, and distinguish the locked primary prior from secondary architecture robustness analyses.

Add supplementary tables for:

1. all screened candidate families and decisions;
2. checkpoint, tokenizer, parameter, license, and provenance metadata;
3. O2 smoke/full-run validation and failure logs;
4. architecture-family sensitivity and multiplicity handling.

Update model counts, family counts, ladder figures, results, discussion, references, data availability,
code availability, and compute-reproducibility statements without changing existing primary estimates.

## Acceptance criteria

- All existing 24 priors remain present and unchanged.
- The frozen manifest contains the four additive primary candidates and the two conditional candidates.
- Each included new model has an auditable primary source, checkpoint revision, tokenizer, and rationale.
- Every included new model passes the O2 smoke and 951/951 full-coverage gates.
- O2 logs and output digests are retained.
- No new model is selected or removed based on its observed authorship or drift result.
- Existing Qwen headline estimates are byte-identical before and after the extension.
- Tests cover model registration, manifest integrity, candidate screening, full-context validation, and
  additive preservation of the existing ladder.
- The assembled manuscript accurately reports the expanded counts and the primary/secondary roles.

## Non-goals

- No removal or replacement of current models.
- No change to the primary prior, co-primary outcomes, fusion rule, or decoder.
- No local heavy compute.
- No clinical deployment claim or external-validity claim without an independent dataset.
