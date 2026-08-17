# Design spec: Neural contribution and phantom agreement in language-model-assisted P300 spelling

Date: 2026-07-26
Status: locked for implementation
Project: `study_bci_llm_authorship`

## Working title

Neural contribution and phantom agreement during language-model-assisted P300 spelling in
amyotrophic lateral sclerosis: a retrospective re-decoding study of real online sessions.

## The question

When a language model supplies the prior for a brain-computer interface speller, part of every
emitted message is determined by the model rather than by the user's brain. Reported accuracy and
information transfer rate cannot separate the two. This study measures the separation directly, per
selection, in real online sessions recorded from people with amyotrophic lateral sclerosis.

## Why this cell is empty

- Speier, Arnold and Pouratian (PLoS One 2013;8(10):e78432) established that language priors inflate
  reported communication rate and proposed a mutual-information correction. That metric is
  aggregate: it yields a single information rate over an output string. It performs no per-selection
  attribution and identifies no individual case where the output was correct without neural support.
  The authors stated that the intermediate per-selection data needed for a fuller treatment were not
  available in published studies.
- Two conditions have since changed. The intermediate data now exist publicly, because BigP3BCI
  releases raw EEG, per-symbol stimulus channels, and feedback streams from online sessions. And the
  prior is no longer an n-gram; it is a language model capable of supplying most of a message.
- Recent language-model speller work reports only speed and accuracy. The nearest prior claims
  near-optimal performance and concludes that the bottleneck has moved to neural decoding
  (bioRxiv 2025.10.28.685216). That conclusion is the premise this study interrogates: if the model
  supplies the message, user determination is what is being spent.
- The augmentative and alternative communication literature names authorship and agency loss
  repeatedly and qualitatively (Valencia et al. CHI 2023; CHI 2026 autoethnographic account of
  ultra-personalized AI-powered AAC) without quantifying it.
- An information-theoretic framework for human contribution in AI-assisted generation exists
  (arXiv 2408.14792) and has never been applied to a brain-computer interface.

Positioning: this is the per-selection, language-model-era successor to Speier 2013, executed on
real ALS online sessions.

## Separation from companion studies

- `study_bci_llm_intent_drift` measures **errors of meaning** introduced when a language model
  post-edits a noisy decode. This study measures **provenance of the emitted selection, including
  when it is correct**. A selection can be fully faithful and almost entirely model-determined; that
  case is invisible to the drift construct and is the core object here.
- `study_bigp3_als_calibration` uses the same archive to ask whether calibration EEG predicts
  session accuracy. It is decoder-side and outcome-side; it contains no language model. Its
  classifier and trial-reconstruction code are reused here as infrastructure.
- Both will be disclosed to the editor as companion submissions from the same public archive.

## Data

- Source: BigP3BCI v1.0.0, already on disk at
  `study_bigp3_als_calibration/data/source_cache/bigP3BCI-data`. No download required.
- Cohort: ALS source studies B, F, L, N; shared 16-channel montage; Train and Test phases.
- Unit of analysis: an eligible online feedback selection. The companion recon establishes
  approximately 3,318 eligible selections across 113 participant-sessions and 194
  participant-session-condition records.
- Ground truth: `CurrentTarget` gives the intended symbol for every trial. Because the task is
  copy-spelling, the ordered sequence of intended symbols within a file recovers the intended
  phrase. True intent is therefore known, not simulated.
- Per-symbol flash channels (`A_1_1` through `9_6_6`) give the flash membership of all 36 grid
  symbols directly, which supports checkerboard as well as row-column paradigms.

## Measures

1. **Neural posterior** `P_neural(symbol | EEG)` for each selection. A per-participant P300
   classifier is trained on Train-phase epochs only, applied to Test-phase flashes, and flash scores
   are accumulated into a 36-way posterior using the per-symbol flash channels. No Test data inform
   training.
2. **Language-model prior** `P_LM(symbol | preceding intended context)` across a capability ladder:
   uniform (null), character n-gram, GPT-2, and modern open models (Llama, Qwen, Gemma) scored for
   log-probabilities on the HMS O2 cluster. Scoring only, no generation.
3. **Fused posterior** proportional to `P_neural^beta * P_LM`, the standard Bayesian speller fusion.
   `beta` is fixed at the pre-specified value with sensitivity across a grid.

## Co-primary outcomes

The user has chosen a co-primary structure. Multiplicity is handled by a pre-registered alpha split
of 0.025 per co-primary with hierarchical gatekeeping: secondary aims are tested only if at least
one co-primary meets its threshold.

- **Co-primary 1, neural contribution fraction (NCF).** Per selection, the share of the decisive
  log-posterior odds attributable to the neural term rather than the prior term, reported as a
  fraction in [0, 1] and as bits. Aggregated per participant, then pooled.
- **Co-primary 2, phantom agreement rate.** The proportion of emitted selections where the fused
  output equals the intended symbol while the neural posterior alone would not have selected it.
  Correct output, absent neural support.

## Secondary outcomes

- **Prior capture error.** Fused output is wrong while the neural posterior alone was right: the
  model broke a selection the brain had correct. This is the harm counterpart and the explicit link
  to the drift study without duplicating it.
- **Neural override.** Fused output right, neural right, prior wrong.
- Moderators: language-model capability tier, participant decoder quality (the companion's
  calibration discriminability score), ALSFRS-R where recorded, within-phrase position (prior
  strength grows with accumulated context), paradigm and condition, source study.

## Analysis plan

- Effect sizes with 95% confidence intervals throughout. Confidence intervals use participant-cluster
  bootstrap with 2,000 deterministic replicates, matching the companion's resampling scheme.
- Co-primaries tested at alpha 0.025 each with the gatekeeping rule above. Benjamini-Hochberg
  correction across the secondary family.
- Mixed-effects logistic models with participant random effects for the binary outcomes; if the
  fit fails to converge, fall back to cluster-robust logistic regression and report the deviation
  explicitly rather than silently.
- Continuous NCF modelled with participant random effects; report marginal means by language-model
  tier with contrasts against the n-gram reference.
- Pre-specified sensitivity analyses: fusion exponent grid, classifier family (regularized logistic
  and shrinkage LDA), exclusion of low-accuracy sessions, and per-source-study leave-one-out.

## Threats and mitigations

| Threat | Mitigation |
|---|---|
| Offline re-decoding is not online behaviour: no closed-loop adaptation, no error potentials, the user cannot correct in the loop | Stated as the principal limitation. All claims are about the decision rule applied to real neural evidence, not about clinical performance. |
| Copy-spelling phrases may be unusually predictable, inflating the prior | Report stratified by phrase predictability; include the least predictable stratum as a pre-specified subgroup. |
| The fusion rule itself determines attribution | Fusion exponent sensitivity grid; report attribution under each. |
| Classifier quality varies by participant and confounds attribution | Adjust for the companion calibration score; report by decoder-quality tertile. |
| Unmeasured severity confounding | Not applicable to the primary contribution, which is measurement rather than causal effect estimation. No causal language anywhere. |
| Pooling ALS studies with differing matrix sizes and electrodes | Source-study stratification and leave-one-study-out sensitivity. |

## Scope guardrails

- One paper. No closed-loop simulation, no new decoder architecture, no generation experiments.
- Descriptive and measurement language only. Never "caused", "led to", or "improves outcomes".
- Every reference verified by DOI against Crossref before it ships.
- Local git checkpoints only. Nothing pushed or submitted.
- Author list, affiliations, funding, competing interests, ethics determination, registration and
  repository URL remain visible placeholders for the human authors.

## Target venue

Decided at the journal-fitting phase after results exist. Leading candidates: npj Digital Medicine
(measurement plus agency framing), Journal of Neural Engineering (methodological home, but the
companion calibration paper is already targeted there), Journal of Medical Internet Research.
