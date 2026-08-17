# Neural Contribution and Phantom Agreement Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Execution is inline in
> this session.

**Goal:** Measure, per online P300 selection in real ALS sessions, how much of the emitted symbol was
determined by neural evidence versus a language-model prior, and how often a correct output had no
neural support.

**Architecture:** Reuse the companion package `bigp3_als` for EDF access, trial reconstruction and
P300 feature extraction. Add a new package `authorship` that (1) turns Test-phase flashes into a
36-way neural posterior per selection, (2) computes a language-model prior over the same 36 symbols
for a ladder of models, (3) fuses them and derives attribution metrics, (4) runs the statistics.

**Tech Stack:** Python 3.11, mne, numpy, pandas, scikit-learn, scipy, statsmodels, pyarrow,
matplotlib, transformers (GPT-2 local; open models scored on HMS O2).

## Global Constraints

- Source data is read-only from `../study_bigp3_als_calibration/data/source_cache/bigP3BCI-data`.
  Never write into the companion project.
- No Test-phase data may inform classifier training. Train phase only.
- Alphabet is exactly the 36 BigP3BCI grid symbols recovered from EDF channel names.
- Co-primaries tested at alpha 0.025 each with hierarchical gatekeeping; Benjamini-Hochberg across
  the secondary family.
- All confidence intervals from 2,000 deterministic participant-cluster bootstrap replicates.
- No causal language in any output. Measurement framing only.
- Every intermediate stage checkpoints to Parquet under `output/intermediate/`.

---

### Task 1: Alphabet, grid map, and Test-phase trial frame

**Files:**
- Create: `authorship/grid.py`
- Create: `tests/test_grid.py`

**Interfaces:**
- Produces: `SYMBOLS: tuple[str, ...]` (36 display symbols in channel order),
  `FLASH_CHANNELS: tuple[str, ...]`, `parse_symbol_channel(name: str) -> tuple[str, int, int]`,
  `target_code_to_symbol(code: int) -> str`.

- [ ] **Step 1: Write the failing test**

```python
from authorship.grid import parse_symbol_channel, SYMBOLS, target_code_to_symbol

def test_parse_symbol_channel():
    assert parse_symbol_channel("A_1_1") == ("A", 1, 1)
    assert parse_symbol_channel("Sp_5_3") == (" ", 5, 3)
    assert parse_symbol_channel("9_6_6") == ("9", 6, 6)

def test_alphabet_is_36_unique():
    assert len(SYMBOLS) == 36
    assert len(set(SYMBOLS)) == 36

def test_target_code_maps_to_symbol():
    assert target_code_to_symbol(1) == "A"
```

- [ ] **Step 2: Run it and confirm it fails** (`pytest tests/test_grid.py -v`).

- [ ] **Step 3: Implement `grid.py`**

```python
"""BigP3BCI 6x6 grid alphabet recovered from EDF per-symbol flash channels."""
from __future__ import annotations

FLASH_CHANNELS = (
    "A_1_1","B_1_2","C_1_3","D_1_4","E_1_5","F_1_6",
    "G_2_1","H_2_2","I_2_3","J_2_4","K_2_5","L_2_6",
    "M_3_1","N_3_2","O_3_3","P_3_4","Q_3_5","R_3_6",
    "S_4_1","T_4_2","U_4_3","V_4_4","W_4_5","X_4_6",
    "Y_5_1","Z_5_2","Sp_5_3","1_5_4","2_5_5","3_5_6",
    "4_6_1","5_6_2","6_6_3","7_6_4","8_6_5","9_6_6",
)

def parse_symbol_channel(name: str) -> tuple[str, int, int]:
    label, row, column = name.rsplit("_", 2)
    return (" " if label == "Sp" else label, int(row), int(column))

SYMBOLS = tuple(parse_symbol_channel(name)[0] for name in FLASH_CHANNELS)

def target_code_to_symbol(code: int) -> str:
    if not 1 <= code <= 36:
        raise ValueError(f"target code out of range: {code}")
    return SYMBOLS[code - 1]
```

- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Validate the code convention against real data.** Load one Test EDF, take trials
  where `correct` is True, and confirm `target_code_to_symbol(target)` reproduces a sensible phrase
  when concatenated in trial order. If the concatenation is gibberish, the code base is not 1-indexed
  channel order; fix the mapping before proceeding. Record the verdict in
  `docs/validation_grid.md`.
- [ ] **Step 6: Commit.**

---

### Task 2: Per-participant P300 classifier trained on Train phase only

**Files:**
- Create: `authorship/decoder.py`
- Create: `tests/test_decoder.py`

**Interfaces:**
- Consumes: `bigp3_als.features` bandpass and epoch helpers.
- Produces: `fit_participant_classifier(train_paths: list[Path]) -> ParticipantDecoder` with method
  `score_epochs(epochs: np.ndarray) -> np.ndarray` returning one calibrated P300 score per epoch,
  and attribute `train_auc: float`.

- [ ] **Step 1: Write the failing test** using a synthetic epoch array where target epochs carry a
  positive deflection at the expected latency and non-target epochs do not; assert `train_auc > 0.9`
  and that `score_epochs` returns finite scores of the right length.
- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement.** Bandpass 0.5-30 Hz, epoch -200 to 800 ms, baseline correct, drop epochs
  exceeding 150 microvolts, temporally downsample, then
  `make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`. Standardization is required;
  unscaled logistic regression overflows on raw EEG features.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Commit.**

---

### Task 3: Test-phase flash epoching and 36-way neural posterior

**Files:**
- Create: `authorship/neural_posterior.py`
- Create: `tests/test_neural_posterior.py`

**Interfaces:**
- Produces: `build_selection_posteriors(edf_path, decoder) -> pd.DataFrame` with one row per
  eligible Test trial and columns `trial_number, target_symbol, selected_symbol, n_flashes,`
  `p_neural` (a length-36 float array), plus source identifiers.

**Why this is the error-prone task:** flash membership must come from the 36 per-symbol channels, not
from row/column codes, because checkerboard conditions flash arbitrary symbol subsets. Evidence
accumulates additively in log space over every flash in the trial's phase-2 window.

- [ ] **Step 1: Write the failing test.** Synthesize a stream where symbol index 7 is present in every
  flash group that receives a high score; assert `argmax(p_neural) == 7` and `p_neural.sum() == 1`.

```python
def test_posterior_concentrates_on_consistently_scored_symbol():
    posterior = accumulate_evidence(
        flash_membership=membership,   # (n_flashes, 36) 0/1
        flash_scores=scores,           # (n_flashes,) log-odds
    )
    assert posterior.shape == (36,)
    assert abs(posterior.sum() - 1.0) < 1e-9
    assert int(posterior.argmax()) == 7
```

- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement `accumulate_evidence` then the EDF-level builder.**

```python
def accumulate_evidence(flash_membership: np.ndarray, flash_scores: np.ndarray) -> np.ndarray:
    """Additive log-odds accumulation over flashes, normalized to a 36-way posterior."""
    log_evidence = flash_membership.T @ flash_scores          # (36,)
    log_evidence -= log_evidence.max()                        # numerical stability
    weights = np.exp(log_evidence)
    return weights / weights.sum()
```

  For the EDF builder: read the 16 EEG channels plus `PhaseInSequence`, `StimulusBegin`,
  `CurrentTarget` and the 36 flash channels. For each eligible trial from
  `bigp3_als.trials.reconstruct_online_trials`, take the contiguous phase-2 window preceding the
  feedback phase, find rising `StimulusBegin` edges inside it, epoch each, score with the
  participant decoder, and read flash membership as the 36 channel values at the flash onset sample.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Validate against reality.** The neural-posterior argmax must agree with the archive's
  recorded `SelectedTarget` far above chance. Report agreement; if it is near chance the epoching
  window or membership read is wrong. Record in `docs/validation_posterior.md`.
- [ ] **Step 6: Commit.**

---

### Task 4: Intended-phrase reconstruction and context strings

**Files:**
- Create: `authorship/context.py`
- Create: `tests/test_context.py`

**Interfaces:**
- Produces: `add_context(trials: pd.DataFrame) -> pd.DataFrame` adding `intended_phrase`,
  `context_prefix` (all intended symbols before this trial within the file), `position_in_phrase`.

**Decision:** context is built from **intended** prior symbols, not previously emitted ones. This
isolates attribution from error propagation, which is a separate phenomenon; the choice is
pre-specified and a sensitivity analysis using emitted context is run in Task 8.

- [ ] **Step 1: Write the failing test** asserting that for the third trial of a file whose intended
  symbols are `["C","A","T"]`, `context_prefix == "CA"` and `position_in_phrase == 2`.
- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement** with a per-file ordered cumulative concatenation.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Commit.**

---

### Task 5: Language-model prior ladder over the 36-symbol alphabet

**Files:**
- Create: `authorship/priors.py`
- Create: `tests/test_priors.py`

**Interfaces:**
- Produces: `uniform_prior() -> np.ndarray`, `NgramPrior(order: int).prior(context: str) -> np.ndarray`,
  `TransformerPrior(model_name: str).prior(context: str) -> np.ndarray`. All return length-36
  probability vectors aligned to `grid.SYMBOLS`.

**The subtle part:** transformer tokenizers do not align to characters. The next-symbol distribution
is obtained by one forward pass over the context, then marginalizing the next-token distribution onto
the first character of each token, restricted and renormalized to the 36-symbol alphabet
(case-folded, with `Sp` mapped from a leading space token).

- [ ] **Step 1: Write the failing tests.**

```python
def test_priors_are_valid_distributions():
    for prior in (uniform_prior(), NgramPrior(order=5).prior("TH"), TransformerPrior("gpt2").prior("TH")):
        assert prior.shape == (36,)
        assert abs(prior.sum() - 1.0) < 1e-6
        assert (prior >= 0).all()

def test_transformer_prior_is_context_sensitive():
    prior = TransformerPrior("gpt2").prior("THE CA")
    assert prior[grid.SYMBOLS.index("T")] > prior[grid.SYMBOLS.index("Q")]
```

- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement.** Character n-gram fitted on a public English corpus with add-k smoothing.
  Transformer prior via first-token-character marginalization as above.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Validate the marginalization.** On a random 200-context subsample, compare the
  marginalized prior against an exact 36-way full-continuation scoring (score `context + symbol` for
  all 36 symbols and normalize). Report Spearman correlation and mean absolute difference in
  `docs/validation_prior.md`. Proceed only if agreement is high; otherwise switch the pipeline to
  exact scoring and note the compute cost.
- [ ] **Step 6: Commit.**

---

### Task 6: Fusion and attribution metrics

**Files:**
- Create: `authorship/attribution.py`
- Create: `tests/test_attribution.py`

**Interfaces:**
- Produces: `fuse(p_neural, p_lm, beta=1.0) -> np.ndarray` and
  `attribution_row(p_neural, p_lm, fused, target_index) -> dict` returning
  `ncf, bits_neural, bits_prior, phantom_agreement, prior_capture, neural_override,`
  `fused_correct, neural_correct, prior_correct`.

- [ ] **Step 1: Write the failing tests**, including the three definitional cases.

```python
def test_phantom_agreement_when_prior_rescues_a_wrong_neural_argmax():
    p_neural = np.full(36, 0.01); p_neural[5] = 0.6          # neural says index 5
    p_lm = np.full(36, 0.001); p_lm[9] = 0.9                 # prior says index 9
    fused = fuse(p_neural, p_lm)
    row = attribution_row(p_neural, p_lm, fused, target_index=9)
    assert row["fused_correct"] and not row["neural_correct"]
    assert row["phantom_agreement"] is True
    assert row["prior_capture"] is False

def test_prior_capture_when_prior_breaks_a_correct_neural_argmax():
    p_neural = np.full(36, 0.01); p_neural[5] = 0.5
    p_lm = np.full(36, 0.001); p_lm[9] = 0.99
    fused = fuse(p_neural, p_lm)
    row = attribution_row(p_neural, p_lm, fused, target_index=5)
    assert row["neural_correct"] and not row["fused_correct"]
    assert row["prior_capture"] is True

def test_ncf_is_one_under_a_uniform_prior():
    row = attribution_row(p_neural, np.full(36, 1/36), fuse(p_neural, np.full(36, 1/36)), 5)
    assert abs(row["ncf"] - 1.0) < 1e-6
```

- [ ] **Step 2: Run and confirm failure.**
- [ ] **Step 3: Implement.** NCF is the neural share of the decisive log-evidence for the emitted
  symbol relative to the alphabet mean:
  `bits_neural = log2 p_neural[k] - mean_j log2 p_neural[j]`, likewise `bits_prior`, and
  `ncf = bits_neural / (bits_neural + bits_prior)` clipped to [0, 1] with both-terms-zero returning
  NaN. Guard against zero probabilities with a floor.
- [ ] **Step 4: Run tests, confirm pass.**
- [ ] **Step 5: Commit.**

---

### Task 7: Assemble the analysis frame

**Files:**
- Create: `scripts/build_analysis_frame.py`
- Output: `output/intermediate/selections.parquet`, `output/intermediate/attribution.parquet`

- [ ] **Step 1:** Build per-participant decoders from Train phase, then selection posteriors for every
  eligible Test trial across studies B, F, L, N. Checkpoint to `selections.parquet`.
- [ ] **Step 2:** Reconcile N against the companion recon (approximately 3,318 eligible selections,
  113 participant-sessions). Any discrepancy must be explained in
  `docs/validation_cohort.md`, not silently accepted.
- [ ] **Step 3:** Cross every selection with every prior in the ladder, fuse, and emit one attribution
  row per selection-by-prior pair to `attribution.parquet`.
- [ ] **Step 4:** Assert invariants: probabilities sum to 1, no leakage of Test data into training,
  every row has a known target symbol, phantom agreement and prior capture are mutually exclusive.
- [ ] **Step 5: Commit.**

---

### Task 8: Statistics

**Files:**
- Create: `scripts/run_stats.py`
- Output: `output/results_digest.json`, `output/stats_digest.json`

- [ ] **Step 1:** Co-primary 1, NCF: participant-level means by prior tier, pooled estimate with
  participant-cluster bootstrap CI, mixed-effects model with participant random intercepts and prior
  tier as fixed effect, tested at alpha 0.025.
- [ ] **Step 2:** Co-primary 2, phantom agreement rate: pooled proportion with cluster bootstrap CI
  and a mixed-effects logistic model, tested at alpha 0.025. Record the gatekeeping outcome.
- [ ] **Step 3:** Secondaries with Benjamini-Hochberg: prior capture rate, neural override rate,
  moderator models (decoder quality tertile, ALSFRS-R where present, position in phrase, condition,
  source study).
- [ ] **Step 4:** Pre-specified sensitivities: fusion exponent grid, shrinkage-LDA decoder, emitted
  rather than intended context, low-accuracy session exclusion, leave-one-study-out.
- [ ] **Step 5:** If any mixed model fails to converge, fall back to cluster-robust logistic
  regression and record the deviation verbatim in the digest under `deviations`.
- [ ] **Step 6: Commit.**

---

### Task 9: Figures

**Files:**
- Create: `scripts/make_figures.py`

- [ ] **Step 1:** Figure 1: neural contribution fraction across the language-model ladder, with
  participant-level points and cluster bootstrap intervals.
- [ ] **Step 2:** Figure 2: phantom agreement and prior capture against decoder quality, by model tier.
- [ ] **Step 3:** Figure 3: worked single-selection example showing neural posterior, prior, and fused
  posterior for one phantom-agreement case.
- [ ] **Step 4:** eFigures: fusion-exponent sensitivity, per-source-study estimates.
- [ ] **Step 5:** Apply the `scientific-figures` and `hochberg-figure-style` conventions, 600 DPI PDF
  plus PNG, TrueType fonts embedded. Commit.

---

### Task 10: Manuscript, supplement, references, cover letter, build

- [ ] **Step 1:** Draft with `clinical-methods`, `clinical-results`, `clinical-introduction`,
  `clinical-discussion`, `clinical-abstract` in that order.
- [ ] **Step 2:** Run `de-ai-writing` to zero tells.
- [ ] **Step 3:** Verify every reference by DOI against Crossref with `ama-citation-zotero`; emit
  `references.bib`.
- [ ] **Step 4:** Build the supplement with `clinical-supplement`; cover letter with
  `clinical-cover-letter`.
- [ ] **Step 5:** Select the journal, fetch its current instructions, fit the exact word, reference and
  display-item limits, renumber all callouts.
- [ ] **Step 6:** Build docx and PDF via `_pub_assets/build_pub.sh`; assemble the submission packet.
- [ ] **Step 7:** Final commit and packet inventory listing the human-only placeholders.

---

## Self-review

- Spec coverage: every spec measure maps to a task (neural posterior Task 3, prior ladder Task 5,
  fusion and both co-primaries Task 6 and 8, secondaries Task 8, sensitivities Task 8, threats
  addressed by the Task 3/5/7 validation gates).
- No placeholders: each task carries concrete paths, signatures and code for the error-prone parts.
- Type consistency: `p_neural` and `p_lm` are length-36 float arrays aligned to `grid.SYMBOLS`
  throughout; `attribution_row` returns the same key set used by Tasks 7, 8 and 9.
