# Manuscript source of truth

Every number in the manuscript, supplement, abstract and cover letter must come from this file.
Generated from `output/stats_digest.json`, `output/cohort_summary.json`, `output/figure_example.json`.
Locked 2026-07-26.

## Design

Retrospective secondary analysis of BigP3BCI v1.0.0. Online P300-speller sessions recorded from
people with amyotrophic lateral sclerosis were re-decoded offline. A per-session decoder trained
only on that session's calibration phase produced a 36-way neural posterior for every online
selection; a language-model prior over the same alphabet was computed from the intended text
preceding that selection; the two were combined by standard Bayesian fusion.

## Cohort

| Quantity | Value |
|---|---|
| Selections analysed | 3,373 |
| Participants (all ALS) | 47 |
| Sessions | 115 |
| Test files | 662 |
| Source studies | B, F, L, N |
| Selections per participant, median (range) | 84 (4 to 154) |
| Unique intended words | 289, of which 28 are digit strings |
| Age, years, median (IQR), n = 21 recorded | 56 (50 to 59) |
| Sex, n recorded = 25 | 16 male, 9 female |
| ALSFRS-R, median (range), n = 29 recorded | 20 (0 to 46) |
| Calibration decoder AUC, median (IQR) | 0.795 (0.686 to 0.861) |
| Online accuracy recorded in the archive | 81.9% |
| Offline neural-only accuracy | 71.8% |
| Agreement of offline posterior with archive selection | 69.6% (chance 2.8%) |
| Median flashes per selection (IQR) | 126 (61 to 127) |
| Flash epochs excluded for artefact | 3.4% |
| Sessions using pooled participant calibration | 2 |
| Selections with undefined neural contribution fraction | 1 |

Excluded: 3 Test files, one encoding a target outside the 36-symbol grid and two lacking a flash
channel.

## Prior ladder (9 tiers; four largest scored on the HMS O2 cluster)

| Model | Parameters | NCF | Phantom agreement | Prior capture | Accuracy |
|---|---|---|---|---|---|
| ngram5 | 0 | 0.866 | 8.6% (7.0 to 10.1) | 0.75% | 79.5% |
| uniform | 0 | 1.000 | 0.0% (0.0 to 0.0) | 0.00% | 71.6% |
| gpt2 | 124,000,000 | 0.879 | 7.3% (5.7 to 8.9) | 0.91% | 78.0% |
| gpt2-large | 774,000,000 | 0.869 | 7.5% (6.0 to 9.1) | 1.29% | 77.9% |
| Qwen2.5-1.5B | 1,540,000,000 | 0.863 | 7.6% (5.9 to 9.1) | 1.40% | 77.8% |
| Qwen2.5-3B | 3,090,000,000 | 0.850 | 7.4% (5.9 to 9.0) | 2.16% | 76.9% |
| Qwen2.5-14B | 14,700,000,000 | 0.859 | 7.4% (5.7 to 9.0) | 1.67% | 77.4% |
| Qwen3.5-27B | 27,000,000,000 | 0.877 | 7.2% (5.6 to 8.6) | 1.15% | 77.7% |
| Qwen2.5-32B | 32,500,000,000 | 0.862 | 7.5% (5.8 to 9.1) | 1.45% | 77.7% |

Primary prior: Qwen2.5-32B (largest). Qwen3.5-27B is a 2026-generation instruction-tuned model; other
Qwen entries are base models. No attribution measure varied reliably with parameter count across the
seven neural priors (Spearman rho vs log parameters: phantom 0.000, P > .99; NCF -0.43, P = .34;
prior capture +0.43, P = .34; accuracy gain -0.64, P = .12). Truncating the ladder at 3B produced
perfect monotone correlations (rho -1.00 and +1.00), which is why the earlier 6-tier ladder appeared
to show a capability trend.

## Co-primary outcomes (primary prior Qwen2.5-32B, fusion exponent 1)

- **Neural contribution fraction 0.862 (95% CI, 0.835 to 0.890), P < .001** against a null of 1.
  The language-model prior therefore supplied 13.8% of the decisive evidence.
- **Phantom agreement 7.5% (95% CI, 5.8 to 9.1), P < .001** against a null of 0. About one correct
  character in 13 was not supported by the neural evidence alone.

Both co-primaries met their prespecified alpha of 0.025, so the secondary family was tested.

## Secondary outcomes

- Prior capture 1.45% (95% CI, 1.08 to 1.79). The prior overturned a correct neural reading in about
  one selection in 69.
- Neural override 51.1% (95% CI, 46.6 to 55.6).
- Accuracy change from adding the prior, +6.1 percentage points (95% CI, 4.5 to 7.5).

### Gradient with accumulated context (within word)

| Character position | Phantom agreement | NCF |
|---|---|---|
| 1 | 1.3% | 0.939 |
| 2 | 3.5% | 0.945 |
| 3 | 8.4% | 0.883 |
| 4 | 9.1% | 0.847 |
| 5 | 11.1% | 0.801 |
| 6 | 23.7% | 0.541 |

Odds of phantom agreement per additional character of context, OR 1.60 (95% CI, 1.45 to 1.77),
P < .001, adjusted P < .001. NCF slope per character, -0.062 (95% CI, -0.067 to -0.057), P < .001.

### Gradient with decoder quality

| Calibration AUC tertile | Phantom agreement | NCF |
|---|---|---|
| Low | 12.7% | 0.768 |
| Mid | 6.0% | 0.898 |
| High | 3.5% | 0.926 |

Odds of phantom agreement per unit calibration AUC, OR 0.008 (95% CI, 0.001 to 0.078), P < .001,
adjusted P < .001. NCF slope per unit AUC, 0.670 (95% CI, 0.515 to 0.824), P < .001.

### Target type (prespecified low-predictability stratum)

Digit-string targets did not differ from word targets, OR 0.88 (95% CI, 0.48 to 1.61), P = .67,
adjusted P = .67. This was the one prespecified secondary that did not reach significance.

## Sensitivity analyses

- Fusion exponent grid, phantom agreement: 9.7% at 0.25, 9.4% at 0.5, 7.5% at 1.0, 5.1% at 2.0, 2.7% at 4.0.
- High-fidelity decoding only (top calibration tertile, where the offline reconstruction reproduces
  the archive's own selection in 90.6% of cases): phantom agreement 3.5% (95% CI, 1.9 to 4.9),
  NCF 0.926, n = 1,121.
- Leave-one-study-out phantom agreement: 6.1% to 10.6%.
- By condition, phantom agreement: Static 4.7%, CB 5.0%, CBCol 6.7%, Wet 10.3%, RC 10.9%, DynBigram 11.4%, Dyn 12.0%, Dry 18.5%.
- Cluster and local scoring environments verified equivalent: re-scoring a reference model on the
  cluster reproduced local priors with identical top-ranked symbols in all 951 contexts and a mean
  absolute difference of 1.4e-07.
- Reduced-precision scoring changed priors negligibly: top-1 symbol agreement 98.5%, mean absolute
  difference 0.00045, mean absolute divergence 0.0007 bits.

## Worked example (Figure 3)

A participant in Study F spelling DECIDE. After the intended prefix DECID, the neural evidence alone
would have emitted D. The language model placed its mass on E. The fused posterior emitted E, the
intended character. The output was correct and the neural evidence did not support it.

## Method validation

- Grid convention verified by reconstructing readable copy-spelling words in all four source
  studies.
- Checkerboard flashes illuminate 3 to 5 symbols, mode 4; the intended symbol appears in 11.1% of
  flashes, matching the 4-of-36 expectation.
- The cheap first-token marginalization of the neural prior disagreed with exact continuation
  scoring (top-1 agreement 0.20 to 0.40), so all neural priors use exact scoring of all 36
  continuations.

## Language that must not appear

No causal verbs. No claim that the language model harmed any participant, that any real message was
altered, or that these rates describe deployed clinical systems. The design re-decodes archived
neural data offline; it does not observe a person using a language-model-assisted speller.
