# Cohort reconciliation and pipeline validation

All counts produced by `scripts/build_analysis_frame.py` on 2026-07-26.

## Grid convention (Task 1 gate)

Concatenating intended symbols across consecutive Test trials reproduces readable
copy-spelling targets in every source study: `THE`, `LAZY`, `DOG` (Study B), `ENOUGH`,
`EASILY`, `REGION`, `WINDOW` (Study F), `PLENTY`, `THREAT`, `KNIGHT`, `ADVICE` (Study L),
`VALLEY`, `PLACES`, `SINGLE`, `SENIOR` (Study N). The one-based channel-order convention is
therefore correct. Verdict: PASS.

## Flash membership (Task 3 gate)

Flash channels are binary. Checkerboard conditions illuminate 3 to 5 symbols per flash
(mode 4), confirming that a row-and-column reading would assign evidence to the wrong
symbols. The intended symbol appears in 11.1% of flashes, matching the 4-of-36 expectation.
Verdict: PASS.

## Neural posterior fidelity (Task 3 gate)

The offline posterior is rebuilt from raw EEG using a decoder trained only on the same
session's calibration phase. Its argmax agrees with the selection the archive actually made
online in 69.6% of selections, against a chance rate of 2.8%.

Agreement tracks decoder quality closely:

| Calibration AUC tertile | Agreement with archive selection |
|---|---|
| Low | 0.392 |
| Middle | 0.805 |
| High | 0.906 |

By source study: B 0.846, L 0.740, F 0.624, N 0.496.

Offline neural-only accuracy is 0.718 against the archive's online accuracy of 0.819. Our
reimplemented decoder is therefore somewhat weaker than the systems that produced the
original sessions. This matters for interpretation: weaker neural evidence inflates the
apparent contribution of the prior. Two consequences were pre-specified before any
attribution result was computed.

1. Decoder fidelity enters the analysis as a stratifying variable, reported by calibration
   AUC tertile.
2. The high-fidelity subgroup, where the offline reconstruction reproduces the archive's own
   selections in 90.6% of cases, is a key sensitivity analysis. A finding that survives
   there is not an artefact of a weak reimplementation.

Verdict: PASS with the stated caveat carried into the limitations.

## Selection counts

| Quantity | Value |
|---|---|
| Selections in the analysis frame | 3,373 |
| Participants | 47 |
| Sessions | 115 |
| Test files | 662 |
| Unique intended phrases | 289 |
| Unique contexts requiring a prior | 951 |
| Median flashes per selection | 126 (IQR 61 to 127) |
| Flash epochs dropped for artefact | 3.4% |
| Sessions using pooled participant calibration | 2 |

Reconciliation against the independent trial reconstruction, which counted 3,395 eligible
selections: three Test files were excluded, one because it encoded a target outside the
36-symbol grid (code 38, indicating a different matrix size) and two because the EDF lacked
the `9_6_6` flash channel. The remaining difference comes from trials whose feedback phase
was not preceded by a stimulation window, and trials retaining fewer than two artefact-free
flash epochs. No selection was dropped on the basis of its outcome.

## Analysis strata available in the data

- Conditions: checkerboard (1,176), static (360), dynamic stopping (359), dynamic stopping
  with a bigram model (348), checkerboard column (330), row-column (330), dry electrodes
  (236), wet electrodes (234). The bigram condition is notable because those sessions ran a
  language prior online.
- Position within the intended word ranges from 0 to 5, so the amount of context available
  to the prior increases within every word.
- Numeric targets contribute 168 selections across 28 phrases and form the pre-specified
  low-predictability stratum, where an English prior carries little usable information.
