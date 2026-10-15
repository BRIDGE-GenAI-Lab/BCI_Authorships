# Neural contribution and phantom agreement in language-model-assisted P300 spelling

Analysis code, tests and derived results for the study *Quantifying Language Model Influence in Brain-Computer Interface Communication for Amyotrophic Lateral Sclerosis*.

The study re-decodes archived online P300-speller sessions from people with amyotrophic lateral sclerosis (ALS) and attributes each emitted character to two sources: the participant's neural evidence and a language-model prior.

## What is measured

For every copy-spelled selection the neural posterior over the 36-symbol grid is fused with a language prior, and three per-selection quantities are computed:

| Measure | Question it answers |
|---|---|
| Neural contribution fraction (NCF) | How much of the displacement of the fused posterior from the prior came from neural evidence? |
| Phantom agreement | How often did fusion emit the intended character when neural evidence alone would have selected another? |
| Prior capture | How often did the prior overturn a correct neural choice? |

Priors range from a uniform null and character 5-grams to 21 neural language models across 11 architecture families.

## Data

BigP3BCI v1.0.0 (PhysioNet, doi:10.13026/0byy-ry86, CC BY 4.0), ALS source studies B, F, L and N: 3,373 online selections from 47 participants in 115 sessions. The data are not redistributed here; download them from PhysioNet.

## Layout

```
authorship/   attribution, calibration, context, decoder, grid, neural posterior, priors
scripts/      pipeline, validation, statistics and figure scripts
tests/        unit and regression tests
results/
  digests/    summary statistics produced by the pipeline (JSON)
  figures/    manuscript and supplementary figures (PDF, PNG)
```

## Pipeline

Run from the repository root. Each script's module docstring describes its inputs and outputs.

| Stage | Script |
|---|---|
| Validate grid coding and the reconstructed posterior | `validate_grid.py`, `validate_posterior.py` |
| Selection frame with neural posteriors | `build_analysis_frame.py` |
| Language-model prior ladder | `build_priors.py`, `ingest_o2_priors.py`, `validate_prior.py`, `validate_prefix_tokenization_stability.py`, `compare_scorer_methods.py` |
| Per-selection attribution | `build_attribution.py` |
| Controls | `build_permuted_context_prior.py`, `build_attribution_emitted_context.py`, `simulate_ncf_properties.py` |
| Statistics | `run_stats.py` |
| Reversal strength, confirmation triage, fidelity-restricted and stronger-decoder analyses | `build_strong_decoder_selections.py`, `revision_analyses.py` |
| Figures | `make_figures.py`, `make_revision_figures.py` |
| Check that a regenerated digest matches the reference | `check_digest_reproduces.py` |

Scoring the larger language models needs a GPU; `score_emitted_context_o2.sbatch` is a Slurm template for the closed-loop analysis.

## Requirements

Python 3.9 or later with `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `mne`, `torch` and `transformers`.

The EDF parsing and trial reconstruction helpers (`bigp3_als`) come from the companion BigP3BCI calibration project. The scripts expect it at `../study_bigp3_als_calibration/src` and the downloaded archive under `../study_bigp3_als_calibration/data/source_cache`.

## Tests

```
python -m pytest -q
```

Tests that exercise the full pipeline need the intermediate outputs in `output/` and are skipped without them.

## Results and figure labels

`results/digests/stats_digest.json` and `revision_digest.json` hold every statistic reported in the manuscript. Files named `eFigure1` to `eFigure6` are Supplementary Figures S1 to S6.

## Licence

MIT. See `LICENSE`.
