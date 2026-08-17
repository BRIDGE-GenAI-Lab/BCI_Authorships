# Neural contribution and phantom agreement during language-model-assisted P300 spelling

Analysis code and result digests for a retrospective attribution study: how much of each character
emitted by a language-model-assisted P300 speller is determined by the user's neural evidence versus
by the language-model prior, and whether that share changes as the prior grows more capable.

Gorenshtein A, Adiniaev Y, Liba T, Klang E, Daniel O. *Neural Contribution and Phantom Agreement
During Language-Model-Assisted P300 Spelling in Amyotrophic Lateral Sclerosis.* Manuscript under
review, npj Digital Medicine.

## What the study measures

3,373 archived online P300-speller selections from 47 people with amyotrophic lateral sclerosis
(bigP3BCI, source studies B/F/L/N) were re-decoded offline and fused with a ladder of 25
language-model priors: a uniform null, 3 classical character 5-grams, and 21 causal language models
spanning eleven architecture families (GPT-2, Qwen, Llama, Gemma, Mistral, DeepSeek, OLMo, GPT-OSS,
and others) from 124 million to 46.7 billion parameters. Two co-primary outcomes quantify attribution
per selection: the neural contribution fraction (a Kullback-Leibler displacement share, bounded in
[0, 1]) and phantom agreement (a correct emission the neural evidence alone would not have selected).

| | |
|---|---|
| Eligible online selections | 3,373 |
| Participants | 47 |
| Recording sessions | 115 |
| Prior ladder | 25 entries: null + 3 classical n-grams + 21 neural models across 11 families |
| Bootstrap replicates | 2,000 (participant-cluster) |

## Headline results

- **The language-model prior accounted for a participant-weighted mean 8.6% of posterior
  displacement** (unweighted median, 2.9%), concentrated in a minority of selections. Under held-out,
  per-source temperature calibration (the primary fusion rule), the neural contribution fraction was
  0.914 (95% CI, 0.896 to 0.934).
- **About 1 selection in 23 emitted a character the neural evidence alone would not have selected.**
  Phantom agreement was 4.4% (95% CI, 3.5 to 5.3).
- **Attribution shifted toward the model as context accumulated within a word** (phantom agreement
  2.5% at a word's first character to 7.7% at its sixth) **and as the neural signal weakened**
  (phantom agreement 7.2%, 3.9%, and 2.7% across ascending tertiles of calibration-decoder quality).
- **No monotonic association with parameter count was detected, nor an architecture-family effect.**
  Across the 21 neural language models, no attribution measure correlated reliably with size
  (Spearman |rho| &lt; 0.2, all *P* &gt; .4) or varied by family (permutation test). A truncated ladder
  showed a trend that weakened as more of the range was sampled, a truncation-artefact signature
  reported as a caution rather than evidence of scale invariance. Character 5-grams exceeded every
  neural language model on calibrated fused accuracy (79.6% to 80.4% versus 72.1% to 74.0%).

## Layout

```
authorship/          analysis package
  grid.py             36-symbol P300 grid alphabet
  decoder.py          per-session calibration-trained logistic decoder
  neural_posterior.py 36-way posterior over the grid from accumulated flash evidence
  priors.py           the prior ladder: family_of() classifier, n-gram (add-k + Kneser-Ney) and
                       transformer priors, WikiText-103/Brown corpus loaders
  context.py          intended- and emitted-context construction
  attribution.py      neural contribution fraction, phantom agreement, prior capture, neural override
  calibration.py      held-out, per-source temperature calibration for fusion
scripts/              pipeline entry points: build_priors, ingest_o2_priors, build_attribution,
                       build_attribution_emitted_context, build_permuted_context_prior,
                       compare_scorer_methods, simulate_ncf_properties,
                       validate_prefix_tokenization_stability, run_stats, make_figures,
                       assemble_manuscript, build_references, build_cover_letter
tests/                pytest suite; 119 tests run standalone against this repository (see
                       "Reproducing"). Tests that check consistency against the manuscript text or
                       against generated intermediates not distributed here live in the primary
                       study repository.
results/digests/      the JSON digests behind every number in the manuscript
                       (stats_digest.json, attribution_summary.json, cohort_summary.json,
                       word_counts.json)
results/figures/      Figure 1-4 and eFigure 1-4, 600 DPI, TrueType fonts embedded
```

## Reproducing

```bash
pip install -e .            # or install the packages imported by authorship/ and scripts/ directly
python -m pytest tests -q   # 119 pass standalone; the rest are skipped without optional
                             # dependencies or the raw data
```

Regenerating the digests from scratch requires the raw bigP3BCI recordings and the per-model prior
shards (some scored locally, some on an academic GPU cluster), which are not distributed here.
`scripts/build_priors.py`, `scripts/ingest_o2_priors.py`, `scripts/build_attribution.py`, and
`scripts/run_stats.py` show the exact entry points and are the same scripts that produced
`results/digests/`.

## Data

- **bigP3BCI** is openly available from PhysioNet and is not redistributed here. DOI
  [10.13026/0byy-ry86](https://doi.org/10.13026/0byy-ry86).
- Language-model weights for the prior ladder were obtained from the Hugging Face Hub; none are
  redistributed here. Every checkpoint used is named in `scripts/build_priors.py`'s `LADDER` and
  cited in the manuscript.

## Two things worth knowing if you reuse this code

1. **Boolean model outcomes silently invert their odds ratio if passed to the GEE/mixed-model formula
   layer as raw booleans.** A boolean column is expanded into a two-level categorical whose first
   level is `False`, which the fitter treats as the reference — the reported OR ends up describing the
   complement of the intended effect. Binary outcomes are cast to integers before modelling, and a
   regression test guards this.
2. **The prior ladder's truncation threshold changed mid-study, deliberately.** An early, narrow-range
   ladder showed a near-perfect rank correlation between model size and attribution. Extending the
   ladder to span five orders of magnitude in parameter count weakened the trend to a
   non-significant, non-monotonic pattern. Both stages are reported (`results/digests/
   stats_digest.json`) as evidence that the original correlation was a property of a narrow sampled
   range, not of capability — see `authorship/priors.py`'s `family_of()` and `scripts/run_stats.py`.

## License

MIT (see `LICENSE`). The license covers this code. It does not extend to the bigP3BCI dataset, which
carries its own PhysioNet terms.
