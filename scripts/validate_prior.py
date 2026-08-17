"""Compare the production prior against the deprecated continuation scorer.

Production scoring is `TransformerPrior.prior`, i.e. `prefix_locked_prior`: one forward
pass per context, with the next-token distribution marginalized onto the 36 grid symbols
by each vocabulary token's own first character. It tokenizes the context exactly once, so
it cannot be affected by the retokenization instability that
`scripts/validate_prefix_tokenization_stability.py` measured (33.9% of sampled pairs).

`TransformerPrior.exact_prior` is the deprecated comparator kept from an earlier revision:
it re-tokenizes `context + candidate` separately for each of the 36 candidates, which is
the method that instability invalidates. It is the unbatched equivalent of
`_legacy_full_sequence_prior`, which scores the same 36 sequences in one padded batch;
tests/test_priors.py asserts the two agree. This script quantifies how far the two diverge on
real copy-spelling contexts. The divergence is the size of the correction, not evidence
against the production scorer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.grid import SYMBOLS  # noqa: E402
from authorship.priors import CharNgramPrior, TransformerPrior  # noqa: E402

SAMPLE_CONTEXTS = [
    "", "S", "SP", "SPE", "SPEE", "SPEEC",
    "T", "TH", "THR", "THRE", "THREA",
    "K", "KN", "KNI", "KNIG", "KNIGH",
    "A", "AD", "ADV", "ADVI", "ADVIC",
    "2", "22", "222", "2224", "22247",
    "V", "VA", "VAL", "VALL", "VALLE",
]


def main(model_name: str = "gpt2") -> None:
    prior_model = TransformerPrior(model_name)
    print(f"model={model_name} device={prior_model.device}")

    differences: list[float] = []
    correlations: list[float] = []
    top1_agreements: list[bool] = []
    for context in SAMPLE_CONTEXTS:
        production = prior_model.prior(context)
        deprecated = prior_model.exact_prior(context)
        differences.append(float(np.abs(production - deprecated).mean()))
        correlations.append(float(spearmanr(production, deprecated).statistic))
        top1_agreements.append(int(production.argmax()) == int(deprecated.argmax()))

    print("production = prefix_locked_prior; comparator = deprecated continuation scorer (exact_prior)")
    print(f"contexts compared: {len(SAMPLE_CONTEXTS)}")
    print(f"mean absolute difference: {np.mean(differences):.5f}")
    print(f"median Spearman correlation: {np.median(correlations):.3f}")
    print(f"top-1 symbol agreement: {np.mean(top1_agreements):.3f}")

    print("\n=== worked examples (top 3 symbols) ===")
    ngram = CharNgramPrior(order=5)
    for context in ["SPEEC", "THREA", "KNIGH", "22247", ""]:
        neural = prior_model.prior(context)
        classical = ngram.prior(context)
        top = lambda vector: ", ".join(  # noqa: E731
            f"{SYMBOLS[index]!r}={vector[index]:.2f}" for index in np.argsort(vector)[::-1][:3]
        )
        print(f"  context={context!r:9s} ngram5: {top(classical):28s} {model_name}: {top(neural)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gpt2")
