"""Diagnose whether appending a candidate character changes the tokenization of an
already-fixed context, which would break the assumption that context-prefix log-
probability cancels when normalizing across the 36 candidates in batched continuation
scoring (`authorship/priors.py`'s `TransformerPrior._legacy_full_sequence_prior`).

`_legacy_full_sequence_prior` re-tokenizes `context + candidate` fresh for each of the 36 grid
symbols and normalizes the 36 resulting total log-probabilities against one another.
That normalization step treats the log-probability mass assigned to the shared context
prefix as a constant that cancels out. It is only a genuine constant if
`tokenizer.encode(context)` is always a strict prefix of `tokenizer.encode(context +
candidate)`'s token list for every candidate: the context's own tokenization does
not change once a character is appended to it. Byte-pair tokenizers can violate this at
a word or token boundary (e.g. "he" + "y" merging into a token distinct from "he"
alone).

This is a directly testable, CPU-only, tokenizer-only question: no model forward pass
is needed, so this diagnostic never touches GPU or model weights. It measures how often
the defect actually occurs on real study contexts against the real primary-prior
tokenizer. Its result is what moved production scoring back to `prefix_locked_prior`,
which never re-tokenizes per candidate and so cannot exhibit this defect at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.grid import SYMBOLS  # noqa: E402

OUTPUT_PATH = PROJECT / "output" / "prefix_tokenization_stability.json"
SAMPLE_SIZE = 200
SAMPLE_SEED = 20260805

# Same primary-prior model name as scripts/run_stats.py's PRIMARY_PRIOR and
# scripts/build_attribution_emitted_context.py's --model default. TransformerPrior
# itself (authorship/priors.py) is not instantiated here: its __init__ unconditionally
# also loads the full causal-LM weights (AutoModelForCausalLM.from_pretrained), which
# for this 32.5B-parameter model is a many-tens-of-gigabytes download this diagnostic
# does not need and normally only runs on the O2 cluster (see build_priors.py's
# "cluster" tier). This loads the tokenizer the identical way TransformerPrior.__init__
# does (`AutoTokenizer.from_pretrained(model_name)`, no extra arguments) without paying
# for the model weights.
PRIMARY_MODEL = "Qwen/Qwen2.5-32B"


def prefix_is_stable(tokenizer, context: str, candidate: str) -> bool:
    """True iff tokenizing `context` alone is a strict prefix of tokenizing
    `context + candidate`.

    Calls `.encode()` with no `add_special_tokens` override, exactly matching the
    literal call `TransformerPrior._legacy_full_sequence_prior` makes on
    `self.tokenizer`: this measures the tokenizer's real default behavior, the one that
    scorer actually exercised, not an idealized one.
    """
    context_tokens = tokenizer.encode(context)
    full_tokens = tokenizer.encode(context + candidate)
    return full_tokens[: len(context_tokens)] == context_tokens


def run(tokenizer, contexts: list[str]) -> dict[str, object]:
    """Check prefix stability for every (context, candidate) pair, lower-casing both
    exactly as `TransformerPrior._prepare` does under the study's only casing setting
    (`casing="lower"`, the default and the only value ever used for the primary prior;
    no call site in this codebase passes `casing="upper"`). Contexts and candidates
    are reported back in their original (upper-case grid) form in `examples_unstable`
    so they read naturally against `context_prefix`/`authorship.grid.SYMBOLS`.
    """
    total = 0
    stable = 0
    examples_unstable = []
    for context in contexts:
        for candidate in SYMBOLS:
            total += 1
            if prefix_is_stable(tokenizer, context.lower(), candidate.lower()):
                stable += 1
            elif len(examples_unstable) < 20:
                examples_unstable.append({"context": context, "candidate": candidate})
    return {
        "n_contexts_sampled": len(contexts),
        "n_pairs_checked": total,
        "fraction_stable": stable / total if total else float("nan"),
        "fraction_unstable": 1 - (stable / total) if total else float("nan"),
        "examples_unstable": examples_unstable,
    }


def main() -> None:
    from transformers import AutoTokenizer

    selections = pd.read_parquet(PROJECT / "output" / "intermediate" / "selections.parquet")
    unique_contexts = selections["context_prefix"].dropna().drop_duplicates()
    contexts = unique_contexts.sample(
        n=min(SAMPLE_SIZE, unique_contexts.nunique()), random_state=SAMPLE_SEED
    ).tolist()

    tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
    result = run(tokenizer, contexts)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "examples_unstable"}, indent=2))


if __name__ == "__main__":
    main()
