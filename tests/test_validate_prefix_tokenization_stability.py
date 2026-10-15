"""Tests for the prefix-tokenization-stability diagnostic.

`_legacy_full_sequence_prior` (authorship/priors.py) re-tokenizes `context + candidate` fresh for
each of the 36 candidate symbols and normalizes the resulting totals against one
another. That normalization is only exact relative to a fixed conditioning prefix if
`tokenizer.encode(context)` is always a strict prefix of `tokenizer.encode(context +
candidate)` — i.e. appending the candidate character never changes how the context
itself gets tokenized. `prefix_is_stable` checks exactly that, for one context/candidate
pair.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_prefix_stable_flags_identical_prefix_tokenization():
    from validate_prefix_tokenization_stability import prefix_is_stable

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            # word-level toy tokenizer: splits on nothing, just returns char codes,
            # standing in for a real BPE tokenizer's .encode() interface
            return [ord(c) for c in text]

    tok = FakeTokenizer()
    assert prefix_is_stable(tok, context="he", candidate="l") is True


def test_prefix_stable_flags_boundary_merging_tokenizer():
    from validate_prefix_tokenization_stability import prefix_is_stable

    class MergingTokenizer:
        def encode(self, text, add_special_tokens=False):
            # toy tokenizer that merges "he" + "y" into one token distinct from "he" alone
            if text == "hey":
                return [999]
            return [ord(c) for c in text]

    tok = MergingTokenizer()
    assert prefix_is_stable(tok, context="he", candidate="y") is False


def test_prefix_stable_checks_token_identity_not_just_token_count():
    """A same-length-but-different-content re-tokenization must count as unstable.

    A buggy implementation that only compares `len(full_tokens) >= len(context_tokens)`
    (ignoring token identity) would wrongly call this pair "stable", and would also
    happen to pass both tests above by coincidence, since those two examples differ in
    token count as well as content. This tokenizer keeps the token count the same (2
    tokens either way) but changes what the first two token ids actually are once the
    candidate is appended, which must be caught.
    """
    from validate_prefix_tokenization_stability import prefix_is_stable

    class SameLengthRelabelingTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "hel":
                return [1, 2]
            if text == "he":
                return [104, 101]
            return [ord(c) for c in text]

    tok = SameLengthRelabelingTokenizer()
    assert prefix_is_stable(tok, context="he", candidate="l") is False


def test_run_aggregates_stability_across_contexts_and_all_36_symbols():
    from authorship.grid import SYMBOLS
    from validate_prefix_tokenization_stability import run

    class AlwaysStableTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    result = run(AlwaysStableTokenizer(), ["A", "SPEEC"])
    assert result["n_contexts_sampled"] == 2
    assert result["n_pairs_checked"] == 2 * len(SYMBOLS)
    assert result["fraction_stable"] == 1.0
    assert result["fraction_unstable"] == 0.0
    assert result["examples_unstable"] == []


def test_run_records_unstable_examples_up_to_a_cap():
    from validate_prefix_tokenization_stability import run

    class AlwaysUnstableTokenizer:
        def encode(self, text, add_special_tokens=False):
            # every appended candidate is scored as a single fused token, so the
            # context-only tokenization is never a prefix of the extended one
            return [len(text)]

    result = run(AlwaysUnstableTokenizer(), ["A", "B"])
    assert result["fraction_stable"] == 0.0
    assert result["fraction_unstable"] == 1.0
    assert len(result["examples_unstable"]) == 20
    assert all({"context", "candidate"} <= example.keys() for example in result["examples_unstable"])
