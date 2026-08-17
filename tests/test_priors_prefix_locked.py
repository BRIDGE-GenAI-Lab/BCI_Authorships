import numpy as np
import pytest

from authorship.grid import SYMBOLS, symbol_index
from authorship.priors import build_symbol_projection, prefix_locked_prior


class FakeTokenizer:
    """A tokenizer whose vocabulary is entirely under test control: token id i decodes
    to `vocab[i]`. Encoding is greedy longest-match against `vocab`, so it can be made
    to retokenize at word boundaries exactly like a real BPE tokenizer, without needing
    real model weights.
    """

    def __init__(self, vocab: list[str], bos_token_id: int = 0):
        self.vocab = vocab
        self.bos_token_id = bos_token_id
        self.eos_token_id = bos_token_id

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def encode(self, text, add_special_tokens=True):
        tokens = []
        i = 0
        while i < len(text):
            for length in range(min(8, len(text) - i), 0, -1):
                piece = text[i : i + length]
                if piece in self.vocab:
                    tokens.append(self.vocab.index(piece))
                    i += length
                    break
            else:
                raise ValueError(f"no vocab entry covers {text[i:]!r}")
        return tokens


class FakeLogitsModel:
    """A stand-in causal LM: __call__ returns fixed logits for the position(s) requested,
    driven entirely by a `next_token_probs` dict keyed on the exact input id tuple, so
    tests can assert prefix_locked_prior marginalizes correctly without a real network.
    """

    def __init__(self, vocab_size: int, next_token_probs: dict[tuple[int, ...], np.ndarray]):
        self.vocab_size = vocab_size
        self.next_token_probs = next_token_probs

    def __call__(self, input_ids, attention_mask=None):
        import torch

        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -1e9)
        for row in range(batch):
            key = tuple(int(x) for x in input_ids[row].tolist())
            probs = self.next_token_probs[key]
            logits[row, -1] = torch.log(torch.tensor(probs, dtype=torch.float32))
        return type("Output", (), {"logits": logits})()

    def eval(self):
        return self

    def parameters(self):
        yield __import__("torch").zeros(1)


def test_prefix_locked_prior_is_immune_to_retokenization():
    """Construct a tokenizer where appending "h" to "spee" + "c" merges into a single
    "spe"+"ech" tokenization distinct from "spee"+"c" alone -- exactly the retokenization
    failure mode the diagnostic measures on real tokenizers. prefix_locked_prior must
    still return the correct marginal because it never re-encodes context+candidate.
    """
    vocab = ["<bos>", "spee", "c", "h", "z", "spe", "ech"]
    tokenizer = FakeTokenizer(vocab, bos_token_id=0)
    context_ids = (0, 1)  # <bos> spee
    probs = np.zeros(len(vocab))
    probs[2] = 0.7  # "c"
    probs[3] = 0.2  # "h"
    probs[4] = 0.1  # "z"
    model = FakeLogitsModel(len(vocab), {context_ids: probs})
    projection = build_symbol_projection(tokenizer, len(vocab))
    result = prefix_locked_prior(
        "spee", tokenizer, model, device="cpu", seed_ids=[0], projection=projection,
    )
    assert result.shape == (36,)
    assert abs(float(result.sum()) - 1.0) < 1e-6
    assert result[symbol_index("C")] > result[symbol_index("H")] > result[symbol_index("Z")]
    assert abs(float(result[symbol_index("C")]) - 0.7) < 1e-6


def test_build_symbol_projection_classifies_by_first_character():
    vocab = ["<bos>", "hello", " world", "9x", "!!"]
    tokenizer = FakeTokenizer(vocab)
    projection = build_symbol_projection(tokenizer, len(vocab))
    assert projection[1, symbol_index("H")] == 1.0
    assert projection[2, symbol_index(" ")] == 1.0
    assert projection[3, symbol_index("9")] == 1.0
    assert projection[4].sum() == 0.0  # "!!" has no grid-symbol first character
    assert projection[0].sum() == 0.0  # "<bos>" decodes to non-alphabet text


def test_prefix_locked_prior_renormalizes_over_valid_symbols_only():
    """If some next-token mass falls on tokens outside the 36-symbol alphabet, the
    returned distribution must still sum to one over the 36 symbols (conditioned on
    the next token mapping to a valid symbol at all), matching how every other prior
    in this codebase is defined.
    """
    vocab = ["<bos>", "a", "!", "b"]
    tokenizer = FakeTokenizer(vocab)
    context_ids = (0,)
    probs = np.array([0.0, 0.4, 0.5, 0.1])
    model = FakeLogitsModel(len(vocab), {context_ids: probs})
    projection = build_symbol_projection(tokenizer, len(vocab))
    result = prefix_locked_prior(
        "", tokenizer, model, device="cpu", seed_ids=[0], projection=projection,
    )
    assert abs(float(result.sum()) - 1.0) < 1e-6
    assert result[symbol_index("A")] > result[symbol_index("B")]


def test_prefix_locked_prior_never_retokenizes_context_plus_candidate():
    """Regression guard: prefix_locked_prior must call tokenizer.encode at most once
    per invocation (on `context` alone). A FakeTokenizer that raises if asked to encode
    any string containing more than the bare context proves no candidate-dependent
    re-encoding happens anywhere in the call path.
    """

    class RaisingOnCandidateTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            if len(text) > len("spee"):
                raise AssertionError(f"must not encode a candidate-extended string: {text!r}")
            return super().encode(text, add_special_tokens=add_special_tokens)

    vocab = ["<bos>", "spee", "c", "h"]
    tokenizer = RaisingOnCandidateTokenizer(vocab)
    context_ids = (0, 1)
    probs = np.array([0.0, 0.0, 0.6, 0.4])
    model = FakeLogitsModel(len(vocab), {context_ids: probs})
    projection = build_symbol_projection(tokenizer, len(vocab))
    result = prefix_locked_prior(
        "spee", tokenizer, model, device="cpu", seed_ids=[0], projection=projection,
    )
    assert abs(float(result.sum()) - 1.0) < 1e-6
