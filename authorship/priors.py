"""Language-model priors over the 36-symbol speller alphabet.

Three tiers are provided: a uniform null, a character n-gram in the tradition of the
classical speller literature, and neural language models. Every prior returns a length-36
probability vector aligned to ``grid.SYMBOLS``.

Neural tokenizers do not align to characters, so the next-symbol distribution is obtained
by one forward pass over the context followed by marginalizing the next-token distribution
onto the first character of each token, via the module-level ``prefix_locked_prior``. This
is the primary scorer: it tokenizes the context exactly once and is immune to the
retokenization instability that affects ``TransformerPrior``'s legacy full-sequence
scorer (``_legacy_full_sequence_prior`` / ``exact_prior``), which are retained only for
comparison, not for production scoring.
"""

from __future__ import annotations

import functools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from authorship.grid import SYMBOLS, symbol_index

ALPHABET = set(SYMBOLS)
N_SYMBOLS = len(SYMBOLS)
SMOOTHING_K = 0.1

_FAMILY_PREFIXES = (
    ("uniform", "null"),
    ("ngram", "classical"),
    ("gpt2", "gpt2"),
    ("Qwen/", "qwen"),
    ("meta-llama/", "llama"),
    ("google/gemma", "gemma"),
    ("mistralai/", "mistral"),
    ("deepseek-ai/", "deepseek"),
    ("allenai/", "olmo"),
    ("openai/", "gptoss"),
    ("EleutherAI/", "pythia"),
    ("state-spaces/mamba", "mamba"),
    ("google/recurrentgemma", "recurrentgemma"),
)


def family_of(prior_model: str) -> str:
    """Classify a ladder entry name into its architecture family.

    Single source of truth for family grouping, used by both the stats and figure
    scripts so the two never disagree about which models belong together.
    """
    for prefix, family in _FAMILY_PREFIXES:
        if prior_model == prefix or prior_model.startswith(prefix):
            return family
    raise ValueError(f"unknown prior family for {prior_model!r}")


def uniform_prior() -> np.ndarray:
    """The null prior: every grid symbol equally likely."""
    return np.full(N_SYMBOLS, 1.0 / N_SYMBOLS)


def normalize_corpus(text: str) -> str:
    """Fold arbitrary text onto the speller alphabet."""
    upper = text.upper()
    kept = "".join(character if character in ALPHABET else " " for character in upper)
    return re.sub(r" +", " ", kept)


class CharNgramPrior:
    """Character n-gram with add-k smoothing and backoff to shorter contexts."""

    def __init__(self, order: int = 5, corpus_text: str | None = None) -> None:
        if order < 1:
            raise ValueError("n-gram order must be at least 1")
        if corpus_text is None:
            corpus_text = load_brown_corpus()
        text = normalize_corpus(corpus_text)
        if not text.strip():
            raise ValueError("corpus is empty after normalization")
        self.order = order
        self.name = f"ngram{order}"
        self._counts: list[dict[str, np.ndarray]] = []
        for context_length in range(order):
            table: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_SYMBOLS))
            for position in range(context_length, len(text)):
                symbol = text[position]
                if symbol not in ALPHABET:
                    continue
                context = text[position - context_length : position]
                if any(character not in ALPHABET for character in context):
                    continue
                table[context][symbol_index(symbol)] += 1.0
            self._counts.append(dict(table))

    def prior(self, context: str) -> np.ndarray:
        context = normalize_corpus(context)
        for context_length in range(min(self.order - 1, len(context)), -1, -1):
            key = context[len(context) - context_length :] if context_length else ""
            counts = self._counts[context_length].get(key)
            if counts is not None and counts.sum() > 0:
                smoothed = counts + SMOOTHING_K
                return smoothed / smoothed.sum()
        return uniform_prior()


class CharNgramKNPrior:
    """Character n-gram with interpolated Kneser-Ney smoothing, via nltk.lm.

    Unlike CharNgramPrior's hand-rolled add-k backoff, interpolation across orders is
    handled internally by nltk's KneserNeyInterpolated — the caller always queries with
    the full available context and never manages backoff manually. Kept as a separate
    class rather than a mode of CharNgramPrior because the two have unrelated training
    and scoring code paths.
    """

    def __init__(self, order: int = 5, corpus_text: str | None = None) -> None:
        from nltk.lm import KneserNeyInterpolated
        from nltk.lm.preprocessing import padded_everygram_pipeline

        if order < 1:
            raise ValueError("n-gram order must be at least 1")
        if corpus_text is None:
            corpus_text = load_brown_corpus()
        text = normalize_corpus(corpus_text)
        if not text.strip():
            raise ValueError("corpus is empty after normalization")
        self.order = order
        self.name = f"ngram{order}_kn"
        characters = list(text)
        train_data, vocab = padded_everygram_pipeline(order, [characters])
        model = KneserNeyInterpolated(order=order)
        model.fit(train_data, vocab)
        self._model = model

    def prior(self, context: str) -> np.ndarray:
        context = normalize_corpus(context)
        window = context[-(self.order - 1) :] if self.order > 1 else ""
        context_tuple = tuple(window)
        scores = np.array([self._model.score(symbol, context_tuple) for symbol in SYMBOLS])
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            return uniform_prior()
        return scores / total


@functools.lru_cache(maxsize=1)
def load_brown_corpus() -> str:
    """Load the Brown corpus, the reference corpus of the classical speller literature."""
    from nltk.corpus import brown

    return " ".join(brown.words())


@functools.lru_cache(maxsize=1)
def load_wikitext_corpus(character_limit: int = 20_000_000, source: str | None = None) -> str:
    """Load a size-capped slice of WikiText-103 as a larger, more contemporary corpus
    than Brown (1961). Capped because a full character-level n-gram count table over the
    ~500M-character corpus is not memory-tractable; the cap is a disclosed methods
    decision, not silent truncation.

    Loaded via the ``datasets`` library rather than the archive's original direct-download
    URL: the original ``s3.amazonaws.com/research.metamind.io`` bucket now returns
    ``AccessDenied`` (the region-corrected endpoint confirms the object is no longer
    publicly reachable, not merely relocated).

    `source` is a testing seam: a local path or literal text, bypassing the real download.
    """
    if source is not None:
        text = Path(source).read_text() if Path(source).exists() else source
        return text[:character_limit]

    import datasets

    dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    chunks: list[str] = []
    total = 0
    for row in dataset:
        chunk = row["text"]
        chunks.append(chunk)
        total += len(chunk)
        if total >= character_limit:
            break
    return "".join(chunks)[:character_limit]


def build_symbol_projection(tokenizer, vocabulary_size: int) -> np.ndarray:
    """Map every vocabulary token onto the grid symbol its own, context-free decoded
    text starts with (space for a leading space, uppercased letter/digit otherwise).
    Tokens whose first character is not a grid symbol get an all-zero row and
    contribute no mass to any candidate. Context-free: decoding a single token id in
    isolation is deterministic for byte-level BPE and SentencePiece tokenizers, so this
    matrix does not depend on which context it will later be used with, and is built
    once per model, not once per context.
    """
    projection = np.zeros((vocabulary_size, N_SYMBOLS))
    for token_id in range(vocabulary_size):
        text = tokenizer.decode([token_id])
        if not text:
            continue
        first = text[0]
        symbol = " " if first == " " else first.upper()
        if symbol in ALPHABET:
            projection[token_id, symbol_index(symbol)] = 1.0
    return projection


def prefix_locked_prior(
    context: str,
    tokenizer,
    model,
    device,
    seed_ids: list[int],
    projection: np.ndarray,
    casing: str = "lower",
    frame: str | None = None,
) -> np.ndarray:
    """Exact P(next character | context) under one fixed tokenization of `context`.

    Tokenizes `context` exactly once (never `context + candidate`), runs one forward
    pass, and marginalizes the model's real next-token distribution by which grid
    symbol each vocabulary token's own first character represents. This is immune to
    the retokenization instability `scripts/validate_prefix_tokenization_stability.py`
    measures, because no candidate-dependent re-encoding ever happens: the shared
    prefix is not merely assumed constant across candidates, it is never computed more
    than once.
    """
    import torch

    text = context.lower() if casing == "lower" else context.upper()
    if frame:
        text = frame.format(prefix=text)
    token_ids = seed_ids + (tokenizer.encode(text) if text else [])
    if len(token_ids) == len(seed_ids) and not text:
        token_ids = seed_ids
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]
    probabilities = torch.softmax(logits.float(), dim=-1)
    projection_t = torch.as_tensor(projection, dtype=probabilities.dtype, device=probabilities.device)
    symbol_mass = (probabilities @ projection_t).cpu().numpy()
    total = float(symbol_mass.sum())
    if total <= 0:
        return uniform_prior()
    return symbol_mass / total


class TransformerPrior:
    """Next-symbol prior from a causal language model.

    The context is presented in lower case because natural-language models are trained on
    mixed-case text; the speller alphabet is upper case, so predicted first characters are
    folded back to upper case. Casing can be switched for sensitivity analysis.
    """

    def __init__(
        self,
        model_name: str,
        casing: str = "lower",
        device: str | None = None,
        frame: str | None = None,
        dtype: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if casing not in {"lower", "upper"}:
            raise ValueError(f"unknown casing: {casing}")
        if frame is not None and "{prefix}" not in frame:
            raise ValueError("frame must contain a {prefix} placeholder")
        self.model_name = model_name
        self.name = model_name
        self.casing = casing
        self.frame = frame
        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = getattr(torch, dtype) if dtype else None
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
            .to(device)
            .eval()
        )
        self._seed_ids = self._build_seed_ids()
        self._projection = self._build_projection()
        self._cache: dict[str, np.ndarray] = {}

    def _build_projection(self) -> "np.ndarray":
        """Map every vocabulary token onto the grid symbol its first character implies.

        Thin wrapper around the module-level ``build_symbol_projection`` so
        ``self._projection`` construction stays a single source of truth.
        """
        vocabulary_size = int(self.model.get_output_embeddings().weight.shape[0])
        return build_symbol_projection(self.tokenizer, vocabulary_size)

    def _build_seed_ids(self) -> list[int]:
        for candidate in (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id):
            if candidate is not None:
                return [int(candidate)]
        raise ValueError(f"{self.model_name} exposes no beginning-of-sequence token")

    def _prepare(self, context: str) -> str:
        """Apply the framing and casing under which the model is queried."""
        text = context.lower() if self.casing == "lower" else context.upper()
        return self.frame.format(prefix=text) if self.frame else text

    def prior(self, context: str) -> np.ndarray:
        """Prefix-locked next-symbol prior, cached because copy-spelling contexts repeat
        heavily. Uses prefix_locked_prior (module-level), which tokenizes `context`
        exactly once and never re-tokenizes it per candidate.
        """
        if context not in self._cache:
            self._cache[context] = prefix_locked_prior(
                context, self.tokenizer, self.model, self.device,
                self._seed_ids, self._projection, casing=self.casing, frame=self.frame,
            )
        return self._cache[context]

    def priors(self, contexts: list[str]) -> np.ndarray:
        return np.stack([self.prior(context) for context in contexts])

    def _legacy_full_sequence_prior(self, context: str) -> np.ndarray:
        """Score all 36 continuations as complete re-tokenized sequences (legacy method).

        Superseded by prefix_locked_prior (module-level function): this method
        re-tokenizes `context + candidate` independently per candidate and assumes the
        context's own tokenization is a stable prefix across all 36 candidates, which
        validate_prefix_tokenization_stability.py measured false 33.9% of the time for
        this study's primary prior. Retained only so scripts/compare_scorer_methods.py
        can report the magnitude of the correction; do not use for new analyses.
        """
        import torch

        text = self._prepare(context)
        sequences = [
            self._seed_ids
            + self.tokenizer.encode(text + (symbol.lower() if self.casing == "lower" else symbol))
            for symbol in SYMBOLS
        ]
        lengths = [len(sequence) for sequence in sequences]
        width = max(lengths)
        pad_id = self.tokenizer.eos_token_id or 0
        padded = torch.full((N_SYMBOLS, width), pad_id, dtype=torch.long, device=self.device)
        for row, sequence in enumerate(sequences):
            padded[row, : len(sequence)] = torch.tensor(sequence, device=self.device)
        attention = torch.zeros((N_SYMBOLS, width), dtype=torch.long, device=self.device)
        for row, length in enumerate(lengths):
            attention[row, :length] = 1
        with torch.no_grad():
            logits = self.model(padded, attention_mask=attention).logits
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        gathered = log_probabilities[:, :-1].gather(2, padded[:, 1:].unsqueeze(2)).squeeze(2)
        position = torch.arange(width - 1, device=self.device).unsqueeze(0)
        valid = position < (torch.tensor(lengths, device=self.device).unsqueeze(1) - 1)
        totals = (gathered * valid).sum(dim=1).cpu().numpy()
        weights = np.exp(totals - totals.max())
        return weights / weights.sum()

    def _sequence_log_probability(self, text: str) -> float:
        """Total log probability of a string under its canonical tokenization."""
        import torch

        token_ids = self._seed_ids + self.tokenizer.encode(text)
        input_ids = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]
        log_probabilities = torch.log_softmax(logits.float(), dim=-1)
        positions = torch.arange(len(token_ids) - 1, device=self.device)
        targets = torch.tensor(token_ids[1:], device=self.device)
        return float(log_probabilities[positions, targets].sum())

    def exact_prior(self, context: str) -> np.ndarray:
        """Legacy/comparison-only: score all 36 continuations explicitly, one full
        sequence per candidate.

        Calls ``_sequence_log_probability``, which shares the same retokenization
        assumption as ``_legacy_full_sequence_prior``: byte-pair tokenization means
        ``context + symbol`` is not a token-level extension of ``context``, so each
        continuation is re-tokenized as a complete sequence and the shared prefix is
        assumed (not guaranteed) to cancel when the 36 totals are normalized against one
        another. Superseded by prefix_locked_prior for production scoring; retained only
        for the Task 3 comparison against the corrected method.
        """
        text = self._prepare(context)
        scores = np.array(
            [
                self._sequence_log_probability(
                    text + (symbol.lower() if self.casing == "lower" else symbol)
                )
                for symbol in SYMBOLS
            ]
        )
        weights = np.exp(scores - scores.max())
        return weights / weights.sum()
