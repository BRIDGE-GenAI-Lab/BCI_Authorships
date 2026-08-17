# Model-Ladder and Classical N-Gram Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `study_bci_llm_authorship` prior ladder with 7 cross-family neural priors (Llama, Gemma, Mistral/Mixtral, DeepSeek) and 2 classical n-gram variants (Kneser-Ney smoothing, WikiText-103 corpus), then propagate the new priors through stats, figures, and manuscript so the "model capability doesn't change the picture" claim becomes a "model capability and architecture family don't change the picture" claim.

**Architecture:** Every new prior implements the existing `prior(context: str) -> np.ndarray` interface (`authorship/priors.py`), so `build_attribution.py`, and `run_stats.py`'s `sensitivity["prior_ladder"]` groupby, require **no changes** — they already iterate generically over whatever `prior_model` values appear in `priors.parquet`. The actual new work is: (1) two new local prior classes/corpora, (2) extending `LADDER` in `build_priors.py` with 7 cluster-tier specs scored on O2 via the existing `ingest_o2_priors.py` contract, (3) a new cross-family scale-check in `run_stats.py`, (4) a family-grouped redesign of the ladder figure, (5) manuscript prose/table/citation updates.

**Tech Stack:** Python, pandas/numpy, statsmodels, scipy, nltk (`nltk.lm.KneserNeyInterpolated`), transformers/torch (HF `AutoModelForCausalLM`), HMS O2 SLURM cluster (`gpu_sun`, L40S GPUs), matplotlib (`hochberg-figure-style` kit), pytest.

## Global Constraints

- **Qwen2.5-32B remains the primary prior** for headline NCF/phantom-agreement co-primary numbers — never change `PRIMARY_PRIOR` in `scripts/run_stats.py:32`.
- **`ngram5` (order-5, add-k, Brown corpus) remains the reference** for co-primary contrasts — it is never removed or renamed.
- All new neural priors are **base (non-instruct) checkpoints** — matches every existing ladder entry except the already-disclosed Qwen3.5-27B deviation.
- **Llama-3.1-70B substitutes for Llama-3.3-70B** (which only ships Instruct weights) — see design spec `2026-07-30-model-ladder-extension-design.md`.
- WikiText-103 corpus is capped at a **disclosed 20,000,000 characters**, never silently truncated further without a documented reason.
- Local git commits only; nothing pushed or submitted, per the standing convention for this study.
- Every new reference verified by DOI against Crossref (via the `ama-citation-zotero` skill) before it ships.

---

## Adjustment flagged for review before execution

The design spec called for a per-family Spearman scale check ("re-run the Spearman scale check within each family separately, not only pooled"). Most new families have only **2** members (Llama: 8B/70B; Gemma: 9B/27B; Mistral: 7B/47B; DeepSeek: 1 member) — too few points for a meaningful per-family correlation (n=2 gives a trivial rho of ±1 or is undefined). Task 7 below implements the closest honest equivalent instead:

1. A **pooled cross-family** Spearman check (NCF/phantom vs log-parameters) across all ~15 neural priors — larger n, more power than the original 8-prior, Qwen-only check.
2. A **descriptive per-family comparison** of NCF/phantom group means/ranges (not a per-family hypothesis test, since n is too small per family).
3. A **repeated truncation-sensitivity check** on the new, broader ladder — the same "what if we'd stopped early" check that caught the original 3B-truncation artefact, now applied cross-family.

If 3+ members per new family (for a real per-family correlation) is wanted instead, that requires adding a 3rd size per family (e.g., Llama-3.2-3B, Gemma-2-2b) — more O2 compute, out of the approved 7-model scope. Flagging here rather than silently deciding.

---

### Task 1: `family_of()` prior-family classifier

**Files:**
- Modify: `authorship/priors.py` (add function near the top, after `ALPHABET`/`N_SYMBOLS`)
- Test: `tests/test_priors.py` (add tests)

**Interfaces:**
- Produces: `family_of(prior_model: str) -> str`, importable as `from authorship.priors import family_of`. Later tasks (`run_stats.py`, `make_figures.py`) import this directly — do not re-derive family membership elsewhere.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_priors.py
from authorship.priors import family_of


def test_family_of_existing_ladder_entries():
    assert family_of("uniform") == "null"
    assert family_of("ngram5") == "classical"
    assert family_of("gpt2") == "gpt2"
    assert family_of("gpt2-large") == "gpt2"
    assert family_of("Qwen/Qwen2.5-1.5B") == "qwen"
    assert family_of("Qwen/Qwen2.5-32B") == "qwen"
    assert family_of("Qwen/Qwen3.5-27B") == "qwen"


def test_family_of_new_ladder_entries():
    assert family_of("ngram5_kn") == "classical"
    assert family_of("ngram5_wiki_kn") == "classical"
    assert family_of("meta-llama/Llama-3.1-8B") == "llama"
    assert family_of("meta-llama/Llama-3.1-70B") == "llama"
    assert family_of("google/gemma-2-9b") == "gemma"
    assert family_of("google/gemma-2-27b") == "gemma"
    assert family_of("mistralai/Mistral-7B-v0.3") == "mistral"
    assert family_of("mistralai/Mixtral-8x7B-v0.1") == "mistral"
    assert family_of("deepseek-ai/DeepSeek-V2-Lite") == "deepseek"


def test_family_of_rejects_unknown_model():
    import pytest

    with pytest.raises(ValueError, match="unknown prior family"):
        family_of("some-vendor/some-model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_priors.py -k family_of -v`
Expected: FAIL with `ImportError: cannot import name 'family_of'`

- [ ] **Step 3: Implement `family_of`**

```python
# add to authorship/priors.py, after the ALPHABET/N_SYMBOLS constants
_FAMILY_PREFIXES = (
    ("uniform", "null"),
    ("ngram", "classical"),
    ("gpt2", "gpt2"),
    ("Qwen/", "qwen"),
    ("meta-llama/", "llama"),
    ("google/gemma", "gemma"),
    ("mistralai/", "mistral"),
    ("deepseek-ai/", "deepseek"),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_priors.py -k family_of -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add authorship/priors.py tests/test_priors.py
git commit -m "Add family_of() prior-family classifier"
```

---

### Task 2: Kneser-Ney character n-gram prior (`ngram5_kn`)

**Files:**
- Modify: `authorship/priors.py` (add `CharNgramKNPrior` class after `CharNgramPrior`)
- Modify: `scripts/build_priors.py` (add `ngram5_kn` to `LADDER`, extend `build_model`)
- Test: `tests/test_priors.py` (add tests)

**Interfaces:**
- Consumes: `SYMBOLS`, `symbol_index`, `normalize_corpus`, `uniform_prior`, `load_brown_corpus` (all already defined in `authorship/priors.py`).
- Produces: `CharNgramKNPrior(order: int = 5, corpus_text: str | None = None)` with `.prior(context: str) -> np.ndarray` and `.name` attribute, same shape as `CharNgramPrior`. Task 3 reuses this class with a different corpus.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_priors.py
from authorship.priors import CharNgramKNPrior


def test_kn_prior_is_a_distribution():
    model = CharNgramKNPrior(order=3, corpus_text="THE CAT SAT ON THE MAT ")
    assert is_distribution(model.prior(""))
    assert is_distribution(model.prior("TH"))
    assert is_distribution(model.prior("ZZZZZ"))


def test_kn_prior_is_context_sensitive():
    model = CharNgramKNPrior(order=3, corpus_text="THE THE THE QAT " * 50)
    prior = model.prior("TH")
    assert prior[symbol_index("E")] > prior[symbol_index("Q")]


def test_kn_prior_assigns_positive_probability_to_every_observed_symbol():
    corpus = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 " * 30
    model = CharNgramKNPrior(order=3, corpus_text=corpus)
    prior = model.prior("AB")
    assert (prior > 0).all()


def test_kn_prior_rejects_empty_corpus():
    with pytest.raises(ValueError):
        CharNgramKNPrior(order=3, corpus_text="")


def test_kn_prior_name_is_distinct_from_add_k_prior():
    kn = CharNgramKNPrior(order=5, corpus_text="THE CAT SAT ON THE MAT ")
    add_k = CharNgramPrior(order=5, corpus_text="THE CAT SAT ON THE MAT ")
    assert kn.name != add_k.name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_priors.py -k kn_prior -v`
Expected: FAIL with `ImportError: cannot import name 'CharNgramKNPrior'`

- [ ] **Step 3: Implement `CharNgramKNPrior`**

```python
# add to authorship/priors.py, after CharNgramPrior, before load_brown_corpus
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
        window = context[-(self.order - 1):] if self.order > 1 else ""
        context_tuple = tuple(window)
        scores = np.array([self._model.score(symbol, context_tuple) for symbol in SYMBOLS])
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            return uniform_prior()
        return scores / total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_priors.py -k "kn_prior" -v`
Expected: PASS (5 tests). Note: `nltk.lm` requires the `punkt` resource only for word tokenization, not used here — no new `nltk.download()` calls needed since input is pre-split into characters.

- [ ] **Step 5: Wire `ngram5_kn` into the LADDER**

In `scripts/build_priors.py`, add to `LADDER` (after the existing `ngram5` entry) and extend `build_model`:

```python
# in LADDER list, after {"name": "ngram5", ...}:
    {"name": "ngram5_kn", "tier": "classical_kn", "kind": "ngram_kn", "parameters": 0},
```

```python
# in build_model(), add a branch before the ngram branch:
    if specification["kind"] == "ngram_kn":
        return CharNgramKNPrior(order=5)
```
And update the import line: `from authorship.priors import CharNgramKNPrior, CharNgramPrior, TransformerPrior, uniform_prior`.

- [ ] **Step 6: Verify build_priors.py still runs end-to-end for the local tiers**

Run: `python scripts/build_priors.py --only uniform,ngram5,ngram5_kn`
Expected: prints timings for all three, writes `output/intermediate/priors.parquet` with `ngram5_kn` rows present, and the `priors do not sum to one` check does not raise.

- [ ] **Step 7: Commit**

```bash
git add authorship/priors.py scripts/build_priors.py tests/test_priors.py
git commit -m "Add Kneser-Ney n-gram prior (ngram5_kn) on the Brown corpus"
```

---

### Task 3: WikiText-103-capped corpus and `ngram5_wiki_kn`

**Files:**
- Modify: `authorship/priors.py` (add `load_wikitext_corpus`)
- Modify: `scripts/build_priors.py` (add `ngram5_wiki_kn` to `LADDER`, extend `build_model`)
- Test: `tests/test_priors.py` (add tests)

**Interfaces:**
- Consumes: `CharNgramKNPrior` from Task 2.
- Produces: `load_wikitext_corpus(character_limit: int = 20_000_000, source: str | None = None) -> str`. The `source` parameter is a testing seam: pass a local path or literal text to avoid a real download in tests.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_priors.py
from authorship.priors import load_wikitext_corpus


def test_wikitext_loader_respects_character_cap(tmp_path):
    fake_corpus = "the quick brown fox jumps over the lazy dog. " * 1000
    source_file = tmp_path / "fake_wikitext.txt"
    source_file.write_text(fake_corpus)
    text = load_wikitext_corpus(character_limit=50, source=str(source_file))
    assert len(text) <= 50


def test_wikitext_loader_reads_full_text_when_under_the_cap(tmp_path):
    fake_corpus = "hello world "
    source_file = tmp_path / "fake_wikitext.txt"
    source_file.write_text(fake_corpus)
    text = load_wikitext_corpus(character_limit=10_000, source=str(source_file))
    assert text == fake_corpus
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_priors.py -k wikitext -v`
Expected: FAIL with `ImportError: cannot import name 'load_wikitext_corpus'`

- [ ] **Step 3: Implement `load_wikitext_corpus`**

Before writing this, verify the direct-download URL is still reachable (URLs rot): `curl -sI https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip | head -5`. If it 404s or times out, use the `datasets.load_dataset("wikitext", "wikitext-103-raw-v1")` fallback path below instead (`pip install datasets` if not already present) — do not silently substitute a different, smaller corpus without updating the design spec's corpus description.

```python
# add to authorship/priors.py, after CharNgramKNPrior
@functools.lru_cache(maxsize=1)
def load_wikitext_corpus(character_limit: int = 20_000_000, source: str | None = None) -> str:
    """Load a size-capped slice of WikiText-103 as a larger, more contemporary corpus
    than Brown (1961). Capped because a full character-level n-gram count table over the
    ~500M-character corpus is not memory-tractable; the cap is a disclosed methods
    decision, not silent truncation. `source` is a testing seam: a local path or literal
    text, bypassing the real download.
    """
    if source is not None:
        text = Path(source).read_text() if Path(source).exists() else source
        return text[:character_limit]

    import io
    import zipfile

    import requests

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("wikitext-103-raw/wiki.train.raw") as handle:
            text = handle.read().decode("utf-8", errors="ignore")
    return text[:character_limit]
```

Add `from pathlib import Path` to the top of `authorship/priors.py` if not already imported (it is not, currently).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_priors.py -k wikitext -v`
Expected: PASS (2 tests, both using the `source=` seam — no network access in the test suite).

- [ ] **Step 5: Wire `ngram5_wiki_kn` into the LADDER**

In `scripts/build_priors.py`:

```python
# in LADDER list, after the ngram5_kn entry:
    {"name": "ngram5_wiki_kn", "tier": "classical_kn_wiki", "kind": "ngram_kn_wiki", "parameters": 0},
```

```python
# in build_model(), add a branch:
    if specification["kind"] == "ngram_kn_wiki":
        return CharNgramKNPrior(order=5, corpus_text=load_wikitext_corpus())
```
Update the import: `from authorship.priors import CharNgramKNPrior, CharNgramPrior, TransformerPrior, load_wikitext_corpus, uniform_prior`.

- [ ] **Step 6: Run the real (uncapped-source) build once locally to confirm the live download path works**

Run: `python scripts/build_priors.py --only ngram5_wiki_kn`
Expected: downloads the zip (one-time, cached by `functools.lru_cache` for the process lifetime only — not persisted to disk, so this download re-runs on every fresh `build_priors.py` invocation; note this as a known cost, not a bug, since the shard itself is cached at `output/intermediate/prior_shards/ngram5_wiki_kn.parquet` afterward and `build_priors.py` skips already-cached shards). Fits the model, writes the shard, prints a timing line.

- [ ] **Step 7: Commit**

```bash
git add authorship/priors.py scripts/build_priors.py tests/test_priors.py
git commit -m "Add WikiText-103-capped Kneser-Ney n-gram prior (ngram5_wiki_kn)"
```

---

### Task 4: Extend `LADDER` with the 7 cross-family neural priors

**Files:**
- Modify: `scripts/build_priors.py`
- Test: `tests/test_priors.py` (LADDER integrity check — put it in a new small test file since it tests the script, not the package)
- Test: `tests/test_build_priors_ladder.py` (new file)

**Interfaces:**
- Consumes: `family_of` from Task 1.
- Produces: the 7 new `LADDER` entries, each `{"name", "tier", "kind": "cluster", "parameters"}`, matching the existing Qwen cluster-tier entry shape exactly so `ingest_o2_priors.py` needs no changes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_priors_ladder.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_priors import LADDER
from authorship.priors import family_of


def test_ladder_names_are_unique():
    names = [spec["name"] for spec in LADDER]
    assert len(names) == len(set(names))


def test_every_ladder_entry_resolves_to_a_known_family():
    for spec in LADDER:
        family_of(spec["name"])  # raises ValueError if unknown; test fails on exception


def test_ladder_contains_the_seven_new_cross_family_models():
    names = {spec["name"] for spec in LADDER}
    expected = {
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-v0.1",
        "deepseek-ai/DeepSeek-V2-Lite",
    }
    assert expected <= names


def test_new_cluster_entries_are_marked_for_cluster_scoring():
    by_name = {spec["name"]: spec for spec in LADDER}
    for name in (
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-v0.1",
        "deepseek-ai/DeepSeek-V2-Lite",
    ):
        assert by_name[name]["kind"] == "cluster"
        assert by_name[name]["parameters"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_priors_ladder.py -v`
Expected: FAIL on `test_ladder_contains_the_seven_new_cross_family_models` (the other 3 pass trivially against the current 11-entry ladder, that's fine — they're regression guards for this task, not just for the new entries).

- [ ] **Step 3: Add the 7 new entries to `LADDER`**

```python
# in scripts/build_priors.py, append to LADDER after the Qwen3.5-27B entry:
    {
        "name": "meta-llama/Llama-3.1-8B",
        "tier": "cross_family_8b",
        "kind": "cluster",
        "parameters": 8_000_000_000,
    },
    {
        "name": "google/gemma-2-9b",
        "tier": "cross_family_9b",
        "kind": "cluster",
        "parameters": 9_000_000_000,
    },
    {
        "name": "mistralai/Mistral-7B-v0.3",
        "tier": "cross_family_7b",
        "kind": "cluster",
        "parameters": 7_000_000_000,
    },
    {
        "name": "deepseek-ai/DeepSeek-V2-Lite",
        "tier": "cross_family_moe_16b_total",
        "kind": "cluster",
        "parameters": 15_700_000_000,
    },
    {
        "name": "google/gemma-2-27b",
        "tier": "cross_family_27b",
        "kind": "cluster",
        "parameters": 27_000_000_000,
    },
    {
        "name": "mistralai/Mixtral-8x7B-v0.1",
        "tier": "cross_family_moe_47b_total",
        "kind": "cluster",
        "parameters": 46_700_000_000,
    },
    {
        "name": "meta-llama/Llama-3.1-70B",
        "tier": "cross_family_70b",
        "kind": "cluster",
        "parameters": 70_000_000_000,
    },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_priors_ladder.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full local test suite to confirm nothing else broke**

Run: `pytest tests/ -v`
Expected: PASS. `build_model()` already raises `SystemExit` for any `kind == "cluster"` spec when run locally (`scripts/build_priors.py:53-56`), so these 7 entries are correctly inert until Task 5 scores them on O2 — running `python scripts/build_priors.py` locally with no `--only` filter would hit that `SystemExit` for the first cluster entry, same as it already does for the existing Qwen cluster entries; this is expected, existing behavior, not a new failure mode.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_priors.py tests/test_build_priors_ladder.py
git commit -m "Extend LADDER with 7 cross-family neural priors (Llama, Gemma, Mistral/Mixtral, DeepSeek)"
```

---

### Task 5: Score the 7 new priors on O2 and retrieve the shards

This task is cluster ops, not local TDD — there is no failing-test cycle for a SLURM job. Each step below has an explicit, checkable expected outcome instead.

**Files:** none locally until Step 8 (shards land in `output/intermediate/prior_shards/`, handled by Task 6's ingest step).

- [ ] **Step 1: Establish the O2 connection**

Invoke the `using-o2` skill to set up the ControlMaster session to `alg7274@o2.hms.harvard.edu` — do not hand-roll a new SSH connection.

- [ ] **Step 2: Read the existing remote scoring script before editing it**

```bash
ssh o2 'cat /n/scratch/users/a/alg7274/authorship/score_priors_o2.py'
ssh o2 'cat /n/scratch/users/a/alg7274/authorship/score_priors.sbatch'
```
Confirm its spec-list shape mirrors `scripts/build_priors.py`'s `LADDER` (per the 2026-07-26 O2 extension notes: it reuses `idrift_env` — torch 2.6.0+cu124, transformers 5.14.1 — and reads a spec list of `{name, tier, parameters}` dicts, writing one `priors_<index>.jsonl` per array task). Do not assume the exact variable names without reading them first.

- [ ] **Step 3: Extend the remote spec list with the 7 new entries**

Edit the remote script's spec list to add the same 7 entries from Task 4, Step 3 (name/tier/parameters only — `kind` is a local-only concept, the remote script does not need it). Split into two arrays by GPU requirement, since a single `sbatch --array` job shares one `--gres` request across all indices:

- **Tier A (single L40S GPU each):** `meta-llama/Llama-3.1-8B`, `google/gemma-2-9b`, `mistralai/Mistral-7B-v0.3`, `deepseek-ai/DeepSeek-V2-Lite` — by analogy to the existing 14B run (2m51s on a single GPU), expect a few minutes each.
- **Tier B (multi-GPU):** `google/gemma-2-27b` (2 GPUs, by analogy to the existing 32B run at 6m15s), `mistralai/Mixtral-8x7B-v0.1` (2 GPUs, ~94GB fp16 weights), `meta-llama/Llama-3.1-70B` (4 GPUs / a full `gpu_sun` node, ~140GB fp16 weights).

- [ ] **Step 4: Confirm gated-repo access before submitting**

Llama and Gemma are gated HuggingFace repos requiring license acceptance. Confirm an `HF_TOKEN` is set in the O2 environment and that the token's account has accepted the Llama 3.1 and Gemma 2 license terms:
```bash
ssh o2 'source /n/scratch/users/a/alg7274/idrift_env/bin/activate && python -c "from huggingface_hub import HfApi; HfApi().model_info(\"meta-llama/Llama-3.1-8B\")"'
```
Expected: no `GatedRepoError`. If it raises one, this is a human step (accept the license on huggingface.co) — do not attempt to work around it.

- [ ] **Step 5: Smoke-test each new model before the full 951-context run**

The existing `ingest_o2_priors.py` already skips files containing `"smoke"` in the name (line 30: `if "smoke" in path.name: continue`), confirming a smoke-test convention already exists in this pipeline. Run each new model against a small context sample (reuse `scripts/validate_prior.py`'s `SAMPLE_CONTEXTS` list) before committing to the full run:
```bash
ssh o2 'cd /n/scratch/users/a/alg7274/authorship && sbatch --job-name=smoke_llama8b score_smoke.sbatch meta-llama/Llama-3.1-8B'
```
(Reuse whatever smoke-test entry point Step 2 revealed; if none exists yet, run `scripts/validate_prior.py <model_name>` remotely for one context batch as the smoke test instead — it works for any `AutoModelForCausalLM`-loadable model with no changes.) Expected: completes without an architecture-specific error (e.g., Gemma-2's logit soft-capping, Mixtral/DeepSeek MoE routing producing malformed logits) before the expensive full run is submitted.

- [ ] **Step 6: Submit the two array jobs**

```bash
ssh o2 'cd /n/scratch/users/a/alg7274/authorship && sbatch --gres=gpu:l40s:1 --array=0-3 score_priors_tier_a.sbatch'
ssh o2 'cd /n/scratch/users/a/alg7274/authorship && sbatch --gres=gpu:l40s:4 --array=0-2 score_priors_tier_b.sbatch'
```
Record both job IDs.

- [ ] **Step 7: Poll until both jobs complete**

```bash
ssh o2 'sacct -j <tier_a_job_id>,<tier_b_job_id> --format=JobID,JobName,State,Elapsed'
```
Use `Monitor` (not a manual sleep-poll loop) if the harness supports it, or a bounded `until`-loop via Bash. Expected: all 7 array tasks reach `COMPLETED`. If any reach `FAILED` or `OUT_OF_MEMORY`, read the SLURM log before resubmitting — do not silently retry with reduced scope.

- [ ] **Step 8: Retrieve the output shards**

```bash
mkdir -p /tmp/o2_priors
scp o2:/n/scratch/users/a/alg7274/authorship/priors_*.jsonl /tmp/o2_priors/
```
Expected: 7 new `priors_*.jsonl` files locally, one per new model, matching the `ingest_o2_priors.py` input contract (each line a JSON object with `prior_model`, `prior_tier`, `prior_parameters`, `context_prefix`, `p_lm`).

No commit for this task — it produces no repo changes, only files under `/tmp/o2_priors/` consumed by Task 6.

---

### Task 6: Ingest the new shards and regression-guard the headline numbers

**Files:**
- No code changes expected (this task runs existing scripts); if `ingest_o2_priors.py` needs a tweak, keep it minimal and covered by the existing validation checks it already performs (context set / alphabet / sum-to-one / finiteness).

- [ ] **Step 1: Snapshot the current headline numbers before touching anything**

```bash
cp output/stats_digest.json /tmp/stats_digest_before_extension.json
```

- [ ] **Step 2: Ingest the 7 new shards**

Run: `python scripts/ingest_o2_priors.py /tmp/o2_priors`
Expected: 7 lines printed (one per model), each showing `n=951` (the unique-context count), no `SystemExit` raised. If the context set differs from `expected_contexts`, the ingest script already raises `SystemExit(f"{name}: context set differs from the local ladder")` — do not bypass this check.

- [ ] **Step 3: Rebuild the concatenated priors table**

Run: `python scripts/build_priors.py --only ngram5_kn,ngram5_wiki_kn` (the two local classical entries from Tasks 2-3, if not already cached) then run the full concatenation step. Since `build_priors.py`'s `main()` already concatenates every shard in `prior_shards/` regardless of `--only` (lines 108-113), a final no-filter invocation will raise `SystemExit` on hitting the first still-unscored `kind: cluster` spec if any remain unscored — at this point all 7 should be ingested, so re-run without `--only` to get the fresh `priors.parquet`:

Run: `python scripts/build_priors.py`
Expected: completes without `SystemExit`, reports `rows` in the printed JSON matching `951 contexts * 18 priors` (9 original + 9 new = 18; verify the exact count against `len(LADDER)`).

- [ ] **Step 4: Rebuild attribution and stats**

Run: `python scripts/build_attribution.py && python scripts/run_stats.py`
Expected: `build_attribution.py`'s existing invariant checks pass (`each prior model must cover every selection`, `phantom agreement and prior capture must be mutually exclusive` — lines 91-94). `run_stats.py` completes and rewrites `output/stats_digest.json`.

- [ ] **Step 5: Verify the headline co-primary numbers are unchanged**

```python
import json

before = json.load(open("/tmp/stats_digest_before_extension.json"))
after = json.load(open("output/stats_digest.json"))
for key in ("coprimary_1_neural_contribution_fraction", "coprimary_2_phantom_agreement"):
    assert before[key]["estimate"] == after[key]["estimate"], f"{key} point estimate changed"
    assert before[key]["ci_low"] == after[key]["ci_low"]
    assert before[key]["ci_high"] == after[key]["ci_high"]
print("headline co-primary numbers unchanged")
```
Expected: prints the confirmation line. If any assertion fails, stop — something in Tasks 1-5 altered the primary-prior computation, which is out of scope for this extension and must be root-caused before continuing (do not proceed to the manuscript task with silently shifted headline numbers).

- [ ] **Step 6: Commit the regenerated intermediate outputs** (if intermediate outputs are tracked in this repo — check `git status` first; if `output/` is gitignored, skip this step, there is nothing to commit)

```bash
git status --short output/
```

---

### Task 7: Cross-family scale-check in `run_stats.py`

**Files:**
- Modify: `scripts/run_stats.py`
- Test: `tests/test_stats_helpers.py`

**Interfaces:**
- Consumes: `family_of` from Task 1, `spearmanr` from `scipy.stats` (already a project dependency, used in `scripts/validate_prior.py`).
- Produces: `digest["sensitivity"]["cross_family_scale_check"]`, a new top-level key under `sensitivity` (alongside the existing `prior_ladder` key), containing the pooled correlation, per-family descriptive means, and the truncation-sensitivity repeat described in the "Adjustment flagged for review" section above.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_stats_helpers.py
import numpy as np

from run_stats import cross_family_scale_check


def ladder_frame():
    """A toy ladder: 3 families, enough points per family to exercise grouping,
    with a deliberately flat NCF/phantom relationship (no true trend) so the test
    checks structure and doesn't assert a spurious correlation direction.
    """
    rows = []
    specs = [
        ("uniform", "null", 0, 1.00, 0.00),
        ("ngram5", "classical", 0, 0.87, 0.08),
        ("gpt2", "gpt2", 124_000_000, 0.88, 0.07),
        ("Qwen/Qwen2.5-1.5B", "qwen", 1_500_000_000, 0.86, 0.08),
        ("Qwen/Qwen2.5-14B", "qwen", 14_000_000_000, 0.86, 0.07),
        ("Qwen/Qwen2.5-32B", "qwen", 32_000_000_000, 0.86, 0.08),
        ("meta-llama/Llama-3.1-8B", "llama", 8_000_000_000, 0.87, 0.08),
        ("meta-llama/Llama-3.1-70B", "llama", 70_000_000_000, 0.86, 0.07),
    ]
    for name, family, params, ncf, phantom in specs:
        for participant in range(10):
            rows.append(
                {
                    "prior_model": name,
                    "prior_parameters": params,
                    "study_participant_id": f"P{participant}",
                    "ncf": ncf,
                    "phantom_agreement": phantom,
                }
            )
    return pd.DataFrame(rows)


def test_cross_family_scale_check_reports_pooled_and_per_family_blocks():
    result = cross_family_scale_check(ladder_frame())
    assert "pooled" in result
    assert "spearman_rho_ncf_vs_log_params" in result["pooled"]
    assert "spearman_rho_phantom_vs_log_params" in result["pooled"]
    assert "by_family" in result
    assert set(result["by_family"]) == {"qwen", "llama"}  # null/classical/gpt2 excluded: no size variation
    assert "truncated_below_10b" in result


def test_cross_family_scale_check_excludes_zero_parameter_priors_from_the_correlation():
    frame = ladder_frame()
    result = cross_family_scale_check(frame)
    # uniform, ngram5 have 0 parameters and must not enter a log-parameter correlation
    assert result["pooled"]["n_priors"] == frame.loc[frame["prior_parameters"] > 0, "prior_model"].nunique()
```

Add `import pandas as pd` near the top of `tests/test_stats_helpers.py` if not already present (check first).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stats_helpers.py -k cross_family -v`
Expected: FAIL with `ImportError: cannot import name 'cross_family_scale_check'`

- [ ] **Step 3: Implement `cross_family_scale_check`**

```python
# add to scripts/run_stats.py, after fit_binary(), before main()
from scipy.stats import spearmanr

from authorship.priors import family_of


def _ladder_points(ladder: pd.DataFrame) -> pd.DataFrame:
    """One row per prior_model with its parameter count and participant-weighted
    outcome means, restricted to priors with parameters > 0 (excludes uniform/ngram,
    which have no capability axis to correlate against).
    """
    points = []
    for name, group in ladder.groupby("prior_model"):
        parameters = int(group["prior_parameters"].iloc[0])
        if parameters <= 0:
            continue
        points.append(
            {
                "prior_model": name,
                "family": family_of(name),
                "log_parameters": float(np.log10(parameters)),
                "ncf": participant_mean(group, "ncf"),
                "phantom_agreement": participant_mean(group, "phantom_agreement"),
            }
        )
    return pd.DataFrame(points)


def cross_family_scale_check(ladder: pd.DataFrame) -> dict:
    """Does capability predict attribution once every architecture family is pooled,
    and does any single family behave differently on its own? Extends the original
    Qwen-only scale check (which was fooled by a ladder truncated at 3B) across
    families, and repeats the same truncation check on the broader ladder.
    """
    points = _ladder_points(ladder)

    def spearman(column: str, subset: pd.DataFrame) -> dict:
        if len(subset) < 3:
            return {"n": int(len(subset)), "rho": None, "p_value": None}
        statistic = spearmanr(subset["log_parameters"], subset[column])
        return {
            "n": int(len(subset)),
            "rho": float(statistic.statistic),
            "p_value": float(statistic.pvalue),
        }

    pooled = {
        "n_priors": int(points["prior_model"].nunique()),
        "spearman_rho_ncf_vs_log_params": spearman("ncf", points),
        "spearman_rho_phantom_vs_log_params": spearman("phantom_agreement", points),
    }

    by_family = {
        family: {
            "n_priors": int(len(group)),
            "ncf_mean": float(group["ncf"].mean()),
            "ncf_range": [float(group["ncf"].min()), float(group["ncf"].max())],
            "phantom_agreement_mean": float(group["phantom_agreement"].mean()),
            "phantom_agreement_range": [
                float(group["phantom_agreement"].min()),
                float(group["phantom_agreement"].max()),
            ],
        }
        for family, group in points.groupby("family")
        if len(group) >= 2
    }

    truncated = points[points["log_parameters"] < 10.0]  # 10^10 = 10B parameters
    truncated_below_10b = {
        "n_priors": int(len(truncated)),
        "spearman_rho_ncf_vs_log_params": spearman("ncf", truncated),
        "spearman_rho_phantom_vs_log_params": spearman("phantom_agreement", truncated),
    }

    return {"pooled": pooled, "by_family": by_family, "truncated_below_10b": truncated_below_10b}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stats_helpers.py -k cross_family -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire it into `main()`**

In `scripts/run_stats.py`'s `main()`, after `sensitivity["prior_ladder"] = {...}` (around line 365), add:

```python
    sensitivity["cross_family_scale_check"] = cross_family_scale_check(ladder)
```

- [ ] **Step 6: Run the full stats pipeline and inspect the new block**

Run: `python scripts/run_stats.py` and inspect `output/stats_digest.json`'s new `sensitivity.cross_family_scale_check` key. Sanity-check: `by_family` should now include `qwen`, `llama`, `gemma`, `mistral` (each with ≥2 members); `deepseek` will be absent (only 1 member, filtered by the `len(group) >= 2` guard) — this is expected, not a bug.

- [ ] **Step 7: Run the full local test suite**

Run: `pytest tests/ -v`
Expected: PASS, all tests including Tasks 1-6's new tests and the pre-existing 47.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_stats.py tests/test_stats_helpers.py
git commit -m "Add cross-family scale-check: pooled correlation, per-family descriptives, truncation-sensitivity repeat"
```

---

### Task 8: Family-grouped ladder figure

**Files:**
- Modify: `scripts/make_figures.py`

**Interfaces:**
- Consumes: `family_of` from Task 1, `CAT` categorical palette from `hochberg_kit` (`['#C0392B', '#1B7340', '#6A4C93', '#C65102', '#2C5FAA', '#117A8B', '#A61E4D']` — 7 colors for the 7 families: null, classical, gpt2, qwen, llama, gemma, mistral, deepseek is an 8th family needing an 8th color, see Step 1).

- [ ] **Step 1: Extend `LADDER_ORDER` and `LADDER_LABEL`, add a family color map**

```python
# in scripts/make_figures.py, replace the existing LADDER_ORDER/LADDER_LABEL block
from authorship.priors import family_of  # add to the existing import block

LADDER_ORDER = [
    "uniform", "ngram5", "ngram5_kn", "ngram5_wiki_kn", "gpt2", "gpt2-large",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-14B", "Qwen/Qwen3.5-27B", "Qwen/Qwen2.5-32B",
    "mistralai/Mistral-7B-v0.3", "deepseek-ai/DeepSeek-V2-Lite", "meta-llama/Llama-3.1-8B",
    "google/gemma-2-9b", "google/gemma-2-27b", "mistralai/Mixtral-8x7B-v0.1", "meta-llama/Llama-3.1-70B",
]
LADDER_LABEL = {
    "uniform": "Uniform",
    "ngram5": "5-gram",
    "ngram5_kn": "5-gram KN",
    "ngram5_wiki_kn": "5-gram KN\n(WikiText)",
    "gpt2": "GPT-2 124M",
    "gpt2-large": "GPT-2 774M",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5 1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5 3B",
    "Qwen/Qwen2.5-14B": "Qwen2.5 14B",
    "Qwen/Qwen3.5-27B": "Qwen3.5 27B",
    "Qwen/Qwen2.5-32B": "Qwen2.5 32B",
    "mistralai/Mistral-7B-v0.3": "Mistral 7B",
    "deepseek-ai/DeepSeek-V2-Lite": "DeepSeek-V2-Lite",
    "meta-llama/Llama-3.1-8B": "Llama-3.1 8B",
    "google/gemma-2-9b": "Gemma-2 9B",
    "google/gemma-2-27b": "Gemma-2 27B",
    "mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B",
    "meta-llama/Llama-3.1-70B": "Llama-3.1 70B",
}
FAMILY_COLOR = {
    "null": CONTEXT,
    "classical": "#6A4C93",
    "gpt2": "#117A8B",
    "qwen": KEY,
    "llama": "#C65102",
    "gemma": "#1B7340",
    "mistral": "#A61E4D",
    "deepseek": "#C0392B",
}
```

- [ ] **Step 2: Replace `bars_with_ci`'s single-color bars with per-bar family colors in `figure2`**

`bars_with_ci` currently takes one `color` for the whole bar series (`scripts/make_figures.py:57-74`). Add an optional `colors` (list, one per bar) parameter that overrides `color` when given:

```python
def bars_with_ci(ax, labels, estimates, lows, highs, color=KEY, colors=None, ylabel="", rotation=0):
    positions = np.arange(len(labels))
    ax.bar(positions, estimates, color=(colors if colors is not None else color), width=0.62, zorder=2)
    ax.errorbar(
        positions,
        estimates,
        yerr=[np.array(estimates) - np.array(lows), np.array(highs) - np.array(estimates)],
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha="center" if rotation == 0 else "right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
```

- [ ] **Step 3: Update `figure2` to pass family colors and a family legend**

In `figure2(digest)`, after computing `names`/`labels`:

```python
    families = [family_of(name) for name in names]
    colors = [FAMILY_COLOR[family] for family in families]
```

Pass `colors=colors` to each `bars_with_ci(...)` call in panels (a) and (b) (remove the fixed `color=` argument on those two calls, keep panel (c)'s line-plot as-is since it's already legended by series, not by prior). Add a family legend below the figure, alongside the existing accuracy-gain/prior-capture legend, listing each family present with its `FAMILY_COLOR` swatch (use the existing `legend_below`/`swatch` helpers already imported at the top of the file).

- [ ] **Step 4: Regenerate the figures and inspect visually**

Run: `python scripts/make_figures.py`
Expected: completes without error, `output/figures/Figure2.pdf` (and `.png` if `save()` writes both) now shows 18 bars grouped/colored by family, legible at the existing figure width (8.4 inches) — if font size 5.8 (line 171/180) makes 18 rotated labels illegible, increase the figure width in the `plt.subplots(1, 3, figsize=(8.4, 3.0))` call (e.g., to `(10.5, 3.2)`) rather than shrinking font further; check the journal's figure-width limit in `manuscript/methods.md` or the npj Digital Medicine author guidelines before finalizing the width.

- [ ] **Step 5: Add a smoke test**

```python
# tests/test_make_figures_ladder.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_figures import FAMILY_COLOR, LADDER_LABEL, LADDER_ORDER
from authorship.priors import family_of


def test_every_ladder_order_entry_has_a_label():
    for name in LADDER_ORDER:
        assert name in LADDER_LABEL


def test_every_ladder_order_entry_resolves_to_a_colored_family():
    for name in LADDER_ORDER:
        assert family_of(name) in FAMILY_COLOR
```

Run: `pytest tests/test_make_figures_ladder.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/make_figures.py tests/test_make_figures_ladder.py
git commit -m "Redesign ladder figure to group/color 18 priors by architecture family"
```

---

### Task 9: Manuscript, table, citations, and final regression check

This task edits prose in `manuscript/*.md` and cannot be pre-filled with exact numbers now — the real values only exist after Tasks 5-7 run. Each step below names the exact file/line to change and the exact source-of-truth to pull the replacement number from; do not fabricate a number that isn't read from `output/stats_digest.json` or `output/attribution_summary.json`.

**Files:**
- Modify: `manuscript/abstract.md`, `manuscript/methods.md`, `manuscript/results.md`, `manuscript/discussion.md`, `manuscript/cover_letter.md`, `manuscript/supplement.md`, `manuscript/references.md`, `manuscript/references.bib`
- Run: `scripts/assemble_manuscript.py`, `scripts/build_references.py`

- [ ] **Step 1: Verify reference count/word budget headroom before adding content**

Current packet: body 3,976 words, refs ≤60 (npj Digital Medicine cap), 18 refs used. Adding 5 new references (Llama 3, Gemma 2, Mistral 7B, Mixtral, DeepSeek-V2) brings refs to 23 — within cap. Run `python scripts/assemble_manuscript.py` once now (before prose edits) to get the current baseline word count, so Step 6's post-edit count has something to diff against.

- [ ] **Step 2: Update `manuscript/methods.md`**

Replace the "ladder of six priors" description (`manuscript/methods.md:54-58`):
> "A ladder of six priors was used: a uniform null; a character 5-gram with add-k smoothing and backoff, fitted on the Brown corpus through NLTK 3.9.2; and four... five from Alibaba Cloud, Qwen2.5 at 1.5, 3, 14 and 32.5 billion parameters and the 2026-generation Qwen3.5 at 27 billion."

with a description covering all 18 entries: the uniform null; three classical n-grams (add-k/Brown, Kneser-Ney/Brown, Kneser-Ney/WikiText-103 capped at 20,000,000 characters); GPT-2 at two sizes; and 13 neural models across five families (Qwen2.5/Qwen3.5, Llama 3.1, Gemma 2, Mistral/Mixtral, DeepSeek-V2-Lite), naming each checkpoint and citing its technical report. State explicitly that Llama-3.1-70B was scored instead of Llama-3.3-70B because Meta released 3.3 only as an instruct-tuned checkpoint (mirrors the existing disclosure pattern already used for Qwen3.5-27B's instruct status at `manuscript/methods.md:58`).

- [ ] **Step 3: Update `manuscript/results.md`**

Replace "Across eight priors spanning a character 5-gram to a 32.5-billion-parameter model, phantom agreement varied only between 7.2% and 8.6%..." (`manuscript/results.md:64-65`, mirrored in `manuscript/discussion.md:23-27`) with numbers pulled from `output/stats_digest.json`'s new `sensitivity.cross_family_scale_check` block:
- Total neural prior count → `sensitivity.cross_family_scale_check.pooled.n_priors`
- Pooled Spearman rho/p for NCF and phantom agreement → `sensitivity.cross_family_scale_check.pooled.spearman_rho_*`
- Range of phantom agreement across the full ladder → compute `min`/`max` of `phantom_agreement.estimate` across `sensitivity.prior_ladder` for all non-zero-parameter priors
- Per-family means/ranges → `sensitivity.cross_family_scale_check.by_family`

Add one new sentence reporting the truncated-below-10B check (`sensitivity.cross_family_scale_check.truncated_below_10b`) as the direct extension of the existing truncation-artefact finding, now shown to hold across families too, not just within Qwen.

- [ ] **Step 4: Update `manuscript/discussion.md`**

Extend the "Model capability did not change the picture" paragraph (`manuscript/discussion.md:23-29`) to "Model capability and architecture family did not change the picture," citing the pooled cross-family result. Keep the existing sentence about the 3B-truncation artefact; add one sentence noting the same check was repeated on the full cross-family ladder (per Task 7) and did not reintroduce a spurious trend. Update the Limitations section (`manuscript/discussion.md:84`, "the ladder reached 32.5 billion parameters and included one 2026-generation model, but not...") to reflect the new largest model (Llama-3.1-70B) and drop or revise this limitation if it's now substantially addressed — read the full limitation sentence in context before editing, since only the excerpt was grepped here.

- [ ] **Step 5: Update Table 2 in `manuscript/manuscript.md`**

Add the 9 new rows (2 new classical n-gram variants — `ngram5_kn`, `ngram5_wiki_kn` — plus the 7 new neural cross-family entries) to the existing 9-row table (`manuscript/manuscript.md:481-496`), each populated from `output/stats_digest.json`'s `sensitivity.prior_ladder[<prior_model>]` and `output/attribution_summary.json`'s `by_model[<prior_model>]`, matching the exact column format already used (parameters, NCF with 95% CI, phantom agreement % with 95% CI, prior capture %, accuracy %). Update the table caption sentence ("Across the seven neural priors, no attribution measure varied reliably with parameter count") to the new neural-prior count.

- [ ] **Step 6: Update Figure 2's legend**

`manuscript/manuscript.md`'s Figure 2 legend ("from a uniform null and a character 5-gram to a 32.5-billion-parameter model... across the seven neural priors") needs the family-grouping language added, matching Task 8's figure redesign: mention that priors are grouped/colored by architecture family and that the largest model is now Llama-3.1-70B.

- [ ] **Step 7: Update `manuscript/abstract.md` and `manuscript/cover_letter.md`**

Both currently say "priors from a character 5-gram to 32.5 billion parameters" / "across eight priors spanning five orders of magnitude" — update the parameter range to "...to 70 billion parameters" and the prior count/orders-of-magnitude language to match the new ladder size, pulling exact figures from Step 3's sourced numbers, not re-deriving independently.

- [ ] **Step 8: Add and verify the 5 new references**

Invoke the `ama-citation-zotero` skill to add and DOI-verify: Llama 3 (Grattafiori et al. 2024), Gemma 2 (Gemma Team 2024), Mistral 7B (Jiang et al. 2023), Mixtral (Jiang et al. 2024), DeepSeek-V2 (DeepSeek-AI 2024). Do not hand-write these citations — every reference in this study has been DOI-verified against Crossref before shipping (per the design spec's scope guardrails and the existing house convention).

- [ ] **Step 9: Reassemble and check the word/reference budget**

Run: `python scripts/assemble_manuscript.py`
Expected: prints word counts per section; body total must stay under npj Digital Medicine's cap (check the exact cap referenced in `manuscript/methods.md` or the original design spec — 3,976 was the pre-extension baseline, confirm the post-edit total against whatever limit was used to size the original submission). If over budget, trim Discussion prose elsewhere rather than cutting the new cross-family content — that content is the entire point of this extension.

Run: `python scripts/build_references.py` (or whatever the reference-numbering step is called — confirm the exact script name from `manuscript/refs_numbering.json`'s provenance if `build_references.py` doesn't already handle renumbering) to make sure reference numbers stay sequential after the 5 additions.

- [ ] **Step 10: Grep for stale ladder language**

```bash
grep -n "eight prior\|seven neural prior\|five orders of magnitude\|32.5-billion-parameter model\|six priors" manuscript/*.md
```
Expected: no remaining matches outside of intentionally-preserved historical statements (e.g., a sentence explicitly describing "the original 6-tier ladder" as a past truncation-artefact episode, which should stay as accurate history, not be rewritten). Every other match must be updated to the new ladder size.

- [ ] **Step 11: Rebuild the `_submission_ready/` packet**

Follow whatever packet-build step was used for the original submission-ready packet (check `_submission_ready/study_bci_llm_authorship/0_README` for the exact commands) to regenerate the docx/pdf outputs from the updated `.md` sources.

- [ ] **Step 12: Full test suite and final commit**

```bash
pytest tests/ -v
```
Expected: PASS, all tests (pre-existing 47 + all new tests from Tasks 1-8).

```bash
git add manuscript/ _submission_ready/study_bci_llm_authorship/ output/
git commit -m "Extend manuscript to the 18-prior cross-family ladder: prose, Table 2, Figure 2 legend, 5 new references"
```
