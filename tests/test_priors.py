import numpy as np
import pytest

from authorship.grid import symbol_index
from authorship.priors import CharNgramPrior, uniform_prior


def is_distribution(vector):
    return (
        vector.shape == (36,)
        and abs(float(vector.sum()) - 1.0) < 1e-6
        and bool((vector >= 0).all())
    )


def test_uniform_prior_is_flat():
    prior = uniform_prior()
    assert is_distribution(prior)
    assert np.allclose(prior, 1 / 36)


def test_ngram_prior_is_a_distribution():
    model = CharNgramPrior(order=3, corpus_text="THE CAT SAT ON THE MAT ")
    assert is_distribution(model.prior(""))
    assert is_distribution(model.prior("TH"))
    assert is_distribution(model.prior("ZZZZZ"))


def test_ngram_prior_is_context_sensitive():
    model = CharNgramPrior(order=3, corpus_text="THE THE THE QAT " * 50)
    prior = model.prior("TH")
    assert prior[symbol_index("E")] > prior[symbol_index("Q")]


def test_ngram_prior_backs_off_to_shorter_context_when_unseen():
    model = CharNgramPrior(order=5, corpus_text="ABCDE ABCDX ZZZZE " * 20)
    prior = model.prior("QQQQ")
    assert is_distribution(prior)
    assert prior.max() < 1.0


def test_ngram_prior_never_assigns_zero_probability():
    model = CharNgramPrior(order=3, corpus_text="AAAA ")
    prior = model.prior("AA")
    assert (prior > 0).all()


def test_ngram_prior_rejects_empty_corpus():
    with pytest.raises(ValueError):
        CharNgramPrior(order=3, corpus_text="")


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


@pytest.fixture(scope="module")
def gpt2_prior():
    from authorship.priors import TransformerPrior

    return TransformerPrior("gpt2")


def test_transformer_prior_is_a_distribution(gpt2_prior):
    for context in ("", "T", "SPEEC"):
        assert is_distribution(gpt2_prior.prior(context))


def test_transformer_prior_completes_a_common_english_word(gpt2_prior):
    prior = gpt2_prior.prior("SPEEC")
    assert int(prior.argmax()) == symbol_index("H")
    # Corrected value is ~0.5075 -- a thin but real margin vs. the legacy method's
    # ~0.9995; not a flake if this stays close to 0.5.
    assert prior[symbol_index("H")] > 0.5


def test_legacy_batched_and_single_sequence_scoring_agree(gpt2_prior):
    """The two legacy methods (both sharing the same retokenization assumption) must
    still agree with each other -- this is a consistency check on the retired code
    path, not a correctness check (see test_priors_prefix_locked.py for correctness).
    """
    batched = gpt2_prior._legacy_full_sequence_prior("KNIGH")
    single = gpt2_prior.exact_prior("KNIGH")
    assert np.abs(batched - single).max() < 1e-5


def test_prior_is_cached_by_context(gpt2_prior):
    first = gpt2_prior.prior("ADVIC")
    second = gpt2_prior.prior("ADVIC")
    assert first is second


def test_frame_requires_a_prefix_placeholder():
    from authorship.priors import TransformerPrior

    with pytest.raises(ValueError):
        TransformerPrior("gpt2", frame="no placeholder here")


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
    assert family_of("allenai/OLMo-2-1124-7B-Instruct") == "olmo"
    assert family_of("openai/gpt-oss-20b") == "gptoss"
    assert family_of("google/gemma-2-2b-it") == "gemma"
    assert family_of("google/gemma-4-12b-it") == "gemma"
    assert family_of("Qwen/Qwen3.6-35B-A3B") == "qwen"


def test_family_of_rejects_unknown_model():
    import pytest

    with pytest.raises(ValueError, match="unknown prior family"):
        family_of("some-vendor/some-model")


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
