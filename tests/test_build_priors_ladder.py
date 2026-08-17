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


def test_ladder_contains_the_eleven_new_cross_family_models():
    names = {spec["name"] for spec in LADDER}
    expected = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-v0.1",
        "deepseek-ai/DeepSeek-V2-Lite",
        "google/gemma-2-2b-it",
        "allenai/OLMo-2-1124-7B-Instruct",
        "google/gemma-4-12b-it",
        "Qwen/Qwen3.6-35B-A3B",
        "openai/gpt-oss-20b",
    }
    assert expected <= names


def test_new_cluster_entries_are_marked_for_cluster_scoring():
    by_name = {spec["name"]: spec for spec in LADDER}
    for name in (
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b",
        "google/gemma-2-27b",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-v0.1",
        "deepseek-ai/DeepSeek-V2-Lite",
        "google/gemma-2-2b-it",
        "allenai/OLMo-2-1124-7B-Instruct",
        "google/gemma-4-12b-it",
        "Qwen/Qwen3.6-35B-A3B",
        "openai/gpt-oss-20b",
    ):
        assert by_name[name]["kind"] == "cluster"
        assert by_name[name]["parameters"] > 0
