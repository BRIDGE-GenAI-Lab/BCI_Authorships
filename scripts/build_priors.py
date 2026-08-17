"""Compute the language-model prior ladder over every unique copy-spelling context.

Copy-spelling contexts repeat heavily across sessions, so each unique context is scored
once per model and reused. Neural priors use `prefix_locked_prior`: the context is
tokenized once, one forward pass produces the next-token distribution, and that
distribution is marginalized onto the 36 grid symbols by each vocabulary token's own first
character. Continuation scoring, which re-tokenizes `context + candidate` per candidate, is
retained only as a comparator (`TransformerPrior._legacy_full_sequence_prior`, batched, and
`exact_prior`, its unbatched equivalent), because that re-tokenization was measured to change
the context's own tokenization for 33.9% of sampled context-candidate pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.priors import (  # noqa: E402
    CharNgramKNPrior,
    CharNgramPrior,
    TransformerPrior,
    load_wikitext_corpus,
    uniform_prior,
)

OUTPUT = PROJECT / "output" / "intermediate"

LADDER = [
    {"name": "uniform", "tier": "null", "kind": "uniform", "parameters": 0},
    {"name": "ngram5", "tier": "classical", "kind": "ngram", "parameters": 0},
    {"name": "ngram5_kn", "tier": "classical_kn", "kind": "ngram_kn", "parameters": 0},
    {"name": "ngram5_wiki_kn", "tier": "classical_kn_wiki", "kind": "ngram_kn_wiki", "parameters": 0},
    {"name": "gpt2", "tier": "neural_small", "kind": "transformer", "parameters": 124_000_000},
    {"name": "gpt2-large", "tier": "neural_medium", "kind": "transformer", "parameters": 774_000_000},
    {
        "name": "Qwen/Qwen2.5-1.5B",
        "tier": "neural_large",
        "kind": "transformer",
        "parameters": 1_540_000_000,
    },
    {
        "name": "Qwen/Qwen2.5-3B",
        "tier": "neural_xlarge",
        "kind": "transformer",
        "parameters": 3_090_000_000,
    },
    # Scored on the HMS O2 cluster; shards are ingested, not recomputed locally.
    {"name": "Qwen/Qwen2.5-14B", "tier": "neural_14b", "kind": "cluster", "parameters": 14_700_000_000},
    {"name": "Qwen/Qwen2.5-32B", "tier": "neural_32b", "kind": "cluster", "parameters": 32_500_000_000},
    {"name": "Qwen/Qwen3.5-27B", "tier": "neural_next_gen", "kind": "cluster", "parameters": 27_000_000_000},
    {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "tier": "cross_family_8b_instruct",
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
        "name": "google/gemma-2-2b-it",
        "tier": "cross_family_2b_instruct",
        "kind": "cluster",
        "parameters": 2_000_000_000,
    },
    {
        "name": "allenai/OLMo-2-1124-7B-Instruct",
        "tier": "cross_family_olmo_7b",
        "kind": "cluster",
        "parameters": 7_000_000_000,
    },
    {
        "name": "google/gemma-4-12b-it",
        "tier": "cross_family_gemma4_12b",
        "kind": "cluster",
        "parameters": 12_000_000_000,
    },
    {
        "name": "Qwen/Qwen3.6-35B-A3B",
        "tier": "cross_family_qwen36_35b",
        "kind": "cluster",
        "parameters": 35_000_000_000,
    },
    {
        # Corrected 2026-08-06 from 20_000_000_000 to OpenAI's own model-card figure
        # (arXiv:2508.10925, Table 1: 20.91B total / 3.61B active). This edit does NOT
        # propagate to already-materialized artifacts: output/intermediate/prior_shards/
        # openai_gpt-oss-20b.parquet, output/intermediate/priors.parquet,
        # output/intermediate/attribution.parquet, and output/stats_digest.json all still
        # store the old 20_000_000_000 label as of this commit (verified directly by
        # reading each file's prior_parameters column). For "kind": "cluster" entries this
        # constant is consulted only when build_model() constructs a NEW shard, which never
        # happens for a cluster entry (build_model() raises immediately for kind=="cluster";
        # main()'s per-model loop also skips any entry whose shard file already exists). The
        # shard was populated once by scripts/ingest_o2_priors.py, which copies whatever
        # "prior_parameters" value is already present in each row of the raw O2-scored JSONL
        # verbatim — it does not import or consult this LADDER list at all. So neither a
        # plain rerun of this script nor a rerun of ingest_o2_priors.py against the same
        # JSONL will pick up this correction. No GPU/O2 rescoring is needed to fix the
        # downstream artifacts (the neural posteriors/p_lm columns are untouched; only a
        # metadata label is wrong): either (a) patch the "prior_parameters" column in-place
        # in the three cached parquet files above and rerun build_attribution.py,
        # run_stats.py, and make_figures.py (all CPU-only), or (b) if the raw O2-scored
        # JSONL for gpt-oss-20b still exists, correct its "prior_parameters" field and
        # re-run ingest_o2_priors.py before doing (a).
        "name": "openai/gpt-oss-20b",
        "tier": "cross_family_gptoss_20b_mxfp4",
        "kind": "cluster",
        "parameters": 20_900_000_000,
    },
    {
        "name": "EleutherAI/pythia-12b",
        "tier": "cross_family_pythia_12b",
        "kind": "cluster",
        "parameters": 12_000_000_000,
    },
    {
        "name": "state-spaces/mamba-2.8b-hf",
        "tier": "cross_family_mamba_2_8b",
        "kind": "cluster",
        "parameters": 2_800_000_000,
    },
    {
        "name": "google/recurrentgemma-2b",
        "tier": "cross_family_recurrentgemma_2b",
        "kind": "cluster",
        "parameters": 2_700_000_000,
    },
]


def build_model(specification: dict, frame: str | None, dtype: str | None = None):
    if specification["kind"] == "cluster":
        raise SystemExit(
            f"{specification['name']} is scored on the cluster; run scripts/ingest_o2_priors.py"
        )
    if specification["kind"] == "uniform":
        return None
    if specification["kind"] == "ngram_kn":
        return CharNgramKNPrior(order=5)
    if specification["kind"] == "ngram_kn_wiki":
        return CharNgramKNPrior(order=5, corpus_text=load_wikitext_corpus())
    if specification["kind"] == "ngram":
        return CharNgramPrior(order=5)
    return TransformerPrior(specification["name"], frame=frame, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=None, help="optional prompt frame with {prefix}")
    parser.add_argument("--suffix", default="", help="output filename suffix")
    parser.add_argument("--only", default=None, help="comma-separated model names to run")
    parser.add_argument("--dtype", default=None, help="torch dtype for transformer priors")
    arguments = parser.parse_args()

    selections = pd.read_parquet(OUTPUT / "selections.parquet")
    contexts = sorted(selections["context_prefix"].unique())
    print(f"unique contexts: {len(contexts)}")

    ladder = LADDER
    if arguments.only:
        wanted = set(arguments.only.split(","))
        ladder = [spec for spec in ladder if spec["name"] in wanted]

    shard_directory = OUTPUT / f"prior_shards{arguments.suffix}"
    shard_directory.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    for specification in ladder:
        shard_path = shard_directory / f"{specification['name'].replace('/', '_')}.parquet"
        if shard_path.exists():
            print(f"  {specification['name']}: cached", flush=True)
            continue
        started = time.time()
        model = build_model(specification, arguments.frame, arguments.dtype)
        rows = []
        for context in contexts:
            prior = uniform_prior() if specification["kind"] == "uniform" else model.prior(context)
            rows.append(
                {
                    "prior_model": specification["name"],
                    "prior_tier": specification["tier"],
                    "prior_parameters": specification["parameters"],
                    "context_prefix": context,
                    "p_lm": prior.astype(float).tolist(),
                }
            )
        pd.DataFrame(rows).to_parquet(shard_path, index=False)
        timings[specification["name"]] = round(time.time() - started, 1)
        print(f"  {specification['name']}: {timings[specification['name']]}s", flush=True)
        del model

    shards = sorted(
        path for path in shard_directory.glob("*.parquet") if not path.name.startswith("._")
    )
    priors = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
    path = OUTPUT / f"priors{arguments.suffix}.parquet"
    priors.to_parquet(path, index=False)

    matrix = np.stack(priors["p_lm"].map(np.asarray).to_numpy())
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("priors do not sum to one")
    print(json.dumps({"rows": len(priors), "timings_seconds": timings, "path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
