"""Closed-loop emitted-context sensitivity analysis for the primary prior (O2 only).

The primary neural contribution fraction analysis scores the language-model prior against
the archive's intended preceding characters (see authorship/context.py: "Context is built
from the symbols the participant was instructed to spell, not from the symbols the system
emitted. That choice isolates attribution from error propagation, which is a separate
phenomenon; a sensitivity analysis repeats the pipeline with emitted context.").

This script is that sensitivity analysis. For each session, selections are walked in
phrase order (matching authorship/context.py's own grouping: one phrase per relative_path,
context reset at the start of each phrase). Instead of scoring the prior against the
archive's ground-truth prefix, it is scored against the characters the fused decision rule
itself emitted at positions 1..i-1 in the same phrase. If an earlier selection in the
phrase was mis-emitted, every later prior lookup in that phrase sees the wrong prefix,
compounding the error the way a real closed-loop speller would.

Because every selection can carry a unique, session-specific emitted context, contexts
cannot be scored once per unique string and reused the way the parallel batch scorer does
for the intended-context ladder. This script instead scores one selection at a time, in
sequence, using the corrected prefix-locked method from `authorship/priors.py`:
`build_symbol_projection` and `prefix_locked_prior`. This method tokenizes the context
exactly once, runs one forward pass, and marginalizes the model's real next-token
distribution by which grid symbol each vocabulary token's first character represents.
This is immune to the retokenization instability that affects legacy full-sequence
scorers. The O2 cluster's own operational scoring script for this study's language-model
ladder (`score_priors_o2.py` on `/n/scratch/users/a/alg7274`) implements the same
method for GPU hardware but has never been committed to this repository; `load`/`seed_ids`
below are adapted from it for consistency with how the rest of this study's ladder was
actually scored on the cluster, and now use the same canonical implementation as the
primary analysis.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.attribution import attribution_row, fuse  # noqa: E402
from authorship.grid import SYMBOLS, symbol_index  # noqa: E402
from authorship.priors import build_symbol_projection, prefix_locked_prior  # noqa: E402

OUTPUT = PROJECT / "output" / "intermediate"

CARRY_COLUMNS = [
    "study",
    "participant_id",
    "study_participant_id",
    "session_id",
    "condition",
    "relative_path",
    "trial_number",
    "target_symbol",
    "archive_selected_symbol",
    "archive_correct",
    "context_prefix",
    "intended_phrase",
    "position_in_phrase",
    "phrase_length",
]


def emitted_contexts_for_session(
    session: list[dict[str, object]], fused_emissions: dict[object, str]
) -> dict[object, str]:
    """Reconstruct the context each selection in a session actually saw.

    Concatenates the fused decision rule's own emissions in order, not the archive's
    ``context_prefix`` (which encodes intended, not emitted, characters).
    """
    contexts: dict[object, str] = {}
    running = ""
    for selection in session:
        selection_id = selection["selection_id"]
        contexts[selection_id] = running
        running += fused_emissions[selection_id]
    return contexts


def ordered_phrases(selections: pd.DataFrame) -> list[pd.DataFrame]:
    """Group selections into phrase-ordered blocks.

    Matches authorship/context.py's own grouping exactly: one phrase per relative_path,
    walked in trial_number order, so the emitted-context reset points line up with where
    the intended-context primary analysis resets context_prefix to "".
    """
    ordered = selections.sort_values(["relative_path", "trial_number"])
    return [group for _, group in ordered.groupby("relative_path", sort=False)]


def load(
    model_name: str,
    dtype: str = "bfloat16",
    revision: str | None = None,
    tokenizer_revision: str | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=tokenizer_revision)
    n_gpus = torch.cuda.device_count()
    device_map = {"": 0} if n_gpus == 1 else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
        revision=revision,
    ).eval()
    return tokenizer, model


def seed_ids(tokenizer) -> list[int]:
    for candidate in (tokenizer.bos_token_id, tokenizer.eos_token_id):
        if candidate is not None:
            return [int(candidate)]
    raise ValueError("tokenizer exposes no beginning-of-sequence token")


def score_phrase_sequential(
    phrase: pd.DataFrame, tokenizer, model, seed: list[int], device, projection: np.ndarray
) -> list[dict[str, object]]:
    """Walk one phrase's selections in order, scoring the prior against the fused decision
    rule's own emitted prefix rather than the archive's intended prefix, then fusing with
    the already-computed p_neural from selections.parquet.
    """
    session = [{"selection_id": index} for index in phrase.index]
    fused_emissions: dict[object, str] = {}
    used_context: dict[object, str] = {}
    results: list[dict[str, object]] = []
    running = ""
    for index, row in phrase.iterrows():
        context = running
        used_context[index] = context
        p_lm = prefix_locked_prior(context, tokenizer, model, device, seed, projection)
        p_neural = np.asarray(row.p_neural)
        target_index = symbol_index(row.target_symbol)
        fused = fuse(p_neural, p_lm)
        measures = attribution_row(p_neural, p_lm, fused, target_index)
        emitted_symbol = SYMBOLS[measures["fused_argmax"]]
        fused_emissions[index] = emitted_symbol
        running += emitted_symbol

        result = dict(measures)
        result["emitted_context"] = context
        result["p_lm"] = [float(x) for x in p_lm]
        result.update({column: getattr(row, column) for column in CARRY_COLUMNS})
        results.append(result)

    reconstructed = emitted_contexts_for_session(session, fused_emissions)
    if any(reconstructed[index] != used_context[index] for index in phrase.index):
        raise ValueError("emitted context used for scoring does not match the replay reconstruction")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B")
    parser.add_argument("--tier", default="neural_32b")
    parser.add_argument("--parameters", type=int, default=32_500_000_000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--selections", default=str(OUTPUT / "selections.parquet"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer-revision", "--tokenizer_revision", default=None,
                         dest="tokenizer_revision")
    arguments = parser.parse_args()
    if arguments.tokenizer_revision is None:
        arguments.tokenizer_revision = arguments.revision

    selections = pd.read_parquet(arguments.selections)
    phrases = ordered_phrases(selections)
    print(f"model={arguments.model} selections={len(selections)} phrases={len(phrases)} "
          f"gpus={torch.cuda.device_count()}", flush=True)

    started = time.time()
    tokenizer, model = load(
        arguments.model, arguments.dtype, arguments.revision, arguments.tokenizer_revision
    )
    device = next(model.parameters()).device
    seed = seed_ids(tokenizer)
    vocabulary_size = int(model.get_output_embeddings().weight.shape[0])
    projection = build_symbol_projection(tokenizer, vocabulary_size)
    print(f"loaded in {time.time() - started:.0f}s onto {device}", flush=True)

    rows: list[dict[str, object]] = []
    started = time.time()
    for position, phrase in enumerate(phrases):
        rows.extend(score_phrase_sequential(phrase, tokenizer, model, seed, device, projection))
        if (position + 1) % 50 == 0 or (position + 1) == len(phrases):
            elapsed = time.time() - started
            rate = elapsed / (position + 1)
            print(f"  phrase {position + 1}/{len(phrases)}  rows={len(rows)}  "
                  f"{rate:.2f}s/phrase", flush=True)

    result = pd.DataFrame(rows)
    result["seed"] = arguments.seed
    result["prior_model"] = arguments.model
    result["prior_tier"] = arguments.tier
    result["prior_parameters"] = arguments.parameters
    if len(result) != len(selections):
        raise ValueError(
            f"expected {len(selections)} scored selections, produced {len(result)}"
        )
    result.to_parquet(arguments.out, index=False)
    print(f"wrote {len(result)} rows to {arguments.out} in {time.time() - started:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
