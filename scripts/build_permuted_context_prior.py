"""Reassign each selection to a random, wrong context for the primary prior, as a negative
control: if attribution reflects genuine linguistic relevance rather than generic posterior
sharpening, phantom agreement under a permuted context should be lower than under the real one.

A same-index derangement is not sufficient here: many rows share an identical context_prefix
value (662 of 3,373 selections carry the empty-string context alone, at word position 1), so
avoiding only a row's own index does not avoid its own context value. Rows are instead grouped
by context_prefix value and the group blocks are rotated by an offset at least as large as the
largest group, which guarantees every row receives a context_prefix value that differs from its
own while still drawing from exactly the same multiset of real contexts (a true permutation).

`permute()` also supports a stratified mode (round-2 review Item 7): rather than drawing a
wrong context from the whole file, restrict the swap to rows that share the same character
position, source study, and context-length bucket, so the "wrong" context a row receives is
still structurally realistic for where it occurs. Some strata are irreducibly unpermutable --
most notably every position-0 selection has context_prefix == "" by construction (there is no
other valid zero-length context to draw from), so no within-stratum permutation can give those
rows a value that differs from their own. Such rows are marked `permutable=False` and excluded
from the stratified output rather than silently left at their true value or forced to raise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

OUTPUT = PROJECT / "output" / "intermediate"
PERMUTATION_SEED = 20260802
DEFAULT_STRATUM_COLUMNS = ["position_in_phrase", "study"]
LENGTH_BUCKET_BINS = [-1, 0, 2, 4, 6, 100]


def _rotate_block(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Block-rotate a 1-D array of values so no position keeps its own value.

    Groups positions by value, shuffles the group order and each group's internal order,
    then rolls the resulting block order by an offset at least as large as the largest
    group -- large enough that no group can land back on its own original slot range.
    Raises ValueError if a single value occupies more than half the array, in which case
    no value-safe permutation exists.
    """
    n = len(values)
    unique_values = pd.unique(values)
    rng.shuffle(unique_values)
    grouped_positions = []
    for value in unique_values:
        positions = np.where(values == value)[0].copy()
        rng.shuffle(positions)
        grouped_positions.append(positions)
    block_order = np.concatenate(grouped_positions)

    max_block = max(len(positions) for positions in grouped_positions)
    if max_block > n - max_block:
        raise ValueError(
            "a single value occupies more than half the group; no value-safe permutation exists"
        )
    offset = int(rng.integers(max_block, n - max_block + 1))

    shifted_block_order = np.roll(block_order, -offset)
    assigned_by_block_slot = values[shifted_block_order]
    permuted_values = np.empty(n, dtype=values.dtype)
    permuted_values[block_order] = assigned_by_block_slot

    if np.any(permuted_values == values):
        raise ValueError("permutation produced a self-matching value; this should be unreachable")
    return permuted_values


def permute(
    selections: pd.DataFrame,
    seed: int = PERMUTATION_SEED,
    stratum_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Permute context_prefix, optionally restricted to within-stratum swaps.

    With `stratum_columns=None` (default), this is the original whole-file block rotation,
    unchanged, so existing single-permutation callers and tests are unaffected. With
    `stratum_columns` given (e.g. `["position_in_phrase", "study"]`), context_prefix is
    additionally never swapped across an internally-computed context-length bucket, and the
    swap is restricted to rows sharing all of those strata. Rows in a stratum where no
    value-safe permutation exists (see `_rotate_block`) get `permutable=False` and a null
    `permuted_context_prefix`, rather than raising -- this is a structural property of the
    data (e.g. every position-0 context is ""), not a per-call failure.
    """
    rng = np.random.default_rng(seed)
    frame = selections.reset_index(drop=True).copy()
    n = len(frame)

    if stratum_columns is None:
        values = frame["context_prefix"].to_numpy()
        frame["permuted_context_prefix"] = _rotate_block(values, rng)
        return frame

    length_bucket = pd.cut(frame["context_prefix"].str.len(), bins=LENGTH_BUCKET_BINS)
    group_key = frame[list(stratum_columns)].assign(_length_bucket=length_bucket.astype(str))

    permuted_values = np.full(n, None, dtype=object)
    permutable_mask = np.zeros(n, dtype=bool)
    skipped_strata: list[tuple[object, int]] = []
    for stratum_key, positions in group_key.groupby(list(group_key.columns), sort=False).groups.items():
        idx = np.asarray(positions)
        stratum_values = frame.loc[idx, "context_prefix"].to_numpy()
        try:
            stratum_permuted = _rotate_block(stratum_values, rng)
        except ValueError:
            skipped_strata.append((stratum_key, len(idx)))
            continue
        permuted_values[idx] = stratum_permuted
        permutable_mask[idx] = True

    frame["permuted_context_prefix"] = permuted_values
    frame["permutable"] = permutable_mask
    if skipped_strata:
        frame.attrs["skipped_strata"] = skipped_strata
    return frame


def _generate_stratified_permutations(
    selections: pd.DataFrame, n_permutations: int, seed_start: int
) -> pd.DataFrame:
    """Build the combined long-format frame for `n_permutations` stratified permutations.

    Only carries `selection_row_id` (position in `selections`), `permutation_index`, and
    `permuted_context_prefix` -- everything else (p_neural, target_symbol, true
    context_prefix, ...) is re-joined from `selections` downstream by `selection_row_id`,
    so a 36-float p_neural vector is never duplicated per permutation.
    """
    base = selections.reset_index(drop=True)
    frames = []
    skipped_strata = None
    for i in range(n_permutations):
        permuted = permute(base, seed=seed_start + i, stratum_columns=DEFAULT_STRATUM_COLUMNS)
        if skipped_strata is None:
            skipped_strata = permuted.attrs.get("skipped_strata", [])
        permutable = permuted[permuted["permutable"]]
        frames.append(
            pd.DataFrame(
                {
                    "selection_row_id": permutable.index.to_numpy(),
                    "permutation_index": i,
                    "permuted_context_prefix": permutable["permuted_context_prefix"].to_numpy(),
                }
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    n_excluded = len(base) - len(frames[0])
    print(
        f"stratified permutations: {n_permutations} draws, "
        f"{len(frames[0])}/{len(base)} selections permutable per draw "
        f"({n_excluded} excluded, {len(skipped_strata or [])} unpermutable strata)"
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    selections = pd.read_parquet(OUTPUT / "selections.parquet")

    if args.n_permutations is None:
        permuted = permute(selections)
        permuted.to_parquet(OUTPUT / "permuted_context_selections.parquet", index=False)
        print(f"permuted {len(permuted)} selections, seed={PERMUTATION_SEED}")
        return

    combined = _generate_stratified_permutations(selections, args.n_permutations, args.seed_start)
    combined.to_parquet(OUTPUT / "permuted_context_selections_stratified.parquet", index=False)
    print(
        f"wrote {len(combined)} rows "
        f"({args.n_permutations} permutations x permutable selections) to "
        "permuted_context_selections_stratified.parquet"
    )


if __name__ == "__main__":
    main()
