"""Reconstruct the intended copy-spelling string and the context available at each selection.

Context is built from the symbols the participant was instructed to spell, not from the
symbols the system emitted. That choice isolates attribution from error propagation, which
is a separate phenomenon; a sensitivity analysis repeats the pipeline with emitted context.
"""

from __future__ import annotations

import pandas as pd


def add_context(trials: pd.DataFrame) -> pd.DataFrame:
    """Add the intended phrase, the preceding context, and the position within it."""
    required = {"relative_path", "trial_number", "target_symbol"}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")

    ordered = trials.sort_values(["relative_path", "trial_number"]).copy()
    grouped = ordered.groupby("relative_path", sort=False)["target_symbol"]
    ordered["intended_phrase"] = grouped.transform(lambda symbols: "".join(symbols))
    ordered["position_in_phrase"] = grouped.cumcount()
    ordered["context_prefix"] = grouped.transform(
        lambda symbols: pd.Series(symbols).shift(1, fill_value="").cumsum().to_numpy()
    )
    ordered["phrase_length"] = ordered["intended_phrase"].str.len()
    ordered["phrase_is_numeric"] = ordered["intended_phrase"].str.strip().str.isdigit()
    return ordered.reset_index(drop=True)
