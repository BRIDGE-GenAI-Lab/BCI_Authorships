"""Cross every selection with every prior in the ladder and compute attribution measures.

The primary analysis fuses at beta = 1, the standard Bayesian speller combination. The
sensitivity grid varies how heavily the neural evidence is weighted relative to the prior,
because the fusion rule itself partly determines attribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.attribution import attribution_row, fuse  # noqa: E402
from authorship.grid import symbol_index  # noqa: E402

OUTPUT = PROJECT / "output" / "intermediate"
PRIMARY_BETA = 1.0
BETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)

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
    "n_flashes",
    "context_prefix",
    "intended_phrase",
    "position_in_phrase",
    "phrase_length",
    "phrase_is_numeric",
    "train_auc",
    "calibration_fallback",
]


def main() -> None:
    selections = pd.read_parquet(OUTPUT / "selections.parquet")
    priors = pd.read_parquet(OUTPUT / "priors.parquet")

    selections["target_index"] = selections["target_symbol"].map(symbol_index)
    neural_matrix = np.stack(selections["p_neural"].map(np.asarray).to_numpy())
    prior_lookup = {
        (row.prior_model, row.context_prefix): np.asarray(row.p_lm)
        for row in priors.itertuples(index=False)
    }
    model_metadata = (
        priors[["prior_model", "prior_tier", "prior_parameters"]]
        .drop_duplicates()
        .set_index("prior_model")
    )

    records: list[dict[str, object]] = []
    for beta in BETA_GRID:
        for model_name in model_metadata.index:
            for position, selection in enumerate(selections.itertuples(index=False)):
                prior = prior_lookup[(model_name, selection.context_prefix)]
                neural = neural_matrix[position]
                fused = fuse(neural, prior, beta=beta)
                row = attribution_row(neural, prior, fused, int(selection.target_index))
                row.update(
                    {
                        "beta": beta,
                        "prior_model": model_name,
                        "prior_tier": model_metadata.loc[model_name, "prior_tier"],
                        "prior_parameters": int(
                            model_metadata.loc[model_name, "prior_parameters"]
                        ),
                    }
                )
                row.update({column: getattr(selection, column) for column in CARRY_COLUMNS})
                records.append(row)
        print(f"beta={beta} done ({len(records)} rows)", flush=True)

    attribution = pd.DataFrame(records)
    attribution.to_parquet(OUTPUT / "attribution.parquet", index=False)

    primary = attribution[attribution["beta"] == PRIMARY_BETA]
    if (primary["phantom_agreement"] & primary["prior_capture"]).any():
        raise ValueError("phantom agreement and prior capture must be mutually exclusive")
    if primary.groupby("prior_model").size().nunique() != 1:
        raise ValueError("each prior model must cover every selection")

    summary = {
        "rows": int(len(attribution)),
        "selections": int(len(selections)),
        "beta_grid": list(BETA_GRID),
        "primary_beta": PRIMARY_BETA,
        "by_model": {
            name: {
                "ncf_mean": round(float(group["ncf"].mean()), 4),
                "phantom_agreement": round(float(group["phantom_agreement"].mean()), 4),
                "prior_capture": round(float(group["prior_capture"].mean()), 4),
                "fused_accuracy": round(float(group["fused_correct"].mean()), 4),
                "neural_accuracy": round(float(group["neural_correct"].mean()), 4),
                "prior_accuracy": round(float(group["prior_correct"].mean()), 4),
            }
            for name, group in primary.groupby("prior_model")
        },
    }
    (PROJECT / "output" / "attribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
