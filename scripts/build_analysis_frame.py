"""Build the selection-level analysis frame: neural posteriors for every online selection.

One decoder is fitted per recording session from that session's calibration phase. Where a
session has no calibration phase of its own, the participant's remaining calibration files
are pooled and the substitution is recorded. Test-phase data never inform training.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
COMPANION = PROJECT.parent / "study_bigp3_als_calibration"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(COMPANION / "src"))

from bigp3_als.edf import parse_source_path, select_edf_paths  # noqa: E402

from authorship.context import add_context  # noqa: E402
from authorship.decoder import fit_session_decoder  # noqa: E402
from authorship.grid import symbol_index  # noqa: E402
from authorship.neural_posterior import build_selection_posteriors  # noqa: E402

CACHE_ROOT = (COMPANION / "data" / "source_cache").resolve()
OUTPUT = PROJECT / "output" / "intermediate"


def index_sources() -> dict[tuple[str, str, str], dict[str, list[Path]]]:
    index: dict[tuple[str, str, str], dict[str, list[Path]]] = {}
    for path in select_edf_paths(CACHE_ROOT):
        relative = path.relative_to(CACHE_ROOT).as_posix()
        source = parse_source_path(relative)
        key = (source.study, source.participant_id, source.session_id)
        index.setdefault(key, {"Train": [], "Test": []})[source.phase].append(path)
    return index


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = index_sources()

    participant_train: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for (study, participant, _session), entry in index.items():
        participant_train[(study, participant)].extend(entry["Train"])

    frames: list[pd.DataFrame] = []
    decoder_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    keys = sorted(key for key, entry in index.items() if entry["Test"])
    for position, key in enumerate(keys, start=1):
        study, participant, session = key
        entry = index[key]
        train_paths = entry["Train"]
        fallback = False
        if not train_paths:
            train_paths = [
                path
                for path in participant_train[(study, participant)]
                if path not in entry["Test"]
            ]
            fallback = True
        if not train_paths:
            skipped.append({"session": ":".join(key), "reason": "no_calibration_available"})
            continue
        try:
            decoder = fit_session_decoder(train_paths)
        except ValueError as error:
            skipped.append({"session": ":".join(key), "reason": str(error)})
            continue
        decoder_rows.append(
            {
                "study": study,
                "participant_id": participant,
                "study_participant_id": f"{study}:{participant}",
                "session_id": session,
                "train_auc": decoder.train_auc,
                "n_target_epochs": decoder.n_target,
                "n_nontarget_epochs": decoder.n_nontarget,
                "calibration_fallback": fallback,
                "n_calibration_files": len(train_paths),
            }
        )
        for path in entry["Test"]:
            relative = path.relative_to(CACHE_ROOT).as_posix()
            try:
                frames.append(build_selection_posteriors(path, relative, decoder))
            except ValueError as error:
                skipped.append({"session": relative, "reason": str(error)})
        print(f"[{position}/{len(keys)}] {':'.join(key)} auc={decoder.train_auc:.3f}", flush=True)

    selections = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    selections = add_context(selections)
    decoders = pd.DataFrame(decoder_rows)
    selections = selections.merge(
        decoders[["study", "participant_id", "session_id", "train_auc", "calibration_fallback"]],
        on=["study", "participant_id", "session_id"],
        how="left",
        validate="many_to_one",
    )

    posterior_matrix = np.stack(selections["p_neural"].to_numpy())
    if not np.allclose(posterior_matrix.sum(axis=1), 1.0):
        raise ValueError("neural posteriors do not sum to one")
    selections["p_neural"] = [row.tolist() for row in posterior_matrix]

    selections.to_parquet(OUTPUT / "selections.parquet", index=False)
    decoders.to_parquet(OUTPUT / "decoders.parquet", index=False)

    summary = {
        "n_selections": int(len(selections)),
        "n_participants": int(selections["study_participant_id"].nunique()),
        "n_sessions": int(selections.groupby(["study_participant_id", "session_id"]).ngroups),
        "n_files": int(selections["relative_path"].nunique()),
        "n_unique_contexts": int(selections["context_prefix"].nunique()),
        "n_unique_phrases": int(selections["intended_phrase"].nunique()),
        "by_study": selections.groupby("study").size().to_dict(),
        "by_condition": selections.groupby("condition").size().to_dict(),
        "archive_accuracy": float(selections["archive_correct"].mean()),
        "offline_neural_accuracy": float(
            (
                posterior_matrix.argmax(axis=1)
                == selections["target_symbol"].map(symbol_index).to_numpy()
            ).mean()
        ),
        "median_flashes": float(selections["n_flashes"].median()),
        "decoder_auc_median": float(decoders["train_auc"].median()),
        "calibration_fallback_sessions": int(decoders["calibration_fallback"].sum()),
        "skipped": skipped,
    }
    (PROJECT / "output" / "cohort_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "skipped"}, indent=2))
    print(f"skipped entries: {len(skipped)}")


if __name__ == "__main__":
    main()
