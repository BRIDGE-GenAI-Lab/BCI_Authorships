"""Validate the reconstructed neural posterior against the archive's own online record.

The archive stores the selection each session actually made online. Our offline posterior
is rebuilt from raw EEG with a decoder trained only on that session's calibration phase,
so it will not match perfectly, but its argmax must agree with the recorded selection far
above the 1-in-36 chance rate. If it does not, the epoching window or the flash-membership
read is wrong and every attribution number downstream would be wrong with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
COMPANION = PROJECT.parent / "study_bigp3_als_calibration"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(COMPANION / "src"))

from bigp3_als.edf import parse_source_path, select_edf_paths  # noqa: E402

from authorship.decoder import fit_session_decoder  # noqa: E402
from authorship.grid import symbol_index  # noqa: E402
from authorship.neural_posterior import build_selection_posteriors  # noqa: E402

CACHE_ROOT = (COMPANION / "data" / "source_cache").resolve()


def session_key(relative_path: str) -> tuple[str, str, str]:
    source = parse_source_path(relative_path)
    return source.study, source.participant_id, source.session_id


def main(max_sessions: int = 6) -> None:
    paths = select_edf_paths(CACHE_ROOT)
    by_session: dict[tuple[str, str, str], dict[str, list[Path]]] = {}
    for path in paths:
        relative = path.relative_to(CACHE_ROOT).as_posix()
        source = parse_source_path(relative)
        entry = by_session.setdefault(session_key(relative), {"Train": [], "Test": []})
        entry[source.phase].append(path)

    usable = [key for key, value in by_session.items() if value["Train"] and value["Test"]]
    records: list[dict[str, object]] = []
    for key in usable[:max_sessions]:
        entry = by_session[key]
        try:
            decoder = fit_session_decoder(entry["Train"])
        except ValueError as error:
            print(f"{key}: decoder skipped ({error})")
            continue
        frames = []
        for path in entry["Test"]:
            relative = path.relative_to(CACHE_ROOT).as_posix()
            frames.append(build_selection_posteriors(path, relative, decoder))
        selections = pd.concat(frames, ignore_index=True)
        if selections.empty:
            print(f"{key}: no eligible selections")
            continue
        posterior = np.stack(selections["p_neural"].to_numpy())
        argmax = posterior.argmax(axis=1)
        archive = selections["archive_selected_symbol"].map(symbol_index).to_numpy()
        target = selections["target_symbol"].map(symbol_index).to_numpy()
        records.append(
            {
                "session": ":".join(key),
                "n": len(selections),
                "train_auc": round(decoder.train_auc, 3),
                "agree_with_archive": round(float((argmax == archive).mean()), 3),
                "offline_accuracy": round(float((argmax == target).mean()), 3),
                "archive_accuracy": round(float(selections["archive_correct"].mean()), 3),
                "median_flashes": int(np.median(selections["n_flashes"])),
            }
        )
        print(records[-1])

    summary = pd.DataFrame(records)
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\nchance agreement = 0.028")
    print(f"mean agreement with archive: {summary['agree_with_archive'].mean():.3f}")
    print(f"mean offline accuracy: {summary['offline_accuracy'].mean():.3f}")
    print(f"mean archive accuracy: {summary['archive_accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()
