"""Rebuild the selection frame with a stronger per-session decoder (decoder-strength sensitivity).

The session loop, calibration fallback, posterior construction and context merge are those
of scripts/build_analysis_frame.py; only the decoder differs. Each decoder is fitted on the
session's calibration (Train) files only. The output is row-aligned with
output/intermediate/selections.parquet, and the script fails loudly if any baseline row is
missing or extra rather than dropping it.

Epoch parity: _extract_file_epochs (training) and build_selection_posteriors (scoring) use
the same 0.5 to 30 Hz zero-phase Butterworth, the same -200 to 800 ms window, the same
pre-stimulus baseline and the same 150 uV artifact threshold, so no scoring-side change is
needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
COMPANION = PROJECT.parent / "study_bigp3_als_calibration"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(COMPANION / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from bigp3_als.features import (  # noqa: E402
    MIN_NONTARGET_EPOCHS,
    MIN_TARGET_EPOCHS,
    _extract_file_epochs,
)

from authorship.context import add_context  # noqa: E402
from authorship.grid import symbol_index  # noqa: E402
from authorship.neural_posterior import build_selection_posteriors  # noqa: E402
from authorship.strong_decoders import CV_SCHEMES, EpochDecoder, fit_strong_decoder  # noqa: E402
from build_analysis_frame import CACHE_ROOT, OUTPUT, index_sources  # noqa: E402

DECODERS = ("xdawn_riemann", "eegnet")
KEY_COLUMNS = ["relative_path", "trial_number"]
SessionKey = tuple[str, str, str]


def fit_strong_session_decoder(train_paths: list[Path], *, classifier: str) -> EpochDecoder:
    """Fit a stronger decoder from every calibration EDF of one session.

    Applies the same eligibility rules as authorship.decoder.fit_from_epochs so that the
    set of fitted sessions matches the original decoder's.
    """
    if not train_paths:
        raise ValueError("no calibration files supplied")
    epoch_blocks, label_blocks, group_blocks, sampling_frequencies = [], [], [], []
    for path in train_paths:
        epochs, labels, sampling_frequency, _ = _extract_file_epochs(path)
        if not len(labels):
            continue
        epoch_blocks.append(epochs)
        label_blocks.append(labels)
        group_blocks.append(np.full(len(labels), path.name))
        sampling_frequencies.append(sampling_frequency)
    if not epoch_blocks:
        raise ValueError("calibration files contained no usable epochs")
    labels = np.concatenate(label_blocks).astype(int)
    groups = np.concatenate(group_blocks)
    n_target, n_nontarget = int((labels == 1).sum()), int((labels == 0).sum())
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration labels must contain target and non-target epochs")
    if n_target < MIN_TARGET_EPOCHS or n_nontarget < MIN_NONTARGET_EPOCHS:
        raise ValueError(
            f"insufficient calibration epochs: {n_target} target, {n_nontarget} non-target"
        )
    if len(np.unique(groups)) < 2:
        raise ValueError("at least two calibration files are required for grouped validation")
    return fit_strong_decoder(
        np.concatenate(epoch_blocks),
        labels,
        groups,
        classifier=classifier,
        sampling_frequency=float(np.mean(sampling_frequencies)),
    )


def build_selections(
    fit_decoder: Callable[[list[Path]], object],
    only_sessions: Iterable[SessionKey] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Run the build_analysis_frame session loop with an arbitrary decoder factory."""
    index = index_sources()
    participant_train: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for (study, participant, _session), entry in index.items():
        participant_train[(study, participant)].extend(entry["Train"])

    frames: list[pd.DataFrame] = []
    decoder_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    keys = sorted(key for key, entry in index.items() if entry["Test"])
    if only_sessions is not None:
        wanted = set(only_sessions)
        keys = [key for key in keys if key in wanted]
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
            decoder = fit_decoder(train_paths)
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
                "cv_scheme": getattr(decoder, "cv_scheme", None),
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
    return selections, decoders, skipped


def align_to_baseline(selections: pd.DataFrame, baseline_keys: pd.DataFrame) -> pd.DataFrame:
    """Reorder to the baseline's rows; raise SystemExit listing any missing or extra keys."""
    baseline_keys = baseline_keys.reset_index(drop=True)
    if selections.duplicated(KEY_COLUMNS).any():
        raise SystemExit("strong-decoder selections contain duplicate (relative_path, trial_number)")
    built = set(map(tuple, selections[KEY_COLUMNS].itertuples(index=False)))
    wanted = list(map(tuple, baseline_keys.itertuples(index=False)))
    missing = [key for key in wanted if key not in built]
    extra = sorted(built - set(wanted))
    if missing or extra:
        raise SystemExit(
            "strong-decoder selections do not align row-for-row with selections.parquet: "
            f"{len(missing)} missing {missing}, {len(extra)} extra {extra}"
        )
    aligned = baseline_keys.merge(selections, on=KEY_COLUMNS, how="left", validate="one_to_one")
    aligned = aligned[selections.columns]
    if not baseline_keys.equals(aligned[KEY_COLUMNS].reset_index(drop=True)):
        raise SystemExit("strong-decoder selections do not align row-for-row with selections.parquet")
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decoder", choices=DECODERS, required=True)
    args = parser.parse_args()

    baseline = pd.read_parquet(OUTPUT / "selections.parquet")
    selections, decoders, skipped = build_selections(
        lambda paths: fit_strong_session_decoder(paths, classifier=args.decoder)
    )
    selections = align_to_baseline(selections, baseline[KEY_COLUMNS])
    if list(selections.columns) != list(baseline.columns):
        raise SystemExit("strong-decoder selections do not carry the baseline columns")

    selections.to_parquet(OUTPUT / f"selections_{args.decoder}.parquet", index=False)
    decoders.to_parquet(OUTPUT / f"decoders_{args.decoder}.parquet", index=False)

    posterior_matrix = np.stack(selections["p_neural"].to_numpy())
    summary = {
        "decoder": args.decoder,
        "n_selections": int(len(selections)),
        "n_participants": int(selections["study_participant_id"].nunique()),
        "n_sessions": int(selections.groupby(["study_participant_id", "session_id"]).ngroups),
        "n_sessions_fit": int(len(decoders)),
        "decoder_auc_median": float(decoders["train_auc"].median()),
        "decoder_auc_cv_scheme": CV_SCHEMES[args.decoder],
        "n_sessions_auc_nan": int(decoders["train_auc"].isna().sum()),
        "archive_accuracy": float(selections["archive_correct"].mean()),
        "offline_neural_accuracy": float(
            (
                posterior_matrix.argmax(axis=1)
                == selections["target_symbol"].map(symbol_index).to_numpy()
            ).mean()
        ),
        "calibration_fallback_sessions": int(decoders["calibration_fallback"].sum()),
        "skipped": skipped,
    }
    (OUTPUT / f"strong_decoder_summary_{args.decoder}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "skipped"}, indent=2))
    print(f"skipped entries: {len(skipped)}")


if __name__ == "__main__":
    main()
