"""Turn Test-phase flashes into a 36-way neural posterior for every online selection.

Evidence accumulates additively in log space across the flashes of one selection. Flash
membership is read from the per-symbol EDF channels rather than from row and column
codes, because checkerboard conditions illuminate arbitrary symbol subsets and a
row-column reading would attribute evidence to the wrong symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

_COMPANION_SRC = Path(__file__).resolve().parents[2] / "study_bigp3_als_calibration" / "src"
if str(_COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(_COMPANION_SRC))

from bigp3_als.edf import SHARED_EEG_CHANNELS, parse_source_path  # noqa: E402
from bigp3_als.features import (  # noqa: E402
    ARTIFACT_THRESHOLD_UV,
    EPOCH_END_SECONDS,
    EPOCH_START_SECONDS,
    _bandpass,
)
from bigp3_als.trials import reconstruct_online_trials  # noqa: E402

from authorship.decoder import ParticipantDecoder
from authorship.grid import FLASH_CHANNELS, target_code_to_symbol

REQUIRED_STREAMS = (
    "PhaseInSequence",
    "CurrentTarget",
    "SelectedTarget",
    "DisplayResults",
    "FakeFeedback",
    "StimulusBegin",
)


def accumulate_evidence(flash_membership: np.ndarray, flash_scores: np.ndarray) -> np.ndarray:
    """Accumulate per-flash log-odds onto the symbols illuminated by each flash."""
    membership = np.asarray(flash_membership, dtype=float)
    scores = np.asarray(flash_scores, dtype=float)
    if membership.ndim != 2 or membership.shape[1] != len(FLASH_CHANNELS):
        raise ValueError("flash membership must be (n_flashes, 36)")
    if scores.ndim != 1 or len(scores) != len(membership):
        raise ValueError("flash scores must align with flash membership")
    log_evidence = membership.T @ scores
    log_evidence = log_evidence - log_evidence.max()
    weights = np.exp(log_evidence)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("evidence accumulation produced a degenerate posterior")
    return weights / total


def phase2_window(phase: np.ndarray, phase3_start: int) -> tuple[int, int] | None:
    """Return the contiguous stimulation window immediately preceding a feedback phase."""
    values = np.rint(np.asarray(phase)).astype(int)
    preceding = phase3_start - 1
    if preceding < 0 or values[preceding] != 2:
        return None
    start = preceding
    while start > 0 and values[start - 1] == 2:
        start -= 1
    return start, phase3_start


def build_selection_posteriors(
    edf_path: Path, relative_path: str, decoder: ParticipantDecoder
) -> pd.DataFrame:
    """Return one row per eligible online selection with its 36-way neural posterior."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    missing = sorted(
        (set(REQUIRED_STREAMS) | set(FLASH_CHANNELS) | set(SHARED_EEG_CHANNELS)) - set(raw.ch_names)
    )
    if missing:
        raise ValueError(f"missing channels in {edf_path}: {', '.join(missing)}")
    sampling_frequency = float(raw.info["sfreq"])
    eeg = _bandpass(raw.get_data(picks=list(SHARED_EEG_CHANNELS)), sampling_frequency)
    streams = {name: raw.get_data(picks=[name])[0] for name in REQUIRED_STREAMS}
    membership_all = np.rint(raw.get_data(picks=list(FLASH_CHANNELS))).astype(int)
    phase = np.rint(streams["PhaseInSequence"]).astype(int)
    stimulus_begin = np.rint(streams["StimulusBegin"]).astype(int)
    rising = (stimulus_begin == 1) & np.r_[True, stimulus_begin[:-1] != 1]

    pre_samples = round(-EPOCH_START_SECONDS * sampling_frequency)
    post_samples = round(EPOCH_END_SECONDS * sampling_frequency)
    source = parse_source_path(relative_path)
    trials = reconstruct_online_trials(
        {name: streams[name] for name in REQUIRED_STREAMS if name != "StimulusBegin"}
    )

    rows: list[dict[str, object]] = []
    for trial in trials:
        if not trial.eligible:
            continue
        window = phase2_window(phase, trial.phase3_sample)
        if window is None:
            continue
        start, end = window
        onsets = np.flatnonzero(rising[start:end]) + start
        usable = (onsets >= pre_samples) & (onsets + post_samples <= eeg.shape[1])
        onsets = onsets[usable]
        if not len(onsets):
            continue
        epochs = np.stack(
            [eeg[:, onset - pre_samples : onset + post_samples] for onset in onsets]
        )
        epochs = epochs - epochs[:, :, :pre_samples].mean(axis=2, keepdims=True)
        artifact_free = np.max(np.abs(epochs), axis=(1, 2)) * 1e6 <= ARTIFACT_THRESHOLD_UV
        if artifact_free.sum() < 2:
            continue
        scores = decoder.score_epochs(epochs[artifact_free])
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite decoder scores in {relative_path}")
        membership = membership_all[:, onsets[artifact_free]].T.astype(float)
        posterior = accumulate_evidence(membership, scores)
        rows.append(
            {
                "study": source.study,
                "participant_id": source.participant_id,
                "study_participant_id": source.study_participant_id,
                "session_id": source.session_id,
                "condition": source.condition,
                "relative_path": relative_path,
                "trial_number": trial.trial_number,
                "target_code": trial.target,
                "target_symbol": target_code_to_symbol(int(trial.target)),
                "archive_selected_symbol": target_code_to_symbol(int(trial.selected)),
                "archive_correct": bool(trial.correct),
                "n_flashes": int(artifact_free.sum()),
                "n_flashes_dropped": int((~artifact_free).sum()),
                "p_neural": posterior.astype(float),
            }
        )
    return pd.DataFrame(rows)
