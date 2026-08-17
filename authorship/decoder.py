"""Per-session P300 decoders fitted on calibration (Train) epochs only.

The decoder converts a single flash epoch into a log-odds score for the presence of a
P300 response. Every decoder is fitted on the calibration phase of one recording session
and is never shown Test-phase data, so the Test-phase posteriors it produces carry no
information from the online outcomes being analysed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_COMPANION_SRC = Path(__file__).resolve().parents[2] / "study_bigp3_als_calibration" / "src"
if str(_COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(_COMPANION_SRC))

from bigp3_als.features import (  # noqa: E402
    MIN_NONTARGET_EPOCHS,
    MIN_TARGET_EPOCHS,
    _downsampled_epoch_features,
    _extract_file_epochs,
    calibration_discriminability,
    shrinkage_lda_discriminability,
)

RANDOM_STATE = 20260726


@dataclass
class ParticipantDecoder:
    """A fitted calibration decoder plus the provenance needed to report it."""

    model: object
    train_auc: float
    n_target: int
    n_nontarget: int
    classifier: str = "logistic"
    sampling_frequency: float | None = None
    source_files: tuple[str, ...] = field(default_factory=tuple)

    def score_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """Return one P300 log-odds score per epoch."""
        if epochs.ndim != 3:
            raise ValueError("epochs must be (n_epochs, n_channels, n_samples)")
        features = _downsampled_epoch_features(epochs)
        return np.asarray(self.model.decision_function(features), dtype=float)


def _build_model(classifier: str):
    if classifier == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
        )
    if classifier == "shrinkage_lda":
        return make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        )
    raise ValueError(f"unknown classifier: {classifier}")


def fit_from_epochs(
    epochs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    classifier: str = "logistic",
    sampling_frequency: float | None = None,
    source_files: tuple[str, ...] = (),
) -> ParticipantDecoder:
    """Fit a calibration decoder and record its grouped cross-validated discriminability."""
    labels = np.asarray(labels, dtype=int)
    n_target = int((labels == 1).sum())
    n_nontarget = int((labels == 0).sum())
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration labels must contain target and non-target epochs")
    if n_target < MIN_TARGET_EPOCHS or n_nontarget < MIN_NONTARGET_EPOCHS:
        raise ValueError(
            f"insufficient calibration epochs: {n_target} target, {n_nontarget} non-target"
        )
    discriminability = (
        calibration_discriminability if classifier == "logistic" else shrinkage_lda_discriminability
    )
    train_auc = float(discriminability(epochs, labels, groups))
    model = _build_model(classifier)
    model.fit(_downsampled_epoch_features(epochs), labels)
    return ParticipantDecoder(
        model=model,
        train_auc=train_auc,
        n_target=n_target,
        n_nontarget=n_nontarget,
        classifier=classifier,
        sampling_frequency=sampling_frequency,
        source_files=source_files,
    )


def fit_session_decoder(
    train_paths: list[Path], *, classifier: str = "logistic"
) -> ParticipantDecoder:
    """Fit a decoder from every calibration EDF belonging to one session."""
    if not train_paths:
        raise ValueError("no calibration files supplied")
    epoch_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    sampling_frequencies: list[float] = []
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
    return fit_from_epochs(
        np.concatenate(epoch_blocks),
        np.concatenate(label_blocks),
        np.concatenate(group_blocks),
        classifier=classifier,
        sampling_frequency=float(np.mean(sampling_frequencies)),
        source_files=tuple(path.name for path in train_paths),
    )
