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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_COMPANION_SRC = Path(__file__).resolve().parents[2] / "study_bigp3_als_calibration" / "src"
if str(_COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(_COMPANION_SRC))

from bigp3_als.features import (  # noqa: E402
    MIN_NONTARGET_EPOCHS,
    MIN_TARGET_EPOCHS,
    _extract_file_epochs,
)

RANDOM_STATE = 20260726

# The reported analysis represented each epoch by every 12th sample. The companion package
# later moved its own feature stride to 4, so this repository keeps the stride it was run
# with rather than importing the companion's current feature function. At 256 Hz a stride of
# 12 leaves a 10.7 Hz Nyquist, so 10.7 to 30 Hz content folds onto lower frequencies; this is
# a property of the reported decoder and is kept only so the reported numbers regenerate
# exactly.
REPORTED_FEATURE_STRIDE = 12
CALIBRATION_CV_RANDOM_STATE = 20260718


def _epoch_features(epochs: np.ndarray, stride: int = REPORTED_FEATURE_STRIDE) -> np.ndarray:
    """Flatten epochs after keeping every stride-th sample, as the reported decoder did."""
    if stride < 1:
        raise ValueError("feature stride must be at least 1")
    return epochs[:, :, ::stride].reshape(len(epochs), -1)


def calibration_auc(
    epochs: np.ndarray, labels: np.ndarray, groups: np.ndarray, classifier: str = "logistic"
) -> float:
    """Grouped cross-validated calibration AUC on the reported feature representation.

    Mirrors the companion package's grouped estimator (StratifiedGroupKFold over calibration
    files, up to five folds, fixed seed) but on _epoch_features, so it reproduces the
    train_auc recorded in the reported analysis.
    """
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    if epochs.ndim != 3 or len(epochs) != len(labels) or len(labels) != len(groups):
        raise ValueError("epochs, labels, and groups must align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration labels must contain target and non-target epochs")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two calibration files are required for grouped validation")
    features = _epoch_features(epochs)
    splitter = StratifiedGroupKFold(
        n_splits=min(5, len(unique_groups)), shuffle=True, random_state=CALIBRATION_CV_RANDOM_STATE
    )
    predictions = np.full(len(labels), np.nan)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        model = _build_model(classifier, random_state=CALIBRATION_CV_RANDOM_STATE)
        model.fit(features[train_indices], labels[train_indices])
        predictions[test_indices] = model.predict_proba(features[test_indices])[:, 1]
    if np.isnan(predictions).any():
        raise ValueError("grouped cross-validation did not predict every calibration epoch")
    return float(roc_auc_score(labels, predictions))


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
        features = _epoch_features(epochs)
        return np.asarray(self.model.decision_function(features), dtype=float)


def _build_model(classifier: str, random_state: int = RANDOM_STATE):
    if classifier == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=random_state
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
    train_auc = calibration_auc(epochs, labels, groups, classifier)
    model = _build_model(classifier)
    model.fit(_epoch_features(epochs), labels)
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
