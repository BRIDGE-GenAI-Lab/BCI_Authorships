"""Stronger per-session P300 decoders for the decoder-strength sensitivity analysis.

Same contract as authorship.decoder.ParticipantDecoder: fitted on one session's
calibration (Train) epochs only, and score_epochs returns one log-odds score per
flash epoch, so authorship.neural_posterior.build_selection_posteriors can use either
decoder unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 20260726
TARGET_HZ = 128.0

LEAVE_ONE_FILE_OUT = "leave_one_file_out"
TWO_FOLD_GROUPED = "two_fold_grouped"
# EEGNet trains roughly a thousand times slower than xDAWN, so its calibration AUC uses a
# two-fold file split instead of leave-one-file-out; the final model is fitted on every
# calibration file either way.
CV_SCHEMES = {"xdawn_riemann": LEAVE_ONE_FILE_OUT, "eegnet": TWO_FOLD_GROUPED}


def resample_epochs(epochs: np.ndarray, from_hz: float, to_hz: float = TARGET_HZ) -> np.ndarray:
    from fractions import Fraction

    ratio = Fraction(to_hz / from_hz).limit_denominator(1000)
    return resample_poly(epochs, ratio.numerator, ratio.denominator, axis=-1)


def _xdawn_riemann():
    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace

    return make_pipeline(
        XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    )


def _torch():
    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch


class EEGNet:  # thin factory so importing this module does not require torch
    def __new__(cls, n_channels: int = 16, n_samples: int = 128, F1: int = 8, D: int = 2,
                F2: int = 16, kernel_length: int = 64, dropout: float = 0.25):
        torch = _torch()
        nn = torch.nn

        class _EEGNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                    nn.BatchNorm2d(F1),
                    nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                    nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                    nn.Conv2d(F1 * D, F2, 1, bias=False),
                    nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
                )
                with torch.no_grad():
                    flat = self.block2(self.block1(torch.zeros(1, 1, n_channels, n_samples))).numel()
                self.classify = nn.Linear(flat, 1)

            def forward(self, x):
                return self.classify(self.block2(self.block1(x.unsqueeze(1))).flatten(1)).squeeze(1)

        return _EEGNet()


class EEGNetClassifier:
    """sklearn-style wrapper: per-channel standardization fitted on training epochs,
    class-weighted BCE, Adam, early stopping on a held-out calibration file."""

    def __init__(self, max_epochs: int = 200, patience: int = 20, batch_size: int = 64, lr: float = 1e-3):
        self.max_epochs, self.patience, self.batch_size, self.lr = max_epochs, patience, batch_size, lr

    def _standardize(self, x):
        return (x - self.mean_) / self.std_

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        torch = _torch()
        torch.manual_seed(RANDOM_STATE)
        rng = np.random.default_rng(RANDOM_STATE)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            valid = groups == unique_groups[-1]
        else:
            valid = rng.random(len(y)) < 0.2
        self.mean_ = x[~valid].mean(axis=(0, 2), keepdims=True)
        self.std_ = x[~valid].std(axis=(0, 2), keepdims=True) + 1e-12
        xt = torch.tensor(self._standardize(x), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        train_idx, valid_idx = np.flatnonzero(~valid), np.flatnonzero(valid)
        self.model_ = EEGNet(n_channels=x.shape[1], n_samples=x.shape[2])
        positive = max(float(y[train_idx].sum()), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((len(train_idx) - positive) / positive))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        best_loss, best_state, stale = np.inf, None, 0
        self.n_epochs_run_ = 0
        for _ in range(self.max_epochs):
            self.n_epochs_run_ += 1
            self.model_.train()
            order = rng.permutation(train_idx)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                if len(batch) < 2:
                    continue
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xt[batch]), yt[batch])
                loss.backward()
                optimizer.step()
            self.model_.eval()
            with torch.no_grad():
                valid_loss = float(loss_fn(self.model_(xt[valid_idx]), yt[valid_idx]))
            if valid_loss < best_loss - 1e-6:
                best_loss, stale = valid_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is None:
            raise ValueError("EEGNet validation loss was never finite")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            return self.model_(torch.tensor(self._standardize(x), dtype=torch.float32)).numpy().astype(float)


@dataclass
class EpochDecoder:
    model: object
    train_auc: float
    n_target: int
    n_nontarget: int
    classifier: str
    sampling_frequency: float
    cv_scheme: str
    target_hz: float = TARGET_HZ

    def score_epochs(self, epochs: np.ndarray) -> np.ndarray:
        if epochs.ndim != 3:
            raise ValueError("epochs must be (n_epochs, n_channels, n_samples)")
        x = resample_epochs(epochs, self.sampling_frequency, self.target_hz)
        return np.asarray(self.model.decision_function(x), dtype=float)


def _fit_model(classifier: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    if classifier == "xdawn_riemann":
        return _xdawn_riemann().fit(x, y)
    if classifier == "eegnet":
        return EEGNetClassifier().fit(x, y, groups)
    raise ValueError(f"unknown classifier: {classifier}")


def two_fold_file_split(groups: np.ndarray) -> np.ndarray:
    """Assign calibration files alternately to fold 0 and 1, in sorted file-name order."""
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)  # sorted
    fold_of_group = {group: position % 2 for position, group in enumerate(unique_groups)}
    return np.array([fold_of_group[group] for group in groups], dtype=int)


def _held_out_auc(classifier: str, x, y, groups, held_out_masks) -> float:
    scores = np.full(len(y), np.nan)
    for test in held_out_masks:
        if len(np.unique(y[~test])) < 2:
            continue
        model = _fit_model(classifier, x[~test], y[~test], groups[~test])
        scores[test] = model.decision_function(x[test])
    finite = np.isfinite(scores)
    if len(np.unique(y[finite])) < 2:
        return float("nan")
    return float(roc_auc_score(y[finite], scores[finite]))


def _grouped_cv_auc(classifier: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    scheme: str) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return float("nan")
    if scheme == LEAVE_ONE_FILE_OUT:
        masks = [groups == held for held in unique_groups]
    elif scheme == TWO_FOLD_GROUPED:
        folds = two_fold_file_split(groups)
        masks = [folds == 0, folds == 1]
    else:
        raise ValueError(f"unknown cross-validation scheme: {scheme}")
    return _held_out_auc(classifier, x, y, groups, masks)


def fit_strong_decoder(epochs, labels, groups, *, classifier: str, sampling_frequency: float) -> EpochDecoder:
    labels = np.asarray(labels, dtype=int)
    x = resample_epochs(np.asarray(epochs, dtype=float), sampling_frequency, TARGET_HZ)
    groups = np.asarray(groups)
    if classifier not in CV_SCHEMES:
        raise ValueError(f"unknown classifier: {classifier}")
    scheme = CV_SCHEMES[classifier]
    return EpochDecoder(
        model=_fit_model(classifier, x, labels, groups),
        train_auc=_grouped_cv_auc(classifier, x, labels, groups, scheme),
        n_target=int((labels == 1).sum()),
        n_nontarget=int((labels == 0).sum()),
        classifier=classifier,
        sampling_frequency=float(sampling_frequency),
        cv_scheme=scheme,
    )
