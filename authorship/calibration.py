"""Held-out, per-source temperature calibration for fusing neural and language-model
posteriors. Fits an independent temperature for each source (unlike a shared fusion
exponent, which can only reweight one source against the other; unlike scaling both
sources by the same power, which cannot move either source's own argmax) on training
folds grouped by participant or session, then applies it to a held-out fold -- so no
selection's calibrated distributions are ever informed by that selection's own group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROBABILITY_FLOOR = 1e-12


def apply_temperature(distribution: np.ndarray, temperature: float) -> np.ndarray:
    """Raise to power 1/T and renormalize. T=1 is the identity."""
    array = np.asarray(distribution, dtype=float)
    scaled = np.power(np.maximum(array, PROBABILITY_FLOOR), 1.0 / temperature)
    return scaled / scaled.sum()


def _mean_negative_log_likelihood(temperature: float, distributions: np.ndarray, target_indices: np.ndarray) -> float:
    scaled = np.power(np.maximum(distributions, PROBABILITY_FLOOR), 1.0 / temperature)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    target_probabilities = scaled[np.arange(len(target_indices)), target_indices]
    return float(-np.mean(np.log(np.maximum(target_probabilities, PROBABILITY_FLOOR))))


def mean_negative_log_likelihood(
    distributions: np.ndarray, target_indices: np.ndarray, temperature: float = 1.0
) -> float:
    """Public entry point for the NLL primitive fit_temperature already optimizes
    internally -- exposed so callers can report the pre-scaling (T=1) or post-scaling
    NLL of any source's own probability distributions, not only use it as an
    optimization objective.
    """
    return _mean_negative_log_likelihood(temperature, np.asarray(distributions), np.asarray(target_indices))


def fit_temperature(distributions: np.ndarray, target_indices: np.ndarray) -> float:
    """Fit a single scalar temperature minimizing mean NLL of the true class, bounded
    to [0.05, 20] (a distribution this far from T=1 is already numerically degenerate
    in either direction for a 36-way alphabet).
    """
    result = minimize_scalar(
        _mean_negative_log_likelihood,
        bounds=(0.05, 20.0),
        method="bounded",
        args=(np.asarray(distributions), np.asarray(target_indices)),
    )
    return float(result.x)


def calibrated_fusion_frame(
    frame: pd.DataFrame, group_column: str, n_folds: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Add p_neural_calibrated / p_lm_calibrated columns via grouped K-fold temperature
    fitting: for each fold, fit T_neural and T_lm on every OTHER fold's rows, apply to
    this fold's rows.

    Also records, per row, the exact (t_neural, t_lm) pair that row was calibrated with
    (t_neural_used / t_lm_used columns) -- callers that need to re-apply the SAME
    real-data-fit temperature to a different (e.g. permuted) view of that same row, without
    refitting, can look these up directly instead of re-deriving fold membership.
    """
    from sklearn.model_selection import GroupKFold

    groups = frame[group_column].to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_folds:
        raise ValueError(
            f"{len(unique_groups)} groups is fewer than n_folds={n_folds}; "
            "reduce n_folds or supply a group_column with more distinct values"
        )
    neural_matrix = np.stack(frame["p_neural"].map(np.asarray).to_numpy())
    lm_matrix = np.stack(frame["p_lm"].map(np.asarray).to_numpy())
    target_indices = frame["target_index"].to_numpy(dtype=int)

    calibrated_neural = np.zeros_like(neural_matrix)
    calibrated_lm = np.zeros_like(lm_matrix)
    t_neural_used = np.zeros(len(frame))
    t_lm_used = np.zeros(len(frame))
    fold_temperatures = []
    splitter = GroupKFold(n_splits=n_folds)
    for train_index, held_out_index in splitter.split(neural_matrix, groups=groups):
        t_neural = fit_temperature(neural_matrix[train_index], target_indices[train_index])
        t_lm = fit_temperature(lm_matrix[train_index], target_indices[train_index])
        for row in held_out_index:
            calibrated_neural[row] = apply_temperature(neural_matrix[row], t_neural)
            calibrated_lm[row] = apply_temperature(lm_matrix[row], t_lm)
            t_neural_used[row] = t_neural
            t_lm_used[row] = t_lm
        fold_temperatures.append({"t_neural": t_neural, "t_lm": t_lm, "n_held_out": len(held_out_index)})

    result = frame.copy()
    result["p_neural_calibrated"] = list(calibrated_neural)
    result["p_lm_calibrated"] = list(calibrated_lm)
    result["t_neural_used"] = t_neural_used
    result["t_lm_used"] = t_lm_used
    result.attrs["fold_temperatures"] = fold_temperatures
    return result
