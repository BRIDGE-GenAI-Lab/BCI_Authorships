"""Attribute each emitted selection to neural evidence or to the language-model prior.

Two families of measure are produced for every selection.

The continuous measure is the neural contribution fraction. It compares how far the fused
posterior has been displaced away from what each source alone would have said. If the
neural evidence is uninformative the fused posterior equals the prior, the neural
displacement is zero, and the fraction is zero. If the prior is uniform the fused
posterior equals the neural posterior and the fraction is one. Displacement is measured
by Kullback-Leibler divergence in bits, so the fraction is bounded in [0, 1] by
construction without clipping.

The categorical measures ask what the emitted symbol would have been under each source
alone. Phantom agreement is a correct emission that the neural evidence alone would not
have produced. Prior capture is the reverse: a correct neural reading that the prior
overturned.
"""

from __future__ import annotations

import numpy as np

from authorship.grid import SYMBOLS

N_SYMBOLS = len(SYMBOLS)
PROBABILITY_FLOOR = 1e-12


def _validate(vector: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (N_SYMBOLS,):
        raise ValueError(f"{name} must have shape ({N_SYMBOLS},)")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    total = array.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1, got {total}")
    return np.maximum(array, PROBABILITY_FLOOR)


def fuse(p_neural: np.ndarray, p_lm: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Combine neural evidence and prior into the posterior a speller would act on."""
    neural = _validate(p_neural, "p_neural")
    prior = _validate(p_lm, "p_lm")
    log_fused = beta * np.log(neural) + np.log(prior)
    log_fused -= log_fused.max()
    weights = np.exp(log_fused)
    return weights / weights.sum()


def _kl_bits(first: np.ndarray, second: np.ndarray) -> float:
    """Kullback-Leibler divergence from ``second`` to ``first``, in bits."""
    return float(np.sum(first * (np.log2(first) - np.log2(second))))


def _centered_bits(distribution: np.ndarray, index: int) -> float:
    """Log-evidence for one symbol relative to the alphabet mean, in bits."""
    log_probabilities = np.log2(distribution)
    return float(log_probabilities[index] - log_probabilities.mean())


def attribution_row(
    p_neural: np.ndarray,
    p_lm: np.ndarray,
    fused: np.ndarray,
    target_index: int,
) -> dict[str, object]:
    """Return the attribution measures for a single emitted selection."""
    neural = _validate(p_neural, "p_neural")
    prior = _validate(p_lm, "p_lm")
    posterior = _validate(fused, "fused")
    if not 0 <= target_index < N_SYMBOLS:
        raise ValueError(f"target index out of range: {target_index}")

    neural_argmax = int(neural.argmax())
    prior_argmax = int(prior.argmax())
    fused_argmax = int(posterior.argmax())

    neural_displacement = _kl_bits(posterior, prior)
    prior_displacement = _kl_bits(posterior, neural)
    total_displacement = neural_displacement + prior_displacement
    ncf = neural_displacement / total_displacement if total_displacement > 1e-12 else np.nan

    fused_correct = fused_argmax == target_index
    neural_correct = neural_argmax == target_index
    prior_correct = prior_argmax == target_index

    neural_rank_of_target = int(np.sum(neural > neural[target_index]))
    neural_margin_to_target = float(neural[neural_argmax] - neural[target_index])

    return {
        "neural_argmax": neural_argmax,
        "prior_argmax": prior_argmax,
        "fused_argmax": fused_argmax,
        "emitted_symbol": SYMBOLS[fused_argmax],
        "neural_only_symbol": SYMBOLS[neural_argmax],
        "prior_only_symbol": SYMBOLS[prior_argmax],
        "fused_correct": bool(fused_correct),
        "neural_correct": bool(neural_correct),
        "prior_correct": bool(prior_correct),
        "phantom_agreement": bool(fused_correct and not neural_correct),
        "prior_capture": bool(neural_correct and not fused_correct),
        "neural_override": bool(fused_correct and neural_correct and not prior_correct),
        "ncf": float(ncf) if np.isfinite(ncf) else np.nan,
        "neural_displacement_bits": neural_displacement,
        "prior_displacement_bits": prior_displacement,
        "bits_neural_emitted": _centered_bits(neural, fused_argmax),
        "bits_prior_emitted": _centered_bits(prior, fused_argmax),
        "p_fused_emitted": float(posterior[fused_argmax]),
        "p_neural_emitted": float(neural[fused_argmax]),
        "p_prior_emitted": float(prior[fused_argmax]),
        "p_fused_target": float(posterior[target_index]),
        "neural_rank_of_target": neural_rank_of_target,
        "neural_margin_to_target": neural_margin_to_target,
        "neural_p_target": float(neural[target_index]),
    }


def log_odds_contribution(
    p_neural: np.ndarray, p_lm: np.ndarray, fused: np.ndarray, target_index: int
) -> float:
    """Share of the total log-odds shift toward the target symbol contributed by the prior.

    An alternative to the KL-based neural contribution fraction that does not require a
    full-distribution divergence: each source's own log-probability on the target symbol,
    relative to the uniform baseline, stands in for that source's independent conviction.
    """
    neural = _validate(p_neural, "p_neural")
    prior = _validate(p_lm, "p_lm")
    _validate(fused, "fused")
    baseline = np.log(1.0 / N_SYMBOLS)
    neural_shift = np.log(neural[target_index]) - baseline
    prior_shift = np.log(prior[target_index]) - baseline
    total = neural_shift + prior_shift
    if abs(total) < 1e-12:
        return float("nan")
    return float(prior_shift / total)


def shapley_contribution(
    p_neural: np.ndarray, p_lm: np.ndarray, fused: np.ndarray, target_index: int
) -> tuple[float, float]:
    """Exact two-player Shapley value for neural and prior evidence.

    The coalition value function is the log-probability the source(s) in the coalition
    assign the target symbol, relative to the uniform baseline for the empty coalition.
    With exactly two players the Shapley value reduces to averaging each player's marginal
    contribution to the coalition value across both join orders (no approximation needed).
    """
    neural = _validate(p_neural, "p_neural")
    prior = _validate(p_lm, "p_lm")
    posterior = _validate(fused, "fused")
    v_empty = np.log(1.0 / N_SYMBOLS)
    v_neural = np.log(neural[target_index])
    v_prior = np.log(prior[target_index])
    v_both = np.log(posterior[target_index])
    neural_shapley = 0.5 * (v_neural - v_empty) + 0.5 * (v_both - v_prior)
    prior_shapley = 0.5 * (v_prior - v_empty) + 0.5 * (v_both - v_neural)
    return float(neural_shapley), float(prior_shapley)
