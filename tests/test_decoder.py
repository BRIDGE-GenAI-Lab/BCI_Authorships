import numpy as np
import pytest

from authorship.decoder import ParticipantDecoder, fit_from_epochs


def synthetic_epochs(n_target=60, n_nontarget=240, n_channels=16, n_samples=256, seed=0):
    """Target epochs carry a positive deflection in the P300 latency band."""
    rng = np.random.default_rng(seed)
    n_total = n_target + n_nontarget
    epochs = rng.normal(0, 1e-6, size=(n_total, n_channels, n_samples))
    labels = np.r_[np.ones(n_target, dtype=int), np.zeros(n_nontarget, dtype=int)]
    p300_window = slice(int(0.45 * n_samples), int(0.65 * n_samples))
    epochs[labels == 1, :, p300_window] += 4e-6
    groups = np.array([f"file{index % 4}" for index in range(n_total)])
    order = rng.permutation(n_total)
    return epochs[order], labels[order], groups[order]


def test_fit_from_epochs_learns_a_discriminable_p300():
    epochs, labels, groups = synthetic_epochs()
    decoder = fit_from_epochs(epochs, labels, groups)
    assert isinstance(decoder, ParticipantDecoder)
    assert decoder.train_auc > 0.9
    assert decoder.n_target == 60
    assert decoder.n_nontarget == 240


def test_score_epochs_returns_finite_log_odds_ordered_by_label():
    epochs, labels, groups = synthetic_epochs()
    decoder = fit_from_epochs(epochs, labels, groups)
    scores = decoder.score_epochs(epochs)
    assert scores.shape == (len(epochs),)
    assert np.isfinite(scores).all()
    assert scores[labels == 1].mean() > scores[labels == 0].mean()


def test_fit_rejects_single_class_input():
    epochs, labels, groups = synthetic_epochs()
    with pytest.raises(ValueError):
        fit_from_epochs(epochs, np.zeros_like(labels), groups)


def test_fit_rejects_too_few_calibration_epochs():
    epochs, labels, groups = synthetic_epochs(n_target=3, n_nontarget=8)
    with pytest.raises(ValueError):
        fit_from_epochs(epochs, labels, groups)
