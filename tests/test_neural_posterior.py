import numpy as np
import pytest

from authorship.neural_posterior import accumulate_evidence, phase2_window


def make_membership(n_flashes=120, n_active=4, target_index=7, seed=0):
    """Checkerboard-like membership where each flash illuminates n_active symbols."""
    rng = np.random.default_rng(seed)
    membership = np.zeros((n_flashes, 36), dtype=float)
    for flash in range(n_flashes):
        others = rng.choice([i for i in range(36) if i != target_index], n_active - 1, replace=False)
        membership[flash, others] = 1.0
        if flash % 9 == 0:
            membership[flash, target_index] = 1.0
    return membership


def test_posterior_is_a_distribution_over_the_alphabet():
    membership = make_membership()
    scores = np.zeros(len(membership))
    posterior = accumulate_evidence(membership, scores)
    assert posterior.shape == (36,)
    assert abs(posterior.sum() - 1.0) < 1e-9
    assert (posterior >= 0).all()


def test_posterior_concentrates_on_the_consistently_scored_symbol():
    target_index = 7
    membership = make_membership(target_index=target_index)
    scores = np.where(membership[:, target_index] == 1, 2.0, -0.5)
    posterior = accumulate_evidence(membership, scores)
    assert int(posterior.argmax()) == target_index
    assert posterior[target_index] > 0.5


def test_uniform_scores_give_a_near_uniform_posterior_under_balanced_membership():
    membership = np.zeros((36, 36))
    for index in range(36):
        membership[index, index] = 1.0
    posterior = accumulate_evidence(membership, np.zeros(36))
    assert np.allclose(posterior, 1 / 36)


def test_accumulate_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        accumulate_evidence(np.zeros((10, 36)), np.zeros(9))


def test_phase2_window_returns_the_contiguous_block_before_feedback():
    phase = np.array([1, 1, 2, 2, 2, 3, 3, 1, 2, 2, 3])
    assert phase2_window(phase, phase3_start=5) == (2, 5)
    assert phase2_window(phase, phase3_start=10) == (8, 10)


def test_phase2_window_returns_none_when_feedback_is_not_preceded_by_stimulation():
    phase = np.array([1, 1, 3, 3])
    assert phase2_window(phase, phase3_start=2) is None
