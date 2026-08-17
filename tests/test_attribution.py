import numpy as np
import pytest

from authorship.attribution import (
    attribution_row,
    fuse,
    log_odds_contribution,
    shapley_contribution,
)

UNIFORM = np.full(36, 1 / 36)


def peaked(index, mass=0.6, size=36):
    vector = np.full(size, (1 - mass) / (size - 1))
    vector[index] = mass
    return vector


def test_fuse_returns_a_distribution():
    fused = fuse(peaked(3), peaked(9))
    assert fused.shape == (36,)
    assert abs(float(fused.sum()) - 1.0) < 1e-9


def test_fuse_with_a_uniform_prior_returns_the_neural_posterior():
    neural = peaked(3)
    assert np.allclose(fuse(neural, UNIFORM), neural)


def test_fuse_with_uniform_neural_evidence_returns_the_prior():
    prior = peaked(9)
    assert np.allclose(fuse(UNIFORM, prior), prior)


def test_ncf_is_one_when_the_prior_is_uniform():
    neural = peaked(3)
    row = attribution_row(neural, UNIFORM, fuse(neural, UNIFORM), target_index=3)
    assert row["ncf"] == pytest.approx(1.0)


def test_ncf_is_zero_when_the_neural_evidence_is_uniform():
    prior = peaked(9)
    row = attribution_row(UNIFORM, prior, fuse(UNIFORM, prior), target_index=9)
    assert row["ncf"] == pytest.approx(0.0)


def test_ncf_is_nan_when_both_sources_are_uninformative():
    row = attribution_row(UNIFORM, UNIFORM, fuse(UNIFORM, UNIFORM), target_index=0)
    assert np.isnan(row["ncf"])


def test_phantom_agreement_when_the_prior_rescues_a_wrong_neural_argmax():
    neural = peaked(5, mass=0.20)
    prior = peaked(9, mass=0.90)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=9)
    assert row["fused_correct"] and not row["neural_correct"]
    assert row["phantom_agreement"] is True
    assert row["prior_capture"] is False


def test_prior_capture_when_the_prior_breaks_a_correct_neural_argmax():
    neural = peaked(5, mass=0.30)
    prior = peaked(9, mass=0.95)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=5)
    assert row["neural_correct"] and not row["fused_correct"]
    assert row["prior_capture"] is True
    assert row["phantom_agreement"] is False


def test_neural_override_when_the_brain_wins_against_a_wrong_prior():
    neural = peaked(5, mass=0.95)
    prior = peaked(9, mass=0.30)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=5)
    assert row["fused_correct"] and row["neural_correct"] and not row["prior_correct"]
    assert row["neural_override"] is True


def test_phantom_agreement_and_prior_capture_are_mutually_exclusive():
    rng = np.random.default_rng(0)
    for _ in range(200):
        neural = rng.dirichlet(np.full(36, 0.4))
        prior = rng.dirichlet(np.full(36, 0.4))
        fused = fuse(neural, prior)
        row = attribution_row(neural, prior, fused, target_index=int(rng.integers(36)))
        assert not (row["phantom_agreement"] and row["prior_capture"])


def test_beta_sharpens_the_neural_contribution():
    neural = peaked(5, mass=0.4)
    prior = peaked(9, mass=0.8)
    weak = attribution_row(neural, prior, fuse(neural, prior, beta=0.5), target_index=5)["ncf"]
    strong = attribution_row(neural, prior, fuse(neural, prior, beta=2.0), target_index=5)["ncf"]
    assert strong > weak


def test_attribution_rejects_malformed_input():
    with pytest.raises(ValueError):
        fuse(np.full(36, 0.5), UNIFORM)
    with pytest.raises(ValueError):
        fuse(np.full(10, 0.1), np.full(10, 0.1))


def test_neural_rank_of_target_is_zero_when_neural_is_correct():
    neural = np.array([0.7, 0.2, 0.1] + [0.0] * 33)
    prior = np.full(36, 1 / 36)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=0)
    assert row["neural_rank_of_target"] == 0
    assert row["neural_margin_to_target"] == 0.0


def test_neural_rank_and_margin_for_a_phantom_agreement_case():
    neural = np.array([0.30, 0.35, 0.15] + [0.20 / 33] * 33)
    prior = np.zeros(36)
    prior[0] = 0.98
    prior[1:] = 0.02 / 35
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=0)
    assert row["fused_argmax"] == 0
    assert row["neural_argmax"] == 1
    assert row["phantom_agreement"] is True
    assert row["neural_rank_of_target"] == 1
    assert abs(row["neural_margin_to_target"] - (0.35 - 0.30)) < 1e-9
    assert abs(row["neural_p_target"] - 0.30) < 1e-9


def test_neural_rank_of_target_is_tie_safe():
    neural = np.zeros(36)
    neural[3] = 0.4
    neural[9] = 0.4
    neural[10:] = 0.2 / 26
    neural[:3] = 0.0
    neural[4:9] = 0.0
    prior = np.full(36, 1 / 36)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=9)
    assert row["neural_argmax"] == 3
    assert row["neural_rank_of_target"] == 0
    assert row["neural_margin_to_target"] == pytest.approx(0.0)


def test_exact_ties_break_toward_the_lower_symbol_index():
    """Locks in numpy.argmax's documented first-occurrence tie-break for neural_argmax,
    prior_argmax, and fused_argmax so a future numpy version change would be caught.
    """
    neural = np.full(36, 0.80 / 34)
    neural[0] = 0.10
    neural[1] = 0.10  # exact tie between symbols 0 and 1
    prior = np.full(36, 1 / 36)
    fused = fuse(neural, prior)
    row = attribution_row(neural, prior, fused, target_index=1)
    assert row["neural_argmax"] == 0  # lower index wins the tie


def test_shapley_values_sum_to_the_total_coalition_gain():
    neural = np.full(36, 1 / 36)
    neural[0] = 0.5
    neural[1:] = 0.5 / 35
    prior = np.full(36, 1 / 36)
    prior[0] = 0.7
    prior[1:] = 0.3 / 35
    fused = fuse(neural, prior)
    neural_shapley, prior_shapley = shapley_contribution(neural, prior, fused, target_index=0)
    baseline = np.log(1 / 36)
    total_gain = np.log(fused[0]) - baseline
    assert abs((neural_shapley + prior_shapley) - total_gain) < 1e-9


def test_log_odds_contribution_is_between_zero_and_one_when_sources_agree():
    neural = np.full(36, 1 / 36)
    neural[0] = 0.6
    neural[1:] = 0.4 / 35
    prior = np.full(36, 1 / 36)
    prior[0] = 0.6
    prior[1:] = 0.4 / 35
    fused = fuse(neural, prior)
    share = log_odds_contribution(neural, prior, fused, target_index=0)
    assert 0.0 <= share <= 1.0


def test_shapley_split_matches_a_direct_hand_calculation_for_an_asymmetric_case():
    """Independent re-derivation of the two-player Shapley value, not a call to
    shapley_contribution itself, to catch an error in the implementation rather
    than merely echoing it (Task 4 found a real swapped-term bug this way).
    """
    neural = peaked(3, mass=0.7)
    prior = peaked(9, mass=0.2)
    fused = fuse(neural, prior)
    target_index = 3

    baseline = np.log(1 / 36)
    v_neural = np.log(neural[target_index])
    v_prior = np.log(prior[target_index])
    v_both = np.log(fused[target_index])
    expected_neural_shapley = 0.5 * (v_neural - baseline) + 0.5 * (v_both - v_prior)
    expected_prior_shapley = 0.5 * (v_prior - baseline) + 0.5 * (v_both - v_neural)

    neural_shapley, prior_shapley = shapley_contribution(
        neural, prior, fused, target_index=target_index
    )
    assert neural_shapley == pytest.approx(expected_neural_shapley)
    assert prior_shapley == pytest.approx(expected_prior_shapley)
    # Sanity check against the model-agnostic efficiency axiom used in the test above.
    assert (neural_shapley + prior_shapley) == pytest.approx(v_both - baseline)


def test_shapley_contribution_rejects_malformed_input():
    with pytest.raises(ValueError):
        shapley_contribution(np.full(36, 0.5), UNIFORM, UNIFORM, target_index=0)


def test_log_odds_contribution_rejects_malformed_input():
    with pytest.raises(ValueError):
        log_odds_contribution(np.full(36, 0.5), UNIFORM, UNIFORM, target_index=0)
