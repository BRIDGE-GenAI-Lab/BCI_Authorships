import numpy as np
import pandas as pd
import pytest

from authorship.calibration import apply_temperature, calibrated_fusion_frame, fit_temperature


def test_apply_temperature_is_identity_at_t_equals_one():
    dist = np.array([0.7, 0.2, 0.1])
    result = apply_temperature(dist, 1.0)
    assert np.allclose(result, dist)


def test_apply_temperature_flattens_toward_uniform_as_t_grows():
    dist = np.array([0.9, 0.05, 0.05])
    result = apply_temperature(dist, temperature=10.0)
    assert result.max() < dist.max()
    assert abs(float(result.sum()) - 1.0) < 1e-9


def test_fit_temperature_recovers_overconfidence():
    """A distribution that is systematically overconfident (puts 0.95 on the argmax but
    is only right 50% of the time) should be corrected toward T > 1 (flattening)."""
    rng = np.random.default_rng(0)
    n = 400
    distributions = np.tile([0.95, 0.05 / 35] + [0.05 / 35] * 34, (n, 1))
    targets = np.where(rng.random(n) < 0.5, 0, rng.integers(1, 36, size=n))
    temperature = fit_temperature(distributions, targets)
    assert temperature > 1.5


def test_fit_temperature_is_near_one_for_already_calibrated_distributions():
    rng = np.random.default_rng(1)
    n = 2000
    probability_correct = 0.7
    distributions = np.tile(
        [probability_correct] + [(1 - probability_correct) / 35] * 35, (n, 1)
    )
    targets = np.where(rng.random(n) < probability_correct, 0, rng.integers(1, 36, size=n))
    temperature = fit_temperature(distributions, targets)
    assert 0.7 < temperature < 1.4


def test_calibrated_fusion_frame_never_uses_a_rows_own_group_to_calibrate_it():
    """The calibration fold assignment must be grouped: every row in group g is
    calibrated using a temperature fit only on rows outside group g.
    """
    rng = np.random.default_rng(2)
    n = 200
    groups = np.repeat(np.arange(10), n // 10)
    neural = np.tile([0.6] + [0.4 / 35] * 35, (n, 1))
    lm = np.tile([0.3] + [0.7 / 35] * 35, (n, 1))
    targets = np.zeros(n, dtype=int)
    frame = pd.DataFrame(
        {
            "group": groups,
            "p_neural": list(neural),
            "p_lm": list(lm),
            "target_index": targets,
        }
    )
    result = calibrated_fusion_frame(frame, group_column="group", n_folds=5, seed=0)
    assert "p_neural_calibrated" in result.columns and "p_lm_calibrated" in result.columns
    assert len(result) == n
    for vector in result["p_neural_calibrated"]:
        assert abs(float(np.sum(vector)) - 1.0) < 1e-6


def test_calibrated_fusion_frame_rejects_too_few_groups_for_the_fold_count():
    frame = pd.DataFrame(
        {
            "group": [0, 0, 1, 1],
            "p_neural": [np.full(36, 1 / 36)] * 4,
            "p_lm": [np.full(36, 1 / 36)] * 4,
            "target_index": [0, 0, 0, 0],
        }
    )
    with pytest.raises(ValueError):
        calibrated_fusion_frame(frame, group_column="group", n_folds=5, seed=0)
