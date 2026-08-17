import numpy as np

from scripts.compare_scorer_methods import compare_priors


def test_compare_priors_reports_flip_when_argmax_differs():
    legacy = {"AB": np.array([0.9] + [0.1 / 35] * 35)}
    corrected = {"AB": np.array([0.1 / 35, 0.9] + [0.1 / 35] * 34)}
    result = compare_priors(legacy, corrected)
    assert result["argmax_flip_fraction"] == 1.0
    assert result["n_contexts"] == 1
    assert result["mean_l1_distance"] > 0.5


def test_compare_priors_reports_no_flip_when_priors_match():
    same = {"AB": np.array([1 / 36] * 36)}
    result = compare_priors(same, dict(same))
    assert result["argmax_flip_fraction"] == 0.0
    assert result["mean_l1_distance"] < 1e-9
