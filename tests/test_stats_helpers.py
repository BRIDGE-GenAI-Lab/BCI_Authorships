"""Regression tests for the statistics helpers.

The boolean-endog test guards a real defect found during the analysis: a boolean model
outcome is expanded by the formula layer into a two-level categorical whose first level is
False, so the model silently describes the complement and every odds ratio comes out
inverted.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_stats import (
    cast_binary_columns,
    cluster_bootstrap,
    cross_family_scale_check,
    fit_binary,
    participant_mean,
)


def gradient_frame(seed=0, n_participants=20, n_per=60):
    """Phantom agreement rises with position, so a correct model must return OR > 1."""
    rng = np.random.default_rng(seed)
    rows = []
    for participant in range(n_participants):
        for _ in range(n_per):
            position = int(rng.integers(0, 6))
            probability = 0.02 + 0.05 * position
            rows.append(
                {
                    "study_participant_id": f"P{participant}",
                    "position_in_phrase": position,
                    "phantom_agreement": bool(rng.random() < probability),
                }
            )
    return pd.DataFrame(rows)


def test_cast_binary_columns_converts_booleans_to_integers():
    frame = cast_binary_columns(gradient_frame())
    assert frame["phantom_agreement"].dtype.kind in "iu"
    assert set(frame["phantom_agreement"].unique()) <= {0, 1}


def test_fit_binary_refuses_a_boolean_outcome():
    with pytest.raises(ValueError, match="cast to integer"):
        fit_binary(gradient_frame(), "phantom_agreement ~ position_in_phrase")


def test_fit_binary_recovers_a_rising_gradient_as_an_odds_ratio_above_one():
    frame = cast_binary_columns(gradient_frame())
    result = fit_binary(frame, "phantom_agreement ~ position_in_phrase")
    odds_ratio = result["terms"]["position_in_phrase"]["odds_ratio"]
    assert odds_ratio > 1.0
    assert result["terms"]["position_in_phrase"]["p_value"] < 0.01


def test_cluster_bootstrap_brackets_the_point_estimate():
    frame = cast_binary_columns(gradient_frame())
    result = cluster_bootstrap(
        frame, lambda data: participant_mean(data, "phantom_agreement"), n_replicates=200
    )
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["bootstrap_replicates"] == 200


def ladder_frame():
    """A toy ladder: 3 families, enough points per family to exercise grouping,
    with a deliberately flat NCF/phantom relationship (no true trend) so the test
    checks structure and doesn't assert a spurious correlation direction. Prior
    capture and the accuracy inputs vary irregularly across priors so a
    correlation computed on them is not spuriously monotonic either.
    """
    rows = []
    specs = [
        ("uniform", "null", 0, 1.00, 0.00, 0.00, 0.50, 0.50),
        ("ngram5", "classical", 0, 0.87, 0.08, 0.05, 0.55, 0.52),
        ("gpt2", "gpt2", 124_000_000, 0.88, 0.07, 0.04, 0.60, 0.55),
        ("Qwen/Qwen2.5-1.5B", "qwen", 1_500_000_000, 0.86, 0.08, 0.06, 0.63, 0.58),
        ("Qwen/Qwen2.5-14B", "qwen", 14_000_000_000, 0.86, 0.07, 0.03, 0.65, 0.62),
        ("Qwen/Qwen2.5-32B", "qwen", 32_000_000_000, 0.86, 0.08, 0.07, 0.68, 0.64),
        ("meta-llama/Llama-3.1-8B", "llama", 8_000_000_000, 0.87, 0.08, 0.05, 0.64, 0.60),
        ("meta-llama/Llama-3.1-70B", "llama", 70_000_000_000, 0.86, 0.07, 0.02, 0.70, 0.69),
    ]
    for name, family, params, ncf, phantom, prior_capture, fused_correct, neural_correct in specs:
        for participant in range(10):
            rows.append(
                {
                    "prior_model": name,
                    "prior_parameters": params,
                    "study_participant_id": f"P{participant}",
                    "ncf": ncf,
                    "phantom_agreement": phantom,
                    "prior_capture": prior_capture,
                    "fused_correct": fused_correct,
                    "neural_correct": neural_correct,
                }
            )
    return pd.DataFrame(rows)


def test_cross_family_scale_check_reports_pooled_and_per_family_blocks():
    result = cross_family_scale_check(ladder_frame())
    assert "pooled" in result
    assert "spearman_rho_ncf_vs_log_params" in result["pooled"]
    assert "spearman_rho_phantom_vs_log_params" in result["pooled"]
    assert "spearman_rho_prior_capture_vs_log_params" in result["pooled"]
    assert "spearman_rho_accuracy_gained_vs_log_params" in result["pooled"]
    assert "by_family" in result
    assert set(result["by_family"]) == {"qwen", "llama"}  # null/classical/gpt2 excluded: no size variation
    assert "truncated_below_10b" in result
    assert "spearman_rho_prior_capture_vs_log_params" in result["truncated_below_10b"]
    assert "spearman_rho_accuracy_gained_vs_log_params" in result["truncated_below_10b"]


def test_cross_family_scale_check_excludes_zero_parameter_priors_from_the_correlation():
    frame = ladder_frame()
    result = cross_family_scale_check(frame)
    # uniform, ngram5 have 0 parameters and must not enter a log-parameter correlation
    assert result["pooled"]["n_priors"] == frame.loc[frame["prior_parameters"] > 0, "prior_model"].nunique()


def test_cross_family_scale_check_prior_capture_and_accuracy_gained_match_independent_computation():
    """eTable 9 reports prior capture and accuracy-gained correlations alongside ncf and
    phantom agreement; cross-check the digest values against a correlation computed
    directly from the fixture rather than through the function under test.
    """
    frame = ladder_frame()
    result = cross_family_scale_check(frame)

    positive_params = frame[frame["prior_parameters"] > 0]
    per_prior = positive_params.drop_duplicates("prior_model").set_index("prior_model")
    log_params = np.log10(per_prior["prior_parameters"])
    prior_capture = per_prior["prior_capture"]
    accuracy_gained = 100 * (per_prior["fused_correct"] - per_prior["neural_correct"])

    expected_capture = spearmanr(log_params, prior_capture)
    expected_gained = spearmanr(log_params, accuracy_gained)

    pooled = result["pooled"]
    assert pooled["spearman_rho_prior_capture_vs_log_params"]["rho"] == pytest.approx(
        expected_capture.statistic
    )
    assert pooled["spearman_rho_prior_capture_vs_log_params"]["p_value"] == pytest.approx(
        expected_capture.pvalue
    )
    assert pooled["spearman_rho_accuracy_gained_vs_log_params"]["rho"] == pytest.approx(
        expected_gained.statistic
    )
    assert pooled["spearman_rho_accuracy_gained_vs_log_params"]["p_value"] == pytest.approx(
        expected_gained.pvalue
    )

    truncated = per_prior[np.log10(per_prior["prior_parameters"]) < 10.0]
    truncated_log_params = np.log10(truncated["prior_parameters"])
    expected_truncated_capture = spearmanr(truncated_log_params, truncated["prior_capture"])
    truncated_block = result["truncated_below_10b"]
    assert truncated_block["spearman_rho_prior_capture_vs_log_params"]["rho"] == pytest.approx(
        expected_truncated_capture.statistic
    )
    assert truncated_block["n_priors"] == len(truncated)
