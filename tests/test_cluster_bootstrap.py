"""A participant drawn k times in one bootstrap replicate must count as k clusters.

The original bootstrap concatenated the resampled participant blocks but kept each block's
original participant id, so participant_mean regrouped a participant drawn twice into one
group of weight 1. Each replicate then behaved like roughly 30 of 47 participants drawn
without replacement, and every participant-weighted interval came out about a fifth too
narrow. make_figures.participant_bootstrap copied the same behaviour (a set of drawn names)
so that it would match the digest.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_stats import (  # noqa: E402
    CLUSTER,
    _relabelled_resample,
    cluster_bootstrap,
    participant_mean,
)
from make_figures import _mean_over_draws, participant_bootstrap  # noqa: E402


def three_participants():
    return pd.DataFrame(
        {
            CLUSTER: ["a", "a", "b", "c", "c"],
            "v": [0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )


def test_relabelled_resample_counts_a_repeated_draw_as_two_clusters():
    frame = three_participants()
    blocks = {name: block for name, block in frame.groupby(CLUSTER)}
    resampled = _relabelled_resample(blocks, ["c", "c", "a"])
    assert resampled[CLUSTER].nunique() == 3
    assert len(resampled) == 6
    assert participant_mean(resampled, "v") == pytest.approx(2 / 3)


def test_relabelled_resample_leaves_the_source_blocks_untouched():
    frame = three_participants()
    blocks = {name: block for name, block in frame.groupby(CLUSTER)}
    _relabelled_resample(blocks, ["c", "c", "a"])
    assert sorted(blocks["c"][CLUSTER].unique()) == ["c"]


def test_figure_bootstrap_mean_counts_a_repeated_draw_twice():
    values = {"a": 0.0, "b": 0.0, "c": 1.0}
    assert _mean_over_draws(values, ["c", "c", "a"]) == pytest.approx(2 / 3)


def test_figure_bootstrap_matches_the_study_bootstrap():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            CLUSTER: [f"P{i}" for i in range(12) for _ in range(5)],
            "v": rng.random(60),
        }
    )
    study = cluster_bootstrap(frame, lambda d: participant_mean(d, "v"), n_replicates=300)
    figure = participant_bootstrap(frame, "v", n_replicates=300)
    assert figure["estimate"] == pytest.approx(study["estimate"], rel=1e-12)
    assert figure["ci_low"] == pytest.approx(study["ci_low"], rel=1e-12)
    assert figure["ci_high"] == pytest.approx(study["ci_high"], rel=1e-12)
