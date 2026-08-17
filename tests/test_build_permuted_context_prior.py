"""Regression tests for the context-permutation negative control."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_permutation_reassigns_contexts_without_repeating_the_true_context(tmp_path, monkeypatch):
    import build_permuted_context_prior as bp

    selections = pd.DataFrame({
        "context_prefix": ["A", "AB", "ABC", "B", "BA"],
        "target_symbol": ["B", "C", "D", "A", "B"],
    })
    permuted = bp.permute(selections, seed=1)
    assert len(permuted) == len(selections)
    assert (permuted["permuted_context_prefix"] != permuted["context_prefix"]).all()
    assert set(permuted["permuted_context_prefix"]) == set(selections["context_prefix"])


def test_permutation_avoids_value_level_collisions_when_a_context_repeats():
    """A same-index derangement is not sufficient: many rows in the real data share an
    identical context_prefix (e.g. the empty string at word position 1), so avoiding only
    the row's own index is not the same as avoiding its own context value. This is the
    degenerate case that would silently defeat the negative control.
    """
    import build_permuted_context_prior as bp

    n = 40
    # 20 of 40 rows share the empty-string context, matching the real file's dominant value.
    context = np.array([""] * 20 + [f"ctx{i}" for i in range(20)])
    selections = pd.DataFrame({
        "context_prefix": context,
        "target_symbol": ["X"] * n,
    })
    permuted = bp.permute(selections, seed=7)
    assert (permuted["permuted_context_prefix"].to_numpy() != permuted["context_prefix"].to_numpy()).all()
    assert sorted(permuted["permuted_context_prefix"]) == sorted(selections["context_prefix"])


def test_permutation_is_deterministic_given_a_fixed_seed():
    import build_permuted_context_prior as bp

    selections = pd.DataFrame({
        "context_prefix": ["A", "AB", "ABC", "B", "BA", "BAC"],
        "target_symbol": ["B", "C", "D", "A", "B", "A"],
    })
    first = bp.permute(selections, seed=42)
    second = bp.permute(selections, seed=42)
    assert first["permuted_context_prefix"].tolist() == second["permuted_context_prefix"].tolist()


def _stratified_fixture() -> pd.DataFrame:
    """Three strata with disjoint value alphabets ('a*', 'b*', 'c*'), so any donor value
    that crosses a (position_in_phrase, study) boundary is immediately detectable -- a bug
    that ignored stratification (but still avoided same-value self-matches) would still
    pass a same-value check, since it would draw from the full a/b/c pool and only
    occasionally collide with its own value, but it would very likely assign some row a
    value from a foreign stratum's alphabet.
    """
    return pd.DataFrame({
        "context_prefix": [
            "a1", "a2", "a3", "a1",
            "b1", "b2", "b3", "b1",
            "c1", "c2", "c3", "c1",
        ],
        "position_in_phrase": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        "study": ["S1"] * 4 + ["S2"] * 4 + ["S1"] * 4,
    })


def test_stratified_permutation_never_crosses_a_position_study_boundary():
    """Every permuted context_prefix must come from a row sharing the SAME
    (position_in_phrase, study) stratum as the row it replaced, not merely some other
    row's value from anywhere in the file (round-2 review Item 7).
    """
    import build_permuted_context_prior as bp

    selections = _stratified_fixture()
    permuted = bp.permute(selections, seed=3, stratum_columns=["position_in_phrase", "study"])

    assert permuted["permutable"].all()
    for row in permuted.itertuples():
        own_stratum = permuted[
            (permuted["position_in_phrase"] == row.position_in_phrase)
            & (permuted["study"] == row.study)
        ]
        assert row.permuted_context_prefix in set(own_stratum["context_prefix"])
        assert row.permuted_context_prefix != row.context_prefix


def test_unstratified_permutation_can_cross_the_same_boundary():
    """Negative control for the test above: confirms the boundary-crossing check is not
    vacuously true. Without stratum_columns, the same fixture's rows may legitimately
    receive a foreign-alphabet value, which is exactly what stratification is meant to
    prevent.
    """
    import build_permuted_context_prior as bp

    selections = _stratified_fixture()
    crossed_any_seed = False
    for seed in range(20):
        permuted = bp.permute(selections, seed=seed)
        for row in permuted.itertuples():
            own_stratum = permuted[
                (permuted["position_in_phrase"] == row.position_in_phrase)
                & (permuted["study"] == row.study)
            ]
            if row.permuted_context_prefix not in set(own_stratum["context_prefix"]):
                crossed_any_seed = True
                break
        if crossed_any_seed:
            break
    assert crossed_any_seed


def _length_bucket_fixture() -> pd.DataFrame:
    """A single (position_in_phrase, study) pair whose rows span two different
    context-length buckets (per LENGTH_BUCKET_BINS = [-1, 0, 2, 4, 6, 100]): a
    length-1 "short" sub-group (bucket (0, 2]) and a length-8 "long" sub-group
    (bucket (6, 100]), each with a disjoint value alphabet ('s'/'t'/'u' vs
    'longval1'/'longval2'/'longval3'). Every row shares the same position_in_phrase
    and study, so a regression that dropped context_length_bucket from the
    stratification key (while still correctly stratifying on position_in_phrase and
    study) would treat all 8 rows as one stratum and would very likely assign a
    short-bucket row a long-bucket donor value (or vice versa) -- immediately
    detectable via the disjoint alphabets, the same technique
    `_stratified_fixture` uses for the (position_in_phrase, study) axis.
    """
    return pd.DataFrame({
        "context_prefix": [
            "s", "t", "u", "s",
            "longval1", "longval2", "longval3", "longval1",
        ],
        "position_in_phrase": [5] * 8,
        "study": ["S1"] * 8,
    })


def test_stratified_permutation_never_crosses_a_context_length_bucket_boundary():
    """Every permuted context_prefix must come from a row sharing the SAME
    context-length bucket as the row it replaced, even when position_in_phrase and
    study are identical across the whole fixture (Task 14 fix-round Finding 1: the
    existing `_stratified_fixture` gave every row within a (position_in_phrase,
    study) group the same context length, so it could not independently detect a
    regression that dropped context_length_bucket from the groupby key).
    """
    import build_permuted_context_prior as bp

    selections = _length_bucket_fixture()
    permuted = bp.permute(selections, seed=11, stratum_columns=["position_in_phrase", "study"])

    assert permuted["permutable"].all()

    short_alphabet = {"s", "t", "u"}
    long_alphabet = {"longval1", "longval2", "longval3"}
    for row in permuted.itertuples():
        assert row.permuted_context_prefix != row.context_prefix
        if row.context_prefix in short_alphabet:
            assert row.permuted_context_prefix in short_alphabet
        else:
            assert row.context_prefix in long_alphabet
            assert row.permuted_context_prefix in long_alphabet


def test_stratified_permutation_excludes_structurally_unpermutable_strata():
    """A stratum where every row shares one identical value (e.g. position 0's real
    empty-string context) has no other real value to assign -- such rows must come back
    marked permutable=False with a null permuted_context_prefix, not silently left at
    their own true value and not a crash.
    """
    import build_permuted_context_prior as bp

    selections = pd.DataFrame({
        "context_prefix": ["", "", "", "x1", "x2", "x3"],
        "position_in_phrase": [0, 0, 0, 1, 1, 1],
        "study": ["S1"] * 6,
    })
    permuted = bp.permute(selections, seed=5, stratum_columns=["position_in_phrase", "study"])

    degenerate = permuted[permuted["position_in_phrase"] == 0]
    assert not degenerate["permutable"].any()
    assert degenerate["permuted_context_prefix"].isna().all()

    permutable_stratum = permuted[permuted["position_in_phrase"] == 1]
    assert permutable_stratum["permutable"].all()
    assert (permutable_stratum["permuted_context_prefix"] != permutable_stratum["context_prefix"]).all()
