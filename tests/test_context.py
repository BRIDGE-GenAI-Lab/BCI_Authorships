import pandas as pd

from authorship.context import add_context


def frame(rows):
    return pd.DataFrame(rows)


def test_context_prefix_accumulates_intended_symbols_within_a_file():
    trials = frame(
        [
            {"relative_path": "f1", "trial_number": 1, "target_symbol": "C"},
            {"relative_path": "f1", "trial_number": 2, "target_symbol": "A"},
            {"relative_path": "f1", "trial_number": 3, "target_symbol": "T"},
        ]
    )
    result = add_context(trials).sort_values("trial_number")
    assert list(result["context_prefix"]) == ["", "C", "CA"]
    assert list(result["position_in_phrase"]) == [0, 1, 2]
    assert set(result["intended_phrase"]) == {"CAT"}


def test_context_does_not_leak_across_files():
    trials = frame(
        [
            {"relative_path": "f1", "trial_number": 1, "target_symbol": "C"},
            {"relative_path": "f2", "trial_number": 1, "target_symbol": "D"},
            {"relative_path": "f2", "trial_number": 2, "target_symbol": "O"},
        ]
    )
    result = add_context(trials).set_index(["relative_path", "trial_number"])
    assert result.loc[("f2", 1), "context_prefix"] == ""
    assert result.loc[("f2", 2), "context_prefix"] == "D"


def test_context_is_ordered_by_trial_number_not_row_order():
    trials = frame(
        [
            {"relative_path": "f1", "trial_number": 3, "target_symbol": "T"},
            {"relative_path": "f1", "trial_number": 1, "target_symbol": "C"},
            {"relative_path": "f1", "trial_number": 2, "target_symbol": "A"},
        ]
    )
    result = add_context(trials).set_index("trial_number")
    assert result.loc[3, "context_prefix"] == "CA"


def test_numeric_targets_are_flagged_as_a_low_predictability_stratum():
    trials = frame(
        [
            {"relative_path": "f1", "trial_number": index + 1, "target_symbol": symbol}
            for index, symbol in enumerate("222475")
        ]
    )
    result = add_context(trials)
    assert result["phrase_is_numeric"].all()

    words = frame(
        [
            {"relative_path": "f2", "trial_number": index + 1, "target_symbol": symbol}
            for index, symbol in enumerate("SPEECH")
        ]
    )
    assert not add_context(words)["phrase_is_numeric"].any()
