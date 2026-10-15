import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from revision_analyses import (  # noqa: E402
    MARGIN_BANDS,
    STRONG_MARGIN,
    add_reversal_columns,
    load_calibrated_primary,
    reversal_strength_profile,
    write_revision_digest,
)
from authorship.grid import symbol_index  # noqa: E402
from run_stats import INTERMEDIATE, participant_mean  # noqa: E402

SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # all in the 36-symbol grid


def _row(p_top_symbol, p_top, emitted, p_emitted, target, participant):
    vector = np.full(36, 0.0)
    vector[symbol_index(p_top_symbol)] = p_top
    vector[symbol_index(emitted)] = max(vector[symbol_index(emitted)], p_emitted)
    remainder = 1.0 - vector.sum()
    others = [i for i in range(36) if vector[i] == 0]
    vector[others] = remainder / len(others)
    neural_correct = int(p_top_symbol == target)
    fused_correct = int(emitted == target)
    return {
        "p_neural_calibrated": vector,
        "neural_only_symbol": p_top_symbol,
        "emitted_symbol": emitted,
        "phantom_agreement": int(fused_correct and not neural_correct),
        "prior_capture": int(neural_correct and not fused_correct),
        "fused_correct": fused_correct,
        "neural_correct": neural_correct,
        "neural_margin_to_target": p_top - vector[symbol_index(target)],
        "ncf": 0.9,
        "study_participant_id": participant,
    }


@pytest.fixture
def frame():
    rows = [
        _row("A", 0.40, "B", 0.38, "B", "p1"),  # phantom, near tie (0.02)
        _row("A", 0.45, "B", 0.35, "B", "p1"),  # phantom, moderate (0.10), not confident
        _row("A", 0.70, "B", 0.10, "B", "p2"),  # phantom, strong (0.60), confident neural
        _row("A", 0.60, "B", 0.30, "A", "p2"),  # prior capture, strong (0.30)
        _row("A", 0.60, "A", 0.60, "A", "p3"),  # no reversal
        _row("A", 0.30, "C", 0.25, "B", "p3"),  # reversal, both wrong, moderate (0.05)
    ]
    return add_reversal_columns(pd.DataFrame(rows))


def test_margin_equals_existing_target_margin_on_phantom_rows(frame):
    phantom = frame[frame["phantom_agreement"] == 1]
    np.testing.assert_allclose(phantom["reversal_margin"], phantom["neural_margin_to_target"])


def test_every_phantom_and_capture_is_a_reversal(frame):
    assert (frame.loc[frame["phantom_agreement"] == 1, "reversal"] == 1).all()
    assert (frame.loc[frame["prior_capture"] == 1, "reversal"] == 1).all()
    assert frame.loc[frame["reversal"] == 0, "reversal_margin"].eq(0).all()


def test_bands_partition_reversals(frame):
    profile = reversal_strength_profile(frame)
    counts = [profile["bands"][name]["n_reversals"] for _, _, name in MARGIN_BANDS]
    assert sum(counts) == profile["n_reversals"] == 5
    assert profile["bands"]["near_tie"]["phantom_agreement"] == 1
    assert profile["bands"]["moderate"]["phantom_agreement"] == 1
    assert profile["bands"]["moderate"]["both_wrong"] == 1
    assert profile["bands"]["strong"]["phantom_agreement"] == 1
    assert profile["bands"]["strong"]["prior_capture"] == 1


def test_strong_and_confident_phantom_counts(frame):
    profile = reversal_strength_profile(frame)
    assert profile["phantom_strong_reversal_count"] == 1
    assert profile["phantom_confident_neural_count"] == 1
    assert profile["n_phantom_agreement"] == 3


@pytest.mark.skipif(
    not (INTERMEDIATE / "selections.parquet").exists(),
    reason="real intermediates not present in this checkout",
)
def test_real_data_consistency():
    selections = pd.read_parquet(INTERMEDIATE / "selections.parquet")
    priors = pd.read_parquet(INTERMEDIATE / "priors.parquet")
    calibrated = load_calibrated_primary(selections, priors)

    n_phantom_agreement = int((calibrated["phantom_agreement"] == 1).sum())
    assert n_phantom_agreement == 166

    digest = json.loads((INTERMEDIATE.parent / "stats_digest.json").read_text())
    target = digest["calibrated_primary_analysis"]["phantom_agreement"]["estimate"]
    assert target == pytest.approx(0.04431587669497745, abs=1e-12)
    participant_weighted = participant_mean(calibrated, "phantom_agreement")
    assert participant_weighted == pytest.approx(target, abs=1e-12)

    phantom = calibrated[calibrated["phantom_agreement"] == 1]
    np.testing.assert_array_equal(
        phantom["reversal_margin"].to_numpy(), phantom["neural_margin_to_target"].to_numpy()
    )


from revision_analyses import (  # noqa: E402
    attribution_discrimination,
    flag_policy_row,
    flag_policy_table,
    heldout_veto,
)


def test_flag_policy_row_counts(frame):
    flagged = (frame["reversal"] == 1).to_numpy()
    row = flag_policy_row(frame, flagged, "all_reversals")
    assert row["n_flagged"] == 5
    assert row["prior_captures_flagged"] == 1
    assert row["fraction_of_prior_captures_flagged"] == 1.0
    assert row["phantom_agreements_flagged"] == 3
    # veto: flagged rows emit the neural argmax; accuracy counts neural_correct there
    expected = np.where(flagged, frame["neural_correct"], frame["fused_correct"]).mean()
    assert row["veto_accuracy_selection_weighted"] == pytest.approx(expected)


def test_heldout_veto_thresholds_come_from_the_candidate_grid(frame):
    # 5 copies of the 6-row fixture, relabelled across 10 participants
    big = pd.concat([frame] * 5, ignore_index=True)
    big["study_participant_id"] = [f"p{i % 10}" for i in range(len(big))]
    result = heldout_veto(big, n_folds=5, seed=0)
    assert set(result["veto_threshold"].unique()) <= {0.0, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf}


def _reversal_row(participant, neural_correct, fused_correct, margin=0.30):
    return {
        "study_participant_id": participant,
        "reversal": 1,
        "reversal_margin": margin,
        "neural_correct": neural_correct,
        "fused_correct": fused_correct,
    }


def test_heldout_veto_threshold_ignores_own_fold_but_reacts_to_training_fold():
    # 4 participants, 3 rows each, all reversals with the same margin (0.30). Grid
    # restricted to {0.0, inf} (veto every reversal vs never veto) so the winning
    # threshold is fully controlled by which source (neural vs fused) is correct on
    # the TRAIN rows: with n_folds=2, one fold's veto_threshold is chosen entirely
    # from the OTHER fold's rows, and must not move when that fold's OWN (held-out)
    # rows change.
    participants = ["q0", "q1", "q2", "q3"]
    # "prior capture" shape: neural is right, fused is wrong, so vetoing wins on train.
    base = pd.DataFrame(
        [_reversal_row(p, neural_correct=1, fused_correct=0) for p in participants for _ in range(3)]
    )

    baseline = heldout_veto(base, grid=(0.0,), n_folds=2, seed=0)
    fold_of_participant = baseline.groupby("study_participant_id")["fold"].first().to_dict()
    target_fold = fold_of_participant["q0"]
    held_out_participants = {p for p, f in fold_of_participant.items() if f == target_fold}
    train_participants = set(participants) - held_out_participants
    assert train_participants  # sanity: the fixture really does split into two folds

    baseline_threshold = baseline.loc[baseline["fold"] == target_fold, "veto_threshold"].iloc[0]
    assert baseline_threshold == 0.0  # vetoing wins: every train row is prior-capture shaped

    # Flip the HELD-OUT participants' own outcome columns. The target fold's threshold
    # is selected without ever looking at its own rows, so it must not change.
    perturbed_heldout = base.copy()
    is_heldout = perturbed_heldout["study_participant_id"].isin(held_out_participants)
    perturbed_heldout.loc[is_heldout, ["neural_correct", "fused_correct"]] = [0, 1]
    result_heldout_perturbed = heldout_veto(perturbed_heldout, grid=(0.0,), n_folds=2, seed=0)
    threshold_after_heldout_perturbation = result_heldout_perturbed.loc[
        result_heldout_perturbed["fold"] == target_fold, "veto_threshold"
    ].iloc[0]
    assert threshold_after_heldout_perturbation == baseline_threshold

    # Flip the TRAIN participants' outcome columns the same way (now "phantom" shaped:
    # neural is wrong, fused is right, so never-veto wins). The target fold's threshold
    # is chosen from exactly these rows, so it must change.
    perturbed_train = base.copy()
    is_train = perturbed_train["study_participant_id"].isin(train_participants)
    perturbed_train.loc[is_train, ["neural_correct", "fused_correct"]] = [0, 1]
    result_train_perturbed = heldout_veto(perturbed_train, grid=(0.0,), n_folds=2, seed=0)
    threshold_after_train_perturbation = result_train_perturbed.loc[
        result_train_perturbed["fold"] == target_fold, "veto_threshold"
    ].iloc[0]
    assert threshold_after_train_perturbation == np.inf
    assert threshold_after_train_perturbation != baseline_threshold


def test_strong_band_and_flag_policy_agree_one_ulp_below_cutpoint():
    # A margin one ULP below 0.20 (float64 subtraction noise, e.g. 0.3 - 0.10 landing
    # just under 0.20) must still be classified "strong" by reversal_strength_profile
    # and flagged by the reversal_margin_ge_0.20 policy: both go through the same
    # _at_least helper, so they cannot disagree.
    margin = np.nextafter(STRONG_MARGIN, 0.0)
    row = pd.DataFrame([{
        "reversal": 1,
        "reversal_margin": margin,
        "neural_top_p": 0.9,
        "phantom_agreement": 0,
        "prior_capture": 1,
        "fused_correct": 0,
        "neural_correct": 1,
        "ncf": 0.5,
    }])

    profile = reversal_strength_profile(row)
    assert profile["bands"]["strong"]["n_reversals"] == 1

    policy_rows = {entry["policy"]: entry for entry in flag_policy_table(row)}
    assert policy_rows["reversal_margin_ge_0.20"]["n_flagged"] == 1


def test_margin_of_exactly_one_is_in_the_strong_band():
    row = pd.DataFrame([{
        "reversal": 1,
        "reversal_margin": 1.0,
        "neural_top_p": 1.0,
        "phantom_agreement": 1,
        "prior_capture": 0,
        "fused_correct": 1,
        "neural_correct": 0,
        "ncf": 0.9,
    }])

    profile = reversal_strength_profile(row)
    assert profile["bands"]["strong"]["n_reversals"] == 1
    # every band together must still partition all reversals (coverage of [0, 1])
    total = sum(profile["bands"][name]["n_reversals"] for _, _, name in MARGIN_BANDS)
    assert total == profile["n_reversals"] == 1


def test_attribution_discrimination_keys(frame):
    out = attribution_discrimination(frame)
    assert set(out) >= {"auroc_low_ncf_for_prior_capture", "auroc_low_ncf_for_fused_error",
                        "auroc_margin_for_capture_among_reversals"}


from revision_analyses import normalized_rates, outcome_block, session_fidelity  # noqa: E402


def test_phantom_cannot_exceed_neural_error_rate(frame):
    by_participant = frame.groupby("study_participant_id").agg(
        phantom=("phantom_agreement", "mean"), neural=("neural_correct", "mean"))
    assert (by_participant["phantom"] <= 1 - by_participant["neural"] + 1e-12).all()


def test_session_fidelity_uses_archive_selection(frame):
    data = frame.assign(session_id="s1", archive_selected_symbol=["A", "B", "A", "A", "A", "C"])
    fidelity = session_fidelity(data)
    p1 = fidelity[fidelity["study_participant_id"] == "p1"].iloc[0]
    assert p1["archive_agreement"] == pytest.approx(0.5)  # neural argmax A,A vs archive A,B


def test_rescue_rate_is_phantom_per_neural_error(frame):
    rates = normalized_rates(frame)
    # p1: 2 neural errors, 2 phantom -> 1.0; p2: 1 error, 1 phantom -> 1.0; p3: 1 error, 0 -> 0.0
    assert rates["rescue_rate"]["estimate"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_revision_digest_write_is_strict_json(tmp_path):
    # inf (a held-out veto "never veto" sentinel) and nan (an undefined AUROC) both
    # occur in real digest values; neither is valid strict JSON on its own.
    digest = {
        "heldout_veto": {"thresholds_by_fold": [0.3, float("inf")]},
        "attribution_discrimination": {"auroc_margin_for_capture_among_reversals": float("nan")},
    }
    serialized = write_revision_digest(digest, path=tmp_path / "digest.json")

    reloaded = json.loads(serialized)
    assert reloaded["heldout_veto"]["thresholds_by_fold"] == [0.3, "never"]
    assert reloaded["attribution_discrimination"]["auroc_margin_for_capture_among_reversals"] is None
    # No NaN/Infinity token survived: re-dumping with allow_nan=False must not raise.
    json.dumps(reloaded, allow_nan=False)
    assert (tmp_path / "digest.json").read_text() == serialized


from revision_analyses import ladder_point_summary  # noqa: E402


def test_ladder_point_summary_median_and_range():
    points = {"m1": {"ncf": 0.9, "phantom_agreement": 0.04, "prior_capture": 0.02},
              "m2": {"ncf": 0.8, "phantom_agreement": 0.06, "prior_capture": 0.03},
              "m3": {"ncf": 0.95, "phantom_agreement": 0.02, "prior_capture": 0.01}}
    summary = ladder_point_summary(points)
    assert summary["ncf"] == {"median": 0.9, "min": 0.8, "max": 0.95}
    assert summary["phantom_agreement"]["median"] == 0.04


from revision_analyses import (  # noqa: E402
    _selection_weighted_descriptives,
    n_participants_with_neural_error,
    thresholds_per_fold,
)


def test_zero_margin_reversals_are_counted(frame):
    tied = add_reversal_columns(pd.DataFrame([_row("A", 0.35, "B", 0.35, "B", "p4")]))
    profile = reversal_strength_profile(pd.concat([frame, tied], ignore_index=True))
    assert profile["n_zero_margin_reversals"] == 1
    assert profile["n_zero_margin_phantom"] == 1
    assert reversal_strength_profile(frame)["n_zero_margin_reversals"] == 0


def test_thresholds_per_fold_lists_one_threshold_per_fold_in_fold_order():
    vetoed = pd.DataFrame({"fold": [1, 0, 2, 1, 0], "veto_threshold": [np.inf, 0.3, 0.3, np.inf, 0.3]})
    assert thresholds_per_fold(vetoed) == [0.3, np.inf, 0.3]


def test_n_participants_with_neural_error(frame):
    # p1 and p2 each have a neural error; p3's only remaining row is neural-correct.
    assert n_participants_with_neural_error(frame.iloc[:5]) == 2
    assert n_participants_with_neural_error(frame) == 3


def test_selection_weighted_descriptives_count_each_selection_once(frame):
    data = frame.assign(archive_selected_symbol=["A", "B", "A", "A", "A", "C"])
    out = _selection_weighted_descriptives(data)
    assert out["neural_accuracy_selection_weighted"] == pytest.approx(2 / 6)
    assert out["archive_agreement_selection_weighted"] == pytest.approx(4 / 6)


def test_bootstrap_correction_summary_counts_participant_cluster_intervals_only():
    from revision_analyses import bootstrap_correction_summary

    def block(low, high, **extra):
        return {"estimate": 0.5, "ci_low": low, "ci_high": high, "bootstrap_replicates": 2000, **extra}

    uncorrected = {
        "a": block(0.4, 0.6),                       # widens to 0.25 (ratio 1.25)
        "b": {"c": [block(0.1, 0.2)]},              # widens to 0.15 (ratio 1.5), inside a list
        "uniform": block(0.0, 0.0),                 # zero width, fixed by construction
        "narrow": block(0.3, 0.7),                  # narrows to 0.2 (ratio 0.5)
        "prior_level": block(0.1, 0.9, n_priors=21),  # resamples priors: excluded
        "model_term": {"estimate": 1.0, "ci_low": 0.5, "ci_high": 1.5},  # no bootstrap: excluded
    }
    corrected = {
        "a": block(0.375, 0.625),
        "b": {"c": [block(0.075, 0.225)]},
        "uniform": block(0.0, 0.0),
        "narrow": block(0.4, 0.6),
        "prior_level": block(0.0, 1.0, n_priors=21),
        "model_term": {"estimate": 1.0, "ci_low": 0.4, "ci_high": 1.6},
    }
    summary = bootstrap_correction_summary(uncorrected, corrected)
    assert summary["n_intervals"] == 4
    assert summary["n_intervals_changed"] == 3
    assert summary["n_wider"] == 2
    assert summary["n_narrower"] == 1
    assert summary["n_unchanged_zero_width"] == 1
    assert summary["median_width_ratio"] == pytest.approx(1.25)


from revision_analyses import paired_decoder_frame, PAIRED_MEASURES  # noqa: E402


def _decoder_frame(ncf, phantom, capture, fused, neural, participants):
    return pd.DataFrame({
        "ncf": ncf, "phantom_agreement": phantom, "prior_capture": capture,
        "fused_correct": fused, "neural_correct": neural,
        "study_participant_id": participants,
    })


def test_paired_decoder_frame_differences_rowwise():
    base = _decoder_frame([0.9, 0.8], [1, 0], [0, 0], [1, 1], [0, 1], ["p1", "p2"])
    other = _decoder_frame([0.95, 0.8], [0, 0], [0, 1], [1, 0], [1, 1], ["p1", "p2"])
    paired = paired_decoder_frame(base, other)
    assert list(paired["diff_prior_share"].round(10)) == [-0.05, 0.0]
    assert list(paired["diff_phantom_agreement"]) == [-1, 0]
    assert list(paired["diff_prior_capture"]) == [0, 1]
    # accuracy_gain = fused - neural: base [1, 0]; other [0, -1]
    assert list(paired["diff_accuracy_gain"]) == [-1, -1]
    assert list(paired["diff_neural_correct"]) == [1, 0]
    assert set(f"diff_{m}" for m in PAIRED_MEASURES) <= set(paired.columns)


def test_paired_decoder_frame_rejects_misaligned_rows():
    base = _decoder_frame([0.9], [1], [0], [1], [0], ["p1"])
    other = _decoder_frame([0.9], [1], [0], [1], [0], ["p2"])
    with pytest.raises(ValueError):
        paired_decoder_frame(base, other)


def test_paired_decoder_frame_rejects_unequal_length():
    base = _decoder_frame([0.9, 0.8], [1, 0], [0, 0], [1, 1], [0, 1], ["p1", "p2"])
    other = _decoder_frame([0.9], [1], [0], [1], [0], ["p1"])
    with pytest.raises(ValueError):
        paired_decoder_frame(base, other)


def test_paired_decoder_frame_rejects_selection_key_mismatch_when_cluster_matches():
    # Same CLUSTER values in the same order on both sides, so a CLUSTER-only check would
    # pass, but trial_number disagrees on the second row: the two decoders are not
    # actually scoring the same selections there, so pairing them would be wrong.
    base = _decoder_frame([0.9, 0.8], [1, 0], [0, 0], [1, 1], [0, 1], ["p1", "p2"])
    other = _decoder_frame([0.9, 0.8], [1, 0], [0, 0], [1, 1], [0, 1], ["p1", "p2"])
    base["relative_path"] = ["a.edf", "b.edf"]
    base["trial_number"] = [1, 2]
    other["relative_path"] = ["a.edf", "b.edf"]
    other["trial_number"] = [1, 3]
    with pytest.raises(ValueError):
        paired_decoder_frame(base, other)
