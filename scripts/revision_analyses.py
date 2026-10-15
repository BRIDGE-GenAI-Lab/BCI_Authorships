"""Additional sensitivity analyses: reversal strength, confirmation triage, fidelity-restricted
estimates, and attribution under stronger decoders.

Reads the intermediates run_stats.py reads and writes output/revision_digest.json.
Never writes output/stats_digest.json: the primary analysis stays frozen.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from authorship.grid import symbol_index  # noqa: E402
from run_stats import (  # noqa: E402
    CLUSTER,
    INTERMEDIATE,
    OUTPUT,
    PRIMARY_PRIOR,
    calibrated_attribution_frame,
    cast_binary_columns,
    cluster_bootstrap,
    full_prior_frame,
    participant_mean,
    strip,
)

REVISION_DIGEST = OUTPUT / "revision_digest.json"
MARGIN_BANDS = ((0.0, 0.05, "near_tie"), (0.05, 0.20, "moderate"), (0.20, 1.0 + 1e-9, "strong"))
STRONG_MARGIN = 0.20
CONFIDENT_NEURAL_P = 0.50
CARRIED_COLUMNS = (
    "session_id", "archive_selected_symbol", "position_in_phrase", "train_auc",
    "relative_path", "trial_number", "study",
)


def add_reversal_columns(calibrated: pd.DataFrame) -> pd.DataFrame:
    """Flag selections where adding the prior changed the argmax, and by how much the
    calibrated neural posterior preferred its own choice. Both quantities are available
    at run time: neither uses the intended character."""
    out = calibrated.copy()
    neural = np.stack(out["p_neural_calibrated"].map(np.asarray).to_numpy())
    rows = np.arange(len(out))
    neural_index = out["neural_only_symbol"].map(symbol_index).to_numpy()
    emitted_index = out["emitted_symbol"].map(symbol_index).to_numpy()
    reversal = neural_index != emitted_index
    out["reversal"] = reversal.astype(int)
    out["reversal_margin"] = np.where(
        reversal, neural[rows, neural_index] - neural[rows, emitted_index], 0.0
    )
    out["neural_top_p"] = neural[rows, neural_index]
    return out


def _outcome_counts(block: pd.DataFrame) -> dict[str, int]:
    return {
        "n_reversals": int(len(block)),
        "phantom_agreement": int(block["phantom_agreement"].sum()),
        "prior_capture": int(block["prior_capture"].sum()),
        "both_wrong": int(((block["fused_correct"] == 0) & (block["neural_correct"] == 0)).sum()),
    }

# Margins are float64 differences of two entries of the same probability vector
# (e.g. 0.3 - 0.25 == 0.049999999999999996 in IEEE 754), so a margin that is exactly
# at a MARGIN_BANDS/STRONG_MARGIN/MARGIN_GRID cutpoint at the source precision can land
# fractionally below it. Real calibrated probabilities carry no meaningful information
# at the 1e-9 level, so every "is this margin at least this cutpoint" comparison in this
# module goes through this one helper instead of a bare >=, so the same value is always
# classified the same way regardless of which analysis is asking.
_BAND_EPS = 1e-9


def _at_least(values, cut: float):
    """Robust 'values >= cut', tolerant of one-ULP-below-cut float64 noise."""
    return np.asarray(values) >= (cut - _BAND_EPS)


def reversal_strength_profile(frame: pd.DataFrame) -> dict[str, object]:
    """How many prior-mediated reversals overturned strong neural evidence, and
    what each strength band did (helped, harmed, or swapped one error for another)."""
    reversals = frame[frame["reversal"] == 1]
    bands = {}
    last_index = len(MARGIN_BANDS) - 1
    for index, (low, high, name) in enumerate(MARGIN_BANDS):
        at_or_above_low = _at_least(reversals["reversal_margin"], low)
        if index == last_index:
            # The last band's "high" is only a padding sentinel (1.0 + 1e-9), not a
            # real next cutpoint: margins are bounded above by 1.0 (a probability
            # difference), so the last band simply has no upper edge to test against.
            in_band = reversals[at_or_above_low]
        else:
            in_band = reversals[at_or_above_low & ~_at_least(reversals["reversal_margin"], high)]
        bands[name] = {"margin_low": low, "margin_high": min(high, 1.0), **_outcome_counts(in_band)}
    phantom = frame["phantom_agreement"] == 1
    strong_phantom = phantom & _at_least(frame["reversal_margin"], STRONG_MARGIN)
    confident_phantom = phantom & (frame["neural_top_p"] >= CONFIDENT_NEURAL_P)
    confident = frame[frame["neural_top_p"] >= CONFIDENT_NEURAL_P]
    zero_margin = np.abs(reversals["reversal_margin"].to_numpy()) <= _BAND_EPS
    return {
        "n_selections": int(len(frame)),
        "n_reversals": int(len(reversals)),
        "n_phantom_agreement": int(phantom.sum()),
        "bands": bands,
        "phantom_strong_reversal_count": int(strong_phantom.sum()),
        "phantom_confident_neural_count": int(confident_phantom.sum()),
        "n_confident_neural_selections": int(len(confident)),
        "neural_accuracy_when_confident": float(confident["neural_correct"].mean())
        if len(confident) else None,
        # Reversals whose margin is zero (to the band tolerance): the calibrated neural
        # posterior's top symbol and the emitted symbol are tied.
        "n_zero_margin_reversals": int(zero_margin.sum()),
        "n_zero_margin_phantom": int((zero_margin & (reversals["phantom_agreement"] == 1)).sum()),
    }


def reversal_strength_rates(frame: pd.DataFrame) -> dict[str, object]:
    """Participant-weighted cohort rates with cluster-bootstrap CIs for the two
    run-time-definable strong-conflict subsets of phantom agreement."""
    data = frame.assign(
        strong_phantom=((frame["phantom_agreement"] == 1) & _at_least(frame["reversal_margin"], STRONG_MARGIN)).astype(int),
        confident_phantom=((frame["phantom_agreement"] == 1) & (frame["neural_top_p"] >= CONFIDENT_NEURAL_P)).astype(int),
        strong_capture=((frame["prior_capture"] == 1) & _at_least(frame["reversal_margin"], STRONG_MARGIN)).astype(int),
    )
    return {
        column: strip(cluster_bootstrap(data, lambda d, c=column: participant_mean(d, c)))
        for column in ("strong_phantom", "confident_phantom", "strong_capture")
    }


def load_calibrated_primary(
    selections: pd.DataFrame, priors: pd.DataFrame, prior_model: str = PRIMARY_PRIOR
) -> pd.DataFrame:
    frame = full_prior_frame(selections, priors, prior_model, extra_columns=CARRIED_COLUMNS)
    calibrated = cast_binary_columns(
        calibrated_attribution_frame(frame, group_column=CLUSTER, n_folds=5, seed=0)
    )
    return add_reversal_columns(calibrated)


# Run-time confirmation triage from attribution signals
MARGIN_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
NCF_GRID = (0.50, 0.70, 0.90)


def flag_policy_row(frame: pd.DataFrame, flagged: np.ndarray, label: str) -> dict[str, object]:
    flagged = np.asarray(flagged, dtype=bool)
    captures = frame["prior_capture"].to_numpy() == 1
    phantoms = frame["phantom_agreement"].to_numpy() == 1
    errors = frame["fused_correct"].to_numpy() == 0
    veto_correct = np.where(flagged, frame["neural_correct"], frame["fused_correct"])
    return {
        "policy": label,
        "n_flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "prior_captures_flagged": int((flagged & captures).sum()),
        "fraction_of_prior_captures_flagged": float((flagged & captures).sum() / max(captures.sum(), 1)),
        "phantom_agreements_flagged": int((flagged & phantoms).sum()),
        "fused_errors_flagged": int((flagged & errors).sum()),
        "fraction_of_fused_errors_flagged": float((flagged & errors).sum() / max(errors.sum(), 1)),
        "veto_accuracy_selection_weighted": float(np.mean(veto_correct)),
        "fused_accuracy_selection_weighted": float(frame["fused_correct"].mean()),
    }


def flag_policy_table(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    reversal = frame["reversal"].to_numpy() == 1
    margin = frame["reversal_margin"].to_numpy()
    for threshold in MARGIN_GRID:
        rows.append(flag_policy_row(frame, reversal & _at_least(margin, threshold),
                                    f"reversal_margin_ge_{threshold:.2f}"))
    for threshold in NCF_GRID:
        rows.append(flag_policy_row(frame, frame["ncf"].to_numpy() < threshold,
                                    f"ncf_lt_{threshold:.2f}"))
    return rows


def _participant_accuracy(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby(CLUSTER)[column].mean().mean())


def heldout_veto(frame: pd.DataFrame, grid=MARGIN_GRID, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Choose the veto margin on the other folds' participants, apply it to the held-out
    fold. np.inf means 'never veto' and is always a candidate, so the procedure can
    decline to intervene. Ties go to the larger threshold (fewer vetoes)."""
    rng = np.random.default_rng(seed)
    participants = np.array(sorted(frame[CLUSTER].unique()))
    rng.shuffle(participants)
    fold_of = {p: i % n_folds for i, p in enumerate(participants)}
    out = frame.copy()
    out["fold"] = out[CLUSTER].map(fold_of)
    candidates = sorted(set(grid) | {np.inf})
    reversal = out["reversal"].to_numpy() == 1
    margin = out["reversal_margin"].to_numpy()
    out["veto_threshold"] = np.nan
    out["veto_correct"] = out["fused_correct"].astype(float)
    for fold in range(n_folds):
        train = out["fold"].to_numpy() != fold
        best, best_accuracy = np.inf, -1.0
        for threshold in candidates:
            flagged = reversal & _at_least(margin, threshold)
            scored = out.loc[train].assign(
                candidate=np.where(flagged[train], out.loc[train, "neural_correct"], out.loc[train, "fused_correct"])
            )
            accuracy = _participant_accuracy(scored, "candidate")
            if accuracy > best_accuracy + 1e-12 or (abs(accuracy - best_accuracy) <= 1e-12 and threshold > best):
                best, best_accuracy = threshold, accuracy
        held = ~train
        flagged = reversal & _at_least(margin, best)
        out.loc[held, "veto_threshold"] = best
        out.loc[held, "veto_correct"] = np.where(flagged[held], out.loc[held, "neural_correct"], out.loc[held, "fused_correct"])
    return out


def thresholds_per_fold(vetoed: pd.DataFrame) -> list[float]:
    """The veto threshold each fold chose, in fold order (np.inf means never veto)."""
    return [float(t) for t in vetoed.groupby("fold")["veto_threshold"].first().sort_index()]


def heldout_veto_summary(frame: pd.DataFrame) -> dict[str, object]:
    vetoed = heldout_veto(frame)
    vetoed["veto_minus_fused"] = vetoed["veto_correct"] - vetoed["fused_correct"]
    return {
        "thresholds_by_fold": sorted({float(t) for t in vetoed["veto_threshold"].unique()}),
        "threshold_per_fold": thresholds_per_fold(vetoed),
        "veto_accuracy": strip(cluster_bootstrap(vetoed, lambda d: participant_mean(d, "veto_correct"))),
        "fused_accuracy": strip(cluster_bootstrap(vetoed, lambda d: participant_mean(d, "fused_correct"))),
        "difference": strip(cluster_bootstrap(vetoed, lambda d: participant_mean(d, "veto_minus_fused"))),
    }


def attribution_discrimination(frame: pd.DataFrame) -> dict[str, float]:
    low_ncf = 1.0 - frame["ncf"].to_numpy()
    reversals = frame[frame["reversal"] == 1]
    return {
        "auroc_low_ncf_for_prior_capture": float(roc_auc_score(frame["prior_capture"], low_ncf)),
        "auroc_low_ncf_for_fused_error": float(roc_auc_score(1 - frame["fused_correct"], low_ncf)),
        "auroc_margin_for_capture_among_reversals": float(
            roc_auc_score(reversals["prior_capture"], reversals["reversal_margin"])
        ) if reversals["prior_capture"].nunique() == 2 else float("nan"),
    }


# Fidelity-restricted and decoder-normalized estimates


def session_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.assign(
        archive_match=(frame["neural_only_symbol"] == frame["archive_selected_symbol"]).astype(int)
    )
    return (
        data.groupby([CLUSTER, "session_id"])
        .agg(archive_agreement=("archive_match", "mean"),
             neural_accuracy=("neural_correct", "mean"),
             n=("archive_match", "size"))
        .reset_index()
    )


def _rate_among(data: pd.DataFrame, numerator: str, condition_column: str, condition_value: int) -> float:
    subset = data[data[condition_column] == condition_value]
    if subset.empty:
        return float("nan")
    return float(subset.groupby(CLUSTER)[numerator].mean().mean())


def normalized_rates(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rescue_rate": strip(cluster_bootstrap(
            frame, lambda d: _rate_among(d, "phantom_agreement", "neural_correct", 0))),
        "capture_rate": strip(cluster_bootstrap(
            frame, lambda d: _rate_among(d, "prior_capture", "neural_correct", 1))),
    }


def n_participants_with_neural_error(frame: pd.DataFrame) -> int:
    """Participants contributing to the rescue rate: those with at least one neural error."""
    return int((frame.groupby(CLUSTER)["neural_correct"].min() == 0).sum())


def outcome_block(frame: pd.DataFrame) -> dict[str, object]:
    data = frame.assign(
        prior_share=1.0 - frame["ncf"],
        accuracy_gain=frame["fused_correct"] - frame["neural_correct"],
    )
    block = {
        name: strip(cluster_bootstrap(data, lambda d, c=name: participant_mean(d, c)))
        for name in ("ncf", "prior_share", "phantom_agreement", "prior_capture",
                     "accuracy_gain", "neural_correct", "fused_correct")
    }
    block.update(normalized_rates(data))
    block["n_selections"] = int(len(data))
    block["n_participants"] = int(data[CLUSTER].nunique())
    block["n_participants_with_neural_error"] = n_participants_with_neural_error(data)
    block["n_sessions"] = int(data.groupby([CLUSTER, "session_id"]).ngroups)
    return block


def fidelity_restricted_estimates(frame: pd.DataFrame, thresholds=(0.80, 0.90)) -> dict[str, object]:
    fidelity = session_fidelity(frame)
    out = {"session_fidelity_quartiles": fidelity["archive_agreement"].quantile([0.25, 0.5, 0.75]).tolist()}
    for threshold in thresholds:
        keep = fidelity.loc[fidelity["archive_agreement"] >= threshold, [CLUSTER, "session_id"]]
        subset = frame.merge(keep, on=[CLUSTER, "session_id"], how="inner")
        out[f"archive_agreement_ge_{threshold:.2f}"] = outcome_block(subset) if len(subset) else None
    return out


def _json_safe(value):
    """Make a digest value strict-JSON-safe, recursively: NaN becomes null and +/-inf
    becomes the string "never" (the only real source is a held-out veto threshold of
    np.inf, meaning "never veto"). Native Python floats/ints/bools are passed through
    unchanged; a stray numpy scalar (should not normally reach here, since the rest of
    this module casts explicitly) is converted to its native type first.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "never"
        return value
    return value


def write_revision_digest(digest: dict[str, object], path: Path = REVISION_DIGEST) -> str:
    """Serialize and write the digest as strict JSON (allow_nan=False), so a NaN or
    +/-Infinity that slipped past _json_safe raises loudly here instead of writing a
    non-standard JSON token that a strict downstream parser would reject silently."""
    serialized = json.dumps(_json_safe(digest), indent=2, allow_nan=False)
    path.write_text(serialized)
    return serialized


# Attribution under stronger decoders
DECODERS = {
    "original": "selections.parquet",
    "xdawn_riemann": "selections_xdawn_riemann.parquet",
    "eegnet": "selections_eegnet.parquet",
}
CLASSICAL_PRIORS = {"uniform", "ngram5", "ngram5_kn", "ngram5_wiki_kn"}
# No decoders_original.parquet exists (the original decoder predates strong_decoders.py's
# per-decoder metadata file); its calibration AUC is the StratifiedGroupKFold scheme
# authorship/decoder.py's calibration_auc always used, recorded here as a fixed label so
# every decoder's scheme sits next to its AUC in the digest the same way.
ORIGINAL_CALIBRATION_CV_SCHEME = "grouped, up to 5 folds"


def ladder_point_summary(points: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {}
    for measure in ("ncf", "phantom_agreement", "prior_capture"):
        values = np.array([p[measure] for p in points.values()], dtype=float)
        out[measure] = {"median": float(np.median(values)), "min": float(values.min()),
                        "max": float(values.max())}
    return out


def _point_estimates(calibrated: pd.DataFrame) -> dict[str, float]:
    return {m: participant_mean(calibrated, m) for m in ("ncf", "phantom_agreement", "prior_capture")}


def _decoder_cv_scheme(name: str) -> str:
    """The calibration AUC's cross-validation scheme, so a reader comparing AUCs across
    decoders sees they are not on the same split."""
    if name == "original":
        return ORIGINAL_CALIBRATION_CV_SCHEME
    schemes = pd.read_parquet(INTERMEDIATE / f"decoders_{name}.parquet", columns=["cv_scheme"])
    unique = schemes["cv_scheme"].unique()
    if len(unique) != 1:
        raise ValueError(f"decoders_{name}.parquet does not have one cv_scheme: {unique}")
    return str(unique[0])


def _selection_weighted_descriptives(frame: pd.DataFrame) -> dict[str, float]:
    """One row one vote: the weighting of Table 1's offline accuracy and archive agreement."""
    return {
        "archive_agreement_selection_weighted": float(
            (frame["neural_only_symbol"] == frame["archive_selected_symbol"]).mean()),
        "neural_accuracy_selection_weighted": float(frame["neural_correct"].mean()),
    }


def decoder_strength_comparison() -> dict[str, object]:
    """Does attribution look different under decoders stronger than the
    original session logistic model? Recomputes the reference-prior outcome block, the
    archive-agreement and calibration-AUC descriptives, the reversal-strength counts, and
    the full 21-model ladder's point estimates, once per decoder in DECODERS."""
    priors = pd.read_parquet(INTERMEDIATE / "priors.parquet")
    neural_priors = sorted(set(priors["prior_model"]) - CLASSICAL_PRIORS)
    assert len(neural_priors) == 21, neural_priors
    baseline_keys = pd.read_parquet(INTERMEDIATE / "selections.parquet",
                                    columns=["relative_path", "trial_number"])
    result: dict[str, object] = {}
    for name, filename in DECODERS.items():
        path = INTERMEDIATE / filename
        if not path.exists():
            result[name] = None
            continue
        selections = pd.read_parquet(path)
        keys = selections[["relative_path", "trial_number"]].reset_index(drop=True)
        if not keys.equals(baseline_keys.reset_index(drop=True)):
            raise ValueError(f"{filename} is not row-aligned with selections.parquet")
        reference = load_calibrated_primary(selections, priors, PRIMARY_PRIOR)
        fidelity = session_fidelity(reference)
        ladder = {model: _point_estimates(load_calibrated_primary(selections, priors, model))
                  for model in neural_priors}
        result[name] = {
            "reference_prior": outcome_block(reference),
            **_selection_weighted_descriptives(reference),
            "session_archive_agreement_median": float(fidelity["archive_agreement"].median()),
            "calibration_auc_median": float(
                selections.groupby([CLUSTER, "session_id"])["train_auc"].first().median()),
            "calibration_auc_cv_scheme": _decoder_cv_scheme(name),
            "reversal_strength": reversal_strength_profile(reference),
            "ladder_points": ladder,
            "ladder_summary": ladder_point_summary(ladder),
            "ngram5": _point_estimates(load_calibrated_primary(selections, priors, "ngram5")),
        }
    return result


# Participant-paired contrasts between the stronger decoders and the original
PAIRED_MEASURES = ("prior_share", "phantom_agreement", "prior_capture", "accuracy_gain", "neural_correct")


def _paired_values(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "prior_share": 1.0 - frame["ncf"].to_numpy(dtype=float),
        "phantom_agreement": frame["phantom_agreement"].to_numpy(dtype=float),
        "prior_capture": frame["prior_capture"].to_numpy(dtype=float),
        "accuracy_gain": (frame["fused_correct"] - frame["neural_correct"]).to_numpy(dtype=float),
        "neural_correct": frame["neural_correct"].to_numpy(dtype=float),
    })


def paired_decoder_frame(base: pd.DataFrame, other: pd.DataFrame, other_name: str = "other") -> pd.DataFrame:
    """Per-selection differences (other minus base) for two decoders scored on the same
    selections. Row order must match; resampling participants once per replicate over
    this frame gives a participant-paired bootstrap.

    Alignment is checked on CLUSTER plus, when both frames carry them (as
    load_calibrated_primary's output does, via CARRIED_COLUMNS), the selection keys
    relative_path and trial_number, mirroring decoder_strength_comparison's row-alignment
    check. CLUSTER alone would not catch two decoders whose rows were shuffled within the
    same participant's block."""
    key_columns = [CLUSTER] + [
        column for column in ("relative_path", "trial_number")
        if column in base.columns and column in other.columns
    ]
    aligned = len(base) == len(other) and all(
        np.array_equal(base[column].to_numpy(), other[column].to_numpy()) for column in key_columns
    )
    if not aligned:
        raise ValueError(f"{other_name} is not row-aligned with the original decoder frame")
    base_values, other_values = _paired_values(base), _paired_values(other)
    out = pd.DataFrame({CLUSTER: base[CLUSTER].to_numpy()})
    for measure in PAIRED_MEASURES:
        out[f"diff_{measure}"] = other_values[measure].to_numpy() - base_values[measure].to_numpy()
    return out


def paired_decoder_contrasts() -> dict[str, object]:
    priors = pd.read_parquet(INTERMEDIATE / "priors.parquet")
    frames = {
        name: load_calibrated_primary(pd.read_parquet(INTERMEDIATE / filename), priors, PRIMARY_PRIOR)
        for name, filename in DECODERS.items()
        if (INTERMEDIATE / filename).exists()
    }
    result: dict[str, object] = {}
    for name in ("xdawn_riemann", "eegnet"):
        if name not in frames:
            continue
        paired = paired_decoder_frame(frames["original"], frames[name], other_name=name)
        result[f"{name}_minus_original"] = {
            measure: strip(cluster_bootstrap(paired, lambda d, c=f"diff_{measure}": participant_mean(d, c)))
            for measure in PAIRED_MEASURES
        }
    return result


UNCORRECTED_STATS_DIGEST = PROJECT / "results" / "digests" / "stats_digest_uncorrected_bootstrap.json"
CORRECTED_STATS_DIGEST = PROJECT / "results" / "digests" / "stats_digest.json"


def _participant_cluster_widths(uncorrected, corrected):
    """Yield (uncorrected width, corrected width) for every participant-cluster bootstrap
    interval present in both digests: a dict carrying bootstrap_replicates and numeric
    ci_low/ci_high. Prior-level bootstraps (the scale-equivalence tests resample priors,
    marked by n_priors) and model-based intervals (no bootstrap_replicates) are skipped."""
    if isinstance(uncorrected, dict) and isinstance(corrected, dict):
        bounds = ("ci_low", "ci_high")
        if (
            "bootstrap_replicates" in uncorrected
            and "n_priors" not in uncorrected
            and all(isinstance(uncorrected.get(k), (int, float)) for k in bounds)
            and all(isinstance(corrected.get(k), (int, float)) for k in bounds)
        ):
            yield (uncorrected["ci_high"] - uncorrected["ci_low"], corrected["ci_high"] - corrected["ci_low"])
        for key in uncorrected:
            if key in corrected:
                yield from _participant_cluster_widths(uncorrected[key], corrected[key])
    elif isinstance(uncorrected, list) and isinstance(corrected, list):
        for left, right in zip(uncorrected, corrected):
            yield from _participant_cluster_widths(left, right)


def bootstrap_correction_summary(uncorrected: dict, corrected: dict) -> dict[str, object]:
    """Count how the participant-cluster bootstrap correction (relabelled draws) moved
    interval widths between the uncorrected and corrected stats digests. The width ratio is
    corrected over uncorrected, summarized over the intervals that changed."""
    widths = list(_participant_cluster_widths(uncorrected, corrected))
    changed = [(old, new) for old, new in widths if old != new]
    ratios = [new / old for old, new in changed if old > 0]
    return {
        "n_intervals": len(widths),
        "n_intervals_changed": len(changed),
        "n_wider": sum(new > old for old, new in changed),
        "n_narrower": sum(new < old for old, new in changed),
        "n_unchanged_zero_width": sum(old == new == 0 for old, new in widths),
        "median_width_ratio": float(np.median(ratios)) if ratios else None,
        "min_width_ratio": float(min(ratios)) if ratios else None,
        "max_width_ratio": float(max(ratios)) if ratios else None,
    }


def main() -> None:
    selections = pd.read_parquet(INTERMEDIATE / "selections.parquet")
    priors = pd.read_parquet(INTERMEDIATE / "priors.parquet")
    calibrated = load_calibrated_primary(selections, priors)
    digest: dict[str, object] = {"primary_prior": PRIMARY_PRIOR}
    digest["reversal_strength"] = {
        **reversal_strength_profile(calibrated),
        "rates": reversal_strength_rates(calibrated),
    }
    digest["flag_policies"] = flag_policy_table(calibrated)
    digest["heldout_veto"] = heldout_veto_summary(calibrated)
    digest["attribution_discrimination"] = attribution_discrimination(calibrated)
    digest["full_cohort_outcomes"] = outcome_block(calibrated)
    digest["fidelity_restricted"] = fidelity_restricted_estimates(calibrated)
    digest["decoder_strength"] = decoder_strength_comparison()
    digest["decoder_paired_contrasts"] = paired_decoder_contrasts()
    if UNCORRECTED_STATS_DIGEST.exists():
        digest["bootstrap_correction"] = bootstrap_correction_summary(
            json.loads(UNCORRECTED_STATS_DIGEST.read_text()), json.loads(CORRECTED_STATS_DIGEST.read_text())
        )
    serialized = write_revision_digest(digest)
    print(serialized[:3000])


if __name__ == "__main__":
    main()
