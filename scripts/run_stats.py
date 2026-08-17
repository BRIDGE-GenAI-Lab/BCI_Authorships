"""Statistical analysis for the authorship study.

Co-primary outcomes are estimated under the primary prior, the most capable language model
in the ladder, so that the two co-primaries are not inflated by counting each selection
once per prior. Each co-primary is tested at alpha 0.025, and the secondary family is
tested only if at least one co-primary meets its threshold.

Confidence intervals come from 2,000 deterministic participant-cluster bootstrap
replicates, matching the resampling scheme of the companion calibration study.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm, spearmanr
from statsmodels.stats.multitest import multipletests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.attribution import (  # noqa: E402
    attribution_row,
    fuse,
    log_odds_contribution,
    shapley_contribution,
)
from authorship.grid import symbol_index  # noqa: E402
from authorship.priors import family_of  # noqa: E402

OUTPUT = PROJECT / "output"
INTERMEDIATE = OUTPUT / "intermediate"

PRIMARY_BETA = 1.0
PRIMARY_PRIOR = "Qwen/Qwen2.5-32B"
# Mirrors build_attribution.py's BETA_GRID: the same five prespecified fusion exponents,
# reused here for the calibrated-basis sensitivity grid (sensitivity.fusion_exponent_calibrated)
# so the two grids sit on the same axis.
BETA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
N_BOOTSTRAP = 2000
ALPHA_COPRIMARY = 0.025
CLUSTER = "study_participant_id"
SEED = 20260726


def cluster_bootstrap(
    frame: pd.DataFrame, statistic, n_replicates: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, float]:
    """Resample participants with replacement and summarize a statistic."""
    rng = np.random.default_rng(seed)
    clusters = frame[CLUSTER].unique()
    blocks = {name: block for name, block in frame.groupby(CLUSTER)}
    estimates = np.empty(n_replicates)
    for replicate in range(n_replicates):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        resampled = pd.concat([blocks[name] for name in drawn], ignore_index=True)
        estimates[replicate] = statistic(resampled)
    point = float(statistic(frame))
    finite = estimates[np.isfinite(estimates)]
    return {
        "estimate": point,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "bootstrap_replicates": int(len(finite)),
        "_draws": finite,
    }


def participant_mean(frame: pd.DataFrame, column: str) -> float:
    """Participant-weighted mean, so that prolific participants do not dominate."""
    return float(frame.groupby(CLUSTER)[column].mean().mean())


def strip(result: dict) -> dict:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def one_sided_p(draws: np.ndarray, null_value: float, direction: str) -> float:
    """Bootstrap one-sided p value against a null value."""
    if direction == "less":
        proportion = float(np.mean(draws >= null_value))
    else:
        proportion = float(np.mean(draws <= null_value))
    return max(proportion, 1.0 / (len(draws) + 1))


def two_sided_p(draws: np.ndarray, null_value: float) -> float:
    """Two-sided bootstrap p value against a null value, for comparisons where either
    direction is a priori plausible (unlike one_sided_p's use elsewhere in this file,
    always testing "is the real effect greater than an a priori-signed null").
    """
    above = float(np.mean(draws >= null_value))
    below = float(np.mean(draws <= null_value))
    return max(min(2 * min(above, below), 1.0), 1.0 / (len(draws) + 1))


def model_frame(frame: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Drop rows the formula cannot use, keeping the cluster column aligned.

    The neural contribution fraction is undefined when neither source displaces the fused
    posterior, so those rows carry NaN. Dropping them here keeps the cluster vector the
    same length as the design matrix.
    """
    variables = [
        token
        for token in formula.replace("~", " ").replace("+", " ").split()
        if token in frame.columns
    ]
    return frame.dropna(subset=variables + [CLUSTER])


def fit_mixed_continuous(
    frame: pd.DataFrame, formula: str, *, vc_groups: dict[str, str] | None = None
) -> dict:
    """Linear mixed model with a participant random intercept, with a documented fallback.

    `vc_groups`, when given, maps a variance-component name to a categorical column (for
    example ``{"participant": CLUSTER, "word": "intended_phrase", "session": "_session_key"}``)
    and fits every named column as its own independent, fully CROSSED random-effect grouping via
    statsmodels' variance-components mechanism (`vc_formula`), instead of the single nested
    `groups=` clustering used otherwise. MixedLM's `groups=` argument supports only one nested
    hierarchy, so representing genuinely crossed structure -- the same word recurring across many
    different participants rather than being nested inside one -- requires assigning every
    observation to a single dummy top-level group and letting `vc_formula` carry all of the real
    grouping structure instead; this is the standard statsmodels workaround for fully crossed
    random effects (see the package's own crossed-variance-component test cases). The default
    optimizer left this crossed specification unconverged on this study's data; `lbfgs` reached a
    reported convergence and was corroborated by three other optimizers landing on the same
    coefficient, so it is used only for this branch, leaving the plain nested case untouched.
    """
    frame = model_frame(frame, formula)
    if vc_groups is not None:
        frame = frame.dropna(subset=list(vc_groups.values()))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if vc_groups is not None:
                vc_formula = {name: f"0 + C({column})" for name, column in vc_groups.items()}
                model = smf.mixedlm(
                    formula,
                    frame,
                    groups=np.ones(len(frame)),
                    re_formula="0",
                    vc_formula=vc_formula,
                ).fit(method="lbfgs", maxiter=500)
            else:
                model = smf.mixedlm(formula, frame, groups=frame[CLUSTER]).fit()
        if not model.converged:
            raise ValueError("mixed model did not converge")
        confidence = model.conf_int()
        term_names = [
            name
            for name in model.params.index
            if name not in {"Group Var"} and not name.endswith(" Var")
        ]
        result = {
            "engine": "mixedlm_crossed_vc" if vc_groups is not None else "mixedlm",
            "terms": {
                name: {
                    "coefficient": float(model.params[name]),
                    "ci_low": float(confidence.loc[name, 0]),
                    "ci_high": float(confidence.loc[name, 1]),
                    "p_value": float(model.pvalues[name]),
                }
                for name in term_names
            },
        }
        if vc_groups is not None:
            result["vc_groups"] = list(vc_groups)
            result["optimizer"] = "lbfgs"
            result["variance_components"] = {
                name: {
                    "variance": float(model.params[f"{name} Var"]),
                    "ci_low": float(confidence.loc[f"{name} Var", 0]),
                    "ci_high": float(confidence.loc[f"{name} Var", 1]),
                    "p_value": float(model.pvalues[f"{name} Var"]),
                }
                for name in vc_groups
            }
        return result
    except Exception as error:  # noqa: BLE001 - fallback is reported, not hidden
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols(formula, frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame[CLUSTER]}
            )
        deviation = f"mixed model unavailable ({error}); cluster-robust ordinary least squares used"
        if vc_groups is not None:
            deviation += (
                "; the word/session crossed adjustment could not be estimated, so this"
                " fallback reflects participant-only clustering, not the adjusted model"
            )
        return {
            "engine": "cluster_robust_ols",
            "deviation": deviation,
            "terms": {
                name: {
                    "coefficient": float(model.params[name]),
                    "ci_low": float(model.conf_int().loc[name, 0]),
                    "ci_high": float(model.conf_int().loc[name, 1]),
                    "p_value": float(model.pvalues[name]),
                }
                for name in model.params.index
            },
        }


BINARY_COLUMNS = (
    "phantom_agreement",
    "prior_capture",
    "neural_override",
    "fused_correct",
    "neural_correct",
    "prior_correct",
    "archive_correct",
    "phrase_is_numeric",
)


def cast_binary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Represent binary outcomes as integers.

    A boolean model outcome is expanded by the formula layer into a two-level categorical
    and the first level is taken as the response, which silently models the complement and
    inverts every odds ratio. Casting to integer removes that failure mode.
    """
    converted = frame.copy()
    for column in BINARY_COLUMNS:
        if column in converted.columns:
            converted[column] = converted[column].astype(bool).astype(int)
    return converted


def fit_binary(frame: pd.DataFrame, formula: str) -> dict:
    """Logistic model with participant-clustered generalized estimating equations."""
    frame = model_frame(frame, formula)
    outcome = formula.split("~")[0].strip()
    if frame[outcome].dtype == bool:
        raise ValueError(f"binary outcome {outcome} must be cast to integer before modelling")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.gee(
            formula,
            groups=frame[CLUSTER],
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    confidence = model.conf_int()
    return {
        "engine": "gee_exchangeable_logit",
        "terms": {
            name: {
                "log_odds": float(model.params[name]),
                "odds_ratio": float(np.exp(model.params[name])),
                "or_ci_low": float(np.exp(confidence.loc[name, 0])),
                "or_ci_high": float(np.exp(confidence.loc[name, 1])),
                "p_value": float(model.pvalues[name]),
            }
            for name in model.params.index
        },
    }


def fit_binary_crossed(frame: pd.DataFrame, formula: str, vc_groups: dict[str, str]) -> dict:
    """Crossed-random-effects logistic model: the binary-outcome analogue of
    fit_mixed_continuous's vc_formula branch.

    fit_binary's generalized estimating equations admit only one clustering variable, so
    word identity and recording session cannot be adjusted for alongside participant
    there. BinomialBayesMixedGLM accepts the same `vc_formula` idiom with a Binomial
    family and so can, but it estimates a variational (mean-field Gaussian) approximation
    to the posterior rather than solving estimating equations. Two consequences must
    travel with any citation of this model: the interval reported here is a 95% credible
    interval from that approximate posterior, not a sandwich-based confidence interval,
    and no p value is defined, which is why this model is deliberately absent from the
    Benjamini-Hochberg family in main().

    statsmodels jitters fit_vb's starting posterior standard deviations off the GLOBAL
    numpy random state (`-0.5 + 0.1 * np.random.normal(size=n)`), which would make an
    otherwise seed-deterministic pipeline write a different coefficient into the tracked
    stats digest on every run. Passing the unjittered centre of that draw as an explicit
    start removes the dependence on global random state and reaches the same optimum.
    """
    frame = model_frame(frame, formula).dropna(subset=list(vc_groups.values()))
    outcome = formula.split("~")[0].strip()
    if frame[outcome].dtype == bool:
        raise ValueError(f"binary outcome {outcome} must be cast to integer before modelling")
    vc_formula = {name: f"0 + C({column})" for name, column in vc_groups.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.BinomialBayesMixedGLM.from_formula(formula, vc_formula, frame, vcp_p=1.0)
        fitted = model.fit_vb(
            sd=np.full(model.k_fep + model.k_vcp + model.k_vc, np.exp(-0.5))
        )
    if not fitted.optim_retvals["success"]:
        raise ValueError(
            f"variational fit for {formula!r} did not converge: {fitted.optim_retvals['message']}"
        )
    return {
        "engine": "binomial_bayes_mixed_glm_crossed_vc",
        "interval": "95% credible interval from the variational posterior; no p value is defined",
        "vc_groups": list(vc_groups),
        "terms": {
            name: {
                "log_odds": float(mean),
                "odds_ratio": float(np.exp(mean)),
                "or_ci_low": float(np.exp(mean - 1.96 * sd)),
                "or_ci_high": float(np.exp(mean + 1.96 * sd)),
            }
            for name, mean, sd in zip(model.exog_names, fitted.fe_mean, fitted.fe_sd)
        },
        "random_effect_sd": {
            name: {
                "sd": float(np.exp(mean)),
                "sd_ci_low": float(np.exp(mean - 1.96 * sd)),
                "sd_ci_high": float(np.exp(mean + 1.96 * sd)),
            }
            for name, mean, sd in zip(model.vcp_names, fitted.vcp_mean, fitted.vcp_sd)
        },
    }


def predicted_contrast(model, term: str, low: float, high: float) -> dict:
    """Difference in fitted outcome between two values of `term`, the interpretable effect
    size for a fractional-logit fit.

    The native coefficient of these fits is a log-odds, and an odds ratio is an awkward way
    to describe an effect on a genuinely continuous fractional quantity like NCF. Worse, the
    Results used to quote the LINEAR mixed model's per-unit slope for a relationship whose
    curve in Figure 2B is this bounded fit -- two different fitted objects, so a reader
    comparing the plotted S-shape against a "per unit" slope had no way to reconcile them.
    This evaluates the plotted model itself at two moderator values and reports the change in
    its fitted mean, which is on the outcome's own scale and reads directly off the curve.

    The interval is a delta-method interval on the same participant-clustered sandwich
    covariance the stored coefficient intervals come from, rather than a separate bootstrap:
    for the logit link mu'(eta) = mu (1 - mu), so the gradient of (mu_high - mu_low) with
    respect to the coefficients is w_high * x_high - w_low * x_low.
    """
    names = ["Intercept", term]
    params = np.array([float(model.params[name]) for name in names])
    covariance = np.asarray(model.cov_params().loc[names, names], dtype=float)
    design = np.array([[1.0, float(low)], [1.0, float(high)]])
    predicted = 1.0 / (1.0 + np.exp(-(design @ params)))
    weights = predicted * (1.0 - predicted)
    gradient = weights[1] * design[1] - weights[0] * design[0]
    standard_error = float(np.sqrt(gradient @ covariance @ gradient))
    difference = float(predicted[1] - predicted[0])
    return {
        "term": term,
        "low_value": float(low),
        "high_value": float(high),
        "predicted_low": float(predicted[0]),
        "predicted_high": float(predicted[1]),
        "difference": difference,
        "ci_low": difference - 1.96 * standard_error,
        "ci_high": difference + 1.96 * standard_error,
        "standard_error": standard_error,
        "p_value": float(2 * norm.sf(abs(difference / standard_error))),
        "method": "delta method on the participant-clustered GEE sandwich covariance",
    }


def fit_fractional_logit(
    frame: pd.DataFrame, formula: str, contrast: tuple[str, float, float] | None = None
) -> dict:
    """Quasi-binomial GEE for a fractional [0,1] outcome (e.g. NCF) -- same family and
    participant-clustered exchangeable covariance fit_binary already uses for a true
    binary outcome, valid here because Binomial-family quasi-likelihood only requires y
    in [0,1], not y in {0,1}, and its logistic mean function cannot predict outside that
    range, unlike the existing linear-mixed-model treatment of the same outcome.

    `contrast`, when given as (term, low, high), adds the `predicted_contrast` summary this
    model is reported by; see predicted_contrast for why the log-odds coefficient alone is
    not a reportable effect size here.
    """
    frame = model_frame(frame, formula)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.gee(
            formula,
            groups=frame[CLUSTER],
            data=frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
    confidence = model.conf_int()
    fit = {
        "engine": "gee_exchangeable_fractional_logit",
        "terms": {
            name: {
                "log_odds": float(model.params[name]),
                "odds_ratio": float(np.exp(model.params[name])),
                "or_ci_low": float(np.exp(confidence.loc[name, 0])),
                "or_ci_high": float(np.exp(confidence.loc[name, 1])),
                "p_value": float(model.pvalues[name]),
            }
            for name in model.params.index
        },
    }
    if contrast is not None:
        fit["predicted_contrast"] = predicted_contrast(model, *contrast)
    return fit


# Shared so the raw and calibrated phantom-agreement fits cannot drift into different
# crossed structures, which would make the two bases silently incomparable.
PHANTOM_VC_GROUPS = {
    "participant": CLUSTER,
    "word": "intended_phrase",
    "session": "_session_key",
}


def six_char_words_only_position_effect(frame: pd.DataFrame) -> dict:
    """The eTable 2 position-in-word effect restricted to 6-character words only.

    With variable-length words, "later position" is confounded with "closer to the end
    of a shorter word", so a position gradient could partly reflect word length rather
    than position itself. Restricting to phrase_length == 6, the single length that
    reaches every one of the six positions (the corpus maximum), holds length fixed
    while position still varies from 1 to 6. Reuses the identical fitting call
    ("ncf ~ position_in_phrase" via fit_mixed_continuous) already used for the
    unrestricted eTable 2 model rather than a separately shaped fit.
    """
    subset = frame[frame["phrase_length"] == 6]
    fit = fit_mixed_continuous(subset, "ncf ~ position_in_phrase")
    return {"n_selections": int(len(subset)), **fit}


def _composite_session_key(frame: pd.DataFrame) -> pd.Series:
    """A session identity unique across the whole cohort, not just within one participant.

    session_id alone is not a valid grouping key for a crossed session random effect: its
    labels are recycled per participant (every participant's own first recorded session is
    "SE001", the next "SE002", and so on -- only 9 distinct raw values across the entire
    archive), so two different participants' first sessions would otherwise be silently
    pooled as if they were the same session. Combining it with the participant identifier
    recovers one distinct key per real recording session, matching the 115 sessions
    reported in eTable 1. Factored out from word_and_session_adjusted_position_effect so
    this specific fix is directly unit-testable rather than only observable indirectly
    through a downstream model fit.
    """
    return frame[CLUSTER].astype(str) + "::" + frame["session_id"].astype(str)


def word_and_session_adjusted_position_effect(frame: pd.DataFrame) -> dict:
    """The eTable 2 position-in-word effect adjusted for word identity and session.

    The eTable 2 model carries only a participant random intercept, so it cannot rule out
    two confounds: the same word recurring across the corpus (word identity), and
    clustering within a recording session. Both are added here as CROSSED random-effect
    structure, alongside participant, via fit_mixed_continuous's `vc_groups` mechanism, with
    session identified by _composite_session_key rather than the raw, per-participant-recycled
    session_id column.
    """
    keyed = frame.copy()
    keyed["_session_key"] = _composite_session_key(keyed)
    return fit_mixed_continuous(
        keyed,
        "ncf ~ position_in_phrase",
        vc_groups={
            "participant": CLUSTER,
            "word": "intended_phrase",
            "session": "_session_key",
        },
    )


def _ladder_points(ladder: pd.DataFrame) -> pd.DataFrame:
    """One row per prior_model with its parameter count and participant-weighted
    outcome means, restricted to priors with parameters > 0 (excludes uniform/ngram,
    which have no capability axis to correlate against).
    """
    points = []
    for name, group in ladder.groupby("prior_model"):
        parameters = int(group["prior_parameters"].iloc[0])
        if parameters <= 0:
            continue
        points.append(
            {
                "prior_model": name,
                "family": family_of(name),
                "log_parameters": float(np.log10(parameters)),
                "ncf": participant_mean(group, "ncf"),
                "phantom_agreement": participant_mean(group, "phantom_agreement"),
                "prior_capture": participant_mean(group, "prior_capture"),
                "accuracy_gained": 100
                * (
                    participant_mean(group, "fused_correct")
                    - participant_mean(group, "neural_correct")
                ),
            }
        )
    return pd.DataFrame(points)


def cross_family_scale_check(ladder: pd.DataFrame) -> dict:
    """Does capability predict attribution once every architecture family is pooled,
    and does any single family behave differently on its own? Extends the original
    Qwen-only scale check (which was fooled by a ladder truncated at 3B) across
    families, and repeats the same truncation check on the broader ladder.
    """
    points = _ladder_points(ladder)

    def spearman(column: str, subset: pd.DataFrame) -> dict:
        if len(subset) < 3:
            return {"n": int(len(subset)), "rho": None, "p_value": None}
        statistic = spearmanr(subset["log_parameters"], subset[column])
        return {
            "n": int(len(subset)),
            "rho": float(statistic.statistic),
            "p_value": float(statistic.pvalue),
        }

    pooled = {
        "n_priors": int(points["prior_model"].nunique()),
        "spearman_rho_ncf_vs_log_params": spearman("ncf", points),
        "spearman_rho_phantom_vs_log_params": spearman("phantom_agreement", points),
        "spearman_rho_prior_capture_vs_log_params": spearman("prior_capture", points),
        "spearman_rho_accuracy_gained_vs_log_params": spearman("accuracy_gained", points),
    }

    by_family = {
        family: {
            "n_priors": int(len(group)),
            "ncf_mean": float(group["ncf"].mean()),
            "ncf_range": [float(group["ncf"].min()), float(group["ncf"].max())],
            "phantom_agreement_mean": float(group["phantom_agreement"].mean()),
            "phantom_agreement_range": [
                float(group["phantom_agreement"].min()),
                float(group["phantom_agreement"].max()),
            ],
        }
        for family, group in points.groupby("family")
        if len(group) >= 2
    }

    truncated = points[points["log_parameters"] < 10.0]  # 10^10 = 10B parameters
    truncated_below_10b = {
        "n_priors": int(len(truncated)),
        "spearman_rho_ncf_vs_log_params": spearman("ncf", truncated),
        "spearman_rho_phantom_vs_log_params": spearman("phantom_agreement", truncated),
        "spearman_rho_prior_capture_vs_log_params": spearman("prior_capture", truncated),
        "spearman_rho_accuracy_gained_vs_log_params": spearman("accuracy_gained", truncated),
    }

    return {"pooled": pooled, "by_family": by_family, "truncated_below_10b": truncated_below_10b}


def _between_within_variance_ratio(labels: np.ndarray, values: np.ndarray) -> float:
    """One-way-ANOVA-style F statistic for how much of the variance in `values`
    falls between the groups named in `labels`, versus within them.
    """
    grand_mean = values.mean()
    unique_labels = np.unique(labels)
    ss_between = 0.0
    ss_within = 0.0
    for label in unique_labels:
        group_values = values[labels == label]
        group_mean = group_values.mean()
        ss_between += len(group_values) * (group_mean - grand_mean) ** 2
        ss_within += float(np.sum((group_values - group_mean) ** 2))
    df_between = len(unique_labels) - 1
    df_within = len(values) - len(unique_labels)
    if df_between <= 0 or df_within <= 0 or ss_within == 0:
        return float("inf")
    return (ss_between / df_between) / (ss_within / df_within)


def architecture_family_permutation_test(
    points: pd.DataFrame, metric: str, *, seed: int, n_permutations: int = 10000
) -> dict:
    """Permutation test for whether architecture family explains variance in `metric`
    beyond chance, across the priors in `points` (a `_ladder_points`-style table).

    Family labels are shuffled across priors 10,000 times (fixed seed) and the
    observed between/within variance ratio is compared against that permutation
    null. The statistic is non-negative and larger values are always more extreme
    in favor of a family effect, so the one-sided upper-tail count already gives
    the two-sided permutation P value.
    """
    families = points["family"].to_numpy()
    values = points[metric].to_numpy(dtype=float)
    observed = _between_within_variance_ratio(families, values)

    rng = np.random.default_rng(seed)
    permuted_stats = np.empty(n_permutations)
    for replicate in range(n_permutations):
        shuffled_labels = rng.permutation(families)
        permuted_stats[replicate] = _between_within_variance_ratio(shuffled_labels, values)

    p_value = float((np.sum(permuted_stats >= observed) + 1) / (n_permutations + 1))

    return {
        "metric": metric,
        "n_priors": int(len(points)),
        "n_families": int(points["family"].nunique()),
        "observed_f_statistic": float(observed),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "p_value": p_value,
    }


def scale_equivalence_test(
    points: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    margin: float = 0.3,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Two-one-sided-tests (TOST) equivalence test for the Spearman rho between
    log parameter count and `metric`, against a predefined margin of +/- `margin`.

    `cluster_bootstrap` resamples participants and does not apply to a per-prior
    correlation, so this resamples the priors in `points` directly (with
    replacement), recomputing rho each time. The reported 90% CI is the two-sided
    interval appropriate for a two-one-sided-tests equivalence conclusion at the
    conventional 0.05 level: equivalence is concluded when that interval falls
    entirely within +/- `margin`.
    """
    log_parameters = points["log_parameters"].to_numpy(dtype=float)
    values = points[metric].to_numpy(dtype=float)
    n = len(points)
    observed = float(spearmanr(log_parameters, values).statistic)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for replicate in range(n_bootstrap):
        resample = rng.integers(0, n, size=n)
        draws[replicate] = spearmanr(log_parameters[resample], values[resample]).statistic
    finite = draws[np.isfinite(draws)]

    ci_low = float(np.percentile(finite, 5.0))
    ci_high = float(np.percentile(finite, 95.0))

    return {
        "metric": metric,
        "n_priors": int(n),
        "observed_rho": observed,
        "margin": margin,
        "ci_level": 0.90,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_replicates": int(len(finite)),
        "seed": int(seed),
        "equivalent": bool(ci_low > -margin and ci_high < margin),
    }


def phantom_agreement_margin_distribution(primary: pd.DataFrame) -> dict[str, object]:
    """Among phantom-agreement selections, how much neural support did the intended
    character actually have? Answers whether a "phantom agreement" reflects near-zero
    neural evidence or a close call the prior narrowly tipped.
    """
    cases = primary[primary["phantom_agreement"].astype(bool)]
    n = len(cases)
    if n == 0:
        return {"n_phantom_agreement_cases": 0}
    p_target = cases["neural_p_target"]
    ranks = cases["neural_rank_of_target"].clip(upper=3)
    rank_counts = ranks.value_counts().to_dict()
    rank_distribution = {
        str(rank): int(rank_counts.get(rank, 0)) for rank in (1, 2)
    }
    rank_distribution["3_or_worse"] = int(rank_counts.get(3, 0))
    quartiles = p_target.quantile([0.25, 0.5, 0.75])
    return {
        "n_phantom_agreement_cases": int(n),
        "neural_p_target_median": float(quartiles.loc[0.5]),
        "neural_p_target_iqr": [float(quartiles.loc[0.25]), float(quartiles.loc[0.75])],
        "neural_p_target_below_0.10_fraction": float((p_target < 0.10).mean()),
        "neural_p_target_between_0.10_and_0.40_fraction": float(
            ((p_target >= 0.10) & (p_target < 0.40)).mean()
        ),
        "neural_p_target_above_0.40_fraction": float((p_target >= 0.40).mean()),
        "neural_rank_of_target_distribution": rank_distribution,
        "neural_margin_to_target_median": float(cases["neural_margin_to_target"].median()),
    }


def primary_prior_frame(
    selections: pd.DataFrame, priors: pd.DataFrame, prior_model: str
) -> pd.DataFrame:
    """Join raw p_neural/p_lm vectors for one prior, following build_attribution.py's join.

    Reuses the same (prior_model, context_prefix) lookup that produces attribution.parquet,
    so the selection set matches the primary analysis exactly. Also carries CLUSTER
    (study_participant_id) so callers needing participant-clustered bootstrap or grouped
    calibration folds (calibrated_primary_analysis) do not need a second join back to
    selections.
    """
    selections = selections.copy()
    selections["target_index"] = selections["target_symbol"].map(symbol_index)
    prior_lookup = {
        (row.prior_model, row.context_prefix): np.asarray(row.p_lm)
        for row in priors.itertuples(index=False)
        if row.prior_model == prior_model
    }
    return pd.DataFrame(
        {
            "p_neural": selections["p_neural"].map(np.asarray),
            "p_lm": [
                prior_lookup[(prior_model, prefix)] for prefix in selections["context_prefix"]
            ],
            "target_index": selections["target_index"],
            CLUSTER: selections[CLUSTER],
        }
    )


def full_prior_frame(
    selections: pd.DataFrame,
    priors: pd.DataFrame,
    prior_model: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Like primary_prior_frame, but also carries named covariate columns straight from
    `selections` in its own row order. primary_prior_frame never reorders or filters
    selections's rows (it copies selections and derives target_index in place), so
    columns pulled from the original `selections` object align 1:1 by position with its
    output -- no join/merge key is needed or safer than direct assignment here.
    """
    frame = primary_prior_frame(selections, priors, prior_model)
    for column in extra_columns:
        frame[column] = selections[column].to_numpy()
    return frame


def calibrated_attribution_frame(
    frame: pd.DataFrame, group_column: str = CLUSTER, n_folds: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Generalizes calibrated_primary_analysis: returns the full per-row calibrated
    frame (ncf, phantom_agreement, prior_capture, fused_correct, neural_correct columns
    added, plus whatever covariate columns the caller's `frame` already carried) instead
    of only a bootstrapped aggregate, so stratified/grouped calibrated analyses (position,
    AUC, condition, LOSO, the model ladder) can filter/groupby on it directly the same way
    they already do on the raw `primary` frame.
    """
    from authorship.calibration import calibrated_fusion_frame

    calibrated = calibrated_fusion_frame(frame, group_column=group_column, n_folds=n_folds, seed=seed)
    ncf_values, phantom_values, prior_capture_values = [], [], []
    fused_correct_values, neural_correct_values = [], []
    neural_override_values = []
    # Task 24: Figure 4's phantom-agreement cohort needs these five per-selection measures
    # (previously read straight off attribution.parquet) on the calibrated basis too, so they
    # are kept the same additive way as the five columns above.
    neural_p_target_values, neural_rank_of_target_values = [], []
    neural_margin_to_target_values = []
    neural_only_symbol_values, prior_only_symbol_values, emitted_symbol_values = [], [], []
    for row in calibrated.itertuples(index=False):
        neural = np.asarray(row.p_neural_calibrated)
        prior = np.asarray(row.p_lm_calibrated)
        fused = fuse(neural, prior)
        result = attribution_row(neural, prior, fused, int(row.target_index))
        ncf_values.append(result["ncf"])
        phantom_values.append(result["phantom_agreement"])
        prior_capture_values.append(result["prior_capture"])
        fused_correct_values.append(result["fused_correct"])
        neural_correct_values.append(result["neural_correct"])
        neural_override_values.append(result["neural_override"])
        neural_p_target_values.append(result["neural_p_target"])
        neural_rank_of_target_values.append(result["neural_rank_of_target"])
        neural_margin_to_target_values.append(result["neural_margin_to_target"])
        neural_only_symbol_values.append(result["neural_only_symbol"])
        prior_only_symbol_values.append(result["prior_only_symbol"])
        emitted_symbol_values.append(result["emitted_symbol"])
    calibrated["ncf"] = ncf_values
    calibrated["phantom_agreement"] = phantom_values
    calibrated["prior_capture"] = prior_capture_values
    calibrated["fused_correct"] = fused_correct_values
    calibrated["neural_correct"] = neural_correct_values
    calibrated["neural_override"] = neural_override_values
    calibrated["neural_p_target"] = neural_p_target_values
    calibrated["neural_rank_of_target"] = neural_rank_of_target_values
    calibrated["neural_margin_to_target"] = neural_margin_to_target_values
    calibrated["neural_only_symbol"] = neural_only_symbol_values
    calibrated["prior_only_symbol"] = prior_only_symbol_values
    calibrated["emitted_symbol"] = emitted_symbol_values
    return calibrated


def calibrated_primary_analysis(
    frame: pd.DataFrame, n_folds: int = 5, seed: int = 0
) -> dict[str, object]:
    """The new principal co-primary analysis: fuse each source AFTER independently
    fitting it a held-out (grouped-by-participant) calibration temperature, at
    beta=1. Compare against sensitivity.uncalibrated_beta_1 (the previous round's
    principal analysis, now demoted to a labeled sensitivity comparison) to show how
    much of the headline number depended on the uncalibrated choice.

    `frame` must be primary_prior_frame()'s output: it needs p_neural, p_lm,
    target_index, and CLUSTER all present. Folds are grouped by CLUSTER (participant),
    not by session, so one participant's multiple sessions never split across a
    train/held-out boundary within the same fold.
    """
    calibrated = calibrated_attribution_frame(frame, group_column=CLUSTER, n_folds=n_folds, seed=seed)

    ncf_result = cluster_bootstrap(calibrated, lambda data: participant_mean(data, "ncf"))
    phantom_result = cluster_bootstrap(
        calibrated, lambda data: participant_mean(data, "phantom_agreement")
    )
    fold_temperatures = calibrated.attrs.get("fold_temperatures", [])
    return {
        "n": int(len(calibrated)),
        "ncf": strip(ncf_result),
        "phantom_agreement": strip(phantom_result),
        "fold_temperatures": fold_temperatures,
        "mean_t_neural": float(np.mean([f["t_neural"] for f in fold_temperatures]))
        if fold_temperatures
        else None,
        "mean_t_lm": float(np.mean([f["t_lm"] for f in fold_temperatures]))
        if fold_temperatures
        else None,
    }


def _temperature_scale(distribution: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.power(distribution, 1.0 / temperature)
    return scaled / scaled.sum()


def temperature_scaled_attribution(frame: pd.DataFrame, temperature: float) -> dict[str, float]:
    """Recompute NCF and phantom agreement with both sources temperature-scaled before fusion."""
    ncf_values, phantom_values = [], []
    for row in frame.itertuples(index=False):
        neural = _temperature_scale(np.asarray(row.p_neural), temperature)
        prior = _temperature_scale(np.asarray(row.p_lm), temperature)
        fused = fuse(neural, prior)
        result = attribution_row(neural, prior, fused, int(row.target_index))
        ncf_values.append(result["ncf"])
        phantom_values.append(result["phantom_agreement"])
    return {
        "ncf_mean": float(np.nanmean(ncf_values)),
        "phantom_agreement_mean": float(np.mean(phantom_values)),
    }


def calibrated_fusion_exponent_points(
    calibrated: pd.DataFrame, beta_grid: tuple[float, ...] = BETA_GRID
) -> dict[str, dict[str, float]]:
    """Calibrated-basis counterpart of sensitivity["fusion_exponent"]: re-fuses
    calibrated_primary's held-out temperature-calibrated posteriors (p_neural_calibrated,
    p_lm_calibrated) at each exponent in beta_grid, instead of refitting calibration
    temperatures per exponent. Beta only enters at the fusion step (calibrated_attribution_
    frame's `fuse(neural, prior)` call, at its default beta=1), so the same per-source
    calibrated posteriors are reused across the whole grid rather than refit five times.
    """
    points: dict[str, dict[str, float]] = {}
    for beta in beta_grid:
        ncf_values, phantom_values, prior_capture_values = [], [], []
        for row in calibrated.itertuples(index=False):
            neural = np.asarray(row.p_neural_calibrated)
            prior = np.asarray(row.p_lm_calibrated)
            fused = fuse(neural, prior, beta=beta)
            result = attribution_row(neural, prior, fused, int(row.target_index))
            ncf_values.append(result["ncf"])
            phantom_values.append(result["phantom_agreement"])
            prior_capture_values.append(result["prior_capture"])
        group = pd.DataFrame({
            CLUSTER: calibrated[CLUSTER].to_numpy(),
            "ncf": ncf_values,
            "phantom_agreement": phantom_values,
            "prior_capture": prior_capture_values,
        })
        points[str(beta)] = {
            "ncf": float(participant_mean(group, "ncf")),
            "phantom_agreement": float(participant_mean(group, "phantom_agreement")),
            "prior_capture": float(participant_mean(group, "prior_capture")),
        }
    return points


def attribution_method_comparison(frame: pd.DataFrame) -> dict[str, float]:
    """Compare the KL-based NCF against two non-KL attribution measures, at the primary
    prior and beta, to test whether the near-uniform prior share is a property of the
    specific divergence measure or replicates under alternative formalisms.
    """
    ncf_values, log_odds_values, shapley_share_values = [], [], []
    for row in frame.itertuples(index=False):
        neural = np.asarray(row.p_neural)
        prior = np.asarray(row.p_lm)
        target_index = int(row.target_index)
        fused = fuse(neural, prior)
        ncf_values.append(attribution_row(neural, prior, fused, target_index)["ncf"])
        log_odds_values.append(log_odds_contribution(neural, prior, fused, target_index))
        neural_shapley, prior_shapley = shapley_contribution(
            neural, prior, fused, target_index
        )
        total_shapley = neural_shapley + prior_shapley
        shapley_share_values.append(
            prior_shapley / total_shapley if abs(total_shapley) > 1e-12 else np.nan
        )
    return {
        "n": int(len(ncf_values)),
        "ncf_mean": float(np.nanmean(ncf_values)),
        "ncf_median": float(np.nanmedian(ncf_values)),
        "log_odds_contribution_mean": float(np.nanmean(log_odds_values)),
        "log_odds_contribution_median": float(np.nanmedian(log_odds_values)),
        "shapley_share_mean": float(np.nanmean(shapley_share_values)),
        "shapley_share_median": float(np.nanmedian(shapley_share_values)),
    }


def _ece(confidences: pd.Series, correct: pd.Series, n_bins: int = 10) -> float:
    """Expected calibration error: bin observations into n_bins equal-width bins on
    confidence in [0, 1], and accumulate the selection-count-weighted absolute gap
    between each bin's mean confidence and its mean empirical accuracy. Shared by
    every calibration measure below -- what differs between them is what is passed
    in as "confidence" and what correctness label it is checked against, not how
    the binning itself is done.
    """
    confidence_values = np.asarray(confidences, dtype=float)
    correct_values = np.asarray(correct, dtype=bool)
    n = len(confidence_values)
    bin_index = np.minimum((confidence_values * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_number in range(n_bins):
        in_bin = bin_index == bin_number
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence_values[in_bin].mean())
        bin_accuracy = float(correct_values[in_bin].mean())
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def source_calibration_diagnostics(
    distributions: np.ndarray, target_indices: np.ndarray, n_bins: int = 10
) -> dict[str, object]:
    """Standard multiclass calibration diagnostics for any single source's own
    probability distributions: NLL, Brier (full 36-way, 0-2 scale), and top-label ECE
    (confidence = max probability, correctness = argmax == target -- the standard
    calibration-literature definition, not the P(true class)-binned nonstandard
    variant neural_posterior_calibration reported alongside it).
    """
    from authorship.calibration import mean_negative_log_likelihood

    n = len(target_indices)
    one_hot = np.zeros_like(distributions)
    one_hot[np.arange(n), target_indices] = 1.0
    brier = float(((distributions - one_hot) ** 2).sum(axis=1).mean())
    nll = mean_negative_log_likelihood(distributions, target_indices)
    confidence = distributions.max(axis=1)
    correct = distributions.argmax(axis=1) == target_indices
    ece = _ece(pd.Series(confidence), pd.Series(correct), n_bins=n_bins)
    return {"n": n, "nll": nll, "brier": brier, "ece": ece}


def _neural_argmax_correct(frame: pd.DataFrame) -> np.ndarray:
    """Whether the neural posterior's own top choice (argmax over all 36 symbols)
    matches the intended target, for each row. Shared by both calibration measures
    below: it is exactly the correctness label neural_correct in attribution.parquet
    encodes, recomputed here from the raw p_neural/target_index frame used by
    primary_prior_frame, which does not carry that column.
    """
    neural_matrix = np.stack(frame["p_neural"].map(np.asarray).to_numpy())
    target_index = frame["target_index"].to_numpy(dtype=int)
    return neural_matrix.argmax(axis=1) == target_index


def neural_posterior_calibration(primary: pd.DataFrame) -> dict[str, float]:
    """Is the neural posterior itself calibrated, independent of the fusion rule?

    Brier score is the mean squared error between the full neural posterior and a
    one-hot target vector, over the primary prior's selections. ECE here bins
    selections into 10 equal-width bins on neural_p_target -- the probability the
    neural posterior places on the *true* label, standing in for "confidence" -- and
    compares each bin's mean confidence against neural_correct, whether the neural
    posterior's own arg max equals the intended symbol. Binning on P(true label)
    rather than on the classifier's own top probability is a nonstandard,
    fusion-motivated choice: it is exactly the quantity multiplied into the fused
    posterior, so it is the calibration diagnostic the Bayesian fusion rule directly
    depends on, but it answers a different question from the standard top-label ECE
    (see top_label_calibration below), which bins on the classifier's own max
    probability instead. Both are reported, honestly labeled, rather than one in
    place of the other.
    """
    neural_matrix = np.stack(primary["p_neural"].map(np.asarray).to_numpy())
    target_index = primary["target_index"].to_numpy(dtype=int)
    n = len(target_index)

    one_hot = np.zeros_like(neural_matrix)
    one_hot[np.arange(n), target_index] = 1.0
    brier_score = float(((neural_matrix - one_hot) ** 2).sum(axis=1).mean())

    neural_p_target = neural_matrix[np.arange(n), target_index]
    neural_correct = _neural_argmax_correct(primary)

    n_bins = 10
    ece = _ece(pd.Series(neural_p_target), pd.Series(neural_correct), n_bins=n_bins)

    return {"n": n, "brier_score": brier_score, "ece": ece, "n_bins": n_bins}


def top_label_calibration(primary: pd.DataFrame) -> dict[str, object]:
    """Standard top-label ECE: confidence is the neural posterior's own top
    probability (its max over all 36 symbols), correctness is whether that top
    choice matches the intended symbol. This is the standard multiclass
    calibration question ("when the classifier is N% confident in its own top
    choice, is it right N% of the time?"), distinct from neural_posterior_calibration
    above, which bins on P(true label) instead -- a different probabilistic event
    that does not answer the same question.
    """
    p_max = primary["p_neural"].apply(lambda vector: float(np.max(vector)))
    correct = primary["neural_correct"].astype(bool)
    n = len(primary)
    ece = _ece(p_max, correct)
    brier = float(np.mean((p_max.to_numpy() - correct.to_numpy().astype(float)) ** 2))
    return {
        "n": n,
        "ece": ece,
        "brier": brier,
        "confidence_definition": "max(p_neural)",
        "correctness_definition": "neural_argmax == target",
    }


def permuted_context_frame(
    permuted_selections: pd.DataFrame, priors: pd.DataFrame, prior_model: str
) -> pd.DataFrame:
    """Join each selection's real p_neural with the WRONG prior's p_lm, following the same
    (prior_model, context_prefix) lookup as primary_prior_frame and build_attribution.py, but
    keyed on permuted_context_prefix instead of the selection's own true context_prefix.

    Carries `selection_row_id` straight through when the input has it (the combined
    stratified-permutation parquet always does; see stratified_permutation_frames) so
    calibrated callers can join each row back to its real-data-fit fold temperature
    without re-deriving fold membership.
    """
    frame = permuted_selections.copy()
    frame["target_index"] = frame["target_symbol"].map(symbol_index)
    prior_lookup = {
        (row.prior_model, row.context_prefix): np.asarray(row.p_lm)
        for row in priors.itertuples(index=False)
        if row.prior_model == prior_model
    }
    columns = {
        "p_neural": frame["p_neural"].map(np.asarray),
        "p_lm": [
            prior_lookup[(prior_model, prefix)]
            for prefix in frame["permuted_context_prefix"]
        ],
        "target_index": frame["target_index"],
    }
    if "selection_row_id" in frame.columns:
        columns["selection_row_id"] = frame["selection_row_id"]
    return pd.DataFrame(columns)


def _mean_ncf_phantom_agreement(frame: pd.DataFrame) -> tuple[float, float]:
    """Unweighted mean NCF and phantom agreement across one frame's rows, matching the
    convention used for the other formalism/sensitivity comparisons (eTable 13, eTable
    14). Shared by negative_control_attribution's per-permutation loop and by the caller
    computing the real, unpermuted comparison value.
    """
    ncf_values, phantom_values = [], []
    for row in frame.itertuples(index=False):
        neural = np.asarray(row.p_neural)
        prior = np.asarray(row.p_lm)
        target_index = int(row.target_index)
        fused = fuse(neural, prior)
        result = attribution_row(neural, prior, fused, target_index)
        ncf_values.append(result["ncf"])
        phantom_values.append(result["phantom_agreement"])
    return float(np.nanmean(ncf_values)), float(np.mean(phantom_values))


def negative_control_attribution(
    frames_by_permutation: list[pd.DataFrame],
    real_ncf: float | None = None,
    real_phantom_agreement: float | None = None,
) -> dict[str, object]:
    """Mean NCF and phantom agreement across many stratified permutations (preserving
    character position, context length, and source study), reporting the permutation-
    null distribution's own mean/CI and an exceedance p-value against the real,
    unpermuted result. Supersedes the single-fixed-permutation version (round-2 review
    Item 7).

    Each element of `frames_by_permutation` holds the rows for one permutation draw, in
    the same p_neural/p_lm/target_index shape as negative_control_attribution's previous
    single-frame input. `real_ncf`/`real_phantom_agreement`, when given, are the
    unpermuted (real-context) values to test the null distribution against: the exceedance
    p-value is one-sided (mean(null_draws >= real_value)) because the reviewer's concern
    is specifically whether the real result is implausibly extreme relative to a proper
    null, not whether it differs in either direction.

    That upper-tail p-value alone is directionally uninformative: a real value that sits
    in the extreme LOWER tail of the null (as NCF does in the real run -- below every one
    of 200 permutation draws) reads as "not significant" (p=1.0) under the upper-tail
    convention alone, which a reader who only checks that one field would misinterpret as
    no evidence of an effect. So each *_p_value is paired with a complementary
    *_p_value_lower_tail (mean(null_draws <= real_value)), and the returned dict always
    carries a general interpretation_note flagging that both tails should be checked
    (Task 14 fix-round Finding 2).
    """
    per_permutation_ncf, per_permutation_phantom = [], []
    for frame in frames_by_permutation:
        ncf_mean, phantom_mean = _mean_ncf_phantom_agreement(frame)
        per_permutation_ncf.append(ncf_mean)
        per_permutation_phantom.append(phantom_mean)

    ncf_null_draws = np.asarray(per_permutation_ncf)
    phantom_null_draws = np.asarray(per_permutation_phantom)
    result = {
        "n_permutations": len(frames_by_permutation),
        "ncf_null_mean": float(np.mean(ncf_null_draws)),
        "ncf_null_ci": [
            float(np.percentile(ncf_null_draws, 2.5)),
            float(np.percentile(ncf_null_draws, 97.5)),
        ],
        "phantom_agreement_null_mean": float(np.mean(phantom_null_draws)),
        "phantom_agreement_null_ci": [
            float(np.percentile(phantom_null_draws, 2.5)),
            float(np.percentile(phantom_null_draws, 97.5)),
        ],
    }
    result["interpretation_note"] = (
        "ncf_p_value/phantom_agreement_p_value are one-sided, upper-tail exceedance "
        "p-values (mean(null_draws >= real_value)); a real result can instead be "
        "extreme in the LOWER tail of the null (as NCF is in the real run, sitting "
        "below every permutation draw), in which case the upper-tail p-value alone "
        "reads as non-significant. Always check the paired lower-tail values "
        "(ncf_p_value_lower_tail/phantom_agreement_p_value_lower_tail, "
        "mean(null_draws <= real_value)) alongside the upper-tail ones before "
        "concluding there is no effect."
    )
    if real_ncf is not None:
        result["real_ncf"] = float(real_ncf)
        result["ncf_p_value"] = float(np.mean(ncf_null_draws >= real_ncf))
        result["ncf_p_value_lower_tail"] = float(np.mean(ncf_null_draws <= real_ncf))
    if real_phantom_agreement is not None:
        result["real_phantom_agreement"] = float(real_phantom_agreement)
        result["phantom_agreement_p_value"] = float(
            np.mean(phantom_null_draws >= real_phantom_agreement)
        )
        result["phantom_agreement_p_value_lower_tail"] = float(
            np.mean(phantom_null_draws <= real_phantom_agreement)
        )
    return result


def _load_combined_stratified_permutations() -> pd.DataFrame:
    return pd.read_parquet(INTERMEDIATE / "permuted_context_selections_stratified.parquet")


def stratified_permutation_frames(
    selections: pd.DataFrame, priors: pd.DataFrame, prior_model: str
) -> list[pd.DataFrame]:
    """Load the combined long-format stratified-permutation parquet (Task 14, 200 draws
    by default) and rebuild one permuted_context_frame() input per permutation_index.

    The parquet only carries selection_row_id/permutation_index/permuted_context_prefix
    (build_permuted_context_prior.py never duplicates p_neural per permutation); this
    rejoins each draw back to `selections` by that row id before handing it to the
    existing permuted_context_frame(), unchanged.
    """
    combined = _load_combined_stratified_permutations()
    base = selections.reset_index(drop=True).reset_index().rename(
        columns={"index": "selection_row_id"}
    )
    frames = []
    for _, draw in combined.groupby("permutation_index", sort=True):
        merged = draw.merge(base, on="selection_row_id", how="left")
        frames.append(permuted_context_frame(merged, priors, prior_model))
    return frames


def calibrated_stratified_permutation_frames(
    permutation_frames: list[pd.DataFrame], fold_temperature_lookup: pd.DataFrame
) -> list[pd.DataFrame]:
    """Calibrated counterpart of stratified_permutation_frames: apply each selection's
    already-fit fold temperature -- fit once on the REAL (unpermuted) permutable subset,
    never on permuted data -- to that same selection's p_neural/p_lm in every permuted
    draw it appears in.

    `fold_temperature_lookup` must carry selection_row_id/t_neural_used/t_lm_used, i.e.
    calibrated_attribution_frame's output restricted to the permutable subset (which
    calibrated_fusion_frame stamps with the fold-fitted temperature actually applied to
    each row -- see authorship.calibration.calibrated_fusion_frame). Every row of every
    permutation_frames[i] is a permutable selection (stratified_permutation_frames only
    ever draws from the permutable rows), so the join below has no unmatched rows and
    never falls back to fitting anything fresh on the permuted p_neural/p_lm it produces.
    """
    from authorship.calibration import apply_temperature

    calibrated_frames = []
    for frame in permutation_frames:
        merged = frame.merge(fold_temperature_lookup, on="selection_row_id", how="left")
        if merged["t_neural_used"].isna().any() or merged["t_lm_used"].isna().any():
            raise ValueError(
                "a permuted-draw row failed to join a real-data fold temperature; "
                "fold_temperature_lookup must cover every selection_row_id the "
                "permutation frames carry"
            )
        calibrated_frames.append(
            pd.DataFrame(
                {
                    "p_neural": [
                        apply_temperature(np.asarray(p), t)
                        for p, t in zip(merged["p_neural"], merged["t_neural_used"])
                    ],
                    "p_lm": [
                        apply_temperature(np.asarray(p), t)
                        for p, t in zip(merged["p_lm"], merged["t_lm_used"])
                    ],
                    "target_index": merged["target_index"],
                }
            )
        )
    return calibrated_frames


def permutable_selections(selections: pd.DataFrame, keep_row_id: bool = False) -> pd.DataFrame:
    """The subset of `selections` eligible for the stratified permutation null -- i.e.
    excluding rows in structurally unpermutable strata (every position_in_phrase == 0
    selection has context_prefix == "" by construction, plus one singleton stratum; see
    build_permuted_context_prior.permute()). The same row set is excluded for every
    permutation_index (exclusion is a property of the stratum, not the seed), so any one
    draw's selection_row_id set is representative.

    A valid permutation test requires the observed statistic and the null distribution to
    be computed over the identical rows -- comparing a null built from 2,710 permutable
    rows against a real value computed on all 3,373 would mix populations, and would be
    biased in a known direction here since the excluded rows are exactly position 0, where
    phantom agreement is elevated and NCF is lower (round-2 review Item 7 follow-up).

    `keep_row_id`, when True, retains the `selection_row_id` column instead of dropping
    it (default) -- the calibrated negative control needs it to later join each row's
    real-data-fit fold temperature back onto the permutation frames, which are keyed on
    the same id (see stratified_permutation_frames).
    """
    combined = _load_combined_stratified_permutations()
    first_index = combined["permutation_index"].min()
    permutable_ids = combined.loc[combined["permutation_index"] == first_index, "selection_row_id"]
    base = selections.reset_index(drop=True).reset_index().rename(
        columns={"index": "selection_row_id"}
    )
    filtered = base[base["selection_row_id"].isin(permutable_ids)]
    return filtered if keep_row_id else filtered.drop(columns=["selection_row_id"])


def emitted_context_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Mean NCF and phantom agreement under the closed-loop emitted-context primary-prior
    attribution (scripts/build_attribution_emitted_context.py, run on O2). Reports unweighted
    means, in the same shape as negative_control_attribution() so the intended- and
    emitted-context sides of eTable 16 are directly comparable, plus participant-clustered
    bootstrap CIs and participant-weighted means (round-2 review Item 4: "Confidence
    intervals and participant weighting should also be provided for the closed-loop
    estimate"). Also reports how often the emitted context actually diverged from the
    archive's intended context (a wrong earlier emission in the same phrase), the statistic
    reported in eMethods S1.14 and eTable 16.
    """
    diverged = int((frame["emitted_context"] != frame["context_prefix"]).sum())
    ncf_result = cluster_bootstrap(frame, lambda data: participant_mean(data, "ncf"))
    phantom_result = cluster_bootstrap(
        frame, lambda data: participant_mean(data, "phantom_agreement")
    )
    return {
        "n": int(len(frame)),
        "ncf_mean": float(np.nanmean(frame["ncf"])),
        "ncf": strip(ncf_result),
        "phantom_agreement_mean": float(frame["phantom_agreement"].mean()),
        "phantom_agreement": strip(phantom_result),
        "context_diverged_from_intended": {
            "n": diverged,
            "fraction": diverged / len(frame),
        },
    }


def emitted_context_raw_frame(
    emitted_context: pd.DataFrame, selections: pd.DataFrame, pair_key: list[str]
) -> pd.DataFrame:
    """Build a calibratable p_neural/p_lm/target_index/CLUSTER frame for the emitted-
    context side, for calibrated_attribution_frame to consume.

    emitted_context_attribution.parquet (scripts/build_attribution_emitted_context.py)
    carries the emitted-context p_lm it scored plus derived scalar measures, but -- like
    attribution.parquet -- not the raw p_neural vector each row was fused against, since
    p_neural does not depend on context and was already available in selections.parquet
    when the emitted-context score was computed. Recover it here the same way that
    script did: joined from `selections` on the same per-selection identity
    (relative_path + trial_number), not by row position (this is a join across two
    independently built files, so positional alignment is not guaranteed and is not
    assumed here; see test_relative_path_trial_number_uniquely_pairs_selections_and_
    emitted_context for the verification that this key is a duplicate-free 1:1 pairing).
    """
    frame = emitted_context[[*pair_key, "target_symbol", "p_lm"]].merge(
        selections[[*pair_key, "p_neural", CLUSTER]],
        on=pair_key, how="inner", validate="one_to_one",
    )
    assert len(frame) == len(emitted_context)
    frame["target_index"] = frame["target_symbol"].map(symbol_index)
    return frame


def emitted_context_shared_temperature_frame(
    raw_frame: pd.DataFrame, temperature_lookup: pd.DataFrame, pair_key: list[str],
) -> pd.DataFrame:
    """Score the emitted-context side under the intended side's already-fit per-row
    temperatures (calibrated_primary's t_neural_used/t_lm_used, Task 9), instead of
    independently refitting a temperature on the emitted arm.

    For a paired difference, holding the calibration transformation fixed across both
    arms is the standard way to isolate the contrast actually being measured (context
    change) from a second, confounding source of variation (temperature change).
    Refitting independently on the emitted side lets the calibrator partially absorb the
    very effect under study: the emitted-context LM prior is more confidently wrong than
    the intended-context one, so an independent fit assigns it a systematically higher
    (flatter) t_lm, which shrinks the measured gap -- Task 10 fix-round-1 found this
    attenuates the NCF gap roughly threefold relative to this shared-temperature version.
    See emitted_context_paired_calibrated_comparison's caller in main() for the
    independently-refit variant, kept as a labeled sensitivity comparison rather than
    deleted.

    Only apply_temperature (pure power/renormalize math) is called here, never
    fit_temperature -- reusing a temperature already fit once on real, intended-context
    data keeps this fully held-out with no new fitting on the emitted arm.
    """
    from authorship.calibration import apply_temperature

    merged = raw_frame.merge(temperature_lookup, on=pair_key, how="left", validate="one_to_one")
    if merged["t_neural_used"].isna().any() or merged["t_lm_used"].isna().any():
        raise ValueError(
            "an emitted-context row failed to join a shared fold temperature; "
            "temperature_lookup must cover every pair_key value raw_frame carries"
        )

    ncf_values, phantom_values = [], []
    for row in merged.itertuples(index=False):
        neural = apply_temperature(np.asarray(row.p_neural), row.t_neural_used)
        prior = apply_temperature(np.asarray(row.p_lm), row.t_lm_used)
        fused = fuse(neural, prior)
        result = attribution_row(neural, prior, fused, int(row.target_index))
        ncf_values.append(result["ncf"])
        phantom_values.append(result["phantom_agreement"])

    output = merged[pair_key].copy()
    output["ncf"] = ncf_values
    output["phantom_agreement"] = phantom_values
    return cast_binary_columns(output)


def emitted_context_paired_calibrated_comparison(
    intended_frame: pd.DataFrame, emitted_frame: pd.DataFrame, pair_key: list[str],
) -> dict[str, object]:
    """Paired comparison of intended- vs emitted-context attribution, both on calibrated
    fusion, both weighted and resampled identically -- replaces the two independently
    weighted/CI'd rows in sensitivity.emitted_context_primary_prior with one paired
    participant-cluster bootstrap difference, so the two sides are on the same
    statistical footing by construction rather than merely both being reported.
    """
    paired = intended_frame.merge(
        emitted_frame, on=pair_key, suffixes=("_intended", "_emitted"), validate="one_to_one"
    )
    assert len(paired) == len(intended_frame) == len(emitted_frame)
    ncf_result = cluster_bootstrap(
        paired, lambda data: participant_mean(data, "ncf_emitted") - participant_mean(data, "ncf_intended")
    )
    phantom_result = cluster_bootstrap(
        paired,
        lambda data: participant_mean(data, "phantom_agreement_emitted")
        - participant_mean(data, "phantom_agreement_intended"),
    )
    return {
        "n": int(len(paired)),
        "ncf_difference": strip(ncf_result),
        "phantom_agreement_difference": strip(phantom_result),
    }


def excluding_dynamic_stopping(primary: pd.DataFrame, condition: str) -> dict[str, object]:
    """Co-primary estimates with "Dynamic stopping with bigram model" removed (Major Comment #8).

    That condition is documented (eTable 8) as having run a bigram language-model prior
    online during the original session, so its post-hoc attribution could be circular: the
    live decision was already language-informed before this study's own prior was ever
    applied. Plain "Dynamic stopping" also halts a trial early, but on the neural classifier's
    own confidence alone, with no language model in the loop, so it carries no such concern
    and stays in the remainder.
    """
    excluded = primary[primary["condition"] == condition]
    remainder = primary[primary["condition"] != condition]
    return {
        "condition_excluded": condition,
        "n_excluded": int(len(excluded)),
        "n_remaining": int(len(remainder)),
        "ncf_mean": strip(
            cluster_bootstrap(remainder, lambda data: participant_mean(data, "ncf"))
        ),
        "phantom_agreement_mean": strip(
            cluster_bootstrap(remainder, lambda data: participant_mean(data, "phantom_agreement"))
        ),
    }


def main() -> None:
    attribution = cast_binary_columns(pd.read_parquet(INTERMEDIATE / "attribution.parquet"))
    primary = attribution[
        (attribution["beta"] == PRIMARY_BETA) & (attribution["prior_model"] == PRIMARY_PRIOR)
    ].copy()
    primary["auc_tertile"] = pd.qcut(primary["train_auc"], 3, labels=["low", "mid", "high"])
    ladder = attribution[attribution["beta"] == PRIMARY_BETA].copy()

    selections = pd.read_parquet(INTERMEDIATE / "selections.parquet")
    priors = pd.read_parquet(INTERMEDIATE / "priors.parquet")
    primary_raw = primary_prior_frame(selections, priors, PRIMARY_PRIOR)

    # Shared calibrated-primary covariate frame, built once and reused by Tasks 5-10
    # (calibrated_attribution_frame refits held-out temperatures via 5-fold minimize_scalar,
    # which is not free to repeat six times for the same primary prior).
    # cast_binary_columns matches the raw `primary` frame's convention: calibrated_
    # attribution_frame emits phantom_agreement/fused_correct as booleans, which
    # fit_binary and fit_binary_crossed refuse outright because a boolean outcome is
    # expanded into a two-level categorical whose complement gets modelled.
    calibrated_primary = cast_binary_columns(calibrated_attribution_frame(
        full_prior_frame(
            selections, priors, PRIMARY_PRIOR,
            extra_columns=(
                "condition", "study", "train_auc", "position_in_phrase", "phrase_length",
                "calibration_fallback", "intended_phrase", "session_id",
                "relative_path", "trial_number", "phrase_is_numeric",
            ),
        ),
        group_column=CLUSTER, n_folds=5, seed=0,
    ))
    calibrated_primary["auc_tertile"] = pd.qcut(
        calibrated_primary["train_auc"], 3, labels=["low", "mid", "high"]
    )
    calibrated_primary["_session_key"] = _composite_session_key(calibrated_primary)

    digest: dict[str, object] = {
        "specification": {
            "primary_prior": PRIMARY_PRIOR,
            "primary_beta": PRIMARY_BETA,
            "alpha_per_coprimary": ALPHA_COPRIMARY,
            "bootstrap_replicates": N_BOOTSTRAP,
            "cluster_unit": "participant",
            "n_selections": int(len(primary)),
            "n_participants": int(primary[CLUSTER].nunique()),
            "n_sessions": int(primary.groupby([CLUSTER, "session_id"]).ngroups),
            "n_ncf_undefined": int(primary["ncf"].isna().sum()),
        }
    }

    # Principal co-primary analysis: fusion after held-out, per-source temperature
    # calibration (Task 11/12). The previous round's raw, uncalibrated beta=1 result is
    # retained below as sensitivity["uncalibrated_beta_1"], not deleted.
    digest["calibrated_primary_analysis"] = calibrated_primary_analysis(primary_raw)

    # Co-primary 1: neural contribution fraction, null value 1 (entirely neural).
    ncf_result = cluster_bootstrap(primary, lambda data: participant_mean(data, "ncf"))
    ncf_p = one_sided_p(ncf_result["_draws"], 1.0, "less")
    coprimary_1 = {
        **strip(ncf_result),
        "null_value": 1.0,
        "alternative": "less",
        "p_value": ncf_p,
        "meets_alpha": bool(ncf_p < ALPHA_COPRIMARY),
        "prior_share": 1.0 - ncf_result["estimate"],
    }

    # Co-primary 2: phantom agreement rate, null value 0 (every correct emission supported).
    phantom_result = cluster_bootstrap(
        primary, lambda data: participant_mean(data, "phantom_agreement")
    )
    phantom_p = one_sided_p(phantom_result["_draws"], 0.0, "greater")
    coprimary_2 = {
        **strip(phantom_result),
        "null_value": 0.0,
        "alternative": "greater",
        "p_value": phantom_p,
        "meets_alpha": bool(phantom_p < ALPHA_COPRIMARY),
    }

    gate_open = coprimary_1["meets_alpha"] or coprimary_2["meets_alpha"]
    digest["gatekeeping"] = {
        "rule": "secondary family tested only if at least one co-primary meets alpha 0.025",
        "secondary_family_tested": bool(gate_open),
    }

    if gate_open:
        secondary: dict[str, object] = {}
        secondary["prior_capture"] = strip(
            cluster_bootstrap(primary, lambda data: participant_mean(data, "prior_capture"))
        )
        secondary["neural_override"] = strip(
            cluster_bootstrap(primary, lambda data: participant_mean(data, "neural_override"))
        )
        secondary["accuracy_change_with_prior"] = strip(
            cluster_bootstrap(
                primary,
                lambda data: participant_mean(data, "fused_correct")
                - participant_mean(data, "neural_correct"),
            )
        )

        secondary["by_position_in_word"] = {
            str(position): {
                "phantom_agreement": strip(
                    cluster_bootstrap(
                        group, lambda data: participant_mean(data, "phantom_agreement")
                    )
                ),
                "ncf": strip(
                    cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))
                ),
                "n": int(len(group)),
            }
            for position, group in primary.groupby("position_in_phrase")
        }
        secondary["by_position_in_word"]["word_and_session_adjusted"] = (
            word_and_session_adjusted_position_effect(primary)
        )
        secondary["by_position_in_word"]["six_char_words_only"] = (
            six_char_words_only_position_effect(primary)
        )

        secondary["calibrated"] = secondary.get("calibrated", {})

        # Calibrated-basis counterparts of the three population-level secondary outcomes
        # computed on `primary` above, so every headline secondary number is reportable on
        # the same held-out-calibrated basis as the co-primary analysis.
        secondary["calibrated"]["prior_capture"] = strip(
            cluster_bootstrap(
                calibrated_primary, lambda data: participant_mean(data, "prior_capture")
            )
        )
        secondary["calibrated"]["neural_override"] = strip(
            cluster_bootstrap(
                calibrated_primary, lambda data: participant_mean(data, "neural_override")
            )
        )
        secondary["calibrated"]["accuracy_change_with_prior"] = strip(
            cluster_bootstrap(
                calibrated_primary,
                lambda data: participant_mean(data, "fused_correct")
                - participant_mean(data, "neural_correct"),
            )
        )

        # Calibrated-basis counterpart: same position-in-word NCF analysis, recomputed
        # on calibrated_primary (Task 4's held-out temperature-calibrated fusion frame)
        # instead of the raw beta=1 primary frame, alongside (not replacing) the block
        # above.
        secondary["calibrated"]["by_position_in_word"] = {
            str(position): {
                "phantom_agreement": strip(cluster_bootstrap(
                    group, lambda data: participant_mean(data, "phantom_agreement")
                )),
                "ncf": strip(cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))),
                "n": int(len(group)),
            }
            for position, group in calibrated_primary.groupby("position_in_phrase")
        }
        secondary["calibrated"]["by_position_in_word"]["word_and_session_adjusted"] = (
            word_and_session_adjusted_position_effect(calibrated_primary)
        )
        secondary["calibrated"]["by_position_in_word"]["six_char_words_only"] = (
            six_char_words_only_position_effect(calibrated_primary)
        )
        secondary["calibrated"]["models"] = secondary["calibrated"].get("models", {})
        secondary["calibrated"]["models"]["ncf_by_position"] = fit_mixed_continuous(
            calibrated_primary, "ncf ~ position_in_phrase"
        )
        secondary["calibrated"]["models"]["phantom_by_position"] = fit_binary(
            calibrated_primary, "phantom_agreement ~ position_in_phrase"
        )
        secondary["calibrated"]["models"]["phantom_by_position_word_session_adjusted"] = (
            fit_binary_crossed(
                calibrated_primary,
                "phantom_agreement ~ position_in_phrase",
                PHANTOM_VC_GROUPS,
            )
        )
        secondary["calibrated"]["models"]["ncf_by_decoder_quality"] = fit_mixed_continuous(
            calibrated_primary, "ncf ~ train_auc"
        )
        # Both bounded fits carry the predicted-value contrast the Results report them by
        # (Task 2). The AUC contrast reuses this paper's existing tertile framework, running
        # between the lower and upper tertiles' median AUC -- the same two x positions panel
        # b's outer tertile ticks already sit at -- so the reported number is readable off
        # the plotted curve. The position contrast spans the observed range of the moderator,
        # read from the frame rather than hard-coded, since "first to sixth character" is
        # only correct while the corpus maximum stays six.
        auc_tertile_median = (
            calibrated_primary.groupby("auc_tertile", observed=True)["train_auc"].median()
        )
        secondary["calibrated"]["models"]["ncf_by_decoder_quality_bounded"] = fit_fractional_logit(
            calibrated_primary,
            "ncf ~ train_auc",
            contrast=(
                "train_auc",
                float(auc_tertile_median["low"]),
                float(auc_tertile_median["high"]),
            ),
        )
        secondary["calibrated"]["models"]["ncf_by_position_bounded"] = fit_fractional_logit(
            calibrated_primary,
            "ncf ~ position_in_phrase",
            contrast=(
                "position_in_phrase",
                float(calibrated_primary["position_in_phrase"].min()),
                float(calibrated_primary["position_in_phrase"].max()),
            ),
        )
        # Task 29: calibrated-basis counterpart of secondary["models"]["phantom_by_
        # decoder_quality"] below, mirroring the phantom_by_position sibling above; this
        # was the one calibrated model in the by-decoder-quality pair left uncomputed.
        secondary["calibrated"]["models"]["phantom_by_decoder_quality"] = fit_binary(
            calibrated_primary, "phantom_agreement ~ train_auc"
        )

        # Calibrated-basis counterpart of secondary["by_decoder_quality"] below: same
        # per-tertile phantom/NCF estimates, grouped by calibrated_primary's own auc_tertile
        # column instead of primary's. Task 12b: Figure 2 panel b's NCF scatter, tertile
        # markers, and rug move onto this basis so they match the calibrated NCF curve
        # already drawn there.
        secondary["calibrated"]["by_decoder_quality"] = {
            str(tertile): {
                "phantom_agreement": strip(
                    cluster_bootstrap(
                        group, lambda data: participant_mean(data, "phantom_agreement")
                    )
                ),
                "ncf": strip(
                    cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))
                ),
                "n": int(len(group)),
                # The x position of this tertile's tick in Figure 2B, and, for the outer two,
                # an endpoint of the bounded model's reported predicted contrast (Task 2).
                "train_auc_median": float(group["train_auc"].median()),
            }
            for tertile, group in calibrated_primary.groupby("auc_tertile", observed=True)
        }

        # Task 29: calibrated-basis counterpart of secondary["by_target_type"] and
        # secondary["models"]["phantom_by_target_type"] below, completing the calibrated
        # sibling of every member of the raw five-test secondary family.
        secondary["calibrated"]["by_target_type"] = {
            ("numeric" if numeric else "word"): {
                "phantom_agreement": strip(
                    cluster_bootstrap(
                        group, lambda data: participant_mean(data, "phantom_agreement")
                    )
                ),
                "ncf": strip(
                    cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))
                ),
                "n": int(len(group)),
            }
            for numeric, group in calibrated_primary.groupby("phrase_is_numeric")
        }
        secondary["calibrated"]["models"]["phantom_by_target_type"] = fit_binary(
            calibrated_primary, "phantom_agreement ~ phrase_is_numeric"
        )

        secondary["by_decoder_quality"] = {
            str(tertile): {
                "phantom_agreement": strip(
                    cluster_bootstrap(
                        group, lambda data: participant_mean(data, "phantom_agreement")
                    )
                ),
                "ncf": strip(
                    cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))
                ),
                "n": int(len(group)),
            }
            for tertile, group in primary.groupby("auc_tertile", observed=True)
        }
        secondary["by_target_type"] = {
            ("numeric" if numeric else "word"): {
                "phantom_agreement": strip(
                    cluster_bootstrap(
                        group, lambda data: participant_mean(data, "phantom_agreement")
                    )
                ),
                "ncf": strip(
                    cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))
                ),
                "n": int(len(group)),
            }
            for numeric, group in primary.groupby("phrase_is_numeric")
        }

        models = {
            "ncf_by_position": fit_mixed_continuous(primary, "ncf ~ position_in_phrase"),
            "ncf_by_decoder_quality": fit_mixed_continuous(primary, "ncf ~ train_auc"),
            "phantom_by_position": fit_binary(
                primary, "phantom_agreement ~ position_in_phrase"
            ),
            "phantom_by_position_word_session_adjusted": fit_binary_crossed(
                primary.assign(_session_key=_composite_session_key(primary)),
                "phantom_agreement ~ position_in_phrase",
                PHANTOM_VC_GROUPS,
            ),
            "phantom_by_decoder_quality": fit_binary(primary, "phantom_agreement ~ train_auc"),
            "phantom_by_target_type": fit_binary(
                primary, "phantom_agreement ~ phrase_is_numeric"
            ),
        }
        secondary["models"] = models

        family_keys = [
            ("phantom_by_position", "position_in_phrase"),
            ("phantom_by_decoder_quality", "train_auc"),
            ("phantom_by_target_type", "phrase_is_numeric"),
            ("ncf_by_position", "position_in_phrase"),
            ("ncf_by_decoder_quality", "train_auc"),
        ]
        raw_p = []
        for model_key, term in family_keys:
            terms = models[model_key]["terms"]
            matches = [name for name in terms if name == term or name.startswith(f"{term}[")]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot resolve term {term!r} in {model_key}; candidates {sorted(terms)}"
                )
            raw_p.append(terms[matches[0]]["p_value"])
        rejected, adjusted, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")
        secondary["multiplicity"] = {
            "method": "benjamini_hochberg",
            "family": [f"{key}:{term}" for key, term in family_keys],
            "raw_p": raw_p,
            "adjusted_p": list(adjusted),
            "rejected": [bool(value) for value in rejected],
        }

        # Task 29: calibrated-basis counterpart of the multiplicity correction above, same
        # five-member family, same procedure, evaluated on the calibrated models added
        # alongside calibrated_primary earlier in this function (phantom_by_decoder_quality
        # and phantom_by_target_type were the two missing siblings; the other three already
        # existed). Kept as a separate key rather than replacing secondary["multiplicity"],
        # which remains the adjustment for the raw-basis family reported as the sensitivity
        # comparison.
        calibrated_family_keys = [
            ("phantom_by_position", "position_in_phrase"),
            ("phantom_by_decoder_quality", "train_auc"),
            ("phantom_by_target_type", "phrase_is_numeric"),
            ("ncf_by_position", "position_in_phrase"),
            ("ncf_by_decoder_quality", "train_auc"),
        ]
        calibrated_models = secondary["calibrated"]["models"]
        calibrated_raw_p = []
        for model_key, term in calibrated_family_keys:
            terms = calibrated_models[model_key]["terms"]
            matches = [name for name in terms if name == term or name.startswith(f"{term}[")]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot resolve term {term!r} in calibrated {model_key}; "
                    f"candidates {sorted(terms)}"
                )
            calibrated_raw_p.append(terms[matches[0]]["p_value"])
        calibrated_rejected, calibrated_adjusted, _, _ = multipletests(
            calibrated_raw_p, alpha=0.05, method="fdr_bh"
        )
        secondary["calibrated"]["multiplicity"] = {
            "method": "benjamini_hochberg",
            "family": [f"{key}:{term}" for key, term in calibrated_family_keys],
            "raw_p": calibrated_raw_p,
            "adjusted_p": list(calibrated_adjusted),
            "rejected": [bool(value) for value in calibrated_rejected],
        }

        secondary["phantom_agreement_margin_distribution"] = (
            phantom_agreement_margin_distribution(primary)
        )
        # Task 24: Figure 4 moved onto the calibrated basis; phantom_agreement_margin_
        # distribution is generic over any frame carrying phantom_agreement/neural_p_target/
        # neural_rank_of_target/neural_margin_to_target, which calibrated_primary now does
        # too (extended in calibrated_attribution_frame), so the raw-basis function is reused
        # unchanged rather than duplicated.
        secondary["calibrated"]["phantom_agreement_margin_distribution"] = (
            phantom_agreement_margin_distribution(calibrated_primary)
        )
        primary_raw_with_correct = primary_raw.assign(
            neural_correct=_neural_argmax_correct(primary_raw)
        )
        secondary["neural_posterior_calibration"] = {
            "target_probability": neural_posterior_calibration(primary_raw),
            "top_label": top_label_calibration(primary_raw_with_correct),
        }

        # Symmetric 4-cell calibration diagnostics: both sources (neural decoder,
        # language-model prior), both bases (pre-scaling raw posteriors, post-scaling
        # held-out temperature-calibrated posteriors from calibrated_primary/Task 4).
        secondary["calibration_diagnostics"] = {
            "neural": {
                "pre_scaling": source_calibration_diagnostics(
                    np.stack(primary_raw["p_neural"].to_numpy()),
                    primary_raw["target_index"].to_numpy(),
                ),
                "post_scaling": source_calibration_diagnostics(
                    np.stack(calibrated_primary["p_neural_calibrated"].to_numpy()),
                    calibrated_primary["target_index"].to_numpy(),
                ),
            },
            "prior": {
                "pre_scaling": source_calibration_diagnostics(
                    np.stack(primary_raw["p_lm"].to_numpy()),
                    primary_raw["target_index"].to_numpy(),
                ),
                "post_scaling": source_calibration_diagnostics(
                    np.stack(calibrated_primary["p_lm_calibrated"].to_numpy()),
                    calibrated_primary["target_index"].to_numpy(),
                ),
            },
        }
        digest["secondary"] = secondary

    # Pre-specified sensitivity analyses.
    sensitivity: dict[str, object] = {}
    sensitivity["uncalibrated_beta_1"] = {
        "coprimary_1_neural_contribution_fraction": coprimary_1,
        "coprimary_2_phantom_agreement": coprimary_2,
    }
    sensitivity["prior_ladder"] = {
        name: {
            "ncf": strip(cluster_bootstrap(group, lambda data: participant_mean(data, "ncf"))),
            "phantom_agreement": strip(
                cluster_bootstrap(group, lambda data: participant_mean(data, "phantom_agreement"))
            ),
            "prior_capture": strip(
                cluster_bootstrap(group, lambda data: participant_mean(data, "prior_capture"))
            ),
            "prior_parameters": int(group["prior_parameters"].iloc[0]),
            "fused_accuracy": float(participant_mean(group, "fused_correct")),
            "fused_accuracy_ci": strip(
                cluster_bootstrap(group, lambda data: participant_mean(data, "fused_correct"))
            ),
            "neural_accuracy": float(participant_mean(group, "neural_correct")),
        }
        for name, group in ladder.groupby("prior_model")
    }
    sensitivity["cross_family_scale_check"] = cross_family_scale_check(ladder)

    ladder_points = _ladder_points(ladder)
    scale_metrics = ["ncf", "phantom_agreement", "prior_capture", "accuracy_gained"]
    sensitivity["architecture_family_permutation_test"] = {
        metric: architecture_family_permutation_test(
            ladder_points, metric, seed=SEED, n_permutations=10000
        )
        for metric in scale_metrics
    }
    sensitivity["scale_equivalence_test"] = {
        metric: scale_equivalence_test(ladder_points, metric, seed=SEED, margin=0.3)
        for metric in scale_metrics
    }

    # Calibrated ladder: an independently fit held-out temperature per non-uniform model
    # (uniform has no informative shape for a temperature to sharpen or flatten, so
    # fit_temperature would be fitting noise), redoing the raw ladder's per-model summary
    # and the parameter-count/architecture-family correlation checks on that basis. Each
    # model's calibrated_attribution_frame is computed exactly once below and reused for
    # both the per-model summary dict and the row-level concatenated frame the correlation
    # checks need, instead of calling calibrated_attribution_frame 48 times (24 models x 2
    # uses).
    calibrated_ladder_frames = {
        model_name: calibrated_attribution_frame(
            full_prior_frame(selections, priors, model_name), group_column=CLUSTER, n_folds=5, seed=0,
        )
        for model_name in sorted(priors["prior_model"].unique())
        if model_name != "uniform"
    }

    prior_ladder_calibrated = {}
    for model_name, model_frame_calibrated in calibrated_ladder_frames.items():
        parameters = int(
            (ladder[ladder["prior_model"] == model_name]["prior_parameters"]).iloc[0]
        )
        prior_ladder_calibrated[model_name] = {
            "ncf": strip(
                cluster_bootstrap(model_frame_calibrated, lambda data: participant_mean(data, "ncf"))
            ),
            "phantom_agreement": strip(cluster_bootstrap(
                model_frame_calibrated, lambda data: participant_mean(data, "phantom_agreement")
            )),
            "prior_capture": strip(cluster_bootstrap(
                model_frame_calibrated, lambda data: participant_mean(data, "prior_capture")
            )),
            "prior_parameters": parameters,
            "fused_accuracy": float(participant_mean(model_frame_calibrated, "fused_correct")),
            "fused_accuracy_ci": strip(cluster_bootstrap(
                model_frame_calibrated, lambda data: participant_mean(data, "fused_correct")
            )),
            # Prior-independent: neural_argmax only ever depends on p_neural, so reuse
            # `primary`'s value instead of recomputing it identically 24 times.
            "neural_accuracy": float(participant_mean(primary, "neural_correct")),
        }
    sensitivity["prior_ladder_calibrated"] = prior_ladder_calibrated

    neural_priors_only = {
        name: entry for name, entry in prior_ladder_calibrated.items()
        if "ngram" not in name
    }
    sensitivity["ladder_summary_calibrated"] = {
        "n_priors": len(neural_priors_only),
        "ncf_median": float(np.median([e["ncf"]["estimate"] for e in neural_priors_only.values()])),
        "ncf_mean": float(np.mean([e["ncf"]["estimate"] for e in neural_priors_only.values()])),
        "phantom_agreement_median": float(
            np.median([e["phantom_agreement"]["estimate"] for e in neural_priors_only.values()])
        ),
        "phantom_agreement_mean": float(
            np.mean([e["phantom_agreement"]["estimate"] for e in neural_priors_only.values()])
        ),
    }

    # Best character n-gram vs. best/primary LLM, both by calibrated fused_accuracy: a
    # paired participant-cluster bootstrap on the same selections, reusing the per-model
    # calibrated frames already built above (calibrated_ladder_frames) rather than
    # recomputing calibrated_attribution_frame a second time for these two models.
    ngram_models = [name for name in prior_ladder_calibrated if "ngram" in name]
    llm_models = [name for name in prior_ladder_calibrated if name not in ngram_models]
    best_ngram = max(ngram_models, key=lambda name: prior_ladder_calibrated[name]["fused_accuracy"])
    best_llm = max(llm_models, key=lambda name: prior_ladder_calibrated[name]["fused_accuracy"])

    ngram_paired = calibrated_ladder_frames[best_ngram][[CLUSTER, "fused_correct"]].rename(
        columns={"fused_correct": "correct_ngram"}
    ).reset_index(drop=True)
    llm_paired = calibrated_ladder_frames[best_llm][[CLUSTER, "fused_correct"]].rename(
        columns={"fused_correct": "correct_llm"}
    ).reset_index(drop=True)
    if ngram_paired[CLUSTER].tolist() != llm_paired[CLUSTER].tolist():
        raise ValueError(
            f"calibrated frames for {best_ngram!r} and {best_llm!r} are not aligned by row "
            "position; a positional pairing here would compare unrelated selections"
        )
    paired = ngram_paired
    paired["correct_llm"] = llm_paired["correct_llm"].to_numpy()
    character_vs_llm_result = cluster_bootstrap(
        paired,
        lambda data: participant_mean(data, "correct_ngram") - participant_mean(data, "correct_llm"),
    )
    sensitivity["character_model_vs_llm_accuracy"] = {
        "best_character_model": best_ngram,
        "best_llm": best_llm,
        "accuracy_difference": strip(character_vs_llm_result),
        "p_value": two_sided_p(character_vs_llm_result["_draws"], 0.0),
    }

    # Each model's own raw (pre-fusion, pre-calibration) prediction quality -- a property
    # of the language model itself, not of how it fuses with the neural decoder -- so this
    # is computed on p_lm straight from priors.parquet, never on calibrated_attribution_frame
    # or calibrated_primary.
    from authorship.calibration import mean_negative_log_likelihood

    prior_quality_metrics = {}
    for model_name in sorted(priors["prior_model"].unique()):
        if model_name == "uniform":
            continue
        frame = full_prior_frame(selections, priors, model_name)
        p_lm_matrix = np.stack(frame["p_lm"].to_numpy())
        target_indices = frame["target_index"].to_numpy()
        entropy_per_row = -np.sum(p_lm_matrix * np.log(np.maximum(p_lm_matrix, 1e-12)), axis=1)
        target_probability = p_lm_matrix[np.arange(len(frame)), target_indices]
        top1_correct = p_lm_matrix.argmax(axis=1) == target_indices
        prior_quality_metrics[model_name] = {
            "nll_intended_character": mean_negative_log_likelihood(p_lm_matrix, target_indices),
            "entropy_mean": float(entropy_per_row.mean()),
            "ece": _ece(pd.Series(target_probability), pd.Series(top1_correct)),
            "top1_accuracy": float(top1_correct.mean()),
        }
    sensitivity["prior_quality_metrics"] = prior_quality_metrics

    quality_points = pd.DataFrame([
        {
            "prior_model": name,
            "nll": metrics["nll_intended_character"],
            "entropy": metrics["entropy_mean"],
            "phantom_agreement": prior_ladder_calibrated[name]["phantom_agreement"]["estimate"],
            "ncf": prior_ladder_calibrated[name]["ncf"]["estimate"],
        }
        for name, metrics in prior_quality_metrics.items()
    ])
    sensitivity["prior_quality_correlations"] = {
        "spearman_rho_phantom_vs_nll": {
            "n": int(len(quality_points)),
            "rho": float(spearmanr(quality_points["nll"], quality_points["phantom_agreement"]).statistic),
            "p_value": float(spearmanr(quality_points["nll"], quality_points["phantom_agreement"]).pvalue),
        },
        "spearman_rho_phantom_vs_entropy": {
            "n": int(len(quality_points)),
            "rho": float(spearmanr(quality_points["entropy"], quality_points["phantom_agreement"]).statistic),
            "p_value": float(spearmanr(quality_points["entropy"], quality_points["phantom_agreement"]).pvalue),
        },
    }

    # Restricted to the 21 neural priors: the 3 n-gram variants are not comparable
    # "language model quality" entities (no NLL-vs-attribution relationship is
    # hypothesized for them), so this is a distinct, correctly-scoped analysis from
    # the n=24 block above rather than a subset view of it.
    neural_only_quality_points = quality_points[
        quality_points["prior_model"].apply(lambda name: "ngram" not in name)
    ]
    sensitivity["prior_quality_correlations_neural_only"] = {
        "spearman_rho_phantom_vs_nll": {
            "n": int(len(neural_only_quality_points)),
            "rho": float(
                spearmanr(
                    neural_only_quality_points["nll"], neural_only_quality_points["phantom_agreement"]
                ).statistic
            ),
            "p_value": float(
                spearmanr(
                    neural_only_quality_points["nll"], neural_only_quality_points["phantom_agreement"]
                ).pvalue
            ),
        },
        "spearman_rho_ncf_vs_nll": {
            "n": int(len(neural_only_quality_points)),
            "rho": float(
                spearmanr(neural_only_quality_points["nll"], neural_only_quality_points["ncf"]).statistic
            ),
            "p_value": float(
                spearmanr(neural_only_quality_points["nll"], neural_only_quality_points["ncf"]).pvalue
            ),
        },
    }

    calibrated_ladder_rows = []
    for model_name, model_frame_calibrated in calibrated_ladder_frames.items():
        row_frame = model_frame_calibrated.copy()
        row_frame["prior_model"] = model_name
        row_frame["prior_parameters"] = prior_ladder_calibrated[model_name]["prior_parameters"]
        calibrated_ladder_rows.append(row_frame)
    calibrated_ladder = pd.concat(calibrated_ladder_rows, ignore_index=True)
    sensitivity["cross_family_scale_check_calibrated"] = cross_family_scale_check(calibrated_ladder)

    calibrated_ladder_points = _ladder_points(calibrated_ladder)
    sensitivity["architecture_family_permutation_test_calibrated"] = {
        metric: architecture_family_permutation_test(
            calibrated_ladder_points, metric, seed=SEED, n_permutations=10000
        )
        for metric in scale_metrics
    }
    # Task 29 / Step 0b: calibrated-basis counterpart of sensitivity["scale_equivalence_
    # test"] above, same TOST procedure and margin, applied to calibrated_ladder_points
    # instead of the raw ladder_points -- the one remaining raw-only test in the Effect of
    # Parameter Count section now has a calibrated sibling, following the same pattern as
    # architecture_family_permutation_test_calibrated immediately above.
    sensitivity["scale_equivalence_test_calibrated"] = {
        metric: scale_equivalence_test(calibrated_ladder_points, metric, seed=SEED, margin=0.3)
        for metric in scale_metrics
    }

    sensitivity["fusion_exponent"] = {
        str(beta): {
            "ncf": float(participant_mean(group, "ncf")),
            "phantom_agreement": float(participant_mean(group, "phantom_agreement")),
            "prior_capture": float(participant_mean(group, "prior_capture")),
        }
        for beta, group in attribution[attribution["prior_model"] == PRIMARY_PRIOR].groupby("beta")
    }
    # Calibrated-basis counterpart: the same five-point exponent grid, re-fused from
    # calibrated_primary's held-out temperature-calibrated posteriors instead of the raw
    # p_neural/p_lm vectors above, alongside (not replacing) sensitivity["fusion_exponent"].
    sensitivity["fusion_exponent_calibrated"] = calibrated_fusion_exponent_points(calibrated_primary)
    high_fidelity = primary[primary["auc_tertile"] == "high"]
    sensitivity["high_fidelity_decoding_only"] = {
        "ncf": strip(cluster_bootstrap(high_fidelity, lambda data: participant_mean(data, "ncf"))),
        "phantom_agreement": strip(
            cluster_bootstrap(
                high_fidelity, lambda data: participant_mean(data, "phantom_agreement")
            )
        ),
        "n": int(len(high_fidelity)),
    }
    excluded_fallback = primary[~primary["calibration_fallback"].fillna(False)]
    sensitivity["excluding_pooled_calibration_sessions"] = {
        "phantom_agreement": float(participant_mean(excluded_fallback, "phantom_agreement")),
        "ncf": float(participant_mean(excluded_fallback, "ncf")),
        "n": int(len(excluded_fallback)),
    }
    sensitivity["leave_one_study_out"] = {
        study: {
            "phantom_agreement": float(
                participant_mean(primary[primary["study"] != study], "phantom_agreement")
            ),
            "ncf": float(participant_mean(primary[primary["study"] != study], "ncf")),
        }
        for study in sorted(primary["study"].unique())
    }
    sensitivity["by_condition"] = {
        condition: {
            "phantom_agreement": float(participant_mean(group, "phantom_agreement")),
            "ncf": float(participant_mean(group, "ncf")),
            "n": int(len(group)),
        }
        for condition, group in primary.groupby("condition")
    }

    # Calibrated-basis counterparts of the three sensitivity analyses above, recomputed
    # on calibrated_primary (Task 4's held-out temperature-calibrated fusion frame)
    # instead of the raw beta=1 primary frame, alongside (not replacing) the blocks above.
    sensitivity["calibrated"] = sensitivity.get("calibrated", {})
    calibrated_high_fidelity = calibrated_primary[calibrated_primary["auc_tertile"] == "high"]
    sensitivity["calibrated"]["high_fidelity_decoding_only"] = {
        "ncf": strip(cluster_bootstrap(calibrated_high_fidelity, lambda data: participant_mean(data, "ncf"))),
        "phantom_agreement": strip(cluster_bootstrap(
            calibrated_high_fidelity, lambda data: participant_mean(data, "phantom_agreement")
        )),
        "n": int(len(calibrated_high_fidelity)),
    }
    sensitivity["calibrated"]["leave_one_study_out"] = {
        study: {
            "phantom_agreement": float(
                participant_mean(calibrated_primary[calibrated_primary["study"] != study], "phantom_agreement")
            ),
            "ncf": float(participant_mean(calibrated_primary[calibrated_primary["study"] != study], "ncf")),
        }
        for study in sorted(calibrated_primary["study"].unique())
    }
    sensitivity["calibrated"]["by_condition"] = {
        condition: {
            "phantom_agreement": float(participant_mean(group, "phantom_agreement")),
            "ncf": float(participant_mean(group, "ncf")),
            "n": int(len(group)),
        }
        for condition, group in calibrated_primary.groupby("condition")
    }

    # Calibrated-basis counterparts of the two exclusions that were previously raw-only.
    # Both filter calibrated_primary rather than refitting the 5-fold held-out temperatures
    # on the reduced frame, so each excluded selection still contributed to the temperature
    # its retained neighbours were scored with; the temperatures are shared, per-source
    # scalars, so that contribution is diluted across thousands of selections (S1.15).
    sensitivity["calibrated"]["excluding_dynamic_stopping"] = excluding_dynamic_stopping(
        calibrated_primary, "DynBigram"
    )
    calibrated_excluded_fallback = calibrated_primary[
        ~calibrated_primary["calibration_fallback"].fillna(False)
    ]
    sensitivity["calibrated"]["excluding_pooled_calibration_sessions"] = {
        "phantom_agreement": float(
            participant_mean(calibrated_excluded_fallback, "phantom_agreement")
        ),
        "ncf": float(participant_mean(calibrated_excluded_fallback, "ncf")),
        "n": int(len(calibrated_excluded_fallback)),
    }

    sensitivity["temperature_scaling"] = {
        str(temperature): temperature_scaled_attribution(primary_raw, temperature)
        for temperature in (0.5, 1.0, 2.0)
    }
    sensitivity["attribution_method_comparison"] = attribution_method_comparison(primary_raw)

    real_ncf_mean, real_phantom_mean = _mean_ncf_phantom_agreement(primary_raw)
    permutable_raw = primary_prior_frame(permutable_selections(selections), priors, PRIMARY_PRIOR)
    real_ncf_subset, real_phantom_subset = _mean_ncf_phantom_agreement(permutable_raw)
    permutation_frames = stratified_permutation_frames(selections, priors, PRIMARY_PRIOR)
    n_permutable = int(len(permutation_frames[0])) if permutation_frames else 0
    sensitivity["stratified_permutation_negative_control"] = {
        "stratum_columns": ["position_in_phrase", "study", "context_length_bucket"],
        "n_selections_total": int(len(primary_raw)),
        "n_selections_permutable": n_permutable,
        "n_selections_structurally_unpermutable": int(len(primary_raw)) - n_permutable,
        "unpermutable_reason": (
            "every position_in_phrase == 0 selection has context_prefix == '' by "
            "construction (no prior selections yet), so no other same-position, "
            "same-study context value exists to draw from; one additional singleton "
            "stratum (n=1) has no other row to swap with"
        ),
        "real_context_full_set": {
            "n": int(len(primary_raw)),
            "ncf_mean": real_ncf_mean,
            "phantom_agreement_mean": real_phantom_mean,
            "note": "all selections, including the structurally unpermutable ones; NOT "
            "used for the exceedance p-value below, reported only for transparency",
        },
        "real_context_permutable_subset": {
            "n": int(len(permutable_raw)),
            "ncf_mean": real_ncf_subset,
            "phantom_agreement_mean": real_phantom_subset,
            "note": "restricted to the same rows the permutation null is built from; "
            "this is the value the exceedance p-value below tests against, since a valid "
            "permutation test requires the observed statistic and the null to be computed "
            "over identical rows",
        },
        "permuted_context_null": negative_control_attribution(
            permutation_frames,
            real_ncf=real_ncf_subset,
            real_phantom_agreement=real_phantom_subset,
        ),
    }

    # Calibrated-basis counterpart: same stratified-permutation negative control,
    # recomputed on calibrated fusion. The permutation null stays statistically valid
    # only if every row's temperature is fit from REAL (unpermuted) training-fold data
    # and the SAME fold-fitted temperature is then applied to both that row's real
    # evaluation and every permuted draw it appears in -- fitting a fresh temperature per
    # permutation draw would let the null "see" permuted labels during calibration and
    # invalidate the test. calibrated_attribution_frame is therefore called ONCE here, on
    # the real (unpermuted) permutable subset; permutation_frames (already built above
    # from real p_neural joined to permuted p_lm) are never touched by calibration
    # fitting, only by apply_temperature using each row's already-fit lookup value.
    permutable_selections_with_id = permutable_selections(selections, keep_row_id=True)
    permutable_raw_with_id = full_prior_frame(
        permutable_selections_with_id, priors, PRIMARY_PRIOR, extra_columns=("selection_row_id",)
    )
    calibrated_permutable = calibrated_attribution_frame(
        permutable_raw_with_id, group_column=CLUSTER, n_folds=5, seed=0
    )
    fold_temperature_lookup = calibrated_permutable[
        ["selection_row_id", "t_neural_used", "t_lm_used"]
    ]
    real_ncf_subset_calibrated = float(np.nanmean(calibrated_permutable["ncf"]))
    real_phantom_subset_calibrated = float(np.mean(calibrated_permutable["phantom_agreement"]))
    calibrated_permutation_frames = calibrated_stratified_permutation_frames(
        permutation_frames, fold_temperature_lookup
    )
    sensitivity["calibrated"]["stratified_permutation_negative_control"] = {
        "fold_temperatures_source": "real_context_permutable_subset",
        "fold_temperatures_note": (
            "t_neural/t_lm are fit once via 5-fold grouped calibration on the REAL "
            "(unpermuted) permutable subset only; the same fold-fitted temperature is "
            "then applied to a given selection's real evaluation and to every permuted "
            "draw involving it -- never refit per permutation draw and never fit on "
            "permuted data"
        ),
        "fold_temperatures": calibrated_permutable.attrs.get("fold_temperatures", []),
        "n_selections_permutable": int(len(calibrated_permutable)),
        "real_context_permutable_subset_calibrated": {
            "n": int(len(calibrated_permutable)),
            "ncf_mean": real_ncf_subset_calibrated,
            "phantom_agreement_mean": real_phantom_subset_calibrated,
            "note": "calibrated counterpart of "
            "sensitivity.stratified_permutation_negative_control."
            "real_context_permutable_subset; this is the value the calibrated "
            "exceedance p-value below tests against",
        },
        "permuted_context_null": negative_control_attribution(
            calibrated_permutation_frames,
            real_ncf=real_ncf_subset_calibrated,
            real_phantom_agreement=real_phantom_subset_calibrated,
        ),
    }

    emitted_context = pd.read_parquet(INTERMEDIATE / "emitted_context_attribution.parquet")
    sensitivity["emitted_context_primary_prior"] = {
        "intended_context": {
            "n": int(len(primary_raw)),
            "ncf_mean": real_ncf_mean,
            "phantom_agreement_mean": real_phantom_mean,
        },
        "emitted_context": emitted_context_summary(emitted_context),
    }

    # Paired, identically-weighted, identically-calibrated counterpart of the block above
    # (round-2 review: the two rows there are not on the same statistical footing --
    # unweighted mean vs participant-weighted bootstrap, both on raw uncalibrated fusion).
    # Primary: the emitted arm is scored under the intended arm's already-fit per-row
    # temperature (t_neural_used/t_lm_used, from calibrated_primary -- Task 9), not
    # refit independently, so the paired difference holds the calibration transformation
    # fixed across arms and isolates the context-change contrast alone (fix round 1:
    # independent refitting was found to let the calibrator partially absorb the very
    # emitted-context-drift effect under study, attenuating the NCF gap ~3x).
    pair_key = ["relative_path", "trial_number"]
    intended_frame = calibrated_primary[[*pair_key, CLUSTER, "ncf", "phantom_agreement"]]
    emitted_raw = emitted_context_raw_frame(emitted_context, selections, pair_key)
    temperature_lookup = calibrated_primary[[*pair_key, "t_neural_used", "t_lm_used"]]
    emitted_frame_shared_temperature = emitted_context_shared_temperature_frame(
        emitted_raw, temperature_lookup, pair_key
    )
    sensitivity["emitted_context_paired_calibrated"] = emitted_context_paired_calibrated_comparison(
        intended_frame, emitted_frame_shared_temperature, pair_key=pair_key
    )

    # Sensitivity comparison, retained (not deleted) per this plan's global convention of
    # relabeling rather than discarding an alternate-but-defensible analysis: the emitted
    # arm independently refits its own held-out temperature instead of reusing the
    # intended arm's. "Each arm optimally calibrated" is a coherent estimand, just the
    # less conservative one for isolating a paired contrast -- see
    # emitted_context_shared_temperature_frame's docstring.
    calibrated_emitted_independent = cast_binary_columns(calibrated_attribution_frame(
        emitted_raw, group_column=CLUSTER, n_folds=5, seed=0,
    ))
    emitted_frame_independent = calibrated_emitted_independent[[*pair_key, "ncf", "phantom_agreement"]]
    sensitivity["emitted_context_paired_calibrated_independently_recalibrated"] = {
        **emitted_context_paired_calibrated_comparison(
            intended_frame, emitted_frame_independent, pair_key=pair_key
        ),
        "note": (
            "Sensitivity comparison, not the primary estimate. Independently refits a "
            "held-out temperature on the emitted arm instead of reusing the intended "
            "arm's already-fit temperature (see the primary "
            "sensitivity.emitted_context_paired_calibrated, which shares temperatures "
            "across both arms). Refitting to the emitted context's own probability shape "
            "lets the calibrator partially absorb the context-drift effect under study, "
            "which attenuates the NCF gap roughly threefold relative to the "
            "shared-temperature primary; phantom agreement is not materially affected."
        ),
    }

    sensitivity["excluding_dynamic_stopping"] = excluding_dynamic_stopping(primary, "DynBigram")
    digest["sensitivity"] = sensitivity

    (OUTPUT / "stats_digest.json").write_text(json.dumps(digest, indent=2, default=float))
    print(json.dumps(digest, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
