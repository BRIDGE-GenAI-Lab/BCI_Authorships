"""Figure 2 is drawn entirely on the calibrated-fusion basis, with no raw-basis element left.

Panel a in full, and phantom agreement's row of panel b, used to be drawn on raw, uncalibrated
fusion while only the neural contribution fraction's row of panel b was calibrated. The mix was
disclosed rather than resolved, and it outlived the Results text's own move to calibrated-primary
reporting, so figure and text printed different numbers for the same quantities (raw OR 1.18 per
character against calibrated 1.19; raw OR 0.042 per AUC unit against calibrated 0.030). Task 4 of
the editorial revision round moved the whole figure onto the basis the Results lead with.

The guard is deliberately two-sided at every point. Asserting only that a drawn value equals its
calibrated digest key would pass for a quantity whose two bases happen to agree; each check below
therefore also asserts the drawn value is *not* the raw counterpart, so a silent revert to
`secondary[...]` in place of `secondary["calibrated"][...]` fails rather than passes. The two
quantities that genuinely are basis-independent (`train_auc`, and the tertile `n` counts derived
from it) are checked for equality instead, and named as such.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402

import make_figures as mf  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
NEEDED = [
    OUTPUT / "stats_digest.json",
    OUTPUT / "intermediate" / "attribution.parquet",
    OUTPUT / "intermediate" / "selections.parquet",
    OUTPUT / "intermediate" / "priors.parquet",
]
TERTILES = ("low", "mid", "high")


def _skip_if_missing():
    if any(not path.exists() for path in NEEDED):
        pytest.skip("the analysis outputs Figure 2 is built from have not been written")


@pytest.fixture(scope="module")
def digest():
    _skip_if_missing()
    return mf.load()[0]


@pytest.fixture(scope="module")
def raw_primary():
    """The raw, uncalibrated frame the figure used to be drawn from, rebuilt here purely so the
    tests can assert the figure is no longer drawn from it."""
    _skip_if_missing()
    _digest, attribution = mf.load()
    frame = mf.cast_binary_columns(attribution)
    return frame[
        (frame["beta"] == mf.PRIMARY_BETA) & (frame["prior_model"] == mf.PRIMARY_PRIOR)
    ].copy()


@pytest.fixture(scope="module")
def figure():
    _skip_if_missing()
    captured = {}
    original = mf.save
    mf.save = lambda fig, path: captured.setdefault("fig", fig)
    try:
        mf.apply_style()
        digest, attribution = mf.load()
        mf.figure2(digest, attribution)
    finally:
        mf.save = original
    fig = captured["fig"]
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _positions(container):
    return sorted((key for key in container if key.isdigit()), key=int)


def _rows(figure):
    """Panel a and panel b's phantom-agreement and NCF axes, keyed by (panel, row). The four
    share two y-limit pairs, so the panels are told apart by the session scatter, which only
    panel b draws: `axis.collections` alone cannot do it, because `ax.errorbar`'s bars are
    LineCollections and panel a's rows carry those from `_estimate_points`.
    """
    found = {}
    for axis in figure.axes:
        limits = tuple(axis.get_ylim())
        if limits == mf.PHANTOM_YLIM:
            row = "phantom"
        elif limits == mf.NCF_YLIM:
            row = "ncf"
        else:
            continue
        panel = "b" if any(isinstance(c, PathCollection) for c in axis.collections) else "a"
        found[(panel, row)] = axis
    assert set(found) == {("a", "phantom"), ("a", "ncf"), ("b", "phantom"), ("b", "ncf")}
    return found


def _estimate_series(axis, color):
    """The y values of the point estimates `_estimate_points` drew: the markers of the one
    errorbar container on this axis in `color`."""
    containers = [c for c in axis.containers if c.lines and c.lines[0].get_color() == color]
    assert len(containers) == 1
    return np.asarray(containers[0].lines[0].get_ydata(), dtype=float)


def _annotations(axis):
    return " | ".join(text.get_text() for text in axis.texts)


# --------------------------------------------------------------------------- panel a estimates


def test_panel_a_phantom_points_are_the_calibrated_by_position_estimates(figure, digest):
    axis = _rows(figure)[("a", "phantom")]
    drawn = _estimate_series(axis, mf.PHANTOM_COLOR)
    secondary = digest["secondary"]
    calibrated = secondary["calibrated"]["by_position_in_word"]
    raw = secondary["by_position_in_word"]
    keys = _positions(calibrated)
    expected = 100 * np.array([calibrated[k]["phantom_agreement"]["estimate"] for k in keys])
    rejected = 100 * np.array([raw[k]["phantom_agreement"]["estimate"] for k in keys])
    np.testing.assert_allclose(drawn, expected, atol=1e-9)
    assert not np.allclose(drawn, rejected, atol=1e-4)


def test_panel_a_ncf_points_are_the_calibrated_by_position_estimates(figure, digest):
    axis = _rows(figure)[("a", "ncf")]
    drawn = _estimate_series(axis, mf.NCF_COLOR)
    secondary = digest["secondary"]
    calibrated = secondary["calibrated"]["by_position_in_word"]
    raw = secondary["by_position_in_word"]
    keys = _positions(calibrated)
    expected = np.array([calibrated[k]["ncf"]["estimate"] for k in keys])
    rejected = np.array([raw[k]["ncf"]["estimate"] for k in keys])
    np.testing.assert_allclose(drawn, expected, atol=1e-9)
    assert not np.allclose(drawn, rejected, atol=1e-4)


def test_overall_phantom_reference_line_is_the_calibrated_primary_estimate(figure, digest):
    """The dashed gray rule on both phantom-agreement rows. It used to read the raw
    `sensitivity.uncalibrated_beta_1` co-primary estimate (3.8%) while Figure 1 and the
    Abstract led with the calibrated one (4.4%), so the same headline number appeared twice in
    the paper with two values."""
    expected = 100 * digest["calibrated_primary_analysis"]["phantom_agreement"]["estimate"]
    rejected = 100 * (
        digest["sensitivity"]["uncalibrated_beta_1"]["coprimary_2_phantom_agreement"]["estimate"]
    )
    for panel in ("a", "b"):
        axis = _rows(figure)[(panel, "phantom")]
        rules = [
            line for line in axis.get_lines()
            if len(line.get_ydata()) == 2
            and np.isclose(line.get_ydata()[0], line.get_ydata()[1])
            and line.get_color() == mf.CONTEXT
        ]
        assert len(rules) == 1, f"panel {panel} should carry exactly one overall reference rule"
        assert np.isclose(rules[0].get_ydata()[0], expected, atol=1e-9)
        assert not np.isclose(rules[0].get_ydata()[0], rejected, atol=1e-3)
    label = _annotations(_rows(figure)[("a", "phantom")])
    assert f"overall {expected:.1f}%" in label


# ---------------------------------------------------------------------------- panel b estimates


def test_panel_b_phantom_scatter_is_the_calibrated_per_session_means(figure, raw_primary):
    axis = _rows(figure)[("b", "phantom")]
    scatter = next(c for c in axis.collections if isinstance(c, PathCollection))
    drawn = np.sort(np.asarray(scatter.get_offsets())[:, 1])
    calibrated_frame = mf._calibrated_quality_frame()
    expected = 100 * np.sort(
        calibrated_frame.groupby([mf.CLUSTER, "session_id"])["phantom_agreement"]
        .mean().to_numpy()
    )
    rejected = 100 * np.sort(
        raw_primary.groupby([mf.CLUSTER, "session_id"])["phantom_agreement"].mean().to_numpy()
    )
    np.testing.assert_allclose(drawn, expected, atol=1e-8)
    assert not np.allclose(drawn, rejected, atol=1e-6)


def test_panel_b_phantom_curve_is_the_calibrated_logistic_model(figure, digest):
    """Phantom agreement's curve and its annotated odds ratio come from one logistic GEE, and
    always did; what changed is which basis that single model is fit on."""
    calibrated_frame = mf._calibrated_quality_frame()
    fits = mf._quality_trend_fits(digest, calibrated_frame)
    models = digest["secondary"]
    for source, terms in (
        ("calibrated", models["calibrated"]["models"]["phantom_by_decoder_quality"]["terms"]),
        ("raw", models["models"]["phantom_by_decoder_quality"]["terms"]),
    ):
        curve = 1.0 / (1.0 + np.exp(
            -(terms["Intercept"]["log_odds"] + terms["train_auc"]["log_odds"] * fits["grid"])
        ))
        if source == "calibrated":
            np.testing.assert_allclose(fits["phantom"][0], curve, atol=1e-8)
        else:
            assert not np.allclose(fits["phantom"][0], curve, atol=1e-3)

    axis = _rows(figure)[("b", "phantom")]
    lines = [line for line in axis.get_lines() if len(line.get_ydata()) == len(fits["grid"])]
    assert len(lines) == 1
    np.testing.assert_allclose(
        np.asarray(lines[0].get_ydata(), dtype=float), 100 * fits["phantom"][0], atol=1e-8
    )


def test_panel_b_phantom_tertile_ticks_are_the_calibrated_estimates(figure, digest):
    axis = _rows(figure)[("b", "phantom")]
    ticks = sorted(
        line.get_ydata().mean() for line in axis.get_lines()
        if len(line.get_xdata()) == 2
        and np.isclose(line.get_xdata()[0], line.get_xdata()[1])
        and line.get_color() == mf.PHANTOM_COLOR
        and np.isclose(line.get_linewidth(), 1.6)
    )
    secondary = digest["secondary"]
    expected = sorted(
        100 * secondary["calibrated"]["by_decoder_quality"][t]["phantom_agreement"]["estimate"]
        for t in TERTILES
    )
    rejected = sorted(
        100 * secondary["by_decoder_quality"][t]["phantom_agreement"]["estimate"]
        for t in TERTILES
    )
    assert len(ticks) == 3
    np.testing.assert_allclose(ticks, expected, atol=1e-6)
    assert not np.allclose(ticks, rejected, atol=1e-3)


# ------------------------------------------------------------------------------- annotations


def test_every_annotated_effect_estimate_is_the_calibrated_one(figure, digest):
    """The four in-panel numbers, each checked against its calibrated digest value in the
    exact rendered format, and against the raw value it must no longer show."""
    secondary = digest["secondary"]
    rows = _rows(figure)

    calibrated_or = secondary["calibrated"]["models"]["phantom_by_position"]["terms"]
    raw_or = secondary["models"]["phantom_by_position"]["terms"]
    text = _annotations(rows[("a", "phantom")])
    assert f"OR {calibrated_or['position_in_phrase']['odds_ratio']:.2f}/character" in text
    assert f"OR {raw_or['position_in_phrase']['odds_ratio']:.2f}/character" not in text

    # Panel a's NCF row annotates the bounded fractional-logit fit's own predicted contrast,
    # parallel to panel b's, not the linear mixed model's per-character slope that model's
    # demotion to a sensitivity comparison left behind.
    position_contrast = (
        secondary["calibrated"]["models"]["ncf_by_position_bounded"]["predicted_contrast"]
    )
    text = _annotations(rows[("a", "ncf")])
    assert f"Predicted {position_contrast['difference']:+.3f} across positions" in text
    for models in (secondary["calibrated"]["models"], secondary["models"]):
        slope = models["ncf_by_position"]["terms"]["position_in_phrase"]["coefficient"]
        assert f"Slope {slope:.4f}/character" not in text

    calibrated_quality = secondary["calibrated"]["models"]["phantom_by_decoder_quality"]["terms"]
    raw_quality = secondary["models"]["phantom_by_decoder_quality"]["terms"]
    text = _annotations(rows[("b", "phantom")])
    assert f"OR {calibrated_quality['train_auc']['odds_ratio']:.3f}/AUC unit" in text
    assert f"OR {raw_quality['train_auc']['odds_ratio']:.3f}/AUC unit" not in text

    contrast = (
        secondary["calibrated"]["models"]["ncf_by_decoder_quality_bounded"]["predicted_contrast"]
    )
    text = _annotations(rows[("b", "ncf")])
    assert f"Predicted {contrast['difference']:+.3f} between tertile medians" in text


def test_no_annotation_claims_a_multiplicity_adjusted_p(figure):
    """Task 4's resolution of the two-P-types asymmetry Task 2's review handed over. Two of the
    four annotated models belong to the five-member Benjamini-Hochberg family and the two
    bounded fractional-logit contrasts do not, so labelling some "adjusted P" and leaving the
    others bare put two kinds of P in one panel with nothing on the figure to separate them.
    Every annotation now shows its own model's unadjusted P and the caption says so once.
    """
    rows = _rows(figure)
    for key in (("a", "phantom"), ("a", "ncf"), ("b", "phantom"), ("b", "ncf")):
        assert "adjusted" not in _annotations(rows[key]).lower()


def test_annotated_p_values_are_each_models_own_unadjusted_p(figure, digest):
    secondary = digest["secondary"]
    rows = _rows(figure)
    for key, terms, term in (
        (("a", "phantom"),
         secondary["calibrated"]["models"]["phantom_by_position"]["terms"], "position_in_phrase"),
        (("b", "phantom"),
         secondary["calibrated"]["models"]["phantom_by_decoder_quality"]["terms"], "train_auc"),
    ):
        assert mf._p_text(terms[term]["p_value"]) in _annotations(rows[key])
    for key, name in (
        (("a", "ncf"), "ncf_by_position_bounded"),
        (("b", "ncf"), "ncf_by_decoder_quality_bounded"),
    ):
        contrast = secondary["calibrated"]["models"][name]["predicted_contrast"]
        assert mf._p_text(contrast["p_value"]) in _annotations(rows[key])


# --------------------------------------------------------- the two basis-independent quantities


def test_tile_strip_counts_come_from_the_calibrated_position_container(figure, digest):
    """These counts are identical on both bases (calibration does not move a selection between
    positions), so equality alone cannot tell which key they were read from. Pin the equality
    as the standing claim, and read the counts from the calibrated container in the figure."""
    secondary = digest["secondary"]
    calibrated = secondary["calibrated"]["by_position_in_word"]
    raw = secondary["by_position_in_word"]
    keys = _positions(calibrated)
    assert [calibrated[k]["n"] for k in keys] == [raw[k]["n"] for k in keys]
    drawn = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
    for key in keys:
        assert f"n = {calibrated[key]['n']:,}" in drawn


def test_figure2_raises_if_the_tertile_cut_stops_being_basis_independent(digest):
    """`figure2()` keeps building the raw frame from its `attribution` argument for one check:
    that both bases carry the same calibration-decoder AUCs, which is what lets the tertile
    boundaries and the strip's n counts stay put while every estimate moved. Perturbing one AUC
    must stop the render rather than produce a figure whose tertile rules and counts describe a
    different partition than the estimates above them.
    """
    _digest, attribution = mf.load()
    tampered = attribution.copy()
    target = tampered.index[
        (tampered["beta"] == mf.PRIMARY_BETA) & (tampered["prior_model"] == mf.PRIMARY_PRIOR)
    ][0]
    tampered.loc[target, "train_auc"] = 0.123456

    original = mf.save
    mf.save = lambda fig, path: None
    try:
        with pytest.raises(RuntimeError, match="same calibration-decoder AUCs"):
            mf.figure2(digest, tampered)
    finally:
        mf.save = original
        plt.close("all")
