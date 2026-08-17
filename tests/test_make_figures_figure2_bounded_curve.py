"""Figure 2 panel b's neural contribution fraction curve must never predict outside [0, 1],
and (Task 12b) its session scatter, tertile markers, and rug must be drawn on the same
calibrated-fusion basis as that curve.

The panel used to draw a linear mixed-model fit of NCF against calibration-decoder AUC, which
crossed 1.0 inside the observed AUC range (visible in the rendered figure, not just in principle).
It now draws Task 11's fractional-logit fit instead, bounded by construction, computed from
`secondary.calibrated.models.ncf_by_decoder_quality_bounded` -- the tests here check that
property directly on the curve the figure actually renders, and confirm the drawn curve is the
digest-reported model rather than some other fit that happens to also be bounded.

Task 12 wired that curve in but left the points it was drawn against (session circles, tertile
markers, rug) on the raw, uncalibrated basis -- a real mismatch, disclosed in the figure's
footer at the time. Task 12b closes that gap: the tests added at the bottom of this file pin
the new `secondary.calibrated.by_decoder_quality` digest key's tertile sizes against the raw
key's (they must match exactly -- same underlying selections, same train_auc column, same
tertile cut) and confirm the figure's NCF scatter is the calibrated per-session values, not
the raw ones it replaced.

Task 4 of the editorial revision round moved the rest of the figure (all of panel a, and
phantom agreement's row of panel b) onto that same calibrated basis, so nothing here is drawn
on raw fusion any more. `tests/test_make_figures_figure2_calibrated_basis.py` is the guard for
that; this file keeps only the bounded-curve properties, which are unchanged by it.
"""

import re
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


def _skip_if_missing():
    if any(not path.exists() for path in NEEDED):
        pytest.skip("the analysis outputs Figure 2 is built from have not been written")


@pytest.fixture(scope="module")
def digest_and_primary():
    _skip_if_missing()
    digest, attribution = mf.load()
    primary = mf.cast_binary_columns(attribution)
    primary = primary[
        (primary["beta"] == mf.PRIMARY_BETA) & (primary["prior_model"] == mf.PRIMARY_PRIOR)
    ].copy()
    return digest, primary


@pytest.fixture(scope="module")
def calibrated_frame():
    _skip_if_missing()
    return mf._calibrated_quality_frame()


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


def test_quality_trend_fits_ncf_curve_never_exceeds_one(digest_and_primary, calibrated_frame):
    digest, _primary = digest_and_primary
    fits = mf._quality_trend_fits(digest, calibrated_frame)
    curve, low, high = fits["ncf"]
    assert curve.min() >= 0.0 and curve.max() <= 1.0
    assert low.min() >= 0.0 and high.max() <= 1.0


def test_quality_trend_fits_ncf_curve_matches_digest_bounded_model(
    digest_and_primary, calibrated_frame
):
    digest, _primary = digest_and_primary
    fits = mf._quality_trend_fits(digest, calibrated_frame)
    terms = digest["secondary"]["calibrated"]["models"]["ncf_by_decoder_quality_bounded"]["terms"]
    intercept = terms["Intercept"]["log_odds"]
    slope = terms["train_auc"]["log_odds"]
    expected = 1.0 / (1.0 + np.exp(-(intercept + slope * fits["grid"])))
    np.testing.assert_allclose(fits["ncf"][0], expected, atol=1e-8)


def test_quality_trend_fits_ncf_curve_is_not_the_old_linear_model(
    digest_and_primary, calibrated_frame
):
    """Guards against a regression back to the unbounded fit: the old linear model's slope
    and intercept, evaluated on the same grid, differ from the bounded curve almost
    everywhere, so an accidental revert would fail this even though both curves are smooth
    monotonic functions of AUC."""
    digest, _primary = digest_and_primary
    fits = mf._quality_trend_fits(digest, calibrated_frame)
    linear_terms = digest["secondary"]["models"]["ncf_by_decoder_quality"]["terms"]
    linear_curve = (
        linear_terms["Intercept"]["coefficient"]
        + linear_terms["train_auc"]["coefficient"] * fits["grid"]
    )
    assert not np.allclose(fits["ncf"][0], linear_curve, atol=1e-3)


def test_figure2_ncf_curve_artist_never_exceeds_one(figure):
    """End-to-end: the actual Line2D panel b draws for the NCF row, not just the values
    `_quality_trend_fits` returns, stays within [0, 1]. Panel b's NCF axis carries no
    ylabel of its own (it shares panel a's), so it is identified by its y-limits instead,
    which are set once to the module-level NCF_YLIM shared by both panels' NCF row.
    """
    ncf_axes = [axis for axis in figure.axes if tuple(axis.get_ylim()) == mf.NCF_YLIM]
    assert len(ncf_axes) == 2, "expected exactly panel a and panel b's NCF axes"
    for ncf_axis in ncf_axes:
        lines = [line for line in ncf_axis.get_lines() if len(line.get_ydata()) > 2]
        if not lines:
            continue  # panel a has no fitted curve, only point-to-point markers
        curve_line = max(lines, key=lambda line: len(line.get_xdata()))
        ydata = np.asarray(curve_line.get_ydata(), dtype=float)
        assert ydata.min() >= 0.0 and ydata.max() <= 1.0


def test_figure2_note_states_one_uniform_calibrated_basis(figure):
    """The paragraph-length footer that used to carry the basis disclosure is gone (figure
    decluttering); a compact tag in panel b's italic note replaced it, reading "phantom raw,
    NCF calibrated" for as long as panel b really did mix two bases within itself. Task 4
    ended the mix, so that tag is now a false statement about the figure and must not come
    back in any form: the note states the single calibrated basis and names no raw component.

    The "raw" check is word-bounded on purpose. A bare substring test also matches "drawn" and
    "drawing", which are ordinary words for describing a figure, so Task 9's legend trim could
    have failed this test over a rewording that has nothing to do with basis.
    """
    panel_b_note = next(
        text.get_text() for text in figure.texts
        if "one circle and one rug tick per session" in text.get_text()
    )
    assert "calibrated" in panel_b_note
    assert not re.search(r"\braw\b", panel_b_note)

    all_texts = " ".join(text.get_text() for text in figure.texts)
    normalized = all_texts.replace("-\n", "-").replace("\n", " ")
    assert "different basis" not in normalized
    assert "phantom raw" not in normalized


def test_calibrated_by_decoder_quality_n_matches_raw(digest_and_primary):
    """Task 12b Step 1's consistency claim, pinned as a permanent regression guard: the new
    calibrated-basis tertile key partitions the same selections into the same three tertiles
    the raw key does, so their `n` values must match exactly."""
    digest, _primary = digest_and_primary
    raw = digest["secondary"]["by_decoder_quality"]
    calibrated = digest["secondary"]["calibrated"]["by_decoder_quality"]
    for tertile in ("low", "mid", "high"):
        assert calibrated[tertile]["n"] == raw[tertile]["n"]


def test_figure2_panel_b_ncf_scatter_is_calibrated_not_raw(figure, digest_and_primary):
    """End-to-end regression guard: panel b's NCF scatter must be the per-session calibrated
    NCF means, not the raw per-session means it replaced. Identifies panel b's NCF axis (as
    opposed to panel a's, which shares the same y-limits but never calls `.scatter`) by which
    of the two NCF-ylim axes carries a PathCollection -- `axis.collections` alone is not
    enough to tell them apart, since `ax.errorbar`'s error bars are also LineCollections and
    panel a's NCF axis has those from `_estimate_points`.
    """
    _digest, primary = digest_and_primary
    ncf_axes = [axis for axis in figure.axes if tuple(axis.get_ylim()) == mf.NCF_YLIM]
    scatter_axes = [
        axis for axis in ncf_axes
        if any(isinstance(c, PathCollection) for c in axis.collections)
    ]
    assert len(scatter_axes) == 1, "expected exactly one NCF axis with a session scatter"
    scatter = next(c for c in scatter_axes[0].collections if isinstance(c, PathCollection))
    drawn_y = np.sort(np.asarray(scatter.get_offsets())[:, 1])

    calibrated_frame = mf._calibrated_quality_frame()
    calibrated_sessions_ncf = np.sort(
        calibrated_frame.groupby([mf.CLUSTER, "session_id"])["ncf"].mean().to_numpy()
    )
    raw_sessions_ncf = np.sort(
        primary.groupby([mf.CLUSTER, "session_id"])["ncf"].mean().to_numpy()
    )
    np.testing.assert_allclose(drawn_y, calibrated_sessions_ncf, atol=1e-8)
    assert not np.allclose(drawn_y, raw_sessions_ncf, atol=1e-6)


def _all_text(figure):
    """Every rendered string in the figure, figure-level (titles/notes/footer, drawn with
    `fig.text`) and axes-level (callouts and annotations, drawn with `ax.text`/`ax.annotate`)."""
    texts = [t.get_text() for t in figure.texts]
    for axis in figure.axes:
        texts.extend(t.get_text() for t in axis.texts)
    return " ".join(t for t in texts if t and t.strip())


def test_figure2_no_stale_already_emitted_or_tertile_callout_wording(figure):
    """Task 22 fixed the panel a tile label's "characters already emitted" wording to match
    Figure 1's "intended preceding characters" (both describe the same oracle-context primary
    analysis basis), and removed the "tertile estimate, 95% CI" leader-line annotation along
    with the open-square markers it pointed at.
    """
    texts = _all_text(figure)
    assert "already emitted" not in texts
    assert "tertile estimate" not in texts
    assert "intended preceding characters" in texts


def test_figure2_panel_titles_match_current_framing(figure):
    """Task 22 retitled both panel headers; pinned so a future edit can't silently drift the
    rendered title away from what the caption in assemble_manuscript.py describes."""
    texts = _all_text(figure)
    assert "LLM influence increases with available linguistic context" in texts
    assert "Calibration decoder discriminability" in texts


def _vertical_ticks(axis, color, linewidth):
    """Line2D artists that are a single vertical segment (two points, equal x) in `color` at
    `linewidth`, the shape `_tertile_ticks` draws. Panel b's rug (`_rug`) draws hundreds of
    2-point vertical segments in this same color too, so linewidth is the discriminator: the
    rug is 0.7 pt at alpha 0.45, the tertile ticks are 1.6 pt and fully opaque.
    """
    ticks = []
    for line in axis.get_lines():
        xdata = line.get_xdata()
        if (len(xdata) == 2 and np.isclose(xdata[0], xdata[1])
                and line.get_color() == color
                and np.isclose(line.get_linewidth(), linewidth)):
            ticks.append(line)
    return ticks


def test_figure2_tertile_markers_are_minimal_unlabeled_ticks(figure):
    """Task 22 replaced the open-square-with-whiskers tertile markers (fmt="s" errorbar) with
    three minimal, unlabeled vertical ticks per row -- declutters panel b, which already
    carries a fitted curve, a confidence band, ~115 session circles, and a rug. This confirms
    the replacement actually happened on the rendered figure, not just in the helper function:
    exactly one phantom-agreement axis and one NCF axis in panel b carry 3 such ticks, and no
    axis in the figure still carries a square-marker artist.
    """
    ncf_axes = [axis for axis in figure.axes if tuple(axis.get_ylim()) == mf.NCF_YLIM]
    phantom_axes = [axis for axis in figure.axes if tuple(axis.get_ylim()) == mf.PHANTOM_YLIM]

    ncf_tick_axes = [a for a in ncf_axes if len(_vertical_ticks(a, mf.NCF_COLOR, 1.6)) == 3]
    phantom_tick_axes = [
        a for a in phantom_axes if len(_vertical_ticks(a, mf.PHANTOM_COLOR, 1.6)) == 3
    ]
    assert len(ncf_tick_axes) == 1, "expected exactly one NCF axis with the 3 tertile ticks"
    assert len(phantom_tick_axes) == 1, (
        "expected exactly one phantom-agreement axis with the 3 tertile ticks"
    )

    for axis in ncf_axes + phantom_axes:
        for line in axis.get_lines():
            assert line.get_marker() != "s", "an open-square tertile marker survived Task 22"
