"""Figure 1 and Figure 3 at the size the journal actually prints them.

Both were drawn wider than the page for a while - 205 mm and 227 mm against npj Digital
Medicine's 183 mm double column - and a journal reproducing an oversized figure scales it down,
taking every glyph with it. That put their smallest type at 5.00 and 5.01 pt on the page,
exactly on the floor of the 5-7 pt band Nature Portfolio asks for at final size, and nothing in
the suite would have noticed the next canvas change pushing it under.

The measurement here is the one that found that. save() exports with bbox_inches="tight", so
the declared figsize is not the published size: what ships is the tight bounding box of what was
actually drawn, plus save()'s own 0.08 inch pad on each side. Everything below is taken from
that, and cross-checked against the committed PNG's own pixel dimensions at its 600 dpi.

Figure4 was added to these checks once its own canvas allocation changed for the first time
(panel d's row grown to close a dead-space gap): it never had the width overflow the other two
did, but nothing was guarding it against picking one up later either.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import make_figures  # noqa: E402
from make_figures import apply_style, figure1, figure3, figure4  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
DIGEST = PROJECT / "output" / "stats_digest.json"
FIGURES = PROJECT / "output" / "figures"

MM_PER_INCH = 25.4
SAVE_PAD_IN = 0.08          # what hochberg_kit.save passes as pad_inches
DOUBLE_COLUMN_MM = 183.0    # Nature Portfolio's full double-column width
FINAL_SIZE_FLOOR_PT = 5.0   # and the smallest type it accepts at final size
# The margin this file exists to hold. Not a journal number: the floor above is. 183 mm is the
# widest a double-column figure is printed, so a figure that only just clears 5 pt there has no
# room left if it is placed any narrower, and none of these two figures' hand-placed text can be
# re-tuned cheaply. Half a point of headroom is what the re-layout bought, and asserting it
# means a later canvas change fails here while there is still slack to absorb it, rather than
# passing silently on the way under the floor.
REQUIRED_MARGIN_PT = 0.5


def _undrawn_tick_labels(fig):
    """Tick labels matplotlib keeps but does not draw, because the tick is outside the view.

    They carry stale positions and put no ink on the page, so counting them would let a
    phantom 10 pt label vouch for a figure whose real smallest type is 6 pt.
    """
    dead = set()
    for axes in fig.axes:
        for axis in (axes.xaxis, axes.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    dead.update({id(tick.label1), id(tick.label2)})
    return dead


def _measure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    dead = _undrawn_tick_labels(fig)
    sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(Text)
        if text.get_visible() and text.get_text().strip() and id(text) not in dead
    ]
    width_mm = (tight.width + 2 * SAVE_PAD_IN) * MM_PER_INCH
    return {
        "width_mm": width_mm,
        "smallest_pt": min(sizes),
        # What the smallest type measures on the page once the figure is laid into a full
        # double column. Above 1.0 the figure is being enlarged, below it, shrunk.
        "smallest_pt_at_final_size": min(sizes) * DOUBLE_COLUMN_MM / width_mm,
    }


@pytest.fixture(scope="module")
def measured():
    if not DIGEST.exists():
        pytest.skip("run_stats has not written output/stats_digest.json")
    digest = json.loads(DIGEST.read_text())
    captured = {}

    def capture(fig, path_no_ext, dpi=600):
        captured[Path(path_no_ext).name] = _measure(fig)
        plt.close(fig)
        return f"{path_no_ext}.png"

    original = make_figures.save
    make_figures.save = capture
    try:
        # figure3 and figure4 do not call apply_style themselves; in the pipeline figure1 has
        # already done it, and without it every rc-driven size here would be matplotlib's 10 pt
        # default.
        apply_style()
        figure1(digest)
        figure3(digest)
        figure4(digest)
    finally:
        make_figures.save = original
    return captured


@pytest.mark.parametrize("figure", ["Figure1", "Figure3", "Figure4"])
def test_the_rendered_figure_fits_the_double_column_it_is_printed_in(measured, figure):
    width = measured[figure]["width_mm"]
    assert width <= DOUBLE_COLUMN_MM, (
        f"{figure} renders {width:.1f} mm wide, past the {DOUBLE_COLUMN_MM:.0f} mm double "
        "column, so the journal will scale it down and take every glyph on it with it"
    )


@pytest.mark.parametrize("figure", ["Figure1", "Figure3", "Figure4"])
def test_the_smallest_type_clears_the_final_size_floor_with_margin(measured, figure):
    final = measured[figure]["smallest_pt_at_final_size"]
    assert final >= FINAL_SIZE_FLOOR_PT + REQUIRED_MARGIN_PT, (
        f"{figure}'s smallest type measures {final:.2f} pt once the figure is laid into a "
        f"{DOUBLE_COLUMN_MM:.0f} mm column, which leaves under {REQUIRED_MARGIN_PT} pt over the "
        f"{FINAL_SIZE_FLOOR_PT:.0f} pt final-size floor. Widen no further; bring the canvas in "
        "or lift the type."
    )


@pytest.mark.parametrize("figure", ["Figure1", "Figure3", "Figure4"])
def test_the_committed_png_is_the_width_this_code_renders(measured, figure):
    """The figure in output/figures is the artifact that gets submitted, not the one this
    process just drew. If they have drifted apart, the measurements above are about a file
    nobody is going to publish."""
    png = FIGURES / f"{figure}.png"
    if not png.exists():
        pytest.skip(f"{png.name} has not been rendered")
    from PIL import Image

    with Image.open(png) as image:
        dpi = image.info.get("dpi", (600, 600))[0] or 600
        committed_mm = image.size[0] / dpi * MM_PER_INCH
    assert committed_mm == pytest.approx(measured[figure]["width_mm"], abs=0.5), (
        f"the committed {png.name} is {committed_mm:.1f} mm wide but this code now renders "
        f"{measured[figure]['width_mm']:.1f} mm; re-run scripts/make_figures.py"
    )


@pytest.fixture(scope="module")
def figure1_live():
    """Figure 1's own Figure object, left open (not swapped for a captured measurement dict
    and closed) so its patches and text artists can be inspected directly."""
    if not DIGEST.exists():
        pytest.skip("run_stats has not written output/stats_digest.json")
    digest = json.loads(DIGEST.read_text())
    captured = {}

    def capture(fig, path_no_ext, dpi=600):
        captured["fig"] = fig
        return f"{path_no_ext}.png"

    original = make_figures.save
    make_figures.save = capture
    try:
        apply_style()
        figure1(digest)
    finally:
        make_figures.save = original
    fig = captured["fig"]
    fig.canvas.draw()
    yield fig
    plt.close(fig)


BOX_REQUIRED_CLEARANCE_PT = 1.0  # Task 29b review F-1 measured 0.42 pt before this test existed


def _panel_d_axes(fig):
    """Panel d's estimate axis: the only inset in Figure 1 with no y ticks and a percentage
    x axis, which is enough to name it without reaching back into figure1()'s coordinates.

    Figure 1's insets are children of its schematic axis, not of the figure, so they are in
    `child_axes` and never appear in `fig.axes`.
    """
    found = [
        axes for axes in fig.axes[0].child_axes
        if not len(axes.get_yticks())
        and [label.get_text() for label in axes.get_xticklabels()][:1] == ["0%"]
    ]
    assert len(found) == 1, f"expected 1 panel d estimate axis, found {len(found)}"
    return found[0]


def test_panel_d_draws_both_co_primary_estimates_as_points_with_intervals(figure1_live):
    """Panel d used to be two rounded cards printing a percentage, a CI line and a sentence:
    styled text, no plotted data, which is what the author asked to see replaced by a chart.
    It is now one dot and one 95% interval bar per co-primary measure on a shared percentage
    axis. This asserts the marks themselves exist - two dots, two error bars, both intervals
    bracketing their own estimate - so a future edit cannot quietly return the panel to text.
    """
    d_axis = _panel_d_axes(figure1_live)

    dots = [line for line in d_axis.lines if line.get_marker() == "o"]
    assert len(dots) == 2, f"expected panel d's 2 estimate dots, found {len(dots)}"

    # errorbar() draws its bars as a LineCollection per container, one container per call.
    intervals = []
    for container in d_axis.containers:
        (estimate_x,), (_,) = container[0].get_data()
        segments = container.lines[2][0].get_segments()[0]
        intervals.append((float(segments[0][0]), float(estimate_x), float(segments[1][0])))
    assert len(intervals) == 2, f"expected 2 interval bars in panel d, found {len(intervals)}"
    for low, estimate, high in intervals:
        assert low < estimate < high, (
            f"panel d drew an interval [{low:.2f}, {high:.2f}] that does not bracket its own "
            f"estimate {estimate:.2f}; the complement's interval ends have to be swapped"
        )
    assert max(high for _, _, high in intervals) <= d_axis.get_xlim()[1], (
        "panel d's widest interval runs past its own axis, so the whisker is clipped"
    )


def test_panel_d_row_text_clears_the_marks_it_labels(figure1_live):
    """Each panel d row carries three hand-placed pieces of text in the schematic's own 0-1
    space - the measure's name, its headline percentage, and its interval - positioned by one
    row-centre number each, while the dot they belong to is placed inside an inset axis. Those
    two coordinate systems are only kept in register by arithmetic in figure1(), and every
    number in the row is interpolated from the digest, so a stats rerun changes the text's
    width but not its anchor. This measures the actual gap between a row's own text and the
    axis it sits beside, in points, rather than trusting that arithmetic.
    """
    ax = figure1_live.axes[0]
    renderer = figure1_live.canvas.get_renderer()
    dpi = figure1_live.dpi
    d_axis = _panel_d_axes(figure1_live)
    axis_bbox = d_axis.get_window_extent(renderer)

    # The row text is what sits at the estimate axis's own height but outside it: three pieces
    # per row (name, percentage, interval), plus the panel letter, which is at the upper row's
    # height and is measured here for the same reason the rows are.
    beside = []
    for text in ax.texts:
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        mid_y = (bbox.y0 + bbox.y1) / 2
        if axis_bbox.y0 <= mid_y <= axis_bbox.y1:
            beside.append((text, bbox))
    assert len(beside) == 7, (
        f"expected panel d's 6 row labels and its panel letter beside the estimate axis, "
        f"found {len(beside)}: " + ", ".join(repr(text.get_text()) for text, _ in beside)
    )

    for text, bbox in beside:
        if bbox.x1 <= axis_bbox.x0:
            clearance_pt = (axis_bbox.x0 - bbox.x1) / dpi * 72
        elif bbox.x0 >= axis_bbox.x1:
            clearance_pt = (bbox.x0 - axis_bbox.x1) / dpi * 72
        else:
            clearance_pt = -1.0
        assert clearance_pt >= BOX_REQUIRED_CLEARANCE_PT, (
            f"panel d's {text.get_text()!r} clears the estimate axis by {clearance_pt:.2f} pt, "
            f"under the {BOX_REQUIRED_CLEARANCE_PT} pt this test requires; move the row's own "
            "x anchor in figure1(), or narrow the axis"
        )


def test_every_figure_1_box_contains_the_text_it_encloses(figure1_live):
    """Figure 1 draws its prose inside hand-sized boxes, and the sizes were pinned while the
    text inside them is interpolated and wrapped at render time. Panel c's definition box was
    pinned at 0.180 tall and its phantom-agreement caption's last line ("classified") fell
    3.35 pt below its own border, which no test saw: the card test this replaced measured one
    edge of one box. This measures all four edges of every box in the schematic.
    """
    ax = figure1_live.axes[0]
    renderer = figure1_live.canvas.get_renderer()
    dpi = figure1_live.dpi

    boxes = [
        patch for patch in ax.patches
        if isinstance(patch, FancyBboxPatch)
        and patch.get_window_extent(renderer).width > 20
        and patch.get_window_extent(renderer).height > 20
    ]
    assert boxes, "found no boxes in Figure 1's schematic axis"

    for box in boxes:
        box_bbox = box.get_window_extent(renderer)
        for text in ax.texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            bbox = text.get_window_extent(renderer)
            centre = ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)
            if not (box_bbox.x0 <= centre[0] <= box_bbox.x1
                    and box_bbox.y0 <= centre[1] <= box_bbox.y1):
                continue  # this text belongs to a different box
            clearance_pt = min(
                bbox.x0 - box_bbox.x0, box_bbox.x1 - bbox.x1,
                bbox.y0 - box_bbox.y0, box_bbox.y1 - bbox.y1,
            ) / dpi * 72
            assert clearance_pt >= BOX_REQUIRED_CLEARANCE_PT, (
                f"{text.get_text()!r} clears its own box border by {clearance_pt:.2f} pt, under "
                f"the {BOX_REQUIRED_CLEARANCE_PT} pt this test requires; give the box more room "
                "or narrow the character budget of the _flow call that wrote the text"
            )


def test_no_stale_raw_basis_or_already_emitted_wording(figure1_live):
    """Task 21 moved panel e's closed-loop and parameter-count cards from raw, uncalibrated
    fusion onto the calibrated basis the rest of the figure and its own footer declare, and
    fixed the panel a context-tile label's "characters already emitted" wording, which
    described the emitted-context arm rather than the oracle-context arm this figure's
    primary analysis actually plots. Both are easy to reintroduce on a future card edit or
    digest-key rename, and neither is caught by the clearance or width tests above.
    """
    texts = " ".join(
        text.get_text() for text in figure1_live.axes[0].texts
        if text.get_visible() and text.get_text().strip()
    )
    assert "already emitted" not in texts
    assert "under raw, uncalibrated fusion" not in texts
    assert "log parameters, raw fusion" not in texts


