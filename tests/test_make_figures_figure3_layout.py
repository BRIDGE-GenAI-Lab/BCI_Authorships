"""Figure 3's two geometric promises: no marker is hidden, and no leader points at the wrong one.

Both have been broken by hand-placed layout more than once - a leader that ran tangent to
OLMo-2 7B, its replacement that stopped inside the filled Llama-3.1-8B-Instruct marker, and a
prior whose marker was covered completely by a same-sized one drawn on top of it. The measures
here are the ones that caught those defects, taken from the artists the figure actually draws
rather than from the values it was asked to draw.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from assemble_manuscript import TABLES  # noqa: E402
from make_figures import (  # noqa: E402
    FIGURE3_A_INSET_RECT,
    FIGURE3_A_INSET_ZOOM,
    FIGURE3_A_XLIM,
    FIGURE3_A_YLIM,
    FIGURE3_B_XLIM,
    FIGURE3_B_YLIM,
    FIGURE3_LEADER_CLEARANCE_PT,
    FIGURE3_MARKER_AREA_PT2,
    FIGURE3_MARKER_HALO_PT,
    FIGURE3_MARKER_LINEWIDTH_PT,
    FIGURE3_MIN_SEPARATION_PT,
    FIGURE3_PANELS,
    FIGURE3_RING_AREA_PT2,
    FIGURE3_RING_LINEWIDTH_PT,
    FIGURE3_SIZE,
    MUTED,
    PARAM_SCATTER_XLIM,
    PARAM_SCATTER_YLIM,
    PRIMARY_PRIOR,
    _drawn_radius_pt,
    _figure3_panel_a,
    _figure3_panel_b,
    _figure3_rows,
    _pareto_frontier,
    _parameter_count_phantom_scatter,
    _points_per_unit,
    _points_per_unit_rect,
    _raw_ladder_rows,
    _segment_to_box,
    _separated_x,
)

DIGEST = Path(__file__).resolve().parents[1] / "output" / "stats_digest.json"
# The marker's own radius, plus half its stroke, plus the white halo drawn outside that stroke:
# read from the figure's own constants rather than restated, so shrinking a marker there cannot
# leave these measurements checking a clearance the figure no longer leaves.
MARKER_RADIUS_PT = _drawn_radius_pt(FIGURE3_MARKER_AREA_PT2, FIGURE3_MARKER_LINEWIDTH_PT,
                                    FIGURE3_MARKER_HALO_PT)
MARKER_AREA = FIGURE3_MARKER_AREA_PT2   # what _draw_prior_point passes to scatter
RING_AREA = FIGURE3_RING_AREA_PT2       # and what _ring passes, which finds the primary prior


@pytest.fixture(scope="module")
def digest():
    if not DIGEST.exists():
        pytest.skip("run_stats has not written output/stats_digest.json")
    return json.loads(DIGEST.read_text())


def _panel(digest, letter):
    """Draw one panel and read back what it drew: marker centres and the leader, in points."""
    rows = _figure3_rows(digest)
    figure = plt.figure(figsize=(4, 3))
    axis = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    if letter == "a":
        _figure3_panel_a(axis, rows, digest)
        scale = np.array(_points_per_unit("a", FIGURE3_A_XLIM, FIGURE3_A_YLIM))
    else:
        _figure3_panel_b(axis, rows, digest)
        scale = np.array(_points_per_unit("b", FIGURE3_B_XLIM, FIGURE3_B_YLIM))
    markers = np.array([collection.get_offsets()[0] for collection in axis.collections
                        if collection.get_sizes()[0] == MARKER_AREA]) * scale
    rings = [collection.get_offsets()[0] for collection in axis.collections
             if collection.get_sizes()[0] == RING_AREA]
    assert len(rings) == 1, "the panel should ring exactly one prior, the primary"
    leaders = [line for line in axis.get_lines() if line.get_color() == MUTED]
    assert len(leaders) == 1, "the panel should draw exactly one leader"
    leader = np.column_stack(leaders[0].get_data()) * scale
    plt.close(figure)
    return markers, leader, np.array(rings[0]) * scale


def _clearance(leader, marker):
    """Distance in points from a marker's edge to the nearest point of the leader segment."""
    start, end = leader
    span = end - start
    along = np.clip((marker - start) @ span / (span @ span), 0, 1)
    return float(np.linalg.norm(start + along * span - marker)) - MARKER_RADIUS_PT


@pytest.mark.parametrize("letter", ["a", "b"])
def test_the_leader_stays_clear_of_every_marker_it_does_not_label(digest, letter):
    markers, leader, primary = _panel(digest, letter)
    # The primary is the ringed prior, the one the leader points at; the leader stops short of
    # it by design, so it is the one marker the clearance rule cannot apply to.
    clearances = [_clearance(leader, marker) for marker in markers
                  if not np.allclose(marker, primary)]
    assert min(clearances) > 2.0, (
        f"panel {letter}'s leader comes within {min(clearances):.2f} points of another prior's "
        "marker edge; a reader following it lands on the wrong prior"
    )


@pytest.mark.parametrize("letter", ["a", "b"])
def test_the_leader_aims_at_the_prior_it_labels(digest, letter):
    _, leader, primary = _panel(digest, letter)
    start, end = leader
    span = (end - start) / np.linalg.norm(end - start)
    # Distance from the ringed prior's centre to the leader's line, extended past its near end.
    offset = primary - start
    miss = abs(float(span[0] * offset[1] - span[1] * offset[0]))
    assert miss < 0.1, f"panel {letter}'s leader misses the centre it labels by {miss:.2f} points"


def _drawn_panel_at_publication_size(digest, letter):
    """Draw one panel on the canvas and panel rectangle Figure 3 actually ships at, and read back
    the box the primary's label occupies, the boxes of every other label, and every marker
    centre, all in points.

    The canvas has to be the real one here. The leader's bearing is chosen in a point space
    derived from FIGURE3_SIZE and FIGURE3_PANELS rather than from whatever figure it is drawn on,
    so it comes out the same anywhere; the rendered extent of a label, measured back off the
    canvas, does not. Measuring a 4x3 test figure would compare the shipped placement against a
    panel of the wrong physical size.
    """
    rows = _figure3_rows(digest)
    figure = plt.figure(figsize=FIGURE3_SIZE)
    axis = figure.add_axes(FIGURE3_PANELS[letter])
    (_figure3_panel_a if letter == "a" else _figure3_panel_b)(axis, rows, digest)
    renderer = figure.canvas.get_renderer()
    scale = 72.0 / figure.dpi

    def box(artist):
        extent = artist.get_window_extent(renderer)
        return np.array([extent.x0, extent.x1, extent.y0, extent.y1]) * scale

    labels = {artist.get_text(): box(artist) for artist in axis.texts}
    primary = next(text for text in labels if text.startswith("primary prior"))
    markers = np.array([collection.get_offsets()[0] for collection in axis.collections
                        if collection.get_sizes()[0] == MARKER_AREA])
    markers = axis.transData.transform(markers) * scale
    ring = axis.transData.transform(
        [collection.get_offsets()[0] for collection in axis.collections
         if collection.get_sizes()[0] == RING_AREA]
    )[0] * scale
    limits = np.array(axis.get_window_extent(renderer).extents)[[0, 2, 1, 3]] * scale
    plt.close(figure)
    return labels.pop(primary), labels, markers, ring, limits


def _gap_to_box(point, box):
    return float(np.hypot(max(box[0] - point[0], point[0] - box[1], 0.0),
                          max(box[2] - point[1], point[1] - box[3], 0.0)))


@pytest.mark.parametrize("letter", ["a", "b"])
def test_the_primary_labels_own_box_stays_clear_of_everything_the_panel_draws(digest, letter):
    """The leader was measured against marker centres, but its label never was against anything.

    That gap put "primary prior, Qwen2.5 32B" in panel b directly on top of a base Qwen marker -
    the marker's ring struck out the "2" of "32B" - while the leader line itself was, correctly,
    nowhere near a marker. A leader that lands its own text on a prior names the wrong one just
    as plainly as a line that stops inside it, so the box is now measured too, and this is the
    measurement.
    """
    label, others, markers, ring, limits = _drawn_panel_at_publication_size(digest, letter)
    ring_radius = _drawn_radius_pt(FIGURE3_RING_AREA_PT2, FIGURE3_RING_LINEWIDTH_PT)

    nearest = min((_gap_to_box(marker, label) - MARKER_RADIUS_PT, tuple(marker))
                  for marker in markers if not np.allclose(marker, ring))
    assert nearest[0] >= FIGURE3_LEADER_CLEARANCE_PT, (
        f"panel {letter}'s primary label comes within {nearest[0]:.2f} points of another prior's "
        f"marker edge at {nearest[1]}; a reader reads the label as naming that prior"
    )
    assert _gap_to_box(ring, label) - ring_radius >= FIGURE3_LEADER_CLEARANCE_PT, (
        f"panel {letter}'s primary label sits on the highlight ring it names"
    )

    for text, box in others.items():
        horizontal = max(box[0] - label[1], label[0] - box[1])
        vertical = max(box[2] - label[3], label[2] - box[3])
        gap = (np.hypot(max(horizontal, 0.0), max(vertical, 0.0))
               if horizontal >= 0 or vertical >= 0 else max(horizontal, vertical))
        assert gap >= FIGURE3_LEADER_CLEARANCE_PT, (
            f"panel {letter}'s primary label is {gap:.2f} points from {text.splitlines()[0]!r}"
        )

    assert (label[0] >= limits[0] and label[1] <= limits[1]
            and label[2] >= limits[2] and label[3] <= limits[3]), (
        f"panel {letter}'s primary label runs outside the panel"
    )


def _labelled_points(digest, letter):
    """Every label a panel hangs off a single marker, with the box it occupies and the marker
    it names, all in points on the canvas the figure ships at.

    Panel a's magnified inset is a child axis with its own scale, so the labels it carries are
    read back off it separately and reported in its own points. What matters about a label is
    the same in either: how far it is from the marker it names against how far it is from every
    marker it does not.
    """
    rows = _figure3_rows(digest)
    figure = plt.figure(figsize=FIGURE3_SIZE)
    axis = figure.add_axes(FIGURE3_PANELS[letter])
    (_figure3_panel_a if letter == "a" else _figure3_panel_b)(axis, rows, digest)
    renderer = figure.canvas.get_renderer()
    scale = 72.0 / figure.dpi
    by_label = {row["label"]: row for row in rows}

    found = []
    axes = [axis] + [child for child in axis.get_children()
                     if isinstance(child, matplotlib.axes.Axes)]
    for drawn in axes:
        centres = np.array([collection.get_offsets()[0] for collection in drawn.collections
                            if collection.get_sizes()[0] == MARKER_AREA])
        if not len(centres):
            continue
        centres = drawn.transData.transform(centres) * scale
        for artist in drawn.texts:
            row = by_label.get(artist.get_text())
            if row is None:
                continue
            extent = artist.get_window_extent(renderer)
            box = np.array([extent.x0, extent.x1, extent.y0, extent.y1]) * scale
            own = drawn.transData.transform(
                [[_drawn_x(digest)[row["name"]] if letter == "a" else row["capture"],
                  row["phantom"] if letter == "a" else row["gain"]]]
            )[0] * scale
            gaps = sorted(float(_gap_to_box(centre, box)) - MARKER_RADIUS_PT
                          for centre in centres if not np.allclose(centre, own))
            found.append((artist.get_text(), float(_gap_to_box(own, box)) - MARKER_RADIUS_PT,
                          gaps[0], box, drawn))
    plt.close(figure)
    return found


def _drawn_x(digest):
    rows = _figure3_rows(digest)
    x_per_unit, y_per_unit = _points_per_unit("a", FIGURE3_A_XLIM, FIGURE3_A_YLIM)
    return dict(zip(
        [row["name"] for row in rows],
        _separated_x([row["nll"] for row in rows], [row["phantom"] for row in rows],
                     x_per_unit, y_per_unit, passes=50),
    ))


@pytest.mark.parametrize("letter", ["a", "b"])
def test_every_point_label_is_nearer_the_prior_it_names_than_any_other(digest, letter):
    """A label hung off a marker has to be readable as that marker's label and no other's.

    "Mixtral 8x7B" used to be set 8 points below its own marker on a fixed offset, in the middle
    of panel a's cluster, and came to rest against a RecurrentGemma marker it did not name; the
    same fixed offset put "5-gram KN" in panel b hard against the neighbouring 5-gram diamond.
    Nearness is what a reader has to go on when a label carries no leader, so nearness is what
    this measures: the gap to the named marker against the gap to the closest of the rest.
    """
    labelled = _labelled_points(digest, letter)
    assert labelled, f"panel {letter} should label at least one prior"
    for text, own, nearest, _, _ in labelled:
        assert own < nearest, (
            f"panel {letter}'s {text!r} is {own:.2f} points from the prior it names and "
            f"{nearest:.2f} from the nearest one it does not; a reader reads it as naming that "
            "one instead"
        )
        assert nearest >= FIGURE3_LEADER_CLEARANCE_PT, (
            f"panel {letter}'s {text!r} comes within {nearest:.2f} points of a marker it does "
            "not name"
        )


def test_panel_bs_frontier_labels_are_not_struck_through_by_the_frontier(digest):
    """The step drawn between two frontier priors of equal gain runs through their own labels.

    Both of panel b's frontier priors gain about 8.7 points, so where="post" draws a horizontal
    run between them at exactly the height a label set level with its own diamond sits at. It
    struck "5-gram KN" through at mid-cap height and carried on into the neighbouring 5-gram
    diamond, which reads as the label naming that diamond rather than its own.
    """
    rows = _figure3_rows(digest)
    frontier = _pareto_frontier(rows)
    scale = np.array(_points_per_unit("b", FIGURE3_B_XLIM, FIGURE3_B_YLIM))
    labels = {text: box for text, _, _, box, _ in _labelled_points(digest, "b")}

    figure = plt.figure(figsize=FIGURE3_SIZE)
    axis = figure.add_axes(FIGURE3_PANELS["b"])
    _figure3_panel_b(axis, rows, digest)
    origin = np.array(axis.transData.transform([[0.0, 0.0]])[0]) * 72.0 / figure.dpi
    plt.close(figure)

    segments = [((near["capture"], near["gain"]), (far["capture"], near["gain"]))
                for near, far in zip(frontier, frontier[1:])]
    segments += [((far["capture"], near["gain"]), (far["capture"], far["gain"]))
                 for near, far in zip(frontier, frontier[1:])]
    segments.append(((FIGURE3_B_XLIM[0], 0.0), (FIGURE3_B_XLIM[1], 0.0)))
    assert segments, "the frontier should draw at least one segment"

    for row in frontier + [min(rows, key=lambda item: item["gain"])]:
        box = labels[row["label"]] - origin[[0, 0, 1, 1]]
        for start, end in segments:
            assert _segment_to_box(np.array(start) * scale, np.array(end) * scale, box) > 0.0, (
                f"panel b's {row['label']!r} is struck through by a line the panel draws"
            )


def test_panel_a_magnifies_its_cluster_and_cuts_no_marker_in_half(digest):
    """The inset exists, holds most of the ladder, separates it, and clips nothing.

    The region it magnifies is bounded by markers, not by their centres: a marker is a disc a
    little over three points across, so a region drawn to the centres cuts discs in half at its
    edges. Every prior is either wholly inside the dashed box or wholly outside it, and inside
    it every pair is far enough apart to be counted, which is what the inset is for.
    """
    rows = _figure3_rows(digest)
    figure = plt.figure(figsize=FIGURE3_SIZE)
    axis = figure.add_axes(FIGURE3_PANELS["a"])
    _figure3_panel_a(axis, rows, digest)
    insets = [child for child in axis.get_children() if isinstance(child, matplotlib.axes.Axes)]
    assert len(insets) == 1, "panel a should draw exactly one inset"
    inset = insets[0]
    assert inset.get_xlim() == FIGURE3_A_INSET_ZOOM[:2]
    assert inset.get_ylim() == FIGURE3_A_INSET_ZOOM[2:]
    drawn = np.array([collection.get_offsets()[0] for collection in inset.collections
                      if collection.get_sizes()[0] == MARKER_AREA])
    plt.close(figure)
    assert len(drawn) == len(rows), "the inset should redraw every prior and let the axis clip"

    panel = np.array(_points_per_unit("a", FIGURE3_A_XLIM, FIGURE3_A_YLIM))
    zoomed = np.array(_points_per_unit_rect(FIGURE3_A_INSET_RECT, FIGURE3_A_INSET_ZOOM))
    assert (zoomed > 2.0 * panel).all(), (
        f"the inset magnifies by {(zoomed / panel).round(2)}, which is not enough to separate a "
        "cluster whose closest pair is only 3.5 points apart on the panel"
    )

    left, right, bottom, top = FIGURE3_A_INSET_ZOOM
    x_of = _drawn_x(digest)
    inside = []
    for row in rows:
        point = np.array([x_of[row["name"]], row["phantom"]])
        outward = np.hypot(*(np.maximum(np.maximum([left, bottom] - point,
                                                   point - [right, top]), 0.0) * panel))
        inward = min((point[0] - left) * panel[0], (right - point[0]) * panel[0],
                     (point[1] - bottom) * panel[1], (top - point[1]) * panel[1])
        assert outward > MARKER_RADIUS_PT or inward > MARKER_RADIUS_PT, (
            f"the dashed box cuts through {row['label']}'s marker"
        )
        if inward > MARKER_RADIUS_PT:
            inside.append(point * zoomed)
    assert len(inside) >= 12, (
        f"the inset shows only {len(inside)} priors; the cluster it exists to open up holds "
        "more than that"
    )
    gaps = [float(np.linalg.norm(inside[i] - inside[j]))
            for i in range(len(inside)) for j in range(i + 1, len(inside))]
    assert min(gaps) > 2 * MARKER_RADIUS_PT, (
        f"two of the inset's markers are {min(gaps):.2f} points apart, which still overlaps at "
        f"a drawn radius of {MARKER_RADIUS_PT:.2f}"
    )


def test_the_primary_prior_is_labelled_on_the_panel_even_though_it_sits_inside_the_inset_region(
    digest,
):
    """The primary prior falls inside the magnified region but is named outside it.

    Panel a splits its labels on the zoom region: a named prior inside it is labelled in the
    inset, where there is room. The primary prior is the one exception, because the ring and the
    leader are what make it the panel's subject and they belong where the reader is looking. It
    does fall inside the region, so the exception is live rather than hypothetical, and the
    caption has to say so: a first draft of that caption claimed everything inside the box was
    labelled in the inset, which the figure visibly contradicted by running the primary's leader
    into the box. This pins both halves - that the primary is inside, and that it is named on
    the panel anyway - so the caption cannot drift back to the simpler, wrong sentence.
    """
    rows = _figure3_rows(digest)
    primary = next(row for row in rows if row["name"] == PRIMARY_PRIOR)
    left, right, bottom, top = FIGURE3_A_INSET_ZOOM
    x = _drawn_x(digest)[PRIMARY_PRIOR]
    assert left <= x <= right and bottom <= primary["phantom"] <= top, (
        "the primary prior no longer falls inside the magnified region, so panel a's caption "
        "should stop carrying the exception this test exists to protect"
    )

    figure = plt.figure(figsize=FIGURE3_SIZE)
    axis = figure.add_axes(FIGURE3_PANELS["a"])
    _figure3_panel_a(axis, rows, digest)
    inset = next(child for child in axis.get_children()
                 if isinstance(child, matplotlib.axes.Axes))
    on_panel = [artist.get_text() for artist in axis.texts]
    in_inset = [artist.get_text() for artist in inset.texts]
    plt.close(figure)

    assert any(text.startswith("primary prior") and primary["label"] in text
               for text in on_panel), (
        "panel a should name the primary prior on the panel itself"
    )
    assert not any(primary["label"] in text for text in in_inset), (
        "the primary prior should not also be named inside the inset"
    )


def test_no_two_markers_in_panel_a_are_close_enough_to_hide_each_other(digest):
    markers, _, _ = _panel(digest, "a")
    gaps = [float(np.linalg.norm(markers[i] - markers[j]))
            for i in range(len(markers)) for j in range(i + 1, len(markers))]
    assert min(gaps) >= FIGURE3_MIN_SEPARATION_PT - 1e-6, (
        f"two of panel a's markers are {min(gaps):.2f} points apart, so the one drawn second "
        "covers the first and the panel shows fewer priors than the caption promises"
    )


def test_a_segment_that_crosses_a_label_box_reports_no_clearance_from_it():
    """A wide box and a steep segment through it: both ends outside, every corner off the line.

    This is the shape that let eFigure 4's leader run through the "B" of its own "32B" - the
    clearance check saw the tip 3 points below the box, the far end well above it, and each
    corner some distance from the line, and reported the smallest of those as the gap.
    """
    box = (-54.6, 21.0, -37.8, -32.4)
    assert _segment_to_box((4.2, -9.1), (19.0, -40.8), box) == 0.0
    # and the case it has to keep getting right: a leader stopping just short of its own label
    assert _segment_to_box((4.2, -9.1), (19.0, -40.8), (-54.6, 21.0, -50.0, -44.6)) > 3.0


def test_separation_moves_only_coincident_markers_and_moves_them_symmetrically():
    xs, ys = [1.0, 1.0, 5.0], [2.0, 2.0, 2.0]
    moved = _separated_x(xs, ys, x_per_unit=10.0, y_per_unit=10.0, minimum=4.0)
    assert moved[2] == 5.0, "a marker with room around it should not move"
    assert moved[0] - moved[1] == pytest.approx(0.4), "the pair should end up 4 points apart"
    assert moved[0] + moved[1] == pytest.approx(2.0), "and straddle the value they share"


def test_the_primary_prior_is_among_the_markers_both_panels_draw(digest):
    rows = _figure3_rows(digest)
    assert any(row["name"] == PRIMARY_PRIOR for row in rows)
    for letter in ("a", "b"):
        markers, _, _ = _panel(digest, letter)
        # The calibrated ladder has 24 priors (the uniform null has no calibrated basis;
        # Table 2's note), and both panels now draw every one of them.
        assert len(markers) >= 24


def _parameter_count_panel(digest):
    """Draw _parameter_count_phantom_scatter and read back what it drew, exactly as _panel does
    for the two panels figure3() still calls. This function is no longer wired into any figure
    (Task 23 moved Figure 3 panel a onto next-character NLL), but the layout it preserves is
    unchanged, and a future supplementary figure (Task 27) depends on it still rendering
    correctly.
    """
    rows = _raw_ladder_rows(digest)
    figure = plt.figure(figsize=(4, 3))
    axis = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    _parameter_count_phantom_scatter(axis, rows, digest)
    scale = np.array(_points_per_unit("a", PARAM_SCATTER_XLIM, PARAM_SCATTER_YLIM))
    markers = np.array([collection.get_offsets()[0] for collection in axis.collections
                        if collection.get_sizes()[0] == MARKER_AREA]) * scale
    plt.close(figure)
    return markers


def test_figure3_caption_discloses_the_all_priors_correlation_the_panel_no_longer_annotates(
    digest,
):
    """Panel a plots all 24 priors (21 neural + 3 character 5-grams) in one NLL space. The
    all-priors correlation (sensitivity.prior_quality_correlations) was originally computed but
    never surfaced anywhere; docs/superpowers/plans/2026-08-14-reviewer-feedback-fixes.md Task 8
    then added it as a headline in-panel annotation beside the neural-only correlation, in
    response to a reviewer's disclosure request, and this test was written to pin that in-panel
    placement.

    docs/superpowers/plans/2026-08-16-editorial-revision-round.md Task 6 reverses that placement
    decision, not the disclosure itself: a different, later external review asked to demote the
    number from headline status, because it pools two model classes while the panel's primary
    correlation is restricted to the 21 neural priors, and a co-equal in-panel annotation
    overstated how comparable the two numbers are. The number itself still has to reach a
    reader, so this test now checks the caption instead of the panel's own annotation text.

    The 2026-08-16 post-revision round's Task 5 swapped the caption's stated reason for that
    restriction, from "the 5-grams' NLL measures corpus frequency, not pretraining" to avoiding
    a conflation of predictive quality with model class. The assertions below are unaffected:
    they pin where the number appears, not why.
    """
    all_priors_rho = digest["sensitivity"]["prior_quality_correlations"][
        "spearman_rho_phantom_vs_nll"]["rho"]
    figure3_caption = TABLES[TABLES.index("**Figure 3."):TABLES.index("**Figure 4.")]
    assert f"{all_priors_rho:+.2f}" in figure3_caption, (
        "Figure 3's caption must disclose the all-priors correlation "
        f"({all_priors_rho:+.2f}) even though the panel itself no longer annotates it"
    )

    rows = _figure3_rows(digest)
    figure = plt.figure(figsize=(4, 3))
    axis = figure.add_axes([0.1, 0.1, 0.8, 0.8])
    _figure3_panel_a(axis, rows, digest)
    texts = [t.get_text() for t in axis.texts]
    plt.close(figure)
    assert not any(f"{all_priors_rho:+.2f}" in text for text in texts), (
        "Figure 3 panel a should no longer annotate the all-priors correlation in-panel; it is "
        "demoted to the caption by Task 6 of the 2026-08-16 editorial-revision-round plan"
    )


def test_the_preserved_parameter_count_scatter_still_draws_every_prior_without_hiding_one(digest):
    markers = _parameter_count_panel(digest)
    # The raw ladder has 25 priors, the uniform null included; unlike the calibrated ladder the
    # two panels above draw, this is the one place in the module that still has a use for it.
    assert len(markers) >= 25
    gaps = [float(np.linalg.norm(markers[i] - markers[j]))
            for i in range(len(markers)) for j in range(i + 1, len(markers))]
    assert min(gaps) >= FIGURE3_MIN_SEPARATION_PT - 1e-6, (
        f"two of the preserved scatter's markers are {min(gaps):.2f} points apart, so the one "
        "drawn second covers the first"
    )
