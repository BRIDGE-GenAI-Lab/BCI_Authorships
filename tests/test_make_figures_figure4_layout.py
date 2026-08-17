"""Figure 4's three layout promises, measured on the rendered figure rather than asserted.

All three have been broken in the making of this panel set. Text placed by eye ran a band label
from panel a straight through panel b's category names; a line of running text left unwrapped ran
off the right-hand edge, which save()'s tight bounding box turns into a silently wider export with
the panels crowded into two thirds of it; and an opaque badge dropped onto panel a's swarm to mark
one of the three profiles covered up to nine of the observations beside it, in a panel whose whole
subject is the shape of that distribution.
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

import make_figures as mf  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
NEEDED = [
    OUTPUT / "stats_digest.json",
    OUTPUT / "intermediate" / "attribution.parquet",
    OUTPUT / "intermediate" / "selections.parquet",
    OUTPUT / "intermediate" / "priors.parquet",
]


@pytest.fixture(scope="module")
def figure():
    if any(not path.exists() for path in NEEDED):
        pytest.skip("the analysis outputs Figure 4 is built from have not been written")
    captured = {}
    original = mf.save
    mf.save = lambda fig, path: captured.setdefault("fig", fig)
    try:
        mf.apply_style()
        digest, _attribution = mf.load()
        mf.figure4(digest)
    finally:
        mf.save = original
    fig = captured["fig"]
    fig.canvas.draw()
    yield fig
    plt.close(fig)


def _renderer(fig):
    return fig.canvas.get_renderer()


def _drawn_texts(fig):
    """Every text the figure actually draws, with a name for the failure message.

    Ticks outside their own axis limits, and the ticks and labels of the decoration-only badge
    axes, are returned by matplotlib but never rendered, so they are excluded here.
    """
    items = [("figure", text) for text in fig.texts
             if text.get_visible() and text.get_text().strip()]
    for index, axis in enumerate(fig.axes):
        if not axis.get_visible():
            continue
        items += [(f"ax{index}", text) for text in axis.texts
                  if text.get_visible() and text.get_text().strip()]
        if not axis.axison:
            continue
        for label, name in ((axis.xaxis.label, "xlabel"), (axis.yaxis.label, "ylabel")):
            if label.get_text():
                items.append((f"ax{index}:{name}", label))
        pairs = (list(zip(axis.get_xticklabels(), axis.get_xticks(), [sorted(axis.get_xlim())]
                          * len(axis.get_xticks())))
                 + list(zip(axis.get_yticklabels(), axis.get_yticks(), [sorted(axis.get_ylim())]
                            * len(axis.get_yticks()))))
        for tick, position, limits in pairs:
            if tick.get_text() and limits[0] <= position <= limits[1]:
                items.append((f"ax{index}:tick '{tick.get_text()}'", tick))
    return items


def test_no_two_pieces_of_text_overlap(figure):
    renderer = _renderer(figure)
    items = _drawn_texts(figure)
    assert len(items) > 60, "the figure should carry its full complement of labels"
    collisions = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (name_a, first), (name_b, second) = items[i], items[j]
            box_a = first.get_window_extent(renderer)
            box_b = second.get_window_extent(renderer)
            overlap_x = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            overlap_y = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            if overlap_x > 0 and overlap_y > 0:
                collisions.append(
                    f"[{name_a}] {first.get_text()[:40]!r} over [{name_b}] "
                    f"{second.get_text()[:40]!r}"
                )
    assert not collisions, "overlapping text: " + "; ".join(collisions)


def test_no_text_runs_off_the_canvas(figure):
    """save() trims to a tight bounding box, so text past the edge widens the export instead of
    being clipped: the panels end up occupying part of a wider image with the overrun beside
    them."""
    renderer = _renderer(figure)
    width, height = figure.get_size_inches() * figure.dpi
    escapes = []
    for name, text in _drawn_texts(figure):
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            escapes.append(f"[{name}] {text.get_text()[:40]!r}")
    assert not escapes, "text outside the figure canvas: " + "; ".join(escapes)


def test_panel_a_profile_badges_hide_no_observation(figure):
    """The numbered badges sit in a lane beside the swarm, not on it. Measured edge to edge in
    points, which is where marker sizes live."""
    panel_a = figure.axes[0]
    per_point = figure.dpi / 72.0
    swarm = max(panel_a.collections, key=lambda collection: len(collection.get_offsets()))
    assert len(swarm.get_offsets()) > 100, "panel a should draw every phantom-agreement case"
    points = panel_a.transData.transform(np.asarray(swarm.get_offsets())) / per_point
    swarm_radius = np.sqrt(swarm.get_sizes()).max() / 2

    badges = [collection for collection in panel_a.collections
              if len(collection.get_offsets()) == 1
              and np.isclose(collection.get_sizes().max(), 52)]
    assert len(badges) == len(mf.FIGURE4_CAPTION_PROFILES)
    for number, badge in enumerate(badges, start=1):
        centre = panel_a.transData.transform(np.asarray(badge.get_offsets()))[0] / per_point
        radius = np.sqrt(badge.get_sizes()).max() / 2
        clearance = (np.linalg.norm(points - centre, axis=1) - radius - swarm_radius).min()
        assert clearance > 0, (
            f"panel a's badge {number} overlaps an observation by {-clearance:.2f} points"
        )
