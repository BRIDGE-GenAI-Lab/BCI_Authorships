import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import make_revision_figures  # noqa: E402
from make_revision_figures import efigure5, efigure6, figure1_panel_d_rows  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "output"
STATS_DIGEST = OUTPUT / "stats_digest.json"
REVISION_DIGEST = OUTPUT / "revision_digest.json"


def test_panel_d_rows_cover_reference_fidelity_and_range():
    stats = {"calibrated_primary_analysis": {
        "ncf": {"estimate": 0.914, "ci_low": 0.896, "ci_high": 0.934},
        "phantom_agreement": {"estimate": 0.044, "ci_low": 0.035, "ci_high": 0.053}}}
    revision = {
        "fidelity_restricted": {"archive_agreement_ge_0.90": {
            "prior_share": {"estimate": 0.05, "ci_low": 0.03, "ci_high": 0.07},
            "phantom_agreement": {"estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03}}},
        "decoder_strength": {"original": {"ladder_summary": {
            "ncf": {"median": 0.919, "min": 0.895, "max": 0.971},
            "phantom_agreement": {"median": 0.043, "min": 0.021, "max": 0.052}}}},
    }
    rows = figure1_panel_d_rows(stats, revision)
    prior_share = [r for r in rows if r["measure"] == "prior_share"]
    assert [r["kind"] for r in prior_share] == ["ci", "ci", "range"]
    assert abs(prior_share[0]["estimate"] - 0.086) < 1e-9
    assert abs(prior_share[2]["low"] - 0.029) < 1e-9 and abs(prior_share[2]["high"] - 0.105) < 1e-9


def test_panel_d_rows_phantom_agreement_bounds_are_not_swapped():
    """Phantom agreement is not a complement of anything stored in the digest, unlike prior
    share (NCF's complement): its own ci_low/ci_high already bracket its own estimate, so the
    row builder must pass them through unswapped."""
    stats = {"calibrated_primary_analysis": {
        "ncf": {"estimate": 0.914, "ci_low": 0.896, "ci_high": 0.934},
        "phantom_agreement": {"estimate": 0.044, "ci_low": 0.035, "ci_high": 0.053}}}
    revision = {
        "fidelity_restricted": {"archive_agreement_ge_0.90": {
            "prior_share": {"estimate": 0.05, "ci_low": 0.03, "ci_high": 0.07},
            "phantom_agreement": {"estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03}}},
        "decoder_strength": {"original": {"ladder_summary": {
            "ncf": {"median": 0.919, "min": 0.895, "max": 0.971},
            "phantom_agreement": {"median": 0.043, "min": 0.021, "max": 0.052}}}},
    }
    rows = figure1_panel_d_rows(stats, revision)
    phantom = [r for r in rows if r["measure"] == "phantom_agreement"]
    assert [r["kind"] for r in phantom] == ["ci", "ci", "range"]
    reference = phantom[0]
    assert reference["estimate"] == pytest.approx(0.044)
    assert reference["low"] == pytest.approx(0.035) and reference["high"] == pytest.approx(0.053)
    range_row = phantom[2]
    assert range_row["low"] == pytest.approx(0.021) and range_row["high"] == pytest.approx(0.052)
    for row in phantom:
        assert row["low"] < row["estimate"] < row["high"]


def test_panel_d_rows_fall_back_to_the_lower_fidelity_threshold_when_090_is_none():
    """fidelity_restricted_estimates() (scripts/revision_analyses.py) stores None for a
    threshold when no sessions clear it. Panel d has to fall back to >=0.80 rather than crash
    or silently draw an empty row."""
    stats = {"calibrated_primary_analysis": {
        "ncf": {"estimate": 0.914, "ci_low": 0.896, "ci_high": 0.934},
        "phantom_agreement": {"estimate": 0.044, "ci_low": 0.035, "ci_high": 0.053}}}
    revision = {
        "fidelity_restricted": {
            "archive_agreement_ge_0.90": None,
            "archive_agreement_ge_0.80": {
                "prior_share": {"estimate": 0.06, "ci_low": 0.04, "ci_high": 0.08},
                "phantom_agreement": {"estimate": 0.025, "ci_low": 0.015, "ci_high": 0.035}},
        },
        "decoder_strength": {"original": {"ladder_summary": {
            "ncf": {"median": 0.919, "min": 0.895, "max": 0.971},
            "phantom_agreement": {"median": 0.043, "min": 0.021, "max": 0.052}}}},
    }
    rows = figure1_panel_d_rows(stats, revision)
    prior_share = [r for r in rows if r["measure"] == "prior_share"]
    assert prior_share[1]["estimate"] == pytest.approx(0.06)
    assert "0.80" in prior_share[1]["label"]


NEURAL_21 = [
    "gpt2", "gpt2-large", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-14B", "Qwen/Qwen3.5-27B", "Qwen/Qwen2.5-32B",
    "Qwen/Qwen3.6-35B-A3B", "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it", "google/gemma-2-9b", "google/gemma-2-27b",
    "google/gemma-4-12b-it", "mistralai/Mistral-7B-v0.3",
    "mistralai/Mixtral-8x7B-v0.1", "deepseek-ai/DeepSeek-V2-Lite",
    "allenai/OLMo-2-1124-7B-Instruct", "openai/gpt-oss-20b",
    "EleutherAI/pythia-12b", "state-spaces/mamba-2.8b-hf",
    "google/recurrentgemma-2b",
]


def test_panel_d_rows_reject_a_stale_ladder_summary():
    """The ladder-range row is a recomputation guard (see _neural_ladder_summary), not a
    blind read of decoder_strength.original.ladder_summary: if that stored summary no longer
    matches stats_digest's own per-prior ladder, the row builder must refuse to render rather
    than draw a Figure 1 range the underlying digest disagrees with."""
    stats = {
        "calibrated_primary_analysis": {
            "ncf": {"estimate": 0.914, "ci_low": 0.896, "ci_high": 0.934},
            "phantom_agreement": {"estimate": 0.044, "ci_low": 0.035, "ci_high": 0.053}},
        "sensitivity": {"prior_ladder_calibrated": {
            name: {"ncf": {"estimate": 0.92}, "phantom_agreement": {"estimate": 0.04}}
            for name in NEURAL_21
        }},
    }
    revision = {
        "fidelity_restricted": {"archive_agreement_ge_0.90": {
            "prior_share": {"estimate": 0.05, "ci_low": 0.03, "ci_high": 0.07},
            "phantom_agreement": {"estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03}}},
        # Deliberately stale: every per-prior estimate above is 0.92/0.04, so the true median
        # is 0.92/0.04, not the 0.919/0.043 this stored summary claims.
        "decoder_strength": {"original": {"ladder_summary": {
            "ncf": {"median": 0.919, "min": 0.895, "max": 0.971},
            "phantom_agreement": {"median": 0.043, "min": 0.021, "max": 0.052}}}},
    }
    with pytest.raises(RuntimeError, match="ladder range recomputed"):
        figure1_panel_d_rows(stats, revision)


def test_panel_d_rows_accept_the_real_digest_pair():
    """The one check that matters beyond the fixtures above: the row builder runs clean, with
    its recomputation guard active, against the actual committed digests."""
    if not STATS_DIGEST.exists() or not REVISION_DIGEST.exists():
        pytest.skip("output digests have not been generated")
    stats = json.loads(STATS_DIGEST.read_text())
    revision = json.loads(REVISION_DIGEST.read_text())
    rows = figure1_panel_d_rows(stats, revision)
    assert len(rows) == 6
    for row in rows:
        assert row["low"] <= row["estimate"] <= row["high"]
        assert row["kind"] in ("ci", "range")


@pytest.fixture(scope="module")
def digests():
    if not STATS_DIGEST.exists() or not REVISION_DIGEST.exists():
        pytest.skip("output digests have not been generated")
    return json.loads(STATS_DIGEST.read_text()), json.loads(REVISION_DIGEST.read_text())


def _measure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    sizes = [float(t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    width_mm = (tight.width + 2 * 0.08) * 25.4
    return sizes, width_mm


def _capture(monkeypatch):
    captured = {}

    def fake_save(fig, path_no_ext, dpi=600):
        captured["fig"] = fig
        return f"{path_no_ext}.png"

    monkeypatch.setattr(make_revision_figures, "save", fake_save)
    return captured


def test_efigure5_clears_the_double_column_and_the_font_floor(digests, monkeypatch):
    stats, revision = digests
    captured = _capture(monkeypatch)
    efigure5(revision, "unused")
    sizes, width_mm = _measure(captured["fig"])
    plt.close(captured["fig"])
    assert width_mm <= 183.0, f"eFigure5 renders {width_mm:.1f} mm wide"
    assert min(sizes) >= 6.0, f"eFigure5's smallest type is {min(sizes):.2f} pt"


def test_efigure6_clears_the_double_column_and_the_font_floor(digests, monkeypatch):
    _, revision = digests
    captured = _capture(monkeypatch)
    efigure6(revision, "unused")
    sizes, width_mm = _measure(captured["fig"])
    plt.close(captured["fig"])
    assert width_mm <= 183.0, f"eFigure6 renders {width_mm:.1f} mm wide"
    assert min(sizes) >= 6.0, f"eFigure6's smallest type is {min(sizes):.2f} pt"


def test_efigure6_draws_no_line_connecting_the_three_decoders(digests, monkeypatch):
    """The three decoders' CIs overlap, so this figure must not imply a monotonic trend across them. A
    connecting `ax.plot` line across the three decoder x-positions would draw exactly that
    trend; decoder points here are drawn only through `errorbar`, whose own zero-length
    connector never spans more than one x. This checks the rendered figure directly: no Line2D
    in either panel visits more than one of the three decoder x-positions in sequence."""
    _, revision = digests
    captured = _capture(monkeypatch)
    efigure6(revision, "unused")
    fig = captured["fig"]

    decoder_x = sorted(
        100 * revision["decoder_strength"][name]["reference_prior"]["neural_correct"]["estimate"]
        for name in make_revision_figures.DECODER_ORDER
    )

    for axes in fig.axes:
        for line in axes.get_lines():
            xdata = np.asarray(line.get_xdata(), dtype=float)
            if xdata.size < 2:
                continue
            hits = sum(any(abs(x - dx) < 1e-6 for x in xdata) for dx in decoder_x)
            assert hits < 2, (
                f"a line in {axes} visits {hits} of the 3 decoder x-positions in sequence, "
                "which draws a trend across decoders that the estimates do not support"
            )
    plt.close(fig)


def test_efigure6_plots_only_the_three_decoders_no_tertile_reference_points(digests, monkeypatch):
    """The calibration-AUC tertile points (grey, open markers) are not drawn in
    any panel: they have no neural-only accuracy of their own to plot on this figure's
    shared x-axis, and mixing calibration AUC into an axis labelled accuracy would put two
    different quantities on one scale. This checks the rendered figure has exactly 3 point
    series (the 3 decoders) in each of the 3 panels, not 6, and that CONTEXT grey never appears
    as a marker or line colour anywhere in the figure."""
    _, revision = digests
    captured = _capture(monkeypatch)
    efigure6(revision, "unused")
    fig = captured["fig"]

    assert len(fig.axes) == 3, f"expected 3 panels, found {len(fig.axes)}"
    for axes in fig.axes:
        assert len(axes.containers) == 3, (
            f"expected 3 point series (one per decoder) in {axes}, found "
            f"{len(axes.containers)} -- a tertile reference point may have crept back in"
        )

    from make_revision_figures import CONTEXT

    for axes in fig.axes:
        for container in axes.containers:
            colour = container[0].get_color()
            assert colour != CONTEXT, f"a CONTEXT-grey point series is still drawn in {axes}"
        for line in axes.get_lines():
            if line.get_marker() not in (None, "None", ""):
                assert line.get_markerfacecolor() != CONTEXT, (
                    f"a CONTEXT-grey marker is still drawn in {axes}"
                )
    plt.close(fig)


def _figure_texts(fig):
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_visible()]
    for ax in fig.axes:
        texts += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
    return texts


def test_revision_figures_carry_no_en_or_em_dash(digests, monkeypatch):
    """House style: no en or em dashes in figure artwork (ranges read "to" or use a hyphen)."""
    stats, revision = digests
    assert not any(ch in row["label"] for row in figure1_panel_d_rows(stats, revision)
                   for ch in "–—")
    for draw in (efigure5, efigure6):
        captured = _capture(monkeypatch)
        draw(revision, "unused")
        texts = _figure_texts(captured["fig"])
        plt.close(captured["fig"])
        dashed = [t for t in texts if "–" in t or "—" in t]
        assert not dashed, f"{draw.__name__} has dashed text: {dashed}"


def test_efigure6_names_eegnet_as_in_the_text_and_labels_each_decoder_at_its_point(
        digests, monkeypatch):
    """The legend and labels use "EEGNet-8,2", the name used in the text, and each
    decoder's panel-a label sits level with its own estimate, beside its own marker."""
    _, revision = digests
    captured = _capture(monkeypatch)
    efigure6(revision, "unused")
    fig = captured["fig"]
    texts = _figure_texts(fig)
    assert "EEGNet-8,2" in texts
    assert "EEGNet" not in texts
    ax_a = fig.axes[0]
    for name in make_revision_figures.DECODER_ORDER:
        block = revision["decoder_strength"][name]["reference_prior"]
        label = [a for a in ax_a.texts if a.get_text() == make_revision_figures.DECODER_LABEL[name]]
        assert len(label) == 1
        assert label[0].xy == pytest.approx(
            (100 * block["neural_correct"]["estimate"], 100 * block["prior_share"]["estimate"]))
    plt.close(fig)
