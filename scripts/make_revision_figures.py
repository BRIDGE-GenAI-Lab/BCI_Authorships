"""Figure 1 panel d's rows, and two supplementary figures (eFigure 5 and eFigure 6).

eFigure 5 profiles the confirmation-triage question: what a reversal's margin band
contains (helped by the prior, harmed by it, or neither), and what a confirmation policy built
on that margin, or on NCF directly, would catch. eFigure 6 profiles the decoder-strength
question: whether a stronger neural decoder changes how much the prior displaces
the fused posterior. figure1_panel_d_rows() feeds Figure 1 panel d itself,
which scripts/make_figures.py imports and draws; it does not draw anything on its own.

Colour grammar throughout, matching every other figure in this study: KEY (blue) is the
neural decoder, ACCENT3 (vermilion) is the language-model prior and anything the prior is
responsible for (phantom agreement, prior share, prior capture), INK (charcoal) is the fused
decision, and CONTEXT (grey) is reference material the panel did not set out to estimate
(a tertile computed a different way, a ceiling implied by another measure). eFigure 5's
"prior capture" bar is the one quantity in either figure that is a genuine negative outcome
(the prior overturning a neural pick that would have been correct), so it alone uses the kit's
reserved ERROR red rather than reusing ACCENT3 for both a good and a bad outcome in the same
bar; ERROR sits only 12.8 CIEDE2000 units and a 1.2:1 WCAG luminance ratio from ACCENT3 (both
checked against tests/test_make_figures_palette.py's own reference implementation of each
measure), too close to trust on hue or luminance alone, so that bar segment is additionally
hatched.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT = Path(__file__).resolve().parents[1]
KIT = Path(__file__).resolve().parent / "figure_style"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(KIT))

import make_figures  # noqa: E402
from hochberg_kit import (  # noqa: E402
    ACCENT3,
    CONTEXT,
    ERROR,
    INK,
    KEY,
    MUTED,
    apply_style,
    legend_below,
    panel,
    save,
)

OUTPUT = make_figures.OUTPUT
FIGURES = make_figures.FIGURES

# The four priors the calibrated ladder carries that are not a neural language model (a
# uniform null and three classical n-grams). Figure 3's own LADDER_ORDER lists all 25; the
# panel d range row and its recomputation guard below both need just the 21 that are left.
NON_NEURAL_PRIORS = frozenset({"uniform", "ngram5", "ngram5_kn", "ngram5_wiki_kn"})


def _neural_ladder_summary(stats_digest: dict) -> dict:
    """Recompute the calibrated ladder's median/min/max across the 21 neural language models
    directly from output/stats_digest.json's own per-prior digest (sensitivity.
    prior_ladder_calibrated), independent of whatever revision_digest.json's decoder_strength.
    original.ladder_summary stores. The same recompute-and-compare pattern every other figure
    in this file already uses on its own inputs (see the eFigure 1 and Figure 2 tertile
    guards): a hand-carried summary can drift from the source it was computed from without
    either file's own schema change catching it.
    """
    per_prior = stats_digest["sensitivity"]["prior_ladder_calibrated"]
    neural = [name for name in per_prior if name not in NON_NEURAL_PRIORS]
    if len(neural) != 21:
        raise RuntimeError(
            "Figure 1 panel d's ladder range expects 21 neural language models in the "
            f"calibrated prior ladder; output/stats_digest.json's sensitivity."
            f"prior_ladder_calibrated has {len(neural)}: {sorted(neural)}."
        )
    summary = {}
    for measure in ("ncf", "phantom_agreement"):
        values = [per_prior[name][measure]["estimate"] for name in neural]
        summary[measure] = {
            "median": float(np.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return summary


def figure1_panel_d_rows(stats_digest: dict, revision_digest: dict) -> list[dict]:
    """The rows Figure 1 panel d plots, per co-primary measure (prior share, phantom
    agreement): the reference prior's full-cohort estimate, the highest-fidelity archive
    subset, and the range across the full 21-model neural ladder.

    Returns one dict per row: measure ("prior_share" | "phantom_agreement"), label, estimate,
    low, high, kind ("ci" for a point with a 95% interval, "range" for a median with a
    min-max bar). Prior share is not a digest key on the reference row; it is NCF's
    complement, so its interval ends swap (low = 1 - ncf's ci_high) the way any complement's
    do. The fidelity-restricted and ladder blocks already carry prior_share as its own key
    (revision_analyses.outcome_block computes it once, the same way), so no swap is needed
    there except for the range row, whose source (ladder_summary) only carries ncf.
    """
    calibrated_primary = stats_digest["calibrated_primary_analysis"]
    ncf = calibrated_primary["ncf"]
    phantom = calibrated_primary["phantom_agreement"]

    fidelity_all = revision_digest["fidelity_restricted"]
    fidelity = fidelity_all.get("archive_agreement_ge_0.90")
    fidelity_threshold = "0.90"
    if fidelity is None:
        fidelity = fidelity_all["archive_agreement_ge_0.80"]
        fidelity_threshold = "0.80"

    ladder = revision_digest["decoder_strength"]["original"]["ladder_summary"]
    if "sensitivity" in stats_digest and "prior_ladder_calibrated" in stats_digest.get(
        "sensitivity", {}
    ):
        recomputed = _neural_ladder_summary(stats_digest)
        for measure in ("ncf", "phantom_agreement"):
            for stat in ("median", "min", "max"):
                stored = ladder[measure][stat]
                found = recomputed[measure][stat]
                if not math.isclose(stored, found, rel_tol=0, abs_tol=1e-9):
                    raise RuntimeError(
                        f"Figure 1 panel d's ladder range recomputed {measure} {stat} as "
                        f"{found:.6g} from output/stats_digest.json's own per-prior ladder, "
                        f"but output/revision_digest.json's decoder_strength.original."
                        f"ladder_summary reports {stored:.6g}."
                    )

    fidelity_label = f"Highest-fidelity subset (archive agreement ≥ {fidelity_threshold})"
    reference_label = "Reference prior (Qwen2.5-32B), full cohort"
    range_label = "21 neural language models, median (min to max)"

    rows: list[dict] = []
    for measure, complement, headline in (
        ("prior_share", True, ncf),
        ("phantom_agreement", False, phantom),
    ):
        rows.append({
            "measure": measure,
            "label": reference_label,
            "kind": "ci",
            "estimate": (1 - headline["estimate"]) if complement else headline["estimate"],
            "low": (1 - headline["ci_high"]) if complement else headline["ci_low"],
            "high": (1 - headline["ci_low"]) if complement else headline["ci_high"],
        })

        fidelity_block = fidelity[measure]
        rows.append({
            "measure": measure,
            "label": fidelity_label,
            "kind": "ci",
            "estimate": fidelity_block["estimate"],
            "low": fidelity_block["ci_low"],
            "high": fidelity_block["ci_high"],
        })

        ladder_block = ladder["ncf"] if measure == "prior_share" else ladder[measure]
        if complement:
            estimate = 1 - ladder_block["median"]
            low, high = 1 - ladder_block["max"], 1 - ladder_block["min"]
        else:
            estimate = ladder_block["median"]
            low, high = ladder_block["min"], ladder_block["max"]
        rows.append({
            "measure": measure,
            "label": range_label,
            "kind": "range",
            "estimate": estimate,
            "low": low,
            "high": high,
        })

    return rows


# ---------------------------------------------------------------------------- eFigure 5
EFIGURE5_SIZE = (7.2, 3.55)
BAND_ORDER = ("near_tie", "moderate", "strong")
BAND_LABEL = {"near_tie": "Near tie", "moderate": "Moderate", "strong": "Strong"}


def efigure5(revision_digest: dict, path) -> None:
    """Confirmation-triage profile of prior-mediated reversals.

    Panel a: what each reversal-margin band contains -- the phantom agreements the prior's
    disagreement with neural evidence helped (correct only because the prior overruled a wrong
    neural pick), the prior captures it harmed (wrong only because the prior overruled a
    correct neural pick), and the reversals where the fused pick was wrong either way. The
    three counts sum to the band's own n_reversals exactly (reversal_strength.bands in
    output/revision_digest.json), so the three bars stack to 100% by construction.

    Panel b: a confirmation policy's operating curve -- what share of selections it would
    flag for human confirmation (flag_rate) against what share of the actual prior captures
    that catches (fraction_of_prior_captures_flagged), one point per policy in
    output/revision_digest.json's flag_policies. The reversal-margin family (the margin
    between the prior's favoured symbol and the neural posterior's own top choice) is drawn as
    a solid line of filled circles; the NCF family (thresholding the neural contribution
    fraction directly, ignoring reversal margin) as a dashed line of open squares, so the two
    policy families are never read as one curve.
    """
    apply_style()
    bands = revision_digest["reversal_strength"]["bands"]
    policies = revision_digest["flag_policies"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=EFIGURE5_SIZE)
    panel(ax_a, "a")
    panel(ax_b, "b")

    x = np.arange(len(BAND_ORDER))
    helped = np.array([100 * bands[b]["phantom_agreement"] / bands[b]["n_reversals"] for b in BAND_ORDER])
    harmed = np.array([100 * bands[b]["prior_capture"] / bands[b]["n_reversals"] for b in BAND_ORDER])
    neither = np.array([100 * bands[b]["both_wrong"] / bands[b]["n_reversals"] for b in BAND_ORDER])

    ax_a.bar(x, helped, width=0.62, color=ACCENT3, zorder=2,
             label="phantom agreement (helped)")
    ax_a.bar(x, harmed, width=0.62, bottom=helped, color=ERROR, hatch="///",
             edgecolor="white", linewidth=0.5, zorder=2, label="prior capture (harmed)")
    ax_a.bar(x, neither, width=0.62, bottom=helped + harmed, color=CONTEXT, zorder=2,
             label="both wrong")
    for index, b in enumerate(BAND_ORDER):
        ax_a.text(index, 103.5, f"n={bands[b]['n_reversals']}", ha="center", va="bottom",
                   fontsize=6.4, color=MUTED)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([
        f"{BAND_LABEL[b]}\n[{bands[b]['margin_low']:.2f} to {bands[b]['margin_high']:.2f}"
        + ("]" if b == "strong" else ")")
        for b in BAND_ORDER
    ], fontsize=6.6)
    ax_a.set_ylim(0, 116)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_ylabel("share of the band's reversals (%)", fontsize=7.6)
    ax_a.set_xlabel("reversal margin band", fontsize=7.6, labelpad=4.0)
    ax_a.grid(axis="y", color="#E5E7EB", lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1, fontsize=6.4,
                frameon=False)

    margin_policies = [p for p in policies if p["policy"].startswith("reversal_margin_ge_")]
    ncf_policies = [p for p in policies if p["policy"].startswith("ncf_lt_")]

    mx = [100 * p["flag_rate"] for p in margin_policies]
    my = [100 * p["fraction_of_prior_captures_flagged"] for p in margin_policies]
    ax_b.plot(mx, my, color=INK, lw=1.1, marker="o", ms=5.2, mfc=INK, mec="white", mew=0.7,
              zorder=3, label="reversal-margin policy")
    # The lowest-flag-rate point of each family sits within a couple of points of the origin,
    # on top of the other family's own lowest point, of the x-axis tick labels, and of the
    # line's own steep early climb toward the next point up: a fixed small offset that reads
    # cleanly for the other 4-5 points on each curve puts the text directly across the line's
    # path there (an offset such as (-2, 11) still has the solid line bisecting the "0" in
    # "0.50"). Both families' closest-to-origin threshold is 0.50, so that one gets a
    # short leader arrow into genuinely empty space between the two curves instead of a flush
    # label, rather than trying to find a flush offset small enough to stay near the point and
    # large enough to clear the line.
    for p, xx, yy in zip(margin_policies, mx, my):
        threshold = p["policy"].rsplit("_ge_", 1)[1]
        if threshold == "0.50":
            ax_b.annotate(
                f"≥{threshold}", (xx, yy), xytext=(20, 22), textcoords="offset points",
                fontsize=6.0, color=INK, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=1, shrinkB=3),
            )
        else:
            ax_b.annotate(f"≥{threshold}", (xx, yy), textcoords="offset points",
                          xytext=(4, 3), fontsize=6.0, color=INK)

    nx = [100 * p["flag_rate"] for p in ncf_policies]
    ny = [100 * p["fraction_of_prior_captures_flagged"] for p in ncf_policies]
    order = np.argsort(nx)
    nx, ny, ncf_policies = (np.asarray(nx)[order], np.asarray(ny)[order],
                             [ncf_policies[i] for i in order])
    ax_b.plot(nx, ny, color=INK, lw=0.9, ls=(0, (3, 2)), marker="s", ms=5.2, mfc="white",
              mec=INK, mew=1.0, zorder=3, label="NCF policy")
    for p, xx, yy in zip(ncf_policies, nx, ny):
        threshold = p["policy"].rsplit("_lt_", 1)[1]
        offset = (9, -3) if threshold == "0.50" else (4, -8)
        ax_b.annotate(f"NCF<{threshold}", (xx, yy), textcoords="offset points",
                      xytext=offset, fontsize=6.0, color=INK)

    ax_b.set_xlim(0, max(list(mx) + list(nx)) * 1.18)
    ax_b.set_ylim(0, 108)
    ax_b.set_xlabel("selections flagged for confirmation (%)", fontsize=7.6)
    ax_b.set_ylabel("prior captures surfaced (%)", fontsize=7.6)
    ax_b.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right", fontsize=6.4, frameon=False)

    fig.subplots_adjust(wspace=0.42, bottom=0.30, top=0.94, left=0.09, right=0.98)
    save(fig, str(path))


# ---------------------------------------------------------------------------- eFigure 6
EFIGURE6_SIZE = (7.2, 9.0)
DECODER_ORDER = ("original", "xdawn_riemann", "eegnet")
DECODER_LABEL = {
    "original": "Original decoder",
    "xdawn_riemann": "xDAWN-Riemannian",
    "eegnet": "EEGNet-8,2",
}
DECODER_MARKER = {"original": "o", "xdawn_riemann": "s", "eegnet": "^"}
# Each decoder's name sits level with its own estimate, immediately beside its own marker:
# the original decoder and EEGNet-8,2 to the right, xDAWN-Riemannian to the left, so that the
# two stronger decoders' labels point away from each other across the 2.5-point gap between
# them. Anchoring the labels at the whisker ends would leave the xDAWN-Riemannian label
# floating above and left of its own whisker, far enough from the square that readers could
# not tell which point it named.
DECODER_LABEL_STYLE = {
    "original": {"anchor": "estimate", "xytext": (8, 0), "ha": "left", "va": "center"},
    "xdawn_riemann": {"anchor": "estimate", "xytext": (-8, 0), "ha": "right", "va": "center"},
    "eegnet": {"anchor": "estimate", "xytext": (8, 0), "ha": "left", "va": "center"},
}


def _draw_point(ax, x, block, measure, colour, marker, ms=6.5, open_marker=False):
    estimate = 100 * block[measure]["estimate"]
    low = 100 * block[measure]["ci_low"]
    high = 100 * block[measure]["ci_high"]
    ax.errorbar(
        [x], [estimate], yerr=[[estimate - low], [high - estimate]], fmt=marker, ms=ms,
        color=colour, ecolor=colour, elinewidth=1.2, capsize=2.6, zorder=4,
        mfc=("white" if open_marker else colour), mec=(colour if open_marker else "white"),
        mew=(1.1 if open_marker else 0.7),
    )


def efigure6(revision_digest: dict, path) -> None:
    """Decoder-strength profile of the prior's contribution.

    One point per decoder (original, xDAWN-Riemannian, EEGNet) at its own reference-prior
    neural-only accuracy, participant-weighted (decoder_strength.<name>.reference_prior.
    neural_correct in output/revision_digest.json): panel a plots prior share of posterior
    displacement, panel b plots phantom agreement, panel c plots rescue rate. Both are the
    language model's own contribution to a wrong neural decision, but on different
    denominators (phantom agreement is a rate over every selection; rescue rate is a rate over
    only the selections where the neural decoder was wrong, so it is not bounded by the neural
    error rate the way phantom agreement is), which is why they were split into their own
    panels rather than sharing one axis and one ceiling line: the ceiling drawn in panel b
    bounds phantom agreement only, and drawing rescue rate on the same axis would put points
    near a line that says nothing about them. Marker shape
    carries decoder identity in every panel (circle/square/triangle); marker colour carries
    the measure (ACCENT3, vermilion, for the prior's contribution in panels a-b; INK, charcoal,
    for rescue rate in panel c, the fused rule's own recovery of an error).

    The three calibration-AUC tertile points from output/stats_digest.json's
    secondary.calibrated.by_decoder_quality (the same low/mid/high split Figure 2 draws) are
    not plotted: that digest block carries no neural-only accuracy figure for those three
    tertiles, so plotting them alongside the three decoders would put a calibration-AUC value
    and a neural-accuracy value, two different quantities, on one shared x-axis labelled as
    accuracy. This figure shows only the three decoders, all on the one basis the x-axis
    states.

    Overlapping marginal intervals are not a test of a decoder difference (the same
    participants and selections are scored under every decoder); the participant-paired
    contrasts in eTable 36 are. Three decoders do not define a trend, so this figure draws
    no line connecting them; the dashed line in panel b is a
    ceiling implied by arithmetic (phantom agreement cannot exceed the neural error rate,
    1 - accuracy), not a fitted trend, and its label sits in the panel's empty top-right
    corner rather than on the line itself.
    """
    apply_style()
    decoders = revision_digest["decoder_strength"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=EFIGURE6_SIZE, sharex=True)
    panel(ax_a, "a")
    panel(ax_b, "b")
    panel(ax_c, "c")

    decoder_x = {}
    prior_share_bounds = []
    for name in DECODER_ORDER:
        block = decoders[name]["reference_prior"]
        x = 100 * block["neural_correct"]["estimate"]
        decoder_x[name] = x
        marker = DECODER_MARKER[name]
        _draw_point(ax_a, x, block, "prior_share", ACCENT3, marker)
        prior_share_bounds.append(100 * block["prior_share"]["ci_low"])
        prior_share_bounds.append(100 * block["prior_share"]["ci_high"])
        style = DECODER_LABEL_STYLE[name]
        anchor_y = 100 * block["prior_share"][style["anchor"]]
        ax_a.annotate(DECODER_LABEL[name], (x, anchor_y), textcoords="offset points",
                      xytext=style["xytext"], ha=style["ha"], va=style["va"], fontsize=6.2,
                      color=INK)
        _draw_point(ax_b, x, block, "phantom_agreement", ACCENT3, marker)
        _draw_point(ax_c, x, block, "rescue_rate", INK, marker)

    # Explicit, generous margin rather than matplotlib's default autoscale (~5% of the data
    # range): with only the 3 decoders' own CIs setting panel a's range, that default margin
    # leaves EEGNet's own label -- anchored just past its ci_low, the panel's lowest point --
    # close enough to the axis's own bottom spine that the label sits on top of it.
    y_lo, y_hi = min(prior_share_bounds), max(prior_share_bounds)
    y_pad = 0.22 * (y_hi - y_lo)
    ax_a.set_ylim(y_lo - y_pad, y_hi + y_pad)

    # Padded from just the 3 decoders' own x values: a wider fixed padding would leave the 3
    # decoders bunched in the middle of a mostly empty axis.
    xmin, xmax = min(decoder_x.values()) - 4, max(decoder_x.values()) + 4
    line_x = np.linspace(xmin, xmax, 50)
    ax_b.plot(line_x, 100 - line_x, ls=(0, (4, 3)), lw=0.9, color=MUTED, zorder=1)
    # Anchored in the panel's own empty top-right corner (axes fraction, not a data point on
    # or near the line) rather than at the line's own right end: a label placed there and
    # offset upward still sat inside the line's swept area, since the line's slope is steep
    # enough in data units that a modest vertical offset does not clear it over the label's
    # own width. The corner is empty in every rendering because every point in this panel
    # sits at 0-40% on the y-axis, well under the line there.
    ax_b.text(0.985, 0.95, "ceiling: phantom agreement\ncannot exceed the neural error rate",
              transform=ax_b.transAxes, fontsize=6.2, color=MUTED, ha="right", va="top")

    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylabel("prior share of posterior\ndisplacement (%)", fontsize=7.4)
    ax_a.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    # Top-left corner: empty in every rendering (the leftmost decoder's own dot and label sit
    # well to the right of it); in panel c, EEGNet's own error bar would cross it.
    ax_a.text(0.02, 0.97,
              "prior influence persisted under both stronger decoders;\n"
              "paired differences in eTable 36",
              transform=ax_a.transAxes, fontsize=6.4, color=MUTED, ha="left", va="top")

    ax_b.set_ylabel("phantom agreement (%)", fontsize=7.4)
    ax_b.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax_b.set_axisbelow(True)

    ax_c.set_ylabel("rescue rate (%)", fontsize=7.4)
    ax_c.set_xlabel("neural-only accuracy (%), participant-weighted", fontsize=7.4)
    ax_c.grid(color="#E5E7EB", lw=0.6, zorder=0)
    ax_c.set_axisbelow(True)

    handles = [
        Line2D([], [], marker=DECODER_MARKER[n], color=ACCENT3, mfc=ACCENT3, mec="white",
               mew=0.7, ls="", ms=6.5, label=DECODER_LABEL[n])
        for n in DECODER_ORDER
    ] + [
        Line2D([], [], marker="o", color=INK, mfc=INK, mec="white", mew=0.7, ls="", ms=6.5,
               label="rescue rate (charcoal, panel c)"),
    ]
    legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=-0.010,
                 fontsize=6.4)

    fig.subplots_adjust(hspace=0.16, top=0.97, bottom=0.145, left=0.13, right=0.97)
    save(fig, str(path))


def main() -> None:
    revision_digest = json.loads((OUTPUT / "revision_digest.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    efigure5(revision_digest, FIGURES / "eFigure5")
    efigure6(revision_digest, FIGURES / "eFigure6")
    print("revision figures written to", FIGURES)


if __name__ == "__main__":
    main()
