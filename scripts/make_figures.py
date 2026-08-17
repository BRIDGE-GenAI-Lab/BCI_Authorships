"""Publication figures for the authorship study, in the BrainGate neural-decoding style."""

from __future__ import annotations

import json
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withStroke

PROJECT = Path(__file__).resolve().parents[1]
KIT = Path.home() / ".claude" / "skills" / "hochberg-figure-style" / "scripts"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(KIT))

from hochberg_kit import (  # noqa: E402
    ACCENT3,
    CONTEXT,
    INK,
    KEY,
    MUTED,
    apply_style,
    legend_below,
    panel,
    save,
    swatch,
)

from authorship.attribution import attribution_row, fuse  # noqa: E402
from authorship.calibration import calibrated_fusion_frame  # noqa: E402
from authorship.grid import FLASH_CHANNELS, SYMBOLS, parse_symbol_channel, symbol_index  # noqa: E402
from authorship.priors import family_of  # noqa: E402
from run_stats import (  # noqa: E402
    BETA_GRID,
    CLUSTER,
    N_BOOTSTRAP,
    PRIMARY_BETA,
    PRIMARY_PRIOR,
    SEED,
    calibrated_attribution_frame,
    cast_binary_columns,
    full_prior_frame,
    model_frame,
    participant_mean,
    primary_prior_frame,
)

OUTPUT = PROJECT / "output"
FIGURES = OUTPUT / "figures"
LADDER_ORDER = [
    "uniform", "ngram5", "ngram5_kn", "ngram5_wiki_kn", "gpt2", "gpt2-large",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-14B", "Qwen/Qwen3.5-27B",
    "Qwen/Qwen2.5-32B", "Qwen/Qwen3.6-35B-A3B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it", "google/gemma-2-9b", "google/gemma-2-27b", "google/gemma-4-12b-it",
    "mistralai/Mistral-7B-v0.3", "mistralai/Mixtral-8x7B-v0.1",
    "deepseek-ai/DeepSeek-V2-Lite",
    "allenai/OLMo-2-1124-7B-Instruct",
    "openai/gpt-oss-20b",
    "EleutherAI/pythia-12b", "state-spaces/mamba-2.8b-hf",
    "google/recurrentgemma-2b",
]
LADDER_LABEL = {
    "uniform": "Uniform",
    "ngram5": "5-gram",
    "ngram5_kn": "5-gram KN",
    "ngram5_wiki_kn": "5-gram KN\n(WikiText)",
    "gpt2": "GPT-2 124M",
    "gpt2-large": "GPT-2 774M",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5 1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5 3B",
    "Qwen/Qwen2.5-14B": "Qwen2.5 14B",
    "Qwen/Qwen3.5-27B": "Qwen3.5 27B",
    "Qwen/Qwen2.5-32B": "Qwen2.5 32B",
    "Qwen/Qwen3.6-35B-A3B": "Qwen3.6 35B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1 8B",
    "google/gemma-2-2b-it": "Gemma-2 2B",
    "google/gemma-2-9b": "Gemma-2 9B",
    "google/gemma-2-27b": "Gemma-2 27B",
    "google/gemma-4-12b-it": "Gemma-4 12B",
    "mistralai/Mistral-7B-v0.3": "Mistral 7B",
    "mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B",
    "deepseek-ai/DeepSeek-V2-Lite": "DeepSeek-V2-Lite",
    "allenai/OLMo-2-1124-7B-Instruct": "OLMo-2 7B",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "EleutherAI/pythia-12b": "Pythia 12B",
    "state-spaces/mamba-2.8b-hf": "Mamba 2.8B",
    "google/recurrentgemma-2b": "RecurrentGemma 2B",
}
# Architecture family, the only thing colour encodes in Figure 3. Four of these used to be a
# reserved colour from the manuscript's own semantic palette: Qwen was drawn in KEY, the blue
# that means neural evidence in Figures 1, 2 and 4; Llama in ACCENT3, the vermilion that means
# the language-model prior and, in Figure 2 and eFigure 1, phantom agreement; DeepSeek in the
# kit's ERROR red, which is reserved for genuine negative outcomes and stands for no outcome
# here; and Pythia in a blue 5.4 CIEDE2000 units from KEY, which is indistinguishable from it
# at this marker size. A reader who has learned that blue is the participant's neural evidence
# should not meet it again as the name of a model family. The replacements were chosen against
# a measured constraint rather than by eye, and tests/test_make_figures_palette.py holds both
# halves of it: every family colour is at least 17 CIEDE2000 units from KEY, ACCENT3, INK,
# ERROR, CONTEXT and MUTED, and the eleven circle-marker families that share panel a's legend
# are at least 14 units from each other. The classical 5-grams sit closer than that to GPT-OSS
# (8.3) and Mamba (10.6), which is allowed because they are the only priors drawn as diamonds
# and are labelled where they sit rather than decoded from the legend. The uniform null keeps
# CONTEXT deliberately, because grey is the reference role in every figure and a prior that
# cannot displace the neural posterior is exactly reference material.
FAMILY_COLOR = {
    "null": CONTEXT,
    "classical": "#6A4C93",
    "gpt2": "#117A8B",
    "qwen": "#7A4A10",
    "llama": "#77700F",
    "gemma": "#1B7340",
    "mistral": "#A61E4D",
    "deepseek": "#A8348E",
    "olmo": "#B8860B",
    "gptoss": "#4B3F72",
    "pythia": "#3C9A3C",
    "mamba": "#7B2CBF",
    "recurrentgemma": "#00B0B9",
}
# Proper display names for the Figure 3 family legend: f.capitalize() mangles multi-word or
# stylized family names (Gpt2, Gptoss, Deepseek, Olmo, Recurrentgemma) into forms that don't
# match this same figure's own tick labels (GPT-2 124M, GPT-OSS 20B, DeepSeek-V2-Lite,
# OLMo-2 7B, RecurrentGemma 2B).
FAMILY_LABEL = {
    "null": "Uniform",
    "classical": "Classical",
    "gpt2": "GPT-2",
    "qwen": "Qwen",
    "llama": "Llama",
    "gemma": "Gemma",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "olmo": "OLMo",
    "gptoss": "GPT-OSS",
    "pythia": "Pythia",
    "mamba": "Mamba",
    "recurrentgemma": "RecurrentGemma",
}
# Base versus instruction-tuned status for the 21 neural priors, the fill encoding in Figure 3.
# SOURCE: eTable 19, the full prior-ladder inventory in manuscript/supplement.md, whose Type
# column carries this classification per prior and whose note records that 14 of the 21 neural
# priors are base models and 7 are instruction-tuned; scripts/assemble_manuscript.py repeats
# the same split in prose. eTable 19 is the only place that classification existed before this
# figure needed it, so if the ladder or that table changes, this set has to change with it:
# tests/test_make_figures_ladder.py re-reads eTable 19 and fails when the two disagree.
INSTRUCTION_TUNED = frozenset({
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-4-12b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "openai/gpt-oss-20b",
})
# What the archive's short condition codes mean, for eFigure 2's axis.
# SOURCE: eTable 8 in manuscript/supplement.md, the manuscript's own per-condition table and
# the only place these expansions are written down; its Condition column carries exactly these
# names. Nothing here is invented: tests/test_make_figures_efigures.py re-reads eTable 8 and
# fails if the two ever disagree, so a renamed condition cannot reach a figure axis unnoticed.
CONDITION_LABEL = {
    "Static": "Static",
    "CB": "Checkerboard",
    "CBCol": "Checkerboard column",
    "RC": "Row-column",
    "DynBigram": "Dynamic stopping with bigram model",
    "Wet": "Wet electrodes",
    "Dyn": "Dynamic stopping",
    "Dry": "Dry electrodes",
}
# The four source studies, in the order eTable 7 lists them.
STUDY_LABEL = {
    "StudyB": "Study B",
    "StudyF": "Study F",
    "StudyL": "Study L",
    "StudyN": "Study N",
}


def load():
    digest = json.loads((OUTPUT / "stats_digest.json").read_text())
    attribution = pd.read_parquet(OUTPUT / "intermediate" / "attribution.parquet")
    return digest, attribution


def bars_with_ci(ax, labels, estimates, lows, highs, color=KEY, colors=None, ylabel="", rotation=0):
    positions = np.arange(len(labels))
    ax.bar(positions, estimates, color=(colors if colors is not None else color), width=0.62, zorder=2)
    ax.errorbar(
        positions,
        estimates,
        yerr=[np.array(estimates) - np.array(lows), np.array(highs) - np.array(estimates)],
        fmt="none",
        ecolor=INK,
        elinewidth=1.0,
        capsize=2.5,
        zorder=3,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha="center" if rotation == 0 else "right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _schematic_block(ax, xy, wh, n, title):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=INK, facecolor="white", zorder=1,
    )
    ax.add_patch(box)
    ax.text(x + 0.028, y + h - 0.032, str(n), fontsize=11, fontweight="bold", color="white",
             va="center", ha="center", zorder=2,
             bbox=dict(boxstyle="circle,pad=0.30", fc=INK, ec="none"))
    ax.text(x + 0.052, y + h - 0.03, title, fontsize=9.5, fontweight="bold", color=INK,
             va="top", ha="left")
    return x, y, w, h


# Drawn at the journal's 183 mm double column rather than wider. This figure used to be 10.2
# inches across, which renders 205 mm, and a journal reproducing it at 183 mm shrinks every
# glyph by a ninth: the 5.6 pt speller-grid letters landed at 5.00 pt on the page, exactly on
# the floor of the 5-7 pt band Nature asks for at final size and with nothing guarding it.
# That change was to the width alone; the wrap budgets that depend on it are rescaled by
# _budget and the speller grid's own type was lifted to this figure's 6.0 pt floor. The height
# has since changed too, but not in a way any hand-placed y can feel: see FIGURE1_Y_FLOOR.
# Nothing is drawn below FIGURE1_Y_FLOOR, and save()'s bbox_inches="tight" measures the axes
# box, not the ink inside it, so every fraction of this schematic's 0-1 space left empty at the
# bottom shipped as blank paper: the card row this panel d replaced ended at y=0.078, which put
# half an inch of white under the figure. Cropping the y-axis to this floor and taking the same
# fraction off the canvas height together hold inches-per-unit-of-y fixed at what it has always
# been, so every hand-placed y above the floor keeps the physical size and spacing it was tuned
# at. That is what FIGURE1_VERTICAL_REFERENCE_IN is: the canvas height the vertical layout was
# measured on, which is no longer the canvas height. Raise the floor to crop further; the two
# places that convert points or inches into y units read the reference, never FIGURE1_SIZE[1].
FIGURE1_Y_FLOOR = 0.1075
FIGURE1_VERTICAL_REFERENCE_IN = 7.6
FIGURE1_SIZE = (8.9, FIGURE1_VERTICAL_REFERENCE_IN * (1 - FIGURE1_Y_FLOOR))
FIGURE1_WRAP_REFERENCE_IN = 10.2  # the width every _flow character budget below was measured at
# Figure 1's worked example is auto-selected by a fixed rule (closest to the median prior
# share), so a change in the underlying data can silently swap it for a different selection
# while the prose caption in scripts/assemble_manuscript.py keeps describing the old one.
# That is exactly how the Figure 4 caption drifted. figure1() checks the selection it picked
# against this record of what the caption says and refuses to render if they disagree.
FIGURE1_CAPTION_EXAMPLE = {
    "context": "ENOU",
    "target": "G",
    "neural_only": "B",
    "phantom_agreement": True,
}


def _wrap(text, width):
    """Hard-wrap schematic body text to a character budget. Figure 1 places its text in
    hand-positioned boxes with no layout engine to reflow it, so the wrap width is what
    keeps a sentence inside its own card instead of running across a neighbour."""
    return textwrap.fill(text, width=width)


def _budget(chars, size, reference):
    """Rescale a character budget measured on one canvas to the canvas actually in use.

    A budget is really a width. Figures 1 and 3 place text by hand at a fixed point size, so
    narrowing the canvas narrows every card and box while leaving the text exactly as long as
    it was, and the text runs out of its own container. Both figures were narrowed to the
    journal's 183 mm double column after being drawn 205 and 227 mm wide, which is where these
    budgets were measured; tying them to the canvas rather than pinning the numbers means the
    next width change reflows the text instead of overflowing it.
    """
    return max(1, round(chars * size[0] / reference))


def _flow(ax, x, y, text, fontsize, width, weight="normal", colour=INK, gap=0.006):
    """Draw one wrapped block at (x, y) in Figure 1's 0-1 schematic space and return the y
    the next block should start at, so stacked text cannot silently overlap when a wrapped
    line count changes. `width` is a character budget measured at FIGURE1_WRAP_REFERENCE_IN
    inches wide and rescaled to the current canvas by _budget."""
    body = _wrap(text, _budget(width, FIGURE1_SIZE, FIGURE1_WRAP_REFERENCE_IN))
    ax.text(x, y, body, fontsize=fontsize, fontweight=weight, color=colour, ha="left",
            va="top", zorder=3)
    line_height = fontsize * 1.25 / 72 / FIGURE1_VERTICAL_REFERENCE_IN
    return y - (body.count("\n") + 1) * line_height - gap


def _p_value(value):
    """AMA style: no leading zero, floored at .001, and three decimals between .001 and .01
    rather than two, which rounded the position-in-word adjusted P of .0045 to ".00"."""
    if value < 0.001:
        return "<.001"
    digits = 3 if value < 0.01 else 2
    return f"{value:.{digits}f}".lstrip("0")


def _p_text(value, label="P"):
    """A whole P value clause, so a floored value reads "P < .001" rather than "P = <.001"."""
    formatted = _p_value(value)
    return f"{label} < .001" if formatted.startswith("<") else f"{label} = {formatted}"


def participant_bootstrap(frame, column, n_replicates=N_BOOTSTRAP, seed=SEED):
    """95% participant-cluster bootstrap interval for a participant-weighted mean.

    run_stats.cluster_bootstrap is the study's bootstrap: resample participants with
    replacement, rebuild the frame, recompute the statistic. For a participant-weighted mean
    that rebuild is redundant work, because participant_mean regroups by participant and a
    participant drawn twice contributes exactly one group mean either way. It is also slow
    enough (about ten seconds a call on the primary frame) that the per-exponent and per-cell
    intervals the supplementary figures need would add minutes to every render.

    This is the same resampling over the per-participant means, with the same generator, seed
    and draw sequence, so it returns the same numbers rather than a second convention.
    _fusion_exponent_points() and efigure2() check that claim against intervals and estimates
    run_stats already stored in output/stats_digest.json and refuse to render if they differ.
    Only valid for statistics of participant_mean's form.
    """
    per_participant = frame.groupby(CLUSTER)[column].mean()
    values = per_participant.to_dict()
    clusters = frame[CLUSTER].unique()
    rng = np.random.default_rng(seed)
    draws = np.empty(n_replicates)
    for replicate in range(n_replicates):
        drawn = set(rng.choice(clusters, size=len(clusters), replace=True).tolist())
        draws[replicate] = np.mean([values[name] for name in drawn])
    return {
        "estimate": float(per_participant.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def prior_share_ecdf(share):
    """Sorted per-selection prior share of posterior displacement plus its ECDF, mean and
    median (sort 1 - ncf, cumulative rank / n).

    Both Figure 1's inset and eFigure 3 draw this distribution and both call this, so the two
    cannot drift apart in how it is derived. They deliberately differ in what they derive it
    from: Figure 1's inset takes the calibrated per-selection shares from
    calibrated_primary_selections(), the manuscript's principal basis, while eFigure 3 takes
    the raw, uncalibrated shares in attribution.parquet, the basis eTable 14 and the whole
    secondary and sensitivity family are computed on. Each figure says which basis it used.
    """
    share = np.sort(np.asarray(share, dtype=float))
    cumulative = np.arange(1, len(share) + 1) / len(share)
    return share, cumulative, float(np.mean(share)), float(np.median(share))


def calibrated_primary_selections():
    """Per-selection calibrated fusion for the primary prior, with attribution measures.

    Reproduces run_stats.calibrated_primary_analysis()'s per-row procedure exactly
    (primary_prior_frame -> grouped 5-fold temperature calibration on participants ->
    fuse -> attribution_row) but keeps every row rather than only the participant-weighted
    bootstrap means the digest stores. Figure 1's worked example and its distribution inset,
    and eFigure 3's calibrated curve, all need per-selection values on the manuscript's
    principal (calibrated) basis, which output/intermediate/attribution.parquet does not
    contain: that file holds the raw, uncalibrated fusion now reported only as a labeled
    sensitivity comparison. Carries CLUSTER so callers can take a participant-weighted mean
    of any column here with run_stats.participant_mean directly, with no second join back to
    selections.parquet needed.
    """
    selections = pd.read_parquet(OUTPUT / "intermediate" / "selections.parquet")
    priors = pd.read_parquet(OUTPUT / "intermediate" / "priors.parquet")
    frame = primary_prior_frame(selections, priors, PRIMARY_PRIOR)
    calibrated = calibrated_fusion_frame(frame, group_column=CLUSTER, n_folds=5, seed=0)

    measures, fused_posteriors = [], []
    for row in calibrated.itertuples(index=False):
        neural = np.asarray(row.p_neural_calibrated)
        prior = np.asarray(row.p_lm_calibrated)
        fused = fuse(neural, prior)
        measures.append(attribution_row(neural, prior, fused, int(row.target_index)))
        fused_posteriors.append(fused)

    table = pd.DataFrame(measures, index=calibrated.index)
    table["p_neural_calibrated"] = list(calibrated["p_neural_calibrated"])
    table["p_lm_calibrated"] = list(calibrated["p_lm_calibrated"])
    table["p_fused"] = fused_posteriors
    table["prior_share"] = 1.0 - table["ncf"]
    context = selections.loc[
        table.index,
        ["context_prefix", "target_symbol", "intended_phrase", "study", "position_in_phrase",
         CLUSTER],
    ]
    return pd.concat([context, table], axis=1), selections


def calibrated_fusion_exponent_selections(beta_grid=BETA_GRID):
    """Independent recomputation of the calibrated fusion-exponent grid, for eFigure 1's
    calibrated panel self-verification guard.

    Like calibrated_primary_selections(), this reads selections/priors fresh and refits the
    held-out per-source calibration temperatures itself via calibrated_fusion_frame, rather
    than importing run_stats.py's already-built calibrated_primary. The calibrated grid is
    not in attribution.parquet (that file holds only the raw, uncalibrated fusion), so there
    is no on-disk source of truth to read here the way _fusion_exponent_points reads the raw
    grid off attribution.parquet; refitting calibration from scratch is what keeps this check
    independent of run_stats.py's own in-memory state instead of re-reading the same stored
    digest number back at itself.

    Calibration is fit once (beta only enters at the fusion step, exactly as
    run_stats.calibrated_fusion_exponent_points reuses its own calibrated_primary), then
    re-fused at each beta in beta_grid. Returns {beta: DataFrame} with CLUSTER, ncf and
    phantom_agreement columns, one row per selection.
    """
    selections = pd.read_parquet(OUTPUT / "intermediate" / "selections.parquet")
    priors = pd.read_parquet(OUTPUT / "intermediate" / "priors.parquet")
    frame = primary_prior_frame(selections, priors, PRIMARY_PRIOR)
    calibrated = calibrated_fusion_frame(frame, group_column=CLUSTER, n_folds=5, seed=0)

    per_beta = {}
    for beta in beta_grid:
        ncf_values, phantom_values = [], []
        for row in calibrated.itertuples(index=False):
            neural = np.asarray(row.p_neural_calibrated)
            prior = np.asarray(row.p_lm_calibrated)
            fused = fuse(neural, prior, beta=beta)
            result = attribution_row(neural, prior, fused, int(row.target_index))
            ncf_values.append(result["ncf"])
            phantom_values.append(result["phantom_agreement"])
        per_beta[beta] = pd.DataFrame({
            CLUSTER: calibrated[CLUSTER].to_numpy(),
            "ncf": ncf_values,
            "phantom_agreement": phantom_values,
        })
    return per_beta


def _draw_speller_grid(ax, x, y, cell_w, cell_h, flash_column, target_symbol):
    """The real 6x6 BigP3BCI grid (authorship.grid.FLASH_CHANNELS), with one flashed
    column shaded and the intended symbol outlined."""
    for channel in FLASH_CHANNELS:
        symbol, row, column = parse_symbol_channel(channel)
        left = x + (column - 1) * cell_w
        bottom = y + (6 - row) * cell_h
        flashed = column == flash_column
        ax.add_patch(plt.Rectangle(
            (left, bottom), cell_w, cell_h, facecolor="#EDEFF2" if flashed else "white",
            edgecolor="#C9CDD2", linewidth=0.6, zorder=1,
        ))
        label = "Sp" if symbol == " " else symbol
        # 6.0 pt, this figure's floor everywhere else. These 36 letters were its one smaller
        # size, so they were the type a journal's reproduction would starve first.
        ax.text(left + cell_w / 2, bottom + cell_h / 2, label, fontsize=6.0,
                color=INK if flashed else MUTED, ha="center", va="center", zorder=2)
        if symbol == target_symbol:
            ax.add_patch(plt.Rectangle(
                (left, bottom), cell_w, cell_h, facecolor="none", edgecolor=KEY,
                linewidth=1.6, zorder=3,
            ))


def _draw_context_tiles(ax, x, y, cell_w, cell_h, prefix):
    """The intended preceding characters in the word (oracle context; the primary analysis's
    basis), plus the open slot the prior scores."""
    for index, character in enumerate(prefix):
        left = x + index * cell_w
        ax.add_patch(plt.Rectangle((left, y), cell_w, cell_h, facecolor="white",
                                    edgecolor=ACCENT3, linewidth=0.9, zorder=2))
        ax.text(left + cell_w / 2, y + cell_h / 2, character, fontsize=8, color=INK,
                ha="center", va="center", zorder=3)
    left = x + len(prefix) * cell_w
    ax.add_patch(plt.Rectangle((left, y), cell_w, cell_h, facecolor="white",
                                edgecolor=ACCENT3, linewidth=0.9, ls=(0, (2, 1.6)), zorder=2))
    ax.text(left + cell_w / 2, y + cell_h / 2, "?", fontsize=8, color=ACCENT3,
            ha="center", va="center", zorder=3)


def _stage_box(ax, x, y, w, h, accent, title, body):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0.9, edgecolor="#C9CDD2", facecolor="white", zorder=2,
    ))
    ax.add_patch(plt.Rectangle((x, y), 0.006, h, facecolor=accent, edgecolor="none", zorder=3))
    ax.text(x + 0.018, y + h - 0.018, title, fontsize=8.2, fontweight="bold", color=accent,
            ha="left", va="top", zorder=3)
    ax.text(x + 0.018, y + h - 0.045, body, fontsize=7.0, color=INK, ha="left", va="top",
            zorder=3)


def _arrow(ax, start, end, color=INK):
    ax.annotate("", xy=end, xytext=start, xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=color, shrinkA=0, shrinkB=0),
                zorder=4)


def figure1(digest):
    """Signal to symbol: how each emitted character is attributed between the participant's
    neural evidence and the language-model prior.

    Colour grammar, consistent across the figure and with Figures 2 and 3: KEY (deep blue)
    is neural evidence, ACCENT3 (vermilion) is the language prior, INK (charcoal) is the
    fused decision, CONTEXT (grey) is reference/no-effect material.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=FIGURE1_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(FIGURE1_Y_FLOOR, 1)
    ax.axis("off")

    calibrated, _ = calibrated_primary_selections()

    # ---------------------------------------------------------------- worked example
    # Chosen by a rule fixed in advance, not by inspection: among selections with at least
    # four characters of preceding context (figure4's own filter, so the prior has real
    # linguistic context to condition on), the one whose prior share of posterior
    # displacement sits closest to the cohort median. That is the opposite of picking a
    # dramatic case; it is the median-attribution selection.
    eligible = calibrated[calibrated["context_prefix"].str.len() >= 4]
    median_share = float(np.nanmedian(calibrated["prior_share"]))
    example = eligible.iloc[
        (eligible["prior_share"] - median_share).abs().to_numpy().argmin()
    ]
    neural = np.asarray(example["p_neural_calibrated"])
    prior = np.asarray(example["p_lm_calibrated"])
    fused = np.asarray(example["p_fused"])
    label_of = lambda index: "SP" if SYMBOLS[index] == " " else SYMBOLS[index]  # noqa: E731

    context_prefix = example["context_prefix"]
    target_symbol = example["target_symbol"]
    target_index = symbol_index(target_symbol)
    neural_pick = int(neural.argmax())
    prior_pick = int(prior.argmax())
    emitted_index = int(fused.argmax())
    fused_correct = emitted_index == target_index
    is_phantom = bool(example["phantom_agreement"])
    # The eight candidates panel b plots, needed here as well so panel a's readout can say
    # whether a source's own top choice is one of them rather than asserting it silently.
    order = np.argsort(fused)[::-1][:8]

    described = FIGURE1_CAPTION_EXAMPLE
    observed = {
        "context": context_prefix,
        "target": target_symbol,
        "neural_only": label_of(neural_pick),
        "phantom_agreement": is_phantom,
    }
    if observed != described:
        raise RuntimeError(
            "Figure 1's auto-selected worked example changed from "
            f"{described} to {observed}. Update the Figure 1 caption in "
            "scripts/assemble_manuscript.py and the FIGURE1_CAPTION_EXAMPLE record in "
            "this file before re-rendering, so the caption cannot describe a selection "
            "the figure no longer shows."
        )

    ladder = digest["sensitivity"]["prior_ladder"]

    # ---------------------------------------------------------------- panel a: pipeline
    ax.text(0.010, 0.996, "a", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    neural_y, prior_y, fuse_y = 0.895, 0.747, 0.821

    cell_w = 0.0140
    cell_h = cell_w * FIGURE1_SIZE[0] / FIGURE1_VERTICAL_REFERENCE_IN
    grid_x, grid_y = 0.046, 0.845
    # The shaded column has to be the one that actually contains this example's intended
    # symbol, because the evoked-response trace beside it is labelled "target flash".
    # Hardcoding a column would quietly mislabel the figure if the example ever changes.
    target_column = parse_symbol_channel(FLASH_CHANNELS[target_index])[2]
    _draw_speller_grid(ax, grid_x, grid_y, cell_w, cell_h, flash_column=target_column,
                        target_symbol=target_symbol)
    ax.text(grid_x + 3 * cell_w, grid_y + 6 * cell_h + 0.010,
            "6 x 6 speller grid", fontsize=7.4, fontweight="bold", color=INK,
            ha="center", va="bottom")
    ax.text(grid_x + 3 * cell_w, grid_y - 0.010,
            f"one flash shaded,\nintended '{target_symbol}' outlined",
            fontsize=6.4, color=MUTED, ha="center", va="top")

    erp = ax.inset_axes([0.150, 0.860, 0.105, 0.090], transform=ax.transData)
    milliseconds = np.linspace(-100, 600, 400)
    target_erp = (5.0 * np.exp(-0.5 * ((milliseconds - 320) / 85) ** 2)
                  - 2.0 * np.exp(-0.5 * ((milliseconds - 160) / 45) ** 2))
    nontarget_erp = (0.9 * np.exp(-0.5 * ((milliseconds - 240) / 130) ** 2)
                     - 0.7 * np.exp(-0.5 * ((milliseconds - 150) / 55) ** 2))
    erp.plot(milliseconds, nontarget_erp, color=CONTEXT, lw=1.4)
    erp.plot(milliseconds, target_erp, color=KEY, lw=1.8)
    erp.axvline(0, color="#C9CDD2", lw=0.7)
    erp.annotate("P300", xy=(320, 5.0), xytext=(375, 5.4), fontsize=6.4, color=KEY,
                 fontweight="bold", arrowprops=dict(arrowstyle="->", lw=0.7, color=KEY))
    erp.text(-90, 4.3, "target flash", fontsize=6.0, color=KEY, va="bottom")
    erp.text(-85, -3.0, "non-target flash", fontsize=6.0, color=CONTEXT, va="bottom")
    erp.set_xlim(-100, 600)
    erp.set_ylim(-3.4, 6.6)
    erp.set_yticks([])
    erp.set_xticks([0, 300, 600])
    erp.tick_params(labelsize=6.0, length=2)
    erp.set_xlabel("ms from flash", fontsize=6.4, labelpad=1.0)
    for side in ("top", "right", "left"):
        erp.spines[side].set_visible(False)
    # Both sit above the inset's own top edge (0.860 + 0.090). "schematic" used to be anchored
    # exactly on that edge, and an inset axes is drawn at zorder 5 over the schematic axis's
    # zorder-3 text, so its opaque background painted the word out completely: removing the
    # text changed no pixel of the rendered figure. The higher zorder is deliberate belt and
    # braces - if this label is ever pushed back under the inset it will now be drawn over the
    # trace, which is visibly wrong, rather than silently disappearing again.
    ax.text(0.2025, 0.9685, "evoked response", fontsize=7.4, fontweight="bold",
            color=INK, ha="center", va="bottom", zorder=6)
    ax.text(0.2025, 0.9655, "schematic", fontsize=6.4, color=MUTED, ha="center", va="top",
            style="italic", zorder=6)

    _arrow(ax, (0.268, neural_y), (0.318, neural_y))
    _stage_box(
        ax, 0.325, 0.845, 0.265, 0.100, KEY, "Neural evidence",
        "Posterior over 36 symbols,\ncalibration-trained decoder,\nheld-out temperature",
    )

    tiles_x, tile_w = 0.046, 0.020
    _draw_context_tiles(ax, tiles_x, 0.724, tile_w, 0.046, context_prefix)
    tiles_center = tiles_x + tile_w * (len(context_prefix) + 1) / 2
    ax.text(tiles_center, 0.784, "intended preceding characters", fontsize=7.4, fontweight="bold",
            color=INK, ha="center", va="bottom")
    ax.text(tiles_center, 0.712, "(primary analysis basis)", fontsize=6.4, color=MUTED,
            ha="center", va="top")
    _arrow(ax, (tiles_x + tile_w * (len(context_prefix) + 1) + 0.012, prior_y), (0.318, prior_y),
           color=ACCENT3)
    # "priors", not "models": 4 of the ladder's entries are a uniform null and 3 n-gram
    # priors, which are not neural models, and the ladder count would otherwise appear to
    # contradict the "21 neural priors" in the parameter-count card below.
    _stage_box(
        ax, 0.325, 0.697, 0.265, 0.100, ACCENT3, "Language-model prior",
        "Next-symbol distribution,\n"
        f"{len(ladder)}-prior ladder,\nheld-out temperature",
    )

    _arrow(ax, (0.596, neural_y - 0.005), (0.660, fuse_y + 0.026), color=KEY)
    _arrow(ax, (0.596, prior_y + 0.005), (0.660, fuse_y - 0.026), color=ACCENT3)
    ax.add_patch(plt.Circle((0.684, fuse_y), 0.024, facecolor="white", edgecolor=INK,
                             linewidth=1.4, zorder=3))
    ax.text(0.684, fuse_y, "×", fontsize=13, fontweight="bold", color=INK, ha="center",
            va="center", zorder=4)
    # Dropped clear of the incoming prior arrow, whose shaft used to run through the first
    # word of this label on its way up to the fusion node.
    ax.text(0.684, fuse_y - 0.060, "fuse, then renormalize", fontsize=7.2, fontweight="bold",
            color=INK, ha="center", va="top")
    ax.text(0.684, fuse_y - 0.082, "held-out calibrated fusion", fontsize=6.6, color=MUTED,
            ha="center", va="top")

    _arrow(ax, (0.712, fuse_y), (0.775, fuse_y))
    ax.add_patch(FancyBboxPatch(
        (0.784, fuse_y - 0.035), 0.052, 0.070, boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=1.4, edgecolor=INK, facecolor="white", zorder=3,
    ))
    ax.text(0.810, fuse_y, label_of(emitted_index), fontsize=15, fontweight="bold", color=INK,
            ha="center", va="center", zorder=4)
    ax.text(0.810, fuse_y + 0.045, "emitted symbol", fontsize=7.4, fontweight="bold", color=INK,
            ha="center", va="bottom")

    # Both readout claims are derived, not asserted: the prior's own top choice is flagged
    # when it falls outside the eight candidates panel b plots (otherwise a reader cannot
    # check it against that panel), and the fused line states whether the emitted symbol
    # actually was the intended one rather than assuming this example stays a correct one.
    prior_note = "" if prior_pick in order else " (not in panel b)"
    ax.text(0.784, fuse_y - 0.048,
            f"neural evidence alone: '{label_of(neural_pick)}'\n"
            f"prior alone: '{label_of(prior_pick)}'{prior_note}\n"
            f"fused: '{label_of(emitted_index)}', "
            + ("the intended symbol" if fused_correct else "not the intended symbol"),
            fontsize=6.9, color=INK, ha="left", va="top")

    # ------------------------------------------- panel b: the three aligned distributions
    ax.text(0.010, 0.640, "b", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    # Wrapped rather than set on one line: at this width one line reached past panel b's own
    # three sub-panels and into panel c's letter.
    ax.text(0.036, 0.638,
            _wrap(f"A selection at the median prior share: after '{context_prefix}', "
                  f"intended '{target_symbol}'",
                  _budget(56, FIGURE1_SIZE, FIGURE1_WRAP_REFERENCE_IN)),
            fontsize=8.4, fontweight="bold", color=INK, ha="left", va="top")

    tick_labels = [label_of(index) for index in order]
    top = 0.585
    height, gap = 0.070, 0.014
    panels = [
        (neural, KEY, "neural evidence alone"),
        (prior, ACCENT3, "language-model prior alone"),
        (fused, INK, "fused posterior, held-out calibrated fusion"),
    ]
    ceiling = 1.18 * max(float(neural[order].max()), float(fused[order].max()))
    axes_b = []
    for index, (values, colour, label) in enumerate(panels):
        bottom = top - (index + 1) * height - index * gap
        sub = ax.inset_axes([0.045, bottom, 0.400, height], transform=ax.transData)
        sub.bar(np.arange(len(order)), values[order], width=0.66, color=colour, zorder=2)
        sub.set_xlim(-0.6, len(order) - 0.4)
        sub.set_ylim(0, ceiling)
        sub.set_yticks([0, 0.15, 0.30])
        sub.tick_params(labelsize=6.8, length=2)
        sub.grid(axis="y", color="#EDEFF2", lw=0.5, zorder=0)
        sub.set_axisbelow(True)
        sub.text(0.995, 0.86, label, transform=sub.transAxes, fontsize=7.4, fontweight="bold",
                 color=colour, ha="right", va="top")
        for side in ("top", "right"):
            sub.spines[side].set_visible(False)
        if index < 2:
            sub.set_xticks(np.arange(len(order)))
            sub.set_xticklabels([])
        axes_b.append(sub)

    # Anchored near the y-axis, not beside the bar: at the larger font this task raised
    # panel b's type to, text starting further right ran into "neural evidence alone"'s
    # own right-anchored label on the same row.
    # relpos anchors the arrow to the text's own bottom-left corner rather than
    # matplotlib's default center, which at this text's width sent the arrow's shaft
    # straight through the middle of the words on the way to a nearby bar.
    axes_b[0].annotate(
        f"its own top choice '{label_of(neural_pick)}'",
        xy=(list(order).index(neural_pick), float(neural[neural_pick])),
        xytext=(0.02, 0.99), textcoords="axes fraction", fontsize=7.0, color=KEY,
        ha="left", va="top",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=KEY, relpos=(0, 0), shrinkB=4),
    )
    uniform = 1.0 / len(SYMBOLS)
    axes_b[1].axhline(uniform, color=CONTEXT, ls=(0, (3, 2)), lw=0.9, zorder=3)
    axes_b[1].text(0.012, 0.96, "dashed line: uniform prior, 1/36", transform=axes_b[1].transAxes,
                   fontsize=6.8, color=CONTEXT, ha="left", va="top")
    # On a scale shared with the neural and fused panels the calibrated prior is a row of
    # near-identical slivers, so the difference the caption credits with the tie-break is not
    # readable off the bars. Print the two probabilities that decided it. A phantom agreement
    # guarantees the prior favoured the intended symbol over the neural posterior's own top
    # choice (the fused ranking flipped while the neural ranking did not), so the comparison
    # below holds by construction whenever it is drawn.
    if is_phantom and target_index in order and neural_pick in order:
        positions = list(order)
        for index, colour in ((target_index, ACCENT3), (neural_pick, ACCENT3)):
            axes_b[1].text(positions.index(index), float(prior[index]) + 0.008,
                           f"{prior[index]:.3f}", fontsize=6.6, color=colour, ha="center",
                           va="bottom", fontweight="bold")
        # Flush right, under the sub-panel's own label. On a narrower panel these two lines take
        # a wider share of it, and set flush left in the middle of the panel they had nowhere to
        # go: above, they ran into "language-model prior alone", and below, into the two
        # probabilities they are about. Flush right they clear the label vertically and the
        # probabilities horizontally, which the left edge could not do at once.
        axes_b[1].text(
            0.995, 0.60,
            f"near-uniform, but higher on '{label_of(target_index)}' than '"
            f"{label_of(neural_pick)}': that gap tips the fused posterior",
            transform=axes_b[1].transAxes, fontsize=6.8, color=ACCENT3, ha="right", va="top",
        )
    # Same reasoning as the neural-panel annotation above: anchored near the y-axis so it
    # clears "fused posterior, held-out calibrated fusion"'s own label at the larger font.
    axes_b[2].annotate(
        f"emitted '{label_of(emitted_index)}'",
        xy=(list(order).index(emitted_index), float(fused[emitted_index])),
        xytext=(0.02, 0.99), textcoords="axes fraction", fontsize=7.0, color=INK,
        ha="left", va="top",
        arrowprops=dict(arrowstyle="->", lw=0.7, color=INK, relpos=(0, 0), shrinkB=4),
    )
    axes_b[2].set_xticks(np.arange(len(order)))
    axes_b[2].set_xticklabels(tick_labels, fontsize=7.2)
    axes_b[2].set_xlabel("eight highest-ranked candidate symbols", fontsize=7.4, labelpad=1.5)
    axes_b[1].set_ylabel("probability", fontsize=7.4, labelpad=2.0)

    # ------------------------------------------------------- panel c: the two definitions
    # Widened to reach the row's own right margin once panel d's ECDF inset was deleted from
    # this row -- a box that stopped well short of that margin left a bare rectangle of unused
    # canvas to its right. The extra width goes to a second column rather than to wider
    # single-column text, since NCF and phantom agreement are two independent definitions that
    # do not need to share a reading order.
    ax.text(0.468, 0.640, "c", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    phantom_rate = 100 * digest["calibrated_primary_analysis"]["phantom_agreement"]["estimate"]

    col_top = 0.598
    left_x0, left_x1 = 0.506, 0.722
    right_x0, right_x1 = 0.746, 0.962

    # ---- neural contribution fraction (left column): a small annotated 0-to-1 axis in place
    # of the formula sentence, so the ratio this measure actually is can be read off a
    # position rather than parsed out of "d(neural) / [d(neural) + d(prior)]".
    cursor = _flow(ax, left_x0, col_top, "Neural contribution fraction (NCF)", 8.0, 38,
                   weight="bold", colour=KEY, gap=0.014)
    cursor = _flow(ax, left_x0, cursor, "KL-divergence share toward neural evidence", 6.6, 45,
                   colour=MUTED, gap=0.036)

    axis_x0, axis_x1 = left_x0 + 0.014, left_x1
    axis_y = cursor
    ax.plot([axis_x0, axis_x1], [axis_y, axis_y], color=CONTEXT, lw=1.3, zorder=2)
    for value, label in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        tick_x = axis_x0 + value * (axis_x1 - axis_x0)
        ax.plot([tick_x, tick_x], [axis_y - 0.009, axis_y + 0.009], color=CONTEXT, lw=1.1,
                zorder=2)
        ax.text(tick_x, axis_y - 0.021, label, fontsize=6.8, color=MUTED, ha="center",
                va="top")
    ax.text((axis_x0 + axis_x1) / 2, axis_y - 0.046, "0 = all prior   ·   1 = all neural",
            fontsize=6.4, color=MUTED, ha="center", va="top", style="italic")

    example_ncf_x = axis_x0 + example["ncf"] * (axis_x1 - axis_x0)
    ax.plot([example_ncf_x], [axis_y], marker="D", markersize=7.5, color=KEY,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    # Below the axis, not overlaid above the marker: at this example's NCF (near 1, close to
    # the "all neural" end) a callout placed above the marker ran into the subtitle line.
    left_bottom = _flow(
        ax, left_x0, axis_y - 0.082,
        f"This selection (marker): NCF {example['ncf']:.3f}, prior share "
        f"{100 * example['prior_share']:.1f}%.", 6.9, 40, weight="bold",
        colour=INK, gap=0.014)

    # ---- phantom agreement (right column): a compact "neural-only to fused" chip pair with
    # an arrow, in place of the full-sentence definition, echoing panel a's own pipeline
    # arrows. Starts at the same col_top as the NCF column rather than continuing that
    # column's cursor, since the two columns now sit side by side, not stacked.
    cursor = _flow(ax, right_x0, col_top, "Phantom agreement", 8.0, 38, weight="bold",
                   colour=ACCENT3, gap=0.016)
    cursor = _flow(ax, right_x0, cursor,
                   "Correct, but not neural's own top choice.", 6.6, 45,
                   colour=MUTED, gap=0.034)

    # Tags sit below each chip, not above: with the chips' vertical center placed close to
    # the definition text above them, a label crowded into that same gap collided with the
    # wrapped sentence on top of it.
    chip_y = cursor - 0.022
    chip_w, chip_h = 0.060, 0.054
    left_x, right_x = 0.780, 0.920
    for cx, letter, colour, tag in (
        (left_x, label_of(neural_pick), KEY, "neural-only"),
        (right_x, label_of(emitted_index), INK, "fused"),
    ):
        ax.add_patch(FancyBboxPatch(
            (cx - chip_w / 2, chip_y - chip_h / 2), chip_w, chip_h,
            boxstyle="round,pad=0.003,rounding_size=0.006", linewidth=1.2,
            edgecolor=colour, facecolor="white", zorder=3,
        ))
        ax.text(cx, chip_y, letter, fontsize=12, fontweight="bold", color=colour,
                ha="center", va="center", zorder=4)
        ax.text(cx, chip_y - chip_h / 2 - 0.015, tag, fontsize=6.6, color=colour,
                ha="center", va="top")
    _arrow(ax, (left_x + chip_w / 2 + 0.008, chip_y), (right_x - chip_w / 2 - 0.008, chip_y),
           color=ACCENT3 if is_phantom else CONTEXT)

    # Conditional on the measure this figure computed, not on the example that happened to
    # be selected: the median-prior-share rule guarantees neither correctness nor phantom
    # status, so a hardcoded caption here would go quietly wrong on a rerun.
    if is_phantom:
        caption = f"one of the {phantom_rate:.1f}% classified as phantom agreement"
        caption_colour = ACCENT3
    else:
        caption = f"not one of the {phantom_rate:.1f}% classified as phantom agreement"
        caption_colour = MUTED
    right_bottom = _flow(ax, right_x0, chip_y - chip_h / 2 - 0.050, caption, 6.6, 34,
                         colour=caption_colour, weight="bold")

    # The box is drawn last, around what the two columns actually came to, rather than at a
    # height fixed in advance. Pinned at 0.180 it was already too short for its own contents:
    # both columns ran past the border, and the phantom caption's last line ("classified") sat
    # entirely below it. Every quantity above is interpolated from the digest or the selected
    # example, so a line count here changes on a stats rerun and a fixed height cannot follow
    # it. Measuring instead means the border encloses its text by construction, and it is what
    # lets this box reach down toward panel b's own floor: the row used to end 0.15 above it,
    # which left an empty block a full inch tall across the right half of the figure.
    box_top, box_clearance = 0.610, 0.010
    box_bottom = min(left_bottom, right_bottom) - box_clearance
    ax.add_patch(FancyBboxPatch(
        (0.490, box_bottom), 0.488, box_top - box_bottom,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=0.9, edgecolor="#C9CDD2", facecolor="#FAFAFB", zorder=1,
    ))

    # mean_share/median_share are still needed below: panel d marks the median on the prior
    # share's own row, so the value survives even though the ECDF that used to visualize the
    # distribution (redundant with eFigure 3's fuller version on both bases) is gone.
    _, _, mean_share, median_share = prior_share_ecdf(calibrated["prior_share"])

    # --------------------------------------------------------- panel d: co-primary estimates
    # The two headline estimates as marks on a shared percentage axis rather than as numerals
    # in two boxes: a dot at the point estimate over a bar spanning its 95% interval. The two
    # measures become comparable by position, and each one's precision is something the reader
    # sees rather than reads off a printed interval. Same estimates, same intervals, same
    # digest keys the two result cards carried before; only the encoding changed.
    ax.text(0.012, 0.243, "d", fontsize=13, fontweight="bold", color=INK, ha="left", va="top")
    calibrated_primary = digest["calibrated_primary_analysis"]
    ncf = calibrated_primary["ncf"]
    phantom = calibrated_primary["phantom_agreement"]

    # NCF's complement, not NCF itself: on the reviewer's readability critique, "0.914" reads
    # as an opaque index while "8.6%" states in one glance how much of the fused posterior's
    # displacement the language model supplied. Taking a complement swaps an interval's ends,
    # so the prior share's lower bound comes from ncf's ci_high.
    rows = (
        ("Prior share of posterior displacement",
         100 * (1 - ncf["estimate"]), 100 * (1 - ncf["ci_high"]), 100 * (1 - ncf["ci_low"])),
        ("Phantom agreement",
         100 * phantom["estimate"], 100 * phantom["ci_low"], 100 * phantom["ci_high"]),
    )
    # Row centres are held in this figure's own 0-1 space, not the inset's, because each row's
    # name and value are drawn in that space alongside the dot: one number per row positions
    # all three, so a label cannot drift off the mark it belongs to.
    d_left, d_bottom, d_width, d_height = 0.355, 0.140, 0.405, 0.096
    row_centres = (0.212, 0.164)
    d_axis = ax.inset_axes([d_left, d_bottom, d_width, d_height], transform=ax.transData)
    d_axis.set_xlim(0, 12)
    d_axis.set_ylim(0, 1)
    d_axis.set_yticks([])
    ticks = [0, 2, 4, 6, 8, 10, 12]
    d_axis.set_xticks(ticks)
    d_axis.set_xticklabels([f"{tick}%" for tick in ticks])
    d_axis.tick_params(labelsize=6.8, length=2.5, pad=1.6)
    d_axis.grid(axis="x", color="#EDEFF2", lw=0.5, zorder=0)
    d_axis.set_axisbelow(True)
    for side in ("top", "right", "left"):
        d_axis.spines[side].set_visible(False)

    # Both rows are this study's co-primary outcomes and both are the language model's
    # contribution, so both take ACCENT3 -- the colour this figure already gives the prior --
    # and render identically. There is no primary/secondary distinction to draw.
    for (name, estimate, low, high), centre in zip(rows, row_centres):
        unit_y = (centre - d_bottom) / d_height
        d_axis.errorbar(
            [estimate], [unit_y], xerr=[[estimate - low], [high - estimate]], fmt="o",
            ms=6.5, color=ACCENT3, ecolor=ACCENT3, elinewidth=1.6, capsize=3.0,
            mfc=ACCENT3, mec="white", mew=0.8, zorder=4,
        )
        ax.text(0.036, centre, name, fontsize=7.4, fontweight="bold", color=INK,
                ha="left", va="center")
        ax.text(0.822, centre, f"{estimate:.1f}%", fontsize=11, fontweight="bold",
                color=ACCENT3, ha="right", va="center")
        ax.text(0.834, centre, f"95% CI {low:.1f} to {high:.1f}", fontsize=6.9, color=MUTED,
                ha="left", va="center")

    # The prior share's distribution is right-skewed enough that its mean sits well above its
    # median, which is why the card this row replaces printed the median in parentheses. A
    # tick on the row itself keeps that comparison visible instead of parenthetical. Phantom
    # agreement is a rate over selections and has no counterpart, so only this row carries one.
    median_unit_y = (row_centres[0] - d_bottom) / d_height
    d_axis.plot([100 * median_share] * 2, [median_unit_y - 0.13, median_unit_y + 0.13],
                color=CONTEXT, lw=1.4, solid_capstyle="butt", zorder=5)
    d_axis.text(100 * median_share, median_unit_y + 0.16,
                f"median {100 * median_share:.1f}%", fontsize=6.4, color=MUTED, ha="center",
                va="bottom")

    save(fig, str(FIGURES / "Figure1"))
    plt.close(fig)

    return {
        "context": context_prefix,
        "target": target_symbol,
        "neural_only": label_of(neural_pick),
        "prior_only": label_of(prior_pick),
        "emitted": label_of(emitted_index),
        "fused_correct": fused_correct,
        "ncf": float(example["ncf"]),
        "prior_share": float(example["prior_share"]),
        "phantom_agreement": is_phantom,
        "median_prior_share": median_share,
        "mean_prior_share": mean_share,
    }


FIGURE2_SIZE = (7.2, 5.2)
# One colour per measure, in every panel of the figure, matching the colours Figure 1 already
# gives the same two measures on its callout cards and in its definition panel: ACCENT3
# (vermilion) is phantom agreement, the correct characters that exist only because the prior
# outvoted the neural evidence, and KEY (deep blue) is the neural contribution fraction. The
# previous version of this figure drew phantom agreement in blue in one panel and in vermilion
# in another. Greys are reserved here for reference material -- the overall rate, the
# entirely-neural ceiling, the tertile boundaries -- so neither measure's colour ever stands
# for anything except that measure. Raw observations are drawn in a transparent wash of the
# same colour, so saturation, not hue, separates a session from an estimate.
PHANTOM_COLOR = ACCENT3
NCF_COLOR = KEY
# Same colour, different meaning. ACCENT3 stands for the language-model prior wherever the
# prior itself is the quantity drawn (Figure 1's fusion schematic, eFigure 3's share of
# posterior displacement); PHANTOM_COLOR is the phantom-agreement measure specifically. They
# coincide today because phantom agreement is the prior overruling the neural evidence, but a
# plot of the prior is not a plot of phantom agreement: naming them apart keeps a future
# recolour of one measure from silently repainting the other.
PRIOR_COLOR = ACCENT3
CONTEXT_TILE_FILL = "#F3D9C7"  # ACCENT3 at ~22% over white, for the emitted-context tiles
FIGURE2_COLUMNS = ((0.075, 0.385), (0.575, 0.385))  # (left, width) in figure fraction
FIGURE2_ROWS = {"strip": (0.845, 0.075), "phantom": (0.505, 0.305), "ncf": (0.165, 0.305)}
# Shared per row across both panels, so the two gradients are read on one scale. The head
# room above the phantom data and the strip below the neural-contribution data hold the
# in-panel model annotations and the session rug respectively.
PHANTOM_YLIM = (-2.4, 19.5)
NCF_YLIM = (0.395, 1.075)
NCF_RUG = (0.410, 0.450)
PHANTOM_RUG = (-2.1, -0.9)


def _square_height(rect, xlim, ylim, width):
    """Height, in y data units, that draws a `width`-wide rectangle square on the page.

    The tile strip is a hand-drawn schematic inside an axis whose x and y units are
    unrelated, so tile height has to be derived from the axis's own geometry rather than
    guessed; figure2 places its axes with explicit rectangles and never calls tight_layout,
    which is what makes that geometry knowable at draw time.
    """
    inches_per_x = rect[2] * FIGURE2_SIZE[0] / (xlim[1] - xlim[0])
    inches_per_y = rect[3] * FIGURE2_SIZE[1] / (ylim[1] - ylim[0])
    return width * inches_per_x / inches_per_y


def _word_progress_tiles(ax, rect, n_positions, y_center):
    """One row of character slots per position: what the prior had to condition on.

    At position k the k slots to its left are the intended preceding characters (filled) and
    the slot being decided is outlined; the slots after it are the rest of the word. Read down
    the strip, the filled block grows, which is the quantity panel a is about.
    """
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    pitch = 0.88 / n_positions
    tile_w = pitch * 0.80
    tile_h = _square_height(rect, xlim, ylim, tile_w)
    for index in range(n_positions):
        origin = index - 0.88 / 2 + (pitch - tile_w) / 2
        for slot in range(n_positions):
            left = origin + slot * pitch
            bottom = y_center - tile_h / 2
            if slot < index:
                style = dict(facecolor=CONTEXT_TILE_FILL, edgecolor=PHANTOM_COLOR, lw=0.5)
            elif slot == index:
                style = dict(facecolor="white", edgecolor=INK, lw=0.9)
            else:
                style = dict(facecolor="white", edgecolor="#D8DCE0", lw=0.5)
            ax.add_patch(plt.Rectangle((left, bottom), tile_w, tile_h, zorder=2, **style))


def _estimate_points(ax, x, records, scale, color, connect=True):
    """Point estimates with their 95% intervals, drawn as points and bars rather than bars
    from zero: the reviewer's objection to the previous version was that a bar chart implies
    a distribution that the participant-clustered bootstrap interval already describes."""
    estimate = np.array([record["estimate"] for record in records]) * scale
    low = np.array([record["ci_low"] for record in records]) * scale
    high = np.array([record["ci_high"] for record in records]) * scale
    if connect:
        ax.plot(x, estimate, color=color, lw=1.0, alpha=0.40, zorder=2)
    ax.errorbar(x, estimate, yerr=[estimate - low, high - estimate], fmt="o", ms=4.6,
                color=color, ecolor=color, elinewidth=1.2, capsize=2.4, mfc=color,
                mec="white", mew=0.6, zorder=5)
    return estimate


def _tertile_ticks(ax, centers, records, scale, color, fraction=0.045):
    """Minimal, unlabeled landmarks for the three decoder-quality tertile estimates the
    Results quote: a short vertical tick at each tertile's median AUC and point estimate,
    with no marker face, error bar, or callout box. The continuous fitted curve and its band
    are the panel's primary content; the previous open-square-with-whiskers markers plus a
    leader-line annotation duplicated that continuum's own value and uncertainty at three
    points and read as visual clutter on top of an already dense scatter (reviewer
    decluttering request). The tick height is a fixed fraction of the axis's own y-range
    rather than a hardcoded value, so it stays proportionate whether it is drawn on the
    phantom-agreement row or the neural-contribution-fraction row.
    """
    estimate = np.array([record["estimate"] for record in records]) * scale
    half_height = fraction * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for center, value in zip(centers, estimate):
        ax.plot([center, center], [value - half_height, value + half_height], color=color,
                lw=1.6, solid_capstyle="butt", zorder=6)


def _rug(ax, values, span, color):
    """One tick per session along the quality axis, so the reader can see where the
    continuum is actually supported and where its ends are thin."""
    for value in values:
        ax.plot([value, value], list(span), color=color, lw=0.7, alpha=0.45,
                solid_capstyle="butt", zorder=3, clip_on=False)


def _linear_predictor(model, grid, names=("Intercept", "train_auc")):
    """Fitted linear predictor and its standard error over a grid of the covariate."""
    params = np.array([float(model.params[name]) for name in names])
    covariance = np.asarray(model.cov_params().loc[list(names), list(names)], dtype=float)
    design = np.column_stack([np.ones_like(grid), grid])
    predictor = design @ params
    variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    return predictor, np.sqrt(variance)


def _calibrated_quality_frame():
    """Per-selection calibrated fusion carrying `train_auc` and `session_id`, built the same
    way run_stats.py's main() built its shared `calibrated_primary` frame (full_prior_frame
    with the extra covariate columns, then calibrated_attribution_frame at n_folds=5, seed=0
    -- that seed is the fold-split argument calibrated_fusion_frame ignores in practice
    (Task 3), fixed here only to match the call site's convention, not because it changes the
    result). `train_auc` drives the decoder-quality NCF fit; `session_id` (added for Task
    12b) lets callers build a per-session calibrated-basis frame the same way figure2()
    already does for the raw basis. The other covariates in run_stats.py's version are
    unused by either and stay out.
    """
    selections = pd.read_parquet(OUTPUT / "intermediate" / "selections.parquet")
    priors = pd.read_parquet(OUTPUT / "intermediate" / "priors.parquet")
    frame = full_prior_frame(
        selections, priors, PRIMARY_PRIOR, extra_columns=("train_auc", "session_id")
    )
    return cast_binary_columns(
        calibrated_attribution_frame(frame, group_column=CLUSTER, n_folds=5, seed=0)
    )


def _quality_trend_fits(digest, calibrated_frame):
    """The two calibration-quality curves panel b draws, refitted here to obtain a
    curve-level confidence band.

    Both are fit on `calibrated_frame`, the calibrated-fusion basis of Figure 1's co-primary
    estimates and of everything else this figure now draws. Phantom agreement is
    `phantom_agreement ~ train_auc` by participant-clustered logistic GEE; NCF is
    `ncf ~ train_auc` by the same GEE family with a logistic mean function (a fractional-logit
    fit, bounded in [0, 1] by construction, replacing the linear mixed model this panel used
    to draw, which could and did predict above 1). Phantom agreement used to be fit on the
    raw, uncalibrated frame while NCF was already calibrated, a split the panel note
    disclosed rather than resolved; it is gone (Task 4). `calibrated_frame` is built once by
    the caller (`_calibrated_quality_frame()`) and passed in here so the curve fits and the
    `sessions` frame figure2() draws from come from the exact same rows, rather than each
    calling `_calibrated_quality_frame()` separately. Neither refit is a second analysis:
    every refitted coefficient is checked against the coefficient the digest already stores
    and rendering stops if they differ, so the curves this panel draws are the models the
    Results state, and the band is each model's own 95% interval for the fitted mean (delta
    method on the linear predictor, logistic-transformed).
    """
    calibrated_stored = digest["secondary"]["calibrated"]["models"]
    auc = calibrated_frame["train_auc"].to_numpy(dtype=float)
    grid = np.linspace(auc.min(), auc.max(), 200)

    phantom_frame = model_frame(calibrated_frame, "phantom_agreement ~ train_auc")
    ncf_frame = model_frame(calibrated_frame, "ncf ~ train_auc")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        phantom_model = smf.gee(
            "phantom_agreement ~ train_auc",
            groups=phantom_frame[CLUSTER],
            data=phantom_frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        ncf_model = smf.gee(
            "ncf ~ train_auc",
            groups=ncf_frame[CLUSTER],
            data=ncf_frame,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()

    for model, terms, label in (
        (phantom_model, calibrated_stored["phantom_by_decoder_quality"]["terms"],
         "Figure 2 refitted the calibrated-basis phantom_by_decoder_quality"),
        (ncf_model, calibrated_stored["ncf_by_decoder_quality_bounded"]["terms"],
         "Figure 2 refitted the calibrated-basis ncf_by_decoder_quality_bounded"),
    ):
        for name, term in terms.items():
            if not np.isclose(float(model.params[name]), term["log_odds"], rtol=0, atol=1e-8):
                raise RuntimeError(
                    f"{label} and got {name} = {float(model.params[name]):.6g}, but "
                    f"output/stats_digest.json reports {term['log_odds']:.6g}. The figure must "
                    "draw the model the Results report; re-run scripts/run_stats.py, or "
                    "reconcile the two specifications, before re-rendering."
                )

    phantom_eta, phantom_se = _linear_predictor(phantom_model, grid)
    ncf_eta, ncf_se = _linear_predictor(ncf_model, grid)
    logistic = lambda value: 1.0 / (1.0 + np.exp(-value))  # noqa: E731
    return {
        "grid": grid,
        "phantom": (logistic(phantom_eta), logistic(phantom_eta - 1.96 * phantom_se),
                    logistic(phantom_eta + 1.96 * phantom_se)),
        "ncf": (logistic(ncf_eta), logistic(ncf_eta - 1.96 * ncf_se),
                logistic(ncf_eta + 1.96 * ncf_se)),
    }


def figure2(digest, attribution):
    """The attribution landscape: the two axes along which the language model's share of an
    emitted character moves, context accumulating inside a word (a) and the calibration
    decoder's discriminability for the session (b).

    Colour grammar, unchanged across panels and matching Figure 1: ACCENT3 (vermilion) is
    phantom agreement wherever it appears, KEY (deep blue) is the neural contribution
    fraction wherever it appears, and grey is reference material only.

    Every quantity drawn here is on the calibrated-fusion basis of Figure 1's co-primary
    estimates. Panel a in full, and phantom agreement's row of panel b, used to be drawn on
    raw, uncalibrated fusion while only NCF's row of panel b was calibrated; the split was
    disclosed rather than resolved, and it survived the move of the Results text to
    calibrated-primary reporting, so figure and text printed different numbers for the same
    quantities. Task 4 put the whole figure on the basis the Results lead with. The raw
    estimates remain the labeled sensitivity comparison, in eTable 2, eTable 3, and eTable 5.
    """
    secondary = digest["secondary"]
    calibrated = secondary["calibrated"]
    by_position = calibrated["by_position_in_word"]
    by_quality = calibrated["by_decoder_quality"]
    models = calibrated["models"]

    # Same non-positional sensitivity-check keys as in figure1(); restrict to digit keys.
    positions = sorted((key for key in by_position if key.isdigit()), key=int)
    tertiles = ["low", "mid", "high"]

    # The per-selection frame every quantity in this figure is computed on: held-out
    # calibrated fusion at the primary prior. `_calibrated_quality_frame()` has already cast
    # the binary outcomes to integer (a boolean outcome is expanded into a two-level
    # categorical and silently inverts every odds ratio).
    calibrated_frame = _calibrated_quality_frame()

    # `attribution` is no longer plotted, but it is still the check on the one thing that did
    # not have to move when everything else did: `train_auc` is a property of the calibration
    # decoder, fitted before fusion, so the tertile cut, its boundaries, and its per-tertile
    # n counts are identical on both bases. If that ever stopped holding, the strip's counts
    # and the dotted tertile rules would describe a different partition than the estimates
    # above them, silently.
    raw_auc = np.sort(
        cast_binary_columns(attribution)
        .loc[lambda frame: (frame["beta"] == PRIMARY_BETA)
             & (frame["prior_model"] == PRIMARY_PRIOR), "train_auc"]
        .to_numpy(dtype=float)
    )
    calibrated_auc = np.sort(calibrated_frame["train_auc"].to_numpy(dtype=float))
    if raw_auc.shape != calibrated_auc.shape or not np.array_equal(raw_auc, calibrated_auc):
        raise RuntimeError(
            "Figure 2's calibrated frame and the raw attribution frame no longer carry the "
            "same calibration-decoder AUCs, so the tertile cut is not basis-independent as "
            "this figure assumes. Re-run scripts/run_stats.py, or reconcile the two frames, "
            "before re-rendering."
        )

    calibrated_frame["auc_tertile"] = pd.qcut(calibrated_frame["train_auc"], 3, labels=tertiles)
    edges = np.asarray(pd.qcut(calibrated_frame["train_auc"], 3, retbins=True)[1], dtype=float)
    # The tertile cut is recomputed here rather than read from the digest, so it can silently
    # disagree with the one run_stats made; the group sizes are the check that it does not.
    for name in tertiles:
        drawn = int((calibrated_frame["auc_tertile"] == name).sum())
        if drawn != by_quality[name]["n"]:
            raise RuntimeError(
                f"Figure 2's calibration-AUC tertile '{name}' holds {drawn} selections but "
                f"output/stats_digest.json reports {by_quality[name]['n']}. The tertile "
                "markers would not be the estimates the Results quote; re-run "
                "scripts/run_stats.py before re-rendering."
            )
    tertile_center = (
        calibrated_frame.groupby("auc_tertile", observed=True)["train_auc"].median()
    )
    # These medians are where the tertile ticks land, and the outer two are also the endpoints
    # run_stats evaluated the annotated predicted contrast at. It computed them from its own
    # frame, so without this check the annotation could describe a span other than the one
    # between the ticks a reader sees. Same raise-on-mismatch guard the tertile sizes use above.
    for name in tertiles:
        stored = by_quality[name]["train_auc_median"]
        if not np.isclose(float(tertile_center[name]), stored, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Figure 2's calibrated tertile '{name}' has median AUC "
                f"{float(tertile_center[name]):.6g} but output/stats_digest.json "
                f"reports {stored:.6g}. The annotated predicted contrast would not span the "
                "ticks drawn; re-run scripts/run_stats.py before re-rendering."
            )
    sessions = (
        calibrated_frame.groupby([CLUSTER, "session_id"])
        .agg(train_auc=("train_auc", "first"),
             phantom_agreement=("phantom_agreement", "mean"),
             ncf=("ncf", "mean"),
             n=("ncf", "size"))
        .reset_index()
    )

    fits = _quality_trend_fits(digest, calibrated_frame)

    fig = plt.figure(figsize=FIGURE2_SIZE)
    axes = {}
    for column, (left, width) in zip(("a", "b"), FIGURE2_COLUMNS):
        for row, (bottom, height) in FIGURE2_ROWS.items():
            rect = [left, bottom, width, height]
            axes[(column, row)] = (fig.add_axes(rect), rect)

    position_lim = (-0.5, len(positions) - 0.5)
    quality_lim = (float(edges[0]) - 0.018, float(edges[-1]) + 0.018)
    for column, limits in (("a", position_lim), ("b", quality_lim)):
        for row in FIGURE2_ROWS:
            axes[(column, row)][0].set_xlim(*limits)
    for column in ("a", "b"):
        strip = axes[(column, "strip")][0]
        strip.set_ylim(0, 1)
        strip.axis("off")
        for row, limits in (("phantom", PHANTOM_YLIM), ("ncf", NCF_YLIM)):
            axis = axes[(column, row)][0]
            axis.set_ylim(*limits)
            axis.grid(axis="y", color="#EDEFF2", lw=0.6)
            axis.set_axisbelow(True)
        axes[(column, "phantom")][0].set_yticks([0, 5, 10, 15])
        axes[(column, "ncf")][0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # labelbottom/labelleft rather than empty label lists, which a later set_xticks would
        # silently undo. The two panels share a y range per row and panel a carries the
        # labels for both.
        axes[(column, "phantom")][0].tick_params(labelbottom=False)
        if column == "b":
            for row in ("phantom", "ncf"):
                axes[(column, row)][0].tick_params(labelleft=False)

    # ------------------------------------------------------------ headers and tile strip
    for column, (left, _), letter, title, note in (
        ("a", FIGURE2_COLUMNS[0], "a", "LLM influence increases with available linguistic context",
         "tiles: intended preceding characters (filled) before the one being decided (outlined)"),
        ("b", FIGURE2_COLUMNS[1], "b", "Calibration decoder discriminability",
         "one circle and one rug tick per session, area scales with selections; both panels "
         "calibrated"),
    ):
        fig.text(left - 0.055, 0.982, letter, fontsize=13, fontweight="bold", color=INK,
                 ha="left", va="top")
        fig.text(left - 0.028, 0.980, title, fontsize=8.6, fontweight="bold", color=INK,
                 ha="left", va="top")
        fig.text(left - 0.028, 0.950, note, fontsize=6.2, color=MUTED, ha="left", va="top",
                 style="italic")

    strip_a, strip_a_rect = axes[("a", "strip")]
    _word_progress_tiles(strip_a, strip_a_rect, len(positions), y_center=0.66)
    for index, key in enumerate(positions):
        strip_a.text(index, 0.16, f"n = {by_position[key]['n']:,}", fontsize=6.2, color=MUTED,
                     ha="center", va="center")

    strip_b = axes[("b", "strip")][0]
    for name, label in zip(tertiles, ("Lower third", "Middle third", "Upper third")):
        index = tertiles.index(name)
        span = (max(edges[index], quality_lim[0]), min(edges[index + 1], quality_lim[1]))
        middle = sum(span) / 2
        strip_b.plot(span, [0.50, 0.50], color="#D8DCE0", lw=0.9, solid_capstyle="butt")
        for end in span:
            strip_b.plot([end, end], [0.44, 0.56], color="#D8DCE0", lw=0.9)
        strip_b.text(middle, 0.80, label, fontsize=6.8, fontweight="bold", color=MUTED,
                     ha="center", va="center")
        strip_b.text(middle, 0.16, f"n = {by_quality[name]['n']:,}", fontsize=6.2,
                     color=MUTED, ha="center", va="center")

    # ------------------------------------------------------------------ panel a: context
    x = np.arange(len(positions))
    phantom_axis = axes[("a", "phantom")][0]
    _estimate_points(
        phantom_axis, x,
        [by_position[key]["phantom_agreement"] for key in positions], 100, PHANTOM_COLOR,
    )
    overall = digest["calibrated_primary_analysis"]["phantom_agreement"]
    phantom_axis.axhline(100 * overall["estimate"], color=CONTEXT, ls=(0, (4, 3)), lw=0.9,
                         zorder=1)
    # Above the line and hard left, not beside it: sitting on the line at the right-hand end
    # put this label straight through the sixth position's confidence bar.
    phantom_axis.text(position_lim[0] + 0.06, 100 * overall["estimate"] + 1.6,
                      f"overall {100 * overall['estimate']:.1f}%", color=CONTEXT, fontsize=6.4,
                      ha="left", va="bottom")
    phantom_axis.set_ylabel("Phantom agreement (%)", color=PHANTOM_COLOR)

    ncf_axis = axes[("a", "ncf")][0]
    _estimate_points(ncf_axis, x, [by_position[key]["ncf"] for key in positions], 1, NCF_COLOR)
    ncf_axis.axhline(1.0, color=CONTEXT, ls=":", lw=1.0, zorder=1)
    ncf_axis.text(position_lim[0] + 0.06, 1.006, "entirely neural", color=CONTEXT,
                  fontsize=6.4, ha="left", va="bottom")
    ncf_axis.set_ylabel("Neural contribution fraction", color=NCF_COLOR)
    for axis in (phantom_axis, ncf_axis):
        axis.set_xticks(x)
    phantom_axis.tick_params(labelbottom=False)
    ncf_axis.set_xticklabels([str(int(key) + 1) for key in positions])
    ncf_axis.set_xlabel("Character position in word")

    # Each annotation carries its own model's P value, unadjusted, and the caption says so
    # once. Two of these four models are members of the five-member Benjamini-Hochberg family
    # and the two bounded fractional-logit contrasts are not, so labelling the adjusted ones
    # "adjusted P" and leaving the others bare, as this figure did after Task 2, put two kinds
    # of P side by side in one panel with nothing on the figure to separate them. Adjusting is
    # also a property of a family whose fifth member (target type) is not drawn here at all,
    # which makes an inline "adjusted P" read as if the correction covered what the reader can
    # see. The adjusted values are in the Results and eTable 30; none of them changes a
    # conclusion (every P in this figure is below .002 on either treatment).
    position_or = models["phantom_by_position"]["terms"]["position_in_phrase"]
    # Same reasoning as panel b's NCF annotation below: the number must come from the model
    # the panel's story is told from. This row used to annotate the linear mixed model's
    # per-character slope, which is now the labeled sensitivity comparison rather than the
    # primary summary, leaving the two NCF rows of one figure headlining two different model
    # families. It is now the bounded fractional-logit fit's own predicted-value contrast
    # across the plotted positions, exactly parallel to panel b.
    position_contrast = models["ncf_by_position_bounded"]["predicted_contrast"]
    phantom_axis.text(
        position_lim[0] + 0.06, 18.6,
        f"OR {position_or['odds_ratio']:.2f}/character, {_p_text(position_or['p_value'])}",
        fontsize=6.2, color=PHANTOM_COLOR, ha="left", va="top",
    )
    ncf_axis.text(
        position_lim[0] + 0.06, 0.640,
        f"Predicted {position_contrast['difference']:+.3f} across positions "
        f"{int(position_contrast['low_value']) + 1} to "
        f"{int(position_contrast['high_value']) + 1}, "
        f"{_p_text(position_contrast['p_value'])}",
        fontsize=6.2, color=NCF_COLOR, ha="left", va="top",
    )

    # ---------------------------------------------------------- panel b: decoder discriminability
    grid = fits["grid"]
    quality_phantom = axes[("b", "phantom")][0]
    quality_ncf = axes[("b", "ncf")][0]
    for axis, color, scale, (curve, low, high), rug_span in (
        (quality_phantom, PHANTOM_COLOR, 100, fits["phantom"], PHANTOM_RUG),
        (quality_ncf, NCF_COLOR, 1, fits["ncf"], NCF_RUG),
    ):
        for edge in edges[1:-1]:
            axis.axvline(edge, color="#D8DCE0", ls=":", lw=0.9, zorder=1)
        axis.fill_between(grid, scale * low, scale * high, color=color, alpha=0.16, lw=0,
                          zorder=2)
        axis.plot(grid, scale * curve, color=color, lw=1.8, zorder=4)
        _rug(axis, sessions["train_auc"].to_numpy(), rug_span, color)
    for edge in edges[1:-1]:
        strip_b.axvline(edge, color="#D8DCE0", ls=":", lw=0.9, zorder=1)

    marker_area = 3.0 + 0.32 * sessions["n"].to_numpy(dtype=float)
    quality_phantom.scatter(sessions["train_auc"], 100 * sessions["phantom_agreement"],
                            s=marker_area, color=PHANTOM_COLOR, alpha=0.28, lw=0, zorder=3)
    quality_ncf.scatter(sessions["train_auc"], sessions["ncf"],
                        s=marker_area, color=NCF_COLOR, alpha=0.28, lw=0, zorder=3)
    _tertile_ticks(quality_phantom, tertile_center.to_numpy(),
                   [by_quality[name]["phantom_agreement"] for name in tertiles], 100,
                   PHANTOM_COLOR)
    _tertile_ticks(quality_ncf, tertile_center.to_numpy(),
                   [by_quality[name]["ncf"] for name in tertiles], 1, NCF_COLOR)
    quality_phantom.axhline(100 * overall["estimate"], color=CONTEXT, ls=(0, (4, 3)), lw=0.9,
                            zorder=1)
    quality_ncf.axhline(1.0, color=CONTEXT, ls=":", lw=1.0, zorder=1)
    quality_ncf.set_xlabel("Calibration decoder AUC of the session")

    quality_or = models["phantom_by_decoder_quality"]["terms"]["train_auc"]
    # The annotated number must come from the model this row actually draws. It used to be the
    # calibrated LINEAR mixed model's per-unit slope while the curve beneath it was already the
    # bounded fractional-logit fit, so the "+0.442/AUC unit" a reader saw described neither the
    # S-shape in front of them nor any quantity derivable from it (Task 2). It is now that same
    # bounded fit's predicted-value contrast between the outer two tertile ticks drawn on this
    # axis, so the annotation, the curve, and the Results sentence are one model.
    quality_contrast = models["ncf_by_decoder_quality_bounded"]["predicted_contrast"]
    # Right-aligned, where the NCF row's annotation already sits, rather than hard left where
    # this one used to. On the raw basis the confidence band's left end stayed low enough to
    # leave the top-left corner empty; the calibrated band is higher there and rises to 18.5%
    # at the lowest observed AUC, straight through this text. It falls below 6% past AUC 0.75,
    # and the highest session circle in that window is at 14.3%, so the top right corner is
    # the one clear block of space this row now has.
    quality_phantom.text(
        quality_lim[1] - 0.006, 18.6,
        f"OR {quality_or['odds_ratio']:.3f}/AUC unit, {_p_text(quality_or['p_value'])}",
        fontsize=6.2, color=PHANTOM_COLOR, ha="right", va="top",
    )
    quality_ncf.text(
        quality_lim[1] - 0.006, 0.775,
        f"Predicted {quality_contrast['difference']:+.3f} between tertile medians, "
        f"{_p_text(quality_contrast['p_value'])}",
        fontsize=6.2, color=NCF_COLOR, ha="right", va="top",
        # No opaque backing on this one, unlike the italic note below it: the only placement
        # that clears the upper tertile boundary also puts the box over a session circle, and
        # erasing an observation to protect text from a hairline dotted rule is the worse
        # trade. The rule is #D8DCE0 at 0.9 pt and disappears behind the glyphs.
    )
    # No callout box or leader line for the tertile ticks themselves: they are deliberately
    # unlabeled landmarks (see _tertile_ticks), and the caption states what they mark.

    save(fig, str(FIGURES / "Figure2"))
    plt.close(fig)


# Drawn at the journal's 183 mm double column rather than wider. This figure used to be 9.2 by
# 4.9 inches, which renders 227 mm, and a journal reproducing it at 183 mm shrinks every glyph
# by a fifth: the 6.2 pt chrome landed at 5.01 pt on the page, on the floor of the 5-7 pt band
# Nature asks for at final size. Only the width really came in. The height is nearly what it
# was, deliberately: type does not shrink with the canvas, so a proportionally shorter figure
# would have spent its panels' vertical room on a legend and two notes that stayed the same
# size, and the family key would have come down on the priors it exists to explain. At this
# height the panels are almost exactly as square as they were, which is what both scatters want.
FIGURE3_SIZE = (7.2, 4.3)
FIGURE3_WRAP_REFERENCE_IN = 9.2  # the width every _wrap character budget below was measured at
# Panel b moved left and both panels widened when the canvas came in. A panel title is set in
# 8.2 pt type that does not shrink with the canvas, and panel b's is both the longer of the two
# and the one anchored past the middle of the figure: left at the old x on a 183 mm canvas it
# ran off the right-hand edge and set the rendered width itself, which is how a narrower canvas
# produced a figure no narrower.
# The gutter between the panels was the place to find the room - it was an inch wide and panel
# b's y-axis label and tick labels need about a third of that - so the panels are now equal,
# wider than they were as a fraction, and separated by what panel b's axis furniture needs.
FIGURE3_PANELS = {  # [left, bottom, width, height] in figure fraction
    "a": (0.062, 0.150, 0.415, 0.672),
    "b": (0.553, 0.150, 0.415, 0.672),
}
# Panel a's x axis is next-character NLL (Task 15's prior_quality_metrics), which every prior
# in the calibrated ladder has a value for -- including the character 5-grams, which had no
# position at all on the old log-parameter-count axis and needed a separate gutter. There is no
# gutter here: every prior plots directly on this axis. Limits pad the observed NLL range, 2.27
# (the best character n-gram) to 4.96 (the weakest LLM), and the calibrated phantom-agreement
# range, 2.09% to 10.73%.
FIGURE3_A_XLIM = (2.05, 5.15)
FIGURE3_A_YLIM = (-0.75, 12.10)
# Panel a's magnified inset, both rectangles in that panel's own data units as (x0, x1, y0, y1).
#
# ZOOM is the region of the cluster the inset redraws: the fourteen priors from GPT-2 774M to
# Qwen2.5 14B in NLL, which is the run _separated_x has to push apart. Its edges are set by the
# markers rather than by the numbers, and a marker is a disc a little over three points across,
# not a point: every prior inside is inside whole, with room to spare, and the two Mistral
# priors just outside it are missed by a clear margin rather than sliced through. An earlier
# region drawn to the marker centres cut the bottom off GPT-OSS 20B's marker and ran its own
# left edge through Mixtral 8x7B's.
#
# RECT is where the inset itself sits, and is the largest empty rectangle panel a has: bounded
# below by Mistral 7B's marker and the band its own x tick labels need, on the left by the three
# character 5-grams and the band its y tick labels need, and above by the room its title takes
# under the top of the panel. No prior falls inside it, which matters because the inset is an
# opaque axis drawn over the panel. The panel's correlation annotation used to hold the top
# right corner and is now in the empty bottom left one, which is what let the inset have this
# much height: the annotation is a fixed block of two lines that fits either corner equally, and
# the inset is not.
#
# Together the two set the magnification, 3.1 times on x and 2.6 on y. Not equal, because the
# region is taller than it is wide in drawn points while the space for the inset is wider than
# it is tall; x is the axis given the larger share because x is the axis _separated_x moves
# markers along, so it is where the pairs this inset exists to separate are separated.
FIGURE3_A_INSET_ZOOM = (3.49, 4.13, 3.15, 5.09)
FIGURE3_A_INSET_RECT = (2.90, 4.90, 6.30, 11.25)
FIGURE3_A_INSET_TITLE_PAD_PT = 2.6
FIGURE3_B_XLIM = (-0.28, 3.95)
FIGURE3_B_YLIM = (-1.05, 8.95)
# Geometry for _parameter_count_phantom_scatter alone: the parameter-count-vs-phantom-agreement
# scatter that was Figure 3 panel A before Task 23 moved that panel onto the NLL basis above.
# This figure no longer calls that function; a future supplementary figure wires it in instead,
# using this same geometry so the extraction changes nothing about how it renders.
PARAM_SCATTER_GUTTER_X = 7.45  # where the priors with no parameter count sit
PARAM_SCATTER_BREAK_X = 7.88  # the break between that gutter and the log-parameter scale
PARAM_SCATTER_XLIM = (7.02, 10.95)
PARAM_SCATTER_YLIM = (-0.75, 9.95)
PARAM_SCATTER_LEADER_SPAN_PT = (10.0, 45.0)
PARAM_SCATTER_LEADER_ARC = (286.0, 302.0)
# Minimum distance, in points, between two marker centres in panel a. Two priors placed close in
# both NLL and phantom agreement land near the same pixel, and the marker drawn second covers
# the first completely, so the panel shows fewer priors than the caption promises. Below this
# separation the markers are pushed apart along x. A marker is 5.8 points across and carries a
# 1.1 point white halo, so this leaves the marker underneath a crescent about 2.4 points wide
# with a clean white edge between the two rather than a colour boundary the reader has to find.
# It was 3.0 against a 6.6 point marker with no halo, where the crescent was the same width but
# ran straight into its neighbour's fill. Panel b needs none of this: its cluster is dense by
# construction, but no two of its markers fall closer than 1.9 points, so every one of the 24
# keeps a visible crescent where it belongs. Its x axis also carries a measured quantity rather
# than a scale, so displacing a marker there would move it off the value it stands for, which is
# why the separation is panel a's alone.
# Changing this moves a printed number in a different figure. _parameter_count_phantom_scatter
# takes the same default, and eFigure 4's footer discloses the largest nudge it produces as a
# fraction of a parameter count: going from 3.0 to 3.5 moved that from 0.07 to 0.08. The footer
# measures it off the layout rather than restating it, so it cannot go stale, but a reader
# comparing eFigure 4 across versions will see the number move and should find the reason here.
FIGURE3_MIN_SEPARATION_PT = 3.5
# The drawn size of the things a leader and its label have to stay clear of. scatter takes an
# area in square points, so the radius is the square root of that over two, plus half the stroke
# the marker is drawn with, plus the white halo drawn outside that stroke. All three are what
# _draw_prior_point and _ring actually pass, so a marker that grows there grows the clearance
# the search below enforces rather than leaving it measuring a size the figure no longer draws.
# The marker area was 44 before this figure's cluster was decluttered; 34 is the same marker a
# little over a tenth narrower, which buys back most of a marker width of white in the dense
# middle of panel a without dropping the smallest priors below the 5-7 pt band the journal
# prints chrome at.
FIGURE3_MARKER_AREA_PT2 = 34.0
FIGURE3_MARKER_LINEWIDTH_PT = 1.15
# The white stroke drawn outside every marker's own edge. Occlusion in panel a's cluster used to
# be total: a marker drawn over another shared its neighbour's boundary, so the one underneath
# survived only as a crescent of colour, and two neighbours of similar hue merged into one shape
# with no boundary at all. The halo puts white between them instead, which is what makes a partly
# hidden marker still countable at the size the journal prints this figure.
FIGURE3_MARKER_HALO_PT = 1.1
FIGURE3_RING_AREA_PT2 = 132.0
FIGURE3_RING_LINEWIDTH_PT = 0.9
FIGURE3_RING_HALO_PT = 1.1
# Every label inside a Figure 3 panel is set at this size, the primary prior's included.
FIGURE3_LABEL_FONTSIZE = 6.2
# How far the primary's label sits from the far end of its leader, in points: two along the
# leader's own direction, so the tip reads as touching the text, and three across it, so the
# line does not run into the glyphs. Which way each is applied follows the bearing; see
# _leader_label_placement.
FIGURE3_LEADER_LABEL_OFFSET_PT = (2.0, 3.0)
# White space, in points, that the leader and the label at its end have to keep from every
# marker's drawn edge, from every other label's box, and from the panel's own limits. The
# leader-to-marker half of this is the threshold tests/test_make_figures_figure3_layout.py has
# enforced since a leader was found running tangent to the wrong prior; the label half is the
# same rule applied to the box the label occupies, which the search used not to model at all.
FIGURE3_LEADER_CLEARANCE_PT = 2.0
# Where each panel's leader to the primary prior starts and ends, as distances in points from
# that prior's centre. Both start well outside the ring because the primary's nearest neighbours
# crowd the ring itself in both panels: a leader leaving the ring's edge among them passes close
# to a neighbour whichever way it goes, and only distance from the ring buys the clearance back.
# Panel b's used to start at 7 and end at 22, which is inside its own cluster: no bearing at that
# reach could put the two-line label anywhere it did not overlap something, and the best on offer
# ran it through a base Qwen marker. Starting at 14 and ending at 38 clears every marker and
# every other label by 6.5 points, measured; test_the_primary_labels_box_stays_clear pins it.
# Panel a's near end moved out from 12 to 14 when every marker gained a white halo: 12 points
# from the primary's centre is 1.7 points from the nearest neighbour's halo, and a leader that
# starts inside a neighbour's halo cuts a white notch out of it. 14 clears it by 2.9 points and
# leaves the leader running on the same bearing, a degree and a half over.
FIGURE3_LEADER_SPAN_PT = {"a": (14.0, 46.0), "b": (14.0, 38.0)}
# Bearings the leader may take, in degrees, when only part of the circle has room for the label.
# Panel a's cluster of large LLMs sits in the lower-right of the panel (high NLL, mid phantom
# agreement), and the panel's whole upper half is now the inset, so the sweep is confined to the
# arc below; the bearing within that arc is still measured, not chosen, and the inset is passed
# to the search as an obstacle as well, so the two agree rather than one relying on the other.
# Panel b's label has room wherever its leader points, so the whole circle is swept.
FIGURE3_LEADER_ARC = {"a": (250.0, 340.0), "b": None}
# The placements _measured_label sweeps for a label hung off a single marker, as
# (dx, dy, ha, va) in points. The four square offsets come first and the four diagonals after,
# so a tie goes to the placement a reader expects; the gap is 7 points on a square offset and
# 5.5 on each axis of a diagonal, which is the same reach measured from the marker's centre.
FIGURE3_LABEL_CANDIDATES = (
    (7.0, 0.0, "left", "center"),
    (-7.0, 0.0, "right", "center"),
    (0.0, 7.0, "center", "bottom"),
    (0.0, -7.0, "center", "top"),
    (5.5, 5.5, "left", "bottom"),
    (5.5, -5.5, "left", "top"),
    (-5.5, 5.5, "right", "bottom"),
    (-5.5, -5.5, "right", "top"),
)


def _figure3_rows(digest):
    """One record per prior in the calibrated ladder, carrying everything both panels plot.

    Every value is read from output/stats_digest.json on the held-out, per-source calibrated
    basis (Table 2's primary basis): participant-weighted phantom agreement and prior capture
    with their bootstrap intervals, the accuracy gained by adding the prior as fused minus
    neural-only accuracy in percentage points, and each prior's own next-character prediction
    quality (mean negative log-likelihood of the intended character, and mean entropy) - a
    property of the language model itself, not of how it fuses with the neural decoder. The
    uniform null has no calibrated basis (Table 2's note: there is no shape for a temperature to
    correct on a prior that carries no information) and so is absent from prior_ladder_calibrated
    and from these rows; _raw_ladder_rows below is where it still appears.
    """
    ladder = digest["sensitivity"]["prior_ladder_calibrated"]
    quality = digest["sensitivity"]["prior_quality_metrics"]
    rows = []
    for name in LADDER_ORDER:
        if name not in ladder:
            continue
        entry = ladder[name]
        metrics = quality[name]
        family = family_of(name)
        parameters = int(entry["prior_parameters"])
        rows.append({
            "name": name,
            "label": LADDER_LABEL[name].replace("\n", " "),
            "family": family,
            "color": FAMILY_COLOR[family],
            "parameters": parameters,
            "nll": metrics["nll_intended_character"],
            "entropy": metrics["entropy_mean"],
            "phantom": 100 * entry["phantom_agreement"]["estimate"],
            "capture": 100 * entry["prior_capture"]["estimate"],
            "gain": 100 * (entry["fused_accuracy"] - entry["neural_accuracy"]),
            # A parameter count is what makes a prior part of the scale question; the n-grams
            # carry 0, which is how the preserved parameter-count scatter below excludes them
            # from its Spearman correlation. It plays no such role here: every row has a
            # well-defined NLL, character n-grams included.
            "neural": parameters > 0,
            "instruction_tuned": name in INSTRUCTION_TUNED,
        })
    return rows


def _raw_ladder_rows(digest):
    """One record per prior in the raw, uncalibrated ladder, including the uniform null.

    This is the row shape Figure 3 panel A used before Task 23 moved that panel onto the
    calibrated, NLL-based basis of _figure3_rows above. Kept only for
    _parameter_count_phantom_scatter, which this figure no longer calls; a future supplementary
    figure wires it in instead.
    """
    ladder = digest["sensitivity"]["prior_ladder"]
    rows = []
    for name in LADDER_ORDER:
        if name not in ladder:
            continue
        entry = ladder[name]
        family = family_of(name)
        parameters = int(entry["prior_parameters"])
        rows.append({
            "name": name,
            "label": LADDER_LABEL[name].replace("\n", " "),
            "family": family,
            "color": FAMILY_COLOR[family],
            "parameters": parameters,
            "log_parameters": float(np.log10(parameters)) if parameters > 0 else None,
            "phantom": 100 * entry["phantom_agreement"]["estimate"],
            "capture": 100 * entry["prior_capture"]["estimate"],
            "gain": 100 * (entry["fused_accuracy"] - entry["neural_accuracy"]),
            # A parameter count is what makes a prior part of the scale question; the uniform
            # null and the n-grams carry 0, which is how run_stats._ladder_points excludes
            # them from the Spearman correlation this figure reports.
            "neural": parameters > 0,
            "instruction_tuned": name in INSTRUCTION_TUNED,
        })
    return rows


def _points_per_unit(letter, xlim, ylim):
    """How many typographic points one data unit spans in a Figure 3 panel, on each axis.

    The panels are placed with fig.add_axes at fixed figure fractions and given fixed limits,
    so this is exact and needs no rendered figure.
    """
    _, _, width, height = FIGURE3_PANELS[letter]
    return (72 * width * FIGURE3_SIZE[0] / (xlim[1] - xlim[0]),
            72 * height * FIGURE3_SIZE[1] / (ylim[1] - ylim[0]))


def _axes_fraction(rect, xlim, ylim):
    """An (x0, x1, y0, y1) rectangle in data units, as the [left, bottom, width, height] in axes
    fractions that inset_axes wants. Stating an inset's place in the units of the panel it sits
    in is what lets it be checked against the panel's own contents."""
    x0, x1, y0, y1 = rect
    return [(x0 - xlim[0]) / (xlim[1] - xlim[0]), (y0 - ylim[0]) / (ylim[1] - ylim[0]),
            (x1 - x0) / (xlim[1] - xlim[0]), (y1 - y0) / (ylim[1] - ylim[0])]


def _points_per_unit_rect(rect, zoom):
    """How many points one data unit spans inside panel a's inset, on each axis.

    _points_per_unit answers this for a panel placed with add_axes at a fixed figure fraction;
    the inset is placed inside one of those panels instead, so its own width in inches is the
    panel's width times the fraction of the panel the inset takes.
    """
    _, _, panel_width, panel_height = FIGURE3_PANELS["a"]
    _, _, width, height = _axes_fraction(rect, FIGURE3_A_XLIM, FIGURE3_A_YLIM)
    return (72 * width * panel_width * FIGURE3_SIZE[0] / (zoom[1] - zoom[0]),
            72 * height * panel_height * FIGURE3_SIZE[1] / (zoom[3] - zoom[2]))


def _separated_x(xs, ys, x_per_unit, y_per_unit, minimum=FIGURE3_MIN_SEPARATION_PT, passes=6):
    """Marker x positions with coincident markers pushed apart along x.

    Two priors with the same parameter count and nearly the same phantom agreement land on the
    same pixel, and the second one drawn hides the first. Any pair whose rendered centres fall
    closer than `minimum` points is separated symmetrically about its own mean, by the smallest
    offset that reaches that separation, so neither member is displaced more than the other and
    the pair still reads as the near-tie it is. Only x moves: y carries the measure the panel is
    testing. The sweep runs in the caller's order and repeats until nothing moves, so the result
    is a deterministic function of the ladder rather than of a random seed.
    """
    xs = list(xs)
    for _ in range(passes):
        crowded = False
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dx = (xs[i] - xs[j]) * x_per_unit
                dy = (ys[i] - ys[j]) * y_per_unit
                if float(np.hypot(dx, dy)) >= minimum:
                    continue
                crowded = True
                needed = float(np.sqrt(max(minimum ** 2 - dy ** 2, 0.0)))
                shift = (needed - abs(dx)) / 2 / x_per_unit
                right, left = (i, j) if xs[i] >= xs[j] else (j, i)
                xs[right] += shift
                xs[left] -= shift
        if not crowded:
            break
    return xs


def _prior_marker(row):
    """(marker, facecolor, edgecolor) for one prior. Shape carries the kind of prior and fill
    carries tuning status, so neither encoding has to borrow the other's channel: circles are
    neural language models, filled when instruction-tuned and hollow when base; diamonds are
    the character 5-grams and the square is the uniform null, neither of which has a tuning
    status to show. Colour stays free for architecture family throughout.

    Keeping colour here was tested, not assumed. Family colour decodes only against Table 2 and
    eTable 19 now that the in-panel key is gone, and the paper's own permutation test found no
    between-family variance above its null, so the encoding names a grouping this study found
    nothing in - which is a real argument for dropping it to one ink. Drawn and compared at the
    183 mm the journal prints, though, the single-ink version of panel a's cluster is the least
    readable of the three tried: fourteen charcoal markers overlapping within a marker width of
    each other read as one ornament rather than as fourteen priors, because the white halo in
    _draw_prior_point separates a marker from its neighbour but gives a reader nothing to tell
    the two apart by once separated. Hue does that work even when it is not being decoded. So
    colour stays, the halo fixes the occlusion it was the wrong tool for, and the inset in
    _figure3_panel_a is where the cluster is actually made countable.
    """
    if not row["neural"]:
        return ("s" if row["family"] == "null" else "D"), row["color"], row["color"]
    if row["instruction_tuned"]:
        return "o", row["color"], row["color"]
    return "o", "white", row["color"]


def _drawn_radius_pt(area, linewidth, halo=0.0):
    return float(np.sqrt(area)) / 2 + linewidth / 2 + halo


def _text_size_pt(ax, renderer, text, fontsize):
    """The width and height, in points, that `text` renders to at `fontsize`.

    Measured from a probe artist and the live renderer rather than counted from the string,
    because the box a label occupies is a font-metric fact: it depends on which glyphs the
    string holds, on how many lines it breaks over, and on the font apply_style installed. A
    label's own box is the thing the leader search below has to keep off the markers, and every
    earlier attempt to reason about it from a character count put a label on a marker anyway.
    The result is in points and so does not depend on the canvas the probe is drawn on.
    """
    probe = ax.text(0, 0, text, fontsize=fontsize)
    extent = probe.get_window_extent(renderer)
    probe.remove()
    scale = 72.0 / ax.figure.dpi
    return extent.width * scale, extent.height * scale


def _label_box(size, anchor, ha, va):
    """(left, right, bottom, top) of a label of `size` points hung off `anchor` by its alignment.

    Matplotlib places text by moving the box so that the edge named by ha and va lands on the
    anchor, which is what makes a label's footprint reach back across a panel from the point it
    is anchored to. Everything here is in points relative to whatever the anchor is measured
    from.
    """
    width, height = size
    left = anchor[0] - {"left": 0.0, "center": width / 2, "right": width}[ha]
    bottom = anchor[1] - {"bottom": 0.0, "center": height / 2, "top": height}[va]
    return np.array([left, left + width, bottom, bottom + height])


def _leader_label_offset(ha, va):
    """Where the label's anchor sits relative to the leader's tip, for one alignment.

    The alignment decides which quadrant of the tip the box occupies, so it cannot be fixed and
    it cannot be read off the bearing either. Fixed at ha="right", va="top" the box always hangs
    down and to the left, which reads correctly only while the leader runs down and to the left
    as well; when the measured bearing came out near vertical the line ran the full height of its
    own label. Flipping it to follow the bearing instead sent panel a's label, which is one long
    line, off the right-hand edge of its panel. So all four are offered to the sweep and the one
    that clears the panel's contents wins, like the bearing itself.
    """
    horizontal, vertical = FIGURE3_LEADER_LABEL_OFFSET_PT
    return np.array([horizontal if ha == "right" else -horizontal,
                     -vertical if va == "top" else vertical])


def _points_to_box(points, box):
    """Distance in points from each point to the nearest edge of `box`, and zero inside it."""
    points = np.atleast_2d(points)
    return np.hypot(np.maximum(np.maximum(box[0] - points[:, 0], points[:, 0] - box[1]), 0.0),
                    np.maximum(np.maximum(box[2] - points[:, 1], points[:, 1] - box[3]), 0.0))


def _points_to_segment(points, start, end):
    span = np.asarray(end) - np.asarray(start)
    along = np.clip((np.atleast_2d(points) - start) @ span / (span @ span), 0.0, 1.0)
    return np.linalg.norm(start + np.outer(along, span) - points, axis=1)


def _box_gap(one, other):
    """Separation between two boxes: positive when they are apart, negative when they overlap."""
    horizontal = max(other[0] - one[1], one[0] - other[1])
    vertical = max(other[2] - one[3], one[2] - other[3])
    if horizontal >= 0.0 or vertical >= 0.0:
        return float(np.hypot(max(horizontal, 0.0), max(vertical, 0.0)))
    return float(max(horizontal, vertical))


def _segment_to_box(start, end, box):
    """Distance from a segment to a box, and zero when the segment crosses it.

    Once the two are known to be apart, both are convex polygons, so the closest pair includes a
    corner of one of them and checking the two ends against the box and the four corners against
    the segment is exact rather than a sampling of the segment. Whether they are apart has to be
    settled first, though: a segment that runs clean through a wide box has both ends outside it
    and can pass every corner at a distance, which those two checks alone report as clearance.
    That is not a hypothetical - it put eFigure 4's leader through the "B" of its own "32B" - so
    the slab clip below runs first.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    span = end - start
    entering, leaving = 0.0, 1.0
    for axis, edges in enumerate(((box[0], box[1]), (box[2], box[3]))):
        if abs(span[axis]) < 1e-12:
            if not edges[0] <= start[axis] <= edges[1]:
                entering, leaving = 1.0, 0.0
                break
        else:
            crossings = sorted((edge - start[axis]) / span[axis] for edge in edges)
            entering, leaving = max(entering, crossings[0]), min(leaving, crossings[1])
    if entering <= leaving:
        return 0.0
    corners = np.array([[box[0], box[2]], [box[1], box[2]], [box[0], box[3]], [box[1], box[3]]])
    return min(float(_points_to_box(np.array([start, end]), box).min()),
               float(_points_to_segment(corners, start, end).min()))


def _draw_prior_point(ax, x, y, row, size=FIGURE3_MARKER_AREA_PT2):
    marker, face, edge = _prior_marker(row)
    # The primary prior draws above its neighbours. It is the one point both panels ring and
    # name, so its own fill has to be readable: the largest priors sit within a marker width of
    # each other, and in ladder order the instruction-tuned Qwen3.6-35B lands on top of the
    # base, hollow Qwen2.5-32B and makes the ringed point read as instruction-tuned.
    zorder = 6 if row["name"] == PRIMARY_PRIOR else 4
    # The halo is a path effect rather than a second, larger white marker drawn underneath,
    # because a second scatter would be a second artist at the same size and position, and the
    # layout tests read this figure's markers back off its collections by their size.
    ax.scatter([x], [y], s=size, marker=marker, facecolors=face, edgecolors=edge,
               linewidths=FIGURE3_MARKER_LINEWIDTH_PT, zorder=zorder,
               path_effects=[withStroke(
                   linewidth=FIGURE3_MARKER_LINEWIDTH_PT + 2 * FIGURE3_MARKER_HALO_PT,
                   foreground="white")])


def _ring(ax, x, y, size=FIGURE3_RING_AREA_PT2):
    """The primary prior's highlight: a wide open circle around the one prior both panels name.

    Drawn in INK: the fused decision's own colour, and the one colour in this figure that no
    architecture family uses. Haloed on both sides of its own stroke, unlike the markers, whose
    halo only needs to sit outside them, because the ring is drawn across the densest part of
    panel a's cluster: at the marker sizes this figure prints at, the neighbours it crosses used
    to run right up against it on both sides and left a reader hunting for which point was the
    ringed one.
    """
    ax.scatter([x], [y], s=size, marker="o", facecolors="none", edgecolors=INK,
               linewidths=FIGURE3_RING_LINEWIDTH_PT, zorder=5,
               path_effects=[withStroke(
                   linewidth=FIGURE3_RING_LINEWIDTH_PT + 2 * FIGURE3_RING_HALO_PT,
                   foreground="white")])


def _primary_leader(ax, label, centre, others, x_per_unit, y_per_unit, span, arc=None,
                    step_degrees=0.5, placed=(), extra_boxes=()):
    """Draw the leader from the primary prior out to its label, on a measured bearing.

    A leader that crosses, or stops inside, a marker other than the one it points at tells the
    reader that the wrong prior is the labelled one, and a label that lands on a marker does the
    same thing more plainly still. Both panels have had both bugs from labels placed by eye, and
    in clusters this dense the directions that avoid them are narrow and depend on the ladder, so
    the bearing is measured instead of chosen.

    What is measured is the whole annotation, not just the line: for each candidate bearing the
    leader is the segment between `span` points from the prior's centre, the label is the box
    that segment's far end hangs, and the clearance scored is the smallest gap either of them
    leaves to anything already on the panel - every other prior's marker edge, the primary's own
    highlight ring, every label in `placed`, and the panel's own limits. The bearing with the
    largest minimum wins. Modelling the label's box is what this function was missing: scoring
    the segment alone let panel b's two-line label sit squarely on a base Qwen marker while the
    line itself was, correctly, nowhere near one.

    `placed` carries the labels the panel has already drawn, each as
    (text, fontsize, anchor in data units, offset in points, ha, va, draw kwargs) - the same
    arguments they were drawn with, so the box measured here is the box on the page. Only the
    geometry is read; the kwargs are the caller's. The sweep is a fixed grid from a fixed origin,
    so it returns the same answer on every run, and `arc` narrows it to the part of the circle
    worth sweeping.
    """
    scale = np.array([x_per_unit, y_per_unit])
    centre = np.array(centre)
    offsets = (np.array(others) - centre) * scale
    renderer = ax.figure.canvas.get_renderer()
    size = _text_size_pt(ax, renderer, label, FIGURE3_LABEL_FONTSIZE)
    marker_radius = _drawn_radius_pt(FIGURE3_MARKER_AREA_PT2, FIGURE3_MARKER_LINEWIDTH_PT,
                                     FIGURE3_MARKER_HALO_PT)
    ring_radius = _drawn_radius_pt(FIGURE3_RING_AREA_PT2, FIGURE3_RING_LINEWIDTH_PT)
    obstacles = [
        _label_box(_text_size_pt(ax, renderer, text, fontsize),
                   (np.array(anchor) - centre) * scale + np.array(offset), ha, va)
        for text, fontsize, anchor, offset, ha, va, _ in placed
    ]
    # Anything on the panel that is neither a marker nor a label, given as an (x0, x1, y0, y1)
    # rectangle in data units - panel a's inset is the only one so far, and it covers a fifth of
    # that panel, which is far too much of the sweep's reach for the search not to know about.
    obstacles += [(np.array([box[0], box[1], box[2], box[3]], dtype=float)
                   - centre[[0, 0, 1, 1]]) * scale[[0, 0, 1, 1]] for box in extra_boxes]
    limits = np.array([(ax.get_xlim()[0] - centre[0]) * x_per_unit,
                       (ax.get_xlim()[1] - centre[0]) * x_per_unit,
                       (ax.get_ylim()[0] - centre[1]) * y_per_unit,
                       (ax.get_ylim()[1] - centre[1]) * y_per_unit])

    lo, hi = arc if arc is not None else (0.0, 360.0)
    best, chosen = -np.inf, (lo, "right", "top")
    for degrees in np.arange(lo, hi, step_degrees):
        direction = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
        near, far = (reach * direction for reach in span)
        along = np.clip(offsets @ direction, span[0], span[1])
        shaft = (float(np.linalg.norm(offsets - np.outer(along, direction), axis=1).min())
                 - marker_radius)
        for ha, va in (("left", "top"), ("left", "bottom"),
                       ("right", "top"), ("right", "bottom")):
            box = _label_box(size, far + _leader_label_offset(ha, va), ha, va)
            # The leader has to stop at a corner of its own label rather than run under the
            # glyphs, which is the alignment's whole job. This is a gate rather than another term
            # in the minimum below because the tip sits a fixed offset from the box: a placement
            # that reads correctly always scores about 3 points here and one that hangs the text
            # back over the line scores 0, so scoring it would cap every candidate at 3 and leave
            # the sweep unable to tell six points of clearance from three.
            if _segment_to_box(near, far, box) < FIGURE3_LEADER_CLEARANCE_PT:
                continue
            clearance = min(
                shaft,
                float(_points_to_box(offsets, box).min()) - marker_radius,
                float(_points_to_box(np.zeros(2), box)[0]) - ring_radius,
                box[0] - limits[0], limits[1] - box[1], box[2] - limits[2], limits[3] - box[3],
            )
            for obstacle in obstacles:
                clearance = min(clearance, _box_gap(box, obstacle),
                                _segment_to_box(near, far, obstacle))
            if clearance > best:
                best, chosen = clearance, (degrees, ha, va)

    degrees, ha, va = chosen
    direction = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
    start, end = (centre + reach * direction / scale for reach in span)
    ax.plot([start[0], end[0]], [start[1], end[1]], color=MUTED, lw=0.7, zorder=3,
            solid_capstyle="butt")
    # The line is drawn plainly rather than as an annotate arrow because annotate starts its
    # arrow from the label's box, not from the point given to it, which makes the bearing it
    # draws a consequence of the rendered font metrics rather than of the bearing measured above.
    ax.annotate(label, end, textcoords="offset points",
                xytext=tuple(_leader_label_offset(ha, va)),
                fontsize=FIGURE3_LABEL_FONTSIZE, color=INK, ha=ha, va=va)
    return chosen, best


def _measured_label(ax, renderer, text, fontsize, anchor, scale, markers, segments, boxes,
                    limits, candidates=FIGURE3_LABEL_CANDIDATES):
    """Offset and alignment for one point's label, chosen by measuring the box it would occupy.

    The same rule _primary_leader applies to the primary prior's label, applied to the labels
    the panels hang straight off a marker: try each candidate placement, measure the rendered
    box it puts on the page against everything already there, and keep the one that leaves the
    most white. `markers` are the priors, in data units, and the one at `anchor` is skipped
    because being close to its own point is what a label is for. `segments` are the lines a box
    must not be struck through by - a reference line, the frontier's own step - and `boxes` are
    the labels the panel has already committed to, in points.

    A fixed offset is what these labels used to get, and it produced exactly the two failures a
    fixed offset produces. Panel b's frontier labels each sat 7 points to the right of their
    diamond, which was correct until the frontier's two priors came out at the same accuracy
    gain: the step drawn between them then ran horizontally through "5-gram KN" at mid-cap
    height and carried on into the neighbouring diamond, so the label read as struck out and as
    naming the wrong point. The worst-gain label sat 7 points below its marker, which put the
    y = 0 reference line through "Gemma-2 2B" for the same reason - a marker close enough to a
    line that the fixed offset lands the text on it.

    Returns the placement, the box it occupies in points, and the clearance it achieved, so a
    caller can both draw it and hand the box on as an obstacle to whatever it places next.
    """
    anchor_point = np.array(anchor, dtype=float) * scale
    size = _text_size_pt(ax, renderer, text, fontsize)
    others = np.array([point for point in markers
                       if not np.allclose(point, anchor)], dtype=float) * scale
    marker_radius = _drawn_radius_pt(FIGURE3_MARKER_AREA_PT2, FIGURE3_MARKER_LINEWIDTH_PT,
                                     FIGURE3_MARKER_HALO_PT)
    best, chosen = -np.inf, candidates[0]
    for candidate in candidates:
        offset, ha, va = candidate[:2], candidate[2], candidate[3]
        box = _label_box(size, anchor_point + np.array(offset), ha, va)
        clearance = min(
            float(_points_to_box(others, box).min()) - marker_radius,
            box[0] - limits[0], limits[1] - box[1], box[2] - limits[2], limits[3] - box[3],
        )
        for start, end in segments:
            clearance = min(clearance, _segment_to_box(np.array(start) * scale,
                                                       np.array(end) * scale, box))
        for other in boxes:
            clearance = min(clearance, _box_gap(box, other))
        if clearance > best:
            best, chosen = clearance, candidate
    offset, ha, va = chosen[:2], chosen[2], chosen[3]
    return offset, ha, va, _label_box(size, anchor_point + np.array(offset), ha, va), best


def _pareto_frontier(rows):
    """The priors no other prior beats on both axes of panel b, computed from the data rather
    than read off the plot: a prior is dominated when some other prior overturned no more
    correct neural readings AND gained at least as much accuracy, strictly better on one of
    the two. What survives is the benefit-harm frontier, sorted by capture.
    """
    frontier = [
        row for row in rows
        if not any(
            other["capture"] <= row["capture"]
            and other["gain"] >= row["gain"]
            and (other["capture"] < row["capture"] or other["gain"] > row["gain"])
            for other in rows if other["name"] != row["name"]
        )
    ]
    return sorted(frontier, key=lambda row: (row["capture"], row["gain"]))


def _parameter_count_phantom_scatter(ax, rows, digest):
    """Phantom agreement against parameter count, one point per prior.

    This drew Figure 3 panel A before Task 23 moved that panel onto the calibrated, NLL-based
    axis in _figure3_panel_a below; it is no longer called from figure3(). Preserved standalone,
    unchanged, so a future supplementary figure can wire it in without re-deriving it from git
    history. Takes `rows` in the shape _raw_ladder_rows produces - the raw, uncalibrated ladder,
    including the uniform null and each row's `log_parameters` - not the calibrated,
    NLL-carrying rows _figure3_rows produces, which have neither. Its on-panel annotation reads
    the raw-basis `cross_family_scale_check`, `scale_equivalence_test`, and
    `architecture_family_permutation_test` digest keys accordingly; a caller supplying
    calibrated rows would need to also swap in `cross_family_scale_check_calibrated` and
    `architecture_family_permutation_test_calibrated`, and note that `scale_equivalence_test`
    itself has no calibrated counterpart in the digest (Results, "Effect of Parameter Count":
    "raw basis only").
    """
    neural = [row for row in rows if row["neural"]]
    benchmarks = [row for row in rows if not row["neural"]]
    scale = digest["sensitivity"]["cross_family_scale_check"]["pooled"][
        "spearman_rho_phantom_vs_log_params"
    ]
    equivalence = digest["sensitivity"]["scale_equivalence_test"]["phantom_agreement"]
    families = digest["sensitivity"]["architecture_family_permutation_test"][
        "phantom_agreement"
    ]["n_families"]

    ax.set_xlim(*PARAM_SCATTER_XLIM)
    ax.set_ylim(*PARAM_SCATTER_YLIM)
    ax.axvspan(PARAM_SCATTER_XLIM[0], PARAM_SCATTER_BREAK_X, color="#F5F6F8", lw=0, zorder=0)
    ax.axvline(PARAM_SCATTER_BREAK_X, color="#C9CDD2", ls=(0, (3, 2)), lw=0.8, zorder=1)
    ax.grid(axis="y", color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([8, 9, 10])
    ax.set_xticklabels(["100M", "1B", "10B"])
    ax.set_xticks([8.5, 9.5, 10.5], minor=True)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.set_xlabel("Prior parameter count (log scale)")
    ax.set_ylabel("Phantom agreement (%)", color=PHANTOM_COLOR)
    # Every label below is collected here as it is decided and drawn from the list further down,
    # so the boxes the primary's leader is searched against are the boxes on the page.
    placed = [("priors with no\nparameter count", FIGURE3_LABEL_FONTSIZE,
               (PARAM_SCATTER_GUTTER_X, PARAM_SCATTER_YLIM[1] - 0.15), (0, 0), "center", "top",
               dict(color=MUTED, style="italic"))]

    # Priors that share a parameter count would otherwise stack into one marker; see
    # FIGURE3_MIN_SEPARATION_PT. Every x this panel draws - markers, labels, the ring and the
    # leader anchor - comes from this one lookup, so nothing can point at a position no marker
    # occupies.
    x_per_unit, y_per_unit = _points_per_unit("a", PARAM_SCATTER_XLIM, PARAM_SCATTER_YLIM)
    drawn_x = dict(zip(
        [row["name"] for row in neural],
        _separated_x([row["log_parameters"] for row in neural],
                     [row["phantom"] for row in neural], x_per_unit, y_per_unit),
    ))
    # What the separation costs, as a fraction of the parameter count, for the footer to state.
    separation_cost = max(
        10 ** abs(drawn_x[row["name"]] - row["log_parameters"]) - 1 for row in neural
    )

    for row in neural:
        _draw_prior_point(ax, drawn_x[row["name"]], row["phantom"], row)
    # The benchmark priors sit at one shared gutter x, labelled to their right: at this height
    # the parameter-count region of the panel is empty, so the labels have room and nothing
    # about the gutter position can be read as a parameter count.
    for row in benchmarks:
        _draw_prior_point(ax, PARAM_SCATTER_GUTTER_X, row["phantom"], row)
        placed.append((row["label"], FIGURE3_LABEL_FONTSIZE,
                       (PARAM_SCATTER_GUTTER_X + 0.14, row["phantom"]), (0, 0), "left", "center",
                       dict(color=INK)))

    # Which neural priors get a name: the primary prior, the two ends of the parameter range,
    # and the two ends of the phantom-agreement range. All five come from the data, so a
    # changed ladder relabels itself rather than keeping a stale hand-picked list. The offsets
    # are layout only, in points, and are checked against the rendered figure.
    named = {
        PRIMARY_PRIOR: (0, 0),
        max(neural, key=lambda row: row["parameters"])["name"]: (0, 8),
        # Below and to the right rather than level with its point: the smallest prior and the
        # next one up the ladder are 0.8 of a decade apart, which on this canvas is almost
        # exactly the width of this label, so set level it ended inside the neighbour's marker.
        min(neural, key=lambda row: row["parameters"])["name"]: (7, -8),
        # Beside its point rather than above it: above is the band the family key sits in, and a
        # label set on the key's last row reads as one more entry in it whatever the gap.
        max(neural, key=lambda row: row["phantom"])["name"]: (-7, 0),
        # Beside its point rather than under it: below is where the primary prior's leader
        # text lands, and the two labels stacked there ran into each other.
        min(neural, key=lambda row: row["phantom"])["name"]: (-7, 0),
    }
    alignment = {(7, -8): ("left", "top"), (-7, 0): ("right", "center")}
    for row in neural:
        if row["name"] not in named or row["name"] == PRIMARY_PRIOR:
            continue
        offset = named[row["name"]]
        ha, va = alignment.get(offset, ("center", "bottom"))
        placed.append((row["label"], FIGURE3_LABEL_FONTSIZE,
                       (drawn_x[row["name"]], row["phantom"]), offset, ha, va, dict(color=INK)))
    placed.append((
        f"Spearman rho {scale['rho']:+.2f} against log parameter count\n"
        f"(90% CI, {equivalence['ci_low']:+.2f} to {equivalence['ci_high']:+.2f}); "
        f"{_p_text(scale['p_value'])}. {scale['n']} neural priors, {families} families.",
        FIGURE3_LABEL_FONTSIZE, (PARAM_SCATTER_XLIM[1] - 0.05, 1.15), (0, 0), "right", "top",
        dict(color=INK),
    ))
    for text, fontsize, anchor, offset, ha, va, style in placed:
        ax.annotate(text, anchor, textcoords="offset points", xytext=offset, fontsize=fontsize,
                    ha=ha, va=va, **style)

    primary = next(row for row in neural if row["name"] == PRIMARY_PRIOR)
    _ring(ax, drawn_x[primary["name"]], primary["phantom"])
    # A leader rather than a label at the point: the four largest priors overlap there. Where it
    # runs is measured, not chosen; see _primary_leader. The priors in the gutter count as
    # obstacles too even though they sit far to the left, and so does every label above, so that
    # nothing on the panel is invisible to the sweep.
    _primary_leader(
        ax, f"primary prior, {primary['label']}",
        (drawn_x[primary["name"]], primary["phantom"]),
        [(drawn_x[row["name"]], row["phantom"]) for row in neural
         if row["name"] != primary["name"]]
        + [(PARAM_SCATTER_GUTTER_X, row["phantom"]) for row in benchmarks],
        x_per_unit, y_per_unit, PARAM_SCATTER_LEADER_SPAN_PT, PARAM_SCATTER_LEADER_ARC,
        placed=placed,
    )

    present = [family for family in FAMILY_COLOR
               if any(row["family"] == family for row in neural)]
    # The old figure decoded 13 architecture families in a five-column swatch strip below both
    # panels. Only the neural families need decoding now (the gutter priors are labelled where
    # they sit), and the key rides inside the panel whose colours it explains, in the empty
    # band between the neural priors and the 5-gram benchmarks.
    ax.legend(
        [Line2D([], [], color="none", marker="o", ms=3.6, mfc=FAMILY_COLOR[family],
                mec=FAMILY_COLOR[family], lw=0) for family in present],
        [FAMILY_LABEL[family] for family in present],
        loc="upper left", bbox_to_anchor=(0.235, 0.685), ncol=4, fontsize=6.2, frameon=False,
        handlelength=0.65, handletextpad=0.22, columnspacing=0.55, labelspacing=0.35,
        borderaxespad=0.0,
    )
    return separation_cost


def _figure3_panel_a(ax, rows, digest):
    """Phantom agreement against next-character prediction quality, one point per prior.

    Every prior in the calibrated ladder - the 21 neural language models and the 3 character
    5-grams alike - has a next-character NLL (eTable 25), unlike the parameter-count axis
    _parameter_count_phantom_scatter plots, which the 3 character 5-grams and the uniform null
    have no position on at all. Nothing here needs a gutter: every row draws at its own
    measured x.
    """
    correlation = digest["sensitivity"]["prior_quality_correlations_neural_only"][
        "spearman_rho_phantom_vs_nll"
    ]
    # The all-24-prior correlation (5-grams included) is reported in the caption only, not
    # annotated here: an earlier round put it beside this one as a headline in-panel number in
    # response to a reviewer's disclosure request, but the current external review asked to
    # demote it, since it pools two model classes where this one stays inside the neural
    # language models, and a co-equal headline number here overstated how comparable the two
    # correlations are.

    ax.set_xlim(*FIGURE3_A_XLIM)
    ax.set_ylim(*FIGURE3_A_YLIM)
    ax.grid(axis="y", color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
    # Two lines: the axis is only 41.5% of the figure's width, and this label set on one line
    # renders wider than the whole page and runs off its left edge.
    ax.set_xlabel(
        "Projected next-character negative log-likelihood\n"
        "of the intended character (lower = more predictive)"
    )
    ax.set_ylabel("Phantom agreement (%)", color=PHANTOM_COLOR)

    neural = [row for row in rows if row["neural"]]
    ngrams = [row for row in rows if not row["neural"]]

    # Every row shares one continuous x now, so the crowding _parameter_count_phantom_scatter
    # only had to solve for exact parameter-count ties can happen between any two priors placed
    # close in both NLL and phantom agreement; see FIGURE3_MIN_SEPARATION_PT. This cluster is
    # denser than that one - 24 priors span a 3-nat NLL range where the old axis spread the same
    # count over 4 decades - so resolving one pair's overlap here more often reopens another's;
    # the default 6 passes leave one pair 3.450 points apart against a 3.5 point target, and 50
    # converges every pair to 3.500 to within floating-point noise.
    x_per_unit, y_per_unit = _points_per_unit("a", FIGURE3_A_XLIM, FIGURE3_A_YLIM)
    drawn_x = dict(zip(
        [row["name"] for row in rows],
        _separated_x([row["nll"] for row in rows], [row["phantom"] for row in rows],
                     x_per_unit, y_per_unit, passes=50),
    ))
    # What the separation costs, in nats of NLL, for the footer to state.
    separation_cost = max(abs(drawn_x[row["name"]] - row["nll"]) for row in rows)

    for row in rows:
        _draw_prior_point(ax, drawn_x[row["name"]], row["phantom"], row)

    primary = next(row for row in rows if row["name"] == PRIMARY_PRIOR)
    # Which priors get a name: the primary prior (via its leader, below), the neural prior with
    # the best (lowest) NLL, the largest and smallest neural priors by parameter count, and the
    # character 5-gram with the best NLL - the reviewer's requested set, all five derived from
    # the data so a changed ladder relabels itself rather than keeping a stale hand-picked list.
    named = [
        min(neural, key=lambda row: row["nll"]),
        max(neural, key=lambda row: row["parameters"]),
        min(neural, key=lambda row: row["parameters"]),
        min(ngrams, key=lambda row: row["nll"]),
    ]
    # A named prior inside the magnified region is named there rather than here, and the split is
    # read off the region rather than chosen. The one that falls inside it, "GPT-2 124M", sits in
    # the middle of the cluster with priors on all four sides of it within half a label's width,
    # so the panel has nowhere to put a thirty-point label that does not rest on a marker it does
    # not name; in the inset the same label has three times the room and sits against its own
    # marker.
    # The primary prior is not in `named` and so is never subject to this split, which matters
    # because it does fall inside the region: it keeps the ring and the leader that make it the
    # panel's subject wherever it lands, and the caption says so, because a reader who sees that
    # leader running into the dashed box has to be told it is the exception rather than left to
    # decide the figure contradicts itself.
    inside = _in_zoom(drawn_x, FIGURE3_A_INSET_ZOOM)
    magnified = _figure3_panel_a_inset(
        ax, rows, drawn_x, primary, [row for row in named if inside(row)])

    renderer = ax.figure.canvas.get_renderer()
    scale = np.array([x_per_unit, y_per_unit])
    limits = np.array([FIGURE3_A_XLIM[0] * x_per_unit, FIGURE3_A_XLIM[1] * x_per_unit,
                       FIGURE3_A_YLIM[0] * y_per_unit, FIGURE3_A_YLIM[1] * y_per_unit])
    markers = [(drawn_x[row["name"]], row["phantom"]) for row in rows]
    # The dashed outline of the magnified region: a label may sit inside it, next to the marker
    # it names, but must not be struck through by it, which is the same rule the reference line
    # and the frontier step get in panel b.
    left, right, bottom, top = FIGURE3_A_INSET_ZOOM
    segments = [((left, bottom), (right, bottom)), ((right, bottom), (right, top)),
                ((right, top), (left, top)), ((left, top), (left, bottom))]

    # Descriptive, not the prespecified equivalence test _parameter_count_phantom_scatter
    # reports (eTable 18 is parameter-count-specific): restricted to the 21 neural priors so that
    # the correlation measures predictive quality within one model class rather than partly
    # measuring the gap between classes. The 5-grams have the same well-defined NLL and are
    # plotted on this axis; only eTable 25's correlation leaves them out. That reason is
    # stated in the caption rather than repeated here, because a version of this box
    # long enough to explain it right-aligned into the best 5-gram's own label regardless of
    # which empty corner it was moved to: at this panel's width, ha="right" only fixes the box's
    # right edge, and its longest line reached most of the way back across the panel from there.
    # Kept short, and in the bottom left corner: the panel's cloud runs down and to the right, so
    # that corner is the emptiest part of it, and the top right one it used to hold is the only
    # place with room for the inset above.
    placed = [(
        f"rho {correlation['rho']:+.2f}, {_p_text(correlation['p_value'])}",
        FIGURE3_LABEL_FONTSIZE, (FIGURE3_A_XLIM[0] + 0.05, 1.55), (0, 0),
        "left", "top", dict(color=INK),
    )]
    boxes = [_label_box(_text_size_pt(ax, renderer, text, fontsize),
                        np.array(anchor) * scale + np.array(offset), ha, va)
             for text, fontsize, anchor, offset, ha, va, _ in placed]
    boxes.append(magnified * np.array([x_per_unit, x_per_unit, y_per_unit, y_per_unit]))

    # Where each label outside the magnified region goes is measured rather than fixed; see
    # _measured_label.
    for row in [row for row in named if not inside(row)]:
        anchor = (drawn_x[row["name"]], row["phantom"])
        offset, ha, va, box, _ = _measured_label(
            ax, renderer, row["label"], FIGURE3_LABEL_FONTSIZE, anchor, scale,
            markers, segments, boxes, limits,
        )
        placed.append((row["label"], FIGURE3_LABEL_FONTSIZE, anchor, offset, ha, va,
                       dict(color=INK)))
        boxes.append(box)

    # As in panel b: one list, drawn from and searched against, so the leader below cannot be
    # placed against a set of boxes that differs from the set on the page.
    for text, fontsize, anchor, offset, ha, va, style in placed:
        ax.annotate(text, anchor, textcoords="offset points", xytext=offset, fontsize=fontsize,
                    ha=ha, va=va, **style)

    _ring(ax, drawn_x[primary["name"]], primary["phantom"])
    # A leader rather than a label at the point: the dense middle of the NLL cluster overlaps
    # there. Where it runs is measured, not chosen; see _primary_leader.
    _primary_leader(
        ax, f"primary prior, {primary['label']}",
        (drawn_x[primary["name"]], primary["phantom"]),
        [(drawn_x[row["name"]], row["phantom"]) for row in rows
         if row["name"] != primary["name"]],
        x_per_unit, y_per_unit, FIGURE3_LEADER_SPAN_PT["a"], FIGURE3_LEADER_ARC["a"],
        placed=placed, extra_boxes=[magnified],
    )

    return separation_cost


def _in_zoom(drawn_x, zoom):
    """Is this prior inside the magnified region, at the x the panel actually draws it at."""
    left, right, bottom, top = zoom

    def inside(row):
        return (left <= drawn_x[row["name"]] <= right and bottom <= row["phantom"] <= top)

    return inside


def _figure3_panel_a_inset(ax, rows, drawn_x, primary, named):
    """Panel a's dense cluster, drawn again about three times larger.

    Fourteen of the twenty-four priors fall inside a region a fifth of the panel's width and an
    eighth of its height, where _separated_x can only guarantee that each marker keeps a
    crescent of itself visible - enough that no prior is hidden, which is what that separation
    is for, and not enough to read the cluster as fourteen priors. Magnifying the region is what
    makes it countable: the same markers at the same drawn size, the closest pair of them 3.5
    points apart on the panel, land 9.9 points apart here.

    Nothing is recomputed. Every marker draws at the x _separated_x already chose for it on the
    main axis, so a prior sits at the same NLL in both views, and the ring marks the same primary
    prior.

    `named` are the priors the panel names that fall inside this region; they are labelled here,
    where there is room to hang a label off a marker without it landing on another one.

    Returns the rectangle the inset occupies on the main axis, in that axis's data units, for
    the label search and the leader search to treat as occupied.
    """
    inset = ax.inset_axes(_axes_fraction(FIGURE3_A_INSET_RECT, FIGURE3_A_XLIM, FIGURE3_A_YLIM))
    left, right, bottom, top = FIGURE3_A_INSET_ZOOM
    inset.set_xlim(left, right)
    inset.set_ylim(bottom, top)
    inset.set_facecolor("white")
    inset.grid(axis="y", color="#EDEFF2", lw=0.5)
    inset.set_axisbelow(True)
    inset.set_xticks([3.6, 3.8, 4.0])
    inset.set_yticks([3.5, 4.0, 4.5, 5.0])
    inset.tick_params(labelsize=5.8, length=2, pad=1.4, color=MUTED, labelcolor=MUTED)
    # All four sides, against the house style, which drops the top and the right. A panel reads
    # as a panel from its axes; a magnified region reads as a cut-out of the panel behind it, and
    # only a closed frame says where that cut-out ends.
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(0.7)
    for row in rows:
        _draw_prior_point(inset, drawn_x[row["name"]], row["phantom"], row)
    _ring(inset, drawn_x[primary["name"]], primary["phantom"])

    renderer = ax.figure.canvas.get_renderer()
    scale = np.array(_points_per_unit_rect(FIGURE3_A_INSET_RECT, FIGURE3_A_INSET_ZOOM))
    limits = np.array([left * scale[0], right * scale[0], bottom * scale[1], top * scale[1]])
    markers = [(drawn_x[row["name"]], row["phantom"]) for row in rows]
    boxes = []
    for row in named:
        anchor = (drawn_x[row["name"]], row["phantom"])
        offset, ha, va, box, _ = _measured_label(
            inset, renderer, row["label"], FIGURE3_LABEL_FONTSIZE, anchor, scale,
            markers, (), boxes, limits,
        )
        inset.annotate(row["label"], anchor, textcoords="offset points", xytext=offset,
                       fontsize=FIGURE3_LABEL_FONTSIZE, color=INK, ha=ha, va=va, zorder=8)
        boxes.append(box)

    inside = _in_zoom(drawn_x, FIGURE3_A_INSET_ZOOM)
    counted = sum(1 for row in rows if inside(row))
    inset.set_title(f"magnified: the {counted} priors inside the dashed box", loc="left",
                    fontsize=6.2, color=MUTED, style="italic", pad=FIGURE3_A_INSET_TITLE_PAD_PT)
    # The region itself, marked on the main axis. Drawn rather than left to
    # indicate_inset_zoom, whose connector lines would have to run from the corners of this
    # rectangle up across the panel to the corners of the inset: the two on the left would cross
    # the leader to the primary prior, and the pair of them enclose the empty band the panel's
    # correlation annotation sits in. eFigure 3 marks its own magnified region the same way.
    ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom, facecolor="none",
                               edgecolor=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=7))

    # What the rest of the panel has to keep off is not the frame: it is the frame plus the tick
    # labels hanging below and to the left of it and the title sitting above it, and how far
    # those reach is a font-metric fact rather than a number to guess. Measured off the drawn
    # inset and handed back in the main axis's own data units. Guessing at it printed the
    # inset's own x tick labels straight through "Mistral 7B" and "Mixtral 8x7B".
    extent = inset.get_tightbbox(ax.figure.canvas.get_renderer())
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [[extent.x0, extent.y0], [extent.x1, extent.y1]])
    return np.array([x0, x1, y0, y1])


def _figure3_panel_b(ax, rows, digest):
    """Accuracy gained against correct neural readings overturned, with the frontier."""
    ax.set_xlim(*FIGURE3_B_XLIM)
    ax.set_ylim(*FIGURE3_B_YLIM)
    ax.grid(color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=CONTEXT, lw=0.8, zorder=1)
    ax.set_xlabel("Correct neural selections overturned (%)")
    ax.set_ylabel("Accuracy gain from language prior (percentage points)")

    frontier = _pareto_frontier(rows)
    ax.step([row["capture"] for row in frontier], [row["gain"] for row in frontier],
            where="post", color=CONTEXT, ls=(0, (4, 2)), lw=1.0, zorder=2)
    for row in rows:
        _draw_prior_point(ax, row["capture"], row["gain"], row)

    renderer = ax.figure.canvas.get_renderer()
    x_per_unit, y_per_unit = _points_per_unit("b", FIGURE3_B_XLIM, FIGURE3_B_YLIM)
    scale = np.array([x_per_unit, y_per_unit])
    limits = np.array([FIGURE3_B_XLIM[0] * x_per_unit, FIGURE3_B_XLIM[1] * x_per_unit,
                       FIGURE3_B_YLIM[0] * y_per_unit, FIGURE3_B_YLIM[1] * y_per_unit])
    markers = [(row["capture"], row["gain"]) for row in rows]
    # The lines a label must not be struck through by: the y = 0 reference, and the frontier's
    # own step, reconstructed here as the segments where="post" draws rather than as the two
    # points it is given. The step's horizontal run between two frontier priors of equal gain is
    # what used to cross "5-gram KN"; it exists only in the drawn path, not in the data.
    segments = [((FIGURE3_B_XLIM[0], 0.0), (FIGURE3_B_XLIM[1], 0.0)),
                ((2.62, 8.30), (3.52, 6.55))]
    for near, far in zip(frontier, frontier[1:]):
        segments.append(((near["capture"], near["gain"]), (far["capture"], near["gain"])))
        segments.append(((far["capture"], near["gain"]), (far["capture"], far["gain"])))

    # Backs this panel's title: the paired participant-cluster bootstrap comparing each side's
    # own best performer on calibrated accuracy (eMethods; digest key
    # character_model_vs_llm_accuracy), restricted to the two models the ladder actually
    # produced rather than to any fixed pair.
    gap = digest["sensitivity"]["character_model_vs_llm_accuracy"]
    difference = gap["accuracy_difference"]
    summary = _wrap(
        f"{len(frontier)} of {len(rows)} priors on the frontier; "
        f"best-observed 5-gram exceeds best-observed LLM by "
        f"{100 * difference['estimate']:.1f} points",
        _budget(45, FIGURE3_SIZE, FIGURE3_WRAP_REFERENCE_IN),
    )
    # The two blocks that belong to the panel rather than to any one marker keep their fixed
    # corners, and are measured first so the labels that do belong to a marker can see them.
    placed = [
        (summary, FIGURE3_LABEL_FONTSIZE, (FIGURE3_B_XLIM[1] - 0.07, 5.55), (0, 0),
         "right", "top", dict(color=INK)),
        ("better", 6.4, (3.58, 6.45), (0, 0), "right", "top",
         dict(color=MUTED, style="italic")),
    ]
    boxes = [_label_box(_text_size_pt(ax, renderer, text, fontsize),
                        np.array(anchor) * scale + np.array(offset), ha, va)
             for text, fontsize, anchor, offset, ha, va, _ in placed]

    # Labelled from the data: the primary prior (via its leader, below), every prior on the
    # frontier (which is what the panel is about), and the prior that gained the least accuracy.
    # Where each label goes is measured rather than fixed; see _measured_label.
    worst = min(rows, key=lambda row: row["gain"])
    for row in list(frontier) + [worst]:
        anchor = (row["capture"], row["gain"])
        offset, ha, va, box, _ = _measured_label(
            ax, renderer, row["label"], FIGURE3_LABEL_FONTSIZE, anchor, scale,
            markers, segments, boxes, limits,
        )
        placed.append((row["label"], FIGURE3_LABEL_FONTSIZE, anchor, offset, ha, va,
                       dict(color=INK)))
        boxes.append(box)

    # Every label the panel places is drawn from that one list, so the box the leader search is
    # told to avoid below is the box on the page; a label drawn with arguments the list does not
    # carry would be one the search cannot see.
    for text, fontsize, anchor, offset, ha, va, style in placed:
        ax.annotate(text, anchor, textcoords="offset points", xytext=offset, fontsize=fontsize,
                    ha=ha, va=va, **style)
    ax.annotate("", xy=(2.62, 8.30), xytext=(3.52, 6.55),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=MUTED))

    primary = next(row for row in rows if row["name"] == PRIMARY_PRIOR)
    _ring(ax, primary["capture"], primary["gain"])
    # The primary sits inside the cluster, so its leader has to leave through whatever bearing
    # has room; _primary_leader measures which one that is. Three labels placed here without
    # measuring the box they occupy each landed on something: one ran tangent to OLMo-2 7B on its
    # way to the lower right, its replacement stopped inside the filled Llama-3.1-8B-Instruct
    # marker 1.9 points from that centre, and the leader-drawn one that replaced both put
    # "Qwen2.5 32B" over a base Qwen marker. It is placed last, against every label above, and
    # the label breaks over two lines to keep its box clear of the frontier's dashed riser.
    _primary_leader(
        ax, "primary prior,\n" + primary["label"], (primary["capture"], primary["gain"]),
        [(row["capture"], row["gain"]) for row in rows if row["name"] != primary["name"]],
        x_per_unit, y_per_unit, FIGURE3_LEADER_SPAN_PT["b"], FIGURE3_LEADER_ARC["b"],
        placed=placed,
    )


def figure3(digest):
    """The prior ladder along the two questions asked of it: does a prior's own next-character
    prediction quality move its contribution (a), and what does each prior buy against what it
    costs (b).

    Replaces three stacked bar charts over 25 rotated model labels, which asked the reader to
    decode a 13-swatch family legend and to compare bar heights across panels to see either
    relationship. Both are scatter plots now, on the axes the questions are actually about. Task
    23 moved panel a off parameter count, which model scale did not predict (eTable 9, eTable
    23; still drawn in _parameter_count_phantom_scatter above for a future supplementary
    figure), onto next-character NLL, which it does (eTable 25).
    """
    rows = _figure3_rows(digest)
    fig = plt.figure(figsize=FIGURE3_SIZE)
    axes = {letter: fig.add_axes(rect) for letter, rect in FIGURE3_PANELS.items()}

    _figure3_panel_a(axes["a"], rows, digest)
    _figure3_panel_b(axes["b"], rows, digest)

    # Both titles name what the panel plots. Panel a's used to read "No monotonic association
    # ... was detected", which put a null finding where a reader expects a description of the
    # axes and asked them to accept the conclusion before reading the scatter it rests on; the
    # finding itself is still on the panel, as the Spearman rho with its P value, and in the
    # figure caption. Panel b's title is the one exception to "state the axes, not the result":
    # it names who occupies the frontier the panel draws, which is itself a fact about what is
    # plotted, not an interpretation of it, and is gated on Task 16's paired comparison actually
    # holding on the calibrated basis (character_model_vs_llm_accuracy: ngram5 beats
    # DeepSeek-V2-Lite by 6.4 points, 95% CI 4.6 to 8.1, P < .001).
    for letter, title, note in (
        ("a", "Phantom agreement against next-character NLL",
         "one point per prior; circles are neural language models, filled when instruction-tuned "
         "(eTable 19) and hollow when base; diamonds are character 5-grams; colour is family"),
        ("b", "Character 5-grams occupy the accuracy-override frontier",
         "all 24 priors on one plane; up and to the left is better; the ringed point is the "
         "primary prior and diamonds are the character 5-grams"),
    ):
        left, _, width, _ = FIGURE3_PANELS[letter]
        fig.text(left - 0.052, 0.988, letter, fontsize=13, fontweight="bold", color=INK,
                 ha="left", va="top")
        fig.text(left - 0.026, 0.986, title, fontsize=8.2, fontweight="bold", color=INK,
                 ha="left", va="top")
        # Raised with the titles: both are one line now where they were two, and a note left at
        # the old y would hang in a band of white rather than sitting under its own title.
        # 6.2 pt is this figure's floor everywhere, and the canvas is now the 183 mm the journal
        # prints at, so 6.2 pt is what a reader gets rather than the 5.01 pt it used to shrink to.
        # The wrap budget is 112 characters per 0.392 of the canvas's width no longer but 101,
        # which on these panels is 84 characters a line. At 92, panel b's note ran to within
        # three pixels of the rightmost ink in the whole figure, so it had no margin of its own
        # at all: every other block of text on this canvas stops well inside its panel, and that
        # one ended level with panel b's right-hand axis furniture. 84 pulls it back about a
        # third of an inch, on both panels.
        # It does not change the figure's size. save() crops to the tight bounding box, but that
        # box is set by panel b's own axis furniture at 6.97 inches either way, not by this note,
        # which is why the saved canvas is the same width before and after.
        fig.text(left - 0.026, 0.940,
                 _wrap(note, _budget(int(101 * width / 0.392), FIGURE3_SIZE,
                                     FIGURE3_WRAP_REFERENCE_IN)),
                 fontsize=6.2, color=MUTED, ha="left", va="top", style="italic")

    save(fig, str(FIGURES / "Figure3"))
    plt.close(fig)


FIGURE4_SIZE = (7.2, 6.7)
FIGURE4_COLUMNS = ((0.088, 0.222), (0.418, 0.222), (0.748, 0.222))  # (left, width)
FIGURE4_TOP_ROW = (0.660, 0.215)  # (bottom, height) for panels a, b and c
FIGURE4_PROFILE_ROW = (0.160, 0.364)  # (bottom, height) for panel d's three profiles
FIGURE4_FLOOR = 1e-4  # probabilities below this are drawn at the floor of panel d's log axis
FIGURE4_A_XLIM = (-1.30, 1.30)
FIGURE4_A_BADGE_X = -0.75  # the lane panel a's profile badges sit in, clear of the swarm
FIGURE4_A_YLIM = (-0.018, 0.560)
FIGURE4_D_XLIM = (-0.55, 3.10)
FIGURE4_D_YLIM = (4.0e-5, 2.6)
# Panel d's three profiles are the phantom-agreement selections nearest these quantiles of the
# neural margin, among those with at least four characters of preceding context (the eligibility
# rule the previous version of this figure already used for its single worked example, kept so
# that the prior had real linguistic context to condition on). The margin is the quantity the
# reviewer's objection was about, and the quantiles are wide enough that the first profile is an
# actual near-tie: at the quartiles the lowest profile still has the intended character a tenth
# of a probability behind, which is not the phenomenon that label names.
FIGURE4_MIN_CONTEXT = 4
FIGURE4_PROFILE_QUANTILES = (0.10, 0.50, 0.90)
FIGURE4_PROFILE_LABELS = ("Near-tie", "Median case", "Strong conflict")
# What the Figure 4 caption and the Results say the three profiles are. Same guard as Figure 1's,
# and for the same reason, except that here it is not hypothetical: this figure's worked example
# silently drifted from the DECIDE selection the caption described to a different selection when
# the corrected scorer changed the ranking its old rule sorted on, and nothing caught it. The
# selection rule is data-dependent by design, so the figure checks what it picked against what the
# prose claims and refuses to render if they disagree.
FIGURE4_CAPTION_PROFILES = (
    {"context": "OFFI", "target": "C", "neural_only": "K", "margin_percentile": 8},
    {"context": "REMA", "target": "I", "neural_only": "Y", "margin_percentile": 56},
    {"context": "USEF", "target": "U", "neural_only": "1", "margin_percentile": 93},
)
# The cohort numbers the Figure 4 caption and the Results paragraph beside it state outright.
# _check_against_digest already ties the panels to output/stats_digest.json; this ties the prose
# to the panels, which is the half that drifted last time.
FIGURE4_CAPTION_COUNTS = {
    "n_cases": 166,
    "tied": 4,
    "runner_up": 98,
    "third": 35,
    "fourth_or_worse": 29,
    "median_neural_p_target": 0.16,
    "median_margin": 0.11,
}


def _rank_label(rank, short=False):
    """How a neural rank of the intended character is named, in one place.

    ``neural_rank_of_target`` counts the symbols with strictly greater neural probability, so 1
    is the neural posterior's runner-up and 0 means the intended character was tied with its own
    top symbol and lost only the deterministic tie-break. eTable 11 and eTable 26 name the
    categories the same way; panel b's category labels and panel d's profile headers both come
    from here so they cannot drift apart.
    """
    if rank == 0:
        return "tied for top" if short else "Tied with the top symbol"
    if rank == 1:
        return "neural runner-up" if short else "Neural runner-up (2nd)"
    if rank == 2:
        return "third-ranked" if short else "Third-ranked"
    return "fourth-ranked or worse" if short else "Fourth-ranked or worse"


def _calibrated_phantom_frame():
    """Per-selection calibrated fusion carrying every column Figure 4's and eFigure 2's panels
    need, built the same way run_stats.py's main() built its shared `calibrated_primary` frame
    (full_prior_frame with the extra covariate columns, then calibrated_attribution_frame at
    n_folds=5, seed=0 -- see _calibrated_quality_frame's docstring for why seed is fixed here
    only for convention).
    `context_prefix` drives panel d's eligibility rule; `target_symbol` and `intended_phrase`
    are what panel d's per-profile caption names the selection by; `study` is that caption's
    source-study label and also eFigure 2's facet key; `condition` is eFigure 2's per-facet row
    key (Task 26 -- added alongside the other four rather than built as a second, near-identical
    frame). p_neural_calibrated/p_lm_calibrated already ride on every row this frame produces
    (calibrated_fusion_frame's own columns), so unlike the pre-Task-24 version of Figure 4, no
    second join back to selections/priors is needed to rebuild a profile's posteriors.
    """
    selections = pd.read_parquet(OUTPUT / "intermediate" / "selections.parquet")
    priors = pd.read_parquet(OUTPUT / "intermediate" / "priors.parquet")
    frame = full_prior_frame(
        selections, priors, PRIMARY_PRIOR,
        extra_columns=("context_prefix", "target_symbol", "intended_phrase", "study", "condition"),
    )
    return calibrated_attribution_frame(frame, group_column=CLUSTER, n_folds=5, seed=0)


def _phantom_cases(calibrated_frame):
    """The phantom-agreement selections of the primary analysis, one row each.

    Panels a to c are the per-selection distributions behind eTable 26's summary, so this is the
    same subset that table is computed on: calibrated fusion at the primary prior (Task 24 --
    previously raw, uncalibrated fusion, which eTable 11 still reports as the disclosed
    sensitivity comparison), every selection where the fused posterior emitted the intended
    character and the neural posterior alone would not have.
    """
    return calibrated_frame[calibrated_frame["phantom_agreement"].astype(bool)].copy()


def _check_against_digest(cases, digest):
    """Every number panels a to c draw, checked against the digest the manuscript quotes.

    The panels are computed here from a freshly rebuilt calibrated frame rather than from the
    digest, because the digest stores only quantiles and bucket counts and these panels draw the
    cases themselves. That is the same data by two paths, so the two have to agree.
    """
    stored = digest["secondary"]["calibrated"]["phantom_agreement_margin_distribution"]
    p_target = cases["neural_p_target"]
    ranks = cases["neural_rank_of_target"]
    drawn = {
        "n_phantom_agreement_cases": len(cases),
        "neural_p_target_median": float(p_target.median()),
        "neural_p_target_below_0.10_fraction": float((p_target < 0.10).mean()),
        "neural_p_target_between_0.10_and_0.40_fraction": float(
            ((p_target >= 0.10) & (p_target < 0.40)).mean()
        ),
        "neural_p_target_above_0.40_fraction": float((p_target >= 0.40).mean()),
        "neural_margin_to_target_median": float(cases["neural_margin_to_target"].median()),
    }
    for key, value in drawn.items():
        if not np.isclose(value, stored[key], rtol=0, atol=1e-9):
            raise RuntimeError(
                f"Figure 4 computed {key} = {value!r} from the calibrated frame but "
                f"output/stats_digest.json reports {stored[key]!r}. The figure must draw the "
                "distribution eTable 26 summarises; re-run scripts/run_stats.py before "
                "re-rendering."
            )
    # The digest's rank breakdown counts only ranks 1, 2 and 3-or-worse, so it omits the cases
    # tied at the top; eTable 26 states that omission and gives the three shares out of the
    # non-tied cases. Panel b draws all four categories, so only the three are checked here.
    for key, rank in (("1", 1), ("2", 2), ("3_or_worse", 3)):
        drawn_count = int((ranks >= 3).sum() if rank == 3 else (ranks == rank).sum())
        if drawn_count != stored["neural_rank_of_target_distribution"][key]:
            raise RuntimeError(
                f"Figure 4 counted {drawn_count} phantom-agreement cases at neural rank {key} "
                f"but output/stats_digest.json reports "
                f"{stored['neural_rank_of_target_distribution'][key]}; re-run "
                "scripts/run_stats.py before re-rendering."
            )

    # And the same numbers against what the prose beside the figure claims they are.
    claimed = FIGURE4_CAPTION_COUNTS
    counted = {
        "n_cases": len(cases),
        "tied": int((ranks == 0).sum()),
        "runner_up": int((ranks == 1).sum()),
        "third": int((ranks == 2).sum()),
        "fourth_or_worse": int((ranks >= 3).sum()),
        "median_neural_p_target": round(float(p_target.median()), 2),
        "median_margin": round(float(cases["neural_margin_to_target"].median()), 2),
    }
    if counted != claimed:
        raise RuntimeError(
            f"Figure 4's phantom-agreement cohort is {counted}, but the Figure 4 caption in "
            f"scripts/assemble_manuscript.py and the paragraph in manuscript/results.md state "
            f"{claimed}. Update both, and the FIGURE4_CAPTION_COUNTS record in this file, before "
            "re-rendering."
        )


def _figure4_profiles(cases):
    """The three selections panel d draws, with the posteriors behind each.

    Selected by a rule fixed in advance rather than by inspection: among phantom-agreement
    selections with at least four characters of preceding context, the ones nearest the 10th,
    50th and 90th percentiles of the neural margin against the intended character. The posteriors
    are the calibrated per-fold p_neural_calibrated/p_lm_calibrated vectors already carried on
    `cases`'s own rows (Task 24 -- previously a second raw lookup into selections/priors), fused
    the same way calibrated_attribution_frame fused them, and every measure recomputed from that
    fusion is checked against the row `cases` already holds, so the panel cannot draw a
    distribution that disagrees with the case it names.
    """
    pool = cases[cases["context_prefix"].str.len() >= FIGURE4_MIN_CONTEXT]
    profiles = []
    for quantile, label in zip(FIGURE4_PROFILE_QUANTILES, FIGURE4_PROFILE_LABELS):
        wanted = float(pool["neural_margin_to_target"].quantile(quantile))
        row = pool.iloc[
            (pool["neural_margin_to_target"] - wanted).abs().to_numpy().argmin()
        ]
        neural = np.asarray(row["p_neural_calibrated"])
        prior = np.asarray(row["p_lm_calibrated"])
        fused = fuse(neural, prior)
        target_index = symbol_index(row["target_symbol"])
        measures = attribution_row(neural, prior, fused, target_index)
        for key in ("neural_p_target", "neural_margin_to_target", "neural_rank_of_target"):
            if not np.isclose(measures[key], row[key], rtol=0, atol=1e-9):
                raise RuntimeError(
                    f"Figure 4 rebuilt the posteriors for the {label} profile and got {key} = "
                    f"{measures[key]!r}, but the calibrated frame's own already-computed "
                    f"{row[key]!r} disagrees for the same selection."
                )
        if not measures["phantom_agreement"]:
            raise RuntimeError(
                f"Figure 4's {label} profile is not a phantom-agreement selection once its "
                "posteriors are rebuilt; the panel would mislabel it."
            )
        profiles.append({
            "label": label,
            "quantile": quantile,
            "row": row,
            "neural": neural,
            "prior": prior,
            "fused": fused,
            "target_index": target_index,
            "measures": measures,
            # Where this case sits in the whole phantom-agreement cohort, not just in the pool
            # it was chosen from: what panels a and c mark.
            "margin_percentile": float(
                (cases["neural_margin_to_target"] < measures["neural_margin_to_target"]).mean()
            ),
        })

    described = FIGURE4_CAPTION_PROFILES
    observed = tuple(
        {
            "context": profile["row"]["context_prefix"],
            "target": profile["row"]["target_symbol"],
            "neural_only": profile["row"]["neural_only_symbol"],
            "margin_percentile": int(round(100 * profile["margin_percentile"])),
        }
        for profile in profiles
    )
    if observed != described:
        raise RuntimeError(
            f"Figure 4's auto-selected profiles changed from {described} to {observed}. Update "
            "the Figure 4 caption in scripts/assemble_manuscript.py, the paragraph in "
            "manuscript/results.md, and the FIGURE4_CAPTION_PROFILES record in this file before "
            "re-rendering, so the prose cannot describe selections the figure no longer shows."
        )
    return profiles


def _swarm_offsets(values, bin_height, spacing):
    """Lateral offsets for a beeswarm, dealt outwards from the centre line within each band of
    the value axis. Deterministic by construction, unlike a random jitter, so the same data
    always draws the same panel and a changed panel means changed data."""
    values = np.asarray(values, dtype=float)
    offsets = np.zeros(len(values))
    bands = np.floor((values - values.min()) / bin_height).astype(int)
    for band in np.unique(bands):
        members = np.flatnonzero(bands == band)
        members = members[np.argsort(values[members], kind="stable")]
        place = np.arange(len(members))
        offsets[members] = spacing * np.where(place % 2 == 0, place // 2, -(place // 2 + 1))
    return offsets


def _profile_badge(ax, x, y, number, size=52, fontsize=5.6, zorder=8):
    """The numbered badge that ties a panel d profile to its position in panels a to c. Drawn in
    INK, the fused decision's colour and the one colour in this figure that stands for neither
    source on its own."""
    ax.scatter([x], [y], s=size, marker="o", facecolors=INK, edgecolors="white", linewidths=0.8,
               zorder=zorder, clip_on=False)
    ax.text(x, y, str(number), fontsize=fontsize, fontweight="bold", color="white", ha="center",
            va="center", zorder=zorder + 1, clip_on=False)


def _figure4_panel_a(ax, cases, profiles):
    """Every phantom-agreement case as one point, by the neural probability on the intended
    character."""
    values = cases["neural_p_target"].to_numpy(dtype=float)
    ax.set_xlim(*FIGURE4_A_XLIM)
    ax.set_ylim(*FIGURE4_A_YLIM)
    ax.grid(axis="y", color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks([])
    ax.set_ylabel("Neural probability on the\nintended character")

    for threshold in (0.10, 0.40):
        ax.axhline(threshold, color="#C9CDD2", ls=(0, (3, 2)), lw=0.8, zorder=1)
    bands = (
        (values < 0.10, 0.045, "below 0.10"),
        ((values >= 0.10) & (values < 0.40), 0.245, "0.10 to 0.40"),
        (values >= 0.40, 0.470, "0.40 or above"),
    )
    for mask, y, name in bands:
        ax.text(FIGURE4_A_XLIM[1] - 0.05, y, f"{100 * mask.mean():.1f}%\n{name}", fontsize=5.8,
                color=MUTED, ha="right", va="center")

    offsets = _swarm_offsets(values, bin_height=0.0165, spacing=0.062)
    ax.scatter(offsets, values, s=11, color=KEY, alpha=0.55, lw=0, zorder=3)

    # The median and interquartile range in their own lane at the left, with the values given in
    # the panel's note rather than beside the bar: the swarm is at its widest across the
    # interquartile range, which is exactly where a label there would land.
    median = float(np.median(values))
    quartiles = np.percentile(values, [25, 75])
    ax.plot([-1.02, -1.02], quartiles, color=INK, lw=1.2, solid_capstyle="butt", zorder=4)
    ax.plot([-1.08, -0.96], [median, median], color=INK, lw=1.4, zorder=4)

    # The three profiles, on their own points: a reader can see where each drawn case sits in
    # the cohort rather than taking the panel d headers on trust. The observation itself is
    # simply redrawn in INK, and the numbered badge goes in the empty lane to the left at the
    # same height. Neither an opaque badge dropped on the swarm nor a ring around the point
    # works here: measured against the rendered swarm, a badge covers up to nine of the
    # observations beside it and a ring wide enough to be seen encloses twelve, so it would mark
    # a neighbourhood rather than a case. A panel whose subject is the shape of a distribution
    # cannot erase part of that distribution to label it. The lane's x is measured to clear the
    # widest band of the swarm.
    lookup = dict(zip(cases.index, offsets))
    for number, profile in enumerate(profiles, start=1):
        value = profile["measures"]["neural_p_target"]
        position = lookup[profile["row"].name]
        # Under the swarm, not over it: the badge and its observation share a height, and the
        # hairline that joins them passes beneath every point it crosses.
        ax.plot([FIGURE4_A_BADGE_X + 0.09, position], [value, value], color="#C9CDD2", lw=0.5,
                zorder=1)
        ax.scatter([position], [value], s=18, color=INK, lw=0, zorder=6)
        _profile_badge(ax, FIGURE4_A_BADGE_X, value, number)


def _figure4_panel_b(ax, cases, profiles):
    """Where the intended character stood in the neural posterior's own ranking."""
    counts = [
        int((cases["neural_rank_of_target"] == 0).sum()),
        int((cases["neural_rank_of_target"] == 1).sum()),
        int((cases["neural_rank_of_target"] == 2).sum()),
        int((cases["neural_rank_of_target"] >= 3).sum()),
    ]
    total = len(cases)
    positions = np.arange(len(counts))[::-1]  # best rank at the top
    ax.barh(positions, counts, height=0.44, color=KEY, zorder=2)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.72, 3.78)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel(f"Phantom-agreement selections (n = {total})")
    ax.grid(axis="x", color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    # Category names above their own bars rather than as y tick labels: as tick labels they are
    # wide enough to run out of this panel and into panel a's plotting area.
    for rank, (position, count) in enumerate(zip(positions, counts)):
        ax.text(0.8, position + 0.27, _rank_label(rank), fontsize=6.4, color=INK, ha="left",
                va="bottom")
        ax.text(count + 3.0, position, f"{count} ({100 * count / total:.1f}%)", fontsize=6.2,
                color=INK, ha="left", va="center")

    # A badge inside the bar its profile belongs to. Two profiles share the runner-up bar, so
    # they are spaced along it rather than stacked on one x.
    used = {}
    for number, profile in enumerate(profiles, start=1):
        rank = min(int(profile["measures"]["neural_rank_of_target"]), 3)
        slot = used.get(rank, 0)
        used[rank] = slot + 1
        _profile_badge(ax, 7.0 + 11.0 * slot, positions[rank], number)


def _figure4_panel_c(ax, cases, profiles):
    """The distribution of how far behind the intended character actually was."""
    margins = np.sort(cases["neural_margin_to_target"].to_numpy(dtype=float))
    cumulative = np.arange(1, len(margins) + 1) / len(margins)
    ax.step(margins, cumulative, where="post", color=KEY, lw=1.6, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Neural margin against the\nintended character")
    ax.set_ylabel("Cumulative fraction of cases")
    ax.grid(color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)

    quartiles = np.percentile(margins, [25, 50, 75])
    ax.plot([quartiles[1], quartiles[1]], [0, 0.5], color=CONTEXT, ls=":", lw=1.1, zorder=2)
    ax.plot([-0.03, quartiles[1]], [0.5, 0.5], color=CONTEXT, ls=":", lw=1.1, zorder=2)
    ax.text(quartiles[1] + 0.03, 0.10,
            f"median {quartiles[1]:.2f}\nIQR {quartiles[0]:.2f} to {quartiles[2]:.2f}",
            fontsize=5.8, color=MUTED, ha="left", va="bottom")

    for number, profile in enumerate(profiles, start=1):
        margin = profile["measures"]["neural_margin_to_target"]
        _profile_badge(ax, margin, profile["margin_percentile"], number)


def _figure4_panel_d(ax, profile, number, first):
    """One selection's three posteriors as aligned dot columns, the intended character's own
    trajectory drawn through them."""
    neural, prior, fused = profile["neural"], profile["prior"], profile["fused"]
    target_index = profile["target_index"]
    neural_pick = int(neural.argmax())
    prior_pick = int(prior.argmax())

    # Which symbols are worth drawing: the candidates any of the three distributions ranks near
    # the top, plus the intended character, which in a phantom-agreement case is by definition
    # not the neural posterior's own choice.
    candidates = list(dict.fromkeys(
        [target_index]
        + [int(i) for i in np.argsort(fused)[::-1][:4]]
        + [int(i) for i in np.argsort(neural)[::-1][:3]]
        + [int(i) for i in np.argsort(prior)[::-1][:3]]
    ))
    label_of = lambda index: "SP" if SYMBOLS[index] == " " else SYMBOLS[index]  # noqa: E731

    ax.set_yscale("log")
    ax.set_xlim(*FIGURE4_D_XLIM)
    ax.set_ylim(*FIGURE4_D_YLIM)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Neural", "Prior", "Fused"], fontsize=6.4)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.grid(axis="y", color="#EDEFF2", lw=0.6)
    ax.set_axisbelow(True)
    if first:
        ax.set_ylabel("Probability (log scale)")
    else:
        ax.tick_params(labelleft=False)

    height = lambda index: np.maximum(  # noqa: E731
        [neural[index], prior[index], fused[index]], FIGURE4_FLOOR
    )
    for index in candidates:
        if index == target_index:
            continue
        ax.plot([0, 1, 2], height(index), color=CONTEXT, lw=0.7, alpha=0.55, zorder=2)
    ax.plot([0, 1, 2], height(target_index), color=INK, lw=1.6, zorder=4)
    for x, source, colour in ((0, neural, KEY), (1, prior, ACCENT3), (2, fused, INK)):
        values = np.maximum([source[index] for index in candidates], FIGURE4_FLOOR)
        ax.scatter([x] * len(candidates), values, s=13, color=colour, lw=0, zorder=5)
        ax.scatter([x], [max(source[target_index], FIGURE4_FLOOR)], s=30, facecolors=colour,
                   edgecolors="white", linewidths=0.8, zorder=6)

    # Three labels, each anchored on the dot it names and offset in points, with no arrows: the
    # intended character where the fused posterior leaves it, and each source's own top choice
    # above its own column, which is the highest dot in that column and so has clear space above
    # it. The two top-choice labels lean away from each other, the neural one to the right of its
    # column and the prior one to the left of its own, so that neither can reach the other and
    # neither runs off the edge of a panel this narrow.
    ax.annotate(f"'{label_of(target_index)}' intended,\nemitted", (2, max(fused[target_index],
                FIGURE4_FLOOR)), textcoords="offset points", xytext=(6, 0), fontsize=6.7,
                fontweight="bold", color=INK, ha="left", va="center")
    ax.annotate(f"top '{label_of(neural_pick)}'", (0, neural[neural_pick]),
                textcoords="offset points", xytext=(3, 4), fontsize=6.7, color=KEY,
                ha="left", va="bottom")
    if prior_pick != target_index:
        ax.annotate(f"top '{label_of(prior_pick)}'", (1, prior[prior_pick]),
                    textcoords="offset points", xytext=(-3, 4), fontsize=6.7, color=ACCENT3,
                    ha="right", va="bottom")


def figure4(digest):
    """The phantom-agreement cohort, not one case from it: how much neural evidence the intended
    character actually had in every selection the calibrated fusion rule corrected (a to c), and
    three of those selections in full (d).

    Replaces a single worked example, which a reviewer read as one dramatic case standing in for
    a population it is not representative of. The single case is still here, as the third of
    three profiles, but now with the distribution it came from drawn around it.

    Colour grammar, matching Figures 1 to 3: KEY (deep blue) is neural evidence, ACCENT3
    (vermilion) is the language-model prior, INK (charcoal) is the fused decision, and grey is
    reference material only. The previous version of this figure had blue and grey the other way
    round, drawing the neural posterior in the reference grey and the fused posterior in blue.
    """
    cases = _phantom_cases(_calibrated_phantom_frame())
    _check_against_digest(cases, digest)
    profiles = _figure4_profiles(cases)

    fig = plt.figure(figsize=FIGURE4_SIZE)
    top = [fig.add_axes([left, FIGURE4_TOP_ROW[0], width, FIGURE4_TOP_ROW[1]])
           for left, width in FIGURE4_COLUMNS]
    _figure4_panel_a(top[0], cases, profiles)
    _figure4_panel_b(top[1], cases, profiles)
    _figure4_panel_c(top[2], cases, profiles)

    p_target = cases["neural_p_target"]
    for (left, width), letter, title, note in zip(
        FIGURE4_COLUMNS, "abc",
        ("Neural probability on the\nintended character",
         "Neural rank of the\nintended character",
         "Neural margin from its\nown top choice"),
        (f"median {p_target.median():.2f}, IQR {p_target.quantile(0.25):.2f}-"
         f"{p_target.quantile(0.75):.2f} (bar at left)",
         "rank = symbols scored higher by the neural posterior",
         "gap: neural top probability minus intended character's"),
    ):
        fig.text(left - 0.062, 0.988, letter, fontsize=13, fontweight="bold", color=INK,
                 ha="left", va="top")
        fig.text(left - 0.036, 0.986, title, fontsize=8.2, fontweight="bold", color=INK,
                 ha="left", va="top")
        fig.text(left - 0.036, 0.928, _wrap(note, 46), fontsize=5.9, color=MUTED, ha="left",
                 va="top", style="italic")

    fig.text(FIGURE4_COLUMNS[0][0] - 0.062, 0.590, "d", fontsize=13, fontweight="bold",
             color=INK, ha="left", va="top")
    fig.text(FIGURE4_COLUMNS[0][0] - 0.036, 0.588,
             "Representative prior-mediated reclassifications", fontsize=8.2, fontweight="bold",
             color=INK, ha="left", va="top")

    for number, ((left, width), profile) in enumerate(zip(FIGURE4_COLUMNS, profiles), start=1):
        axis = fig.add_axes([left, FIGURE4_PROFILE_ROW[0], width, FIGURE4_PROFILE_ROW[1]])
        _figure4_panel_d(axis, profile, number, first=number == 1)
        row = profile["row"]
        # Sits close above its own axis, not centred in the band below the "d" header: with a
        # single line of label text per card, centring it there would leave a gap that reads as
        # missing content rather than as deliberate spacing. y=0.535/0.548 (was 0.490/0.503) close
        # a measured 0.045-fraction (0.30 in) blank strip left when the header's own italic
        # subtitle moved into the caption; the profile row below grew by the same amount it freed.
        badge = fig.add_axes([left - 0.036, 0.535, 0.02, 0.02])
        badge.set_xlim(0, 1)
        badge.set_ylim(0, 1)
        badge.axis("off")
        _profile_badge(badge, 0.5, 0.5, number, size=64, fontsize=6.2)
        fig.text(left - 0.014, 0.548, f"{profile['label']}: {row['intended_phrase']}",
                 fontsize=7.4, fontweight="bold", color=INK, ha="left", va="top")

    save(fig, str(FIGURES / "Figure4"))
    plt.close(fig)

    return {
        "n_phantom_agreement_cases": len(cases),
        "neural_p_target_median": float(cases["neural_p_target"].median()),
        "neural_margin_to_target_median": float(cases["neural_margin_to_target"].median()),
        "profiles": [
            {
                "label": profile["label"],
                "quantile": profile["quantile"],
                "phrase": profile["row"]["intended_phrase"],
                "context": profile["row"]["context_prefix"],
                "target": profile["row"]["target_symbol"],
                "neural_only": profile["row"]["neural_only_symbol"],
                "prior_only": profile["row"]["prior_only_symbol"],
                "emitted": profile["row"]["emitted_symbol"],
                "study": profile["row"]["study"],
                "neural_p_neural_argmax": float(profile["neural"].max()),
                "neural_p_target": profile["measures"]["neural_p_target"],
                "neural_margin_to_target": profile["measures"]["neural_margin_to_target"],
                "neural_rank_of_target": profile["measures"]["neural_rank_of_target"],
                "margin_percentile": profile["margin_percentile"],
            }
            for profile in profiles
        ],
    }


def _fusion_exponent_points(digest, attribution):
    """Both co-primaries at each point of the fusion-exponent grid, with intervals.

    The digest stores only a point estimate per exponent, so the intervals are bootstrapped
    here from the same per-selection frame run_stats reads. Every point estimate is checked
    against the one the digest already holds, and the beta = 1 column against the interval
    run_stats bootstrapped for sensitivity.uncalibrated_beta_1, so the figure cannot draw a
    grid the Results do not describe or an interval computed a second way.
    """
    grid = digest["sensitivity"]["fusion_exponent"]
    frame = cast_binary_columns(attribution)
    frame = frame[frame["prior_model"] == PRIMARY_PRIOR]
    stored_primary = digest["sensitivity"]["uncalibrated_beta_1"]
    checked = {
        "ncf": stored_primary["coprimary_1_neural_contribution_fraction"],
        "phantom_agreement": stored_primary["coprimary_2_phantom_agreement"],
    }

    records = []
    for key in sorted(grid, key=float):
        beta = float(key)
        group = frame[frame["beta"] == beta]
        record = {"beta": beta, "n": int(len(group))}
        for measure in ("phantom_agreement", "ncf"):
            interval = participant_bootstrap(group, measure)
            expected = [("estimate", grid[key][measure])]
            if beta == PRIMARY_BETA:
                expected += [(field, checked[measure][field])
                             for field in ("ci_low", "ci_high")]
            for field, stored in expected:
                if not np.isclose(interval[field], stored, rtol=0, atol=1e-9):
                    raise RuntimeError(
                        f"eFigure 1 recomputed {measure} at fusion exponent {beta:g} and got "
                        f"{field} = {interval[field]:.6g}, but output/stats_digest.json "
                        f"reports {stored:.6g}. The figure must draw the grid the Results "
                        "describe; re-run scripts/run_stats.py before re-rendering."
                    )
            record[measure] = interval
        records.append(record)
    return records


def _fusion_exponent_points_calibrated(digest):
    """Calibrated-basis counterpart of _fusion_exponent_points: same self-verification guard,
    against sensitivity.fusion_exponent_calibrated instead of the raw grid, plus a cross-check
    of the beta = 1 column against calibrated_primary_analysis, the calibrated co-primary
    estimate that beta = 1 must reproduce exactly (calibrated_fusion_exponent_points fuses the
    same calibrated_primary posteriors calibrated_primary_analysis does, at the same default
    beta). The calibrated grid is not in attribution.parquet -- that file holds only the raw,
    uncalibrated fusion -- so there is no on-disk source of truth to filter the way the raw
    guard filters attribution.parquet; calibrated_fusion_exponent_selections() instead refits
    the held-out calibration temperatures from scratch, independently of run_stats.py's own
    calibrated_primary, so this checks a fresh recomputation against the stored digest rather
    than reading the same number back at itself.
    """
    grid = digest["sensitivity"]["fusion_exponent_calibrated"]
    per_beta = calibrated_fusion_exponent_selections(tuple(float(key) for key in grid))
    checked = digest["calibrated_primary_analysis"]

    records = []
    for key in sorted(grid, key=float):
        beta = float(key)
        group = per_beta[beta]
        record = {"beta": beta, "n": int(len(group))}
        for measure in ("phantom_agreement", "ncf"):
            interval = participant_bootstrap(group, measure)
            expected = [("estimate", grid[key][measure])]
            if beta == PRIMARY_BETA:
                expected += [(field, checked[measure][field])
                             for field in ("ci_low", "ci_high")]
            for field, stored in expected:
                if not np.isclose(interval[field], stored, rtol=0, atol=1e-9):
                    raise RuntimeError(
                        f"eFigure 1 recomputed calibrated-basis {measure} at fusion exponent "
                        f"{beta:g} and got {field} = {interval[field]:.6g}, but "
                        f"output/stats_digest.json reports {stored:.6g}. The figure must draw "
                        "the grid the Results describe; re-run scripts/run_stats.py before "
                        "re-rendering."
                    )
            record[measure] = interval
        records.append(record)
    return records


def _fusion_exponent_row(ax_phantom, ax_ncf, records, basis_title):
    """Shared plotting body for one row of eFigure 1's fusion-exponent grid (phantom agreement,
    then NCF), reused for the raw row and the calibrated row so the two carry identical
    categorical-axis and filled/open-point styling and cannot silently drift apart. basis_title
    ("Raw, uncalibrated fusion" / "Calibrated fusion") is set as ax_phantom's left-aligned title,
    since it is what tells the two rows' otherwise-identical axes apart; the y-axis labels
    themselves stay exactly as before so they cannot grow long enough to crowd the panel letter
    in the corner above them.
    """
    positions = np.array([record["beta"] for record in records])
    primary = next(index for index, record in enumerate(records)
                   if record["beta"] == PRIMARY_BETA)
    sensitivity_only = [index for index in range(len(records)) if index != primary]

    for ax, measure, colour, scale, ylabel in (
        (ax_phantom, "phantom_agreement", PHANTOM_COLOR, 100.0, "Phantom agreement (%)"),
        (ax_ncf, "ncf", NCF_COLOR, 1.0, "Neural contribution fraction"),
    ):
        estimate = np.array([record[measure]["estimate"] for record in records]) * scale
        low = np.array([record[measure]["ci_low"] for record in records]) * scale
        high = np.array([record[measure]["ci_high"] for record in records]) * scale
        ax.axvline(positions[primary], color="#D8DCE0", ls=(0, (3, 2)), lw=0.9, zorder=0)
        ax.plot(positions, estimate, color=colour, lw=1.1, alpha=0.45, zorder=2)
        ax.errorbar(positions, estimate, yerr=[estimate - low, high - estimate], fmt="none",
                    ecolor=colour, elinewidth=1.2, capsize=2.6, zorder=4)
        # Filled is the prespecified exponent the whole paper reports; open are the four
        # sensitivity-only points, which no other result is computed at.
        ax.scatter(positions[sensitivity_only], estimate[sensitivity_only], s=32,
                   facecolor="white", edgecolor=colour, linewidths=1.3, zorder=5)
        ax.scatter([positions[primary]], [estimate[primary]], s=46, color=colour,
                   edgecolor="white", linewidths=0.7, zorder=6)
        # log2, not log10: the grid (0.25, 0.5, 1, 2, 4) is exactly 2**-2..2**2, so base-2 places
        # every point at an integer power and all five land evenly spaced; minor ticks are off
        # because only these five values are meaningful on this axis.
        ax.set_xscale("log", base=2)
        ax.set_xticks(positions)
        ax.set_xticklabels([
            f"{record['beta']:g}\n(prespecified)" if index == primary else f"{record['beta']:g}"
            for index, record in enumerate(records)
        ])
        ax.get_xticklabels()[primary].set_fontweight("bold")
        ax.minorticks_off()
        ax.set_xlim(0.20, 5.0)
        ax.set_xlabel("Fusion exponent on neural evidence (beta)")
        # In its measure's own colour, as in Figures 2 and 3, which is the rule wherever two
        # measures sit side by side and the colour is what says which axis belongs to which:
        # this panel pair had its data correctly split into vermilion and blue but labelled
        # both axes in black, so the split stopped at the edge of the plotting area.
        ax.set_ylabel(ylabel, color=colour)
        span = high.max() - low.min()
        ax.set_ylim(low.min() - 0.14 * span, high.max() + 0.10 * span)
        ax.grid(axis="y", color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
    # Left-aligned, starting at the axis's own left edge (x = 0 in axes fraction): the panel
    # letter panel() draws sits further left still (x = -0.16, outside the axis), so the two
    # never compete for the same horizontal space the way a longer y-axis label would.
    ax_phantom.set_title(basis_title, loc="left", fontsize=7.6, fontweight="bold", color=INK,
                          pad=10)


def efigure1(digest, attribution):
    """Sensitivity of both co-primaries to the fusion exponent, raw basis (a, b) and
    calibrated basis (c, d).

    Colour grammar as in Figure 2: ACCENT3 is phantom agreement, KEY is the neural
    contribution fraction. The x-axis is log2-scaled, not log10: the grid (0.25, 0.5, 1, 2, 4)
    is exactly 2**-2 through 2**2, so a log10 axis clustered four of the five points near the
    origin (the earlier version of this panel, which used log10, showed a single labelled
    value for that reason and was replaced with an equal-spacing categorical axis as a
    workaround). Base 2 places every grid point at an integer power, so all five land evenly
    spaced with genuine numerical meaning instead of an arbitrary equal-spacing convention.
    The calibrated row is additive, alongside the raw row this
    panel always reported, not a replacement for it: raw fusion remains the basis every other
    sensitivity analysis in this supplement is computed on (a reviewer's own suggestion was
    that the calibrated exponent sensitivity could sit in "a second panel").
    """
    records = _fusion_exponent_points(digest, attribution)
    records_calibrated = _fusion_exponent_points_calibrated(digest)

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.8))
    _fusion_exponent_row(axes[0, 0], axes[0, 1], records, "Raw, uncalibrated fusion")
    _fusion_exponent_row(axes[1, 0], axes[1, 1], records_calibrated, "Calibrated fusion")
    panel(axes[0, 0], "a")
    panel(axes[0, 1], "b")
    panel(axes[1, 0], "c")
    panel(axes[1, 1], "d")

    fig.tight_layout(h_pad=3.0)
    save(fig, str(FIGURES / "eFigure1"))
    plt.close(fig)


EFIGURE2_SIZE = (6.5, 4.0)
EFIGURE2_N_COLUMN = 1.035  # axes fraction: the selections column, in the right-hand margin
# Widened from the raw basis's (-0.55, 10.8): the calibrated basis's largest cell (Study N, dry
# electrodes) has a 95% CI upper bound of 12.3%, above the raw basis's 10.0% high (Task 26).
EFIGURE2_XLIM = (-0.55, 13.3)


def _efigure2_cells(digest):
    """One record per study and condition that actually occurs, with its interval, on the
    calibrated-fusion basis (Methods) -- the same basis as Figure 1's co-primary estimates
    and of Table 2 (Task 26; previously the raw, uncalibrated fusion eTable 8 still reports).

    Conditions are nested in studies in this archive: each source study ran its own set, so
    most study-by-condition combinations do not exist and a study-by-condition matrix is
    mostly empty. These are the combinations that exist, which is what the figure draws.
    """
    frame = _calibrated_phantom_frame()

    # The per-condition estimates pooled across studies are already in the digest
    # (sensitivity.calibrated.by_condition, Task 8). Checking against them ties this frame,
    # and this measure, to the number the digest carries before any of it is drawn a second,
    # finer way.
    for condition, group in frame.groupby("condition"):
        stored = digest["sensitivity"]["calibrated"]["by_condition"][condition]
        pooled = participant_mean(group, "phantom_agreement")
        if not np.isclose(pooled, stored["phantom_agreement"], rtol=0, atol=1e-9):
            raise RuntimeError(
                f"eFigure 2 recomputed calibrated phantom agreement in condition {condition} "
                f"as {pooled:.6g}, but output/stats_digest.json reports "
                f"{stored['phantom_agreement']:.6g}. Re-run scripts/run_stats.py before "
                "re-rendering."
            )
        if int(len(group)) != int(stored["n"]):
            raise RuntimeError(
                f"eFigure 2 found {len(group)} selections in condition {condition}, but "
                f"output/stats_digest.json reports {stored['n']}."
            )

    cells = []
    for (study, condition), group in frame.groupby(["study", "condition"]):
        cells.append({
            "study": study,
            "condition": condition,
            "n": int(len(group)),
            "participants": int(group[CLUSTER].nunique()),
            **participant_bootstrap(group, "phantom_agreement"),
        })
    return cells


def efigure2(digest, attribution):
    """Phantom agreement by source study and stimulation condition, as a faceted dot plot, on
    the calibrated-fusion basis (Methods) -- the same basis as Figure 1's co-primary estimates
    and of Table 2 (Task 26; previously the raw, uncalibrated fusion eTable 8 still reports as
    a disclosed sensitivity comparison).

    One facet per source study, one row per condition that study ran, point area increasing
    with the selections behind the estimate and bars its participant-clustered 95% interval.
    The area mapping is affine, not proportional - a floor keeps the smallest sample's point
    legible - so the legend below the panels and the count printed beside each row, not the
    ratio of two areas, are what a reader should decode the sample from.
    The previous version was a study-by-condition heatmap, of which 23 of 32 cells were empty
    because conditions are nested in studies; the sample behind each shaded cell, which ranges
    over threefold, was not shown at all.

    `attribution` is accepted but no longer read: the raw attribution frame this figure used to
    draw from is superseded by _efigure2_cells's own calibrated frame (Task 26). The parameter
    stays so the call in main() and the shared `_rendered(draw)` test helper, both of which pass
    `(digest, attribution)` to every supplementary figure, do not need a figure-specific branch.
    """
    cells = _efigure2_cells(digest)
    cohort = digest["calibrated_primary_analysis"]["phantom_agreement"]
    studies = [study for study in STUDY_LABEL if any(c["study"] == study for c in cells)]
    per_study = {
        study: sorted((c for c in cells if c["study"] == study),
                      key=lambda cell: -cell["estimate"])
        for study in studies
    }
    largest_n_participants = max(cell["participants"] for cell in cells)

    fig = plt.figure(figsize=EFIGURE2_SIZE)
    grid = fig.add_gridspec(
        len(studies), 1, height_ratios=[len(per_study[study]) + 0.75 for study in studies],
        left=0.30, right=0.875, top=0.945, bottom=0.215, hspace=0.55,
    )
    axes = [fig.add_subplot(grid[index]) for index in range(len(studies))]

    for ax, study in zip(axes, studies):
        rows = per_study[study]
        y = np.arange(len(rows))
        estimate = np.array([100 * row["estimate"] for row in rows])
        low = np.array([100 * row["ci_low"] for row in rows])
        high = np.array([100 * row["ci_high"] for row in rows])
        ax.axvline(100 * cohort["estimate"], color="#D8DCE0", ls=(0, (3, 2)), lw=0.9, zorder=1)
        ax.errorbar(estimate, y, xerr=[estimate - low, high - estimate], fmt="none",
                    ecolor=PHANTOM_COLOR, elinewidth=1.2, capsize=2.4, zorder=4)
        ax.scatter(estimate, y, s=[16 + 74 * row["participants"] / largest_n_participants
                                    for row in rows],
                   color=PHANTOM_COLOR, edgecolor="white", linewidths=0.7, zorder=5)
        for index, row in enumerate(rows):
            ax.text(EFIGURE2_N_COLUMN, index, f"{row['n']:,}", fontsize=7, color=INK,
                    ha="left", va="center", transform=ax.get_yaxis_transform(),
                    clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels([CONDITION_LABEL[row["condition"]] for row in rows])
        ax.set_ylim(len(rows) - 0.45, -0.75)
        ax.set_xlim(*EFIGURE2_XLIM)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.set_title(f"{STUDY_LABEL[study]}, {rows[0]['participants']} participants",
                     loc="left", fontsize=7.6, pad=3.0)
        ax.grid(axis="x", color="#EDEFF2", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        if ax is not axes[-1]:
            ax.tick_params(axis="x", labelbottom=False)

    axes[-1].set_xlabel("Phantom agreement (%)")
    axes[0].text(EFIGURE2_N_COLUMN, -0.60, "Selections", fontsize=7, color=MUTED,
                 ha="left", va="center", style="italic",
                 transform=axes[0].get_yaxis_transform(), clip_on=False)
    axes[0].text(100 * cohort["estimate"] + 0.25, -0.60,
                 f"whole cohort, {100 * cohort['estimate']:.1f}%", fontsize=6.6, color=MUTED,
                 ha="left", va="center", style="italic")

    reference = [8, 13, 18]
    legend_below(
        fig,
        [plt.scatter([], [], s=16 + 74 * value / largest_n_participants, color=PHANTOM_COLOR,
                     edgecolor="white", linewidths=0.7) for value in reference],
        [f"{value} participants" for value in reference],
        ncol=3, y=0.075, fontsize=7,
    )

    save(fig, str(FIGURES / "eFigure2"))
    plt.close(fig)


EFIGURE3_THRESHOLDS = (0.01, 0.05, 0.10, 0.50)
EFIGURE3_ZOOM = 0.10  # the inset's x range: the region holding most of the selections


def efigure3(digest, attribution):
    """Per-selection distribution of the prior's share of posterior displacement
    (1 - neural contribution fraction), raw and calibrated bases overlaid.

    The distribution is almost all at zero with a long thin tail, which a single ECDF drawn
    across the full range renders as a vertical wall at the origin. The inset magnifies that
    wall, the two-row strip beneath shows where the selections themselves lie on each basis
    (one tick per percentile, so tick density is data density), and the threshold block reads
    eight percentages -- four thresholds by two bases -- off the curves rather than leaving
    them to be eyeballed.

    This is the figure a reviewer asked to "transparently show what calibration actually
    changed": Task 23 through 26 already moved every other figure onto the calibrated basis
    outright, but this one keeps the raw curve alongside it rather than replacing it, because
    the comparison itself, not either curve alone, is what the reviewer's complaint was about.

    Colour grammar as in every other figure in this study: PRIOR_COLOR (vermilion) is raw,
    uncalibrated fusion, the same value the prior's share has carried since Figure 2, and KEY
    (blue) is the calibrated basis, the colour of the headline result throughout Figures 1-4
    and eFigure 1's calibrated row. Grey is reserved for structural chrome (the inset's frame,
    the zero-line grid), not for either basis's own curve or marks, which is a change from the
    single-curve version: with two bases to key apart, a grey median or mean mark would no
    longer say which curve it summarizes.
    """
    primary = attribution[
        (attribution["beta"] == PRIMARY_BETA) & (attribution["prior_model"] == PRIMARY_PRIOR)
    ].copy()
    primary["prior_share"] = 1.0 - primary["ncf"]
    share_raw, cumulative_raw, mean_raw, median_raw = prior_share_ecdf(primary["prior_share"])
    # The manuscript reports this quantity participant-weighted (7.5%, the raw co-primary's
    # complement) and this figure draws it unweighted across selections. The two differ by
    # less than half a percentage point, far less than the width of a drawn line here, so the
    # figure marks one mean per basis and states both numbers rather than implying a visible gap.
    weighted_raw = participant_mean(primary, "prior_share")
    stored_raw = digest["sensitivity"]["uncalibrated_beta_1"][
        "coprimary_1_neural_contribution_fraction"
    ]["prior_share"]
    if not np.isclose(weighted_raw, stored_raw, rtol=0, atol=1e-9):
        raise RuntimeError(
            f"eFigure 3 recomputed the raw, participant-weighted prior share as "
            f"{weighted_raw:.6g}, but output/stats_digest.json reports {stored_raw:.6g}. "
            "Re-run scripts/run_stats.py before re-rendering."
        )

    calibrated, _ = calibrated_primary_selections()
    share_cal, cumulative_cal, mean_cal, median_cal = prior_share_ecdf(calibrated["prior_share"])
    weighted_cal = participant_mean(calibrated, "prior_share")
    # calibrated_primary_analysis's ncf estimate is itself a participant-weighted mean
    # (run_stats.cluster_bootstrap over participant_mean), and prior_share = 1 - ncf is linear
    # in ncf, so the weighted mean of prior_share is exactly 1 minus the weighted mean of ncf;
    # no separate calibrated-basis digest key is needed to check this against.
    stored_cal = 1.0 - digest["calibrated_primary_analysis"]["ncf"]["estimate"]
    if not np.isclose(weighted_cal, stored_cal, rtol=0, atol=1e-9):
        raise RuntimeError(
            f"eFigure 3 recomputed the calibrated, participant-weighted prior share as "
            f"{weighted_cal:.6g}, but output/stats_digest.json's calibrated_primary_analysis "
            f"ncf estimate implies {stored_cal:.6g}. Re-run scripts/run_stats.py before "
            "re-rendering."
        )

    below_raw = {t: float(np.mean(share_raw < t)) for t in EFIGURE3_THRESHOLDS}
    below_cal = {t: float(np.mean(share_cal < t)) for t in EFIGURE3_THRESHOLDS}

    fig = plt.figure(figsize=(5.4, 4.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 0.075, 0.075], hspace=0.10,
                            left=0.120, right=0.985, top=0.975, bottom=0.275)
    ax = fig.add_subplot(grid[0])
    strip_raw = fig.add_subplot(grid[1], sharex=ax)
    strip_cal = fig.add_subplot(grid[2], sharex=ax)

    ax.step(share_raw, cumulative_raw, where="post", color=PRIOR_COLOR, lw=1.6, zorder=4)
    ax.step(share_cal, cumulative_cal, where="post", color=KEY, lw=1.6, zorder=4)
    ax.axvline(median_raw, color=PRIOR_COLOR, ls=":", lw=1.3, zorder=3)
    ax.axvline(mean_raw, color=PRIOR_COLOR, ls="--", lw=1.0, zorder=3, alpha=0.75)
    ax.axvline(median_cal, color=KEY, ls=":", lw=1.3, zorder=3)
    ax.axvline(mean_cal, color=KEY, ls="--", lw=1.0, zorder=3, alpha=0.75)
    ax.scatter(EFIGURE3_THRESHOLDS, [below_raw[t] for t in EFIGURE3_THRESHOLDS], s=22,
               facecolor="white", edgecolor=PRIOR_COLOR, linewidths=1.1, zorder=6)
    ax.scatter(EFIGURE3_THRESHOLDS, [below_cal[t] for t in EFIGURE3_THRESHOLDS], s=22,
               facecolor="white", edgecolor=KEY, linewidths=1.1, zorder=6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cumulative fraction of selections")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False)
    # The mean and median marks of a given basis share that basis's colour, so the legend keys
    # curve identity and both summary marks in one entry per basis rather than making a reader
    # cross-reference a colour-neutral line style against two different curves.
    ax.legend(
        handles=[
            Line2D([], [], color=PRIOR_COLOR, lw=1.6,
                   label=f"raw, uncalibrated fusion\nmean {mean_raw:.3f} "
                         f"(participant-weighted, {weighted_raw:.3f})"),
            Line2D([], [], color=KEY, lw=1.6,
                   label=f"calibrated fusion (primary)\nmean {mean_cal:.3f} "
                         f"(participant-weighted, {weighted_cal:.3f})"),
        ],
        loc="center left", bbox_to_anchor=(0.118, 0.355), fontsize=5.8, frameon=False,
        handlelength=1.1, labelspacing=0.9, handletextpad=0.5, borderaxespad=0.0,
    )

    # ------------------------------------------------------------------ threshold block
    # Both curves are already close to their ceiling by the x this block sits at, so an
    # opaque panel behind the text keeps the block readable where the single-curve version
    # had empty axis to work with.
    ax.add_patch(plt.Rectangle((0.478, 0.600), 0.400, 0.370, facecolor="white",
                               edgecolor="none", zorder=5))
    ax.text(0.500, 0.930, "Selections with a prior share below", fontsize=7,
            fontweight="bold", color=INK, ha="left", va="center", zorder=6)
    ax.text(0.700, 0.878, "raw", fontsize=6.4, color=PRIOR_COLOR, fontweight="bold",
            ha="center", va="center", zorder=6)
    ax.text(0.800, 0.878, "calib.", fontsize=6.4, color=KEY, fontweight="bold",
            ha="center", va="center", zorder=6)
    for index, threshold in enumerate(EFIGURE3_THRESHOLDS):
        y = 0.822 - 0.060 * index
        ax.text(0.628, y, f"{threshold:.2f}", fontsize=7, color=INK, ha="right", va="center",
                zorder=6)
        ax.text(0.700, y, f"{100 * below_raw[threshold]:.0f}%", fontsize=7, color=PRIOR_COLOR,
                ha="center", va="center", fontweight="bold", zorder=6)
        ax.text(0.800, y, f"{100 * below_cal[threshold]:.0f}%", fontsize=7, color=KEY,
                ha="center", va="center", fontweight="bold", zorder=6)

    # ------------------------------------------------------------------------- the inset
    # The region the whole distribution piles into, drawn at ten times the scale. The box on
    # the main axis is where it comes from; no leader lines, which at this scale would have to
    # start inside the curve. Both medians are annotated directly here, where they are far
    # enough apart to read (0.0006 and 0.029 are indistinguishable at the main axis's scale).
    inset = ax.inset_axes([0.535, 0.150, 0.435, 0.395])
    inset.step(share_raw, cumulative_raw, where="post", color=PRIOR_COLOR, lw=1.4, zorder=4)
    inset.step(share_cal, cumulative_cal, where="post", color=KEY, lw=1.4, zorder=4)
    inset.axvline(median_raw, color=PRIOR_COLOR, ls=":", lw=1.1, zorder=3)
    inset.axvline(mean_raw, color=PRIOR_COLOR, ls="--", lw=0.9, zorder=3, alpha=0.75)
    inset.axvline(median_cal, color=KEY, ls=":", lw=1.1, zorder=3)
    inset.axvline(mean_cal, color=KEY, ls="--", lw=0.9, zorder=3, alpha=0.75)
    zoomed = [t for t in EFIGURE3_THRESHOLDS if t <= EFIGURE3_ZOOM]
    inset.scatter(zoomed, [below_raw[t] for t in zoomed], s=18, facecolor="white",
                  edgecolor=PRIOR_COLOR, linewidths=1.0, zorder=6)
    inset.scatter(zoomed, [below_cal[t] for t in zoomed], s=18, facecolor="white",
                  edgecolor=KEY, linewidths=1.0, zorder=6)
    inset.text(median_raw, 0.885, f"{median_raw:.3f}", fontsize=5.6, color=PRIOR_COLOR,
              ha="left", va="top", rotation=90)
    inset.text(median_cal, 0.885, f"{median_cal:.3f}", fontsize=5.6, color=KEY,
              ha="left", va="top", rotation=90)
    inset.set_xlim(-0.002, EFIGURE3_ZOOM + 0.002)
    inset.set_ylim(0, 0.92)
    inset.set_xticks([0, 0.05, 0.10])
    inset.set_yticks([0, 0.4, 0.8])
    inset.tick_params(labelsize=6.2, length=2)
    inset.grid(axis="y", color="#EDEFF2", lw=0.5)
    inset.set_axisbelow(True)
    inset.set_title(f"magnified: 0 to {EFIGURE3_ZOOM:g}", fontsize=6.6, fontweight="normal",
                    color=MUTED, pad=2.5)
    ax.add_patch(plt.Rectangle((0, 0), EFIGURE3_ZOOM, 0.92, facecolor="none",
                               edgecolor=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=5))

    # -------------------------------------------------------------------- density strips
    # One row per basis, stacked directly on top of each other so a reader can compare tick
    # density at the same x straight down the page, the same purpose the single raw-only strip
    # served before there was a second basis to compare it against.
    percentiles_raw = np.percentile(share_raw, np.arange(0.5, 100.0, 1.0))
    percentiles_cal = np.percentile(share_cal, np.arange(0.5, 100.0, 1.0))
    for strip, percentiles, colour, tag in (
        (strip_raw, percentiles_raw, PRIOR_COLOR, "raw"),
        (strip_cal, percentiles_cal, KEY, "calib."),
    ):
        strip.vlines(percentiles, 0.08, 0.92, color=colour, lw=0.65, alpha=0.55)
        strip.set_ylim(0, 1)
        strip.set_yticks([])
        strip.set_xlim(-0.02, 1.02)
        strip.text(-0.010, 0.5, tag, fontsize=5.8, color=colour, ha="right", va="center",
                   fontweight="bold", transform=strip.get_yaxis_transform())
        for side in ("left", "top", "right"):
            strip.spines[side].set_visible(False)
    strip_raw.tick_params(axis="x", labelbottom=False)
    strip_cal.set_xlabel("KL-based prior displacement share (1 - NCF)")
    # Keyed on the main axis, not the strips: the gap between them is a twentieth of an inch
    # and text placed there lands on the axis line above or the ticks below.
    ax.text(1.02, 0.018, "one tick per percentile of selections, per basis", fontsize=6.2,
            color=MUTED, ha="right", va="bottom")

    save(fig, str(FIGURES / "eFigure3"))
    plt.close(fig)


def efigure4(digest, attribution):
    """Phantom agreement against parameter count on the raw, uncalibrated ladder.

    This was Figure 3 panel a before Task 23 moved that panel onto each prior's own
    next-character NLL (eTable 25), which parameter count does not predict on either basis
    (eTable 9 raw, eTable 23 calibrated). Task 23 preserved the plotting logic, unchanged, in
    _parameter_count_phantom_scatter for a later supplementary figure to place; eFigure 1 (a
    2x2 grid) and eFigure 2 (a variable-height per-study facet plot) were both already at
    capacity by the time this task landed, and eFigure 3's own redesign above turned it into a
    dense raw-vs-calibrated overlay with no room for a third, structurally unrelated scatter,
    so this scatter gets its own supplementary figure instead of a fifth or sixth panel
    somewhere it does not fit.

    `attribution` is accepted but not read, matching efigure2's convention: the shared
    `_rendered(draw)` test helper and main()'s call site both pass every supplementary figure
    `(digest, attribution)`, so this parameter stays even though _parameter_count_phantom_scatter
    reads only the raw ladder rows and the digest.

    Reuses Figure 3's own canvas size and panel a rect (FIGURE3_SIZE, FIGURE3_PANELS["a"])
    rather than a new geometry: _parameter_count_phantom_scatter's marker separation and
    primary-prior leader line are both measured in typographic points through
    _points_per_unit("a", ...), which reads those two constants directly rather than the
    actual axes passed to it, so only this exact canvas and rect reproduce the spacing Task 23
    tuned. save()'s tight bounding box then crops away the unused canvas where Figure 3 panel
    b used to sit, so nothing is wasted on the rendered page.
    """
    fig = plt.figure(figsize=FIGURE3_SIZE)
    ax = fig.add_axes(FIGURE3_PANELS["a"])
    _parameter_count_phantom_scatter(ax, _raw_ladder_rows(digest), digest)

    save(fig, str(FIGURES / "eFigure4"))
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_style()
    digest, attribution = load()
    figure1_example = figure1(digest)
    figure2(digest, attribution)
    figure3(digest)
    example = figure4(digest)
    efigure1(digest, attribution)
    efigure2(digest, attribution)
    efigure3(digest, attribution)
    efigure4(digest, attribution)
    # figure_example.json stays Figure 4's own record (docs/manuscript_facts.md reads it): the
    # phantom-agreement cohort summary and the three profiles panel d draws. Figure 1's own
    # median-attribution example is printed for caption checking only.
    (OUTPUT / "figure_example.json").write_text(json.dumps(example, indent=2))
    print(json.dumps({"figure1": figure1_example, "figure4": example}, indent=2))
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
