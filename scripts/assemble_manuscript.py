"""Assemble the manuscript from its section files and report word counts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT / "manuscript"
sys.path.insert(0, str(PROJECT))

SECTIONS = ["abstract", "introduction", "results", "discussion", "methods"]
BODY_SECTIONS = ["introduction", "results", "discussion", "methods"]

DIGEST = PROJECT / "output" / "stats_digest.json"
LADDER_ROWS_MARKER = "<!-- prior ladder rows, generated from output/stats_digest.json -->"

# Display label for each prior in output/stats_digest.json's sensitivity.prior_ladder, in the
# order Table 2 presents them. Table 2's numbers are generated from the digest rather than
# written out, so the table cannot drift from the analysis it reports.
LADDER_DISPLAY = [
    ("uniform", "Uniform"),
    ("ngram5", "Character 5-gram"),
    ("ngram5_kn", "5-gram, Kneser-Ney"),
    ("ngram5_wiki_kn", "5-gram, Kneser-Ney (WikiText-103)"),
    ("gpt2", "GPT-2"),
    ("gpt2-large", "GPT-2 large"),
    ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
    ("Qwen/Qwen2.5-14B", "Qwen2.5-14B"),
    ("Qwen/Qwen3.5-27B", "Qwen3.5-27B (Instruct)"),
    ("Qwen/Qwen2.5-32B", "Qwen2.5-32B"),
    ("Qwen/Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B (Instruct)"),
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B (Instruct)"),
    ("google/gemma-2-2b-it", "Gemma-2-2B (Instruct)"),
    ("google/gemma-2-9b", "Gemma-2-9B"),
    ("google/gemma-2-27b", "Gemma-2-27B"),
    ("google/gemma-4-12b-it", "Gemma-4-12B (Instruct)"),
    ("mistralai/Mistral-7B-v0.3", "Mistral-7B"),
    ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B"),
    ("deepseek-ai/DeepSeek-V2-Lite", "DeepSeek-V2-Lite"),
    ("allenai/OLMo-2-1124-7B-Instruct", "OLMo-2-7B (Instruct)"),
    ("openai/gpt-oss-20b", "GPT-OSS-20B (Instruct)"),
    ("EleutherAI/pythia-12b", "Pythia-12B"),
    ("state-spaces/mamba-2.8b-hf", "Mamba-2.8B"),
    ("google/recurrentgemma-2b", "RecurrentGemma-2B"),
]

TITLE_PAGE = """# Attributing Large Language Model Influence in Brain Computer Interface Communication for Amyotrophic Lateral Sclerosis

**Authors:** Alon Gorenshtein, MD^1,2^; Yosef Adiniaev^2^; Tom Liba, MD^3^; Eyal Klang, MD^2,4^; Oved Daniel, MD^5^

**Affiliations:** ^1^ Department of Neurology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA. ^2^ BRIDGE GenAI Lab, Beth Israel Deaconess Medical Center, Boston, MA, USA. ^3^ Azrieli Faculty of Medicine, Bar-Ilan University, Safed, Israel. ^4^ Department of Radiology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA. ^5^ Neurology Division, Tel Aviv Sourasky University Medical Center, Tel Aviv, Israel.

**Corresponding author:** Alon Gorenshtein, MD, Department of Neurology, Beth Israel Deaconess Medical Center, Harvard Medical School, Boston, MA, USA (agorensh@bidmc.harvard.edu)

**Word counts:** {counts}

---
"""

END_MATTER = """
**Data availability:** BigP3BCI version 1.0.0 is publicly available from PhysioNet
(doi:10.13026/0byy-ry86).

**Code availability:** Analysis code will be made publicly available at
https://github.com/Alon-Gorenshtein/study_bci_llm_authorship.

## Acknowledgments

This work received no external funding.

**Author contributions:** Contributions are described using the CRediT taxonomy. A.G.: conceptualization, methodology, software, formal analysis, data curation, validation, visualization, and writing of the original draft. Y.A.: software, data curation, formal analysis, validation, and review and editing of the manuscript. T.L.: investigation, validation, and review and editing of the manuscript. E.K.: conceptualization, methodology, supervision, resources, and review and editing of the manuscript. O.D.: conceptualization, methodology, investigation, formal analysis, clinical interpretation, supervision, writing of the original draft, and review and editing of the manuscript. All authors critically reviewed the manuscript and approved the final version submitted for publication. A.G. (corresponding author) had full access to all data in the study and takes responsibility for the integrity of the data and the accuracy of the analysis.

**Competing interests:** The authors declare that they have no competing interests.
"""

TABLES = """
## Table 1. Cohort characteristics

| Characteristic | Value |
|---|---|
| Participants, No. | 47 |
| Recording sessions, No. | 115 |
| Online selections analysed, No. | 3,373 |
| Source studies, No. | 4 |
| Selections per participant, median (range) | 84 (4 to 154) |
| Age, y, median (IQR), among 21 with age recorded | 56 (50 to 59) |
| Male sex, No., among 25 with sex recorded | 16 |
| ALSFRS-R, median (range), among 29 with score recorded | 20 (0 to 46) |
| Distinct intended words, No. | 289 |
| Digit-string targets, No. of words | 28 |
| Flashes per selection, median (IQR) | 126 (61 to 127) |
| Flash epochs excluded for artefact, % | 3.4 |
| Calibration decoder AUC, median (IQR) | 0.795 (0.686 to 0.861) |
| Online accuracy recorded in archive, % | 81.9 |
| Offline neural-only accuracy, % | 71.8 |
| Agreement of offline posterior with archive selection, % | 69.6 |

Abbreviations: ALSFRS-R, Amyotrophic Lateral Sclerosis Functional Rating Scale-Revised; AUC, area
under the receiver operating characteristic curve; IQR, interquartile range. Offline neural-only
accuracy is a selection-weighted mean (each of the 3,373 selections counted once); Table 2 reports
accuracy as a participant-weighted mean (each participant weighted equally regardless of selection
count), which is why the uniform-prior row of eTable 22 shows 71.6% rather than 71.8%.

## Table 2. Attribution measures across the language-model prior ladder

Table 2 reports the calibrated ladder, held-out per-source temperature calibration applied to
every prior, as the primary basis; eTable 22 reports the same ladder on raw, uncalibrated fusion
as the disclosed sensitivity comparison. Table 2 omits the uniform prior, for which a calibrated
basis is not defined, because temperature scaling has nothing to correct on a prior that carries
no information; the uniform-prior row appears only in eTable 22. Prior capture is calculated
separately for each prior using the same definition as the secondary outcome; the Qwen2.5-32B row
corresponds to the primary-prior estimate reported in Results.

| Prior | Parameters | Neural contribution fraction (95% CI) | Phantom agreement, % (95% CI) | Prior capture, % (95% CI) | Accuracy, % (95% CI) |
|---|---|---|---|---|---|
<!-- prior ladder rows, generated from output/stats_digest.json -->

Qwen2.5-32B became the primary prior after an initial six-tier ladder (ending at Qwen2.5-3B) showed
a parameter-count trend on 2026-07-26; the ladder was extended the same day specifically to test
whether that trend would hold, and Qwen2.5-32B, the largest model in the resulting nine-entry
ladder, was designated primary at that point, not in the original six-tier design, and remained
primary through every later extension without a separate, independently justified criterion for
choosing it specifically over any other ladder point (Supplementary Note S1.17).
Qwen3.5-27B, Qwen3.6-35B-A3B, Llama-3.1-8B, Gemma-2-2B, Gemma-4-12B, OLMo-2-7B, and
GPT-OSS-20B are instruction-tuned; the remaining 14 neural entries are base models. Llama-3.1-8B was
substituted for a base Llama checkpoint because live Hugging Face Hub access to Llama-family
repositories was not available at scoring time. Across the 21
neural language models spanning eleven architecture families (GPT-2, Qwen, Llama, Gemma, Mistral, DeepSeek,
OLMo, GPT-OSS, Pythia, Mamba, RecurrentGemma), no attribution measure varied reliably with parameter
count on either basis (eTable 9 raw; eTable 23 calibrated). An architecture search additionally
screened RWKV-7, xLSTM, StripedHyena, RetNet,
and Gated DeltaNet; none passed a documented O2 loadability check performed before scoring
(eTable 10). The full per-model
inventory for this table, including exact parameter counts, base-versus-instruction-tuned status,
and source organization, is reported in eTable 19.

## Figure Legends

**Figure 1. From flashed grid to emitted character: how each selection was attributed between the
participant's neural evidence and the language-model prior.** (a) The reconstruction pipeline.
Flashes of the 36-symbol grid evoke responses that a calibration-trained decoder accumulates into a
posterior over the alphabet; the language model, conditioned on the intended preceding characters,
supplies a prior over the next symbol; each source is then rescaled by a temperature
fitted on held-out participant folds, and the two are multiplied and renormalized, the fused
posterior's highest-probability symbol being the emitted character. The grid is drawn with one
flash shown as an illuminated column, although checkerboard conditions instead illuminate arbitrary
symbol subsets, and the evoked-response trace is schematic rather than averaged data. (b) One
selection, chosen by a rule fixed in advance as the selection whose prior share of posterior
displacement lies closest to the cohort median among those with at least four preceding characters:
the neural posterior, the prior, and the fused posterior across the eight highest-ranked candidate
symbols, all under the same held-out calibrated fusion. The neural posterior's own highest
probability fell on B; the prior, close to uniform after calibration, nonetheless placed more
probability on the intended G than on B, and that difference, annotated on the middle panel because
it is too small to read off bars drawn on the shared scale, was enough to raise G above B in the
fused posterior. (c) The two per-selection measures, with this selection's values. (d) Headline
co-primary estimates, each drawn as a point estimate with a bar spanning its 95% confidence
interval on a shared percentage axis: the calibrated neural contribution fraction's complement (the
prior's mean share of posterior displacement) and phantom agreement, both under held-out calibrated
fusion with oracle context. The gray tick on the upper row marks the median prior share across
selections. Intervals are from 2,000 participant-cluster bootstrap replicates.

**Figure 2. Attribution shifts toward the language model as context accumulates and as calibration
decoder discriminability decreases.** Both measures keep one color throughout: vermilion is phantom agreement, blue is
the neural contribution fraction, and gray marks reference lines. The two panels share a y-axis
range within each row. (a) The two measures by character position within the intended word. The
tile strip above the panels shows what the prior had to condition on at each position, with the
contributing selection count given beneath it; later positions occur only in longer words and so
rest on fewer selections. The dashed gray line is the overall phantom agreement rate. (b) The same
two measures against the calibration decoder AUC of the session, treated as the continuous
quantity it is rather than as three bins; the fitted curve is the participant-clustered model
reported in the Results (logistic for phantom agreement, a bounded fractional-logit fit for the
neural contribution fraction that cannot predict outside [0, 1]) and the band is that model's own
95% interval for the fitted mean. The small unlabeled ticks mark each tertile's median AUC and
point estimate, the value the Results quote, and the dotted verticals are the tertile boundaries.
Error bars on the estimates are 95% confidence intervals from 2,000 participant-cluster bootstrap
replicates. All estimates, curves, points, and annotations in both panels are on the
calibrated-fusion basis of Figure 1 and the Results; raw, uncalibrated counterparts are the
sensitivity comparisons in eTable 2 (position), eTable 3 (decoder quality), and eTable 5
(hypothesis tests). The neural contribution fraction's annotation in each panel is the bounded
fit's own predicted change, across the plotted character positions in (a) and between the lower and
upper tertiles' median AUC in (b), not a per-unit slope from the separate linear mixed model
reported as a sensitivity comparison in the Results and eTable 30. Annotated P values are each model's own, unadjusted for multiplicity;
Benjamini-Hochberg-adjusted values for the five-member secondary family are in the Results and
eTable 30.

**Figure 3. Next-character predictive quality is associated with phantom agreement, whereas no
monotonic association with parameter count was detected, and character 5-grams alone occupy the
frontier of accuracy gained against correct neural decisions overturned.** One point per prior in
both panels. The ringed point is the primary prior, Qwen2.5-32B. Both panels use the calibrated,
held-out fusion basis of Figure 1 and Table 2; the raw basis is eTable 22's sensitivity comparison.
(a) The densest cluster sits in a dashed box, magnified in the inset for individual reading. Across
the 21 neural language models, phantom agreement correlates with next-character NLL at rho = -0.64
(P = .002; eTable 25), a descriptive check, not the prespecified parameter-count equivalence test
(eTable 18, eTable 23). Restricting the correlation to that class keeps predictive quality from
being conflated with model class: the 5-grams predict characters natively, the unit the speller
selects, whereas the neural language models are general-purpose checkpoints scored without
task-specific adaptation. Across all 24 priors the correlation is stronger (rho = -0.76, P < .001),
reported here as a descriptive comparison rather than in-panel because it spans both classes.
Labeled: the primary prior, the best-NLL neural language model and 5-gram, and the largest and
smallest neural language models by parameter count. The primary is labeled on the panel itself even
inside the dashed box; the rest, in the inset when inside it. (b) What each prior buys against what
it costs, defined on the panel's own axes. Prior capture here is per-prior, from the calibrated
ladder (Table 2); the ringed point's prior capture is the primary-prior estimate reported in
Results. The dashed step marks the benefit-harm frontier, the two points no prior beats on both
axes: both are character 5-grams; the best of them, ngram5, exceeded the best neural model,
DeepSeek-V2-Lite, by 6.4 percentage points on calibrated fused accuracy, a descriptive, not
prespecified, comparison. Parameter counts for panel a's largest/smallest labels are totals,
including inactive experts of the four sparsely activated models (eTable 19).

**Figure 4. Neural support among prior-mediated reclassifications: how much evidence the intended
character had in the 166 selections the calibrated fusion rule corrected.**
Every selection in which the fused posterior emitted the intended
character and the neural posterior alone would not have; blue is neural evidence, vermilion the
language-model prior, and charcoal the fused decision. (a) The neural probability on the intended
character, one point per selection. (b) Where the intended character stood in the neural posterior's
own ranking; the 4 cases tied with its top symbol are shown separately, because a tie is resolved by
a deterministic tie-break that can select against the intended character, and all four categories are
shares of the full cohort (eTable 26). (c) The cumulative distribution of the neural margin. (d)
Three of these selections in full, the ones nearest the 10th, 50th, and 90th percentiles of that
margin among selections with at least four preceding characters, chosen by a rule fixed in advance.
Numbered badges tie each card to its position in (a) to (c), and each source's own top-ranked
symbol is labeled above its column. The first, typing "OFFI" toward the intended "C" (OFFICE),
is a near-tie the prior settled at a neural margin of 0.01 (8th percentile); the intended character
was already the neural posterior's own runner-up. The second, typing "REMA" toward the intended
"I" (REMAIN), is the median case at a
neural margin of 0.11 (56th percentile), the intended character again its own runner-up. The third,
at the 93rd percentile of the cohort's margins, is one in which the neural posterior favored a different
symbol by a wide margin, and so is atypical of the rest of the cohort. Probabilities in (d) are on a
logarithmic scale and values below 10^-4^ are drawn at that floor. All panels are computed on the
calibrated, held-out per-source fusion basis of Figure 1's co-primary estimates and of Table 2, not
the raw, uncalibrated basis eTable 11 reports as a disclosed sensitivity comparison.

![](../output/figures/Figure1.png){width=92%}

![](../output/figures/Figure2.png){width=92%}

![](../output/figures/Figure3.png){width=92%}

![](../output/figures/Figure4.png){width=92%}
"""


def parameter_label(count: int) -> str:
    if not count:
        return "None"
    if count < 1_000_000_000:
        return f"{count / 1e6:.0f}M"
    billions = count / 1e9
    return f"{billions:.1f}B" if round(billions, 1) == round(billions, 2) else f"{billions:.2f}B"


def ladder_rows(ladder: dict) -> str:
    missing = set(ladder) - {key for key, _ in LADDER_DISPLAY}
    if missing:
        raise RuntimeError(
            f"output/stats_digest.json carries priors Table 2 has no row for: {sorted(missing)}. "
            "Add them to LADDER_DISPLAY rather than letting the table under-report the ladder."
        )
    rows = []
    for key, label in LADDER_DISPLAY:
        entry = ladder[key]
        accuracy = entry["fused_accuracy_ci"]
        cells = [
            label,
            parameter_label(entry["prior_parameters"]),
            f"{entry['ncf']['estimate']:.3f}",
            f"{100 * entry['phantom_agreement']['estimate']:.1f}",
            f"{100 * entry['prior_capture']['estimate']:.2f}",
            f"{100 * accuracy['estimate']:.1f} "
            f"({100 * accuracy['ci_low']:.1f} to {100 * accuracy['ci_high']:.1f})",
        ]
        if key != "uniform":
            # The uniform null's first three measures are fixed by construction, so the table
            # note explains why they alone carry no interval.
            for column, (measure, scale, places) in enumerate(
                    (("ncf", 1, 3), ("phantom_agreement", 100, 1), ("prior_capture", 100, 2)),
                    start=2):
                interval = entry[measure]
                cells[column] += (f" ({scale * interval['ci_low']:.{places}f} to "
                                  f"{scale * interval['ci_high']:.{places}f})")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def ladder_rows_calibrated(ladder: dict) -> str:
    missing = set(ladder) - {key for key, _ in LADDER_DISPLAY}
    if missing:
        raise RuntimeError(
            f"output/stats_digest.json carries calibrated priors Table 2 has no row for: "
            f"{sorted(missing)}. Add them to LADDER_DISPLAY rather than letting the table "
            "under-report the ladder."
        )
    rows = []
    for key, label in LADDER_DISPLAY:
        if key not in ladder:
            # The calibrated ladder omits the uniform prior: a calibrated basis is not
            # defined for it, since temperature scaling has nothing to correct on a prior
            # that carries no information.
            continue
        entry = ladder[key]
        accuracy = entry["fused_accuracy_ci"]
        ncf = entry["ncf"]
        phantom = entry["phantom_agreement"]
        capture = entry["prior_capture"]
        cells = [
            label,
            parameter_label(entry["prior_parameters"]),
            f"{ncf['estimate']:.3f} ({ncf['ci_low']:.3f} to {ncf['ci_high']:.3f})",
            f"{100 * phantom['estimate']:.1f} "
            f"({100 * phantom['ci_low']:.1f} to {100 * phantom['ci_high']:.1f})",
            f"{100 * capture['estimate']:.2f} "
            f"({100 * capture['ci_low']:.2f} to {100 * capture['ci_high']:.2f})",
            f"{100 * accuracy['estimate']:.1f} "
            f"({100 * accuracy['ci_low']:.1f} to {100 * accuracy['ci_high']:.1f})",
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def word_count(text: str) -> int:
    text = re.sub(r"^#+ .*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\|.*\|", "", text)
    text = re.sub(r"\[REF-[A-Z0-9-]+\]", "", text)
    return len(text.split())


def main() -> None:
    parts = {name: (MANUSCRIPT / f"{name}.md").read_text() for name in SECTIONS}
    abstract_text = parts["abstract"].split("## Abstract", 1)[1]
    counts = {
        "body": sum(word_count(parts[name]) for name in BODY_SECTIONS),
        "abstract": word_count(abstract_text),
    }
    header = TITLE_PAGE.format(
        counts=f"body {counts['body']:,}; "
        f"abstract {counts['abstract']} / 150"
    )
    references = (MANUSCRIPT / "references.md").read_text()
    digest = json.loads(DIGEST.read_text())
    tables = TABLES.replace(LADDER_ROWS_MARKER,
                            ladder_rows_calibrated(digest["sensitivity"]["prior_ladder_calibrated"]))
    document = (header + "\n" + "\n\n".join(parts[name] for name in SECTIONS)
                + "\n" + END_MATTER + "\n" + references + "\n" + tables)
    (MANUSCRIPT / "manuscript.md").write_text(document)
    (PROJECT / "output" / "word_counts.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
