"""Build manuscript/cover_letter.docx and .pdf with single line-spacing.

The shared `_pub_assets/build_pub.sh` pipeline (used to build manuscript.md and supplement.md too)
bakes in double line-spacing for every file it touches: `-V linestretch=2` for the PDF, and
`reference_cmu.docx`'s docDefaults (`w:line="480"`) for the docx. That is correct for the
manuscript body but wrong for a cover letter. House convention -- confirmed against the
`study_bigp3_als_calibration` companion's own cover-letter rebuild (2026-07-29, see that study's
`_submission_ready/0_README.md`: "rendered single-spaced (the house double-spacing convention
applies to the manuscript body, not correspondence) to fit one page") -- is that cover letters
render single-spaced while the manuscript body renders double-spaced.

Running `build_pub.sh` unmodified on this project's `manuscript/` directory will silently
regress `cover_letter.pdf`/`.docx` back to double-spaced and multi-page, with no error. Use this
script instead for `cover_letter.md` specifically: it reuses the same CMU-Serif/xelatex pipeline
(same fonts, margins, lua filter) as `build_pub.sh`, but overrides line spacing to single for this
one file, and does not modify `_pub_assets/build_pub.sh` or `reference_cmu.docx` themselves --
both are shared across every paper in the lab and out of scope to change here.

Usage: python3 scripts/build_cover_letter.py
Requires: pandoc with xelatex, CMU Serif installed, and the `python-docx` package.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import docx

PROJECT = Path(__file__).resolve().parents[1]
MANUSCRIPT = PROJECT / "manuscript"
ASSETS = Path("/Volumes/Extreme SSD/Mimic-IV/_pub_assets")
PREPPED = Path("/tmp/cover_letter.pub.md")


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    md = MANUSCRIPT / "cover_letter.md"
    if not md.exists():
        sys.exit(f"missing {md}")
    if not ASSETS.exists():
        sys.exit(f"missing shared pipeline assets at {ASSETS}")

    prep = subprocess.run(
        ["python3", str(ASSETS / "prep.py"), str(md)],
        check=True, capture_output=True, text=True,
    )
    PREPPED.write_text(prep.stdout, encoding="utf-8")

    docx_path = MANUSCRIPT / "cover_letter.docx"
    pdf_path = MANUSCRIPT / "cover_letter.pdf"

    run([
        "pandoc", str(PREPPED), "-f", "gfm+superscript+attributes",
        "--syntax-highlighting=none", "--lua-filter", str(ASSETS / "colwidths.lua"),
        "-o", str(docx_path), "--reference-doc", str(ASSETS / "reference_cmu.docx"),
    ])

    run([
        "pandoc", str(PREPPED), "-f", "gfm+superscript+attributes",
        "--syntax-highlighting=none", "--lua-filter", str(ASSETS / "colwidths.lua"),
        "--pdf-engine=xelatex", "-V", "mainfont=CMU Serif", "-V", "monofont=CMU Serif",
        "-V", "geometry:margin=1in", "-V", "fontsize=11pt", "-V", "linestretch=1",
        "--include-in-header", str(ASSETS / "header.tex"),
        "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=black",
        "-V", "citecolor=black", "-V", "toccolor=black",
        "-o", str(pdf_path),
    ])

    # --reference-doc inherits reference_cmu.docx's double-spaced docDefaults (w:line="480").
    # Override every style's and every paragraph's line spacing to single (1.0) so the docx
    # matches the single-spaced PDF built above.
    d = docx.Document(str(docx_path))
    for style in d.styles:
        pf = getattr(style, "paragraph_format", None)
        if pf is not None:
            pf.line_spacing = 1.0
    for para in d.paragraphs:
        para.paragraph_format.line_spacing = 1.0
    d.save(str(docx_path))

    print(f"[ok] {docx_path}")
    print(f"[ok] {pdf_path}")


if __name__ == "__main__":
    main()
