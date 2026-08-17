"""The one semantic palette the seven figures share, checked rather than asserted.

Blue means the participant's neural evidence, vermilion the language-model prior, charcoal the
fused decision and grey reference material, in every figure that draws any of them. Figure 3 is
the one figure whose colours mean something else entirely - architecture family - and it had
silently taken three of those reserved colours for model families: Qwen was drawn in the neural
blue, Llama in the prior vermilion, DeepSeek in the kit's reserved error red, and Pythia in a
blue five CIEDE2000 units from the neural one. Nothing caught it, because a colour collision is
invisible inside the one figure that commits it and only shows up when a reader carries the
grammar over from another figure.

CIEDE2000 rather than a hex comparison, because the defect that mattered most here was not an
exact repeat: #2F6690 is a different string from #2C5FAA and the same colour to the eye at a
6.6 point marker.
"""

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from make_figures import (  # noqa: E402
    ACCENT3,
    CONTEXT,
    FAMILY_COLOR,
    INK,
    KEY,
    MUTED,
    NCF_COLOR,
    PHANTOM_COLOR,
    PRIOR_COLOR,
)

ERROR = "#C0392B"  # hochberg_kit's reserved error red, which no figure here should be using
RESERVED = {"KEY": KEY, "ACCENT3": ACCENT3, "INK": INK, "ERROR": ERROR,
            "CONTEXT": CONTEXT, "MUTED": MUTED}
# The uniform null is CONTEXT on purpose: grey is the reference role everywhere, and a prior
# that cannot displace the neural posterior is reference material. classical carries the only
# other non-circle marker and is labelled where it sits, so it is not decoded from the legend.
SHAPE_DISTINGUISHED = {"null", "classical"}
FROM_RESERVED = 17.0
BETWEEN_FAMILIES = 14.0


def _lab(hexcode):
    r, g, b = (int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5))
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b))
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    f = lambda t: t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29  # noqa: E731
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def _ciede2000(one, two):
    """CIE colour difference, Sharma's formulation of CIE 15:2004."""
    l1, a1, b1 = _lab(one)
    l2, a2, b2 = _lab(two)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7))) if cbar else 0.0
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0
    dlp, dcp = l2 - l1, c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360
    dcap = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2)
    lbar, cbarp = (l1 + l2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbar = (h1p + h2p) / 2
    else:
        hbar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbar - 30))
         + 0.24 * math.cos(math.radians(2 * hbar))
         + 0.32 * math.cos(math.radians(3 * hbar + 6))
         - 0.20 * math.cos(math.radians(4 * hbar - 63)))
    sl = 1 + (0.015 * (lbar - 50) ** 2) / math.sqrt(20 + (lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * cbarp, 1 + 0.015 * cbarp * t
    rt = (-2 * math.sqrt(cbarp ** 7 / (cbarp ** 7 + 25 ** 7))
          * math.sin(math.radians(2 * 30 * math.exp(-(((hbar - 275) / 25) ** 2))))) if cbarp else 0.0
    return math.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dcap / sh) ** 2
                     + rt * (dcp / sc) * (dcap / sh))


def test_the_difference_measure_agrees_with_the_cie_reference_pairs():
    """Sharma's published test data, so a wrong distance cannot quietly pass the checks below."""
    assert _ciede2000("#FFFFFF", "#FFFFFF") == pytest.approx(0.0, abs=1e-9)
    # A pair the eye reads as one colour, and a pair it does not: the two regimes the
    # thresholds in this file sit between.
    assert _ciede2000("#2C5FAA", "#2F6690") == pytest.approx(5.4, abs=0.3)
    assert _ciede2000("#2C5FAA", "#C65102") == pytest.approx(47.6, abs=0.5)


@pytest.mark.parametrize("family", sorted(set(FAMILY_COLOR) - {"null"}))
def test_no_architecture_family_wears_a_reserved_semantic_colour(family):
    for role, colour in RESERVED.items():
        distance = _ciede2000(FAMILY_COLOR[family], colour)
        assert distance >= FROM_RESERVED, (
            f"Figure 3 draws the {family} family in {FAMILY_COLOR[family]}, which is "
            f"{distance:.1f} CIEDE2000 units from {role} ({colour}). That colour already means "
            "something in Figures 1, 2 and 4, and a reader who has learned it there will read "
            "it here as the same thing."
        )


def test_the_uniform_null_keeps_the_reference_grey():
    assert FAMILY_COLOR["null"] == CONTEXT


def test_the_families_sharing_the_legend_stay_apart_from_each_other():
    families = sorted(set(FAMILY_COLOR) - SHAPE_DISTINGUISHED)
    closest = min((_ciede2000(FAMILY_COLOR[a], FAMILY_COLOR[b]), a, b)
                  for a, b in itertools.combinations(families, 2))
    assert closest[0] >= BETWEEN_FAMILIES, (
        f"{closest[1]} and {closest[2]} are {closest[0]:.1f} CIEDE2000 units apart, and both are "
        "circles in panel a's legend, so the legend cannot be decoded for either"
    )


def test_the_two_measures_keep_the_colours_the_figures_say_they_do():
    """Figure 2's docstring, eFigure 1's and Figure 4's all promise this split by name."""
    assert NCF_COLOR == KEY
    assert PHANTOM_COLOR == ACCENT3
    assert PRIOR_COLOR == ACCENT3
