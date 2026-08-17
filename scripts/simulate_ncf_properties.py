"""Six synthetic scenarios probing what the neural contribution fraction does and does not measure.

Each scenario is a hand-constructed pair of 36-way distributions (compressed here to a small
number of symbols for readability; the remaining mass is spread uniformly over the rest of the
alphabet) chosen to separate distributional displacement from any claim about "authorship."

The NCF formula reimplemented in ``_ncf`` below matches the definition in ``authorship.attribution``
and manuscript ``supplement.md`` S1.4: NCF = D(f||p) / [D(f||p) + D(f||n)], where f is the fused
posterior, p is the prior, n is the neural posterior, and D is Kullback-Leibler divergence in bits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.attribution import N_SYMBOLS, PROBABILITY_FLOOR, fuse  # noqa: E402

N = N_SYMBOLS
OUTPUT_PATH = PROJECT / "output" / "ncf_simulation.json"


def _dist(peaks: dict[int, float]) -> np.ndarray:
    """Build a 36-way distribution with the given peaks and uniform mass elsewhere.

    Entries are floored at a small positive value and the array renormalized so that no
    symbol carries literally zero probability. Without this, peaks summing to exactly 1.0
    (as in the near-tie scenario's 0.51/0.49 split) leave the remaining 34 symbols at exactly
    0.0, which makes the KL divergence against that symbol undefined rather than merely large.
    """
    array = np.full(N, 0.0)
    remaining = 1.0 - sum(peaks.values())
    fill = remaining / (N - len(peaks))
    array[:] = fill
    for index, mass in peaks.items():
        array[index] = mass
    array = np.maximum(array, PROBABILITY_FLOOR)
    return array / array.sum()


def _ncf(neural: np.ndarray, prior: np.ndarray) -> float:
    fused = fuse(neural, prior)
    neural_displacement = float(np.sum(fused * (np.log2(fused) - np.log2(prior))))
    prior_displacement = float(np.sum(fused * (np.log2(fused) - np.log2(neural))))
    total = neural_displacement + prior_displacement
    return neural_displacement / total if total > 1e-12 else float("nan")


def _scenario(
    scenario: str, neural: np.ndarray, prior: np.ndarray, description: str,
) -> dict[str, object]:
    fused = fuse(neural, prior)
    return {
        "scenario": scenario,
        "p_neural": neural.tolist(),
        "p_lm": prior.tolist(),
        "ncf": _ncf(neural, prior),
        "fused_argmax_matches_neural": bool(int(fused.argmax()) == int(neural.argmax())),
        "fused_argmax_matches_prior": bool(int(fused.argmax()) == int(prior.argmax())),
        "description": description,
    }


def build_scenarios() -> list[dict[str, object]]:
    scenarios = []

    scenarios.append(_scenario(
        "agreement", _dist({0: 0.90}), _dist({0: 0.90}),
        "Both sources sharply agree on the same symbol.",
    ))

    scenarios.append(_scenario(
        "conflict", _dist({0: 0.90}), _dist({1: 0.90}),
        "Both sources are confident but disagree.",
    ))

    scenarios.append(_scenario(
        "near_tie", _dist({0: 0.51, 1: 0.49}), _dist({1: 0.90}),
        "Neural evidence is a near-coin-flip the prior tips toward its own choice.",
    ))

    scenarios.append(_scenario(
        "sharply_peaked", _dist({0: 0.99}), _dist({1: 0.34}),
        "Neural evidence is extremely confident; the prior is comparatively diffuse.",
    ))

    scenarios.append(_scenario(
        "diffuse", _dist({}), _dist({0: 0.90}),
        "Neural evidence is uninformative (uniform); the prior alone determines the fused posterior.",
    ))

    scenarios.append(_scenario(
        "miscalibrated", _dist({0: 0.60, 1: 0.05, 2: 0.05}), _dist({0: 0.15, 1: 0.15, 2: 0.15}),
        "Neural evidence is overconfident relative to its true discriminability (a calibration "
        "failure), which the KL-based measure cannot detect from the distributions alone.",
    ))

    return scenarios


def main() -> None:
    scenarios = build_scenarios()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(scenarios, indent=2))
    print(json.dumps(scenarios, indent=2))


if __name__ == "__main__":
    main()
