"""Quantify the correction from the legacy retokenization-vulnerable scorer to the
prefix-locked scorer (Tasks 1-2): magnitude of per-context score change and how often
the top-ranked candidate flips. Required by the round-2 review's Item 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from authorship.grid import SYMBOLS  # noqa: E402

OUTPUT = PROJECT / "output" / "scorer_correction_impact.json"
STABILITY = PROJECT / "output" / "prefix_tokenization_stability.json"


def compare_priors(legacy: dict[str, np.ndarray], corrected: dict[str, np.ndarray]) -> dict:
    contexts = sorted(legacy.keys() & corrected.keys())
    l1_distances, kl_bits, flips = [], [], []
    flip_examples = []
    for context in contexts:
        old = legacy[context]
        new = corrected[context]
        l1_distances.append(float(np.abs(old - new).sum()))
        floor = 1e-12
        kl = float(np.sum(new * (np.log2(np.maximum(new, floor)) - np.log2(np.maximum(old, floor)))))
        kl_bits.append(kl)
        old_top, new_top = int(old.argmax()), int(new.argmax())
        flipped = old_top != new_top
        flips.append(flipped)
        if flipped and len(flip_examples) < 20:
            flip_examples.append(
                {"context": context, "legacy_top": SYMBOLS[old_top], "corrected_top": SYMBOLS[new_top]}
            )
    return {
        "n_contexts": len(contexts),
        "mean_l1_distance": float(np.mean(l1_distances)) if contexts else float("nan"),
        "mean_kl_bits": float(np.mean(kl_bits)) if contexts else float("nan"),
        "argmax_flip_fraction": float(np.mean(flips)) if contexts else float("nan"),
        "argmax_flip_examples": flip_examples,
    }


def main() -> None:
    from authorship.priors import TransformerPrior

    selections = pd.read_parquet(PROJECT / "output" / "intermediate" / "selections.parquet")
    contexts = sorted(selections["context_prefix"].unique())

    # NOTE (2026-08-06): the committed output/scorer_correction_impact.json's
    # Qwen/Qwen2.5-3B entry was hand-patched to {"status": "incomplete", ...} after this
    # loop hung on that checkpoint locally; re-running this script attempts it fresh and
    # may reproduce the same slowdown -- see task-3-report.md for the full story.
    model_names = ["gpt2", "gpt2-large", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
    per_model = {}
    for name in model_names:
        prior_obj = TransformerPrior(name)
        legacy = {c: prior_obj._legacy_full_sequence_prior(c) for c in contexts}
        corrected = {c: prior_obj.prior(c) for c in contexts}
        per_model[name] = compare_priors(legacy, corrected)
        print(f"{name}: {json.dumps({k: v for k, v in per_model[name].items() if k != 'argmax_flip_examples'})}")
        del prior_obj

    stability = json.loads(STABILITY.read_text()) if STABILITY.exists() else {}
    result = {
        "per_model": per_model,
        "diagnostic_fraction_unstable": stability.get("fraction_unstable"),
        "note": (
            "Cluster-tier models (including the primary prior) are compared separately "
            "once Tasks 5-9 produce before/after O2 shards; see Task 10's extension of "
            "this report."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
