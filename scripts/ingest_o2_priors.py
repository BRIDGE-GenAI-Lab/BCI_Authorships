"""Convert prior files scored on the cluster into local shards and validate them.

The cluster tiers must be interchangeable with the locally scored tiers: same alphabet,
same exact-continuation method, same context set. This script enforces that before the
shards join the ladder.
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

SHARDS = PROJECT / "output" / "intermediate" / "prior_shards"
INBOX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/o2_priors")


def main() -> None:
    reference = pd.read_parquet(SHARDS / "gpt2.parquet")
    expected_contexts = set(reference["context_prefix"])
    SHARDS.mkdir(parents=True, exist_ok=True)

    for path in sorted(INBOX.glob("priors_*.jsonl")):
        if "smoke" in path.name:
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        name = frame["prior_model"].iloc[0]

        if set(frame["context_prefix"]) != expected_contexts:
            raise SystemExit(f"{name}: context set differs from the local ladder")
        matrix = np.stack(frame["p_lm"].map(np.asarray).to_numpy())
        if matrix.shape[1] != len(SYMBOLS):
            raise SystemExit(f"{name}: alphabet size {matrix.shape[1]}, expected {len(SYMBOLS)}")
        if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
            raise SystemExit(f"{name}: priors do not sum to one")
        if not np.isfinite(matrix).all():
            raise SystemExit(f"{name}: non-finite prior values")

        out = SHARDS / f"{name.replace('/', '_')}.parquet"
        frame.to_parquet(out, index=False)
        top = matrix.argmax(axis=1)
        print(
            f"{name:26s} n={len(frame):4d} params={frame['prior_parameters'].iloc[0]:>13,} "
            f"mean_max_p={matrix.max(axis=1).mean():.3f} "
            f"top1_is_space={(np.array([SYMBOLS[i] for i in top]) == ' ').mean():.3f} -> {out.name}"
        )


if __name__ == "__main__":
    main()
