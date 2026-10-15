"""Compare a freshly regenerated stats digest with the reference one.

Usage: python scripts/check_digest_reproduces.py results/digests/stats_digest.json output/stats_digest.json
Exits 1 and prints every mismatching JSON path if they differ.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def compare_digests(expected, actual, rtol: float = 1e-12, path: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                mismatches.append(child)
            else:
                mismatches.extend(compare_digests(expected[key], actual[key], rtol, child))
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatches.extend(compare_digests(left, right, rtol, f"{path}[{index}]"))
        return mismatches
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            if math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=1e-15):
                return []
        except (TypeError, ValueError):
            pass
        return [path]
    return [] if expected == actual else [path]


def main() -> None:
    expected = json.loads(Path(sys.argv[1]).read_text())
    actual = json.loads(Path(sys.argv[2]).read_text())
    mismatches = compare_digests(expected, actual)
    if mismatches:
        print(f"{len(mismatches)} mismatching paths:")
        for item in mismatches[:200]:
            print("  ", item)
        sys.exit(1)
    print("digest reproduces exactly")


if __name__ == "__main__":
    main()
