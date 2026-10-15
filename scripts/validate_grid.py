"""Validate the grid code convention by reconstructing intended phrases from real EDFs.

The archive codes targets as integers. If our one-based channel-order convention is
right, concatenating the intended symbols of consecutive Test trials must reproduce
readable copy-spelling phrases. If it produces gibberish the convention is wrong and
every downstream attribution number would be wrong with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COMPANION = PROJECT.parent / "study_bigp3_als_calibration"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(COMPANION / "src"))

from bigp3_als.edf import parse_source_path, select_edf_paths  # noqa: E402
from bigp3_als.trials import reconstruct_edf_trials  # noqa: E402

from authorship.grid import target_code_to_symbol  # noqa: E402

CACHE = COMPANION / "data" / "source_cache" / "bigP3BCI-data"


def phrase_for_file(edf_path: Path, relative_path: str) -> str:
    trials = reconstruct_edf_trials(edf_path, relative_path)
    trials = trials[trials["target"].notna()].sort_values("trial_number")
    return "".join(target_code_to_symbol(int(code)) for code in trials["target"])


def main() -> None:
    cache_root = CACHE.parent
    test_paths = [
        path
        for path in select_edf_paths(cache_root)
        if parse_source_path(path.relative_to(cache_root).as_posix()).phase == "Test"
    ]
    print(f"test EDFs available: {len(test_paths)}")
    by_study: dict[str, list[str]] = {}
    for path in test_paths:
        relative = path.relative_to(cache_root).as_posix()
        study = parse_source_path(relative).study
        if len(by_study.setdefault(study, [])) >= 4:
            continue
        try:
            by_study[study].append(f"{Path(relative).name}: {phrase_for_file(path, relative)!r}")
        except Exception as error:  # noqa: BLE001 - diagnostic script
            by_study[study].append(f"{Path(relative).name}: FAILED {error}")
    for study, samples in sorted(by_study.items()):
        print(f"\n=== {study} ===")
        for sample in samples:
            print(" ", sample)


if __name__ == "__main__":
    main()
