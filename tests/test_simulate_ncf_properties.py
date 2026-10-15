"""Regression test for the NCF construct-validity simulation (eTable 12)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_simulation_covers_six_required_scenarios(tmp_path, monkeypatch):
    import simulate_ncf_properties as sim

    monkeypatch.setattr(sim, "OUTPUT_PATH", tmp_path / "ncf_simulation.json")
    sim.main()
    results = json.loads((tmp_path / "ncf_simulation.json").read_text())
    scenarios = {r["scenario"] for r in results}
    assert scenarios == {
        "agreement", "conflict", "near_tie", "sharply_peaked", "diffuse", "miscalibrated",
    }
    for row in results:
        assert 0.0 <= row["ncf"] <= 1.0


def test_diffuse_neural_scenario_yields_ncf_near_zero(tmp_path, monkeypatch):
    import simulate_ncf_properties as sim

    monkeypatch.setattr(sim, "OUTPUT_PATH", tmp_path / "ncf_simulation.json")
    sim.main()
    results = {r["scenario"]: r for r in json.loads((tmp_path / "ncf_simulation.json").read_text())}
    assert results["diffuse"]["ncf"] < 0.01


def test_agreement_scenario_places_ncf_near_one_half(tmp_path, monkeypatch):
    import simulate_ncf_properties as sim

    monkeypatch.setattr(sim, "OUTPUT_PATH", tmp_path / "ncf_simulation.json")
    sim.main()
    results = {r["scenario"]: r for r in json.loads((tmp_path / "ncf_simulation.json").read_text())}
    assert results["agreement"]["ncf"] == 0.5
