"""Regression tests for the closed-loop emitted-context sensitivity analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_emitted_context_uses_prior_fused_emissions_not_intended_characters():
    import build_attribution_emitted_context as bec

    session = [
        {"selection_id": 1, "intended": "D", "context_prefix": ""},
        {"selection_id": 2, "intended": "E", "context_prefix": "D"},
        {"selection_id": 3, "intended": "C", "context_prefix": "DE"},
    ]
    fused_emissions = {1: "D", 2: "E", 3: "X"}  # selection 3's fused output was wrong: "X" not "C"
    contexts = bec.emitted_contexts_for_session(session, fused_emissions)
    assert contexts[1] == ""
    assert contexts[2] == "D"
    assert contexts[3] == "DE"
    # a hypothetical 4th selection would see "DEX", the actually-emitted (wrong) prefix:
    session.append({"selection_id": 4, "intended": "I", "context_prefix": "DEC"})
    fused_emissions[4] = "I"
    contexts = bec.emitted_contexts_for_session(session, fused_emissions)
    assert contexts[4] == "DEX"
