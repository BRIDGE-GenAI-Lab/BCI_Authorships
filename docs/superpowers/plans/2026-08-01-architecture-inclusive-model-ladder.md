# Additive Architecture-Inclusive Model Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add scientifically justified causal-language-model architectures to the existing authorship prior ladder while preserving every current model and running all model and CPU work on O2.

**Architecture:** Keep the existing prior, attribution, bootstrap, and manuscript interfaces. Add a frozen screening registry and six candidates: RWKV-7, RecurrentGemma/Griffin, xLSTM-7B, StripedHyena-Hessian-7B, RetNet, and Gated DeltaNet. Use O2 smoke validation to determine which conditional candidates enter the full additive ladder.

**Tech Stack:** Python 3, pandas, NumPy, pytest, Hugging Face Transformers, PyTorch, SLURM, HMS O2 gpu_sun, parquet/JSONL, Markdown manuscript assembly.

---

## File map

- Create docs/model_screening_2026-08-01.json: frozen candidate metadata, evidence links, roles, and decisions.
- Modify authorship/priors.py: recognize new model namespaces as families.
- Modify scripts/build_priors.py: register six cluster-scored specifications; never enable local scoring for them.
- Modify scripts/run_stats.py: preserve Qwen primary calculations and add architecture-expansion metadata with Holm-adjusted optional comparisons.
- Modify scripts/make_figures.py: append new rows, labels, colors, and family legend entries.
- Create scripts/o2_architecture_smoke.sbatch: O2-only 31-context preflight.
- Create scripts/o2_architecture_score.sbatch: O2-only 951-context scoring.
- Create scripts/o2_architecture_postprocess.sbatch: O2 CPU ingestion, attribution, bootstrap, and statistics.
- Create tests/test_model_screening.py: registry and O2-wrapper invariants.
- Modify tests/test_priors.py, tests/test_build_priors_ladder.py, and tests/test_make_figures_ladder.py.
- Modify the actual-study assembly and manuscript files under /Volumes/Extreme SSD/Mimic-IV/study_bci_llm_authorship/ only after verified O2 results exist.

## Candidate registry

Create docs/model_screening_2026-08-01.json with these exact candidates:

| Name | Family | Role | Parameters | Architecture |
|---|---|---:|---:|---|
| RWKV/RWKV7-Goose-World3-2.9B-HF | rwkv | additive primary | 2.9B | RWKV-7 recurrent/linear-attention |
| google/recurrentgemma-2b | recurrentgemma | additive primary | 2.7B | Griffin gated recurrence plus local attention |
| NX-AI/xLSTM-7b | xlstm | additive primary | 7B | xLSTM recurrent memory |
| togethercomputer/StripedHyena-Hessian-7B | stripedhyena | additive primary | 8B | attention plus gated Hyena convolution |
| fla-hub/retnet-2.7B-100B | retnet | conditional | 2.7B | Retentive Network |
| m-a-p/1.3B-100B-GatedDeltaNet-pure | gated_deltanet | conditional | 1.3B | gated delta-rule linear attention |

Each record must include name, family, role, parameters, base_or_instruction, architecture, paper, model_card, license_or_access, decision, and search_date. Use the primary evidence URLs documented in the approved design spec. Record explicit exclusions for HyenaDNA/Evo2, TTT, Zamba/Jamba/Falcon-Mamba, and BitNet/MoE/GQA/RoPE/LongNet variants. Copy every current LADDER name into preserved_existing_models; no preserved name may appear in exclusions.

### Task 1: Freeze the registry

**Files:**
- Create docs/model_screening_2026-08-01.json
- Create tests/test_model_screening.py

- [ ] Step 1: Write failing registry tests.

~~~python
def test_registry_has_six_candidates_and_four_additive_primary_families():
    registry = load_registry()
    assert len(registry["candidates"]) == 6
    assert sum(row["role"] == "additive_primary" for row in registry["candidates"]) == 4
    assert sum(row["role"] == "conditional" for row in registry["candidates"]) == 2


def test_candidate_records_have_evidence_and_reproducibility_fields():
    required = {"name", "family", "role", "parameters", "architecture",
                "paper", "model_card", "decision", "search_date"}
    for row in load_registry()["candidates"]:
        assert required <= row.keys()
        assert row["parameters"] > 0
        assert row["paper"].startswith("https://")
        assert row["model_card"].startswith("https://")


def test_exclusions_do_not_remove_current_models():
    registry = load_registry()
    assert set(registry["preserved_existing_models"]).isdisjoint(
        row["name"] for row in registry["excluded"]
    )
    assert all(row["reason"] for row in registry["excluded"])
~~~

- [ ] Step 2: Run pytest tests/test_model_screening.py -q and verify failure because the registry and test file do not exist.

- [ ] Step 3: Create the registry with search_date 2026-08-01, the six exact IDs, the authoritative paper/model-card URLs, explicit exclusions, and the complete current ladder preservation list.

- [ ] Step 4: Run pytest tests/test_model_screening.py -q. Expected: PASS.

- [ ] Step 5: Commit only the registry files.

~~~bash
git add docs/model_screening_2026-08-01.json tests/test_model_screening.py
git commit -m "Add frozen architecture screening registry"
~~~

### Task 2: Register families and additive ladder entries

**Files:**
- Modify authorship/priors.py
- Modify scripts/build_priors.py
- Modify tests/test_priors.py
- Modify tests/test_build_priors_ladder.py

- [ ] Step 1: Add failing assertions for these family mappings:

~~~python
assert family_of("RWKV/RWKV7-Goose-World3-2.9B-HF") == "rwkv"
assert family_of("google/recurrentgemma-2b") == "recurrentgemma"
assert family_of("NX-AI/xLSTM-7b") == "xlstm"
assert family_of("togethercomputer/StripedHyena-Hessian-7B") == "stripedhyena"
assert family_of("fla-hub/retnet-2.7B-100B") == "retnet"
assert family_of("m-a-p/1.3B-100B-GatedDeltaNet-pure") == "gated_deltanet"
~~~

Also assert that all six entries are kind cluster, have positive parameter counts, and that representative current names uniform, ngram5, gpt2, Qwen/Qwen2.5-32B, EleutherAI/pythia-12b, and state-spaces/mamba-2.8b-hf remain present.

- [ ] Step 2: Run pytest tests/test_priors.py tests/test_build_priors_ladder.py -q and verify only the new assertions fail.

- [ ] Step 3: Append these family prefixes without altering existing mappings:

~~~python
("RWKV/", "rwkv"),
("google/recurrentgemma", "recurrentgemma"),
("NX-AI/xLSTM", "xlstm"),
("togethercomputer/StripedHyena", "stripedhyena"),
("fla-hub/retnet", "retnet"),
("m-a-p/1.3B-100B-GatedDeltaNet", "gated_deltanet"),
~~~

- [ ] Step 4: Append six kind cluster specifications to scripts/build_priors.py LADDER using the exact registry IDs and parameter counts. Keep existing specifications unchanged. Do not add a local scoring branch.

- [ ] Step 5: Run the targeted tests; expected: PASS.

- [ ] Step 6: Commit the additive registration.

~~~bash
git add authorship/priors.py scripts/build_priors.py tests/test_priors.py tests/test_build_priors_ladder.py
git commit -m "Register additive recurrent and convolutional prior families"
~~~

### Task 3: Create O2 smoke and full-scoring wrappers

**Files:**
- Create scripts/o2_architecture_smoke.sbatch
- Create scripts/o2_architecture_score.sbatch
- Modify tests/test_model_screening.py

- [ ] Step 1: Write wrappers using gpu_sun, account sun_hs285_contrib, idrift_env, the existing authorship scratch payload, HF_HOME, tokenizer parallelism disabled, set -euo pipefail, and nvidia-smi. Neither wrapper may run model scoring locally.

The smoke array must score all six candidates on 31 contexts and write one immutable JSONL result and log per candidate. The full array must score only candidates whose smoke record is PASS on all 951 contexts, use exact model/tier/parameter arrays, and reject an existing output unless it passes the 951-row validator. Pass trust_remote_code for xLSTM, StripedHyena, RetNet, and Gated DeltaNet when required by their model cards.

- [ ] Step 2: Validate shell syntax without executing compute.

~~~bash
bash -n scripts/o2_architecture_smoke.sbatch
bash -n scripts/o2_architecture_score.sbatch
~~~

Expected: exit code 0.

- [ ] Step 3: Add tests asserting both scripts contain gpu_sun, sun_hs285_contrib, idrift_env, all six exact model IDs, and no local-only scoring invocation.

- [ ] Step 4: Run pytest tests/test_model_screening.py -q and commit the wrappers.

~~~bash
git add scripts/o2_architecture_smoke.sbatch scripts/o2_architecture_score.sbatch tests/test_model_screening.py
git commit -m "Add O2 architecture preflight and scoring jobs"
~~~

### Task 4: Add manifest-driven O2 postprocessing and statistics

**Files:**
- Create scripts/o2_architecture_postprocess.sbatch
- Modify scripts/ingest_o2_priors.py
- Modify scripts/run_stats.py
- Modify tests/test_model_screening.py

- [ ] Step 1: Add failing validation tests for smoke failure, fewer than 951 full contexts, context-set mismatch, and preservation of existing models when a new candidate fails.

- [ ] Step 2: Implement manifest-driven validation requiring exact model/revision/context matches, finite 36-way vectors summing to one, and 951 unique contexts before inclusion. Retain failures in a status report. Reject partial rows rather than imputing or silently dropping them.

- [ ] Step 3: Make the O2 postprocess job run ingestion, build_attribution.py, and run_stats.py on O2 CPU resources, writing combined priors, attribution parquet, statistics digest, candidate status, and SHA-256 hashes.

~~~bash
python scripts/ingest_o2_priors.py "$WORK/inbox" --manifest "$WORK/model_screening_2026-08-01.json"
python scripts/build_attribution.py
python scripts/run_stats.py
~~~

- [ ] Step 4: Preserve PRIMARY_PRIOR, co-primary calculations, bootstrap seed, and current primary digest. Add an architecture_expansion block with estimates, 95% bootstrap intervals, raw optional comparison P values, and Holm-adjusted P values. Never use these P values to decide inclusion.

- [ ] Step 5: Run pytest tests/test_model_screening.py tests/test_priors.py tests/test_build_priors_ladder.py -q and commit.

~~~bash
git add scripts/ingest_o2_priors.py scripts/run_stats.py scripts/o2_architecture_postprocess.sbatch tests/test_model_screening.py
git commit -m "Add manifest-driven O2 validation and architecture statistics"
~~~

### Task 5: Extend the family-grouped figure additively

**Files:**
- Modify scripts/make_figures.py
- Modify tests/test_make_figures_ladder.py

- [ ] Step 1: Add failing tests requiring all six new names in LADDER_ORDER, labels, family mappings, and colors, while requiring every pre-existing LADDER_ORDER name to remain.

- [ ] Step 2: Add labels RWKV-7 2.9B, RecurrentGemma 2B, xLSTM 7B, StripedHyena 7B, RetNet 2.7B, and Gated DeltaNet 1.3B. Add distinct colors without changing existing colors.

- [ ] Step 3: Append new rows after the current ladder. Plot conditional models only when validated result rows exist; do not draw zero-valued placeholders.

- [ ] Step 4: Run pytest tests/test_make_figures_ladder.py -q; expected: PASS.

~~~bash
git add scripts/make_figures.py tests/test_make_figures_ladder.py
git commit -m "Add architecture families to prior ladder figure"
~~~

### Task 6: Execute the model workflow on O2 only

**Files:**
- Use scripts/o2_architecture_smoke.sbatch
- Use scripts/o2_architecture_score.sbatch
- Use scripts/o2_architecture_postprocess.sbatch

- [ ] Step 1: Check the O2 ControlMaster with ssh -O check. If dead, use the configured login helper once; never request credentials in chat or retry in a loop.

- [ ] Step 2: Stage exact committed files, registry, contexts, and runner payload to /n/scratch/users/a/alg7274/authorship_architecture_2026_08_01/ and record hashes. Verify support for all six types before requesting GPUs.

- [ ] Step 3: Submit the six-candidate smoke array with sbatch. Do not cancel or modify existing iDrift jobs. Require 31/31 contexts, finite probabilities, normalization, deterministic repeat, and complete metadata for PASS.

- [ ] Step 4: Write PASS, FAIL, or EXCLUDED plus a reason for every candidate. Submit only PASS candidates to the full array; never silently substitute.

- [ ] Step 5: Submit the full 951-context array using immutable output names and the same objective as the current 24 priors. Monitor with squeue and finalize with sacct. All scoring and heavy CPU work remain on O2.

- [ ] Step 6: Postprocess on O2 CPU, require POSTPROCESS_OK, verify current/new rows, and compare the Qwen primary digest byte-for-byte with the pre-expansion digest.

- [ ] Step 7: Retrieve only verified digests, status reports, parquet outputs, logs, and hashes. Do not ingest incomplete shards.

### Task 7: Update the manuscript after verified O2 results

**Files:**
- Modify /Volumes/Extreme SSD/Mimic-IV/study_bci_llm_authorship/scripts/assemble_manuscript.py
- Modify actual-study manuscript/methods.md, results.md, discussion.md, cover_letter.md, references.bib, and refs_verified.json

- [ ] Step 1: Add an Architecture search and prior-model inclusion Methods subsection stating the search cutoff, authoritative sources, frozen registry, additive preservation, base-model preference, family rationale, conditional preflight, and explicit exclusions.

- [ ] Step 2: Add supplementary provenance tables with model ID, revision, tokenizer, parameters, base/instruction status, license/access, architecture, evidence links, O2 status, and failure/exclusion reason.

- [ ] Step 3: Add only verified result rows. Preserve every existing table row and the Qwen headline values. Report conditional failures as screened/failed rather than unexplained missing models.

- [ ] Step 4: Add and verify RWKV, RecurrentGemma/Griffin, xLSTM, StripedHyena, RetNet, and Gated DeltaNet references corresponding to included or screened candidates.

- [ ] Step 5: Regenerate with /usr/bin/python3 /Volumes/Extreme SSD/Mimic-IV/study_bci_llm_authorship/scripts/assemble_manuscript.py. Check expanded counts, historical pre-expansion counts, and failed-candidate language.

### Task 8: Final verification

- [ ] Step 1: Run pytest -q. Local tests must not download models or perform heavy scoring.

- [ ] Step 2: Run git diff --check and scan the plan/registry for unresolved placeholder markers.

- [ ] Step 3: Confirm all current model names remain, no excluded candidate has attribution rows, all six candidates have final status records, and the Qwen primary digest is byte-identical before and after expansion.

- [ ] Step 4: Commit only scoped repository changes.

~~~bash
git add authorship scripts tests docs/model_screening_2026-08-01.json
git commit -m "Add architecture-inclusive prior ladder implementation"
~~~

Do not stage unrelated pre-existing worktree changes. Keep actual-study manuscript/output changes in their existing workflow and report their paths explicitly.
