#!/usr/bin/env bash
# O2-only closed-loop emitted-context sensitivity for the primary prior (Qwen/Qwen2.5-32B).
#
# Unlike the study's existing parallel scoring (one forward pass per UNIQUE context, reused
# across every selection that shares it; see authorship/priors.py's TransformerPrior, the
# git-committed source of the exact-continuation method both this study's cluster scoring
# and this script use), this job scores selections in a strictly sequential per-phrase loop:
# see scripts/build_attribution_emitted_context.py. Every selection's prior context depends
# on the fused decision rule's own emission at the prior selection in the same phrase, so
# contexts are session-specific and cannot be cached or parallelized the way the
# intended-context ladder is. There is exactly one model and one fixed input here (no
# candidate array, no smoke preflight), so this script is a leaner single-job variant of the
# conventions already used by this study's other O2 cluster jobs (which live only on O2
# scratch, never committed to this repository): idrift_env, set -Eeuo pipefail, immutable
# output, checksum verification, environment capture, and a written status record for
# provenance.
#
# Partition/account: gpu_sun/sun_hs285_contrib (the lab's contributed hardware) is the
# default below and is what actually produced eTable 16. It stayed queued for 9h once
# (behind other large pending arrays on the same account) and was briefly moved to the
# shared `gpu` partition under the sibling `sun_hs285` (non-contrib) account as a
# congestion workaround -- confirmed viable via `sbatch --test-only` after checking
# `sacctmgr show associations user=alg7274` showed no partition ACL restriction -- before
# gpu_sun capacity was freed by deferring other pending jobs on the account and the job was
# resubmitted back to gpu_sun for the run that finished successfully. Both are accepted by
# the guard below (override with --partition/--account on the sbatch command line if
# congestion recurs). `gpu_requeue` was also tried but is genuinely blocked (`Setting QOS to
# gpuquad_qos` / `Invalid qos specification` -- this account has no such QOS grant), so it
# is not used.
#SBATCH --job-name=auth_emitted_ctx
#SBATCH --partition=gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/n/scratch/users/a/alg7274/authorship_emitted_context/logs/%x_%j.out

set -Eeuo pipefail

VALID_PARTITIONS=("gpu_sun" "gpu")
partition_ok="false"
for candidate in "${VALID_PARTITIONS[@]}"; do
  [[ "${SLURM_JOB_PARTITION:-}" == "$candidate" ]] && partition_ok="true"
done
if [[ "$partition_ok" != "true" || -z "${SLURM_JOB_ID:-}" ]]; then
  echo "refusing emitted-context scoring outside a known O2 GPU allocation (gpu_sun or gpu)" >&2
  exit 70
fi

SEED="${SEED:-20260802}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-/n/scratch/users/a/alg7274/hf_cache}"
SOURCE_ROOT="${SOURCE_ROOT:-/n/scratch/users/a/alg7274/authorship}"
RUN_ROOT="${RUN_ROOT:-/n/scratch/users/a/alg7274/authorship_emitted_context}"
PYTHON="${PYTHON:-/n/scratch/users/a/alg7274/idrift_env/bin/python}"

CODE_ROOT="$RUN_ROOT/code"
RESULT_ROOT="$RUN_ROOT/results"
STATUS_ROOT="$RUN_ROOT/status"
LOG_ROOT="$RUN_ROOT/logs"
RUNNER="$CODE_ROOT/scripts/build_attribution_emitted_context.py"
ATTRIBUTION_MODULE="$CODE_ROOT/authorship/attribution.py"
GRID_MODULE="$CODE_ROOT/authorship/grid.py"
SELECTIONS="$CODE_ROOT/output/intermediate/selections.parquet"

MODEL="Qwen/Qwen2.5-32B"
TIER="neural_32b"
PARAMETERS=32500000000
: "${CHECKPOINT_REVISION:?set the immutable checkpoint commit sha (40 hex chars) pinned for this model on O2}"
TOKENIZER_REVISION="${TOKENIZER_REVISION:-$CHECKPOINT_REVISION}"
: "${EXPECTED_CODE_DIGEST:?set staged code bundle sha256 (runner + attribution.py + grid.py)}"
: "${EXPECTED_DATA_DIGEST:?set staged selections.parquet sha256}"

SAFE_MODEL="${MODEL//\//_}"
STATUS="$STATUS_ROOT/emitted_context_${SAFE_MODEL}.json"
OUT="$RESULT_ROOT/emitted_context_attribution_${SAFE_MODEL}.parquet"
TMP_OUT="$OUT.tmp.${SLURM_JOB_ID}"
LOG="$LOG_ROOT/score_emitted_context_${SAFE_MODEL}_${SLURM_JOB_ID}.log"
START_EPOCH="$(date +%s)"
HOSTNAME="$(hostname)"
GPU_IDENTITY=""
EXIT_CODE=0
PYTHON_VERSION="unavailable"
TORCH_VERSION="unavailable"
TRANSFORMERS_VERSION="unavailable"
CUDA_RUNTIME_VERSION="unavailable"
CUDA_AVAILABLE="unavailable"
CUDA_DRIVER_VERSION="unavailable"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unavailable}"
WARNINGS=()
ENV_CAPTURE_OK="false"

mkdir -p "$CODE_ROOT/authorship" "$CODE_ROOT/scripts" "$CODE_ROOT/output/intermediate" \
  "$RESULT_ROOT" "$STATUS_ROOT" "$LOG_ROOT"
[[ ! -e "$LOG" ]] || { echo "immutable log already exists: $LOG" >&2; exit 73; }
exec > >(tee "$LOG") 2>&1

capture_value() {
  local target="$1" code="$2" value
  if value="$($PYTHON -c "$code" 2>/dev/null)" && [[ -n "$value" ]]; then
    printf -v "$target" '%s' "$value"
  else
    printf -v "$target" '%s' "unavailable"
    WARNINGS+=("environment capture failed: $target")
  fi
}

capture_environment() {
  WARNINGS=()
  capture_value PYTHON_VERSION 'import platform; print(platform.python_version())'
  capture_value TORCH_VERSION 'import torch; print(torch.__version__)'
  capture_value TRANSFORMERS_VERSION 'import transformers; print(transformers.__version__)'
  capture_value CUDA_RUNTIME_VERSION 'import torch; print(torch.version.cuda or "unavailable")'
  capture_value CUDA_AVAILABLE 'import torch; print(str(torch.cuda.is_available()).lower())'
  if ! GPU_IDENTITY="$(nvidia-smi --query-gpu=name,uuid --format=csv,noheader 2>/dev/null | tr '\n' ';')" || [[ -z "$GPU_IDENTITY" ]]; then
    GPU_IDENTITY="unavailable"
    WARNINGS+=("environment capture failed: GPU identity")
  fi
  if ! CUDA_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)" || [[ -z "$CUDA_DRIVER_VERSION" ]]; then
    CUDA_DRIVER_VERSION="unavailable"
    WARNINGS+=("environment capture failed: CUDA driver")
  fi
  if [[ "$CUDA_VISIBLE_DEVICES" == "unavailable" ]]; then
    WARNINGS+=("CUDA_VISIBLE_DEVICES was unset")
  fi
  if [[ "${#WARNINGS[@]}" -eq 0 ]]; then
    ENV_CAPTURE_OK="true"
  fi
}

write_status() {
  local state="$1" reason="$2" rows="${3:-0}"
  if [[ -e "$STATUS" ]]; then
    echo "refusing to write status: existing status requires validation" >&2
    return 73
  fi
  local output_sha256=""
  [[ -f "$OUT" ]] && output_sha256="$(sha256sum "$OUT" | awk '{print $1}')"
  local elapsed_seconds=$(( $(date +%s) - START_EPOCH ))
  local warning_text=""
  if [[ "${#WARNINGS[@]}" -gt 0 ]]; then local old_ifs="$IFS"; IFS='|'; warning_text="${WARNINGS[*]}"; IFS="$old_ifs"; fi
  "$PYTHON" - "$STATUS" "$MODEL" "$TIER" "$PARAMETERS" "$state" "$reason" "$rows" "$SEED" \
    "$EXPECTED_CODE_DIGEST" "$EXPECTED_DATA_DIGEST" "$CHECKPOINT_REVISION" "$TOKENIZER_REVISION" \
    "$output_sha256" "$GPU_IDENTITY" "$elapsed_seconds" "$EXIT_CODE" "$HOSTNAME" \
    "$PYTHON_VERSION" "$TORCH_VERSION" "$TRANSFORMERS_VERSION" "$CUDA_RUNTIME_VERSION" \
    "$CUDA_AVAILABLE" "$CUDA_DRIVER_VERSION" "$CUDA_VISIBLE_DEVICES" "$warning_text" <<'PY'
import json, os, sys
from pathlib import Path
(path, model, tier, parameters, state, reason, rows, seed, code_digest, data_digest,
 checkpoint_revision, tokenizer_revision, output_sha256, gpu_identity, elapsed_seconds,
 exit_code, hostname, python_version, torch_version, transformers_version,
 cuda_runtime_version, cuda_available, cuda_driver_version, cuda_visible_devices,
 warning_text) = sys.argv[1:]
warnings = [item for item in warning_text.split("|") if item]
environment = {"python_version": python_version, "torch_version": torch_version,
               "transformers_version": transformers_version, "cuda_runtime_version": cuda_runtime_version,
               "cuda_available": cuda_available, "cuda_driver_version": cuda_driver_version,
               "cuda_visible_devices": cuda_visible_devices, "device": "cuda",
               "gpu_identity": gpu_identity}
payload = {"model": model, "tier": tier, "parameters": int(parameters), "status": state,
           "reason": reason, "rows": int(rows), "seed": int(seed),
           "code_digest": code_digest, "data_digest": data_digest,
           "checkpoint_id": model, "tokenizer_id": model,
           "checkpoint_revision": checkpoint_revision, "tokenizer_revision": tokenizer_revision,
           "output_sha256": output_sha256, "hostname": hostname, "gpu_identity": gpu_identity,
           "elapsed_seconds": int(elapsed_seconds), "exit_code": int(exit_code),
           "warnings": warnings, "environment": environment,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "runner": "scripts/build_attribution_emitted_context.py"}
tmp = Path(f"{path}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
tmp.replace(path)
PY
}

on_exit() {
  local rc=$?
  EXIT_CODE=$rc
  if (( rc != 0 )); then
    write_status "FAIL" "wrapper exited with status $rc" 0 || true
  fi
  exit "$rc"
}
trap on_exit EXIT

capture_environment

stage_once() {
  local source="$1" destination="$2"
  [[ -e "$destination" ]] || install -m 0644 "$source" "$destination"
}
stage_once "$SOURCE_ROOT/scripts/build_attribution_emitted_context.py" "$RUNNER"
stage_once "$SOURCE_ROOT/authorship/attribution.py" "$ATTRIBUTION_MODULE"
stage_once "$SOURCE_ROOT/authorship/grid.py" "$GRID_MODULE"
stage_once "$SOURCE_ROOT/output/intermediate/selections.parquet" "$SELECTIONS"
[[ -s "$RUNNER" && -s "$ATTRIBUTION_MODULE" && -s "$GRID_MODULE" && -s "$SELECTIONS" ]]

actual_code_digest="$("$PYTHON" - "$RUNNER" "$ATTRIBUTION_MODULE" "$GRID_MODULE" <<'PY'
import hashlib, pathlib, sys
digest = hashlib.sha256()
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
[[ "$actual_code_digest" == "$EXPECTED_CODE_DIGEST" ]]
actual_data_digest="$(sha256sum "$SELECTIONS" | awk '{print $1}')"
[[ "$actual_data_digest" == "$EXPECTED_DATA_DIGEST" ]]

[[ "$ENV_CAPTURE_OK" == "true" ]] || { echo "environment capture failed; refusing PASS status" >&2; exit 78; }

validate_output() {
  local candidate_path="$1"
  "$PYTHON" - "$candidate_path" "$SELECTIONS" "$MODEL" <<'PY'
import json, math, sys
import pandas as pd
path, selections_path, model = sys.argv[1:]
selections = pd.read_parquet(selections_path)
frame = pd.read_parquet(path)
required = {"emitted_context", "p_lm", "ncf", "phantom_agreement", "prior_capture",
            "fused_correct", "neural_correct", "prior_correct", "relative_path",
            "trial_number", "target_symbol", "study_participant_id", "prior_model"}
missing = required - set(frame.columns)
if missing:
    raise SystemExit(f"output is missing required columns: {sorted(missing)}")
if len(frame) != len(selections):
    raise SystemExit(f"expected {len(selections)} rows, found {len(frame)}")
if (frame["prior_model"] != model).any():
    raise SystemExit("output contains rows not labeled with the scored model")
for values in frame["p_lm"]:
    values = [float(v) for v in values]
    if len(values) != 36 or not all(math.isfinite(v) and v >= 0 for v in values):
        raise SystemExit("output contains invalid 36-way probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise SystemExit("output probabilities do not sum to one")
## ncf = neural_displacement / (neural_displacement + prior_displacement), where each
## displacement is a KL divergence computed directly in log2-space (authorship/attribution.py
## _kl_bits). KL divergence is non-negative by construction, but when the fused posterior is
## numerically near-identical to one source distribution, floating-point cancellation in that
## direct-formula computation can leave the corresponding term a few ULPs below zero, pushing
## ncf a few ULPs outside [0, 1]. This is a property of the existing, already-used KL formula
## (present in the primary intended-context attribution too, just never gated this strictly
## there), not a defect in this sequential scorer -- so tolerate it at float64-noise scale
## (1e-6, several orders above the largest observed violation) while still failing loudly on
## any real, large-magnitude violation that would indicate an actual bug.
tolerance = 1e-6
ncf_values = frame["ncf"].dropna().to_numpy()
violations = ncf_values[(ncf_values < -tolerance) | (ncf_values > 1 + tolerance)]
if len(violations):
    raise SystemExit(
        f"ncf values must lie in [0, 1] within floating-point tolerance ({tolerance}) or be "
        f"NaN; found {len(violations)} real violations, e.g. {violations[:5].tolist()}"
    )
merge_keys = ["relative_path", "trial_number"]
joined = frame.merge(selections[merge_keys + ["target_symbol"]], on=merge_keys,
                      suffixes=("", "_archive"))
if len(joined) != len(frame) or (joined["target_symbol"] != joined["target_symbol_archive"]).any():
    raise SystemExit("output rows do not exactly match the staged selections by phrase/trial")
print(f"validated {len(frame)} rows")
PY
}

if [[ -e "$OUT" ]]; then
  echo "immutable output already exists; validating without overwriting: $OUT"
  validate_output "$OUT"
else
  rm -f "$TMP_OUT"
  PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" "$RUNNER" \
    --model "$MODEL" --tier "$TIER" --parameters "$PARAMETERS" \
    --selections "$SELECTIONS" --out "$TMP_OUT" --seed "$SEED" \
    --revision "$CHECKPOINT_REVISION" --tokenizer-revision "$TOKENIZER_REVISION"
  validate_output "$TMP_OUT"
  [[ ! -e "$OUT" ]] || { echo "refusing to replace immutable output: $OUT" >&2; exit 73; }
  mv "$TMP_OUT" "$OUT"
fi

if [[ -e "$STATUS" ]]; then
  echo "existing status found; not rewriting immutable status: $STATUS"
else
  rows="$("$PYTHON" -c "import pandas as pd; print(len(pd.read_parquet('$OUT')))")"
  write_status "PASS" "emitted-context sequential scoring validated against staged selections" "$rows"
fi
echo "EMITTED_CONTEXT_PASS model=$MODEL output=$OUT"
