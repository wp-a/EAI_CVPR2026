#!/usr/bin/env bash
set -euo pipefail

# Four-GPU sharded offline inference for the AxisTilted2 EAI Qwen3-32B LoRA.
# By default this runs VirtualHome Action Sequencing, which is the module the
# downloaded LoRA was trained for.

OUTPUT_DIR="${OUTPUT_DIR:-sample_submission_axistilted2_qwen3_32b_lora_vas_4gpu}"
LOG_DIR="${LOG_DIR:-logs/axistilted2_qwen3_lora_vas_4gpu_$(date +%Y%m%d_%H%M%S)}"
IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
NUM_SHARDS="${NUM_SHARDS:-4}"
FILES="${FILES:-virtualhome_action_sequencing_prompts.json}"
LIMIT="${LIMIT:-}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_axistilted2_qwen3_lora_offline_4gpu_sharded.sh [generator args]

Default task:
  FILES=virtualhome_action_sequencing_prompts.json

Common examples:
  LIMIT=8 bash scripts/run_axistilted2_qwen3_lora_offline_4gpu_sharded.sh
  FILES="virtualhome_action_sequencing_prompts.json" OUTPUT_DIR=sample_submission_axistilted2_vas bash scripts/run_axistilted2_qwen3_lora_offline_4gpu_sharded.sh
EOF
  exit 0
fi

mkdir -p "${LOG_DIR}"

extra_args=()
merge_args=()
if [[ -n "${LIMIT}" ]]; then
  extra_args+=(--limit "${LIMIT}")
  merge_args+=(--limit "${LIMIT}")
fi
if [[ -n "${FILES}" ]]; then
  # shellcheck disable=SC2206
  files_array=(${FILES})
  extra_args+=(--files "${files_array[@]}")
  merge_args+=(--files "${files_array[@]}")
fi
extra_args+=("${@}")

start_worker() {
  local gpu="$1"
  local log_file="${LOG_DIR}/gpu${gpu}_shard${gpu}.log"

  echo "Starting GPU ${gpu}: shard ${gpu}/${NUM_SHARDS}"
  (
    export GPU="${gpu}"
    export OUTPUT_DIR
    export BATCH_SIZE
    export MAX_MODEL_LEN
    export GPU_MEMORY_UTILIZATION
    bash scripts/run_axistilted2_qwen3_lora_offline.sh \
      "${extra_args[@]}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${gpu}" \
      --shard-output
  ) >"${log_file}" 2>&1 &
  echo "$!" >"${LOG_DIR}/gpu${gpu}_shard${gpu}.pid"
}

for gpu in 0 1 2 3; do
  start_worker "${gpu}"
done

echo "Logs: ${LOG_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Files: ${FILES}"
echo "Waiting for workers..."

failed=0
for pid_file in "${LOG_DIR}"/*.pid; do
  pid="$(cat "${pid_file}")"
  if ! wait "${pid}"; then
    echo "Worker failed: ${pid_file}"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit "${failed}"
fi

docker run --rm \
  --entrypoint bash \
  -v "$(pwd):/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  -lc "chmod -R a+rwX '${OUTPUT_DIR}'"

python3 scripts/merge_vllm_shards_selected.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  "${merge_args[@]}"

echo "Finished at $(date)"
