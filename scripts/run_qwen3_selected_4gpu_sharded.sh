#!/usr/bin/env bash
set -euo pipefail

# Run Qwen3-AWQ on selected prompt files using 4 one-GPU workers, then merge
# only those selected shards.  This avoids the full-run merge script trying to
# merge every prompt file in llm_prompts.

OUTPUT_DIR="${OUTPUT_DIR:-qwen32_awq_selected}"
LOG_DIR="${LOG_DIR:-logs/qwen3_selected_$(date +%Y%m%d_%H%M%S)}"
IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
PROMPT_DIR="${PROMPT_DIR:-llm_prompts}"
FILES="${FILES:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
LIMIT="${LIMIT:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"

if [[ -z "${FILES}" ]]; then
  echo "FILES must be set, e.g. FILES=behavior_subgoal_decomposition_prompts.json" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

extra_args=(--prompt-dir "${PROMPT_DIR}")
merge_args=(--prompt-dir "${PROMPT_DIR}")

read -r -a selected_files <<< "${FILES}"
extra_args+=(--files "${selected_files[@]}")
merge_args+=(--files "${selected_files[@]}")

if [[ -n "${LIMIT}" ]]; then
  extra_args+=(--limit "${LIMIT}")
  merge_args+=(--limit "${LIMIT}")
fi

extra_args+=("$@")

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
    bash scripts/run_qwen3_vllm_offline.sh \
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
