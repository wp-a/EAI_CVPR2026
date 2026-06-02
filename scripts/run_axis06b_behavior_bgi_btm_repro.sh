#!/usr/bin/env bash
set -euo pipefail

# Reproduce the near-ceiling BEHAVIOR BGI/BTM 0.6B pipeline.
# BGI: reconstructed compact prompt + b_gi_Qwen-0.6B-FINALSUB.
# BTM: official prompt + b_compiled_model_Qwen-0.6B-TM-2-FINALSUB + JSON/PDDL cleanup.

IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_HOST}/models/AxisTilted2}"
CACHE_HOST="${CACHE_HOST:-/home/user/.cache/vllm_axis06b}"
RUN_ID="${RUN_ID:-axis06b_bgi_btm_repro_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-/home/user/eai_runs/${RUN_ID}}"

BGI_PROMPT_DIR="${BGI_PROMPT_DIR:-llm_prompts_axis06b_behavior_compact}"
BTM_PROMPT_DIR="${BTM_PROMPT_DIR:-llm_prompts}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-outputs/${RUN_ID}_raw}"
FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR:-outputs/${RUN_ID}}"
REPORT_ROOT="${REPORT_ROOT:-${FINAL_OUTPUT_DIR}/_reports}"

NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-0}"
LIMIT="${LIMIT:-}"
TASKS="${TASKS:-bgi btm}"

mkdir -p "${LOG_ROOT}" "${CACHE_HOST}" "${RAW_OUTPUT_DIR}" "${FINAL_OUTPUT_DIR}" "${REPORT_ROOT}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

echo "RUN_ID=${RUN_ID}"
echo "WORKSPACE_HOST=${WORKSPACE_HOST}"
echo "RAW_OUTPUT_DIR=${RAW_OUTPUT_DIR}"
echo "FINAL_OUTPUT_DIR=${FINAL_OUTPUT_DIR}"
echo "TASKS=${TASKS}"
echo "TEMPERATURE=${TEMPERATURE} TOP_P=${TOP_P} TOP_K=${TOP_K} SEED=${SEED}"
echo "START=$(date)"

want_task() {
  local task="$1"
  [[ " ${TASKS} " == *" ${task} "* ]]
}

run_sharded_generation() {
  local key="$1"
  local model_name="$2"
  local prompt_dir="$3"
  local prompt_file="$4"
  local max_tokens="$5"
  local extra_flags="${6:-}"
  local task_log="${LOG_ROOT}/${key}"
  mkdir -p "${task_log}"

  echo "[$(date)] start ${key} model=${model_name} prompt_dir=${prompt_dir} file=${prompt_file}"

  local failed=0
  local pids=()
  for gpu in $(seq 0 $((NUM_SHARDS - 1))); do
    (
      # shellcheck disable=SC2086
      docker run --rm \
        --gpus "device=${gpu}" \
        --ipc=host \
        --entrypoint python3 \
        -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
        -v "${MODEL_ROOT}/${model_name}:/workspace/model:ro" \
        -v "${CACHE_HOST}:/root/.cache/vllm" \
        -w /workspace/eai-starter-kit \
        "${IMAGE}" \
        scripts/generate_axis06b_compact_offline.py \
        --model /workspace/model \
        --prompt-dir "${prompt_dir}" \
        --output-dir "${RAW_OUTPUT_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --seed "${SEED}" \
        --max-tokens "${max_tokens}" \
        --num-shards "${NUM_SHARDS}" \
        --shard-index "${gpu}" \
        --shard-output \
        --files "${prompt_file}" \
        --no-resume \
        ${extra_flags} \
        "${limit_args[@]}"
    ) >"${task_log}/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"${task_log}/gpu${gpu}.pid"
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date)] FAILED ${key}; see ${task_log}" >&2
    return 1
  fi

  docker run --rm \
    --entrypoint python3 \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -w /workspace/eai-starter-kit \
    "${IMAGE}" \
    scripts/merge_vllm_shards_selected.py \
    --prompt-dir "${prompt_dir}" \
    --output-dir "${RAW_OUTPUT_DIR}" \
    --num-shards "${NUM_SHARDS}" \
    --files "${prompt_file}" \
    "${limit_args[@]}" \
    >"${task_log}/merge.log" 2>&1

  echo "[$(date)] done ${key}"
}

if want_task bgi; then
  run_sharded_generation \
    bgi \
    b_gi_Qwen-0.6B-FINALSUB \
    "${BGI_PROMPT_DIR}" \
    behavior_goal_interpretation_prompts.json \
    1024

  docker run --rm \
    --entrypoint python3 \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -w /workspace/eai-starter-kit \
    "${IMAGE}" \
    scripts/postprocess_axis06b_behavior_outputs.py \
    --raw-output-dir "${RAW_OUTPUT_DIR}" \
    --official-prompt-dir "${BGI_PROMPT_DIR}" \
    --output-dir "${FINAL_OUTPUT_DIR}" \
    --tasks gi \
    "${limit_args[@]}" \
    >"${LOG_ROOT}/bgi_postprocess.log" 2>&1

  if [[ -f "${FINAL_OUTPUT_DIR}/validation_report.json" ]]; then
    mv "${FINAL_OUTPUT_DIR}/validation_report.json" "${REPORT_ROOT}/bgi_validation_report.json"
  fi
fi

if want_task btm; then
  run_sharded_generation \
    btm \
    b_compiled_model_Qwen-0.6B-TM-2-FINALSUB \
    "${BTM_PROMPT_DIR}" \
    behavior_transition_modeling_prompts.json \
    4096 \
    "--enforce-eager"

  docker run --rm \
    --entrypoint python3 \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -w /workspace/eai-starter-kit \
    "${IMAGE}" \
    scripts/postprocess_axis06b_behavior_tm_outputs.py \
    --raw-output-dir "${RAW_OUTPUT_DIR}" \
    --prompt-dir "${BTM_PROMPT_DIR}" \
    --output-dir "${FINAL_OUTPUT_DIR}" \
    --report-path "${REPORT_ROOT}/btm_validation_report.json" \
    "${limit_args[@]}" \
    >"${LOG_ROOT}/btm_postprocess.log" 2>&1
fi

docker run --rm \
  --entrypoint bash \
  -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  -lc "chmod -R a+rwX '${RAW_OUTPUT_DIR}' '${FINAL_OUTPUT_DIR}' '${REPORT_ROOT}'"

echo "[$(date)] all done final=${FINAL_OUTPUT_DIR}"
find "${FINAL_OUTPUT_DIR}" -maxdepth 2 -type f | sort
