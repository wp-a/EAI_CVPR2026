#!/usr/bin/env bash
set -euo pipefail

# Reproduce the best BAS/BSD-only C02 compact run.
# This script does not run BGI or BTM.

RUN_ID="${RUN_ID:-axis06b_bas_bsd_c02_repro_$(date +%Y%m%d_%H%M%S)}"
IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE_HOST}/models/AxisTilted2}"
CACHE_HOST="${CACHE_HOST:-${WORKSPACE_HOST}/.cache/vllm_axis06b}"
PROMPT_DIR="${PROMPT_DIR:-llm_prompts_axis06b_behavior_compact}"
OUT_ROOT="${OUT_ROOT:-outputs/${RUN_ID}}"
RAW_ROOT="${RAW_ROOT:-outputs/${RUN_ID}_raw}"
REPORT_ROOT="${REPORT_ROOT:-${OUT_ROOT}/_reports}"
CANDIDATE="${CANDIDATE:-C02_compact_t02}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-202}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUT_ROOT}" "${RAW_ROOT}" "${REPORT_ROOT}" "${CACHE_HOST}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

final_output_dir="${OUT_ROOT}/${CANDIDATE}"
raw_output_dir="${RAW_ROOT}/${CANDIDATE}"
log_dir="${LOG_ROOT:-${WORKSPACE_HOST}/logs/${RUN_ID}/${CANDIDATE}}"
mkdir -p "${final_output_dir}" "${raw_output_dir}" "${log_dir}"

echo "RUN_ID=${RUN_ID}"
echo "CANDIDATE=${CANDIDATE}"
echo "PROMPT_DIR=${PROMPT_DIR}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "RAW_ROOT=${RAW_ROOT}"
echo "temperature=${TEMPERATURE} top_p=${TOP_P} top_k=${TOP_K} seed=${SEED}"
echo "START=$(date)"

run_sharded_generation() {
  local task_name="$1"
  local model_mount="$2"
  local model_path="$3"
  local prompt_file="$4"
  local extra_arg="${5:-}"

  local pids=()
  local failed=0
  for gpu in 0 1 2 3; do
    (
      docker run --rm \
        --gpus "device=${gpu}" \
        --ipc=host \
        --entrypoint python3 \
        -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
        -v "${model_mount}:${model_path}:ro" \
        -v "${CACHE_HOST}:/root/.cache/vllm" \
        -w /workspace/eai-starter-kit \
        "${IMAGE}" \
        scripts/generate_axis06b_compact_offline.py \
        --model "${model_path}" \
        --prompt-dir "${PROMPT_DIR}" \
        --output-dir "${raw_output_dir}" \
        --batch-size "${BATCH_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        ${extra_arg} \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --top-k "${TOP_K}" \
        --seed "${SEED}" \
        --num-shards 4 \
        --shard-index "${gpu}" \
        --shard-output \
        --files "${prompt_file}" \
        --no-resume \
        "${limit_args[@]}"
    ) >"${log_dir}/${task_name}_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "FAILED ${task_name}; see ${log_dir}/${task_name}_gpu*.log" >&2
    return 1
  fi

  docker run --rm \
    --entrypoint python3 \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -w /workspace/eai-starter-kit \
    "${IMAGE}" \
    scripts/merge_vllm_shards_selected.py \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${raw_output_dir}" \
    --num-shards 4 \
    --files "${prompt_file}" \
    "${limit_args[@]}" \
    >"${log_dir}/${task_name}_merge.log" 2>&1
}

echo "[$(date)] start BSD"
run_sharded_generation \
  bsd \
  "${MODEL_ROOT}/b_sd_Qwen-0.6B-FINALSUB" \
  /workspace/model_sd \
  behavior_subgoal_decomposition_prompts.json
echo "[$(date)] done BSD"

echo "[$(date)] start BAS"
run_sharded_generation \
  bas \
  "${MODEL_ROOT}/b_as_2_Qwen-0.6B-FINALSUB" \
  /workspace/model_as \
  behavior_action_sequencing_prompts.json \
  --enforce-eager
echo "[$(date)] done BAS"

docker run --rm \
  --entrypoint python3 \
  -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  scripts/postprocess_axis06b_behavior_outputs.py \
  --raw-output-dir "${raw_output_dir}" \
  --official-prompt-dir "${PROMPT_DIR}" \
  --output-dir "${final_output_dir}" \
  --tasks sd as \
  "${limit_args[@]}" \
  >"${log_dir}/postprocess.log" 2>&1

docker run --rm \
  --entrypoint bash \
  -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  -lc "mkdir -p '${REPORT_ROOT}' && if [[ -f '${final_output_dir}/validation_report.json' ]]; then mv '${final_output_dir}/validation_report.json' '${REPORT_ROOT}/${CANDIDATE}_validation_report.json'; fi && chmod -R a+rwX '${final_output_dir}' '${REPORT_ROOT}' '${raw_output_dir}'"

file_count="$(find "${final_output_dir}" -maxdepth 1 -type f | wc -l | tr -d ' ')"
if [[ "${file_count}" != "2" ]]; then
  echo "Unexpected file count in ${final_output_dir}: ${file_count}" >&2
  exit 1
fi

echo "[$(date)] all done"
echo "final_output_dir=${final_output_dir}"
echo "raw_output_dir=${raw_output_dir}"
echo "report=${REPORT_ROOT}/${CANDIDATE}_validation_report.json"
