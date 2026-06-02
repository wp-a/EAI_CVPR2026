#!/usr/bin/env bash
set -euo pipefail

# Single-GPU Qwen3-AWQ offline inference helper used by
# scripts/run_qwen3_selected_4gpu_sharded.sh.

IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
GPU="${GPU:-0}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(pwd)}"
MODEL_HOST="${MODEL_HOST:-${WORKSPACE_HOST}/models/Qwen3-32B-AWQ}"
MODEL_IN="${MODEL_IN:-/workspace/models/Qwen3-32B-AWQ}"
CACHE_HOST="${CACHE_HOST:-${WORKSPACE_HOST}/.cache/vllm}"
PROMPT_DIR="${PROMPT_DIR:-llm_prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen32_awq_selected}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_qwen3_vllm_offline.sh [generator args]

Common examples:
  FILES=virtualhome_subgoal_decomposition_prompts.json bash scripts/run_qwen3_selected_4gpu_sharded.sh
  GPU=0 OUTPUT_DIR=outputs/qwen32_awq_sample bash scripts/run_qwen3_vllm_offline.sh --files virtualhome_action_sequencing_prompts.json --limit 4

Environment overrides:
  IMAGE, GPU, WORKSPACE_HOST, MODEL_HOST, MODEL_IN, CACHE_HOST, PROMPT_DIR,
  OUTPUT_DIR, BATCH_SIZE, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION
EOF
  exit 0
fi

mkdir -p "${CACHE_HOST}"

docker run --rm \
  --gpus "device=${GPU}" \
  --ipc=host \
  --entrypoint python3 \
  -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
  -v "${MODEL_HOST}:${MODEL_IN}:ro" \
  -v "${CACHE_HOST}:/root/.cache/vllm" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  scripts/generate_axis06b_compact_offline.py \
    --model "${MODEL_IN}" \
    --prompt-dir "${PROMPT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    "${@}"
