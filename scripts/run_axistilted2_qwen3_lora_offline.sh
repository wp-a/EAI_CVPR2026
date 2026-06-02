#!/usr/bin/env bash
set -euo pipefail

# Offline vLLM generation with the AxisTilted2 EAI Qwen3-32B LoRA.
# Defaults are aimed at VirtualHome Action Sequencing, the module this LoRA
# was trained for according to its train_meta.json.

IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
GPU="${GPU:-0}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(pwd)}"
MODEL_HOST="${MODEL_HOST:-${WORKSPACE_HOST}/models/Qwen3-32B-AWQ}"
LORA_HOST="${LORA_HOST:-${WORKSPACE_HOST}/models/AxisTilted2/qwen3-32b-domain2ep-vas20ep}"
CACHE_HOST="${CACHE_HOST:-${WORKSPACE_HOST}/.cache/vllm}"
OUTPUT_DIR="${OUTPUT_DIR:-sample_submission_axistilted2_qwen3_32b_lora_vas}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/run_axistilted2_qwen3_lora_offline.sh [generator args]

Common examples:
  bash scripts/run_axistilted2_qwen3_lora_offline.sh --files virtualhome_action_sequencing_prompts.json --limit 4

Environment overrides:
  GPU=0 OUTPUT_DIR=sample_submission_axistilted2_vas BATCH_SIZE=4 MAX_MODEL_LEN=8192
EOF
  exit 0
fi

mkdir -p "${CACHE_HOST}"

docker run --rm \
  --gpus "device=${GPU}" \
  --ipc=host \
  --entrypoint python3 \
  -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
  -v "${MODEL_HOST}:/workspace/models/Qwen3-32B-AWQ:ro" \
  -v "${LORA_HOST}:/workspace/models/AxisTilted2/qwen3-32b-domain2ep-vas20ep:ro" \
  -v "${CACHE_HOST}:/root/.cache/vllm" \
  -w /workspace/eai-starter-kit \
  "${IMAGE}" \
  scripts/generate_with_vllm_offline_lora.py \
  --model /workspace/models/Qwen3-32B-AWQ \
  --lora-path /workspace/models/AxisTilted2/qwen3-32b-domain2ep-vas20ep \
  --lora-name axistilted2-qwen3-32b-domain2ep-vas20ep \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  "${@}"
