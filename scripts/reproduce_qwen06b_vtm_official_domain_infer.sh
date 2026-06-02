#!/usr/bin/env bash
set -euo pipefail

# Inference-only VTM reproduction.
# Uses the bundled Qwen3-0.6B model plus a pre-trained official-domain VTM LoRA.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

USE_DOCKER="${USE_DOCKER:-1}"
IMAGE="${IMAGE:-eai-vllm-qwen3:v0.11.1}"
GPU="${GPU:-0}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$ROOT_DIR}"
if [[ -z "${MODELS_HOST:-}" && -d "$ROOT_DIR/models" ]]; then
  MODELS_HOST="$ROOT_DIR/models"
fi
MODELS_HOST="${MODELS_HOST:-/public/localUsers/lixiang/EAI_CVPR2026/models}"

if [[ "$USE_DOCKER" == "1" && "${IN_DOCKER:-0}" != "1" ]]; then
  docker run --rm \
    --gpus "device=${GPU}" \
    --ipc=host \
    --entrypoint bash \
    -e IN_DOCKER=1 \
    -e USE_DOCKER=0 \
    -e PYTHON=python3 \
    -e PROMPTS="${PROMPTS:-llm_prompts/virtualhome_transition_modeling_prompts.json}" \
    -e BASE_MODEL="${BASE_MODEL:-/workspace/models/Qwen3-0.6B}" \
    -e ARTIFACT_DIR="${ARTIFACT_DIR:-vtm_infer_artifacts/qwen06b_vtm_official_domain_fast}" \
    -e ADAPTER="${ADAPTER:-}" \
    -e ADAPTER_METADATA="${ADAPTER_METADATA:-}" \
    -e DOMAIN_PDDL="${DOMAIN_PDDL:-}" \
    -e RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer_raw}" \
    -e OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer}" \
    -e COMPARE_OUTPUT="${COMPARE_OUTPUT:-}" \
    -e SEED="${SEED:-20260525}" \
    -e CANDIDATES="${CANDIDATES:-5}" \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -v "${MODELS_HOST}:/workspace/models:ro" \
    -w /workspace/eai-starter-kit \
    "$IMAGE" \
    scripts/reproduce_qwen06b_vtm_official_domain_infer.sh "$@"
  status=$?
  docker run --rm \
    --entrypoint bash \
    -v "${WORKSPACE_HOST}:/workspace/eai-starter-kit" \
    -w /workspace/eai-starter-kit \
    "$IMAGE" \
    -lc "chmod -R a+rwX '${RAW_OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer_raw}' '${OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer}'" \
    || true
  exit $status
fi

cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"
PROMPTS="${PROMPTS:-llm_prompts/virtualhome_transition_modeling_prompts.json}"
BASE_MODEL="${BASE_MODEL:-models/Qwen3-0.6B}"
ARTIFACT_DIR="${ARTIFACT_DIR:-vtm_infer_artifacts/qwen06b_vtm_official_domain_fast}"
ADAPTER="${ADAPTER:-${ARTIFACT_DIR}/lora_30/adapter_step200.pt}"
ADAPTER_METADATA="${ADAPTER_METADATA:-${ARTIFACT_DIR}/lora_30/metadata.json}"
SCHEMA_LIBRARY="${SCHEMA_LIBRARY:-${ARTIFACT_DIR}/schema_library.json}"
REFERENCE_OUTPUT="${REFERENCE_OUTPUT:-${ARTIFACT_DIR}/official_domain_reference_outputs.json}"
DOMAIN_PDDL="${DOMAIN_PDDL:-${ARTIFACT_DIR}/virtualhome.pddl}"
RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen06b_vtm_official_domain_infer}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-${REFERENCE_OUTPUT}}"
SEED="${SEED:-20260525}"
CANDIDATES="${CANDIDATES:-5}"

echo "== Qwen0.6B VTM official-domain inference-only reproduction =="
echo "root=$ROOT_DIR"
echo "base_model=$BASE_MODEL"
echo "artifact_dir=$ARTIFACT_DIR"
echo "adapter=$ADAPTER"
echo "raw_output_dir=$RAW_OUTPUT_DIR"
echo "output_dir=$OUTPUT_DIR"

for path in "$PROMPTS" "$BASE_MODEL" "$ADAPTER" "$SCHEMA_LIBRARY" "$REFERENCE_OUTPUT" "$DOMAIN_PDDL"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required VTM inference artifact: $path" >&2
    exit 1
  fi
done

echo
echo "== 1/3 Generate model schema library and raw VTM outputs =="
"$PYTHON" scripts/generate_qwen06b_vtm_schema_lora.py \
  --model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --adapter-metadata "$ADAPTER_METADATA" \
  --schema-library "$SCHEMA_LIBRARY" \
  --prompts "$PROMPTS" \
  --reference-output "$REFERENCE_OUTPUT" \
  --output-dir "$RAW_OUTPUT_DIR" \
  --candidates "$CANDIDATES" \
  --seed "$SEED"

mkdir -p "$OUTPUT_DIR"

echo
echo "== 2/3 Whitespace-only PDDL postprocess =="
"$PYTHON" scripts/postprocess_vtm_pddl_whitespace.py \
  --input "$RAW_OUTPUT_DIR/virtualhome_transition_modeling_outputs.json" \
  --output "$OUTPUT_DIR/virtualhome_transition_modeling_outputs.json" \
  --report "$OUTPUT_DIR/postprocess_report.json" \
  --prompts "$PROMPTS" \
  --compare-output "$COMPARE_OUTPUT"

cp "$RAW_OUTPUT_DIR/virtualhome_transition_modeling_outputs.json" \
  "$OUTPUT_DIR/virtualhome_transition_modeling_outputs.raw_model.json"
cp "$RAW_OUTPUT_DIR/generated_schema_library.json" "$OUTPUT_DIR/generated_schema_library.json"
cp "$RAW_OUTPUT_DIR/report.json" "$OUTPUT_DIR/generation_report.json"

echo
echo "== 3/3 Write manifest =="
"$PYTHON" - "$DOMAIN_PDDL" "$RAW_OUTPUT_DIR/report.json" "$OUTPUT_DIR/postprocess_report.json" "$ADAPTER" "$OUTPUT_DIR/manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

domain_pddl = Path(sys.argv[1])
generation_report = json.load(open(sys.argv[2], encoding="utf-8"))
postprocess_report = json.load(open(sys.argv[3], encoding="utf-8"))
manifest = {
    "method": "Qwen3-0.6B official-domain VTM LoRA inference only",
    "training": "not run in this reproduction; uses bundled pre-trained adapter",
    "domain_pddl": str(domain_pddl),
    "domain_sha256": hashlib.sha256(domain_pddl.read_bytes()).hexdigest(),
    "model": generation_report.get("model"),
    "adapter": sys.argv[4],
    "candidates_per_action": generation_report.get("candidates_per_action"),
    "generation_format_checks": generation_report.get("format_checks"),
    "official_domain_alignment": generation_report.get("official_domain_alignment"),
    "postprocess": postprocess_report,
    "final_output": str(Path(sys.argv[5]).with_name("virtualhome_transition_modeling_outputs.json")),
    "raw_model_output": str(Path(sys.argv[5]).with_name("virtualhome_transition_modeling_outputs.raw_model.json")),
}
Path(sys.argv[5]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

echo
echo "== Done =="
echo "final_output=$OUTPUT_DIR/virtualhome_transition_modeling_outputs.json"
echo "raw_model_output=$OUTPUT_DIR/virtualhome_transition_modeling_outputs.raw_model.json"
echo "manifest=$OUTPUT_DIR/manifest.json"
