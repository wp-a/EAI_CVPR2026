#!/usr/bin/env bash
set -euo pipefail

# Reproduce the best VirtualHome SD candidate:
#   Qwen3-32B-AWQ no-think raw inference
#   -> conservative VSD postprocess
#   -> restore only GRAB(...) intermediate actions removed by postprocess.
#
# Intended to run on the remote server from the eai-starter-kit repo.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-outputs/vsd_baseline_restore_grab_repro_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/vsd_baseline_restore_grab_repro_${STAMP}}"

PROMPTS="${PROMPTS:-llm_prompts/virtualhome_subgoal_decomposition_prompts.json}"
PROMPT_FILE="virtualhome_subgoal_decomposition_prompts.json"

RAW_DIR="${EXP_ROOT}/raw_qwen32_no_think"
POST_DIR="${EXP_ROOT}/baseline_postprocessed"
REPORT_DIR="${EXP_ROOT}/reports"
ABLATION_DIR="${EXP_ROOT}/ablation"
FINAL_DIR="${EXP_ROOT}/baseline_restore_grab"

mkdir -p "${RAW_DIR}" "${POST_DIR}" "${REPORT_DIR}" "${ABLATION_DIR}" "${FINAL_DIR}" "${LOG_ROOT}"

echo "[$(date)] root=${ROOT}"
echo "[$(date)] exp_root=${EXP_ROOT}"
echo "[$(date)] log_root=${LOG_ROOT}"

echo "[$(date)] START raw Qwen32B no-think inference"
FILES="${PROMPT_FILE}" \
OUTPUT_DIR="${RAW_DIR}" \
LOG_DIR="${LOG_ROOT}/inference" \
BATCH_SIZE=4 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.75 \
bash scripts/run_qwen3_selected_4gpu_sharded.sh \
  --temperature 0.0 \
  --top-p 0.95 \
  --no-resume
echo "[$(date)] DONE raw Qwen32B no-think inference"

echo "[$(date)] START conservative postprocess"
python3 scripts/postprocess_vsd_outputs.py \
  --prompts "${PROMPTS}" \
  --input "${RAW_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --output "${POST_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --report "${REPORT_DIR}/baseline_postprocess_report.json"
echo "[$(date)] DONE conservative postprocess"

echo "[$(date)] START restore-grab ablation"
python3 scripts/build_vsd_ablation_candidates.py \
  --prompts "${PROMPTS}" \
  --original "${RAW_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --baseline "${POST_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --report "${REPORT_DIR}/baseline_postprocess_report.json" \
  --output-dir "${ABLATION_DIR}"

cp "${ABLATION_DIR}/baseline_restore_grab/virtualhome_subgoal_decomposition_outputs.json" \
  "${FINAL_DIR}/virtualhome_subgoal_decomposition_outputs.json"
echo "[$(date)] DONE restore-grab ablation"

echo "[$(date)] START validation"
python3 scripts/validate_vsd_control_outputs.py \
  --prompts "${PROMPTS}" \
  --output "${REPORT_DIR}/validation_report.json" \
  --file "raw=${RAW_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --file "baseline_postprocessed=${POST_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --file "baseline_restore_grab=${FINAL_DIR}/virtualhome_subgoal_decomposition_outputs.json" \
  --postprocess-report "baseline=${REPORT_DIR}/baseline_postprocess_report.json" \
  --ablation-manifest "baseline=${ABLATION_DIR}/manifest.json" \
  > "${REPORT_DIR}/validation_report.stdout.json"
echo "[$(date)] DONE validation"

python3 - <<PY
import hashlib
import json
from pathlib import Path

final_path = Path("${FINAL_DIR}/virtualhome_subgoal_decomposition_outputs.json")
rows = json.loads(final_path.read_text(encoding="utf-8"))
parse_ok = 0
think_tags = 0
for row in rows:
    raw = row.get("llm_output", "")
    think_tags += int("<think>" in raw.lower() or "</think>" in raw.lower())
    try:
        obj = json.loads(raw)
        parse_ok += int(isinstance(obj, dict) and isinstance(obj.get("output"), list))
    except Exception:
        pass
print(json.dumps({
    "final_path": str(final_path),
    "rows": len(rows),
    "unique_identifiers": len({row.get("identifier") for row in rows}),
    "parse_ok": parse_ok,
    "think_tag_rows": think_tags,
    "md5": hashlib.md5(final_path.read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2))
PY

echo "[$(date)] FINAL ${FINAL_DIR}/virtualhome_subgoal_decomposition_outputs.json"
