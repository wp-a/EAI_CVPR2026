#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_ID="${RUN_ID:-repro8_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_ID}}"
FINAL_DIR="${FINAL_DIR:-sample_submission_repro_${RUN_ID}}"
D02_CANDIDATE="${D02_CANDIDATE:-D02_complete_7_short_direct_tasks}"
BAS_OUT_ROOT="${BAS_OUT_ROOT:-${RUN_ROOT}/behavior_bas_bsd}"
BAS_RAW_ROOT="${BAS_RAW_ROOT:-${RUN_ROOT}/behavior_bas_bsd_raw}"
BAS_D02_ROOT="${BAS_D02_ROOT:-${RUN_ROOT}/behavior_bas_bsd_d02}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${FINAL_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_ROOT}/run_all.log"
}

run_cmd() {
  local name="$1"
  shift
  log "START ${name}"
  (
    set -x
    "$@"
  ) > >(tee "${LOG_ROOT}/${name}.stdout.log") 2>&1
  log "DONE ${name}"
}

copy_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Missing expected output: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dst}"
}

log "repo=${REPO_ROOT}"
log "run_id=${RUN_ID}"
log "run_root=${RUN_ROOT}"
log "final_dir=${FINAL_DIR}"

run_cmd behavior_bgi_btm \
  env \
    RUN_ID="${RUN_ID}/behavior_bgi_btm" \
    LOG_ROOT="${LOG_ROOT}/behavior_bgi_btm" \
    RAW_OUTPUT_DIR="${RUN_ROOT}/behavior_bgi_btm_raw" \
    FINAL_OUTPUT_DIR="${RUN_ROOT}/behavior_bgi_btm" \
    REPORT_ROOT="${RUN_ROOT}/behavior_bgi_btm/_reports" \
    TASKS="bgi btm" \
    bash scripts/run_axis06b_behavior_bgi_btm_repro.sh

BAS_RUN_ID="${RUN_ID}/behavior_bas_bsd"
run_cmd behavior_bas_bsd \
  env \
    RUN_ID="${BAS_RUN_ID}" \
    OUT_ROOT="${BAS_OUT_ROOT}" \
    RAW_ROOT="${BAS_RAW_ROOT}" \
    D02_ROOT="${BAS_D02_ROOT}" \
    LOG_ROOT="${LOG_ROOT}/behavior_bas_bsd" \
    D02_CANDIDATE="${D02_CANDIDATE}" \
    bash scripts/run_axis06b_bas_bsd_c02_d02_repro.sh

run_cmd virtualhome_action_sequencing \
  env \
    RUN_INFERENCE=1 \
    VERIFY_SHA=0 \
    EXP_ROOT="${RUN_ROOT}/vas" \
    LOG_ROOT="${LOG_ROOT}/vas" \
    AXIS_RAW_DIR="${RUN_ROOT}/vas_raw_axistilted2" \
    QWEN_THINK_DIR="${RUN_ROOT}/vas_raw_qwen32_think" \
    bash scripts/run_vas_v6_repro.sh

run_cmd virtualhome_goal_interpretation \
  env \
    WORKSPACE="${REPO_ROOT}" \
    ADAPTER="vgi_sft_work/qwen06b_vgi97/lora/adapter.pt" \
    GEN="${RUN_ROOT}/vgi/generation" \
    OUT="${RUN_ROOT}/vgi/virtualhome_goal_interpretation_outputs.json" \
    REPORT="${RUN_ROOT}/vgi/report.json" \
    bash scripts/reproduce_vgi_qwen06b_sft97_vote.sh

run_cmd virtualhome_subgoal_decomposition \
  env \
    ROOT="${REPO_ROOT}" \
    EXP_ROOT="${RUN_ROOT}/vsd" \
    LOG_ROOT="${LOG_ROOT}/vsd" \
    bash scripts/reproduce_vsd_baseline_restore_grab.sh

run_cmd virtualhome_transition_modeling \
  env \
    GPU="${VTM_GPU:-0}" \
    RAW_OUTPUT_DIR="${RUN_ROOT}/vtm/raw" \
    OUTPUT_DIR="${RUN_ROOT}/vtm" \
    bash scripts/reproduce_qwen06b_vtm_official_domain_infer.sh

rm -rf "${FINAL_DIR}"
mkdir -p "${FINAL_DIR}"

copy_required "${RUN_ROOT}/behavior_bgi_btm/behavior_goal_interpretation_outputs.json" \
  "${FINAL_DIR}/behavior_goal_interpretation_outputs.json"
copy_required "${RUN_ROOT}/behavior_bgi_btm/behavior_transition_modeling_outputs.json" \
  "${FINAL_DIR}/behavior_transition_modeling_outputs.json"
copy_required "${BAS_D02_ROOT}/${D02_CANDIDATE}/behavior_action_sequencing_outputs.json" \
  "${FINAL_DIR}/behavior_action_sequencing_outputs.json"
copy_required "${BAS_D02_ROOT}/${D02_CANDIDATE}/behavior_subgoal_decomposition_outputs.json" \
  "${FINAL_DIR}/behavior_subgoal_decomposition_outputs.json"
copy_required "${RUN_ROOT}/vas/v6_2/virtualhome_action_sequencing_outputs.json" \
  "${FINAL_DIR}/virtualhome_action_sequencing_outputs.json"
copy_required "${RUN_ROOT}/vgi/virtualhome_goal_interpretation_outputs.json" \
  "${FINAL_DIR}/virtualhome_goal_interpretation_outputs.json"
copy_required "${RUN_ROOT}/vsd/baseline_restore_grab/virtualhome_subgoal_decomposition_outputs.json" \
  "${FINAL_DIR}/virtualhome_subgoal_decomposition_outputs.json"
copy_required "${RUN_ROOT}/vtm/virtualhome_transition_modeling_outputs.json" \
  "${FINAL_DIR}/virtualhome_transition_modeling_outputs.json"

python3 - "${FINAL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

final_dir = Path(sys.argv[1])
expected = {
    "behavior_action_sequencing_outputs.json": 100,
    "behavior_goal_interpretation_outputs.json": 100,
    "behavior_subgoal_decomposition_outputs.json": 100,
    "behavior_transition_modeling_outputs.json": 100,
    "virtualhome_action_sequencing_outputs.json": 1500,
    "virtualhome_goal_interpretation_outputs.json": 1500,
    "virtualhome_subgoal_decomposition_outputs.json": 1500,
    "virtualhome_transition_modeling_outputs.json": 1500,
}

seen = sorted(path.name for path in final_dir.glob("*.json"))
if seen != sorted(expected):
    raise SystemExit(f"Unexpected final files: {seen}")

summary = {}
for name, count in expected.items():
    rows = json.loads((final_dir / name).read_text(encoding="utf-8"))
    if len(rows) != count:
        raise SystemExit(f"{name}: expected {count} rows, got {len(rows)}")
    bad_schema = [
        index
        for index, row in enumerate(rows)
        if sorted(row.keys()) != ["identifier", "llm_output"]
    ]
    if bad_schema:
        raise SystemExit(f"{name}: bad schema rows begin at {bad_schema[:5]}")
    summary[name] = count

report_path = final_dir.parent / f"{final_dir.name}_validation_summary.json"
report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"final_dir": str(final_dir), "files": summary}, ensure_ascii=False, indent=2))
PY

(
  cd "${FINAL_DIR}"
  zip -q -r "../${FINAL_DIR}.zip" .
)

log "FINAL ${FINAL_DIR}"
log "ZIP ${FINAL_DIR}.zip"
