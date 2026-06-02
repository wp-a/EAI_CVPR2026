#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current VirtualHome Action Sequencing v6.2 result.
#
# Default mode reuses the canonical raw candidates if they already exist:
#   - AxisTilted2 VAS LoRA output
#   - Qwen3-32B-AWQ thinking-mode output
#
# To regenerate candidates, set RUN_INFERENCE=1.  Fresh inference may produce a
# different final sha because vLLM/model kernels can be nondeterministic across
# environments, so exact-sha verification is automatic only for the canonical
# candidate hashes.  The final postprocess uses a frozen v6.2 repair script so
# this remains reproducible even if active ablation scripts keep evolving.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROMPT_FILE="${PROMPT_FILE:-llm_prompts/virtualhome_action_sequencing_prompts.json}"
PROMPT_DIR="$(dirname "${PROMPT_FILE}")"
PROMPT_BASENAME="$(basename "${PROMPT_FILE}")"
OUTPUT_NAME="virtualhome_action_sequencing_outputs.json"

RUN_INFERENCE="${RUN_INFERENCE:-auto}"  # auto, 0, or 1
FORCE_INFERENCE="${FORCE_INFERENCE:-0}"
LIMIT="${LIMIT:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"

EXP_ROOT="${EXP_ROOT:-outputs/vas_v6_2_repro}"
LOG_ROOT="${LOG_ROOT:-logs/vas_v6_2_repro_$(date +%Y%m%d_%H%M%S)}"

AXIS_RAW_DIR="${AXIS_RAW_DIR:-outputs/vas_axistilted2_20260524_093512_raw}"
QWEN_THINK_DIR="${QWEN_THINK_DIR:-outputs/qwen32_awq_think}"
V5_DIR="${V5_DIR:-${EXP_ROOT}/v5_score_warn}"
V6_DIR="${V6_DIR:-${EXP_ROOT}/v6}"
FINAL_DIR="${FINAL_DIR:-${EXP_ROOT}/v6_2}"

AXIS_RAW_FILE="${AXIS_RAW_DIR}/${OUTPUT_NAME}"
QWEN_THINK_FILE="${QWEN_THINK_DIR}/${OUTPUT_NAME}"
V5_FILE="${V5_DIR}/${OUTPUT_NAME}"
V6_FILE="${V6_DIR}/${OUTPUT_NAME}"
FINAL_FILE="${FINAL_DIR}/${OUTPUT_NAME}"

EXPECTED_ROWS="${EXPECTED_ROWS:-${LIMIT:-1500}}"
EXPECTED_SHA="${EXPECTED_SHA:-20aad203fd8befc6bd211bf3a0fd3fab9c5e84463717d0bbdd68978ecec5883a}"
VERIFY_SHA="${VERIFY_SHA:-auto}"  # auto, 0, or 1
V6_FALLBACK_POLICY="${V6_FALLBACK_POLICY:-structural}"

KNOWN_AXIS_SHA="342b789becb20941be1e4b040d31d15550c261372b0b2a020f5b7762e8c4d2eb"
KNOWN_QWEN_THINK_SHA="f12918c60db210bbe8ab85bb7f8fd51886572251baa2ce7fb2f75215d9272fb6"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

json_row_count() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(-1)
else:
    print(len(json.loads(path.read_text(encoding="utf-8"))))
PY
}

sha256_file() {
  python3 - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "")
PY
}

has_expected_rows() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  [[ "$(json_row_count "${path}")" == "${EXPECTED_ROWS}" ]]
}

should_generate() {
  local path="$1"
  [[ "${FORCE_INFERENCE}" == "1" ]] && return 0
  case "${RUN_INFERENCE}" in
    0) return 1 ;;
    1) return 0 ;;
    auto)
      if has_expected_rows "${path}"; then
        return 1
      fi
      return 0
      ;;
    *) die "RUN_INFERENCE must be auto, 0, or 1; got ${RUN_INFERENCE}" ;;
  esac
}

run_axis_inference() {
  echo "== AxisTilted2 inference -> ${AXIS_RAW_DIR}"
  mkdir -p "${LOG_ROOT}/axis"
  FILES="${PROMPT_BASENAME}" \
  OUTPUT_DIR="${AXIS_RAW_DIR}" \
  LOG_DIR="${LOG_ROOT}/axis" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  LIMIT="${LIMIT}" \
    bash scripts/run_axistilted2_qwen3_lora_offline_4gpu_sharded.sh \
      --prompt-dir "${PROMPT_DIR}"
}

run_qwen_think_inference() {
  echo "== Qwen3-32B-AWQ think inference -> ${QWEN_THINK_DIR}"
  mkdir -p "${LOG_ROOT}/qwen32_think"
  FILES="${PROMPT_BASENAME}" \
  PROMPT_DIR="${PROMPT_DIR}" \
  OUTPUT_DIR="${QWEN_THINK_DIR}" \
  LOG_DIR="${LOG_ROOT}/qwen32_think" \
  BATCH_SIZE="${BATCH_SIZE}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  LIMIT="${LIMIT}" \
    bash scripts/run_qwen3_selected_4gpu_sharded.sh --think
}

echo "Repository: ${REPO_ROOT}"
echo "Prompts: ${PROMPT_FILE}"
echo "Experiment root: ${EXP_ROOT}"
echo "RUN_INFERENCE=${RUN_INFERENCE} FORCE_INFERENCE=${FORCE_INFERENCE}"

if should_generate "${AXIS_RAW_FILE}"; then
  run_axis_inference
else
  echo "== Reusing AxisTilted2 candidate: ${AXIS_RAW_FILE}"
fi

if should_generate "${QWEN_THINK_FILE}"; then
  run_qwen_think_inference
else
  echo "== Reusing Qwen think candidate: ${QWEN_THINK_FILE}"
fi

has_expected_rows "${AXIS_RAW_FILE}" || die "Axis candidate row count is not ${EXPECTED_ROWS}: ${AXIS_RAW_FILE}"
has_expected_rows "${QWEN_THINK_FILE}" || die "Qwen think candidate row count is not ${EXPECTED_ROWS}: ${QWEN_THINK_FILE}"

mkdir -p "${V5_DIR}" "${V6_DIR}" "${FINAL_DIR}"

echo "== v5 candidate fusion + structural repair"
optimizer_args=(
  --prompts "${PROMPT_FILE}"
  --candidate "axistilted2=${AXIS_RAW_FILE}"
  --candidate "qwen32think=${QWEN_THINK_FILE}"
  --output "${V5_FILE}"
  --report "${V5_DIR}/validation_report.json"
  --selection-mode score
  --property-policy warn
)
if [[ -n "${LIMIT}" ]]; then
  optimizer_args+=(--limit "${LIMIT}")
fi
python3 scripts/optimize_virtualhome_action_sequencing.py "${optimizer_args[@]}"

echo "== frozen v6.2 evaluator-guided postprocess"
python3 scripts/repair_virtualhome_action_sequencing_v6_2_frozen.py \
  --prompts "${PROMPT_FILE}" \
  --input "${V5_FILE}" \
  --v6-output "${V6_FILE}" \
  --output "${FINAL_FILE}" \
  --report "${FINAL_DIR}/v6_2_repair_report.json" \
  --fallback-policy "${V6_FALLBACK_POLICY}"

echo "== v6.2 manifest + sha verification"
python3 - \
  "${V5_FILE}" \
  "${V6_FILE}" \
  "${FINAL_FILE}" \
  "${FINAL_DIR}/v6_2_repair_report.json" \
  "${FINAL_DIR}/manifest.json" \
  "${PROMPT_FILE}" \
  "${AXIS_RAW_FILE}" \
  "${QWEN_THINK_FILE}" \
  "${EXPECTED_ROWS}" \
  "${EXPECTED_SHA}" \
  "${VERIFY_SHA}" \
  "${KNOWN_AXIS_SHA}" \
  "${KNOWN_QWEN_THINK_SHA}" <<'PY'
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

(
    v5_path,
    v6_path,
    final_path,
    report_path,
    manifest_path,
    prompt_path,
    axis_path,
    qwen_path,
    expected_rows_raw,
    expected_sha,
    verify_sha,
    known_axis_sha,
    known_qwen_sha,
) = [Path(arg) if index < 8 else arg for index, arg in enumerate(sys.argv[1:])]

expected_rows = int(expected_rows_raw)

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def action_count(llm_output: str) -> int:
    return len(re.findall(r'"[^"]+"\s*:', llm_output or ""))

v5_rows = read_json(v5_path)
v6_rows = read_json(v6_path)
if len(v5_rows) != len(v6_rows):
    raise SystemExit(f"v5/v6 row mismatch: {len(v5_rows)} vs {len(v6_rows)}")
if len(v6_rows) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows, got {len(v6_rows)}")

final_rows = read_json(final_path)
if len(final_rows) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} final rows, got {len(final_rows)}")
final_sha = sha256(final_path)
axis_sha = sha256(axis_path)
qwen_sha = sha256(qwen_path)
v5_sha = sha256(v5_path)
v6_sha = sha256(v6_path)

lengths = [action_count(str(row.get("llm_output", ""))) for row in final_rows]
length_stats = {
    "mean": round(statistics.mean(lengths), 4) if lengths else 0,
    "p90": sorted(lengths)[int(0.90 * (len(lengths) - 1))] if lengths else 0,
    "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))] if lengths else 0,
    "p99": sorted(lengths)[int(0.99 * (len(lengths) - 1))] if lengths else 0,
    "max": max(lengths) if lengths else 0,
}

report = read_json(report_path) if report_path.exists() else {}
verification = {
    "base": str(v6_path),
    "fallback": str(v5_path),
    "rows": len(final_rows),
    "fallback_policy": report.get("summary", {}).get("fallback_policy"),
    "fallback_changed": report.get("summary", {}).get("fallback_changed", []),
    "sha256": final_sha,
}
report["verification"] = verification
write_json(report_path, report)

canonical_candidates = axis_sha == str(known_axis_sha) and qwen_sha == str(known_qwen_sha)
verify_final = verify_sha == "1" or (verify_sha == "auto" and canonical_candidates and expected_rows == 1500)
if verify_sha not in {"0", "1", "auto"}:
    raise SystemExit(f"VERIFY_SHA must be auto, 0, or 1; got {verify_sha}")
if verify_final and expected_sha and final_sha != str(expected_sha):
    raise SystemExit(f"Final sha mismatch: expected {expected_sha}, got {final_sha}")

manifest = {
    "recommended_output": str(final_path),
    "rows": len(final_rows),
    "output_sha256": final_sha,
    "expected_sha256": str(expected_sha) if expected_sha else None,
    "sha_verified": bool(verify_final),
    "prompt_file": str(prompt_path),
    "scripts": {
        "repro": "scripts/run_vas_v6_repro.sh",
        "optimizer": "scripts/optimize_virtualhome_action_sequencing.py",
        "repair": "scripts/repair_virtualhome_action_sequencing_v6_2_frozen.py",
    },
    "candidates": {
        "axistilted2": {"path": str(axis_path), "sha256": axis_sha},
        "qwen32think": {"path": str(qwen_path), "sha256": qwen_sha},
    },
    "intermediate_outputs": {
        "v5_score_warn": {"path": str(v5_path), "sha256": v5_sha},
        "v6_before_structural_fallback": {"path": str(v6_path), "sha256": v6_sha},
        "v6_2_report": str(report_path),
    },
    "method": {
        "summary": "AxisTilted2 raw + Qwen32B thinking candidate -> ordered-pair JSON parsing -> score-based candidate fusion -> VirtualHome structural repair -> frozen evaluator-guided v6.2 templates -> structural fallback to v5 for empty or standalone-STANDUP repairs.",
        "selection_mode": "score",
        "property_policy": "warn",
        "fallback_policy": report.get("summary", {}).get("fallback_policy"),
        "v6_templates": [
            "watch_tv_template",
            "drink_template",
            "food_inside_container_template",
            "light_template",
            "read_template",
            "phone_template",
            "prefix_goal_trim",
        ],
    },
    "length_stats": length_stats,
    "official_scene_1_dev_eval_reference": {
        "task_success_rate": 76.3934,
        "execution_success_rate": 95.7,
        "total_goal": 84.9835,
        "parsing": 0.0,
        "hallucination": 1.3115,
    },
}
write_json(manifest_path, manifest)

print(json.dumps({
    "final": str(final_path),
    "rows": len(final_rows),
    "sha256": final_sha,
    "sha_verified": verify_final,
    "fallback_changed": verification["fallback_changed"],
}, ensure_ascii=False, indent=2))
PY

if [[ -n "${SUBMISSION_DIR:-}" ]]; then
  mkdir -p "${SUBMISSION_DIR}"
  cp "${FINAL_FILE}" "${SUBMISSION_DIR}/${OUTPUT_NAME}"
  cp "${FINAL_DIR}/manifest.json" "${SUBMISSION_DIR}/manifest.json"
  cp "${FINAL_DIR}/v6_2_repair_report.json" "${SUBMISSION_DIR}/v6_2_repair_report.json"
  echo "== Copied final output to ${SUBMISSION_DIR}"
fi

echo "== Done"
echo "Final output: ${FINAL_FILE}"
echo "Manifest: ${FINAL_DIR}/manifest.json"
