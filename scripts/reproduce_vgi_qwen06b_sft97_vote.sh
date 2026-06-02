#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${ROOT:-vgi_sft_work/qwen06b_vgi97}"
WORKSPACE="${WORKSPACE:-${REPO_ROOT}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-eai-vllm-qwen3:v0.11.1}"
MODEL="${MODEL:-${REPO_ROOT}/models/Qwen3-0.6B}"
MODEL_IN="${MODEL_IN:-/workspace/models/Qwen3-0.6B}"
HF_CACHE="${HF_CACHE:-${WORKSPACE}/.cache/huggingface}"
ADAPTER="${ADAPTER:-${ROOT}/lora/adapter.pt}"
PROMPT_FILE="${PROMPT_FILE:-llm_prompts/virtualhome_goal_interpretation_prompts.json}"
GEN="${GEN:-${ROOT}/generation}"
OUT="${OUT:-outputs/best_vgi_qwen06b_sft97_vote_virtualhome_goal_interpretation_outputs.json}"
REPORT="${REPORT:-${ROOT}/best_vgi_qwen06b_sft97_vote_report.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-12288}"
SEED="${SEED:-20260523}"

cd "${WORKSPACE}"

test -f "scripts/generate_qwen06b_vgi_lora.py"
test -f "scripts/make_vgi_qwen06b_sft_vote.py"
test -f "${PROMPT_FILE}"
test -f "${ADAPTER}"

rm -rf "${GEN}"
rm -f "${OUT}" "${REPORT}" "${ROOT}/GEN_DONE"
mkdir -p "${GEN}/greedy" "${GEN}/t02" "${GEN}/t06" "$(dirname "${OUT}")" "$(dirname "${REPORT}")"

echo "[vgi-qwen06b-sft97] reproduce start $(date)"
echo "[vgi-qwen06b-sft97] output: ${OUT}"

run_one() {
  local gpu="$1"
  local name="$2"
  local temp="$3"
  local top_p="$4"
  local seed_offset="$5"
  docker run --rm --gpus "device=${gpu}" --ipc=host --entrypoint python3 \
    -v "${WORKSPACE}:/workspace/eai-starter-kit" \
    -v "${MODEL}:${MODEL_IN}:ro" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -w /workspace/eai-starter-kit \
    "${DOCKER_IMAGE}" \
    scripts/generate_qwen06b_vgi_lora.py \
      --model "${MODEL_IN}" \
      --adapter "${ADAPTER}" \
      --prompt-file "${PROMPT_FILE}" \
      --output "${GEN}/${name}/virtualhome_goal_interpretation_outputs.json" \
      --temperature "${temp}" \
      --top-p "${top_p}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-input-tokens "${MAX_INPUT_TOKENS}" \
      --seed "$((SEED + seed_offset))"
}

run_one 0 greedy 0.0 0.9 0 > "${GEN}/greedy.log" 2>&1 & echo $! > "${GEN}/greedy.pid"
run_one 1 t02 0.2 0.9 1 > "${GEN}/t02.log" 2>&1 & echo $! > "${GEN}/t02.pid"
run_one 2 t06 0.6 0.9 2 > "${GEN}/t06.log" 2>&1 & echo $! > "${GEN}/t06.pid"

failed=0
for pid_file in "${GEN}"/*.pid; do
  pid="$(cat "${pid_file}")"
  if ! wait "${pid}"; then
    echo "[vgi-qwen06b-sft97] generation failed: ${pid_file}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit "${failed}"
fi

docker run --rm --entrypoint python3 \
  -v "${WORKSPACE}:/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${DOCKER_IMAGE}" \
  scripts/make_vgi_qwen06b_sft_vote.py \
    --prompt-file "${PROMPT_FILE}" \
    --candidate-dir "${GEN}" \
    --output "${OUT}" \
    --report "${REPORT}"

docker run --rm --entrypoint python3 \
  -v "${WORKSPACE}:/workspace/eai-starter-kit" \
  -w /workspace/eai-starter-kit \
  "${DOCKER_IMAGE}" \
  - <<'PY' "${PROMPT_FILE}" "${OUT}" "${REPORT}"
import json
import sys
from pathlib import Path

prompt_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
report_file = Path(sys.argv[3])
prompts = json.loads(prompt_file.read_text(encoding="utf-8"))
rows = json.loads(output_file.read_text(encoding="utf-8"))
identifiers = [row["identifier"] for row in prompts]
if [row.get("identifier") for row in rows] != identifiers:
    raise SystemExit("identifier order mismatch")
json_ok = schema_ok = think_count = fence_count = self_loop = action_gt_2 = 0
node_total = edge_total = action_total = empty = 0
for row in rows:
    text = row.get("llm_output", "")
    think_count += int("<think>" in text)
    fence_count += int("```" in text)
    decoded = json.loads(text)
    json_ok += 1
    schema_ok += int(all(key in decoded for key in ["node goals", "edge goals", "action goals"]))
    nodes = decoded.get("node goals") or []
    edges = decoded.get("edge goals") or []
    actions = decoded.get("action goals") or []
    node_total += len(nodes)
    edge_total += len(edges)
    action_total += len(actions)
    empty += int(not nodes and not edges and not actions)
    action_gt_2 += int(len(actions) > 2)
    for edge in edges:
        self_loop += int(edge.get("from_name") == edge.get("to_name"))
validation = {
    "rows": len(rows),
    "json_ok": json_ok,
    "schema_ok": schema_ok,
    "think_count": think_count,
    "fence_count": fence_count,
    "self_loop": self_loop,
    "action_gt_2": action_gt_2,
    "total_node": node_total,
    "total_edge": edge_total,
    "total_action": action_total,
    "empty": empty,
}
if len(rows) != 1500 or json_ok != 1500 or schema_ok != 1500 or think_count or fence_count or self_loop or action_gt_2:
    raise SystemExit(f"validation failed: {validation}")
report = json.loads(report_file.read_text(encoding="utf-8"))
report["reproduce_validation"] = validation
report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
PY

docker run --rm --entrypoint chown \
  -v "${WORKSPACE}:/workspace/eai-starter-kit" \
  "${DOCKER_IMAGE}" \
  -R 1000:1000 "/workspace/eai-starter-kit/${GEN}" "/workspace/eai-starter-kit/${OUT}" "/workspace/eai-starter-kit/${REPORT}"

touch "${ROOT}/GEN_DONE"
echo "[vgi-qwen06b-sft97] reproduce done $(date)"
