#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best BEHAVIOR BAS/BSD recipe:
# 1) run C02 compact 0.6B inference for BSD/BAS
# 2) apply prompt-derived conservative direct-goal BSD completion
#
# This script does not run BGI or BTM. If SUBMISSION_SOURCE_DIR is provided,
# it also creates a full 8-JSON submission directory by copying the other tasks.

RUN_ID="${RUN_ID:-axis06b_bas_bsd_c02_d02_repro_$(date +%Y%m%d_%H%M%S)}"
WORKSPACE_HOST="${WORKSPACE_HOST:-$(pwd)}"
CANDIDATE="${CANDIDATE:-C02_compact_t02}"
D02_CANDIDATE="${D02_CANDIDATE:-D02_complete_7_short_direct_tasks}"
SUBMISSION_SOURCE_DIR="${SUBMISSION_SOURCE_DIR:-}"
PROMPT_DIR="${PROMPT_DIR:-llm_prompts_axis06b_behavior_compact}"
OUT_ROOT="${OUT_ROOT:-outputs/${RUN_ID}}"
RAW_ROOT="${RAW_ROOT:-outputs/${RUN_ID}_raw}"
D02_ROOT="${D02_ROOT:-outputs/${RUN_ID}_d02}"
LOG_ROOT="${LOG_ROOT:-}"

export RUN_ID
export CANDIDATE
export OUT_ROOT
export RAW_ROOT
export LOG_ROOT

bash scripts/run_axis06b_bas_bsd_c02_repro.sh

base_dir="${OUT_ROOT}/${CANDIDATE}"
d02_root="${D02_ROOT}"
d02_dir="${d02_root}/${D02_CANDIDATE}"
mkdir -p "${d02_dir}" "${d02_root}/_reports"

python3 scripts/apply_bsd_d02_direct_patch.py \
  --input-bsd "${base_dir}/behavior_subgoal_decomposition_outputs.json" \
  --prompt-file "${PROMPT_DIR}/behavior_subgoal_decomposition_prompts.json" \
  --output-bsd "${d02_dir}/behavior_subgoal_decomposition_outputs.json" \
  --report "${d02_root}/_reports/${D02_CANDIDATE}_patch_report.json"

cp "${base_dir}/behavior_action_sequencing_outputs.json" "${d02_dir}/behavior_action_sequencing_outputs.json"

if [[ -n "${SUBMISSION_SOURCE_DIR}" ]]; then
  full_dir="${d02_root}/${D02_CANDIDATE}_full_submission"
  python3 scripts/apply_bsd_d02_direct_patch.py \
    --input-bsd "${base_dir}/behavior_subgoal_decomposition_outputs.json" \
    --prompt-file "${PROMPT_DIR}/behavior_subgoal_decomposition_prompts.json" \
    --output-bsd "${d02_dir}/behavior_subgoal_decomposition_outputs.json" \
    --copy-from-dir "${SUBMISSION_SOURCE_DIR}" \
    --output-dir "${full_dir}" \
    --report "${d02_root}/_reports/${D02_CANDIDATE}_full_submission_report.json"
  cp "${base_dir}/behavior_action_sequencing_outputs.json" "${full_dir}/behavior_action_sequencing_outputs.json"
  echo "full_submission_dir=${full_dir}"
fi

echo "bas_bsd_dir=${d02_dir}"
echo "raw_c02_dir=${base_dir}"
echo "report_dir=${d02_root}/_reports"
