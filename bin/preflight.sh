#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

missing=0
warn=0

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "MISSING file: $1"
    missing=1
  fi
}

need_dir() {
  if [[ ! -d "$1" ]]; then
    echo "MISSING dir: $1"
    missing=1
  fi
}

soft_file() {
  if [[ ! -f "$1" ]]; then
    echo "WARN missing optional artifact: $1"
    warn=1
  fi
}

soft_dir() {
  if [[ ! -d "$1" ]]; then
    echo "WARN missing optional artifact dir: $1"
    warn=1
  fi
}

note_file() {
  if [[ ! -f "$1" ]]; then
    echo "NOTE optional cached output not present: $1"
  fi
}

need_dir scripts
need_dir llm_prompts
need_dir llm_prompts_axis06b_behavior_compact

for path in \
  scripts/run_axis06b_behavior_bgi_btm_repro.sh \
  scripts/postprocess_axis06b_behavior_tm_outputs.py \
  scripts/run_axis06b_bas_bsd_c02_d02_repro.sh \
  scripts/apply_bsd_d02_direct_patch.py \
  scripts/run_axis06b_bas_bsd_c02_repro.sh \
  scripts/run_vas_v6_repro.sh \
  scripts/reproduce_vgi_qwen06b_sft97_vote.sh \
  scripts/reproduce_vsd_baseline_restore_grab.sh \
  scripts/reproduce_qwen06b_vtm_official_domain_infer.sh \
  scripts/run_qwen3_vllm_offline.sh \
  scripts/run_qwen3_selected_4gpu_sharded.sh \
  scripts/run_axistilted2_qwen3_lora_offline.sh \
  scripts/run_axistilted2_qwen3_lora_offline_4gpu_sharded.sh \
  scripts/generate_axis06b_compact_offline.py \
  scripts/generate_with_vllm_offline_lora.py \
  scripts/merge_vllm_shards_selected.py \
  scripts/postprocess_axis06b_behavior_outputs.py \
  scripts/optimize_virtualhome_action_sequencing.py \
  scripts/repair_virtualhome_action_sequencing_v6_2_frozen.py \
  scripts/repair_virtualhome_action_sequencing_v6.py \
  scripts/virtualhome_two_stage_planner.py \
  scripts/postprocess_vsd_outputs.py \
  scripts/build_vsd_ablation_candidates.py \
  scripts/validate_vsd_control_outputs.py \
  scripts/vsd_common.py \
  scripts/generate_qwen06b_vtm_schema_lora.py \
  scripts/postprocess_vtm_pddl_whitespace.py \
  scripts/vtm_schema_utils.py \
  scripts/lora_runtime.py \
  scripts/generate_qwen06b_vgi_lora.py \
  scripts/make_vgi_qwen06b_sft_vote.py \
  scripts/make_vgi_e4_learned_rules_vote.py \
  scripts/make_vgi_vote_submission.py
do
  need_file "${path}"
done

for path in \
  llm_prompts/behavior_transition_modeling_prompts.json \
  llm_prompts/virtualhome_action_sequencing_prompts.json \
  llm_prompts/virtualhome_goal_interpretation_prompts.json \
  llm_prompts/virtualhome_subgoal_decomposition_prompts.json \
  llm_prompts/virtualhome_transition_modeling_prompts.json \
  llm_prompts_axis06b_behavior_compact/behavior_action_sequencing_prompts.json \
  llm_prompts_axis06b_behavior_compact/behavior_goal_interpretation_prompts.json \
  llm_prompts_axis06b_behavior_compact/behavior_subgoal_decomposition_prompts.json
do
  need_file "${path}"
done

if ! command -v docker >/dev/null 2>&1; then
  echo "WARN docker is not on PATH; inference scripts use Docker."
  warn=1
fi

soft_file vgi_sft_work/qwen06b_vgi97/lora/adapter.pt
soft_file vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/lora_30/adapter_step200.pt
soft_file vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/schema_library.json
soft_file vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/virtualhome.pddl
soft_file docker/eai-vllm-qwen3_v0.11.1.tar
note_file outputs/vas_axistilted2_20260524_093512_raw/virtualhome_action_sequencing_outputs.json
note_file outputs/qwen32_awq_think/virtualhome_action_sequencing_outputs.json
soft_dir models/AxisTilted2
soft_dir models/Qwen3-32B-AWQ
soft_dir models/Qwen3-0.6B

if [[ "${missing}" -ne 0 ]]; then
  echo "Preflight failed: required files are missing."
  exit 1
fi

if [[ "${warn}" -ne 0 ]]; then
  echo "Preflight passed with warnings about external artifacts."
else
  echo "Preflight passed."
fi
