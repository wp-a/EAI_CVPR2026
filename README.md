# EAI CVPR 2026 8-Task Repro Kit

This repository contains reproducible scripts and lightweight resources for the
eight CVPR 2026 EAI tasks:

- B-BGI and B-BTM
- B-BAS and B-BSD
- V-AS, V-GI, V-SD, and V-TM

Large model weights, LoRA adapters, Docker images, runtime outputs, and account
credentials are intentionally not committed. See `ARTIFACTS.md` for the expected
external artifact layout.

## Reviewer Notes

This repository is intended for challenge review and method reproducibility. It
contains the code paths used to generate the eight EvalAI submission JSON files,
plus the prompt resources required by those scripts. It does not include:

- EvalAI tokens, account credentials, SSH keys, or local server paths
- generated submission outputs or previous EvalAI result files
- model weight binaries, LoRA adapter binaries, Docker image tar files, caches,
  or runtime logs
- starter-kit sample outputs or answer-bearing artifacts

The expected external artifacts and their target paths are documented in
`ARTIFACTS.md` and `MODEL_INVENTORY.md`.

## Layout

- `scripts/`: reproducible entrypoints and their direct helper scripts
- `llm_prompts/`: required official prompt files for BTM and VirtualHome tasks
- `llm_prompts_axis06b_behavior_compact/`: compact BEHAVIOR prompts for BGI/BAS/BSD
- `MANIFEST.json`: task-to-command mapping
- `docs/task_matrix.md`: compact task summary
- `bin/preflight.sh`: local file and external-artifact sanity check
- `models/`: README files only; download model weights separately
- `docker/`: Docker image pull/load notes

## Quick Check

```bash
cd EAI_CVPR2026
bash bin/preflight.sh
```

Warnings about cached V-AS candidates are expected if you have not copied the
canonical raw candidate outputs. Model, adapter, and Docker tar warnings are
expected until you download or mount the external artifacts described below.

## Output Files

The full reproduction pipeline writes the eight expected EAI output files:

```text
behavior_goal_interpretation_outputs.json
behavior_subgoal_decomposition_outputs.json
behavior_action_sequencing_outputs.json
behavior_transition_modeling_outputs.json
virtualhome_goal_interpretation_outputs.json
virtualhome_subgoal_decomposition_outputs.json
virtualhome_action_sequencing_outputs.json
virtualhome_transition_modeling_outputs.json
```

## Compact BEHAVIOR Prompts

The directory `llm_prompts_axis06b_behavior_compact/` is reproducible from the
official BEHAVIOR prompt files in `llm_prompts/`. The builder extracts only the
dynamic fields needed by the AxisTilted2 BEHAVIOR models, then writes compact
prompt JSON files with the same EvalAI-style row shape:

```json
[
  {"identifier": "...", "llm_prompt": "..."}
]
```

The output filenames stay the same for the three BEHAVIOR prompt classes:

- `behavior_goal_interpretation_prompts.json`
- `behavior_subgoal_decomposition_prompts.json`
- `behavior_action_sequencing_prompts.json`

Run the compact prompt builder with:

```bash
python3 scripts/make_axis06b_behavior_compact_prompts.py \
  --input-dir llm_prompts \
  --output-dir llm_prompts_axis06b_behavior_compact
```

Field layouts:

```text
GI prompt:
{relevant_objects}
--
{initial_states}
--
{task_name}
{goal_instructions}
```

```text
SD prompt:
{task_name}
--
{relevant_objects}
--
{initial_states}
--
{goal_states}
```

```text
AS prompt:
{initial_states}
--
{target_states}
--
{interactable_objects}
```

`scripts/generate_axis06b_compact_offline.py` is the offline inference script
that consumes these compact prompts during B-GI/B-AS/B-SD reproduction.

## Commands

Run B-BGI:

```bash
TASKS=bgi bash scripts/run_axis06b_behavior_bgi_btm_repro.sh
```

Run B-BTM:

```bash
TASKS=btm bash scripts/run_axis06b_behavior_bgi_btm_repro.sh
```

Run both B-BGI and B-BTM:

```bash
bash scripts/run_axis06b_behavior_bgi_btm_repro.sh
```

Run B-BAS and B-BSD with prompt-derived direct-goal BSD completion:

```bash
bash scripts/run_axis06b_bas_bsd_c02_d02_repro.sh
```

Run V-AS from local or mounted models:

```bash
RUN_INFERENCE=1 EXP_ROOT=outputs/vas_v6_2_repro bash scripts/run_vas_v6_repro.sh
```

If you separately copy the canonical raw V-AS candidates into `outputs/`, you
can use `RUN_INFERENCE=0` to skip inference.

Run V-GI:

```bash
WORKSPACE="$(pwd)" bash scripts/reproduce_vgi_qwen06b_sft97_vote.sh
```

Run V-SD:

```bash
bash scripts/reproduce_vsd_baseline_restore_grab.sh
```

Run V-TM:

```bash
GPU=0 bash scripts/reproduce_qwen06b_vtm_official_domain_infer.sh
```

## External Artifacts

See `ARTIFACTS.md`, `MODEL_INVENTORY.md`, `external_artifacts/README.md`, and `docker/README.md`
for model, Docker, adapter, and optional cached-output locations. Most paths can
be overridden with environment variables such as `MODEL_ROOT`, `MODEL_HOST`,
`WORKSPACE_HOST`, `ROOT`, `EXP_ROOT`, `CACHE_HOST`, `RUN_INFERENCE`, and
`SUBMISSION_SOURCE_DIR`.

Large artifacts intentionally excluded from Git:

- `models/**/*.safetensors`
- `vgi_sft_work/**/*.pt`
- `vtm_infer_artifacts/**/*.pt`
- `docker/*.tar`
- runtime logs, outputs, caches, and EvalAI credentials

## Notes

- `scripts/run_qwen3_vllm_offline.sh` is included as the missing single-GPU
  helper required by `scripts/run_qwen3_selected_4gpu_sharded.sh`.
- V-TM uses a pre-trained Qwen3-0.6B official-domain VTM LoRA and runs
  inference only.
- B-BAS and B-BSD share the same entrypoint because the reproducible recipe first
  creates the C02 BAS/BSD outputs, then applies conservative BSD direct-goal
  completion derived from each prompt's goal-state section. This completion is
  not keyed by task identifier.
