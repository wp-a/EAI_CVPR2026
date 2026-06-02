# Reviewer Guide

This repository is prepared for CVPR 2026 EAI challenge review. It contains the
reproduction scripts, prompt files, task manifest, lightweight resources, and
artifact layout documentation needed to inspect the method.

## Scope

Included in Git:

- task entrypoint scripts under `scripts/`
- official prompt JSON files under `llm_prompts/`
- compact BEHAVIOR prompt files under `llm_prompts_axis06b_behavior_compact/`
- `MANIFEST.json`, which maps each task to its reproduction command
- lightweight resources such as `resources/virtualhome.pddl`
- documentation for external artifacts

Excluded from Git:

- EvalAI credentials, SSH keys, local server paths, and account tokens
- generated submission outputs and previous EvalAI result files
- model weight binaries, LoRA adapter binaries, Docker image tar files, caches,
  and runtime logs
- starter-kit sample outputs or answer-bearing artifacts

## Task Entrypoints

| Task | Entrypoint |
| --- | --- |
| BEHAVIOR Goal Interpretation | `TASKS=bgi bash scripts/run_axis06b_behavior_bgi_btm_repro.sh` |
| BEHAVIOR Transition Modeling | `TASKS=btm bash scripts/run_axis06b_behavior_bgi_btm_repro.sh` |
| BEHAVIOR Action Sequencing | `bash scripts/run_axis06b_bas_bsd_c02_d02_repro.sh` |
| BEHAVIOR Subgoal Decomposition | `bash scripts/run_axis06b_bas_bsd_c02_d02_repro.sh` |
| VirtualHome Action Sequencing | `RUN_INFERENCE=1 EXP_ROOT=outputs/vas_v6_2_repro bash scripts/run_vas_v6_repro.sh` |
| VirtualHome Goal Interpretation | `WORKSPACE="$(pwd)" bash scripts/reproduce_vgi_qwen06b_sft97_vote.sh` |
| VirtualHome Subgoal Decomposition | `bash scripts/reproduce_vsd_baseline_restore_grab.sh` |
| VirtualHome Transition Modeling | `GPU=0 bash scripts/reproduce_qwen06b_vtm_official_domain_infer.sh` |

The same mapping is available in machine-readable form in `MANIFEST.json`.

## Preflight

Run:

```bash
bash bin/preflight.sh
```

The preflight script checks required repository files and reports missing
external artifacts. Missing model weights, adapters, Docker image tar files, and
optional cached outputs are expected until those artifacts are downloaded or
mounted.

## External Artifacts

See:

- `ARTIFACTS.md`
- `MODEL_INVENTORY.md`
- `docker/README.md`
- `external_artifacts/README.md`

Most artifact paths can be overridden with environment variables such as
`MODEL_HOST`, `MODEL`, `MODELS_HOST`, `LORA_HOST`, `ADAPTER`, `ARTIFACT_DIR`,
`WORKSPACE_HOST`, `CACHE_HOST`, and `DOCKER_IMAGE`.
