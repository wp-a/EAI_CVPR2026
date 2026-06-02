# Task Matrix

| Task | Repro entrypoint | Notes |
| --- | --- | --- |
| B-BGI | `scripts/run_axis06b_behavior_bgi_btm_repro.sh` | Set `TASKS=bgi` to run only BGI. |
| B-BTM | `scripts/run_axis06b_behavior_bgi_btm_repro.sh` | Uses `scripts/postprocess_axis06b_behavior_tm_outputs.py`. |
| B-BAS | `scripts/run_axis06b_bas_bsd_c02_d02_repro.sh` | Shares the BAS/BSD C02 run and copies BAS into the D02 candidate dir. |
| B-BSD | `scripts/run_axis06b_bas_bsd_c02_d02_repro.sh` | Applies prompt-derived short direct-goal completion via `scripts/apply_bsd_d02_direct_patch.py`; candidate name is `D02_complete_7_short_direct_tasks`. |
| V-AS | `scripts/run_vas_v6_repro.sh` | Use `RUN_INFERENCE=1` to regenerate raw candidates from local/mounted models; use `RUN_INFERENCE=0` only when cached raw candidates are present. |
| V-GI | `scripts/reproduce_vgi_qwen06b_sft97_vote.sh` | Requires the Qwen 0.6B SFT LoRA adapter under `vgi_sft_work/qwen06b_vgi97/lora/adapter.pt`. |
| V-SD | `scripts/reproduce_vsd_baseline_restore_grab.sh` | Runs Qwen32 raw inference, conservative postprocess, restore-GRAB ablation, and validation. |
| V-TM | `scripts/reproduce_qwen06b_vtm_official_domain_infer.sh` | Qwen3-0.6B official-domain VTM LoRA inference only; run with `GPU=0`. |
