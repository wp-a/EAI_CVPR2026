# Model Inventory

The inference scripts expect model files under `models/` and LoRA/adapters under
the paths listed below. The Git repository contains only README files for model
directories; download or mount the real weights before running full inference.

Required copied directories/files:

- `models/AxisTilted2/b_gi_Qwen-0.6B-FINALSUB`
- `models/AxisTilted2/b_compiled_model_Qwen-0.6B-TM-2-FINALSUB`
- `models/AxisTilted2/b_as_2_Qwen-0.6B-FINALSUB`
- `models/AxisTilted2/b_sd_Qwen-0.6B-FINALSUB`
- `models/AxisTilted2/qwen3-32b-domain2ep-vas20ep`
- `models/Qwen3-0.6B`
- `models/Qwen3-32B-AWQ`
- `vgi_sft_work/qwen06b_vgi97/lora/adapter.pt`
- `resources/virtualhome.pddl` for the V-TM official-domain SFT data builder
- `vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/lora_30/adapter_step200.pt`
  for V-TM inference-only reproduction

Approximate artifact sizes:

- `models/Qwen3-0.6B`: 2.9G
- `models/Qwen3-32B-AWQ`: 18G
- `models/AxisTilted2`: 12G total, with only the required subdirectories copied
- `vgi_sft_work/qwen06b_vgi97/lora/adapter.pt`: 39M
