# External Artifacts

This repro kit vendors model weights and the Docker image tar in the server
bundle. Large historical output directories are still optional.

Expected external artifacts:

- Docker image: `eai-vllm-qwen3:v0.11.1`
- Docker tar: `docker/eai-vllm-qwen3_v0.11.1.tar`
- AxisTilted2 model root: `models/AxisTilted2`
- Qwen3-32B-AWQ model: `models/Qwen3-32B-AWQ`
- Qwen3-0.6B model: `models/Qwen3-0.6B`
- V-GI LoRA adapter: `vgi_sft_work/qwen06b_vgi97/lora/adapter.pt`
- Optional V-AS cached raw candidates:
  - `outputs/vas_axistilted2_20260524_093512_raw/virtualhome_action_sequencing_outputs.json`
  - `outputs/qwen32_awq_think/virtualhome_action_sequencing_outputs.json`
