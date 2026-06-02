# External Artifact Layout

This repository keeps code, prompts, lightweight resources, and documentation in
Git. Large binary artifacts should be downloaded or mounted separately.

## Docker

The scripts expect a vLLM image named:

```bash
eai-vllm-qwen3:v0.11.1
```

One known mirror command is:

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.1
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.1 eai-vllm-qwen3:v0.11.1
```

## Models And Adapters

Place the following files/directories before running full reproduction:

```text
models/AxisTilted2/b_gi_Qwen-0.6B-FINALSUB/
models/AxisTilted2/b_sd_Qwen-0.6B-FINALSUB/
models/AxisTilted2/b_as_2_Qwen-0.6B-FINALSUB/
models/AxisTilted2/b_compiled_model_Qwen-0.6B-TM-2-FINALSUB/
models/AxisTilted2/qwen3-32b-domain2ep-vas20ep/
models/Qwen3-0.6B/
models/Qwen3-32B-AWQ/
vgi_sft_work/qwen06b_vgi97/lora/adapter.pt
vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/lora_30/adapter_step200.pt
vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/schema_library.json
vtm_infer_artifacts/qwen06b_vtm_official_domain_fast/virtualhome.pddl
```

Most scripts allow overriding these locations through environment variables such
as `MODEL_HOST`, `MODEL`, `MODELS_HOST`, `LORA_HOST`, `ADAPTER`,
`ARTIFACT_DIR`, `WORKSPACE_HOST`, `CACHE_HOST`, and `DOCKER_IMAGE`.

## Runtime Outputs

Runtime outputs are written under `outputs/` by default and are ignored by Git.
Submission files should be generated and validated locally before use.
