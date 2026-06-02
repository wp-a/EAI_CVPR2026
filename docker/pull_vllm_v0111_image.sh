#!/usr/bin/env bash
set -euo pipefail

SOURCE_IMAGE="${SOURCE_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.1}"
TARGET_IMAGE="${TARGET_IMAGE:-eai-vllm-qwen3:v0.11.1}"

docker pull "${SOURCE_IMAGE}"
docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE}"

echo "Ready: ${TARGET_IMAGE}"
