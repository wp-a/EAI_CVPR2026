#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAR="${IMAGE_TAR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eai-vllm-qwen3_v0.11.1.tar}"

docker load -i "${IMAGE_TAR}"
docker image inspect eai-vllm-qwen3:v0.11.1 --format '{{.Id}} {{json .RepoTags}}'
