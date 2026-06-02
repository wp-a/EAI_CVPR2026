# Docker Image

The reproducible scripts use:

```bash
eai-vllm-qwen3:v0.11.1
```

The image is also available from the mirror used on the L40s server:

```bash
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.1
docker tag swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.1 eai-vllm-qwen3:v0.11.1
```

If `docker/eai-vllm-qwen3_v0.11.1.tar` is present, load it directly:

```bash
docker load -i docker/eai-vllm-qwen3_v0.11.1.tar
```

The expected local image id on the source server was:

```text
sha256:e4896bdb93ffab61032f8992624928c198363b5085a0c9ac2af8a7f992de89a2
```
