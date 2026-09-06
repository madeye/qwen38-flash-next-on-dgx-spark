#!/usr/bin/env bash
# Download the official NVIDIA NVFP4 checkpoint (~124 GiB) used by serve.sh.
# Resumable — safe to re-run if the connection drops.
#
#   scripts/download-weights.sh
#
# Needs ~130 GB free on the filesystem holding ~/.cache/huggingface.
set -euo pipefail

MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
MODEL_HOST="${MODEL_HOST:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
mkdir -p "$MODEL_HOST"

# hf authenticates via HF_TOKEN (or the older HUGGING_FACE_HUB_TOKEN name).
# docker -e NAME (no value) copies the host env var into the container.
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HF_TOKEN)
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN="$HUGGING_FACE_HUB_TOKEN")
else
  echo ">> no HF_TOKEN in the environment; Hub will rate-limit unauthenticated downloads"
fi

echo ">> downloading $MODEL into $MODEL_HOST (resumable)"
# HF_HUB_DISABLE_XET=1: the Xet backend stalled on some Spark setups; plain HTTPS
# is reliable and saturates the link.
docker run --rm --name qwen38-dl \
  -e HF_HUB_DISABLE_XET=1 \
  "${TOKEN_ARGS[@]}" \
  -v "$MODEL_HOST:/models" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --local-dir /models --max-workers 8"

echo ">> done. Verify with: scripts/serve.sh"
