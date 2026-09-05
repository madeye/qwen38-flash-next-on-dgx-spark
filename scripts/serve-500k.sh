#!/usr/bin/env bash
# Persistent local Flash-Next profile. Build Dockerfile.performance first.
set -euo pipefail
cd "$(dirname "$0")/.."
export IMAGE="${IMAGE:-qwen38-flash-dgx:performance}"
export MODE="${MODE:-hybrid}"
export CTX="${CTX:-524288}" YARN="${YARN:-1}"
export MTP="${MTP:-3}" SEQS="${SEQS:-1}"
export GPU_MEM="${GPU_MEM:-0.80}" KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
# Explicit KV allocation bypasses GPU_MEM sizing. 9 GiB targets >600k FP8
# tokens while leaving memory for hybrid weights, MTP and desktop processes.
export KV_BYTES="${KV_BYTES:-9663676416}"
export BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
exec bash scripts/serve.sh
