#!/usr/bin/env bash
# Persistent local Flash-Next profile. Build Dockerfile.performance first.
set -euo pipefail
cd "$(dirname "$0")/.."
export IMAGE="${IMAGE:-qwen38-flash-dgx:performance}"
export MODE="${MODE:-hybrid}"
export CTX="${CTX:-524288}" YARN="${YARN:-1}"
export MTP="${MTP:-3}" SEQS="${SEQS:-1}"
export GPU_MEM="${GPU_MEM:-0.80}" KV_DTYPE="${KV_DTYPE:-auto}"
# Explicit KV allocation bypasses GPU_MEM sizing. 16 GiB targets >580k BF16
# tokens while leaving memory for hybrid weights, MTP and desktop processes.
export KV_BYTES="${KV_BYTES:-17179869184}"
export BIND_ADDR="${BIND_ADDR:-127.0.0.1}"
exec bash scripts/serve.sh
