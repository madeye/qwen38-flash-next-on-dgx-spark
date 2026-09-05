#!/usr/bin/env bash
# Optional BF16 KV profile; the default 500k profile uses FP8 KV.
set -euo pipefail
cd "$(dirname "$0")/.."
export KV_DTYPE="${KV_DTYPE:-auto}"
export KV_BYTES="${KV_BYTES:-17179869184}"
exec bash scripts/serve-500k.sh
