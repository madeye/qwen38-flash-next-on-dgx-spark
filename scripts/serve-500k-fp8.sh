#!/usr/bin/env bash
# FP8 KV storage with in-kernel BF16 conversion, using the existing QSA patch.
set -euo pipefail
cd "$(dirname "$0")/.."
export KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
# Similar token capacity to the 16 GiB BF16 profile, with 7 GiB less allocation.
export KV_BYTES="${KV_BYTES:-9663676416}"
exec bash scripts/serve-500k.sh
