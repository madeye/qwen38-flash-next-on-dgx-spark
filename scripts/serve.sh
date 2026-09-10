#!/usr/bin/env bash
# Default single-DGX-Spark Qwen3.8 Flash recipe: NVIDIA's NVFP4 checkpoint,
# staged disk PLE gather, reduced-vocabulary MTP3 drafting, and decode graphs.
#
#   scripts/serve.sh
#   docker logs -f qwen38-flash
#
# Defaults reproduce the validated TP1 recipe: 262k context, 6 sequences,
# MTP3, FP8 KV, BF16 recurrent state, a 4096-token prefill chunk,
# MiaAI's code-tuned draft vocabulary, and prefix caching disabled.
# Override MODEL_HOST only when the official checkpoint lives elsewhere.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c}"
MODEL_HOST="${MODEL_HOST:-/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-nvidia}"
PATCH_DIR="${PATCH_DIR:-$ROOT/src/full-recipe-patch}"
CACHE_HOST="${CACHE_HOST:-/var/tmp/qwen38fn-vllm-cache}"
PORT="${PORT:-18300}"
HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-30}"
GMU="${GMU:-}"
MAXLEN="${MAXLEN:-262144}"
SEQS="${SEQS:-6}"
MTP="${MTP:-3}"
CHUNK="${CHUNK-4096}"
CAPTURE_SIZES="${CAPTURE_SIZES-auto}"
PLE_MODE="${PLE_MODE:-staged}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"
DRAFT_VOCAB="${DRAFT_VOCAB-$ROOT/src/miaai/draft_vocab_en_code_47k.txt}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE-bfloat16}"
PREFIX_CACHE="${PREFIX_CACHE:-0}"
EXTRA="${EXTRA:-}"

for value in PORT MAXLEN SEQS MTP; do
  [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || { echo "!! $value must be a positive integer" >&2; exit 2; }
done
[[ "$CHUNK" =~ ^[1-9][0-9]*$ || -z "$CHUNK" ]] || { echo "!! CHUNK must be a positive integer or empty" >&2; exit 2; }
[[ "$CAPTURE_SIZES" == auto || "$CAPTURE_SIZES" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ || -z "$CAPTURE_SIZES" ]] || { echo "!! CAPTURE_SIZES must be auto, comma-separated positive integers, or empty" >&2; exit 2; }
[[ -z "$MAMBA_SSM_CACHE_DTYPE" || "$MAMBA_SSM_CACHE_DTYPE" == float32 || "$MAMBA_SSM_CACHE_DTYPE" == bfloat16 ]] || { echo "!! MAMBA_SSM_CACHE_DTYPE must be bfloat16, float32, or empty" >&2; exit 2; }
[[ "$PREFIX_CACHE" == 0 || "$PREFIX_CACHE" == 1 ]] || { echo "!! PREFIX_CACHE must be 0 or 1" >&2; exit 2; }
[[ "$HOST_RESERVE_GIB" =~ ^[1-9][0-9]*$ ]] || { echo "!! HOST_RESERVE_GIB must be a positive integer" >&2; exit 2; }

# GB10 shares memory with the host. Derive the GPU budget from an explicit
# host reserve instead of a fixed fraction. The measured 30 GiB reserve yields
# GMU=0.7535, 7.44 GiB of KV budget, and 507,810 cached tokens on this machine.
# A 36 GiB reserve left only 0.97 GiB usable KV, below the 3.84 GiB needed for
# one 262K request. Increasing the reserve may require reducing MAXLEN too.
# See docs/miaai-optimization-validation-2026-09-10.md; profiling varies by setup.
# Explicit GMU overrides the calculation; GMU=0.80 reproduces the older budget.
if [[ -z "$GMU" ]]; then
  GMU="$(awk -v r="$HOST_RESERVE_GIB" '/^MemTotal:/ {t=$2/1048576; if (t<=r) exit 1; printf "%.4f", (t-r)/t}' /proc/meminfo)" \
    || { echo "!! HOST_RESERVE_GIB=$HOST_RESERVE_GIB exceeds MemTotal" >&2; exit 2; }
fi
[[ "$GMU" =~ ^(0\.[0-9]+|1|1\.0+)$ ]] || { echo "!! GMU must be a fraction in (0,1] like 0.78 (got '$GMU')" >&2; exit 2; }
[[ -f "$MODEL_HOST/config.json" ]] || { echo "!! official NVIDIA checkpoint missing at $MODEL_HOST" >&2; exit 3; }
for file in ple_layer.py ple_mmap.py model_state.py mtp_draft_vocab.py upstream-overlays/modelopt.py; do
  [[ -f "$PATCH_DIR/$file" ]] || { echo "!! recipe patch missing: $PATCH_DIR/$file" >&2; exit 3; }
done

mkdir -p "$CACHE_HOST"
PLE_ENV=()
case "$PLE_MODE" in
  staged)
    PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_STAGED=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}"
      -v "$PATCH_DIR/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py:ro"
      -v "$PATCH_DIR/ple_mmap.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro"
      -v "$PATCH_DIR/model_state.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py:ro")
    ;;
  mmap)
    PLE_ENV=(-e QWEN4EXP_PLE_MMAP=1 -e QWEN4EXP_PLE_MMAP_THREADS="${PLE_WORKERS:-64}"
      -v "$PATCH_DIR/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py:ro"
      -v "$PATCH_DIR/ple_mmap.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py:ro")
    ;;
  none) ;;
  *) echo "!! PLE_MODE must be staged, mmap, or none" >&2; exit 2 ;;
esac

DRAFT_ENV=() DRAFT_MOUNT=()
if [[ -n "$DRAFT_VOCAB" && "$DRAFT_VOCAB" != 0 ]]; then
  DRAFT_MOUNT=(-v "$PATCH_DIR/mtp_draft_vocab.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py:ro")
  if [[ "$DRAFT_VOCAB" =~ ^[1-9][0-9]*$ ]]; then
    DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB="$DRAFT_VOCAB")
  else
    [[ "$DRAFT_VOCAB" == /* ]] || DRAFT_VOCAB="$ROOT/$DRAFT_VOCAB"
    [[ -f "$DRAFT_VOCAB" ]] || { echo "!! draft vocabulary missing: $DRAFT_VOCAB" >&2; exit 3; }
    DRAFT_ENV=(-e QWEN4EXP_DRAFT_VOCAB=/opt/qwen38-draft-vocab.txt)
    DRAFT_MOUNT+=(-v "$DRAFT_VOCAB:/opt/qwen38-draft-vocab.txt:ro")
  fi
  python3 - "$DRAFT_VOCAB" "$MODEL_HOST/config.json" <<'PYVOCAB'
import json, sys
from pathlib import Path
value, config_path = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
size = config.get("text_config", config)["vocab_size"]
if value.isdecimal():
    assert 0 < int(value) <= size, "draft vocabulary count exceeds model vocabulary"
else:
    ids = [int(line) for line in Path(value).read_text().splitlines() if line.strip()]
    assert ids and len(ids) == len(set(ids)), "draft vocabulary is empty or has duplicate IDs"
    assert min(ids) >= 0 and max(ids) < size, "draft vocabulary contains out-of-range IDs"
PYVOCAB
fi

VP=/usr/local/lib/python3.12/dist-packages/vllm
OVERLAY_MOUNT=(
  -v "$PATCH_DIR/upstream-overlays/ops_ple.py:$VP/models/qwen4_exp/nvidia/ops/ple.py:ro"
  -v "$PATCH_DIR/upstream-overlays/ops_qsa.py:$VP/models/qwen4_exp/nvidia/ops/qsa.py:ro"
  -v "$PATCH_DIR/upstream-overlays/qsa.py:$VP/models/qwen4_exp/nvidia/qsa.py:ro"
  -v "$PATCH_DIR/upstream-overlays/platforms_interface.py:$VP/platforms/interface.py:ro"
  -v "$PATCH_DIR/upstream-overlays/modelopt.py:$VP/model_executor/layers/quantization/modelopt.py:ro"
)

GRAPH_ARGS=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}")
if [[ "$CAPTURE_SIZES" == auto ]]; then
  CAPTURE_SIZES=$(python3 - "$MTP" "$SEQS" <<'PYGRAPHS'
import sys
k, sequences = map(int, sys.argv[1:])
print(",".join(str((k + 1) * s) for s in range(1, sequences + 1)))
PYGRAPHS
)
fi
if [[ -n "$CAPTURE_SIZES" ]]; then
  GRAPH_ARGS=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}")
fi
PC_ARG=--no-enable-prefix-caching
[[ "$PREFIX_CACHE" == 1 ]] && PC_ARG=--enable-prefix-caching
CHUNK_ARGS=()
[[ -n "$CHUNK" ]] && CHUNK_ARGS=(--max-num-batched-tokens "$CHUNK")
SSM_ARGS=()
[[ -n "$MAMBA_SSM_CACHE_DTYPE" ]] && SSM_ARGS=(--mamba-ssm-cache-dtype "$MAMBA_SSM_CACHE_DTYPE")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  docker() { printf '%q ' "$@"; printf '\n'; }
else
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sync
  sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1 || true
fi

# shellcheck disable=SC2086
docker run --gpus all -d --name "$NAME" --restart unless-stopped \
  --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 \
  -v "$MODEL_HOST:/models/qwen38fn:ro" -v "$CACHE_HOST:/root/.cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  "${PLE_ENV[@]}" "${DRAFT_ENV[@]}" "${DRAFT_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" \
  "$IMAGE" /models/qwen38fn --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code --quantization modelopt --tensor-parallel-size 1 \
    --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GMU" "${CHUNK_ARGS[@]}" \
    --no-enable-flashinfer-autotune "$PC_ARG" --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}" \
    "${GRAPH_ARGS[@]}" "${SSM_ARGS[@]}" --kv-cache-dtype "$KV_DTYPE" $EXTRA

echo ">> $NAME starting on http://127.0.0.1:$PORT with the NVIDIA full TP1 recipe"
echo ">> gmu=$GMU (host reserve ${HOST_RESERVE_GIB} GiB of $(awk '/^MemTotal:/ {printf "%.1f", $2/1048576}' /proc/meminfo) GiB), maxlen=$MAXLEN seqs=$SEQS mtp=$MTP kv=$KV_DTYPE"
echo ">> ready when: curl -fsS http://127.0.0.1:$PORT/health"
