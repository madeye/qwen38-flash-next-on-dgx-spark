#!/usr/bin/env python3
"""fp8-e4m3 storage for the Qwen3.8-Flash-Next QSA main KV cache (VLLM_QSA_FP8_KV=1).

K/V are stored in the paged cache as float8_e4m3 and dequantized to bf16 inside the
Triton sparse-attention kernel, immediately after the paged gather. Q stays bf16,
accumulation stays fp32, and the output stays bf16 -- only *storage* changes.

Why it is worth doing here: 12 of the 48 layers are ``full_attention`` (QSA) and they
dominate the KV pool --
    2 (K,V) x 2 kv_heads x 256 head_dim x 2 B = 2048 B / token / layer
    x 12 layers                               = 24.0 KiB / token
against a measured 28.4 KiB/token blended total, i.e. ~85%. The remainder is GDN
recurrent state plus the indexer's raw-key ring, which this patch does not touch -- they
are separate KV groups, and indexer_qsa.py hardcodes bf16 rather than reading
cache_config.cache_dtype, so its unpatched kernel is unaffected.

Measured end to end (GB10, 500k ctx, hybrid, MTP=2): 29,069 -> 15,782 B per token, a
45.7% cut; a 16 GiB fp8 pool holds 1,088,571 tokens against 887,878 for a 24 GiB bf16
pool. It costs ~14% single-stream decode (48.6 -> 41.7 tok/s, TPOT 20.6 -> 24.0 ms):
partly the in-loop dequant, partly MTP acceptance length falling 2.694 -> 2.529, since
fp8 perturbs the target's logits and the draft head agrees with it slightly less often.
Generation quality was not regression-tested beyond smoke checks; a real eval is owed.

Upstream refuses fp8 KV with NotImplementedError in four places, not because the math
is wrong but because the QSA Triton kernel had no dequant path. This adds one.

Scales follow vLLM's convention: store fp8 = bf16 / scale, dequantize bf16 = fp8 * scale.
They come from the layer's ``_k_scale``/``_v_scale`` buffers, which qsa.py already
registers via ``set_default_quant_scales``. Both are folded rather than applied
per-element:
  - k_scale into the softmax scale (scores are linear in K),
  - v_scale into the normalized output (the accumulator is linear in V, and the
    softmax normalizer does not involve V).
This keeps the inner loop free of extra multiplies and is exact, not an approximation.

The kernel edit is gated on the cache's actual dtype, so with a bf16 cache the compiled
code is unchanged. Only the *guards* are gated on VLLM_QSA_FP8_KV, so with the env var
unset this image behaves exactly like upstream.

Note on storage dtype: vLLM's kv_cache_dtype_str_to_dtype("fp8_e4m3") returns
torch.uint8, not torch.float8_e4m3fn -- fp8 KV pages are raw byte storage that each
backend reinterprets. The wrapper therefore .view()s them to float8_e4m3fn (free, same
itemsize) before launching. Skipping that view does not raise; Triton would read the fp8
bit patterns as integers 0..255 and return silently wrong attention output.

Usage:  VLLM_QSA_FP8_KV=1  plus  --kv-cache-dtype fp8_e4m3
"""
import ast
import sys
from pathlib import Path

NVIDIA_DIR = Path(sys.argv[1])
QSA = NVIDIA_DIR / "qsa.py"
OPS = NVIDIA_DIR / "ops" / "qsa.py"


def sub(src: str, old: str, new: str, what: str) -> str:
    n = src.count(old)
    assert n == 1, f"{what}: expected 1 occurrence, found {n}"
    return src.replace(old, new)


# --------------------------------------------------------------------------------
# 1. ops/qsa.py -- teach the sparse attention kernel to dequantize
# --------------------------------------------------------------------------------
ops = OPS.read_text()

# 1a. kernel signature: two runtime scales + one constexpr switch
ops = sub(
    ops,
    """    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,""",
    """    num_rows,
    num_cache_blocks,
    num_requests,
    k_scale_arg,
    v_scale_arg,
    FP8_KV: tl.constexpr,
    TOPK: tl.constexpr,""",
    "kernel signature",
)

# 1b. fold k_scale into the softmax scale (was a constexpr; now a runtime value)
ops = sub(
    ops,
    "    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634\n",
    "    softmax_scale_log2 = (HEAD_DIM**-0.5) * 1.4426950408889634 * k_scale_arg\n",
    "softmax scale",
)

# 1c. dequantize the gathered tiles before the dots. `values` must become bf16 here:
#     the accumulator dot casts `probabilities.to(values.dtype)`, which would otherwise
#     quantize the softmax probabilities to fp8.
ops = sub(
    ops,
    "        scores = tl.dot(query, keys)\n",
    """        if FP8_KV:
            keys = keys.to(tl.bfloat16)
            values = values.to(tl.bfloat16)
        scores = tl.dot(query, keys)
""",
    "dequant insertion point",
)

# 1d. fold v_scale into the normalized output. Correct for both the NUM_SPLITS == 1
#     store and the split-k partials: the merge is a weighted sum, so a uniform scale
#     on every partial survives it.
ops = sub(
    ops,
    """    output_mask = head_offsets[:, None] < GROUP_SIZE
""",
    """    normalized_output = normalized_output * v_scale_arg
    output_mask = head_offsets[:, None] < GROUP_SIZE
""",
    "v_scale fold",
)

# 1e. wrapper signature
ops = sub(
    ops,
    """    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    \"\"\"Run sparse GQA directly over paged BF16 K/V caches.\"\"\"""",
    """    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    \"\"\"Run sparse GQA over paged BF16 or FP8-E4M3 K/V caches.\"\"\"""",
    "wrapper signature",
)

# 1f. relax the dtype assert and reinterpret vLLM's fp8 byte storage.
#     kv_cache_dtype_str_to_dtype("fp8_e4m3") returns torch.uint8, not float8_e4m3fn:
#     vLLM allocates fp8 KV as raw bytes and lets each backend reinterpret them. Without
#     this view, Triton would load uint8 and .to(tl.bfloat16) would read the fp8 bit
#     patterns as integers 0..255 -- silently wrong output rather than an error. The view
#     is free (both dtypes are 1 byte) and preserves strides.
ops = sub(
    ops,
    "    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16\n",
    """    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
    if k_cache.dtype == torch.uint8:
        k_cache = k_cache.view(torch.float8_e4m3fn)
        v_cache = v_cache.view(torch.float8_e4m3fn)
""",
    "dtype assert",
)

# 1g. launch site
ops = sub(
    ops,
    """        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],""",
    """        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        k_scale,
        v_scale,
        FP8_KV=(k_cache.dtype == torch.float8_e4m3fn),
        TOPK=logical_indices.shape[1],""",
    "kernel launch",
)

ast.parse(ops)
OPS.write_text(ops)
print("ops/qsa.py: fp8 dequant path added")


# --------------------------------------------------------------------------------
# 2. qsa.py -- relax the four guards and thread the scales through
# --------------------------------------------------------------------------------
qsa = QSA.read_text()

qsa = sub(
    qsa,
    "import torch\nfrom torch import nn\n",
    "import os\n\nimport torch\nfrom torch import nn\n",
    "os import",
)

# 2a. backend capability declaration
qsa = sub(
    qsa,
    '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n',
    "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(_QSA_OK_CACHE_DTYPES)\n",
    "backend supported dtypes",
)

# 2a2. FlashAttentionImpl.__init__ (flash_attn.py) raises for any quantized KV when
#      FlashAttention's own kernels lack fp8 on this device -- true on sm_121. QSA
#      inherits that __init__ but never calls those kernels; attention runs in the
#      Triton sparse kernel. Hide the dtype from the base ctor, then put it back:
#      the inherited do_kv_cache_update() needs the real value to quantize on write.
qsa = sub(
    qsa,
    """    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():""",
    """    def __init__(self, *args, **kwargs) -> None:
        args, kwargs, _qsa_real_kv_dtype = _qsa_mask_kv_dtype(args, kwargs)
        super().__init__(*args, **kwargs)
        if _qsa_real_kv_dtype is not None:
            self.kv_cache_dtype = _qsa_real_kv_dtype
        if not is_flash_attn_varlen_func_available():""",
    "flash-attn fp8 guard bypass",
)

# 2b. impl __init__ guard
qsa = sub(
    qsa,
    """        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )""",
    """        if self.kv_cache_dtype not in _QSA_OK_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
                " (set VLLM_QSA_FP8_KV=1 to allow fp8_e4m3)"
            )""",
    "impl init guard",
)

# 2c. forward_qsa dtype check -- Q must stay bf16, the cache may be fp8
qsa = sub(
    qsa,
    """        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")""",
    """        if query.dtype != torch.bfloat16 or key_cache.dtype not in _QSA_OK_TORCH_DTYPES:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 query and a BF16/FP8 K/V cache"
            )""",
    "forward dtype check",
)

# 2d. pass the layer's dequant scales to the kernel
qsa = sub(
    qsa,
    """            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )""",
    """            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            *_qsa_kv_scales(layer),
        )""",
    "kernel call scales",
)

# 2e. owner __init__ cache_dtype guard
qsa = sub(
    qsa,
    """        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )""",
    """        if cache_config.cache_dtype not in _QSA_OK_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
                " (set VLLM_QSA_FP8_KV=1 to allow fp8_e4m3)"
            )""",
    "owner cache_dtype guard",
)

# 2f. owner __init__ storage-dtype guard
qsa = sub(
    qsa,
    """        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )""",
    """        if self.kv_cache_torch_dtype not in _QSA_OK_TORCH_DTYPES:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
                " (set VLLM_QSA_FP8_KV=1 to allow fp8_e4m3)"
            )""",
    "owner storage dtype guard",
)

qsa += '''

# --- qwen38-flash-dgx: fp8-e4m3 QSA KV storage (VLLM_QSA_FP8_KV=1) -----------------
# Off by default: with the env var unset the tuples below are exactly the upstream
# allow-lists, so every guard above behaves as it did before.
_QSA_FP8_KV = os.environ.get("VLLM_QSA_FP8_KV", "0").lower() in ("1", "true", "yes")
_QSA_OK_CACHE_DTYPES = ("auto", "bfloat16") + (
    ("fp8", "fp8_e4m3") if _QSA_FP8_KV else ()
)
# vLLM hands back torch.uint8 for "fp8_e4m3": the pages are raw byte storage that each
# backend reinterprets. Accept both spellings so this does not depend on that detail.
_QSA_OK_TORCH_DTYPES = (torch.bfloat16,) + (
    (torch.float8_e4m3fn, torch.uint8) if _QSA_FP8_KV else ()
)
_QSA_SCALE_CACHE: dict = {}


def _qsa_mask_kv_dtype(args, kwargs):
    """Present a non-quantized kv_cache_dtype to FlashAttentionImpl.__init__.

    Returns (args, kwargs, real_dtype_or_None). Only ever acts when fp8 is enabled and
    the value actually looks like an fp8 string in the expected slot, so a signature
    change upstream degrades to a no-op rather than silently corrupting an argument.
    """
    if not _QSA_FP8_KV:
        return args, kwargs, None
    kw = "kv_cache_dtype"
    if kw in kwargs:
        real = kwargs[kw]
        if isinstance(real, str) and real.startswith("fp8"):
            return args, {**kwargs, kw: "auto"}, real
        return args, kwargs, None
    # FlashAttentionImpl(num_heads, head_size, scale, num_kv_heads,
    #                    alibi_slopes, sliding_window, kv_cache_dtype, ...)
    idx = 6
    if len(args) > idx and isinstance(args[idx], str) and args[idx].startswith("fp8"):
        return args[:idx] + ("auto",) + args[idx + 1 :], kwargs, args[idx]
    return args, kwargs, None


def _qsa_kv_scales(layer) -> tuple[float, float]:
    """Dequant scales as Python floats, cached per layer.

    vLLM stores ``fp8 = bf16 / scale``, so dequant multiplies by the same scale.
    Reading the 0-dim buffers every call would force a device sync in the decode hot
    path; they are constant after weight load, so one read per layer is enough.
    """
    key = id(layer)
    cached = _QSA_SCALE_CACHE.get(key)
    if cached is not None:
        return cached

    def _one(name: str) -> float:
        v = getattr(layer, f"_{name}_float", None)
        if v is None:
            t = getattr(layer, f"_{name}", None)
            v = 1.0 if t is None else float(t)
        return float(v) if v else 1.0

    scales = (_one("k_scale"), _one("v_scale"))
    _QSA_SCALE_CACHE[key] = scales
    return scales
'''

# The module-level names must exist before the class bodies that reference them are
# executed, so hoist the block above the first use rather than leaving it at the end.
marker = "\n\n# --- qwen38-flash-dgx: fp8-e4m3 QSA KV storage"
head, tail = qsa.split(marker, 1)
block = marker + tail
anchor = "\nclass Qwen3_8FlashNextQSAMetadataBuilder("
assert head.count(anchor) == 1, "class anchor for hoist not found"
qsa = head.replace(anchor, block.rstrip("\n") + "\n\n" + anchor, 1)

ast.parse(qsa)
QSA.write_text(qsa)
print("qsa.py: fp8 guards relaxed, scales threaded")
