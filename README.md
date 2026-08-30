# Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark

Field notes from bringing up **Qwen3.8-Flash-Next** — a ~176B-parameter multimodal
MoE (125B main + 51B n-gram embedding table, 6B active per token) — on **one**
DGX Spark / GB10, using vLLM with the model's built-in MTP speculative decoding.

Recipe source: [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(patched official vLLM image `vllm/vllm-openai:qwen38-flash-next`).
Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).

## Hardware

- NVIDIA DGX Spark (GB10, Blackwell sm_121), 128 GB **unified** memory (~116 GiB free)
- 20-core Grace CPU (Cortex-X925 @ 4.0 GHz), aarch64
- Driver 580.82.09, CUDA 13.0, Docker 28.3.3, 3.7 TB NVMe

## Why it fits at all

The NVFP4 checkpoint is 122 GiB — larger than the usable unified pool once you
add KV cache. The trick from the recipe: 44 GiB of that is the n-gram embedding
("PLE") table, a pure lookup that a token only touches 16 rows of. The patched
image serves it from NVMe via `mmap` instead of keeping it resident:

- Resident weights drop to **~76 GiB**; the rest of the pool is KV cache.
- On unified memory, "CPU offload" saves nothing (same pool) — only serving
  from disk actually frees memory.

## Setup (as run)

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX
docker build -t qwen38-flash-dgx .
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4   # ~122 GiB, resumable
scripts/serve.sh            # MODE=nvfp4, MTP=2, prefix caching, exact top-k
scripts/smoke-test.sh
```

Effective serving config: native 262,144-token context, MTP=2 speculative tokens,
`--enable-prefix-caching`, deterministic exact QSA top-k, 8 concurrent sequences,
`--gpu-memory-utilization 0.85`, PIECEWISE CUDA graphs (the mmap'd PLE gather is a
splitting op), bf16 KV cache (fp8 KV is refused by the QSA layers).

## Measured results (single request, greedy)

### MODE=nvfp4 (checkpoint as published)

| Metric | Result |
| --- | --- |
| First boot (weight load) | ~10 min |
| Prefill, cold, 10.7k-token prompt | **1,042 tok/s** |
| Same prompt again (prefix-cache hit) | 1.48 s TTFT |
| Determinism at T=0 | first-token logprobs identical across runs |
| Decode, 400-token real answer incl. TTFT | **19.1 tok/s** |

### MODE=hybrid (NVFP4 experts + blockwise-fp8 side layers)

One-time prep: `scripts/prepare-hybrid.sh` (~10 min, +13 GB disk) — 300 dense
side-layer tensors (GDN in/out projections, QSA q/k/v/o, shared experts) converted
bf16 → fp8-e4m3, worst per-tensor max relative error 3.54%. Routed experts stay
NVFP4.

| Metric | nvfp4 | hybrid |
| --- | --- | --- |
| Prefill, cold, 10.7k-token prompt | 1,042 tok/s | 916 tok/s |
| Prefix-cache hit TTFT | 1.48 s | 1.50 s |
| Deterministic at T=0 | yes | yes |
| Decode, 400-token real answer incl. TTFT | 19.1 tok/s | **21.6 tok/s (+13%)** |

Hybrid trades a little cold-prefill speed for meaningfully faster decode and
~7 GiB less resident weight — the right default for an interactive/agentic box.

## Reducing memory footprint on a MoE (what actually works here)

- **PLE table mmap** (already on): −44 GiB, the single biggest win.
- **Hybrid mode**: −7 GiB more, plus faster decode.
- **Lower `GPU_MEM` to 0.80** for long-running service: the recipe authors saw
  0.85 drift into swap after a day.
- **Lower `CTX`/`SEQS`** if you don't need 262k context — KV is the other big
  consumer.
- Expert streaming/offload is **not** useful on this chip: unified memory means
  host RAM is the same pool, and experts are touched every token so disk paging
  would thrash. vLLM has no expert-mmap path anyway.
- GGUF IQ3/IQ2 quants via llama.cpp shrink further but cost MTP, prefill speed
  (~540 tok/s vs ~1,000+), and quality.

## DFlash note

No DFlash drafter exists for this checkpoint (the [z-lab/dflash](https://github.com/z-lab/dflash)
zoo covers Qwen3.5/3.6 and Qwen3.8-27B). Not needed: the model ships a built-in
MTP head, enabled in vLLM with
`--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`.

## Known limitations (from the recipe, confirmed relevant)

- One big model at a time — this uses most of the 128 GB pool.
- 1M context is out of reach on one box; 500k with YaRN is the validated ceiling
  (`YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve.sh`).
- The stock GB10 `persistent_topk` kernel is non-deterministic — the image's
  `EXACT_TOPK=1` default fixes it at some long-prefill cost.
