# Running Qwen3.8-Flash-Next (NVFP4 + MTP) on a single NVIDIA DGX Spark

Field notes from bringing up **Qwen3.8-Flash-Next** — a ~176B-parameter multimodal
MoE (125B main + 51B n-gram embedding table, 6B active per token) — on **one**
DGX Spark / GB10, using vLLM with the model's built-in MTP speculative decoding.

Recipe source: [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(patched official vLLM image `vllm/vllm-openai:qwen38-flash-next`). The recipe files
(`Dockerfile`, `src/`, `tools/`, `scripts/`) are **vendored in this repo** so it is
self-contained; they remain Apache-2.0 © blazux (see `LICENSE`).
`scripts/gateway.py` and `scripts/serve-public.sh` are additions of this repo, not
part of that recipe.
Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4).

## Local single-stream 524k profile (2026-09-05)

```bash
docker build -f Dockerfile.performance -t qwen38-flash-dgx:performance .
bash scripts/serve-500k.sh
# After /health succeeds:
python3 scripts/bench-single-stream.py
python3 scripts/validate-context.py --tokens 500000
```

The incremental image requires the existing `qwen38-flash-dgx:latest` image.
On a clean machine, first build the main Dockerfile, download weights and run
`scripts/prepare-hybrid.sh`. This profile uses the prepared hybrid checkpoint,
524,288 total context tokens, YaRN factor 2, MTP=3, one concurrent sequence,
prefix caching, exact QSA top-k, and an explicit 16 GiB **BF16** KV pool.
The API listens on `127.0.0.1:18300`. The separate 27B service on port 8080
is not connected to this endpoint.

This profile prioritizes per-stream speed. BF16 avoids FP8 KV dequantization
and its effect on speculative acceptance. Hybrid weight quantization remains
enabled. Exact top-k retains the local correctness fix; `EXACT_TOPK=0` can
be faster but restores the stock kernel's known candidate-selection issue.
`SEQS`, `MTP`, and other launcher variables can still be overridden.

Inspired by [MiaAI-Lab's single-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
PLE mappings now use `MADV_RANDOM` to avoid reading adjacent pages for scattered
lookups. Set `VLLM_PLE_MMAP_RANDOM=0` for an A/B comparison. YaRN scaling is
derived from the requested context and preserves the checkpoint's remaining
RoPE parameters. Context beyond 524,288 is refused, as is context beyond
262,144 without YaRN.

`KV_BYTES` explicitly sizes KV and bypasses `GPU_MEM` sizing; raising it consumes
the desktop's shared memory. This is a one-model-at-a-time profile. A 500,000-token
prompt leaves 24,288 tokens for output. The validation script checks actual API
usage and records TTFT and host memory; its repeated archive text is a capacity
test, not evidence of reliable retrieval across arbitrary 500k documents.
`DRY_RUN=1 bash scripts/serve-500k.sh` prints the launch without replacing a container.
The single-stream benchmark uses three sequential short prompts, thinking disabled,
and 384 generated tokens per prompt; it writes `/tmp/flash-single-stream.json`.
The capacity test writes `/tmp/flash-context-validation.json`.

Validated on this host: the 16 GiB KV pool holds **587,382 tokens**. An exact
**500,000-token prompt** completed successfully and returned the correct arithmetic
answer, with **414.3 s TTFT** and **10.24 GiB minimum MemAvailable**. Short-prompt
single-stream decode measured **26.3–33.4 tok/s**; warm TTFT was **0.24–0.30 s**
(first request: 3.16 s). These are final-profile measurements, not a matched
speedup comparison against the old profile. Full prompts, outputs, configuration,
and usage are in [the validation report](docs/performance-2026-09-05.json).

### FP8 KV memory profile

```bash
bash scripts/serve-500k-fp8.sh
# Restore the BF16 profile:
bash scripts/serve-500k.sh
```

The FP8 profile uses the same image, hybrid weights, 524,288 context, MTP=3,
single sequence, and exact top-k. It sets `KV_DTYPE=fp8_e4m3` and a **9 GiB**
KV pool, saving **7 GiB of allocation** versus the BF16 profile. The installed
QSA patch stores the main KV pages in FP8 and converts gathered tiles to BF16
inside the attention kernel, with FP32 accumulation. It does not allocate a
second full BF16 cache. The side/compressor caches remain BF16. Conversion
does not recover precision lost during FP8 storage.

Measured on 2026-09-05 with the same three short prompts and generation settings:

| Metric | BF16 KV | FP8 KV |
| --- | ---: | ---: |
| KV allocation | 16 GiB | 9 GiB |
| KV token capacity | 587,382 | 603,639 |
| Database prose decode | 26.3 tok/s | 26.4 tok/s |
| Code decode | 33.4 tok/s | 34.8 tok/s |
| TCP prose decode | 26.5 tok/s | 24.4 tok/s |
| Warm TTFT | 0.24–0.30 s | 0.26–0.31 s |
| 500k synthetic-prompt TTFT | 414.3 s | 411.0 s |
| Minimum available RAM during 500k test | 10.24 GiB | 16.95 GiB |

Both profiles completed exactly 500,000 prompt tokens and returned the correct
arithmetic answer. Short-prompt performance was mixed and broadly comparable in
these three single-run samples; generated text differed, affecting MTP acceptance.
This is a capacity and speed comparison, with general reasoning/retrieval quality
equivalence still unverified. FP8 was left running after validation.
See [the FP8 comparison report](docs/performance-fp8-2026-09-05.json) for raw outputs,
usage, timings, configuration, and limitations.

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
git clone https://github.com/madeye/qwen38-flash-next-on-dgx-spark.git
cd qwen38-flash-next-on-dgx-spark
docker build -t qwen38-flash-dgx .
hf download RadixArk/Qwen3.8-Flash-Next-NVFP4   # ~122 GiB, resumable
scripts/serve.sh            # MODE=nvfp4, MTP=2, prefix caching, exact top-k
scripts/smoke-test.sh
scripts/serve-public.sh     # optional: loopback vLLM + authenticating gateway on :8080
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

## Serving it publicly

`scripts/serve.sh` publishes the API on `0.0.0.0` with no authentication of its
own — fine on a private box, not something to leave on a LAN. `serve-public.sh`
pins the container's port to loopback instead and fronts it with `gateway.py`:

```bash
scripts/serve-public.sh              # container + gateway on 0.0.0.0:8080
MODE=hybrid scripts/serve-public.sh  # every serve.sh variable passes through
GW_PORT=9000 scripts/serve-public.sh
```

The gateway proxies `/v1/*` and `/metrics`, requires `Authorization: Bearer
<key>` on every one of them, and 404s everything else — vLLM's other routes
(`/tokenize`, `/sleep`, the shutdown endpoints) never reach the public
interface. Streaming passes through chunk-by-chunk, so SSE latency is
unaffected. It needs only `aiohttp`, declared inline in the script, so
`uv run scripts/gateway.py` installs nothing permanently.

Keys live in the gateway rather than in the container's argv, which is the
point: rotating one takes effect on the next request instead of restarting a
container that loads ~76 GiB of weights over ~10 minutes. Startup prints the
dashboard URL with its admin token:

```
gateway    0.0.0.0:8080  ->  http://127.0.0.1:18300
api base   http://192.168.0.4:8080/v1
dashboard  http://192.168.0.4:8080/?token=<admin token>
```

The dashboard shows upstream health and the served model, lets you edit the
advertised API base URL (override it if you front the gateway with a tunnel or
domain), and manages keys — create, label, reveal, copy, rotate, revoke, with
per-key request counts and last-used times. It also renders a ready-to-paste
curl and OpenAI-SDK snippet. State lives in `gateway.json` (mode 0600,
gitignored); the admin token is generated on first run and persists there.

Ctrl-C stops the gateway but leaves the container running — it is detached with
`--restart unless-stopped`, and a 10-minute weight load is not worth throwing
away on a terminal hangup. Stop it with `docker rm -f qwen38-flash`.

#### Getting and setting keys

Three equivalent routes, in rough order of convenience:

```bash
# the dashboard: show / copy / rotate / revoke, per key
python3 -c "import json;d=json.load(open('gateway.json'));\
print(f\"http://127.0.0.1:8080/?token={d['admin_token']}\")"

# the admin API
curl -s -H "X-Admin-Token: $TOK" localhost:8080/admin/state          # list
curl -s -H "X-Admin-Token: $TOK" -d '{"label":"laptop"}' \
     -H 'Content-Type: application/json' localhost:8080/admin/keys   # create
curl -s -H "X-Admin-Token: $TOK" -d '{}' \
     localhost:8080/admin/keys/<id>/rotate                           # rotate

# or just edit gateway.json -- the only way to set a *chosen* value,
# since the dashboard and API only generate random ones
```

`gateway.json` is re-read within a second of changing, so a hand-edited key,
`public_url`, or `admin_token` takes effect on the next request with no restart.
Per-key request counters are preserved across a reload, an unparseable file is
ignored (and warned about once) rather than crashing the gateway, and emptying
the `keys` list regenerates one instead of locking everyone out.

**This is bearer auth over plain HTTP.** It is enough for a trusted LAN. Before
exposing it further, put TLS in front of it — a Cloudflare tunnel, Tailscale, or
a reverse proxy — and set the dashboard's API base URL to that public address.

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
