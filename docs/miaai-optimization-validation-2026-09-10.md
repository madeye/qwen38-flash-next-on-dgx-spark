# MiaAI optimization integration, 2026-09-10

Source: [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark,
d03809008834124e80223c3482f2ddb59577a48f](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/commit/d03809008834124e80223c3482f2ddb59577a48f).

Applied to the existing NVIDIA checkpoint and pinned vLLM `8a728663` on one
GB10, TP1. DFlash remains disabled; native MTP uses three speculative tokens.

## Changes

- Vendored the upstream 47,149-token code vocabulary with its AGPL-3.0 license
  and provenance. SHA256:
  `20e36b6e8eae2598019298959a578ef8adc2948bbed7189e43a8da9b9d84a0b1`.
  It includes 10,247 IDs outside the previous `[0, 65536)` selection.
- Extended the local NVIDIA MTP head implementation to select arbitrary token
  IDs and scatter the resulting logits into the original full vocabulary.
  Numeric prefix selections and full-vocabulary fallback remain available.
- Enabled BF16 GDN recurrent state; added a setting to restore FP32.
- Derived decode graph widths automatically from MTP depth and sequence count.
  The six-sequence default still captures widths `[4,8,12,16,20,24]`.

The NVIDIA checkpoint and staged disk-backed PLE implementation were retained.
MiaAI's packed PLE loader targets a different checkpoint format.

## Checks completed

- Shell syntax, Python parsing, and whitespace checks.
- Launcher dry runs: defaults, eight sequences (widths through 32), numeric
  vocabulary, full-vocabulary fallback, checkpoint state-precision fallback,
  and rejection of missing/out-of-range vocabularies and invalid graph sizes.
- PyTorch CPU comparison against the full-head reference: arbitrary and prefix
  selections, global token mapping, excluded-token masking, repeated calls,
  and partitioned head contributions. Invalid token IDs were rejected.
- On-device GB10 test: BF16 selected-head logits captured in a CUDA graph and
  replayed against changing inputs, checked against the full-head reference.
- Installed vLLM accepts `mamba_ssm_cache_dtype=bfloat16`.
- Live startup at 05:57 UTC confirmed `MTP draft vocabulary: 47149/248320 rows`,
  attention block size 1664 (previously 3200 with FP32 state), and successful
  graph capture (0.14 GiB). Model loading reported 76.48 GiB.

## Live serving validation

The final container started at 05:59 UTC with a 30 GiB host reserve and
`gpu_memory_utilization=0.7535`. It completed startup with 7.44 GiB available
for KV cache and a reported 507,810-token capacity (1.94x a 262,144-token request).
DFlash remained disabled; TP1, MTP3, the 47K vocabulary, BF16 state, FP8 KV,
six sequence slots, and 4,096-token chunks were active.

At 11:21–11:23 UTC, all 16 requests in the
[live performance measurement](performance-live-2026-09-10-112330.json) succeeded:

| Workload | Speed | First token |
| --- | ---: | ---: |
| Prose, one request | 36.83 output tok/s | 0.236 s |
| Code, one request | 45.75 output tok/s | 0.201 s |
| Four mixed requests | 100.18 output tok/s aggregate, including prefill | 1.017 s median |
| Fresh 8,215-token input | 1,606.5 input tok/s | 5.114 s |
| Fresh 32,791-token input | 1,824.95 input tok/s | 17.968 s |

Single-request decode medians use three 384-token samples per workload, excluding
first-token latency. The concurrency result is one four-request batch. Prefill
medians use two fresh documents per size, with prefix caching disabled. The
report includes raw outputs, token counts, timing definitions, and limitations.
No other completed generation requests were observed in the metric deltas.

The server was healthy with zero restarts after the run. Available host memory
bottomed at 19.94 GiB; aggregate MTP draft acceptance was 65.7%. These measurements
are not a matched comparison against the old recipe, and do not establish
quality equivalence or behavior at the full 262K context limit.

## Memory-budget failure reproduced

An earlier launch at `gpu_memory_utilization=0.7042` (36 GiB host reserve) passed
model loading and graph capture but failed KV sizing: 0.97 GiB usable versus
3.84 GiB required for one 262,144-token request. The estimated maximum length
was 44,928. This was a capacity check, not a CUDA OOM. The final 30 GiB reserve
provides enough KV capacity while keeping more host headroom than `GMU=0.80`.

## Additional quality checks

A separate functional/retrieval harness is available:

```bash
python3 tools/validate_miaai_update.py \
  --output docs/miaai-serving-check-2026-09-10.json
```

It covers arithmetic, Chinese output, six simultaneous requests, three facts at
different depths of a long prompt, and a follow-up recall turn. This harness has
not been run; its checks must not be inferred from the successful performance run.
