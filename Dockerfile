# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and layers a handful of
# patches on it. Every one of them is a no-op unless its runtime flag is set, so
# the image still behaves like upstream with the flags off:
#
#   1. PLE table served from disk via mmap            (VLLM_PLE_MMAP=1)      — the one that makes it fit
#   2. GB10 FLA fixes                                  (always on, harmless elsewhere)
#   3. Mamba state-copy race fix + bounds guard        (always on)
#   4. Prefix-caching block_size fix                   (always on; needed for --enable-prefix-caching)
#   5. Exact, deterministic QSA top-k                  (VLLM_QSA_EXACT_TOPK=1)
#   6. NVFP4 experts + fp8 side-layers "hybrid" mode   (VLLM_FP8_HYBRID=1)
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

# --- 1. PLE n-gram table from disk ------------------------------------------------
COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# --- 2. GB10 FLA fixes, contributed by @Saren-Arterius ----------------------------
#     (https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
# 1) sm_121 reports 99 KiB of shared memory per block; the flash-linear-attention
#    gate asks for 100 KiB, so all 36 GDN layers silently fell back to small tiles.
# 2) fla#953: tl.dot race on Blackwell with num_warps=4 in chunk_delta_h -> pin 2.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# --- 3. Mamba state copy: vllm#50729 + bounds guard --------------------------------
# src/mamba_utils_guarded.py = vLLM's v1/worker/mamba_utils.py with
#   - vllm#50729 "[Bugfix][Mamba] Fix overlapping state copy race" (@AndreasKaratzas), and
#   - a bounds guard by @Saren-Arterius: a state copy with an out-of-range block id is
#     skipped and counted ("mamba state-copy guard: N out-of-range ...") instead of
#     taking down the CUDA context with an illegal memory access.
ARG MU=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MU} ${MU}.orig
COPY src/mamba_utils_guarded.py ${MU}
RUN python3 -c "import ast; ast.parse(open('${MU}').read()); print('mamba_utils.py guarded OK')"

# --- 4. Prefix caching: block_size fix ----------------------------------------------
# vLLM's EngineCore overwrites cache_config.block_size with the smallest KV-group block
# size (here the QSA raw-key ring: 8/16 tokens) while the Mamba block is 1600. Two
# consumers used the former as the latter, so a prefix hit restored an all-zero Mamba
# state. See docs/HOW-IT-WORKS.md. With this, --enable-prefix-caching is correct.
COPY src/patch_mamba_block_size.py /tmp/patch_mamba_block_size.py
RUN python3 /tmp/patch_mamba_block_size.py ${SP} && rm /tmp/patch_mamba_block_size.py

# --- 5. Exact QSA top-k (VLLM_QSA_EXACT_TOPK=1) ---------------------------------------
# The stock persistent_topk kernel is non-deterministic on GB10 and can drop real
# top-k candidates (vllm#51782; reported here by @k3dani, issue #3). The exact path
# uses torch.topk over the visible columns. Opt-in; scripts/serve.sh enables it.
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py && rm /tmp/patch_qsa_exact_topk.py

# --- 6. Hybrid mode: NVFP4 experts + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ----
# Lets the GDN in/out projections, QSA q/k/v/o and shared experts be stored as
# blockwise fp8 (DeepSeek layout, produced by tools/fp8_convert.py) while the MoE
# experts stay NVFP4. Port of @Saren-Arterius's int4+fp8 dispatch to ModelOpt-NVFP4.
# No-op unless VLLM_FP8_HYBRID=1 (and harmless on a plain NVFP4 checkpoint).
ARG MO=${SP}/vllm/model_executor/layers/quantization/modelopt.py
ARG QSA=${SP}/vllm/models/qwen3_8_flash_next/nvidia/qsa.py
COPY src/vllm_fp8_hybrid_modelopt.py ${SP}/vllm_fp8_hybrid_modelopt.py
RUN cp ${MO} ${MO}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: NVFP4 + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${MO} \
 && python3 -c "import ast; ast.parse(open('${MO}').read()); print('modelopt.py hooked OK')" \
 && cp ${QSA} ${QSA}.orig \
 && sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' ${QSA} \
 && sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded/' ${QSA} \
 && grep -q "_fp8_hybrid_excluded(quant_config)" ${QSA} && grep -q "^from vllm_fp8_hybrid_modelopt import" ${QSA} \
 && python3 -c "import ast; ast.parse(open('${QSA}').read()); print('qsa.py hooked OK')"
