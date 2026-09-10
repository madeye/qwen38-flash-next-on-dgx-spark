# MiaAI draft vocabulary

`draft_vocab_en_code_47k.txt` is copied unchanged from
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
at commit `d03809008834124e80223c3482f2ddb59577a48f` (2026-09-09).
It contains 47,149 token IDs selected from the upstream author's code and docs.
The upstream AGPL-3.0 license is preserved in this directory as `LICENSE`.

Only the token-ID data is vendored here. The NVIDIA MTP implementation in
`../full-recipe-patch/mtp_draft_vocab.py` independently supports an arbitrary
token-ID selection as well as the original numeric vocabulary prefix.

The target retains its full vocabulary. Coverage of a draft selection can
affect acceptance and speed, especially for non-code or non-English text.
Upstream's speed measurements are not measurements of this NVIDIA checkpoint.
