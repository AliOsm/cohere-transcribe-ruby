# Vendored native sources

This is a deliberately reduced source snapshot of [CrispASR](https://github.com/CrispStrobe/CrispASR) at commit `a68dd64092c4c44b41799a15976f5c6d27af13d7` (version 0.8.9).

It contains only the Cohere Transcribe implementation, the shared helpers that implementation uses, and the upstream MIT license and author record. The gem's small session adapter lives in `ext/cohere_transcribe_native/cohere_abi.cpp`.

The nested `ggml` snapshot is from [CrispStrobe/ggml](https://github.com/CrispStrobe/ggml) at commit `0714117daca2471b00e09554c7eaa74a06b0b2c5`. Its build metadata, public headers, core/CPU implementation, and opt-in CUDA and Metal implementations are included. Tests, examples, documentation, and unused accelerator backends were omitted from the gem snapshot.

Both projects are distributed under the MIT license. Their original `LICENSE` and `AUTHORS` files are retained at this directory and `ggml/` respectively.
