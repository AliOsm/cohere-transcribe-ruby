# Changelog

## [0.1.0] - 2026-07-15

- Release the independent Ruby gem with output schema 8 and profile schema 9.
- License original gem code and documentation under Apache License 2.0 and package Ruby-specific notices for every retained third-party component.
- Add Ruby 4 Linux/macOS CI with native ABI checks, monthly dependency updates, and GitHub Release publishing to RubyGems.org; WER and performance benchmarks remain outside CI.
- Unify directory-sync, staged-output cleanup, rollback reporting, backup preservation, target-mode lookup, and finite-JSON behavior across transcript, profile, checkpoint, and manifest publication.
- Add bounded parallel feature extraction, fused Conformer SiGLU execution, flattened decoder projections, device-side token selection, and the fast padded-encoder projection path. Across three fresh-process runs of the 69.34-minute audio benchmark, the final CUDA path has a median throughput of 118.82x real time; the matching Python package has a median throughput of 119.64x.
- Add a standalone installed-gem WER benchmark that remains outside the test suite and CI. On the balanced-500 corpus the final public API scores 22.7534% lexical WER and the native B24 lane scores 23.1724%, compared with 22.9551% for the retained Python BF16 reference.
- Refresh the reproducible Ruby 4 source gem, isolated install checks, native ABI checks, and real CPU/CUDA Dense-model coverage.
- Port the complete public Ruby API and 54-option CLI from the Python package.
- Add native Dense Safetensors/PyTorch-state-dict-to-GGUF conversion for F16, BF16, and F32 weights, including compatible Hub and local Dense fine-tunes. Cache keys fingerprint dtype and source metadata, while completion markers bind the source fingerprint to the converted file's filesystem identity.
- Add 24-row logical native ASR batches: padded encoder work is microbatched at eight rows and gathered into one ragged decoder call. Include bounded adaptive growth, typed native failure diagnostics, OOM-only cap learning and recursive failure splitting, exact EOS/token-limit provenance, targeted token retries, and repetition stopping.
- Add ordered, memory-bounded decode/VAD worker groups with one-group ASR look-ahead, worker-confined Silero sessions, and deterministic error isolation.
- Make packed-compatible Silero tuning effective through the packaged ONNX substitute: `vad_threads` configures CPU intra-op execution, `vad_block_frames` bounds temporal calls exactly, and `vad_batch_size` caps independent-file session concurrency. Report exact single-stream call/frame telemetry and ONNX provider introspection in JSON and profiles.
- Add two-phase multi-file word execution with one Dense session, a hard Dense/MMS eviction barrier, bounded concrete-backend PCM reload, and direct checkpoint-to-alignment resume.
- Add source-bound ASR checkpoints and preflight verification. Verified skips avoid PCM decode, while render-only segment/none resumes avoid decode, VAD, and Dense model loading. Profiles receive private batch/checkpoint telemetry and raw-versus-selected speech-span measurements without changing the public statistics type.
- Add exact Silero v6 ONNX, Auditok-compatible energy, and fixed-window VAD.
- Add subprocess-free FFmpeg 4–8 C-ABI audio decoding for all 15 accepted media extensions, cooperative cancellation and decode/probe deadlines, metadata duration probing, plus libsndfile/libsamplerate fallback, word/segment timing, subtitle rendering, transactional publication, profiling, diagnostics, and immutable result types.
- Add source-buildable native runtime packaging for Ruby 4 on Linux and macOS.
