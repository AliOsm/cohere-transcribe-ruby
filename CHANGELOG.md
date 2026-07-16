# Changelog

## [0.1.3] - 2026-07-16

- Close every libsndfile handle and native session exactly once, preserve caller interruption through FFmpeg cancellation cleanup, reject empty native inference rows locally, and consolidate detached session ownership without retaining wrapper objects.
- Match native FFmpeg mono mixing for common multichannel layouts, classify decoded-audio ceiling failures explicitly, prepare unknown sizes concurrently, retry missing or low estimates sequentially with the full ceiling, and limit metadata probing to the current and next groups.
- Restore warm-cache Hub operation for temporary connection, rate-limit, server, and malformed-response failures while retaining online branch revalidation, short request coalescing, and definitive authentication or missing-repository errors.
- Reject invalid byte paths before transcription or publication, treat publication-parent changes during read-only verification as reprocessing decisions, reject inverted segment bounds, and keep cross-process output locks in the stable per-user cache.
- Combine full-storage tensor checksum validation with conversion, simplify ZIP64 locator and checksum bookkeeping, preserve the discovered CUDA toolkit runtime path, and exclude untracked native build artifacts from locally built gems.
- Match the pinned Python word-aligner geometry for sub-30-second audio, require exact command-line enum values, and make the public `OUTPUT_FORMATS` constant directly usable with `PublicationOptions`.

## [0.1.2] - 2026-07-16

- Correct macOS creation-mode delivery for descriptor-relative staged files and turn concurrent output replacement during preflight into a normal reprocessing decision.
- Correct ZIP and ZIP64 boundary handling, verify streamed PyTorch tensor storage checksums before conversion, and make vocabulary validation bounded for malformed declared shapes.
- Close abandoned and partially constructed native sessions exactly once, release native decoder metadata allocations deterministically, and guarantee decoded FFmpeg buffers are released when interruption arrives at the native return boundary.
- Make preparation groups size-aware so long files retain the full single-file memory ceiling without overlapping another group, unify stereo downmix energy across audio backends, and return typed public errors for binary-encoded Unix paths.
- Keep diagnostics and early command-line interruption contained, revalidate symbolic Hub revisions while online with an explicit offline cache mode, require CMake 3.15+, and install and exercise the built source gem on both CI platforms.

## [0.1.1] - 2026-07-15

- Fix CUDA source-gem installation across Unix Makefiles and Ninja by making NVCC's generated response file available from each compiler working directory, including when the gem is installed beneath a path containing spaces.
- Preserve RubyGems build parallelism when the native extension delegates compilation to CMake.

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
