# cohere-transcribe

Ruby 4 transcription for native Transformers Cohere ASR checkpoints. The gem mirrors the Python package's public API and command-line interface while keeping inference entirely in Ruby and native C/C++ ABIs—there is no Python runtime, worker, subprocess, or model server.

The core Dense path supports the pinned Arabic/English model, compatible Hub fine-tunes, and local Dense checkpoints. It includes 16 kHz audio decoding/resampling, Silero v6 ONNX VAD, Auditok-compatible energy VAD, fixed-window segmentation, native batched inference, exact decoder stop/retry metadata, pinned MMS CTC word alignment, subtitle cues, transactional publication, progress callbacks, and reusable model sessions.

## Requirements

- 64-bit Ruby `>= 4.0, < 5.0` (the Dense model exceeds a 32-bit address space)
- 64-bit Linux or macOS; the packaged native source build does not currently support Windows
- CMake 3.14+ and a C++20 compiler when installing from source
- FFmpeg 4–8 shared libraries for the full accepted container/codec set
- `libsndfile` and `libsamplerate` shared libraries for the fallback used when the native FFmpeg adapter is unavailable
- A Hugging Face token when the selected model requires authentication
- Enough storage for the source checkpoint and its cached F16, BF16, or F32 GGUF conversion; word alignment additionally downloads a pinned 1.18 GiB FP32 MMS ONNX model (or 603 MiB FP16 model on CUDA)

The native extension builds a portable CPU backend by default. Accelerator availability depends on the build flags and platform; set `COHERE_TRANSCRIBE_NATIVE_LIBRARY` to use a separately built CrispASR library when needed.

Dense conversion and inference support F16, BF16, and F32 GGUF weights for the pinned checkpoint and compatible Dense fine-tunes. CPU execution resolves to FP32. CUDA `auto` selects BF16 when the loaded runtime reports hardware support and otherwise selects FP16; MPS `auto` selects FP16. An explicit unsupported BF16 accelerator request fails instead of silently changing precision.

Audio decoding never launches the `ffmpeg` executable. The packaged `libcohere_audio` adapter dynamically binds a compatible libavformat/libavcodec/libavutil/libswresample tuple and decodes the first audio stream directly to mono 16 kHz float PCM. This path covers AAC, AIFF, ALAC, FLAC, M4A, MP3, MP4, Ogg/Vorbis, Opus, WAV, WebM, and WMA. Explicit `torchcodec` and `librosa` compatibility modes use the same FFmpeg codec runtime through that C ABI, preserving the complete accepted-format set while reporting the concrete decoder in result provenance. If the adapter is absent, `auto` and `librosa` can fall back to libsndfile/libsamplerate for the formats those libraries support.

The FFmpeg adapter gives duration probes and full decodes monotonic deadlines (30 seconds and one hour, respectively) and shares a generation-based cooperative cancellation hook. It supplies FFmpeg's public `AVIOInterruptCB` to input I/O and also checks cancellation between codec and resampler calls. Cancellation therefore takes effect when FFmpeg invokes that callback or returns control; it does not forcibly unwind an uninterruptible codec or system call.

## Installation

The release artifact is a source gem. RubyGems installs the Ruby dependencies automatically and then builds the native runtime locally. Before continuing, confirm that `ruby --version` reports a 64-bit Ruby 4.x installation.

### Ubuntu or Debian: CPU

```bash
sudo apt update
sudo apt install build-essential cmake ffmpeg libsndfile1 libsamplerate0
gem install cohere-transcribe
```

This creates the portable CPU build. The `ffmpeg` package supplies the shared codec libraries used by the native audio adapter; the gem does not invoke the `ffmpeg` executable or require FFmpeg development headers.

### Ubuntu or Debian: CUDA

Install the NVIDIA driver and CUDA toolkit by following NVIDIA's [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and confirm that `nvcc` is available. CUDA builds require CMake 3.18 or newer.

```bash
sudo apt update
sudo apt install build-essential cmake ffmpeg libsndfile1 libsamplerate0
COHERE_TRANSCRIBE_CUDA=1 gem install cohere-transcribe
```

If the same gem version is already installed as a CPU build, add `--force` to the CUDA installation command so RubyGems rebuilds the native extension.

### macOS: CPU or Metal

Install Apple's command-line build tools and the runtime libraries from Homebrew:

```bash
xcode-select --install
brew install cmake ffmpeg libsndfile libsamplerate
```

For CPU:

```bash
gem install cohere-transcribe
```

For Metal:

```bash
COHERE_TRANSCRIBE_METAL=1 gem install cohere-transcribe
```

The Metal build additionally requires Xcode's `metal` and `metallib` tools; Apple documents the separately installable [Metal toolchain](https://developer.apple.com/documentation/Xcode/downloading-and-installing-additional-xcode-components). Add `--force` when replacing an existing CPU installation of the same version.

### Bundler

Inside an application, `bundle add` can replace `gem install`; the CUDA and Metal environment flags work the same way:

```bash
bundle add cohere-transcribe
```

### Verify the installation

```bash
cohere-transcribe-doctor
cohere-transcribe-doctor --model-access
```

The first command checks the local runtime without loading ASR weights. The second also resolves the selected model metadata and may require `HF_TOKEN`. Model weights are downloaded on first use rather than during `gem install`.

See [the native runtime build notes](ext/cohere_transcribe_native/README.md) for host CPU tuning, OpenMP, parallel build jobs, custom CMake arguments, and external native-library overrides.

## Ruby API

One-shot transcription creates and closes a session automatically:

```ruby
require "cohere/transcribe"

options = Cohere::Transcribe::TranscriptionOptions.new(
  language: "ar",
  device: "auto",
  vad: "silero",
  vad_engine: "auto",
  alignment: "word"
)

run = Cohere::Transcribe.transcribe("speech.wav", options: options)
result = run.single

puts result.text
result.words.each do |word|
  puts "%.2f..%.2f %s" % [word.start, word.end, word.text]
end
```

Retain the model for repeated calls with `Transcriber`:

```ruby
transcriber = Cohere::Transcribe::Transcriber.new(options)
begin
  first = transcriber.transcribe("first.wav")
  second = transcriber.transcribe(["second.wav", "recordings/"])
ensure
  transcriber.close
end
```

Inputs may be a string, a path-like object, or an ordered array of files and directories. Directory expansion is deterministic, recursive by default, and deduplicates canonical paths. A run preserves that expanded order and provides `successful`, `failed`, `skipped`, `single`, and `ok?` helpers.

Per-file media failures are returned as failed results so the rest of a batch can finish. Pass `raise_on_error: true` to raise `BatchTranscriptionError`; its `run` still contains every completed result.

Progress callbacks receive immutable `ProgressEvent` values:

```ruby
callback = ->(event) do
  if event.message
    warn event.message
  elsif event.total
    warn "#{event.stage}: #{event.current}/#{event.total}"
  end
end

run = Cohere::Transcribe.transcribe("speech.wav", progress: callback)
```

## Durable outputs

The in-memory API writes nothing by default. Add `PublicationOptions` to create TXT, SRT, VTT, and JSON files:

```ruby
publication = Cohere::Transcribe::PublicationOptions.new(
  formats: %w[txt srt vtt json],
  output_dir: "transcripts",
  existing: "error", # error, overwrite, or skip
  profile_json: "transcripts/profile.json"
)

options = Cohere::Transcribe::TranscriptionOptions.new(publication: publication)
run = Cohere::Transcribe.transcribe("recordings", options: options)
```

All formats for one source are staged before commit. Existing outputs are preserved if staging fails, and directory-relative structure is retained beneath `output_dir`.

Durable publication binds the planned output root and parent directory inodes, then performs staging, backup, commit, rollback, and cleanup relative to a retained directory descriptor. This prevents a concurrent directory rename or symlink replacement from redirecting transcript, checkpoint, manifest, or profile bytes. Publication fails closed if the planned path identity changes or if the platform lacks `O_NOFOLLOW` and the POSIX `openat`, `renameat`, and `unlinkat` primitives; the supported Linux and macOS targets provide them.

Publication state records the source's canonical path, device, inode, size, nanosecond mtime, and nanosecond ctime, plus checksums for the state payload and published artifacts. With `existing: "skip"`, a verified manifest is resolved before PCM decode, VAD, or a Dense model session is opened; a best-effort metadata duration probe supplies the skipped result's duration without materializing PCM.

When publication is enabled, completed Dense ASR is checkpointed before timing and rendering. A later run validates the source snapshot, ASR contract, and checkpoint contents during the same preflight. Render-only changes can then rebuild segment/none-aligned outputs without decoding audio, running VAD, or opening Dense. Word-aligned resumes likewise skip Dense and VAD preparation, but re-decode through the recorded concrete audio backend because MMS needs the full waveform.

`profile_json` writes the Python-compatible profile schema while keeping controller details private to the profile rather than expanding the stable `TranscriptionStatistics` value. Its ASR section includes effective batch minimum/maximum, final size/cap, batch history, and checkpoint written/resumed counts. Per-file rows retain raw VAD span counts/durations and selected-audio duration even when `vad_merge` combines those spans; JSON transcript output similarly keeps the original spans in `segmentation_details.speech_spans`.

## CLI

```bash
cohere-transcribe interview.wav
cohere-transcribe recordings/ --language ar --formats txt srt vtt json
cohere-transcribe speech.wav --vad auditok --alignment segment
cohere-transcribe speech.wav --vad none --max-dur 30 --text-only
```

Run `cohere-transcribe --help` for the complete Python-compatible 54-option interface. Exit status is `0` on success, `1` when files fail, `2` for command-line errors, `130` for interruption, and `143` for termination.

## Native batching and generation controls

The segment runtime uses CrispASR's padded Cohere encoder and ragged greedy decoder for true native batches. `batch_size`, `batch_max_size`, `batch_audio_seconds`, and `adaptive_batch` control row count, padded-audio budget, and bounded growth. One native session call accepts up to 24 logical rows. Internally it runs consecutive padded-encoder microbatches of at most eight rows, gathers their valid encoder states, and feeds all logical rows to one ragged decoder call. The Ruby controller caps itself to the capacity reported by the loaded session, and the session ABI rejects calls above it. Allocation failures are split recursively without discarding successful rows. Word-aligned requests retain the same native ASR batching. They run as two explicit phases: every file completes ASR with one retained Dense session, then Dense is evicted once before one retained MMS session aligns and publishes the completed files. The phases therefore never co-reside the 2B ASR checkpoint and 300M aligner.

Native inference failures carry a thread-local error kind and diagnostic message across the C ABI. Ruby maps invalid arguments and invariant violations to fatal failures, allocator failures to OOM, and ordinary runtime failures to isolatable errors. Only typed OOM failures teach a smaller adaptive batch cap; fatal failures open the retained session's circuit breaker.

For multi-file runs, `preprocess_workers` concurrently decodes and segments one ordered preparation group while `pipeline_preparation` permits exactly one next group to overlap current ASR. Each pipelined group is capped at the smaller of half `audio_memory_gb` and 512 MiB of retained mono float PCM, so the current and next groups remain within the configured PCM budget; native codec transients can add short-lived overhead. Disable `pipeline_preparation` to use the full per-file budget on a sequential path, which is the escape hatch for a single file larger than the half-budget pipeline slot. Automatic worker selection uses one worker for one file and at most two otherwise; explicit counts are capped by available processors and the group size. Decode/VAD failures remain isolated per file, and results, progress events, and publication always follow input order. Word alignment does not retain those prepared waveforms across its phase barrier. It re-decodes one file at a time through the concrete backend recorded during ASR (with one look-ahead only when the adjacent PCM pair fits `audio_memory_gb`) and rejects backend or sample-count drift. Resumable ASR checkpoints enter the alignment phase directly without reopening Dense or repeating decode/VAD preparation.

`pin_memory` remains accepted for Python API/CLI compatibility, but resolves to `false` in this ggml runtime. Ruby passes float PCM directly through the native session ABI, so there is no PyTorch host tensor to pin and no nonblocking tensor transfer for the option to accelerate.

Generated-token counts come directly from decoder IDs rather than rendered word estimates. Rows that reach `max_new_tokens` without EOS follow `truncation_policy` and retry only the affected rows up to `max_retry_tokens`. `stop_repetition_loops` applies the same conservative 96-token, four-repeat, 8–32-token-period guard as the Python runtime, and all stop/retry decisions are recorded per segment in provenance.

## VAD and timing modes

- `vad: "silero"` runs the packaged Silero v6 ONNX graph with recurrent state and the same sample-domain timestamp state machine as Silero 6.2.1. `auto` and `onnx` select it directly; `torch` and `jit` requests use the equivalent ONNX graph and record that executor substitution in provenance.

The packed-Torch tuning options remain effective without Python. For requested `auto` or `torch`, `vad_block_frames` is the exact maximum number of temporal frames sent to one sequence-ONNX call, and `vad_threads` sets ONNX Runtime's CPU intra-op SessionOption (one thread when omitted). The reference profile schema has no ONNX-session thread field, so the effective value is disclosed in its legacy `vad.torch_intraop_threads` slot. Requested `onnx` or `jit` retains the reference sequence runner's 256-frame block and ignores these packed-only knobs; `vad_threads` remains rejected for those engines as in Python.

The packaged graph has temporal input `[seq_len, 576]` but recurrent h/c inputs fixed at `[1, 1, 128]`; it has no file-batch axis, lengths, or mask. Concatenating files would therefore leak recurrent state between recordings. Ruby instead uses one thread-confined session per active file and makes `vad_batch_size` an upper bound on that independent-file concurrency for requested `auto`/`torch`. The effective count is also bounded by `preprocess_workers`, CPU availability, group size, and the sequential preparation mode. Profiles consequently report `max_files_per_call: 1`, exact temporal model-call/frame counts, and provider options matching ONNX Runtime introspection (`CPUExecutionProvider: {}`), while the configured batch/block values and effective temporal block remain visible.
- `vad: "auditok"` uses a native Ruby implementation of Auditok's 50 ms PCM16 log-RMS tokenizer.
- `vad: "none"` creates bounded fixed windows using `max_dur`.
- `alignment: "word"` computes full-file emissions with the exact pinned MMS-300M forced-aligner ONNX export, then runs a pure-Ruby float32 CTC Viterbi kernel. An unalignable segment alone falls back to bounded uniform timing without dropping transcript words.
- `alignment: "segment"` distributes words uniformly over the segment's speech spans.
- `alignment: "none"` returns plain text without words or cues.

## Models and cache

The default model and revision are pinned:

```text
CohereLabs/cohere-transcribe-arabic-07-2026
0a8193caa4f3f92131471ab08824e488141cb392
```

Hub artifacts reuse the standard Hugging Face cache. Dense Safetensors and `pytorch_model.bin` weights are streamed into a GGUF conversion once and reused from `~/.cache/cohere-transcribe`. PyTorch metadata is decoded by a restricted, allowlist-only Ruby reader; it never imports Python or executes pickle globals. Current `torch.save` ZIP files and the preceding raw-storage stream format are supported, including sharded indexes and strided tensors. Ancient tar-format checkpoints are rejected because they cannot be interpreted with the same restricted weights-only contract; re-save those as Safetensors or a current state dict. The retained-session identity includes the resolved device and dtype as well as the model identity. Each converted artifact has an independent cache key for its output dtype and a SHA-256 source fingerprint over the model ID/revision, relative source paths, sizes, mtimes, and ctimes, so even a same-size, mtime-preserving rewrite invalidates a local fine-tune through its changed ctime. A sidecar completion marker binds that source fingerprint, output dtype, and cache layout to the converted GGUF's device, inode, size, mtime, and ctime. This marker is stored beside the model as `*.complete.json`. The cache accepts only regular, non-symlink GGUF/marker files; conversion locks use no-follow opens where available and verify that the opened descriptor still matches the path's device and inode.

Word mode uses `onnx-community/mms-300m-1130-forced-aligner-ONNX@2100fb247d8e43962eef24491597fbeb8b469531`, an ONNX export of `MahmoudAshraf/mms-300m-1130-forced-aligner@49402e9577b1158620820667c218cd494cc44486`. The default `align_dtype: "fp32"` works with the CPU provider even when Dense ASR runs on CUDA. `align_dtype: "fp16"` requires a CUDA-enabled ONNX Runtime; point `COHERE_TRANSCRIBE_ONNXRUNTIME_LIBRARY` at that runtime when it is not the one supplied by the installed `onnxruntime` gem. The runtime verifies the complete model file against a pinned byte size and SHA-256 before loading it. These downloaded model weights are CC-BY-NC-4.0 and are not distributed inside the gem; see `lib/cohere/transcribe/alignment/ATTRIBUTION.md` for provenance and notices.

Useful environment variables:

- `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT`
- `COHERE_TRANSCRIBE_CACHE`
- `COHERE_TRANSCRIBE_NATIVE_LIBRARY`
- `COHERE_TRANSCRIBE_AUDIO_LIBRARY`
- `COHERE_TRANSCRIBE_GGML_LIBRARY`
- `COHERE_TRANSCRIBE_SNDFILE_LIBRARY`
- `COHERE_TRANSCRIBE_SAMPLERATE_LIBRARY`
- `COHERE_TRANSCRIBE_ONNXRUNTIME_LIBRARY`
- `COHERE_TRANSCRIBE_THREADS`

Advanced deployments may pin the four dynamically loaded codec libraries with `COHERE_TRANSCRIBE_AVFORMAT_LIBRARY`, `COHERE_TRANSCRIBE_AVCODEC_LIBRARY`, `COHERE_TRANSCRIBE_AVUTIL_LIBRARY`, and `COHERE_TRANSCRIBE_SWRESAMPLE_LIBRARY`; set all four together.

### Saved quantization and adapters

Saved bitsandbytes checkpoints and PEFT/LoRA adapters are detected and rejected explicitly; they are not silently treated as Dense weights.

The official [Transformers bitsandbytes contract](https://huggingface.co/docs/transformers/quantization/bitsandbytes) defines INT8 execution with an FP16 outlier path and INT4 NF4/FP4 storage with optional nested quantization. Those layouts and execution rules are not GGML Q8/Q4 tensor formats. A bounded-memory converter could dequantize either format to this gem's Dense GGUF contract, but the result would be a roughly Dense-sized model with Dense inference behavior, not native bitsandbytes support or equivalent memory/performance characteristics.

Likewise, the official [PEFT LoRA contract](https://huggingface.co/docs/peft/main/package_reference/lora) is broader than a pair of low-rank matrices on `Linear` weights: supported layouts can include embedding and convolution adapters, DoRA magnitude vectors, saved module replacements, bias updates, and sparse trainable-token state. A future adapter importer must define and validate a narrower mergeable profile, merge it before GGUF conversion, and bind the adapter configuration and weights into the conversion fingerprint and completion marker. Until that contract exists, explicit rejection avoids a plausible-looking but incomplete merge.

## Development

```bash
bundle install
bundle exec rake test
```

The test suite contains pure-Ruby unit coverage, ABI-boundary tests, real audio decode/resample tests, exact Silero differential tests against Python, converter fixtures, and optional live Dense-model smoke tests.

Installed-gem WER measurements use the standalone runner documented in [`benchmark/README.md`](benchmark/README.md). The inference benchmark is intentionally separate from `bundle exec rake` and CI.

### Releasing

Add a repository Actions secret named `RUBYGEMS_AUTH_TOKEN` containing a RubyGems API key with permission to push `cohere-transcribe`. Publishing a GitHub Release tagged `v0.1.0` runs the release workflow, verifies that the tag matches `Cohere::Transcribe::VERSION`, builds the exact `cohere-transcribe-0.1.0.gem` artifact, and pushes it to RubyGems.org.

Normal CI runs the Ruby suite, style checks, signature validation, native CPU build and ABI smoke checks, and source-gem build on Linux and macOS. It does not run the installed-gem WER or performance benchmarks.

## License

The gem's original Ruby code and documentation are licensed under Apache License 2.0. Packaged third-party components retain their own terms: CC-BY-NC-4.0 for the retained Fairseq MMS and ctc-forced-aligner normalization/span behavior, BSD-2-Clause for the TorchAudio forced-alignment port, Uroman's permissive license for generated romanization data, and MIT for Auditok-derived segmentation, CrispASR, GGML, Silero VAD, and faster-whisper assets. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the license files beside those components.

MMS model weights downloaded when word alignment is used are not distributed in the gem and remain separately licensed CC-BY-NC-4.0. Other runtime-downloaded models remain subject to the terms published by their owners.
