# Native Cohere runtime

This extension builds the vendored Cohere backend and ggml into one private `libcrispasr` shared library, plus a small `libcohere_audio` adapter for direct FFmpeg C-ABI decoding. It does not require Python, the `ffmpeg` executable, or a neighboring CrispASR checkout. The audio adapter dynamically supports FFmpeg 4 through 8 runtime libraries and fails cleanly when no compatible tuple is installed; no FFmpeg development headers are needed to build the gem.

The Cohere session ABI exposes a logical batch capacity of 24. A logical call runs the padded encoder in consecutive microbatches of at most eight lanes, gathers their valid states, then invokes the ragged decoder once for all logical lanes. Calls above 24 are rejected. Failures expose a thread-local kind (`invalid_argument`, `out_of_memory`, `invariant`, `runtime`, or `cancelled`) and bounded diagnostic message so Ruby can split only OOM batches and distinguish fatal session errors without parsing text.

Dense inference is cooperatively cancellable through `crispasr_session_cancel`. Cancelling an idle session is a no-op, so a late request cannot poison the next operation. CPU and Metal execution use ggml's abort callback, with additional checks between inference stages and decoder steps. CUDA observes those bounded checkpoints after the current ggml graph returns because its graph execution is not asynchronously interruptible. The Ruby binding runs the foreign call on a private worker so its caller remains interruptible; on `Interrupt` or another caller-side asynchronous exception it requests cancellation, hard-joins that worker, and then re-raises the original caller exception. Session close and other inference calls remain serialized until that worker has fully unwound.

The audio ABI uses `avio_open2` with FFmpeg's public `AVIOInterruptCB` for cooperative cancellation and monotonic deadlines. Duration inspection is a metadata probe and does not materialize PCM. Interruption is cooperative: the callback can wake FFmpeg I/O, while codec/resampler work is checked whenever it returns control to the adapter.

The gemspec should register `ext/cohere_transcribe_native/extconf.rb` in `spec.extensions` and include both `ext/` and `vendor/` in `spec.files`. The build installs the library into `lib/cohere/transcribe/native`, which is already one of the runtime binding's search paths.

Packaged builds define `GGML_BACKTRACE_NO_DEBUGGER`: fatal ggml diagnostics use the platform's in-process backtrace implementation and never fork or execute `gdb`/`lldb`. The upstream debugger-attaching behavior remains unchanged when the vendored ggml snapshot is built outside this gem's CMake project.

CPU is the default. Optional build flags are:

- `COHERE_TRANSCRIBE_NATIVE=1`: tune CPU code for the build machine.
- `COHERE_TRANSCRIBE_OPENMP=1`: enable OpenMP.
- `COHERE_TRANSCRIBE_CUDA=1`: enable CUDA on Linux (CUDA toolkit and CMake 3.18+ required).
- `COHERE_TRANSCRIBE_METAL=1`: enable embedded Metal on macOS (Xcode tools required).
- `COHERE_TRANSCRIBE_NATIVE_JOBS=N`: set CMake build parallelism.
- `COHERE_TRANSCRIBE_CMAKE_ARGS=...`: append advanced CMake arguments.

CUDA follows ggml's upstream architecture set. To make a smaller build for one known GPU, append (for example) `COHERE_TRANSCRIBE_CMAKE_ARGS=-DCMAKE_CUDA_ARCHITECTURES=86`.

Run `make check` after `extconf.rb` to load both produced libraries and verify the C ABIs expected by the Ruby Fiddle bindings.
