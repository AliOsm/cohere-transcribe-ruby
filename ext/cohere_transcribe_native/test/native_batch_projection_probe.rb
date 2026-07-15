# frozen_string_literal: true

# Focused real-model timing probe for one retained native session.
#
# Usage:
#   ruby native_batch_projection_probe.rb \
#     LIBCRISPASR LIBCOHERE_AUDIO MODEL_GGUF AUDIO [BATCH_SIZE] [ITERATIONS]

require "digest"
require "fiddle"
require "json"

native_library, audio_library, model_path, audio_path, batch_text, iterations_text = ARGV
unless native_library && audio_library && model_path && audio_path
  warn "usage: #{$PROGRAM_NAME} LIBCRISPASR LIBCOHERE_AUDIO MODEL_GGUF AUDIO [BATCH_SIZE] [ITERATIONS]"
  exit 2
end

batch_size = Integer(batch_text || 24)
iterations = Integer(iterations_text || 2)
raise ArgumentError, "BATCH_SIZE must be between 1 and 24" unless batch_size.between?(1, 24)
raise ArgumentError, "ITERATIONS must be positive" unless iterations.positive?

ENV["COHERE_TRANSCRIBE_NATIVE_LIBRARY"] = File.expand_path(native_library)
ENV["COHERE_TRANSCRIBE_AUDIO_LIBRARY"] = File.expand_path(audio_library)

$LOAD_PATH.unshift(File.expand_path("../../../lib", __dir__))
require "cohere/transcribe"

options = Cohere::Transcribe::TranscriptionOptions.new(
  device: ENV.fetch("COHERE_PROBE_DEVICE", "cuda"),
  dtype: "fp16",
  vad: "none",
  alignment: "segment",
  max_new_tokens: 445
)
samples = Cohere::Transcribe::Audio::Decoder.decode(audio_path, backend: "ffmpeg").samples
session = Cohere::Transcribe::ASR::NativeSession.new(model_path, options, threads: 6)
warmup_iterations = Integer(ENV.fetch("COHERE_PROBE_WARMUP_ITERATIONS", "0"))
raise ArgumentError, "COHERE_PROBE_WARMUP_ITERATIONS must not be negative" if warmup_iterations.negative?

profiler = if ENV["COHERE_PROBE_CUDA_CAPTURE"] == "1"
             handle = Fiddle::Handle.new("libcudart.so.12")
             [
               handle,
               Fiddle::Function.new(handle["cudaProfilerStart"], [], Fiddle::TYPE_INT),
               Fiddle::Function.new(handle["cudaProfilerStop"], [], Fiddle::TYPE_INT)
             ]
           end
profiler_started = false

begin
  warmup_iterations.times do
    session.transcribe_batch(Array.new(batch_size, samples), language: "ar")
  end
  if profiler
    status = profiler[1].call
    raise "cudaProfilerStart failed with status #{status}" unless status.zero?

    profiler_started = true
  end

  iterations.times do |iteration|
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    output = session.transcribe_batch(Array.new(batch_size, samples), language: "ar")
    elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
    puts JSON.generate(
      iteration: iteration + 1,
      batch_size: batch_size,
      elapsed_seconds: elapsed,
      text_sha256: Digest::SHA256.hexdigest(output.map(&:text).join("\0")),
      generated_tokens: output.map(&:generated_tokens)
    )
  end
ensure
  if profiler_started
    status = profiler[2].call
    warn "cudaProfilerStop failed with status #{status}" unless status.zero?
  end
  session.close
end
