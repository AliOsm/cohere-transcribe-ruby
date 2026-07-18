# frozen_string_literal: true

require "fiddle"
require "rbconfig"
require "tmpdir"

directory = File.expand_path(ARGV.fetch(0))
patterns = case RbConfig::CONFIG.fetch("host_os")
           when /darwin/ then ["libcrispasr*.dylib"]
           when /mswin|mingw/ then ["crispasr*.dll", "libcrispasr*.dll"]
           else ["libcrispasr.so", "libcrispasr.so.*"]
           end
library = patterns.flat_map { |pattern| Dir.glob(File.join(directory, pattern)) }.min
abort "no packaged libcrispasr found in #{directory}" unless library

handle = Fiddle::Handle.new(library, Fiddle::RTLD_NOW | Fiddle::RTLD_GLOBAL)
symbols = %w[
  crispasr_bf16_to_fp32_row
  crispasr_fp16_to_fp32_row
  crispasr_fp32_to_fp16_row
  crispasr_fp32_to_bf16_row
  crispasr_last_error_kind
  crispasr_last_error_message
  crispasr_set_gpu_backend
  crispasr_runtime_resolve_device
  crispasr_runtime_supports_bf16
  crispasr_session_open_with_params
  crispasr_session_backend
  crispasr_session_compute_backend
  crispasr_session_memory
  crispasr_session_batch_capacity
  crispasr_session_cancel
  crispasr_session_transcribe_lang
  crispasr_session_transcribe_batch_lang
  crispasr_session_batch_result_count
  crispasr_session_batch_result_stats_v1
  crispasr_session_batch_result_at
  crispasr_session_batch_result_free
  crispasr_session_result_n_segments
  crispasr_session_result_segment_text
  crispasr_session_result_segment_t0
  crispasr_session_result_segment_t1
  crispasr_session_result_n_words
  crispasr_session_result_word_text
  crispasr_session_result_word_t0
  crispasr_session_result_word_t1
  crispasr_session_result_word_p
  crispasr_session_result_generated_tokens
  crispasr_session_result_generation_limit
  crispasr_session_result_generation_capacity
  crispasr_session_result_stopped_by_max_tokens
  crispasr_session_result_repetition_stopped
  crispasr_session_result_free
  crispasr_session_close
  crispasr_session_set_max_new_tokens
  crispasr_session_set_beam_size
  crispasr_session_set_repetition_loop_guard
]
symbols.each { |symbol| handle[symbol] }

backend = Fiddle::Function.new(handle["crispasr_session_backend"], [Fiddle::TYPE_VOIDP], Fiddle::TYPE_VOIDP)
compute_backend = Fiddle::Function.new(
  handle["crispasr_session_compute_backend"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_VOIDP
)
resolve_device = Fiddle::Function.new(
  handle["crispasr_runtime_resolve_device"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_VOIDP
)
supports_bf16 = Fiddle::Function.new(
  handle["crispasr_runtime_supports_bf16"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_INT
)
segments = Fiddle::Function.new(
  handle["crispasr_session_result_n_segments"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_INT
)
batch_capacity = Fiddle::Function.new(
  handle["crispasr_session_batch_capacity"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_INT
)
cancel_session = Fiddle::Function.new(
  handle["crispasr_session_cancel"],
  [Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_INT
)
last_error_kind = Fiddle::Function.new(
  handle["crispasr_last_error_kind"],
  [],
  Fiddle::TYPE_INT
)
last_error_message = Fiddle::Function.new(
  handle["crispasr_last_error_message"],
  [],
  Fiddle::TYPE_VOIDP
)
batch_transcribe = Fiddle::Function.new(
  handle["crispasr_session_transcribe_batch_lang"],
  [Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP],
  Fiddle::TYPE_VOIDP
)
max_tokens = Fiddle::Function.new(
  handle["crispasr_session_set_max_new_tokens"],
  [Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT],
  Fiddle::TYPE_INT
)
raise "initial native error kind must be zero" unless last_error_kind.call.zero?
raise "initial native error message must be empty" unless Fiddle::Pointer.new(
  last_error_message.call
).to_s.empty?
raise "null session backend must be an empty C string" unless Fiddle::Pointer.new(backend.call(0)).to_s.empty?
raise "null session compute backend must be an empty C string" unless Fiddle::Pointer.new(
  compute_backend.call(0)
).to_s.empty?

cpu = Fiddle::Pointer["cpu\0"]
auto = Fiddle::Pointer["auto\0"]
raise "CPU must always resolve" unless Fiddle::Pointer.new(resolve_device.call(cpu)).to_s == "cpu"
raise "CPU precision contract must use FP32" unless supports_bf16.call(cpu).zero?
unless %w[cpu cuda mps].include?(Fiddle::Pointer.new(resolve_device.call(auto)).to_s)
  raise "auto device resolution returned an unsupported value"
end
raise "null result must contain no segments" unless segments.call(0).zero?
raise "null session must have no batch capacity" unless batch_capacity.call(0).zero?
raise "null session cancellation must fail" unless cancel_session.call(0) == -1
raise "null session setter must fail" unless max_tokens.call(0, 1) == -1

failed, failure_kind, failure_message = Thread.new do
  result = batch_transcribe.call(0, 0, 0, 1, 0)
  [result.null?, last_error_kind.call, Fiddle::Pointer.new(last_error_message.call).to_s]
end.value
raise "invalid batch arguments must fail" unless failed
raise "invalid batch arguments must be classified" unless failure_kind == 1
raise "invalid batch arguments must include a diagnostic message" unless failure_message.include?("invalid batch transcription")
raise "native diagnostics must remain thread-local" unless last_error_kind.call.zero?

puts "verified #{symbols.length} Cohere session ABI symbols in #{File.basename(library)}"

audio_patterns = case RbConfig::CONFIG.fetch("host_os")
                 when /darwin/ then ["libcohere_audio*.dylib"]
                 when /mswin|mingw/ then ["cohere_audio*.dll", "libcohere_audio*.dll"]
                 else ["libcohere_audio.so", "libcohere_audio.so.*"]
                 end
audio_library = audio_patterns.flat_map do |pattern|
  Dir.glob(File.join(directory, pattern))
end.min
abort "no packaged libcohere_audio found in #{directory}" unless audio_library

audio_handle = Fiddle::Handle.new(audio_library, Fiddle::RTLD_NOW)
audio_symbols = %w[
  cohere_audio_ffmpeg_probe
  cohere_audio_ffmpeg_versions
  cohere_audio_ffmpeg_decode
  cohere_audio_ffmpeg_duration
  cohere_audio_ffmpeg_cancel
  cohere_audio_ffmpeg_free
]
audio_symbols.each { |symbol| audio_handle[symbol] }
probe = Fiddle::Function.new(
  audio_handle["cohere_audio_ffmpeg_probe"],
  [Fiddle::TYPE_VOIDP, Fiddle::TYPE_SIZE_T],
  Fiddle::TYPE_INT
)
diagnostic = Fiddle::Pointer.malloc(1_024, Fiddle::RUBY_FREE)
diagnostic[0, 1_024] = "\0" * 1_024
probe_status = probe.call(diagnostic, 1_024)
raise "audio probe returned an invalid status" unless [0, 1].include?(probe_status)
raise "audio probe returned no diagnostic" if diagnostic.to_s.empty?

versions = Fiddle::Function.new(
  audio_handle["cohere_audio_ffmpeg_versions"],
  [Fiddle::TYPE_VOIDP, Fiddle::TYPE_SIZE_T],
  Fiddle::TYPE_INT
)
raise "audio versions accepted a null output" unless versions.call(0, 0) == 2

if probe_status.zero?
  tuple_bytes = 4 * Fiddle::SIZEOF_INT
  tuple = Fiddle::Pointer.malloc(tuple_bytes, Fiddle::RUBY_FREE)
  tuple[0, tuple_bytes] = [0, 0, 0, 0].pack("i!*")
  raise "audio versions call failed" unless versions.call(tuple, 4).zero?

  format_major, codec_major, util_major, resample_major = tuple[0, tuple_bytes].unpack("i!4")
  raise "audio versions returned an incompatible format/codec tuple" unless format_major == codec_major
  raise "audio versions returned an incompatible avutil major" unless util_major == format_major - 2

  expected_resample = if format_major == 58
                        3
                      elsif format_major <= 60
                        4
                      else
                        format_major - 56
                      end
  raise "audio versions returned an incompatible swresample major" unless resample_major == expected_resample

  duration = Fiddle::Function.new(
    audio_handle["cohere_audio_ffmpeg_duration"],
    [Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_SIZE_T],
    Fiddle::TYPE_INT
  )
  decode = Fiddle::Function.new(
    audio_handle["cohere_audio_ffmpeg_decode"],
    [
      Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT, Fiddle::TYPE_UINT64_T,
      Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_SIZE_T
    ],
    Fiddle::TYPE_INT
  )
  cancel = Fiddle::Function.new(audio_handle["cohere_audio_ffmpeg_cancel"], [], Fiddle::TYPE_VOID)
  release = Fiddle::Function.new(
    audio_handle["cohere_audio_ffmpeg_free"],
    [Fiddle::TYPE_VOIDP],
    Fiddle::TYPE_VOID
  )
  Dir.mktmpdir("cohere-audio-smoke") do |temporary_directory|
    fixture = File.join(temporary_directory, "fixture.wav")
    frames = 400
    pcm = Array.new(frames) { [16_384, -8_192] }.flatten.pack("s<*")
    header = [
      "RIFF", 36 + pcm.bytesize, "WAVE", "fmt ", 16, 1, 2,
      16_000, 64_000, 4, 16, "data", pcm.bytesize
    ].pack("a4Va4a4VvvVVvva4V")
    File.binwrite(fixture, header + pcm)

    duration_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
    duration_slot[0, Fiddle::SIZEOF_DOUBLE] = [-1.0].pack("d")
    status = duration.call(Fiddle::Pointer["#{fixture}\0"], duration_slot, diagnostic, 1_024)
    raise "audio duration smoke failed: #{diagnostic}" unless status.zero?

    seconds = duration_slot[0, Fiddle::SIZEOF_DOUBLE].unpack1("d")
    raise "audio duration smoke mismatch: #{seconds}" unless (seconds - 0.025).abs <= 1e-9

    output_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_VOIDP, Fiddle::RUBY_FREE)
    output_slot[0, Fiddle::SIZEOF_VOIDP] = [0].pack("J")
    count_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_INT64_T, Fiddle::RUBY_FREE)
    count_slot[0, Fiddle::SIZEOF_INT64_T] = [0].pack("q")
    cancel.call # A previous cancellation must never poison a newly started operation.
    status = decode.call(
      Fiddle::Pointer["#{fixture}\0"], 16_000, frames * Fiddle::SIZEOF_FLOAT,
      output_slot, count_slot, diagnostic, 1_024
    )
    address = output_slot[0, Fiddle::SIZEOF_VOIDP].unpack1("J")
    count = count_slot[0, Fiddle::SIZEOF_INT64_T].unpack1("q")
    begin
      raise "audio decode smoke failed: #{diagnostic}" unless status.zero?
      raise "audio decode smoke returned #{count} samples" unless count == frames && !address.zero?
    ensure
      release.call(address) unless address.zero?
    end
  end
end

puts "verified #{audio_symbols.length} audio ABI symbols in #{File.basename(audio_library)}: #{diagnostic}"
