# frozen_string_literal: true

require "test_helper"
require "tmpdir"
require "cohere/transcribe/audio/decoder"

module Cohere
  module Transcribe
    module Audio
      class DecoderTest < Minitest::Test
        def test_decodes_pcm16_stereo_and_mixes_channels
          with_wav(sample_rate: SAMPLE_RATE, channels: 2, frames: 400) do |path|
            decoded = Decoder.decode(path)

            assert_equal SAMPLE_RATE, decoded.sample_rate
            assert_includes %w[ffmpeg libsndfile], decoded.backend
            assert_nil decoded.fallback_reason
            assert_equal 400, decoded.samples.length
            assert decoded.samples.frozen?
            assert_in_delta Math.sqrt(0.5) * 0.25, decoded.samples[200], 1e-6
          end
        end

        def test_duration_probe_reads_metadata_without_decoding_pcm
          with_wav(sample_rate: 8_000, channels: 1, frames: 2_000) do |path|
            assert_in_delta 0.25, Decoder.probe_duration(path), 1e-9
          end
        end

        def test_duration_probe_is_best_effort_for_missing_and_corrupt_inputs
          Dir.mktmpdir do |directory|
            assert_nil Decoder.probe_duration(File.join(directory, "missing.wav"))
            corrupt = File.join(directory, "corrupt.wav")
            File.binwrite(corrupt, "not audio")
            assert_nil Decoder.probe_duration(corrupt)
          end
        end

        def test_decode_size_estimate_covers_the_libsndfile_stereo_input_buffer
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)

          with_wav(sample_rate: SAMPLE_RATE, channels: 2, frames: 400) do |path|
            replace_singleton_methods(FFmpegNative, available?: -> { false }) do
              estimate = Decoder.estimate_decoded_bytes(
                path,
                backend: "librosa",
                sample_rate: SAMPLE_RATE
              )

              assert_equal 400 * 2 * Fiddle::SIZEOF_FLOAT, estimate
            end
          end
        end

        def test_libsndfile_metadata_probes_close_every_open_handle
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)

          closed_handles = []
          original_close = sound_file.method(:sf_close)
          close = lambda do |handle|
            closed_handles << handle.to_i
            original_close.call(handle)
          end

          with_wav(sample_rate: 8_000, channels: 2, frames: 400) do |path|
            replace_singleton_methods(sound_file, sf_close: close) do
              replace_singleton_methods(FFmpegNative, available?: -> { false }) do
                assert_in_delta 0.05, Decoder.probe_duration(path), 1e-9
                assert_equal ((400 * 2) + 64) * Fiddle::SIZEOF_FLOAT,
                             Decoder.estimate_decoded_bytes(path, backend: "librosa")
              end
            end
          end

          assert_equal 2, closed_handles.length
          assert closed_handles.all?(&:positive?)
        end

        def test_kill_after_libsndfile_open_closes_the_handle_before_termination
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)

          closed_handles = []
          original_open = sound_file.method(:sf_open)
          original_close = sound_file.method(:sf_close)
          open = lambda do |*arguments|
            handle = original_open.call(*arguments)
            opening_thread = Thread.current
            Thread.new { opening_thread.kill }.join
            handle
          end
          close = lambda do |handle|
            closed_handles << handle.to_i
            original_close.call(handle)
          end

          with_wav(sample_rate: 8_000, channels: 2, frames: 400) do |path|
            replace_singleton_methods(sound_file, sf_open: open, sf_close: close) do
              replace_singleton_methods(FFmpegNative, available?: -> { false }) do
                caller = Thread.new { Decoder.probe_duration(path) }
                caller.report_on_exception = false
                assert caller.join(2), "libsndfile probe remained stuck after termination"
                assert_nil caller.value
              ensure
                caller&.kill
                caller&.join
              end
            end
          end

          assert_equal 1, closed_handles.length
          assert closed_handles.first.positive?
        end

        def test_libsndfile_metadata_allocations_register_release_callbacks
          sound_file = Audio.const_get(:SoundFileABI, false)
          sample_rate = Audio.const_get(:SampleRateABI, false)
          skip "libsndfile or libsamplerate is unavailable" unless sound_file.const_get(:AVAILABLE) && sample_rate.const_get(:AVAILABLE)

          sound_file_releases = []
          sample_rate_releases = []
          original_sound_file_malloc = sound_file::SFInfo.method(:malloc)
          original_sample_rate_malloc = sample_rate::SRCData.method(:malloc)
          sound_file_malloc = lambda do |release = nil, &block|
            sound_file_releases << release
            original_sound_file_malloc.call(release, &block)
          end
          sample_rate_malloc = lambda do |release = nil, &block|
            sample_rate_releases << release
            original_sample_rate_malloc.call(release, &block)
          end

          replace_singleton_methods(sound_file::SFInfo, malloc: sound_file_malloc) do
            replace_singleton_methods(sample_rate::SRCData, malloc: sample_rate_malloc) do
              replace_singleton_methods(FFmpegNative, available?: -> { false }) do
                with_wav(sample_rate: 8_000, channels: 2, frames: 400) do |path|
                  Decoder.probe_duration(path)
                  Decoder.decode(path, backend: "librosa", sample_rate: SAMPLE_RATE)
                end
              end
            end
          end

          assert_equal [Fiddle::RUBY_FREE, Fiddle::RUBY_FREE], sound_file_releases
          assert_equal [Fiddle::RUBY_FREE], sample_rate_releases
        end

        def test_darwin_library_candidates_include_homebrew_and_macports
          candidates = Audio.const_get(:SharedLibraryCandidates, false).build(
            "COHERE_TRANSCRIBE_UNUSED_LIBRARY",
            ["libexample.dylib"],
            formula: "example",
            host_os: "darwin"
          )

          assert_includes candidates, "/opt/homebrew/opt/example/lib/libexample.dylib"
          assert_includes candidates, "/usr/local/opt/example/lib/libexample.dylib"
          assert_includes candidates, "/opt/local/lib/libexample.dylib"
        end

        def test_resamples_real_wav_through_libsamplerate
          with_wav(sample_rate: 8_000, channels: 2, frames: 800) do |path|
            decoded = Decoder.decode(path, sample_rate: SAMPLE_RATE, max_decoded_bytes: nil)

            assert_equal SAMPLE_RATE, decoded.sample_rate
            assert_in_delta 1_600, decoded.samples.length, 1
            assert_in_delta Math.sqrt(0.5) * 0.25, decoded.samples[decoded.samples.length / 2], 0.005
          end
        end

        def test_explicit_compatibility_backend_is_reported
          with_wav(sample_rate: SAMPLE_RATE, channels: 1, frames: 10) do |path|
            decoded = Decoder.decode(path, backend: "librosa")

            assert_includes %w[ffmpeg libsndfile], decoded.backend
            assert_match(/librosa compatibility mode/, decoded.fallback_reason)
          end
        end

        def test_librosa_compatibility_uses_libsndfile_when_native_ffmpeg_is_unavailable
          with_wav(sample_rate: SAMPLE_RATE, channels: 1, frames: 10) do |path|
            replace_singleton_methods(FFmpegNative, available?: -> { false }) do
              decoded = Decoder.decode(path, backend: "librosa")

              assert_equal "libsndfile", decoded.backend
              assert_match(/librosa compatibility mode/, decoded.fallback_reason)
            end
          end
        end

        def test_libsndfile_stereo_downmix_matches_native_ffmpeg_energy
          with_wav(sample_rate: SAMPLE_RATE, channels: 2, frames: 400) do |path|
            replace_singleton_methods(FFmpegNative, available?: -> { false }) do
              decoded = Decoder.decode(path, backend: "librosa")

              assert_equal "libsndfile", decoded.backend
              assert_in_delta Math.sqrt(0.5) * 0.25, decoded.samples[200], 1e-6
            end
          end
        end

        def test_libsndfile_common_5_1_downmix_matches_native_ffmpeg_matrix
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)

          with_wav(
            sample_rate: SAMPLE_RATE,
            channels: 6,
            frames: 400,
            channel_values: [8_192, 8_192, 8_192, 8_192, 8_192, 8_192]
          ) do |path|
            replace_singleton_methods(FFmpegNative, available?: -> { false }) do
              decoded = Decoder.decode(path, backend: "librosa")
              expected = 0.25 * ((2 * Math.sqrt(0.5)) + 1.0 + (2 * 0.5))

              assert_equal "libsndfile", decoded.backend
              assert_in_delta expected, decoded.samples[200], 1e-6
            end
          end
        end

        def test_libsndfile_and_native_ffmpeg_match_common_5_1_layout
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile or native FFmpeg is unavailable" unless sound_file.const_get(:AVAILABLE) && FFmpegNative.available?

          with_wav(
            sample_rate: SAMPLE_RATE,
            channels: 6,
            frames: 400,
            channel_values: [8_192, 8_192, 8_192, 8_192, 8_192, 8_192]
          ) do |path|
            fallback = Decoder.decode(path, backend: "libsndfile")
            native = Decoder.decode(path, backend: "ffmpeg")

            assert_in_delta native.samples[200], fallback.samples[200], 1e-6
          end
        end

        def test_libsndfile_and_native_ffmpeg_match_standard_height_layouts
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)
          skip "native FFmpeg audio adapter is unavailable" unless FFmpegNative.available?

          layouts = {
            "5.1.2" => [0x503f, 8],
            "5.1.4" => [0x2d03f, 10],
            "7.1.4" => [0x2d63f, 12]
          }
          layouts.each do |name, (channel_mask, channels)|
            values = Array.new(channels) { |index| (index + 1) * 1_024 }
            with_wav_extensible(
              sample_rate: SAMPLE_RATE,
              channel_values: values,
              channel_mask: channel_mask
            ) do |path|
              fallback = Decoder.decode(path, backend: "libsndfile")
              native = Decoder.decode(path, backend: "ffmpeg")

              assert_in_delta native.samples[200], fallback.samples[200], 1e-6, name
            end
          end
        end

        def test_libsndfile_and_native_ffmpeg_match_unspecified_standard_layouts_above_eight_channels
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)
          skip "native FFmpeg audio adapter is unavailable" unless FFmpegNative.available?

          channels = [10, 12, 16, 24]
          channels << 14 if FFmpegNative.avutil_major.to_i >= 59
          channels.sort.each do |channel_count|
            values = Array.new(channel_count) { |index| (index + 1) * 512 }
            with_wav(
              sample_rate: SAMPLE_RATE,
              channels: channel_count,
              frames: 400,
              channel_values: values
            ) do |path|
              fallback = Decoder.decode(path, backend: "libsndfile")
              native = Decoder.decode(path, backend: "ffmpeg")

              assert_in_delta native.samples[200], fallback.samples[200], 1e-6, "#{channel_count} channels"
            end
          end
        end

        def test_unspecified_14_and_16_channel_defaults_follow_the_loaded_ffmpeg_generation
          old_sixteen = Decoder.const_get(:DEFAULT_MONO_MIXES).fetch(16)
          seven_fourteen = Decoder.const_get(:FFMPEG_7_DEFAULT_MONO_MIXES).fetch(14)
          eight_sixteen = Decoder.const_get(:FFMPEG_8_DEFAULT_MONO_MIXES).fetch(16)

          replace_singleton_methods(FFmpegNative, avutil_major: -> { 58 }) do
            assert_nil Decoder.send(:default_mono_mix, 14)
            assert_same old_sixteen, Decoder.send(:default_mono_mix, 16)
          end
          replace_singleton_methods(FFmpegNative, avutil_major: -> { 59 }) do
            assert_same seven_fourteen, Decoder.send(:default_mono_mix, 14)
            assert_same old_sixteen, Decoder.send(:default_mono_mix, 16)
          end
          replace_singleton_methods(FFmpegNative, avutil_major: -> { 60 }) do
            assert_same seven_fourteen, Decoder.send(:default_mono_mix, 14)
            assert_same eight_sixteen, Decoder.send(:default_mono_mix, 16)
          end
        end

        def test_libsndfile_uses_an_explicit_three_channel_layout
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)

          with_wav_extensible(
            sample_rate: SAMPLE_RATE,
            channel_values: [0, 0, 8_192],
            channel_mask: 0x7
          ) do |path|
            decoded = Decoder.decode(path, backend: "libsndfile")

            assert_in_delta 0.25, decoded.samples[200], 1e-6
            if FFmpegNative.available?
              native = Decoder.decode(path, backend: "ffmpeg")
              assert_in_delta native.samples[200], decoded.samples[200], 1e-6
            end
          end
        end

        def test_libsndfile_and_native_ffmpeg_match_unspecified_three_channel_2_1_layout
          sound_file = Audio.const_get(:SoundFileABI, false)
          skip "libsndfile is unavailable" unless sound_file.const_get(:AVAILABLE)
          skip "native FFmpeg audio adapter is unavailable" unless FFmpegNative.available?

          with_wav(
            sample_rate: SAMPLE_RATE,
            channels: 3,
            frames: 400,
            channel_values: [8_192, 4_096, 16_384]
          ) do |path|
            fallback = Decoder.decode(path, backend: "libsndfile")
            native = Decoder.decode(path, backend: "ffmpeg")
            expected = Math.sqrt(0.5) * 0.375

            assert_in_delta expected, fallback.samples[200], 1e-6
            assert_in_delta native.samples[200], fallback.samples[200], 1e-6
          end
        end

        def test_explicit_ffmpeg_uses_native_abi_or_fails_cleanly_when_adapter_is_absent
          with_wav(sample_rate: SAMPLE_RATE, channels: 1, frames: 10) do |path|
            if FFmpegNative.available?
              decoded = Decoder.decode(path, backend: "ffmpeg")

              assert_equal "ffmpeg", decoded.backend
              assert_nil decoded.fallback_reason
            else
              error = assert_raises(TranscriptionRuntimeError) do
                Decoder.decode(path, backend: "ffmpeg")
              end
              assert_match(/native FFmpeg audio adapter/, error.message)
            end
          end
        end

        def test_native_ffmpeg_serves_auto_and_compatible_named_backends
          require "numo/narray"
          samples = Numo::SFloat[0.0, 0.25, -0.25]
          test_case = self
          decode = lambda do |path, sample_rate:, max_decoded_bytes:|
            test_case.assert_equal "fixture.wav", File.basename(path.to_s)
            test_case.assert_equal SAMPLE_RATE, sample_rate
            test_case.assert_equal 64, max_decoded_bytes
            samples
          end

          with_wav(sample_rate: SAMPLE_RATE, channels: 1, frames: 10) do |path|
            replace_singleton_methods(FFmpegNative, available?: -> { true }, decode: decode) do
              automatic = Decoder.decode(path, backend: "auto", max_decoded_bytes: 64)
              torchcodec = Decoder.decode(path, backend: "torchcodec", max_decoded_bytes: 64)
              librosa = Decoder.decode(path, backend: "librosa", max_decoded_bytes: 64)

              assert_equal ["ffmpeg", nil], [automatic.backend, automatic.fallback_reason]
              assert_equal "ffmpeg", torchcodec.backend
              assert_match(/torchcodec compatibility mode/, torchcodec.fallback_reason)
              assert_equal "ffmpeg", librosa.backend
              assert_match(/librosa compatibility mode/, librosa.fallback_reason)
              assert_equal samples, automatic.samples
            end
          end
        end

        def test_zero_frame_wav_decodes_without_resampling_failure
          with_wav(sample_rate: 8_000, channels: 1, frames: 0) do |path|
            decoded = Decoder.decode(path)

            assert_equal 0, decoded.samples.length
          end
        end

        def test_memory_limit_is_checked_before_allocation
          with_wav(sample_rate: SAMPLE_RATE, channels: 2, frames: 100) do |path|
            error = assert_raises(DecodedAudioLimitError) do
              Decoder.decode(path, max_decoded_bytes: 16)
            end
            assert_match(/memory limit/, error.message)
          end
        end

        def test_resample_memory_limit_uses_the_typed_limit_error
          sample_rate = Audio.const_get(:SampleRateABI, false)
          skip "libsamplerate is unavailable" unless sample_rate.const_get(:AVAILABLE)
          require "numo/narray"

          error = assert_raises(DecodedAudioLimitError) do
            Decoder.resample(Numo::SFloat.zeros(100), 8_000, SAMPLE_RATE, 16)
          end
          assert_match(/memory limit/, error.message)
        end

        def test_missing_and_nonregular_inputs_are_typed
          Dir.mktmpdir do |directory|
            missing = File.join(directory, "missing.wav")
            error = assert_raises(TranscriptionInputError) { Decoder.decode(missing) }
            assert_match(/does not exist/, error.message)

            error = assert_raises(TranscriptionInputError) { Decoder.decode(directory) }
            assert_match(/not a regular file/, error.message)
          end
        end

        def test_decode_validates_backend_rates_and_memory_limit_types
          assert_raises(ArgumentError) { Decoder.decode("unused.wav", backend: :auto) }
          assert_raises(ArgumentError) { Decoder.decode("unused.wav", backend: "unknown") }
          [0, -1, 16_000.0, true].each do |rate|
            assert_raises(ArgumentError) { Decoder.decode("unused.wav", sample_rate: rate) }
          end
          [0, -1, 1.0, true].each do |limit|
            assert_raises(ArgumentError) { Decoder.decode("unused.wav", max_decoded_bytes: limit) }
          end
        end

        def test_finite_sample_validation_is_chunked_and_rejects_nan_and_infinity
          require "numo/narray"

          assert_equal(
            Numo::SFloat[0.0, 1.0],
            Decoder.send(:validate_finite!, Numo::SFloat[0.0, 1.0])
          )
          assert_raises(TranscriptionRuntimeError) do
            Decoder.send(:validate_finite!, Numo::SFloat[0.0, Float::NAN])
          end
          assert_raises(TranscriptionRuntimeError) do
            Decoder.send(:validate_finite!, Numo::SFloat[Float::INFINITY])
          end
        end

        private

        def replace_singleton_methods(object, replacements)
          originals = replacements.to_h { |name, _replacement| [name, object.method(name)] }
          replacements.each do |name, replacement|
            object.singleton_class.define_method(name, replacement)
          end
          yield
        ensure
          originals&.each do |name, original|
            object.singleton_class.define_method(name) do |*arguments, **keywords, &block|
              original.call(*arguments, **keywords, &block)
            end
          end
        end

        def with_wav(sample_rate:, channels:, frames:, channel_values: nil)
          Dir.mktmpdir do |directory|
            path = File.join(directory, "fixture.wav")
            frame_values = channel_values || (channels == 1 ? [4_096] : [16_384, -8_192])
            raise ArgumentError, "channel_values must contain one sample per channel" unless frame_values.length == channels

            values = Array.new(frames, frame_values).flatten
            pcm = values.pack("s<*")
            header = [
              "RIFF", 36 + pcm.bytesize, "WAVE", "fmt ", 16, 1, channels,
              sample_rate, sample_rate * channels * 2, channels * 2, 16,
              "data", pcm.bytesize
            ].pack("a4Va4a4VvvVVvva4V")
            File.binwrite(path, header + pcm)
            yield path
          end
        rescue TranscriptionRuntimeError => e
          skip e.message if e.message.match?(/libsndfile|libsamplerate/)

          raise
        end

        def with_wav_extensible(sample_rate:, channel_values:, channel_mask:, frames: 400)
          Dir.mktmpdir do |directory|
            path = File.join(directory, "fixture.wav")
            channels = channel_values.length
            pcm = Array.new(frames, channel_values).flatten.pack("s<*")
            format = [
              0xfffe, channels, sample_rate, sample_rate * channels * 2,
              channels * 2, 16, 22, 16, channel_mask
            ].pack("vvVVvvvvV")
            format << ["0100000000001000800000aa00389b71"].pack("H*")
            header = ["RIFF", 60 + pcm.bytesize, "WAVE", "fmt ", 40].pack("a4Va4a4V")
            data = ["data", pcm.bytesize].pack("a4V")
            File.binwrite(path, header + format + data + pcm)
            yield path
          end
        rescue TranscriptionRuntimeError => e
          skip e.message if e.message.match?(/libsndfile|libsamplerate/)

          raise
        end
      end
    end
  end
end
