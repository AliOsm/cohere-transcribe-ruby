# frozen_string_literal: true

require "test_helper"
require "tmpdir"

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
            expected = decoded.backend == "ffmpeg" ? Math.sqrt(0.5) * 0.25 : 0.125
            assert_in_delta expected, decoded.samples[200], 1e-6
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
            expected = decoded.backend == "ffmpeg" ? Math.sqrt(0.5) * 0.25 : 0.125
            assert_in_delta expected, decoded.samples[decoded.samples.length / 2], 0.005
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
            error = assert_raises(TranscriptionRuntimeError) do
              Decoder.decode(path, max_decoded_bytes: 16)
            end
            assert_match(/memory limit/, error.message)
          end
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

        def with_wav(sample_rate:, channels:, frames:)
          Dir.mktmpdir do |directory|
            path = File.join(directory, "fixture.wav")
            values = Array.new(frames) do
              channels == 1 ? [4_096] : [16_384, -8_192]
            end.flatten
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
      end
    end
  end
end
