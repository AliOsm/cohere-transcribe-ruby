# frozen_string_literal: true

require "test_helper"
require "fiddle"
require "tmpdir"
require "timeout"

module Cohere
  module Transcribe
    module Audio
      class FFmpegNativeTest < Minitest::Test
        def setup
          skip FFmpegNative.diagnostic unless FFmpegNative.available?
        end

        def test_probe_and_pcm_decode
          assert_match(/FFmpeg ABI avformat (?:58|59|60|61|62)/, FFmpegNative.diagnostic)
          format_major, codec_major, util_major, resample_major = FFmpegNative.ffmpeg_versions
          assert_equal format_major, codec_major
          assert_equal format_major - 2, util_major
          assert_equal(util_major, FFmpegNative.avutil_major)
          expected_resample = if format_major == 58
                                3
                              elsif format_major <= 60
                                4
                              else
                                format_major - 56
                              end
          assert_equal expected_resample, resample_major
          with_wav(frames: 400) do |path|
            samples = FFmpegNative.decode(
              path,
              sample_rate: SAMPLE_RATE,
              max_decoded_bytes: 400 * Fiddle::SIZEOF_FLOAT
            )

            assert_equal 400, samples.length
            assert_in_delta Math.sqrt(0.5) * 0.25, samples[200], 1e-6
            assert_in_delta 400.fdiv(SAMPLE_RATE), FFmpegNative.duration(path), 1e-9
          end
        end

        def test_memory_limit_and_corrupt_input_fail_cleanly
          with_wav(frames: 400) do |path|
            error = assert_raises(DecodedAudioLimitError) do
              FFmpegNative.decode(
                path,
                sample_rate: SAMPLE_RATE,
                max_decoded_bytes: (400 * Fiddle::SIZEOF_FLOAT) - 1
              )
            end
            assert_match(/memory limit/, error.message)
          end

          Dir.mktmpdir do |directory|
            path = File.join(directory, "corrupt.m4a")
            File.binwrite(path, "not an audio container")
            error = assert_raises(TranscriptionRuntimeError) do
              FFmpegNative.decode(path, sample_rate: SAMPLE_RATE, max_decoded_bytes: 1_024)
            end
            assert_match(/FFmpeg could not open/, error.message)
          end
        end

        def test_cancellation_generation_does_not_poison_later_decodes
          FFmpegNative.cancel_all!
          with_wav(frames: 32) do |path|
            assert_equal 32, FFmpegNative.decode(
              path,
              sample_rate: SAMPLE_RATE,
              max_decoded_bytes: 32 * Fiddle::SIZEOF_FLOAT
            ).length
          end
        end

        def test_active_stream_decode_is_cooperatively_cancelled
          skip "named pipes are unavailable" unless File.respond_to?(:mkfifo)

          Dir.mktmpdir do |directory|
            path = File.join(directory, "stream.wav")
            File.mkfifo(path)
            ready = Queue.new
            outcome = Queue.new
            stop = false
            writer = Thread.new do
              File.open(path, "wb") do |io|
                io.sync = true
                io.write(streaming_wav_header)
                ready << true
                chunk = "\0" * 2_048
                until stop
                  io.write(chunk)
                  sleep 0.002
                end
              end
            rescue Errno::EPIPE, IOError
              nil
            end
            decoder = Thread.new do
              samples = FFmpegNative.decode(
                path,
                sample_rate: SAMPLE_RATE,
                max_decoded_bytes: 64 * 1024 * 1024
              )
              outcome << samples
            rescue Exception => e # rubocop:disable Lint/RescueException -- cancellation is an Interrupt
              outcome << e
            end

            Timeout.timeout(3) { ready.pop }
            sleep 0.03
            FFmpegNative.cancel_all!
            assert decoder.join(3), "native streaming decode ignored cancellation"
            error = outcome.pop
            assert_instance_of Interrupt, error
            assert_match(/cancelled/, error.message)
          ensure
            stop = true
            FFmpegNative.cancel_all!
            [writer, decoder].compact.each do |thread|
              next if thread.join(1)

              thread.kill
              thread.join
            end
          end
        end

        def test_repeated_duration_and_decode_calls_do_not_leak_file_descriptors
          descriptors = Pathname("/proc/self/fd")
          skip "file-descriptor accounting is unavailable" unless descriptors.directory?

          with_wav(frames: 32) do |path|
            baseline = descriptors.children.length
            100.times do
              assert_in_delta 32.fdiv(SAMPLE_RATE), FFmpegNative.duration(path), 1e-9
              FFmpegNative.decode(
                path,
                sample_rate: SAMPLE_RATE,
                max_decoded_bytes: 32 * Fiddle::SIZEOF_FLOAT
              )
            end
            GC.start
            assert_operator descriptors.children.length, :<=, baseline + 1
          end
        end

        private

        def with_wav(frames:)
          Dir.mktmpdir do |directory|
            path = File.join(directory, "fixture.wav")
            pcm = Array.new(frames) { [16_384, -8_192] }.flatten.pack("s<*")
            header = [
              "RIFF", 36 + pcm.bytesize, "WAVE", "fmt ", 16, 1, 2,
              SAMPLE_RATE, SAMPLE_RATE * 4, 4, 16, "data", pcm.bytesize
            ].pack("a4Va4a4VvvVVvva4V")
            File.binwrite(path, header + pcm)
            yield path
          end
        end

        def streaming_wav_header
          data_bytes = 0x7FFF_FF00
          [
            "RIFF", 36 + data_bytes, "WAVE", "fmt ", 16, 1, 1,
            SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16, "data", data_bytes
          ].pack("a4Va4a4VvvVVvva4V")
        end
      end

      class FFmpegNativeOwnershipTest < Minitest::Test
        def test_interrupt_immediately_after_native_return_releases_the_pcm_buffer_once
          bytes = 2 * Fiddle::SIZEOF_FLOAT
          address = Fiddle.malloc(bytes)
          Fiddle::Pointer.new(address)[0, bytes] = [0.25, -0.25].pack("f*")
          freed = []
          decode = lambda do |_path, _sample_rate, _maximum, output_slot, count_slot, _message, _capacity|
            output_slot[0, Fiddle::SIZEOF_VOIDP] = [address].pack("J")
            count_slot[0, Fiddle::SIZEOF_INT64_T] = [2].pack("q")
            target = Thread.current
            Thread.new { target.raise Interrupt }.join
            0
          end
          library = fake_library(decode: decode, free: lambda { |pointer|
            freed << pointer
            Fiddle.free(pointer)
          })

          assert_raises(Interrupt) do
            library.decode("fixture.wav", sample_rate: SAMPLE_RATE, max_decoded_bytes: bytes)
          end
          assert_equal [address], freed
        ensure
          Fiddle.free(address) if defined?(address) && address && defined?(freed) && freed.empty?
        end

        def test_kill_immediately_after_native_return_releases_the_pcm_buffer_once
          bytes = 2 * Fiddle::SIZEOF_FLOAT
          address = Fiddle.malloc(bytes)
          Fiddle::Pointer.new(address)[0, bytes] = [0.25, -0.25].pack("f*")
          freed = []
          decode = lambda do |_path, _sample_rate, _maximum, output_slot, count_slot, _message, _capacity|
            output_slot[0, Fiddle::SIZEOF_VOIDP] = [address].pack("J")
            count_slot[0, Fiddle::SIZEOF_INT64_T] = [2].pack("q")
            worker = Thread.current
            Thread.new { worker.kill }.join
            0
          end
          library = fake_library(decode: decode, free: lambda { |pointer|
            freed << pointer
            Fiddle.free(pointer)
          })

          error = assert_raises(TranscriptionRuntimeError) do
            library.decode("fixture.wav", sample_rate: SAMPLE_RATE, max_decoded_bytes: bytes)
          end
          assert_match(/without reporting an outcome/, error.message)
          assert_equal [address], freed
        ensure
          Fiddle.free(address) if defined?(address) && address && defined?(freed) && freed.empty?
        end

        def test_native_limit_status_uses_the_typed_limit_error
          decode = lambda do |_path, _sample_rate, _maximum, _output_slot, _count_slot, message, _capacity|
            message[0, 13] = "memory limit\0"
            FFmpegNative::DECODED_AUDIO_LIMIT_STATUS
          end
          library = fake_library(decode: decode, free: ->(_pointer) {})

          error = assert_raises(DecodedAudioLimitError) do
            library.decode("fixture.wav", sample_rate: SAMPLE_RATE, max_decoded_bytes: 4)
          end
          assert_match(/memory limit/, error.message)
        end

        def test_wrapper_limit_check_uses_the_typed_limit_error_and_frees_output
          bytes = 2 * Fiddle::SIZEOF_FLOAT
          address = Fiddle.malloc(bytes)
          Fiddle::Pointer.new(address)[0, bytes] = [0.25, -0.25].pack("f*")
          freed = []
          decode = lambda do |_path, _sample_rate, _maximum, output_slot, count_slot, _message, _capacity|
            output_slot[0, Fiddle::SIZEOF_VOIDP] = [address].pack("J")
            count_slot[0, Fiddle::SIZEOF_INT64_T] = [2].pack("q")
            0
          end
          library = fake_library(decode: decode, free: lambda { |pointer|
            freed << pointer
            Fiddle.free(pointer)
          })

          assert_raises(DecodedAudioLimitError) do
            library.decode("fixture.wav", sample_rate: SAMPLE_RATE, max_decoded_bytes: Fiddle::SIZEOF_FLOAT)
          end
          assert_equal [address], freed
        ensure
          Fiddle.free(address) if defined?(address) && address && defined?(freed) && freed.empty?
        end

        def test_caller_interrupt_wins_when_the_decode_worker_fails_during_cleanup
          started = Queue.new
          release = Queue.new
          cancel_calls = 0
          decode = lambda do |*_arguments|
            started << true
            release.pop
            raise "worker failure after cancellation"
          end
          cancel = lambda do
            cancel_calls += 1
            release << true if release.empty?
          end
          library = fake_library(decode: decode, free: ->(_pointer) {}, cancel: cancel)
          caller = Thread.current
          interrupter = Thread.new do
            started.pop
            caller.raise(Interrupt, "original caller interruption")
          end

          error = assert_raises(Interrupt) do
            library.decode("fixture.wav", sample_rate: SAMPLE_RATE, max_decoded_bytes: 4)
          end
          assert_equal "original caller interruption", error.message
          assert_operator cancel_calls, :>=, 1
        ensure
          interrupter&.join
        end

        private

        def fake_library(decode:, free:, cancel: -> {})
          FFmpegNative::Library.allocate.tap do |library|
            library.instance_variable_set(
              :@functions,
              {
                decode: decode,
                free: free,
                cancel: cancel
              }
            )
          end
        end
      end

      class FFmpegNativeLoaderTest < Minitest::Test
        def test_failed_adapter_load_is_retryable
          attempts = 0
          loaded = Object.new
          loader = Class.new(FFmpegNative::Library)
          loader.define_singleton_method(:load_uncached) do
            attempts += 1
            raise TranscriptionRuntimeError, "adapter unavailable" if attempts == 1

            loaded
          end

          assert_raises(TranscriptionRuntimeError) { loader.load }
          assert_same loaded, loader.load
          assert_equal 2, attempts
        end
      end
    end
  end
end
