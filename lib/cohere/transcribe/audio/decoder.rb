# frozen_string_literal: true

require "fiddle/import"
require "rbconfig"

require_relative "../constants"
require_relative "../errors"
require_relative "ffmpeg_native"

module Cohere
  module Transcribe
    module Audio
      Decoded = Data.define(:samples, :sample_rate, :backend, :fallback_reason)

      module SharedLibraryCandidates
        module_function

        def build(environment_key, names, formula:, host_os: RbConfig::CONFIG.fetch("host_os"))
          candidates = [ENV.fetch(environment_key, nil), *names].compact
          return candidates.uniq unless host_os.match?(/darwin/)

          directories = []
          homebrew_prefix = ENV.fetch("HOMEBREW_PREFIX", nil)
          directories << File.join(homebrew_prefix, "opt", formula, "lib") if homebrew_prefix && !homebrew_prefix.empty?
          directories.push(
            "/opt/homebrew/opt/#{formula}/lib",
            "/usr/local/opt/#{formula}/lib",
            "/opt/homebrew/lib",
            "/usr/local/lib",
            "/opt/local/lib"
          )
          candidates.concat(directories.product(names).map { |directory, name| File.join(directory, name) })
          candidates.uniq
        end
      end
      private_constant :SharedLibraryCandidates

      module SoundFileABI
        extend Fiddle::Importer

        begin
          candidates = SharedLibraryCandidates.build(
            "COHERE_TRANSCRIBE_SNDFILE_LIBRARY",
            %w[libsndfile.so.1 libsndfile.so libsndfile.1.dylib libsndfile.dylib],
            formula: "libsndfile"
          )
          library = candidates.find do |candidate|
            Fiddle.dlopen(candidate)
            true
          rescue Fiddle::DLError
            false
          end
          raise Fiddle::DLError, "libsndfile was not found" unless library

          dlload library
          SFInfo = struct([
                            "long long frames",
                            "int samplerate",
                            "int channels",
                            "int format",
                            "int sections",
                            "int seekable"
                          ])
          extern "void* sf_open(const char*, int, void*)"
          extern "long long sf_readf_float(void*, void*, long long)"
          extern "int sf_close(void*)"
          extern "const char* sf_strerror(void*)"
          AVAILABLE = true
        rescue Fiddle::DLError => e
          LOAD_ERROR = e
          AVAILABLE = false
        end
      end
      private_constant :SoundFileABI

      module SampleRateABI
        extend Fiddle::Importer

        begin
          candidates = SharedLibraryCandidates.build(
            "COHERE_TRANSCRIBE_SAMPLERATE_LIBRARY",
            %w[libsamplerate.so.0 libsamplerate.so libsamplerate.0.dylib libsamplerate.dylib],
            formula: "libsamplerate"
          )
          library = candidates.find do |candidate|
            Fiddle.dlopen(candidate)
            true
          rescue Fiddle::DLError
            false
          end
          raise Fiddle::DLError, "libsamplerate was not found" unless library

          dlload library
          SRCData = struct([
                             "void* data_in",
                             "void* data_out",
                             "long input_frames",
                             "long output_frames",
                             "long input_frames_used",
                             "long output_frames_gen",
                             "int end_of_input",
                             "double src_ratio"
                           ])
          extern "int src_simple(void*, int, int)"
          extern "const char* src_strerror(int)"
          AVAILABLE = true
        rescue Fiddle::DLError => e
          LOAD_ERROR = e
          AVAILABLE = false
        end
      end
      private_constant :SampleRateABI

      module Decoder
        module_function

        SFM_READ = 0x10
        SRC_SINC_FASTEST = 2
        BACKENDS = %w[auto ffmpeg torchcodec librosa libsndfile].freeze

        # Best-effort metadata probe used for public skipped-result parity. It
        # may inspect container headers and demuxer probe packets but never
        # launches ffprobe or materializes decoded PCM in Ruby.
        def probe_duration(path)
          source = Pathname(path).expand_path
          return nil unless source.file?

          if FFmpegNative.available?
            duration = FFmpegNative.duration(source)
            return duration if duration
          end
          return nil unless SoundFileABI::AVAILABLE

          SoundFileABI::SFInfo.malloc(Fiddle::RUBY_FREE) do |info|
            handle = SoundFileABI.sf_open(source.to_s, SFM_READ, info.to_ptr)
            return nil if handle.null?

            frames = Integer(info.frames)
            sample_rate = Integer(info.samplerate)
            return nil unless frames >= 0 && sample_rate.positive?

            seconds = frames.fdiv(sample_rate)
            return seconds if seconds.finite? && seconds >= 0.0

            nil
          end
        rescue Fiddle::DLError, SystemCallError, TranscriptionRuntimeError
          nil
        ensure
          SoundFileABI.sf_close(handle) if defined?(handle) && handle && !handle.null?
        end

        # Best-effort upper bound for the buffers governed by max_decoded_bytes.
        # The preparation scheduler uses it only for grouping; decode performs
        # the authoritative check again against the per-file ceiling.
        def estimate_decoded_bytes(path, backend: "auto", sample_rate: SAMPLE_RATE)
          requested = backend
          return unless requested.is_a?(String) && BACKENDS.include?(requested)
          return unless sample_rate.is_a?(Integer) && sample_rate.positive?

          source = Pathname(path).expand_path
          return unless source.file?

          native_ffmpeg_available = %w[auto librosa].include?(requested) && FFmpegNative.available?
          use_ffmpeg = requested == "ffmpeg" || requested == "torchcodec" ||
                       (%w[auto librosa].include?(requested) && native_ffmpeg_available)
          if use_ffmpeg
            duration = FFmpegNative.duration(source)
            return unless duration&.finite? && duration >= 0.0

            return ((duration * sample_rate).ceil + 1) * Fiddle::SIZEOF_FLOAT
          end
          return unless SoundFileABI::AVAILABLE

          SoundFileABI::SFInfo.malloc(Fiddle::RUBY_FREE) do |info|
            handle = SoundFileABI.sf_open(source.to_s, SFM_READ, info.to_ptr)
            next if handle.null?

            frames = Integer(info.frames)
            channels = Integer(info.channels)
            source_rate = Integer(info.samplerate)
            next unless frames >= 0 && channels.positive? && source_rate.positive?

            input_bytes = frames * channels * Fiddle::SIZEOF_FLOAT
            output_frames = (frames * sample_rate.fdiv(source_rate)).ceil + 64
            [input_bytes, output_frames * Fiddle::SIZEOF_FLOAT].max
          end
        rescue Fiddle::DLError, SystemCallError, TranscriptionRuntimeError
          nil
        ensure
          SoundFileABI.sf_close(handle) if defined?(handle) && handle && !handle.null?
        end

        def decode(path, backend: "auto", sample_rate: SAMPLE_RATE, max_decoded_bytes: 4 * (1024**3))
          requested = backend
          unless requested.is_a?(String) && BACKENDS.include?(requested)
            raise ArgumentError, "Unsupported audio backend: #{backend.inspect}"
          end
          raise ArgumentError, "sample_rate must be a positive integer" unless sample_rate.is_a?(Integer) && sample_rate.positive?
          unless max_decoded_bytes.nil? || (max_decoded_bytes.is_a?(Integer) && max_decoded_bytes.positive?)
            raise ArgumentError, "max_decoded_bytes must be a positive integer or nil"
          end

          source = Pathname(path).expand_path
          raise TranscriptionInputError, "Input does not exist: #{source}" unless source.exist?
          raise TranscriptionInputError, "Input is not a regular file: #{source}" unless source.file?

          native_ffmpeg_available = %w[auto librosa].include?(requested) && FFmpegNative.available?
          use_ffmpeg = requested == "ffmpeg" || requested == "torchcodec" ||
                       (%w[auto librosa].include?(requested) && native_ffmpeg_available)
          if use_ffmpeg
            samples = FFmpegNative.decode(
              source,
              sample_rate: sample_rate,
              max_decoded_bytes: max_decoded_bytes
            )
            validate_finite!(samples)
            return Decoded.new(
              samples: samples.freeze,
              sample_rate: sample_rate,
              backend: "ffmpeg",
              fallback_reason: if %w[torchcodec librosa].include?(requested)
                                 "Ruby #{requested} compatibility mode uses FFmpeg through the native C ABI"
                               end
            )
          end

          unless SoundFileABI::AVAILABLE
            sound_file_error = if SoundFileABI.const_defined?(:LOAD_ERROR, false)
                                 SoundFileABI::LOAD_ERROR.message
                               else
                                 "not found"
                               end
            if requested == "auto"
              raise TranscriptionRuntimeError,
                    "Automatic audio decoding requires the native FFmpeg adapter or libsndfile " \
                    "(FFmpeg: #{FFmpegNative.diagnostic}; libsndfile: #{sound_file_error})"
            end
            raise TranscriptionRuntimeError, "libsndfile is required for native audio decoding: #{sound_file_error}"
          end

          SoundFileABI::SFInfo.malloc(Fiddle::RUBY_FREE) do |info|
            handle = SoundFileABI.sf_open(source.to_s, SFM_READ, info.to_ptr)
            raise TranscriptionRuntimeError, "Cannot decode #{source}: #{SoundFileABI.sf_strerror(handle)}" if handle.null?

            begin
              frames = Integer(info.frames)
              channels = Integer(info.channels)
              source_rate = Integer(info.samplerate)
              unless frames >= 0 && channels.positive? && source_rate.positive?
                raise TranscriptionRuntimeError, "Decoder returned invalid audio metadata for #{source}"
              end

              input_bytes = frames * channels * Fiddle::SIZEOF_FLOAT
              output_frames = (frames * sample_rate.fdiv(source_rate)).ceil + 64
              projected_bytes = [input_bytes, output_frames * Fiddle::SIZEOF_FLOAT].max
              if max_decoded_bytes && projected_bytes > max_decoded_bytes
                raise TranscriptionRuntimeError,
                      "Decoded audio exceeds the configured memory limit for #{source} " \
                      "(#{projected_bytes} > #{max_decoded_bytes} bytes)"
              end

              raw = Fiddle::Pointer.malloc([input_bytes, 1].max, Fiddle::RUBY_FREE)
              read_frames = SoundFileABI.sf_readf_float(handle, raw, frames)
              raise TranscriptionRuntimeError, "Cannot decode #{source}: #{SoundFileABI.sf_strerror(handle)}" if read_frames.negative?
              raise TranscriptionRuntimeError, "Decoder returned more frames than allocated for #{source}" if read_frames > frames

              frames = read_frames
              begin
                require "numo/narray"
              rescue LoadError => e
                raise TranscriptionRuntimeError, "numo-narray is required for decoded audio: #{e.message}"
              end
              interleaved = if frames.zero?
                              nil
                            else
                              Numo::SFloat.from_binary(raw[0, frames * channels * Fiddle::SIZEOF_FLOAT])
                            end
              mono = if frames.zero?
                       Numo::SFloat.zeros(0)
                     elsif channels == 1
                       interleaved
                     elsif channels == 2
                       (interleaved.reshape(frames, channels).sum(1) * Math.sqrt(0.5)).cast_to(Numo::SFloat)
                     else
                       interleaved.reshape(frames, channels).mean(1).cast_to(Numo::SFloat)
                     end
              samples = source_rate == sample_rate ? mono : resample(mono, source_rate, sample_rate, max_decoded_bytes)
              validate_finite!(samples)
              Decoded.new(
                samples: samples.freeze,
                sample_rate: sample_rate,
                backend: "libsndfile",
                fallback_reason: if %w[auto libsndfile].include?(requested)
                                   nil
                                 else
                                   "Ruby #{requested} compatibility mode uses the native libsndfile ABI"
                                 end
              )
            ensure
              SoundFileABI.sf_close(handle)
            end
          end
        end

        def resample(samples, source_rate, target_rate, max_decoded_bytes)
          unless source_rate.is_a?(Integer) && source_rate.positive? &&
                 target_rate.is_a?(Integer) && target_rate.positive?
            raise ArgumentError, "source_rate and target_rate must be positive integers"
          end
          return Numo::SFloat.zeros(0) if samples.empty?

          unless SampleRateABI::AVAILABLE
            error = SampleRateABI.const_defined?(:LOAD_ERROR, false) ? SampleRateABI::LOAD_ERROR.message : "not found"
            raise TranscriptionRuntimeError, "libsamplerate is required to resample audio: #{error}"
          end
          ratio = target_rate.fdiv(source_rate)
          output_capacity = (samples.length * ratio).ceil + 64
          bytes = output_capacity * Fiddle::SIZEOF_FLOAT
          if max_decoded_bytes && bytes > max_decoded_bytes
            raise TranscriptionRuntimeError, "Resampled audio exceeds the configured memory limit"
          end

          input_string = samples.to_binary
          input_pointer = Fiddle::Pointer[input_string]
          output_pointer = Fiddle::Pointer.malloc([bytes, 1].max, Fiddle::RUBY_FREE)
          SampleRateABI::SRCData.malloc(Fiddle::RUBY_FREE) do |data|
            data.data_in = input_pointer
            data.data_out = output_pointer
            data.input_frames = samples.length
            data.output_frames = output_capacity
            data.input_frames_used = 0
            data.output_frames_gen = 0
            data.end_of_input = 1
            data.src_ratio = ratio
            error_code = SampleRateABI.src_simple(data.to_ptr, SRC_SINC_FASTEST, 1)
            raise TranscriptionRuntimeError, "Audio resampling failed: #{SampleRateABI.src_strerror(error_code)}" unless error_code.zero?

            generated = Integer(data.output_frames_gen)
            consumed = Integer(data.input_frames_used)
            unless consumed == samples.length && generated.between?(0, output_capacity)
              raise TranscriptionRuntimeError,
                    "Audio resampling returned invalid frame counts " \
                    "(consumed #{consumed}/#{samples.length}, generated #{generated}/#{output_capacity})"
            end
            return Numo::SFloat.from_binary(output_pointer[0, generated * Fiddle::SIZEOF_FLOAT])
          end
        end

        def validate_finite!(samples)
          return samples if samples.empty?

          (0...samples.length).step(1_048_576) do |offset|
            finish = [offset + 1_048_576, samples.length].min
            raise TranscriptionRuntimeError, "Decoded audio contains NaN or infinite samples" unless samples[offset...finish].isfinite.all?
          end
          samples
        end
        private_class_method :validate_finite!
      end
    end
  end
end
