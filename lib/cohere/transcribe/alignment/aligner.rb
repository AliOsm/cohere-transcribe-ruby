# frozen_string_literal: true

require "digest"
require "fileutils"
require "numo/narray"

require_relative "../constants"
require_relative "../errors"
require_relative "../hub"
require_relative "../output/timing"
require_relative "../python_text"
require_relative "../types"
require_relative "ctc"
require_relative "text"

module Cohere
  module Transcribe
    module Alignment
      class BackendUnavailable < TranscriptionRuntimeError; end

      ModelArtifact = Data.define(:filename, :size, :sha256)

      # Integrity-pinned access to an ONNX export of the exact reference MMS
      # checkpoint. Model weights remain in the standard Hugging Face cache and
      # are never included in the gem or passed through another language.
      class ModelProvider
        REPOSITORY = "onnx-community/mms-300m-1130-forced-aligner-ONNX"
        REVISION = "2100fb247d8e43962eef24491597fbeb8b469531"
        SOURCE_REPOSITORY = "MahmoudAshraf/mms-300m-1130-forced-aligner"
        SOURCE_REVISION = "49402e9577b1158620820667c218cd494cc44486"
        LICENSE = "CC-BY-NC-4.0"
        UTILITY_REPOSITORY = "https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
        UTILITY_REVISION = "11855d1de76af2b490dd2e8e2db2661805ae90a0"
        UROMAN_COMPATIBILITY_VERSION = "1.3.1.1-compatible-ruby-port"
        ARTIFACTS = {
          "fp32" => ModelArtifact.new(
            filename: "onnx/model.onnx",
            size: 1_262_529_881,
            sha256: "429e5d05c62acc8a9264db874a1b131e359fc626e40c253ac7b1fe52b11149b4"
          ),
          "fp16" => ModelArtifact.new(
            filename: "onnx/model_fp16.onnx",
            size: 631_591_191,
            sha256: "e98082b382375f3528ec7514e175b5cd0eb77fcc4d4531a7142b9e45a1ce6deb"
          )
        }.freeze

        attr_reader :hub

        def initialize(hub: Hub.new, artifacts: ARTIFACTS)
          @hub = hub
          @artifacts = artifacts
        end

        def fetch(dtype)
          artifact = @artifacts.fetch(dtype) do
            raise ArgumentError, "Unsupported aligner dtype: #{dtype.inspect}"
          end
          path = hub.download(REPOSITORY, artifact.filename, revision: REVISION)
          with_integrity_lock(Pathname("#{path}.integrity.lock")) do |lock|
            lock.flock(File::LOCK_EX)
            return path if valid?(path, artifact)

            FileUtils.rm_f(path)
            path = hub.download(REPOSITORY, artifact.filename, revision: REVISION)
            unless valid?(path, artifact)
              FileUtils.rm_f(path)
              raise BackendUnavailable,
                    "Downloaded aligner #{artifact.filename} failed its pinned size/SHA-256 check"
            end
          end
          path
        rescue Hub::Error, SystemCallError => e
          raise BackendUnavailable, "Cannot prepare MMS aligner: #{e.message}"
        end

        private

        def with_integrity_lock(path)
          flags = File::RDWR | File::CREAT
          flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
          flags |= File::CLOEXEC if defined?(File::CLOEXEC)
          descriptor = ::IO.sysopen(path.to_s, flags, 0o600)
          lock = File.new(descriptor, "r+", autoclose: true)
          descriptor = nil
          opened = lock.stat
          current = path.lstat
          unless opened.file? && !current.symlink? && opened.dev == current.dev && opened.ino == current.ino
            raise BackendUnavailable,
                  "Aligner integrity lock changed while it was being opened or is not regular: #{path}"
          end

          yield lock
        rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
          raise BackendUnavailable, "Aligner integrity lock is not a regular file: #{path}", cause: e
        ensure
          lock&.close
          ::IO.new(descriptor).close if descriptor
        end

        def valid?(path, artifact)
          File.file?(path) && File.size(path) == artifact.size &&
            Digest::SHA256.file(path).hexdigest == artifact.sha256
        rescue SystemCallError
          false
        end
      end

      # Lazy ONNX Runtime session with an explicitly reported execution
      # provider. FP16 is admitted only when the CUDA provider is genuinely
      # available; the packaged CPU runtime always uses the full FP32 export.
      class Session
        CPU_PROVIDER = "CPUExecutionProvider"
        CUDA_PROVIDER = "CUDAExecutionProvider"
        SESSION_OPTIONS = {
          graph_optimization_level: :all,
          log_severity_level: 4
        }.freeze

        attr_reader :dtype, :device, :load_seconds, :provider

        def initialize(dtype: "fp32", device: "cpu", model_provider: ModelProvider.new,
                       session: nil, session_factory: nil, available_providers: nil)
          @dtype = dtype
          @device = device
          @model_provider = model_provider
          @session = session
          @session_factory = session_factory
          @available_providers = available_providers
          @provider = session ? CPU_PROVIDER : nil
          @load_seconds = 0.0
          validate_configuration!
        end

        def run(input_values)
          values = session.run(["logits"], { "input_values" => input_values }, output_type: :numo)
          raise TranscriptionRuntimeError, "MMS ONNX aligner must return one logits tensor" unless values.is_a?(Array) && values.length == 1

          values.first
        rescue BackendUnavailable, TranscriptionRuntimeError
          raise
        rescue StandardError => e
          raise BackendUnavailable, "MMS ONNX inference failed: #{e.class}: #{e.message}"
        end

        def load!
          session
          self
        end

        def close
          @session.close if @session.respond_to?(:close)
        ensure
          @session = nil
        end

        private

        def validate_configuration!
          raise ArgumentError, "aligner dtype must be fp32 or fp16" unless %w[fp32 fp16].include?(dtype)
          return unless dtype == "fp16" && device != "cuda"

          raise BackendUnavailable, "FP16 word alignment requires a CUDA ONNX Runtime provider"
        end

        def session
          return @session if @session

          started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          path = @model_provider.fetch(dtype)
          providers = available_providers
          @provider = device == "cuda" && providers.include?(CUDA_PROVIDER) ? CUDA_PROVIDER : CPU_PROVIDER
          if dtype == "fp16" && @provider != CUDA_PROVIDER
            raise BackendUnavailable,
                  "FP16 word alignment requested, but CUDAExecutionProvider is unavailable " \
                  "(available: #{providers.join(", ")})"
          end
          options = SESSION_OPTIONS.merge(providers: [@provider])
          @session = if @session_factory
                       @session_factory.call(path.to_s, **options)
                     else
                       require_onnxruntime!
                       OnnxRuntime::InferenceSession.new(path.to_s, **options)
                     end
          validate_contract!(@session)
          @session
        ensure
          @load_seconds += Process.clock_gettime(Process::CLOCK_MONOTONIC) - started if started
        end

        def available_providers
          return @available_providers if @available_providers

          require_onnxruntime!
          @available_providers = OnnxRuntime::InferenceSession.allocate.send(:providers)
        rescue StandardError => e
          raise BackendUnavailable, "Cannot inspect ONNX Runtime providers: #{e.class}: #{e.message}"
        end

        def require_onnxruntime!
          require "onnxruntime"
          library = ENV.fetch("COHERE_TRANSCRIBE_ONNXRUNTIME_LIBRARY", nil)
          OnnxRuntime.ffi_lib = [File.expand_path(library)] if library && OnnxRuntime.autoload?(:FFI)
        rescue LoadError => e
          raise BackendUnavailable, "ONNX Runtime is unavailable: #{e.message}"
        end

        def validate_contract!(candidate)
          return unless candidate.respond_to?(:inputs) && candidate.respond_to?(:outputs)

          inputs = candidate.inputs
          outputs = candidate.outputs
          valid_input = inputs.length == 1 && inputs.first[:name] == "input_values" &&
                        inputs.first[:type] == "tensor(float)"
          valid_output = outputs.length == 1 && outputs.first[:name] == "logits" &&
                         outputs.first[:type] == "tensor(float)"
          return if valid_input && valid_output

          raise BackendUnavailable, "Pinned MMS ONNX graph has an unexpected input/output contract"
        end
      end

      # Full-file MMS emissions plus per-ASR-segment CTC Viterbi alignment.
      # The 30 s windows, 2 s context, 20 ms stride, wildcard column, and
      # uniform per-segment recovery are the same as the Python reference.
      class Aligner
        SAMPLE_RATE = 16_000
        INPUTS_TO_LOGITS_RATIO = 320
        WINDOW_SECONDS = 30
        CONTEXT_SECONDS = 2
        WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
        CONTEXT_SAMPLES = CONTEXT_SECONDS * SAMPLE_RATE
        INPUT_SAMPLES = WINDOW_SAMPLES + (2 * CONTEXT_SAMPLES)
        MINIMUM_INPUT_SAMPLES = 400
        WINDOW_FRAMES = WINDOW_SAMPLES / INPUTS_TO_LOGITS_RATIO
        CONTEXT_FRAMES = CONTEXT_SAMPLES / INPUTS_TO_LOGITS_RATIO
        STRIDE_MS = INPUTS_TO_LOGITS_RATIO * 1_000.0 / SAMPLE_RATE
        ISO3 = { "ar" => "ara", "en" => "eng" }.freeze
        VOCABULARY = {
          "<blank>" => 0, "<pad>" => 1, "</s>" => 2, "<unk>" => 3,
          "a" => 4, "i" => 5, "e" => 6, "n" => 7, "o" => 8, "u" => 9,
          "t" => 10, "s" => 11, "r" => 12, "m" => 13, "k" => 14,
          "l" => 15, "d" => 16, "g" => 17, "h" => 18, "y" => 19,
          "b" => 20, "p" => 21, "w" => 22, "c" => 23, "v" => 24,
          "j" => 25, "z" => 26, "f" => 27, "'" => 28, "q" => 29,
          "x" => 30
        }.freeze
        BLANK_ID = VOCABULARY.fetch("<blank>")

        attr_reader :session, :batch_size, :emissions_seconds, :viterbi_seconds

        def initialize(dtype: "fp32", device: "cpu", batch_size: 4, session: nil, **session_options)
          raise ArgumentError, "aligner batch_size must be positive" unless batch_size.is_a?(Integer) && batch_size.positive?

          @session = session || Session.new(dtype: dtype, device: device, **session_options)
          @batch_size = batch_size
          @learned_batch_size = batch_size
          @emissions_seconds = 0.0
          @viterbi_seconds = 0.0
        end

        def provider
          session.provider
        end

        def load_seconds
          session.load_seconds
        end

        def load!
          session.load! if session.respond_to?(:load!)
          self
        end

        def close
          session.close
        end

        def align(audio, segment_times, segment_texts, language:)
          raise ArgumentError, "segment_times and segment_texts must have equal lengths" unless segment_times.length == segment_texts.length

          emissions, stride_ms = compute_emissions(audio)
          started = monotonic
          align_emissions(emissions, stride_ms, segment_times, segment_texts, language: language)
        ensure
          @viterbi_seconds += monotonic - started if started
        end

        def compute_emissions(audio)
          started = monotonic
          samples = mono_float32(audio)
          raise ArgumentError, "Cannot compute CTC emissions for empty audio" if samples.empty?
          return [compute_direct_emissions(samples), STRIDE_MS] if samples.length < WINDOW_SAMPLES

          total_windows = (samples.length + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES
          extension_samples = (total_windows * WINDOW_SAMPLES) - samples.length
          extension_frames = extension_samples / INPUTS_TO_LOGITS_RATIO
          frame_count = (total_windows * WINDOW_FRAMES) - extension_frames
          raise TranscriptionRuntimeError, "MMS aligner produced no usable CTC frames" unless frame_count.positive?

          emissions = nil
          write_offset = 0
          first_window = 0
          current_batch_size = [batch_size, @learned_batch_size].min
          while first_window < total_windows
            window_count = [current_batch_size, total_windows - first_window].min
            begin
              input = build_window_batch(samples, first_window, window_count)
              logits = session.run(input)
              batch_log_probs = crop_and_normalize(logits, window_count)
            rescue NoMemoryError, StandardError => e
              raise unless out_of_memory?(e) && current_batch_size > 1

              current_batch_size = [1, current_batch_size / 2].max
              @learned_batch_size = current_batch_size
              next
            end

            expected_frames = window_count * WINDOW_FRAMES
            unless batch_log_probs.shape[0] == expected_frames
              raise TranscriptionRuntimeError, "MMS aligner returned an unexpected frame count"
            end

            class_count = batch_log_probs.shape[1]
            validate_class_count!(class_count)
            emissions ||= Numo::SFloat.zeros(frame_count, class_count + 1)
            if first_window + window_count == total_windows && extension_frames.positive?
              kept = batch_log_probs.shape[0] - extension_frames
              batch_log_probs = batch_log_probs[0...kept, true]
            end

            next_offset = write_offset + batch_log_probs.shape[0]
            raise TranscriptionRuntimeError, "MMS aligner produced too many CTC frames" if next_offset > emissions.shape[0]

            emissions[write_offset...next_offset, 0...class_count] = batch_log_probs
            write_offset = next_offset
            first_window += window_count
          end
          unless emissions && write_offset == emissions.shape[0]
            raise TranscriptionRuntimeError,
                  "MMS emission assembly mismatch: wrote #{write_offset}, expected #{frame_count}"
          end

          [emissions, STRIDE_MS]
        ensure
          @emissions_seconds += monotonic - started if started
        end

        def align_emissions(emissions, stride_ms, segment_times, segment_texts, language:)
          frame_count, emission_classes = emissions.shape
          star_id = emission_classes - 1
          raise TranscriptionRuntimeError, "MMS wildcard column collides with its tokenizer vocabulary" if VOCABULARY.value?(star_id)

          dictionary = VOCABULARY.merge("<star>" => star_id).freeze
          index_to_token = dictionary.invert.freeze
          iso_language = ISO3.fetch(language, language)
          words = []
          fallback_count = 0

          segment_times.zip(segment_texts).each_with_index do |((start_time, end_time), text), segment_index|
            text = PythonText.strip(text.to_s)
            next if text.empty?

            first_frame = [(start_time * 1_000 / stride_ms).round(half: :even), 0].max
            last_frame = [(end_time * 1_000 / stride_ms).round(half: :even), frame_count].min
            if last_frame - first_frame < 2
              fallback_count += 1
              words.concat(uniform_fallback(text, start_time, end_time, segment_index))
              next
            end

            begin
              tokens_starred, text_starred = Text.preprocess(text, iso_language)
              targets = tokens_starred.join(" ").split.filter_map { |token| dictionary[token] }
              raise ArgumentError, "Transcript produced no aligner vocabulary tokens" if targets.empty?

              path = CTC.forced_align(emissions[first_frame...last_frame, true], targets, blank: BLANK_ID)
              segments = CTC.merge_repeats(path, index_to_token)
              spans = CTC.spans(tokens_starred, segments, index_to_token.fetch(BLANK_ID))
              results = Text.postprocess(text_starred, spans, stride_ms)
              expected_tokens = PythonText.split(text)
              unless results.map { |word| word.fetch(:text) } == expected_tokens
                raise ArgumentError,
                      "forced alignment did not preserve the complete ASR transcript " \
                      "(#{results.length}/#{expected_tokens.length} words)"
              end
            rescue StandardError
              fallback_count += 1
              words.concat(uniform_fallback(text, start_time, end_time, segment_index))
              next
            end

            results.each_with_index do |word, word_index|
              absolute_start = (start_time + word.fetch(:start)).clamp(start_time, end_time)
              absolute_end = (start_time + word.fetch(:end)).clamp(absolute_start, end_time)
              words << TranscriptionWord.new(
                start: absolute_start,
                end: absolute_end,
                text: word.fetch(:text),
                segment_index: segment_index,
                segment_word_index: word_index,
                timing_source: "ctc"
              )
            end
          end
          [words.freeze, fallback_count]
        end

        private

        def mono_float32(audio)
          if audio.respond_to?(:ndim)
            raise ArgumentError, "MMS alignment expects mono audio" unless audio.ndim == 1

            audio.cast_to(Numo::SFloat)
          elsif audio.is_a?(Array) && audio.none?(Array)
            Numo::SFloat.cast(audio)
          else
            raise ArgumentError, "MMS alignment expects an indexable mono waveform"
          end
        rescue TypeError, ArgumentError => e
          raise ArgumentError, "MMS alignment audio must contain real samples: #{e.message}"
        end

        def build_window_batch(audio, first_window, window_count)
          batch = Numo::SFloat.zeros(window_count, INPUT_SAMPLES)
          window_count.times do |row|
            window_index = first_window + row
            requested_start = (window_index * WINDOW_SAMPLES) - CONTEXT_SAMPLES
            requested_end = requested_start + INPUT_SAMPLES
            source_start = [0, requested_start].max
            source_end = [audio.length, requested_end].min
            next if source_end <= source_start

            destination_start = source_start - requested_start
            destination_end = destination_start + (source_end - source_start)
            batch[row, destination_start...destination_end] = audio[source_start...source_end]
          end
          batch
        end

        # The pinned ctc-forced-aligner passes sub-30-second waveforms to MMS
        # directly, without window context or tail padding. Its Wav2Vec2 feature
        # extractor needs at least its 400-sample convolutional receptive field,
        # so only shorter inputs receive the minimum right padding needed to run.
        def compute_direct_emissions(audio)
          input_samples = [audio.length, MINIMUM_INPUT_SAMPLES].max
          input = Numo::SFloat.zeros(1, input_samples)
          input[0, 0...audio.length] = audio
          logits = session.run(input)
          logits = Numo::SFloat.cast(logits) unless logits.is_a?(Numo::NArray)
          unless logits.ndim == 3 && logits.shape[0] == 1 && logits.shape[1].positive?
            raise TranscriptionRuntimeError,
                  "MMS aligner returned invalid direct logits shape #{logits.shape.inspect}"
          end

          class_count = logits.shape[2]
          validate_class_count!(class_count)
          log_probs = log_softmax(logits.reshape(logits.shape[1], class_count))
          emissions = Numo::SFloat.zeros(log_probs.shape[0], class_count + 1)
          emissions[true, 0...class_count] = log_probs
          emissions
        end

        def crop_and_normalize(logits, expected_batch)
          logits = Numo::SFloat.cast(logits) unless logits.is_a?(Numo::NArray)
          unless logits.ndim == 3 && logits.shape[0] == expected_batch &&
                 logits.shape[1] >= CONTEXT_FRAMES + WINDOW_FRAMES
            raise TranscriptionRuntimeError,
                  "MMS aligner returned invalid logits shape #{logits.shape.inspect}"
          end

          classes = logits.shape[2]
          cropped = logits[true, CONTEXT_FRAMES...(CONTEXT_FRAMES + WINDOW_FRAMES), true]
                    .reshape(expected_batch * WINDOW_FRAMES, classes)
          log_softmax(cropped)
        end

        def log_softmax(logits)
          maxima = logits.max(1).reshape(logits.shape[0], 1)
          shifted = logits - maxima
          denominators = Numo::NMath.log(Numo::NMath.exp(shifted).sum(1)).reshape(logits.shape[0], 1)
          (shifted - denominators).cast_to(Numo::SFloat)
        end

        def validate_class_count!(class_count)
          return if class_count == VOCABULARY.length

          raise TranscriptionRuntimeError,
                "MMS aligner returned #{class_count} classes; expected #{VOCABULARY.length}"
        end

        def uniform_fallback(text, start_time, end_time, segment_index)
          Output::Timing.uniform_words(
            text, start_time, end_time, segment_index, "uniform_fallback"
          )
        end

        def out_of_memory?(error)
          error.is_a?(NoMemoryError) || /out of memory|bad_alloc|failed to allocate|memory allocation/i.match?(error.message)
        end

        def monotonic
          Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end
      end
    end
  end
end
