# frozen_string_literal: true

require_relative "../constants"
require_relative "../errors"
require_relative "timestamps"

module Cohere
  module Transcribe
    module VAD
      class SileroBackendUnavailable < TranscriptionRuntimeError; end

      # Thread-confined, bounded ONNX sequence runner for Silero VAD v6.
      #
      # The model consumes one row per 512-sample frame. Each row contains the
      # preceding 64 waveform samples followed by the current frame. Recurrent
      # h/c state and waveform context are carried across bounded temporal
      # calls, while every new audio file starts from zero state.
      class Silero
        WINDOW_SAMPLES = Timestamps::WINDOW_SAMPLES
        CONTEXT_SAMPLES = 64
        MODEL_ROW_SAMPLES = WINDOW_SAMPLES + CONTEXT_SAMPLES
        MAX_SEQUENCE_FRAMES = 256
        DEFAULT_INTRA_OP_THREADS = 1
        STATE_WIDTH = 128
        MODEL_PATH = File.expand_path("silero_vad_v6.onnx", __dir__).freeze
        ENGINE = "onnx"
        PROVIDER = "CPUExecutionProvider"
        SESSION_OPTIONS = {
          providers: [PROVIDER].freeze,
          inter_op_num_threads: 1,
          intra_op_num_threads: 1,
          enable_cpu_mem_arena: false,
          log_severity_level: 4
        }.freeze

        Execution = Data.define(
          :model_calls,
          :valid_frames,
          :padded_frames,
          :max_files_per_call,
          :effective_block_frames,
          :model_load_seconds,
          :inference_seconds,
          :postprocess_seconds
        ) do
          def initialize(model_calls:, valid_frames:, padded_frames:, max_files_per_call:, effective_block_frames:,
                         model_load_seconds: 0.0, inference_seconds: 0.0, postprocess_seconds: 0.0)
            super
          end
        end

        attr_reader :model_path, :block_frames, :intra_op_threads, :last_execution

        def initialize(model_path: MODEL_PATH, session: nil, session_factory: nil,
                       block_frames: MAX_SEQUENCE_FRAMES, threads: DEFAULT_INTRA_OP_THREADS)
          @model_path = File.expand_path(model_path.to_s).freeze
          raise ArgumentError, "block_frames must be a positive integer" unless block_frames.is_a?(Integer) && block_frames.positive?
          raise ArgumentError, "threads must be a positive integer" unless threads.is_a?(Integer) && threads.positive?

          @block_frames = block_frames
          @intra_op_threads = threads
          @session = session
          @session_factory = session_factory
          @current_model_load_seconds = 0.0
          @current_inference_seconds = 0.0
          @last_execution = execution(model_calls: 0, valid_frames: 0)
        end

        def engine
          ENGINE
        end

        def provider
          PROVIDER
        end

        # Match ONNX Runtime's execution-provider introspection. Thread counts
        # belong to SessionOptions and are reported separately in the profile.
        def provider_options
          @provider_options ||= { PROVIDER => {}.freeze }.freeze
        end

        def speech_probabilities(audio)
          @current_model_load_seconds = 0.0
          @current_inference_seconds = 0.0
          audio_length = validate_mono_audio!(audio)
          if audio_length.zero?
            @last_execution = execution(model_calls: 0, valid_frames: 0)
            return []
          end

          return speech_probabilities_numo(audio, audio_length) if numo_array?(audio)

          hidden = zero_state
          cell = zero_state
          previous_context = Array.new(CONTEXT_SAMPLES, 0.0)
          frame_count = (audio_length + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES
          outputs = []

          model_calls = 0
          (0...frame_count).step(block_frames) do |frame_start|
            chunk_frames = [block_frames, frame_count - frame_start].min
            frames = build_frames(audio, audio_length, frame_start, chunk_frames)
            contexts = build_contexts(frames, previous_context)
            previous_context = frames.last.last(CONTEXT_SAMPLES).dup
            model_rows = contexts.zip(frames).map { |context, frame| context + frame }

            probabilities, hidden, cell = run_model(model_rows, hidden, cell)
            chunk_probabilities = one_dimensional_output(probabilities)
            unless chunk_probabilities.length == chunk_frames
              raise TranscriptionRuntimeError,
                    "Silero ONNX returned #{chunk_probabilities.length} probabilities " \
                    "for #{chunk_frames} frames"
            end
            outputs.concat(chunk_probabilities)
            model_calls += 1
          end

          @last_execution = execution(model_calls: model_calls, valid_frames: frame_count)
          outputs
        end

        def speech_timestamps(
          audio,
          sampling_rate: SAMPLE_RATE,
          threshold: 0.5,
          min_speech_duration_ms: 250,
          max_speech_duration_s: Float::INFINITY,
          min_silence_duration_ms: 100,
          speech_pad_ms: 30,
          neg_threshold: nil,
          min_silence_at_max_speech: 98,
          use_max_poss_sil_at_max_speech: true,
          cancel_check: nil
        )
          probabilities = speech_probabilities(audio)
          started = monotonic
          begin
            Timestamps.from_probabilities(
              audio.length,
              probabilities,
              sampling_rate: sampling_rate,
              threshold: threshold,
              min_speech_duration_ms: min_speech_duration_ms,
              max_speech_duration_s: max_speech_duration_s,
              min_silence_duration_ms: min_silence_duration_ms,
              speech_pad_ms: speech_pad_ms,
              neg_threshold: neg_threshold,
              min_silence_at_max_speech: min_silence_at_max_speech,
              use_max_poss_sil_at_max_speech: use_max_poss_sil_at_max_speech,
              cancel_check: cancel_check
            )
          ensure
            @last_execution = @last_execution.with(postprocess_seconds: monotonic - started)
          end
        end

        private

        def speech_probabilities_numo(audio, audio_length)
          hidden = ::Numo::SFloat.zeros(1, 1, STATE_WIDTH)
          cell = ::Numo::SFloat.zeros(1, 1, STATE_WIDTH)
          previous_context = ::Numo::SFloat.zeros(CONTEXT_SAMPLES)
          frame_count = (audio_length + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES
          outputs = []

          model_calls = 0
          (0...frame_count).step(block_frames) do |frame_start|
            chunk_frames = [block_frames, frame_count - frame_start].min
            sample_start = frame_start * WINDOW_SAMPLES
            sample_end = [audio_length, (frame_start + chunk_frames) * WINDOW_SAMPLES].min
            chunk_audio = audio[sample_start...sample_end].cast_to(::Numo::SFloat)
            frames = ::Numo::SFloat.zeros(chunk_frames, WINDOW_SAMPLES)
            full_frames, remainder = chunk_audio.size.divmod(WINDOW_SAMPLES)
            if full_frames.positive?
              frames[0...full_frames, true] =
                chunk_audio[0...(full_frames * WINDOW_SAMPLES)].reshape(full_frames, WINDOW_SAMPLES)
            end
            frames[full_frames, 0...remainder] = chunk_audio[(full_frames * WINDOW_SAMPLES)...chunk_audio.size] if remainder.positive?

            model_rows = ::Numo::SFloat.zeros(chunk_frames, MODEL_ROW_SAMPLES)
            model_rows[0, 0...CONTEXT_SAMPLES] = previous_context
            if chunk_frames > 1
              model_rows[1...chunk_frames, 0...CONTEXT_SAMPLES] =
                frames[0...(chunk_frames - 1), (WINDOW_SAMPLES - CONTEXT_SAMPLES)...WINDOW_SAMPLES]
            end
            model_rows[true, CONTEXT_SAMPLES...MODEL_ROW_SAMPLES] = frames
            previous_context =
              frames[chunk_frames - 1, (WINDOW_SAMPLES - CONTEXT_SAMPLES)...WINDOW_SAMPLES].dup

            probabilities, hidden, cell = run_model(model_rows, hidden, cell, output_type: :numo)
            chunk_probabilities = one_dimensional_output(probabilities)
            unless chunk_probabilities.length == chunk_frames
              raise TranscriptionRuntimeError,
                    "Silero ONNX returned #{chunk_probabilities.length} probabilities " \
                    "for #{chunk_frames} frames"
            end
            outputs.concat(chunk_probabilities)
            model_calls += 1
          end

          @last_execution = execution(model_calls: model_calls, valid_frames: frame_count)
          outputs
        end

        def validate_mono_audio!(audio)
          if audio.respond_to?(:ndim) && audio.ndim != 1
            shape = audio.respond_to?(:shape) ? audio.shape.inspect : audio.ndim.to_s
            raise ArgumentError, "Silero VAD expects mono audio, got shape #{shape}"
          end
          raise ArgumentError, "Silero VAD expects an indexable mono waveform" unless audio.respond_to?(:length) && audio.respond_to?(:[])
          raise ArgumentError, "Silero VAD expects mono audio, got nested samples" if audio.is_a?(Array) && audio.any?(Array)

          length = audio.length
          raise ArgumentError, "Silero VAD expects a non-negative audio length" unless length.is_a?(Integer) && length >= 0

          length
        end

        def build_frames(audio, audio_length, frame_start, chunk_frames)
          frames = Array.new(chunk_frames) { Array.new(WINDOW_SAMPLES, 0.0) }
          sample_start = frame_start * WINDOW_SAMPLES
          sample_end = [audio_length, (frame_start + chunk_frames) * WINDOW_SAMPLES].min
          (sample_start...sample_end).each do |sample_index|
            local_index = sample_index - sample_start
            frame_index, within_frame = local_index.divmod(WINDOW_SAMPLES)
            frames[frame_index][within_frame] = Float(audio[sample_index])
          end
          frames
        rescue TypeError, ArgumentError => e
          raise ArgumentError, "Silero VAD audio must contain real samples: #{e.message}"
        end

        def build_contexts(frames, previous_context)
          contexts = Array.new(frames.length) { Array.new(CONTEXT_SAMPLES, 0.0) }
          contexts[0] = previous_context.dup
          1.upto(frames.length - 1) do |index|
            contexts[index] = frames[index - 1].last(CONTEXT_SAMPLES).dup
          end
          contexts
        end

        def run_model(model_rows, hidden, cell, output_type: nil)
          feed = {
            "input" => model_rows,
            "h" => hidden,
            "c" => cell
          }
          active_session = session
          started = monotonic
          values = if output_type
                     active_session.run(nil, feed, output_type: output_type)
                   else
                     active_session.run(nil, feed)
                   end
          unless values.is_a?(Array) && values.length == 3
            raise TranscriptionRuntimeError, "Silero ONNX must return probabilities, hidden state, and cell state"
          end

          values
        rescue StandardError => e
          raise SileroBackendUnavailable, e.message if onnxruntime_error?(e)

          raise
        ensure
          @current_inference_seconds += monotonic - started if started
        end

        def one_dimensional_output(values)
          if values.respond_to?(:ndim) && values.ndim != 1
            raise TranscriptionRuntimeError, "Silero ONNX returned a non-vector probability tensor"
          end
          raise TranscriptionRuntimeError, "Silero ONNX returned an invalid probability tensor" unless values.respond_to?(:to_a)

          array = values.to_a
          raise TranscriptionRuntimeError, "Silero ONNX returned a non-vector probability tensor" if array.any?(Array)

          array.map { |value| Float(value) }
        rescue TypeError, ArgumentError => e
          raise TranscriptionRuntimeError, "Silero ONNX returned invalid probabilities: #{e.message}"
        end

        def zero_state
          [[Array.new(STATE_WIDTH, 0.0)]]
        end

        def execution(model_calls:, valid_frames:)
          Execution.new(
            model_calls: model_calls,
            valid_frames: valid_frames,
            # The sequence graph receives one unpadded temporal stream. Only
            # samples in the final 32 ms waveform frame can be zero-filled.
            padded_frames: valid_frames,
            max_files_per_call: model_calls.positive? ? 1 : 0,
            effective_block_frames: block_frames,
            model_load_seconds: @current_model_load_seconds,
            inference_seconds: @current_inference_seconds,
            postprocess_seconds: 0.0
          )
        end

        def numo_array?(audio)
          defined?(::Numo::NArray) && audio.is_a?(::Numo::NArray)
        end

        def session
          return @session if @session

          started = monotonic
          @session = build_session
        ensure
          @current_model_load_seconds += monotonic - started if started
        end

        def build_session
          raise SileroBackendUnavailable, "packaged Silero ONNX asset is missing: #{model_path}" unless File.file?(model_path)

          return @session_factory.call(model_path, **session_options) if @session_factory

          require "onnxruntime"
          OnnxRuntime::InferenceSession.new(model_path, **session_options)
        rescue LoadError, SystemCallError => e
          raise SileroBackendUnavailable, "Silero ONNX backend is unavailable: #{e.message}"
        rescue StandardError => e
          raise SileroBackendUnavailable, e.message if onnxruntime_error?(e)

          raise
        end

        def onnxruntime_error?(error)
          error.class.name.to_s.start_with?("OnnxRuntime::")
        end

        def session_options
          SESSION_OPTIONS.merge(intra_op_num_threads: intra_op_threads).freeze
        end

        def monotonic
          Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end
      end
    end
  end
end
