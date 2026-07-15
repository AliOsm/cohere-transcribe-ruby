# frozen_string_literal: true

module Cohere
  module Transcribe
    module VAD
      # Silero 6.2.1's sample-domain timestamp state machine.
      #
      # Keeping this separate from the inference backend makes every Silero
      # engine share exactly the same threshold, hysteresis, duration, and
      # padding behavior.
      module Timestamps
        WINDOW_SAMPLES = 512
        CANCELLATION_CHECK_FRAMES = 4_096

        module_function

        def from_probabilities(
          audio_length_samples,
          speech_probabilities,
          sampling_rate: 16_000,
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
          check_cancellation(cancel_check)
          raise ArgumentError, "The vectorized Silero v6 export requires 16 kHz audio" unless sampling_rate == 16_000
          unless audio_length_samples.is_a?(Integer) && audio_length_samples >= 0
            raise ArgumentError, "audio_length_samples must be a non-negative integer"
          end

          speech_probs = one_dimensional_float32(speech_probabilities)
          expected_frames = (audio_length_samples + WINDOW_SAMPLES - 1) / WINDOW_SAMPLES
          unless speech_probs.length == expected_frames
            raise ArgumentError,
                  "Silero probability count does not match the audio length: " \
                  "expected #{expected_frames}, got #{speech_probs.length}"
          end
          raise ArgumentError, "Silero probabilities contain non-finite values" unless speech_probs.all?(&:finite?)
          unless speech_probs.all? { |probability| probability.between?(0.0, 1.0) }
            raise ArgumentError, "Silero probabilities must be between zero and one"
          end

          check_cancellation(cancel_check)

          min_speech_samples = sampling_rate * min_speech_duration_ms / 1000.0
          speech_pad_samples = sampling_rate * speech_pad_ms / 1000.0
          max_speech_samples = (
            (sampling_rate * max_speech_duration_s) - WINDOW_SAMPLES - (2 * speech_pad_samples)
          )
          min_silence_samples = sampling_rate * min_silence_duration_ms / 1000.0
          min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000.0
          # NumPy keeps scalar comparisons in the probability array's float32
          # dtype. Coerce the comparison thresholds likewise so decimal values
          # such as 0.01 and 0.35 do not move a frame across a boundary merely
          # because Ruby Float is binary64.
          positive_threshold = float32(threshold)
          negative_threshold = float32(neg_threshold || [threshold - 0.15, 0.01].max)

          triggered = false
          speeches = []
          current_speech = {}
          temp_end = 0
          prev_end = 0
          next_start = 0
          possible_ends = []
          next_cancellation_check = CANCELLATION_CHECK_FRAMES

          speech_probs.each_with_index do |speech_prob, index|
            if index == next_cancellation_check
              check_cancellation(cancel_check)
              next_cancellation_check += CANCELLATION_CHECK_FRAMES
            end
            current_sample = WINDOW_SAMPLES * index

            if speech_prob >= positive_threshold && temp_end.positive?
              silence_duration = current_sample - temp_end
              possible_ends << [temp_end, silence_duration] if silence_duration > min_silence_samples_at_max_speech
              temp_end = 0
              next_start = current_sample if next_start < prev_end
            end

            if speech_prob >= positive_threshold && !triggered
              triggered = true
              current_speech[:start] = current_sample
              next
            end

            if triggered && current_sample - current_speech.fetch(:start) > max_speech_samples
              if use_max_poss_sil_at_max_speech && !possible_ends.empty?
                prev_end, silence_duration = possible_ends.max_by { |_end_sample, duration| duration }
                current_speech[:end] = prev_end
                speeches << current_speech
                current_speech = {}
                next_start = prev_end + silence_duration
                if next_start < prev_end + current_sample
                  current_speech[:start] = next_start
                else
                  triggered = false
                end
                prev_end = next_start = temp_end = 0
                possible_ends = []
              elsif prev_end.positive?
                current_speech[:end] = prev_end
                speeches << current_speech
                current_speech = {}
                if next_start < prev_end
                  triggered = false
                else
                  current_speech[:start] = next_start
                end
                prev_end = next_start = temp_end = 0
                possible_ends = []
              else
                current_speech[:end] = current_sample
                speeches << current_speech
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = false
                possible_ends = []
                next
              end
            end

            next unless speech_prob < negative_threshold && triggered

            temp_end = current_sample unless temp_end.positive?
            current_silence_duration = current_sample - temp_end
            if !use_max_poss_sil_at_max_speech &&
               current_silence_duration > min_silence_samples_at_max_speech
              prev_end = temp_end
            end
            next if current_silence_duration < min_silence_samples

            current_speech[:end] = temp_end
            speeches << current_speech if current_speech.fetch(:end) - current_speech.fetch(:start) > min_speech_samples
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = false
            possible_ends = []
          end

          check_cancellation(cancel_check)
          if !current_speech.empty? &&
             audio_length_samples - current_speech.fetch(:start) > min_speech_samples
            current_speech[:end] = audio_length_samples
            speeches << current_speech
          end

          pad_speeches!(speeches, audio_length_samples, speech_pad_samples, cancel_check)
          check_cancellation(cancel_check)
          speeches
        end

        def one_dimensional_float32(values)
          if values.respond_to?(:ndim) && values.ndim != 1
            shape = values.respond_to?(:shape) ? values.shape.inspect : values.ndim.to_s
            raise ArgumentError, "Silero probabilities must be one-dimensional, got #{shape}"
          end
          raise ArgumentError, "Silero probabilities must be one-dimensional" unless values.respond_to?(:to_a)

          array = values.to_a
          if array.any? { |value| value.is_a?(Array) || value.respond_to?(:ndim) }
            raise ArgumentError, "Silero probabilities must be one-dimensional"
          end

          array.map { |value| float32(value) }
        rescue TypeError, ArgumentError => e
          raise e if e.message.start_with?("Silero probabilities")

          raise ArgumentError, "Silero probabilities must contain real numbers: #{e.message}"
        end
        private_class_method :one_dimensional_float32

        def float32(value)
          [Float(value)].pack("f").unpack1("f")
        end
        private_class_method :float32

        def pad_speeches!(speeches, audio_length_samples, speech_pad_samples, cancel_check)
          next_cancellation_check = CANCELLATION_CHECK_FRAMES
          speeches.each_with_index do |speech, index|
            if index == next_cancellation_check
              check_cancellation(cancel_check)
              next_cancellation_check += CANCELLATION_CHECK_FRAMES
            end

            speech[:start] = [0, speech.fetch(:start) - speech_pad_samples].max.to_i if index.zero?
            if index == speeches.length - 1
              speech[:end] = [audio_length_samples, speech.fetch(:end) + speech_pad_samples].min.to_i
            else
              following = speeches[index + 1]
              silence_duration = following.fetch(:start) - speech.fetch(:end)
              if silence_duration < 2 * speech_pad_samples
                half_silence = silence_duration.div(2)
                speech[:end] += half_silence
                following[:start] = [0, following.fetch(:start) - half_silence].max.to_i
              else
                speech[:end] = [audio_length_samples, speech.fetch(:end) + speech_pad_samples].min.to_i
                following[:start] = [0, following.fetch(:start) - speech_pad_samples].max.to_i
              end
            end
          end
        end
        private_class_method :pad_speeches!

        def check_cancellation(cancel_check)
          cancel_check&.call
        end
        private_class_method :check_cancellation
      end
    end
  end
end
