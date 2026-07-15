# frozen_string_literal: true

require_relative "../constants"

module Cohere
  module Transcribe
    module Audio
      module Segmentation
        module_function

        def fixed(audio_or_length, window_seconds)
          total_samples = audio_or_length.is_a?(Integer) ? audio_or_length : audio_or_length.length
          return [] if total_samples.zero?
          raise ArgumentError, "audio length must be non-negative" if total_samples.negative?
          raise ArgumentError, "window_seconds must be finite and positive" unless real_finite?(window_seconds) && window_seconds.positive?

          window_samples = [1, (window_seconds * SAMPLE_RATE).round(half: :even)].max
          (0...total_samples).step(window_samples).map do |start_sample|
            [start_sample.fdiv(SAMPLE_RATE), [start_sample + window_samples, total_samples].min.fdiv(SAMPLE_RATE)]
          end
        end

        def validate(segments, duration, max_duration: nil)
          raise ArgumentError, "Audio duration must be finite and non-negative" unless real_finite?(duration) && duration >= 0

          tolerance = 2.0 / SAMPLE_RATE
          previous_end = 0.0
          segments.each_with_index.filter_map do |segment, index|
            row = segment.respond_to?(:to_ary) ? segment.to_ary : nil
            raise ArgumentError, "Segment #{index} must contain exactly two bounds" unless row.is_a?(Array) && row.length == 2

            raw_start, raw_end = row
            start_time = Float(raw_start)
            end_time = Float(raw_end)
            raise ArgumentError, "Segment #{index} has non-finite bounds" unless start_time.finite? && end_time.finite?
            if start_time < -tolerance || end_time > duration + tolerance
              raise ArgumentError,
                    format("Segment %<index>d lies outside the audio: %<start>.6f..%<end>.6f for %<duration>.6fs",
                           index: index, start: start_time, end: end_time, duration: duration)
            end

            start_time = start_time.clamp(0.0, duration)
            end_time = end_time.clamp(0.0, duration)
            next if end_time <= start_time
            raise ArgumentError, "Segment #{index} overlaps or is out of order" if start_time < previous_end - tolerance

            start_time = previous_end if start_time < previous_end
            if max_duration && end_time - start_time > max_duration + tolerance
              raise ArgumentError,
                    format("Segment %<index>d is %<duration>.6fs, exceeding the %<max>gs single-row limit",
                           index: index, duration: end_time - start_time, max: max_duration)
            end
            previous_end = end_time
            [start_time, end_time]
          rescue TypeError, ArgumentError => e
            raise e if e.message.start_with?("Segment ")

            raise ArgumentError, "Segment #{index} has invalid bounds: #{e.message}"
          end
        end

        def samples_to_seconds(timestamps, audio_samples)
          duration = audio_samples.fdiv(SAMPLE_RATE)
          segments = timestamps.filter_map do |item|
            start_sample = fetch(item, :start)
            end_sample = fetch(item, :end)
            start_time = [0.0, Float(start_sample).fdiv(SAMPLE_RATE)].max
            end_time = [duration, Float(end_sample).fdiv(SAMPLE_RATE)].min
            [start_time, end_time] if end_time > start_time
          end
          validate(segments, duration)
        end

        def merge_speech(segments, max_duration)
          raise ArgumentError, "max_duration must be finite and positive" unless real_finite?(max_duration) && max_duration.positive?
          return [] if segments.empty?

          first = segment_bounds(segments.first, 0)
          current_start, current_end = first.map { |value| Float(value) }
          if current_start.negative? || current_end < current_start
            raise ArgumentError, "VAD segments must have non-negative ordered bounds"
          end

          merged = []
          segments.drop(1).each_with_index do |segment, offset|
            raw_start, raw_end = segment_bounds(segment, offset + 1)
            start_time = Float(raw_start)
            end_time = Float(raw_end)
            raise ArgumentError, "VAD segments must be sorted and non-overlapping" if start_time < current_end || end_time < start_time

            if end_time - current_start <= max_duration + 1e-9
              current_end = end_time
            else
              merged << [current_start, current_end]
              current_start = start_time
              current_end = end_time
            end
          end
          merged << [current_start, current_end]
        end

        def segment_bounds(segment, index)
          row = segment.respond_to?(:to_ary) ? segment.to_ary : nil
          return row if row.is_a?(Array) && row.length == 2

          raise ArgumentError, "VAD segment #{index} must contain exactly two bounds"
        end
        private_class_method :segment_bounds

        # Auditok-equivalent energy segmentation. Auditok evaluates 50 ms PCM16
        # windows with 20*log10(RMS), tolerates internal silence, and cuts at a
        # hard maximum duration. Keeping this native avoids a Python dependency.
        def energy(audio, min_duration:, max_duration:, max_silence:, threshold:, analysis_window: 0.05)
          unless real_finite?(analysis_window) && analysis_window.positive?
            raise ArgumentError, "analysis_window must be finite and positive"
          end
          raise ArgumentError, "min_duration must be finite and positive" unless real_finite?(min_duration) && min_duration.positive?
          raise ArgumentError, "max_duration must be finite and positive" unless real_finite?(max_duration) && max_duration.positive?
          raise ArgumentError, "max_silence must be finite and non-negative" unless real_finite?(max_silence) && max_silence >= 0
          raise ArgumentError, "threshold must be finite" unless real_finite?(threshold)

          window_samples = (analysis_window * SAMPLE_RATE).to_i
          raise ArgumentError, "analysis_window must cover at least one audio sample" if window_samples.zero?

          frame_seconds = window_samples.fdiv(SAMPLE_RATE)
          minimum_frames = (min_duration / frame_seconds).ceil
          maximum_frames = ((max_duration / frame_seconds) + 1e-10).floor
          silence_frames = ((max_silence / frame_seconds) + 1e-10).floor
          raise ArgumentError, "max_duration must cover min_duration" if maximum_frames < minimum_frames
          raise ArgumentError, "max_silence must cover fewer analysis windows than max_duration" if silence_frames >= maximum_frames

          active = []
          frame_lengths = []
          (0...audio.length).step(window_samples) do |offset|
            finish = [offset + window_samples, audio.length].min
            count = finish - offset
            # The Python path multiplies decoded float32 samples by 32767 in
            # NumPy before converting them to PCM16. Round the products back
            # to IEEE-754 binary32 as one block so values just below an integer
            # PCM boundary behave identically (for example, 3 / 32767).
            scaled_pcm = Array.new(count) do |relative_index|
              Float(audio[offset + relative_index]).clamp(-1.0, 1.0) * 32_767.0
            end
            sum_squares = scaled_pcm.pack("e*").unpack("e*").sum do |value|
              pcm = value.to_i
              pcm * pcm
            end
            rms = Math.sqrt(sum_squares.fdiv([count, 1].max))
            energy_db = 20.0 * Math.log10([rms, 1e-10].max)
            active << (energy_db >= threshold)
            frame_lengths << count
          end

          tokenize_energy_frames(
            active,
            frame_lengths,
            minimum_frames,
            maximum_frames,
            silence_frames,
            frame_seconds
          )
        end

        def tokenize_energy_frames(active, frame_lengths, minimum, maximum, maximum_silence, frame_seconds)
          results = []
          state = :silence
          start_frame = nil
          data_frames = []
          silence_length = 0
          contiguous_token = false

          deliver = lambda do |current_frame, truncated|
            accepted = data_frames.length >= minimum || (!data_frames.empty? && contiguous_token)
            if accepted
              start_time = start_frame * frame_seconds
              sample_count = data_frames.sum { |frame| frame_lengths.fetch(frame) }
              duration = sample_count.fdiv(SAMPLE_RATE)
              results << [start_time, start_time + duration]
              if truncated
                start_frame = current_frame + 1
                contiguous_token = true
              else
                contiguous_token = false
              end
            else
              contiguous_token = false
            end
            data_frames = []
          end

          active.each_with_index do |speech, frame|
            case state
            when :silence
              next unless speech

              start_frame = frame
              silence_length = 0
              data_frames = [frame]
              state = :noise
              deliver.call(frame, true) if data_frames.length >= maximum
            when :noise
              if speech
                data_frames << frame
                deliver.call(frame, true) if data_frames.length >= maximum
              elsif maximum_silence <= 0
                state = :silence
                deliver.call(frame, false)
              else
                silence_length = 1
                data_frames << frame
                state = :possible_silence
                deliver.call(frame, true) if data_frames.length == maximum
              end
            when :possible_silence
              if speech
                data_frames << frame
                silence_length = 0
                state = :noise
                deliver.call(frame, true) if data_frames.length >= maximum
              elsif silence_length >= maximum_silence
                state = :silence
                if silence_length < data_frames.length
                  deliver.call(frame, false)
                else
                  data_frames = []
                  silence_length = 0
                end
              else
                data_frames << frame
                silence_length += 1
                deliver.call(frame, true) if data_frames.length >= maximum
              end
            end
          end

          if %i[noise possible_silence].include?(state) &&
             !data_frames.empty? && data_frames.length > silence_length
            deliver.call(active.length, false)
          end
          results
        end
        private_class_method :tokenize_energy_frames

        def fetch(value, key)
          return value.public_send(key) if value.respond_to?(key)
          return value[key] if value.respond_to?(:key?) && value.key?(key)
          return value[key.to_s] if value.respond_to?(:key?) && value.key?(key.to_s)

          raise KeyError, "missing #{key}"
        end
        private_class_method :fetch

        def real_finite?(value)
          value.is_a?(Numeric) && !value.is_a?(Complex) && value.finite?
        end
        private_class_method :real_finite?
      end
    end
  end
end
