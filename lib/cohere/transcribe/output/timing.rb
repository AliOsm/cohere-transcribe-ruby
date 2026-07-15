# frozen_string_literal: true

require_relative "../python_text"
require_relative "../types"

module Cohere
  module Transcribe
    module Output
      module Timing
        module_function

        def uniform_words(text, start_time, end_time, segment_index, timing_source = "uniform_segment")
          tokens = PythonText.split(text.to_s)
          return [] if tokens.empty?

          token_duration = [0.0, end_time - start_time].max.fdiv(tokens.length)
          tokens.each_with_index.map do |token, index|
            TranscriptionWord.new(
              start: start_time + (index * token_duration),
              end: start_time + ((index + 1) * token_duration),
              text: token,
              segment_index: segment_index,
              segment_word_index: index,
              timing_source: timing_source
            )
          end.freeze
        end

        def uniform_words_across_spans(text, spans, segment_index, timing_source = "uniform_speech_spans")
          tokens = PythonText.split(text.to_s)
          valid_spans = spans.select { |start_time, end_time| end_time > start_time }
          return [] if tokens.empty? || valid_spans.empty?

          counts = proportional_counts(tokens.length, valid_spans)
          token_offset = 0
          words = valid_spans.zip(counts).flat_map do |(start_time, end_time), count|
            next [] unless count.positive?

            duration = (end_time - start_time).fdiv(count)
            Array.new(count) do |local_index|
              token_index = token_offset + local_index
              TranscriptionWord.new(
                start: start_time + (local_index * duration),
                end: start_time + ((local_index + 1) * duration),
                text: tokens.fetch(token_index),
                segment_index: segment_index,
                segment_word_index: token_index,
                timing_source: timing_source
              )
            end.tap { token_offset += count }
          end
          raise "Speech-span token allocation did not preserve every token" unless token_offset == tokens.length

          words.freeze
        end

        def proportional_counts(token_count, spans)
          raise ArgumentError, "token_count must be non-negative" if token_count.negative?

          durations = spans.map { |start_time, end_time| [0.0, end_time - start_time].max }
          total = durations.sum
          return Array.new(spans.length, 0).freeze if token_count.zero? || total <= 0

          exact = durations.map { |duration| token_count * duration.fdiv(total) }
          counts = exact.map(&:floor)
          remaining = token_count - counts.sum
          order = spans.each_index.sort_by do |index|
            [-(exact[index] - counts[index]), -durations[index], index]
          end
          order.first(remaining).each { |index| counts[index] += 1 }
          counts.freeze
        end

        def spans_within(speech_spans, start_time, end_time)
          speech_spans.filter_map do |speech_start, speech_end|
            next unless speech_end > start_time && speech_start < end_time

            clipped_start = [start_time, speech_start].max
            clipped_end = [end_time, speech_end].min
            [clipped_start, clipped_end] if clipped_end > clipped_start
          end.freeze
        end
      end
    end
  end
end
