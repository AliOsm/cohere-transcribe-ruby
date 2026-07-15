# frozen_string_literal: true

require "json"

require_relative "../python_text"
require_relative "../types"

module Cohere
  module Transcribe
    module Output
      module Rendering
        module_function

        SENTENCE_ENDINGS = [".", "!", "?", "؟", "،", "؛", "…"].freeze

        def build_cues(words, max_chars:, max_duration:, max_gap:, min_cue_duration: 0.30, media_duration: nil)
          cue_words = []
          current = []
          words.each do |word|
            if current.any?
              candidate = (current.map { |item| field(item, :text) } + [field(word, :text)]).join(" ")
              gap = field(word, :start) - field(current.last, :end)
              duration = field(word, :end) - field(current.first, :start)
              if candidate.length > max_chars || duration > max_duration || gap > max_gap
                cue_words << current
                current = []
              end
            end
            current << word
            if SENTENCE_ENDINGS.any? { |ending| field(word, :text).end_with?(ending) }
              cue_words << current
              current = []
            end
          end
          cue_words << current if current.any?

          cues = cue_words.map do |items|
            SubtitleCue.new(
              start: [0.0, field(items.first, :start)].max,
              end: [field(items.first, :start), field(items.last, :end)].max,
              text: PythonText.strip(items.map { |item| field(item, :text) }.join(" "))
            )
          end
          cues.each_with_index.map do |cue, index|
            start_time = media_duration ? [cue.start, media_duration].min : cue.start
            end_time = media_duration ? [cue.end, media_duration].min.clamp(start_time, media_duration) : cue.end
            next_start = index + 1 < cues.length ? cues[index + 1].start : Float::INFINITY
            upper_bound = [next_start, media_duration || Float::INFINITY].min
            desired_end = [end_time, start_time + min_cue_duration].max
            cue.with(start: start_time, end: [[desired_end, upper_bound].min, start_time].max)
          end.freeze
        end

        def timestamp(seconds, include_hours: false, marker: ".")
          milliseconds = ([0.0, seconds].max * 1000).round(half: :even)
          hours, milliseconds = milliseconds.divmod(3_600_000)
          minutes, milliseconds = milliseconds.divmod(60_000)
          whole_seconds, milliseconds = milliseconds.divmod(1000)
          if include_hours || hours.positive?
            format("%<hours>02d:%<minutes>02d:%<seconds>02d%<marker>s%<milliseconds>03d",
                   hours: hours, minutes: minutes, seconds: whole_seconds, marker: marker, milliseconds: milliseconds)
          else
            format("%<minutes>02d:%<seconds>02d%<marker>s%<milliseconds>03d",
                   minutes: minutes, seconds: whole_seconds, marker: marker, milliseconds: milliseconds)
          end
        end

        def plain_text(lines)
          normalized = lines.filter_map do |line|
            text = PythonText.strip(line.to_s)
            text unless text.empty?
          end
          "#{normalized.join("\n")}\n"
        end

        def srt(cues)
          cues.each_with_index.map do |cue, index|
            "#{index + 1}\n#{timestamp(field(cue, :start), include_hours: true, marker: ",")} --> " \
              "#{timestamp(field(cue, :end), include_hours: true, marker: ",")}\n#{field(cue, :text)}\n\n"
          end.join
        end

        def vtt(cues)
          body = cues.map do |cue|
            "#{timestamp(field(cue, :start))} --> #{timestamp(field(cue, :end))}\n#{field(cue, :text)}\n\n"
          end.join
          "WEBVTT\n\n#{body}"
        end

        def json(payload)
          "#{JSON.pretty_generate(payload)}\n"
        end

        def field(value, name)
          return value.public_send(name) if value.respond_to?(name)
          return value[name] if value.respond_to?(:key?) && value.key?(name)
          return value[name.to_s] if value.respond_to?(:key?) && value.key?(name.to_s)

          raise KeyError, "missing #{name}"
        end
        private_class_method :field
      end
    end
  end
end
