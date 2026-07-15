# frozen_string_literal: true

require "test_helper"

module Cohere
  module Transcribe
    module Output
      class RenderingTest < Minitest::Test
        def test_build_cues_splits_on_character_duration_gap_and_sentence_limits
          assert_equal %w[abc def], cue_texts(words(%w[abc def]), max_chars: 6)

          duration_words = [word(0.0, 0.5, "one"), word(1.5, 2.5, "two")]
          assert_equal %w[one two], cue_texts(duration_words, max_duration: 2.0)

          gap_words = [word(0.0, 0.2, "one"), word(1.0, 1.2, "two")]
          assert_equal %w[one two], cue_texts(gap_words, max_gap: 0.5)

          sentence_words = [word(0.0, 0.2, "مرحبا؟"), word(0.2, 0.4, "التالي")]
          assert_equal ["مرحبا؟", "التالي"], cue_texts(sentence_words)
        end

        def test_build_cues_accepts_hashes_and_normalizes_bounds
          input = [
            { "start" => -0.2, "end" => -0.1, "text" => "first" },
            { start: 0.1, end: 0.2, text: "second." }
          ]
          cues = Rendering.build_cues(input, max_chars: 80, max_duration: 6.0, max_gap: 0.6)

          assert_equal 1, cues.length
          assert_equal 0.0, cues.first.start
          assert_equal 0.3, cues.first.end
          assert_equal "first second.", cues.first.text
          assert cues.frozen?
        end

        def test_minimum_cue_duration_never_overlaps_next_cue_or_media_end
          input = [word(0.0, 0.1, "one."), word(0.2, 0.25, "two.")]
          cues = Rendering.build_cues(
            input,
            max_chars: 80,
            max_duration: 6.0,
            max_gap: 0.6,
            min_cue_duration: 0.3,
            media_duration: 0.4
          )

          assert_equal([[0.0, 0.2], [0.2, 0.4]], cues.map { |cue| [cue.start, cue.end] })

          beyond_media = Rendering.build_cues(
            [word(2.0, 3.0, "late")],
            max_chars: 80,
            max_duration: 6.0,
            max_gap: 0.6,
            media_duration: 1.0
          )
          assert_equal [1.0, 1.0], [beyond_media.first.start, beyond_media.first.end]
        end

        def test_timestamp_uses_python_half_even_rounding_and_rollover
          assert_equal "00:00.000", Rendering.timestamp(-1)
          assert_equal "00:00.000", Rendering.timestamp(0.0005)
          assert_equal "00:00.002", Rendering.timestamp(0.0015)
          assert_equal "01:01:01,234", Rendering.timestamp(3_661.234, include_hours: true, marker: ",")
          assert_equal "01:00:00.000", Rendering.timestamp(3_600.0)
        end

        def test_plain_text_strips_blank_lines_and_ends_with_newline
          assert_equal "one\ntwo\n", Rendering.plain_text([" one ", "", " \n", "two"])
          assert_equal "\n", Rendering.plain_text([])
          assert_equal "one\ntwo\n", Rendering.plain_text(["\u00A0one\u2003", "\u202F", "two\u001C"])
        end

        def test_cue_text_uses_python_unicode_strip_semantics
          input = [word(0.0, 0.2, "\u00A0one"), word(0.2, 0.4, "two\u3000")]

          assert_equal ["one two"], cue_texts(input)
        end

        def test_srt_and_vtt_render_exact_formats_from_data_and_hash_cues
          cues = [
            SubtitleCue.new(start: 0.0, end: 1.25, text: "Hello"),
            { "start" => 61.0, "end" => 62.5, "text" => "World" }
          ]

          assert_equal(
            "1\n00:00:00,000 --> 00:00:01,250\nHello\n\n" \
            "2\n00:01:01,000 --> 00:01:02,500\nWorld\n\n",
            Rendering.srt(cues)
          )
          assert_equal(
            "WEBVTT\n\n00:00.000 --> 00:01.250\nHello\n\n" \
            "01:01.000 --> 01:02.500\nWorld\n\n",
            Rendering.vtt(cues)
          )
        end

        def test_json_is_pretty_unicode_and_newline_terminated
          rendered = Rendering.json({ "text" => "مرحبا", "items" => [1, 2] })

          assert rendered.end_with?("\n")
          assert_includes rendered, "مرحبا"
          assert_equal({ "text" => "مرحبا", "items" => [1, 2] }, JSON.parse(rendered))
          assert_includes rendered, "\n  \"items\": [\n"
        end

        private

        def word(start_time, end_time, text)
          TranscriptionWord.new(
            start: start_time,
            end: end_time,
            text: text,
            segment_index: 0,
            segment_word_index: 0,
            timing_source: "test"
          )
        end

        def words(texts)
          texts.each_with_index.map { |text, index| word(index * 0.2, (index + 1) * 0.2, text) }
        end

        def cue_texts(input, max_chars: 80, max_duration: 6.0, max_gap: 0.6)
          Rendering.build_cues(
            input,
            max_chars: max_chars,
            max_duration: max_duration,
            max_gap: max_gap
          ).map(&:text)
        end
      end
    end
  end
end
