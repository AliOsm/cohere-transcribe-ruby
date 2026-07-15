# frozen_string_literal: true

require "test_helper"

module Cohere
  module Transcribe
    module Output
      class TimingTest < Minitest::Test
        def test_uniform_words_preserve_every_token_and_metadata
          words = Timing.uniform_words("  one\n two   three ", 1.0, 4.0, 7)

          assert_equal %w[one two three], words.map(&:text)
          assert_equal([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], words.map { |word| [word.start, word.end] })
          assert_equal [7, 7, 7], words.map(&:segment_index)
          assert_equal [0, 1, 2], words.map(&:segment_word_index)
          assert_equal ["uniform_segment"] * 3, words.map(&:timing_source)
          assert words.frozen?
        end

        def test_uniform_words_handle_blank_and_negative_duration
          assert_empty Timing.uniform_words(" \n ", 0.0, 1.0, 0)

          words = Timing.uniform_words("one two", 2.0, 1.0, 0, "fallback")
          assert_equal([[2.0, 2.0], [2.0, 2.0]], words.map { |word| [word.start, word.end] })
          assert_equal %w[fallback fallback], words.map(&:timing_source)
        end

        def test_uniform_words_split_every_python_unicode_whitespace_character
          separators = [
            *(0x0009..0x000D), *(0x001C..0x0020), 0x0085, 0x00A0, 0x1680,
            *(0x2000..0x200A), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000
          ].map { |codepoint| codepoint.chr(Encoding::UTF_8) }
          text = "#{separators.join}one#{separators.join}two#{separators.join}"

          assert_equal %w[one two], Timing.uniform_words(text, 0.0, 2.0, 0).map(&:text)
          assert_equal %w[one two], Timing.uniform_words_across_spans(text, [[0.0, 2.0]], 0).map(&:text)
          assert_equal ["one\u200Btwo"], Timing.uniform_words("one\u200Btwo", 0.0, 1.0, 0).map(&:text)
        end

        def test_proportional_counts_use_python_stable_tie_order
          assert_equal [1, 0], Timing.proportional_counts(1, [[0.0, 1.0], [2.0, 3.0]])
          assert_equal [0, 2], Timing.proportional_counts(2, [[0.0, 1.0], [2.0, 5.0]])
          assert_equal [1, 0, 0], Timing.proportional_counts(1, [[0, 1], [1, 2], [2, 3]])
        end

        def test_proportional_counts_cover_zero_and_invalid_durations
          assert_equal [0, 0], Timing.proportional_counts(0, [[0, 1], [2, 3]])
          assert_equal [0, 0], Timing.proportional_counts(4, [[1, 0], [2, 2]])
          assert_raises(ArgumentError) { Timing.proportional_counts(-1, []) }
        end

        def test_uniform_words_across_spans_do_not_stretch_over_silence
          words = Timing.uniform_words_across_spans(
            "one two three four",
            [[0.0, 1.0], [1.0, 1.0], [3.0, 4.0]],
            5
          )

          assert_equal %w[one two three four], words.map(&:text)
          assert_equal([[0.0, 0.5], [0.5, 1.0], [3.0, 3.5], [3.5, 4.0]], words.map { |word| [word.start, word.end] })
          assert_equal [0, 1, 2, 3], words.map(&:segment_word_index)
          assert_equal ["uniform_speech_spans"] * 4, words.map(&:timing_source)
          assert words.frozen?
        end

        def test_uniform_words_across_more_spans_than_tokens_is_deterministic
          words = Timing.uniform_words_across_spans("only", [[0, 1], [2, 3], [4, 5]], 0)

          assert_equal 1, words.length
          assert_equal [0, 1], [words.first.start, words.first.end]
          assert_empty Timing.uniform_words_across_spans("", [[0, 1]], 0)
          assert_empty Timing.uniform_words_across_spans("word", [[1, 1]], 0)
        end

        def test_spans_within_clips_boundaries_and_excludes_touches
          spans = [[-1, 0], [0, 1], [1, 2], [2, 3], [4, 5]]

          result = Timing.spans_within(spans, 0.5, 2.5)
          assert_equal [[0.5, 1], [1, 2], [2, 2.5]], result
          assert result.frozen?
        end
      end
    end
  end
end
