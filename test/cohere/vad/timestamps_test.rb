# frozen_string_literal: true

require "test_helper"
require "cohere/transcribe/vad/timestamps"

class Cohere::Transcribe::VAD::TimestampsTest < Minitest::Test
  Timestamps = Cohere::Transcribe::VAD::Timestamps

  def test_reference_hysteresis_and_padding_fixtures
    assert_equal(
      [{ start: 256, end: 1_792 }],
      timestamps([0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
    )
    assert_equal(
      [{ start: 256, end: 2_304 }],
      timestamps([0.1, 0.8, 0.4, 0.36, 0.34, 0.34, 0.34, 0.34])
    )
    assert_equal(
      [{ start: 0, end: 1_280 }, { start: 2_816, end: 4_352 }],
      timestamps([0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1])
    )
  end

  def test_reference_maximum_duration_cut_fixture
    probabilities = Array.new(20, 0.8)
    assert_equal(
      [
        { start: 0, end: 3_328 },
        { start: 3_328, end: 6_912 },
        { start: 6_912, end: 10_240 }
      ],
      timestamps(probabilities, max_speech_duration_s: 0.25)
    )
  end

  def test_minimum_speech_duration_is_strict
    options = default_options.merge(min_speech_duration_ms: 32, speech_pad_ms: 0)
    assert_empty Timestamps.from_probabilities(512, [0.8], **options)
    assert_equal(
      [{ start: 0, end: 1_024 }],
      Timestamps.from_probabilities(1_024, [0.8, 0.8], **options)
    )
  end

  def test_partial_last_frame_and_padding_are_clamped_to_audio
    assert_equal(
      [{ start: 0, end: 1_100 }],
      Timestamps.from_probabilities(
        1_100,
        [0.1, 0.8, 0.8],
        **default_options, speech_pad_ms: 60
      )
    )
  end

  def test_threshold_comparisons_follow_numpy_float32_coercion
    options = default_options.merge(
      min_speech_duration_ms: 0,
      min_silence_duration_ms: 0,
      speech_pad_ms: 0
    )

    # np.float32(0.01) is slightly below the binary64 representation of 0.01,
    # but NumPy casts its scalar threshold to float32 for this comparison.
    assert_equal(
      [{ start: 0, end: 512 }],
      Timestamps.from_probabilities(1_024, [0.01, 0.0], **options, threshold: 0.01)
    )
    # The same coercion applies to an explicit negative hysteresis threshold.
    assert_equal(
      [{ start: 0, end: 1_536 }],
      Timestamps.from_probabilities(
        2_048,
        [0.8, 0.35, 0.35, 0.0],
        **options,
        threshold: 0.5,
        neg_threshold: 0.35
      )
    )
  end

  def test_probability_contract_is_strict
    error = assert_raises(ArgumentError) do
      Timestamps.from_probabilities(513, [0.1], **default_options)
    end
    assert_match(/expected 2, got 1/, error.message)

    assert_raises(ArgumentError) do
      Timestamps.from_probabilities(512, [[0.1]], **default_options)
    end
    assert_raises(ArgumentError) do
      Timestamps.from_probabilities(512, [Float::NAN], **default_options)
    end
    assert_raises(ArgumentError) do
      Timestamps.from_probabilities(512, [1.01], **default_options)
    end
    assert_raises(ArgumentError) do
      Timestamps.from_probabilities(true, [], **default_options)
    end
    assert_raises(ArgumentError) do
      Timestamps.from_probabilities(0, [], **default_options, sampling_rate: 8_000)
    end
  end

  def test_long_probability_stream_checks_cancellation_periodically
    checks = 0
    cancel_check = -> { checks += 1 }
    Timestamps.from_probabilities(
      8_193 * 512,
      Array.new(8_193, 0.0),
      **default_options, cancel_check: cancel_check
    )
    assert_equal 6, checks
  end

  private

  def timestamps(probabilities, **overrides)
    Timestamps.from_probabilities(
      probabilities.length * 512,
      probabilities,
      **default_options, **overrides
    )
  end

  def default_options
    {
      sampling_rate: 16_000,
      threshold: 0.5,
      min_speech_duration_ms: 0,
      max_speech_duration_s: 30,
      min_silence_duration_ms: 64,
      speech_pad_ms: 16
    }
  end
end
