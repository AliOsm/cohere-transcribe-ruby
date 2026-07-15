# frozen_string_literal: true

require "test_helper"

module Cohere
  module Transcribe
    module Audio
      class SegmentationTest < Minitest::Test
        def test_fixed_windows_cover_every_sample_exactly
          total = (65 * SAMPLE_RATE) + (SAMPLE_RATE / 4)

          assert_equal [[0.0, 30.0], [30.0, 60.0], [60.0, 65.25]], Segmentation.fixed(total, 30.0)
          assert_empty Segmentation.fixed(0, 30.0)

          ranges = Segmentation.fixed(5_000, 1_001.fdiv(SAMPLE_RATE)).map do |start_time, end_time|
            [(start_time * SAMPLE_RATE).round, (end_time * SAMPLE_RATE).round]
          end
          assert_equal [[0, 1_001], [1_001, 2_002], [2_002, 3_003], [3_003, 4_004], [4_004, 5_000]], ranges
        end

        def test_fixed_windows_use_python_half_even_rounding
          segments = Segmentation.fixed(5, 2.5.fdiv(SAMPLE_RATE))

          assert_equal([[0, 2], [2, 4], [4, 5]], segments.map do |bounds|
            bounds.map do |value|
              (value * SAMPLE_RATE).round
            end
          end)
        end

        def test_fixed_windows_validate_length_and_window
          assert_raises(ArgumentError) { Segmentation.fixed(-1, 1.0) }
          [0, -1, Float::NAN, Float::INFINITY, Complex(1, 0), "1"].each do |window|
            assert_raises(ArgumentError) { Segmentation.fixed(1, window) }
          end
        end

        def test_validate_clamps_subsample_drift_drops_empty_and_snaps_tiny_overlap
          tolerance = 2.0 / SAMPLE_RATE
          segments = [
            [-tolerance / 2, 0.5],
            [0.5 - (tolerance / 2), 0.5],
            [0.5 - (tolerance / 2), 1.0 + (tolerance / 2)]
          ]

          # Python checks emptiness before snapping a tolerated overlap to the
          # prior end, so the resulting zero-length boundary row is retained.
          assert_equal [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0]], Segmentation.validate(segments, 1.0)
        end

        def test_validate_rejects_malformed_nonfinite_outside_overlap_and_long_rows
          assert_raises(ArgumentError) { Segmentation.validate([[0, 1, 2]], 2.0) }
          assert_raises(ArgumentError) { Segmentation.validate([[0, Float::NAN]], 2.0) }
          assert_raises(ArgumentError) { Segmentation.validate([[-1, 1]], 2.0) }
          assert_raises(ArgumentError) { Segmentation.validate([[0, 1], [0.5, 2]], 2.0) }
          assert_raises(ArgumentError) { Segmentation.validate([[0, 1.1]], 2.0, max_duration: 1.0) }
          assert_raises(ArgumentError) { Segmentation.validate([], Float::INFINITY) }
        end

        def test_sample_timestamps_are_clipped_and_validated
          timestamps = [
            { start: -100, end: 512 },
            { "start" => 512, "end" => 20_000 },
            Struct.new(:start, :end).new(20_000, 20_001)
          ]

          assert_equal [[0.0, 0.032], [0.032, 1.0]], Segmentation.samples_to_seconds(timestamps, SAMPLE_RATE)
        end

        def test_merge_speech_uses_full_timeline_and_preserves_gaps
          segments = [[0.2, 4.0], [4.4, 9.8], [15.0, 24.0], [24.5, 31.0]]

          assert_equal [[0.2, 9.8], [15.0, 24.0], [24.5, 31.0]], Segmentation.merge_speech(segments, 10.0)
          assert_empty Segmentation.merge_speech([], 10.0)
          assert_raises(ArgumentError) { Segmentation.merge_speech([[0, 2], [1, 3]], 30.0) }
          assert_raises(ArgumentError) { Segmentation.merge_speech([[0, 1, 2]], 30.0) }
          assert_raises(ArgumentError) { Segmentation.merge_speech([nil], 30.0) }
          assert_raises(ArgumentError) { Segmentation.merge_speech([[0, 1]], 0) }
        end

        def test_energy_finds_tone_and_includes_tolerated_trailing_silence
          require "numo/narray"

          audio = Numo::SFloat.zeros(SAMPLE_RATE)
          start_sample = (0.2 * SAMPLE_RATE).to_i
          end_sample = (0.7 * SAMPLE_RATE).to_i
          phase = Numo::SFloat.new(end_sample - start_sample).seq
          audio[start_sample...end_sample] = 0.5 * Numo::NMath.sin(2 * Math::PI * 440 * phase / SAMPLE_RATE)

          segments = Segmentation.energy(
            audio,
            min_duration: 0.1,
            max_duration: 2.0,
            max_silence: 0.05,
            threshold: 50.0
          )

          assert_equal 1, segments.length
          assert_in_delta 0.2, segments[0][0], 0.01
          assert_in_delta 0.75, segments[0][1], 0.01
        end

        def test_energy_hard_maximum_keeps_short_contiguous_remainder
          audio = Array.new((0.23 * SAMPLE_RATE).round, 0.5)

          assert_equal(
            [[0.0, 0.2], [0.2, 0.23]],
            Segmentation.energy(
              audio,
              min_duration: 0.1,
              max_duration: 0.2,
              max_silence: 0.05,
              threshold: 50.0
            )
          )
        end

        def test_energy_rounds_float32_pcm_scaling_like_numpy
          require "numo/narray"

          # Decoded audio is float32. NumPy keeps the multiplication by 32767
          # in float32, so this value rounds back to PCM integer 3 before the
          # int16 cast. Multiplying the extracted scalar as a Ruby Float would
          # instead produce 2.999999997... and incorrectly truncate it to 2.
          audio = Numo::SFloat.ones(3 * 800) * (3.0 / 32_767.0)

          assert_equal(
            [[0.0, 0.15]],
            Segmentation.energy(
              audio,
              min_duration: 0.15,
              max_duration: 1.0,
              max_silence: 0.05,
              threshold: 20.0 * Math.log10(3)
            )
          )
        end

        def test_energy_partial_final_frame_uses_auditok_time_arithmetic
          audio = Array.new(18 * 800, 0.0) + [0.5, 0.5]

          assert_equal(
            [[0.9, 0.9001250000000001]],
            Segmentation.energy(
              audio,
              min_duration: 1.fdiv(SAMPLE_RATE),
              max_duration: 1.0,
              max_silence: 0.05,
              threshold: 50.0
            )
          )
        end

        def test_energy_matches_auditok_silence_state_transitions
          # Each value is one 50 ms analysis frame. A tolerated silent frame is
          # retained, while the frame that exceeds max_silence ends the token.
          waveform = [1, 1, 0, 1, 0, 0, 1, 1].flat_map do |speech|
            Array.new(800, speech == 1 ? 0.5 : 0.0)
          end

          assert_equal(
            # Auditok derives the start from frame_index * frame_duration,
            # retaining the same observable binary64 rounding as Python.
            [[0.0, 0.25], [0.30000000000000004, 0.4]],
            Segmentation.energy(
              waveform,
              min_duration: 0.1,
              max_duration: 1.0,
              max_silence: 0.05,
              threshold: 50.0
            )
          )
        end

        def test_energy_validates_quantized_duration_settings
          base = { min_duration: 0.1, max_duration: 1.0, max_silence: 0.05, threshold: 50.0 }
          assert_raises(ArgumentError) { Segmentation.energy([], **base, analysis_window: 0) }
          assert_raises(ArgumentError) { Segmentation.energy([], **base, min_duration: 0) }
          assert_raises(ArgumentError) { Segmentation.energy([], **base, max_duration: 0.05) }
          assert_raises(ArgumentError) { Segmentation.energy([], **base, max_silence: 1.0) }
          assert_raises(ArgumentError) { Segmentation.energy([], **base, threshold: Float::NAN) }
        end
      end
    end
  end
end
