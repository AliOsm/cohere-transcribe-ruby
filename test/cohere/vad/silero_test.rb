# frozen_string_literal: true

require "digest"
require "test_helper"
require "cohere/transcribe/vad/silero"

class Cohere::Transcribe::VAD::SileroTest < Minitest::Test
  Silero = Cohere::Transcribe::VAD::Silero

  class SyntheticAudio
    attr_reader :length

    def initialize(length)
      @length = length
    end

    def [](index)
      ((index % 2_003) - 1_001).fdiv(2_003)
    end
  end

  class RecordingSession
    attr_reader :calls

    def initialize(probability: 0.1)
      @probability = probability
      @calls = []
    end

    def run(output_names, feed)
      rows = feed.fetch("input")
      call_number = calls.length + 1
      calls << {
        output_names: output_names,
        row_count: rows.length,
        row_widths: rows.map(&:length).uniq,
        first_context: rows.first.first(64),
        second_context: rows.length > 1 ? rows[1].first(64) : nil,
        last_row: rows.last.dup,
        hidden_first: feed.fetch("h").fetch(0).fetch(0).fetch(0),
        cell_first: feed.fetch("c").fetch(0).fetch(0).fetch(0)
      }
      hidden = [[Array.new(128, call_number.to_f)]]
      cell = [[Array.new(128, 100.0 + call_number)]]
      [Array.new(rows.length, @probability), hidden, cell]
    end
  end

  def test_packaged_asset_has_the_validated_identity_and_notices
    assert_equal 1_249_744, File.size(Silero::MODEL_PATH)
    assert_equal(
      "914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9",
      Digest::SHA256.file(Silero::MODEL_PATH).hexdigest
    )
    directory = File.dirname(Silero::MODEL_PATH)
    assert_path_exists File.join(directory, "LICENSE.silero-vad")
    assert_path_exists File.join(directory, "LICENSE.faster-whisper")
    assert_path_exists File.join(directory, "ATTRIBUTION.md")
  end

  def test_exactly_divisible_audio_has_no_extra_frame
    session = RecordingSession.new
    probabilities = Silero.new(session: session).speech_probabilities(SyntheticAudio.new(1_024))

    assert_equal 2, probabilities.length
    assert_equal([2], session.calls.map { |call| call.fetch(:row_count) })
  end

  def test_context_and_recurrent_state_cross_the_256_frame_boundary
    audio = SyntheticAudio.new(257 * 512)
    session = RecordingSession.new
    probabilities = Silero.new(session: session).speech_probabilities(audio)

    assert_equal 257, probabilities.length
    assert_equal([256, 1], session.calls.map { |call| call.fetch(:row_count) })
    assert_equal([[576], [576]], session.calls.map { |call| call.fetch(:row_widths) })
    assert_equal Array.new(64, 0.0), session.calls[0].fetch(:first_context)
    assert_equal(
      (448...512).map { |index| audio[index] },
      session.calls[0].fetch(:second_context)
    )
    assert_equal(
      (((256 * 512) - 64)...(256 * 512)).map { |index| audio[index] },
      session.calls[1].fetch(:first_context)
    )
    assert_equal 1.0, session.calls[1].fetch(:hidden_first)
    assert_equal 101.0, session.calls[1].fetch(:cell_first)
  end

  def test_configured_block_frames_bounds_every_temporal_model_call
    session = RecordingSession.new
    model = Silero.new(session: session, block_frames: 3)

    probabilities = model.speech_probabilities(SyntheticAudio.new(7 * 512))

    assert_equal 7, probabilities.length
    assert_equal([3, 3, 1], session.calls.map { |call| call.fetch(:row_count) })
    assert_equal 3, model.block_frames
    assert_equal 3, model.last_execution.model_calls
    assert_equal 7, model.last_execution.valid_frames
    assert_equal 7, model.last_execution.padded_frames
    assert_equal 1, model.last_execution.max_files_per_call
    assert_equal 3, model.last_execution.effective_block_frames
  end

  def test_configured_block_larger_than_direct_onnx_default_is_honored_exactly
    session = RecordingSession.new
    model = Silero.new(session: session, block_frames: 300)

    model.speech_probabilities(SyntheticAudio.new(601 * 512))

    assert_equal([300, 300, 1], session.calls.map { |call| call.fetch(:row_count) })
    assert_equal 300, model.block_frames
    assert_equal 300, model.last_execution.effective_block_frames
  end

  def test_partial_final_frame_is_zero_padded_without_mutating_context
    audio = SyntheticAudio.new(513)
    session = RecordingSession.new
    Silero.new(session: session).speech_probabilities(audio)

    last_row = session.calls.fetch(0).fetch(:last_row)
    assert_equal (448...512).map { |index| audio[index] }, last_row.first(64)
    assert_equal audio[512], last_row[64]
    assert_equal Array.new(511, 0.0), last_row.drop(65)
  end

  def test_each_audio_file_starts_with_fresh_recurrent_and_waveform_state
    session = RecordingSession.new
    model = Silero.new(session: session)
    2.times { model.speech_probabilities(SyntheticAudio.new(512)) }

    assert_equal([0.0, 0.0], session.calls.map { |call| call.fetch(:hidden_first) })
    assert_equal([0.0, 0.0], session.calls.map { |call| call.fetch(:cell_first) })
    assert_equal([Array.new(64, 0.0)] * 2, session.calls.map { |call| call.fetch(:first_context) })
  end

  def test_session_is_lazy_and_constructed_with_reference_cpu_options
    calls = []
    session = RecordingSession.new
    factory = lambda do |path, **options|
      calls << [path, options]
      session
    end
    model = Silero.new(session_factory: factory)

    assert_empty model.speech_probabilities([])
    assert_empty calls
    model.speech_probabilities([0.0])
    model.speech_probabilities([0.0])

    assert_equal 1, calls.length
    assert_equal Silero::MODEL_PATH, calls[0][0]
    assert_equal Silero::SESSION_OPTIONS, calls[0][1]
  end

  def test_execution_reports_per_call_load_inference_and_timestamp_timings
    session = RecordingSession.new
    factory = lambda do |_path, **_options|
      sleep 0.005
      session
    end
    session.define_singleton_method(:run) do |output_names, feed|
      sleep 0.005
      super(output_names, feed)
    end
    model = Silero.new(session_factory: factory)

    model.speech_timestamps(Array.new(512, 0.0))
    first = model.last_execution

    assert_operator first.model_load_seconds, :>=, 0.004
    assert_operator first.inference_seconds, :>=, 0.004
    assert_operator first.postprocess_seconds, :>, 0.0

    model.speech_timestamps(Array.new(512, 0.0))
    second = model.last_execution
    assert_equal 0.0, second.model_load_seconds
    assert_operator second.inference_seconds, :>=, 0.004
    assert_operator second.postprocess_seconds, :>, 0.0
  end

  def test_requested_threads_configure_the_session_and_reported_options
    calls = []
    factory = lambda do |path, **options|
      calls << [path, options]
      RecordingSession.new
    end
    model = Silero.new(session_factory: factory, threads: 4)

    model.speech_probabilities([0.0])

    assert_equal 4, model.intra_op_threads
    assert_equal 4, calls.dig(0, 1, :intra_op_num_threads)
    assert_equal({ Silero::PROVIDER => {} }, model.provider_options)
  end

  def test_standalone_tuning_values_are_positive_integers
    assert_raises(ArgumentError) { Silero.new(block_frames: 0) }
    assert_raises(ArgumentError) { Silero.new(threads: 0) }
    assert_raises(ArgumentError) { Silero.new(threads: "2") }
  end

  def test_invalid_audio_and_model_output_shapes_are_rejected
    assert_raises(ArgumentError) { Silero.new(session: RecordingSession.new).speech_probabilities([[0.0]]) }

    bad_session = Object.new
    def bad_session.run(_outputs, _feed)
      [[[0.1]], [[Array.new(128, 0.0)]], [[Array.new(128, 0.0)]]]
    end
    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      Silero.new(session: bad_session).speech_probabilities([0.0])
    end
    assert_match(/non-vector/, error.message)
  end

  def test_real_onnx_probabilities_match_the_reference_export
    require "onnxruntime"
    model = Silero.new
    actual = model.speech_probabilities(Array.new(1_025, 0.0))
    expected = [0.0016697943, 0.0068838596, 0.0089106858]

    assert_equal expected.length, actual.length
    expected.zip(actual).each do |reference, probability|
      assert_in_delta reference, probability, 2e-6
    end
  rescue LoadError
    skip "onnxruntime gem is not installed"
  end

  def test_packaged_graph_exposes_one_recurrent_stream_and_no_file_batch_axis
    require "onnxruntime"
    session = OnnxRuntime::InferenceSession.new(Silero::MODEL_PATH, **Silero::SESSION_OPTIONS)
    inputs = session.inputs.to_h { |input| [input.fetch(:name), input.fetch(:shape)] }
    outputs = session.outputs.to_h { |output| [output.fetch(:name), output.fetch(:shape)] }

    assert_equal ["seq_len", 576], inputs.fetch("input")
    assert_equal [1, 1, 128], inputs.fetch("h")
    assert_equal [1, 1, 128], inputs.fetch("c")
    assert_equal [1, 1, 128], outputs.fetch("hn")
    assert_equal [1, 1, 128], outputs.fetch("cn")
    refute_includes session.inputs.flat_map { |input| input.fetch(:shape) }, "batch"
  rescue LoadError
    skip "onnxruntime gem is not installed"
  end

  def test_numo_fast_path_matches_array_path_across_chunk_boundary
    require "numo/narray"
    require "onnxruntime"
    length = (257 * 512) + 17
    audio = ::Numo::SFloat.new(length).seq
    audio = ::Numo::NMath.sin(audio * 0.013) * 0.2
    model = Silero.new

    assert_equal model.speech_probabilities(audio.to_a), model.speech_probabilities(audio)
  rescue LoadError
    skip "numo-narray and onnxruntime gems are not installed"
  end
end
