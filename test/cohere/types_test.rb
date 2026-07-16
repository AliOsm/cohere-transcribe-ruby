# frozen_string_literal: true

require "test_helper"
require "cohere/transcribe/errors"
require "cohere/transcribe/types"

class Cohere::Transcribe::TypesTest < Minitest::Test
  Types = Cohere::Transcribe

  def test_public_output_formats_are_directly_accepted
    assert_equal %w[txt srt vtt json], Cohere::Transcribe::OUTPUT_FORMATS
    assert_equal Cohere::Transcribe::OUTPUT_FORMATS,
                 Types::PublicationOptions.new(formats: Cohere::Transcribe::OUTPUT_FORMATS).formats
  end

  def test_publication_options_defaults_normalization_and_paths
    defaults = Types::PublicationOptions.new
    assert_nil defaults.formats
    assert_nil defaults.output_dir
    assert_equal "error", defaults.existing
    assert_nil defaults.profile_json

    options = Types::PublicationOptions.new(
      formats: %w[json txt json],
      output_dir: "transcripts",
      existing: "overwrite",
      profile_json: "run.json"
    )
    assert_equal %w[json txt], options.formats
    assert_equal Pathname("transcripts"), options.output_dir
    assert_equal "overwrite", options.existing
    assert_equal Pathname("run.json"), options.profile_json
    assert options.frozen?
    assert options.formats.frozen?

    empty_paths = Types::PublicationOptions.new(output_dir: "", profile_json: "")
    assert_equal Pathname("."), empty_paths.output_dir
    assert_equal Pathname("."), empty_paths.profile_json

    normalized_paths = Types::PublicationOptions.new(output_dir: "a//./b/", profile_json: "///")
    assert_equal Pathname("a/b"), normalized_paths.output_dir
    assert_equal Pathname("/"), normalized_paths.profile_json
    assert_equal Pathname("a/../b"), Types::PublicationOptions.new(output_dir: "a/../b").output_dir

    byte_paths = Types::PublicationOptions.new(
      output_dir: "transcripts-\xC3\xA9".b,
      profile_json: "profile-\xC3\xA9.json".b
    )
    assert_equal Encoding::UTF_8, byte_paths.output_dir.to_s.encoding
    assert_equal Encoding::UTF_8, byte_paths.profile_json.to_s.encoding
    assert_equal "transcripts-é", byte_paths.output_dir.to_s
    assert_equal "profile-é.json", byte_paths.profile_json.to_s
  end

  def test_publication_options_validate_formats_and_existing_policy
    error = assert_raises(ArgumentError) { Types::PublicationOptions.new(formats: []) }
    assert_equal "formats must contain at least one output format", error.message

    error = assert_raises(ArgumentError) do
      Types::PublicationOptions.new(formats: %w[txt pdf csv])
    end
    assert_equal "Unsupported output format(s): csv, pdf", error.message

    error = assert_raises(ArgumentError) do
      Types::PublicationOptions.new(existing: "append")
    end
    assert_equal "existing must be 'error', 'overwrite', or 'skip'", error.message

    error = assert_raises(ArgumentError) do
      Types::PublicationOptions.new(output_dir: "transcripts-\xE9".b)
    end
    assert_equal "output_dir must contain valid UTF-8", error.message

    error = assert_raises(ArgumentError) do
      Types::PublicationOptions.new(profile_json: "profile-\xE9.json".b)
    end
    assert_equal "profile_json must contain valid UTF-8", error.message
  end

  def test_transcription_options_mirror_reference_defaults
    options = Types::TranscriptionOptions.new

    assert_equal(
      {
        model: "CohereLabs/cohere-transcribe-arabic-07-2026",
        model_revision: nil,
        adapter: nil,
        adapter_revision: nil,
        language: "ar",
        text_only: false,
        recursive: true,
        device: "auto",
        dtype: "auto",
        audio_backend: "auto",
        audio_memory_gb: 4.0,
        preprocess_workers: nil,
        pipeline_preparation: true,
        vad: "silero",
        vad_engine: "auto",
        vad_batch_size: 16,
        vad_block_frames: 512,
        vad_threads: nil,
        vad_merge: false,
        min_dur: 0.5,
        max_dur: 30.0,
        max_silence: 0.6,
        energy_threshold: 50.0,
        vad_threshold: 0.5,
        min_silence_ms: 300,
        speech_pad_ms: 60,
        batch_size: nil,
        batch_max_size: nil,
        batch_audio_seconds: nil,
        batch_vram_target: 0.9,
        adaptive_batch: false,
        pin_memory: false,
        max_new_tokens: 445,
        max_retry_tokens: 896,
        truncation_policy: "retry",
        stop_repetition_loops: true,
        alignment: "segment",
        align_batch_size: 4,
        align_dtype: "fp32",
        max_chars: 80,
        max_cue_dur: 6.0,
        max_gap: 0.6,
        publication: nil
      },
      options.to_h
    )
    assert options.frozen?
    assert_raises(NoMethodError) { options.language = "en" }
  end

  def test_model_and_adapter_references_normalize_valid_utf8_bytes
    options = Types::TranscriptionOptions.new(
      model: "owner/modèle".b,
      model_revision: "révision".b,
      adapter: "owner/adaptateur-é".b,
      adapter_revision: "révision-adapter".b
    )

    %i[model model_revision adapter adapter_revision].each do |field|
      assert_equal Encoding::UTF_8, options.public_send(field).encoding
      assert_predicate options.public_send(field), :valid_encoding?
    end
  end

  def test_all_result_value_objects_expose_the_reference_fields
    segment = Types::TranscriptionSegment.new(index: 0, start: 0.0, end: 1.0, text: "hello")
    word = Types::TranscriptionWord.new(
      start: 0.0,
      end: 0.5,
      text: "hello",
      segment_index: 0,
      segment_word_index: 0,
      timing_source: "ctc"
    )
    cue = Types::SubtitleCue.new(start: 0.0, end: 1.0, text: "hello")
    provenance = Types::TranscriptionProvenance.new
    result = Types::TranscriptionResult.new(
      path: Pathname("/audio/clip.wav"),
      relative_path: Pathname("clip.wav"),
      status: "completed",
      text: "hello",
      duration: 1.0,
      segments: [segment],
      words: [word],
      cues: [cue]
    )

    assert_equal %i[index start end text], segment.to_h.keys
    assert_equal %i[start end text segment_index segment_word_index timing_source], word.to_h.keys
    assert_equal %i[start end text], cue.to_h.keys
    assert_equal(
      {
        model_id: nil,
        model_revision: nil,
        model_format: nil,
        adapter_id: nil,
        adapter_revision: nil,
        decode_backend: nil,
        decode_fallback_reason: nil,
        vad_engine_requested: nil,
        vad_engine_actual: nil,
        vad_provider: nil,
        vad_fallback_reason: nil,
        fallback_alignment_segments: 0,
        repetition_stopped_segments: [],
        truncation_retried_segments: [],
        token_limit_segments: [],
        generated_tokens_by_segment: [],
        resumed_from_asr_checkpoint: false,
        published: false
      },
      provenance.to_h
    )
    assert_equal [], result.outputs
    assert_nil result.error
    assert_equal provenance, result.provenance
    assert result.segments.frozen?
    assert result.words.frozen?
    assert result.cues.frozen?
    assert result.status.frozen?
  end

  def test_statistics_and_progress_event_fields
    values = Types::TranscriptionStatistics.members.to_h { |member| [member, 0] }
    statistics = Types::TranscriptionStatistics.new(**values)
    assert_equal 20, statistics.to_h.length
    assert_includes statistics.to_h, :real_time_factor_x
    refute_includes statistics.to_h, :real_time_factor

    assert_equal(
      { stage: "ASR", message: nil, current: nil, total: nil },
      Types::ProgressEvent.new(stage: "ASR").to_h
    )
    assert_raises(ArgumentError) { Types::TranscriptionStatistics.new(*Array.new(20, 0)) }
  end

  def test_transcription_run_is_immutable_enumerable_indexable_and_classified
    completed = result("done.wav", "completed")
    failed = result("failed.wav", "failed", error: "decode failed")
    skipped = result("skipped.wav", "skipped", text: nil)
    run = run_with(completed, failed, skipped, errors: ["profile failed"])

    assert_equal 3, run.length
    assert_equal [completed, failed, skipped], run.to_a
    assert_same completed, run[0]
    assert_equal [failed, skipped], run[1..]
    assert run[1..].frozen?
    assert_equal [completed], run.successful
    assert_equal [failed], run.failed
    assert_equal [skipped], run.skipped
    assert_equal Types::TranscriptionRun.members, run.to_h.keys
    assert_equal [completed, failed, skipped], run.to_h.fetch(:results)
    assert_equal ["profile failed"], run.to_h.fetch(:errors)
    refute run.ok?
    refute run.ok
    assert run.results.frozen?
    assert run.errors.frozen?
  end

  def test_single_requires_exactly_one_result
    only = result("only.wav", "completed")
    assert_same only, run_with(only).single

    error = assert_raises(ArgumentError) { run_with.single }
    assert_match(/found 0/, error.message)
    error = assert_raises(ArgumentError) { run_with(only, result("two.wav", "completed")).single }
    assert_match(/found 2/, error.message)
  end

  def test_errors_retain_original_and_partial_run
    assert_operator Types::TranscriptionConfigurationError, :<, Types::TranscriptionError
    assert_operator Types::TranscriptionInputError, :<, Types::TranscriptionError
    assert_operator Types::TranscriptionRuntimeError, :<, Types::TranscriptionError
    assert_operator Types::TranscriberClosedError, :<, Types::TranscriptionError
    assert_operator Types::TranscriberBusyError, :<, Types::TranscriptionError

    original = RuntimeError.new("callback broke")
    callback_error = Types::ProgressCallbackError.new(original)
    assert_same original, callback_error.original
    assert_equal "Progress callback failed: RuntimeError: callback broke", callback_error.message

    partial = run_with(result("bad.wav", "failed", error: "bad input"))
    batch_error = Types::BatchTranscriptionError.new(partial)
    assert_same partial, batch_error.run
    assert_equal "1 transcription file(s) failed", batch_error.message

    run_error = Types::BatchTranscriptionError.new(run_with(errors: ["profile failed"]))
    assert_equal "Transcription run failed with 1 run error(s)", run_error.message
  end

  def test_mutable_inputs_are_detached_and_frozen
    formats = [+"txt"]
    options = Types::PublicationOptions.new(formats: formats)
    formats.first << "-changed"
    formats << "json"
    assert_equal ["txt"], options.formats

    source_text = +"transcript"
    transcript = result("clip.wav", "completed", text: source_text)
    source_text << " changed"
    assert_equal "transcript", transcript.text
    assert transcript.text.frozen?
  end

  def test_every_nested_collection_and_string_is_detached
    model = +"owner/model"
    revision = +"main"
    adapter = +"owner/adapter"
    formats = [+"txt"]
    output_dir = +"output"
    publication = Types::PublicationOptions.new(formats: formats, output_dir: output_dir)
    options = Types::TranscriptionOptions.new(
      model: model,
      model_revision: revision,
      adapter: adapter,
      publication: publication
    )
    generated = [[0, 12]]
    provenance = Types::TranscriptionProvenance.new(generated_tokens_by_segment: generated)

    model << "-changed"
    revision << "-changed"
    adapter << "-changed"
    formats.first << "-changed"
    output_dir << "-changed"
    generated.first << 99
    generated << [1, 4]

    assert_equal "owner/model", options.model
    assert_equal "main", options.model_revision
    assert_equal "owner/adapter", options.adapter
    assert_equal ["txt"], options.publication.formats
    assert_equal Pathname("output"), options.publication.output_dir
    assert_equal [[0, 12]], provenance.generated_tokens_by_segment
    assert provenance.generated_tokens_by_segment.first.frozen?
    assert_raises(FrozenError) { provenance.generated_tokens_by_segment.first << 7 }
  end

  private

  def statistics
    @statistics ||= begin
      values = Types::TranscriptionStatistics.members.to_h { |member| [member, 0] }
      Types::TranscriptionStatistics.new(**values)
    end
  end

  def result(path, status, text: "text", error: nil)
    Types::TranscriptionResult.new(
      path: Pathname(path),
      relative_path: Pathname(path),
      status: status,
      text: text,
      duration: 1.0,
      error: error
    )
  end

  def run_with(*results, errors: [])
    options = Types::TranscriptionOptions.new
    Types::TranscriptionRun.new(
      results: results,
      requested_options: options,
      resolved_options: options,
      statistics: statistics,
      errors: errors
    )
  end
end
