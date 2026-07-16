# frozen_string_literal: true

require "open3"
require "stringio"
require "tmpdir"
require "test_helper"
require "cohere/transcribe/cli"

class Cohere::Transcribe::CLITest < Minitest::Test
  CLI = Cohere::Transcribe::CLI

  MAIN_OPTIONS = %w[
    --adapter --adapter-revision --adaptive-batch --align-batch-size --align-dtype
    --alignment --audio-backend --audio-memory-gb --batch-audio-seconds
    --batch-max-size --batch-size --batch-vram-target --device --dtype
    --energy-threshold --existing --formats --help --language --max-chars
    --max-cue-dur --max-dur --max-gap --max-new-tokens --max-retry-tokens
    --max-silence --min-dur --min-silence-ms --model --model-revision
    --no-adaptive-batch --no-pin-memory --no-pipeline-preparation --no-recursive
    --no-stop-repetition-loops --no-vad-merge --output-dir --pin-memory
    --pipeline-preparation --preprocess-workers --profile-json --recursive
    --speech-pad-ms --stop-repetition-loops --text-only --truncation-policy
    --vad --vad-batch-size --vad-block-frames --vad-engine --vad-merge
    --vad-threads --vad-threshold --version
  ].freeze

  class IntegrationDecoder
    def initialize(failure: nil)
      @failure = failure
    end

    def decode(_path, **)
      raise @failure if @failure

      Cohere::Transcribe::Audio::Decoded.new(
        samples: Array.new(16_000, 0.1),
        sample_rate: 16_000,
        backend: "integration",
        fallback_reason: nil
      )
    end
  end

  class IntegrationSession
    def transcribe(samples, language:, offset:, max_new_tokens:)
      Cohere::Transcribe::ASR::NativeResult.new(
        text: "verified transcript",
        segments: [],
        words: [],
        generated_tokens: 2,
        generation_limit: max_new_tokens,
        generation_capacity: [samples.length, 1].max,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end

    def close; end
  end

  class IntegrationProvider
    attr_reader :open_count

    def initialize
      @open_count = 0
    end

    def resolve(_options)
      Cohere::Transcribe::ResolvedModelIdentity.new(
        model_id: "CohereLabs/cohere-transcribe-arabic-07-2026",
        model_revision: "a" * 40,
        model_format: :dense,
        quantization_config: nil,
        adapter_id: nil,
        adapter_revision: nil
      )
    end

    def open(_identity, _options)
      @open_count += 1
      IntegrationSession.new
    end
  end

  def test_help_is_complete_and_does_not_load_the_public_data_model
    script = <<~RUBY
      require "cohere/transcribe/cli"
      abort "types loaded before help" if $LOADED_FEATURES.any? { |path| path.end_with?("cohere/transcribe/types.rb") }
      status = Cohere::Transcribe::CLI.main(["--help"])
      abort "types loaded by help" if $LOADED_FEATURES.any? { |path| path.end_with?("cohere/transcribe/types.rb") }
      exit status
    RUBY
    output, error, status = Open3.capture3("ruby", "-Ilib", "-e", script, chdir: project_root)

    assert_predicate status, :success?, error
    assert_includes output, "audio [audio ...]"
    assert_equal MAIN_OPTIONS.sort, output.scan(/--[a-z](?:[a-z-]*[a-z])?/).uniq.sort
  end

  def test_version_succeeds_without_audio
    out = StringIO.new
    assert_equal 0, CLI.main(["--version"], out: out, err: StringIO.new)
    assert_equal "cohere-transcribe #{Cohere::Transcribe::VERSION}\n", out.string
  end

  def test_default_command_matches_the_python_contract
    command = CLI.parse_args(["audio.wav"])
    options = command.options

    assert_equal ["audio.wav"], command.audio
    assert_equal "CohereLabs/cohere-transcribe-arabic-07-2026", options.model
    assert_equal "ar", options.language
    assert options.recursive
    assert_equal "auto", options.device
    assert_equal "auto", options.dtype
    assert_equal "auto", options.audio_backend
    assert_equal 4.0, options.audio_memory_gb
    assert options.pipeline_preparation
    assert_equal "silero", options.vad
    assert_equal "auto", options.vad_engine
    assert_equal 16, options.vad_batch_size
    assert_equal 512, options.vad_block_frames
    refute options.vad_merge
    assert_equal 30.0, options.max_dur
    assert_equal "segment", options.alignment
    assert_equal 445, options.max_new_tokens
    assert_equal 896, options.max_retry_tokens
    assert_equal %w[txt srt vtt], options.publication.formats
    assert_equal "error", options.publication.existing
  end

  def test_all_value_and_boolean_option_families_build_public_options
    command = CLI.parse_args(
      %w[
        first.wav second.flac
        --model owner/cohere-finetune --model-revision release
        --adapter owner/adapter --adapter-revision adapter-release
        --language en --formats json txt json --output-dir transcripts
        --no-recursive --existing overwrite --device cpu --dtype fp32
        --audio-backend ffmpeg --audio-memory-gb 2.5 --preprocess-workers 3
        --no-pipeline-preparation --vad auditok --vad-engine jit
        --vad-batch-size -1 --vad-block-frames -1 --no-vad-merge
        --min-dur 0.25 --max-dur 20.5 --max-silence 0.3
        --energy-threshold 42.5 --vad-threshold 0.6 --min-silence-ms 200
        --speech-pad-ms 40 --batch-size 8 --adaptive-batch --batch-max-size 16
        --batch-audio-seconds 160.5 --batch-vram-target 0.85 --pin-memory
        --max-new-tokens 400 --max-retry-tokens 800 --truncation-policy warn
        --no-stop-repetition-loops --alignment word --align-batch-size 3
        --align-dtype fp32 --max-chars 70 --max-cue-dur 5.5 --max-gap 0.4
        --profile-json profiles/run.json
      ]
    )
    options = command.options

    assert_equal %w[first.wav second.flac], command.audio
    assert_equal "owner/cohere-finetune", options.model
    assert_equal "release", options.model_revision
    assert_equal "owner/adapter", options.adapter
    assert_equal "adapter-release", options.adapter_revision
    assert_equal "en", options.language
    refute options.recursive
    assert_equal "cpu", options.device
    assert_equal "fp32", options.dtype
    assert_equal "ffmpeg", options.audio_backend
    assert_equal 2.5, options.audio_memory_gb
    assert_equal 3, options.preprocess_workers
    refute options.pipeline_preparation
    assert_equal "auditok", options.vad
    assert_equal "jit", options.vad_engine
    assert_equal(-1, options.vad_batch_size)
    assert_equal(-1, options.vad_block_frames)
    assert_equal 0.25, options.min_dur
    assert_equal 20.5, options.max_dur
    assert_equal 8, options.batch_size
    assert_equal 16, options.batch_max_size
    assert options.adaptive_batch
    assert options.pin_memory
    refute options.stop_repetition_loops
    assert_equal "word", options.alignment
    assert_equal 3, options.align_batch_size
    assert_equal %w[json txt], options.publication.formats
    assert_equal Pathname("transcripts"), options.publication.output_dir
    assert_equal Pathname("profiles/run.json"), options.publication.profile_json
  end

  def test_boolean_options_accept_both_forms_and_last_form_wins
    command = CLI.parse_args(
      %w[
        audio.wav --no-recursive --recursive --pipeline-preparation
        --no-pipeline-preparation --vad-merge --no-vad-merge
        --adaptive-batch --no-adaptive-batch --pin-memory --no-pin-memory
        --no-stop-repetition-loops --stop-repetition-loops
      ]
    )
    options = command.options
    assert options.recursive
    refute options.pipeline_preparation
    refute options.vad_merge
    refute options.adaptive_batch
    refute options.pin_memory
    assert options.stop_repetition_loops
  end

  def test_enum_values_require_an_exact_choice
    {
      "--language" => "e",
      "--existing" => "o",
      "--device" => "c",
      "--dtype" => "fp",
      "--audio-backend" => "ff",
      "--vad" => "n",
      "--vad-engine" => "on",
      "--truncation-policy" => "r",
      "--alignment" => "w",
      "--align-dtype" => "fp"
    }.each do |option, value|
      error = assert_raises(OptionParser::InvalidArgument, option) do
        CLI.parse_args(["audio.wav", option, value])
      end
      assert_includes error.message, "invalid choice"
    end
  end

  def test_formats_use_reference_whitespace_separation_and_reject_commas
    command = CLI.parse_args(%w[audio.wav --formats json txt json])
    assert_equal %w[json txt], command.options.publication.formats

    [
      %w[audio.wav --formats txt,srt],
      %w[audio.wav --formats=txt,srt]
    ].each do |arguments|
      error = assert_raises(OptionParser::InvalidArgument) { CLI.parse_args(arguments) }
      assert_match(/invalid argument|unsupported output format/, error.message)
    end

    separator_collision = "txt\u001Fjson"
    [
      ["audio.wav", "--formats", separator_collision],
      ["audio.wav", "--formats=#{separator_collision}"]
    ].each do |arguments|
      error = assert_raises(OptionParser::InvalidArgument) { CLI.parse_args(arguments) }
      assert_match(/unsupported output format/, error.message)
    end
  end

  def test_long_option_abbreviations_match_argparse_ambiguity_and_formats_greediness
    %w[--mo --mod --mode --adapte --va].each do |option|
      assert_raises(OptionParser::AmbiguousOption, option) do
        CLI.parse_args([option, "value", "audio.wav"])
      end
      assert_raises(OptionParser::AmbiguousOption, "#{option}=value") do
        CLI.parse_args(["#{option}=value", "audio.wav"])
      end
    end

    %w[--f --fo --for --form --forma --format].each do |option|
      assert_raises(OptionParser::InvalidArgument, option) do
        CLI.parse_args([option, "txt", "audio.wav"])
      end
      command = CLI.parse_args([option, "txt", "--", "audio.wav"])
      assert_equal ["audio.wav"], command.audio
      assert_equal ["txt"], command.options.publication.formats
    end

    assert_equal ["txt"], CLI.parse_args(%w[--formats=txt audio.wav]).options.publication.formats
    assert_equal ["txt"], CLI.parse_args(%w[--f=txt audio.wav]).options.publication.formats
  end

  def test_audio_operands_are_one_contiguous_block_like_argparse
    before = CLI.parse_args(%w[--language en first.wav second.wav])
    after = CLI.parse_args(%w[first.wav second.wav --language en])
    barrier = CLI.parse_args(%w[first.wav -- --literal-option.wav])
    assert_equal %w[first.wav second.wav], before.audio
    assert_equal %w[first.wav second.wav], after.audio
    assert_equal %w[first.wav --literal-option.wav], barrier.audio

    [
      %w[first.wav --language en second.wav],
      %w[first.wav --recursive second.wav],
      %w[first.wav --recursive -- --literal-option.wav]
    ].each do |arguments|
      error = assert_raises(OptionParser::InvalidArgument) { CLI.parse_args(arguments) }
      assert_match(/unrecognized arguments/, error.message)
    end
  end

  def test_double_dash_keeps_format_like_audio_operands_literal
    command = CLI.parse_args(%w[-- --formats txt --recursive])
    assert_equal %w[--formats txt --recursive], command.audio
    assert_equal %w[txt srt vtt], command.options.publication.formats

    error = assert_raises(OptionParser::InvalidArgument) do
      CLI.parse_args(%w[--formats txt json -])
    end
    assert_match(/formats/, error.message)
  end

  def test_text_only_uses_txt_and_conflicts_with_explicit_alignment
    command = CLI.parse_args(%w[audio.wav --text-only])
    assert command.options.text_only
    assert_equal "none", command.options.alignment
    assert_equal ["txt"], command.options.publication.formats

    assert_raises(OptionParser::InvalidOption) do
      CLI.parse_args(%w[audio.wav --text-only --alignment none])
    end
    assert_raises(Cohere::Transcribe::TranscriptionConfigurationError) do
      CLI.parse_args(%w[audio.wav --alignment none --formats txt json])
    end
  end

  def test_reference_numeric_token_grammar
    command = CLI.parse_args(
      ["audio.wav", "--batch-size", "١_٢", "--max-silence", "\u00A01.5\u00A0"]
    )
    assert_equal 12, command.options.batch_size
    assert_equal 1.5, command.options.max_silence
    assert_equal 10.0, CLI.parse_args(%w[audio.wav --max-silence 1_0]).options.max_silence

    assert_raises(Cohere::Transcribe::TranscriptionConfigurationError) do
      CLI.parse_args(%w[audio.wav --max-gap nan])
    end
    assert_raises(OptionParser::InvalidArgument) do
      CLI.parse_args(%w[audio.wav --max-silence 0x1])
    end
    assert_raises(OptionParser::MissingArgument) do
      CLI.parse_args(%w[audio.wav --max-silence -1e2])
    end
    assert_equal(-100.0, CLI.parse_args(%w[audio.wav --max-silence=-1e2]).options.max_silence)
  end

  def test_separate_option_values_follow_argparse_missing_argument_rules
    assert_raises(OptionParser::MissingArgument) do
      CLI.parse_args(%w[--profile-json --max-gap 0.5 audio.wav])
    end
    assert_raises(OptionParser::MissingArgument) do
      CLI.parse_args(%w[--profile-json -- audio.wav])
    end

    command = CLI.parse_args(%w[--profile-json - audio.wav])
    assert_equal Pathname("-"), command.options.publication.profile_json

    assert_raises(Cohere::Transcribe::TranscriptionConfigurationError) do
      CLI.parse_args(%w[--min-dur -.5 audio.wav])
    end
  end

  def test_negative_number_shaped_tokens_are_values_for_string_and_path_options
    command = CLI.parse_args(
      %w[
        --model-revision -1 --output-dir -1 --profile-json -1
        audio.wav
      ]
    )

    assert_equal "-1", command.options.model_revision
    assert_equal Pathname("-1"), command.options.publication.output_dir
    assert_equal Pathname("-1"), command.options.publication.profile_json

    assert_raises(Cohere::Transcribe::TranscriptionConfigurationError) do
      CLI.parse_args(%w[--model -1 audio.wav])
    end
  end

  def test_help_defers_unknown_options_but_not_prior_conflicts
    out = StringIO.new
    err = StringIO.new
    assert_equal 0, CLI.main(["--unknown", "--help"], out: out, err: err)
    assert_includes out.string, "Usage: cohere-transcribe"
    assert_empty err.string

    out = StringIO.new
    err = StringIO.new
    assert_equal(
      2,
      CLI.main(
        %w[audio.wav --text-only --alignment none --help],
        out: out,
        err: err
      )
    )
    assert_empty out.string
    assert_includes err.string, "conflicts with --alignment"

    separator_collision = "txt\u001Fjson"
    [["--help", "--formats", "csv"], ["--version", "--formats=#{separator_collision}"]].each do |arguments|
      assert_raises(CLI::EarlyExit) { CLI.parse_args(arguments, out: StringIO.new) }
    end
    [["--formats", "csv", "--help"], ["--formats=#{separator_collision}", "--version"]].each do |arguments|
      assert_raises(OptionParser::InvalidArgument) { CLI.parse_args(arguments, out: StringIO.new) }
    end
  end

  def test_parser_errors_return_two_and_do_not_call_runtime
    called = false
    transcriber = lambda do |*, **|
      called = true
      raise "unexpected call"
    end
    out = StringIO.new
    err = StringIO.new

    assert_equal 2, CLI.main(["--device", "invalid", "audio.wav"], out: out, err: err, transcriber: transcriber)
    refute called
    assert_includes err.string, "cohere-transcribe: error:"

    err = StringIO.new
    assert_equal 2, CLI.main([], out: StringIO.new, err: err, transcriber: transcriber)
    assert_includes err.string, "audio"
  end

  def test_main_calls_public_shape_prints_summary_and_maps_run_status
    observed = nil
    transcriber = lambda do |audio, options:, progress:, raise_on_error:|
      observed = [audio, options, raise_on_error]
      progress.call(Cohere::Transcribe::ProgressEvent.new(stage: "message", message: "    preparing"))
      run_for(result("done.wav", "completed", outputs: [Pathname("done.txt")]))
    end
    out = StringIO.new

    assert_equal 0, CLI.main(["done.wav", "--text-only"], out: out, err: StringIO.new, transcriber: transcriber)
    assert_equal ["done.wav"], observed[0]
    assert_instance_of Cohere::Transcribe::TranscriptionOptions, observed[1]
    refute observed[2]
    assert_includes out.string, "preparing"
    assert_includes out.string, "1/1 files finished"
    assert_includes out.string, "done.txt"

    failed = lambda do |*, **|
      run_for(result("bad.wav", "failed", text: nil, error: "decode failed"))
    end
    out = StringIO.new
    assert_equal 1, CLI.main(["bad.wav"], out: out, err: StringIO.new, transcriber: failed)
    assert_includes out.string, "Failures:"
    assert_includes out.string, "decode failed"
  end

  def test_all_skipped_run_uses_the_reference_early_summary
    skipped = lambda do |*, **|
      run_for(
        result(
          "cached.wav",
          "skipped",
          text: nil,
          outputs: [Pathname("cached.txt")]
        )
      )
    end
    out = StringIO.new

    assert_equal(
      0,
      CLI.main(
        ["cached.wav", "--existing", "skip"],
        out: out,
        err: StringIO.new,
        transcriber: skipped
      )
    )
    assert_includes out.string, "skipping cached.wav: verified output generation is complete"
    assert_includes out.string, "All inputs were skipped; no model was loaded."
    refute_includes out.string, "[2/4]"
    refute_includes out.string, "[3/4]"
    refute_includes out.string, "[4/4]"
    refute_includes out.string, "0/0 files finished"
  end

  def test_verified_engine_skip_reaches_cli_without_preparation_or_stage_two
    Dir.mktmpdir("cohere-cli-skip") do |directory|
      source = File.join(directory, "audio.wav")
      output_dir = File.join(directory, "out")
      File.binwrite(source, "source")
      common = [
        source,
        "--device", "cpu",
        "--dtype", "fp32",
        "--vad", "none",
        "--max-dur", "1",
        "--formats", "txt",
        "--output-dir", output_dir
      ]
      overwrite = CLI.parse_args(common + ["--existing", "overwrite"])
      first = Cohere::Transcribe::Runtime::Engine.new(
        overwrite.options,
        model_provider: IntegrationProvider.new,
        decoder: IntegrationDecoder.new
      )
      assert_equal "completed", first.transcribe(overwrite.audio).single.status
      first.close

      provider = IntegrationProvider.new
      callable = lambda do |audio, options:, progress:, raise_on_error:|
        engine = Cohere::Transcribe::Runtime::Engine.new(
          options,
          progress: progress,
          model_provider: provider,
          decoder: IntegrationDecoder.new(failure: "verified skip decoded audio")
        )
        engine.transcribe(audio, raise_on_error: raise_on_error)
      ensure
        engine&.close
      end
      out = StringIO.new

      status = CLI.main(
        common + ["--existing", "skip"],
        out: out,
        err: StringIO.new,
        transcriber: callable
      )

      assert_equal 0, status
      assert_equal 0, provider.open_count
      assert_includes out.string, "All inputs were skipped; no model was loaded."
      refute_includes out.string, "preparing (workers="
      refute_includes out.string, "[2/4]"
    ensure
      first&.close
    end
  end

  def test_interrupt_maps_to_130
    transcriber = ->(*, **) { raise Interrupt }
    out = StringIO.new
    assert_equal 130, CLI.main(["audio.wav"], out: out, err: StringIO.new, transcriber: transcriber)
    assert_includes out.string, "Interrupted"
  end

  def test_interrupt_during_argument_parsing_maps_to_130_before_configuration_loads
    script = <<~RUBY
      require "stringio"
      require "cohere/transcribe/cli"
      Cohere::Transcribe::CLI.define_singleton_method(:parse_args) { |*, **| raise Interrupt }
      output = StringIO.new
      status = Cohere::Transcribe::CLI.main([], out: output, err: StringIO.new)
      abort "wrong status: \#{status}" unless status == 130
      abort "missing interrupt message" unless output.string.include?("Interrupted")
    RUBY
    _output, error, status = Open3.capture3("ruby", "-Ilib", "-e", script, chdir: project_root)

    assert_predicate status, :success?, error
  end

  def test_sigterm_maps_to_143_and_other_signals_are_reraised
    term = ->(*, **) { raise SignalException.new("TERM") }
    out = StringIO.new
    assert_equal 143, CLI.main(["audio.wav"], out: out, err: StringIO.new, transcriber: term)
    assert_includes out.string, "Termination requested"

    hup = ->(*, **) { raise SignalException.new("HUP") }
    error = assert_raises(SignalException) do
      CLI.main(["audio.wav"], out: StringIO.new, err: StringIO.new, transcriber: hup)
    end
    assert_equal "SIGHUP", error.signm
  end

  def test_executable_help
    output, error, status = Open3.capture3(File.join(project_root, "exe/cohere-transcribe"), "--help")
    assert_predicate status, :success?, error
    assert_includes output, "Usage: cohere-transcribe"
  end

  private

  def project_root
    File.expand_path("../..", __dir__)
  end

  def statistics
    values = Cohere::Transcribe::TranscriptionStatistics.members.to_h { |member| [member, 0] }
    values[:elapsed_seconds] = 1.0
    values[:successful_audio_seconds] = 1.0
    values[:real_time_factor_x] = 1.0
    Cohere::Transcribe::TranscriptionStatistics.new(**values)
  end

  def result(path, status, text: "text", error: nil, outputs: [])
    Cohere::Transcribe::TranscriptionResult.new(
      path: Pathname(path),
      relative_path: Pathname(path),
      status: status,
      text: text,
      duration: 1.0,
      outputs: outputs,
      error: error
    )
  end

  def run_for(*results)
    options = Cohere::Transcribe::TranscriptionOptions.new
    Cohere::Transcribe::TranscriptionRun.new(
      results: results,
      requested_options: options,
      resolved_options: options,
      statistics: statistics
    )
  end
end
