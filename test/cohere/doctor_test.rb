# frozen_string_literal: true

require "open3"
require "json"
require "stringio"
require "test_helper"
require "cohere/transcribe/doctor"

class Cohere::Transcribe::DoctorTest < Minitest::Test
  Doctor = Cohere::Transcribe::Doctor

  DOCTOR_OPTIONS = %w[
    --adapter --adapter-revision --audio-backend --help --mode --model-access
    --model --model-revision
  ].freeze

  def test_help_is_complete_and_dependency_light
    script = <<~RUBY
      require "cohere/transcribe/doctor"
      heavy = %w[onnxruntime numo/narray]
      abort "runtime loaded before help" if heavy.any? { |name| $LOADED_FEATURES.any? { |path| path.include?(name) } }
      status = Cohere::Transcribe::Doctor.main(["--help"])
      abort "runtime loaded by help" if heavy.any? { |name| $LOADED_FEATURES.any? { |path| path.include?(name) } }
      exit status
    RUBY
    output, error, status = Open3.capture3("ruby", "-Ilib", "-e", script, chdir: project_root)

    assert_predicate status, :success?, error
    assert_equal DOCTOR_OPTIONS.sort, output.scan(/--[a-z](?:[a-z-]*[a-z])?/).uniq.sort
  end

  def test_defaults_and_every_choice
    defaults = Doctor.parse_args([])
    assert_equal "segment", defaults.mode
    refute defaults.model_access
    assert_equal "CohereLabs/cohere-transcribe-arabic-07-2026", defaults.model
    assert_nil defaults.model_revision
    assert_nil defaults.adapter
    assert_nil defaults.adapter_revision
    assert_equal "auto", defaults.audio_backend

    %w[segment word].each do |mode|
      %w[auto torchcodec ffmpeg librosa].each do |backend|
        options = Doctor.parse_args(["--mode", mode, "--audio-backend", backend, "--model-access"])
        assert_equal [mode, backend, true], [options.mode, options.audio_backend, options.model_access]
      end
    end
  end

  def test_rejects_unknown_choices_values_and_positionals
    assert_raises(OptionParser::InvalidArgument) { Doctor.parse_args(%w[--mode invalid]) }
    assert_raises(OptionParser::NeedlessArgument) { Doctor.parse_args(["--model-access=true"]) }
    assert_raises(OptionParser::InvalidOption) { Doctor.parse_args(["--version"]) }
    assert_raises(OptionParser::InvalidArgument) { Doctor.parse_args(["unexpected"]) }

    err = StringIO.new
    assert_equal 2, Doctor.main(["--not-an-option"], out: StringIO.new, err: err)
    assert_includes err.string, "cohere-transcribe-doctor: error:"

    err = StringIO.new
    assert_equal 2, Doctor.main(["--version"], out: StringIO.new, err: err)
    assert_includes err.string, "invalid option: --version"
  end

  def test_long_option_abbreviations_match_argparse_ambiguity
    %w[--m --mo --mod --model- --a --ad --ada --adap --adapte].each do |option|
      assert_raises(OptionParser::AmbiguousOption, option) do
        Doctor.parse_args([option, "value"])
      end
      assert_raises(OptionParser::AmbiguousOption, "#{option}=value") do
        Doctor.parse_args(["#{option}=value"])
      end
    end

    assert_equal "word", Doctor.parse_args(%w[--mode=word]).mode
    assert_equal "release", Doctor.parse_args(%w[--model-r=release]).model_revision
    assert Doctor.parse_args(%w[--model-a]).model_access
    assert_equal "cpu/model", Doctor.parse_args(%w[--model=cpu/model]).model
  end

  def test_main_routes_checks_and_selected_model_implies_access
    calls = []
    checks = fake_checks(calls)
    out = StringIO.new
    status = Doctor.main(
      %w[
        --mode word --audio-backend librosa --model owner/model
        --model-revision release --adapter owner/adapter --adapter-revision adapter-release
      ],
      out: out,
      err: StringIO.new,
      checks: checks
    )

    assert_equal 0, status
    assert_equal :files, calls[0]
    assert_equal :runtime, calls[1]
    assert_equal :silero, calls[2]
    assert_equal :word, calls[3]
    assert_equal [:backend, "librosa"], calls[4]
    assert_equal(
      [
        :access,
        {
          include_aligner: true,
          model_id: "owner/model",
          model_revision: "release",
          adapter_id: "owner/adapter",
          adapter_revision: "adapter-release"
        }
      ],
      calls[5]
    )
    assert_includes out.string, "Validation passed for word mode"
  end

  def test_failures_produce_exit_one
    checks = fake_checks([]) { |results| results.fail("runtime unavailable") }
    out = StringIO.new
    assert_equal 1, Doctor.main([], out: out, err: StringIO.new, checks: checks)
    assert_includes out.string, "Validation failed: 1 failure(s)"
  end

  def test_packaged_asset_and_silero_smoke_checks_pass
    results = Doctor::Results.new(out: StringIO.new)
    Doctor.validate_files(results)
    Doctor.validate_silero(results)
    assert_equal 0, results.failures
  end

  def test_native_device_report_uses_actual_cpu_and_cuda_capabilities
    cpu = native_library_fixture(auto: "cpu", available: %w[cpu], bf16: false)
    out = StringIO.new
    results = Doctor::Results.new(out: out)
    Doctor.report_native_device_capabilities(results, cpu)
    assert_equal 0, results.failures
    assert_equal 1, results.warnings
    assert_includes out.string, "native inference devices: cpu; auto resolves to cpu"
    assert_includes out.string, "accelerator: CPU only"

    cuda = native_library_fixture(auto: "cuda", available: %w[cpu cuda], bf16: true)
    out = StringIO.new
    results = Doctor::Results.new(out: out)
    Doctor.report_native_device_capabilities(results, cuda)
    assert_equal 0, results.failures
    assert_equal 0, results.warnings
    assert_includes out.string, "native inference devices: cpu, cuda; auto resolves to cuda"
    assert_includes out.string, "accelerator: CUDA available through native runtime"
    assert_includes out.string, "BF16 operations supported"
  end

  def test_librosa_compatibility_reports_the_preferred_native_ffmpeg_route
    require "cohere/transcribe/audio/decoder"
    out = StringIO.new
    results = Doctor::Results.new(out: out)
    replacements = {
      available?: -> { true },
      diagnostic: -> { "FFmpeg ABI fixture" }
    }

    replace_singleton_methods(Cohere::Transcribe::Audio::FFmpegNative, replacements) do
      Doctor.report_optional_runtime(results, "librosa")
    end

    assert_equal 0, results.failures
    assert_includes out.string, "native FFmpeg decoder: FFmpeg ABI fixture"
    assert_includes out.string, "librosa compatibility mode uses FFmpeg through the native C ABI"
  end

  def test_word_alignment_smokes_use_pinned_mms_ctc_without_loading_weights
    out = StringIO.new
    results = Doctor::Results.new(out: out)

    Doctor.validate_word_alignment(results)

    assert_equal 0, results.failures
    assert_includes out.string, "pure-Ruby MMS CTC Viterbi"
    assert_includes out.string, "onnx-community/mms-300m-1130-forced-aligner-ONNX@"
    assert_includes out.string, "429e5d05c62acc8a9264db874a1b131e359fc626e40c253ac7b1fe52b11149b4"
    refute_includes out.string, "cross-attention"
  end

  def test_model_access_validates_dense_weights_and_tokenizer_metadata
    Dir.mktmpdir do |directory|
      model = Pathname(directory)
      model.join("config.json").write(
        JSON.generate(
          "model_type" => "cohere_asr",
          "architectures" => ["CohereAsrForConditionalGeneration"]
        )
      )
      model.join("model.safetensors").binwrite("weights")
      write_processor_metadata(model)
      results = Doctor::Results.new(out: StringIO.new)

      Doctor.validate_model_access(
        results, include_aligner: false, model_id: model.to_s
      )
      assert_equal 1, results.failures

      write_tokenizer_metadata(model)
      out = StringIO.new
      results = Doctor::Results.new(out: out)
      Doctor.validate_model_access(
        results, include_aligner: false, model_id: model.to_s
      )
      assert_equal 0, results.failures
      assert_includes out.string, "CohereAsrProcessor one-row limit is 35.0s"
    end
  end

  def test_model_access_rejects_incompatible_processor_and_tokenizer_metadata
    Dir.mktmpdir do |directory|
      model = Pathname(directory)
      model.join("config.json").write(
        JSON.generate(
          "model_type" => "cohere_asr",
          "architectures" => ["CohereAsrForConditionalGeneration"]
        )
      )
      model.join("model.safetensors").binwrite("weights")
      write_tokenizer_metadata(model)

      write_processor_metadata(model, processor_class: "OtherProcessor")
      results = Doctor::Results.new(out: StringIO.new)
      Doctor.validate_model_access(results, include_aligner: false, model_id: model.to_s)
      assert_equal 1, results.failures

      write_processor_metadata(model, max_audio_clip_s: nil)
      results = Doctor::Results.new(out: StringIO.new)
      Doctor.validate_model_access(results, include_aligner: false, model_id: model.to_s)
      assert_equal 1, results.failures

      write_processor_metadata(model)
      model.join("tokenizer.json").write(
        JSON.generate("model" => { "vocab" => { "<unk>" => 0 } })
      )
      results = Doctor::Results.new(out: StringIO.new)
      Doctor.validate_model_access(results, include_aligner: false, model_id: model.to_s)
      assert_equal 1, results.failures
    end
  end

  def test_packaged_default_model_access_rejects_a_non_cohere_config
    Dir.mktmpdir do |directory|
      snapshot = Pathname(directory)
      snapshot.join("config.json").write(JSON.generate("model_type" => "other"))
      write_processor_metadata(snapshot)
      write_tokenizer_metadata(snapshot)
      hub = Object.new
      hub.define_singleton_method(:resolve_revision) do |_repository, revision, filename:|
        raise "wrong metadata" unless filename == "config.json"

        revision.downcase
      end
      hub.define_singleton_method(:download) do |_repository, filename, revision:|
        raise "wrong revision" unless revision == Cohere::Transcribe::DEFAULT_ASR_MODEL_REVISION

        snapshot.join(filename)
      end
      out = StringIO.new
      results = Doctor::Results.new(out: out)

      Doctor.validate_model_access(
        results,
        include_aligner: false,
        model_id: Cohere::Transcribe::DEFAULT_ASR_MODEL_ID,
        hub: hub
      )

      assert_equal 1, results.failures
      assert_includes out.string, "config.json declares model_type=\"other\"; expected cohere_asr"
    end
  end

  def test_model_access_validates_pinned_aligner_stride_source_and_vocabulary
    Dir.mktmpdir do |directory|
      model = Pathname(directory).join("asr")
      model.mkpath
      model.join("config.json").write(
        JSON.generate(
          "model_type" => "cohere_asr",
          "architectures" => ["CohereAsrForConditionalGeneration"]
        )
      )
      model.join("model.safetensors").binwrite("weights")
      write_processor_metadata(model)
      write_tokenizer_metadata(model)

      aligner = Pathname(directory).join("aligner")
      aligner.mkpath
      aligner.join("config.json").write(
        JSON.generate(
          "_name_or_path" => "MahmoudAshraf/mms-300m-1130-forced-aligner",
          "conv_stride" => [5, 2, 2, 2, 2, 2, 2],
          "vocab_size" => 31
        )
      )
      aligner.join("vocab.json").write(
        JSON.generate(Cohere::Transcribe::Alignment::Aligner::VOCABULARY)
      )
      downloads = []
      hub = Object.new
      hub.define_singleton_method(:download) do |repository, filename, revision:|
        downloads << [repository, filename, revision]
        aligner.join(filename)
      end
      out = StringIO.new
      results = Doctor::Results.new(out: out)

      Doctor.validate_model_access(
        results,
        include_aligner: true,
        model_id: model.to_s,
        hub: hub
      )

      assert_equal 0, results.failures
      assert_equal(
        [
          [
            "onnx-community/mms-300m-1130-forced-aligner-ONNX",
            "config.json",
            "2100fb247d8e43962eef24491597fbeb8b469531"
          ],
          [
            "onnx-community/mms-300m-1130-forced-aligner-ONNX",
            "vocab.json",
            "2100fb247d8e43962eef24491597fbeb8b469531"
          ]
        ],
        downloads
      )
      assert_includes out.string, "pinned MMS ONNX aligner metadata"

      aligner.join("vocab.json").write(JSON.generate("<blank>" => 1))
      invalid = Doctor::Results.new(out: StringIO.new)
      Doctor.validate_model_access(
        invalid,
        include_aligner: true,
        model_id: model.to_s,
        hub: hub
      )
      assert_equal 1, invalid.failures
    end
  end

  def test_executable_help
    output, error, status = Open3.capture3(File.join(project_root, "exe/cohere-transcribe-doctor"), "--help")
    assert_predicate status, :success?, error
    assert_includes output, "Usage: cohere-transcribe-doctor"
  end

  private

  def project_root
    File.expand_path("../..", __dir__)
  end

  def write_processor_metadata(model, processor_class: "CohereAsrProcessor", **values)
    model.join("preprocessor_config.json").write(
      JSON.generate({ "processor_class" => processor_class }.merge(values.transform_keys(&:to_s)))
    )
  end

  def write_tokenizer_metadata(model)
    tokens = Cohere::Transcribe::Doctor::REQUIRED_PROMPT_TOKENS
    vocabulary = tokens.each_with_index.to_h
    model.join("tokenizer.json").write(
      JSON.generate("model" => { "vocab" => vocabulary }, "added_tokens" => [])
    )
  end

  def native_library_fixture(auto:, available:, bf16:)
    Object.new.tap do |library|
      library.define_singleton_method(:resolve_device) do |requested|
        return auto if requested == "auto"
        return requested if available.include?(requested)

        raise Cohere::Transcribe::TranscriptionRuntimeError, "#{requested} is unavailable"
      end
      library.define_singleton_method(:supports_bf16?) { |_device| bf16 }
    end
  end

  def fake_checks(calls, &runtime_check)
    Object.new.tap do |checks|
      checks.define_singleton_method(:validate_files) { |_results| calls << :files }
      checks.define_singleton_method(:validate_common_runtime) do |results|
        calls << :runtime
        runtime_check&.call(results)
      end
      checks.define_singleton_method(:validate_silero) { |_results| calls << :silero }
      checks.define_singleton_method(:validate_word_alignment) { |_results| calls << :word }
      checks.define_singleton_method(:report_optional_runtime) do |_results, backend|
        calls << [:backend, backend]
      end
      checks.define_singleton_method(:validate_model_access) do |_results, **keywords|
        calls << [:access, keywords]
      end
    end
  end

  def replace_singleton_methods(object, replacements)
    originals = replacements.to_h { |name, _replacement| [name, object.method(name)] }
    replacements.each { |name, replacement| object.singleton_class.define_method(name, replacement) }
    yield
  ensure
    originals&.each do |name, original|
      object.singleton_class.define_method(name) do |*arguments, **keywords, &block|
        original.call(*arguments, **keywords, &block)
      end
    end
  end
end
