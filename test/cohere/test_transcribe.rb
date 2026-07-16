# frozen_string_literal: true

require "test_helper"
require "open3"
require "rbconfig"

class Cohere::TestTranscribe < Minitest::Test
  class FakeEngine
    attr_reader :audio, :closed

    def initialize(options)
      @options = options
      @closed = false
    end

    def transcribe(audio, **)
      @audio = audio
      statistics = Cohere::Transcribe::TranscriptionStatistics.new(
        elapsed_seconds: 0.0,
        successful_audio_seconds: 0.0,
        real_time_factor_x: 0.0,
        runtime_import_seconds: 0.0,
        serialization_wait_seconds: 0.0,
        input_validation_seconds: 0.0,
        decode_seconds: 0.0,
        vad_seconds: 0.0,
        asr_load_seconds: 0.0,
        asr_seconds: 0.0,
        aligner_load_seconds: 0.0,
        emissions_seconds: 0.0,
        viterbi_seconds: 0.0,
        peak_cuda_allocated_gib: 0.0,
        peak_cuda_reserved_gib: 0.0,
        asr_batches: 0,
        asr_processor_rows: 0,
        generated_tokens: 0,
        oom_retries: 0,
        truncation_retries: 0
      )
      Cohere::Transcribe::TranscriptionRun.new(
        results: [],
        requested_options: @options,
        resolved_options: @options,
        statistics: statistics
      )
    end

    def close
      @closed = true
    end
  end

  def test_that_it_has_a_version_number
    refute_nil ::Cohere::Transcribe::VERSION
    assert_equal "0.1.2", ::Cohere::Transcribe::VERSION
  end

  def test_public_transcriber_is_lazy_reusable_and_closable
    options = Cohere::Transcribe::TranscriptionOptions.new
    engine = FakeEngine.new(options)
    creations = 0
    transcriber = Cohere::Transcribe::Transcriber.new(
      options,
      engine_factory: lambda {
        creations += 1
        engine
      }
    )

    assert_same options, transcriber.options
    assert_equal 0, creations
    assert transcriber.transcribe("one.wav").ok?
    assert transcriber.transcribe(["two.wav"]).ok?
    assert_equal 1, creations
    assert_equal ["two.wav"], engine.audio

    transcriber.close
    transcriber.close
    assert engine.closed
    assert transcriber.closed?
    assert_raises(Cohere::Transcribe::TranscriberClosedError) { transcriber.transcribe("three.wav") }
  end

  def test_constructor_validates_options_and_progress_without_loading_runtime
    options = Cohere::Transcribe::TranscriptionOptions.new

    assert_same options, Cohere::Transcribe::Transcriber.new(options: options).options
    assert_raises(TypeError) { Cohere::Transcribe::Transcriber.new(Object.new) }
    assert_raises(TypeError) { Cohere::Transcribe::Transcriber.new(false) }
    assert_raises(TypeError) { Cohere::Transcribe::Transcriber.new(progress: Object.new) }
    assert_raises(TypeError) { Cohere::Transcribe::Transcriber.new(progress: false) }
    assert_raises(ArgumentError) do
      Cohere::Transcribe::Transcriber.new(options, options: options)
    end
  end

  def test_root_require_is_dependency_light_and_exports_the_documented_api
    script = <<~'RUBY'
      require "cohere/transcribe"

      forbidden = %w[
        numo/narray onnxruntime fiddle runtime/engine audio/decoder
        alignment/aligner asr/native dense_converter safetensors
      ]
      loaded = $LOADED_FEATURES.select do |feature|
        forbidden.any? { |fragment| feature.include?(fragment) }
      end
      abort "heavy features loaded: #{loaded.inspect}" unless loaded.empty?

      package = Cohere::Transcribe
      missing = package::PUBLIC_API.reject do |name|
        name == "transcribe" ? package.respond_to?(:transcribe) : package.const_defined?(name, false)
      end
      abort "missing public API: #{missing.inspect}" unless missing.empty?
    RUBY

    stdout, stderr, status = Open3.capture3(
      RbConfig.ruby,
      "--disable-gems",
      "-Ilib",
      "-e",
      script,
      chdir: File.expand_path("../..", __dir__)
    )
    assert status.success?, "#{stdout}\n#{stderr}"
  end

  def test_constructor_and_injected_engine_do_not_materialize_the_runtime
    script = <<~'RUBY'
      require "cohere/transcribe"

      options = Cohere::Transcribe::TranscriptionOptions.new
      statistics = Cohere::Transcribe::TranscriptionStatistics.new(
        **Cohere::Transcribe::TranscriptionStatistics.members.to_h do |member|
          numeric = member.to_s.end_with?("seconds", "_x", "_gib") ? 0.0 : 0
          [member, numeric]
        end
      )
      run = Cohere::Transcribe::TranscriptionRun.new(
        results: [],
        requested_options: options,
        resolved_options: options,
        statistics: statistics
      )
      engine = Object.new
      engine.define_singleton_method(:transcribe) { |_audio, **| run }
      engine.define_singleton_method(:close) { nil }

      session = Cohere::Transcribe::Transcriber.new(options, engine_factory: -> { engine })
      session.transcribe("unused.wav")
      session.close

      forbidden = %w[runtime/engine audio/decoder asr/native dense_converter numo/narray onnxruntime fiddle]
      loaded = $LOADED_FEATURES.select do |feature|
        forbidden.any? { |fragment| feature.include?(fragment) }
      end
      abort "injected facade loaded runtime: #{loaded.inspect}" unless loaded.empty?
    RUBY

    stdout, stderr, status = Open3.capture3(
      RbConfig.ruby,
      "-Ilib",
      "-e",
      script,
      chdir: File.expand_path("../..", __dir__)
    )
    assert status.success?, "#{stdout}\n#{stderr}"
  end

  def test_runtime_session_construction_defers_model_numo_onnx_and_native_code
    script = <<~'RUBY'
      require "cohere/transcribe"

      engine = Cohere::Transcribe::Runtime::Engine.new(
        Cohere::Transcribe::TranscriptionOptions.new
      )
      engine.close

      forbidden = %w[
        numo/narray onnxruntime alignment/aligner asr/native dense_converter
        safetensors pytorch_checkpoint gguf_writer
      ]
      loaded = $LOADED_FEATURES.select do |feature|
        forbidden.any? { |fragment| feature.include?(fragment) }
      end
      abort "model/runtime dependencies loaded early: #{loaded.inspect}" unless loaded.empty?
    RUBY

    stdout, stderr, status = Open3.capture3(
      RbConfig.ruby,
      "-Ilib",
      "-e",
      script,
      chdir: File.expand_path("../..", __dir__)
    )
    assert status.success?, "#{stdout}\n#{stderr}"
  end

  def test_lazy_default_runtime_load_failure_is_typed_and_retains_its_cause
    script = <<~'RUBY'
      require "cohere/transcribe"

      loader = Cohere::Transcribe.const_get(:Loader, false)
      loader.define_singleton_method(:load_runtime!) do
        raise LoadError, "cannot load such file -- optional_runtime"
      end
      begin
        Cohere::Transcribe.transcribe("unused.wav")
      rescue Cohere::Transcribe::TranscriptionRuntimeError => error
        abort "wrong message: #{error.message}" unless error.message.match?(/LoadError.*optional_runtime/)
        abort "cause was lost" unless error.cause.is_a?(LoadError)
      else
        abort "lazy runtime failure escaped its typed boundary"
      end
    RUBY

    stdout, stderr, status = Open3.capture3(
      RbConfig.ruby,
      "-Ilib",
      "-e",
      script,
      chdir: File.expand_path("../..", __dir__)
    )
    assert status.success?, "#{stdout}\n#{stderr}"
  end
end
