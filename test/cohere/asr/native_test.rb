# frozen_string_literal: true

require "tempfile"
require "weakref"
require "test_helper"
require "cohere/transcribe/types"
require "cohere/transcribe/errors"
require "cohere/transcribe/asr/native"

class Cohere::Transcribe::NativeSessionTest < Minitest::Test
  class FakeLibrary
    attr_reader :close_count, :closed, :language, :opened, :sample_count, :setters

    def initialize(
      batch_capacity: 24,
      batch_failure_kind: nil,
      single_failure_kind: nil,
      failure_message: "native failure",
      segment_text: " hello world ",
      word_texts: [" hello", "world "]
    )
      @buffers = []
      @setters = {}
      @closed = false
      @close_count = 0
      @batch_capacity = batch_capacity
      @batch_failure_kind = batch_failure_kind
      @single_failure_kind = single_failure_kind
      @failure_message = failure_message
      @segment_text = segment_text
      @word_texts = word_texts
    end

    def open_session(model_path:, device:, threads:)
      @opened = { model_path: model_path.to_s, device: device, threads: threads }
      101
    end

    def null_pointer?(pointer)
      pointer.nil? || pointer == 0
    end

    def string(pointer)
      pointer.to_s.force_encoding(Encoding::UTF_8).scrub
    end

    def call(name, *arguments)
      case name
      when :last_error_kind then @active_failure_kind || @batch_failure_kind || 0
      when :last_error_message then pointer(@failure_message)
      when :session_backend
        pointer("cohere")
      when :session_compute_backend
        pointer("CPU")
      when :session_memory
        Fiddle::Pointer.new(arguments[1])[0, 8] = [3_000].pack("Q")
        Fiddle::Pointer.new(arguments[2])[0, 8] = [4_000].pack("Q")
        0
      when :session_batch_capacity then @batch_capacity
      when :session_cancel then 0
      when :session_set_max_new_tokens, :session_set_beam_size, :session_set_repetition_loop_guard
        @setters[name] = arguments.last
        0
      when :session_transcribe_lang
        unless @single_failure_kind.nil?
          @active_failure_kind = @single_failure_kind
          return 0
        end

        @sample_count = arguments[2]
        @language = arguments[3].to_s
        202
      when :session_transcribe_batch_lang
        return 0 unless @batch_failure_kind.nil?

        @batch_size = arguments[3]
        @language = arguments[4].to_s
        303
      when :session_batch_result_count then @batch_size
      when :session_batch_result_at then 202
      when :session_batch_result_free
        @batch_result_freed = true
      when :session_result_n_segments then 1
      when :session_result_segment_text then pointer(@segment_text)
      when :session_result_segment_t0 then 10
      when :session_result_segment_t1 then 125
      when :session_result_n_words then 2
      when :session_result_word_text
        pointer(@word_texts.fetch(arguments[2]))
      when :session_result_word_t0
        arguments[2].zero? ? 10 : 60
      when :session_result_word_t1
        arguments[2].zero? ? 60 : 125
      when :session_result_word_p then 0.75
      when :session_result_generated_tokens then 17
      when :session_result_generation_limit then @setters.fetch(:session_set_max_new_tokens)
      when :session_result_generation_capacity then 900
      when :session_result_stopped_by_max_tokens then 1
      when :session_result_repetition_stopped then 0
      when :session_result_free
        @result_freed = true
      when :session_close
        @close_count += 1
        @closed = true
      else
        raise "unexpected fake call: #{name}"
      end
    end

    def result_freed?
      @result_freed
    end

    def batch_result_freed?
      @batch_result_freed
    end

    private

    def pointer(value)
      @buffers << Fiddle::Pointer["#{value}\0"]
      @buffers.last
    end
  end

  class BatchMetricsLibrary < FakeLibrary
    STATS = [
      1, 21,
      2_000_000, 100_000, 900_000, 10_000,
      20_000, 30_000, 40_000, 500_000, 50_000, 5_000,
      1_200_000, 100_000, 10_000, 1_050_000, 25_000,
      240, 240, 3, 23_040
    ].freeze

    def function?(name)
      name == :session_batch_result_stats_v1
    end

    def call(name, *arguments)
      return super unless name == :session_batch_result_stats_v1

      buffer = Fiddle::Pointer.new(arguments[1])
      capacity = arguments[2]
      buffer[0, [capacity, STATS.length].min * 8] = STATS.first(capacity).pack("q*")
      STATS.length
    end
  end

  class BlockingCancellationLibrary < FakeLibrary
    attr_reader :cancel_calls, :started

    def initialize
      super
      @cancel_calls = 0
      @started = Queue.new
      @release = Queue.new
      @attempt = 0
      @inference_finished = false
      @closed_during_inference = false
    end

    def call(name, *arguments)
      case name
      when :session_transcribe_lang
        @attempt += 1
        if @attempt == 1
          @started << true
          @release.pop
          @active_failure_kind = 5
          @inference_finished = true
          return 0
        end
        @active_failure_kind = nil
      when :session_cancel
        @cancel_calls += 1
        @release << true if @cancel_calls == 1
        return 1
      when :session_close
        @closed_during_inference = !@inference_finished
      end
      super
    end

    def inference_finished?
      @inference_finished
    end

    def closed_during_inference?
      @closed_during_inference
    end
  end

  class KillableWorkerLibrary < FakeLibrary
    attr_reader :started

    def initialize
      super
      @started = Queue.new
      @blocked = Queue.new
    end

    def call(name, *arguments)
      if name == :session_transcribe_lang
        @started << Thread.current
        @blocked.pop
      end
      super
    end
  end

  class InterruptingInitializationLibrary < FakeLibrary
    def call(name, *arguments)
      raise Interrupt, "constructor interrupted after native open" if name == :session_set_beam_size

      super
    end
  end

  def test_candidate_paths_never_search_a_neighboring_development_checkout
    original = ENV.delete("COHERE_TRANSCRIBE_NATIVE_LIBRARY")

    paths = Cohere::Transcribe::ASR::NativeLibrary.candidate_paths
    repository = File.expand_path("../../..", __dir__)
    forbidden = File.expand_path("../research/CrispASR/build/src", repository)

    refute(paths.any? { |path| File.expand_path(path).start_with?(forbidden) })
  ensure
    ENV["COHERE_TRANSCRIBE_NATIVE_LIBRARY"] = original if original
  end

  def test_native_library_open_failure_includes_typed_native_diagnostic
    library = Cohere::Transcribe::ASR::NativeLibrary.allocate
    diagnostic = Fiddle::Pointer["CUDA allocator exhausted\0"]
    library.define_singleton_method(:call) do |name, *_arguments|
      case name
      when :set_gpu_backend then nil
      when :session_open_with_params then 0
      when :last_error_kind then 2
      when :last_error_message then diagnostic
      else raise "unexpected fake call: #{name}"
      end
    end

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      library.open_session(model_path: "/model.gguf", device: "cuda", threads: 3)
    end

    assert_match(/native out_of_memory error \(2\): CUDA allocator exhausted/, error.message)
  end

  def test_materializes_native_segments_and_words_with_absolute_offsets
    with_model_file do |path|
      library = FakeLibrary.new
      options = Cohere::Transcribe::TranscriptionOptions.new(device: "cpu", max_new_tokens: 321)
      session = Cohere::Transcribe::ASR::NativeSession.new(path, options, threads: 3, library: library)

      result = session.transcribe(
        [0.0, 0.25, -0.5],
        language: "ar",
        offset: 4.0,
        max_new_tokens: 640
      )

      assert_equal "hello world", result.text
      assert_equal 1, result.segments.length
      assert_in_delta 4.1, result.segments.first.start
      assert_in_delta 5.25, result.segments.first.end
      assert_equal %w[hello world], result.words.map(&:text)
      assert_in_delta 4.1, result.words.first.start
      assert_in_delta 5.25, result.words.last.end
      assert_equal "ar", library.language
      assert_equal 3, library.sample_count
      assert_equal 640, library.setters.fetch(:session_set_max_new_tokens)
      assert_equal 1, library.setters.fetch(:session_set_beam_size)
      assert_equal 1, library.setters.fetch(:session_set_repetition_loop_guard)
      assert_equal 17, result.generated_tokens
      assert_equal 640, result.generation_limit
      assert_equal 900, result.generation_capacity
      assert result.stopped_by_max_tokens
      refute result.repetition_stopped
      assert library.result_freed?
      assert_equal({ model_path: path, device: "cpu", threads: 3 }, library.opened)

      session.close
      session.close
      assert session.closed?
      assert library.closed
    end
  end

  def test_empty_single_row_is_rejected_before_inference_without_poisoning_the_session
    with_model_file do |path|
      library = FakeLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      error = assert_raises(ArgumentError) { session.transcribe([], language: "en") }
      assert_match(/must not be empty/, error.message)
      assert_nil library.sample_count
      assert_equal "hello world", session.transcribe([0.0], language: "en").text
    ensure
      session&.close
    end
  end

  def test_empty_batch_row_is_rejected_before_inference_without_poisoning_the_session
    with_model_file do |path|
      library = FakeLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      error = assert_raises(ArgumentError) do
        session.transcribe_batch([[0.0], []], language: "en")
      end
      assert_match(/batch row 1 must not be empty/, error.message)
      assert_equal 2, session.transcribe_batch([[0.0], [0.1]], language: "en").length
    ensure
      session&.close
    end
  end

  def test_gc_closes_an_abandoned_native_session_exactly_once
    with_model_file do |path|
      library = FakeLibrary.new
      reference = abandon_native_session(path, library)

      collect_until { !reference.weakref_alive? && library.close_count == 1 }
      3.times { GC.start(full_mark: true, immediate_sweep: true) }

      assert_equal 1, library.close_count
    end
  end

  def test_explicit_close_then_gc_does_not_close_the_native_session_twice
    with_model_file do |path|
      library = FakeLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
      reference = WeakRef.new(session)

      session.close
      session = nil # rubocop:disable Lint/UselessAssignment -- release the final strong reference before GC
      collect_until { !reference.weakref_alive? }

      assert_equal 1, library.close_count
    end
  end

  def test_constructor_interrupt_after_native_open_closes_exactly_once
    with_model_file do |path|
      library = InterruptingInitializationLibrary.new

      error = assert_raises(Interrupt) do
        Cohere::Transcribe::ASR::NativeSession.new(
          path,
          Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
          library: library
        )
      end

      assert_equal "constructor interrupted after native open", error.message
      assert_equal 1, library.close_count
      3.times { GC.start(full_mark: true, immediate_sweep: true) }
      assert_equal 1, library.close_count
    end
  end

  def test_native_text_materialization_uses_python_unicode_whitespace
    with_model_file do |path|
      library = FakeLibrary.new(
        segment_text: "\u00A0hello world\u2003",
        word_texts: ["\u202Fhello", "world\u3000"]
      )
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      result = session.transcribe([0.0], language: "en")

      assert_equal "hello world", result.text
      assert_equal "hello world", result.segments.first.text
      assert_equal %w[hello world], result.words.map(&:text)
    ensure
      session&.close
    end
  end

  def test_reports_native_device_memory
    with_model_file do |path|
      library = FakeLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      assert_equal [3_000, 4_000], session.memory
      assert_equal "CPU", session.compute_backend
      assert_equal "cpu", session.device
      assert_equal 24, session.batch_capacity
    ensure
      session&.close
    end
  end

  def test_rejects_a_missing_native_model_before_open
    library = FakeLibrary.new
    options = Cohere::Transcribe::TranscriptionOptions.new

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      Cohere::Transcribe::ASR::NativeSession.new("/missing/model.gguf", options, library: library)
    end

    assert_match(/does not exist/, error.message)
  end

  def test_rejects_silent_native_device_fallback
    with_model_file do |path|
      library = FakeLibrary.new
      options = Cohere::Transcribe::TranscriptionOptions.new(device: "cuda")

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        Cohere::Transcribe::ASR::NativeSession.new(path, options, library: library)
      end

      assert_match(/selected "CPU" \(cpu\).*"cuda" was resolved/, error.message)
      assert library.closed
      assert_equal 1, library.close_count
    end
  end

  def test_materializes_a_native_batch_and_frees_its_owner
    with_model_file do |path|
      library = FakeLibrary.new
      options = Cohere::Transcribe::TranscriptionOptions.new(device: "cpu", max_new_tokens: 321)
      session = Cohere::Transcribe::ASR::NativeSession.new(path, options, library: library)

      results = session.transcribe_batch(
        [[0.0, 0.1], [0.2, 0.3, 0.4]],
        language: "ar",
        offsets: [1.0, 7.0],
        max_new_tokens: 512
      )

      assert_equal 2, results.length
      assert_equal([1.1, 7.1], results.map { |result| result.segments.first.start })
      assert_equal [512, 512], results.map(&:generation_limit)
      assert_equal 512, library.setters.fetch(:session_set_max_new_tokens)
      assert library.batch_result_freed?
    ensure
      session&.close
    end
  end

  def test_exposes_versioned_native_batch_phase_metrics
    with_model_file do |path|
      library = BatchMetricsLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      session.transcribe_batch([[0.0], [0.1]], language: "en")
      metrics = session.last_batch_metrics

      assert_equal 1, metrics.fetch(:abi_version)
      assert_in_delta 2.0, metrics.fetch(:total_seconds)
      assert_in_delta 0.1, metrics.fetch(:feature_wall_seconds)
      assert_in_delta 0.5, metrics.fetch(:encoder_compute_seconds)
      assert_in_delta 1.2, metrics.fetch(:decoder_total_seconds)
      assert_equal 240, metrics.fetch(:generation_steps)
      assert_equal 3, metrics.fetch(:encoder_microbatches)
      assert_equal 23_040, metrics.fetch(:token_id_readback_bytes)
    ensure
      session&.close
    end
  end

  def test_enforces_the_capacity_queried_from_the_native_session
    with_model_file do |path|
      library = FakeLibrary.new(batch_capacity: 3)
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      assert_equal 3, session.batch_capacity
      assert_equal 3, session.transcribe_batch(
        Array.new(3) { [0.0] },
        language: "en"
      ).length
      error = assert_raises(ArgumentError) do
        session.transcribe_batch(Array.new(4) { [0.0] }, language: "en")
      end
      assert_match(/between 1 and 3 audio rows/, error.message)
    ensure
      session&.close
    end
  end

  def test_classifies_native_batch_failures_without_parsing_messages
    {
      0 => :error,
      1 => :fatal,
      2 => :oom,
      3 => :fatal,
      4 => :error,
      99 => :fatal
    }.each do |native_kind, expected_kind|
      with_model_file do |path|
        library = FakeLibrary.new(
          batch_failure_kind: native_kind,
          failure_message: "allocator diagnostic #{native_kind}"
        )
        session = Cohere::Transcribe::ASR::NativeSession.new(
          path,
          Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
          library: library
        )

        error = assert_raises(Cohere::Transcribe::ASR::ExecutionError) do
          session.transcribe_batch([[0.0]], language: "en")
        end
        assert_equal expected_kind, error.failure_kind
        assert_match(/allocator diagnostic #{native_kind}/, error.message)
        refute library.batch_result_freed?
      ensure
        session&.close
      end
    end
  end

  def test_classifies_a_native_single_row_allocator_failure
    with_model_file do |path|
      library = FakeLibrary.new(single_failure_kind: 2, failure_message: "scheduler allocation failed")
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      error = assert_raises(Cohere::Transcribe::ASR::ExecutionError) do
        session.transcribe([0.0], language: "en")
      end
      assert_equal :oom, error.failure_kind
      assert_match(/scheduler allocation failed/, error.message)
      refute library.result_freed?
    ensure
      session&.close
    end
  end

  def test_native_cancellation_is_an_interrupt_not_a_retryable_execution_failure
    with_model_file do |path|
      library = FakeLibrary.new(single_failure_kind: 5, failure_message: "cooperative abort")
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )

      error = assert_raises(Interrupt) { session.transcribe([0.0], language: "en") }
      assert_match(/cooperative abort/, error.message)
    ensure
      session&.close
    end
  end

  def test_caller_interrupt_cancels_and_hard_joins_before_unwinding
    with_model_file do |path|
      library = BlockingCancellationLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
      caller = Thread.current
      interrupter = Thread.new do
        library.started.pop
        caller.raise(Interrupt, "original caller interruption")
      end

      error = assert_raises(Interrupt) { session.transcribe([0.0], language: "en") }
      assert_equal "original caller interruption", error.message
      assert_operator library.cancel_calls, :>=, 1
      assert library.inference_finished?

      # The cancelled operation returned to idle and did not poison the next
      # inference on the same retained session.
      assert_equal "hello world", session.transcribe([0.0], language: "en").text
      session.close
      refute library.closed_during_inference?
    ensure
      interrupter&.join
      session&.close
    end
  end

  def test_non_signal_caller_exception_also_wins_over_native_cancellation
    with_model_file do |path|
      library = BlockingCancellationLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
      caller = Thread.current
      interrupter = Thread.new do
        library.started.pop
        caller.raise(RuntimeError, "application cancellation")
      end

      error = assert_raises(RuntimeError) { session.transcribe([0.0], language: "en") }
      assert_equal "application cancellation", error.message
      assert library.inference_finished?
    ensure
      interrupter&.join
      session&.close
    end
  end

  def test_externally_killed_native_worker_fails_instead_of_blocking_on_an_empty_outcome
    with_model_file do |path|
      library = KillableWorkerLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
      killer = Thread.new { library.started.pop.kill }
      error = assert_raises(Cohere::Transcribe::ASR::ExecutionError) do
        session.transcribe([0.0], language: "en")
      end
      assert_equal :fatal, error.failure_kind
      assert_match(/worker exited without reporting an outcome/, error.message)
    ensure
      killer&.join
      session&.close
    end
  end

  def test_batch_rejects_a_row_larger_than_the_signed_int_abi_before_packing
    with_model_file do |path|
      library = FakeLibrary.new
      session = Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
      session.define_singleton_method(:float_samples) do |_samples|
        ["".b, Cohere::Transcribe::ASR::NativeSession::MAX_NATIVE_SAMPLE_COUNT + 1]
      end

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        session.transcribe_batch([Object.new], language: "en")
      end
      assert_match(/batch row 0 is too large/, error.message)
    ensure
      session&.close
    end
  end

  private

  def abandon_native_session(path, library)
    WeakRef.new(
      Cohere::Transcribe::ASR::NativeSession.new(
        path,
        Cohere::Transcribe::TranscriptionOptions.new(device: "cpu"),
        library: library
      )
    )
  end

  def collect_until(timeout: 2)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
    loop do
      GC.start(full_mark: true, immediate_sweep: true)
      return if yield
      raise "object was not finalized before the test deadline" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline

      Thread.pass
    end
  end

  def with_model_file
    Tempfile.create(["dense-model", ".gguf"]) do |file|
      file.write("GGUF")
      file.flush
      yield file.path
    end
  end
end
