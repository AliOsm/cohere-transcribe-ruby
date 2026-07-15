# frozen_string_literal: true

require "test_helper"
require "timeout"

module Cohere
  module Transcribe
    class ApiLifecycleTest < Minitest::Test
      class FakeEngine
        attr_reader :closed

        def initialize(run)
          @run = run
          @closed = false
          @lock = Mutex.new
        end

        def transcribe(_audio, **)
          @lock.synchronize do
            raise TranscriberClosedError, "This Transcriber has been closed" if @closed

            @run
          end
        end

        def close
          @lock.synchronize { @closed = true }
          nil
        end
      end

      class OwnedOneShotSession
        attr_reader :closed

        def initialize(outcome)
          @outcome = outcome
          @closed = false
        end

        def transcribe(*)
          raise @outcome if @outcome.is_a?(Exception)

          @outcome
        end

        def close
          @closed = true
          nil
        end
      end

      def test_same_session_calls_serialize_and_include_the_gate_wait
        options = TranscriptionOptions.new
        run_builder = ->(wait) { run_for(options, serialization_wait_seconds: wait) }
        entered = Queue.new
        release = Queue.new
        calls = 0
        activity_lock = Mutex.new
        active = 0
        maximum_active = 0
        engine = Runtime::Engine.new(options)
        engine.define_singleton_method(:execute) do |_audio, measurements, _started|
          call = activity_lock.synchronize do
            calls += 1
            active += 1
            maximum_active = [maximum_active, active].max
            calls
          end
          if call == 1
            entered << true
            release.pop
          end
          activity_lock.synchronize { active -= 1 }
          run_builder.call(measurements.serialization_wait_seconds)
        end
        outcomes = []
        outcomes_lock = Mutex.new
        worker = lambda do |name|
          value = engine.transcribe([name])
          outcomes_lock.synchronize { outcomes << value }
        rescue Exception => e # rubocop:disable Lint/RescueException -- thread assertions retain every outcome
          outcomes_lock.synchronize { outcomes << e }
        end

        first = Thread.new { worker.call("first.wav") }
        entered.pop
        second = Thread.new { worker.call("second.wav") }
        sleep 0.03
        release << true
        [first, second].each { |thread| assert thread.join(2), "transcription thread did not finish" }

        assert_equal 1, maximum_active
        assert_equal 2, outcomes.length
        assert outcomes.all?(TranscriptionRun), outcomes.inspect
        assert_operator outcomes.map { |run| run.statistics.serialization_wait_seconds }.max, :>=, 0.02
      ensure
        engine&.close
      end

      def test_process_gate_serializes_distinct_sessions
        options = TranscriptionOptions.new
        run_builder = ->(wait) { run_for(options, serialization_wait_seconds: wait) }
        entered = Queue.new
        release = Queue.new
        activity_lock = Mutex.new
        active = 0
        maximum_active = 0
        first_engine = Runtime::Engine.new(options)
        second_engine = Runtime::Engine.new(options)
        [first_engine, second_engine].each_with_index do |engine, index|
          engine.define_singleton_method(:execute) do |_audio, measurements, _started|
            activity_lock.synchronize do
              active += 1
              maximum_active = [maximum_active, active].max
            end
            if index.zero?
              entered << true
              release.pop
            end
            activity_lock.synchronize { active -= 1 }
            run_builder.call(measurements.serialization_wait_seconds)
          end
        end
        outcomes = Queue.new
        first = Thread.new { outcomes << first_engine.transcribe(["first.wav"]) }
        entered.pop
        second = Thread.new { outcomes << second_engine.transcribe(["second.wav"]) }
        sleep 0.03
        release << true
        [first, second].each { |thread| assert thread.join(2), "transcription thread did not finish" }
        runs = 2.times.map { outcomes.pop }

        assert_equal 1, maximum_active
        assert_operator runs.map { |run| run.statistics.serialization_wait_seconds }.max, :>=, 0.02
      ensure
        first_engine&.close
        second_engine&.close
      end

      def test_same_thread_reentry_on_same_and_other_sessions_is_rejected_then_recovers
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        same_errors = []
        other_errors = []
        same_calls = 0
        outer_engine = Runtime::Engine.new(options)
        inner_engine = Runtime::Engine.new(options)
        same_session = nil
        inner_session = Transcriber.new(options, engine_factory: -> { inner_engine })
        outer_session = nil

        outer_engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          begin
            inner_session.transcribe("nested-other.wav")
          rescue Exception => e # rubocop:disable Lint/RescueException
            other_errors << e
          end
          run_builder.call
        end
        inner_engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          run_builder.call
        end
        same_engine = Runtime::Engine.new(options)
        same_engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          same_calls += 1
          if same_calls == 1
            begin
              same_session.transcribe("nested-same.wav")
            rescue Exception => e # rubocop:disable Lint/RescueException
              same_errors << e
            end
          end
          run_builder.call
        end
        same_session = Transcriber.new(options, engine_factory: -> { same_engine })
        outer_session = Transcriber.new(options, engine_factory: -> { outer_engine })

        assert same_session.transcribe("outer-same.wav").ok?
        assert outer_session.transcribe("outer-other.wav").ok?
        assert inner_session.transcribe("recovered-inner.wav").ok?
        assert same_session.transcribe("recovered-same.wav").ok?
        assert_instance_of TranscriberBusyError, same_errors.fetch(0)
        assert_match(/one process/, other_errors.fetch(0).message)
      ensure
        same_session&.close
        outer_session&.close
        inner_session&.close
      end

      def test_reentrant_close_is_rejected_without_closing_the_facade
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        errors = []
        calls = 0
        engine = Runtime::Engine.new(options)
        session = nil
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          calls += 1
          if calls == 1
            begin
              session.close
            rescue Exception => e # rubocop:disable Lint/RescueException
              errors << e
            end
          end
          run_builder.call
        end
        session = Transcriber.new(options, engine_factory: -> { engine })

        assert session.transcribe("first.wav").ok?
        refute session.closed?
        assert session.transcribe("second.wav").ok?
        assert_instance_of TranscriberBusyError, errors.fetch(0)
        assert_match(/active/, errors.fetch(0).message)
      ensure
        session&.close
      end

      def test_reentry_from_another_fiber_on_the_same_thread_is_rejected
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        nested_errors = []
        engine = Runtime::Engine.new(options)
        session = nil
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          Fiber.new do
            session.transcribe("nested-fiber.wav")
          rescue Exception => e # rubocop:disable Lint/RescueException
            nested_errors << e
          end.resume
          run_builder.call
        end
        session = Transcriber.new(options, engine_factory: -> { engine })

        assert session.transcribe("outer.wav").ok?
        assert_instance_of TranscriberBusyError, nested_errors.fetch(0)
        assert_match(/one process/, nested_errors.fetch(0).message)
      ensure
        session&.close
      end

      def test_close_waits_for_in_flight_work_and_rejects_new_operations
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        entered = Queue.new
        release = Queue.new
        engine = Runtime::Engine.new(options)
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          entered << true
          release.pop
          run_builder.call
        end
        session = Transcriber.new(options, engine_factory: -> { engine })
        transcription_outcome = []
        close_outcome = []
        transcription = Thread.new do
          transcription_outcome << session.transcribe("in-flight.wav")
        rescue Exception => e # rubocop:disable Lint/RescueException
          transcription_outcome << e
        end
        entered.pop
        closer = Thread.new do
          session.close
          close_outcome << :closed
        rescue Exception => e # rubocop:disable Lint/RescueException
          close_outcome << e
        end
        Timeout.timeout(2) do
          Thread.pass until session.instance_variable_get(:@closing)
        end

        assert_raises(TranscriberBusyError) { session.transcribe("racing.wav") }
        assert_raises(TranscriberBusyError) { session.close }
        release << true
        assert transcription.join(2), "in-flight transcription did not finish"
        assert closer.join(2), "close did not finish"
        assert_instance_of TranscriptionRun, transcription_outcome.fetch(0)
        assert_equal [:closed], close_outcome
        assert session.closed?
        assert_raises(TranscriberClosedError) { session.transcribe("after-close.wav") }
      ensure
        release << true if defined?(release) && release.empty?
        transcription&.join(0.1)
        closer&.join(0.1)
        session&.close unless session&.closed?
      end

      def test_second_close_is_rejected_while_resource_close_is_in_progress
        options = TranscriptionOptions.new
        started = Queue.new
        release = Queue.new
        close_calls = 0
        engine = FakeEngine.new(run_for(options))
        engine.define_singleton_method(:close) do
          close_calls += 1
          started << true
          release.pop
          super()
        end
        session = Transcriber.new(options, engine_factory: -> { engine })
        session.transcribe("materialize.wav")
        first_errors = []
        first = Thread.new do
          session.close
        rescue Exception => e # rubocop:disable Lint/RescueException
          first_errors << e
        end
        started.pop

        error = assert_raises(TranscriberBusyError) { session.close }
        assert_match(/already being closed/, error.message)
        assert_raises(TranscriberBusyError) { session.transcribe("racing.wav") }
        release << true
        assert first.join(2), "first close did not finish"
        assert_empty first_errors
        assert_equal 1, close_calls
        assert session.closed?
      ensure
        release << true if defined?(release) && release.empty?
        first&.join(0.1)
      end

      def test_failed_close_rolls_back_closing_state
        options = TranscriptionOptions.new
        engine = FakeEngine.new(run_for(options))
        close_calls = 0
        engine.define_singleton_method(:close) do
          close_calls += 1
          raise "resource close failed" if close_calls == 1

          super()
        end
        session = Transcriber.new(options, engine_factory: -> { engine })
        session.transcribe("before.wav")

        error = assert_raises(RuntimeError) { session.close }
        assert_equal "resource close failed", error.message
        refute session.closed?
        assert session.transcribe("after-failed-close.wav").ok?
        assert_nil session.close
        assert_equal 2, close_calls
        assert session.closed?
      end

      def test_close_racing_lazy_initialization_does_not_leak_the_created_engine
        options = TranscriptionOptions.new
        run = run_for(options)
        factory_entered = Queue.new
        release_factory = Queue.new
        created = []
        session = Transcriber.new(
          options,
          engine_factory: lambda {
            factory_entered << true
            release_factory.pop
            FakeEngine.new(run).tap { |engine| created << engine }
          }
        )
        outcomes = []
        transcribing = Thread.new do
          outcomes << session.transcribe("racing.wav")
        rescue Exception => e # rubocop:disable Lint/RescueException
          outcomes << e
        end
        factory_entered.pop
        closing = Thread.new { session.close }
        Thread.pass
        release_factory << true
        assert transcribing.join(2), "transcription did not finish"
        assert closing.join(2), "close did not finish"

        assert_equal 1, created.length
        assert created.first.closed
        assert session.closed?
        assert outcomes.first.is_a?(TranscriptionRun) || outcomes.first.is_a?(TranscriberClosedError)
      ensure
        release_factory << true if defined?(release_factory) && release_factory.empty?
        transcribing&.join(0.1)
        closing&.join(0.1)
      end

      def test_runtime_initialization_system_exit_is_typed_and_recoverable
        options = TranscriptionOptions.new
        engine = FakeEngine.new(run_for(options))
        calls = 0
        session = Transcriber.new(
          options,
          engine_factory: lambda {
            calls += 1
            raise SystemExit, "runtime setup stopped" if calls == 1

            engine
          }
        )

        error = assert_raises(TranscriptionRuntimeError) { session.transcribe("first.wav") }
        assert_equal "runtime setup stopped", error.message
        assert_instance_of SystemExit, error.cause
        assert session.transcribe("second.wav").ok?
        assert_equal 2, calls
      ensure
        session&.close
      end

      def test_lazy_dependency_load_failure_is_typed_at_both_runtime_boundaries
        options = TranscriptionOptions.new
        facade = Transcriber.new(
          options,
          engine_factory: -> { raise LoadError, "cannot load such file -- optional_backend" }
        )
        facade_error = assert_raises(TranscriptionRuntimeError) do
          facade.transcribe("first.wav")
        end
        assert_match(/LoadError.*optional_backend/, facade_error.message)
        assert_instance_of LoadError, facade_error.cause

        engine = Runtime::Engine.new(options)
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          raise LoadError, "cannot load such file -- model_backend"
        end
        engine_error = assert_raises(TranscriptionRuntimeError) do
          engine.transcribe(["first.wav"])
        end
        assert_match(/LoadError.*model_backend/, engine_error.message)
        assert_instance_of LoadError, engine_error.cause
      ensure
        facade&.close
        engine&.close
      end

      def test_runtime_execution_system_exit_is_typed_and_engine_recovers
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        calls = 0
        engine = Runtime::Engine.new(options)
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          calls += 1
          raise SystemExit, "backend initialization failed" if calls == 1

          run_builder.call
        end

        error = assert_raises(TranscriptionRuntimeError) { engine.transcribe(["first.wav"]) }
        assert_equal "backend initialization failed", error.message
        assert_instance_of SystemExit, error.cause
        assert engine.transcribe(["second.wav"]).ok?
      ensure
        engine&.close
      end

      def test_bare_system_exit_uses_stable_typed_fallback_messages
        options = TranscriptionOptions.new
        facade = Transcriber.new(options, engine_factory: -> { raise SystemExit })
        facade_error = assert_raises(TranscriptionRuntimeError) do
          facade.transcribe("first.wav")
        end
        assert_equal "Transcription runtime initialization failed", facade_error.message

        engine = Runtime::Engine.new(options)
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          raise SystemExit
        end
        engine_error = assert_raises(TranscriptionRuntimeError) do
          engine.transcribe(["first.wav"])
        end
        assert_equal "Transcription setup failed", engine_error.message
      ensure
        facade&.close
        engine&.close
      end

      def test_worker_progress_callback_reentry_is_rejected_without_blocking
        %i[transcribe close].each do |operation|
          options = TranscriptionOptions.new
          run_builder = -> { run_for(options) }
          outcomes = []
          callback_finished = Queue.new
          session = nil
          callback = lambda do |_event|
            operation == :transcribe ? session.transcribe("nested.wav") : session.close
          rescue Exception => e # rubocop:disable Lint/RescueException
            outcomes << e
          ensure
            callback_finished << true
          end
          engine = Runtime::Engine.new(options, progress: callback)
          engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
            worker = Thread.new { send(:report, ProgressEvent.new(stage: "message", message: "worker")) }
            worker.report_on_exception = false
            callback_finished.pop
            raise "progress worker blocked" unless worker.join(2)

            run_builder.call
          end
          session = Transcriber.new(options, progress: callback, engine_factory: -> { engine })

          assert session.transcribe("outer.wav").ok?
          assert_instance_of TranscriberBusyError, outcomes.fetch(0)
          refute session.closed?
          session.close
        end
      end

      def test_progress_reentry_guard_crosses_fiber_boundaries
        options = TranscriptionOptions.new
        run_builder = -> { run_for(options) }
        nested_errors = []
        session = nil
        callback = lambda do |_event|
          Fiber.new do
            session.transcribe("nested-fiber-callback.wav")
          rescue Exception => e # rubocop:disable Lint/RescueException
            nested_errors << e
          end.resume
        end
        engine = Runtime::Engine.new(options, progress: callback)
        engine.define_singleton_method(:execute) do |_audio, _measurements, _started|
          run_builder.call
        end
        session = Transcriber.new(options, progress: callback, engine_factory: -> { engine })
        session.transcribe("materialize.wav")

        engine.send(:report, ProgressEvent.new(stage: "message", message: "outside run"))

        assert_instance_of TranscriberBusyError, nested_errors.fetch(0)
        assert_match(/progress callback/, nested_errors.fetch(0).message)
      ensure
        session&.close
      end

      def test_progress_callback_is_never_invoked_concurrently
        options = TranscriptionOptions.new
        lock = Mutex.new
        active = 0
        maximum_active = 0
        received = []
        callback = lambda do |event|
          lock.synchronize do
            active += 1
            maximum_active = [maximum_active, active].max
          end
          sleep 0.005
          lock.synchronize do
            received << event.message
            active -= 1
          end
        end
        engine = Runtime::Engine.new(options, progress: callback)
        start = Queue.new
        errors = Queue.new
        threads = 8.times.map do |index|
          Thread.new do
            start.pop
            engine.send(:report, ProgressEvent.new(stage: "message", message: "message-#{index}"))
          rescue Exception => e # rubocop:disable Lint/RescueException
            errors << e
          end.tap { |thread| thread.report_on_exception = false }
        end
        threads.length.times { start << true }
        threads.each { |thread| assert thread.join(2), "progress thread did not finish" }

        thread_error = errors.empty? ? nil : errors.pop
        assert_nil thread_error
        assert_equal 1, maximum_active
        assert_equal 8.times.map { |index| "message-#{index}" }.sort, received.sort
      ensure
        engine&.close
      end

      def test_callback_raising_a_typed_error_is_still_wrapped_as_user_callback_failure
        options = TranscriptionOptions.new
        original = ProgressCallbackError.new(RuntimeError.new("inner callback failure"))
        engine = Runtime::Engine.new(options, progress: ->(_event) { raise original })

        raised = assert_raises(ProgressCallbackError) do
          engine.send(:report, ProgressEvent.new(stage: "message"))
        end

        refute_same original, raised
        assert_same original, raised.original
        assert_same original, raised.cause
      ensure
        engine&.close
      end

      def test_callback_load_errors_are_classified_as_callback_failures
        options = TranscriptionOptions.new
        original = LoadError.new("cannot load such file -- callback_dependency")
        engine = Runtime::Engine.new(options, progress: ->(_event) { raise original })

        raised = assert_raises(ProgressCallbackError) do
          engine.send(:report, ProgressEvent.new(stage: "message"))
        end

        assert_same original, raised.original
        assert_same original, raised.cause
      ensure
        engine&.close
      end

      def test_open_closes_after_the_block_raises
        options = TranscriptionOptions.new
        engine = FakeEngine.new(run_for(options))

        error = assert_raises(RuntimeError) do
          Transcriber.open(options, engine_factory: -> { engine }) do |session|
            session.transcribe("one.wav")
            raise "application failed"
          end
        end

        assert_equal "application failed", error.message
        assert engine.closed
      end

      def test_one_shot_api_closes_its_owned_session_on_every_outcome
        options = TranscriptionOptions.new
        successful = OwnedOneShotSession.new(run_for(options))
        with_owned_session(successful) do
          assert Cohere::Transcribe.transcribe("success.wav").ok?
        end
        assert successful.closed

        original = RuntimeError.new("unexpected runtime failure")
        failed = OwnedOneShotSession.new(original)
        with_owned_session(failed) do
          assert_same original, assert_raises(RuntimeError) {
            Cohere::Transcribe.transcribe("failure.wav")
          }
        end
        assert failed.closed

        partial = run_for(options).with(errors: ["profile failed"])
        batch = OwnedOneShotSession.new(BatchTranscriptionError.new(partial))
        with_owned_session(batch) do
          raised = assert_raises(BatchTranscriptionError) do
            Cohere::Transcribe.transcribe("batch.wav", raise_on_error: true)
          end
          assert_equal partial.results, raised.run.results
          assert_equal partial.errors, raised.run.errors
        end
        assert batch.closed
      end

      def test_configuration_type_failures_cross_the_public_boundary_as_configuration_errors
        invalid_path = Object.new
        invalid_path.define_singleton_method(:to_path) { 123 }
        options = TranscriptionOptions.new(model: invalid_path)
        engine = Runtime::Engine.new(options)
        session = Transcriber.new(options, engine_factory: -> { engine })

        error = assert_raises(TranscriptionConfigurationError) { session.transcribe("unused.wav") }
        assert_match(/resolve to text/, error.message)
      ensure
        session&.close
      end

      private

      def with_owned_session(session)
        singleton = Transcriber.singleton_class
        original = Transcriber.method(:new)
        singleton.define_method(:new) { |*, **| session }
        yield
      ensure
        singleton&.define_method(:new, original) if original
      end

      def run_for(options, serialization_wait_seconds: 0.0)
        statistics = TranscriptionStatistics.new(
          elapsed_seconds: 0.0,
          successful_audio_seconds: 0.0,
          real_time_factor_x: 0.0,
          runtime_import_seconds: 0.0,
          serialization_wait_seconds: serialization_wait_seconds,
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
        TranscriptionRun.new(
          results: [],
          requested_options: options,
          resolved_options: options,
          statistics: statistics
        )
      end
    end
  end
end
