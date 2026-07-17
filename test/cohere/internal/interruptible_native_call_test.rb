# frozen_string_literal: true

require "test_helper"
require "cohere/transcribe/internal/interruptible_native_call"
require "timeout"

class Cohere::Transcribe::InterruptibleNativeCallTest < Minitest::Test
  Runner = Cohere::Transcribe::Internal::InterruptibleNativeCall

  def test_returns_the_worker_value_and_assigns_its_name
    value = Runner.run(
      cancel: -> { flunk "completed calls must not cancel" },
      join_interval: 0.001,
      missing_outcome: RuntimeError.new("missing"),
      thread_name: "native-call-test"
    ) { [Thread.current.name, 42] }

    assert_equal ["native-call-test", 42], value
  end

  def test_worker_exception_is_raised_on_the_caller
    failure = RuntimeError.new("worker failed")

    error = assert_raises(RuntimeError) do
      Runner.run(
        cancel: -> { flunk "worker failures must not cancel" },
        join_interval: 0.001,
        missing_outcome: RuntimeError.new("missing")
      ) { raise failure }
    end

    assert_same failure, error
  end

  def test_caller_exception_wins_while_cancel_retries_until_the_worker_exits
    started = Queue.new
    release = Queue.new
    cancel_calls = 0
    caller = Thread.current
    interrupter = Thread.new do
      started.pop
      caller.raise(Interrupt, "caller stopped")
    end
    cancel = lambda do
      cancel_calls += 1
      raise "first cancel failed" if cancel_calls == 1

      release << true if cancel_calls == 3
    end

    error = assert_raises(Interrupt) do
      Runner.run(
        cancel: cancel,
        join_interval: 0.001,
        missing_outcome: RuntimeError.new("missing")
      ) do
        started << true
        release.pop
        raise "worker failed during cancellation"
      end
    end

    assert_equal "caller stopped", error.message
    assert_operator cancel_calls, :>=, 3
  ensure
    interrupter&.join
    release << true if defined?(release) && release.empty?
  end

  def test_missing_worker_outcome_raises_the_supplied_exception
    started = Queue.new
    missing = RuntimeError.new("worker vanished")
    killer = Thread.new { started.pop.kill }

    error = assert_raises(RuntimeError) do
      Runner.run(
        cancel: -> { flunk "an externally killed worker has already exited" },
        join_interval: 0.001,
        missing_outcome: missing
      ) do
        started << Thread.current
        sleep
      end
    end

    assert_same missing, error
  ensure
    killer&.join
  end

  def test_caller_exception_survives_an_always_failing_cancel_after_the_worker_finishes
    started = Queue.new
    cancel_calls = 0
    failure = nil
    caller = Thread.new do
      Runner.run(
        cancel: lambda {
          cancel_calls += 1
          raise "cancel failed"
        },
        join_interval: 0.001,
        missing_outcome: RuntimeError.new("missing")
      ) do
        started << true
        sleep 0.02
      end
    rescue Exception => e # rubocop:disable Lint/RescueException -- capture caller interruption for assertion
      failure = e
    end
    interrupter = Thread.new do
      started.pop
      caller.raise(Interrupt, "caller stopped")
    end

    assert caller.join(2), "caller remained stuck after its worker exited"
    assert_instance_of Interrupt, failure
    assert_equal "caller stopped", failure.message
    assert_operator cancel_calls, :>, 0
  ensure
    caller&.kill
    caller&.join
    interrupter&.join
  end

  def test_killing_the_caller_cancels_and_joins_the_dedicated_worker
    started = Queue.new
    release = Queue.new
    dedicated_worker = Queue.new
    cancel_calls = 0
    caller = Thread.new do
      Runner.run(
        cancel: lambda {
          cancel_calls += 1
          raise "first cancel failed" if cancel_calls == 1

          release << true if cancel_calls == 3
        },
        join_interval: 0.001,
        missing_outcome: RuntimeError.new("missing")
      ) do
        dedicated_worker << Thread.current
        started << true
        release.pop
        raise "worker failed during cancellation"
      end
    end
    caller.report_on_exception = false

    started.pop
    worker = dedicated_worker.pop
    caller.kill

    assert caller.join(2), "caller remained stuck while cleaning up its dedicated worker"
    assert_nil caller.value
    refute worker.alive?, "dedicated worker survived caller termination"
    assert_operator cancel_calls, :>=, 3
  ensure
    release << true if defined?(worker) && worker&.alive?
    worker&.join
    caller&.kill
    caller&.join
  end

  def test_killing_the_caller_between_worker_creation_and_join_cleans_up_the_worker
    helper_path = Runner.method(:run).source_location.fetch(0)
    metadata_line = File.readlines(helper_path).index { |line| line.include?("worker.name =") } + 1
    reached = Queue.new
    release = Queue.new
    cancel_calls = 0
    trace = TracePoint.new(:line) do |event|
      next unless event.path == helper_path && event.lineno == metadata_line

      reached << event.binding.local_variable_get(:worker)
      sleep
    end
    caller = Thread.new do
      trace.enable(target_thread: Thread.current) do
        Runner.run(
          cancel: lambda {
            cancel_calls += 1
            release << true
          },
          join_interval: 0.001,
          missing_outcome: RuntimeError.new("missing")
        ) { release.pop }
      end
    end
    caller.report_on_exception = false

    worker = Timeout.timeout(2) { reached.pop }
    caller.kill

    assert caller.join(2), "caller remained stuck during pre-join cleanup"
    assert_nil caller.value
    refute worker.alive?, "dedicated worker survived termination before join"
    assert_operator cancel_calls, :>=, 1
  ensure
    trace&.disable
    release << true if defined?(worker) && worker&.alive?
    worker&.join
    caller&.kill
    caller&.join
  end

  def test_kill_during_soft_interrupt_cleanup_waits_for_the_native_worker
    started = Queue.new
    release_worker = Queue.new
    cleanup_started = Queue.new
    release_cleanup = Queue.new
    dedicated_worker = Queue.new
    cancel_calls = 0
    caller = Thread.new do
      Runner.run(
        cancel: lambda {
          cancel_calls += 1
          if cancel_calls == 1
            cleanup_started << true
            release_cleanup.pop
            release_worker << true
          end
        },
        join_interval: 0.001,
        missing_outcome: RuntimeError.new("missing")
      ) do
        dedicated_worker << Thread.current
        started << true
        release_worker.pop
      end
    end
    caller.report_on_exception = false

    started.pop
    worker = dedicated_worker.pop
    caller.raise(Interrupt, "soft stop")
    Timeout.timeout(2) { cleanup_started.pop }
    caller.kill
    Thread.pass
    assert caller.alive?, "caller termination interrupted native cleanup"

    release_cleanup << true
    assert caller.join(2), "caller remained stuck after native cleanup"
    assert_nil caller.value
    refute worker.alive?, "dedicated worker survived escalated caller termination"
    assert_operator cancel_calls, :>=, 1
  ensure
    release_cleanup << true if defined?(release_cleanup) && release_cleanup.empty?
    release_worker << true if defined?(worker) && worker&.alive? && release_worker.empty?
    worker&.join
    caller&.kill
    caller&.join
  end
end
