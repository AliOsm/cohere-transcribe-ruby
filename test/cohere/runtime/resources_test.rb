# frozen_string_literal: true

require "test_helper"
require "timeout"

class Cohere::Transcribe::RuntimeResourcesTest < Minitest::Test
  Resources = Cohere::Transcribe::Runtime::ModelResources

  class Session
    attr_reader :close_count

    def initialize(events: nil, label: nil, close_error: nil)
      @events = events
      @label = label
      @close_error = close_error
      @close_count = 0
    end

    def close
      @close_count += 1
      @events << [:closed, @label] if @events
      raise @close_error if @close_error
    end
  end

  def teardown
    Resources.evict_current_asr_owner
  rescue StandardError
    nil
  end

  def test_same_key_reuses_one_session_and_key_change_reloads
    resources = Resources.new
    first = Session.new
    second = Session.new
    loads = 0

    acquired, loaded = resources.acquire_asr(%w[cpu fp32 model-a]) do
      loads += 1
      first
    end
    reused, reused_loaded = resources.acquire_asr(%w[cpu fp32 model-a]) do
      loads += 1
      raise "must not reload"
    end
    replaced, replacement_loaded = resources.acquire_asr(%w[cuda bf16 model-a]) do
      loads += 1
      second
    end

    assert_same first, acquired
    assert_same first, reused
    assert_same second, replaced
    assert loaded
    refute reused_loaded
    assert replacement_loaded
    assert_equal 2, loads
    assert_equal 1, first.close_count
  ensure
    resources&.close
  end

  def test_second_resource_takes_exclusive_process_ownership
    events = []
    first = Resources.new
    second = Resources.new
    first_session = Session.new(events: events, label: :first)
    second_session = Session.new(events: events, label: :second)

    first.acquire_asr(%w[cpu fp32]) { first_session }
    assert first.has_asr?
    assert_same first, Resources.current_asr_owner

    second.acquire_asr(%w[cpu fp32]) do
      events << %i[loaded second]
      second_session
    end

    refute first.has_asr?
    assert second.has_asr?
    assert_same second, Resources.current_asr_owner
    assert_equal [%i[closed first], %i[loaded second]], events
  ensure
    second&.close
    first&.close
  end

  def test_collected_owner_closes_its_session_before_a_replacement_loads
    events = []
    first_session = Session.new(events: events, label: :first)
    reference = abandon_resources_with(first_session)

    collect_until { !reference.weakref_alive? && first_session.close_count == 1 }
    assert_nil Resources.current_asr_owner

    replacement = Resources.new
    second_session = Session.new(events: events, label: :second)
    replacement.acquire_asr(%w[cuda bf16]) do
      events << %i[loaded second]
      second_session
    end

    assert_equal [%i[closed first], %i[loaded second]], events
    assert_equal 1, first_session.close_count
    3.times { GC.start(full_mark: true, immediate_sweep: true) }
    assert_equal 1, first_session.close_count
  ensure
    replacement&.close
  end

  def test_global_evict_supports_checkpoint_only_word_alignment
    events = []
    owner = Resources.new
    checkpoint_only = Resources.new
    session = Session.new(events: events, label: :asr)
    owner.acquire_asr(%w[cpu fp32]) { session }

    # Constructing a checkpoint-only resource does not steal the lease because
    # it never needs ASR. Alignment still evicts whichever other session owns it.
    refute checkpoint_only.has_asr?
    assert owner.has_asr?
    Resources.evict_current_asr_owner
    events << %i[loaded aligner]

    refute owner.has_asr?
    refute checkpoint_only.has_asr?
    assert_nil Resources.current_asr_owner
    assert_equal [%i[closed asr], %i[loaded aligner]], events
  ensure
    checkpoint_only&.close
    owner&.close
  end

  def test_controller_state_lives_with_the_retained_session
    resources = Resources.new
    resources.acquire_asr(%w[cuda bf16]) { Session.new }
    controller = Cohere::Transcribe::ASR::BatchController.new(
      current_size: 8,
      max_size: 8,
      initial_size: 8,
      audio_budget_seconds: 240.0,
      adaptive: false,
      target_memory_ratio: 0.9
    )
    installed = resources.install_batch_controller(controller)
    controller.open_circuit(RuntimeError.new("CUDA error: illegal memory access"))

    assert_same controller, installed
    assert_same controller, resources.install_batch_controller(Object.new)
    assert resources.asr_circuit_broken?
    resources.evict_asr
    refute resources.asr_circuit_broken?
  ensure
    resources&.close
  end

  def test_failed_loader_does_not_strand_the_process_lease
    resources = Resources.new
    assert_raises(RuntimeError) do
      resources.acquire_asr(%w[cpu fp32]) { raise "load failed" }
    end

    refute resources.has_asr?
    assert_nil Resources.current_asr_owner
  ensure
    resources&.close
  end

  def test_kill_during_loader_releases_the_claimed_process_lease
    resources = Resources.new
    loader_started = Queue.new
    caller = Thread.new do
      resources.acquire_asr(%w[cpu fp32]) do
        loader_started << true
        sleep
      end
    end
    caller.report_on_exception = false

    loader_started.pop
    caller.kill
    assert caller.join(2), "ASR loader remained stuck after termination"
    assert_nil caller.value
    refute resources.has_asr?
    assert_nil Resources.current_asr_owner
  ensure
    caller&.kill
    caller&.join
    resources&.close
  end

  def test_kill_after_loader_return_closes_the_session_and_releases_ownership
    resources = Resources.new
    session = Session.new
    source_path = Resources.instance_method(:acquire_asr).source_location.fetch(0)
    handoff_line = File.readlines(source_path).index do |line|
      line.include?("raise TranscriptionRuntimeError, \"ASR loader returned no native session\"")
    end + 1
    reached = Queue.new
    release = Queue.new
    trace = TracePoint.new(:line) do |event|
      next unless event.path == source_path && event.lineno == handoff_line

      reached << true
      release.pop
    end
    caller = Thread.new do
      trace.enable(target_thread: Thread.current) do
        resources.acquire_asr(%w[cpu fp32]) { session }
      end
    end
    caller.report_on_exception = false

    Timeout.timeout(2) { reached.pop }
    caller.kill
    Thread.pass
    assert caller.alive?, "termination interrupted ASR ownership transfer"
    release << true

    assert caller.join(2), "ASR loader caller remained stuck during rollback"
    assert_nil caller.value
    refute resources.has_asr?
    assert_nil Resources.current_asr_owner
    assert_equal 1, session.close_count
  ensure
    trace&.disable
    release << true if defined?(release) && release.empty?
    caller&.kill
    caller&.join
    resources&.close
  end

  def test_close_is_idempotent_and_permanently_rejects_acquisition
    resources = Resources.new
    session = Session.new
    resources.acquire_asr(%w[cpu fp32]) { session }

    resources.close
    resources.close

    assert resources.closed?
    assert_equal 1, session.close_count
    assert_raises(Cohere::Transcribe::TranscriberClosedError) do
      resources.acquire_asr(%w[cpu fp32]) { Session.new }
    end
  end

  def test_close_failure_still_clears_global_ownership
    resources = Resources.new
    session = Session.new(close_error: RuntimeError.new("native close failed"))
    resources.acquire_asr(%w[cpu fp32]) { session }

    assert_raises(RuntimeError) { resources.close }

    refute resources.has_asr?
    refute resources.closed?
    assert_nil Resources.current_asr_owner
    resources.close
    assert resources.closed?
  end

  private

  def abandon_resources_with(session)
    resources = Resources.new
    resources.acquire_asr(%w[cpu fp32]) { session }
    WeakRef.new(resources)
  end

  def collect_until(timeout: 2)
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + timeout
    loop do
      GC.start(full_mark: true, immediate_sweep: true)
      return if yield
      raise "resources were not finalized before the test deadline" if Process.clock_gettime(Process::CLOCK_MONOTONIC) >= deadline

      Thread.pass
    end
  end
end
