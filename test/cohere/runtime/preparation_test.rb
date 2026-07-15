# frozen_string_literal: true

require "timeout"
require "test_helper"

class Cohere::Transcribe::RuntimePreparationTest < Minitest::Test
  Pipeline = Cohere::Transcribe::Runtime::Preparation::Pipeline

  def test_automatic_worker_count_is_bounded_to_two_and_available_processors
    pipeline = Pipeline.new(
      (0...8).to_a,
      memory_byte_limit: 1_024,
      requested_workers: nil,
      enabled: true
    ) { |item, _limit, _slot| item }

    assert_operator pipeline.effective_workers, :>=, 1
    assert_operator pipeline.effective_workers, :<=, 2
    assert_operator pipeline.effective_workers, :<=, Etc.nprocessors
  end

  def test_explicit_worker_limit_caps_independent_file_concurrency
    pipeline = Pipeline.new(
      (0...8).to_a,
      memory_byte_limit: 1_024,
      requested_workers: 8,
      enabled: true,
      worker_limit: 1
    ) { |item, _limit, _slot| item }

    assert_equal 1, pipeline.effective_workers
    assert_equal (0...8).to_a, pipeline.to_a
  end

  def test_worker_limit_must_be_positive
    error = assert_raises(ArgumentError) do
      Pipeline.new(
        [1],
        memory_byte_limit: 1_024,
        requested_workers: 1,
        enabled: true,
        worker_limit: 0
      ) { |item, _limit, _slot| item }
    end

    assert_equal "worker_limit must be positive", error.message
  end

  def test_pipelining_is_ordered_bounded_and_only_one_group_ahead
    workers = [2, Etc.nprocessors].min
    second_group = (workers...(2 * workers)).to_a
    third_group_start = 2 * workers
    mutex = Mutex.new
    started = []
    release_second_group = Queue.new
    limits = []
    pipeline = Pipeline.new(
      (0...6).to_a,
      memory_byte_limit: 100,
      requested_workers: 2,
      enabled: true
    ) do |item, limit, _slot|
      mutex.synchronize do
        started << item
        limits << [item, limit]
      end
      release_second_group.pop if second_group.include?(item)
      item
    end

    yielded = []
    pipeline.each do |item|
      yielded << item
      next unless item.zero?

      Timeout.timeout(2) do
        Thread.pass until mutex.synchronize { (second_group - started).empty? }
      end
      snapshot = mutex.synchronize { started.dup }
      second_group.each { |index| assert_includes snapshot, index }
      refute_includes snapshot, third_group_start, "a second look-ahead group started"
      workers.times { release_second_group << true }
    end

    assert pipeline.pipelined?
    assert_equal workers, pipeline.effective_workers
    assert_equal 50, pipeline.group_byte_limit
    assert_equal (0...6).to_a, yielded
    assert_equal Array.new(6, 50 / workers), limits.sort.map(&:last)
  end

  def test_disabled_pipeline_is_a_full_budget_sequential_escape_path
    caller = Thread.current
    observed = []
    pipeline = Pipeline.new(
      %i[first second third],
      memory_byte_limit: 123,
      requested_workers: 8,
      enabled: false
    ) do |item, limit, slot|
      observed << [item, limit, slot, Thread.current]
      item
    end

    assert_equal %i[first second third], pipeline.to_a
    refute pipeline.pipelined?
    assert_equal 1, pipeline.effective_workers
    observed_limits = observed.map { |row| row[1] }
    observed_slots = observed.map { |row| row[2] }
    assert_equal [123, 123, 123], observed_limits
    assert_equal [0, 0, 0], observed_slots
    assert(observed.all? { |row| row[3].equal?(caller) })
    assert_equal 0.0, pipeline.wait_seconds
  end

  def test_pipeline_reports_time_blocked_waiting_for_preparation
    pipeline = Pipeline.new(
      %i[first second],
      memory_byte_limit: 100,
      requested_workers: 1,
      enabled: true
    ) do |item, _limit, _slot|
      sleep 0.01 if item == :first
      item
    end

    assert_equal %i[first second], pipeline.to_a
    assert_operator pipeline.wait_seconds, :>=, 0.008
  end

  def test_consumer_failure_cancels_and_joins_the_in_flight_group
    workers = [2, Etc.nprocessors].min
    second_group = (workers...(2 * workers)).to_a
    next_group_started = Queue.new
    pipeline = Pipeline.new(
      (0...4).to_a,
      memory_byte_limit: 100,
      requested_workers: 2,
      enabled: true
    ) do |item, _limit, _slot|
      if second_group.include?(item)
        next_group_started << item
        sleep
      end
      item
    end

    error = assert_raises(RuntimeError) do
      pipeline.each do |item|
        next unless item.zero?

        Timeout.timeout(2) { workers.times { next_group_started.pop } }
        raise "abort consumer"
      end
    end
    assert_equal "abort consumer", error.message
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-audio-") })
  end

  def test_pipeline_cancellation_notifies_an_already_loaded_native_decoder
    workers = [2, Etc.nprocessors].min
    second_group = (workers...(2 * workers)).to_a
    next_group_started = Queue.new
    cancellations = Queue.new
    native = Cohere::Transcribe::Audio::FFmpegNative
    original_cancel = native.method(:cancel_active!)
    native.define_singleton_method(:cancel_active!) { cancellations << true }
    pipeline = Pipeline.new(
      (0...(2 * workers)).to_a,
      memory_byte_limit: 100,
      requested_workers: workers,
      enabled: true
    ) do |item, _limit, _slot|
      if second_group.include?(item)
        next_group_started << true
        sleep
      end
      item
    end

    assert_raises(RuntimeError) do
      pipeline.each do |item|
        next unless item.zero?

        Timeout.timeout(2) { next_group_started.pop }
        raise "abort consumer"
      end
    end
    refute cancellations.empty?, "native decoder cancellation hook was not called"
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-audio-") })
  ensure
    native&.define_singleton_method(:cancel_active!) { original_cancel.call }
  end

  def test_worker_interrupt_propagates_and_sibling_workers_are_joined
    workers = [2, Etc.nprocessors].min
    sibling_started = Queue.new
    pipeline = Pipeline.new(
      %i[interrupt sibling],
      memory_byte_limit: 100,
      requested_workers: 2,
      enabled: true
    ) do |item, _limit, _slot|
      if item == :interrupt
        sibling_started.pop if workers > 1
        raise Interrupt
      end
      sibling_started << true
      sleep
    end

    assert_raises(Interrupt) { pipeline.to_a }
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-audio-") })
  end
end
