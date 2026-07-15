# frozen_string_literal: true

require "test_helper"

class Cohere::Transcribe::ASRBatchingTest < Minitest::Test
  ASR = Cohere::Transcribe::ASR

  Result = Data.define(:generated_tokens)

  def test_device_defaults_and_physical_native_cap
    assert_equal({ "cuda" => 24, "mps" => 8, "cpu" => 4 }, ASR::BatchController::DEVICE_DEFAULTS)
    controller = ASR::BatchController.create(
      options(batch_size: 24),
      device: "cuda",
      durations: [30.0],
      memory: [8 * (1024**3), 12 * (1024**3)],
      physical_max: 8
    )

    assert_equal 8, controller.current_size
    assert_equal 8, controller.max_size
    assert_equal 240.0, controller.audio_budget_seconds
    assert_operator controller.memory_budget_bytes, :>, 0
  end

  def test_static_oom_learning_and_group_audio_budget
    controller = ASR::BatchController.create(
      options(batch_size: 4),
      device: "cpu",
      durations: [0.5]
    )
    assert_equal [4, 4], [controller.current_size, controller.max_size]
    assert_equal 2.0, controller.audio_budget_seconds

    controller.configure_group(options(batch_size: 4), [30.0])
    assert_equal 120.0, controller.audio_budget_seconds
    controller.record_oom(4)

    assert_equal [2, 3], [controller.current_size, controller.max_size]
    assert_equal 2, controller.growth_cooldown
  end

  def test_memory_measured_success_grows_cautiously
    gib = 1024**3
    controller = ASR::BatchController.new(
      current_size: 8,
      max_size: 32,
      initial_size: 8,
      audio_budget_seconds: 240.0,
      adaptive: true,
      target_memory_ratio: 0.9,
      total_memory_bytes: 12 * gib,
      memory_budget_bytes: 10 * gib
    )

    controller.record_success(8, baseline_reserved_bytes: 4 * gib, peak_reserved_bytes: 6 * gib)

    assert_equal 10, controller.current_size
  end

  def test_recursive_oom_siblings_honor_the_learned_cap_and_metrics
    controller = controller(size: 24)
    attempts = []
    rows = Array.new(24) { |index| { index: index, duration: 1.0 } }
    executor = ASR::BatchExecutor.new(controller)

    result = executor.execute(rows, max_new_tokens: 445, base_max_new_tokens: 445) do |batch|
      attempts << batch.length
      raise ASR::ExecutionError.new("synthetic out of memory", failure_kind: :oom) if batch.length > 6

      batch.map { Result.new(generated_tokens: 2) }
    end

    assert result.success?
    assert_equal [24, 12, 6, 6, 6, 6], attempts
    assert_equal 6, controller.current_size
    assert_equal 2, result.telemetry.oom_retries
    assert_equal 4, result.telemetry.asr_batches
    assert_equal 24, result.telemetry.processor_rows
    assert_equal 48, result.telemetry.generated_tokens
    assert_equal [6, 6], [result.telemetry.effective_batch_min, result.telemetry.effective_batch_max]
    assert_equal 6, result.telemetry.final_batch_size
    assert_equal(%w[oom oom], result.telemetry.batch_history.filter_map { |item| item["event"] })
    successful = result.telemetry.batch_history.reject { |item| item["event"] }
    assert_equal([6, 6, 6, 6], successful.map { |item| item.fetch("processor_rows") })
    assert_equal([[2, 2, 2, 2, 2, 2]] * 4, successful.map { |item| item.fetch("generated_tokens_by_row") })
    assert_equal([6.0, 6.0, 6.0, 6.0], successful.map { |item| item.fetch("padded_audio_seconds") })
    assert(successful.all? { |item| item.fetch("generation_call_wall_seconds").positive? })
  end

  def test_native_phase_metrics_fill_existing_profile_fields
    controller = controller(size: 2)
    metrics = {
      feature_wall_seconds: 0.10,
      mel_pack_seconds: 0.01,
      encoder_input_seconds: 0.02,
      encoder_compute_seconds: 0.50,
      decoder_total_seconds: 1.20,
      render_seconds: 0.03
    }.freeze
    executor = ASR::BatchExecutor.new(controller, operation_metrics: -> { metrics })

    result = executor.execute(
      [{ duration: 1.0 }, { duration: 1.0 }],
      max_new_tokens: 445,
      base_max_new_tokens: 445
    ) { [Result.new(generated_tokens: 2), Result.new(generated_tokens: 3)] }
    batch = result.telemetry.batch_history.fetch(0)

    assert_in_delta 0.11, batch.fetch("prepare_seconds")
    assert_in_delta 0.02, batch.fetch("h2d_seconds")
    assert_in_delta 1.70, batch.fetch("generate_device_seconds")
    assert_in_delta 0.03, batch.fetch("generation_analysis_seconds")
    assert_in_delta 0.11, result.telemetry.feature_worker_seconds
    assert_in_delta 1.70, result.telemetry.generate_device_seconds
  end

  def test_high_token_oom_uses_retry_local_cap_without_changing_base_controller
    controller = controller(size: 24)
    attempts = []
    rows = Array.new(24) { |index| { index: index, duration: 1.0 } }

    result = ASR::BatchExecutor.new(controller).execute(
      rows,
      max_new_tokens: 896,
      base_max_new_tokens: 445
    ) do |batch|
      attempts << batch.length
      raise ASR::ExecutionError.new("high-token OOM", failure_kind: :oom) if batch.length > 6

      batch.map { Result.new(generated_tokens: 1) }
    end

    assert result.success?
    assert_equal [24, 12, 6, 6, 6, 6], attempts
    assert_equal [24, 24], [controller.current_size, controller.max_size]
    assert_equal 2, result.telemetry.oom_retries
  end

  def test_data_local_failures_are_isolated_without_poisoning_the_circuit
    controller = controller(size: 5)
    attempts = []
    rows = Array.new(5) { |index| { index: index, duration: 1.0 } }

    result = ASR::BatchExecutor.new(controller).execute(
      rows,
      max_new_tokens: 445,
      base_max_new_tokens: 445
    ) do |batch|
      indices = batch.map { |row| row.fetch(:index) }
      attempts << indices
      raise "malformed sample payload" if indices.any? { |index| index < 3 }

      batch.map { Result.new(generated_tokens: 1) }
    end

    assert_equal([RuntimeError, RuntimeError, RuntimeError, nil, nil], result.errors.map { |error| error&.class })
    assert_nil result.values[0]
    assert_equal 1, result.values.fetch(4).generated_tokens
    assert_includes attempts, [3, 4]
    refute controller.circuit_open?
    assert_equal 0, result.telemetry.oom_retries
  end

  def test_fatal_failure_does_not_bisect_and_blocks_later_work
    controller = controller(size: 8)
    calls = 0
    rows = Array.new(8) { |index| { index: index, duration: 1.0 } }
    executor = ASR::BatchExecutor.new(controller)

    first = executor.execute(rows, max_new_tokens: 445, base_max_new_tokens: 445) do |_batch|
      calls += 1
      raise TypeError, "incompatible model API"
    end
    second = executor.execute(rows.first(2), max_new_tokens: 445, base_max_new_tokens: 445) do |_batch|
      calls += 1
      []
    end

    assert_equal 1, calls
    assert(first.errors.all?(TypeError))
    assert(second.errors.all?(ASR::CircuitOpenError))
    assert controller.circuit_open?
    assert_match(/TypeError: incompatible model API/, controller.circuit_breaker_error)
  end

  def test_wrong_result_cardinality_is_an_invariant_and_opens_the_circuit
    controller = controller(size: 2)
    result = ASR::BatchExecutor.new(controller).execute(
      [{ duration: 1.0 }, { duration: 1.0 }],
      max_new_tokens: 445,
      base_max_new_tokens: 445
    ) { [Result.new(generated_tokens: 1)] }

    assert(result.errors.all?(ASR::ExecutionError))
    assert controller.circuit_open?
    assert_equal 0, result.telemetry.asr_batches
  end

  def test_interrupt_is_never_reclassified_or_retried
    controller = controller(size: 2)
    assert_raises(Interrupt) do
      ASR::BatchExecutor.new(controller).execute(
        [{ duration: 1.0 }, { duration: 1.0 }],
        max_new_tokens: 445,
        base_max_new_tokens: 445
      ) { raise Interrupt }
    end
  end

  private

  def controller(size:)
    ASR::BatchController.new(
      current_size: size,
      max_size: size,
      initial_size: size,
      audio_budget_seconds: size * 30.0,
      adaptive: false,
      target_memory_ratio: 0.9
    )
  end

  def options(**changes)
    Cohere::Transcribe::TranscriptionOptions.new(
      vad: "none",
      alignment: "none",
      **changes
    )
  end
end
