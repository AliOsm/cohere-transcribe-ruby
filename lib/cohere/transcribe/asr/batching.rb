# frozen_string_literal: true

module Cohere
  module Transcribe
    module ASR
      # Mutable, run-local telemetry for native ASR batching. The public
      # statistics object intentionally remains immutable; Engine copies these
      # counters into it after a group completes.
      class BatchTelemetry
        attr_accessor :asr_batches,
                      :processor_rows,
                      :generated_tokens,
                      :oom_retries,
                      :effective_batch_min,
                      :effective_batch_max,
                      :final_batch_size,
                      :final_batch_cap,
                      :feature_worker_seconds,
                      :h2d_seconds,
                      :generate_device_seconds,
                      :generation_analysis_seconds
        attr_reader :batch_history

        def initialize
          @asr_batches = 0
          @processor_rows = 0
          @generated_tokens = 0
          @oom_retries = 0
          @effective_batch_min = 0
          @effective_batch_max = 0
          @final_batch_size = 0
          @final_batch_cap = 0
          @feature_worker_seconds = 0.0
          @h2d_seconds = 0.0
          @generate_device_seconds = 0.0
          @generation_analysis_seconds = 0.0
          @batch_history = []
        end

        def record_success(rows, max_new_tokens:, generated_tokens: 0, generated_tokens_by_row: [],
                           generation_call_wall_seconds: nil, padded_audio_seconds: nil,
                           native_metrics: nil)
          prepare_seconds = native_metrics && (
            native_metrics.fetch(:feature_wall_seconds, 0.0) +
            native_metrics.fetch(:mel_pack_seconds, 0.0)
          )
          h2d_seconds = native_metrics&.fetch(:encoder_input_seconds, nil)
          generate_device_seconds = native_metrics && (
            native_metrics.fetch(:encoder_compute_seconds, 0.0) +
            native_metrics.fetch(:decoder_total_seconds, 0.0)
          )
          generation_analysis_seconds = native_metrics&.fetch(:render_seconds, nil)
          @asr_batches += 1
          @processor_rows += rows
          @generated_tokens += generated_tokens
          @effective_batch_min = @effective_batch_min.zero? ? rows : [@effective_batch_min, rows].min
          @effective_batch_max = [@effective_batch_max, rows].max
          @feature_worker_seconds += prepare_seconds.to_f
          @h2d_seconds += h2d_seconds.to_f
          @generate_device_seconds += generate_device_seconds.to_f
          @generation_analysis_seconds += generation_analysis_seconds.to_f
          @batch_history << {
            "segments" => rows,
            "processor_rows" => rows,
            "max_new_tokens" => max_new_tokens,
            "generated_tokens" => generated_tokens,
            "generated_tokens_by_row" => generated_tokens_by_row.freeze,
            "prepare_seconds" => prepare_seconds,
            "h2d_seconds" => h2d_seconds,
            "generation_call_wall_seconds" => generation_call_wall_seconds,
            "generate_device_seconds" => generate_device_seconds,
            "generation_analysis_seconds" => generation_analysis_seconds,
            "padded_audio_seconds" => padded_audio_seconds
          }.freeze
        end

        def record_oom(rows, max_new_tokens:)
          @oom_retries += 1
          @batch_history << {
            "event" => "oom",
            "segments" => rows,
            "max_new_tokens" => max_new_tokens
          }.freeze
        end

        def finalize(controller)
          @final_batch_size = controller.current_size
          @final_batch_cap = controller.max_size
          self
        end
      end

      # Persistent batch-size controller. OOM learning and fatal circuit state
      # live with the retained native session, not with one audio file.
      class BatchController
        DEVICE_DEFAULTS = { "cuda" => 24, "mps" => 8, "cpu" => 4 }.freeze

        attr_reader :adaptive,
                    :audio_budget_seconds,
                    :initial_size,
                    :max_size,
                    :memory_budget_bytes,
                    :target_memory_ratio,
                    :total_memory_bytes
        attr_accessor :circuit_breaker_error, :current_size, :growth_cooldown

        def self.create(options, device:, durations:, memory: nil, physical_max: nil)
          default_initial = DEVICE_DEFAULTS.fetch(device.to_s, DEVICE_DEFAULTS.fetch("cpu"))
          requested_initial = options.batch_size || if options.batch_max_size
                                                      [default_initial, options.batch_max_size].min
                                                    else
                                                      default_initial
                                                    end
          initial = physical_max ? [requested_initial, physical_max].min : requested_initial
          free_bytes, total_bytes = memory || [0, 0]
          free_bytes = Integer(free_bytes)
          total_bytes = Integer(total_bytes)
          requested_maximum = if !options.adaptive_batch
                                requested_initial
                              elsif options.batch_max_size
                                options.batch_max_size
                              elsif options.batch_size
                                options.batch_size
                              elsif device.to_s == "cuda"
                                total_gib = total_bytes.fdiv(1024**3)
                                [requested_initial, [128, (total_gib * 4).to_i].min].max
                              else
                                default_initial
                              end
          maximum = [requested_initial, requested_maximum].max
          maximum = [maximum, physical_max].min if physical_max
          maximum = [initial, maximum].max
          longest = durations.map { |duration| Float(duration) }.max || 1.0
          audio_budget = options.batch_audio_seconds || (initial * [longest, 0.25].max)

          used_bytes = [total_bytes - free_bytes, 0].max
          memory_budget = if device.to_s == "cuda" && total_bytes.positive?
                            [
                              (options.batch_vram_target * total_bytes).to_i,
                              used_bytes + (0.95 * free_bytes).to_i
                            ].min
                          else
                            0
                          end

          new(
            current_size: initial,
            max_size: maximum,
            initial_size: initial,
            audio_budget_seconds: audio_budget,
            adaptive: options.adaptive_batch,
            target_memory_ratio: options.batch_vram_target,
            total_memory_bytes: total_bytes,
            memory_budget_bytes: memory_budget
          )
        end

        def initialize(current_size:, max_size:, initial_size:, audio_budget_seconds:, adaptive:,
                       target_memory_ratio:, total_memory_bytes: 0, memory_budget_bytes: 0)
          @current_size = positive_integer(current_size, "current_size")
          @max_size = positive_integer(max_size, "max_size")
          @initial_size = positive_integer(initial_size, "initial_size")
          raise ArgumentError, "max_size must be at least current_size" if @max_size < @current_size

          @audio_budget_seconds = Float(audio_budget_seconds)
          raise ArgumentError, "audio_budget_seconds must be positive" unless @audio_budget_seconds.positive?

          raise ArgumentError, "adaptive must be a boolean" unless [true, false].include?(adaptive)

          @adaptive = adaptive
          @target_memory_ratio = Float(target_memory_ratio)
          @total_memory_bytes = Integer(total_memory_bytes)
          @memory_budget_bytes = Integer(memory_budget_bytes)
          @growth_cooldown = 0
          @circuit_breaker_error = nil
        end

        def configure_group(options, durations)
          longest = durations.map { |duration| Float(duration) }.max || 1.0
          @audio_budget_seconds = options.batch_audio_seconds || (@initial_size * [longest, 0.25].max)
          self
        end

        def take_count(rows)
          return 0 if rows.empty?

          longest = [yield(rows.first), 1.0 / SAMPLE_RATE].max
          frame_limited = [(@audio_budget_seconds / longest).floor, 1].max
          [rows.length, @current_size, frame_limited].min
        end

        def record_oom(attempted_rows)
          rows = positive_integer(attempted_rows, "attempted_rows")
          @growth_cooldown = [@growth_cooldown, 2].max
          if rows <= 1
            @current_size = 1
            return self
          end

          @max_size = [@max_size, rows - 1].min
          @current_size = [1, [@max_size, rows / 2].min].max
          self
        end

        def record_success(attempted_rows, baseline_reserved_bytes: 0, peak_reserved_bytes: 0)
          rows = positive_integer(attempted_rows, "attempted_rows")
          if @growth_cooldown.positive?
            @growth_cooldown -= 1
            return self
          end
          return self unless @adaptive && @current_size < @max_size && rows >= @current_size
          return self unless @total_memory_bytes.positive? && peak_reserved_bytes.positive?

          budget = @memory_budget_bytes.positive? ? @memory_budget_bytes : (@target_memory_ratio * @total_memory_bytes).to_i
          headroom = (budget - peak_reserved_bytes).fdiv(@total_memory_bytes)
          return self if headroom <= 0.05

          factor = headroom >= 0.20 ? 1.25 : 1.125
          proposed = [@current_size + 1, (@current_size * factor).ceil].max
          incremental = [1, peak_reserved_bytes - baseline_reserved_bytes].max
          available = [0, budget - baseline_reserved_bytes].max
          memory_estimate = (rows * available).div(incremental)
          proposed = [proposed, [@current_size, memory_estimate].max].min if memory_estimate.positive?
          @current_size = [@max_size, proposed].min
          self
        end

        def open_circuit(error)
          return @circuit_breaker_error if @circuit_breaker_error

          @circuit_breaker_error = FailurePolicy.fingerprint(error)
        end

        def circuit_open?
          !@circuit_breaker_error.nil?
        end

        private

        def positive_integer(value, field)
          integer = Integer(value)
          raise ArgumentError, "#{field} must be positive" unless integer.positive?

          integer
        end
      end

      # A batch cap scoped to higher-token truncation retries. An OOM here must
      # not reduce the ordinary ASR controller learned from the base limit.
      class RetryBatchCap
        attr_reader :current_size

        def initialize(current_size)
          @current_size = Integer(current_size)
          raise ArgumentError, "current_size must be positive" unless @current_size.positive?
        end

        def record_oom(attempted_rows)
          rows = Integer(attempted_rows)
          proposed = rows > 1 ? rows / 2 : 1
          @current_size = [1, [@current_size, proposed].min].max
          self
        end
      end

      BatchExecutionResult = Data.define(:values, :errors, :telemetry) do
        def success?
          errors.all?(&:nil?)
        end
      end

      # Generic recursive recovery around a native batch operation. Results
      # remain in input order, data-local failures are isolated, fatal failures
      # open a persistent circuit, and learned caps are honored by siblings that
      # have not yet been attempted.
      class BatchExecutor
        attr_reader :telemetry

        def initialize(controller, duration: nil, memory: nil, operation_metrics: nil, telemetry: BatchTelemetry.new)
          @controller = controller
          @duration = duration || method(:default_duration)
          @memory = memory
          @operation_metrics = operation_metrics
          @telemetry = telemetry
        end

        def execute(rows, max_new_tokens:, base_max_new_tokens:, retry_cap: nil, &operation)
          raise ArgumentError, "a native batch operation block is required" unless operation

          source = rows.to_a
          values = Array.new(source.length)
          errors = Array.new(source.length)
          indexed = source.each_with_index.map { |row, index| [index, row] }
          high_token_retry = max_new_tokens > base_max_new_tokens
          retry_cap ||= RetryBatchCap.new([source.length, 1].max) if high_token_retry
          process(
            indexed,
            values,
            errors,
            max_new_tokens: max_new_tokens,
            high_token_retry: high_token_retry,
            retry_cap: retry_cap,
            operation: operation
          )
          @telemetry.finalize(@controller)
          BatchExecutionResult.new(values: values.freeze, errors: errors.freeze, telemetry: @telemetry)
        end

        private

        def process(indexed, values, errors, max_new_tokens:, high_token_retry:, retry_cap:, operation:)
          return if indexed.empty?

          if @controller.circuit_open?
            error = CircuitOpenError.new(@controller.circuit_breaker_error)
            indexed.each { |pair| errors[pair.first] = error }
            return
          end

          effective_cap = high_token_retry ? retry_cap.current_size : @controller.current_size
          if indexed.length > effective_cap
            indexed.each_slice([effective_cap, 1].max) do |slice|
              process(
                slice,
                values,
                errors,
                max_new_tokens: max_new_tokens,
                high_token_retry: high_token_retry,
                retry_cap: retry_cap,
                operation: operation
              )
            end
            return
          end

          rows = indexed.map(&:last)
          memory_before = memory_snapshot
          operation_started = monotonic
          output = operation.call(rows)
          operation_seconds = monotonic - operation_started
          native_metrics = @operation_metrics&.call
          memory_after = memory_snapshot
          unless output.is_a?(Array) && output.length == rows.length
            raise ExecutionError.new(
              "Native Cohere batch returned #{output.respond_to?(:length) ? output.length : "a non-array result"} " \
              "for #{rows.length} inputs",
              failure_kind: :fatal
            )
          end

          indexed.each_with_index { |(index, _row), lane| values[index] = output.fetch(lane) }
          generated_tokens_by_row = output.map do |value|
            value.respond_to?(:generated_tokens) ? Integer(value.generated_tokens) : 0
          end
          generated_tokens = generated_tokens_by_row.sum
          @telemetry.record_success(
            rows.length,
            max_new_tokens: max_new_tokens,
            generated_tokens: generated_tokens,
            generated_tokens_by_row: generated_tokens_by_row,
            generation_call_wall_seconds: operation_seconds,
            padded_audio_seconds: rows.length * rows.map { |row| Float(@duration.call(row)) }.max.to_f,
            native_metrics: native_metrics
          )
          baseline_reserved, peak_reserved = reserved_memory(memory_before, memory_after)
          @controller.record_success(
            rows.length,
            baseline_reserved_bytes: baseline_reserved,
            peak_reserved_bytes: peak_reserved
          )
        rescue NoMemoryError, SystemStackError, ScriptError, StandardError => e
          handle_failure(
            e,
            indexed,
            values,
            errors,
            max_new_tokens: max_new_tokens,
            high_token_retry: high_token_retry,
            retry_cap: retry_cap,
            operation: operation
          )
        end

        def handle_failure(error, indexed, values, errors, max_new_tokens:, high_token_retry:, retry_cap:, operation:)
          kind = FailurePolicy.classify(error)
          if kind == :fatal
            @controller.open_circuit(error)
            indexed.each { |pair| errors[pair.first] = error }
            return
          end

          if kind == :oom
            @telemetry.record_oom(indexed.length, max_new_tokens: max_new_tokens)
            if high_token_retry
              retry_cap.record_oom(indexed.length)
            else
              @controller.record_oom(indexed.length)
            end
          end

          if indexed.length == 1
            errors[indexed.first.first] = error
            return
          end

          split = kind == :oom ? balanced_split(indexed.map(&:last)) : indexed.length / 2
          [indexed.first(split), indexed.drop(split)].each do |half|
            process(
              half,
              values,
              errors,
              max_new_tokens: max_new_tokens,
              high_token_retry: high_token_retry,
              retry_cap: retry_cap,
              operation: operation
            )
          end
        end

        def balanced_split(rows)
          return 1 if rows.length < 2

          first_duration = Float(@duration.call(rows.first))
          best_index = rows.length / 2
          best_cost = Float::INFINITY
          (1...rows.length).each do |index|
            left_cost = index * first_duration
            right_cost = (rows.length - index) * Float(@duration.call(rows.fetch(index)))
            cost = [left_cost, right_cost].max
            if cost < best_cost
              best_cost = cost
              best_index = index
            end
          end
          best_index
        end

        def default_duration(row)
          return row.fetch(:duration) if row.respond_to?(:fetch)
          return row.duration if row.respond_to?(:duration)

          1.0
        end

        def memory_snapshot
          return nil unless @memory

          free_bytes, total_bytes = @memory.call
          free_bytes = Integer(free_bytes)
          total_bytes = Integer(total_bytes)
          return nil unless free_bytes >= 0 && total_bytes.positive? && free_bytes <= total_bytes

          [free_bytes, total_bytes]
        rescue StandardError
          nil
        end

        def reserved_memory(before, after)
          samples = [before, after].compact
          return [0, 0] if samples.empty?

          used = samples.map { |free_bytes, total_bytes| total_bytes - free_bytes }
          [used.first, used.max]
        end

        def monotonic
          Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end
      end
    end
  end
end
