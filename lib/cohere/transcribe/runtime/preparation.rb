# frozen_string_literal: true

require "etc"

require_relative "../audio/ffmpeg_native"

module Cohere
  module Transcribe
    module Runtime
      module Preparation
        MAX_PIPELINE_GROUP_BYTES = 512 * (1024**2)
        MAX_GROUP_JOBS = 128
        ESTIMATE_HEADROOM = 0.05

        # The result of decode and VAD preparation. Timings are accumulated by
        # the engine on the consuming thread, keeping statistics deterministic.
        PreparedEntry = Data.define(
          :item,
          :snapshot,
          :decoded,
          :duration,
          :segment_times,
          :speech_spans,
          :vad_details,
          :decode_seconds,
          :vad_seconds,
          :error
        )

        # Bounded ordered preparation with at most one group ahead of ASR.
        #
        # A pipelined group receives at most half of the configured decoded-PCM
        # budget (and never more than 512 MiB). Estimated decode sizes determine
        # group membership and per-file ceilings. Unknown sizes receive an equal
        # share initially. A ceiling failure first retries within the remaining
        # configured PCM budget while successful group entries stay retained. If
        # that is insufficient, later retained audio is released and only those
        # entries are prepared again after the full-ceiling retry. A known file
        # that cannot fit the group cap starts with that exclusive path.
        # Native decoder implementation transients are outside this retained-PCM
        # accounting, as they are in the Python path.
        class Pipeline
          include Enumerable

          Group = Data.define(:items, :limits, :exclusive)

          # Explicit cursor keeps metadata probes on the preparation caller's
          # native stack while preserving lazy current/next-group discovery.
          class GroupCursor
            def initialize(items:, workers:, group_byte_limit:, memory_byte_limit:,
                           estimated_reservation:, unknown_reservation:, bounded_group:)
              @items = items
              @workers = workers
              @group_byte_limit = group_byte_limit
              @memory_byte_limit = memory_byte_limit
              @estimated_reservation = estimated_reservation
              @unknown_reservation = unknown_reservation
              @bounded_group = bounded_group
              @index = 0
              @pending = nil
            end

            def next
              items = []
              estimates = []
              total = 0
              while items.length < @workers
                pair = take
                break unless pair

                item, estimate = pair
                if estimate > @group_byte_limit
                  if items.empty?
                    return Group.new(
                      items: [item].freeze,
                      limits: [@memory_byte_limit].freeze,
                      exclusive: true
                    )
                  end
                  @pending = pair
                  break
                end
                if !items.empty? && total + estimate > @group_byte_limit
                  @pending = pair
                  break
                end

                items << item
                estimates << estimate
                total += estimate
              end
              raise StopIteration if items.empty?

              @bounded_group.call(items, estimates)
            end

            private

            def take
              if @pending
                pair = @pending
                @pending = nil
                return pair
              end
              return if @index >= @items.length

              item = @items.fetch(@index)
              @index += 1
              estimate = @estimated_reservation.call(item) || @unknown_reservation
              [item, estimate]
            end
          end
          private_constant :Group, :GroupCursor

          class WorkerPool
            STOP = Object.new.freeze

            def initialize(size, &prepare)
              @prepare = prepare
              @queues = Array.new(size) { Queue.new }
              @threads = @queues.each_with_index.map do |queue, slot|
                Thread.new do
                  Thread.current.name = "cohere-audio-prep-#{slot}" if Thread.current.respond_to?(:name=)
                  loop do
                    task = queue.pop
                    break if task.equal?(STOP)

                    item, limit, response = task
                    begin
                      response << [:ok, @prepare.call(item, limit, slot)]
                    rescue Exception => e # rubocop:disable Lint/RescueException -- transport fatal worker errors
                      response << [:error, e]
                    end
                  end
                end.tap { |thread| thread.report_on_exception = false }
              end
            end

            def prepare(group)
              responses = group.items.each_with_index.map do |item, slot|
                response = Queue.new
                @queues.fetch(slot) << [item, group.limits.fetch(slot), response]
                response
              end
              responses.map do |response|
                status, value = response.pop
                raise value if status == :error

                value
              end
            end

            def close(cancel:)
              if cancel
                Audio::FFmpegNative.cancel_active!
                @threads.each { |thread| thread.kill if thread.alive? }
              else
                @queues.each { |queue| queue << STOP }
              end
              @threads.each(&:join)
            end
          end
          private_constant :WorkerPool

          attr_reader :effective_workers, :group_byte_limit, :memory_byte_limit, :wait_seconds

          def initialize(items, memory_byte_limit:, requested_workers:, enabled:, worker_limit: nil,
                         estimate_bytes: nil, exclusive_retry: nil, retained_bytes: nil, &prepare)
            raise ArgumentError, "prepare block is required" unless prepare

            @items = items.to_a.freeze
            @memory_byte_limit = Integer(memory_byte_limit)
            raise ArgumentError, "memory_byte_limit must be positive" unless @memory_byte_limit.positive?

            @prepare = prepare
            @estimate_bytes = estimate_bytes
            @exclusive_retry = exclusive_retry
            @retained_bytes = retained_bytes
            @wait_seconds = 0.0
            @enabled = enabled && @items.length > 1
            @worker_limit = worker_limit.nil? ? nil : Integer(worker_limit)
            raise ArgumentError, "worker_limit must be positive" if @worker_limit && !@worker_limit.positive?

            @group_byte_limit = if @enabled
                                  [[@memory_byte_limit / 2, 1].max, MAX_PIPELINE_GROUP_BYTES].min
                                else
                                  @memory_byte_limit
                                end
            @effective_workers = resolve_workers(requested_workers)
          end

          def pipelined?
            @enabled
          end

          def each(&block)
            return enum_for(__method__) unless block
            return if @items.empty?

            unless pipelined?
              @items.each { |item| block.call(@prepare.call(item, @memory_byte_limit, 0)) }
              return
            end

            groups = build_groups
            pool = WorkerPool.new(@effective_workers, &@prepare)
            group = groups.next
            group_index = 0
            pending = submit(group, group_index, pool)
            loop do
              prepared = resolve(pending)
              pending = nil
              next_group = next_group(groups)
              if retry_exclusively?(group, prepared)
                consume_with_exclusive_retry(group, prepared, pool, &block)
              else
                pending = submit(next_group, group_index + 1, pool) if next_group && !group.exclusive && !next_group.exclusive
                consume(prepared, &block)
              end
              break unless next_group

              pending ||= submit(next_group, group_index + 1, pool) if next_group
              group = next_group
              group_index += 1
            end
            completed = true
          ensure
            cancel(pending) if defined?(pending) && pending
            pool&.close(cancel: !completed) if defined?(pool) && pool
          end

          private

          def resolve_workers(requested)
            return 1 if @items.empty?

            processors = Etc.nprocessors
            processors = 1 unless processors.is_a?(Integer) && processors.positive?
            automatic = if @items.length == 1
                          1
                        else
                          [2, @items.length, [processors / 2, 1].max].min
                        end
            value = requested.nil? ? automatic : Integer(requested)
            raise ArgumentError, "requested_workers must be positive" unless value.positive?

            return 1 unless @enabled

            limits = [value, @items.length, processors, MAX_GROUP_JOBS, @group_byte_limit]
            limits << @worker_limit if @worker_limit
            limits.min
          rescue SystemCallError
            value = requested.nil? ? 2 : Integer(requested)
            limits = [value, @items.length, MAX_GROUP_JOBS, @group_byte_limit]
            limits << @worker_limit if @worker_limit
            limits.min
          end

          def submit(group, group_index, pool)
            Thread.new do
              Thread.current.name = "cohere-audio-group-#{group_index}" if Thread.current.respond_to?(:name=)
              pool.prepare(group)
            end.tap { |thread| thread.report_on_exception = false }
          end

          def resolve(thread)
            started = monotonic
            thread.value
          ensure
            @wait_seconds += monotonic - started if started
          end

          def next_group(groups)
            groups.next
          rescue StopIteration
            nil
          end

          def retry_exclusively?(group, prepared)
            return false if group.exclusive || !@exclusive_retry

            prepared.any? { |entry| @exclusive_retry.call(entry) }
          end

          def consume_with_exclusive_retry(group, prepared, pool, &block)
            released = Array.new(prepared.length, false)
            prepared.each_index do |index|
              entry = prepared.fetch(index)
              if entry.nil? && released.fetch(index)
                consume_single_retry(group.items.fetch(index), @memory_byte_limit, pool, &block)
                next
              end
              unless @exclusive_retry.call(entry)
                begin
                  block.call(entry)
                ensure
                  prepared[index] = nil
                end
                next
              end

              prepared[index] = nil
              collect_released_audio
              unless @retained_bytes
                release_retained_entries(prepared, released)
                collect_released_audio
              end
              limit = available_retry_limit(prepared)
              retried = prepare_single(group.items.fetch(index), limit, pool)
              if limit < @memory_byte_limit && @exclusive_retry.call(retried.fetch(0))
                retried.clear
                release_retained_entries(prepared, released)
                collect_released_audio
                retried = prepare_single(group.items.fetch(index), @memory_byte_limit, pool)
              end
              consume_single_result(retried, &block)
            end
          ensure
            release(prepared)
          end

          def consume_single_retry(item, limit, pool, &)
            consume_single_result(prepare_single(item, limit, pool), &)
          end

          def prepare_single(item, limit, pool)
            pool.prepare(
              Group.new(
                items: [item].freeze,
                limits: [limit].freeze,
                exclusive: true
              )
            )
          end

          def consume_single_result(retained)
            yield retained.fetch(0)
          ensure
            retained&.clear
            collect_released_audio
          end

          def available_retry_limit(prepared)
            return @memory_byte_limit unless @retained_bytes

            retained = prepared.compact.sum { |entry| retained_bytes(entry) }
            [@memory_byte_limit - retained, 1].max
          end

          def release_retained_entries(prepared, released)
            unless @retained_bytes
              prepared.each_index do |index|
                next unless prepared[index]

                prepared[index] = nil
                released[index] = true
              end
              return
            end

            prepared.each_index do |index|
              entry = prepared[index]
              next unless entry && retained_bytes(entry).positive?

              prepared[index] = nil
              released[index] = true
            end
          end

          def retained_bytes(entry)
            bytes = Integer(@retained_bytes.call(entry))
            raise ArgumentError, "retained_bytes must return a non-negative integer" if bytes.negative?

            bytes
          end

          def build_groups
            return equal_groups.each unless @estimate_bytes

            GroupCursor.new(
              items: @items,
              workers: @effective_workers,
              group_byte_limit: @group_byte_limit,
              memory_byte_limit: @memory_byte_limit,
              estimated_reservation: method(:estimated_reservation),
              unknown_reservation: unknown_reservation,
              bounded_group: method(:bounded_group)
            )
          end

          def equal_groups
            @items.each_slice(@effective_workers).map do |items|
              limit = [@group_byte_limit / items.length, 1].max
              Group.new(items: items.freeze, limits: Array.new(items.length, limit).freeze, exclusive: false)
            end
          end

          def estimated_reservation(item)
            value = @estimate_bytes.call(item)
            return if value.nil?

            bytes = Integer(value)
            return unless bytes.positive?

            [(bytes * (1.0 + ESTIMATE_HEADROOM)).ceil, 1].max
          rescue ArgumentError, TypeError, SystemCallError
            nil
          end

          def unknown_reservation
            [@group_byte_limit / @effective_workers, 1].max
          end

          def bounded_group(items, estimates)
            limits = estimates.dup
            remaining = @group_byte_limit - limits.sum
            if remaining.positive?
              weight = limits.sum
              extras = limits.map { |limit| (remaining * limit).div(weight) }
              leftover = remaining - extras.sum
              extras[limits.each_index.max_by { |index| limits.fetch(index) }] += leftover
              limits = limits.each_index.map { |index| limits.fetch(index) + extras.fetch(index) }
            end
            Group.new(items: items.freeze, limits: limits.freeze, exclusive: false)
          end

          def consume(prepared, &block)
            prepared.each_index do |index|
              entry = prepared.fetch(index)
              begin
                block.call(entry)
              ensure
                prepared[index] = nil
              end
            end
          ensure
            release(prepared)
          end

          def release(prepared)
            prepared&.clear
            collect_released_audio
          end

          def collect_released_audio
            GC.start(full_mark: false, immediate_mark: true, immediate_sweep: true)
          end

          def cancel(thread)
            Audio::FFmpegNative.cancel_active!
            thread.kill if thread.alive?
            thread.join
          rescue Exception # rubocop:disable Lint/RescueException -- preserve the caller's active exception
            nil
          end

          def monotonic
            Process.clock_gettime(Process::CLOCK_MONOTONIC)
          end
        end
      end
    end
  end
end
