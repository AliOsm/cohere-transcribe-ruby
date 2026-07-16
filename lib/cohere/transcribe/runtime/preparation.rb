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
        # group membership and per-file ceilings. A file that cannot fit that cap
        # is prepared alone with the full configured ceiling and without overlap.
        # Native decoder implementation transients are outside this retained-PCM
        # accounting, as they are in the Python path.
        class Pipeline
          include Enumerable

          Group = Data.define(:items, :limits, :exclusive)
          private_constant :Group

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
                         estimate_bytes: nil, &prepare)
            raise ArgumentError, "prepare block is required" unless prepare

            @items = items.to_a.freeze
            @memory_byte_limit = Integer(memory_byte_limit)
            raise ArgumentError, "memory_byte_limit must be positive" unless @memory_byte_limit.positive?

            @prepare = prepare
            @estimate_bytes = estimate_bytes
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
            pending = submit(groups.first, 0, pool)
            groups.each_with_index do |group, group_index|
              prepared = resolve(pending)
              pending = nil
              next_group = groups[group_index + 1]
              pending = submit(next_group, group_index + 1, pool) if next_group && !group.exclusive && !next_group.exclusive
              consume(prepared, &block)
              pending ||= submit(next_group, group_index + 1, pool) if next_group
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
              prepare_group(group, pool)
            end.tap { |thread| thread.report_on_exception = false }
          end

          def resolve(thread)
            started = monotonic
            thread.value
          ensure
            @wait_seconds += monotonic - started if started
          end

          def prepare_group(group, pool)
            pool.prepare(group)
          end

          def build_groups
            return equal_groups unless @estimate_bytes

            groups = []
            items = []
            estimates = []
            @items.each do |item|
              estimate = estimated_reservation(item)
              if estimate.nil? || estimate > @group_byte_limit
                groups << bounded_group(items, estimates) unless items.empty?
                groups << Group.new(items: [item].freeze, limits: [@memory_byte_limit].freeze, exclusive: true)
                items = []
                estimates = []
                next
              end

              if items.length >= @effective_workers || estimates.sum + estimate > @group_byte_limit
                groups << bounded_group(items, estimates)
                items = []
                estimates = []
              end
              items << item
              estimates << estimate
            end
            groups << bounded_group(items, estimates) unless items.empty?
            groups
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
            prepared&.clear
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
