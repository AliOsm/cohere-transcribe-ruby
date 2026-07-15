# frozen_string_literal: true

module Cohere
  module Transcribe
    module Runtime
      # Scheduling primitives for word-aligned runs. The coordinator creates a
      # hard phase barrier: every ASR outcome is collected before Dense is
      # evicted, and no alignment callback can run before that eviction.
      module WordPipeline
        UNSET = Object.new.freeze
        private_constant :UNSET

        # The complete state needed after ASR. Deliberately absent are decoded
        # samples (or a decoder result): multi-file word mode must re-decode one
        # source at a time instead of retaining every waveform until alignment.
        AlignmentWork = Data.define(
          :index,
          :entry,
          :plan,
          :result,
          :segment_times,
          :speech_spans,
          :generation_id,
          :decode_backend,
          :expected_sample_count,
          :audio_required,
          :source_snapshot
        ) do
          def initialize(index:, entry:, plan:, result:, segment_times:, speech_spans:,
                         generation_id:, decode_backend:, expected_sample_count:, audio_required:,
                         source_snapshot:)
            index = Integer(index)
            expected_sample_count = Integer(expected_sample_count)
            raise ArgumentError, "word-pipeline index must be nonnegative" if index.negative?
            raise ArgumentError, "expected alignment sample count must be nonnegative" if expected_sample_count.negative?

            super(
              index: index,
              entry: entry,
              plan: plan,
              result: result,
              segment_times: immutable_spans(segment_times, "segment_times"),
              speech_spans: immutable_spans(speech_spans, "speech_spans"),
              generation_id: immutable_optional_string(generation_id),
              decode_backend: immutable_optional_string(decode_backend),
              expected_sample_count: expected_sample_count,
              audio_required: !!audio_required,
              source_snapshot: immutable_snapshot(source_snapshot)
            )
          end

          private

          def immutable_spans(value, label)
            raise ArgumentError, "#{label} must be an array" unless value.is_a?(Array)

            value.map do |span|
              raise ArgumentError, "#{label} contains an invalid span" unless
                span.is_a?(Array) && span.length == 2 && span.all?(Numeric)

              [span.fetch(0).to_f, span.fetch(1).to_f].freeze
            end.freeze
          end

          def immutable_optional_string(value)
            return nil if value.nil?
            raise ArgumentError, "word-pipeline metadata strings must be strings" unless value.is_a?(String)

            value.dup.freeze
          end

          def immutable_snapshot(value)
            value.is_a?(Array) ? value.dup.freeze : value
          end
        end

        Final = Data.define(:index, :result) do
          def initialize(index:, result:)
            index = Integer(index)
            raise ArgumentError, "word-pipeline index must be nonnegative" if index.negative?

            super
          end
        end

        ReloadResult = Data.define(:decoded, :error) do
          def ok?
            error.nil?
          end
        end

        # Executes an ordered two-phase batch. The ASR callback returns either
        # AlignmentWork or Final. Alignment errors intended to be per-file must
        # likewise be converted to a final result by the alignment callback;
        # fatal/cancellation exceptions intentionally cross this boundary.
        class Coordinator
          def initialize(total:, evict_asr:, align:, fixed_results: {}, completed: nil, gc_interval: 8,
                         reload: nil, memory_byte_limit: nil, start_alignment: nil)
            @total = Integer(total)
            raise ArgumentError, "word-pipeline total must be nonnegative" if @total.negative?
            raise ArgumentError, "evict_asr must respond to call" unless evict_asr.respond_to?(:call)
            raise ArgumentError, "align must respond to call" unless align.respond_to?(:call)
            raise ArgumentError, "completed must respond to call" if completed && !completed.respond_to?(:call)
            raise ArgumentError, "start_alignment must respond to call" if
              start_alignment && !start_alignment.respond_to?(:call)

            @fixed_results = fixed_results.to_h.freeze
            @evict_asr = evict_asr
            @align = align
            @completed = completed
            @gc_interval = Integer(gc_interval)
            raise ArgumentError, "gc_interval must be positive" unless @gc_interval.positive?
            raise ArgumentError, "reload and memory_byte_limit must be provided together" unless
              reload.nil? == memory_byte_limit.nil?
            raise ArgumentError, "reload must respond to call" if reload && !reload.respond_to?(:call)

            @reload = reload
            @start_alignment = start_alignment
            @memory_byte_limit = Integer(memory_byte_limit) if memory_byte_limit
            raise ArgumentError, "memory_byte_limit must be positive" if
              @memory_byte_limit && !@memory_byte_limit.positive?
          end

          def run(prepared, &asr)
            raise ArgumentError, "an ASR callback is required" unless asr

            slots = Array.new(@total, UNSET)
            @fixed_results.each do |index, result|
              outcome = if result.is_a?(AlignmentWork) || result.is_a?(Final)
                          result
                        else
                          Final.new(index: index, result: result)
                        end
              install!(slots, outcome)
            end
            transitioned = false
            begin
              collect_asr!(slots, prepared, &asr)
              missing = slots.each_index.select { |index| slots.fetch(index).equal?(UNSET) }
              unless missing.empty?
                raise TranscriptionRuntimeError,
                      "Word ASR phase omitted input index(es): #{missing.join(", ")}"
              end

              # Set the flag before calling so an eviction exception is not
              # followed by a second, potentially destructive close attempt.
              transitioned = true
              alignment_required = slots.grep(AlignmentWork).any?(&:audio_required)
              @evict_asr.call if alignment_required

              align_slots!(slots)
              completed_count = 0
              slots.each_with_index do |outcome, index|
                result = outcome.is_a?(Final) ? outcome.result : outcome
                raise TranscriptionRuntimeError, "Word pipeline returned no result for input #{index}" if result.nil?

                slots[index] = result
                completed_count += 1
                @completed&.call(index, completed_count, @total, result)
                GC.start(full_mark: false, immediate_mark: true, immediate_sweep: true) if
                  (completed_count % @gc_interval).zero?
              end
              slots.freeze
            ensure
              @evict_asr.call unless transitioned
            end
          end

          private

          # Keeping ASR collection in its own frame guarantees that the final
          # PreparedEntry (and its PCM) is unreachable before alignment starts.
          def collect_asr!(slots, prepared)
            prepared.each { |prepared_entry| install!(slots, yield(prepared_entry)) }
          end

          def align_slots!(slots)
            works = slots.grep(AlignmentWork)
            if @reload
              PairBudgetReload.new(
                works,
                memory_byte_limit: @memory_byte_limit,
                reload: @reload,
                started: -> { @start_alignment&.call(works) }
              ).each do |work, reload_result|
                slots[work.index] = alignment_result(work, reload_result)
              end
            else
              works.each { |work| slots[work.index] = alignment_result(work) }
            end
          end

          def alignment_result(work, reload_result = UNSET)
            aligned = if reload_result.equal?(UNSET)
                        @align.call(work)
                      else
                        @align.call(work, reload_result)
                      end
            if aligned.is_a?(AlignmentWork) || aligned.is_a?(Final)
              raise TranscriptionRuntimeError,
                    "Word alignment returned a scheduler outcome instead of a result"
            end
            aligned
          end

          def install!(slots, outcome)
            unless outcome.is_a?(AlignmentWork) || outcome.is_a?(Final)
              raise TranscriptionRuntimeError,
                    "Word ASR phase must return AlignmentWork or Final, got #{outcome.class}"
            end
            index = outcome.index
            unless index.between?(0, @total - 1)
              raise TranscriptionRuntimeError,
                    "Word ASR phase returned out-of-range input index #{index}"
            end
            raise TranscriptionRuntimeError, "Word ASR phase returned input #{index} more than once" unless
              slots.fetch(index).equal?(UNSET)

            slots[index] = outcome
          end
        end

        # Reloads alignment PCM with one worker and at most one lookahead. The
        # next file may overlap MMS computation only when the current/next PCM
        # pair fits the configured budget; otherwise the current reference is
        # dropped before the next decode starts.
        class PairBudgetReload
          SAMPLE_BYTES = 4

          def initialize(works, memory_byte_limit:, reload:, started: nil)
            @works = works.to_a.freeze
            @memory_byte_limit = Integer(memory_byte_limit)
            raise ArgumentError, "memory_byte_limit must be positive" unless @memory_byte_limit.positive?
            raise ArgumentError, "reload must respond to call" unless reload.respond_to?(:call)
            raise ArgumentError, "started must respond to call" if started && !started.respond_to?(:call)

            @reload = reload
            @started = started
          end

          def each
            return enum_for(__method__) unless block_given?
            return if @works.empty?

            pending = submit(@works.first, 0)
            @started&.call
            @works.each_with_index do |work, index|
              current = pending
              pending = nil
              reload_result = current.value
              current = nil # rubocop:disable Lint/UselessAssignment -- release Thread#value before a serial decode

              next_work = @works[index + 1]
              pending = submit(next_work, index + 1) if next_work && pair_fits?(work, next_work)
              yield [work, reload_result].freeze
              reload_result = nil # rubocop:disable Lint/UselessAssignment -- release PCM before a serial decode
              unless pending || next_work.nil?
                GC.start(full_mark: false, immediate_mark: true, immediate_sweep: true)
                pending = submit(next_work, index + 1)
              end
            end
            completed = true
          ensure
            cancel(pending) if defined?(pending) && pending
            cancel(current) if defined?(current) && current
            GC.start(full_mark: false, immediate_mark: true, immediate_sweep: true) if defined?(completed) && completed
          end

          private

          def pair_fits?(left, right)
            pcm_bytes(left) + pcm_bytes(right) <= @memory_byte_limit
          end

          def pcm_bytes(work)
            work.audio_required ? work.expected_sample_count * SAMPLE_BYTES : 0
          end

          def submit(work, index)
            Thread.new do
              Thread.current.name = "cohere-alignment-reload-#{index}" if Thread.current.respond_to?(:name=)
              begin
                ReloadResult.new(decoded: @reload.call(work), error: nil)
              rescue StandardError => e
                ReloadResult.new(decoded: nil, error: e)
              end
            end.tap { |thread| thread.report_on_exception = false }
          end

          def cancel(thread)
            begin
              @reload.cancel if @reload.respond_to?(:cancel)
            rescue StandardError
              nil
            end
            thread.kill if thread.alive?
            thread.join
          rescue Exception # rubocop:disable Lint/RescueException -- preserve caller cancellation/failure
            nil
          end
        end

        # Re-decodes one alignment candidate with the concrete backend recorded
        # during ASR/checkpointing. It rejects automatic fallback and sample
        # drift, which makes the second decode reproducible and source-consistent.
        class ConcreteRedecoder
          def initialize(decoder:, sample_rate:, memory_byte_limit:, validate: nil, skip: nil)
            @decoder = decoder
            @sample_rate = Integer(sample_rate)
            @memory_byte_limit = Integer(memory_byte_limit)
            raise ArgumentError, "sample_rate must be positive" unless @sample_rate.positive?
            raise ArgumentError, "memory_byte_limit must be positive" unless @memory_byte_limit.positive?
            raise ArgumentError, "validate must respond to call" if validate && !validate.respond_to?(:call)
            raise ArgumentError, "skip must respond to call" if skip && !skip.respond_to?(:call)

            @validate = validate
            @skip = skip
          end

          def call(work)
            return nil if @skip&.call(work)

            @validate&.call(work)
            return nil unless work.audio_required

            backend = work.decode_backend
            if backend.nil? || backend.empty? || backend == "auto"
              raise TranscriptionRuntimeError,
                    "Cannot re-decode #{work.entry.path} for word alignment without a concrete ASR backend"
            end

            decoded = @decoder.decode(
              work.entry.path,
              backend: backend,
              sample_rate: @sample_rate,
              max_decoded_bytes: @memory_byte_limit
            )
            if decoded.backend != backend
              raise TranscriptionRuntimeError,
                    "Alignment decoder backend changed for #{work.entry.path}: " \
                    "#{backend} -> #{decoded.backend}"
            end
            unless decoded.samples.length == work.expected_sample_count
              raise TranscriptionRuntimeError,
                    "Decoded sample count changed between ASR and alignment for #{work.entry.path}: " \
                    "#{decoded.samples.length} != #{work.expected_sample_count}"
            end

            @validate&.call(work)
            decoded
          end

          def cancel
            Audio::FFmpegNative.cancel_active! if
              defined?(Audio::FFmpegNative) && Audio::FFmpegNative.respond_to?(:cancel_active!)
            nil
          rescue StandardError
            nil
          end
        end
      end
    end
  end
end
