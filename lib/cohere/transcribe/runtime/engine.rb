# frozen_string_literal: true

require_relative "../python_text"

module Cohere
  module Transcribe
    module Runtime
      PROCESS_GATE = Mutex.new
      THREAD_ACTIVE_KEY = :__cohere_transcribe_active
      THREAD_PROGRESS_KEY = :__cohere_transcribe_progress

      class FatalRuntimeError < TranscriptionRuntimeError; end
      private_constant :FatalRuntimeError

      Measurements = Struct.new(
        :runtime_import_seconds,
        :serialization_wait_seconds,
        :input_validation_seconds,
        :decode_seconds,
        :vad_seconds,
        :vad_model_load_seconds,
        :vad_inference_seconds,
        :vad_postprocess_seconds,
        :preparation_wait_seconds,
        :asr_load_seconds,
        :asr_seconds,
        :asr_generation_call_seconds,
        :asr_feature_worker_seconds,
        :asr_h2d_seconds,
        :asr_generate_device_seconds,
        :asr_generation_analysis_seconds,
        :aligner_load_seconds,
        :emissions_seconds,
        :viterbi_seconds,
        :post_asr_seconds,
        :checkpoint_seconds,
        :progressive_output_seconds,
        :cuda_total_gib,
        :cuda_free_start_gib,
        :cuda_free_end_gib,
        :asr_batches,
        :asr_processor_rows,
        :generated_tokens,
        :oom_retries,
        :truncation_retries,
        :effective_batch_min,
        :effective_batch_max,
        :final_batch_size,
        :final_batch_cap,
        :batch_history,
        :checkpoint_written_files,
        :vad_prepared_groups,
        :vad_model_calls,
        :vad_valid_frames,
        :vad_padded_frames,
        :vad_max_files_per_call,
        :vad_effective_block_frames,
        :vad_intraop_threads,
        :file_segmentation,
        keyword_init: true
      ) do
        def initialize(**values)
          super
          members.each do |member|
            self[member] ||= if %i[cuda_total_gib cuda_free_start_gib cuda_free_end_gib].include?(member)
                               nil
                             elsif member == :batch_history
                               []
                             elsif member == :file_segmentation
                               {}
                             elsif member.to_s.end_with?("seconds")
                               0.0
                             else
                               0
                             end
          end
        end
      end
      private_constant :Measurements

      GenerationOutcome = Data.define(:native, :retried, :hit_token_limit)
      private_constant :GenerationOutcome

      PreparationItem = Data.define(:index, :entry, :plan)
      private_constant :PreparationItem

      PreparationSilero = Data.define(:thread, :session)
      private_constant :PreparationSilero

      # Reusable, serialized Ruby execution engine. Heavy model conversion and
      # native session creation remain lazy until at least one speech segment
      # actually needs inference.
      class Engine
        attr_reader :options

        def initialize(options, progress: nil, model_provider: nil, decoder: Audio::Decoder, silero_factory: nil,
                       aligner_factory: nil)
          @options = options
          @progress = progress
          @model_provider = model_provider || ModelProvider.new
          @decoder = decoder
          @silero_factory = silero_factory || ->(**keywords) { VAD::Silero.new(**keywords) }
          @aligner_factory = aligner_factory || ->(**keywords) { Alignment::Aligner.new(**keywords) }
          @gate = Mutex.new
          @progress_gate = Mutex.new
          @closed = false
          @active = false
          @identity = nil
          @resources = ModelResources.new
          @native_session = nil
          @aligner = nil
          @preparation_sileros = []
          @preparation_silero_gate = Mutex.new
        end

        def transcribe(audio, raise_on_error: false, runtime_import_seconds: 0.0, serialization_wait_seconds: 0.0)
          if Thread.current.thread_variable_get(THREAD_PROGRESS_KEY)
            raise TranscriberBusyError, "Reentrant transcription from a progress callback is not supported"
          end
          if Thread.current.thread_variable_get(THREAD_ACTIVE_KEY)
            raise TranscriberBusyError, "Reentrant transcription in one process is not supported"
          end

          started = monotonic
          wait_started = monotonic
          @gate.synchronize do
            measurements = Measurements.new(
              runtime_import_seconds: runtime_import_seconds,
              serialization_wait_seconds: serialization_wait_seconds + monotonic - wait_started
            )
            raise TranscriberClosedError, "This Transcriber has been closed" if @closed
            raise TranscriberBusyError, "Reentrant transcription with one Transcriber is not supported" if @active

            @active = true
            begin
              process_wait = monotonic
              PROCESS_GATE.synchronize do
                measurements.serialization_wait_seconds += monotonic - process_wait
                previous_active = Thread.current.thread_variable_get(THREAD_ACTIVE_KEY)
                Thread.current.thread_variable_set(THREAD_ACTIVE_KEY, true)
                run = execute(audio, measurements, started)
                raise BatchTranscriptionError, run if raise_on_error && !run.ok?

                run
              ensure
                Thread.current.thread_variable_set(THREAD_ACTIVE_KEY, previous_active)
              end
            rescue ProgressCallbackError, BatchTranscriptionError, TranscriptionConfigurationError,
                   TranscriptionInputError, TranscriberBusyError, TranscriberClosedError
              raise
            rescue FatalRuntimeError => e
              evict_inference_sessions
              raise TranscriptionRuntimeError, e.message
            rescue TranscriptionRuntimeError
              evict_inference_sessions
              raise
            rescue Interrupt
              evict_inference_sessions
              raise
            rescue SystemExit => e
              evict_inference_sessions
              detail = e.message.to_s
              detail = "Transcription setup failed" if detail.empty? || detail == "SystemExit"
              raise TranscriptionRuntimeError, detail
            rescue ScriptError => e
              evict_inference_sessions
              raise TranscriptionRuntimeError, "#{e.class}: #{e.message}"
            rescue StandardError => e
              evict_inference_sessions
              raise TranscriptionRuntimeError, "#{e.class}: #{e.message}"
            ensure
              @active = false
            end
          end
        end

        def close
          if Thread.current.thread_variable_get(THREAD_PROGRESS_KEY) ||
             Thread.current.thread_variable_get(THREAD_ACTIVE_KEY)
            raise TranscriberBusyError, "Cannot close a Transcriber while transcription is active"
          end

          @gate.synchronize do
            return if @closed
            raise TranscriberBusyError, "Cannot close a Transcriber while its transcription is active" if @active

            begin
              @resources.close
            ensure
              @native_session = nil
              evict_aligner
            end
            @preparation_sileros.clear
            @closed = true
          end
          nil
        end

        private

        def execute(audio, measurements, started)
          validation_started = monotonic
          validate_configuration!
          if options.adapter
            raise TranscriptionConfigurationError,
                  "PEFT/LoRA adapters are not supported by the native Ruby Dense runtime"
          end
          entries = Input.expand(audio, recursive: options.recursive)
          preliminary_options = resolve_configuration
          # Preserve filesystem/preflight failures before any remote model
          # identity work, then bind durable contracts to the resolved model.
          preliminary_plans = Output::Publication.plan(
            entries,
            state_contract_options(preliminary_options)
          )
          preliminary_profile_binding = Output::Publication.bind_profile_path(
            preliminary_options.publication&.profile_json
          )
          @identity ||= resolve_identity
          resolved_options = resolve_configuration(model_identity: @identity)
          preliminary_plans.each_value do |plan|
            plan.directory_bindings.each(&:verify!)
          end
          preliminary_profile_binding&.verify!
          plans = Output::Publication.plan(entries, state_contract_options(resolved_options))
          profile_binding = Output::Publication.bind_profile_path(
            resolved_options.publication&.profile_json
          )
          Output::Publication.verify_plan_directory_continuity!(preliminary_plans, plans)
          Output::Publication.verify_profile_directory_continuity!(
            preliminary_profile_binding,
            profile_binding
          )
          measurements.input_validation_seconds += monotonic - validation_started

          preflight_skips = {}
          preparation_items = entries.each_with_index.filter_map do |entry, index|
            plan = plans.fetch(entry.path)
            outcome = preflight_outcome(
              index,
              entry,
              plan,
              resolved_options,
              measurements
            )
            next outcome if outcome.is_a?(PreparationItem)

            preflight_skips[index] = outcome
            nil
          end
          memory_byte_limit = [(resolved_options.audio_memory_gb * (1024**3)).to_i, 1].max
          estimate_bytes = if @decoder.respond_to?(:estimate_decoded_bytes)
                             lambda do |item|
                               @decoder.estimate_decoded_bytes(
                                 item.entry.path,
                                 backend: resolved_options.audio_backend,
                                 sample_rate: SAMPLE_RATE
                               )
                             end
                           end
          pipeline = Preparation::Pipeline.new(
            preparation_items,
            memory_byte_limit: memory_byte_limit,
            requested_workers: resolved_options.preprocess_workers,
            enabled: resolved_options.pipeline_preparation,
            worker_limit: vad_file_concurrency_limit(resolved_options),
            estimate_bytes: estimate_bytes,
            exclusive_retry: lambda do |prepared|
              prepared.error.is_a?(Audio::DecodedAudioLimitError)
            end,
            retained_bytes: lambda do |prepared|
              prepared.decoded ? prepared.decoded.samples.byte_size : 0
            end
          ) do |item, decoded_byte_limit, worker_slot|
            prepare_entry(
              item,
              resolved_options,
              decoded_byte_limit: decoded_byte_limit,
              worker_slot: worker_slot
            )
          end

          preparation_mode = pipeline.pipelined? ? "on" : "off"
          unless preparation_items.empty?
            report(
              ProgressEvent.new(
                stage: "message",
                message: "    preparing (workers=#{pipeline.effective_workers}, " \
                         "next-group=#{preparation_mode}, " \
                         "PCM group cap=#{pipeline.group_byte_limit} bytes)"
              )
            )
          end
          fully_preflight_skipped = preparation_items.empty? &&
                                    preflight_skips.length == entries.length &&
                                    preflight_skips.values.all? do |result|
                                      result.respond_to?(:status) && result.status == "skipped"
                                    end
          results = if fully_preflight_skipped
                      entries.each_index.map { |index| preflight_skips.fetch(index) }.freeze
                    elsif resolved_options.alignment == "word"
                      process_word_pipeline(
                        pipeline,
                        total: entries.length,
                        preflight_skips: preflight_skips,
                        resolved_options: resolved_options,
                        measurements: measurements,
                        memory_byte_limit: memory_byte_limit
                      )
                    else
                      process_standard_pipeline(
                        pipeline,
                        total: entries.length,
                        preflight_skips: preflight_skips,
                        resolved_options: resolved_options,
                        measurements: measurements
                      )
                    end
          measurements.preparation_wait_seconds += pipeline.wait_seconds
          record_cuda_memory(measurements, @native_session, start: false)
          evict_native_session if @resources.asr_circuit_broken?

          run = build_run(
            results,
            resolved_options,
            measurements,
            elapsed: monotonic - started
          )
          if (profile = resolved_options.publication&.profile_json)
            begin
              Output::Publication.write_profile(
                profile,
                run,
                runtime_metrics: profile_runtime_metrics(measurements),
                directory_binding: profile_binding
              )
            rescue TranscriptionError => e
              run = run.with(errors: (run.errors + [e.message]).freeze)
            end
          end
          public_run(run)
        end

        def process_standard_pipeline(pipeline, total:, preflight_skips:, resolved_options:, measurements:)
          results = []
          cursor = 0
          while cursor < total && preflight_skips.key?(cursor)
            results << preflight_skips.fetch(cursor)
            cursor += 1
            report(ProgressEvent.new(stage: "files", current: cursor, total: total))
          end

          pipeline.each do |prepared|
            target = prepared.item.index
            while cursor < target
              raise TranscriptionRuntimeError, "Preparation order became inconsistent" unless preflight_skips.key?(cursor)

              results << preflight_skips.fetch(cursor)
              cursor += 1
              report(ProgressEvent.new(stage: "files", current: cursor, total: total))
            end

            raise TranscriptionRuntimeError, "Preparation order became inconsistent" unless cursor == target

            results << process_prepared_entry(prepared, resolved_options, measurements)
            cursor += 1
            report(ProgressEvent.new(stage: "files", current: cursor, total: total))
          end
          while cursor < total
            raise TranscriptionRuntimeError, "Preparation omitted an input" unless preflight_skips.key?(cursor)

            results << preflight_skips.fetch(cursor)
            cursor += 1
            report(ProgressEvent.new(stage: "files", current: cursor, total: total))
          end
          results.freeze
        end

        def preflight_outcome(index, entry, plan, resolved_options, measurements)
          Output::Publication.with_plan_lock(plan) do
            decision = Output::Publication.revalidate(plan, resolved_options)
            case decision.action
            when :skip
              skipped_result(entry, plan)
            when :resume
              preflight_resumed_entry(
                index,
                entry,
                plan,
                decision,
                resolved_options,
                measurements
              )
            else
              PreparationItem.new(index: index, entry: entry, plan: plan)
            end
          end
        end

        def preflight_resumed_entry(index, entry, plan, decision, resolved_options, measurements)
          prepared = Preparation::PreparedEntry.new(
            item: PreparationItem.new(index: index, entry: entry, plan: plan),
            snapshot: source_snapshot(entry.path),
            decoded: nil,
            duration: decision.checkpoint.duration,
            segment_times: decision.checkpoint.segment_times,
            speech_spans: decision.checkpoint.speech_spans,
            vad_details: nil,
            decode_seconds: 0.0,
            vad_seconds: 0.0,
            error: nil
          )
          process_resumed_entry(
            prepared,
            decision.checkpoint,
            decision.generation_id,
            resolved_options,
            measurements,
            defer_word_alignment: resolved_options.alignment == "word"
          )
        rescue FatalRuntimeError, ProgressCallbackError
          raise
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          failed_result(entry, decision.checkpoint.duration, e)
        end

        def process_word_pipeline(pipeline, total:, preflight_skips:, resolved_options:, measurements:,
                                  memory_byte_limit:)
          alignment_state = { error: nil }
          post_asr_started = nil
          redecoder = WordPipeline::ConcreteRedecoder.new(
            decoder: @decoder,
            sample_rate: SAMPLE_RATE,
            memory_byte_limit: memory_byte_limit,
            skip: ->(_work) { !alignment_state.fetch(:error).nil? },
            validate: lambda do |work|
              ensure_source_unchanged!(work.entry.path, work.source_snapshot)
            end
          )
          WordPipeline::Coordinator.new(
            total: total,
            fixed_results: preflight_skips,
            evict_asr: method(:evict_current_asr_owner),
            reload: redecoder,
            memory_byte_limit: memory_byte_limit,
            start_alignment: lambda do |works|
              post_asr_started ||= monotonic
              preload_word_aligner(works, resolved_options, measurements, alignment_state)
            end,
            align: lambda do |work, reload_result|
              process_word_alignment(
                work,
                reload_result,
                resolved_options,
                measurements,
                alignment_state
              )
            end,
            completed: lambda do |_index, current, event_total, _result|
              report(ProgressEvent.new(stage: "files", current: current, total: event_total))
            end
          ).run(pipeline) do |prepared|
            process_word_asr_entry(prepared, resolved_options, measurements)
          end
        ensure
          measurements.post_asr_seconds += monotonic - post_asr_started if post_asr_started
          # Python's MMS owner is one-shot. Closing it after the alignment
          # phase also prevents a different reusable Transcriber from loading
          # Dense while this engine retains an idle aligner.
          evict_aligner
        end

        def resolve_identity
          @model_provider.resolve(options)
        rescue TranscriptionConfigurationError
          raise
        rescue ArgumentError, TypeError, Hub::Error => e
          raise TranscriptionConfigurationError, e.message
        end

        def validate_configuration!
          Configuration.validate!(options)
        rescue TranscriptionConfigurationError
          raise
        rescue ArgumentError, TypeError, NoMethodError => e
          raise TranscriptionConfigurationError, e.message
        end

        def resolve_configuration(model_identity: nil)
          resolved = Configuration.resolved(options, model_identity: model_identity)
          Precision.resolve(resolved)
        rescue TranscriptionConfigurationError
          raise
        rescue ArgumentError, TypeError, NoMethodError => e
          raise TranscriptionConfigurationError, e.message
        end

        def state_contract_options(resolved_options)
          return resolved_options unless resolved_options.vad == "silero"

          resolved_options.with(vad_engine: options.vad_engine)
        end

        def prepare_entry(item, resolved_options, decoded_byte_limit:, worker_slot:)
          snapshot = nil
          duration = nil
          decoded = nil
          segment_times = nil
          speech_spans = nil
          vad_details = nil
          decode_seconds = 0.0
          vad_seconds = 0.0

          snapshot = source_snapshot(item.entry.path)
          decode_started = monotonic
          begin
            decoded = @decoder.decode(
              item.entry.path,
              backend: resolved_options.audio_backend,
              sample_rate: SAMPLE_RATE,
              max_decoded_bytes: decoded_byte_limit
            )
          ensure
            decode_seconds += monotonic - decode_started
          end
          duration = decoded.samples.length.fdiv(SAMPLE_RATE)

          vad_started = monotonic
          begin
            segment_times, speech_spans, vad_details = segment(
              decoded.samples,
              resolved_options,
              silero: preparation_silero(worker_slot, resolved_options)
            )
          ensure
            vad_seconds += monotonic - vad_started
          end

          Preparation::PreparedEntry.new(
            item: item,
            snapshot: snapshot,
            decoded: decoded,
            duration: duration,
            segment_times: segment_times,
            speech_spans: speech_spans,
            vad_details: vad_details,
            decode_seconds: decode_seconds,
            vad_seconds: vad_seconds,
            error: nil
          )
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          Preparation::PreparedEntry.new(
            item: item,
            snapshot: snapshot,
            decoded: nil,
            duration: duration,
            segment_times: nil,
            speech_spans: nil,
            vad_details: nil,
            decode_seconds: decode_seconds,
            vad_seconds: vad_seconds,
            error: e
          )
        end

        def process_prepared_entry(prepared, resolved_options, measurements)
          item = prepared.item
          entry = item.entry
          plan = item.plan
          duration = prepared.duration
          measurements.decode_seconds += prepared.decode_seconds
          measurements.vad_seconds += prepared.vad_seconds
          record_vad_measurements(measurements, prepared.vad_details)
          Output::Publication.with_plan_lock(plan) do
            decision = Output::Publication.revalidate(plan, resolved_options)
            return skipped_result(entry, plan) if decision.action == :skip

            if prepared.error
              raise prepared.error if prepared.error.is_a?(FatalRuntimeError) ||
                                      prepared.error.is_a?(ProgressCallbackError)
              unless decision.action == :resume && resolved_options.alignment != "word"
                return failed_result(entry, duration, prepared.error)
              end
            end

            if decision.action == :resume
              return process_resumed_entry(
                prepared,
                decision.checkpoint,
                decision.generation_id,
                resolved_options,
                measurements
              )
            end

            process_fresh_entry(prepared, resolved_options, measurements)
          end
        rescue FatalRuntimeError, ProgressCallbackError
          raise
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          failed_result(entry, duration, e)
        end

        def process_word_asr_entry(prepared, resolved_options, measurements)
          item = prepared.item
          entry = item.entry
          plan = item.plan
          duration = prepared.duration
          measurements.decode_seconds += prepared.decode_seconds
          measurements.vad_seconds += prepared.vad_seconds
          record_vad_measurements(measurements, prepared.vad_details)
          outcome = Output::Publication.with_plan_lock(plan) do
            decision = Output::Publication.revalidate(plan, resolved_options)
            if decision.action == :skip
              WordPipeline::Final.new(index: item.index, result: skipped_result(entry, plan))
            else
              if prepared.error
                raise prepared.error if prepared.error.is_a?(ProgressCallbackError)

                resumable = decision.action == :resume && prepared.snapshot
                unless resumable
                  raise prepared.error if prepared.error.is_a?(FatalRuntimeError)

                  next WordPipeline::Final.new(
                    index: item.index,
                    result: failed_result(entry, duration, prepared.error)
                  )
                end
              end

              if decision.action == :resume
                process_resumed_entry(
                  prepared,
                  decision.checkpoint,
                  decision.generation_id,
                  resolved_options,
                  measurements,
                  defer_word_alignment: true
                )
              else
                process_fresh_entry(
                  prepared,
                  resolved_options,
                  measurements,
                  defer_word_alignment: true
                )
              end
            end
          end
          return outcome if outcome.is_a?(WordPipeline::AlignmentWork) || outcome.is_a?(WordPipeline::Final)

          raise TranscriptionRuntimeError, "Word ASR phase produced an invalid outcome for #{entry.path}"
        rescue FatalRuntimeError, ProgressCallbackError
          raise
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          WordPipeline::Final.new(index: item.index, result: failed_result(entry, duration, e))
        end

        def process_word_alignment(work, reload_result, resolved_options, measurements, alignment_state)
          entry = work.entry
          plan = work.plan
          Output::Publication.with_plan_lock(plan) do
            decision = Output::Publication.revalidate(plan, resolved_options)
            return skipped_result(entry, plan) if decision.action == :skip
            if (alignment_error = alignment_state.fetch(:error))
              return failed_result(entry, work.result.duration, alignment_error)
            end
            return failed_result(entry, work.result.duration, reload_result.error) unless reload_result.ok?

            ensure_source_unchanged!(entry.path, work.source_snapshot)
            rendered = render_and_publish(
              work.result,
              reload_result.decoded&.samples || [],
              work.segment_times,
              work.speech_spans,
              plan,
              resolved_options,
              measurements,
              work.generation_id,
              asr_evicted: true
            )
            ensure_source_unchanged!(entry.path, work.source_snapshot)
            rendered
          end
        rescue Alignment::BackendUnavailable => e
          alignment_state[:error] = e
          failed_result(entry, work.result.duration, e)
        rescue FatalRuntimeError, ProgressCallbackError
          raise
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          failed_result(entry, work.result.duration, e)
        end

        def preload_word_aligner(works, resolved_options, measurements, alignment_state)
          return unless works.any?(&:audio_required)

          aligner = ensure_aligner(resolved_options)
          before = alignment_measurements(aligner)
          aligner.load! if aligner.respond_to?(:load!)
        rescue Interrupt, SystemExit, ProgressCallbackError
          raise
        rescue StandardError => e
          alignment_state[:error] = e
          evict_aligner
        ensure
          record_alignment_measurements(measurements, before, aligner) if before && aligner
        end

        def process_fresh_entry(prepared, resolved_options, measurements, defer_word_alignment: false)
          item = prepared.item
          entry = item.entry
          plan = item.plan
          snapshot = prepared.snapshot
          decoded = prepared.decoded
          duration = prepared.duration
          segment_times = prepared.segment_times
          speech_spans = prepared.speech_spans
          vad_details = prepared.vad_details
          record_file_segmentation(measurements, entry.path, segment_times, speech_spans)

          segments = []
          repetition_stopped_segments = []
          truncation_retried_segments = []
          token_limit_segments = []
          generated_tokens = []
          report(ProgressEvent.new(stage: "ASR", current: 0, total: segment_times.length))
          generation_outcomes = transcribe_segments(
            decoded.samples,
            segment_times,
            language: resolved_options.language,
            measurements: measurements,
            resolved_options: resolved_options
          )
          segment_times.each_with_index do |(start_time, end_time), segment_index|
            outcome = generation_outcomes.fetch(segment_index)
            native = outcome.native
            retried = outcome.retried
            hit_token_limit = outcome.hit_token_limit
            repetition_stopped_segments << segment_index if native.repetition_stopped
            truncation_retried_segments << segment_index if retried
            token_limit_segments << segment_index if hit_token_limit
            text = PythonText.strip(native.text.to_s)
            segments << TranscriptionSegment.new(
              index: segment_index,
              start: start_time,
              end: end_time,
              text: text
            )
            token_count = native.generated_tokens
            generated_tokens << [segment_index, token_count]
            report(ProgressEvent.new(stage: "ASR", current: segment_index + 1, total: segment_times.length))
          end

          ensure_source_unchanged!(entry.path, snapshot)
          text = segments.map(&:text).reject(&:empty?).join("\n")
          provenance = provenance_for(
            decoded: decoded,
            vad_details: vad_details,
            repetition_stopped_segments: repetition_stopped_segments,
            truncation_retried_segments: truncation_retried_segments,
            token_limit_segments: token_limit_segments,
            generated_tokens: generated_tokens,
            published: false
          )
          result = TranscriptionResult.new(
            path: entry.path,
            relative_path: entry.relative_path,
            status: "completed",
            text: text,
            duration: duration,
            segments: segments,
            provenance: provenance
          )
          checkpoint_started = nil
          generation_id = if plan.checkpoint_path
                            checkpoint_started = monotonic
                            begin
                              State.write_asr_checkpoint(
                                path: plan.checkpoint_path,
                                result: result,
                                source_snapshot: plan.source_snapshot,
                                asr_contract_key: plan.asr_contract_key,
                                speech_spans: speech_spans,
                                vad_provider_options: vad_provider_options(vad_details),
                                directory_binding: plan.directory_bindings.last,
                                guard_bindings: plan.directory_bindings
                              )
                            ensure
                              measurements.checkpoint_seconds += monotonic - checkpoint_started
                            end
                          end
          measurements.checkpoint_written_files += 1 if generation_id
          if defer_word_alignment
            return word_alignment_work(
              prepared,
              result,
              segment_times,
              speech_spans,
              generation_id
            )
          end

          render_and_publish(
            result,
            decoded.samples,
            segment_times,
            speech_spans,
            plan,
            resolved_options,
            measurements,
            generation_id
          )
        end

        def process_resumed_entry(prepared, checkpoint, generation_id, resolved_options, measurements,
                                  defer_word_alignment: false)
          item = prepared.item
          entry = item.entry
          plan = item.plan
          record_file_segmentation(
            measurements,
            entry.path,
            checkpoint.segment_times,
            checkpoint.speech_spans
          )
          segments = checkpoint.segment_times.each_with_index.map do |(start_time, end_time), index|
            TranscriptionSegment.new(
              index: index,
              start: start_time,
              end: end_time,
              text: checkpoint.segment_texts.fetch(index)
            )
          end
          text = segments.map(&:text).reject(&:empty?).join("\n")
          provenance = checkpoint_provenance(checkpoint, resolved_options)
          result = TranscriptionResult.new(
            path: entry.path,
            relative_path: entry.relative_path,
            status: "completed",
            text: text,
            duration: checkpoint.duration,
            segments: segments,
            provenance: provenance
          )
          if defer_word_alignment
            return word_alignment_work(
              prepared,
              result,
              checkpoint.segment_times,
              checkpoint.speech_spans,
              generation_id,
              expected_sample_count: (checkpoint.duration * SAMPLE_RATE).round(half: :even)
            )
          end

          samples = prepared.decoded&.samples || []
          render_and_publish(
            result,
            samples,
            checkpoint.segment_times,
            checkpoint.speech_spans,
            plan,
            resolved_options,
            measurements,
            generation_id
          )
        end

        def word_alignment_work(prepared, result, segment_times, speech_spans, generation_id,
                                expected_sample_count: nil)
          WordPipeline::AlignmentWork.new(
            index: prepared.item.index,
            entry: prepared.item.entry,
            plan: prepared.item.plan,
            result: result,
            segment_times: segment_times,
            speech_spans: speech_spans,
            generation_id: generation_id,
            decode_backend: result.provenance.decode_backend,
            expected_sample_count: expected_sample_count || prepared.decoded.samples.length,
            audio_required: result.segments.any? { |segment| !segment.text.empty? },
            source_snapshot: prepared.snapshot
          )
        end

        def render_and_publish(result, samples, segment_times, speech_spans, plan, resolved_options,
                               measurements, generation_id, asr_evicted: false)
          progressive_started = monotonic unless resolved_options.alignment == "word"
          words, fallback_alignment_segments = words_for_segments(
            samples,
            segment_times,
            result.segments,
            speech_spans,
            resolved_options,
            measurements,
            asr_evicted: asr_evicted
          )
          cues = if resolved_options.alignment == "none"
                   [].freeze
                 else
                   Output::Rendering.build_cues(
                     words,
                     max_chars: resolved_options.max_chars,
                     max_duration: resolved_options.max_cue_dur,
                     max_gap: resolved_options.max_gap,
                     media_duration: result.duration
                   )
                 end
          result = result.with(
            words: words,
            cues: cues,
            provenance: result.provenance.with(
              fallback_alignment_segments: fallback_alignment_segments
            )
          )
          outputs = Output::Publication.write(
            plan,
            result,
            resolved_options,
            generation_id: generation_id,
            speech_spans: speech_spans
          )
          if outputs.any?
            result = result.with(
              outputs: outputs,
              provenance: result.provenance.with(published: true)
            )
          end
          result
        ensure
          measurements.progressive_output_seconds += monotonic - progressive_started if progressive_started
        end

        def checkpoint_provenance(checkpoint, resolved_options)
          requested_vad = resolved_options.vad == "silero" ? options.vad_engine : nil
          TranscriptionProvenance.new(
            model_id: @identity&.model_id,
            model_revision: @identity&.model_revision,
            model_format: @identity&.model_format&.to_s,
            adapter_id: @identity&.adapter_id,
            adapter_revision: @identity&.adapter_revision,
            decode_backend: checkpoint.decode_backend,
            decode_fallback_reason: checkpoint.decode_fallback_reason,
            vad_engine_requested: requested_vad,
            vad_engine_actual: checkpoint.vad_engine_actual,
            vad_provider: checkpoint.vad_provider,
            vad_fallback_reason: checkpoint.vad_fallback_reason,
            repetition_stopped_segments: checkpoint.repetition_stopped_segments,
            truncation_retried_segments: checkpoint.truncation_retried_segments,
            token_limit_segments: checkpoint.token_limit_segments,
            generated_tokens_by_segment: checkpoint.generated_tokens_by_segment,
            resumed_from_asr_checkpoint: true,
            published: false
          )
        end

        def vad_provider_options(vad_details)
          vad_details&.fetch(:provider_options, nil)
        end

        def segment(samples, resolved_options, silero: nil)
          duration = samples.length.fdiv(SAMPLE_RATE)
          case resolved_options.vad
          when "none"
            spans = Audio::Segmentation.fixed(samples, resolved_options.max_dur)
            [spans, spans, { requested: nil, actual: "none (fixed windows)", provider: nil, fallback: nil }]
          when "auditok"
            spans = Audio::Segmentation.energy(
              samples,
              min_duration: resolved_options.min_dur,
              max_duration: resolved_options.max_dur,
              max_silence: resolved_options.max_silence,
              threshold: resolved_options.energy_threshold
            )
            spans = Audio::Segmentation.validate(spans, duration, max_duration: resolved_options.max_dur)
            [spans, spans, { requested: nil, actual: "auditok", provider: nil, fallback: nil }]
          when "silero"
            silero ||= preparation_silero(0, resolved_options)
            timestamps = silero.speech_timestamps(
              samples,
              sampling_rate: SAMPLE_RATE,
              threshold: resolved_options.vad_threshold,
              min_speech_duration_ms: (resolved_options.min_dur * 1000).round(half: :even),
              max_speech_duration_s: resolved_options.max_dur,
              min_silence_duration_ms: resolved_options.min_silence_ms,
              speech_pad_ms: resolved_options.speech_pad_ms
            )
            raw = Audio::Segmentation.samples_to_seconds(timestamps, samples.length)
            spans = resolved_options.vad_merge ? Audio::Segmentation.merge_speech(raw, resolved_options.max_dur) : raw
            spans = Audio::Segmentation.validate(spans, duration, max_duration: resolved_options.max_dur)
            requested = options.vad_engine
            fallback = unless %w[auto onnx].include?(requested)
                         "Ruby maps the requested Silero #{requested} executor to the equivalent packaged ONNX graph"
                       end
            actual = "onnx"
            provider_options = silero.provider_options if silero.respond_to?(:provider_options)
            execution = silero.last_execution if silero.respond_to?(:last_execution)
            intraop_threads = silero.intra_op_threads if silero.respond_to?(:intra_op_threads)
            [
              spans,
              raw,
              {
                requested: requested,
                actual: actual,
                provider: silero.provider,
                provider_options: provider_options,
                execution: execution,
                intraop_threads: intraop_threads,
                fallback: fallback
              }
            ]
          else
            raise TranscriptionConfigurationError, "Unsupported VAD mode: #{resolved_options.vad.inspect}"
          end
        rescue VAD::SileroBackendUnavailable => e
          raise FatalRuntimeError, "Silero VAD is unavailable: #{e.message}"
        end

        def preparation_silero(worker_slot, resolved_options)
          return nil unless resolved_options.vad == "silero"

          @preparation_silero_gate.synchronize do
            cached = @preparation_sileros[worker_slot]
            return cached.session if cached && cached.thread.equal?(Thread.current)

            keywords = silero_runtime_options(resolved_options)
            session = if keywords.empty? || !silero_factory_accepts_keywords?
                        @silero_factory.call
                      else
                        @silero_factory.call(**keywords)
                      end
            @preparation_sileros[worker_slot] = PreparationSilero.new(
              thread: Thread.current,
              session: session
            )
            session
          end
        end

        def silero_runtime_options(resolved_options)
          return {} unless packed_silero_compatibility?(resolved_options)

          {
            block_frames: resolved_options.vad_block_frames,
            threads: resolved_options.vad_threads || VAD::Silero::DEFAULT_INTRA_OP_THREADS
          }
        end

        def silero_factory_accepts_keywords?
          parameters = if @silero_factory.respond_to?(:parameters)
                         @silero_factory.parameters
                       else
                         @silero_factory.method(:call).parameters
                       end
          parameters.any? { |kind, _name| %i[key keyreq keyrest].include?(kind) }
        rescue NameError
          false
        end

        def vad_file_concurrency_limit(resolved_options)
          resolved_options.vad_batch_size if packed_silero_compatibility?(resolved_options)
        end

        def packed_silero_compatibility?(resolved_options)
          resolved_options.vad == "silero" && %w[auto torch].include?(options.vad_engine)
        end

        def transcribe_segments(samples, segment_times, language:, measurements:, resolved_options:)
          return [] if segment_times.empty?

          # A retained MMS session from an earlier file/run must not overlap
          # the much larger Dense ASR checkpoint.
          evict_aligner if resolved_options.alignment == "word"
          session = ensure_native_session(measurements, resolved_options)
          work = segment_times.each_with_index.map do |(start_time, end_time), index|
            { index: index, start: start_time, end: end_time, duration: end_time - start_time }
          end
          work.sort_by! { |row| [-row.fetch(:duration), row.fetch(:index)] }
          controller = ensure_batch_controller(session, resolved_options, work.map { |row| row.fetch(:duration) })
          telemetry = ASR::BatchTelemetry.new
          outcomes = Array.new(segment_times.length)

          begin
            until work.empty?
              count = if work.first.fetch(:duration) > 35.0
                        1
                      else
                        controller.take_count(work) { |row| row.fetch(:duration) }
                      end
              rows = work.shift(count)
              batch_outcomes = transcribe_batch_generation(
                session,
                samples,
                rows,
                language: language,
                measurements: measurements,
                resolved_options: resolved_options,
                max_new_tokens: resolved_options.max_new_tokens,
                controller: controller,
                telemetry: telemetry
              )
              batch_outcomes.each { |index, outcome| outcomes[index] = outcome }
            end
          ensure
            merge_batch_telemetry(measurements, telemetry)
          end

          outcomes.freeze
        end

        def ensure_batch_controller(session, resolved_options, durations)
          controller = @resources.batch_controller
          return controller.configure_group(resolved_options, durations) if controller

          physical_max = if session.respond_to?(:batch_capacity)
                           Integer(session.batch_capacity)
                         elsif session.respond_to?(:transcribe_batch)
                           8
                         else
                           1
                         end
          raise TranscriptionRuntimeError, "Native session reported an invalid batch capacity" unless physical_max.positive?

          controller = ASR::BatchController.create(
            resolved_options,
            device: resolved_options.device,
            durations: durations,
            memory: native_memory(session),
            physical_max: physical_max
          )
          @resources.install_batch_controller(controller)
        end

        def transcribe_batch_generation(session, samples, rows, language:, measurements:, resolved_options:,
                                        max_new_tokens:, controller:, telemetry:, retry_cap: nil)
          executor = ASR::BatchExecutor.new(
            controller,
            memory: (-> { session.memory } if session.respond_to?(:memory)),
            operation_metrics: (-> { session.last_batch_metrics } if session.respond_to?(:last_batch_metrics)),
            telemetry: telemetry
          )
          execution = executor.execute(
            rows,
            max_new_tokens: max_new_tokens,
            base_max_new_tokens: resolved_options.max_new_tokens,
            retry_cap: retry_cap
          ) do |attempt_rows|
            call_native_batch(
              session,
              samples,
              attempt_rows,
              language: language,
              measurements: measurements,
              max_new_tokens: max_new_tokens
            )
          end
          raise execution.errors.compact.first if execution.errors.any?

          natives = execution.values
          outcomes = {}
          retry_groups = Hash.new { |hash, limit| hash[limit] = [] }

          rows.each_with_index do |row, lane|
            native = natives.fetch(lane)
            unless native.stopped_by_max_tokens
              outcomes[row.fetch(:index)] = GenerationOutcome.new(
                native: native,
                retried: false,
                hit_token_limit: false
              )
              next
            end

            next_limit = retry_token_limit(
              max_new_tokens,
              resolved_options.max_retry_tokens,
              native.generation_limit,
              native.generation_capacity
            )
            if resolved_options.truncation_policy == "retry" && next_limit > max_new_tokens
              retry_groups[next_limit] << row
            else
              outcomes[row.fetch(:index)] = GenerationOutcome.new(
                native: native,
                retried: false,
                hit_token_limit: true
              )
            end
          end

          retry_groups.each do |next_limit, retry_rows|
            measurements.truncation_retries += retry_rows.length
            group_retry_cap = retry_cap || ASR::RetryBatchCap.new(retry_rows.length)
            retried = transcribe_batch_generation(
              session,
              samples,
              retry_rows,
              language: language,
              measurements: measurements,
              resolved_options: resolved_options,
              max_new_tokens: next_limit,
              controller: controller,
              telemetry: telemetry,
              retry_cap: group_retry_cap
            )
            retried.each do |index, outcome|
              outcomes[index] = GenerationOutcome.new(
                native: outcome.native,
                retried: true,
                hit_token_limit: outcome.hit_token_limit
              )
            end
          end
          outcomes.freeze
        end

        def call_native_batch(session, samples, rows, language:, measurements:, max_new_tokens:)
          slices = []
          offsets = []
          rows.each do |row|
            start_sample = (row.fetch(:start) * SAMPLE_RATE).round(half: :even).clamp(0, samples.length)
            end_sample = (row.fetch(:end) * SAMPLE_RATE).round(half: :even).clamp(start_sample, samples.length)
            slices << samples[start_sample...end_sample].dup
            offsets << row.fetch(:start)
          end

          asr_started = monotonic
          if rows.length == 1 || !session.respond_to?(:transcribe_batch)
            [session.transcribe(
              slices.first,
              language: language,
              offset: offsets.first,
              max_new_tokens: max_new_tokens
            )]
          else
            session.transcribe_batch(
              slices,
              language: language,
              offsets: offsets,
              max_new_tokens: max_new_tokens
            )
          end
        ensure
          if asr_started
            elapsed = monotonic - asr_started
            measurements.asr_seconds += elapsed
            measurements.asr_generation_call_seconds += elapsed
          end
          record_cuda_memory(measurements, session, start: false)
        end

        def native_memory(session)
          session.memory if session.respond_to?(:memory)
        rescue StandardError
          nil
        end

        def merge_batch_telemetry(measurements, telemetry)
          measurements.asr_batches += telemetry.asr_batches
          measurements.asr_processor_rows += telemetry.processor_rows
          measurements.generated_tokens += telemetry.generated_tokens
          measurements.oom_retries += telemetry.oom_retries
          measurements.asr_feature_worker_seconds += telemetry.feature_worker_seconds
          measurements.asr_h2d_seconds += telemetry.h2d_seconds
          measurements.asr_generate_device_seconds += telemetry.generate_device_seconds
          measurements.asr_generation_analysis_seconds += telemetry.generation_analysis_seconds
          if telemetry.effective_batch_min.positive?
            measurements.effective_batch_min = if measurements.effective_batch_min.zero?
                                                 telemetry.effective_batch_min
                                               else
                                                 [measurements.effective_batch_min,
                                                  telemetry.effective_batch_min].min
                                               end
          end
          measurements.effective_batch_max = [
            measurements.effective_batch_max,
            telemetry.effective_batch_max
          ].max
          measurements.final_batch_size = telemetry.final_batch_size
          measurements.final_batch_cap = telemetry.final_batch_cap
          measurements.batch_history.concat(telemetry.batch_history)
        end

        def retry_token_limit(current_limit, requested_maximum, effective_limit, generation_capacity)
          # A lower native effective limit means the decoder's positional
          # context, rather than the user limit, was exhausted. Retrying cannot
          # create more decoder positions.
          return current_limit if effective_limit.positive? && effective_limit < current_limit

          ceiling = generation_capacity.positive? ? [requested_maximum, generation_capacity].min : requested_maximum
          return current_limit if ceiling <= current_limit

          proposed = [ceiling, [current_limit + 128, current_limit * 2].max].min
          ceiling - proposed < 128 ? ceiling : proposed
        end

        def ensure_native_session(measurements, resolved_options)
          loaded_seconds = 0.0
          retained = @resources.asr_session
          if retained
            @native_session = retained
            record_cuda_memory(measurements, retained, start: true)
            return retained
          end

          evict_aligner
          session, = @resources.acquire_asr(asr_resource_key(resolved_options)) do
            started = monotonic
            begin
              @model_provider.open(@identity, resolved_options)
            ensure
              loaded_seconds += monotonic - started
            end
          end
          @native_session = session
          record_cuda_memory(measurements, session, start: true)
          session
        rescue TranscriptionError => e
          raise FatalRuntimeError, e.message
        rescue StandardError => e
          raise FatalRuntimeError, "Cannot initialize Dense Cohere inference: #{e.class}: #{e.message}"
        ensure
          measurements.asr_load_seconds += loaded_seconds if loaded_seconds
        end

        def asr_resource_key(resolved_options)
          [
            resolved_options.device,
            resolved_options.dtype,
            @identity.model_id,
            @identity.model_revision,
            @identity.model_format,
            @identity.adapter_id,
            @identity.adapter_revision
          ].freeze
        end

        def uniform_words_for_segment(text, start_time, end_time, segment_index, speech_spans, resolved_options)
          return [] if text.empty?

          within = Output::Timing.spans_within(speech_spans, start_time, end_time)
          if resolved_options.vad_merge && within.any?
            Output::Timing.uniform_words_across_spans(
              text, within, segment_index, "uniform_speech_spans"
            )
          else
            Output::Timing.uniform_words(text, start_time, end_time, segment_index)
          end
        end

        def words_for_segments(samples, segment_times, segments, speech_spans, resolved_options, measurements,
                               asr_evicted: false)
          case resolved_options.alignment
          when "none"
            [[], 0]
          when "word"
            align_words(
              samples,
              segment_times,
              segments,
              resolved_options,
              measurements,
              asr_evicted: asr_evicted
            )
          else
            words = segments.flat_map do |segment|
              uniform_words_for_segment(
                segment.text,
                segment.start,
                segment.end,
                segment.index,
                speech_spans,
                resolved_options
              )
            end
            [words.freeze, 0]
          end
        end

        def align_words(samples, segment_times, segments, resolved_options, measurements, asr_evicted: false)
          return [[], 0] unless segments.any? { |segment| !segment.text.empty? }

          # Match the reference runtime's memory invariant: the 2B ASR and
          # 300M MMS checkpoints must not be resident at the same time.
          evict_current_asr_owner unless asr_evicted
          report(ProgressEvent.new(stage: "alignment", current: 0, total: segments.length))
          aligner = ensure_aligner(resolved_options)
          before = alignment_measurements(aligner)
          result = aligner.align(
            samples,
            segment_times,
            segments.map(&:text),
            language: resolved_options.language
          )
          report(ProgressEvent.new(stage: "alignment", current: segments.length, total: segments.length))
          result
        rescue StandardError
          evict_aligner unless asr_evicted
          raise
        ensure
          record_alignment_measurements(measurements, before, aligner) if before && aligner
        end

        def ensure_aligner(resolved_options)
          @aligner ||= @aligner_factory.call(
            dtype: resolved_options.align_dtype,
            device: resolved_options.device,
            batch_size: resolved_options.align_batch_size
          )
        rescue TranscriptionError
          raise
        rescue StandardError => e
          raise TranscriptionRuntimeError, "Cannot initialize MMS word alignment: #{e.class}: #{e.message}"
        end

        def alignment_measurements(aligner)
          [aligner.load_seconds.to_f, aligner.emissions_seconds.to_f, aligner.viterbi_seconds.to_f]
        end

        def record_alignment_measurements(measurements, before, aligner)
          after = alignment_measurements(aligner)
          measurements.aligner_load_seconds += [after.fetch(0) - before.fetch(0), 0.0].max
          measurements.emissions_seconds += [after.fetch(1) - before.fetch(1), 0.0].max
          measurements.viterbi_seconds += [after.fetch(2) - before.fetch(2), 0.0].max
        end

        def skipped_result(entry, plan)
          TranscriptionResult.new(
            path: entry.path,
            relative_path: entry.relative_path,
            status: "skipped",
            text: nil,
            duration: skipped_duration(entry.path),
            outputs: plan.paths.values,
            provenance: provenance_for(published: true)
          )
        end

        def skipped_duration(path)
          @decoder.probe_duration(path) if @decoder.respond_to?(:probe_duration)
        rescue StandardError
          nil
        end

        def failed_result(entry, duration, error)
          TranscriptionResult.new(
            path: entry.path,
            relative_path: entry.relative_path,
            status: "failed",
            text: nil,
            duration: duration,
            error: "#{error.class}: #{error.message}",
            provenance: provenance_for(published: false)
          )
        end

        def provenance_for(published:, decoded: nil, vad_details: nil, fallback_alignment_segments: 0,
                           repetition_stopped_segments: [], truncation_retried_segments: [],
                           token_limit_segments: [], generated_tokens: [])
          TranscriptionProvenance.new(
            model_id: @identity&.model_id,
            model_revision: @identity&.model_revision,
            model_format: @identity&.model_format&.to_s,
            adapter_id: @identity&.adapter_id,
            adapter_revision: @identity&.adapter_revision,
            decode_backend: decoded&.backend,
            decode_fallback_reason: decoded&.fallback_reason,
            vad_engine_requested: vad_details&.fetch(:requested, nil),
            vad_engine_actual: vad_details&.fetch(:actual, nil),
            vad_provider: vad_details&.fetch(:provider, nil),
            vad_fallback_reason: vad_details&.fetch(:fallback, nil),
            fallback_alignment_segments: fallback_alignment_segments,
            repetition_stopped_segments: repetition_stopped_segments,
            truncation_retried_segments: truncation_retried_segments,
            token_limit_segments: token_limit_segments,
            generated_tokens_by_segment: generated_tokens,
            published: published
          )
        end

        def build_run(results, resolved_options, measurements, elapsed:)
          successful_seconds = results.sum do |result|
            result.status == "completed" ? result.duration.to_f : 0.0
          end
          statistics = TranscriptionStatistics.new(
            elapsed_seconds: elapsed,
            successful_audio_seconds: successful_seconds,
            real_time_factor_x: elapsed.positive? ? successful_seconds / elapsed : 0.0,
            runtime_import_seconds: measurements.runtime_import_seconds,
            serialization_wait_seconds: measurements.serialization_wait_seconds,
            input_validation_seconds: measurements.input_validation_seconds,
            decode_seconds: measurements.decode_seconds,
            vad_seconds: measurements.vad_seconds,
            asr_load_seconds: measurements.asr_load_seconds,
            asr_seconds: measurements.asr_seconds,
            aligner_load_seconds: measurements.aligner_load_seconds,
            emissions_seconds: measurements.emissions_seconds,
            viterbi_seconds: measurements.viterbi_seconds,
            peak_cuda_allocated_gib: 0.0,
            peak_cuda_reserved_gib: 0.0,
            asr_batches: measurements.asr_batches,
            asr_processor_rows: measurements.asr_processor_rows,
            generated_tokens: measurements.generated_tokens,
            oom_retries: measurements.oom_retries,
            truncation_retries: measurements.truncation_retries
          )
          TranscriptionRun.new(
            results: results,
            requested_options: options,
            resolved_options: resolved_options,
            statistics: statistics
          )
        end

        def profile_runtime_metrics(measurements)
          {
            vad_model_load_seconds: measurements.vad_model_load_seconds,
            vad_inference_seconds: measurements.vad_inference_seconds,
            vad_postprocess_seconds: measurements.vad_postprocess_seconds,
            preparation_wait_seconds: measurements.preparation_wait_seconds,
            asr_generation_call_seconds: measurements.asr_generation_call_seconds,
            asr_feature_worker_seconds: measurements.asr_feature_worker_seconds,
            asr_h2d_seconds: measurements.asr_h2d_seconds,
            asr_generate_device_seconds: measurements.asr_generate_device_seconds,
            asr_generation_analysis_seconds: measurements.asr_generation_analysis_seconds,
            post_asr_seconds: measurements.post_asr_seconds,
            checkpoint_seconds: measurements.checkpoint_seconds,
            progressive_output_seconds: measurements.progressive_output_seconds,
            cuda_total_gib: measurements.cuda_total_gib,
            cuda_free_start_gib: measurements.cuda_free_start_gib,
            cuda_free_end_gib: measurements.cuda_free_end_gib,
            effective_batch_min: measurements.effective_batch_min,
            effective_batch_max: measurements.effective_batch_max,
            final_batch_size: measurements.final_batch_size,
            final_batch_cap: measurements.final_batch_cap,
            batch_history: measurements.batch_history,
            checkpoint_written_files: measurements.checkpoint_written_files,
            vad_prepared_groups: measurements.vad_prepared_groups,
            vad_model_calls: measurements.vad_model_calls,
            vad_valid_frames: measurements.vad_valid_frames,
            vad_padded_frames: measurements.vad_padded_frames,
            vad_max_files_per_call: measurements.vad_max_files_per_call,
            vad_effective_block_frames: measurements.vad_effective_block_frames,
            vad_intraop_threads: measurements.vad_intraop_threads,
            file_segmentation: measurements.file_segmentation
          }.freeze
        end

        def record_vad_measurements(measurements, vad_details)
          intraop_threads = vad_details&.fetch(:intraop_threads, nil)
          measurements.vad_intraop_threads = intraop_threads if intraop_threads
          execution = vad_details&.fetch(:execution, nil)
          return unless execution

          measurements.vad_prepared_groups += 1
          measurements.vad_model_load_seconds += execution.model_load_seconds if execution.respond_to?(:model_load_seconds)
          measurements.vad_inference_seconds += execution.inference_seconds if execution.respond_to?(:inference_seconds)
          measurements.vad_postprocess_seconds += execution.postprocess_seconds if execution.respond_to?(:postprocess_seconds)
          measurements.vad_model_calls += execution.model_calls
          measurements.vad_valid_frames += execution.valid_frames
          measurements.vad_padded_frames += execution.padded_frames
          measurements.vad_max_files_per_call = [
            measurements.vad_max_files_per_call,
            execution.max_files_per_call
          ].max
          effective = execution.effective_block_frames
          return unless effective&.positive?

          current = measurements.vad_effective_block_frames
          measurements.vad_effective_block_frames = current.zero? ? effective : [current, effective].min
        end

        def record_file_segmentation(measurements, path, segment_times, speech_spans)
          immutable_spans = lambda do |spans|
            spans.map { |start_time, end_time| [start_time.to_f, end_time.to_f].freeze }.freeze
          end
          measurements.file_segmentation[path.to_s] = {
            segment_times: immutable_spans.call(segment_times),
            speech_spans: immutable_spans.call(speech_spans)
          }.freeze
        end

        def record_cuda_memory(measurements, session, start:)
          return unless session.respond_to?(:memory)
          return if session.respond_to?(:device) && session.device.to_s != "cuda"

          free_bytes, total_bytes = native_memory(session)
          free_bytes = Integer(free_bytes)
          total_bytes = Integer(total_bytes)
          return unless free_bytes.between?(0, total_bytes) && total_bytes.positive?

          measurements.cuda_total_gib = total_bytes.fdiv(1024**3)
          free_gib = free_bytes.fdiv(1024**3)
          if start
            measurements.cuda_free_start_gib ||= free_gib
          else
            measurements.cuda_free_end_gib = free_gib
          end
        rescue TypeError, ArgumentError
          nil
        end

        # Publication, checkpoints, and profiles retain every ASR row so their
        # duration and generation telemetry remains complete. The stable public
        # API mirrors Python by omitting blank transcript rows without
        # renumbering the surviving segment indices.
        def public_run(run)
          changed = false
          results = run.results.map do |result|
            next result if result.segments.empty?

            visible = result.segments.reject { |segment| PythonText.blank?(segment.text.to_s) }
            next result if visible.length == result.segments.length

            changed = true
            result.with(segments: visible.freeze)
          end.freeze
          changed ? run.with(results: results) : run
        end

        def report(event)
          return unless @progress

          @progress_gate.synchronize do
            previous = Thread.current.thread_variable_get(THREAD_PROGRESS_KEY)
            Thread.current.thread_variable_set(THREAD_PROGRESS_KEY, true)
            begin
              @progress.call(event)
            rescue ScriptError, StandardError => e
              evict_inference_sessions
              raise ProgressCallbackError, e
            ensure
              Thread.current.thread_variable_set(THREAD_PROGRESS_KEY, previous)
            end
          end
        end

        def source_snapshot(path)
          stat = path.stat
          [stat.dev, stat.ino, stat.size, stat.mtime.to_r, stat.ctime.to_r]
        end

        def ensure_source_unchanged!(path, before)
          unless source_snapshot(path) == before
            raise TranscriptionRuntimeError,
                  "Source changed while processing: #{path}"
          end
        rescue SystemCallError => e
          raise TranscriptionRuntimeError, "Cannot re-check source #{path}: #{e.message}"
        end

        def evict_native_session
          @resources.evict_asr
        ensure
          @native_session = nil
        end

        def evict_current_asr_owner
          ModelResources.evict_current_asr_owner
        ensure
          @native_session = @resources.asr_session
        end

        def evict_aligner
          @aligner&.close
        ensure
          @aligner = nil
        end

        def evict_inference_sessions
          evict_native_session
        ensure
          evict_aligner
        end

        def monotonic
          Process.clock_gettime(Process::CLOCK_MONOTONIC)
        end
      end
    end
  end
end
