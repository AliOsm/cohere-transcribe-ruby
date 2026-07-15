# frozen_string_literal: true

require "digest"
require "etc"
require "json"
require "rbconfig"
require "securerandom"
require "tempfile"
require_relative "../python_text"

module Cohere
  module Transcribe
    module Output
      PublicationPlan = Data.define(
        :paths,
        :state_path,
        :checkpoint_path,
        :source_snapshot,
        :asr_contract_key,
        :render_contract_key,
        :lock_target,
        :directory_bindings,
        :skipped,
        :generation_id
      )
      PublicationDecision = Data.define(:action, :checkpoint, :generation_id, :reason)

      module Publication
        module_function

        OUTPUT_SCHEMA_VERSION = 8
        PROFILE_SCHEMA_VERSION = 9
        REPETITION_DETECTOR_VERSION = 3
        SILERO_VERSION = "6.2.1"
        SUCCESSFUL_BATCH_PROFILE_FIELDS = %w[
          segments processor_rows max_new_tokens generated_tokens generated_tokens_by_row
          prepare_seconds h2d_seconds generation_call_wall_seconds generate_device_seconds
          generation_analysis_seconds padded_audio_seconds padding_ratio peak_allocated_gib
          peak_reserved_gib
        ].freeze

        def plan(entries, options)
          publication = options.publication
          unless publication
            return entries.to_h do |entry|
              [
                entry.path,
                PublicationPlan.new(
                  paths: {}.freeze,
                  state_path: nil,
                  checkpoint_path: nil,
                  source_snapshot: nil,
                  asr_contract_key: nil,
                  render_contract_key: nil,
                  lock_target: nil,
                  directory_bindings: [].freeze,
                  skipped: false,
                  generation_id: nil
                )
              ]
            end.freeze
          end

          root = publication.output_dir&.expand_path
          if root
            root_binding = State.ensure_bound_directory(root)
            root = root_binding.canonical_path
          end

          claimed = {}
          input_paths = entries.to_h { |entry| [entry.path.expand_path.to_s, true] }
          profile_path = publication.profile_json&.expand_path
          profile_key = nil
          if profile_path
            if profile_path.symlink? || (profile_path.exist? && !profile_path.file?)
              raise TranscriptionInputError, "Profile path is not a regular file: #{profile_path}"
            end

            profile_key = prospective_realpath(profile_path).to_s
            if input_paths.key?(profile_key)
              raise TranscriptionInputError, "Profile path collides with an input audio file: #{profile_path}"
            end
          end
          asr_contract_key = State.asr_contract_key(options)
          render_contract_key = State.render_contract_key(options)
          plans = entries.to_h do |entry|
            source_snapshot = source_record(entry.path)
            parent = root ? root.join(entry.relative_path.dirname) : entry.path.dirname
            ensure_within_output_root!(prospective_realpath(parent), root, parent) if root
            parent_binding = State.ensure_bound_directory(parent, root_binding: root_binding)
            parent = parent_binding.canonical_path
            ensure_within_output_root!(parent, root, parent) if root
            directory_bindings = [root_binding, parent_binding].compact.uniq.freeze
            stem = entry.relative_path.basename(entry.relative_path.extname).to_s
            paths = publication.formats.to_h do |format|
              output_path = parent.join("#{stem}.#{format}")
              validate_output_path!(output_path, input_paths, entry.path)
              canonical = output_path.expand_path.to_s
              if (previous = claimed[canonical]) && previous != entry.path
                raise TranscriptionInputError,
                      "Output collision between #{previous} and #{entry.path}: #{output_path}"
              end
              if profile_key == canonical
                raise TranscriptionInputError,
                      "Profile path collides with a transcript output: #{output_path}"
              end
              claimed[canonical] = entry.path
              [format, output_path.freeze]
            end.freeze
            state_path = State.state_path_for_outputs(paths)
            checkpoint_path = State.checkpoint_path_for_outputs(paths)
            lock_target = State.lock_target_for_outputs(paths)
            {
              "State marker" => state_path,
              "ASR checkpoint" => checkpoint_path
            }.each do |label, reserved_path|
              if reserved_path.symlink? || (reserved_path.exist? && !reserved_path.file?)
                raise TranscriptionInputError, "#{label} is not a regular file: #{reserved_path}"
              end

              reserved_key = reserved_path.expand_path.to_s
              if profile_key == reserved_key
                raise TranscriptionInputError,
                      "Profile path collides with a reserved #{label.downcase}: #{reserved_path}"
              end
              if input_paths.key?(reserved_key) || claimed.key?(reserved_key)
                raise TranscriptionInputError, "#{label} collides with an input or output: #{reserved_path}"
              end

              claimed[reserved_key] = entry.path
            end

            existing = paths.values.select(&:exist?)
            if existing.any? && publication.existing == "error"
              rendered = existing.map { |path| "  #{path}" }.join("\n")
              raise TranscriptionInputError,
                    "Output already exists:\n#{rendered}\n" \
                    "Use existing: 'overwrite' or existing: 'skip'."
            end
            verification = if publication.existing == "skip" && existing.length == paths.length
                             State.verify_published_outputs(
                               source_snapshot: source_snapshot,
                               output_paths: paths,
                               state_path: state_path,
                               asr_contract_key: asr_contract_key,
                               render_contract_key: render_contract_key,
                               directory_binding: directory_bindings.last,
                               guard_bindings: directory_bindings
                             )
                           end
            skipped = verification&.verified? || false
            [
              entry.path,
              PublicationPlan.new(
                paths: paths,
                state_path: state_path.freeze,
                checkpoint_path: checkpoint_path.freeze,
                source_snapshot: source_snapshot,
                asr_contract_key: asr_contract_key,
                render_contract_key: render_contract_key,
                lock_target: lock_target,
                directory_bindings: directory_bindings,
                skipped: skipped,
                generation_id: verification&.generation_id
              )
            ]
          end
          plans.freeze
        rescue SystemCallError, ArgumentError, TranscriptionRuntimeError => e
          detail = e.message.to_s.delete("\0")
          raise TranscriptionInputError, "Cannot prepare output paths: #{detail}"
        end

        def with_plan_lock(plan, &block)
          return block.call unless plan.lock_target

          State.with_output_lock(plan.lock_target, &block)
        end

        def verify_plan_directory_continuity!(before_plans, after_plans)
          raise TranscriptionRuntimeError, "Publication inputs changed between planning passes" unless before_plans.keys == after_plans.keys

          before_plans.each do |path, before_plan|
            after_plan = after_plans.fetch(path)
            before_plan.directory_bindings.each(&:verify!)
            after_plan.directory_bindings.each(&:verify!)
            next if before_plan.directory_bindings == after_plan.directory_bindings

            raise TranscriptionRuntimeError,
                  "Publication parent identity changed between planning passes: #{path}"
          end
          nil
        end

        def verify_profile_directory_continuity!(before_binding, after_binding)
          return if before_binding.nil? && after_binding.nil?

          before_binding&.verify!
          after_binding&.verify!
          return if before_binding == after_binding

          raise TranscriptionRuntimeError,
                "Profile parent identity changed between planning passes"
        end

        def revalidate(plan, options)
          return PublicationDecision.new(action: :process, checkpoint: nil, generation_id: nil, reason: nil) if plan.paths.empty?

          plan.directory_bindings.each(&:verify!)
          State.ensure_source_unchanged!(plan.source_snapshot)
          validate_reserved_path!(plan.state_path, "State marker")
          validate_reserved_path!(plan.checkpoint_path, "ASR checkpoint")
          plan.paths.each_value { |path| validate_runtime_output_path!(path) }
          existing = plan.paths.values.select(&:exist?)
          if existing.any? && options.publication.existing == "error"
            rendered = existing.map { |path| "  #{path}" }.join("\n")
            raise TranscriptionInputError,
                  "Output already exists:\n#{rendered}\n" \
                  "Use existing: 'overwrite' or existing: 'skip'."
          end

          if options.publication.existing == "skip" && existing.length == plan.paths.length
            verification = State.verify_published_outputs(
              source_snapshot: plan.source_snapshot,
              output_paths: plan.paths,
              state_path: plan.state_path,
              asr_contract_key: plan.asr_contract_key,
              render_contract_key: plan.render_contract_key,
              directory_binding: plan.directory_bindings.last,
              guard_bindings: plan.directory_bindings
            )
            if verification.verified?
              return PublicationDecision.new(
                action: :skip,
                checkpoint: nil,
                generation_id: verification.generation_id,
                reason: nil
              )
            end
            publication_reason = verification.reason
          elsif existing.any? && options.publication.existing == "skip"
            publication_reason = "requested output set is incomplete"
          end

          checkpoint = State.restore_asr_checkpoint(
            path: plan.checkpoint_path,
            source_snapshot: plan.source_snapshot,
            asr_contract_key: plan.asr_contract_key,
            directory_binding: plan.directory_bindings.last,
            guard_bindings: plan.directory_bindings
          )
          if checkpoint.restored?
            return PublicationDecision.new(
              action: :resume,
              checkpoint: checkpoint.checkpoint,
              generation_id: checkpoint.checkpoint.generation_id,
              reason: publication_reason
            )
          end

          reasons = [publication_reason, checkpoint.reason].compact
          PublicationDecision.new(
            action: :process,
            checkpoint: nil,
            generation_id: nil,
            reason: reasons.empty? ? nil : reasons.join("; ").freeze
          )
        ensure
          plan.directory_bindings.each(&:verify!) if plan&.paths&.any?
        end

        def write(plan, result, options, generation_id: nil, speech_spans: nil)
          return [].freeze if plan.paths.empty?

          contents = plan.paths.to_h do |format, _path|
            [format, render(format, result, options, speech_spans: speech_spans)]
          end
          snapshot = plan.source_snapshot || source_record(result.path)
          generation_id = SecureRandom.hex(16) if generation_id.to_s.empty?
          manifest = State.published_manifest_content(
            source_snapshot: snapshot,
            output_paths: plan.paths,
            contents: contents,
            asr_contract_key: plan.asr_contract_key || State.asr_contract_key(options),
            render_contract_key: plan.render_contract_key || State.render_contract_key(options),
            generation_id: generation_id
          )
          transaction_paths = plan.paths.merge("__manifest__" => plan.state_path).freeze
          transaction_contents = contents.merge("__manifest__" => manifest).freeze
          atomic_write_set(
            transaction_paths,
            transaction_contents,
            before_commit: -> { State.ensure_source_unchanged!(snapshot) },
            directory_bindings: plan.directory_bindings
          )
          plan.paths.values.freeze
        end

        def write_profile(path, run, runtime_metrics: nil, directory_binding: nil)
          return unless path

          destination = Pathname(path).expand_path
          directory_binding ||= State.ensure_bound_directory(destination.dirname)
          payload = profile_payload(run, runtime_metrics: runtime_metrics)
          atomic_write_set(
            { "json" => destination },
            { "json" => Rendering.json(payload) },
            directory_bindings: [directory_binding]
          )
          destination
        rescue Interrupt, SystemExit
          raise
        rescue StandardError => e
          raise TranscriptionRuntimeError,
                "profile output failed: #{e.class}: Cannot write #{path}: #{e.message}"
        end

        def bind_profile_path(path)
          return unless path

          destination = Pathname(path).expand_path
          State.ensure_bound_directory(destination.dirname)
        rescue SystemCallError, ArgumentError, TranscriptionRuntimeError => e
          raise TranscriptionInputError, "Cannot prepare profile output path #{path}: #{e.message}"
        end

        def profile_payload(run, runtime_metrics: nil)
          requested = run.requested_options
          resolved = run.resolved_options
          statistics = run.statistics
          runtime_metrics ||= {}
          file_segmentation = runtime_metrics.fetch(:file_segmentation, {})
          results = run.results
          successful = results.reject { |result| result.status == "failed" }
          representative = results.find { |result| result.provenance.model_id } || results.first
          all_durations = results.flat_map do |result|
            profile_segment_durations(result, file_segmentation[result.path.to_s])
          end
          inferred_durations = results.reject do |result|
            result.provenance.resumed_from_asr_checkpoint
          end.flat_map do |result|
            profile_segment_durations(result, file_segmentation[result.path.to_s])
          end
          requested_engines = results.filter_map(&:provenance)
                                     .filter_map(&:vad_engine_requested).uniq.sort
          actual_engines = results.filter_map(&:provenance)
                                  .filter_map(&:vad_engine_actual).uniq.sort
          vad_prepared_groups = runtime_metrics.fetch(:vad_prepared_groups, 0)
          vad_model_calls = runtime_metrics.fetch(:vad_model_calls, 0)
          vad_valid_frames = runtime_metrics.fetch(:vad_valid_frames, 0)
          vad_padded_frames = runtime_metrics.fetch(:vad_padded_frames, 0)
          vad_max_files_per_call = runtime_metrics.fetch(:vad_max_files_per_call, 0)
          vad_effective_block_frames = runtime_metrics.fetch(:vad_effective_block_frames, 0)
          vad_intraop_threads = runtime_metrics.fetch(:vad_intraop_threads, 0)
          vad_padding_ratio = if vad_padded_frames.zero?
                                0.0
                              else
                                1.0 - vad_valid_frames.fdiv(vad_padded_frames)
                              end

          {
            "schema_version" => PROFILE_SCHEMA_VERSION,
            "created_unix_seconds" => Time.now.to_f,
            "implementation" => implementation_payload,
            "models" => profile_models_payload(representative, resolved),
            "environment" => environment_payload(resolved, runtime_metrics),
            "configuration" => configuration_payload(requested, results, resolved: false),
            "resolved_configuration" => configuration_payload(resolved, results, resolved: true),
            "run" => {
              "elapsed_seconds" => statistics.elapsed_seconds,
              "successful_files" => successful.length,
              "failed_files" => results.count { |result| result.status == "failed" },
              "successful_audio_seconds" => statistics.successful_audio_seconds,
              "real_time_factor_x" => statistics.real_time_factor_x
            },
            "timings" => timings_payload(statistics, runtime_metrics),
            "vad" => {
              "requested_engines" => requested_engines,
              "actual_engines" => actual_engines,
              "torch_device" => vad_prepared_groups.positive? ? "cpu" : nil,
              # This reference-schema field reports the effective CPU
              # intra-op count for Ruby's sequence-ONNX substitution.
              "torch_intraop_threads" => vad_intraop_threads.positive? ? vad_intraop_threads : nil,
              "configured_file_batch_size" => resolved.vad_batch_size,
              "configured_block_frames" => resolved.vad_block_frames,
              "effective_block_frames" => vad_effective_block_frames.positive? ? vad_effective_block_frames : nil,
              "prepared_groups" => vad_prepared_groups,
              "model_calls" => vad_model_calls,
              "valid_frames" => vad_valid_frames,
              "padded_frames" => vad_padded_frames,
              "padding_ratio" => vad_padding_ratio,
              "max_files_per_call" => vad_max_files_per_call
            },
            "asr" => {
              "batches" => statistics.asr_batches,
              "processor_rows" => statistics.asr_processor_rows,
              "generated_tokens" => statistics.generated_tokens,
              "valid_feature_frames" => nil,
              "padded_feature_frames" => nil,
              "discarded_processor_rows" => 0,
              "discarded_valid_feature_frames" => 0,
              "discarded_padded_feature_frames" => 0,
              "padding_ratio" => nil,
              "effective_batch_min" => runtime_metrics[:effective_batch_min],
              "effective_batch_max" => runtime_metrics[:effective_batch_max],
              "final_batch_size" => runtime_metrics[:final_batch_size],
              "final_batch_cap" => runtime_metrics[:final_batch_cap],
              "oom_retries" => statistics.oom_retries,
              "truncation_retries" => statistics.truncation_retries,
              "discarded_feature_batches" => 0,
              "pin_memory_fallbacks" => 0,
              "all_segment_duration_seconds" => duration_quantiles(all_durations),
              "inferred_segment_duration_seconds" => duration_quantiles(inferred_durations),
              "batch_history" => profile_batch_history(runtime_metrics[:batch_history]),
              "checkpoint_resumed_files" => results.count do |result|
                result.provenance.resumed_from_asr_checkpoint
              end,
              "checkpoint_written_files" => runtime_metrics.fetch(:checkpoint_written_files, 0)
            },
            "cuda_memory" => {
              "total_gib" => runtime_metrics[:cuda_total_gib],
              "free_start_gib" => runtime_metrics[:cuda_free_start_gib],
              "free_end_gib" => runtime_metrics[:cuda_free_end_gib],
              "peak_allocated_gib" => nil,
              "peak_reserved_gib" => nil
            },
            "files" => results.map do |result|
              profile_file_payload(
                result,
                resolved,
                segmentation: file_segmentation[result.path.to_s]
              )
            end
          }
        end

        def render(format, result, options, speech_spans: nil)
          case format
          when "txt"
            lines = result.segments.empty? ? [result.text.to_s] : result.segments.map(&:text)
            Rendering.plain_text(lines)
          when "srt" then Rendering.srt(result.cues)
          when "vtt" then Rendering.vtt(result.cues)
          when "json" then Rendering.json(result_payload(result, options, speech_spans: speech_spans))
          else
            raise TranscriptionRuntimeError, "Unsupported publication format: #{format.inspect}"
          end
        end

        def result_payload(result, options, speech_spans: nil)
          provenance = result.provenance
          {
            "schema_version" => OUTPUT_SCHEMA_VERSION,
            "implementation" => implementation_payload,
            "source" => {
              "path" => result.path.to_s,
              "duration_seconds" => result.duration,
              "sample_rate" => SAMPLE_RATE,
              "decode_backend" => provenance.decode_backend,
              "decode_fallback_reason" => provenance.decode_fallback_reason
            },
            "language" => options.language,
            "segmentation" => options.vad,
            "segmentation_details" => {
              "mode" => options.vad,
              "requested_engine" => provenance.vad_engine_requested,
              "actual_engine" => provenance.vad_engine_actual,
              "provider" => provenance.vad_provider,
              "provider_options" => vad_provider_options(provenance, options),
              "fallback_reason" => provenance.vad_fallback_reason,
              "merge" => options.vad_merge,
              "parameters" => segmentation_parameters(options),
              "speech_spans" => speech_spans_payload(result, speech_spans)
            },
            "timing" => options.alignment,
            "models" => {
              "asr" => {
                "id" => provenance.model_id,
                "revision" => provenance.model_revision,
                "format" => provenance.model_format,
                "quantization" => nil,
                "adapter" => provenance.adapter_id && {
                  "id" => provenance.adapter_id,
                  "revision" => provenance.adapter_revision
                }
              },
              "vad" => output_vad_model_payload(options, provenance),
              "aligner" => aligner_payload(options)
            },
            "fallback_alignment_segments" => provenance.fallback_alignment_segments,
            "repetition_detector_version" => REPETITION_DETECTOR_VERSION,
            "repetition_stopped_segments" => provenance.repetition_stopped_segments,
            "truncation_retried_segments" => provenance.truncation_retried_segments,
            "token_limit_segments" => provenance.token_limit_segments,
            "generated_tokens_by_segment" => provenance.generated_tokens_by_segment.map do |index, tokens|
              { "segment_index" => index, "tokens" => tokens }
            end,
            "transcript" => transcript_lines(result),
            "segments" => result.segments.filter_map do |segment|
              text = PythonText.strip(segment.text.to_s)
              next if text.empty?

              {
                "segment_index" => segment.index,
                "start" => segment.start,
                "end" => segment.end,
                "text" => text
              }
            end,
            "words" => result.words.map do |word|
              {
                "start" => word.start,
                "end" => word.end,
                "text" => word.text,
                "segment_index" => word.segment_index,
                "segment_word_index" => word.segment_word_index,
                "timing_source" => word.timing_source
              }
            end,
            "cues" => result.cues.map do |cue|
              { "start" => cue.start, "end" => cue.end, "text" => cue.text }
            end
          }
        end

        def implementation_payload
          @implementation_payload ||= begin
            behavior_root = Pathname(__dir__).parent
            package_root = behavior_root.parent.parent
            suffixes = %w[.rb .onnx .so .bundle .dylib .dll]
            artifacts = Dir.glob(behavior_root.join("**", "*").to_s).filter_map do |name|
              path = Pathname(name)
              next unless path.file? && suffixes.include?(path.extname)

              [path.relative_path_from(package_root).to_s.tr(File::SEPARATOR, "/"), Digest::SHA256.file(path).hexdigest]
            end.to_h
            artifacts.each do |key, value|
              key.freeze
              value.freeze
            end
            artifacts.freeze
            {
              "package_version" => VERSION,
              "artifacts_sha256" => artifacts
            }.freeze
          end
        end

        def configuration_payload(options, results, resolved:)
          publication = options.publication
          formats = publication&.formats
          formats ||= options.text_only || options.alignment == "none" ? ["txt"] : %w[txt srt vtt]
          payload = options.class.members.each_with_object({}) do |member, values|
            next if member == :publication

            values[member.to_s] = json_value(options.public_send(member))
          end
          representative = results.find { |result| result.provenance.model_id }
          payload.merge(
            "audio" => results.map { |result| result.path.to_s },
            "formats" => formats,
            "output_dir" => publication&.output_dir&.to_s,
            "existing" => publication&.existing || "error",
            "profile_json" => publication&.profile_json&.to_s,
            "model_format" => resolved ? representative&.provenance&.model_format : nil,
            "model_quantization" => nil
          )
        end
        private_class_method :configuration_payload

        def timings_payload(statistics, runtime_metrics)
          {
            "runtime_import_seconds" => statistics.runtime_import_seconds,
            "serialization_wait_seconds" => statistics.serialization_wait_seconds,
            "input_validation_seconds" => statistics.input_validation_seconds,
            "decode_worker_seconds" => statistics.decode_seconds,
            "vad_worker_seconds" => statistics.vad_seconds,
            "vad_model_load_seconds" => runtime_metrics[:vad_model_load_seconds],
            "vad_inference_seconds" => runtime_metrics[:vad_inference_seconds],
            "vad_postprocess_seconds" => runtime_metrics[:vad_postprocess_seconds],
            "preparation_wait_seconds" => runtime_metrics[:preparation_wait_seconds],
            "asr_load_seconds" => statistics.asr_load_seconds,
            "asr_wall_seconds" => statistics.asr_seconds,
            "asr_feature_worker_seconds" => runtime_metrics[:asr_feature_worker_seconds],
            "asr_discarded_feature_seconds" => 0.0,
            "asr_feature_wait_seconds" => 0.0,
            "asr_h2d_seconds" => runtime_metrics[:asr_h2d_seconds],
            "asr_generation_call_wall_seconds" => runtime_metrics[:asr_generation_call_seconds],
            "asr_generate_device_seconds" => runtime_metrics[:asr_generate_device_seconds],
            "asr_generation_analysis_seconds" => runtime_metrics[:asr_generation_analysis_seconds],
            "asr_decode_seconds" => nil,
            "aligner_load_seconds" => statistics.aligner_load_seconds,
            "emissions_seconds" => statistics.emissions_seconds,
            "viterbi_seconds" => statistics.viterbi_seconds,
            "post_asr_seconds" => runtime_metrics[:post_asr_seconds],
            "checkpoint_seconds" => runtime_metrics[:checkpoint_seconds],
            "progressive_output_seconds" => runtime_metrics[:progressive_output_seconds]
          }
        end
        private_class_method :timings_payload

        def environment_payload(options, runtime_metrics)
          specs = Gem.loaded_specs
          packages = {
            "cohere-transcribe" => VERSION,
            "numo-narray" => specs["numo-narray"]&.version&.to_s,
            "onnxruntime" => specs["onnxruntime"]&.version&.to_s
          }.compact
          environment = {
            "python" => nil,
            "executable" => RbConfig.ruby,
            "platform" => RUBY_PLATFORM,
            "machine" => RbConfig::CONFIG["host_cpu"],
            "processor" => RbConfig::CONFIG["host_cpu"],
            "cpu_count" => Etc.nprocessors,
            "device" => options.device,
            "dtype" => options.dtype,
            "packages" => packages,
            "torch_cuda_build" => nil,
            "pytorch_alloc_conf" => nil,
            "pytorch_cuda_alloc_conf" => nil,
            "pytorch_effective_alloc_conf" => nil,
            "torch_intraop_threads" => nil,
            "torch_interop_threads" => nil
          }
          if options.device == "cuda"
            environment["cuda"] = {
              "device_index" => nil,
              "name" => nil,
              "compute_capability" => nil,
              "total_memory_gib" => runtime_metrics[:cuda_total_gib],
              "free_memory_at_profile_gib" => runtime_metrics[:cuda_free_end_gib] ||
                                              runtime_metrics[:cuda_free_start_gib],
              "driver_visible_device_count" => nil,
              "cudnn_version" => nil,
              "allocator_backend" => nil
            }
          end
          environment
        end
        private_class_method :environment_payload

        def profile_batch_history(entries)
          return entries unless entries

          entries.map do |entry|
            next entry unless entry.is_a?(Hash) && !entry.key?("event")

            SUCCESSFUL_BATCH_PROFILE_FIELDS.to_h { |field| [field, entry[field]] }.merge(entry)
          end
        end
        private_class_method :profile_batch_history

        def profile_models_payload(result, options)
          provenance = result&.provenance || TranscriptionProvenance.new
          {
            "asr" => {
              "id" => provenance.model_id || json_value(options.model),
              "revision" => provenance.model_revision || options.model_revision,
              "format" => provenance.model_format,
              "quantization" => nil,
              "adapter" => provenance.adapter_id && {
                "id" => provenance.adapter_id,
                "revision" => provenance.adapter_revision
              }
            },
            "vad" => profile_vad_model_payload(options),
            "aligner" => aligner_payload(options)
          }
        end
        private_class_method :profile_models_payload

        def output_vad_model_payload(options, provenance)
          return nil unless options.vad == "silero"

          {
            "source" => "silero-vad",
            "source_version" => SILERO_VERSION,
            "distribution" => "cohere-transcribe",
            "version" => VERSION,
            "weight_asset" => "cohere/transcribe/vad/silero_vad_v6.onnx",
            "implementation" => provenance.vad_engine_actual&.start_with?("onnx") ? "sequence-onnx" : nil
          }
        end
        private_class_method :output_vad_model_payload

        def profile_vad_model_payload(options)
          return nil unless options.vad == "silero"

          {
            "source" => "silero-vad",
            "source_version" => SILERO_VERSION,
            "distribution" => "cohere-transcribe",
            "version" => VERSION,
            "torch_weight_asset" => nil,
            "onnx_weight_asset" => "cohere/transcribe/vad/silero_vad_v6.onnx",
            "packed_torch_implementation" => nil
          }
        end
        private_class_method :profile_vad_model_payload

        def aligner_payload(options)
          return nil unless options.alignment == "word"

          {
            "id" => Alignment::ModelProvider::SOURCE_REPOSITORY,
            "revision" => Alignment::ModelProvider::SOURCE_REVISION,
            "kernel" => {
              "distribution" => "cohere-transcribe",
              "operation" => "Cohere::Transcribe::Alignment::CTC.forced_align",
              "version" => VERSION
            },
            "utility_package" => {
              "distribution" => "cohere-transcribe",
              "location" => "cohere/transcribe/alignment",
              "repository" => Alignment::ModelProvider::UTILITY_REPOSITORY,
              "revision" => Alignment::ModelProvider::UTILITY_REVISION
            },
            "romanizer" => {
              "distribution" => "cohere-transcribe",
              "version" => Alignment::ModelProvider::UROMAN_COMPATIBILITY_VERSION
            }
          }
        end
        private_class_method :aligner_payload

        def vad_provider_options(provenance, _options)
          providers = provenance.vad_provider.to_s.split(",").reject(&:empty?)
          return nil if providers.empty?

          providers.to_h { |provider| [provider, {}] }
        end
        private_class_method :vad_provider_options

        def segmentation_parameters(options)
          parameters = { "max_duration_seconds" => options.max_dur }
          case options.vad
          when "silero"
            parameters.merge!(
              "min_duration_seconds" => options.min_dur,
              "threshold" => options.vad_threshold,
              "min_silence_ms" => options.min_silence_ms,
              "speech_pad_ms" => options.speech_pad_ms
            )
          when "auditok"
            parameters.merge!(
              "min_duration_seconds" => options.min_dur,
              "max_silence_seconds" => options.max_silence,
              "energy_threshold" => options.energy_threshold
            )
          end
          parameters
        end
        private_class_method :segmentation_parameters

        def speech_spans_payload(result, speech_spans)
          spans = speech_spans || result.segments.map { |segment| [segment.start, segment.end] }
          spans.map { |start_time, end_time| { "start" => start_time, "end" => end_time } }
        end
        private_class_method :speech_spans_payload

        def transcript_lines(result)
          lines = result.segments.map(&:text)
          lines = result.text.to_s.lines if lines.empty?
          lines.filter_map do |line|
            stripped = PythonText.strip(line.to_s)
            stripped unless stripped.empty?
          end
        end
        private_class_method :transcript_lines

        def profile_file_payload(result, options, segmentation: nil)
          segment_times = segmentation&.fetch(:segment_times, nil)
          speech_spans = segmentation&.fetch(:speech_spans, nil)
          durations = segment_times ? span_durations(segment_times) : segment_durations(result)
          raw_available = !speech_spans.nil? || !options.vad_merge
          raw_durations = speech_spans ? span_durations(speech_spans) : durations
          {
            "path" => result.path.to_s,
            "relative_path" => result.relative_path.to_s,
            "duration_seconds" => result.duration,
            "segment_count" => segment_times ? segment_times.length : result.segments.length,
            "raw_speech_span_count" => raw_available ? raw_durations.length : nil,
            "raw_speech_seconds" => raw_available ? raw_durations.sum : nil,
            "selected_audio_seconds" => durations.sum,
            "decode_backend" => result.provenance.decode_backend,
            "decode_fallback_reason" => result.provenance.decode_fallback_reason,
            "vad_engine" => result.provenance.vad_engine_actual,
            "vad_provider" => result.provenance.vad_provider,
            "vad_provider_options" => vad_provider_options(result.provenance, options),
            "vad_fallback_reason" => result.provenance.vad_fallback_reason,
            "generated_tokens" => result.provenance.generated_tokens_by_segment.sum { |_index, count| count },
            "repetition_stopped_segments" => result.provenance.repetition_stopped_segments,
            "truncation_retried_segments" => result.provenance.truncation_retried_segments,
            "token_limit_segments" => result.provenance.token_limit_segments,
            "fallback_alignment_segments" => result.provenance.fallback_alignment_segments,
            "outputs" => result.outputs.map(&:to_s),
            "resumed_from_asr_checkpoint" => result.provenance.resumed_from_asr_checkpoint,
            "published" => result.provenance.published,
            "error" => result.error
          }
        end
        private_class_method :profile_file_payload

        def segment_durations(result)
          result.segments.map { |segment| [segment.end - segment.start, 0.0].max }
        end
        private_class_method :segment_durations

        def profile_segment_durations(result, segmentation)
          return segment_durations(result) unless segmentation

          span_durations(segmentation.fetch(:segment_times))
        end
        private_class_method :profile_segment_durations

        def span_durations(spans)
          spans.map { |start_time, end_time| [end_time - start_time, 0.0].max }
        end
        private_class_method :span_durations

        def duration_quantiles(values)
          sorted = values.map(&:to_f).sort
          return nil if sorted.empty?

          {
            "min" => sorted.first,
            "p50" => quantile(sorted, 0.50),
            "p90" => quantile(sorted, 0.90),
            "p99" => quantile(sorted, 0.99),
            "max" => sorted.last
          }
        end
        private_class_method :duration_quantiles

        def quantile(sorted, fraction)
          position = (sorted.length - 1) * fraction
          lower = position.floor
          upper = position.ceil
          return sorted.fetch(lower) if lower == upper

          sorted.fetch(lower) + ((sorted.fetch(upper) - sorted.fetch(lower)) * (position - lower))
        end
        private_class_method :quantile

        def atomic_write_set(paths, contents, before_commit: nil, rename: nil,
                             directory_bindings: nil, operation_hook: nil)
          staged = {}
          backups = {}
          committed = []
          preserved_backups = {}
          open_handles = []
          failed = false
          bound_directories = publication_bound_directories(paths, directory_bindings)
          operation_hook&.call(:directories_opened, nil)
          paths.each do |format, supplied_destination|
            destination = Pathname(supplied_destination).expand_path
            bound = bound_directories.fetch(destination.dirname)
            bound.verify!
            output_mode = bound_output_mode(bound, destination.basename.to_s)
            temporary_name = nil
            temporary = nil
            Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
              temporary_name, temporary = bound.create_temporary(destination.basename.to_s, ".tmp")
              staged[destination] = [bound, temporary_name].freeze
              open_handles << temporary
            end
            begin
              temporary.chmod(output_mode)
              operation_hook&.call(:before_stage_write, destination)
              temporary.write(contents.fetch(format))
              temporary.flush
              temporary.fsync
            ensure
              temporary.close unless temporary.closed?
            end
            bound.verify!
          end

          staged.each do |destination, (bound, _temporary_name)|
            operation_hook&.call(:before_backup_open, destination)
            source = nil
            Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
              source = begin
                bound.open_regular(destination.basename.to_s)
              rescue Errno::ENOENT
                nil
              end
              open_handles << source if source
            end
            unless source
              backups[destination] = nil
              next
            end

            backup_name = nil
            backup = nil
            Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
              backup_name, backup = bound.create_temporary(destination.basename.to_s, ".bak")
              backups[destination] = [bound, backup_name].freeze
              open_handles << backup
            end
            begin
              backup.chmod(source.stat.mode & 0o7777)
              operation_hook&.call(:before_backup_copy, destination)
              IO.copy_stream(source, backup)
              backup.flush
              backup.fsync
            ensure
              source.close
              backup.close
            end
            bound.verify!
          end
          before_commit&.call
          bound_directories.each_value(&:verify!)
          staged.each do |destination, (bound, temporary_name)|
            operation_hook&.call(:before_rename, destination)
            if rename
              Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
                committed << destination
                rename.call(bound.display_path(temporary_name), destination)
                bound.rename(temporary_name, destination.basename.to_s) if bound.regular_entry?(temporary_name)
              end
            else
              Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
                bound.rename(temporary_name, destination.basename.to_s)
                committed << destination
              end
            end
            operation_hook&.call(:after_rename, destination)
            bound.verify!
          end
          bound_directories.each_value(&:fsync)
          bound_directories.each_value(&:verify!)
        rescue Exception => e # rubocop:disable Lint/RescueException -- rollback must include interrupts
          failed = true
          rollback_errors = []
          committed.reverse_each do |destination|
            backup = backups[destination]
            bound, backup_name = backup
            bound ||= staged.fetch(destination).first
            begin
              Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
                bound.unlink(destination.basename.to_s, missing_ok: true)
                bound.rename(backup_name, destination.basename.to_s) if backup_name
              end
            rescue SystemCallError, TranscriptionRuntimeError => rollback_error
              preserved_backups[backup] = true if backup_name
              rollback_errors << "#{destination}: #{rollback_error.message}"
            end
          end
          begin
            bound_directories&.each_value(&:fsync)
          rescue SystemCallError, TranscriptionRuntimeError => rollback_error
            rollback_errors << "directory sync: #{rollback_error.message}"
          end
          if rollback_errors.any?
            detail = rollback_errors.join("; ")
            retained = preserved_backups.keys.filter_map do |backup|
              backup&.then { |bound, name| bound.display_path(name).to_s }
            end.sort
            raise TranscriptionRuntimeError,
                  "Output commit failed and rollback was incomplete (#{detail}); " \
                  "preserved backups: #{retained}",
                  cause: e
          end
          raise e
        ensure
          cleanup_errors = []
          staged&.each_value do |bound, name|
            bound.unlink(name, missing_ok: true)
          rescue SystemCallError, TranscriptionRuntimeError => e
            cleanup_errors << e
          end
          backups&.each_value do |backup|
            next unless backup && !preserved_backups&.key?(backup)

            bound, name = backup
            bound.unlink(name, missing_ok: true)
          rescue SystemCallError, TranscriptionRuntimeError => e
            cleanup_errors << e
          end
          open_handles&.each { |handle| handle.close unless handle.closed? }
          bound_directories&.each_value(&:close)
          raise cleanup_errors.first if cleanup_errors.any? && !failed
        end

        def publication_bound_directories(paths, directory_bindings)
          guards = Array(directory_bindings).compact.uniq.freeze
          bound_directories = {}
          paths.each_value.map { |path| Pathname(path).expand_path.dirname }.uniq.each do |parent|
            binding = guards.rfind do |candidate|
              candidate.access_path == parent || candidate.canonical_path == parent
            end
            binding ||= State.ensure_bound_directory(parent)
            active_guards = (guards + [binding]).uniq
            Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
              bound_directories[parent] = State::BoundDirectory.open(binding, guards: active_guards)
            end
          end
          bound_directories.freeze
        rescue Exception # rubocop:disable Lint/RescueException -- close partially opened directory handles
          bound_directories&.each_value(&:close)
          raise
        end
        private_class_method :publication_bound_directories

        def bound_output_mode(bound, basename)
          existing = nil
          Thread.handle_interrupt(State::DEFERRED_PUBLICATION_EXCEPTIONS) do
            existing = bound.open_regular(basename)
          end
          existing.stat.mode & 0o7777
        rescue Errno::ENOENT
          current_umask = File.umask
          File.umask(current_umask)
          0o666 & ~current_umask
        ensure
          existing&.close
        end
        private_class_method :bound_output_mode

        def output_mode(destination)
          return destination.stat.mode & 0o7777 if destination.exist?

          current_umask = File.umask
          File.umask(current_umask)
          0o666 & ~current_umask
        end
        private_class_method :output_mode

        def fsync_directories(directories)
          directories.each do |directory|
            File.open(directory, File::RDONLY, &:fsync)
          rescue Errno::EACCES, Errno::EBADF, Errno::EINVAL, Errno::EISDIR,
                 Errno::ENOTSUP, Errno::EPERM
            next
          end
        end
        private_class_method :fsync_directories

        def verified_publication?(source_snapshot, paths, state_path, options)
          State.verify_published_outputs(
            source_snapshot: source_snapshot,
            output_paths: paths,
            state_path: state_path,
            asr_contract_key: State.asr_contract_key(options),
            render_contract_key: State.render_contract_key(options)
          ).verified?
        end
        private_class_method :verified_publication?

        def source_record(path)
          State::SourceSnapshot.capture(path)
        end
        private_class_method :source_record

        def json_value(value)
          case value
          when Pathname then value.to_s
          when Array then value.map { |item| json_value(item) }
          when Hash then value.to_h { |key, item| [key.to_s, json_value(item)] }
          else value
          end
        end
        private_class_method :json_value

        def validate_reserved_path!(path, label)
          return unless path.symlink? || (path.exist? && !path.file?)

          raise TranscriptionInputError, "#{label} is not a regular file: #{path}"
        end
        private_class_method :validate_reserved_path!

        def validate_runtime_output_path!(path)
          return unless path.symlink? || (path.exist? && !path.file?)

          raise TranscriptionInputError, "Output path is not a regular file: #{path}"
        end
        private_class_method :validate_runtime_output_path!

        def validate_output_path!(path, input_paths, source)
          raise TranscriptionInputError, "Output path is not a regular file: #{path}" if path.symlink? || (path.exist? && !path.file?)
          return unless input_paths.key?(path.expand_path.to_s)

          raise TranscriptionInputError, "Output path for #{source} collides with an input: #{path}"
        end
        private_class_method :validate_output_path!

        def prospective_realpath(path)
          path = Pathname(path).expand_path
          missing = []
          cursor = path
          until cursor.exist? || cursor.symlink?
            parent = cursor.dirname
            raise ArgumentError, "Cannot resolve output directory #{path}" if parent == cursor

            missing.unshift(cursor.basename)
            cursor = parent
          end
          missing.reduce(cursor.realpath) { |resolved, component| resolved.join(component) }
        end
        private_class_method :prospective_realpath

        def ensure_within_output_root!(path, root, supplied)
          relative = Pathname(path).relative_path_from(Pathname(root))
          return unless relative.each_filename.first == ".."

          raise TranscriptionInputError, "Output directory escapes output root: #{supplied} -> #{path}"
        rescue ArgumentError
          raise TranscriptionInputError, "Output directory escapes output root: #{supplied} -> #{path}"
        end
        private_class_method :ensure_within_output_root!
      end
    end
  end
end
