# frozen_string_literal: true

require "tmpdir"
require "timeout"
require "test_helper"

module Cohere
  module Transcribe
    module Output
      class PublicationSchemaTest < Minitest::Test
        RESULT_KEYS = %w[
          schema_version implementation source language segmentation
          segmentation_details timing models fallback_alignment_segments
          repetition_detector_version repetition_stopped_segments
          truncation_retried_segments token_limit_segments
          generated_tokens_by_segment transcript segments words cues
        ].sort.freeze
        PROFILE_KEYS = %w[
          schema_version created_unix_seconds implementation models environment
          configuration resolved_configuration run timings vad asr cuda_memory files
        ].sort.freeze

        def test_result_payload_matches_the_reference_v8_key_contract
          options = options_with_publication
          payload = Publication.result_payload(result, options)

          assert_equal 8, payload.fetch("schema_version")
          assert_equal RESULT_KEYS, payload.keys.sort
          assert_equal %w[artifacts_sha256 package_version], payload.fetch("implementation").keys.sort
          assert_equal(
            %w[mode requested_engine actual_engine provider provider_options fallback_reason merge parameters speech_spans].sort,
            payload.fetch("segmentation_details").keys.sort
          )
          assert_equal %w[aligner asr vad], payload.fetch("models").keys.sort
          assert_equal %w[adapter format id quantization revision], payload.dig("models", "asr").keys.sort
          assert_nil payload.dig("models", "asr", "quantization")
          assert_nil payload.dig("models", "vad")
          assert_nil payload.dig("models", "aligner")
          assert_nil payload.dig("segmentation_details", "provider_options")
          assert_equal 3, payload.fetch("repetition_detector_version")
          assert_equal ["hello world"], payload.fetch("transcript")
          assert_equal [{ "start" => 0.0, "end" => 1.0 }], payload.dig("segmentation_details", "speech_spans")
        end

        def test_result_payload_omits_blank_segments_without_renumbering_them
          options = options_with_publication
          blank = TranscriptionSegment.new(index: 0, start: 0.0, end: 0.5, text: "  ")
          spoken = result.segments.first.with(index: 1, start: 0.5)
          payload = Publication.result_payload(result.with(segments: [blank, spoken]), options)
          segment_indices = payload.fetch("segments").map { |segment| segment.fetch("segment_index") }

          assert_equal [1], segment_indices
          assert_equal [
            { "start" => 0.0, "end" => 0.5 },
            { "start" => 0.5, "end" => 1.0 }
          ], payload.dig("segmentation_details", "speech_spans")
        end

        def test_result_payload_retains_raw_speech_spans_when_selected_segments_were_merged
          options = options_with_publication.with(vad: "silero", vad_merge: true)
          merged = result.with(
            segments: [TranscriptionSegment.new(index: 0, start: 0.1, end: 0.9, text: "hello world")]
          )
          raw = [[0.1, 0.3], [0.6, 0.9]]

          payload = Publication.result_payload(merged, options, speech_spans: raw)

          assert_equal(
            [{ "start" => 0.1, "end" => 0.3 }, { "start" => 0.6, "end" => 0.9 }],
            payload.dig("segmentation_details", "speech_spans")
          )
        end

        def test_silero_result_reports_real_provider_options_without_session_options
          options = options_with_publication.with(
            vad: "silero",
            vad_engine: "onnx",
            vad_batch_size: 1,
            vad_block_frames: 300,
            vad_threads: nil
          )
          provenance = result.provenance.with(
            vad_engine_requested: "torch",
            vad_engine_actual: "onnx",
            vad_provider: "CPUExecutionProvider"
          )

          payload = Publication.result_payload(result.with(provenance: provenance), options)

          assert_equal(
            { "CPUExecutionProvider" => {} },
            payload.dig("segmentation_details", "provider_options")
          )
        end

        def test_word_aligner_provenance_reports_the_mms_onnx_and_ruby_ctc_path
          options = options_with_publication.with(alignment: "word")
          payload = Publication.result_payload(result, options)
          aligner = payload.dig("models", "aligner")

          assert_equal Alignment::ModelProvider::SOURCE_REPOSITORY, aligner.fetch("id")
          assert_equal Alignment::ModelProvider::SOURCE_REVISION, aligner.fetch("revision")
          assert_equal "cohere-transcribe", aligner.dig("kernel", "distribution")
          assert_equal "Cohere::Transcribe::Alignment::CTC.forced_align",
                       aligner.dig("kernel", "operation")
          assert_equal Alignment::ModelProvider::UTILITY_REVISION,
                       aligner.dig("utility_package", "revision")
          assert_equal Alignment::ModelProvider::UROMAN_COMPATIBILITY_VERSION,
                       aligner.dig("romanizer", "version")
        end

        def test_profile_payload_matches_the_reference_v9_section_contract
          options = options_with_publication
          run = TranscriptionRun.new(
            results: [result],
            requested_options: options.with(device: "auto", dtype: "auto"),
            resolved_options: options.with(device: "cpu", dtype: "fp32"),
            statistics: statistics
          )
          payload = Publication.profile_payload(run)

          assert_equal 9, payload.fetch("schema_version")
          assert_equal PROFILE_KEYS, payload.keys.sort
          assert_equal "auto", payload.dig("configuration", "device")
          assert_equal "cpu", payload.dig("resolved_configuration", "device")
          assert_equal "cpu", payload.dig("environment", "device")
          assert_equal 1, payload.dig("run", "successful_files")
          assert_equal 1, payload.dig("asr", "batches")
          assert_nil payload.dig("asr", "valid_feature_frames")
          assert_equal 0, payload.dig("asr", "discarded_processor_rows")
          assert_equal 0, payload.dig("asr", "discarded_feature_batches")
          assert_equal 0, payload.dig("asr", "pin_memory_fallbacks")
          assert_equal({ "min" => 1.0, "p50" => 1.0, "p90" => 1.0, "p99" => 1.0, "max" => 1.0 },
                       payload.dig("asr", "all_segment_duration_seconds"))
          assert_equal %w[
            runtime_import_seconds serialization_wait_seconds input_validation_seconds
            decode_worker_seconds vad_worker_seconds vad_model_load_seconds
            vad_inference_seconds vad_postprocess_seconds preparation_wait_seconds
            asr_load_seconds asr_wall_seconds asr_feature_worker_seconds
            asr_discarded_feature_seconds asr_feature_wait_seconds asr_h2d_seconds
            asr_generation_call_wall_seconds asr_generate_device_seconds
            asr_generation_analysis_seconds asr_decode_seconds aligner_load_seconds
            emissions_seconds viterbi_seconds post_asr_seconds checkpoint_seconds
            progressive_output_seconds
          ].sort, payload.fetch("timings").keys.sort
          assert_equal %w[
            path relative_path duration_seconds segment_count raw_speech_span_count
            raw_speech_seconds selected_audio_seconds decode_backend
            decode_fallback_reason vad_engine vad_provider vad_provider_options
            vad_fallback_reason generated_tokens repetition_stopped_segments
            truncation_retried_segments token_limit_segments
            fallback_alignment_segments outputs resumed_from_asr_checkpoint
            published error
          ].sort, payload.fetch("files").first.keys.sort
        end

        def test_profile_counts_verified_skips_as_successful_files
          options = options_with_publication
          skipped = result.with(status: "skipped", text: nil)
          run = TranscriptionRun.new(
            results: [result, skipped],
            requested_options: options,
            resolved_options: options,
            statistics: statistics
          )

          payload = Publication.profile_payload(run)
          assert_equal 2, payload.dig("run", "successful_files")
          assert_equal 0, payload.dig("run", "failed_files")
        end

        def test_profile_accepts_private_runtime_batch_telemetry_without_expanding_public_statistics
          options = options_with_publication
          run = TranscriptionRun.new(
            results: [result],
            requested_options: options,
            resolved_options: options,
            statistics: statistics
          )
          batch_history = [
            { "segments" => 4, "max_new_tokens" => 445, "generated_tokens" => 17 },
            { "event" => "oom", "segments" => 8, "max_new_tokens" => 445 }
          ]

          payload = Publication.profile_payload(
            run,
            runtime_metrics: {
              effective_batch_min: 2,
              effective_batch_max: 8,
              final_batch_size: 4,
              final_batch_cap: 7,
              batch_history: batch_history,
              checkpoint_written_files: 1,
              file_segmentation: {
                result.path.to_s => {
                  segment_times: [[0.1, 0.9]],
                  speech_spans: [[0.1, 0.3], [0.6, 0.9]]
                }
              }
            }
          )

          assert_equal 2, payload.dig("asr", "effective_batch_min")
          assert_equal 8, payload.dig("asr", "effective_batch_max")
          assert_equal 4, payload.dig("asr", "final_batch_size")
          assert_equal 7, payload.dig("asr", "final_batch_cap")
          successful_batch, failed_batch = payload.dig("asr", "batch_history")
          assert_equal %w[
            segments processor_rows max_new_tokens generated_tokens generated_tokens_by_row
            prepare_seconds h2d_seconds generation_call_wall_seconds generate_device_seconds
            generation_analysis_seconds padded_audio_seconds padding_ratio peak_allocated_gib
            peak_reserved_gib
          ].sort, successful_batch.keys.sort
          assert_equal 4, successful_batch.fetch("segments")
          assert_equal 445, successful_batch.fetch("max_new_tokens")
          assert_equal 17, successful_batch.fetch("generated_tokens")
          assert_nil successful_batch.fetch("processor_rows")
          assert_equal batch_history.last, failed_batch
          assert_equal 1, payload.dig("asr", "checkpoint_written_files")
          assert_equal 2, payload.fetch("files").first.fetch("raw_speech_span_count")
          assert_in_delta 0.5, payload.fetch("files").first.fetch("raw_speech_seconds")
          assert_in_delta 0.8, payload.fetch("files").first.fetch("selected_audio_seconds")
          refute_includes TranscriptionStatistics.members, :batch_history
        end

        def test_cuda_profile_retains_the_reference_environment_field_contract
          options = options_with_publication.with(device: "cuda", dtype: "bf16")
          run = TranscriptionRun.new(
            results: [result],
            requested_options: options,
            resolved_options: options,
            statistics: statistics
          )

          payload = Publication.profile_payload(
            run,
            runtime_metrics: {
              cuda_total_gib: 12.0,
              cuda_free_start_gib: 8.0,
              cuda_free_end_gib: 7.5
            }
          )
          cuda = payload.dig("environment", "cuda")
          assert_equal %w[
            device_index name compute_capability total_memory_gib free_memory_at_profile_gib
            driver_visible_device_count cudnn_version allocator_backend
          ].sort, cuda.keys.sort
          assert_equal 12.0, cuda.fetch("total_memory_gib")
          assert_equal 7.5, cuda.fetch("free_memory_at_profile_gib")
          assert cuda.values_at(
            "device_index", "name", "compute_capability", "driver_visible_device_count",
            "cudnn_version", "allocator_backend"
          ).all?(&:nil?)
          assert_equal(
            {
              "total_gib" => 12.0,
              "free_start_gib" => 8.0,
              "free_end_gib" => 7.5,
              "peak_allocated_gib" => nil,
              "peak_reserved_gib" => nil
            },
            payload.fetch("cuda_memory")
          )
        end

        def test_profile_reports_effective_sequence_onnx_tuning_and_single_stream_calls
          requested = options_with_publication.with(
            vad: "silero",
            vad_engine: "torch",
            vad_batch_size: 2,
            vad_block_frames: 300,
            vad_threads: 4
          )
          resolved = requested.with(vad_engine: "onnx")
          provenance = result.provenance.with(
            vad_engine_requested: "torch",
            vad_engine_actual: "onnx",
            vad_provider: "CPUExecutionProvider"
          )
          run = TranscriptionRun.new(
            results: [result.with(provenance: provenance)],
            requested_options: requested,
            resolved_options: resolved,
            statistics: statistics
          )

          payload = Publication.profile_payload(
            run,
            runtime_metrics: {
              vad_prepared_groups: 1,
              vad_model_calls: 3,
              vad_valid_frames: 601,
              vad_padded_frames: 601,
              vad_max_files_per_call: 1,
              vad_effective_block_frames: 300,
              vad_intraop_threads: 4
            }
          )

          assert_equal 4, payload.dig("vad", "torch_intraop_threads")
          assert_equal 300, payload.dig("vad", "effective_block_frames")
          assert_equal 3, payload.dig("vad", "model_calls")
          assert_equal 601, payload.dig("vad", "valid_frames")
          assert_equal 601, payload.dig("vad", "padded_frames")
          assert_equal 0.0, payload.dig("vad", "padding_ratio")
          assert_equal 1, payload.dig("vad", "max_files_per_call")
          assert_equal(
            { "CPUExecutionProvider" => {} },
            payload.fetch("files").first.fetch("vad_provider_options")
          )
        end

        def test_profile_writer_publishes_finite_pretty_json
          Dir.mktmpdir do |directory|
            options = options_with_publication(profile_json: File.join(directory, "profile.json"))
            run = TranscriptionRun.new(
              results: [result], requested_options: options, resolved_options: options,
              statistics: statistics
            )
            path = Publication.write_profile(options.publication.profile_json, run)

            document = path.read(encoding: "UTF-8")
            assert document.end_with?("\n")
            refute_match(/\b(?:NaN|Infinity)\b/, document)
            assert_equal 9, JSON.parse(document).fetch("schema_version")
          end
        end

        def test_profile_serialization_failure_is_a_nonfatal_typed_run_error
          Dir.mktmpdir do |directory|
            options = options_with_publication(profile_json: File.join(directory, "profile.json"))
            run = TranscriptionRun.new(
              results: [result],
              requested_options: options,
              resolved_options: options,
              statistics: statistics.with(elapsed_seconds: Float::NAN)
            )

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.write_profile(options.publication.profile_json, run)
            end
            assert_match(/profile output failed:.*NaN not allowed/, error.message)
            refute options.publication.profile_json.exist?
          end
        end

        def test_nested_output_symlink_cannot_escape_the_output_root
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            source_root = root.join("source")
            source = source_root.join("redirect/clip.wav")
            output_root = root.join("out")
            outside = root.join("outside")
            source.dirname.mkpath
            source.binwrite("audio")
            output_root.mkpath
            outside.mkpath
            begin
              output_root.join("redirect").make_symlink(outside)
            rescue NotImplementedError, Errno::EPERM
              skip "symbolic links are unavailable"
            end
            entry = InputEntry.new(path: source.realpath, relative_path: Pathname("redirect/clip.wav"))
            options = options_with_publication.with(
              publication: options_with_publication.publication.with(output_dir: output_root)
            )

            error = assert_raises(TranscriptionInputError) do
              Publication.plan([entry], options)
            end
            assert_match(/escapes output root/, error.message)
            assert_empty outside.children
          end
        end

        def test_symlink_inserted_during_nested_output_mkdir_cannot_create_outside_the_bound_root
          Dir.mktmpdir("cohere-output-mkdir-race") do |directory|
            root = Pathname(directory)
            source = root.join("source/nested/clip.wav")
            output_root = root.join("out")
            outside = root.join("outside")
            source.dirname.mkpath
            source.binwrite("audio")
            output_root.mkdir
            outside.mkdir
            entry = InputEntry.new(
              path: source.realpath,
              relative_path: Pathname("nested/clip.wav")
            )
            options = options_with_publication.with(
              publication: options_with_publication.publication.with(output_dir: output_root)
            )
            nested_output = output_root.realpath.join("nested")
            inserted = false
            operation_hook = lambda do |phase, candidate|
              next unless phase == :before_mkdir && candidate == nested_output && !inserted

              inserted = true
              candidate.make_symlink(outside)
            end
            original = State.method(:ensure_bound_directory)
            injected = operation_hook
            replacement = lambda do |path, root_binding: nil, operation_hook: nil|
              original.call(path, root_binding: root_binding, operation_hook: operation_hook || injected)
            end

            State.define_singleton_method(:ensure_bound_directory, replacement)
            begin
              error = assert_raises(TranscriptionInputError) { Publication.plan([entry], options) }
            ensure
              State.define_singleton_method(:ensure_bound_directory, original)
            end

            assert inserted
            assert_match(/not a real directory/, error.message)
            assert_empty outside.children
          end
        end

        def test_symlink_inserted_during_profile_mkdir_cannot_create_outside_the_bound_ancestor
          Dir.mktmpdir("cohere-profile-mkdir-race") do |directory|
            root = Pathname(directory)
            outside = root.join("outside")
            outside.mkdir
            profile_parent = root.join("profiles")
            profile = profile_parent.join("nested/run.json")
            inserted = false
            injected = lambda do |phase, candidate|
              next unless phase == :before_mkdir && candidate == profile_parent && !inserted

              inserted = true
              candidate.make_symlink(outside)
            end
            original = State.method(:ensure_bound_directory)
            replacement = lambda do |path, root_binding: nil, operation_hook: nil|
              original.call(path, root_binding: root_binding, operation_hook: operation_hook || injected)
            end

            State.define_singleton_method(:ensure_bound_directory, replacement)
            begin
              error = assert_raises(TranscriptionInputError) { Publication.bind_profile_path(profile) }
            ensure
              State.define_singleton_method(:ensure_bound_directory, original)
            end

            assert inserted
            assert_match(/not a real directory/, error.message)
            assert_empty outside.children
          end
        end

        def test_profile_alias_cannot_hide_a_collision_with_a_transcript_output
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            source = root.join("clip.wav")
            output_root = root.join("out")
            alias_root = root.join("output-alias")
            source.binwrite("audio")
            output_root.mkpath
            begin
              alias_root.make_symlink(output_root)
            rescue NotImplementedError, Errno::EPERM
              skip "symbolic links are unavailable"
            end
            entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
            options = TranscriptionOptions.new(
              vad: "none",
              max_dur: 1.0,
              publication: PublicationOptions.new(
                formats: ["txt"],
                output_dir: output_root,
                existing: "overwrite",
                profile_json: alias_root.join("clip.txt")
              )
            )

            error = assert_raises(TranscriptionInputError) do
              Publication.plan([entry], options)
            end
            assert_match(/Profile path collides with a transcript output/, error.message)
          end
        end

        def test_atomic_writer_preserves_existing_mode_and_uses_process_umask_for_new_files
          Dir.mktmpdir do |directory|
            existing = Pathname(directory).join("existing.txt")
            fresh = Pathname(directory).join("fresh.txt")
            existing.write("old")
            existing.chmod(0o640)
            previous_umask = File.umask(0o027)
            begin
              Publication.atomic_write_set(
                { "existing" => existing, "fresh" => fresh },
                { "existing" => "new", "fresh" => "new" }
              )
            ensure
              File.umask(previous_umask)
            end

            assert_equal 0o640, existing.stat.mode & 0o777
            assert_equal 0o640, fresh.stat.mode & 0o777
          end
        end

        def test_atomic_writer_reads_the_target_mode_before_creating_a_staging_file
          original_open_regular = State::BoundDirectory.instance_method(:open_regular)
          original_create_temporary = State::BoundDirectory.instance_method(:create_temporary)
          staging_files_created = 0
          State::BoundDirectory.define_method(:open_regular) do |name|
            raise Errno::EIO, "simulated mode lookup failure" if name == "clip.txt"

            original_open_regular.bind_call(self, name)
          end
          State::BoundDirectory.define_method(:create_temporary) do |*arguments|
            staging_files_created += 1
            original_create_temporary.bind_call(self, *arguments)
          end

          Dir.mktmpdir("cohere-output-mode") do |directory|
            root = Pathname(directory)
            destination = root.join("clip.txt")

            error = assert_raises(Errno::EIO) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" }
              )
            end

            assert_match(/mode lookup failure/, error.message)
            assert_equal 0, staging_files_created
            assert_empty root.children
          end
        ensure
          State::BoundDirectory.define_method(:open_regular, original_open_regular) if original_open_regular
          State::BoundDirectory.define_method(:create_temporary, original_create_temporary) if original_create_temporary
        end

        def test_successful_atomic_writer_reports_a_backup_cleanup_failure
          original_unlink = State::BoundDirectory.instance_method(:unlink)
          State::BoundDirectory.define_method(:unlink) do |name, missing_ok: false|
            raise Errno::EIO, "simulated backup cleanup failure" if name.end_with?(".bak")

            original_unlink.bind_call(self, name, missing_ok: missing_ok)
          end

          Dir.mktmpdir("cohere-output-cleanup") do |directory|
            destination = Pathname(directory).join("clip.txt")
            destination.binwrite("old transcript")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" }
              )
            end

            assert_match(/cleanup was incomplete/, error.message)
            assert_match(/backup cleanup failure/, error.message)
            retained = destination.dirname.children.select { |path| path.extname == ".bak" }
            assert_equal 1, retained.length
            assert_includes error.message, retained.fetch(0).to_s
            assert_equal "new transcript", destination.binread
          end
        ensure
          State::BoundDirectory.define_method(:unlink, original_unlink) if original_unlink
        end

        def test_staging_cleanup_failure_reports_the_primary_write_failure_as_its_cause
          original_unlink = State::BoundDirectory.instance_method(:unlink)
          State::BoundDirectory.define_method(:unlink) do |name, missing_ok: false|
            raise Errno::EIO, "simulated staging cleanup failure" if name.end_with?(".tmp")

            original_unlink.bind_call(self, name, missing_ok: missing_ok)
          end

          Dir.mktmpdir("cohere-output-primary") do |directory|
            destination = Pathname(directory).join("clip.txt")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" },
                operation_hook: lambda do |phase, _path|
                  raise "simulated primary write failure" if phase == :before_stage_write
                end
              )
            end

            assert_match(/recovery was incomplete/, error.message)
            assert_match(/simulated staging cleanup failure/, error.message)
            assert_instance_of RuntimeError, error.cause
            assert_equal "simulated primary write failure", error.cause.message
            retained = Pathname(directory).children.select { |path| path.extname == ".tmp" }
            assert_equal 1, retained.length
            assert_includes error.message, retained.fetch(0).to_s
          end
        ensure
          State::BoundDirectory.define_method(:unlink, original_unlink) if original_unlink
        end

        def test_atomic_writer_aggregates_every_directory_close_failure_and_preserves_the_primary_failure
          original_close = State::BoundDirectory.instance_method(:close)
          closed = []
          State::BoundDirectory.define_method(:close) do
            closed << binding.canonical_path
            original_close.bind_call(self)
            raise Errno::EIO, "forced output directory close failure"
          end

          Dir.mktmpdir("cohere-output-close") do |directory|
            root = Pathname(directory)
            first_parent = root.join("first")
            second_parent = root.join("second")
            first_parent.mkdir
            second_parent.mkdir
            first = first_parent.join("clip.txt")
            second = second_parent.join("clip.json")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.atomic_write_set(
                { "txt" => first, "json" => second },
                { "txt" => "new transcript", "json" => "{}" },
                directory_bindings: [
                  State::DirectoryBinding.capture(first_parent),
                  State::DirectoryBinding.capture(second_parent)
                ],
                operation_hook: lambda do |phase, _path|
                  raise "forced output primary failure" if phase == :before_stage_write
                end
              )
            end

            assert_equal [first_parent.realpath, second_parent.realpath].sort, closed.sort
            assert_equal 2, error.message.scan("forced output directory close failure").length
            assert_instance_of RuntimeError, error.cause
            assert_equal "forced output primary failure", error.cause.message
            assert_empty first_parent.children
            assert_empty second_parent.children
          end
        ensure
          State::BoundDirectory.define_method(:close, original_close) if original_close
        end

        def test_successful_output_rollback_does_not_clean_the_restored_backup_again
          original_unlink = State::BoundDirectory.instance_method(:unlink)
          backup_cleanup_attempts = []
          State::BoundDirectory.define_method(:unlink) do |name, missing_ok: false|
            backup_cleanup_attempts << name if name.end_with?(".bak")
            original_unlink.bind_call(self, name, missing_ok: missing_ok)
          end

          Dir.mktmpdir("cohere-output-restored-backup") do |directory|
            destination = Pathname(directory).join("clip.txt")
            destination.binwrite("old transcript")

            error = assert_raises(RuntimeError) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" },
                operation_hook: lambda do |phase, _path|
                  raise "forced failure after rename" if phase == :after_rename
                end
              )
            end

            assert_equal "forced failure after rename", error.message
            assert_equal "old transcript", destination.binread
            assert_empty backup_cleanup_attempts
            transaction_files = destination.dirname.children.select do |path|
              %w[.tmp .bak].include?(path.extname)
            end
            assert_empty transaction_files
          end
        ensure
          State::BoundDirectory.define_method(:unlink, original_unlink) if original_unlink
        end

        def test_output_backup_copy_closes_both_handles_and_types_the_first_close_failure
          original_open = State::BoundDirectory.instance_method(:open_regular)
          original_create = State::BoundDirectory.instance_method(:create_temporary)
          source_opens = 0
          close_calls = Hash.new(0)
          State::BoundDirectory.define_method(:open_regular) do |name, writable: false|
            handle = original_open.bind_call(self, name, writable: writable)
            source_opens += 1 if name == "clip.txt"
            if name == "clip.txt" && source_opens == 2
              close = handle.method(:close)
              handle.define_singleton_method(:close) do
                close_calls[:source] += 1
                raise Errno::EIO, "forced output source close failure" if close_calls[:source] == 1

                close.call
              end
            end
            handle
          end
          State::BoundDirectory.define_method(:create_temporary) do |basename, suffix|
            name, handle = original_create.bind_call(self, basename, suffix)
            if suffix == ".bak"
              close = handle.method(:close)
              handle.define_singleton_method(:close) do
                close_calls[:backup] += 1
                close.call
              end
            end
            [name, handle]
          end

          Dir.mktmpdir("cohere-output-copy-close") do |directory|
            destination = Pathname(directory).join("clip.txt")
            destination.binwrite("old transcript")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" }
              )
            end

            assert_match(/close .*forced output source close failure/, error.message)
            assert_instance_of Errno::EIO, error.cause
            assert_equal({ source: 2, backup: 1 }, close_calls)
            assert_equal "old transcript", destination.binread
            transaction_files = destination.dirname.children.select do |path|
              %w[.tmp .bak].include?(path.extname)
            end
            assert_empty transaction_files
          end
        ensure
          State::BoundDirectory.define_method(:open_regular, original_open) if original_open
          State::BoundDirectory.define_method(:create_temporary, original_create) if original_create
        end

        def test_atomic_writer_restores_every_committed_output_after_a_later_failure
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            first = root.join("first.txt")
            second = root.join("second.txt")
            first.write("first-old")
            second.write("second-old")
            original_rename = File.method(:rename)
            [Errno::EIO.new("simulated commit failure"), Interrupt.new].each do |failure|
              replacement = lambda do |source, destination|
                raise failure if Pathname(destination) == second && Pathname(source).extname == ".tmp"

                original_rename.call(source, destination)
              end

              assert_raises(failure.class) do
                Publication.atomic_write_set(
                  { "first" => first, "second" => second },
                  { "first" => "first-new", "second" => "second-new" },
                  rename: replacement
                )
              end
              assert_equal "first-old", first.read
              assert_equal "second-old", second.read
              assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
            end
          end
        end

        def test_atomic_writer_restores_an_output_when_cancellation_follows_its_rename
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            destination = root.join("clip.txt")
            destination.binwrite("old transcript")
            replacement = lambda do |source, target|
              File.rename(source, target)
              raise Interrupt, "cancelled immediately after rename"
            end

            assert_raises(Interrupt) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" },
                rename: replacement
              )
            end

            assert_equal "old transcript", destination.binread
            assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          end
        end

        def test_thread_kill_between_output_renames_rolls_back_the_output_set
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            first = root.join("first.txt")
            second = root.join("second.txt")
            first.binwrite("first-old")
            second.binwrite("second-old")
            killed = false
            hook = lambda do |phase, destination|
              next unless phase == :after_rename && destination == first && !killed

              killed = true
              publishing_thread = Thread.current
              Thread.new { publishing_thread.kill }.join
            end
            caller = Thread.new do
              Publication.atomic_write_set(
                { "first" => first, "second" => second },
                { "first" => "first-new", "second" => "second-new" },
                operation_hook: hook
              )
            end
            caller.report_on_exception = false

            assert caller.join(2), "output publisher remained stuck after termination"
            assert_nil caller.value
            assert_equal "first-old", first.binread
            assert_equal "second-old", second.binread
            assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          ensure
            caller&.kill
            caller&.join
          end
        end

        def test_thread_kill_with_an_incomplete_output_rollback_remains_terminal
          original_rename = State::BoundDirectory.instance_method(:rename)
          State::BoundDirectory.define_method(:rename) do |source, destination|
            raise Errno::EIO, "forced restore failure" if source.end_with?(".bak")

            original_rename.bind_call(self, source, destination)
          end

          Dir.mktmpdir("cohere-output-kill-rollback-failure") do |directory|
            root = Pathname(directory)
            destination = root.join("clip.txt")
            destination.binwrite("old transcript")
            hook = lambda do |phase, path|
              next unless phase == :after_rename && path == destination

              publishing_thread = Thread.current
              Thread.new { publishing_thread.kill }.join
            end
            rescued = false
            continued = false
            caller = Thread.new do
              begin
                Publication.atomic_write_set(
                  { "txt" => destination },
                  { "txt" => "new transcript" },
                  operation_hook: hook
                )
              rescue TranscriptionRuntimeError
                rescued = true
              end
              continued = true
            end
            caller.report_on_exception = false

            assert caller.join(2), "output publisher remained stuck after termination"
            assert_nil caller.value
            refute rescued, "recovery failure replaced Thread#kill with a catchable exception"
            refute continued, "output publisher continued after Thread#kill"
            retained = root.children.select { |path| path.extname == ".bak" }
            assert_equal 1, retained.length
            assert_equal "old transcript", retained.fetch(0).binread
            refute destination.exist?
          ensure
            caller&.kill
            caller&.join
          end
        ensure
          State::BoundDirectory.define_method(:rename, original_rename) if original_rename
        end

        def test_throw_reports_a_staging_cleanup_failure_instead_of_silently_leaving_the_file
          original_unlink = State::BoundDirectory.instance_method(:unlink)
          State::BoundDirectory.define_method(:unlink) do |name, missing_ok: false|
            raise Errno::EIO, "forced staging cleanup failure" if name.end_with?(".tmp") && display_path(name).exist?

            original_unlink.bind_call(self, name, missing_ok: missing_ok)
          end

          Dir.mktmpdir("cohere-output-throw-cleanup-failure") do |directory|
            root = Pathname(directory)
            first = root.join("first.txt")
            second = root.join("second.txt")
            first.binwrite("first-old")
            second.binwrite("second-old")
            hook = lambda do |phase, destination|
              throw :stop_publication if phase == :after_rename && destination == first
            end

            error = assert_raises(TranscriptionRuntimeError) do
              catch(:stop_publication) do
                Publication.atomic_write_set(
                  { "first" => first, "second" => second },
                  { "first" => "first-new", "second" => "second-new" },
                  operation_hook: hook
                )
              end
            end

            assert_match(/recovery was incomplete/, error.message)
            assert_match(/forced staging cleanup failure/, error.message)
            assert_equal "first-old", first.binread
            assert_equal "second-old", second.binread
            retained = root.children.select { |path| path.extname == ".tmp" }
            assert_equal 1, retained.length
            assert_includes error.message, retained.fetch(0).to_s
          end
        ensure
          State::BoundDirectory.define_method(:unlink, original_unlink) if original_unlink
        end

        def test_thread_kill_during_failure_rollback_waits_for_cleanup
          original_fsync = State::BoundDirectory.instance_method(:fsync)
          rollback_started = Queue.new
          continue_rollback = Queue.new
          block_once = true
          State::BoundDirectory.define_method(:fsync) do
            if block_once
              block_once = false
              rollback_started << true
              continue_rollback.pop
            end
            original_fsync.bind_call(self)
          end

          Dir.mktmpdir("cohere-output-rollback-kill") do |directory|
            root = Pathname(directory)
            first = root.join("first.txt")
            second = root.join("second.txt")
            first.binwrite("first-old")
            second.binwrite("second-old")
            caller = Thread.new do
              Publication.atomic_write_set(
                { "first" => first, "second" => second },
                { "first" => "first-new", "second" => "second-new" },
                operation_hook: lambda do |phase, destination|
                  raise "force rollback" if phase == :after_rename && destination == first
                end
              )
            end
            caller.report_on_exception = false

            Timeout.timeout(2) { rollback_started.pop }
            caller.kill
            continue_rollback << true

            assert caller.join(2), "output rollback remained stuck after termination"
            assert_nil caller.value
            assert_equal "first-old", first.binread
            assert_equal "second-old", second.binread
            assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          ensure
            continue_rollback << true if continue_rollback.empty?
            caller&.kill
            caller&.join
          end
        ensure
          State::BoundDirectory.define_method(:fsync, original_fsync) if original_fsync
        end

        def test_cleanup_failure_does_not_replace_thread_kill_after_output_commit
          original_unlink = State::BoundDirectory.instance_method(:unlink)
          State::BoundDirectory.define_method(:unlink) do |name, missing_ok: false|
            raise Errno::EIO, "forced cleanup failure" if name.end_with?(".tmp")

            original_unlink.bind_call(self, name, missing_ok: missing_ok)
          end

          Dir.mktmpdir("cohere-output-cleanup-termination") do |directory|
            destination = Pathname(directory).join("output.txt")
            guard_calls = 0
            commit_guard = lambda do
              guard_calls += 1
              next unless guard_calls == 6

              publishing_thread = Thread.current
              Thread.new { publishing_thread.kill }.join
            end
            caller = Thread.new do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new output" },
                commit_guard: commit_guard
              )
            end
            caller.report_on_exception = false

            assert caller.join(2), "output publisher remained stuck after termination"
            assert_nil caller.value
            assert_equal 6, guard_calls
            assert_equal "new output", destination.binread
          ensure
            caller&.kill
            caller&.join
          end
        ensure
          State::BoundDirectory.define_method(:unlink, original_unlink) if original_unlink
        end

        def test_incomplete_output_rollback_preserves_a_backup_when_its_status_cannot_be_read
          original_rename = State::BoundDirectory.instance_method(:rename)
          original_regular_entry = State::BoundDirectory.instance_method(:regular_entry?)
          State::BoundDirectory.define_method(:rename) do |source, destination|
            raise Errno::EIO, "forced restore failure" if source.end_with?(".bak")

            original_rename.bind_call(self, source, destination)
          end
          State::BoundDirectory.define_method(:regular_entry?) do |name|
            raise Errno::EIO, "forced status failure" if name.end_with?(".bak")

            original_regular_entry.bind_call(self, name)
          end

          Dir.mktmpdir("cohere-output-rollback") do |directory|
            root = Pathname(directory)
            destination = root.join("clip.txt")
            destination.binwrite("old transcript")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.atomic_write_set(
                { "txt" => destination },
                { "txt" => "new transcript" },
                directory_bindings: [State::DirectoryBinding.capture(root)],
                operation_hook: lambda do |phase, _path|
                  raise Interrupt, "force rollback" if phase == :after_rename
                end
              )
            end

            assert_match(/recovery was incomplete/, error.message)
            retained = root.children.select { |path| path.extname == ".bak" }
            assert_equal 1, retained.length
            assert_equal "old transcript", retained.fetch(0).binread
            assert_includes error.message, retained.fetch(0).to_s
            refute destination.exist?
          end
        ensure
          State::BoundDirectory.define_method(:rename, original_rename) if original_rename
          State::BoundDirectory.define_method(:regular_entry?, original_regular_entry) if original_regular_entry
        end

        def test_skip_manifest_detects_same_size_source_rewrite_with_restored_mtime
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            source = root.join("clip.wav")
            source.binwrite("audio-one")
            entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
            output_root = root.join("out")
            overwrite = options_with_publication.with(
              publication: PublicationOptions.new(
                formats: %w[txt json], output_dir: output_root, existing: "overwrite"
              )
            )
            completed = result.with(path: source.realpath, relative_path: Pathname("clip.wav"))
            initial_plan = Publication.plan([entry], overwrite).fetch(source.realpath)
            Publication.write(initial_plan, completed, overwrite)

            skip_options = overwrite.with(publication: overwrite.publication.with(existing: "skip"))
            assert Publication.plan([entry], skip_options).fetch(source.realpath).skipped

            original_mtime = source.mtime
            original_ctime = source.ctime
            source.binwrite("audio-two")
            File.utime(original_mtime, original_mtime, source)
            skip "filesystem does not expose a changed ctime" if source.ctime == original_ctime

            refute Publication.plan([entry], skip_options).fetch(source.realpath).skipped
          end
        end

        def test_skip_revalidation_reprocesses_an_output_replaced_during_verification
          original_same_regular_entry = State::BoundDirectory.instance_method(:same_regular_entry?)
          replaced = false

          Dir.mktmpdir("cohere-skip-output-race") do |directory|
            root = Pathname(directory)
            plan, skip_options = published_skip_fixture(root)
            output = plan.paths.fetch("txt")
            parked = output.dirname.join("original-clip.txt")
            State::BoundDirectory.define_method(:same_regular_entry?) do |name, expected_stat|
              if name == output.basename.to_s && !replaced
                replaced = true
                output.rename(parked)
                output.binwrite("replacement transcript\n")
              end
              original_same_regular_entry.bind_call(self, name, expected_stat)
            end

            decision = Publication.with_plan_lock(plan) do
              Publication.revalidate(plan, skip_options)
            end

            assert replaced
            assert_equal :process, decision.action
            assert_match(/txt output changed or is not regular/, decision.reason)
          end
        ensure
          State::BoundDirectory.define_method(:same_regular_entry?, original_same_regular_entry) if original_same_regular_entry
        end

        def test_skip_revalidation_reprocesses_a_nonregular_output_replacement
          original_same_regular_entry = State::BoundDirectory.instance_method(:same_regular_entry?)
          replaced = false

          Dir.mktmpdir("cohere-skip-output-nonregular") do |directory|
            root = Pathname(directory)
            plan, skip_options = published_skip_fixture(root)
            output = plan.paths.fetch("txt")
            parked = output.dirname.join("original-clip.txt")
            State::BoundDirectory.define_method(:same_regular_entry?) do |name, expected_stat|
              if name == output.basename.to_s && !replaced
                replaced = true
                output.rename(parked)
                output.mkdir
              end
              original_same_regular_entry.bind_call(self, name, expected_stat)
            end

            decision = Publication.with_plan_lock(plan) do
              Publication.revalidate(plan, skip_options)
            end

            assert replaced
            assert_equal :process, decision.action
            assert_match(/txt output changed or is not regular/, decision.reason)
          end
        ensure
          State::BoundDirectory.define_method(:same_regular_entry?, original_same_regular_entry) if original_same_regular_entry
        end

        def test_skip_verification_still_propagates_a_changed_directory_guard
          original_same_regular_entry = State::BoundDirectory.instance_method(:same_regular_entry?)
          changed = false

          Dir.mktmpdir("cohere-skip-output-guard") do |directory|
            root = Pathname(directory)
            plan, skip_options = published_skip_fixture(
              root,
              relative_path: Pathname("nested/clip.wav")
            )
            output = plan.paths.fetch("txt")
            output_root = root.join("out")
            parked = root.join("parked-output")
            State::BoundDirectory.define_method(:same_regular_entry?) do |name, expected_stat|
              if name == output.basename.to_s && !changed
                changed = true
                output_root.rename(parked)
                output_root.mkdir
              end
              original_same_regular_entry.bind_call(self, name, expected_stat)
            end

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.with_plan_lock(plan) do
                Publication.revalidate(plan, skip_options)
              end
            end

            assert changed
            assert_match(/Publication parent changed/, error.message)
          end
        ensure
          State::BoundDirectory.define_method(:same_regular_entry?, original_same_regular_entry) if original_same_regular_entry
        end

        def test_publication_rechecks_the_planned_source_immediately_before_commit
          Dir.mktmpdir do |directory|
            root = Pathname(directory)
            source = root.join("clip.wav")
            source.binwrite("original")
            entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
            output_root = root.join("out")
            options = options_with_publication.with(
              publication: PublicationOptions.new(
                formats: %w[txt json], output_dir: output_root, existing: "overwrite"
              )
            )
            plan = Publication.plan([entry], options).fetch(source.realpath)
            completed = result.with(path: source.realpath, relative_path: Pathname("clip.wav"))
            source.binwrite("changed!")

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.write(plan, completed, options)
            end
            assert_match(/Source changed while processing/, error.message)
            refute output_root.join("clip.txt").exist?
            refute output_root.join("clip.json").exist?
            refute plan.state_path.exist?
          end
        end

        def test_publication_refuses_to_commit_after_its_output_lock_is_replaced
          Dir.mktmpdir("cohere-publication-lock-identity") do |directory|
            root = Pathname(directory)
            source = root.join("clip.wav")
            source.binwrite("audio")
            entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
            options = options_with_publication.with(
              publication: PublicationOptions.new(
                formats: %w[txt json], output_dir: root.join("out"), existing: "overwrite"
              )
            )
            plan = Publication.plan([entry], options).fetch(source.realpath)
            completed = result.with(path: source.realpath, relative_path: entry.relative_path)

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.with_plan_lock(plan) do |lock|
                parked = plan.lock_target.path.sub_ext(".parked")
                plan.lock_target.path.rename(parked)
                plan.lock_target.path.binwrite("")
                Publication.write(plan, completed, options, lock: lock)
              end
            end

            assert_match(/changed while held/, error.message)
            plan.paths.each_value { |path| refute path.exist? }
            refute plan.state_path.exist?
          end
        end

        def test_planned_parent_rejects_symlink_and_same_path_inode_replacements
          %i[symlink directory].each do |replacement|
            Dir.mktmpdir("cohere-publication-parent") do |directory|
              root = Pathname(directory)
              source = root.join("source/nested/clip.wav")
              source.dirname.mkpath
              source.binwrite("audio")
              outside = root.join("outside")
              outside.mkpath
              options = options_with_publication.with(
                publication: PublicationOptions.new(
                  formats: %w[txt json], output_dir: root.join("out"), existing: "overwrite"
                )
              )
              entry = InputEntry.new(
                path: source.realpath,
                relative_path: Pathname("nested/clip.wav")
              )
              plan = Publication.plan([entry], options).fetch(source.realpath)
              parent = plan.paths.fetch("txt").dirname
              parked = root.join("parked")
              parent.rename(parked)
              replacement == :symlink ? parent.make_symlink(outside) : parent.mkdir

              error = assert_raises(TranscriptionRuntimeError) do
                Publication.write(
                  plan,
                  result.with(path: source.realpath, relative_path: entry.relative_path),
                  options
                )
              end

              assert_match(/Publication parent changed/, error.message)
              assert_empty outside.children
              assert_empty parent.children if replacement == :directory
              assert_empty parked.children
            end
          end
        end

        def test_descriptor_relative_transaction_contains_parent_swaps_at_every_commit_phase
          phases = %i[before_stage_write before_backup_copy before_rename after_rename]
          phases.each do |phase|
            Dir.mktmpdir("cohere-publication-race") do |directory|
              root = Pathname(directory)
              parent = root.join("bound")
              outside = root.join("outside")
              parent.mkdir
              outside.mkdir
              paths = {
                "txt" => parent.join("clip.txt"),
                "json" => parent.join("clip.json"),
                "__manifest__" => parent.join(".clip.cohere-transcribe.manifest.json")
              }
              paths.each_value { |path| path.binwrite("old-#{path.basename}\n") }
              old_contents = paths.values.to_h { |path| [path, path.binread] }
              binding = State::DirectoryBinding.capture(parent)
              parked = root.join("parked")
              swapped = false
              hook = lambda do |event, _destination|
                next unless event == phase && !swapped

                swapped = true
                parent.rename(parked)
                parent.make_symlink(outside)
              end

              assert_raises(TranscriptionRuntimeError) do
                Publication.atomic_write_set(
                  paths,
                  {
                    "txt" => "secret transcript\n",
                    "json" => "{\"secret\":true}\n",
                    "__manifest__" => "secret manifest\n"
                  },
                  directory_bindings: [binding],
                  operation_hook: hook
                )
              end

              assert swapped, "phase #{phase} was not reached"
              assert_empty outside.children, "phase #{phase} escaped the bound directory"
              old_contents.each do |path, content|
                assert_equal content, parked.join(path.basename).binread
              end
              assert_empty(parked.children.select { |path| %w[.tmp .bak].include?(path.extname) })
            end
          end
        end

        def test_backup_open_rejects_leaf_symlink_fifo_and_directory_without_touching_targets
          kinds = %i[symlink fifo directory]
          kinds.each do |kind|
            Dir.mktmpdir("cohere-publication-leaf") do |directory|
              root = Pathname(directory)
              destination = root.join("clip.txt")
              victim = root.join("victim")
              victim.binwrite("victim remains untouched")
              binding = State::DirectoryBinding.capture(root)
              installed = false
              hook = lambda do |event, _path|
                next unless event == :before_backup_open && !installed

                installed = true
                case kind
                when :symlink
                  destination.make_symlink(victim)
                when :fifo
                  skip "FIFO creation is unavailable" unless File.respond_to?(:mkfifo)

                  File.mkfifo(destination)
                when :directory
                  destination.mkdir
                end
              end

              error = assert_raises(TranscriptionRuntimeError) do
                Publication.atomic_write_set(
                  { "txt" => destination },
                  { "txt" => "secret transcript\n" },
                  directory_bindings: [binding],
                  operation_hook: hook
                )
              end

              assert_match(/not a regular file/, error.message)
              assert_equal "victim remains untouched", victim.binread
              assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
            end
          end
        end

        def test_hardlinked_output_is_detached_without_modifying_the_other_name
          Dir.mktmpdir("cohere-publication-hardlink") do |directory|
            root = Pathname(directory)
            victim = root.join("victim")
            destination = root.join("clip.txt")
            victim.binwrite("victim remains untouched")
            File.link(victim, destination)
            victim_inode = victim.stat.ino

            Publication.atomic_write_set(
              { "txt" => destination },
              { "txt" => "transcript\n" },
              directory_bindings: [State::DirectoryBinding.capture(root)]
            )

            assert_equal "victim remains untouched", victim.binread
            assert_equal victim_inode, victim.stat.ino
            assert_equal "transcript\n", destination.binread
            refute_equal victim.stat.ino, destination.stat.ino
            assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          end
        end

        def test_leaf_symlink_swap_after_backup_open_cannot_redirect_commit_or_rollback
          Dir.mktmpdir("cohere-publication-leaf-race") do |directory|
            root = Pathname(directory)
            destination = root.join("clip.txt")
            victim = root.join("victim")
            destination.binwrite("old transcript\n")
            victim.binwrite("victim remains untouched")
            swapped = false
            hook = lambda do |event, _path|
              next unless event == :before_backup_copy && !swapped

              swapped = true
              destination.delete
              destination.make_symlink(victim)
            end

            Publication.atomic_write_set(
              { "txt" => destination },
              { "txt" => "new transcript\n" },
              directory_bindings: [State::DirectoryBinding.capture(root)],
              operation_hook: hook
            )

            assert swapped
            assert_equal "victim remains untouched", victim.binread
            refute destination.symlink?
            assert_equal "new transcript\n", destination.binread
            assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          end
        end

        def test_profile_write_is_bound_to_the_parent_planned_before_the_run
          Dir.mktmpdir("cohere-profile-parent") do |directory|
            root = Pathname(directory)
            parent = root.join("profiles")
            outside = root.join("outside")
            parent.mkdir
            outside.mkdir
            profile = parent.join("run.json")
            binding = Publication.bind_profile_path(profile)
            parked = root.join("parked")
            parent.rename(parked)
            parent.make_symlink(outside)
            options = options_with_publication(profile_json: profile)
            run = TranscriptionRun.new(
              results: [result], requested_options: options, resolved_options: options,
              statistics: statistics
            )

            error = assert_raises(TranscriptionRuntimeError) do
              Publication.write_profile(profile, run, directory_binding: binding)
            end

            assert_match(/profile output failed:.*Publication parent changed/, error.message)
            assert_empty outside.children
            assert_empty parked.children
          end
        end

        private

        def published_skip_fixture(root, relative_path: Pathname("clip.wav"))
          source = root.join("source", relative_path)
          source.dirname.mkpath
          source.binwrite("audio")
          entry = InputEntry.new(path: source.realpath, relative_path: relative_path)
          overwrite = options_with_publication.with(
            publication: PublicationOptions.new(
              formats: %w[txt json], output_dir: root.join("out"), existing: "overwrite"
            )
          )
          plan = Publication.plan([entry], overwrite).fetch(source.realpath)
          completed = result.with(path: source.realpath, relative_path: relative_path)
          Publication.write(plan, completed, overwrite)
          skip_options = overwrite.with(
            publication: overwrite.publication.with(existing: "skip")
          )
          [plan, skip_options]
        end

        def options_with_publication(profile_json: nil)
          TranscriptionOptions.new(
            vad: "none",
            max_dur: 1.0,
            alignment: "segment",
            publication: PublicationOptions.new(
              formats: %w[txt json], output_dir: "out", existing: "overwrite",
              profile_json: profile_json
            )
          )
        end

        def result
          @result ||= TranscriptionResult.new(
            path: Pathname("/audio/clip.wav"),
            relative_path: Pathname("clip.wav"),
            status: "completed",
            text: "hello world",
            duration: 1.0,
            segments: [TranscriptionSegment.new(index: 0, start: 0.0, end: 1.0, text: "hello world")],
            words: [
              TranscriptionWord.new(
                start: 0.0, end: 1.0, text: "hello world", segment_index: 0,
                segment_word_index: 0, timing_source: "uniform_segment"
              )
            ],
            cues: [SubtitleCue.new(start: 0.0, end: 1.0, text: "hello world")],
            outputs: [Pathname("/audio/clip.txt")],
            provenance: TranscriptionProvenance.new(
              model_id: DEFAULT_ASR_MODEL_ID,
              model_revision: DEFAULT_ASR_MODEL_REVISION,
              model_format: "dense",
              decode_backend: "ffmpeg",
              generated_tokens_by_segment: [[0, 2]],
              published: true
            )
          )
        end

        def statistics
          values = TranscriptionStatistics.members.to_h { |member| [member, 0] }
          values.merge!(
            elapsed_seconds: 1.0,
            successful_audio_seconds: 1.0,
            real_time_factor_x: 1.0,
            asr_batches: 1,
            asr_processor_rows: 1,
            generated_tokens: 2
          )
          TranscriptionStatistics.new(**values)
        end
      end
    end
  end
end
