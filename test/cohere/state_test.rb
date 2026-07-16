# frozen_string_literal: true

require "test_helper"
require "json"
require "open3"
require "rbconfig"
require "timeout"
require "tmpdir"

module Cohere
  module Transcribe
    class StateTest < Minitest::Test
      def test_asr_and_render_contracts_have_separate_semantics
        base = options
        render_change = base.with(
          alignment: "word",
          align_dtype: "fp16",
          max_chars: 37,
          max_gap: 0.2,
          publication: base.publication.with(formats: %w[txt json])
        )

        assert_equal State.asr_contract_key(base), State.asr_contract_key(render_change)
        refute_equal State.render_contract_key(base), State.render_contract_key(render_change)
        refute_equal State.asr_contract_key(base), State.asr_contract_key(base.with(dtype: "fp16"))
        refute_equal State.asr_contract_key(base), State.asr_contract_key(base.with(language: "en"))
        refute_equal State.asr_contract_key(base),
                     State.asr_contract_key(base.with(pipeline_preparation: false))
        assert_equal State.asr_contract_key(base),
                     State.asr_contract_key(base.with(preprocess_workers: 8))
        refute_equal State.asr_contract_key(base),
                     State.asr_contract_key(base, model_format: "alternate-dense")
      end

      def test_checkpoint_round_trip_binds_canonical_source_and_ctime
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          snapshot = State::SourceSnapshot.capture(source)
          checkpoint_path = root.join(".clip#{State::CHECKPOINT_SUFFIX}")
          key = State.asr_contract_key(options)
          generation_id = State.write_asr_checkpoint(
            path: checkpoint_path,
            result: result(source),
            source_snapshot: snapshot,
            asr_contract_key: key,
            speech_spans: [[0.0, 0.4], [0.5, 1.0]],
            vad_provider_options: { "CPUExecutionProvider" => { "threads" => 1 } }
          )

          document = JSON.parse(checkpoint_path.read)
          assert_equal snapshot.ctime_ns, document.dig("payload", "source", "snapshot", "ctime_ns")
          assert_equal source.realpath.to_s, document.dig("payload", "source", "canonical_path")
          restored = State.restore_asr_checkpoint(
            path: checkpoint_path,
            source_snapshot: snapshot,
            asr_contract_key: key
          )

          assert restored.restored?, restored.reason
          assert_equal generation_id, restored.checkpoint.generation_id
          assert_equal [[0.0, 1.0]], restored.checkpoint.segment_times
          assert_equal [[0.0, 0.4], [0.5, 1.0]], restored.checkpoint.speech_spans
          assert_equal ["hello world"], restored.checkpoint.segment_texts
          assert_equal [[0, 2]], restored.checkpoint.generated_tokens_by_segment
          assert restored.checkpoint.segment_times.frozen?
          assert restored.checkpoint.vad_provider_options.frozen?
        end
      end

      def test_invalid_checkpoint_is_rejected_without_mutating_callers
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          snapshot = State::SourceSnapshot.capture(source)
          checkpoint_path = root.join(".clip#{State::CHECKPOINT_SUFFIX}")
          key = State.asr_contract_key(options)
          payload = State.asr_checkpoint_payload(
            result: result(source),
            source_snapshot: snapshot,
            asr_contract_key: key,
            speech_spans: [[0.0, 1.0]],
            vad_provider_options: nil,
            generation_id: "generation"
          )
          payload.fetch("checkpoint")["generated_tokens"] = [[0, 2], [0, 3]]
          State.write_state_atomic(checkpoint_path, payload, source_snapshot: snapshot)
          sentinel = { segments: ["fresh"], tokens: { 0 => 17 } }
          before = Marshal.dump(sentinel)

          restored = State.restore_asr_checkpoint(
            path: checkpoint_path,
            source_snapshot: snapshot,
            asr_contract_key: key
          )

          refute restored.restored?
          assert_match(/invalid/, restored.reason)
          assert_equal before, Marshal.dump(sentinel)
        end
      end

      def test_checkpoint_rejects_integrity_contract_source_and_span_corruption
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          snapshot = State::SourceSnapshot.capture(source)
          checkpoint_path = root.join(".clip#{State::CHECKPOINT_SUFFIX}")
          key = State.asr_contract_key(options)
          State.write_asr_checkpoint(
            path: checkpoint_path, result: result(source), source_snapshot: snapshot,
            asr_contract_key: key, speech_spans: [[0.0, 1.0]]
          )

          tampered = JSON.parse(checkpoint_path.read)
          tampered.dig("payload", "checkpoint")["duration"] = 2.0
          checkpoint_path.write(JSON.pretty_generate(tampered))
          restored = State.restore_asr_checkpoint(
            path: checkpoint_path, source_snapshot: snapshot, asr_contract_key: key
          )
          assert_match(/integrity/, restored.reason)

          State.write_asr_checkpoint(
            path: checkpoint_path, result: result(source), source_snapshot: snapshot,
            asr_contract_key: key, speech_spans: [[0.0, 1.0]]
          )
          restored = State.restore_asr_checkpoint(
            path: checkpoint_path, source_snapshot: snapshot, asr_contract_key: "different"
          )
          assert_match(/contract/, restored.reason)

          other = root.join("other.wav")
          other.binwrite("audio")
          restored = State.restore_asr_checkpoint(
            path: checkpoint_path,
            source_snapshot: State::SourceSnapshot.capture(other),
            asr_contract_key: key
          )
          assert_match(/source snapshot/, restored.reason)

          payload, = State.decode_state(checkpoint_path)
          payload.fetch("checkpoint")["segment_times"] = [[0.0, 0.8], [0.7, 1.0]]
          payload.fetch("checkpoint")["segment_texts"] = %w[first second]
          State.write_state_atomic(checkpoint_path, payload, source_snapshot: snapshot)
          restored = State.restore_asr_checkpoint(
            path: checkpoint_path, source_snapshot: snapshot, asr_contract_key: key
          )
          assert_match(/overlapping/, restored.reason)
        end
      end

      def test_source_snapshot_detects_same_size_mtime_preserving_rewrite
        Dir.mktmpdir do |directory|
          source = Pathname(directory).join("clip.wav")
          source.binwrite("first")
          before = State::SourceSnapshot.capture(source)
          original_stat = source.stat
          source.binwrite("other")
          File.utime(original_stat.atime, original_stat.mtime, source)
          after = State::SourceSnapshot.capture(source)

          assert_equal before.device, after.device
          assert_equal before.inode, after.inode
          assert_equal before.size, after.size
          assert_equal before.mtime_ns, after.mtime_ns
          refute_equal before.ctime_ns, after.ctime_ns
          refute_equal before, after
        end
      end

      def test_manifest_verifies_exact_generation_and_rejects_tampering
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          snapshot = State::SourceSnapshot.capture(source)
          output_paths = { "txt" => root.join("clip.txt"), "json" => root.join("clip.json") }
          contents = { "txt" => "transcript\n", "json" => "{\"ok\":true}\n" }
          contents.each { |format, content| output_paths.fetch(format).binwrite(content) }
          state_path = State.state_path_for_outputs(output_paths)
          state_path.binwrite(
            State.published_manifest_content(
              source_snapshot: snapshot,
              output_paths: output_paths,
              contents: contents,
              asr_contract_key: "asr",
              render_contract_key: "render",
              generation_id: "generation"
            )
          )

          verified = State.verify_published_outputs(
            source_snapshot: snapshot,
            output_paths: output_paths,
            state_path: state_path,
            asr_contract_key: "asr",
            render_contract_key: "render"
          )
          assert verified.verified?, verified.reason
          assert_equal "generation", verified.generation_id

          wrong_render = State.verify_published_outputs(
            source_snapshot: snapshot, output_paths: output_paths, state_path: state_path,
            asr_contract_key: "asr", render_contract_key: "changed"
          )
          refute wrong_render.verified?
          assert_match(/render contract/, wrong_render.reason)

          output_paths.fetch("txt").binwrite("tampered!!\n")
          tampered = State.verify_published_outputs(
            source_snapshot: snapshot, output_paths: output_paths, state_path: state_path,
            asr_contract_key: "asr", render_contract_key: "render"
          )
          refute tampered.verified?
          assert_match(/does not match/, tampered.reason)
        end
      end

      def test_publication_plan_uses_separate_checkpoint_and_manifest_contracts
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
          base = options.with(
            publication: options.publication.with(output_dir: root.join("out"))
          )
          plan = Output::Publication.plan([entry], base).fetch(source.realpath)
          refute_equal plan.state_path, plan.checkpoint_path
          assert_equal State.state_path_for_outputs(plan.paths), plan.state_path
          assert_equal State.checkpoint_path_for_outputs(plan.paths), plan.checkpoint_path
          assert_equal State.lock_target_for_outputs(plan.paths), plan.lock_target

          generation_id = State.write_asr_checkpoint(
            path: plan.checkpoint_path,
            result: result(source),
            source_snapshot: plan.source_snapshot,
            asr_contract_key: plan.asr_contract_key,
            speech_spans: [[0.0, 1.0]]
          )
          Output::Publication.with_plan_lock(plan) do
            Output::Publication.write(plan, result(source), base, generation_id: generation_id)
          end
          checkpoint_before = plan.checkpoint_path.binread
          manifest = JSON.parse(plan.state_path.read).fetch("payload")
          assert_equal generation_id, manifest.fetch("generation_id")
          assert_equal plan.asr_contract_key, manifest.fetch("asr_contract_key")
          assert_equal plan.render_contract_key, manifest.fetch("render_contract_key")

          skip_options = base.with(publication: base.publication.with(existing: "skip"))
          skip_plan = Output::Publication.plan([entry], skip_options).fetch(source.realpath)
          skip_decision = Output::Publication.with_plan_lock(skip_plan) do
            Output::Publication.revalidate(skip_plan, skip_options)
          end
          assert_equal :skip, skip_decision.action
          assert_equal generation_id, skip_decision.generation_id

          plan.paths.fetch("txt").binwrite("tampered output\n")
          tampered_plan = Output::Publication.plan([entry], skip_options).fetch(source.realpath)
          tampered_decision = Output::Publication.with_plan_lock(tampered_plan) do
            Output::Publication.revalidate(tampered_plan, skip_options)
          end
          assert_equal :resume, tampered_decision.action
          assert_equal ["hello world"], tampered_decision.checkpoint.segment_texts

          render_change = base.with(
            max_chars: 37,
            publication: base.publication.with(formats: %w[txt json])
          )
          render_plan = Output::Publication.plan([entry], render_change).fetch(source.realpath)
          render_decision = Output::Publication.with_plan_lock(render_plan) do
            Output::Publication.revalidate(render_plan, render_change)
          end
          assert_equal plan.asr_contract_key, render_plan.asr_contract_key
          refute_equal plan.render_contract_key, render_plan.render_contract_key
          assert_equal :resume, render_decision.action

          asr_change = base.with(language: "en")
          changed_plan = Output::Publication.plan([entry], asr_change).fetch(source.realpath)
          changed_decision = Output::Publication.with_plan_lock(changed_plan) do
            Output::Publication.revalidate(changed_plan, asr_change)
          end
          refute_equal plan.asr_contract_key, changed_plan.asr_contract_key
          assert_equal :process, changed_decision.action
          assert_match(/contract/, changed_decision.reason)
          assert_equal checkpoint_before, plan.checkpoint_path.binread
        end
      end

      def test_publication_revalidation_requires_and_releases_the_stem_lock
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          entry = InputEntry.new(path: source.realpath, relative_path: Pathname("clip.wav"))
          run_options = options.with(
            publication: options.publication.with(output_dir: root.join("out"))
          )
          plan = Output::Publication.plan([entry], run_options).fetch(source.realpath)
          blocker = State::OutputSetLock.acquire(plan.lock_target)
          begin
            assert_raises(TranscriptionRuntimeError) do
              Output::Publication.with_plan_lock(plan) do
                Output::Publication.revalidate(plan, run_options)
              end
            end
          ensure
            blocker.release
          end

          assert_raises(Interrupt) do
            Output::Publication.with_plan_lock(plan) { raise Interrupt, "cancelled" }
          end
          decision = Output::Publication.with_plan_lock(plan) do
            Output::Publication.revalidate(plan, run_options)
          end
          assert_equal :process, decision.action
        end
      end

      def test_atomic_state_write_refuses_symlink_and_source_change
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          snapshot = State::SourceSnapshot.capture(source)
          victim = root.join("victim")
          victim.binwrite("untouched")
          marker = root.join("marker.json")
          begin
            marker.make_symlink(victim)
          rescue NotImplementedError, Errno::EPERM
            skip "symbolic links are unavailable"
          end
          assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(marker, { "kind" => "test" }, source_snapshot: snapshot)
          end
          assert_equal "untouched", victim.read

          marker.delete
          source.binwrite("changed")
          error = assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(marker, { "kind" => "test" }, source_snapshot: snapshot)
          end
          assert_match(/Source changed/, error.message)
          refute marker.exist?
        end
      end

      def test_bound_directory_fsync_tolerates_platform_limitations_and_propagates_io_failures
        tolerated_errors = [
          Errno::EACCES,
          Errno::EBADF,
          Errno::EINVAL,
          Errno::EISDIR,
          Errno::ENOTSUP,
          Errno::EPERM
        ]
        tolerated_errors.each do |error_class|
          handle = Object.new
          handle.define_singleton_method(:fsync) do
            raise error_class, "directory sync is unavailable"
          end

          assert_nil State::BoundDirectory.new(nil, handle, []).fsync
        end

        handle = Object.new
        handle.define_singleton_method(:fsync) { raise Errno::EIO, "directory sync failed" }

        assert_raises(Errno::EIO) { State::BoundDirectory.new(nil, handle, []).fsync }
      end

      def test_bound_directory_create_temporary_passes_the_mode_at_creation
        arguments, = State::BoundDirectory::AT_FUNCTION_SIGNATURES.fetch(:openat)
        assert_equal Fiddle::TYPE_VARIADIC, arguments.last

        Dir.mktmpdir("cohere-bound-openat") do |directory|
          bound = State::BoundDirectory.open(
            State::DirectoryBinding.capture(Pathname(directory))
          )
          previous_umask = File.umask(0)
          temporary_name = nil
          handle = nil
          begin
            temporary_name, handle = bound.create_temporary("entry", ".tmp")
            assert_equal 0o600, handle.stat.mode & 0o777
          ensure
            File.umask(previous_umask)
            handle&.close
            bound.unlink(temporary_name, missing_ok: true) if temporary_name
            bound.close
          end
        end
      end

      def test_atomic_state_write_reports_cleanup_failure_after_success
        Dir.mktmpdir("cohere-state-cleanup-success") do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          marker = root.join("marker.json")
          source.binwrite("audio")

          error = assert_raises(Errno::EIO) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              rename: rename_with_temporary_cleanup_failure
            )
          end

          assert_match(/temporary cleanup failed/, error.message)
          assert_equal "new", JSON.parse(marker.read).dig("payload", "kind")
        end
      end

      def test_atomic_state_write_keeps_primary_failure_when_cleanup_also_fails
        Dir.mktmpdir("cohere-state-cleanup-primary") do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          marker = root.join("marker.json")
          source.binwrite("audio")
          primary_failure = Class.new(StandardError)

          error = assert_raises(primary_failure) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              operation_hook: lambda do |phase, _path|
                raise primary_failure, "primary state write failed" if phase == :after_rename
              end,
              rename: rename_with_temporary_cleanup_failure
            )
          end

          assert_equal "primary state write failed", error.message
          refute marker.exist?
        end
      end

      def test_descriptor_relative_state_write_contains_parent_swaps_during_staging_backup_and_rename
        %i[before_stage_write before_backup_copy before_rename after_rename].each do |phase|
          Dir.mktmpdir("cohere-state-race") do |directory|
            root = Pathname(directory)
            parent = root.join("bound")
            outside = root.join("outside")
            parent.mkdir
            outside.mkdir
            source = root.join("clip.wav")
            source.binwrite("audio")
            snapshot = State::SourceSnapshot.capture(source)
            marker = parent.join(".clip#{State::CHECKPOINT_SUFFIX}")
            marker.binwrite("old checkpoint")
            binding = State::DirectoryBinding.capture(parent)
            parked = root.join("parked")
            swapped = false
            hook = lambda do |event, _path|
              next unless event == phase && !swapped

              swapped = true
              parent.rename(parked)
              parent.make_symlink(outside)
            end

            assert_raises(TranscriptionRuntimeError) do
              State.write_state_atomic(
                marker,
                { "kind" => "asr_complete", "checkpoint" => { "secret" => true } },
                source_snapshot: snapshot,
                directory_binding: binding,
                guard_bindings: [binding],
                operation_hook: hook
              )
            end

            assert swapped, "phase #{phase} was not reached"
            assert_empty outside.children, "phase #{phase} escaped the bound directory"
            assert_equal "old checkpoint", parked.join(marker.basename).binread
            assert_empty(parked.children.select { |path| %w[.tmp .bak].include?(path.extname) })
          end
        end
      end

      def test_bound_parent_is_always_guarded_when_separate_ancestor_guards_are_supplied
        Dir.mktmpdir("cohere-state-parent-guard") do |directory|
          root = Pathname(directory)
          parent = root.join("bound")
          parked = root.join("parked")
          parent.mkdir
          source = root.join("clip.wav")
          source.binwrite("audio")
          marker = parent.join("marker.json")
          marker.binwrite("old state")
          swapped = false
          hook = lambda do |phase, _path|
            next unless phase == :directory_opened && !swapped

            swapped = true
            parent.rename(parked)
            parent.mkdir
          end

          assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(parent),
              guard_bindings: [State::DirectoryBinding.capture(root)],
              operation_hook: hook
            )
          end

          assert swapped
          assert_equal "old state", parked.join(marker.basename).binread
          assert_empty parent.children
          assert_empty(parked.children.select { |path| %w[.tmp .bak].include?(path.extname) })
        end
      end

      def test_bound_manifest_and_checkpoint_reads_reject_parent_redirection
        Dir.mktmpdir("cohere-state-read-race") do |directory|
          root = Pathname(directory)
          parent = root.join("bound")
          outside = root.join("outside")
          parent.mkdir
          outside.mkdir
          marker = parent.join("marker.json")
          binding = State::DirectoryBinding.capture(parent)
          marker.binwrite(JSON.generate(State.envelope("kind" => "inside")))
          outside.join("marker.json").binwrite(JSON.generate(State.envelope("kind" => "forged")))
          parked = root.join("parked")
          parent.rename(parked)
          parent.make_symlink(outside)

          payload, reason = State.decode_state(
            marker,
            directory_binding: binding,
            guard_bindings: [binding]
          )

          assert_nil payload
          assert_match(/Publication parent changed/, reason)
          assert_equal "forged", JSON.parse(outside.join("marker.json").read).dig("payload", "kind")
        end
      end

      def test_atomic_state_write_detaches_a_hardlink_without_modifying_its_peer
        Dir.mktmpdir("cohere-state-hardlink") do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          source.binwrite("audio")
          victim = root.join("victim")
          marker = root.join("marker.json")
          victim.binwrite("victim remains untouched")
          File.link(victim, marker)
          victim_inode = victim.stat.ino

          State.write_state_atomic(
            marker,
            { "kind" => "test" },
            source_snapshot: State::SourceSnapshot.capture(source),
            directory_binding: State::DirectoryBinding.capture(root)
          )

          assert_equal "victim remains untouched", victim.binread
          assert_equal victim_inode, victim.stat.ino
          refute_equal victim.stat.ino, marker.stat.ino
          assert_equal "test", JSON.parse(marker.read).dig("payload", "kind")
          assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
        end
      end

      def test_parent_fifo_swap_after_guard_check_fails_without_blocking_or_leaking_a_descriptor
        skip "FIFO creation is unavailable" unless File.respond_to?(:mkfifo)
        skip "Linux file descriptor accounting is unavailable" unless Pathname("/proc/self/fd").directory?

        Dir.mktmpdir("cohere-parent-fifo") do |directory|
          root = Pathname(directory)
          parent = root.join("bound")
          parked = root.join("parked")
          parent.mkdir
          actual = State::DirectoryBinding.capture(parent)
          proxy = Object.new
          %i[access_path canonical_path device inode].each do |name|
            proxy.define_singleton_method(name) { actual.public_send(name) }
          end
          checks = 0
          proxy.define_singleton_method(:verify!) do
            checks += 1
            actual.verify!
            if checks == 1
              parent.rename(parked)
              File.mkfifo(parent)
            end
            self
          end
          descriptors_before = Pathname("/proc/self/fd").children.length

          error = Timeout.timeout(1) do
            assert_raises(TranscriptionRuntimeError) do
              State::BoundDirectory.open(proxy, guards: [proxy])
            end
          end

          assert_match(/Publication parent changed/, error.message)
          assert_equal descriptors_before, Pathname("/proc/self/fd").children.length
          assert_predicate parent, :pipe?
          assert_empty parked.children
        end
      end

      def test_failed_child_directory_binding_closes_the_opened_descriptor
        skip "Linux file descriptor accounting is unavailable" unless Pathname("/proc/self/fd").directory?

        Dir.mktmpdir("cohere-child-binding-fd") do |directory|
          root = Pathname(directory)
          parent = root.join("parent")
          child_path = parent.join("child")
          parked = parent.join("parked")
          parent.mkdir
          actual = State::DirectoryBinding.capture(parent)
          proxy = Object.new
          %i[access_path canonical_path device inode].each do |name|
            proxy.define_singleton_method(name) { actual.public_send(name) }
          end
          checks = 0
          proxy.define_singleton_method(:verify!) do
            checks += 1
            actual.verify!
            if checks == 3
              child_path.rename(parked)
              child_path.mkdir
            end
            self
          end

          bound = State::BoundDirectory.open(proxy, guards: [proxy])
          bound.mkdir("child")
          descriptors_before = Pathname("/proc/self/fd").children.length
          garbage_collection_was_disabled = GC.disable
          begin
            assert_raises(TranscriptionRuntimeError) do
              bound.open_child_directory(
                "child",
                access_path: child_path,
                canonical_path: child_path
              )
            end
            assert_equal descriptors_before, Pathname("/proc/self/fd").children.length
          ensure
            GC.enable unless garbage_collection_was_disabled
            bound.close
          end
        end
      end

      def test_incomplete_state_rollback_preserves_the_backup_named_in_the_error
        Dir.mktmpdir("cohere-state-rollback") do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          marker = root.join("marker.json")
          source.binwrite("audio")
          marker.binwrite("old state")
          renamer = lambda do |bound, from, to, phase|
            raise Errno::EIO, "forced rollback rename failure" if phase == :rollback

            bound.rename(from, to)
          end
          hook = lambda do |phase, _path|
            raise Interrupt, "force rollback" if phase == :after_rename
          end

          error = assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              operation_hook: hook,
              rename: renamer
            )
          end

          assert_match(/rollback was incomplete/, error.message)
          retained = error.message.match(/preserved backup: (.+)\z/).captures.fetch(0)
          assert File.file?(retained), "reported backup was removed: #{retained}"
          assert_equal "old state", File.binread(retained)
          refute marker.exist?
        end
      end

      def test_atomic_state_writer_restores_the_marker_when_cancellation_follows_its_rename
        Dir.mktmpdir("cohere-state-rename-cancellation") do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          marker = root.join("marker.json")
          source.binwrite("audio")
          marker.binwrite("old state")
          renamer = lambda do |bound, from, to, phase|
            bound.rename(from, to)
            raise Interrupt, "cancelled immediately after rename" if phase == :commit
          end

          assert_raises(Interrupt) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              rename: renamer
            )
          end

          assert_equal "old state", marker.binread
          assert_empty(root.children.select { |path| %w[.tmp .bak].include?(path.extname) })
        end
      end

      def test_output_lock_contends_across_processes_and_releases
        Dir.mktmpdir do |directory|
          target = State.lock_target_for_outputs("txt" => Pathname(directory).join("clip.txt"))
          lock = State::OutputSetLock.acquire(target)
          begin
            assert_equal 23, run_lock_child(target)
          ensure
            lock.release
          end
          assert_equal 0, run_lock_child(target)
        end
      end

      def test_different_output_stems_have_independent_locks
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          first = State.lock_target_for_outputs("txt" => root.join("first.txt"))
          second = State.lock_target_for_outputs("txt" => root.join("second.txt"))
          refute_equal first.path, second.path
          lock = State::OutputSetLock.acquire(first)
          begin
            assert_equal 0, run_lock_child(second)
          ensure
            lock.release
          end
        end
      end

      def test_same_process_lock_is_unique_and_interrupt_releases
        Dir.mktmpdir do |directory|
          target = State.lock_target_for_outputs("txt" => Pathname(directory).join("clip.txt"))
          lock = State::OutputSetLock.acquire(target)
          begin
            error = assert_raises(TranscriptionRuntimeError) { State::OutputSetLock.acquire(target) }
            assert_match(/owns output set/, error.message)
          ensure
            lock.release
          end

          assert_raises(Interrupt) do
            State.with_output_lock(target) { raise Interrupt, "cancelled" }
          end
          reacquired = State::OutputSetLock.acquire(target)
          reacquired.release
        end
      end

      def test_lock_open_rejects_symlink_and_public_permissions
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          private_directory = root.join("private")
          private_directory.mkdir(0o700)
          victim = root.join("victim")
          victim.binwrite("untouched")
          target = State::OutputLockTarget.new(
            path: private_directory.join("output.lock"),
            identity: "test output"
          )
          begin
            target.path.make_symlink(victim)
          rescue NotImplementedError, Errno::EPERM
            skip "symbolic links are unavailable"
          end
          assert_raises(TranscriptionRuntimeError) { State::OutputSetLock.acquire(target) }
          assert_equal "untouched", victim.read

          target.path.delete
          private_directory.chmod(0o755)
          error = assert_raises(TranscriptionRuntimeError) { State::OutputSetLock.acquire(target) }
          assert_match(/permissions.*0700/, error.message)
        end
      end

      def test_lock_rechecks_the_path_identity_after_flock
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          private_directory = root.join("private")
          private_directory.mkdir(0o700)
          target = State::OutputLockTarget.new(
            path: private_directory.join("output.lock"),
            identity: "test output"
          )
          target.path.binwrite("")
          target.path.chmod(0o600)
          parked = private_directory.join("parked.lock")
          hook = lambda do |phase, _candidate|
            next unless phase == :after_flock

            target.path.rename(parked)
            target.path.binwrite("")
            target.path.chmod(0o600)
          end

          error = assert_raises(TranscriptionRuntimeError) do
            State::OutputSetLock.acquire(target, operation_hook: hook)
          end
          assert_match(/changed while acquiring/, error.message)
          assert_empty State::OutputSetLock.active
          File.open(parked, "r+") do |handle|
            assert handle.flock(File::LOCK_EX | File::LOCK_NB), "rejected lock descriptor remained locked"
          end

          replacement = State::OutputSetLock.acquire(target)
          replacement.release
        end
      end

      def test_signal_during_lock_acquisition_closes_and_unlocks_the_descriptor
        skip "Linux file descriptor accounting is unavailable" unless Pathname("/proc/self/fd").directory?

        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          private_directory = root.join("private")
          private_directory.mkdir(0o700)
          target = State::OutputLockTarget.new(
            path: private_directory.join("output.lock"),
            identity: "test output"
          )
          descriptors_before = Pathname("/proc/self/fd").children.length
          garbage_collection_was_disabled = GC.disable
          begin
            assert_raises(SignalException) do
              State::OutputSetLock.acquire(
                target,
                operation_hook: ->(*) { raise SignalException, "TERM" }
              )
            end
            assert_equal descriptors_before, Pathname("/proc/self/fd").children.length
            assert_empty State::OutputSetLock.active

            replacement = State::OutputSetLock.acquire(target)
            replacement.release
          ensure
            GC.enable unless garbage_collection_was_disabled
          end
        end
      end

      def test_sequential_many_stem_locks_do_not_accumulate_descriptors
        skip "Linux file descriptor accounting is unavailable" unless Pathname("/proc/self/fd").directory?

        Dir.mktmpdir do |directory|
          before = Pathname("/proc/self/fd").children.length
          1_500.times do |index|
            target = State.lock_target_for_outputs(
              "txt" => Pathname(directory).join("directory-#{index}", "clip.txt")
            )
            State.with_output_lock(target) { nil }
          end
          after = Pathname("/proc/self/fd").children.length
          assert_operator after - before, :<=, 1
        end
      end

      private

      def rename_with_temporary_cleanup_failure
        lambda do |bound, source, destination, _phase|
          bound.rename(source, destination)
          unlink = bound.method(:unlink)
          bound.define_singleton_method(:unlink) do |name, missing_ok: false|
            raise Errno::EIO, "temporary cleanup failed" if name.end_with?(".tmp")

            unlink.call(name, missing_ok: missing_ok)
          end
        end
      end

      def options
        @options ||= TranscriptionOptions.new(
          model_revision: DEFAULT_ASR_MODEL_REVISION,
          device: "cpu",
          dtype: "fp32",
          audio_backend: "ffmpeg",
          vad: "none",
          max_dur: 30.0,
          alignment: "segment",
          publication: PublicationOptions.new(
            formats: %w[txt srt], output_dir: "out", existing: "overwrite"
          )
        )
      end

      def result(path)
        TranscriptionResult.new(
          path: path,
          relative_path: Pathname("clip.wav"),
          status: "completed",
          text: "hello world",
          duration: 1.0,
          segments: [
            TranscriptionSegment.new(index: 0, start: 0.0, end: 1.0, text: "hello world")
          ],
          provenance: TranscriptionProvenance.new(
            model_id: DEFAULT_ASR_MODEL_ID,
            model_revision: DEFAULT_ASR_MODEL_REVISION,
            model_format: "dense",
            decode_backend: "ffmpeg",
            vad_engine_actual: "none",
            generated_tokens_by_segment: [[0, 2]]
          )
        )
      end

      def run_lock_child(target)
        script = <<~RUBY
          require "cohere/transcribe"
          target = Cohere::Transcribe::State::OutputLockTarget.new(
            path: Pathname(ARGV.fetch(0)), identity: ARGV.fetch(1)
          )
          begin
            lock = Cohere::Transcribe::State::OutputSetLock.acquire(target)
          rescue Cohere::Transcribe::TranscriptionRuntimeError
            exit 23
          end
          lock.release
        RUBY
        library = Pathname(__dir__).join("../../lib").expand_path
        _stdout, _stderr, status = Open3.capture3(
          RbConfig.ruby, "-I#{library}", "-e", script, target.path.to_s, target.identity
        )
        status.exitstatus
      end
    end
  end
end
