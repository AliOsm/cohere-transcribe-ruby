# frozen_string_literal: true

require "test_helper"
require "tmpdir"

module Cohere
  module Transcribe
    class StateAtomicRecoveryTest < Minitest::Test
      def test_nonlocal_throw_after_rename_restores_the_previous_state
        with_state_files("cohere-state-throw") do |root, source, marker|
          outcome = catch(:stop_state_write) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              operation_hook: lambda do |phase, _path|
                throw :stop_state_write, :stopped if phase == :after_rename
              end
            )
          end

          assert_equal :stopped, outcome
          assert_equal "old state", marker.binread
          assert_empty transaction_files(root)
        end
      end

      def test_thread_kill_after_a_custom_rename_restores_the_previous_state
        with_state_files("cohere-state-kill") do |root, source, marker|
          rename = lambda do |bound, from, to, phase|
            bound.rename(from, to)
            Thread.handle_interrupt(Object => :immediate) { Thread.current.kill } if phase == :commit
          end
          caller = Thread.new do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              rename: rename
            )
          end
          caller.report_on_exception = false

          assert caller.join(2), "state writer remained stuck after termination"
          assert_nil caller.value
          assert_equal "old state", marker.binread
          assert_empty transaction_files(root)
        ensure
          caller&.kill
          caller&.join
        end
      end

      def test_incomplete_state_recovery_is_typed_and_preserves_the_primary_failure
        with_state_files("cohere-state-recovery") do |root, source, marker|
          rename = lambda do |bound, from, to, phase|
            raise Errno::EIO, "forced restore failure" if phase == :rollback

            bound.rename(from, to)
          end

          error = assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              operation_hook: lambda do |phase, _path|
                raise "forced primary failure" if phase == :after_rename
              end,
              rename: rename
            )
          end

          assert_match(/recovery was incomplete/, error.message)
          assert_match(/forced restore failure/, error.message)
          assert_instance_of RuntimeError, error.cause
          assert_equal "forced primary failure", error.cause.message
          retained = transaction_files(root, ".bak")
          assert_equal 1, retained.length
          assert_equal "old state", retained.fetch(0).binread
          assert_includes error.message, retained.fetch(0).to_s
          refute marker.exist?
        end
      end

      def test_state_directory_close_failure_is_typed_and_preserves_the_primary_failure
        original_close = State::BoundDirectory.instance_method(:close)
        closed = []
        State::BoundDirectory.define_method(:close) do
          closed << binding.canonical_path
          original_close.bind_call(self)
          raise Errno::EIO, "forced state directory close failure"
        end

        with_state_files("cohere-state-close") do |root, source, marker|
          error = assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root),
              operation_hook: lambda do |phase, _path|
                raise "forced state primary failure" if phase == :before_stage_write
              end
            )
          end

          assert_match(/close .*forced state directory close failure/, error.message)
          assert_instance_of RuntimeError, error.cause
          assert_equal "forced state primary failure", error.cause.message
          assert_equal [root.realpath], closed
          assert_equal "old state", marker.binread
          assert_empty transaction_files(root)
        end
      ensure
        State::BoundDirectory.define_method(:close, original_close) if original_close
      end

      def test_bound_parent_close_failure_preserves_an_ordinary_callers_failure
        original_close = State::BoundDirectory.instance_method(:close)
        State::BoundDirectory.define_method(:close) do
          original_close.bind_call(self)
          raise Errno::EIO, "forced ordinary bound close failure"
        end

        with_state_files("cohere-bound-close") do |root, _source, marker|
          error = assert_raises(TranscriptionRuntimeError) do
            State.with_bound_parent(
              marker,
              directory_binding: State::DirectoryBinding.capture(root)
            ) do
              raise "forced ordinary bound failure"
            end
          end

          assert_match(/Publication directory operation failed/, error.message)
          assert_match(/forced ordinary bound close failure/, error.message)
          assert_instance_of RuntimeError, error.cause
          assert_equal "forced ordinary bound failure", error.cause.message
        end
      ensure
        State::BoundDirectory.define_method(:close, original_close) if original_close
      end

      def test_resource_close_errors_are_aggregated_without_skipping_or_closing_twice
        calls = Hash.new(0)
        first = Object.new
        second = Object.new
        first.define_singleton_method(:close) do
          calls[:first] += 1
          raise IOError, "first close failed"
        end
        second.define_singleton_method(:close) { calls[:second] += 1 }
        labels = { first => "first", second => "second" }
        errors = []

        State.close_atomic_resources(
          [first, second, first],
          errors,
          label: ->(resource) { labels.fetch(resource) }
        )

        assert_equal({ first: 1, second: 1 }, calls)
        assert_equal ["close first: first close failed"], errors
      end

      def test_state_backup_copy_closes_both_handles_and_types_the_first_close_failure
        original_open = State::BoundDirectory.instance_method(:open_regular)
        original_create = State::BoundDirectory.instance_method(:create_temporary)
        source_opens = 0
        close_calls = Hash.new(0)
        State::BoundDirectory.define_method(:open_regular) do |name, writable: false|
          handle = original_open.bind_call(self, name, writable: writable)
          source_opens += 1 if name == "marker.json"
          if name == "marker.json" && source_opens == 2
            close = handle.method(:close)
            handle.define_singleton_method(:close) do
              close_calls[:source] += 1
              raise Errno::EIO, "forced state source close failure" if close_calls[:source] == 1

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

        with_state_files("cohere-state-copy-close") do |root, source, marker|
          error = assert_raises(TranscriptionRuntimeError) do
            State.write_state_atomic(
              marker,
              { "kind" => "new" },
              source_snapshot: State::SourceSnapshot.capture(source),
              directory_binding: State::DirectoryBinding.capture(root)
            )
          end

          assert_match(/close .*forced state source close failure/, error.message)
          assert_instance_of Errno::EIO, error.cause
          assert_equal({ source: 2, backup: 1 }, close_calls)
          assert_equal "old state", marker.binread
          assert_empty transaction_files(root)
        end
      ensure
        State::BoundDirectory.define_method(:open_regular, original_open) if original_open
        State::BoundDirectory.define_method(:create_temporary, original_create) if original_create
      end

      private

      def with_state_files(prefix)
        Dir.mktmpdir(prefix) do |directory|
          root = Pathname(directory)
          source = root.join("clip.wav")
          marker = root.join("marker.json")
          source.binwrite("audio")
          marker.binwrite("old state")
          yield root, source, marker
        end
      end

      def transaction_files(root, extension = nil)
        extensions = extension ? [extension] : %w[.tmp .bak]
        root.children.select { |path| extensions.include?(path.extname) }
      end
    end
  end
end
