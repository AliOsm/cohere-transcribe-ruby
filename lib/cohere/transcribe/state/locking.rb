# frozen_string_literal: true

require "digest"
require "fileutils"
require "tmpdir"

module Cohere
  module Transcribe
    module State
      LOCK_DIRECTORY_PREFIX = "cohere-transcribe"
      LOCK_DIRECTORY_NAME = ".cohere-transcribe-locks"
      LOCK_SUFFIX = ".lock"

      OutputLockTarget = Data.define(:path, :identity) do
        def sort_key
          [path.to_s.downcase, identity.downcase]
        end
      end

      class OutputSetLock
        @guard = Mutex.new
        @active = {}

        class << self
          attr_reader :guard, :active

          def acquire(target, blocking: false, operation_hook: nil, ownership_hook: nil)
            guard.synchronize do
              handles = []
              registered_keys = []
              acquired = false
              lock = nil
              paths = State.lock_paths_for_target(target)
              keys = paths.map(&:to_s)
              if keys.any? { |key| active.key?(key) }
                raise TranscriptionRuntimeError,
                      "Another transcription job owns output set #{target.identity}"
              end

              paths.each_with_index do |path, index|
                handle = State.open_lock_file(path)
                handles << handle
                operation = File::LOCK_EX
                operation |= File::LOCK_NB unless blocking
                unless handle.flock(operation)
                  raise TranscriptionRuntimeError,
                        "Another transcription process owns output set #{target.identity} (lock #{path})"
                end
                operation_hook&.call(:after_flock, target) if index.zero?
                State.verify_lock_identity!(
                  path,
                  handle,
                  message: "Output lock changed while acquiring #{target.identity}"
                )
              end

              lock = new(target, handles, keys)
              keys.each do |key|
                registered_keys << key
                active[key] = lock
              end
              ownership_hook&.call(lock)
              acquired = true
              lock
            rescue Errno::EWOULDBLOCK, Errno::EAGAIN
              raise TranscriptionRuntimeError,
                    "Another transcription process owns output set #{target.identity}"
            ensure
              primary_error = $! # rubocop:disable Style/SpecialGlobalVars
              unless acquired && primary_error.nil?
                cleanup_error = nil
                Thread.handle_interrupt(Exception => :never) do
                  registration_error = delete_active_keys(lock, registered_keys || [])
                  handle_error = close_lock_handles(handles || [])
                  cleanup_error = registration_error || handle_error
                end
                raise cleanup_error if cleanup_error && primary_error.nil?
              end
            end
          end

          def release_all
            guard.synchronize { active.values.uniq }.each(&:release)
          end

          def close_lock_handles(handles)
            first_error = nil
            Thread.handle_interrupt(Exception => :never) do
              handles.reverse_each do |handle|
                begin
                  next if handle.closed?
                rescue Exception => e # rubocop:disable Lint/RescueException
                  first_error ||= e
                end

                begin
                  handle.flock(File::LOCK_UN)
                rescue Exception => e # rubocop:disable Lint/RescueException
                  first_error ||= e
                end
                begin
                  handle.close
                rescue Exception => e # rubocop:disable Lint/RescueException
                  first_error ||= e
                end
              end
            end
            first_error
          end

          def delete_active_keys(lock, keys)
            first_error = nil
            keys.reverse_each do |key|
              active.delete(key) if active[key].equal?(lock)
            rescue Exception => e # rubocop:disable Lint/RescueException
              first_error ||= e
            end
            first_error
          end
        end

        attr_reader :target

        def initialize(target, handles, keys)
          @target = target
          @handles = handles
          @keys = keys
          @released = false
        end

        def release
          cleanup_error = nil
          Thread.handle_interrupt(Exception => :never) do
            self.class.guard.synchronize do
              return if @released

              begin
                handle_error = self.class.close_lock_handles(@handles)
                registration_error = self.class.delete_active_keys(self, @keys)
                cleanup_error = handle_error || registration_error
              ensure
                @released = true
              end
            end
          end
          raise cleanup_error if cleanup_error

          nil
        end

        def released?
          @released
        end
      end

      module_function

      def lock_target_for_outputs(output_paths)
        parent, stem = output_parent_and_stem(output_paths)
        identity = parent.join(stem).expand_path.cleanpath.to_s
        OutputLockTarget.new(
          path: canonical_lock_path(identity).freeze,
          identity: identity.freeze
        )
      end

      def with_output_lock(target, blocking: false)
        lock = nil
        completed = false
        Thread.handle_interrupt(Exception => :never) do
          Thread.handle_interrupt(Exception => :on_blocking) do
            OutputSetLock.acquire(
              target,
              blocking: blocking,
              ownership_hook: ->(acquired_lock) { lock = acquired_lock }
            )
          end
          result = Thread.handle_interrupt(Exception => :immediate) { yield lock }
          completed = true
          result
        ensure
          begin
            lock&.release
          rescue Exception # rubocop:disable Lint/RescueException
            raise if completed
          end
        end
      end

      def lock_paths_for_target(target)
        primary = Pathname(target.path).expand_path.cleanpath
        return [primary].freeze unless primary == canonical_lock_path(target.identity)

        compatibility = [legacy_temporary_lock_path(target.identity)]
        ([primary] + compatibility.reject { |path| path == primary }).freeze
      end

      def canonical_lock_path(identity)
        identity = Pathname(identity).expand_path.cleanpath
        lock_registry_directory(identity.dirname).join(lock_filename(identity.to_s))
      end

      def lock_registry_directory(output_parent)
        Pathname(output_parent).expand_path.cleanpath.join(LOCK_DIRECTORY_NAME)
      end

      def legacy_temporary_lock_path(identity)
        scope = Process.respond_to?(:uid) ? Process.uid.to_s : Digest::SHA256.hexdigest(Dir.home)[0, 16]
        Pathname(Dir.tmpdir).expand_path.join("#{LOCK_DIRECTORY_PREFIX}-#{scope}", lock_filename(identity))
      end

      def lock_filename(identity)
        "#{Digest::SHA256.hexdigest(identity.downcase)}#{LOCK_SUFFIX}"
      end

      def open_lock_file(path)
        path = Pathname(path)
        succeeded = false
        validate_lock_directory!(path.dirname)
        inspect_lock_path!(path)
        flags = File::RDWR | File::CREAT
        flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
        flags |= File::CLOEXEC if defined?(File::CLOEXEC)
        descriptor = ::IO.sysopen(path.to_s, flags, 0o600)
        handle = File.new(descriptor, "r+", autoclose: true)
        descriptor = nil
        opened = verify_lock_identity!(
          path,
          handle,
          message: "Output lock changed while it was being opened or is not regular"
        )
        validate_owned_private_file!(opened, path)
        succeeded = true
        handle
      rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
        raise TranscriptionRuntimeError, "Output lock is not a regular file: #{path}", cause: e
      ensure
        if !succeeded && handle
          handle.close
        elsif descriptor
          ::IO.new(descriptor).close
        end
      end

      def validate_lock_directory!(path)
        begin
          FileUtils.mkdir_p(path, mode: 0o700)
        rescue SystemCallError => e
          raise TranscriptionRuntimeError, "Cannot prepare output lock directory #{path}: #{e.message}"
        end
        stat = path.lstat
        raise TranscriptionRuntimeError, "Output lock directory is not a real directory: #{path}" unless stat.directory? && !stat.symlink?

        return if Gem.win_platform?

        if Process.respond_to?(:uid) && stat.uid != Process.uid
          raise TranscriptionRuntimeError,
                "Output lock directory is not owned by the current user: #{path}"
        end
        return if stat.mode.nobits?(0o077)

        raise TranscriptionRuntimeError,
              "Output lock directory permissions must be private (0700): #{path}"
      rescue Errno::ENOENT, Errno::ENOTDIR => e
        raise TranscriptionRuntimeError, "Cannot prepare output lock directory #{path}: #{e.message}"
      end

      def inspect_lock_path!(path)
        stat = path.lstat
        return if stat.file? && !stat.symlink?

        raise TranscriptionRuntimeError, "Output lock is not a regular file: #{path}"
      rescue Errno::ENOENT
        nil
      end

      def validate_owned_private_file!(stat, path)
        return if Gem.win_platform?

        if Process.respond_to?(:uid) && stat.uid != Process.uid
          raise TranscriptionRuntimeError, "Output lock is not owned by the current user: #{path}"
        end
        return if stat.mode.nobits?(0o077)

        raise TranscriptionRuntimeError,
              "Output lock permissions must be private (0600): #{path}"
      end

      def verify_lock_identity!(path, handle, message:)
        opened = handle.stat
        current = Pathname(path).lstat
        return opened if opened.file? && same_file?(opened, current)

        raise TranscriptionRuntimeError, "#{message}: #{path}"
      rescue SystemCallError => e
        raise TranscriptionRuntimeError, "#{message}: #{path}", cause: e
      end
    end
  end
end
