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
            completed = false
            failed = false
            guard.synchronize do
              handles = []
              registered_keys = []
              lock = nil
              paths = State.lock_paths_for_target(target)
              keys = paths.map(&:to_s)
              if keys.any? { |key| active.key?(key) }
                raise TranscriptionRuntimeError,
                      "Another transcription job owns output set #{target.identity}"
              end

              paths.each_with_index do |path, index|
                shared = path == State.canonical_lock_path(target.identity)
                handle = nil
                Thread.handle_interrupt(Object => :never) do
                  handle = State.open_lock_file(path, shared: shared)
                  handles << handle
                end
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
                State.refresh_legacy_lock(handle) if path == State.legacy_temporary_lock_path(target.identity)
              end

              lock = new(target, handles, keys)
              keys.each do |key|
                registered_keys << key
                active[key] = lock
              end
              ownership_hook&.call(lock)
              completed = true
              lock
            rescue Errno::EWOULDBLOCK, Errno::EAGAIN
              failed = true
              raise TranscriptionRuntimeError,
                    "Another transcription process owns output set #{target.identity}"
            rescue SystemCallError => e
              failed = true
              raise TranscriptionRuntimeError,
                    "Cannot acquire output lock for #{target.identity}: #{e.message}",
                    cause: e
            rescue Exception # rubocop:disable Lint/RescueException -- record only this acquisition's unwind
              failed = true
              raise
            ensure
              unless completed && !failed && Thread.current.status != "aborting"
                Thread.handle_interrupt(Object => :never) do
                  delete_active_keys(lock, registered_keys || [])
                  close_lock_handles(handles || [])
                end
              end
            end
          end

          def release_all
            guard.synchronize { active.values.uniq }.each(&:release)
          end

          def close_lock_handles(handles)
            first_error = nil
            Thread.handle_interrupt(Object => :never) do
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
          Thread.handle_interrupt(Object => :never) do
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
        primary_error = nil
        Thread.handle_interrupt(Object => :never) do
          Thread.handle_interrupt(Object => :on_blocking) do
            OutputSetLock.acquire(
              target,
              blocking: blocking,
              ownership_hook: ->(acquired_lock) { lock = acquired_lock }
            )
          end
          Thread.handle_interrupt(Object => :immediate) { yield lock }
        rescue Exception => e # rubocop:disable Lint/RescueException -- preserve protected-work failures
          primary_error = e
          raise
        ensure
          begin
            lock&.release
          rescue Exception # rubocop:disable Lint/RescueException -- preserve protected-work failures
            raise unless primary_error || Thread.current.status == "aborting"
          end
        end
      end

      def lock_paths_for_target(target)
        primary = Pathname(target.path).expand_path.cleanpath
        return [primary].freeze unless primary == canonical_lock_path(target.identity)

        # Remove this temporary-registry compatibility path in 0.2.0. Until
        # then, acquiring both paths keeps released 0.1.2 processes coordinated
        # with output-adjacent locks.
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

      def open_lock_file(path, shared: false)
        path = Pathname(path)
        succeeded = false
        created = false
        validate_lock_directory!(path.dirname, shared: shared)
        inspect_lock_path!(path)
        flags = File::RDWR
        flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
        flags |= File::CLOEXEC if defined?(File::CLOEXEC)
        mode = shared ? shared_lock_file_mode(path.dirname.lstat) : 0o600
        begin
          descriptor = ::IO.sysopen(path.to_s, flags | File::CREAT | File::EXCL, mode)
          created = true
          access = "r+"
        rescue Errno::EEXIST
          begin
            descriptor = ::IO.sysopen(path.to_s, flags, mode)
            access = "r+"
          rescue Errno::EACCES, Errno::EROFS
            raise unless shared

            readonly_flags = File::RDONLY
            readonly_flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
            readonly_flags |= File::CLOEXEC if defined?(File::CLOEXEC)
            descriptor = ::IO.sysopen(path.to_s, readonly_flags)
            access = "r"
          end
        end
        handle = File.new(descriptor, access, autoclose: true)
        descriptor = nil
        opened = verify_lock_identity!(
          path,
          handle,
          message: "Output lock changed while it was being opened or is not regular"
        )
        if shared
          if created || (access == "r+" && Process.respond_to?(:uid) && opened.uid == Process.uid)
            apply_lock_mode_if_supported(handle, mode)
          end
        else
          validate_owned_private_file!(opened, path)
        end
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

      def validate_lock_directory!(path, shared: false)
        if shared
          create_shared_lock_directory(path)
        else
          begin
            FileUtils.mkdir_p(path, mode: 0o700)
          rescue SystemCallError => e
            raise TranscriptionRuntimeError, "Cannot prepare output lock directory #{path}: #{e.message}"
          end
        end
        stat = path.lstat
        raise TranscriptionRuntimeError, "Output lock directory is not a real directory: #{path}" unless stat.directory? && !stat.symlink?

        return if shared
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

      def create_shared_lock_directory(path)
        parent = path.dirname.lstat
        unless parent.directory? && !parent.symlink?
          raise TranscriptionRuntimeError, "Output lock parent is not a real directory: #{path.dirname}"
        end

        mode = (parent.mode & 0o3777) | 0o700
        begin
          Dir.mkdir(path, mode)
          apply_lock_mode_if_supported(path, mode)
        rescue Errno::EEXIST
          existing = path.lstat
          if !Gem.win_platform? && Process.respond_to?(:uid) && existing.directory? && existing.uid == Process.uid
            apply_lock_mode_if_supported(path, mode)
          end
        rescue SystemCallError => e
          raise TranscriptionRuntimeError, "Cannot prepare output lock directory #{path}: #{e.message}"
        end
      end

      def shared_lock_file_mode(directory_stat)
        mode = 0o600
        mode |= 0o060 if directory_stat.mode.anybits?(0o020) && directory_stat.mode.anybits?(0o010)
        mode |= 0o006 if directory_stat.mode.anybits?(0o002) && directory_stat.mode.anybits?(0o001)
        mode
      end

      def apply_lock_mode_if_supported(target, mode)
        target.chmod(mode)
      rescue SystemCallError => e
        unsupported = [Errno::EPERM::Errno]
        unsupported << Errno::EOPNOTSUPP::Errno if defined?(Errno::EOPNOTSUPP)
        unsupported << Errno::ENOTSUP::Errno if defined?(Errno::ENOTSUP)
        raise unless unsupported.include?(e.errno)

        nil
      end

      def refresh_legacy_lock(handle)
        handle.rewind
        handle.write("#{Process.pid}\n")
        handle.flush
        handle.truncate(handle.pos)
        nil
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
