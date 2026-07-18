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
      LEGACY_LOCK_REMOVAL_VERSION = "0.2.0"

      OutputLockTarget = Data.define(:path, :identity)
      LockPath = Data.define(:path, :role) do
        def canonical?
          role == :canonical
        end

        def legacy?
          role == :legacy
        end
      end
      private_constant :LockPath

      class OutputSetLock
        @guard = Mutex.new
        @active = {}

        class << self
          attr_reader :guard, :active

          def acquire(target, blocking: false, operation_hook: nil, ownership_hook: nil)
            State.ensure_lock_acquisition_allowed!(target)
            completed = false
            failed = false
            guard.synchronize do
              handles = []
              lock = nil
              lock_paths = State.lock_path_specs_for_target(target)
              paths = lock_paths.map(&:path)
              keys = paths.map(&:to_s)
              if keys.any? { |key| active.key?(key) }
                raise TranscriptionRuntimeError,
                      "Another transcription job owns output set #{target.identity}"
              end

              lock_paths.each_with_index do |lock_path, index|
                path = lock_path.path
                handle = nil
                Thread.handle_interrupt(Object => :never) do
                  handle = State.open_lock_file(path, shared: lock_path.canonical?)
                  handles << handle
                end
                operation = File::LOCK_EX
                operation |= File::LOCK_NB unless blocking
                unless State.acquire_exclusive_file_lock(handle, operation, target: target, path: path)
                  raise TranscriptionRuntimeError,
                        "Another transcription process owns output set #{target.identity} (lock #{path})"
                end
                operation_hook&.call(:after_flock, target) if index.zero?
                State.verify_lock_identity!(
                  path,
                  handle,
                  message: "Output lock changed while acquiring #{target.identity}"
                )
                State.refresh_legacy_lock(handle) if lock_path.legacy?
              end

              lock = new(target, handles, keys)
              keys.each do |key|
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
              unless completed && !failed && !State.lock_acquisition_unwinding?
                Thread.handle_interrupt(Object => :never) do
                  delete_active_keys(lock, keys || [])
                  close_lock_handles(handles || [])
                  lock&.mark_released!
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
          @handles = handles.freeze
          @keys = keys.freeze
          @released = false
        end

        def verify!
          self.class.guard.synchronize do
            if @released
              raise TranscriptionRuntimeError,
                    "Output lock was released before committing #{target.identity}"
            end

            @keys.zip(@handles).each do |path, handle|
              State.verify_lock_identity!(
                path,
                handle,
                message: "Output lock changed while held for #{target.identity}"
              )
            end
          end
          nil
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

        def mark_released!
          @released = true
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
            raise unless primary_error || lock_acquisition_unwinding?
          end
        end
      end

      def lock_paths_for_target(target)
        lock_path_specs_for_target(target).map(&:path).freeze
      end

      def lock_path_specs_for_target(target)
        primary = Pathname(target.path).expand_path.cleanpath
        canonical = canonical_lock_path(target.identity)
        legacy = legacy_temporary_lock_path(target.identity)
        unless primary == canonical
          role = primary == legacy ? :legacy : :custom
          return [LockPath.new(path: primary.freeze, role: role)].freeze
        end

        # Remove this temporary-registry compatibility path in 0.2.0. Until
        # then, acquiring both paths keeps released 0.1.2 processes coordinated
        # with output-adjacent locks.
        paths = [LockPath.new(path: primary.freeze, role: :canonical)]
        paths << LockPath.new(path: legacy.freeze, role: :legacy) if VERSION.start_with?("0.1.") && legacy != primary
        paths.freeze
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
          descriptor, created = open_or_create_lock_descriptor(path, flags, mode)
          access = "r+"
        rescue Errno::EACCES, Errno::EROFS => e
          raise unless shared

          handle, access = open_shared_lock_after_write_denial(path, flags, mode, e)
        end
        unless handle
          handle = File.new(descriptor, access, autoclose: true)
          descriptor = nil
        end
        opened = verify_lock_identity!(
          path,
          handle,
          message: "Output lock changed while it was being opened or is not regular"
        )
        if shared
          owned_writable = access == "r+" && Process.respond_to?(:uid) && opened.uid == Process.uid
          apply_lock_mode_if_supported(handle, mode) if (created || owned_writable) && (opened.mode & 0o777) != mode
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

      def open_or_create_lock_descriptor(path, flags, mode)
        2.times do
          return [::IO.sysopen(path.to_s, flags, mode), false]
        rescue Errno::ENOENT
          begin
            return [::IO.sysopen(path.to_s, flags | File::CREAT | File::EXCL, mode), true]
          rescue Errno::EEXIST
            next
          end
        end

        raise Errno::ENOENT, path.to_s
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
        mode |= 0o1000 if mode.anybits?(0o022)
        begin
          Dir.mkdir(path, mode)
          apply_lock_mode_if_supported(path, mode)
        rescue Errno::EEXIST
          existing = path.lstat
          if !Gem.win_platform? && Process.respond_to?(:uid) && existing.directory? &&
             existing.uid == Process.uid && (existing.mode & 0o3777) != mode
            apply_lock_mode_if_supported(path, mode)
          end
        rescue SystemCallError => e
          raise TranscriptionRuntimeError, "Cannot prepare output lock directory #{path}: #{e.message}"
        end
        validate_shared_lock_directory_mode!(path)
      end

      def validate_shared_lock_directory_mode!(path)
        return if Gem.win_platform?

        stat = path.lstat
        return unless stat.mode.anybits?(0o022)
        return if stat.mode.anybits?(0o1000)

        raise TranscriptionRuntimeError,
              "Shared output lock directory must use the sticky bit: #{path}"
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
        unsupported << Errno::EROFS::Errno
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

      def acquire_exclusive_file_lock(handle, operation, target:, path:)
        handle.flock(operation)
      rescue Errno::EBADF => e
        raise TranscriptionRuntimeError,
              "Cannot acquire an exclusive output lock for #{target.identity}: " \
              "the filesystem requires a writable lock file at #{path}",
              cause: e
      end

      def open_shared_lock_after_write_denial(path, flags, mode, original_error)
        descriptor = ::IO.sysopen(path.to_s, readonly_lock_flags)
        readonly = File.new(descriptor, "r", autoclose: true)
        descriptor = nil
        opened = verify_lock_identity!(
          path,
          readonly,
          message: "Output lock changed while opening it read-only"
        )
        if original_error.is_a?(Errno::EACCES) &&
           Process.respond_to?(:uid) && opened.uid == Process.uid
          apply_lock_mode_if_supported(readonly, mode)
          begin
            descriptor = ::IO.sysopen(path.to_s, flags, mode)
            writable = File.new(descriptor, "r+", autoclose: true)
            descriptor = nil
            readonly.close
            readonly = nil
            handle = writable
            writable = nil
            return [handle, "r+"]
          rescue Errno::EACCES, Errno::EROFS
            nil
          end
        end

        handle = readonly
        readonly = nil
        [handle, "r"]
      ensure
        readonly&.close unless readonly&.closed?
        writable&.close unless writable&.closed?
        ::IO.new(descriptor).close if descriptor
      end

      def readonly_lock_flags
        readonly_flags = File::RDONLY
        readonly_flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
        readonly_flags |= File::CLOEXEC if defined?(File::CLOEXEC)
        readonly_flags
      end

      def ensure_lock_acquisition_allowed!(target)
        return unless lock_acquisition_unwinding?

        raise TranscriptionRuntimeError,
              "Cannot acquire output lock for #{target.identity} while the current thread is terminating"
      end

      # Ruby exposes an in-progress Thread#kill unwind only through #status;
      # pending_interrupt? is already false while ensure blocks are running.
      def lock_acquisition_unwinding?
        Thread.current.status == "aborting"
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
