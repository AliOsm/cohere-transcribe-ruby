# frozen_string_literal: true

require "digest"
require "fileutils"
require "tmpdir"

module Cohere
  module Transcribe
    module State
      LOCK_DIRECTORY_PREFIX = "cohere-transcribe"
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

          def acquire(target, blocking: false, operation_hook: nil)
            guard.synchronize do
              handle = nil
              acquired = false
              key = target.path.to_s
              if active.key?(key)
                raise TranscriptionRuntimeError,
                      "Another transcription job owns output set #{target.identity}"
              end

              handle = State.open_lock_file(target.path)
              operation = File::LOCK_EX
              operation |= File::LOCK_NB unless blocking
              unless handle.flock(operation)
                handle.close
                raise TranscriptionRuntimeError,
                      "Another transcription process owns output set #{target.identity} (lock #{target.path})"
              end
              operation_hook&.call(:after_flock, target)
              State.verify_lock_identity!(
                target.path,
                handle,
                message: "Output lock changed while acquiring #{target.identity}"
              )

              lock = new(target, handle, key)
              active[key] = lock
              acquired = true
              lock
            rescue Errno::EWOULDBLOCK, Errno::EAGAIN
              handle&.close
              raise TranscriptionRuntimeError,
                    "Another transcription process owns output set #{target.identity} (lock #{target.path})"
            rescue Interrupt, SystemExit, StandardError
              handle&.close
              raise
            ensure
              handle.close if handle && !acquired && !handle.closed?
            end
          end

          def release_all
            guard.synchronize { active.values.dup }.each(&:release)
          end
        end

        attr_reader :target

        def initialize(target, handle, key)
          @target = target
          @handle = handle
          @key = key
          @released = false
        end

        def release
          self.class.guard.synchronize do
            return if @released

            begin
              @handle.flock(File::LOCK_UN)
            ensure
              @handle.close
              @released = true
              self.class.active.delete(@key)
            end
          end
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
        digest = Digest::SHA256.hexdigest(identity.downcase)
        OutputLockTarget.new(
          path: lock_registry_directory.join("#{digest}#{LOCK_SUFFIX}").freeze,
          identity: identity.freeze
        )
      end

      def with_output_lock(target, blocking: false)
        lock = OutputSetLock.acquire(target, blocking: blocking)
        yield lock
      ensure
        lock&.release
      end

      def lock_registry_directory
        scope = Process.respond_to?(:uid) ? Process.uid.to_s : Digest::SHA256.hexdigest(Dir.home)[0, 16]
        Pathname(Dir.tmpdir).join("#{LOCK_DIRECTORY_PREFIX}-#{scope}")
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
          Dir.mkdir(path, 0o700)
        rescue Errno::EEXIST
          nil
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
