# frozen_string_literal: true

require "digest"
require "fiddle"
require "json"
require "securerandom"

module Cohere
  module Transcribe
    module State
      STATE_SCHEMA_VERSION = 1
      STATE_SUFFIX = ".cohere-transcribe.manifest.json"
      CHECKPOINT_SUFFIX = ".cohere-transcribe.asr.json"
      MAX_STATE_BYTES = 64 * 1024 * 1024
      DEFERRED_PUBLICATION_EXCEPTIONS = { Object => :never }.freeze

      class PublicationDirectoryChangedError < TranscriptionRuntimeError; end
      class PublicationEntryError < TranscriptionRuntimeError; end
      private_constant :PublicationDirectoryChangedError, :PublicationEntryError

      DirectoryBinding = Data.define(
        :access_path,
        :canonical_path,
        :device,
        :inode
      ) do
        def self.capture(path)
          access_path = Pathname(path).expand_path.cleanpath
          canonical_path = access_path.realpath
          stat = canonical_path.lstat
          unless stat.directory? && !stat.symlink?
            raise TranscriptionRuntimeError, "Publication parent is not a real directory: #{access_path}"
          end

          new(
            access_path: access_path.freeze,
            canonical_path: canonical_path.freeze,
            device: stat.dev,
            inode: stat.ino
          )
        rescue SystemCallError, ArgumentError => e
          raise TranscriptionRuntimeError,
                "Cannot bind publication parent #{path}: #{e.message}"
        end

        def verify!
          resolved = access_path.realpath
          stat = canonical_path.lstat
          valid = resolved == canonical_path && stat.directory? && !stat.symlink? &&
                  stat.dev == device && stat.ino == inode
          return self if valid

          raise PublicationDirectoryChangedError,
                "Publication parent changed after planning: #{access_path}"
        rescue SystemCallError, ArgumentError => e
          raise PublicationDirectoryChangedError,
                "Publication parent changed after planning: #{access_path} (#{e.message})"
        end
      end

      # A retained directory descriptor is the authority for publication I/O.
      # Path checks alone cannot close a rename/symlink TOCTOU window: after
      # this object opens and verifies the planned inode, every entry operation
      # is performed with the POSIX *at family against that descriptor.
      class BoundDirectory
        AT_FUNCTION_SIGNATURES = {
          openat: [
            [Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT, Fiddle::TYPE_VARIADIC],
            Fiddle::TYPE_INT
          ],
          renameat: [
            [Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP],
            Fiddle::TYPE_INT
          ],
          unlinkat: [[Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT], Fiddle::TYPE_INT],
          mkdirat: [[Fiddle::TYPE_INT, Fiddle::TYPE_VOIDP, Fiddle::TYPE_INT], Fiddle::TYPE_INT]
        }.freeze

        class << self
          def open(binding, guards: [binding])
            ensure_supported!
            succeeded = false
            guards = guards.uniq.freeze
            guards.each(&:verify!)
            flags = File::RDONLY | File::NONBLOCK | File.const_get(:NOFOLLOW)
            descriptor = IO.sysopen(binding.canonical_path.to_s, flags)
            handle = File.new(descriptor, "rb", autoclose: true)
            descriptor = nil
            handle.close_on_exec = true
            stat = handle.stat
            unless stat.directory? && stat.dev == binding.device && stat.ino == binding.inode
              raise PublicationDirectoryChangedError,
                    "Publication parent changed while it was being opened: #{binding.access_path}"
            end
            guards.each(&:verify!)
            bound = new(binding, handle, guards)
            succeeded = true
            bound
          rescue Interrupt, SystemExit
            raise
          rescue SystemCallError, ArgumentError => e
            guards&.each(&:verify!)
            raise TranscriptionRuntimeError,
                  "Cannot open planned publication parent #{binding.access_path}: #{e.message}"
          ensure
            handle&.close if handle && !succeeded
            IO.new(descriptor).close if descriptor
          end

          def ensure_supported!
            required_constants = %i[NOFOLLOW]
            missing_constants = required_constants.reject { |name| File.const_defined?(name) }
            missing_functions = AT_FUNCTION_SIGNATURES.keys.reject do |name|
              function(name)
              true
            rescue Fiddle::DLError
              false
            end
            return if missing_constants.empty? && missing_functions.empty?

            details = (missing_constants + missing_functions).join(", ")
            raise TranscriptionRuntimeError,
                  "Descriptor-relative publication is unavailable on this platform (missing #{details})"
          end

          def function(name)
            @function_guard ||= Mutex.new
            @functions ||= {}
            @function_guard.synchronize do
              @functions[name] ||= begin
                arguments, result = AT_FUNCTION_SIGNATURES.fetch(name)
                address = Fiddle::Handle::DEFAULT[name.to_s]
                Fiddle::Function.new(address, arguments, result)
              end
            end
          end
        end

        attr_reader :binding

        def initialize(binding, handle, guards)
          @binding = binding
          @handle = handle
          @guards = guards
          @closed = false
        end

        def create_temporary(basename, suffix)
          validate_entry_name!(basename)
          100.times do
            name = ".#{basename}.#{SecureRandom.hex(12)}#{suffix}"
            begin
              handle = open_entry(
                name,
                File::WRONLY | File::CREAT | File::EXCL | File::NOFOLLOW,
                0o600,
                mode: "wb"
              )
              return [name.freeze, handle]
            rescue Errno::EEXIST
              next
            end
          end
          raise TranscriptionRuntimeError,
                "Cannot allocate a unique publication temporary for #{basename}"
        end

        def open_regular(name, writable: false)
          flags = writable ? File::RDWR : File::RDONLY
          flags |= File::NOFOLLOW | File::NONBLOCK
          handle = open_entry(name, flags, 0, mode: writable ? "r+b" : "rb")
          unless handle.stat.file?
            handle.close
            raise PublicationEntryError,
                  "Publication entry is not a regular file: #{display_path(name)}"
          end
          handle
        rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
          raise PublicationEntryError,
                "Publication entry is not a regular file: #{display_path(name)}",
                cause: e
        end

        def regular_entry?(name)
          handle = nil
          Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
            handle = open_regular(name)
          end
          true
        rescue Errno::ENOENT
          false
        ensure
          handle&.close
        end

        def same_regular_entry?(name, expected_stat)
          handle = nil
          Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
            handle = open_regular(name)
          end
          current = handle.stat
          current.dev == expected_stat.dev && current.ino == expected_stat.ino
        rescue Errno::ENOENT, PublicationEntryError
          false
        ensure
          handle&.close
        end

        def rename(source, destination)
          validate_entry_name!(source)
          validate_entry_name!(destination)
          call_at!(:renameat, descriptor, source, descriptor, destination)
          nil
        end

        def mkdir(name, mode = 0o777)
          validate_entry_name!(name)
          call_at!(:mkdirat, descriptor, name, mode)
          nil
        rescue Errno::EEXIST
          nil
        end

        def open_child_directory(name, access_path:, canonical_path:)
          validate_entry_name!(name)
          handle = open_entry(
            name,
            File::RDONLY | File::NONBLOCK | File::NOFOLLOW,
            0,
            mode: "rb"
          )
          stat = handle.stat
          unless stat.directory?
            raise TranscriptionRuntimeError,
                  "Publication directory component is not a real directory: #{access_path}"
          end
          child_binding = DirectoryBinding.new(
            access_path: Pathname(access_path).freeze,
            canonical_path: Pathname(canonical_path).freeze,
            device: stat.dev,
            inode: stat.ino
          )
          child = self.class.new(child_binding, handle, (@guards + [child_binding]).uniq.freeze)
          child.verify!
          handle = nil
          child
        rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
          raise TranscriptionRuntimeError,
                "Publication directory component is not a real directory: #{access_path}",
                cause: e
        ensure
          handle&.close
        end

        def unlink(name, missing_ok: false)
          validate_entry_name!(name)
          call_at!(:unlinkat, descriptor, name, 0)
          nil
        rescue Errno::ENOENT
          raise unless missing_ok

          nil
        end

        def verify!
          stat = @handle.stat
          unless stat.directory? && stat.dev == binding.device && stat.ino == binding.inode
            raise PublicationDirectoryChangedError,
                  "Retained publication parent changed identity: #{binding.access_path}"
          end
          @guards.each(&:verify!)
          self
        end

        def fsync
          @handle.fsync
        rescue Errno::EACCES, Errno::EBADF, Errno::EINVAL, Errno::EISDIR,
               Errno::ENOTSUP, Errno::EPERM
          nil
        end

        def close
          Thread.handle_interrupt(Object => :never) do
            return if @closed

            @handle.close
            @closed = true
          end
          nil
        end

        def display_path(name)
          binding.canonical_path.join(name)
        end

        private

        def descriptor
          @handle.fileno
        end

        def open_entry(name, flags, permissions, mode:)
          validate_entry_name!(name)
          file_descriptor = call_at!(
            :openat,
            descriptor,
            name,
            flags,
            Fiddle::TYPE_INT,
            permissions
          )
          handle = File.new(file_descriptor, mode, autoclose: true)
          file_descriptor = nil
          handle.close_on_exec = true
          handle
        ensure
          IO.new(file_descriptor).close if file_descriptor
        end

        def call_at!(name, *)
          result = self.class.function(name).call(*)
          return result unless result == -1

          error_number = Fiddle.last_error
          raise SystemCallError.new("#{name} #{binding.canonical_path}", error_number)
        end

        def validate_entry_name!(name)
          value = name.to_s
          return if !value.empty? && value != "." && value != ".." &&
                    !value.include?(File::SEPARATOR) && !value.include?("\0")

          raise ArgumentError, "Invalid descriptor-relative publication name: #{value.inspect}"
        end
      end

      SourceSnapshot = Data.define(
        :canonical_path,
        :device,
        :inode,
        :size,
        :mtime_ns,
        :ctime_ns
      ) do
        def self.capture(path)
          canonical = Pathname(path).realpath
          stat = canonical.stat
          new(
            canonical_path: canonical.to_s.freeze,
            device: stat.dev,
            inode: stat.ino,
            size: stat.size,
            mtime_ns: State.time_nanoseconds(stat.mtime),
            ctime_ns: State.time_nanoseconds(stat.ctime)
          )
        end

        def payload
          {
            "canonical_path" => canonical_path,
            "snapshot" => {
              "device" => device,
              "inode" => inode,
              "size" => size,
              "mtime_ns" => mtime_ns,
              "ctime_ns" => ctime_ns
            }
          }
        end
      end

      module_function

      def with_bound_parent(path, directory_binding: nil, guard_bindings: nil)
        path = Pathname(path).expand_path
        directory_binding ||= DirectoryBinding.capture(path.dirname)
        unless [directory_binding.access_path, directory_binding.canonical_path].include?(path.dirname)
          raise TranscriptionRuntimeError,
                "Publication path is outside its planned parent: #{path}"
        end
        guards = Array(guard_bindings).compact
        guards << directory_binding
        bound = nil
        Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
          bound = BoundDirectory.open(directory_binding, guards: guards.uniq)
        end
        yield bound, path.basename.to_s
      ensure
        bound&.close
      end

      def ensure_bound_directory(path, root_binding: nil, operation_hook: nil)
        target = Pathname(path).expand_path.cleanpath
        bindings = []
        current = nil
        if root_binding
          root_binding.verify!
          relative = target.relative_path_from(root_binding.canonical_path)
          if relative.each_filename.first == ".."
            raise TranscriptionRuntimeError,
                  "Publication directory escapes its bound root: #{target}"
          end
          current_binding = root_binding
          access_cursor = root_binding.canonical_path
          canonical_cursor = root_binding.canonical_path
          components = relative.each_filename.reject { |name| name == "." }
        else
          missing = []
          cursor = target
          until cursor.exist? || cursor.symlink?
            parent = cursor.dirname
            raise TranscriptionRuntimeError, "Cannot resolve publication directory #{target}" if parent == cursor

            missing.unshift(cursor.basename.to_s)
            cursor = parent
          end
          current_binding = DirectoryBinding.capture(cursor)
          access_cursor = current_binding.access_path
          canonical_cursor = current_binding.canonical_path
          components = missing
        end

        Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
          current = BoundDirectory.open(current_binding, guards: [current_binding])
          bindings << current
        end
        operation_hook&.call(:ancestor_opened, current.binding.access_path)
        components.each do |component|
          operation_hook&.call(:before_mkdir, access_cursor.join(component))
          current.mkdir(component)
          operation_hook&.call(:after_mkdir, access_cursor.join(component))
          access_cursor = access_cursor.join(component)
          canonical_cursor = canonical_cursor.join(component)
          Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
            current = current.open_child_directory(
              component,
              access_path: access_cursor,
              canonical_path: canonical_cursor
            )
            bindings << current
          end
        end
        bindings.each(&:verify!)
        current.binding
      rescue ArgumentError => e
        raise TranscriptionRuntimeError, "Cannot bind publication directory #{path}: #{e.message}"
      ensure
        bindings&.reverse_each(&:close)
      end

      def time_nanoseconds(time)
        (time.to_i * 1_000_000_000) + time.nsec
      end

      def output_parent_and_stem(output_paths)
        raise ArgumentError, "An output set must contain at least one path" if output_paths.empty?

        paths = output_paths.values.map { |path| Pathname(path) }
        parent = paths.first.dirname
        stem = paths.first.basename(paths.first.extname).to_s
        unless paths.drop(1).all? do |path|
          path.dirname == parent && path.basename(path.extname).to_s == stem
        end
          raise ArgumentError, "All formats in one output set must share a parent and stem"
        end

        [parent, stem]
      end

      def state_path_for_outputs(output_paths)
        parent, stem = output_parent_and_stem(output_paths)
        parent.join(".#{stem}#{STATE_SUFFIX}")
      end

      def checkpoint_path_for_outputs(output_paths)
        parent, stem = output_parent_and_stem(output_paths)
        parent.join(".#{stem}#{CHECKPOINT_SUFFIX}")
      end

      def canonical_json(value)
        JSON.generate(deep_sort(value))
      end

      def envelope(payload)
        {
          "schema_version" => STATE_SCHEMA_VERSION,
          "payload_sha256" => Digest::SHA256.hexdigest(canonical_json(payload)),
          "payload" => payload
        }
      end

      def decode_state(path, directory_binding: nil, guard_bindings: nil)
        path = Pathname(path)
        contents = securely_read_regular_file(
          path,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        )
        decoded = JSON.parse(contents)
        return [nil, "state marker root is not an object"] unless decoded.is_a?(Hash)
        return [nil, "state marker schema is unsupported"] unless decoded["schema_version"] == STATE_SCHEMA_VERSION

        payload = decoded["payload"]
        return [nil, "state marker payload is not an object"] unless payload.is_a?(Hash)

        expected = Digest::SHA256.hexdigest(canonical_json(payload))
        return [nil, "state marker integrity check failed"] unless secure_digest_equal?(decoded["payload_sha256"], expected)

        [payload, nil]
      rescue Errno::ENOENT
        [nil, "state marker is missing or not a regular file"]
      rescue JSON::ParserError, JSON::GeneratorError, EncodingError, SystemCallError,
             TranscriptionRuntimeError, TypeError, ArgumentError => e
        [nil, "state marker is unreadable (#{e.class}: #{e.message})"]
      end

      def write_state_atomic(path, payload, source_snapshot:, directory_binding: nil,
                             guard_bindings: nil, operation_hook: nil, rename: nil,
                             commit_guard: nil)
        path = Pathname(path).expand_path
        directory_binding ||= ensure_bound_directory(path.dirname)
        with_bound_parent(
          path,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        ) do |bound, basename|
          operation_hook&.call(:directory_opened, path)
          temporary_name = nil
          temporary = nil
          backup_name = nil
          backup = nil
          source = nil
          committed = false
          preserved_backup = false
          primary_failed = false
          begin
            mode = bound_state_mode(bound, basename)
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              temporary_name, temporary = bound.create_temporary(basename, ".tmp")
            end
            temporary.chmod(mode)
            operation_hook&.call(:before_stage_write, path)
            temporary.write(JSON.pretty_generate(envelope(payload)))
            temporary.write("\n")
            temporary.flush
            temporary.fsync
            temporary.close
            temporary = nil
            bound.verify!

            operation_hook&.call(:before_backup_open, path)
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              source = begin
                bound.open_regular(basename)
              rescue Errno::ENOENT
                nil
              end
            end
            if source
              Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
                backup_name, backup = bound.create_temporary(basename, ".bak")
              end
              begin
                backup.chmod(source.stat.mode & 0o7777)
                operation_hook&.call(:before_backup_copy, path)
                IO.copy_stream(source, backup)
                backup.flush
                backup.fsync
              ensure
                source.close
                backup.close
              end
            end

            ensure_source_unchanged!(source_snapshot)
            bound.verify!
            operation_hook&.call(:before_rename, path)
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              commit_guard&.call
              if rename
                committed = true
                rename.call(bound, temporary_name, basename, :commit)
              else
                bound.rename(temporary_name, basename)
                committed = true
              end
              operation_hook&.call(:after_rename, path)
              bound.verify!
              bound.fsync
              commit_guard&.call
            end
            path
          rescue Exception => e # rubocop:disable Lint/RescueException -- rollback includes interrupts
            primary_failed = true
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              if committed
                begin
                  bound.unlink(basename, missing_ok: true)
                  if backup_name
                    if rename
                      rename.call(bound, backup_name, basename, :rollback)
                    else
                      bound.rename(backup_name, basename)
                    end
                  end
                  backup_name = nil
                  bound.fsync
                rescue SystemCallError, TranscriptionRuntimeError => rollback_error
                  retained = backup_name ? bound.display_path(backup_name) : nil
                  preserved_backup = !backup_name.nil?
                  raise TranscriptionRuntimeError,
                        "State commit failed and rollback was incomplete (#{rollback_error.message}); " \
                        "preserved backup: #{retained}",
                        cause: e
                end
              end
            end
            raise e
          ensure
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              cleanup_failure = nil
              cleanup_operations = [
                -> { temporary&.close unless temporary&.closed? },
                -> { backup&.close unless backup&.closed? },
                -> { bound.unlink(temporary_name, missing_ok: true) if temporary_name },
                lambda do
                  bound.unlink(backup_name, missing_ok: true) if backup_name && !preserved_backup
                end
              ]
              cleanup_operations.each do |operation|
                operation.call
              rescue SystemCallError, IOError => e
                cleanup_failure ||= e
              end
              raise cleanup_failure if cleanup_failure && !primary_failed &&
                                       Thread.current.status != "aborting"
            end
          end
        end
      end

      def bound_state_mode(bound, basename)
        existing = nil
        Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
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
      private_class_method :bound_state_mode

      def ensure_source_unchanged!(snapshot)
        return if SourceSnapshot.capture(snapshot.canonical_path) == snapshot

        raise TranscriptionRuntimeError, "Source changed while processing: #{snapshot.canonical_path}"
      rescue SystemCallError, ArgumentError => e
        raise TranscriptionRuntimeError,
              "Cannot re-check source #{snapshot.canonical_path}: #{e.message}"
      end

      def securely_read_regular_file(path, directory_binding: nil, guard_bindings: nil)
        if directory_binding
          return with_bound_parent(
            path,
            directory_binding: directory_binding,
            guard_bindings: guard_bindings
          ) do |bound, basename|
            bound.verify!
            handle = nil
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              handle = bound.open_regular(basename)
            end
            opened = handle.stat
            raise TranscriptionRuntimeError, "State marker is too large: #{path}" if opened.size > MAX_STATE_BYTES

            contents = handle.read(MAX_STATE_BYTES + 1)
            raise TranscriptionRuntimeError, "State marker is too large: #{path}" if contents.bytesize > MAX_STATE_BYTES
            unless bound.same_regular_entry?(basename, opened)
              raise TranscriptionRuntimeError, "State marker changed while it was being read: #{path}"
            end

            bound.verify!
            contents
          ensure
            handle&.close
          end
        end

        before = path.lstat
        raise TranscriptionRuntimeError, "State marker is not a regular file: #{path}" unless before.file? && !before.symlink?
        raise TranscriptionRuntimeError, "State marker is too large: #{path}" if before.size > MAX_STATE_BYTES

        flags = File::RDONLY
        flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
        flags |= File::CLOEXEC if defined?(File::CLOEXEC)
        descriptor = ::IO.sysopen(path.to_s, flags)
        handle = ::IO.new(descriptor, "rb", autoclose: true)
        descriptor = nil
        opened = handle.stat
        current = path.lstat
        unless opened.file? && same_file?(opened, before) && same_file?(opened, current)
          raise TranscriptionRuntimeError, "State marker changed while it was being opened: #{path}"
        end

        handle.read(MAX_STATE_BYTES + 1).tap do |contents|
          raise TranscriptionRuntimeError, "State marker is too large: #{path}" if contents.bytesize > MAX_STATE_BYTES
        end
      ensure
        handle&.close
        ::IO.new(descriptor).close if descriptor
      end

      def same_file?(first, second)
        first.dev == second.dev && first.ino == second.ino && first.file? && second.file?
      end

      def secure_digest_equal?(actual, expected)
        return false unless actual.is_a?(String) && actual.bytesize == expected.bytesize

        difference = 0
        actual.bytes.zip(expected.bytes) { |left, right| difference |= left ^ right }
        difference.zero?
      end

      def deep_sort(value)
        case value
        when Hash
          value.keys.map(&:to_s).sort.to_h do |key|
            original_key = value.key?(key) ? key : value.keys.find { |candidate| candidate.to_s == key }
            [key, deep_sort(value.fetch(original_key))]
          end
        when Array
          value.map { |item| deep_sort(item) }
        when Pathname
          value.to_s
        else
          value
        end
      end
    end
  end
end
