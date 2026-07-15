# frozen_string_literal: true

require "digest"
require "fileutils"
require "json"
require "tempfile"

module Cohere
  module Transcribe
    module Runtime
      # Resolves native Transformers Dense checkpoints, downloads only their
      # required artifacts, converts them once to CrispASR GGUF, and retains no
      # Python-side state.
      class ModelProvider
        # Increment whenever conversion semantics or the native tensor types
        # written to GGUF change. Version 2 adds first-class BF16 artifacts.
        CACHE_LAYOUT_VERSION = 2

        attr_reader :cache_dir, :hub

        def initialize(hub: Hub.new, cache_dir: nil, native_session_class: nil,
                       converter: nil)
          @hub = hub
          root = cache_dir || ENV["COHERE_TRANSCRIBE_CACHE"] ||
                 File.join(ENV.fetch("XDG_CACHE_HOME", File.expand_path("~/.cache")), "cohere-transcribe")
          @cache_dir = Pathname(root).expand_path
          @native_session_class = native_session_class
          @converter = converter
        end

        # Planning may discover that every requested output can be resumed from
        # a checkpoint. Defer multi-gigabyte weight discovery until #open unless
        # a caller explicitly asks to validate inference artifacts now.
        def resolve(options, verify_model_weights: false)
          identity = ModelIdentity.resolve(
            path_text(options.model),
            options.model_revision,
            path_text(options.adapter),
            options.adapter_revision,
            hub: hub,
            verify_weight_artifacts: verify_model_weights
          )
          unless identity.model_format == :dense
            raise TranscriptionConfigurationError,
                  "Saved #{identity.model_format} checkpoints are outside the core Dense Ruby inference path"
          end
          if identity.adapter_id
            raise TranscriptionConfigurationError,
                  "PEFT/LoRA adapters are not supported by the native Ruby Dense runtime"
          end

          identity
        end

        def open(identity, options)
          ModelIdentity.verify_model_weight_artifacts(identity, hub: hub)
          path = converted_model_path(identity, options)
          (@native_session_class || ASR::NativeSession).new(path, options)
        rescue ArgumentError, TypeError, Hub::Error => e
          raise TranscriptionRuntimeError, e.message
        end

        def converted_model_path(identity, options)
          model_directory = materialize_source(identity)
          output_type = { "fp32" => :f32, "bf16" => :bf16 }.fetch(options.dtype, :f16)
          fingerprint = source_fingerprint(identity, model_directory, output_type)
          directory = conversion_cache_directory
          output = directory.join("#{fingerprint}-#{output_type}.gguf")
          return output if valid_gguf?(output, fingerprint: fingerprint, output_type: output_type)

          lock_path = Pathname("#{output}.lock")
          open_conversion_lock(lock_path) do |lock|
            raise TranscriptionRuntimeError, "Cannot acquire Dense conversion lock for #{output}" unless lock.flock(File::LOCK_EX)
            return output if valid_gguf?(output, fingerprint: fingerprint, output_type: output_type)

            cleanup_conversion_temporaries(output)
            remove_cache_entry(output)
            remove_cache_entry(conversion_marker(output))
            begin
              converter.convert(
                model_dir: model_directory,
                output_path: output,
                output_type: output_type,
                overwrite: false,
                fsync: true
              )
              unless source_fingerprint(identity, model_directory, output_type) == fingerprint
                raise TranscriptionRuntimeError,
                      "Dense checkpoint changed during Dense conversion; retry with a stable model directory"
              end
              write_conversion_marker(output, fingerprint: fingerprint, output_type: output_type)
            rescue Exception # rubocop:disable Lint/RescueException -- clean partial artifacts before propagating interrupts
              remove_cache_entry(output)
              remove_cache_entry(conversion_marker(output))
              raise
            end
          end
          unless valid_gguf?(output, fingerprint: fingerprint, output_type: output_type)
            raise TranscriptionRuntimeError,
                  "Dense conversion did not produce a valid GGUF file at #{output}"
          end

          output
        rescue DenseConverter::Error, Safetensors::Error, PyTorchCheckpoint::Error, GGUF::Error => e
          raise TranscriptionRuntimeError, "Cannot convert Dense Cohere checkpoint: #{e.message}"
        rescue Hub::Error => e
          raise TranscriptionRuntimeError, e.message
        rescue SystemCallError => e
          raise TranscriptionRuntimeError, "Cannot prepare native Dense model: #{e.message}"
        end

        def materialize_source(identity)
          local = Pathname(identity.model_id).expand_path
          return local.realpath if local.directory?

          revision = identity.model_revision
          raise TranscriptionRuntimeError, "Remote model identity has no immutable revision" unless revision

          DenseConverter::REQUIRED_ARTIFACT_FILENAMES.each do |filename|
            hub.download(identity.model_id, filename, revision: revision)
          end
          snapshot = hub.snapshot_path(identity.model_id, revision)
          return snapshot if hub.cached_file(identity.model_id, "model.safetensors", revision: revision)
          return snapshot if hub.cached_file(identity.model_id, "pytorch_model.bin", revision: revision)

          safetensors_index = hub.cached_file(
            identity.model_id, "model.safetensors.index.json", revision: revision
          )
          if safetensors_index
            download_index_shards(identity, safetensors_index, extension: ".safetensors", label: "Safetensors")
            return snapshot
          end
          pytorch_index = hub.cached_file(
            identity.model_id, "pytorch_model.bin.index.json", revision: revision
          )
          if pytorch_index
            download_index_shards(identity, pytorch_index, extension: ".bin", label: "PyTorch")
            return snapshot
          end

          files = hub.list_files(identity.model_id, revision: revision)
          if files.include?("model.safetensors")
            hub.download(identity.model_id, "model.safetensors", revision: revision)
          elsif files.include?("model.safetensors.index.json")
            index = hub.download(identity.model_id, "model.safetensors.index.json", revision: revision)
            download_index_shards(identity, index, extension: ".safetensors", label: "Safetensors")
          elsif files.include?("pytorch_model.bin")
            hub.download(identity.model_id, "pytorch_model.bin", revision: revision)
          elsif files.include?("pytorch_model.bin.index.json")
            index = hub.download(identity.model_id, "pytorch_model.bin.index.json", revision: revision)
            download_index_shards(identity, index, extension: ".bin", label: "PyTorch")
          else
            raise TranscriptionRuntimeError,
                  "#{identity.model_id}@#{revision} has no supported Dense Safetensors or PyTorch weights"
          end
          snapshot
        end

        private

        def converter
          @converter ||= DenseConverter
        end

        def conversion_cache_directory
          FileUtils.mkdir_p(cache_dir)
          cache_root = cache_dir.realpath
          directory = cache_dir.join("dense-v#{CACHE_LAYOUT_VERSION}")
          begin
            Dir.mkdir(directory, 0o700)
          rescue Errno::EEXIST
            nil
          end
          stat = directory.lstat
          unless stat.directory? && !stat.symlink? && directory.parent.realpath == cache_root
            raise TranscriptionRuntimeError,
                  "Dense conversion cache directory is not a regular directory: #{directory}"
          end

          directory
        rescue TranscriptionRuntimeError
          raise
        rescue SystemCallError => e
          raise TranscriptionRuntimeError,
                "Cannot prepare Dense conversion cache directory: #{e.message}"
        end

        def open_conversion_lock(path)
          flags = File::RDWR | File::CREAT
          flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
          flags |= File::CLOEXEC if defined?(File::CLOEXEC)
          descriptor = ::IO.sysopen(path.to_s, flags, 0o600)
          lock = File.new(descriptor, "r+", autoclose: true)
          descriptor = nil
          opened = lock.stat
          current = path.lstat
          unless opened.file? && !current.symlink? && opened.dev == current.dev && opened.ino == current.ino
            raise TranscriptionRuntimeError,
                  "Dense conversion lock changed while it was being opened or is not regular: #{path}"
          end

          yield lock
        rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
          raise TranscriptionRuntimeError, "Dense conversion lock is not a regular file: #{path}", cause: e
        ensure
          lock&.close
          ::IO.new(descriptor).close if descriptor
        end

        def path_text(value)
          value.respond_to?(:to_path) ? value.to_path : value
        end

        def download_index_shards(identity, index_path, extension:, label:)
          payload = JSON.parse(Pathname(index_path).read(encoding: "UTF-8"))
          weights = payload.is_a?(Hash) ? payload["weight_map"] : nil
          unless weights.is_a?(Hash) && !weights.empty? &&
                 weights.all? { |name, filename| name.is_a?(String) && !name.empty? && filename.is_a?(String) }
            raise TranscriptionRuntimeError, "#{label} index #{index_path} has an invalid weight_map"
          end

          filenames = weights.values.uniq
          filenames.each do |filename|
            if filename.empty? || filename.include?("\0")
              raise TranscriptionRuntimeError, "#{label} index contains an invalid shard name: #{filename.inspect}"
            end

            path = Pathname(filename)
            unless !path.absolute? &&
                   path.each_filename.none? { |part| part == ".." } && filename.end_with?(extension)
              raise TranscriptionRuntimeError, "#{label} index contains an invalid shard name: #{filename.inspect}"
            end

            hub.download(identity.model_id, filename, revision: identity.model_revision)
          end
        rescue JSON::ParserError, EncodingError => e
          raise TranscriptionRuntimeError, "Cannot parse #{label} index #{index_path}: #{e.message}"
        end

        def source_fingerprint(identity, directory, output_type)
          digest = Digest::SHA256.new
          digest << "cohere-transcribe-dense\0#{CACHE_LAYOUT_VERSION}\0#{output_type}\0"
          digest << identity.model_id.to_s << "\0" << identity.model_revision.to_s << "\0"
          source_files(directory).each do |path|
            stat = path.stat
            digest << path.relative_path_from(directory).to_s << "\0"
            digest << stat.size.to_s << "\0" << stat.mtime.to_r.to_s << "\0"
            # A same-size rewrite can deliberately preserve mtime. ctime is
            # kernel-managed and still invalidates that stale conversion.
            digest << stat.ctime.to_r.to_s << "\0"
          end
          digest.hexdigest
        end

        def source_files(directory)
          files = DenseConverter::REQUIRED_ARTIFACT_FILENAMES.map { |name| directory.join(name) }
          single = directory.join("model.safetensors")
          if single.file?
            files << single
          elsif directory.join("model.safetensors.index.json").file?
            index = directory.join("model.safetensors.index.json")
            files << index
            payload = JSON.parse(index.read(encoding: "UTF-8"))
            weights = payload.is_a?(Hash) ? payload["weight_map"] : nil
            unless weights.is_a?(Hash) && !weights.empty? &&
                   weights.all? { |name, file| name.is_a?(String) && !name.empty? && file.is_a?(String) }
              raise TranscriptionRuntimeError,
                    "Cannot fingerprint Dense checkpoint: " \
                    "Safetensors index #{index} has an invalid weight_map"
            end

            shard_names = weights.values.uniq
            shard_names.each do |name|
              next if valid_safetensors_shard_name?(name)

              raise TranscriptionRuntimeError,
                    "Cannot fingerprint Dense checkpoint: " \
                    "Safetensors index #{index} contains an invalid shard path #{name.inspect}"
            end
            shard_names.sort.each { |name| files << directory.join(name) }
          elsif directory.join("pytorch_model.bin").file?
            files << directory.join("pytorch_model.bin")
          else
            index = directory.join("pytorch_model.bin.index.json")
            files << index
            tensor_set = PyTorchCheckpoint::TensorSet.from_directory(directory)
            files.concat(tensor_set.readers.map(&:path))
          end
          missing = files.reject(&:file?)
          raise TranscriptionRuntimeError, "Dense checkpoint is missing: #{missing.join(", ")}" unless missing.empty?

          files.sort_by(&:to_s)
        rescue JSON::ParserError, KeyError => e
          raise TranscriptionRuntimeError, "Cannot fingerprint Dense checkpoint: #{e.message}"
        end

        def valid_safetensors_shard_name?(name)
          return false if name.empty? || name.include?("\0")

          candidate = Pathname(name)
          !candidate.absolute? && candidate.each_filename.none? { |part| part == ".." }
        end

        def conversion_marker(path)
          Pathname("#{path}.complete.json")
        end

        def remove_cache_entry(path)
          path.delete
        rescue Errno::ENOENT
          nil
        end

        def cleanup_conversion_temporaries(output)
          [output, conversion_marker(output)].each do |target|
            pattern = target.dirname.join(".#{target.basename}*.tmp")
            Dir.glob(pattern.to_s).each do |path|
              File.unlink(path)
            rescue Errno::ENOENT
              nil
            end
          end
        end

        def valid_gguf?(path, fingerprint:, output_type:)
          stat = path.lstat
          return false unless stat.file? && !stat.symlink? && stat.size > 32
          return false unless path.open("rb") { |file| file.read(4) == "GGUF" }

          marker = conversion_marker(path)
          marker_stat = marker.lstat
          return false unless marker_stat.file? && !marker_stat.symlink? && marker_stat.size.between?(1, 16_384)

          payload = JSON.parse(marker.read(encoding: "UTF-8"))
          payload == conversion_marker_payload(
            stat,
            fingerprint: fingerprint,
            output_type: output_type
          )
        rescue JSON::ParserError, EncodingError, SystemCallError
          false
        end

        def write_conversion_marker(path, fingerprint:, output_type:)
          stat = path.lstat
          unless stat.file? && !stat.symlink? && stat.size > 32 &&
                 path.open("rb") { |file| file.read(4) == "GGUF" }
            raise TranscriptionRuntimeError, "Dense conversion produced an invalid GGUF file at #{path}"
          end

          marker = conversion_marker(path)
          payload = conversion_marker_payload(
            stat,
            fingerprint: fingerprint,
            output_type: output_type
          )
          Tempfile.create([".#{marker.basename}", ".tmp"], marker.dirname, binmode: true) do |temporary|
            temporary.chmod(0o600)
            temporary.write(JSON.generate(payload))
            temporary.write("\n")
            temporary.flush
            temporary.fsync
            temporary.close
            File.rename(temporary.path, marker)
          end
          begin
            File.open(marker.dirname, File::RDONLY, &:fsync)
          rescue Errno::EINVAL, Errno::ENOTSUP, Errno::EISDIR
            nil
          end
          marker
        end

        def conversion_marker_payload(stat, fingerprint:, output_type:)
          {
            "schema_version" => 1,
            "cache_layout_version" => CACHE_LAYOUT_VERSION,
            "source_fingerprint" => fingerprint,
            "output_type" => output_type.to_s,
            "device" => stat.dev,
            "inode" => stat.ino,
            "size" => stat.size,
            "mtime_ns" => (stat.mtime.to_i * 1_000_000_000) + stat.mtime.nsec,
            "ctime_ns" => (stat.ctime.to_i * 1_000_000_000) + stat.ctime.nsec
          }
        end
      end
    end
  end
end
