# frozen_string_literal: true

require "tmpdir"
require "json"
require "timeout"
require "test_helper"

class Cohere::Transcribe::ModelProviderTest < Minitest::Test
  def test_planning_defers_dense_weight_validation_until_open
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = Pathname(directory).join("source").tap(&:mkpath)
      source.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: Pathname(directory).join("cache")
      )
      options = Cohere::Transcribe::TranscriptionOptions.new(model: source.to_s)

      identity = provider.resolve(options)

      assert_equal :dense, identity.model_format
      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.open(identity, options)
      end
      assert_match(/Transformers model weights/, error.message)
    end
  end

  def test_planning_can_explicitly_validate_model_weights
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = Pathname(directory).join("source").tap(&:mkpath)
      source.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: Pathname(directory).join("cache")
      )
      options = Cohere::Transcribe::TranscriptionOptions.new(model: source.to_s)

      error = assert_raises(ArgumentError) do
        provider.resolve(options, verify_model_weights: true)
      end

      assert_match(/Transformers model weights/, error.message)
    end
  end

  def test_planning_rejects_saved_quantization_before_weight_discovery
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = Pathname(directory).join("source").tap(&:mkpath)
      source.join("config.json").write(
        JSON.generate(
          "model_type" => "cohere_asr",
          "quantization_config" => {
            "quant_method" => "bitsandbytes",
            "load_in_4bit" => true
          }
        )
      )
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: Pathname(directory).join("cache")
      )
      options = Cohere::Transcribe::TranscriptionOptions.new(model: source.to_s)

      error = assert_raises(Cohere::Transcribe::TranscriptionConfigurationError) do
        provider.resolve(options)
      end

      assert_match(/Saved bitsandbytes-int4 checkpoints are outside the core Dense Ruby inference path/, error.message)
    end
  end

  def test_planning_does_not_treat_false_quantization_config_as_dense
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = Pathname(directory).join("source").tap(&:mkpath)
      source.join("config.json").write(
        JSON.generate("model_type" => "cohere_asr", "quantization_config" => false)
      )
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: Pathname(directory).join("cache")
      )
      options = Cohere::Transcribe::TranscriptionOptions.new(model: source.to_s)

      error = assert_raises(ArgumentError) { provider.resolve(options) }

      assert_equal "#{source.realpath} has an invalid quantization_config", error.message
    end
  end

  def test_local_dense_conversion_is_locked_fingerprinted_and_reused
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = File.join(directory, "cache")
      conversions = []
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          conversions << arguments
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      first = provider.converted_model_path(identity, options)
      second = provider.converted_model_path(identity, options)

      assert_equal first, second
      assert first.file?
      assert_equal "GGUF", first.binread(4)
      assert Pathname("#{first}.complete.json").file?
      assert_equal 1, conversions.length
      assert_equal Pathname(source), conversions.first.fetch(:model_dir)
      assert_equal :f16, conversions.first.fetch(:output_type)

      File.open(File.join(source, "model.safetensors"), "ab") { |file| file.write("changed") }
      changed = provider.converted_model_path(identity, options)
      refute_equal first, changed
      assert_equal 2, conversions.length
    end
  end

  def test_fp32_option_selects_an_independent_conversion
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      seen = []
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          seen << arguments.fetch(:output_type)
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache"),
        converter: converter
      )

      provider.converted_model_path(
        dense_identity(source),
        Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp32")
      )

      assert_equal [:f32], seen
    end
  end

  def test_bf16_option_selects_an_independent_conversion
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      seen = []
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          seen << arguments.fetch(:output_type)
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache"), converter: converter
      )

      provider.converted_model_path(
        dense_identity(source),
        Cohere::Transcribe::TranscriptionOptions.new(dtype: "bf16")
      )

      assert_equal [:bf16], seen
    end
  end

  def test_pytorch_bin_source_participates_in_the_conversion_fingerprint
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      File.delete(File.join(source, "model.safetensors"))
      bin = File.join(source, "pytorch_model.bin")
      File.binwrite(bin, "first")
      conversions = 0
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          conversions += 1
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache"), converter: converter
      )
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      first = provider.converted_model_path(dense_identity(source), options)
      File.binwrite(bin, "second-and-different")
      second = provider.converted_model_path(dense_identity(source), options)

      refute_equal first, second
      assert_equal 2, conversions
    end
  end

  def test_invalid_safetensors_index_is_normalized_before_cache_keying
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      source.join("model.safetensors").delete
      index = source.join("model.safetensors.index.json")
      index.write(JSON.generate("weight_map" => []))
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache")
      )

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.converted_model_path(
          dense_identity(source),
          Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
        )
      end

      assert_match(/Safetensors index .* has an invalid weight_map/, error.message)
      refute Pathname(directory).join("cache").exist?
    end
  end

  def test_invalid_safetensors_shard_paths_are_rejected_before_cache_keying
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      source.join("model.safetensors").delete
      outside = Pathname(directory).join("outside.safetensors").tap { |path| path.write("outside") }
      index = source.join("model.safetensors.index.json")
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache")
      )

      ["../#{outside.basename}", outside.to_s, "shard\0.safetensors"].each do |shard|
        index.write(JSON.generate("weight_map" => { "tensor" => shard }))

        error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError, shard.inspect) do
          provider.converted_model_path(
            dense_identity(source),
            Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
          )
        end

        assert_match(/contains an invalid shard path/, error.message)
      end
      assert_equal "outside", outside.read
      refute Pathname(directory).join("cache").exist?

      nonstandard_suffix = source.join("reader-compatible-shard.bin").tap { |path| path.write("fixture") }
      index.write(JSON.generate("weight_map" => { "tensor" => nonstandard_suffix.basename.to_s }))
      assert_includes provider.send(:source_files, source), nonstandard_suffix
    end
  end

  def test_same_size_mtime_preserving_source_rewrite_invalidates_conversion
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = File.join(directory, "cache")
      conversions = 0
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          conversions += 1
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      first = provider.converted_model_path(identity, options)
      weights = File.join(source, "model.safetensors")
      original_mtime = File.mtime(weights)
      original = File.binread(weights)
      replacement = original.bytes.map { |byte| byte ^ 0x01 }.pack("C*")
      File.binwrite(weights, replacement)
      File.utime(original_mtime, original_mtime, weights)
      second = provider.converted_model_path(identity, options)

      refute_equal first, second
      assert_equal 2, conversions
    end
  end

  def test_modified_cached_gguf_is_reconverted_even_when_magic_and_size_survive
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      conversions = 0
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          conversions += 1
          File.binwrite(arguments.fetch(:output_path), "GGUF" + (conversions.chr * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: File.join(directory, "cache"),
        converter: converter
      )
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      path = provider.converted_model_path(identity, options)
      original_mtime = path.mtime
      File.binwrite(path, "GGUF" + ("x" * 64))
      File.utime(original_mtime, original_mtime, path)
      rebuilt = provider.converted_model_path(identity, options)

      assert_equal path, rebuilt
      assert_equal 2, conversions
      assert_equal "\x02", rebuilt.binread(5).byteslice(4)
    end
  end

  def test_conversion_lock_symlink_is_rejected_without_touching_its_target
    skip "File::NOFOLLOW is unavailable on this platform" unless defined?(File::NOFOLLOW)

    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = File.join(directory, "cache")
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(cache_dir: cache)
      identity = dense_identity(source)
      model_directory = provider.materialize_source(identity)
      fingerprint = provider.send(:source_fingerprint, identity, model_directory, :f16)
      output = Pathname(cache).join(
        "dense-v#{Cohere::Transcribe::Runtime::ModelProvider::CACHE_LAYOUT_VERSION}",
        "#{fingerprint}-f16.gguf"
      )
      FileUtils.mkdir_p(output.dirname)
      victim = Pathname(directory).join("victim")
      victim.write("leave me alone")
      File.symlink(victim, "#{output}.lock")

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.converted_model_path(
          identity,
          Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
        )
      end

      assert_match(/lock is not a regular file/, error.message)
      assert_equal "leave me alone", victim.read
    end
  end

  def test_source_mutation_during_conversion_is_not_marked_complete
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      weights = source.join("model.safetensors")
      conversions = 0
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          conversions += 1
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
          File.open(weights, "ab") { |file| file.write("changed-during-conversion") } if conversions == 1
        end
      end
      cache = Pathname(directory).join("cache")
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.converted_model_path(identity, options)
      end

      assert_match(/changed during Dense conversion/, error.message)
      assert_empty Dir.glob(File.join(cache, "dense-v*", "*.gguf"))
      assert_empty Dir.glob(File.join(cache, "dense-v*", "*.complete.json"))

      result = provider.converted_model_path(identity, options)
      assert result.file?
      assert_equal 2, conversions
    end
  end

  def test_interrupted_converter_output_is_removed_before_the_lock_is_released
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      attempts = 0
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          attempts += 1
          if attempts == 1
            File.binwrite(arguments.fetch(:output_path), "partial")
            raise Cohere::Transcribe::DenseConverter::Error, "interrupted fixture"
          end
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      cache = Pathname(directory).join("cache")
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.converted_model_path(identity, options)
      end

      assert_match(/interrupted fixture/, error.message)
      assert_empty Dir.glob(File.join(cache, "dense-v*", "*.gguf"))
      assert_empty Dir.glob(File.join(cache, "dense-v*", "*.complete.json"))
      assert provider.converted_model_path(identity, options).file?
      assert_equal 2, attempts
    end
  end

  def test_stale_conversion_lock_and_temporaries_are_recovered
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = Pathname(directory).join("cache")
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      model_directory = provider.materialize_source(identity)
      fingerprint = provider.send(:source_fingerprint, identity, model_directory, :f16)
      output = provider.send(:conversion_cache_directory).join("#{fingerprint}-f16.gguf")
      lock = Pathname("#{output}.lock").tap { |path| path.write("stale metadata") }
      output_temp = output.dirname.join(".#{output.basename}crashed.tmp").tap do |path|
        path.write("partial")
      end
      marker = Pathname("#{output}.complete.json")
      marker_temp = marker.dirname.join(".#{marker.basename}crashed.tmp").tap do |path|
        path.write("partial marker")
      end

      result = provider.converted_model_path(
        identity,
        Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
      )

      assert_equal output, result
      assert lock.file?
      refute output_temp.exist?
      refute marker_temp.exist?
      assert marker.file?
    end
  end

  def test_dangling_output_symlink_is_unlinked_before_conversion
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = Pathname(directory).join("cache")
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          output = Pathname(arguments.fetch(:output_path))
          raise "converter received a symlink output" if output.symlink?

          output.binwrite("GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )
      identity = dense_identity(source)
      model_directory = provider.materialize_source(identity)
      fingerprint = provider.send(:source_fingerprint, identity, model_directory, :f16)
      output = cache.join(
        "dense-v#{Cohere::Transcribe::Runtime::ModelProvider::CACHE_LAYOUT_VERSION}",
        "#{fingerprint}-f16.gguf"
      )
      output.dirname.mkpath
      victim = Pathname(directory).join("outside-victim")
      File.symlink(victim, output)

      result = provider.converted_model_path(
        identity,
        Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
      )

      assert_equal output, result
      refute result.symlink?
      assert_equal "GGUF", result.binread(4)
      refute victim.exist?
    end
  end

  def test_conversion_cache_subdirectory_symlink_cannot_redirect_outputs
    Dir.mktmpdir("cohere-model-provider") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = Pathname(directory).join("cache").tap(&:mkpath)
      outside = Pathname(directory).join("outside").tap(&:mkpath)
      layout = "dense-v#{Cohere::Transcribe::Runtime::ModelProvider::CACHE_LAYOUT_VERSION}"
      File.symlink(outside, cache.join(layout))
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        cache_dir: cache,
        converter: converter
      )

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        provider.converted_model_path(
          dense_identity(source),
          Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
        )
      end

      assert_match(/conversion cache directory is not a regular directory/, error.message)
      assert_empty outside.children
    end
  end

  def test_conversion_lock_serializes_contending_processes
    skip "fork is unavailable on this platform" unless Process.respond_to?(:fork)

    Dir.mktmpdir("cohere-model-provider-processes") do |directory|
      source = build_source(File.join(directory, "source"))
      cache = Pathname(directory).join("cache")
      log = Pathname(directory).join("conversions.log")
      start_reader, start_writer = IO.pipe
      entered_reader, entered_writer = IO.pipe
      release_reader, release_writer = IO.pipe
      converter = Class.new do
        define_singleton_method(:convert) do |**arguments|
          File.open(log, "a") { |file| file.puts(Process.pid) }
          entered_writer.write("x")
          raise "conversion release pipe closed" unless release_reader.read(1)

          File.binwrite(arguments.fetch(:output_path), "GGUF" + ("\0" * 64))
        end
      end
      identity = dense_identity(source)
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: "fp16")
      children = Array.new(8) do
        fork do
          start_writer.close
          entered_reader.close
          release_writer.close
          start_reader.read(1)
          provider = Cohere::Transcribe::Runtime::ModelProvider.new(
            cache_dir: cache,
            converter: converter
          )
          provider.converted_model_path(identity, options)
          exit! 0
        rescue Exception => e # rubocop:disable Lint/RescueException -- report every child failure to the parent
          warn e.full_message
          exit! 1
        end
      end
      start_reader.close
      entered_writer.close
      release_reader.close
      start_writer.write("x" * children.length)
      start_writer.close
      assert_equal "x", Timeout.timeout(5) { entered_reader.read(1) }
      release_writer.write("x" * children.length)
      release_writer.close

      statuses = children.map { |pid| Process.wait2(pid).last }

      assert statuses.all?(&:success?), statuses.map(&:inspect).join(", ")
      assert_equal 1, log.readlines.length
    ensure
      [start_reader, start_writer, entered_reader, entered_writer,
       release_reader, release_writer].compact.each do |io|
        io.close unless io.closed?
      end
    end
  end

  def test_remote_weight_symlink_outside_the_repository_is_refetched
    Dir.mktmpdir("cohere-model-provider") do |directory|
      commit = "e" * 40
      repository = Pathname(directory).join("hub", "models--owner--model")
      snapshot = repository.join("snapshots", commit).tap(&:mkpath)
      snapshot.join("config.json").write("{}")
      snapshot.join("tokenizer.json").write("{}")
      victim = Pathname(directory).join("victim.safetensors").tap do |path|
        path.write("private")
      end
      weights = snapshot.join("model.safetensors")
      File.symlink(victim, weights)
      requests = []
      hub = Cohere::Transcribe::Hub.new(
        cache_dir: Pathname(directory).join("hub"), endpoint: "https://example.invalid"
      )
      hub.define_singleton_method(:list_files) do |_repo_id, revision:|
        raise "wrong revision" unless revision == commit

        ["model.safetensors"]
      end
      hub.define_singleton_method(:request) do |uri, stream: nil, **_options|
        requests << uri
        stream.write("downloaded weights")
      end
      provider = Cohere::Transcribe::Runtime::ModelProvider.new(
        hub: hub,
        cache_dir: Pathname(directory).join("converted")
      )
      identity = Cohere::Transcribe::ResolvedModelIdentity.new(
        model_id: "owner/model",
        model_revision: commit,
        model_format: :dense,
        quantization_config: nil,
        adapter_id: nil,
        adapter_revision: nil
      )

      result = provider.materialize_source(identity)

      assert_equal snapshot, result
      assert_equal 1, requests.length
      refute weights.symlink?
      assert_equal "downloaded weights", weights.read
      assert_equal "private", victim.read
    end
  end

  private

  def build_source(path)
    FileUtils.mkdir_p(path)
    File.write(File.join(path, "config.json"), "{}")
    File.write(File.join(path, "tokenizer.json"), "{}")
    File.binwrite(File.join(path, "model.safetensors"), "stub")
    Pathname(path).realpath
  end

  def dense_identity(path)
    Cohere::Transcribe::ResolvedModelIdentity.new(
      model_id: path.to_s,
      model_revision: nil,
      model_format: :dense,
      quantization_config: nil,
      adapter_id: nil,
      adapter_revision: nil
    )
  end
end
