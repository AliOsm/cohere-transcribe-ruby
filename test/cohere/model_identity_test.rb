# frozen_string_literal: true

require "json"
require "etc"
require "test_helper"

class Cohere::Transcribe::ModelIdentityTest < Minitest::Test
  ModelIdentity = Cohere::Transcribe::ModelIdentity

  class FakeHub
    attr_reader :resolution_calls

    def initialize(directory, adapter_config:, base_commit: "1" * 40,
                   adapter_commit: "2" * 40, declared_commits: {})
      @directory = Pathname(directory)
      @base_commit = base_commit
      @adapter_commit = adapter_commit
      @declared_commits = declared_commits
      @resolution_calls = []
      @directory.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))
      @directory.join("adapter_config.json").write(JSON.generate(adapter_config))
    end

    def resolve_revision(repository, revision, filename:)
      resolution_calls << [repository, revision, filename]
      return @adapter_commit if filename == "adapter_config.json"

      @declared_commits.fetch(revision, @base_commit)
    end

    def download(_repository, filename, revision:)
      raise "download must use an immutable revision" unless revision

      @directory.join(filename)
    end
  end

  def test_remote_adapter_requires_the_exact_remote_base_repository
    Dir.mktmpdir do |directory|
      missing = FakeHub.new(directory, adapter_config: lora_config.except("base_model_name_or_path"))
      error = assert_raises(ArgumentError) { resolve_with(missing) }
      assert_match(%r{requires base model nil, not "owner/model"}, error.message)

      mismatched = FakeHub.new(directory, adapter_config: lora_config.merge("base_model_name_or_path" => "owner/other"))
      error = assert_raises(ArgumentError) { resolve_with(mismatched) }
      assert_match(%r{requires base model "owner/other"}, error.message)
    end
  end

  def test_adapter_declared_base_tag_is_resolved_before_commit_comparison
    Dir.mktmpdir do |directory|
      hub = FakeHub.new(
        directory,
        adapter_config: lora_config.merge("revision" => "training-base"),
        declared_commits: { "training-base" => "1" * 40 }
      )

      identity = resolve_with(hub)

      assert_equal "owner/adapter", identity.adapter_id
      assert_equal "2" * 40, identity.adapter_revision
      assert_includes hub.resolution_calls, ["owner/model", "training-base", "config.json"]
    end
  end

  def test_adapter_declared_base_revision_must_resolve_to_the_selected_commit
    Dir.mktmpdir do |directory|
      hub = FakeHub.new(
        directory,
        adapter_config: lora_config.merge("revision" => "training-base"),
        declared_commits: { "training-base" => "3" * 40 }
      )

      error = assert_raises(ArgumentError) { resolve_with(hub) }

      assert_match(/requires base revision #{"3" * 40}, not #{"1" * 40}/, error.message)
    end
  end

  def test_local_base_does_not_compare_an_adapters_historical_hub_identity
    Dir.mktmpdir do |directory|
      root = Pathname(directory)
      model = root.join("local-model").tap(&:mkpath)
      model.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))
      hub = FakeHub.new(
        root.join("hub").tap(&:mkpath),
        adapter_config: lora_config.merge(
          "base_model_name_or_path" => "training-machine/base",
          "revision" => "historical-tag"
        )
      )

      identity = ModelIdentity.resolve(
        model.to_s,
        nil,
        "owner/adapter",
        nil,
        hub: hub,
        verify_weight_artifacts: false
      )

      assert_equal model.realpath.to_s, identity.model_id
      refute_includes hub.resolution_calls, [model.to_s, "historical-tag", "config.json"]
    end
  end

  def test_packaged_default_identity_and_weights_need_no_hub_artifacts
    hub = Object.new
    hub.define_singleton_method(:resolve_revision) do |_repository, revision, filename:|
      raise "wrong metadata" unless filename == "config.json"

      expected = Cohere::Transcribe::DEFAULT_ASR_MODEL_REVISION
      raise "wrong revision" unless revision&.casecmp?(expected)

      revision.downcase
    end
    %i[download cached_file list_files].each do |operation|
      hub.define_singleton_method(operation) { |*_args, **_options| raise "unexpected #{operation}" }
    end

    revision = Cohere::Transcribe::DEFAULT_ASR_MODEL_REVISION
    [nil, revision, revision.upcase].each do |requested|
      identity = ModelIdentity.resolve(
        Cohere::Transcribe::DEFAULT_ASR_MODEL_ID,
        requested,
        hub: hub
      )

      assert_equal Cohere::Transcribe::DEFAULT_ASR_MODEL_ID, identity.model_id
      assert_equal Cohere::Transcribe::DEFAULT_ASR_MODEL_REVISION, identity.model_revision
      assert_equal :dense, identity.model_format
      assert_nil identity.quantization_config
    end
  end

  def test_another_default_repository_revision_still_inspects_its_config
    Dir.mktmpdir do |directory|
      revision = "3" * 40
      hub = FakeHub.new(
        directory,
        base_commit: revision,
        adapter_config: lora_config
      )
      Pathname(directory).join("config.json").write(
        JSON.generate(
          "model_type" => "cohere_asr",
          "quantization_config" => {
            "quant_method" => "bitsandbytes",
            "load_in_8bit" => true
          }
        )
      )

      identity = ModelIdentity.resolve(
        Cohere::Transcribe::DEFAULT_ASR_MODEL_ID,
        revision,
        hub: hub,
        verify_weight_artifacts: false
      )

      assert_equal revision, identity.model_revision
      assert_equal :"bitsandbytes-int8", identity.model_format
      assert_includes hub.resolution_calls,
                      [Cohere::Transcribe::DEFAULT_ASR_MODEL_ID, revision, "config.json"]
    end
  end

  def test_packaged_default_still_validates_adapter_metadata_and_weights
    Dir.mktmpdir do |directory|
      root = Pathname(directory)
      adapter_config = root.join("adapter_config.json").tap do |path|
        path.write(
          JSON.generate(
            lora_config.merge(
              "base_model_name_or_path" => Cohere::Transcribe::DEFAULT_ASR_MODEL_ID
            )
          )
        )
      end
      adapter_weight = root.join("adapter_model.safetensors").tap { |path| path.write("adapter") }
      accesses = []
      hub = Object.new
      hub.define_singleton_method(:resolve_revision) do |repository, revision, filename:|
        accesses << [:resolve, repository, revision, filename]
        filename == "config.json" ? revision : "4" * 40
      end
      hub.define_singleton_method(:download) do |repository, filename, revision:|
        accesses << [:download, repository, revision, filename]
        raise "packaged model config must not be downloaded" if filename == "config.json"

        adapter_config
      end
      hub.define_singleton_method(:cached_file) do |repository, filename, revision:|
        accesses << [:cached_file, repository, revision, filename]
        filename == "adapter_model.safetensors" ? adapter_weight : nil
      end
      hub.define_singleton_method(:list_files) do |*_args, **_options|
        raise "cached adapter weight must avoid file listing"
      end

      identity = ModelIdentity.resolve(
        Cohere::Transcribe::DEFAULT_ASR_MODEL_ID,
        nil,
        "owner/adapter",
        "adapter-release",
        hub: hub
      )

      assert_equal "owner/adapter", identity.adapter_id
      assert_equal "4" * 40, identity.adapter_revision
      assert_includes accesses,
                      [:download, "owner/adapter", "4" * 40, "adapter_config.json"]
      assert_includes accesses,
                      [:cached_file, "owner/adapter", "4" * 40, "adapter_model.safetensors"]
      refute(accesses.any? { |item| item[0] == :cached_file && item[1] == identity.model_id })
    end
  end

  def test_nonexistent_named_user_path_is_not_misclassified_as_a_hub_id
    user = Etc.getpwuid.name
    reference = "~#{user}/.cohere-transcribe-missing-#{Process.pid}-#{object_id}"
    refute Pathname(reference).expand_path.exist?

    error = assert_raises(ArgumentError) do
      ModelIdentity.resolve_local_directory(reference, description: "Model")
    end

    assert_match(/Model directory .* does not exist/, error.message)
  end

  def test_parent_traversal_is_resolved_only_when_every_component_exists
    Dir.mktmpdir do |directory|
      Dir.chdir(directory) do
        missing = "missing-#{Process.pid}-#{object_id}/.."
        assert_nil ModelIdentity.resolve_local_directory(missing, description: "Model")

        Pathname("existing").mkdir
        assert_equal Pathname(directory).realpath.to_s,
                     ModelIdentity.resolve_local_directory("existing/..", description: "Model")
      end
    end
  end

  def test_relative_dot_components_are_removed_before_home_expansion
    assert_equal Pathname(Dir.home).realpath.to_s,
                 ModelIdentity.resolve_local_directory("./~", description: "Model")
    assert_equal Pathname(Dir.home).realpath.to_s,
                 ModelIdentity.resolve_local_directory("././~", description: "Model")
  end

  def test_architectures_mirror_python_truthiness_and_membership
    target = "CohereAsrForConditionalGeneration"
    accepted = [nil, false, 0, 0.0, "", [], {}, target, "prefix-#{target}-suffix", [target], { target => false }]
    rejected = ["OtherArchitecture", ["OtherArchitecture"], { "OtherArchitecture" => target }]

    accepted.each do |architectures|
      actual = ModelIdentity.classify_model_config(
        { "model_type" => "cohere_asr", "architectures" => architectures },
        "owner/model@commit"
      )
      assert_equal [:dense, nil], actual, "expected #{architectures.inspect} to be accepted"
    end
    rejected.each do |architectures|
      error = assert_raises(ArgumentError) do
        ModelIdentity.classify_model_config(
          { "model_type" => "cohere_asr", "architectures" => architectures },
          "owner/model@commit"
        )
      end
      assert_equal(
        "owner/model@commit does not declare CohereAsrForConditionalGeneration",
        error.message
      )
    end
  end

  def test_truthy_scalar_architectures_preserve_python_type_errors
    { true => "bool", 1 => "int", -1.5 => "float" }.each do |architectures, python_type|
      error = assert_raises(TypeError) do
        ModelIdentity.classify_model_config(
          { "model_type" => "cohere_asr", "architectures" => architectures },
          "owner/model@commit"
        )
      end

      assert_equal "argument of type '#{python_type}' is not iterable", error.message
    end
  end

  def test_quantization_config_mirrors_python_null_type_method_and_flag_handling
    dense = ModelIdentity.classify_model_config(
      { "model_type" => "cohere_asr", "quantization_config" => nil },
      "owner/model@commit"
    )
    assert_equal [:dense, nil], dense

    [false, true, 0, 1, "", "int8", []].each do |quantization|
      error = assert_raises(ArgumentError) do
        ModelIdentity.classify_model_config(
          { "model_type" => "cohere_asr", "quantization_config" => quantization },
          "owner/model@commit"
        )
      end
      assert_equal "owner/model@commit has an invalid quantization_config", error.message
    end

    {
      {} => "'unknown'",
      { "quant_method" => "" } => "'unknown'",
      { "quant_method" => nil } => "'none'",
      { "quant_method" => false } => "'false'",
      { "quant_method" => 1e-7 } => "'1e-07'",
      { "quant_method" => 6.156602123266126e+15 } => "'6156602123266126.0'",
      { "quant_method" => "ΟΣ" } => "'ος'",
      { "quant_method" => "\u00a0" } => "'\\xa0'",
      { "quant_method" => [] } => "'[]'",
      { "quant_method" => {} } => "'{}'"
    }.each do |quantization, rendered_method|
      error = assert_raises(ArgumentError) do
        ModelIdentity.classify_model_config(
          { "model_type" => "cohere_asr", "quantization_config" => quantization },
          "owner/model@commit"
        )
      end
      assert_equal(
        "owner/model@commit uses unsupported saved quantization configuration #{rendered_method}",
        error.message
      )
    end

    [nil, 0, 1, "true", []].each do |flag|
      error = assert_raises(ArgumentError) do
        ModelIdentity.classify_model_config(
          {
            "model_type" => "cohere_asr",
            "quantization_config" => {
              "quant_method" => "bitsandbytes",
              "load_in_8bit" => flag
            }
          },
          "owner/model@commit"
        )
      end
      assert_equal(
        "owner/model@commit has an invalid quantization_config: " \
        "bitsandbytes load flags must be boolean",
        error.message
      )
    end
  end

  def test_quantization_config_preserves_flag_precedence_and_original_payload
    quantization = {
      "quant_method" => "BITSANDBYTES",
      "load_in_8bit" => false,
      "_load_in_8bit" => true,
      "load_in_4bit" => true
    }

    format, retained = ModelIdentity.classify_model_config(
      { "model_type" => "cohere_asr", "quantization_config" => quantization },
      "owner/model@commit"
    )

    assert_equal :"bitsandbytes-int4", format
    assert_equal quantization, retained
    refute_same quantization, retained
  end

  def test_local_json_requires_a_regular_json_object_file
    Dir.mktmpdir do |directory|
      root = Pathname(directory)
      array = root.join("array.json").tap { |path| path.write("[]") }
      invalid = root.join("invalid.json").tap { |path| path.write("not-json") }
      not_file = root.join("directory.json").tap(&:mkpath)

      error = assert_raises(ArgumentError) { ModelIdentity.read_json_object(array) }
      assert_match(/is not a JSON object/, error.message)
      error = assert_raises(ArgumentError) { ModelIdentity.read_json_object(invalid) }
      assert_match(/Cannot read JSON object/, error.message)
      error = assert_raises(ArgumentError) { ModelIdentity.read_json_object(not_file) }
      assert_match(/missing or is not a file/, error.message)
    end
  end

  def test_local_json_permission_errors_are_normalized
    Dir.mktmpdir do |directory|
      path = Pathname(directory).join("config.json")
      path.write("{}")
      path.chmod(0o000)

      error = assert_raises(ArgumentError) { ModelIdentity.read_json_object(path) }
      assert_match(/Cannot read JSON object/, error.message)
    ensure
      path&.chmod(0o600) if path&.exist?
    end
  end

  def test_adapter_configuration_is_validated_before_base_weight_artifacts
    Dir.mktmpdir do |directory|
      root = Pathname(directory)
      model = root.join("model").tap(&:mkpath)
      model.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))
      adapter = root.join("adapter").tap(&:mkpath)
      adapter.join("adapter_config.json").write(
        JSON.generate("peft_type" => "IA3", "task_type" => "SEQ_2_SEQ_LM")
      )

      error = assert_raises(ArgumentError) do
        ModelIdentity.resolve(model.to_s, nil, adapter.to_s, nil)
      end

      assert_match(/Adapter peft_type must be LORA/, error.message)
      refute_match(/model weights/, error.message)
    end
  end

  def test_local_weight_validation_can_be_deferred_until_inference
    Dir.mktmpdir do |directory|
      model = Pathname(directory)
      model.join("config.json").write(JSON.generate("model_type" => "cohere_asr"))

      identity = ModelIdentity.resolve(
        model.to_s,
        nil,
        verify_weight_artifacts: false
      )
      error = assert_raises(ArgumentError) do
        ModelIdentity.verify_model_weight_artifacts(identity)
      end

      assert_match(/Transformers model weights/, error.message)
    end
  end

  def test_remote_weight_validation_rejects_a_cache_symlink_outside_the_repository
    Dir.mktmpdir("cohere-model-identity") do |directory|
      commit = "f" * 40
      cache = Pathname(directory).join("hub")
      repository = cache.join("models--owner--model")
      snapshot = repository.join("snapshots", commit).tap(&:mkpath)
      victim = Pathname(directory).join("victim.safetensors").tap do |path|
        path.write("private")
      end
      File.symlink(victim, snapshot.join("model.safetensors"))
      list_calls = 0
      hub = Cohere::Transcribe::Hub.new(cache_dir: cache)
      hub.define_singleton_method(:list_files) do |_repo_id, revision:|
        list_calls += 1
        raise "wrong revision" unless revision == commit

        []
      end
      identity = Cohere::Transcribe::ResolvedModelIdentity.new(
        model_id: "owner/model",
        model_revision: commit,
        model_format: :dense,
        quantization_config: nil,
        adapter_id: nil,
        adapter_revision: nil
      )

      error = assert_raises(ArgumentError) do
        ModelIdentity.verify_model_weight_artifacts(identity, hub: hub)
      end

      assert_match(/does not contain supported Transformers model weights/, error.message)
      assert_equal 1, list_calls
      assert_equal "private", victim.read
    end
  end

  def test_remote_weight_validation_reuses_an_internal_blob_symlink_offline
    Dir.mktmpdir("cohere-model-identity") do |directory|
      commit = "9" * 40
      cache = Pathname(directory).join("hub")
      repository = cache.join("models--owner--model")
      snapshot = repository.join("snapshots", commit).tap(&:mkpath)
      blob = repository.join("blobs", "weights").tap do |path|
        path.dirname.mkpath
        path.write("cached")
      end
      File.symlink(Pathname("../../blobs/weights"), snapshot.join("model.safetensors"))
      hub = Cohere::Transcribe::Hub.new(cache_dir: cache)
      hub.define_singleton_method(:list_files) do |*_args, **_options|
        raise "internal cached blob must not contact the Hub"
      end
      identity = Cohere::Transcribe::ResolvedModelIdentity.new(
        model_id: "owner/model",
        model_revision: commit,
        model_format: :dense,
        quantization_config: nil,
        adapter_id: nil,
        adapter_revision: nil
      )

      assert_nil ModelIdentity.verify_model_weight_artifacts(identity, hub: hub)
      assert_equal blob.realpath,
                   hub.cached_file("owner/model", "model.safetensors", revision: commit).realpath
    end
  end

  private

  def resolve_with(hub)
    ModelIdentity.resolve(
      "owner/model",
      nil,
      "owner/adapter",
      "adapter-release",
      hub: hub,
      verify_weight_artifacts: false
    )
  end

  def lora_config
    {
      "base_model_name_or_path" => "owner/model",
      "peft_type" => "LORA",
      "task_type" => "SEQ_2_SEQ_LM"
    }
  end
end
