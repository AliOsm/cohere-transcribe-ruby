# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/cohere/transcribe/dense_converter"
require_relative "converter_test_support"

require "tmpdir"

class DenseConverterTest < Minitest::Test
  include ConverterTestSupport

  Converter = Cohere::Transcribe::DenseConverter

  def test_public_entry_point_and_source_artifact_contract
    assert_equal(
      %w[
        config.json tokenizer.json model.safetensors model.safetensors.index.json
        pytorch_model.bin pytorch_model.bin.index.json
      ],
      Converter.source_artifact_filenames
    )
    assert_equal(
      {
        required: %w[config.json tokenizer.json],
        one_weight_file: %w[
          model.safetensors model.safetensors.index.json
          pytorch_model.bin pytorch_model.bin.index.json
        ]
      },
      Converter.required_source_artifacts
    )
    assert_equal 87, Converter.tensor_mappings(encoder_layers: 1, decoder_layers: 1).length
    assert_equal 2102, Converter.tensor_mappings(encoder_layers: 48, decoder_layers: 8).length
  end

  def test_config_validation_uses_model_identity_python_parity_contract
    target = "CohereAsrForConditionalGeneration"
    accepted = [nil, false, 0, "", [], {}, target, "prefix-#{target}", { target => false }]

    accepted.each do |architectures|
      validate_model_config!(
        "model_type" => "cohere_asr",
        "architectures" => architectures
      )
    end

    error = assert_raises(Converter::Error) do
      validate_model_config!(
        "model_type" => "cohere_asr",
        "quantization_config" => false
      )
    end
    assert_equal(
      "#{File.expand_path("fixture-model")} has an invalid quantization_config",
      error.message
    )
  end

  def test_converts_current_prefixed_f32_checkpoint_with_tokenizer_json_only
    Dir.mktmpdir do |directory|
      vocabulary = write_dense_fixture(directory, layout: :current, dtype: "F32")
      output = File.join(directory, "current.gguf")
      progress = []

      result = Converter.convert(
        model_dir: directory,
        output_path: output,
        chunk_bytes: 7,
        fsync: false,
        progress: ->(**event) { progress << event }
      )
      document = read_gguf(result.path)

      assert_equal File.expand_path(output), result.path.to_s
      assert_equal 87, result.tensor_count
      assert_equal({ "F32" => 87 }, result.source_dtype_counts)
      assert_equal 87, progress.length
      assert_equal 87, progress.last.fetch(:index)
      assert_equal "runtime-generated-fp32", document[:metadata]["cohere_transcribe.frontend"]
      assert_equal vocabulary, document[:metadata]["tokenizer.ggml.tokens"]
      assert_equal 1, document[:metadata]["general.file_type"]
      assert_equal 87, document[:tensors].length
      refute(document[:tensors].keys.any? { |name| name.start_with?("fe.") })
      assert_equal 1, document[:tensors].fetch("enc.proj.weight").fetch(:dtype)
      assert_equal 0, document[:tensors].fetch("enc.proj.bias").fetch(:dtype)
      assert_equal [0x3c00] * 4, gguf_tensor_bytes(document, "enc.proj.weight", 8).unpack("S<*")
      refute File.exist?(File.join(directory, "tokenizer.model"))
    end
  end

  def test_converts_legacy_bf16_checkpoint_and_omits_saved_frontend_constants
    Dir.mktmpdir do |directory|
      write_dense_fixture(directory, layout: :legacy, dtype: "BF16", include_frontend: true)
      output = File.join(directory, "legacy-f32.gguf")

      result = Converter.convert(
        model_dir: directory,
        output_path: output,
        output_type: :f32,
        fsync: false
      )
      document = read_gguf(result.path)

      assert_equal({ "BF16" => 87 }, result.source_dtype_counts)
      assert_equal({ "f32" => 87 }, result.output_dtype_counts)
      assert(document[:tensors].values.all? { |tensor| tensor.fetch(:dtype).zero? })
      assert_equal [1.0] * 4, gguf_tensor_bytes(document, "enc.proj.weight", 16).unpack("e*")
      refute(document[:tensors].keys.any? { |name| name.start_with?("fe.") })
    end
  end

  def test_converts_primary_dense_weights_to_bfloat16
    Dir.mktmpdir do |directory|
      write_dense_fixture(directory, layout: :current, dtype: "F32")
      output = File.join(directory, "current-bf16.gguf")

      result = Converter.convert(
        model_dir: directory,
        output_path: output,
        output_type: :bf16,
        fsync: false
      )
      document = read_gguf(result.path)

      assert_equal 24, document[:metadata]["general.file_type"]
      assert_equal 30, document[:tensors].fetch("enc.proj.weight").fetch(:dtype)
      assert_equal 0, document[:tensors].fetch("enc.proj.bias").fetch(:dtype)
      assert_equal [0x3f80] * 4,
                   gguf_tensor_bytes(document, "enc.proj.weight", 8).unpack("S<*")
      assert_equal({ "bf16" => 32, "f32" => 55 }, result.output_dtype_counts)
    end
  end

  def test_converts_a_dense_pytorch_bin_state_dictionary
    Dir.mktmpdir do |directory|
      write_dense_fixture(directory, layout: :current, dtype: "F32")
      safetensors = File.join(directory, "model.safetensors")
      write_legacy_pytorch_from_safetensors(
        safetensors, File.join(directory, "pytorch_model.bin")
      )
      File.delete(safetensors)
      output = File.join(directory, "pytorch.gguf")

      result = Converter.convert(
        model_dir: directory,
        output_path: output,
        chunk_bytes: 7,
        fsync: false
      )
      document = read_gguf(result.path)

      assert_equal 87, result.tensor_count
      assert_equal({ "F32" => 87 }, result.source_dtype_counts)
      assert_equal [0x3c00] * 4, gguf_tensor_bytes(document, "enc.proj.weight", 8).unpack("S<*")
    end
  end

  def test_rejects_an_unmapped_learned_tensor
    Dir.mktmpdir do |directory|
      write_dense_fixture(directory, layout: :current, dtype: "F32", include_unexpected: true)
      converter = Converter.new(directory, output_path: File.join(directory, "bad.gguf"))

      error = assert_raises(Converter::Error) { converter.plan }

      assert_match(/unmapped model tensors/, error.message)
      assert_match(/model\.unexpected\.weight/, error.message)
    end
  end

  def test_rejects_token_ids_outside_the_embedding_vocabulary
    Dir.mktmpdir do |directory|
      write_dense_fixture(directory, layout: :current, dtype: "F32")
      tokenizer_path = File.join(directory, "tokenizer.json")
      tokenizer = JSON.parse(File.read(tokenizer_path))
      token = tokenizer.fetch("model").fetch("vocab").keys.first
      tokenizer.fetch("model").fetch("vocab")[token] = 1_000_000
      File.write(tokenizer_path, JSON.generate(tokenizer))
      converter = Converter.new(directory, output_path: File.join(directory, "bad-tokenizer.gguf"))

      error = assert_raises(Converter::Error) { converter.plan }

      assert_match(/outside the checkpoint vocabulary size/, error.message)
    end
  end

  private

  def validate_model_config!(config)
    converter = Converter.new("fixture-model", output_path: "fixture.gguf")
    converter.instance_variable_set(:@config, config)
    converter.send(:validate_config!)
  end

  def write_legacy_pytorch_from_safetensors(source_path, destination_path)
    reader = Cohere::Transcribe::Safetensors::Reader.new(source_path)
    records = reader.tensors.values.each_with_index.map do |tensor, index|
      {
        name: tensor.name,
        dtype: tensor.dtype,
        shape: tensor.shape,
        key: index.to_s,
        bytes: File.binread(source_path, tensor.nbytes, tensor.data_start)
      }
    end
    payload = +torch_pickle_integer(Cohere::Transcribe::PyTorchCheckpoint::MAGIC_NUMBER)
    payload << torch_pickle_integer(Cohere::Transcribe::PyTorchCheckpoint::PROTOCOL_VERSION)
    payload << torch_proto << "}."
    payload << torch_state_dict_pickle(records)
    payload << torch_proto << "](" << records.map { |record| torch_unicode(record[:key]) }.join << "e."
    records.each do |record|
      width = Cohere::Transcribe::Safetensors::DTYPE_BYTES.fetch(record[:dtype])
      payload << [record[:bytes].bytesize / width].pack("Q<")
      payload << record[:bytes]
    end
    File.binwrite(destination_path, payload)
  end

  def torch_state_dict_pickle(records)
    payload = +torch_proto
    payload << torch_global("collections", "OrderedDict") << ")R("
    records.each do |record|
      width = Cohere::Transcribe::Safetensors::DTYPE_BYTES.fetch(record[:dtype])
      elements = record[:bytes].bytesize / width
      payload << torch_unicode(record[:name])
      payload << torch_global("torch._utils", "_rebuild_tensor_v2") << "("
      payload << torch_tuple(
        [
          torch_unicode("storage"),
          torch_global("torch", torch_storage_class(record[:dtype])),
          torch_unicode(record[:key]),
          torch_unicode("cpu"),
          torch_integer(elements)
        ]
      )
      payload << "Q" << torch_integer(0)
      payload << torch_tuple(record[:shape].map { |value| torch_integer(value) })
      payload << torch_tuple(contiguous_stride(record[:shape]).map { |value| torch_integer(value) })
      payload << "\x89".b << torch_global("collections", "OrderedDict") << ")RtR"
    end
    payload << "u."
  end

  def contiguous_stride(shape)
    expected = 1
    shape.reverse.map do |dimension|
      value = expected
      expected *= dimension
      value
    end.reverse
  end

  def torch_storage_class(dtype)
    {
      "F32" => "FloatStorage", "F16" => "HalfStorage",
      "BF16" => "BFloat16Storage", "I64" => "LongStorage"
    }.fetch(dtype)
  end

  def torch_proto
    "\x80\x02".b
  end

  def torch_global(mod, name)
    "c#{mod}\n#{name}\n".b
  end

  def torch_unicode(value)
    value = value.to_s.b
    "X".b + [value.bytesize].pack("V") + value
  end

  def torch_tuple(items)
    "(".b + items.join + "t"
  end

  def torch_pickle_integer(value)
    torch_proto + torch_integer(value) + "."
  end

  def torch_integer(value)
    return "K".b + [value].pack("C") if value.between?(0, 0xff)
    return "M".b + [value].pack("v") if value.between?(0, 0xffff)
    return "J".b + [value].pack("l<") if value.between?(-0x8000_0000, 0x7fff_ffff)

    bytes = +"".b
    until value.zero?
      bytes << (value & 0xff)
      value >>= 8
    end
    bytes << 0 if !bytes.empty? && bytes.getbyte(-1).anybits?(0x80)
    "\x8a".b + [bytes.bytesize].pack("C") + bytes
  end

  def write_dense_fixture(directory, layout:, dtype:, include_frontend: false, include_unexpected: false)
    vocabulary = (Converter::REQUIRED_PROMPT_TOKENS + ["<|ar|>"]).uniq
    tokenizer = {
      "model" => { "vocab" => vocabulary.each_with_index.to_h },
      "added_tokens" => vocabulary.each_with_index.map { |token, id| { "id" => id, "content" => token } }
    }
    config = {
      "_name_or_path" => "fixture/#{layout}",
      "model_type" => "cohere_asr",
      "architectures" => ["CohereAsrForConditionalGeneration"],
      "encoder_config" => {
        "num_hidden_layers" => 1,
        "hidden_size" => 2,
        "num_attention_heads" => 1,
        "intermediate_size" => 4,
        "num_mel_bins" => 128,
        "subsampling_factor" => 8
      },
      "transf_decoder" => {
        "config_dict" => {
          "num_layers" => 1,
          "hidden_size" => 2,
          "num_attention_heads" => 1,
          "inner_size" => 4,
          "max_sequence_length" => 8
        }
      },
      "sample_rate" => 16_000,
      "preprocessor" => { "n_fft" => 512, "hop_length" => 160, "win_length" => 400 }
    }
    File.write(File.join(directory, "config.json"), JSON.generate(config))
    File.write(File.join(directory, "tokenizer.json"), JSON.generate(tokenizer))

    tensors = Converter.tensor_mappings(encoder_layers: 1, decoder_layers: 1).to_h do |mapping|
      source_name = checkpoint_tensor_name(mapping.source_name, layout)
      shape = dense_tensor_shape(mapping.source_name, vocabulary.length)
      [source_name, { dtype: dtype, shape: shape, bytes: dense_tensor_bytes(dtype, shape) }]
    end
    counter_name = checkpoint_tensor_name(
      "encoder.layers.0.conv.batch_norm.num_batches_tracked",
      layout
    )
    tensors[counter_name] = { dtype: "I64", shape: [], bytes: [0].pack("q<") }
    if include_frontend
      tensors["preprocessor.featurizer.fb"] = {
        dtype: "F32", shape: [2, 2], bytes: ([1.0] * 4).pack("e*")
      }
      tensors["preprocessor.featurizer.window"] = {
        dtype: "F32", shape: [4], bytes: ([1.0] * 4).pack("e*")
      }
    end
    tensors["model.unexpected.weight"] = { dtype: dtype, shape: [1], bytes: dense_tensor_bytes(dtype, [1]) } if include_unexpected
    write_safetensors(File.join(directory, "model.safetensors"), tensors)
    vocabulary
  end

  def checkpoint_tensor_name(name, layout)
    return name if layout == :legacy || name.start_with?("log_softmax.")

    "model.#{name}"
  end

  def dense_tensor_shape(name, vocabulary_size)
    case name
    when "encoder.pre_encode.conv.0.weight" then [256, 1, 3, 3]
    when "encoder_decoder_proj.weight" then [2, 2]
    when "transf_decoder._embedding.token_embedding.weight" then [vocabulary_size, 2]
    when "transf_decoder._embedding.position_embedding.pos_enc" then [8, 2]
    when "encoder.layers.0.feed_forward1.linear1.weight" then [4, 2]
    when "encoder.layers.0.conv.depthwise_conv.weight" then [2, 1, 9]
    when "transf_decoder._decoder.layers.0.third_sub_layer.dense_in.weight" then [4, 2]
    when "log_softmax.mlp.layer0.weight" then [vocabulary_size, 2]
    else
      name.end_with?(".weight") ? [2, 2] : [2]
    end
  end

  def dense_tensor_bytes(dtype, shape)
    count = shape.empty? ? 1 : shape.reduce(1, :*)
    case dtype
    when "BF16" then ([0x3f80] * count).pack("S<*")
    when "F32" then ([1.0] * count).pack("e*")
    else raise "unsupported fixture dtype #{dtype.inspect}"
    end
  end
end
