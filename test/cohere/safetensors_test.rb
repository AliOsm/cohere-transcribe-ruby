# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/cohere/transcribe/safetensors"
require_relative "converter_test_support"

require "stringio"
require "tmpdir"

class SafetensorsTest < Minitest::Test
  include ConverterTestSupport

  def test_reader_streams_bf16_to_f32_in_bounded_chunks
    Dir.mktmpdir do |directory|
      path = File.join(directory, "model.safetensors")
      write_safetensors(
        path,
        "weights" => {
          dtype: "BF16",
          shape: [4],
          bytes: [0x3f80, 0xc000, 0x0000, 0x7f80].pack("S<*")
        }
      )

      reader = Cohere::Transcribe::Safetensors::Reader.new(path)
      tensor = reader.fetch("weights")
      output = StringIO.new(+"".b)
      written = reader.write_tensor(
        tensor,
        output,
        target_dtype: :f32,
        converter: Cohere::Transcribe::Safetensors::DTypeConverter.new,
        chunk_bytes: 3
      )

      assert_equal 16, written
      assert_equal [1.0, -2.0, 0.0, Float::INFINITY], output.string.unpack("e*")
      assert_equal [4], tensor.shape
      assert_equal 4, tensor.element_count
    end
  end

  def test_portable_converter_rounds_f32_to_ieee_half
    converter = Cohere::Transcribe::Safetensors::DTypeConverter.new
    input = [0.0, -0.0, 1.0, -2.0, 65_504.0, 100_000.0].pack("e*")

    result = converter.convert(input, from: :f32, to: :f16)

    assert_equal [0x0000, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x7c00], result.unpack("S<*")
    assert_equal input, converter.convert(input, from: :f32, to: :f32)
    refute_same input, converter.convert(input, from: :f32, to: :f32)
  end

  def test_portable_converter_rounds_f32_and_f16_to_ieee_bfloat16
    converter = Cohere::Transcribe::Safetensors::DTypeConverter.new
    bits = [
      0x0000_0000, 0x8000_0000, 0x3f80_0000, 0xc000_0000,
      0x3f80_7fff, 0x3f80_8000, 0x3f81_8000, 0x7f80_0000, 0x7f80_0001
    ]

    result = converter.convert(bits.pack("L<*"), from: :f32, to: :bf16)

    assert_equal(
      [0x0000, 0x8000, 0x3f80, 0xc000, 0x3f80, 0x3f80, 0x3f82, 0x7f80, 0x7fc0],
      result.unpack("S<*")
    )
    half = [0x0000, 0x8000, 0x3c00, 0xc000, 0x7c00].pack("S<*")
    assert_equal(
      [0x0000, 0x8000, 0x3f80, 0xc000, 0x7f80],
      converter.convert(half, from: :f16, to: :bf16).unpack("S<*")
    )
  end

  def test_tensor_set_reads_sharded_index
    Dir.mktmpdir do |directory|
      write_safetensors(
        File.join(directory, "model-00001-of-00002.safetensors"),
        "left" => { dtype: "F16", shape: [1], bytes: [0x3c00].pack("S<") }
      )
      write_safetensors(
        File.join(directory, "model-00002-of-00002.safetensors"),
        "right" => { dtype: "F32", shape: [1], bytes: [2.0].pack("e") }
      )
      index = {
        "weight_map" => {
          "left" => "model-00001-of-00002.safetensors",
          "right" => "model-00002-of-00002.safetensors"
        }
      }
      File.write(File.join(directory, "model.safetensors.index.json"), JSON.generate(index))

      set = Cohere::Transcribe::Safetensors::TensorSet.from_directory(directory)

      assert_equal %w[left right], set.names
      assert_equal "F16", set.fetch_any(%w[missing left]).dtype
      assert_equal 2, set.readers.length
    end
  end

  def test_reader_rejects_overlapping_payload_ranges
    Dir.mktmpdir do |directory|
      path = File.join(directory, "model.safetensors")
      header = JSON.generate(
        "left" => { "dtype" => "F16", "shape" => [1], "data_offsets" => [0, 2] },
        "right" => { "dtype" => "F16", "shape" => [1], "data_offsets" => [0, 2] }
      )
      File.binwrite(path, [header.bytesize].pack("Q<") + header + [0].pack("S<"))

      error = assert_raises(Cohere::Transcribe::Safetensors::Error) do
        Cohere::Transcribe::Safetensors::Reader.new(path)
      end
      assert_match(/overlap/, error.message)
    end
  end

  def test_index_rejects_parent_directory_shard_paths
    Dir.mktmpdir do |directory|
      index = { "weight_map" => { "weight" => "../outside.safetensors" } }
      path = File.join(directory, "model.safetensors.index.json")
      File.write(path, JSON.generate(index))

      error = assert_raises(Cohere::Transcribe::Safetensors::Error) do
        Cohere::Transcribe::Safetensors::TensorSet.from_directory(directory)
      end
      assert_match(/invalid shard path/, error.message)
    end
  end

  def test_index_cannot_hide_unindexed_tensors_in_a_shard
    Dir.mktmpdir do |directory|
      shard = "model-00001-of-00001.safetensors"
      write_safetensors(
        File.join(directory, shard),
        "visible" => { dtype: "F16", shape: [1], bytes: [0x3c00].pack("S<") },
        "hidden" => { dtype: "F16", shape: [1], bytes: [0x3c00].pack("S<") }
      )
      index = { "weight_map" => { "visible" => shard } }
      File.write(File.join(directory, "model.safetensors.index.json"), JSON.generate(index))

      error = assert_raises(Cohere::Transcribe::Safetensors::Error) do
        Cohere::Transcribe::Safetensors::TensorSet.from_directory(directory)
      end
      assert_match(/tensors absent from its index/, error.message)
    end
  end

  def test_native_converter_discovers_ggml_beside_configured_runtime
    Dir.mktmpdir do |directory|
      native = File.join(directory, native_library_filename)
      ggml = File.join(directory, ggml_library_filename)
      File.binwrite(native, "")
      File.binwrite(ggml, "")
      previous = ENV.fetch("COHERE_TRANSCRIBE_NATIVE_LIBRARY", nil)
      ENV["COHERE_TRANSCRIBE_NATIVE_LIBRARY"] = native

      candidates = Cohere::Transcribe::Safetensors::NativeDTypeConverter.candidate_libraries("/explicit/ggml")

      assert_equal "/explicit/ggml", candidates.first
      assert_includes candidates, ggml
    ensure
      ENV["COHERE_TRANSCRIBE_NATIVE_LIBRARY"] = previous
    end
  end

  def test_native_converter_never_searches_neighboring_development_builds
    directories = Cohere::Transcribe::Safetensors::NativeDTypeConverter.send(:candidate_directories)

    refute(directories.any? { |path| path.include?("/research/CrispASR/") })
    refute_includes directories, File.expand_path("../../build/ggml/src", __dir__)
  end

  private

  def native_library_filename
    case RbConfig::CONFIG["host_os"]
    when /darwin/ then "libcrispasr.dylib"
    when /mswin|mingw|cygwin/ then "crispasr.dll"
    else "libcrispasr.so"
    end
  end

  def ggml_library_filename
    case RbConfig::CONFIG["host_os"]
    when /darwin/ then "libggml-base.dylib"
    when /mswin|mingw|cygwin/ then "ggml-base.dll"
    else "libggml-base.so"
    end
  end
end
