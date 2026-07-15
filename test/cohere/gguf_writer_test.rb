# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/cohere/transcribe/gguf_writer"
require_relative "converter_test_support"

require "stringio"
require "tmpdir"

class GgufWriterTest < Minitest::Test
  include ConverterTestSupport

  def test_writes_gguf_v3_metadata_tensor_infos_and_aligned_data
    writer = Cohere::Transcribe::GGUF::Writer.new
    writer.add_string("general.architecture", "cohere-transcribe")
    writer.add_uint32("answer", 42)
    writer.add_uint64("large", 2**40)
    writer.add_float32("ratio", 0.5)
    writer.add_bool("ready", true)
    writer.add_string_array("tokens", ["▁", "<eos>"])
    first_bytes = (0...6).to_a.pack("S<*")
    second_bytes = [1.5].pack("e")
    writer.add_tensor("first", shape: [2, 3], dtype: :f16) { |io| io.write(first_bytes) }
    writer.add_tensor("second", shape: [1], dtype: :f32) { |io| io.write(second_bytes) }
    output = StringIO.new(+"".b)

    writer.write_to(output)
    document = read_gguf(output.string)

    assert_equal 3, document[:version]
    assert_equal "cohere-transcribe", document[:metadata]["general.architecture"]
    assert_equal 42, document[:metadata]["answer"]
    assert_equal 2**40, document[:metadata]["large"]
    assert_in_delta 0.5, document[:metadata]["ratio"]
    assert document[:metadata]["ready"]
    assert_equal ["▁", "<eos>"], document[:metadata]["tokens"]
    assert_equal({ dimensions: [3, 2], dtype: 1, offset: 0 }, document[:tensors]["first"])
    assert_equal({ dimensions: [1], dtype: 0, offset: 32 }, document[:tensors]["second"])
    assert_equal first_bytes, gguf_tensor_bytes(document, "first", first_bytes.bytesize)
    assert_equal second_bytes, gguf_tensor_bytes(document, "second", second_bytes.bytesize)
    assert_equal 0, document[:data_start] % 32
  end

  def test_rejects_a_tensor_callback_that_writes_the_wrong_size
    writer = Cohere::Transcribe::GGUF::Writer.new
    writer.add_string("general.architecture", "test")
    writer.add_tensor("broken", shape: [2], dtype: :f16) { |io| io.write("x") }

    error = assert_raises(Cohere::Transcribe::GGUF::Error) do
      writer.write_to(StringIO.new(+"".b))
    end
    assert_match(/wrote 1 bytes; expected 4/, error.message)
  end

  def test_atomic_file_write_refuses_overwrite_by_default
    Dir.mktmpdir do |directory|
      path = File.join(directory, "model.gguf")
      writer = Cohere::Transcribe::GGUF::Writer.new
      writer.add_string("general.architecture", "test")
      writer.add_tensor("weight", shape: [1], dtype: :f32) { |io| io.write([1.0].pack("e")) }

      assert_equal File.expand_path(path), writer.write(path, fsync: false).to_s
      assert File.file?(path)
      assert_raises(Cohere::Transcribe::GGUF::Error) { writer.write(path, fsync: false) }

      writer.write(path, overwrite: true, fsync: false)
      assert_equal "GGUF", File.binread(path, 4)
    end
  end

  def test_no_overwrite_publish_does_not_clobber_a_file_created_during_conversion
    Dir.mktmpdir do |directory|
      path = File.join(directory, "model.gguf")
      writer = Cohere::Transcribe::GGUF::Writer.new
      writer.add_string("general.architecture", "test")
      writer.add_tensor("weight", shape: [1], dtype: :f32) do |io|
        io.write([1.0].pack("e"))
        File.binwrite(path, "concurrent winner")
      end

      error = assert_raises(Cohere::Transcribe::GGUF::Error) do
        writer.write(path, overwrite: false, fsync: false)
      end

      assert_match(/already exists/, error.message)
      assert_equal "concurrent winner", File.binread(path)
    end
  end

  def test_nondefault_alignment_is_declared_in_metadata
    writer = Cohere::Transcribe::GGUF::Writer.new(alignment: 64)
    writer.add_string("general.architecture", "test")
    writer.add_tensor("weight", shape: [1], dtype: :f32) { |io| io.write([1.0].pack("e")) }
    output = StringIO.new(+"".b)

    writer.write_to(output)
    document = read_gguf(output.string)

    assert_equal 64, document[:metadata]["general.alignment"]
    assert_equal 0, document[:data_start] % 64
  end

  def test_unsigned_metadata_rejects_out_of_range_values
    writer = Cohere::Transcribe::GGUF::Writer.new

    assert_raises(ArgumentError) { writer.add_uint32("negative", -1) }
    assert_raises(ArgumentError) { writer.add_uint64("too-large", 2**64) }
  end
end
