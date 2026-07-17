# frozen_string_literal: true

require_relative "../test_helper"
require_relative "../../lib/cohere/transcribe/pytorch_checkpoint"

require "stringio"
require "tmpdir"
require "zlib"

class PyTorchCheckpointTest < Minitest::Test
  Checkpoint = Cohere::Transcribe::PyTorchCheckpoint

  def test_reads_current_torch_save_zip_and_reorders_a_strided_tensor
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "matrix" => tensor(dtype: "F32", key: "0", shape: [3, 2], stride: [1, 3], elements: 6),
        "half" => tensor(dtype: "F16", key: "1", shape: [2], stride: [1], elements: 2)
      }
      storages = {
        "0" => (0..5).map(&:to_f).pack("e*"),
        "1" => [0x3c00, 0xc000].pack("S<*")
      }
      write_zip_checkpoint(path, tensors, storages)

      reader = Checkpoint::Reader.new(path)
      output = StringIO.new(+"".b)
      written = reader.write_tensor(
        reader.fetch("matrix"), output, target_dtype: "F32",
                                        converter: Cohere::Transcribe::Safetensors::DTypeConverter.new,
                                        chunk_bytes: 5
      )

      assert_equal %w[matrix half], reader.names
      assert_equal 24, written
      assert_equal [0.0, 3.0, 1.0, 4.0, 2.0, 5.0], output.string.unpack("e*")
      assert_equal "F16", reader.fetch("half").dtype
    end
  end

  def test_zip_storage_crc_covers_the_complete_payload_before_contiguous_conversion
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "slice" => tensor(dtype: "F32", key: "0", shape: [1], stride: [1], elements: 4)
      }
      write_zip_checkpoint(path, tensors, "0" => [1.0, 2.0, 3.0, 4.0].pack("e*"))
      reader = Checkpoint::Reader.new(path)
      flip_byte(path, reader.fetch("slice").storage_data_start + 12)
      output = StringIO.new(+"".b)

      error = assert_raises(Checkpoint::Error) do
        reader.write_tensor(reader.fetch("slice"), output, target_dtype: "F32")
      end

      assert_match(/storage "0" failed its CRC check/, error.message)
      assert_empty output.string
    end
  end

  def test_zip_storage_crc_covers_unused_bytes_for_a_strided_tensor
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "matrix" => tensor(dtype: "F32", key: "0", shape: [2, 2], stride: [1, 3], elements: 6)
      }
      write_zip_checkpoint(path, tensors, "0" => (0..5).map(&:to_f).pack("e*"))
      reader = Checkpoint::Reader.new(path)
      flip_byte(path, reader.fetch("matrix").storage_data_start + 20)
      output = StringIO.new(+"".b)

      error = assert_raises(Checkpoint::Error) do
        reader.write_tensor(reader.fetch("matrix"), output, target_dtype: "F32")
      end

      assert_match(/storage "0" failed its CRC check/, error.message)
      assert_empty output.string
    end
  end

  def test_shared_zip_storage_is_fully_checked_only_once_for_contiguous_and_strided_tensors
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "matrix" => tensor(dtype: "F32", key: "0", shape: [3, 2], stride: [1, 3], elements: 6),
        "flat" => tensor(dtype: "F32", key: "0", shape: [6], stride: [1], elements: 6)
      }
      write_zip_checkpoint(path, tensors, "0" => (0..5).map(&:to_f).pack("e*"))
      reader = Checkpoint::Reader.new(path)
      checked = []
      probe = Module.new do
        define_method(:verify_storage_crc!) do |storage_key, payload|
          checked << storage_key
          super(storage_key, payload)
        end
      end
      reader.singleton_class.prepend(probe)

      matrix = StringIO.new(+"".b)
      flat = StringIO.new(+"".b)
      reader.write_tensor(reader.fetch("matrix"), matrix, target_dtype: "F32", chunk_bytes: 5)
      reader.write_tensor(reader.fetch("flat"), flat, target_dtype: "F32", chunk_bytes: 5)

      assert_equal ["0"], checked
      assert_equal [0.0, 3.0, 1.0, 4.0, 2.0, 5.0], matrix.string.unpack("e*")
      assert_equal (0..5).map(&:to_f), flat.string.unpack("e*")
    end
  end

  def test_full_contiguous_zip_storage_combines_crc_with_conversion
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "flat" => tensor(dtype: "F32", key: "0", shape: [6], stride: [1], elements: 6)
      }
      write_zip_checkpoint(path, tensors, "0" => (0..5).map(&:to_f).pack("e*"))
      reader = Checkpoint::Reader.new(path)
      separate_checks = []
      probe = Module.new do
        define_method(:verify_storage_crc!) do |storage_key, payload|
          separate_checks << storage_key
          super(storage_key, payload)
        end
      end
      reader.singleton_class.prepend(probe)
      output_bytes = +"".b
      output = Object.new
      output.define_singleton_method(:write) do |bytes|
        output_bytes << bytes
        bytes.bytesize
      end

      reader.write_tensor(reader.fetch("flat"), output, target_dtype: "F32", chunk_bytes: 5)

      assert_empty separate_checks
      assert_equal (0..5).map(&:to_f), output_bytes.unpack("e*")
    end
  end

  def test_inline_crc_completion_is_reused_by_other_views_of_the_storage
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "flat" => tensor(dtype: "F32", key: "0", shape: [6], stride: [1], elements: 6),
        "matrix" => tensor(dtype: "F32", key: "0", shape: [3, 2], stride: [1, 3], elements: 6)
      }
      write_zip_checkpoint(path, tensors, "0" => (0..5).map(&:to_f).pack("e*"))
      reader = Checkpoint::Reader.new(path)
      separate_checks = []
      probe = Module.new do
        define_method(:verify_storage_crc!) do |storage_key, payload|
          separate_checks << storage_key
          super(storage_key, payload)
        end
      end
      reader.singleton_class.prepend(probe)

      flat = StringIO.new(+"".b)
      matrix = StringIO.new(+"".b)
      reader.write_tensor(reader.fetch("flat"), flat, target_dtype: "F32", chunk_bytes: 5)
      reader.write_tensor(reader.fetch("matrix"), matrix, target_dtype: "F32", chunk_bytes: 5)

      assert_empty separate_checks
      assert_equal (0..5).map(&:to_f), flat.string.unpack("e*")
      assert_equal [0.0, 3.0, 1.0, 4.0, 2.0, 5.0], matrix.string.unpack("e*")
    end
  end

  def test_inline_crc_failure_keeps_the_storage_unverified
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "flat" => tensor(dtype: "F32", key: "0", shape: [4], stride: [1], elements: 4)
      }
      write_zip_checkpoint(path, tensors, "0" => [1.0, 2.0, 3.0, 4.0].pack("e*"))
      reader = Checkpoint::Reader.new(path)
      flip_byte(path, reader.fetch("flat").storage_data_start + 12)
      output_bytes = +"".b
      output = Object.new
      output.define_singleton_method(:write) do |bytes|
        output_bytes << bytes
        bytes.bytesize
      end

      2.times do
        offset = output_bytes.bytesize
        error = assert_raises(Checkpoint::Error) do
          reader.write_tensor(reader.fetch("flat"), output, target_dtype: "F32")
        end

        assert_match(/storage "0" failed its CRC check/, error.message)
        assert_equal 16, output_bytes.bytesize - offset
      end
    end
  end

  def test_reads_legacy_pickle_and_raw_storage_stream_without_executing_python
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      tensors = {
        "weights" => tensor(dtype: "BF16", key: "7", shape: [4], stride: [1], elements: 4)
      }
      storages = { "7" => [0x3f80, 0xc000, 0, 0x7f80].pack("S<*") }
      write_legacy_checkpoint(path, tensors, storages)

      reader = Checkpoint::Reader.new(path)
      output = StringIO.new(+"".b)
      reader.write_tensor(
        reader.fetch("weights"), output, target_dtype: "F32",
                                         converter: Cohere::Transcribe::Safetensors::DTypeConverter.new,
                                         chunk_bytes: 3
      )

      assert_equal [1.0, -2.0, 0.0, Float::INFINITY], output.string.unpack("e*")
    end
  end

  def test_restricted_pickle_refuses_unlisted_global_reducers
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      malicious = proto + global("posix", "system") + tuple([unicode("touch /tmp/not-allowed")]) + "R."
      write_zip(path, "archive/data.pkl" => malicious, "archive/byteorder" => "little")

      error = assert_raises(Checkpoint::Error) { Checkpoint::Reader.new(path) }

      assert_match(/refuses reducer posix\.system/, error.message)
      refute File.exist?("/tmp/not-allowed")
    end
  end

  def test_zip_reader_checks_metadata_crc_and_entry_paths
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      write_zip(path, "archive/data.pkl" => proto + "}.", "archive/byteorder" => "little")
      bytes = File.binread(path)
      bytes[bytes.index("little")] = "L"
      File.binwrite(path, bytes)

      error = assert_raises(Checkpoint::Error) { Checkpoint::Reader.new(path) }
      assert_match(/CRC/, error.message)

      malformed = File.join(directory, "malformed.bin")
      write_zip(malformed, "../data.pkl" => proto + "}.")
      error = assert_raises(Checkpoint::Error) { Checkpoint::Reader.new(malformed) }
      assert_match(/invalid entry name/, error.message)

      write_zip(malformed, "archive/data\0.pkl" => proto + "}.")
      error = assert_raises(Checkpoint::Error) { Checkpoint::Reader.new(malformed) }
      assert_match(/invalid entry name/, error.message)
    end
  end

  def test_zip_reader_checks_local_metadata_and_bounds_deflate_expansion
    Dir.mktmpdir do |directory|
      path = File.join(directory, "metadata.bin")
      write_zip(path, "archive/data.pkl" => proto + "}.")
      bytes = File.binread(path)
      bytes[8, 2] = [8].pack("v") # Local method differs from the central entry.
      File.binwrite(path, bytes)

      error = assert_raises(Checkpoint::Error) { Checkpoint::ZipArchive.new(path) }
      assert_match(%r{central/local metadata disagree}, error.message)

      bomb = File.join(directory, "bounded-deflate.bin")
      write_deflated_zip(bomb, "archive/data.pkl", "A" * (2 * 1024 * 1024), declared_size: 16)
      archive = Checkpoint::ZipArchive.new(bomb)
      error = nil
      _output, warnings = capture_io do
        error = assert_raises(Checkpoint::Error) { archive.read("archive/data.pkl", limit: 1024) }
      end
      assert_empty warnings
      assert_match(/expands beyond its declared size/, error.message)
    end
  end

  def test_reads_zip64_end_records_and_central_entry_sizes
    Dir.mktmpdir do |directory|
      path = File.join(directory, "zip64.bin")
      write_zip64(path, "archive/data.pkl", proto + "}.")
      locator_offset = File.binread(path).index([Checkpoint::ZIP64_LOCATOR].pack("V"))
      archive_class = Class.new(Checkpoint::ZipArchive) do
        attr_reader :read_ranges

        def initialize(...)
          @read_ranges = []
          super
        end

        private

        def read_range(offset, length)
          @read_ranges << [offset, length]
          super
        end
      end

      archive = archive_class.new(path)

      assert_equal ["archive/data.pkl"], archive.entries.keys
      assert_equal proto + "}.", archive.read("archive/data.pkl")
      locator_reads = archive.read_ranges.select { |offset, _length| offset == locator_offset }
      assert_equal [[locator_offset, 20]], locator_reads
    end
  end

  def test_classic_zip_accepts_exact_16_bit_values_in_32_bit_entry_fields
    Dir.mktmpdir do |directory|
      size_path = File.join(directory, "exact-size.zip")
      payload = "x" * 0xffff
      write_zip(size_path, "archive/data.pkl" => payload)

      assert_equal payload, Checkpoint::ZipArchive.new(size_path).read("archive/data.pkl")

      offset_path = File.join(directory, "exact-offset.zip")
      write_zip_at_offset(offset_path, "archive/data.pkl", proto + "}.", local_offset: 0xffff)

      assert_equal proto + "}.", Checkpoint::ZipArchive.new(offset_path).read("archive/data.pkl")
    end
  end

  def test_classic_zip_accepts_exact_maximum_entry_count_without_a_zip64_locator
    Dir.mktmpdir do |directory|
      path = File.join(directory, "exact-entry-count.zip")
      write_empty_classic_zip(path, 0xffff)

      archive = Checkpoint::ZipArchive.new(path)

      assert_equal 0xffff, archive.entries.length
      assert archive.entries.key?("entry-00000")
      assert archive.entries.key?("entry-65534")
    end
  end

  def test_zip64_rejects_a_malformed_locator_and_end_record
    Dir.mktmpdir do |directory|
      path = File.join(directory, "malformed-zip64.bin")
      write_zip64(path, "archive/data.pkl", proto + "}.")
      bytes = File.binread(path)
      locator = bytes.index([Checkpoint::ZIP64_LOCATOR].pack("V"))
      bytes[locator + 4, 4] = [1].pack("V")
      File.binwrite(path, bytes)

      error = assert_raises(Checkpoint::Error) { Checkpoint::ZipArchive.new(path) }
      assert_match(/invalid ZIP64 locator/, error.message)

      write_zip64(path, "archive/data.pkl", proto + "}.")
      bytes = File.binread(path)
      locator = bytes.index([Checkpoint::ZIP64_LOCATOR].pack("V"))
      bytes[locator + 8, 8] = [1].pack("Q<")
      File.binwrite(path, bytes)

      error = assert_raises(Checkpoint::Error) { Checkpoint::ZipArchive.new(path) }

      assert_match(/invalid ZIP64 end record/, error.message)
    end
  end

  def test_restricted_pickle_rejects_pathological_long_integers_before_materializing_them
    bytes = proto + "\x8b".b + [Checkpoint::PICKLE_LONG_BYTES_LIMIT + 1].pack("V")
    bytes << ("\0" * (Checkpoint::PICKLE_LONG_BYTES_LIMIT + 1)) << "."

    error = assert_raises(Checkpoint::Error) do
      Checkpoint::RestrictedUnpickler.new(StringIO.new(bytes)).load
    end

    assert_match(/integer exceeds/, error.message)
  end

  def test_rejects_legacy_tar_checkpoint_with_actionable_conversion_message
    Dir.mktmpdir do |directory|
      path = File.join(directory, "pytorch_model.bin")
      bytes = "\0".b * 512
      bytes[257, 5] = "ustar"
      File.binwrite(path, bytes)

      error = assert_raises(Checkpoint::Error) { Checkpoint::Reader.new(path) }

      assert_match(/restricted weights-only reader/, error.message)
      assert_match(/Safetensors/, error.message)
    end
  end

  def test_tensor_set_validates_shard_index_coverage_and_paths
    Dir.mktmpdir do |directory|
      shard = "pytorch_model-00001-of-00001.bin"
      tensors = { "visible" => tensor(dtype: "F32", key: "0", shape: [1], stride: [1], elements: 1) }
      write_zip_checkpoint(File.join(directory, shard), tensors, "0" => [1.0].pack("e"))
      File.write(
        File.join(directory, "pytorch_model.bin.index.json"),
        JSON.generate("weight_map" => { "visible" => shard })
      )

      set = Checkpoint::TensorSet.from_directory(directory)
      assert_equal ["visible"], set.names

      File.write(
        File.join(directory, "pytorch_model.bin.index.json"),
        JSON.generate("weight_map" => { "visible" => "../outside.bin" })
      )
      error = assert_raises(Checkpoint::Error) { Checkpoint::TensorSet.from_directory(directory) }
      assert_match(/invalid shard path/, error.message)

      File.write(
        File.join(directory, "pytorch_model.bin.index.json"),
        JSON.generate("weight_map" => { "visible" => "invalid\0.bin" })
      )
      error = assert_raises(Checkpoint::Error) { Checkpoint::TensorSet.from_directory(directory) }
      assert_match(/invalid shard path/, error.message)
    end
  end

  private

  def tensor(dtype:, key:, shape:, stride:, elements:, offset: 0)
    {
      dtype: dtype, key: key, shape: shape, stride: stride,
      elements: elements, offset: offset
    }
  end

  def write_zip_checkpoint(path, tensors, storages)
    entries = {
      "archive/data.pkl" => state_dict_pickle(tensors),
      "archive/byteorder" => "little"
    }
    storages.each { |key, bytes| entries["archive/data/#{key}"] = bytes }
    write_zip(path, entries)
  end

  def write_legacy_checkpoint(path, tensors, storages)
    payload = +pickle_integer(Checkpoint::MAGIC_NUMBER)
    payload << pickle_integer(Checkpoint::PROTOCOL_VERSION)
    payload << proto << "}." # sys_info; absent little_endian means historical little-endian default
    payload << state_dict_pickle(tensors)
    payload << proto << "](" << storages.keys.map { |key| unicode(key) }.join << "e."
    storages.each do |key, bytes|
      payload << [tensors.values.find { |item| item[:key] == key }.fetch(:elements)].pack("Q<")
      payload << bytes
    end
    File.binwrite(path, payload)
  end

  def state_dict_pickle(tensors)
    payload = +proto
    payload << ordered_dict
    payload << "("
    tensors.each do |name, specification|
      payload << unicode(name)
      payload << global("torch._utils", "_rebuild_tensor_v2")
      payload << "("
      payload << tuple(
        [
          unicode("storage"),
          global("torch", storage_class(specification.fetch(:dtype))),
          unicode(specification.fetch(:key)),
          unicode("cpu"),
          integer(specification.fetch(:elements))
        ]
      )
      payload << "Q"
      payload << integer(specification.fetch(:offset))
      payload << tuple(specification.fetch(:shape).map { |value| integer(value) })
      payload << tuple(specification.fetch(:stride).map { |value| integer(value) })
      payload << "\x89".b
      payload << ordered_dict
      payload << "tR"
    end
    payload << "u."
    payload
  end

  def storage_class(dtype)
    {
      "F32" => "FloatStorage", "F16" => "HalfStorage",
      "BF16" => "BFloat16Storage", "I64" => "LongStorage"
    }.fetch(dtype)
  end

  def proto
    "\x80\x02".b
  end

  def ordered_dict
    global("collections", "OrderedDict") + ")R"
  end

  def global(mod, name)
    "c#{mod}\n#{name}\n".b
  end

  def unicode(value)
    value = value.to_s.b
    "X".b + [value.bytesize].pack("V") + value
  end

  def tuple(items)
    "(".b + items.join + "t"
  end

  def integer(value)
    return "K".b + [value].pack("C") if value.between?(0, 0xff)
    return "M".b + [value].pack("v") if value.between?(0, 0xffff)
    return "J".b + [value].pack("l<") if value.between?(-0x8000_0000, 0x7fff_ffff)

    long_integer(value)
  end

  def pickle_integer(value)
    proto + integer(value) + "."
  end

  def long_integer(value)
    bytes = +"".b
    until value.zero?
      bytes << (value & 0xff)
      value >>= 8
    end
    bytes << 0 if !bytes.empty? && bytes.getbyte(-1).anybits?(0x80)
    "\x8a".b + [bytes.bytesize].pack("C") + bytes
  end

  def write_zip(path, entries)
    local = +"".b
    central = +"".b
    entries.each do |name, payload|
      name = name.b
      payload = payload.b
      offset = local.bytesize
      crc = Zlib.crc32(payload)
      local << [
        Checkpoint::ZIP_LOCAL, 20, 0x800, 0, 0, 0, crc,
        payload.bytesize, payload.bytesize, name.bytesize, 0
      ].pack("VvvvvvVVVvv")
      local << name << payload
      central << [
        Checkpoint::ZIP_CENTRAL, 20, 20, 0x800, 0, 0, 0, crc,
        payload.bytesize, payload.bytesize, name.bytesize, 0, 0, 0, 0, 0, offset
      ].pack("VvvvvvvVVVvvvvvVV")
      central << name
    end
    eocd = [
      Checkpoint::ZIP_EOCD, 0, 0, entries.length, entries.length,
      central.bytesize, local.bytesize, 0
    ].pack("VvvvvVVv")
    File.binwrite(path, local + central + eocd)
  end

  def write_zip_at_offset(path, name, payload, local_offset:)
    name = name.b
    payload = payload.b
    crc = Zlib.crc32(payload)
    local = ("\0".b * local_offset) + [
      Checkpoint::ZIP_LOCAL, 20, 0x800, 0, 0, 0, crc,
      payload.bytesize, payload.bytesize, name.bytesize, 0
    ].pack("VvvvvvVVVvv") + name + payload
    central = [
      Checkpoint::ZIP_CENTRAL, 20, 20, 0x800, 0, 0, 0, crc,
      payload.bytesize, payload.bytesize, name.bytesize, 0, 0, 0, 0, 0, local_offset
    ].pack("VvvvvvvVVVvvvvvVV") + name
    eocd = [
      Checkpoint::ZIP_EOCD, 0, 0, 1, 1, central.bytesize, local.bytesize, 0
    ].pack("VvvvvVVv")
    File.binwrite(path, local + central + eocd)
  end

  def write_empty_classic_zip(path, count)
    local = +"".b
    central = +"".b
    count.times do |index|
      name = format("entry-%05d", index).b
      local_offset = local.bytesize
      local << [
        Checkpoint::ZIP_LOCAL, 20, 0x800, 0, 0, 0, 0,
        0, 0, name.bytesize, 0
      ].pack("VvvvvvVVVvv") << name
      central << [
        Checkpoint::ZIP_CENTRAL, 20, 20, 0x800, 0, 0, 0, 0,
        0, 0, name.bytesize, 0, 0, 0, 0, 0, local_offset
      ].pack("VvvvvvvVVVvvvvvVV") << name
    end
    eocd = [
      Checkpoint::ZIP_EOCD, 0, 0, count, count,
      central.bytesize, local.bytesize, 0
    ].pack("VvvvvVVv")
    File.binwrite(path, local + central + eocd)
  end

  def flip_byte(path, offset)
    File.open(path, "r+b") do |file|
      file.seek(offset)
      byte = file.read(1)
      raise "fixture byte is missing" unless byte

      file.seek(offset)
      file.write([byte.unpack1("C") ^ 0xff].pack("C"))
    end
  end

  def write_deflated_zip(path, name, payload, declared_size:)
    name = name.b
    payload = payload.b
    deflater = Zlib::Deflate.new(Zlib::DEFAULT_COMPRESSION, -Zlib::MAX_WBITS)
    compressed = deflater.deflate(payload, Zlib::FINISH)
    deflater.close
    crc = Zlib.crc32(payload)
    local = [
      Checkpoint::ZIP_LOCAL, 20, 0x800, 8, 0, 0, crc,
      compressed.bytesize, declared_size, name.bytesize, 0
    ].pack("VvvvvvVVVvv") + name + compressed
    central = [
      Checkpoint::ZIP_CENTRAL, 20, 20, 0x800, 8, 0, 0, crc,
      compressed.bytesize, declared_size, name.bytesize, 0, 0, 0, 0, 0, 0
    ].pack("VvvvvvvVVVvvvvvVV") + name
    eocd = [
      Checkpoint::ZIP_EOCD, 0, 0, 1, 1, central.bytesize, local.bytesize, 0
    ].pack("VvvvvVVv")
    File.binwrite(path, local + central + eocd)
  end

  def write_zip64(path, name, payload)
    name = name.b
    payload = payload.b
    crc = Zlib.crc32(payload)
    local = [
      Checkpoint::ZIP_LOCAL, 45, 0x800, 0, 0, 0, crc,
      payload.bytesize, payload.bytesize, name.bytesize, 0
    ].pack("VvvvvvVVVvv") + name + payload
    zip64_payload = [payload.bytesize, payload.bytesize, 0].pack("Q<Q<Q<")
    extra = [0x0001, zip64_payload.bytesize].pack("vv") + zip64_payload
    central = [
      Checkpoint::ZIP_CENTRAL, 45, 45, 0x800, 0, 0, 0, crc,
      0xffff_ffff, 0xffff_ffff, name.bytesize, extra.bytesize, 0, 0, 0, 0, 0xffff_ffff
    ].pack("VvvvvvvVVVvvvvvVV") + name + extra
    zip64_offset = local.bytesize + central.bytesize
    zip64_end = [
      Checkpoint::ZIP64_EOCD, 44, 45, 45, 0, 0, 1, 1, central.bytesize, local.bytesize
    ].pack("VQ<vvVVQ<Q<Q<Q<")
    locator = [Checkpoint::ZIP64_LOCATOR, 0, zip64_offset, 1].pack("VVQ<V")
    eocd = [
      Checkpoint::ZIP_EOCD, 0, 0, 0xffff, 0xffff,
      0xffff_ffff, 0xffff_ffff, 0
    ].pack("VvvvvVVv")
    File.binwrite(path, local + central + zip64_end + locator + eocd)
  end
end
