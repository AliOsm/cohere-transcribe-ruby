# frozen_string_literal: true

require "json"
require "stringio"

module ConverterTestSupport
  GGUF_VALUE_TYPES = {
    0 => :uint8,
    1 => :int8,
    2 => :uint16,
    3 => :int16,
    4 => :uint32,
    5 => :int32,
    6 => :float32,
    7 => :bool,
    8 => :string,
    9 => :array,
    10 => :uint64,
    11 => :int64,
    12 => :float64
  }.freeze

  def write_safetensors(path, tensors)
    offset = 0
    payload = +"".b
    header = tensors.to_h do |name, tensor|
      bytes = tensor.fetch(:bytes).b
      first = offset
      offset += bytes.bytesize
      payload << bytes
      [
        name,
        {
          "dtype" => tensor.fetch(:dtype),
          "shape" => tensor.fetch(:shape),
          "data_offsets" => [first, offset]
        }
      ]
    end

    json = JSON.generate(header)
    json << (" " * ((8 - (json.bytesize % 8)) % 8))
    File.binwrite(path, [json.bytesize].pack("Q<") + json.b + payload)
  end

  def read_gguf(path_or_bytes)
    bytes = if path_or_bytes.respond_to?(:to_path)
              File.binread(path_or_bytes.to_path)
            else
              path_or_bytes.b
            end
    input = StringIO.new(bytes)
    raise "not a GGUF document" unless read_exact(input, 4) == "GGUF"

    version = read_exact(input, 4).unpack1("L<")
    tensor_count, metadata_count = read_exact(input, 16).unpack("Q<Q<")
    metadata = metadata_count.times.to_h do
      key = read_gguf_string(input)
      type = GGUF_VALUE_TYPES.fetch(read_exact(input, 4).unpack1("L<"))
      [key, read_gguf_value(input, type)]
    end
    tensors = tensor_count.times.to_h do
      name = read_gguf_string(input)
      rank = read_exact(input, 4).unpack1("L<")
      dimensions = read_exact(input, rank * 8).unpack("Q<*")
      dtype = read_exact(input, 4).unpack1("L<")
      offset = read_exact(input, 8).unpack1("Q<")
      [name, { dimensions: dimensions, dtype: dtype, offset: offset }]
    end
    alignment = metadata.fetch("general.alignment", 32)
    data_start = align(input.pos, alignment)

    {
      version: version,
      metadata: metadata,
      tensors: tensors,
      data_start: data_start,
      bytes: bytes
    }
  end

  def gguf_tensor_bytes(document, name, length)
    tensor = document.fetch(:tensors).fetch(name)
    start = document.fetch(:data_start) + tensor.fetch(:offset)
    document.fetch(:bytes).byteslice(start, length)
  end

  private

  def read_gguf_value(input, type)
    case type
    when :uint8 then read_exact(input, 1).unpack1("C")
    when :int8 then read_exact(input, 1).unpack1("c")
    when :uint16 then read_exact(input, 2).unpack1("S<")
    when :int16 then read_exact(input, 2).unpack1("s<")
    when :uint32 then read_exact(input, 4).unpack1("L<")
    when :int32 then read_exact(input, 4).unpack1("l<")
    when :float32 then read_exact(input, 4).unpack1("e")
    when :bool then read_exact(input, 1).unpack1("C") == 1
    when :string then read_gguf_string(input)
    when :uint64 then read_exact(input, 8).unpack1("Q<")
    when :int64 then read_exact(input, 8).unpack1("q<")
    when :float64 then read_exact(input, 8).unpack1("E")
    when :array
      element_type = GGUF_VALUE_TYPES.fetch(read_exact(input, 4).unpack1("L<"))
      length = read_exact(input, 8).unpack1("Q<")
      Array.new(length) { read_gguf_value(input, element_type) }
    else
      raise "unsupported GGUF metadata type #{type.inspect}"
    end
  end

  def read_gguf_string(input)
    length = read_exact(input, 8).unpack1("Q<")
    read_exact(input, length).force_encoding(Encoding::UTF_8)
  end

  def read_exact(input, length)
    value = input.read(length)
    raise "unexpected end of document" unless value&.bytesize == length

    value
  end

  def align(value, alignment)
    ((value + alignment - 1) / alignment) * alignment
  end
end
