# frozen_string_literal: true

require "tempfile"

module Cohere
  module Transcribe
    # Small GGUF v3 writer tailored to streaming model conversion. Tensor data
    # is supplied by callbacks, so the writer never retains model weights.
    module GGUF
      class Error < StandardError; end

      MAGIC = "GGUF".b.freeze
      VERSION = 3
      DEFAULT_ALIGNMENT = 32

      VALUE_TYPES = {
        uint8: 0,
        int8: 1,
        uint16: 2,
        int16: 3,
        uint32: 4,
        int32: 5,
        float32: 6,
        bool: 7,
        string: 8,
        array: 9,
        uint64: 10,
        int64: 11,
        float64: 12
      }.freeze

      TENSOR_TYPES = {
        f32: 0,
        f16: 1,
        bf16: 30
      }.freeze
      TENSOR_WIDTHS = {
        f32: 4,
        f16: 2,
        bf16: 2
      }.freeze

      Metadata = Data.define(:type, :value, :element_type)
      Tensor = Data.define(:name, :shape, :dtype, :nbytes, :write_data)

      # Produces a GGUF v3 document from metadata and streaming tensor writers.
      class Writer
        attr_reader :metadata, :tensors, :alignment

        def initialize(alignment: DEFAULT_ALIGNMENT)
          unless alignment.is_a?(Integer) && alignment.positive? && alignment.nobits?(alignment - 1)
            raise ArgumentError, "GGUF alignment must be a positive power of two"
          end

          @alignment = alignment
          @metadata = {}
          @tensors = []
          add_uint32("general.alignment", alignment) unless alignment == DEFAULT_ALIGNMENT
        end

        def add_string(key, value)
          add_metadata(key, :string, String(value))
        end

        def add_uint32(key, value)
          add_metadata(key, :uint32, unsigned_integer(value, bits: 32))
        end

        def add_uint64(key, value)
          add_metadata(key, :uint64, unsigned_integer(value, bits: 64))
        end

        def add_float32(key, value)
          add_metadata(key, :float32, Float(value))
        end

        def add_bool(key, value)
          raise ArgumentError, "GGUF bool metadata must be true or false" unless [true, false].include?(value)

          add_metadata(key, :bool, value)
        end

        def add_string_array(key, values)
          values = values.map { |value| String(value) }.freeze
          raise ArgumentError, "GGUF arrays must not be empty" if values.empty?

          add_metadata(key, :array, values, element_type: :string)
        end

        def add_tensor(name, shape:, dtype:, &write_data)
          name = String(name)
          raise ArgumentError, "GGUF tensor name must not be empty" if name.empty?
          raise ArgumentError, "GGUF tensor #{name.inspect} has no data writer" unless write_data
          raise ArgumentError, "Duplicate GGUF tensor name #{name.inspect}" if tensors.any? { |tensor| tensor.name == name }

          shape = Array(shape)
          unless !shape.empty? && shape.all? { |dimension| dimension.is_a?(Integer) && dimension.positive? }
            raise ArgumentError, "GGUF tensor #{name.inspect} has an invalid shape"
          end

          dtype = dtype.to_sym
          width = TENSOR_WIDTHS[dtype]
          raise ArgumentError, "Unsupported GGUF tensor dtype #{dtype.inspect}" unless width

          elements = shape.reduce(1, :*)
          tensors << Tensor.new(
            name: name.freeze,
            shape: shape.freeze,
            dtype: dtype,
            nbytes: elements * width,
            write_data: write_data
          )
          self
        end

        def write(path, overwrite: false, fsync: true)
          output = Pathname(path).expand_path
          raise Error, "Output directory #{output.dirname} does not exist" unless output.dirname.directory?
          raise Error, "Output file #{output} already exists" if output.exist? && !overwrite

          temporary = Tempfile.new([".#{output.basename}", ".tmp"], output.dirname.to_s)
          temporary.binmode
          begin
            write_to(temporary)
            temporary.flush
            temporary.fsync if fsync
            temporary.close
            raise Error, "Output file #{output} already exists" if output.exist? && !overwrite

            if overwrite
              File.rename(temporary.path, output)
            else
              # A same-directory hard link is an atomic no-replace publish on
              # POSIX filesystems. It closes the existence-check race where a
              # concurrent writer could otherwise be silently overwritten by
              # rename(2).
              begin
                File.link(temporary.path, output)
              rescue Errno::EEXIST
                raise Error, "Output file #{output} already exists"
              end
              temporary.unlink
            end
            sync_directory(output.dirname) if fsync
          ensure
            temporary.close unless temporary.closed?
            temporary.unlink
          end
          output
        rescue Errno::EACCES, Errno::ENOSPC, Errno::EROFS => e
          raise Error, "Cannot write GGUF file #{output}: #{e.message}"
        end

        def write_to(io)
          validate_document!
          io.write(MAGIC)
          io.write([VERSION].pack("L<"))
          io.write([tensors.length, metadata.length].pack("Q<Q<"))
          metadata.each do |key, entry|
            write_string(io, key)
            io.write([VALUE_TYPES.fetch(entry.type)].pack("L<"))
            write_value(io, entry.type, entry.value, element_type: entry.element_type)
          end

          offset = 0
          tensors.each do |tensor|
            write_string(io, tensor.name)
            io.write([tensor.shape.length].pack("L<"))
            io.write(tensor.shape.reverse.pack("Q<*"))
            io.write([TENSOR_TYPES.fetch(tensor.dtype)].pack("L<"))
            io.write([offset].pack("Q<"))
            offset += padded(tensor.nbytes)
          end

          write_padding(io, io.pos)
          tensors.each do |tensor|
            before = io.pos
            tensor.write_data.call(io)
            actual = io.pos - before
            if actual != tensor.nbytes
              raise Error,
                    "Tensor #{tensor.name.inspect} wrote #{actual} bytes; expected #{tensor.nbytes}"
            end
            write_padding(io, tensor.nbytes)
          end
          io
        end

        private

        def sync_directory(path)
          File.open(path, File::RDONLY, &:fsync)
        rescue Errno::EINVAL, Errno::ENOTSUP, Errno::EISDIR
          nil
        end

        def unsigned_integer(value, bits:)
          integer = begin
            Integer(value)
          rescue ArgumentError, TypeError
            raise ArgumentError, "GGUF uint#{bits} metadata must be an integer"
          end
          return integer if integer.between?(0, (1 << bits) - 1)

          raise ArgumentError, "GGUF uint#{bits} metadata is outside its representable range"
        end

        def add_metadata(key, type, value, element_type: nil)
          key = String(key)
          raise ArgumentError, "GGUF metadata key must not be empty" if key.empty?
          raise ArgumentError, "Duplicate GGUF metadata key #{key.inspect}" if metadata.key?(key)

          metadata[key.freeze] = Metadata.new(type: type, value: value, element_type: element_type)
          self
        end

        def validate_document!
          raise Error, "GGUF document contains no tensors" if tensors.empty?
          return if metadata.key?("general.architecture")

          raise Error, "GGUF document is missing general.architecture"
        end

        def write_value(io, type, value, element_type: nil)
          case type
          when :uint8 then io.write([value].pack("C"))
          when :int8 then io.write([value].pack("c"))
          when :uint16 then io.write([value].pack("S<"))
          when :int16 then io.write([value].pack("s<"))
          when :uint32 then io.write([value].pack("L<"))
          when :int32 then io.write([value].pack("l<"))
          when :float32 then io.write([value].pack("e"))
          when :bool then io.write([value ? 1 : 0].pack("C"))
          when :string then write_string(io, value)
          when :uint64 then io.write([value].pack("Q<"))
          when :int64 then io.write([value].pack("q<"))
          when :float64 then io.write([value].pack("E"))
          when :array
            io.write([VALUE_TYPES.fetch(element_type)].pack("L<"))
            io.write([value.length].pack("Q<"))
            value.each { |item| write_value(io, element_type, item) }
          else
            raise Error, "Unsupported GGUF metadata type #{type.inspect}"
          end
        end

        def write_string(io, value)
          encoded = value.encode(Encoding::UTF_8)
          raise Error, "GGUF strings must contain valid UTF-8" unless encoded.valid_encoding?

          io.write([encoded.bytesize].pack("Q<"))
          io.write(encoded.b)
        rescue Encoding::UndefinedConversionError, Encoding::InvalidByteSequenceError => e
          raise Error, "Cannot encode GGUF string as UTF-8: #{e.message}"
        end

        def padded(bytes)
          ((bytes + alignment - 1) / alignment) * alignment
        end

        def write_padding(io, bytes)
          count = padded(bytes) - bytes
          io.write("\0".b * count) if count.positive?
        end
      end
    end
  end
end
