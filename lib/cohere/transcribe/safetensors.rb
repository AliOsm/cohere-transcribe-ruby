# frozen_string_literal: true

require "fiddle"
require "json"
require "rbconfig"

module Cohere
  module Transcribe
    # Minimal, streaming safetensors support for converting Cohere ASR weights.
    # It deliberately does not materialize a multi-gigabyte checkpoint in Ruby.
    module Safetensors
      class Error < StandardError; end

      DTYPE_BYTES = {
        "BF16" => 2,
        "F16" => 2,
        "F32" => 4,
        "I64" => 8
      }.freeze
      FLOAT_DTYPES = %w[BF16 F16 F32].freeze
      HEADER_LIMIT = 64 * 1024 * 1024
      DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024

      Tensor = Data.define(:reader, :name, :dtype, :shape, :data_start, :nbytes) do
        def element_count
          shape.empty? ? 1 : shape.reduce(1, :*)
        end

        def floating_point?
          FLOAT_DTYPES.include?(dtype)
        end
      end

      # Reads and validates one safetensors file. Tensor payloads stay on disk.
      class Reader
        attr_reader :path, :tensors

        def initialize(path)
          @path = Pathname(path).expand_path
          @tensors = read_header.freeze
        end

        def names
          tensors.keys
        end

        def key?(name)
          tensors.key?(name)
        end

        def fetch(name)
          tensors.fetch(name)
        rescue KeyError
          raise Error, "Tensor #{name.inspect} is not present in #{path}"
        end

        def write_tensor(tensor, output, target_dtype:, converter: DTypeConverter.default,
                         chunk_bytes: DEFAULT_CHUNK_BYTES)
          raise ArgumentError, "Tensor #{tensor.name.inspect} belongs to a different safetensors reader" unless tensor.reader.equal?(self)

          target_dtype = target_dtype.to_s.upcase
          unless FLOAT_DTYPES.include?(tensor.dtype) && FLOAT_DTYPES.include?(target_dtype)
            raise Error, "Cannot convert #{tensor.name.inspect} from #{tensor.dtype} to #{target_dtype}"
          end
          raise ArgumentError, "chunk_bytes must be positive" unless chunk_bytes.positive?

          source_width = DTYPE_BYTES.fetch(tensor.dtype)
          elements_per_chunk = [chunk_bytes / source_width, 1].max
          bytes_per_chunk = elements_per_chunk * source_width
          remaining = tensor.nbytes
          written = 0

          File.open(path, "rb") do |source|
            source.seek(tensor.data_start)
            while remaining.positive?
              requested = [remaining, bytes_per_chunk].min
              chunk = source.read(requested)
              raise Error, "Unexpected end of #{path} while reading #{tensor.name.inspect}" if chunk.nil? || chunk.bytesize != requested

              converted = converter.convert(chunk, from: tensor.dtype, to: target_dtype)
              output.write(converted)
              written += converted.bytesize
              remaining -= requested
            end
          end

          expected = tensor.element_count * DTYPE_BYTES.fetch(target_dtype)
          return written if written == expected

          raise Error,
                "Converted tensor #{tensor.name.inspect} wrote #{written} bytes; expected #{expected}"
        end

        private

        def read_header
          size = path.size
          raise Error, "Safetensors file #{path} is shorter than its header prefix" if size < 8

          File.open(path, "rb") do |file|
            header_length = read_exact(file, 8, "header length").unpack1("Q<")
            if header_length.zero? || header_length > HEADER_LIMIT || header_length > size - 8
              raise Error, "Safetensors file #{path} has invalid header length #{header_length}"
            end

            raw_header = read_exact(file, header_length, "JSON header")
            parsed = JSON.parse(raw_header)
            raise Error, "Safetensors header in #{path} is not a JSON object" unless parsed.is_a?(Hash)

            parsed.delete("__metadata__")
            build_tensors(parsed, data_offset: 8 + header_length, file_size: size)
          end
        rescue Errno::ENOENT
          raise Error, "Safetensors file #{path} does not exist"
        rescue Errno::EACCES => e
          raise Error, "Cannot read safetensors file #{path}: #{e.message}"
        rescue JSON::ParserError, EncodingError => e
          raise Error, "Invalid safetensors JSON header in #{path}: #{e.message}"
        end

        def build_tensors(header, data_offset:, file_size:)
          data_size = file_size - data_offset
          ranges = []
          result = {}

          header.each do |name, info|
            validate_tensor_name!(name)
            raise Error, "Safetensors entry #{name.inspect} in #{path} is not an object" unless info.is_a?(Hash)

            dtype = info["dtype"]
            width = DTYPE_BYTES[dtype]
            raise Error, "Tensor #{name.inspect} in #{path} uses unsupported dtype #{dtype.inspect}" unless width

            shape = info["shape"]
            unless shape.is_a?(Array) && shape.all? { |dimension| dimension.is_a?(Integer) && dimension >= 0 }
              raise Error, "Tensor #{name.inspect} in #{path} has an invalid shape"
            end

            offsets = info["data_offsets"]
            unless offsets.is_a?(Array) && offsets.length == 2 &&
                   offsets.all? { |offset| offset.is_a?(Integer) && offset >= 0 }
              raise Error, "Tensor #{name.inspect} in #{path} has invalid data offsets"
            end

            first, last = offsets
            raise Error, "Tensor #{name.inspect} in #{path} points outside the file" unless last.between?(first, data_size)

            count = shape.empty? ? 1 : shape.reduce(1, :*)
            expected = count * width
            actual = last - first
            if actual != expected
              raise Error,
                    "Tensor #{name.inspect} in #{path} has #{actual} data bytes; expected #{expected}"
            end

            ranges << [first, last, name] if actual.positive?
            result[name] = Tensor.new(
              reader: self,
              name: name.freeze,
              dtype: dtype.freeze,
              shape: shape.map(&:to_i).freeze,
              data_start: data_offset + first,
              nbytes: actual
            )
          end

          validate_non_overlapping!(ranges)
          result
        end

        def validate_tensor_name!(name)
          return if name.is_a?(String) && !name.empty?

          raise Error, "Safetensors file #{path} contains an invalid tensor name"
        end

        def validate_non_overlapping!(ranges)
          ranges.sort_by!(&:first)
          ranges.each_cons(2) do |left, right|
            next if left[1] <= right[0]

            raise Error,
                  "Tensors #{left[2].inspect} and #{right[2].inspect} overlap in #{path}"
          end
        end

        def read_exact(io, length, description)
          value = io.read(length)
          return value if value && value.bytesize == length

          raise Error, "Safetensors file #{path} ended while reading its #{description}"
        end
      end

      # Presents one unsharded file or a model.safetensors.index.json shard set
      # through the same tensor-name lookup API.
      class TensorSet
        attr_reader :directory, :readers, :tensors

        def self.from_directory(directory)
          directory = Pathname(directory).expand_path
          single = directory.join("model.safetensors")
          return new(directory, [Reader.new(single)]) if single.file?

          index = directory.join("model.safetensors.index.json")
          raise Error, "#{directory} does not contain model.safetensors or its index" unless index.file?

          from_index(directory, index)
        end

        def self.from_index(directory, index_path)
          payload = JSON.parse(index_path.read(encoding: "UTF-8"))
          weight_map = payload["weight_map"]
          unless weight_map.is_a?(Hash) && !weight_map.empty? &&
                 weight_map.all? { |name, file| name.is_a?(String) && file.is_a?(String) }
            raise Error, "Safetensors index #{index_path} has an invalid weight_map"
          end

          shard_names = weight_map.values.uniq
          shard_names.each do |name|
            candidate = Pathname(name)
            if name.empty? || candidate.absolute? || candidate.each_filename.any? { |part| part == ".." }
              raise Error, "Safetensors index #{index_path} contains an invalid shard path #{name.inspect}"
            end
          end
          readers = shard_names.to_h do |name|
            path = directory.join(name)
            raise Error, "Safetensors shard #{path} is missing" unless path.file?

            [name, Reader.new(path)]
          end

          mapped = weight_map.to_h do |name, shard|
            reader = readers.fetch(shard)
            raise Error, "Safetensors index maps missing tensor #{name.inspect} to #{shard}" unless reader.key?(name)

            [name, reader.fetch(name)]
          end
          readers.each do |shard, reader|
            indexed_names = weight_map.filter_map { |name, filename| name if filename == shard }
            unindexed = reader.names - indexed_names
            next if unindexed.empty?

            raise Error,
                  "Safetensors shard #{reader.path} contains tensors absent from its index: " \
                  "#{unindexed.first(8).map(&:inspect).join(", ")}"
          end
          new(directory, readers.values, tensors: mapped)
        rescue JSON::ParserError, EncodingError => e
          raise Error, "Invalid safetensors index #{index_path}: #{e.message}"
        rescue Errno::EACCES, Errno::ENOENT => e
          raise Error, "Cannot read safetensors index #{index_path}: #{e.message}"
        end

        def initialize(directory, readers, tensors: nil)
          @directory = Pathname(directory).expand_path
          @readers = readers.freeze
          @tensors = (tensors || readers.flat_map { |reader| reader.tensors.to_a }.to_h).freeze
        end

        def names
          tensors.keys
        end

        def key?(name)
          tensors.key?(name)
        end

        def fetch(name)
          tensors.fetch(name)
        rescue KeyError
          raise Error, "Tensor #{name.inspect} is not present in #{directory}"
        end

        def fetch_any(names)
          name = names.find { |candidate| tensors.key?(candidate) }
          return tensors.fetch(name) if name

          raise Error, "None of the tensor aliases are present: #{names.map(&:inspect).join(", ")}"
        end
      end

      # Portable Ruby dtype conversion. Conversion is chunked by Reader, so
      # memory use remains bounded even when this fallback is used.
      class DTypeConverter
        class << self
          def default
            return @default if @default

            @default = NativeDTypeConverter.auto
            @default || (@portable_fallback ||= new)
          end

          attr_writer :default
        end

        def convert(bytes, from:, to:)
          from = from.to_s.upcase
          to = to.to_s.upcase
          validate_input!(bytes, from)
          return bytes.dup if from == to

          case [from, to]
          when %w[BF16 F32] then bf16_to_f32(bytes)
          when %w[BF16 F16] then bf16_to_f16(bytes)
          when %w[F32 F16] then f32_to_f16(bytes)
          when %w[F16 F32] then f16_to_f32(bytes)
          when %w[F32 BF16] then f32_to_bf16(bytes)
          when %w[F16 BF16] then f32_to_bf16(f16_to_f32(bytes))
          else
            raise Error, "Unsupported floating-point conversion #{from} -> #{to}"
          end
        end

        private

        def validate_input!(bytes, dtype)
          width = DTYPE_BYTES[dtype]
          raise Error, "Unsupported source dtype #{dtype.inspect}" unless width
          return if (bytes.bytesize % width).zero?

          raise Error, "#{dtype} input contains a partial element"
        end

        def bf16_to_f32(bytes)
          bytes.unpack("S<*").map! { |bits| bits << 16 }.pack("L<*")
        end

        def bf16_to_f16(bytes)
          bytes.unpack("S<*").map! { |bits| bf16_bits_to_f16(bits) }.pack("S<*")
        end

        def f32_to_f16(bytes)
          bytes.unpack("L<*").map! { |bits| f32_bits_to_f16(bits) }.pack("S<*")
        end

        def f32_to_bf16(bytes)
          bytes.unpack("L<*").map! do |bits|
            if (bits & 0x7fff_ffff) > 0x7f80_0000
              (bits >> 16) | 0x40
            else
              (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
            end
          end.pack("S<*")
        end

        def f16_to_f32(bytes)
          bytes.unpack("S<*").map! { |bits| f16_bits_to_f32(bits) }.pack("L<*")
        end

        def bf16_bits_to_f16(bits)
          sign = (bits & 0x8000)
          exponent = (bits >> 7) & 0xff
          mantissa = bits & 0x7f
          return sign if exponent.zero?
          return sign | 0x7c00 | nan_payload(mantissa << 3) if exponent == 0xff && !mantissa.zero?
          return sign | 0x7c00 if exponent == 0xff

          unbiased = exponent - 127
          return sign | 0x7c00 if unbiased > 15
          return sign | ((unbiased + 15) << 10) | (mantissa << 3) if unbiased >= -14
          return sign if unbiased < -25

          significand = 0x80 | mantissa
          shift = -(unbiased + 17)
          rounded = shift.positive? ? round_right(significand, shift) : significand << -shift
          sign | rounded
        end

        def f32_bits_to_f16(bits)
          sign = (bits >> 16) & 0x8000
          exponent = (bits >> 23) & 0xff
          mantissa = bits & 0x7fffff
          if exponent == 0xff
            return sign | 0x7c00 if mantissa.zero?

            return sign | 0x7c00 | nan_payload(mantissa >> 13)
          end
          return sign if exponent.zero?

          if exponent < 113
            return sign if exponent < 102

            rounded = round_right(0x800000 | mantissa, 126 - exponent)
            return sign | rounded
          end
          return sign | 0x7c00 if exponent > 142

          half_exponent = exponent - 112
          half_mantissa = round_right(mantissa, 13)
          if half_mantissa == 0x400
            half_exponent += 1
            half_mantissa = 0
          end
          return sign | 0x7c00 if half_exponent >= 31

          sign | (half_exponent << 10) | half_mantissa
        end

        def f16_bits_to_f32(bits)
          sign = (bits & 0x8000) << 16
          exponent = (bits >> 10) & 0x1f
          mantissa = bits & 0x3ff
          if exponent.zero?
            return sign if mantissa.zero?

            unbiased = -14
            until mantissa.anybits?(0x400)
              mantissa <<= 1
              unbiased -= 1
            end
            mantissa &= 0x3ff
            return sign | ((unbiased + 127) << 23) | (mantissa << 13)
          end
          return sign | 0x7f800000 | (mantissa << 13) if exponent == 0x1f

          sign | ((exponent - 15 + 127) << 23) | (mantissa << 13)
        end

        def nan_payload(payload)
          payload.zero? ? 1 : payload
        end

        def round_right(value, shift)
          base = value >> shift
          remainder = value & ((1 << shift) - 1)
          halfway = 1 << (shift - 1)
          increment = remainder > halfway || (remainder == halfway && base.odd?)
          increment ? base + 1 : base
        end
      end

      # Uses GGML's vector conversion routines when its base library is
      # available. The Ruby fallback above remains authoritative and testable.
      class NativeDTypeConverter < DTypeConverter
        LIBRARY_NAMES = %w[libggml-base.so libggml-base.dylib ggml-base.dll].freeze
        LIBRARY_PATTERNS = case RbConfig::CONFIG["host_os"]
                           when /darwin/
                             ["libggml-base.dylib", "libggml-base*.dylib"]
                           when /mswin|mingw|cygwin/
                             ["ggml-base.dll", "libggml-base.dll"]
                           else
                             ["libggml-base.so", "libggml-base.so.*"]
                           end.freeze
        FUNCTIONS = {
          bf16_to_f32: "ggml_bf16_to_fp32_row",
          f16_to_f32: "ggml_fp16_to_fp32_row",
          f32_to_f16: "ggml_fp32_to_fp16_row",
          f32_to_bf16: "ggml_fp32_to_bf16_row"
        }.freeze
        PRIVATE_FUNCTIONS = {
          bf16_to_f32: "crispasr_bf16_to_fp32_row",
          f16_to_f32: "crispasr_fp16_to_fp32_row",
          f32_to_f16: "crispasr_fp32_to_fp16_row",
          f32_to_bf16: "crispasr_fp32_to_bf16_row"
        }.freeze

        class << self
          def auto(library = ENV.fetch("COHERE_TRANSCRIBE_GGML_LIBRARY", nil))
            candidate_libraries(library).each do |candidate|
              return new(candidate)
            rescue Fiddle::DLError, LoadError
              next
            end
            nil
          end

          # Ordered candidates used by the lazy default converter. This keeps
          # dense conversion fast when GGML is packaged beside libcrispasr,
          # while retaining system-library and portable Ruby fallbacks.
          def candidate_libraries(library = ENV.fetch("COHERE_TRANSCRIBE_GGML_LIBRARY", nil))
            explicit = library.to_s.empty? ? [] : [library]
            native = native_library_candidates.select { |candidate| path_like?(candidate) && File.file?(candidate) }
            [*explicit, nil, *native, *discovered_library_paths, *LIBRARY_NAMES].uniq.freeze
          end

          private

          def discovered_library_paths
            candidate_directories.flat_map do |directory|
              LIBRARY_PATTERNS.flat_map do |pattern|
                Dir.glob(File.join(directory, pattern)).sort
              end
            end.select { |path| File.file?(path) }.uniq
          end

          def candidate_directories
            packaged = File.join(__dir__, "native")
            directories = [
              packaged,
              *Dir.glob(File.join(packaged, "*")).select { |path| File.directory?(path) }
            ]
            native_library_candidates.each do |candidate|
              next unless path_like?(candidate)

              directory = File.directory?(candidate) ? candidate : File.dirname(candidate)
              directories.push(
                directory, File.join(directory, "ggml", "src"), File.expand_path("../ggml/src", directory)
              )
            end
            directories.map { |directory| File.expand_path(directory) }.uniq
          end

          def native_library_candidates
            candidates = [ENV.fetch("COHERE_TRANSCRIBE_NATIVE_LIBRARY", nil)]
            native_library = defined?(Cohere::Transcribe::ASR::NativeLibrary) &&
                             Cohere::Transcribe::ASR::NativeLibrary
            candidates.concat(native_library.candidate_paths) if native_library
            candidates.compact
          end

          def path_like?(candidate)
            !candidate.to_s.empty? &&
              (Pathname(candidate).absolute? || candidate.include?(File::SEPARATOR) ||
               (File::ALT_SEPARATOR && candidate.include?(File::ALT_SEPARATOR)))
          end
        end

        def initialize(library = nil)
          super()
          @handle = library ? Fiddle::Handle.new(library) : Fiddle::Handle::DEFAULT
          @functions = FUNCTIONS.to_h do |name, symbol|
            address = begin
              @handle[symbol]
            rescue Fiddle::DLError
              @handle[PRIVATE_FUNCTIONS.fetch(name)]
            end
            function = Fiddle::Function.new(
              address,
              [Fiddle::TYPE_VOIDP, Fiddle::TYPE_VOIDP, Fiddle::TYPE_LONG_LONG],
              Fiddle::TYPE_VOID
            )
            [name, function]
          end.freeze
        end

        def convert(bytes, from:, to:)
          from = from.to_s.upcase
          to = to.to_s.upcase
          validate_native_input!(bytes, from)
          return bytes.dup if from == to

          case [from, to]
          when %w[BF16 F32] then native_call(:bf16_to_f32, bytes, 4)
          when %w[F16 F32] then native_call(:f16_to_f32, bytes, 4)
          when %w[F32 F16] then native_call(:f32_to_f16, bytes, 2)
          when %w[F32 BF16] then native_call(:f32_to_bf16, bytes, 2)
          when %w[BF16 F16]
            intermediate = native_call(:bf16_to_f32, bytes, 4)
            native_call(:f32_to_f16, intermediate, 2)
          when %w[F16 BF16]
            intermediate = native_call(:f16_to_f32, bytes, 4)
            native_call(:f32_to_bf16, intermediate, 2)
          else
            super
          end
        end

        private

        def validate_native_input!(bytes, dtype)
          width = DTYPE_BYTES[dtype]
          raise Error, "Unsupported source dtype #{dtype.inspect}" unless width
          return if (bytes.bytesize % width).zero?

          raise Error, "#{dtype} input contains a partial element"
        end

        def native_call(name, source, output_width)
          input_width = %i[f32_to_f16 f32_to_bf16].include?(name) ? 4 : 2
          count = source.bytesize / input_width
          destination = "\0".b * (count * output_width)
          @functions.fetch(name).call(Fiddle::Pointer[source], Fiddle::Pointer[destination], count)
          destination
        end
      end
    end
  end
end
