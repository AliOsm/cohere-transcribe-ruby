# frozen_string_literal: true

require "json"
require "stringio"
require "zlib"

require_relative "safetensors"

module Cohere
  module Transcribe
    # A non-executing reader for dense PyTorch state dictionaries. It accepts
    # the ZIP serialization used by current torch.save and the immediately
    # preceding pickle-plus-storage stream. Pickle GLOBAL/REDUCE operations are
    # interpreted through a small allowlist; no Python classes or code run.
    module PyTorchCheckpoint
      class Error < StandardError; end

      MAGIC_NUMBER = 119_547_037_146_038_801_333_356
      PROTOCOL_VERSION = 1001
      PICKLE_LIMIT = 64 * 1024 * 1024
      PICKLE_OPCODE_LIMIT = 1_000_000
      PICKLE_LONG_BYTES_LIMIT = 16
      ZIP_TAIL_LIMIT = 65_557
      ZIP_ENTRY_LIMIT = 1_000_000
      ZIP_CENTRAL_LIMIT = 256 * 1024 * 1024
      ZIP_COMPRESSED_OVERHEAD = 1024 * 1024
      ZIP_INFLATE_CHUNK = 16 * 1024
      STORAGE_CRC_CHUNK_BYTES = 4 * 1024 * 1024
      ZIP_EOCD = 0x0605_4b50
      ZIP64_EOCD = 0x0606_4b50
      ZIP64_LOCATOR = 0x0706_4b50
      ZIP_CENTRAL = 0x0201_4b50
      ZIP_LOCAL = 0x0403_4b50
      ZIP64_UINT16_MARKER = 0xffff
      ZIP64_UINT32_MARKER = 0xffff_ffff

      STORAGE_DTYPES = {
        "FloatStorage" => ["F32", 4],
        "HalfStorage" => ["F16", 2],
        "BFloat16Storage" => ["BF16", 2],
        "LongStorage" => ["I64", 8]
      }.freeze
      FLOAT_DTYPES = %w[BF16 F16 F32].freeze

      Global = Data.define(:module_name, :name)
      StorageRef = Data.define(:dtype, :key, :elements, :base_offset)
      StoragePayload = Data.define(:data_offset, :size, :crc32)
      TensorSpec = Data.define(:storage, :storage_offset, :shape, :stride)
      Entry = Data.define(
        :name, :compression_method, :flags, :crc32, :compressed_size, :size,
        :local_header_offset, :data_offset
      )
      Tensor = Data.define(
        :reader, :name, :dtype, :shape, :storage_key, :data_start, :nbytes,
        :storage_data_start, :storage_elements, :storage_offset, :stride
      ) do
        def element_count
          shape.empty? ? 1 : shape.reduce(1, :*)
        end

        def floating_point?
          FLOAT_DTYPES.include?(dtype)
        end
      end

      # Minimal ZIP/ZIP64 central-directory reader. Torch stores tensor records
      # without compression, allowing bounded direct reads from multi-GB files.
      class ZipArchive
        attr_reader :path, :entries

        def initialize(path)
          @path = Pathname(path).expand_path
          @entries = read_entries.freeze
        end

        def fetch(name)
          entries.fetch(name)
        rescue KeyError
          raise Error, "PyTorch archive #{path} has no entry #{name.inspect}"
        end

        def read(name, limit: PICKLE_LIMIT)
          entry = fetch(name)
          raise Error, "PyTorch archive entry #{name.inspect} exceeds the #{limit}-byte size limit" if entry.size > limit
          if entry.compressed_size > limit + ZIP_COMPRESSED_OVERHEAD
            raise Error,
                  "PyTorch archive entry #{name.inspect} has an oversized compressed representation"
          end
          if entry.compression_method.zero? && entry.compressed_size != entry.size
            raise Error, "Stored PyTorch archive entry #{name.inspect} has inconsistent sizes"
          end

          compressed = read_range(entry.data_offset, entry.compressed_size)
          payload = case entry.compression_method
                    when 0
                      compressed
                    when 8
                      inflate_bounded(compressed, entry, limit)
                    else
                      raise Error,
                            "PyTorch archive entry #{name.inspect} uses unsupported ZIP method " \
                            "#{entry.compression_method}"
                    end
          if payload.bytesize != entry.size
            raise Error,
                  "PyTorch archive entry #{name.inspect} decoded to #{payload.bytesize} bytes; expected #{entry.size}"
          end
          raise Error, "PyTorch archive entry #{name.inspect} failed its CRC check" unless Zlib.crc32(payload) == entry.crc32

          payload
        rescue Zlib::Error => e
          raise Error, "Cannot inflate PyTorch archive entry #{name.inspect}: #{e.message}"
        end

        private

        def inflate_bounded(compressed, entry, limit)
          inflater = Zlib::Inflate.new(-Zlib::MAX_WBITS)
          payload = +"".b
          offset = 0
          chunk_size = [[entry.size / 1024, 1].max, ZIP_INFLATE_CHUNK].min
          begin
            while offset < compressed.bytesize
              chunk = compressed.byteslice(offset, chunk_size)
              payload << inflater.inflate(chunk)
              if payload.bytesize > entry.size || payload.bytesize > limit
                raise Error,
                      "PyTorch archive entry #{entry.name.inspect} expands beyond its declared size"
              end
              offset += chunk.bytesize
            end
            payload << inflater.finish
            if payload.bytesize > entry.size || payload.bytesize > limit
              raise Error,
                    "PyTorch archive entry #{entry.name.inspect} expands beyond its declared size"
            end
            payload
          ensure
            # Ruby 4 warns when an unfinished zstream is closed and performs
            # this reset implicitly.  Expansion-limit and corrupt-stream
            # failures are expected rejection paths, so reset explicitly
            # before closing without masking the original checkpoint error.
            inflater.reset unless inflater.finished?
            inflater.close
          end
        end

        def read_entries
          size = path.size
          raise Error, "PyTorch archive #{path} is too short" if size < 22

          tail_length = [size, ZIP_TAIL_LIMIT].min
          tail_start = size - tail_length
          tail = read_range(tail_start, tail_length)
          signature = [ZIP_EOCD].pack("V")
          relative = nil
          cursor = tail.bytesize - 22
          while cursor >= 0
            candidate = tail.rindex(signature, cursor)
            break unless candidate

            comment_length = tail.byteslice(candidate + 20, 2)&.unpack1("v")
            if comment_length && candidate + 22 + comment_length == tail.bytesize
              relative = candidate
              break
            end
            cursor = candidate - 1
          end
          raise Error, "PyTorch archive #{path} has no end-of-central-directory record" unless relative

          eocd_offset = tail_start + relative
          eocd = tail.byteslice(relative, 22)
          raise Error, "PyTorch archive #{path} has a truncated end record" unless eocd&.bytesize == 22

          fields = eocd.unpack("VvvvvVVv")
          _signature, disk, central_disk, disk_entries, total_entries,
            central_size, central_offset, comment_length = fields
          raise Error, "PyTorch archive #{path} has a truncated ZIP comment" if relative + 22 + comment_length > tail.bytesize
          unless disk.zero? && central_disk.zero? && disk_entries == total_entries
            raise Error, "Multi-disk PyTorch ZIP archives are not supported"
          end

          zip64_marker = disk_entries == ZIP64_UINT16_MARKER ||
                         total_entries == ZIP64_UINT16_MARKER ||
                         central_size == ZIP64_UINT32_MARKER ||
                         central_offset == ZIP64_UINT32_MARKER
          total_entries, central_size, central_offset = zip64_directory(eocd_offset) if
            zip64_marker && zip64_locator_present?(eocd_offset)
          if total_entries > ZIP_ENTRY_LIMIT || central_size > ZIP_CENTRAL_LIMIT ||
             central_size > size || central_offset > size - central_size
            raise Error, "PyTorch archive #{path} has invalid central-directory bounds"
          end

          parse_central_directory(
            read_range(central_offset, central_size),
            expected_entries: total_entries,
            archive_size: size,
            data_limit: central_offset
          )
        rescue Errno::ENOENT
          raise Error, "PyTorch checkpoint #{path} does not exist"
        rescue Errno::EACCES => e
          raise Error, "Cannot read PyTorch checkpoint #{path}: #{e.message}"
        end

        def zip64_directory(eocd_offset)
          raise Error, "PyTorch archive #{path} has no ZIP64 locator" if eocd_offset < 20

          locator = read_range(eocd_offset - 20, 20)
          signature, disk, record_offset, disks = locator.unpack("VVQ<V")
          raise Error, "PyTorch archive #{path} has an invalid ZIP64 locator" unless signature == ZIP64_LOCATOR && disk.zero? && disks == 1

          record = read_range(record_offset, 56)
          signature, record_size, _made, _needed, disk, central_disk,
            disk_entries, total_entries, central_size, central_offset = record.unpack("VQ<vvVVQ<Q<Q<Q<")
          unless signature == ZIP64_EOCD && record_size >= 44 && disk.zero? && central_disk.zero? &&
                 disk_entries == total_entries && record_offset + 12 + record_size == eocd_offset - 20
            raise Error, "PyTorch archive #{path} has an invalid ZIP64 end record"
          end

          [total_entries, central_size, central_offset]
        end

        def zip64_locator_present?(eocd_offset)
          return false if eocd_offset < 20

          read_range(eocd_offset - 20, 4).unpack1("V") == ZIP64_LOCATOR
        end

        def parse_central_directory(bytes, expected_entries:, archive_size:, data_limit:)
          offset = 0
          result = {}
          expected_entries.times do
            fixed = bytes.byteslice(offset, 46)
            unless fixed&.bytesize == 46 && fixed.unpack1("V") == ZIP_CENTRAL
              raise Error, "PyTorch archive #{path} has a malformed central directory"
            end

            values = fixed.unpack("VvvvvvvVVVvvvvvVV")
            flags = values[3]
            method = values[4]
            crc32 = values[7]
            compressed_size = values[8]
            size = values[9]
            name_length = values[10]
            extra_length = values[11]
            comment_length = values[12]
            disk = values[13]
            local_offset = values[16]
            offset += 46

            variable_length = name_length + extra_length + comment_length
            variable = bytes.byteslice(offset, variable_length)
            unless variable&.bytesize == variable_length
              raise Error,
                    "PyTorch archive #{path} has a truncated central entry"
            end

            raw_name = variable.byteslice(0, name_length)
            extra = variable.byteslice(name_length, extra_length)
            offset += variable_length
            size, compressed_size, local_offset, disk = apply_zip64_extra(
              extra, size: size, compressed_size: compressed_size,
                     local_offset: local_offset, disk: disk
            )
            raise Error, "Multi-disk PyTorch ZIP archives are not supported" unless disk.zero?
            raise Error, "Encrypted PyTorch ZIP entries are not supported" if flags.anybits?(0x1)

            name = decode_name(raw_name, utf8: flags.anybits?(0x800))
            validate_entry_name!(name)
            raise Error, "PyTorch archive #{path} contains duplicate entry #{name.inspect}" if result.key?(name)

            data_offset = local_data_offset(
              local_offset,
              name,
              archive_size,
              central_flags: flags,
              central_method: method
            )
            if local_offset >= data_limit || compressed_size > archive_size ||
               data_offset > archive_size - compressed_size || data_offset + compressed_size > data_limit
              raise Error, "PyTorch archive entry #{name.inspect} points outside #{path}"
            end

            result[name] = Entry.new(
              name: name.freeze,
              compression_method: method,
              flags: flags,
              crc32: crc32,
              compressed_size: compressed_size,
              size: size,
              local_header_offset: local_offset,
              data_offset: data_offset
            )
          end
          raise Error, "PyTorch archive #{path} central-directory size is inconsistent" unless offset == bytes.bytesize

          result
        end

        def apply_zip64_extra(extra, size:, compressed_size:, local_offset:, disk:)
          values = nil
          cursor = 0
          while cursor < extra.bytesize
            raise Error, "PyTorch archive #{path} has a truncated ZIP extra field" if cursor + 4 > extra.bytesize

            tag, length = extra.byteslice(cursor, 4).unpack("vv")
            cursor += 4
            payload = extra.byteslice(cursor, length)
            raise Error, "PyTorch archive #{path} has a truncated ZIP extra payload" unless payload&.bytesize == length

            values = payload if tag == 0x0001
            cursor += length
          end

          required = [size, compressed_size, local_offset].count(ZIP64_UINT32_MARKER)
          required += 1 if disk == ZIP64_UINT16_MARKER
          return [size, compressed_size, local_offset, disk] if required.zero?
          raise Error, "PyTorch archive #{path} lacks required ZIP64 size data" unless values

          cursor = 0
          take_qword = lambda do
            raise Error, "PyTorch archive #{path} has truncated ZIP64 data" if cursor + 8 > values.bytesize

            value = values.byteslice(cursor, 8).unpack1("Q<")
            cursor += 8
            value
          end
          take_dword = lambda do
            raise Error, "PyTorch archive #{path} has truncated ZIP64 data" if cursor + 4 > values.bytesize

            value = values.byteslice(cursor, 4).unpack1("V")
            cursor += 4
            value
          end
          size = take_qword.call if size == ZIP64_UINT32_MARKER
          compressed_size = take_qword.call if compressed_size == ZIP64_UINT32_MARKER
          local_offset = take_qword.call if local_offset == ZIP64_UINT32_MARKER
          disk = take_dword.call if disk == ZIP64_UINT16_MARKER
          [size, compressed_size, local_offset, disk]
        end

        def local_data_offset(local_offset, central_name, archive_size, central_flags:, central_method:)
          raise Error, "PyTorch archive entry #{central_name.inspect} has an invalid local header" if local_offset > archive_size - 30

          fixed = read_range(local_offset, 30)
          signature, _needed, flags, method, _time, _date, _crc, _compressed, _size,
            name_length, extra_length = fixed.unpack("VvvvvvVVVvv")
          raise Error, "PyTorch archive entry #{central_name.inspect} has no local header" unless signature == ZIP_LOCAL

          local_name = read_range(local_offset + 30, name_length)
          raise Error, "PyTorch archive central/local names disagree for #{central_name.inspect}" unless local_name == central_name.b
          unless flags == central_flags && method == central_method
            raise Error,
                  "PyTorch archive central/local metadata disagree for #{central_name.inspect}"
          end

          local_offset + 30 + name_length + extra_length
        end

        def decode_name(bytes, utf8:)
          value = bytes.dup
          value.force_encoding(utf8 ? Encoding::UTF_8 : Encoding::BINARY)
          raise Error, "PyTorch archive #{path} contains an invalid entry name" unless value.valid_encoding?

          value.encode(Encoding::UTF_8)
        rescue EncodingError
          raise Error, "PyTorch archive #{path} contains a non-UTF-8 entry name"
        end

        def validate_entry_name!(name)
          raise Error, "PyTorch archive #{path} contains an invalid entry name #{name.inspect}" if name.empty? || name.include?("\0")

          candidate = Pathname(name)
          return if !candidate.absolute? &&
                    candidate.each_filename.none? { |part| part == ".." }

          raise Error, "PyTorch archive #{path} contains an invalid entry name #{name.inspect}"
        end

        def read_range(offset, length)
          File.open(path, "rb") do |file|
            file.seek(offset)
            value = file.read(length)
            return value if value && value.bytesize == length
          end
          raise Error, "PyTorch archive #{path} ended unexpectedly"
        end
      end

      # Restricted pickle virtual machine for tensor state dictionaries.
      class RestrictedUnpickler
        MARKER = Object.new.freeze

        attr_reader :storages

        def initialize(io, limit: PICKLE_LIMIT, storages: nil)
          @io = io
          @limit = limit
          @start = io.pos
          @stack = []
          @memo = {}
          @storages = storages || {}
          @next_memo = 0
          @opcode_count = 0
        end

        def load
          loop do
            @opcode_count += 1
            raise Error, "Restricted PyTorch pickle exceeds the opcode-count limit" if @opcode_count > PICKLE_OPCODE_LIMIT

            opcode = read_exact(1).getbyte(0)
            case opcode
            when 0x80 then protocol!
            when 0x95 then read_uint64 # FRAME
            when 0x2e then return pop # STOP
            when 0x28 then push(MARKER)
            when 0x30 then pop
            when 0x31 then pop_mark
            when 0x32 then push(peek)
            when 0x4e then push(nil)
            when 0x88 then push(true)
            when 0x89 then push(false)
            when 0x4a then push(read_exact(4).unpack1("l<"))
            when 0x4b then push(read_exact(1).unpack1("C"))
            when 0x4d then push(read_exact(2).unpack1("v"))
            when 0x49 then push(parse_ascii_integer(read_line))
            when 0x4c then push(Integer(read_line.delete_suffix("L"), 10))
            when 0x8a then push(parse_long(read_long_bytes(1)))
            when 0x8b then push(parse_long(read_long_bytes(4)))
            when 0x46 then push(Float(read_line))
            when 0x47 then push(read_exact(8).unpack1("G"))
            when 0x58 then push(read_utf8(read_exact(read_exact(4).unpack1("V"))))
            when 0x8c then push(read_utf8(read_exact(read_exact(1).unpack1("C"))))
            when 0x8d then push(read_utf8(read_exact(read_uint64)))
            when 0x54 then push(read_exact(read_exact(4).unpack1("V")))
            when 0x55 then push(read_exact(read_exact(1).unpack1("C")))
            when 0x42 then push(read_exact(read_exact(4).unpack1("V")))
            when 0x43 then push(read_exact(read_exact(1).unpack1("C")))
            when 0x8e then push(read_exact(read_uint64))
            when 0x96 then push(read_exact(read_uint64).dup)
            when 0x53 then push(parse_quoted_string(read_line))
            when 0x5d then push([])
            when 0x6c then push(pop_mark)
            when 0x61
              value = pop
              list_target << value
            when 0x65
              values = pop_mark
              list_target.concat(values)
            when 0x7d then push({})
            when 0x64 then push(Hash[*pop_mark])
            when 0x73 then set_pair
            when 0x75 then set_pairs
            when 0x29 then push([])
            when 0x74 then push(pop_mark)
            when 0x85 then push([pop])
            when 0x86 then push(pop(2))
            when 0x87 then push(pop(3))
            when 0x8f then push([]) # EMPTY_SET; arrays suffice for metadata
            when 0x90
              values = pop_mark
              list_target.concat(values).uniq!
            when 0x91 then push(pop_mark.uniq.freeze)
            when 0x63 then push(read_global)
            when 0x93
              module_name, name = pop(2)
              raise Error, "Restricted PyTorch pickle has an invalid STACK_GLOBAL" unless module_name.is_a?(String) && name.is_a?(String)

              push(Global.new(module_name: module_name, name: name))
            when 0x52 then reduce!
            when 0x51 then push(persistent_load(pop))
            when 0x50 then push(persistent_load(read_line))
            when 0x62 then build!
            when 0x81 then new_object!
            when 0x92 then new_object_with_keywords!
            when 0x71 then memo_store(read_exact(1).unpack1("C"))
            when 0x72 then memo_store(read_exact(4).unpack1("V"))
            when 0x70 then memo_store(Integer(read_line, 10))
            when 0x94 then memo_store_next
            when 0x68 then push(memo_fetch(read_exact(1).unpack1("C")))
            when 0x6a then push(memo_fetch(read_exact(4).unpack1("V")))
            when 0x67 then push(memo_fetch(Integer(read_line, 10)))
            else
              raise Error, format("Restricted PyTorch pickle uses unsupported opcode 0x%02x", opcode)
            end
          end
        rescue EOFError
          raise Error, "PyTorch pickle ended before STOP"
        rescue ArgumentError, TypeError, RangeError => e
          raise Error, "Invalid restricted PyTorch pickle: #{e.message}"
        end

        private

        def protocol!
          version = read_exact(1).unpack1("C")
          raise Error, "PyTorch pickle protocol #{version} is unsupported" unless version.between?(2, 5)
        end

        def read_global
          module_name = read_line
          name = read_line
          Global.new(module_name: module_name.freeze, name: name.freeze)
        end

        def reduce!
          arguments = pop
          callable = pop
          raise Error, "Restricted PyTorch pickle attempted a non-global reduction" unless callable.is_a?(Global) && arguments.is_a?(Array)

          push(apply_reducer(callable, arguments))
        end

        def apply_reducer(callable, arguments)
          key = [callable.module_name, callable.name]
          case key
          when %w[collections OrderedDict]
            pairs = arguments.first || []
            return {} if pairs.empty?
            unless pairs.is_a?(Array) && pairs.all? { |pair| pair.is_a?(Array) && pair.length == 2 }
              raise Error, "Restricted PyTorch pickle has an invalid OrderedDict"
            end

            pairs.to_h
          when %w[torch Size]
            Array(arguments.first)
          when %w[builtins set], %w[__builtin__ set]
            Array(arguments.first).uniq
          when ["torch._utils", "_rebuild_tensor"],
               ["torch._utils", "_rebuild_tensor_v2"],
               ["torch._utils", "_rebuild_tensor_v3"]
            rebuild_tensor(arguments)
          when ["torch._utils", "_rebuild_parameter"],
               ["torch._utils", "_rebuild_parameter_with_state"]
            tensor = arguments.first
            raise Error, "Restricted PyTorch pickle has an invalid parameter" unless tensor.is_a?(TensorSpec)

            tensor
          else
            raise Error,
                  "Restricted PyTorch pickle refuses reducer #{callable.module_name}.#{callable.name}"
          end
        end

        def rebuild_tensor(arguments)
          storage, offset, shape, stride = arguments.first(4)
          unless storage.is_a?(StorageRef) && offset.is_a?(Integer) &&
                 shape.is_a?(Array) && stride.is_a?(Array)
            raise Error, "Restricted PyTorch pickle has invalid tensor reconstruction arguments"
          end

          TensorSpec.new(
            storage: storage,
            storage_offset: offset,
            shape: integer_vector(shape, "tensor shape"),
            stride: integer_vector(stride, "tensor stride")
          )
        end

        def persistent_load(identifier)
          unless identifier.is_a?(Array) && identifier.length.between?(5, 6) && identifier[0].to_s == "storage"
            raise Error, "Restricted PyTorch pickle contains an unsupported persistent object"
          end

          storage_global, key, _location, elements, view = identifier[1..]
          unless storage_global.is_a?(Global) && storage_global.module_name == "torch"
            raise Error, "Restricted PyTorch pickle has an invalid storage type"
          end

          dtype, = STORAGE_DTYPES.fetch(storage_global.name) do
            raise Error, "PyTorch storage #{storage_global.name.inspect} is unsupported"
          end
          unless (key.is_a?(String) || key.is_a?(Integer)) && elements.is_a?(Integer) && elements >= 0
            raise Error, "Restricted PyTorch pickle has invalid storage metadata"
          end

          key = key.to_s
          raise Error, "Restricted PyTorch pickle has an invalid storage key #{key.inspect}" unless /\A[0-9A-Za-z_.-]+\z/.match?(key)

          existing = storages[key]
          if existing && (existing.dtype != dtype || existing.elements != elements)
            raise Error, "PyTorch storage #{key.inspect} has conflicting metadata"
          end

          root = existing || StorageRef.new(dtype: dtype, key: key.freeze, elements: elements, base_offset: 0)
          storages[key] ||= root
          return root unless view

          unless view.is_a?(Array) && view.length == 3 &&
                 view[1].is_a?(Integer) && view[2].is_a?(Integer) &&
                 view[1] >= 0 && view[2] >= 0 && view[1] + view[2] <= elements
            raise Error, "Restricted PyTorch pickle has invalid storage-view metadata"
          end

          StorageRef.new(dtype: dtype, key: key.freeze, elements: view[2], base_offset: view[1])
        end

        def build!
          state = pop
          object = peek
          return if state.is_a?(Hash) && (object.is_a?(Hash) || object.is_a?(TensorSpec))

          raise Error, "Restricted PyTorch pickle refuses object BUILD state"

          # OrderedDict's _metadata and Tensor's backward state do not affect
          # weight bytes. They are deliberately validated as a Hash then ignored.
        end

        def new_object!
          arguments = pop
          callable = pop
          raise Error, "Restricted PyTorch pickle attempted invalid NEWOBJ" unless callable.is_a?(Global) && arguments.is_a?(Array)

          push(apply_reducer(callable, arguments))
        end

        def new_object_with_keywords!
          keywords = pop
          arguments = pop
          callable = pop
          unless keywords == {} && callable.is_a?(Global) && arguments.is_a?(Array)
            raise Error, "Restricted PyTorch pickle refuses NEWOBJ_EX keyword construction"
          end

          push(apply_reducer(callable, arguments))
        end

        def integer_vector(values, description)
          raise Error, "Restricted PyTorch pickle has an invalid #{description}" unless values.all?(Integer)

          values.map(&:to_i).freeze
        end

        def set_pair
          value = pop
          key = pop
          target = peek
          raise Error, "Restricted PyTorch pickle SETITEM target is not a Hash" unless target.is_a?(Hash)

          target[key] = value
        end

        def set_pairs
          pairs = pop_mark
          raise Error, "Restricted PyTorch pickle has an odd SETITEMS payload" unless pairs.length.even?

          target = peek
          raise Error, "Restricted PyTorch pickle SETITEMS target is not a Hash" unless target.is_a?(Hash)

          pairs.each_slice(2) { |key, value| target[key] = value }
        end

        def list_target
          target = peek
          raise Error, "Restricted PyTorch pickle list target is invalid" unless target.is_a?(Array)

          target
        end

        def memo_store(index)
          @memo[index] = peek
          @next_memo = [@next_memo, index + 1].max
        end

        def memo_store_next
          @memo[@next_memo] = peek
          @next_memo += 1
        end

        def memo_fetch(index)
          @memo.fetch(index)
        rescue KeyError
          raise Error, "Restricted PyTorch pickle references missing memo #{index}"
        end

        def push(value)
          @stack << value
        end

        def peek
          raise Error, "Restricted PyTorch pickle stack is empty" if @stack.empty?

          @stack.last
        end

        def pop(count = nil)
          return @stack.pop unless count
          raise Error, "Restricted PyTorch pickle stack underflow" if @stack.length < count

          @stack.pop(count)
        end

        def pop_mark
          index = @stack.rindex(MARKER)
          raise Error, "Restricted PyTorch pickle has no MARK" unless index

          values = @stack.slice!(index + 1, @stack.length - index - 1)
          @stack.pop
          values
        end

        def read_exact(length)
          raise Error, "Restricted PyTorch pickle requests an invalid byte count" if length.negative?
          raise Error, "PyTorch pickle exceeds the #{@limit}-byte size limit" if @io.pos - @start + length > @limit

          value = @io.read(length)
          raise EOFError unless value && value.bytesize == length

          value
        end

        def read_line
          remaining = @limit - (@io.pos - @start)
          raise Error, "PyTorch pickle exceeds the #{@limit}-byte size limit" unless remaining.positive?

          value = @io.gets("\n", remaining + 1)
          raise EOFError unless value&.end_with?("\n")

          value.delete_suffix("\n")
        end

        def read_uint64
          value = read_exact(8).unpack1("Q<")
          raise Error, "Restricted PyTorch pickle declares an oversized field" if value > @limit

          value
        end

        def read_long_bytes(length_width)
          length = read_exact(length_width).unpack1(length_width == 1 ? "C" : "V")
          if length > PICKLE_LONG_BYTES_LIMIT
            raise Error,
                  "Restricted PyTorch pickle integer exceeds the #{PICKLE_LONG_BYTES_LIMIT}-byte size limit"
          end

          read_exact(length)
        end

        def read_utf8(bytes)
          value = bytes.dup.force_encoding(Encoding::UTF_8)
          raise Error, "Restricted PyTorch pickle contains invalid UTF-8" unless value.valid_encoding?

          value.freeze
        end

        def parse_ascii_integer(value)
          return true if value == "01"
          return false if value == "00"

          Integer(value, 10)
        end

        def parse_long(bytes)
          value = bytes.bytes.each_with_index.sum { |byte, index| byte << (8 * index) }
          value -= 1 << (8 * bytes.bytesize) if !bytes.empty? && bytes.getbyte(-1).anybits?(0x80)
          value
        end

        def parse_quoted_string(value)
          unless value.bytesize >= 2 && ["'", '"'].include?(value[0]) && value[-1] == value[0]
            raise Error, "Restricted PyTorch pickle contains an invalid STRING"
          end

          body = value.byteslice(1, value.bytesize - 2)
          body.gsub(/\\(?:x([0-9a-fA-F]{2})|([\\'"nrt]))/) do
            if Regexp.last_match(1)
              Regexp.last_match(1).to_i(16).chr
            else
              { "\\" => "\\", "'" => "'", '"' => '"', "n" => "\n", "r" => "\r", "t" => "\t" }
                .fetch(Regexp.last_match(2))
            end
          end
        end
      end

      # One checkpoint shard. Tensor bytes are never materialized as a whole.
      class Reader
        attr_reader :path, :tensors

        def initialize(path)
          @path = Pathname(path).expand_path
          @storages = {}
          @storage_payloads = {}
          @verified_storage_payloads = {}
          @storage_verification_mutex = Mutex.new
          @tensors = read_checkpoint.freeze
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

        def write_tensor(tensor, output, target_dtype:, converter: Safetensors::DTypeConverter.default,
                         chunk_bytes: Safetensors::DEFAULT_CHUNK_BYTES)
          raise ArgumentError, "Tensor #{tensor.name.inspect} belongs to a different PyTorch reader" unless tensor.reader.equal?(self)

          target_dtype = target_dtype.to_s.upcase
          unless FLOAT_DTYPES.include?(tensor.dtype) && FLOAT_DTYPES.include?(target_dtype)
            raise Error, "Cannot convert #{tensor.name.inspect} from #{tensor.dtype} to #{target_dtype}"
          end
          raise ArgumentError, "chunk_bytes must be positive" unless chunk_bytes.positive?

          verify_storage_payload!(tensor)

          source_width = Safetensors::DTYPE_BYTES.fetch(tensor.dtype)
          elements_per_chunk = [chunk_bytes / source_width, 1].max
          bytes_per_chunk = elements_per_chunk * source_width
          unless contiguous?(tensor)
            return write_strided_tensor(
              tensor, output, target_dtype: target_dtype, converter: converter,
                              elements_per_chunk: elements_per_chunk, source_width: source_width
            )
          end

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
          expected = tensor.element_count * Safetensors::DTYPE_BYTES.fetch(target_dtype)
          return written if written == expected

          raise Error, "Converted tensor #{tensor.name.inspect} wrote #{written} bytes; expected #{expected}"
        end

        private

        def read_checkpoint
          signature = File.binread(path, 4)
          if signature == [ZIP_LOCAL].pack("V")
            read_zip
          elsif path.size >= 262 && File.binread(path, 5, 257) == "ustar"
            raise Error,
                  "Legacy tar PyTorch checkpoints are not accepted by the restricted weights-only reader; " \
                  "re-save this dense state dictionary as Safetensors or a current torch.save ZIP"
          else
            read_legacy_stream
          end
        rescue Errno::ENOENT
          raise Error, "PyTorch checkpoint #{path} does not exist"
        rescue Errno::EACCES => e
          raise Error, "Cannot read PyTorch checkpoint #{path}: #{e.message}"
        end

        def read_zip
          archive = ZipArchive.new(path)
          data_entries = archive.entries.keys.select { |name| name.end_with?("/data.pkl") }
          raise Error, "PyTorch archive #{path} must contain exactly one data.pkl" unless data_entries.length == 1

          data_name = data_entries.first
          prefix = data_name.delete_suffix("data.pkl")
          byteorder_name = "#{prefix}byteorder"
          if archive.entries.key?(byteorder_name)
            byteorder = archive.read(byteorder_name, limit: 16)
            raise Error, "Big-endian PyTorch checkpoints are not supported" unless byteorder == "little"
          end
          root = RestrictedUnpickler.new(
            StringIO.new(archive.read(data_name)), storages: @storages
          ).load
          build_tensors(root) do |storage|
            entry = archive.fetch("#{prefix}data/#{storage.key}")
            unless entry.compression_method.zero? && entry.compressed_size == entry.size
              raise Error, "PyTorch tensor storage #{storage.key.inspect} is compressed and cannot be streamed"
            end

            expected = @storages.fetch(storage.key).elements * dtype_width(storage.dtype)
            unless entry.size == expected
              raise Error, "PyTorch storage #{storage.key.inspect} has #{entry.size} bytes; expected #{expected}"
            end

            @storage_payloads[storage.key] ||= StoragePayload.new(
              data_offset: entry.data_offset,
              size: entry.size,
              crc32: entry.crc32
            )

            entry.data_offset
          end
        end

        def read_legacy_stream
          File.open(path, "rb") do |file|
            magic = RestrictedUnpickler.new(file, limit: 64).load
            raise Error, "#{path} is not a supported PyTorch state-dictionary checkpoint" unless magic == MAGIC_NUMBER

            version = RestrictedUnpickler.new(file, limit: 64).load
            raise Error, "PyTorch legacy protocol #{version.inspect} is unsupported" unless version == PROTOCOL_VERSION

            system_info = RestrictedUnpickler.new(file, limit: 4096).load
            unless system_info.is_a?(Hash) && system_info["little_endian"] != false
              raise Error, "Big-endian PyTorch checkpoints are not supported"
            end

            root = RestrictedUnpickler.new(file, storages: @storages).load
            storage_keys = RestrictedUnpickler.new(file, limit: PICKLE_LIMIT).load
            unless storage_keys.is_a?(Array) && storage_keys.map(&:to_s).uniq.length == storage_keys.length
              raise Error, "PyTorch legacy checkpoint has an invalid storage-key list"
            end

            offsets = {}
            storage_keys.each do |raw_key|
              key = raw_key.to_s
              storage = @storages.fetch(key) do
                raise Error, "PyTorch legacy checkpoint lists unknown storage #{key.inspect}"
              end
              raw_count = file.read(8)
              unless raw_count&.bytesize == 8
                raise Error,
                      "PyTorch legacy checkpoint ended before storage #{key.inspect}"
              end

              count = raw_count.unpack1("Q<")
              unless count == storage.elements
                raise Error, "PyTorch legacy storage #{key.inspect} has #{count} elements; expected #{storage.elements}"
              end

              offsets[key] = file.pos
              bytes = count * dtype_width(storage.dtype)
              file.seek(bytes, IO::SEEK_CUR)
              raise Error, "PyTorch legacy storage #{key.inspect} points beyond #{path}" if file.pos > path.size
            end
            extra = @storages.keys - offsets.keys
            raise Error, "PyTorch legacy checkpoint omits storage(s): #{extra.first(8).join(", ")}" unless extra.empty?

            build_tensors(root) { |storage| offsets.fetch(storage.key) }
          end
        end

        def build_tensors(root)
          state = extract_state_dict(root)
          state.to_h do |name, specification|
            validate_tensor_spec!(name, specification)
            storage = specification.storage
            root_storage = @storages.fetch(storage.key)
            width = dtype_width(storage.dtype)
            count = element_count(specification.shape)
            relative_elements = storage.base_offset + specification.storage_offset
            last_element = contiguous_extent(specification.shape, specification.stride, relative_elements)
            if relative_elements.negative? || last_element > root_storage.elements
              raise Error, "Tensor #{name.inspect} points outside storage #{storage.key.inspect}"
            end

            base_offset = yield(storage)
            tensor = Tensor.new(
              reader: self,
              name: name.freeze,
              dtype: storage.dtype.freeze,
              shape: specification.shape.freeze,
              storage_key: storage.key,
              data_start: base_offset + (relative_elements * width),
              nbytes: count * width,
              storage_data_start: base_offset,
              storage_elements: root_storage.elements,
              storage_offset: relative_elements,
              stride: specification.stride.freeze
            )
            [name, tensor]
          end
        end

        def extract_state_dict(root)
          candidates = [root]
          candidates << root["state_dict"] if root.is_a?(Hash)
          state = candidates.compact.find do |candidate|
            candidate.is_a?(Hash) && !candidate.empty? &&
              candidate.all? { |name, value| name.is_a?(String) && value.is_a?(TensorSpec) }
          end
          return state if state

          raise Error, "PyTorch checkpoint #{path} does not contain a plain tensor state dictionary"
        end

        def validate_tensor_spec!(name, specification)
          unless name.is_a?(String) && !name.empty? && specification.is_a?(TensorSpec)
            raise Error, "PyTorch checkpoint #{path} contains invalid tensor metadata"
          end

          shape = specification.shape
          stride = specification.stride
          unless shape.length == stride.length && shape.all? { |dimension| dimension >= 0 } &&
                 specification.storage_offset >= 0
            raise Error, "Tensor #{name.inspect} has an invalid shape, stride, or storage offset"
          end
          return if stride.all? { |value| value >= 0 }

          raise Error, "Tensor #{name.inspect} has a negative stride"
        end

        def contiguous_extent(shape, stride, start)
          return start if shape.any?(&:zero?)

          start + shape.each_index.sum { |index| (shape[index] - 1) * stride[index] } + 1
        end

        def element_count(shape)
          shape.empty? ? 1 : shape.reduce(1, :*)
        end

        def dtype_width(dtype)
          Safetensors::DTYPE_BYTES.fetch(dtype)
        rescue KeyError
          raise Error, "PyTorch dtype #{dtype.inspect} is unsupported"
        end

        def contiguous?(tensor)
          expected = 1
          (tensor.shape.length - 1).downto(0) do |index|
            dimension = tensor.shape[index]
            return false if dimension > 1 && tensor.stride[index] != expected

            expected *= dimension unless dimension.zero?
          end
          true
        end

        def verify_storage_payload!(tensor)
          payload = @storage_payloads[tensor.storage_key]
          return unless payload

          @storage_verification_mutex.synchronize do
            return if @verified_storage_payloads.key?(tensor.storage_key)

            verify_storage_crc!(tensor.storage_key, payload)
            @verified_storage_payloads[tensor.storage_key] = true
          end
        end

        def verify_storage_crc!(storage_key, payload)
          crc32 = 0
          remaining = payload.size
          File.open(path, "rb") do |source|
            source.seek(payload.data_offset)
            while remaining.positive?
              requested = [remaining, STORAGE_CRC_CHUNK_BYTES].min
              chunk = source.read(requested)
              if chunk.nil? || chunk.bytesize != requested
                raise Error, "Unexpected end of #{path} while checking storage #{storage_key.inspect}"
              end

              crc32 = Zlib.crc32(chunk, crc32)
              remaining -= requested
            end
          end
          return if crc32 == payload.crc32

          raise Error, "PyTorch tensor storage #{storage_key.inspect} failed its CRC check"
        rescue Errno::ENOENT, Errno::EACCES => e
          raise Error, "Cannot read PyTorch checkpoint #{path}: #{e.message}"
        end

        # General strided tensors occur in legitimate torch state dictionaries
        # (notably transposed projection weights). A read-only file mapping
        # keeps heap usage bounded while preserving logical row-major order.
        def write_strided_tensor(tensor, output, target_dtype:, converter:, elements_per_chunk:, source_width:)
          total = tensor.element_count
          return 0 if total.zero?

          page_size = IO::Buffer::PAGE_SIZE
          map_offset = (tensor.storage_data_start / page_size) * page_size
          map_delta = tensor.storage_data_start - map_offset
          map_length = map_delta + (tensor.storage_elements * source_width)
          coordinates = Array.new(tensor.shape.length, 0)
          source_element = tensor.storage_offset
          written = 0
          previous_warning = Warning[:experimental]
          Warning[:experimental] = false
          File.open(path, "rb") do |source|
            mapped = IO::Buffer.map(source, map_length, map_offset, IO::Buffer::READONLY)
            begin
              remaining = total
              while remaining.positive?
                count = [remaining, elements_per_chunk].min
                gathered = IO::Buffer.new(count * source_width)
                begin
                  count.times do |index|
                    gathered.copy(
                      mapped,
                      index * source_width,
                      source_width,
                      map_delta + (source_element * source_width)
                    )
                    next unless index + 1 < count || remaining > count

                    source_element = advance_index!(
                      coordinates, tensor.shape, tensor.stride, source_element
                    )
                  end
                  raw = gathered.get_string(0, count * source_width)
                ensure
                  gathered.free
                end
                converted = converter.convert(raw, from: tensor.dtype, to: target_dtype)
                output.write(converted)
                written += converted.bytesize
                remaining -= count
              end
            ensure
              mapped.free
            end
          end
          expected = total * Safetensors::DTYPE_BYTES.fetch(target_dtype)
          return written if written == expected

          raise Error, "Converted tensor #{tensor.name.inspect} wrote #{written} bytes; expected #{expected}"
        ensure
          Warning[:experimental] = previous_warning unless previous_warning.nil?
        end

        def advance_index!(coordinates, shape, stride, source_element)
          (shape.length - 1).downto(0) do |dimension|
            coordinates[dimension] += 1
            source_element += stride[dimension]
            return source_element if coordinates[dimension] < shape[dimension]

            source_element -= stride[dimension] * shape[dimension]
            coordinates[dimension] = 0
          end
          source_element
        end
      end

      # One unsharded pytorch_model.bin or its Transformers shard index.
      class TensorSet
        attr_reader :directory, :readers, :tensors

        def self.from_directory(directory)
          directory = Pathname(directory).expand_path
          single = directory.join("pytorch_model.bin")
          return new(directory, [Reader.new(single)]) if single.file?

          index = directory.join("pytorch_model.bin.index.json")
          raise Error, "#{directory} does not contain pytorch_model.bin or its index" unless index.file?

          from_index(directory, index)
        end

        def self.from_index(directory, index_path)
          payload = JSON.parse(index_path.read(encoding: "UTF-8"))
          weight_map = payload["weight_map"]
          unless weight_map.is_a?(Hash) && !weight_map.empty? &&
                 weight_map.all? { |name, file| name.is_a?(String) && file.is_a?(String) }
            raise Error, "PyTorch index #{index_path} has an invalid weight_map"
          end

          shard_names = weight_map.values.uniq
          shard_names.each { |name| validate_shard_name!(name, index_path) }
          readers = shard_names.to_h do |name|
            shard = directory.join(name)
            raise Error, "PyTorch shard #{shard} is missing" unless shard.file?

            [name, Reader.new(shard)]
          end
          mapped = weight_map.to_h do |name, shard|
            reader = readers.fetch(shard)
            raise Error, "PyTorch index maps missing tensor #{name.inspect} to #{shard}" unless reader.key?(name)

            [name, reader.fetch(name)]
          end
          readers.each do |shard, reader|
            indexed = weight_map.filter_map { |name, filename| name if filename == shard }
            unindexed = reader.names - indexed
            next if unindexed.empty?

            raise Error,
                  "PyTorch shard #{reader.path} contains tensors absent from its index: " \
                  "#{unindexed.first(8).map(&:inspect).join(", ")}"
          end
          new(directory, readers.values, tensors: mapped)
        rescue JSON::ParserError, EncodingError => e
          raise Error, "Invalid PyTorch index #{index_path}: #{e.message}"
        rescue Errno::EACCES, Errno::ENOENT => e
          raise Error, "Cannot read PyTorch index #{index_path}: #{e.message}"
        end

        def self.validate_shard_name!(name, index_path)
          raise Error, "PyTorch index #{index_path} contains an invalid shard path #{name.inspect}" if name.empty? || name.include?("\0")

          candidate = Pathname(name)
          return if name.end_with?(".bin") && !candidate.absolute? &&
                    candidate.each_filename.none? { |part| part == ".." }

          raise Error, "PyTorch index #{index_path} contains an invalid shard path #{name.inspect}"
        end
        private_class_method :validate_shard_name!

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
    end
  end
end
