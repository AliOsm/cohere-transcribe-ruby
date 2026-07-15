# frozen_string_literal: true

require "fiddle"
require "rbconfig"

require_relative "../errors"

module Cohere
  module Transcribe
    module Audio
      # Lazy binding for the gem's subprocess-free libav decoder adapter. The
      # adapter itself dynamically selects a compatible FFmpeg 4-8 runtime;
      # Ruby never launches ffmpeg and does not retain a Python codec runtime.
      module FFmpegNative
        ERROR_CAPACITY = 1_024
        CANCELLED_STATUS = 6

        class Library
          FUNCTIONS = {
            probe: [%i[voidp size_t], :int],
            decode: [%i[voidp int uint64 voidp voidp voidp size_t], :int],
            duration: [%i[voidp voidp voidp size_t], :int],
            cancel: [[], :void],
            free: [[:voidp], :void]
          }.freeze

          TYPE_MAP = {
            void: Fiddle::TYPE_VOID,
            voidp: Fiddle::TYPE_VOIDP,
            int: Fiddle::TYPE_INT,
            size_t: Fiddle::TYPE_SIZE_T,
            uint64: Fiddle::TYPE_UINT64_T
          }.freeze

          class << self
            def load
              @mutex ||= Mutex.new
              @mutex.synchronize { @instance ||= load_uncached }
            end

            def loaded
              mutex = @mutex
              mutex&.synchronize { @instance }
            end

            def candidate_paths
              explicit = ENV.fetch("COHERE_TRANSCRIBE_AUDIO_LIBRARY", nil)
              return [explicit] if explicit && !explicit.empty?

              packaged = File.expand_path("../native", __dir__)
              patterns = case RbConfig::CONFIG.fetch("host_os")
                         when /darwin/ then ["libcohere_audio*.dylib"]
                         when /mswin|mingw|cygwin/ then ["cohere_audio*.dll", "libcohere_audio*.dll"]
                         else ["libcohere_audio.so", "libcohere_audio.so.*"]
                         end
              paths = patterns.flat_map { |pattern| Dir.glob(File.join(packaged, pattern)) }.sort
              paths.concat(system_library_names)
              paths.uniq
            end

            private

            def load_uncached
              failures = []
              candidate_paths.each do |candidate|
                library = new(candidate)
                library.probe!
                return library
              rescue Fiddle::DLError, LoadError, TranscriptionRuntimeError => e
                failures << "#{candidate}: #{e.message}"
              end
              detail = failures.empty? ? "no candidate adapter libraries were found" : failures.join("; ")
              raise TranscriptionRuntimeError,
                    "The native FFmpeg audio adapter could not be loaded (#{detail}). " \
                    "Set COHERE_TRANSCRIBE_AUDIO_LIBRARY to libcohere_audio."
            end

            def system_library_names
              case RbConfig::CONFIG.fetch("host_os")
              when /darwin/ then ["libcohere_audio.1.dylib", "libcohere_audio.dylib"]
              when /mswin|mingw|cygwin/ then ["cohere_audio.dll", "libcohere_audio.dll"]
              else ["libcohere_audio.so.1", "libcohere_audio.so"]
              end
            end
          end

          attr_reader :path, :diagnostic

          def initialize(path)
            @path = path
            @handle = Fiddle::Handle.new(path, Fiddle::RTLD_NOW)
            @functions = FUNCTIONS.to_h do |name, (arguments, result)|
              address = @handle["cohere_audio_ffmpeg_#{name}"]
              function = Fiddle::Function.new(
                address,
                arguments.map { |type| TYPE_MAP.fetch(type) },
                TYPE_MAP.fetch(result)
              )
              [name, function]
            end.freeze
          end

          def probe!
            message = Fiddle::Pointer.malloc(ERROR_CAPACITY, Fiddle::RUBY_FREE)
            message[0, ERROR_CAPACITY] = "\0" * ERROR_CAPACITY
            status = @functions.fetch(:probe).call(message, ERROR_CAPACITY)
            @diagnostic = message.to_s.force_encoding(Encoding::UTF_8).scrub.freeze
            return self if status.zero?

            raise TranscriptionRuntimeError, "Native FFmpeg libraries are unavailable: #{@diagnostic}"
          end

          def decode(path, sample_rate:, max_decoded_bytes:)
            output_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_VOIDP, Fiddle::RUBY_FREE)
            output_slot[0, Fiddle::SIZEOF_VOIDP] = [0].pack("J")
            count_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_INT64_T, Fiddle::RUBY_FREE)
            count_slot[0, Fiddle::SIZEOF_INT64_T] = [0].pack("q")
            message = Fiddle::Pointer.malloc(ERROR_CAPACITY, Fiddle::RUBY_FREE)
            message[0, ERROR_CAPACITY] = "\0" * ERROR_CAPACITY
            maximum = max_decoded_bytes || 0

            status = @functions.fetch(:decode).call(
              c_string(path.to_s),
              sample_rate,
              maximum,
              output_slot,
              count_slot,
              message,
              ERROR_CAPACITY
            )
            address = output_slot[0, Fiddle::SIZEOF_VOIDP].unpack1("J")
            count = count_slot[0, Fiddle::SIZEOF_INT64_T].unpack1("q")
            detail = message.to_s.force_encoding(Encoding::UTF_8).scrub
            unless status.zero?
              @functions.fetch(:free).call(address) unless address.zero?
              raise Interrupt, "Native FFmpeg decode was cancelled for #{path}: #{detail}" if status == CANCELLED_STATUS

              raise TranscriptionRuntimeError, "Cannot decode #{path} through native FFmpeg: #{detail}"
            end
            if count.negative? || (count.positive? && address.zero?)
              @functions.fetch(:free).call(address) unless address.zero?
              raise TranscriptionRuntimeError, "Native FFmpeg returned invalid output metadata for #{path}"
            end
            bytes = count * Fiddle::SIZEOF_FLOAT
            if max_decoded_bytes && bytes > max_decoded_bytes
              @functions.fetch(:free).call(address) unless address.zero?
              raise TranscriptionRuntimeError,
                    "Native FFmpeg exceeded the configured decoded-audio memory limit for #{path}"
            end

            begin
              require "numo/narray"
              if count.zero?
                Numo::SFloat.zeros(0)
              else
                Numo::SFloat.from_binary(Fiddle::Pointer.new(address)[0, bytes])
              end
            rescue LoadError => e
              raise TranscriptionRuntimeError, "numo-narray is required for decoded audio: #{e.message}"
            ensure
              @functions.fetch(:free).call(address) unless address.zero?
            end
          end

          def duration(path)
            duration_slot = Fiddle::Pointer.malloc(Fiddle::SIZEOF_DOUBLE, Fiddle::RUBY_FREE)
            duration_slot[0, Fiddle::SIZEOF_DOUBLE] = [-1.0].pack("d")
            message = Fiddle::Pointer.malloc(ERROR_CAPACITY, Fiddle::RUBY_FREE)
            message[0, ERROR_CAPACITY] = "\0" * ERROR_CAPACITY
            status = @functions.fetch(:duration).call(
              c_string(path.to_s),
              duration_slot,
              message,
              ERROR_CAPACITY
            )
            detail = message.to_s.force_encoding(Encoding::UTF_8).scrub
            unless status.zero?
              raise Interrupt, "Native FFmpeg duration probe was cancelled for #{path}: #{detail}" if status == CANCELLED_STATUS

              raise TranscriptionRuntimeError,
                    "Cannot inspect #{path} through native FFmpeg: #{detail}"
            end

            seconds = duration_slot[0, Fiddle::SIZEOF_DOUBLE].unpack1("d")
            return nil unless seconds.finite? && seconds >= 0.0

            seconds
          end

          def cancel_all!
            @functions.fetch(:cancel).call
            nil
          end

          private

          def c_string(value)
            Fiddle::Pointer["#{value}\0"]
          end
        end

        module_function

        def library
          Library.load
        end

        def available?
          library
          true
        rescue TranscriptionRuntimeError
          false
        end

        def diagnostic
          library.diagnostic
        rescue TranscriptionRuntimeError => e
          e.message
        end

        def decode(path, sample_rate:, max_decoded_bytes:)
          library.decode(path, sample_rate: sample_rate, max_decoded_bytes: max_decoded_bytes)
        end

        def duration(path)
          library.duration(path)
        end

        def cancel_all!
          library.cancel_all!
        rescue TranscriptionRuntimeError
          nil
        end

        # Pipeline cleanup must not load FFmpeg merely to discover that no
        # native decode has ever started. If the adapter is already resident,
        # this wakes every operation that captured an older cancellation
        # generation while preserving later independent calls.
        def cancel_active!
          Library.loaded&.cancel_all!
          nil
        rescue Fiddle::DLError, TranscriptionRuntimeError
          nil
        end
      end
    end
  end
end
