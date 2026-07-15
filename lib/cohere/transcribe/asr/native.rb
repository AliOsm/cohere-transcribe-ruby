# frozen_string_literal: true

require "etc"
require "fiddle"
require "rbconfig"
require_relative "../errors"
require_relative "../python_text"
require_relative "failure_policy"

module Cohere
  module Transcribe
    module ASR
      NativeWord = Data.define(:text, :start, :end, :probability)
      NativeSegment = Data.define(:text, :start, :end, :words)
      NativeResult = Data.define(
        :text,
        :segments,
        :words,
        :generated_tokens,
        :generation_limit,
        :generation_capacity,
        :stopped_by_max_tokens,
        :repetition_stopped
      )

      # Thin, process-local binding to CrispASR's stable C session ABI. The
      # extension owns model execution; no Python process or Python runtime is
      # involved. Loading remains lazy so configuration and CLI help stay light.
      class NativeLibrary
        OPEN_PARAM_VERSION = 2
        OPEN_PARAM_INTEGER_COUNT = 12
        NATIVE_ERROR_NAMES = {
          0 => "none",
          1 => "invalid_argument",
          2 => "out_of_memory",
          3 => "invariant",
          4 => "runtime",
          5 => "cancelled"
        }.freeze

        FUNCTIONS = {
          last_error_kind: [[], :int],
          last_error_message: [[], :voidp],
          runtime_resolve_device: [[:voidp], :voidp],
          runtime_supports_bf16: [[:voidp], :int],
          set_gpu_backend: [[:voidp], :void],
          session_open_with_params: [%i[voidp voidp voidp], :voidp],
          session_backend: [[:voidp], :voidp],
          session_compute_backend: [[:voidp], :voidp],
          session_memory: [%i[voidp voidp voidp], :int],
          session_batch_capacity: [[:voidp], :int],
          session_cancel: [[:voidp], :int],
          session_transcribe_lang: [%i[voidp voidp int voidp], :voidp],
          session_transcribe_batch_lang: [%i[voidp voidp voidp int voidp], :voidp],
          session_batch_result_count: [[:voidp], :int],
          session_batch_result_at: [%i[voidp int], :voidp],
          session_batch_result_free: [[:voidp], :void],
          session_result_n_segments: [[:voidp], :int],
          session_result_segment_text: [%i[voidp int], :voidp],
          session_result_segment_t0: [%i[voidp int], :int64],
          session_result_segment_t1: [%i[voidp int], :int64],
          session_result_n_words: [%i[voidp int], :int],
          session_result_word_text: [%i[voidp int int], :voidp],
          session_result_word_t0: [%i[voidp int int], :int64],
          session_result_word_t1: [%i[voidp int int], :int64],
          session_result_word_p: [%i[voidp int int], :float],
          session_result_generated_tokens: [[:voidp], :int],
          session_result_generation_limit: [[:voidp], :int],
          session_result_generation_capacity: [[:voidp], :int],
          session_result_stopped_by_max_tokens: [[:voidp], :int],
          session_result_repetition_stopped: [[:voidp], :int],
          session_result_free: [[:voidp], :void],
          session_close: [[:voidp], :void],
          session_set_max_new_tokens: [%i[voidp int], :int],
          session_set_beam_size: [%i[voidp int], :int],
          session_set_repetition_loop_guard: [%i[voidp int], :int]
        }.freeze
        OPTIONAL_FUNCTIONS = {
          session_batch_result_stats_v1: [%i[voidp voidp int], :int]
        }.freeze

        TYPE_MAP = {
          void: Fiddle::TYPE_VOID,
          voidp: Fiddle::TYPE_VOIDP,
          int: Fiddle::TYPE_INT,
          int64: Fiddle::TYPE_LONG_LONG,
          float: Fiddle::TYPE_FLOAT
        }.freeze

        class << self
          def load
            @mutex ||= Mutex.new
            @mutex.synchronize { @instance ||= load_uncached }
          end

          def available?
            load
            true
          rescue TranscriptionRuntimeError
            false
          end

          def candidate_paths
            explicit = ENV.fetch("COHERE_TRANSCRIBE_NATIVE_LIBRARY", nil)
            return [explicit] if explicit && !explicit.empty?

            root = File.expand_path("../../../..", __dir__)
            packaged = File.join(root, "lib", "cohere", "transcribe", "native")
            patterns = case RbConfig::CONFIG["host_os"]
                       when /darwin/
                         ["libcrispasr*.dylib"]
                       when /mswin|mingw|cygwin/
                         ["crispasr*.dll", "libcrispasr*.dll"]
                       else
                         ["libcrispasr.so", "libcrispasr.so.*"]
                       end

            paths = [packaged].flat_map do |directory|
              patterns.flat_map { |pattern| Dir.glob(File.join(directory, pattern)) }.sort
            end
            paths.concat(system_library_names)
            paths.uniq
          end

          private

          def load_uncached
            failures = []
            candidate_paths.each do |candidate|
              return new(candidate)
            rescue Fiddle::DLError, LoadError => e
              failures << "#{candidate}: #{e.message}"
            end

            detail = failures.empty? ? "no candidate libraries were found" : failures.join("; ")
            raise TranscriptionRuntimeError,
                  "The native Cohere ASR runtime could not be loaded (#{detail}). " \
                  "Set COHERE_TRANSCRIBE_NATIVE_LIBRARY to libcrispasr."
          end

          def system_library_names
            case RbConfig::CONFIG["host_os"]
            when /darwin/
              ["libcrispasr.1.dylib", "libcrispasr.dylib"]
            when /mswin|mingw|cygwin/
              ["crispasr.dll", "libcrispasr.dll"]
            else
              ["libcrispasr.so.1", "libcrispasr.so"]
            end
          end
        end

        attr_reader :path

        def initialize(path)
          @path = path
          preload_sibling_libraries(path) if File.absolute_path(path) == path && File.file?(path)
          flags = Fiddle::RTLD_NOW | Fiddle::RTLD_GLOBAL
          @handle = Fiddle::Handle.new(path, flags)
          @functions = FUNCTIONS.to_h do |name, (arguments, result)|
            symbol = "crispasr_#{name}"
            address = @handle[symbol]
            [name, Fiddle::Function.new(address, arguments.map { |type| TYPE_MAP.fetch(type) }, TYPE_MAP.fetch(result))]
          end
          OPTIONAL_FUNCTIONS.each do |name, (arguments, result)|
            symbol = "crispasr_#{name}"
            address = @handle[symbol]
            @functions[name] = Fiddle::Function.new(
              address,
              arguments.map { |type| TYPE_MAP.fetch(type) },
              TYPE_MAP.fetch(result)
            )
          rescue Fiddle::DLError
            next
          end
          @functions.freeze
        end

        def call(name, *)
          @functions.fetch(name).call(*)
        end

        def function?(name)
          @functions.key?(name)
        end

        def open_session(model_path:, device:, threads:)
          gpu_backend = case device
                        when "cuda" then "cuda"
                        when "mps" then "metal"
                        else ""
                        end
          call(:set_gpu_backend, c_string(gpu_backend))

          use_gpu = device == "cpu" ? 0 : 1
          values = [OPEN_PARAM_VERSION, threads, use_gpu, 0, 1, -1] + Array.new(6, 0)
          unless values.length == OPEN_PARAM_INTEGER_COUNT
            raise TranscriptionRuntimeError, "Native open-parameter ABI is internally inconsistent"
          end

          packed = values.pack("i!*")
          pointer = call(
            :session_open_with_params,
            c_string(model_path.to_s),
            c_string("cohere"),
            Fiddle::Pointer[packed]
          )
          return pointer unless null_pointer?(pointer)

          native_kind = Integer(call(:last_error_kind))
          native_name = NATIVE_ERROR_NAMES.fetch(native_kind, "unknown")
          native_message = string(call(:last_error_message)).strip
          diagnostic = "native #{native_name} error (#{native_kind})"
          diagnostic = "#{diagnostic}: #{native_message}" unless native_message.empty?
          message = "Unable to load Dense Cohere model #{model_path.inspect} " \
                    "on device #{device.inspect}: #{diagnostic}"
          raise TranscriptionRuntimeError, message
        end

        def resolve_device(requested)
          value = requested.to_s
          pointer = call(:runtime_resolve_device, c_string(value))
          resolved = string(pointer)
          return resolved.freeze unless resolved.empty?

          case value
          when "cuda"
            raise TranscriptionRuntimeError,
                  "--device cuda was requested, but CUDA is not available to the native runtime"
          when "mps"
            raise TranscriptionRuntimeError,
                  "--device mps was requested, but Metal is not available to the native runtime"
          else
            raise TranscriptionRuntimeError, "Unsupported native inference device: #{value.inspect}"
          end
        end

        def supports_bf16?(device)
          call(:runtime_supports_bf16, c_string(device.to_s)) == 1
        end

        def string(pointer)
          return "" if null_pointer?(pointer)

          value = pointer.is_a?(Fiddle::Pointer) ? pointer.to_s : Fiddle::Pointer.new(pointer).to_s
          value.force_encoding(Encoding::UTF_8).scrub
        end

        def null_pointer?(pointer)
          pointer.nil? || pointer == 0 || (pointer.respond_to?(:null?) && pointer.null?)
        end

        private

        def c_string(value)
          Fiddle::Pointer["#{value}\0"]
        end

        # A packaged build keeps ggml beside libcrispasr. Preloading those
        # libraries makes that layout work even when a platform ignores
        # $ORIGIN for a manually-loaded shared object.
        def preload_sibling_libraries(path)
          directory = File.dirname(path)
          basenames = %w[libggml-base libggml-cpu libggml-cuda libggml-metal libggml]
          basenames.each do |basename|
            Dir.glob(File.join(directory, "#{basename}.*")).each do |dependency|
              next unless File.file?(dependency)

              Fiddle::Handle.new(dependency, Fiddle::RTLD_NOW | Fiddle::RTLD_GLOBAL)
            rescue Fiddle::DLError
              # The main load below reports a useful loader error if this is a
              # required dependency; optional GPU backends may legitimately fail.
              next
            end
          end
        end
      end

      # One retained native Cohere model session.
      class NativeSession
        MAX_NATIVE_SAMPLE_COUNT = 2_147_483_647
        NATIVE_CANCELLED_KIND = 5
        CANCELLATION_JOIN_INTERVAL = 0.01
        NATIVE_BATCH_STAT_FIELDS = %i[
          abi_version field_count total_us feature_wall_us feature_worker_us mel_pack_us
          encoder_graph_build_us encoder_graph_alloc_us encoder_input_us encoder_compute_us
          encoder_readback_us encoder_repack_us decoder_total_us decoder_cross_kv_us
          decoder_reserve_us decoder_decode_us render_us decoder_calls generation_steps
          encoder_microbatches token_id_readback_bytes
        ].freeze
        NATIVE_BATCH_STAT_ABI_VERSION = 1

        NATIVE_FAILURE_KINDS = {
          1 => :fatal,
          2 => :oom,
          3 => :fatal,
          4 => :error
        }.freeze

        attr_reader :backend, :batch_capacity, :compute_backend, :device, :last_batch_metrics, :model_path

        def initialize(model_path, options, threads: nil, library: NativeLibrary.load)
          @library = library
          @model_path = Pathname.new(model_path).expand_path.freeze
          raise TranscriptionRuntimeError, "Native Dense model does not exist: #{@model_path}" unless @model_path.file?

          @mutex = Mutex.new
          @closed = false
          thread_count = normalize_threads(threads)
          @session = @library.open_session(
            model_path: @model_path,
            device: options.device.to_s,
            threads: thread_count
          )
          @default_max_new_tokens = Integer(options.max_new_tokens)
          @library.call(:session_set_max_new_tokens, @session, @default_max_new_tokens)
          @library.call(:session_set_beam_size, @session, 1)
          @library.call(
            :session_set_repetition_loop_guard,
            @session,
            options.stop_repetition_loops ? 1 : 0
          )
          @backend = @library.string(@library.call(:session_backend, @session)).freeze
          unless @backend == "cohere"
            close
            raise TranscriptionRuntimeError,
                  "Native runtime selected #{@backend.inspect}, expected the Cohere Dense backend"
          end
          @compute_backend = @library.string(
            @library.call(:session_compute_backend, @session)
          ).freeze
          @device = canonical_device(@compute_backend)
          unless @device == options.device.to_s
            close
            raise TranscriptionRuntimeError,
                  "Native runtime selected #{@compute_backend.inspect} (#{@device}) " \
                  "after #{options.device.inspect} was resolved"
          end
          @batch_capacity = Integer(@library.call(:session_batch_capacity, @session))
          unless @batch_capacity.positive?
            close
            raise TranscriptionRuntimeError, "Native runtime reported an invalid Cohere batch capacity"
          end
        rescue StandardError
          close if defined?(@session) && @session
          raise
        end

        def transcribe(samples, language:, offset: 0.0, max_new_tokens: nil)
          @mutex.synchronize do
            ensure_open!
            @last_batch_metrics = nil
            generation_limit = max_new_tokens.nil? ? @default_max_new_tokens : Integer(max_new_tokens)
            set_generation_limit!(generation_limit)
            binary, sample_count = float_samples(samples)
            raise TranscriptionRuntimeError, "Audio segment is too large for the native ABI" \
              if sample_count > MAX_NATIVE_SAMPLE_COUNT

            run_native_inference do
              result = @library.call(
                :session_transcribe_lang,
                @session,
                Fiddle::Pointer[binary],
                sample_count,
                Fiddle::Pointer["#{language}\0"]
              )
              raise_native_failure!("Native Cohere inference returned no result") if @library.null_pointer?(result)

              begin
                materialize_result(result, offset: Float(offset))
              ensure
                @library.call(:session_result_free, result)
              end
            end
          end
        end

        # True padded-encoder/ragged-decoder batching in the native Cohere
        # runtime. The capacity is queried from the loaded adapter because its
        # logical decoder batch may span multiple encoder microbatches.
        def transcribe_batch(sample_batches, language:, offsets: nil, max_new_tokens: nil)
          unless sample_batches.is_a?(Array) && sample_batches.length.between?(1, @batch_capacity)
            raise ArgumentError,
                  "sample_batches must contain between 1 and #{@batch_capacity} audio rows"
          end

          offsets ||= Array.new(sample_batches.length, 0.0)
          unless offsets.is_a?(Array) && offsets.length == sample_batches.length
            raise ArgumentError, "offsets must contain one value per audio row"
          end

          @mutex.synchronize do
            ensure_open!
            @last_batch_metrics = nil
            generation_limit = max_new_tokens.nil? ? @default_max_new_tokens : Integer(max_new_tokens)
            set_generation_limit!(generation_limit)

            buffers_and_counts = sample_batches.map { |samples| float_samples(samples) }
            oversized_lane = buffers_and_counts.index { |_binary, count| count > MAX_NATIVE_SAMPLE_COUNT }
            if oversized_lane
              raise TranscriptionRuntimeError,
                    "Audio batch row #{oversized_lane} is too large for the native ABI"
            end

            run_native_inference do
              pointers = buffers_and_counts.map { |binary, _count| Fiddle::Pointer[binary] }
              pointer_table = pointers.map(&:to_i).pack("J*")
              counts = buffers_and_counts.map(&:last).pack("i!*")
              batch_result = @library.call(
                :session_transcribe_batch_lang,
                @session,
                Fiddle::Pointer[pointer_table],
                Fiddle::Pointer[counts],
                sample_batches.length,
                Fiddle::Pointer["#{language}\0"]
              )
              raise_native_failure!("Native Cohere batch inference returned no result") \
                if @library.null_pointer?(batch_result)

              begin
                @last_batch_metrics = materialize_batch_metrics(batch_result)
                count = @library.call(:session_batch_result_count, batch_result)
                unless count == sample_batches.length
                  raise ExecutionError.new(
                    "Native Cohere batch returned #{count} rows for #{sample_batches.length} inputs",
                    failure_kind: :fatal
                  )
                end
                Array.new(count) do |lane|
                  result = @library.call(:session_batch_result_at, batch_result, lane)
                  if @library.null_pointer?(result)
                    raise ExecutionError.new(
                      "Native Cohere batch result row #{lane} is missing",
                      failure_kind: :fatal
                    )
                  end

                  materialize_result(result, offset: Float(offsets.fetch(lane)))
                end.freeze
              ensure
                @library.call(:session_batch_result_free, batch_result)
              end
            end
          end
        end

        def close
          @mutex ||= Mutex.new
          @mutex.synchronize do
            return if @closed

            @library.call(:session_close, @session) if @library && @session
            @session = nil
            @closed = true
          end
          nil
        end

        def closed?
          @closed
        end

        # Current free/total memory for the selected ggml device, used by the
        # adaptive batch controller. Returns nil when a backend has no memory
        # telemetry rather than guessing from process RSS.
        def memory
          @mutex.synchronize do
            ensure_open!
            free_bytes = [0].pack("Q")
            total_bytes = [0].pack("Q")
            status = @library.call(
              :session_memory,
              @session,
              Fiddle::Pointer[free_bytes],
              Fiddle::Pointer[total_bytes]
            )
            return nil unless status.zero?

            [free_bytes.unpack1("Q"), total_bytes.unpack1("Q")].freeze
          end
        end

        private

        def raise_native_failure!(fallback_message)
          native_kind = Integer(@library.call(:last_error_kind))
          native_message = @library.string(@library.call(:last_error_message)).strip
          message = native_message.empty? ? fallback_message : "#{fallback_message}: #{native_message}"
          raise Interrupt, message if native_kind == NATIVE_CANCELLED_KIND

          failure_kind = NATIVE_FAILURE_KINDS.fetch(native_kind, native_kind.zero? ? :error : :fatal)
          raise ExecutionError.new(message, failure_kind: failure_kind)
        end

        # Ruby cannot deliver Thread#raise (including SIGINT's Interrupt) while
        # the receiving thread is blocked inside Fiddle. Keep the caller at an
        # interruptible join while a private worker owns the foreign call. The
        # caller still holds @mutex, so close and another inference cannot race
        # the session; cancellation is the only concurrent C ABI operation.
        def run_native_inference
          outcome = Queue.new
          worker = nil
          Thread.handle_interrupt(Exception => :on_blocking) do
            worker = Thread.new do
              outcome << [:returned, yield]
            rescue Exception => e # rubocop:disable Lint/RescueException -- transfer native cancellation intact
              outcome << [:raised, e]
            end
            worker.report_on_exception = false

            begin
              worker.join
            rescue Exception => e # rubocop:disable Lint/RescueException -- caller cancellation must win
              cancel_and_hard_join(worker)
              raise e
            end
          end

          status, value = begin
            outcome.pop(true)
          rescue ThreadError
            raise ExecutionError.new(
              "Native Cohere inference worker exited without reporting an outcome",
              failure_kind: :fatal
            )
          end
          raise value if status == :raised

          value
        end

        # A cancellation can arrive after Thread.new but before the worker has
        # entered the C function. Retry the non-poisoning session cancel until
        # the worker exits; never kill a thread that may still own ggml state.
        def cancel_and_hard_join(worker)
          loop do
            @library.call(:session_cancel, @session)
            return if worker.join(CANCELLATION_JOIN_INTERVAL)
          rescue Exception # rubocop:disable Lint/RescueException -- preserve the first caller exception
            next
          end
        end

        def canonical_device(name)
          case name.downcase
          when /\Acuda/ then "cuda"
          when /\Ametal/ then "mps"
          when /\Acpu/ then "cpu"
          else
            raise TranscriptionRuntimeError,
                  "Native runtime reported an unsupported compute backend: #{name.inspect}"
          end
        end

        def ensure_open!
          raise TranscriberClosedError, "The native Cohere model session has been closed" if @closed
        end

        def normalize_threads(threads)
          value = threads || ENV["COHERE_TRANSCRIBE_THREADS"] || Etc.nprocessors
          value = Integer(value)
          raise ArgumentError, "threads must be a positive integer" unless value.positive?

          value
        end

        def set_generation_limit!(generation_limit)
          raise ArgumentError, "max_new_tokens must be a positive integer" unless generation_limit.positive?
          return if @library.call(:session_set_max_new_tokens, @session, generation_limit).zero?

          raise TranscriptionRuntimeError, "Native Cohere inference rejected max_new_tokens=#{generation_limit}"
        end

        def float_samples(samples)
          if defined?(Numo::NArray) && samples.is_a?(Numo::NArray)
            array = samples.cast_to(Numo::SFloat).reshape(samples.size)
            [array.to_binary, array.size]
          elsif samples.is_a?(Array)
            [samples.map { |value| Float(value) }.pack("f*"), samples.length]
          else
            raise TypeError, "samples must be a Numo::NArray or an Array of floats"
          end
        end

        def materialize_result(result, offset:)
          segments = []
          words = []
          count = @library.call(:session_result_n_segments, result)
          count.times do |segment_index|
            segment_text = @library.string(
              @library.call(:session_result_segment_text, result, segment_index)
            )
            segment_text = PythonText.strip(segment_text)
            start_time = offset + centiseconds(
              @library.call(:session_result_segment_t0, result, segment_index)
            )
            end_time = offset + centiseconds(
              @library.call(:session_result_segment_t1, result, segment_index)
            )
            segment_words = materialize_words(result, segment_index, offset: offset)
            words.concat(segment_words)
            segments << NativeSegment.new(
              text: segment_text.freeze,
              start: start_time,
              end: end_time,
              words: segment_words.freeze
            )
          end

          text = PythonText.strip(segments.map(&:text).reject(&:empty?).join(" ")).freeze
          NativeResult.new(
            text: text,
            segments: segments.freeze,
            words: words.freeze,
            generated_tokens: @library.call(:session_result_generated_tokens, result),
            generation_limit: @library.call(:session_result_generation_limit, result),
            generation_capacity: @library.call(:session_result_generation_capacity, result),
            stopped_by_max_tokens: !@library.call(
              :session_result_stopped_by_max_tokens,
              result
            ).zero?,
            repetition_stopped: !@library.call(:session_result_repetition_stopped, result).zero?
          )
        end

        def materialize_batch_metrics(batch_result)
          return nil unless @library.respond_to?(:function?) &&
                            @library.function?(:session_batch_result_stats_v1)

          capacity = NATIVE_BATCH_STAT_FIELDS.length
          buffer = Array.new(capacity, 0).pack("q*")
          reported = Integer(
            @library.call(
              :session_batch_result_stats_v1,
              batch_result,
              Fiddle::Pointer[buffer],
              capacity
            )
          )
          return nil if reported < capacity

          values = buffer.unpack("q*")
          raw = NATIVE_BATCH_STAT_FIELDS.zip(values).to_h
          return nil unless raw.fetch(:abi_version) == NATIVE_BATCH_STAT_ABI_VERSION &&
                            raw.fetch(:field_count) >= capacity

          raw.each_with_object({}) do |(name, value), metrics|
            metrics[name.to_s.end_with?("_us") ? name.to_s.sub(/_us\z/, "_seconds").to_sym : name] =
              name.to_s.end_with?("_us") ? value.fdiv(1_000_000) : value
          end.freeze
        end

        def materialize_words(result, segment_index, offset:)
          count = @library.call(:session_result_n_words, result, segment_index)
          Array.new(count) do |word_index|
            start_cs = @library.call(:session_result_word_t0, result, segment_index, word_index)
            end_cs = @library.call(:session_result_word_t1, result, segment_index, word_index)
            word_text = @library.string(
              @library.call(:session_result_word_text, result, segment_index, word_index)
            )
            NativeWord.new(
              text: PythonText.strip(word_text).freeze,
              start: start_cs.negative? ? nil : offset + centiseconds(start_cs),
              end: end_cs.negative? ? nil : offset + centiseconds(end_cs),
              probability: @library.call(:session_result_word_p, result, segment_index, word_index)
            )
          end
        end

        def centiseconds(value)
          value.to_f / 100.0
        end
      end
    end
  end
end
