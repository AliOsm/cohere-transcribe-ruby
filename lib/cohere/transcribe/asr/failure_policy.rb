# frozen_string_literal: true

require_relative "../python_text"

module Cohere
  module Transcribe
    module ASR
      # An inference failure whose recovery class is known at the boundary
      # where it occurred. Native bindings can use this instead of encoding
      # machine-readable state in an error-message string.
      class ExecutionError < TranscriptionRuntimeError
        KINDS = %i[oom fatal error].freeze

        attr_reader :failure_kind, :original

        def initialize(message, failure_kind:, original: nil)
          kind = failure_kind.to_sym
          raise ArgumentError, "failure_kind must be one of: #{KINDS.join(", ")}" unless KINDS.include?(kind)

          @failure_kind = kind
          @original = original
          super(message)
        end
      end

      # Raised without entering the native runtime after an invariant or
      # device-fatal failure has poisoned the retained inference session.
      class CircuitOpenError < ExecutionError
        def initialize(fingerprint)
          super(
            "runtime failure circuit breaker is open: #{fingerprint}",
            failure_kind: :fatal
          )
        end
      end

      # Classification shared by recursive batching and the native boundary.
      # Only allocator failures are allowed to teach a smaller batch cap.
      module FailurePolicy
        OUT_OF_MEMORY_MARKERS = [
          "out of memory",
          "cannot allocate memory",
          "cublas_status_alloc_failed",
          "cuda error: memory allocation",
          "cuda_error_memory_allocation",
          "cudaerrormemoryallocation",
          "cudnn_status_alloc_failed",
          "defaultcpuallocator"
        ].freeze

        DEVICE_FATAL_MARKERS = [
          "cublas_status",
          "cuda error",
          "cudnn_status",
          "device-side assert",
          "driver shutting down",
          "hip error",
          "illegal memory access",
          "mps backend failed",
          "unspecified launch failure"
        ].freeze

        INVARIANT_ERROR_CLASSES = [
          IndexError,
          KeyError,
          LoadError,
          NameError,
          NoMethodError,
          NotImplementedError,
          ScriptError,
          SystemStackError,
          TypeError
        ].freeze
        OUT_OF_MEMORY_PATTERN = Regexp.union(OUT_OF_MEMORY_MARKERS)
        DEVICE_FATAL_PATTERN = Regexp.union(DEVICE_FATAL_MARKERS)

        module_function

        def classify(error)
          if error.respond_to?(:failure_kind)
            kind = error.failure_kind
            kind = kind.to_sym if kind.respond_to?(:to_sym)
            return kind if ExecutionError::KINDS.include?(kind)
          end
          return :oom if error.is_a?(NoMemoryError)

          message = error.message.to_s.downcase
          return :oom if message.match?(OUT_OF_MEMORY_PATTERN)
          return :fatal if INVARIANT_ERROR_CLASSES.any? { |error_class| error.is_a?(error_class) }
          return :fatal if message.match?(DEVICE_FATAL_PATTERN)

          :error
        end

        def fingerprint(error)
          message = PythonText.split(error.message.to_s).join(" ")
          # Ruby supplies the exception class name as the implicit message for
          # `RuntimeError.new`, while Python's equivalent has an empty message.
          message = "<no message>" if message.empty? || message == error.class.to_s
          message = message[0, 500]
          message = message.gsub(/0x[0-9a-f]+/i, "0x*")
          message = message.gsub(/\b\d+\b/, "#")
          "#{error.class}: #{message}"
        end

        def formatted_message(error)
          "#{error.class}: #{error.message}"
        end
      end
    end
  end
end
