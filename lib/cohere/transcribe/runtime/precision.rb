# frozen_string_literal: true

require "fiddle"

module Cohere
  module Transcribe
    module Runtime
      # Resolves public device/precision requests against the native runtime
      # before any input is decoded. The returned options describe what will
      # actually execute; an unavailable explicit accelerator is never allowed
      # to fall through to ggml's command-line-oriented CPU fallback.
      module Precision
        module_function

        def resolve(options, native_library: nil)
          library = native_library
          device = if options.device == "cpu"
                     "cpu"
                   else
                     library ||= ASR::NativeLibrary.load
                     library.resolve_device(options.device)
                   end
          dtype = resolved_dtype(options.dtype, device, library)
          if options.alignment == "word" && options.align_dtype == "fp16" && device != "cuda"
            raise TranscriptionRuntimeError,
                  "--align-dtype fp16 is supported only with CUDA"
          end

          vad_engine = options.vad == "silero" ? "onnx" : options.vad_engine
          options.with(device: device, dtype: dtype, vad_engine: vad_engine)
        rescue TranscriptionError
          raise
        rescue Fiddle::DLError, LoadError, SystemCallError => e
          raise TranscriptionRuntimeError,
                "Cannot resolve the native inference device: #{e.class}: #{e.message}"
        end

        def resolved_dtype(requested, device, native_library)
          return "fp32" if device == "cpu"

          resolved = if requested == "auto"
                       device == "cuda" && native_library&.supports_bf16?(device) ? "bf16" : "fp16"
                     else
                       requested
                     end
          if resolved == "bf16" && !native_library&.supports_bf16?(device)
            label = device == "mps" ? "MPS device/runtime" : "CUDA device"
            raise TranscriptionRuntimeError,
                  "This #{label} does not support BF16; use --dtype fp16"
          end
          resolved
        end
        private_class_method :resolved_dtype
      end
    end
  end
end
