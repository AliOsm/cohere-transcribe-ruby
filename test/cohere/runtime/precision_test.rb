# frozen_string_literal: true

require "test_helper"

class Cohere::Transcribe::PrecisionTest < Minitest::Test
  Precision = Cohere::Transcribe::Runtime::Precision

  class FakeLibrary
    def initialize(device:, bf16: false)
      @device = device
      @bf16 = bf16
    end

    def resolve_device(_requested)
      @device
    end

    def supports_bf16?(_device)
      @bf16
    end
  end

  def test_cpu_normalizes_every_requested_precision_to_fp32
    %w[auto bf16 fp16 fp32].each do |requested|
      options = Cohere::Transcribe::TranscriptionOptions.new(dtype: requested)
      resolved = Precision.resolve(
        options,
        native_library: FakeLibrary.new(device: "cpu")
      )

      assert_equal "cpu", resolved.device
      assert_equal "fp32", resolved.dtype
      assert_equal requested, options.dtype
    end
  end

  def test_cuda_auto_uses_bf16_only_when_the_device_supports_it
    supported = Precision.resolve(
      Cohere::Transcribe::TranscriptionOptions.new,
      native_library: FakeLibrary.new(device: "cuda", bf16: true)
    )
    unsupported = Precision.resolve(
      Cohere::Transcribe::TranscriptionOptions.new,
      native_library: FakeLibrary.new(device: "cuda", bf16: false)
    )

    assert_equal "bf16", supported.dtype
    assert_equal "fp16", unsupported.dtype
  end

  def test_explicit_unsupported_bf16_is_rejected
    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      Precision.resolve(
        Cohere::Transcribe::TranscriptionOptions.new(dtype: "bf16"),
        native_library: FakeLibrary.new(device: "cuda", bf16: false)
      )
    end

    assert_match(/CUDA device does not support BF16/, error.message)
  end

  def test_mps_auto_uses_fp16_and_word_alignment_fp16_is_cuda_only
    resolved = Precision.resolve(
      Cohere::Transcribe::TranscriptionOptions.new,
      native_library: FakeLibrary.new(device: "mps", bf16: true)
    )
    assert_equal "fp16", resolved.dtype

    assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      Precision.resolve(
        Cohere::Transcribe::TranscriptionOptions.new(
          alignment: "word",
          align_dtype: "fp16"
        ),
        native_library: FakeLibrary.new(device: "mps", bf16: true)
      )
    end
  end

  def test_silero_executor_resolves_to_the_actual_onnx_runtime
    %w[auto torch onnx jit].each do |requested|
      resolved = Precision.resolve(
        Cohere::Transcribe::TranscriptionOptions.new(vad_engine: requested),
        native_library: FakeLibrary.new(device: "cpu")
      )

      assert_equal "onnx", resolved.vad_engine
    end
  end
end
