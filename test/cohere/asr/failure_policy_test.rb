# frozen_string_literal: true

require "test_helper"

class Cohere::Transcribe::ASRFailurePolicyTest < Minitest::Test
  FailurePolicy = Cohere::Transcribe::ASR::FailurePolicy

  def test_recognizes_allocator_failures_before_broad_device_markers
    messages = [
      "DefaultCPUAllocator: out of memory",
      "cannot allocate memory for native buffer",
      "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
      "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
      "cudaErrorMemoryAllocation",
      "CUDA_ERROR_MEMORY_ALLOCATION"
    ]

    messages.each do |message|
      assert_equal :oom, FailurePolicy.classify(RuntimeError.new(message)), message
    end
    assert_equal :oom, FailurePolicy.classify(NoMemoryError.new("heap exhausted"))
  end

  def test_recognizes_invariant_and_device_fatal_failures
    [
      TypeError.new("wrong native result"),
      KeyError.new("missing tensor"),
      NoMethodError.new("missing API"),
      LoadError.new("missing native symbol"),
      NotImplementedError.new("unsupported graph")
    ].each do |error|
      assert_equal :fatal, FailurePolicy.classify(error), error.class.to_s
    end

    [
      "CUDA error: illegal memory access",
      "device-side assert triggered",
      "driver shutting down",
      "MPS backend failed to execute graph",
      "unspecified launch failure"
    ].each do |message|
      assert_equal :fatal, FailurePolicy.classify(RuntimeError.new(message)), message
    end
  end

  def test_unclassified_runtime_errors_remain_data_local
    assert_equal :error, FailurePolicy.classify(RuntimeError.new("malformed sample payload"))
    assert_equal :error, FailurePolicy.classify(ArgumentError.new("empty audio segment"))
  end

  def test_typed_native_failure_overrides_its_message
    error = Cohere::Transcribe::ASR::ExecutionError.new(
      "backend returned no result",
      failure_kind: :oom
    )

    assert_equal :oom, FailurePolicy.classify(error)
    assert_equal :oom, error.failure_kind
    assert_raises(ArgumentError) do
      Cohere::Transcribe::ASR::ExecutionError.new("bad kind", failure_kind: :cancelled)
    end
  end

  def test_fingerprint_is_stable_and_removes_volatile_details
    assert_equal "RuntimeError: <no message>", FailurePolicy.fingerprint(RuntimeError.new)
    error = RuntimeError.new("  CUDA   failure at 0x7ffc1234 on device 12 after 99 steps  ")

    assert_equal(
      "RuntimeError: CUDA failure at 0x* on device # after # steps",
      FailurePolicy.fingerprint(error)
    )
    unicode = RuntimeError.new("\u00A0CUDA\u2003failure\u001Cat\u202F0x7ffc1234\u3000")
    assert_equal "RuntimeError: CUDA failure at 0x*", FailurePolicy.fingerprint(unicode)
  end

  def test_circuit_open_error_preserves_the_fingerprint
    error = Cohere::Transcribe::ASR::CircuitOpenError.new("RuntimeError: CUDA error #")

    assert_equal :fatal, FailurePolicy.classify(error)
    assert_match(/circuit breaker is open/, error.message)
    assert_match(/CUDA error #/, error.message)
  end
end
