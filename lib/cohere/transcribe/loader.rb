# frozen_string_literal: true

module Cohere
  module Transcribe
    # Internal autoload map for the dependency-light public facade. Merely
    # requiring `cohere/transcribe` must not initialize decoder ABIs, Numo,
    # ONNX Runtime, model conversion, or the native inference binding.
    module Loader
      ROOT = File.expand_path(__dir__).freeze
      LOAD_MUTEX = Mutex.new

      module_function

      def load_runtime!
        LOAD_MUTEX.synchronize do
          require_relative "runtime/engine"
        end
      end

      def feature(relative)
        File.join(ROOT, relative).freeze
      end
      private_class_method :feature

      def register(parent, name, relative)
        return if parent.const_defined?(name, false)

        parent.autoload(name, feature(relative))
      end
      private_class_method :register

      def namespace(parent, name)
        if parent.const_defined?(name, false)
          parent.const_get(name, false)
        else
          parent.const_set(name, Module.new)
        end
      end
      private_class_method :namespace

      def install!
        transcribe = Cohere::Transcribe

        {
          Hub: "hub",
          ResolvedModelIdentity: "model_identity",
          ModelIdentity: "model_identity",
          Configuration: "configuration",
          Safetensors: "safetensors",
          PyTorchCheckpoint: "pytorch_checkpoint",
          GGUF: "gguf_writer",
          DenseConverter: "dense_converter",
          State: "state",
          CLI: "cli",
          Doctor: "doctor"
        }.each { |name, path| register(transcribe, name, path) }

        audio = namespace(transcribe, :Audio)
        {
          Decoded: "audio/decoder",
          Decoder: "audio/decoder",
          FFmpegNative: "audio/ffmpeg_native",
          Segmentation: "audio/segmentation"
        }.each { |name, path| register(audio, name, path) }

        vad = namespace(transcribe, :VAD)
        {
          Timestamps: "vad/timestamps",
          SileroBackendUnavailable: "vad/silero",
          Silero: "vad/silero"
        }.each { |name, path| register(vad, name, path) }

        output = namespace(transcribe, :Output)
        {
          Timing: "output/timing",
          Rendering: "output/rendering",
          PublicationPlan: "output/publication",
          PublicationDecision: "output/publication",
          Publication: "output/publication"
        }.each { |name, path| register(output, name, path) }

        alignment = namespace(transcribe, :Alignment)
        {
          Segment: "alignment/ctc",
          CTC: "alignment/ctc",
          Text: "alignment/text",
          BackendUnavailable: "alignment/aligner",
          ModelArtifact: "alignment/aligner",
          ModelProvider: "alignment/aligner",
          Session: "alignment/aligner",
          Aligner: "alignment/aligner"
        }.each { |name, path| register(alignment, name, path) }

        asr = namespace(transcribe, :ASR)
        {
          ExecutionError: "asr/failure_policy",
          CircuitOpenError: "asr/failure_policy",
          FailurePolicy: "asr/failure_policy",
          BatchTelemetry: "asr/batching",
          BatchController: "asr/batching",
          RetryBatchCap: "asr/batching",
          BatchExecutionResult: "asr/batching",
          BatchExecutor: "asr/batching",
          NativeWord: "asr/native",
          NativeSegment: "asr/native",
          NativeResult: "asr/native",
          NativeLibrary: "asr/native",
          NativeSession: "asr/native"
        }.each { |name, path| register(asr, name, path) }

        runtime = namespace(transcribe, :Runtime)
        {
          Precision: "runtime/precision",
          ModelProvider: "runtime/model_provider",
          Preparation: "runtime/preparation",
          ModelResources: "runtime/resources",
          WordPipeline: "runtime/word_pipeline",
          Engine: "runtime/engine"
        }.each { |name, path| register(runtime, name, path) }

        nil
      end
    end
    private_constant :Loader

    Loader.install!
  end
end
