# frozen_string_literal: true

require "digest"
require "json"

module Cohere
  module Transcribe
    module State
      CONTRACT_SCHEMA_VERSION = 3

      ASR_IMPLEMENTATION_FILES = %w[
        asr/native.rb
        audio/decoder.rb
        audio/ffmpeg_native.rb
        audio/segmentation.rb
        configuration.rb
        dense_converter.rb
        gguf_writer.rb
        model_identity.rb
        runtime/engine.rb
        runtime/model_provider.rb
        runtime/preparation.rb
        vad/silero.rb
        vad/silero_vad_v6.onnx
        vad/timestamps.rb
      ].freeze
      RENDER_IMPLEMENTATION_FILES = %w[
        alignment/aligner.rb
        alignment/ctc.rb
        alignment/text.rb
        output/publication.rb
        output/rendering.rb
        output/timing.rb
        types.rb
      ].freeze

      module_function

      def asr_contract_key(options, model_format: "dense", model_quantization: nil)
        silero = options.vad == "silero"
        auditok = options.vad == "auditok"
        model_id = json_value(options.model)
        model_revision = ModelIdentity.default_model_revision(model_id, options.model_revision)
        configuration = {
          "language" => options.language,
          "device" => options.device,
          "dtype" => options.dtype,
          "audio_backend" => options.audio_backend,
          "vad" => options.vad,
          "vad_engine" => silero ? options.vad_engine : nil,
          "vad_batch_size" => silero ? options.vad_batch_size : nil,
          "vad_block_frames" => silero ? options.vad_block_frames : nil,
          "vad_threads" => silero ? options.vad_threads : nil,
          "vad_merge" => silero ? options.vad_merge : nil,
          "min_dur" => options.vad == "none" ? nil : options.min_dur,
          "max_dur" => options.max_dur,
          "max_silence" => auditok ? options.max_silence : nil,
          "energy_threshold" => auditok ? options.energy_threshold : nil,
          "vad_threshold" => silero ? options.vad_threshold : nil,
          "min_silence_ms" => silero ? options.min_silence_ms : nil,
          "speech_pad_ms" => silero ? options.speech_pad_ms : nil,
          "batch_size" => options.batch_size,
          "batch_max_size" => options.batch_max_size,
          "batch_audio_seconds" => options.batch_audio_seconds,
          "batch_vram_target" => options.batch_vram_target,
          "adaptive_batch" => options.adaptive_batch,
          "pin_memory" => options.pin_memory,
          "audio_memory_gb" => options.audio_memory_gb,
          "pipeline_preparation" => options.pipeline_preparation,
          "max_new_tokens" => options.max_new_tokens,
          "max_retry_tokens" => options.max_retry_tokens,
          "truncation_policy" => options.truncation_policy,
          "stop_repetition_loops" => options.stop_repetition_loops
        }
        fingerprint(
          "contract_schema_version" => CONTRACT_SCHEMA_VERSION,
          "configuration" => configuration,
          "models" => {
            "asr" => {
              "id" => model_id,
              "revision" => model_revision,
              "format" => model_format,
              "quantization" => model_quantization,
              "adapter" => options.adapter && {
                "id" => json_value(options.adapter),
                "revision" => options.adapter_revision
              }
            },
            "silero_version" => "6.2.1"
          },
          "implementation_sha256" => implementation_fingerprint(ASR_IMPLEMENTATION_FILES)
        )
      end

      def render_contract_key(options)
        formats = options.publication&.formats || []
        fingerprint(
          "contract_schema_version" => CONTRACT_SCHEMA_VERSION,
          "configuration" => {
            "alignment" => options.alignment,
            "align_batch_size" => options.align_batch_size,
            "align_dtype" => options.align_dtype,
            "formats" => formats.sort,
            "max_chars" => options.max_chars,
            "max_cue_dur" => options.max_cue_dur,
            "max_gap" => options.max_gap
          },
          "models" => {
            "align_revision" => options.alignment == "word" ? Alignment::ModelProvider::REVISION : nil
          },
          "output_schema_version" => Output::Publication::OUTPUT_SCHEMA_VERSION,
          "implementation_sha256" => implementation_fingerprint(RENDER_IMPLEMENTATION_FILES)
        )
      end

      def implementation_fingerprint(files)
        root = Pathname(__dir__).parent
        missing = files.reject { |relative| root.join(relative).file? }
        unless missing.empty?
          raise TranscriptionRuntimeError,
                "Implementation fingerprint references missing package artifacts: #{missing.join(", ")}"
        end

        fingerprint(
          "package_version" => VERSION,
          "artifacts_sha256" => files.to_h do |relative|
            [relative, Digest::SHA256.file(root.join(relative)).hexdigest]
          end
        )
      end

      def fingerprint(value)
        Digest::SHA256.hexdigest(canonical_json(value))
      end

      def json_value(value)
        value.is_a?(Pathname) ? value.to_s : value
      end
    end
  end
end
