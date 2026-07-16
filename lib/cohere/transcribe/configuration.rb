# frozen_string_literal: true

require_relative "constants"
require_relative "errors"
require_relative "model_identity"
require_relative "python_text"
require_relative "types"

module Cohere
  module Transcribe
    module Configuration
      module_function

      CHOICES = {
        language: %w[ar en],
        device: %w[auto mps cuda cpu],
        dtype: %w[auto bf16 fp16 fp32],
        audio_backend: %w[auto torchcodec ffmpeg librosa],
        vad: %w[silero auditok none],
        vad_engine: %w[auto torch onnx jit],
        truncation_policy: %w[retry warn],
        alignment: %w[word segment none],
        align_dtype: %w[fp32 fp16]
      }.freeze
      INTEGER_OPTIONS = %i[
        preprocess_workers vad_batch_size vad_block_frames vad_threads
        min_silence_ms speech_pad_ms batch_size batch_max_size max_new_tokens
        max_retry_tokens align_batch_size max_chars
      ].freeze
      REAL_OPTIONS = %i[
        audio_memory_gb min_dur max_dur max_silence energy_threshold vad_threshold
        batch_audio_seconds batch_vram_target max_cue_dur max_gap
      ].freeze
      BOOLEAN_OPTIONS = %i[
        text_only recursive pipeline_preparation vad_merge adaptive_batch pin_memory
        stop_repetition_loops
      ].freeze
      MAX_TORCH_VAD_PADDED_FRAMES = 32_768
      ASR_FIXED_MIN_SECONDS = 1.0

      def validate!(options)
        unless defined?(TranscriptionOptions) && options.is_a?(TranscriptionOptions)
          raise TypeError, "options must be a TranscriptionOptions instance"
        end

        validate_model_reference!(options.model, option: "--model", description: "Model")
        validate_revision!(options.model_revision, option: "--model-revision")
        if local_reference?(options.model, description: "Model") && options.model_revision
          invalid!("--model-revision cannot be used with a local model directory")
        end
        validate_model_reference!(options.adapter, option: "--adapter", description: "Adapter") unless options.adapter.nil?
        invalid!("--adapter-revision requires --adapter") if options.adapter_revision && options.adapter.nil?
        validate_revision!(options.adapter_revision, option: "--adapter-revision")
        if !options.adapter.nil? &&
           local_reference?(options.adapter, description: "Adapter") && options.adapter_revision
          invalid!("--adapter-revision cannot be used with a local adapter directory")
        end

        BOOLEAN_OPTIONS.each do |name|
          value = options.public_send(name)
          invalid!("--#{name.to_s.tr("_", "-")} must be a boolean") unless [true, false].include?(value)
        end
        INTEGER_OPTIONS.each do |name|
          value = options.public_send(name)
          next if value.nil? || value.is_a?(Integer)

          invalid!("--#{name.to_s.tr("_", "-")} must be an integer")
        end
        REAL_OPTIONS.each do |name|
          value = options.public_send(name)
          next if value.nil? || (value.is_a?(Numeric) && !value.is_a?(Complex))

          invalid!("--#{name.to_s.tr("_", "-")} must be a real number")
        end
        CHOICES.each do |name, choices|
          value = options.public_send(name)
          next if value.is_a?(String) && choices.include?(value)

          invalid!("--#{name.to_s.tr("_", "-")} must be one of: #{choices.sort.join(", ")}")
        end

        invalid!("--text-only conflicts with --alignment word") if options.text_only && options.alignment == "word"
        publication = options.publication
        if !publication.nil? && (!defined?(PublicationOptions) || !publication.is_a?(PublicationOptions))
          invalid!("publication must be a PublicationOptions instance or nil")
        end
        formats = publication&.formats&.map(&:to_s)
        text_mode = options.text_only || options.alignment == "none"
        invalid!("Plain-text mode supports only --formats txt") if text_mode && formats && formats.uniq != ["txt"]

        positive_finite!(options.audio_memory_gb, "--audio-memory-gb")
        positive_integer_if_set!(options.preprocess_workers, "--preprocess-workers")

        packed_vad = options.vad == "silero" && %w[auto torch].include?(options.vad_engine)
        if packed_vad
          unless positive_integer?(options.vad_batch_size) && positive_integer?(options.vad_block_frames)
            invalid!("--vad-batch-size and --vad-block-frames must be positive")
          end
          if options.vad_batch_size * options.vad_block_frames > MAX_TORCH_VAD_PADDED_FRAMES
            invalid!(
              "--vad-batch-size * --vad-block-frames must not exceed " \
              "#{format("%<frames>d", frames: MAX_TORCH_VAD_PADDED_FRAMES)} frames"
            )
          end
        end
        positive_integer_if_set!(options.vad_threads, "--vad-threads")
        invalid!("--vad-threads applies only to packed Torch Silero VAD") if options.vad_threads && !packed_vad

        positive_integer_if_set!(options.batch_size, "--batch-size")
        positive_integer_if_set!(options.batch_max_size, "--batch-max-size")
        if options.batch_size && options.batch_max_size && options.batch_max_size < options.batch_size
          invalid!("--batch-max-size must be at least --batch-size")
        end
        invalid!("--batch-max-size requires --adaptive-batch") if options.batch_max_size && !options.adaptive_batch
        positive_finite_if_set!(options.batch_audio_seconds, "--batch-audio-seconds")
        unless finite?(options.batch_vram_target) && options.batch_vram_target.between?(0.50, 0.98)
          invalid!("--batch-vram-target must be between 0.50 and 0.98")
        end

        invalid!("--align-batch-size must be positive") if (options.alignment == "word") && !positive_integer?(options.align_batch_size)
        positive_finite!(options.max_dur, "--max-dur")
        invalid!("--min-dur must be finite") unless finite?(options.min_dur)
        invalid!("Require 0 <= --min-dur <= --max-dur") if options.vad != "none" && !options.min_dur.between?(0, options.max_dur)
        if options.vad == "none" && options.max_dur < ASR_FIXED_MIN_SECONDS
          invalid!(
            "--vad none requires --max-dur >= #{ASR_FIXED_MIN_SECONDS.to_i} second " \
            "to bound segment count and memory use"
          )
        end

        case options.vad
        when "silero"
          invalid!("--vad-threshold must be between 0 and 1") unless finite?(options.vad_threshold) && options.vad_threshold.between?(0, 1)
          unless options.min_silence_ms.is_a?(Integer) && options.speech_pad_ms.is_a?(Integer)
            invalid!("--min-silence-ms and --speech-pad-ms must be integers")
          end
          if options.min_silence_ms.negative? || options.speech_pad_ms.negative?
            invalid!("--min-silence-ms and --speech-pad-ms must be non-negative")
          end
        when "auditok"
          if (options.max_silence.is_a?(Numeric) && options.max_silence.negative?) ||
             (options.energy_threshold.is_a?(Numeric) && options.energy_threshold.negative?)
            invalid!("Auditok silence and energy thresholds must be finite and non-negative")
          end
          invalid!("--vad auditok requires --min-dur > 0") unless options.min_dur.positive?
          if options.max_silence.is_a?(Numeric) && options.max_silence >= options.max_dur
            invalid!("--vad auditok requires --max-silence < --max-dur")
          end
        end
        invalid!("--vad-merge is supported only with --vad silero") if options.vad_merge && options.vad != "silero"

        invalid!("--max-new-tokens must be positive") unless positive_integer?(options.max_new_tokens)
        invalid!("--max-retry-tokens must be an integer") unless options.max_retry_tokens.is_a?(Integer)
        invalid!("--max-retry-tokens must be at least --max-new-tokens") if options.max_retry_tokens < options.max_new_tokens
        thresholds = [
          options.vad_threshold, options.max_silence, options.energy_threshold,
          options.max_cue_dur, options.max_gap
        ]
        invalid!("All numeric thresholds and cue limits must be finite") unless thresholds.all? { |v| finite?(v) }
        if options.alignment != "none" &&
           (!positive_integer?(options.max_chars) || options.max_cue_dur <= 0 || options.max_gap.negative?)
          invalid!("Subtitle cue limits must be positive (and --max-gap non-negative)")
        end

        options
      end

      def resolved(options, model_identity: nil)
        validate!(options)
        publication = options.publication
        text_mode = options.text_only || options.alignment == "none"
        if publication
          formats = publication.formats || (text_mode ? ["txt"] : %w[txt srt vtt])
          publication = publication.with(formats: formats.map(&:to_s).uniq.freeze)
        end
        changes = {
          alignment: text_mode ? "none" : options.alignment,
          # ggml consumes host float buffers through its native session ABI;
          # there is no PyTorch CPU tensor to page-lock or asynchronous H2D
          # transfer for this compatibility option to affect.
          pin_memory: false,
          publication: publication
        }
        if model_identity
          changes.merge!(
            model: model_identity.model_id,
            model_revision: model_identity.model_revision,
            adapter: model_identity.adapter_id,
            adapter_revision: model_identity.adapter_revision
          )
        end
        options.with(**changes)
      end

      def validate_model_reference!(value, option:, description:)
        unless value.is_a?(String) || value.respond_to?(:to_path)
          raise TypeError, "#{description} must be a string or a path-like local directory"
        end

        text = value.is_a?(String) ? value : value.to_path
        raise TypeError, "#{description} path must resolve to text" unless text.is_a?(String)

        text = validated_utf8_text(text, option)
        stripped = PythonText.strip(text)
        invalid!("#{option} must be a non-empty Hugging Face repository ID or local directory") if stripped.empty?
        invalid!("#{option} must not have leading or trailing whitespace") unless text == stripped
        return if local_reference?(value, description: description)

        valid = valid_repository_id?(text)
        invalid!("#{option} must be a valid Hugging Face repository ID or local directory") unless valid
      rescue EncodingError
        invalid!("#{option} must be a valid Hugging Face repository ID or local directory")
      rescue ArgumentError => e
        invalid!(e.message)
      end

      def local_reference?(value, description:)
        text = value.is_a?(String) ? value : value.to_path
        unless value.is_a?(String)
          local = ModelIdentity.resolve_local_directory(text, description: description)
          raise ArgumentError, "#{description} directory #{text.inspect} does not exist" unless local

          return true
        end
        !ModelIdentity.resolve_local_directory(text, description: description).nil?
      end

      def validate_revision!(value, option:)
        return if value.nil?

        if value.is_a?(String)
          value = validated_utf8_text(value, option)
          stripped = PythonText.strip(value)
          return if !stripped.empty? && value == stripped
        end

        invalid!("#{option} must be a non-empty revision without surrounding whitespace")
      end

      private_class_method :validate_revision!

      def validated_utf8_text(value, option)
        text = value.b.dup.force_encoding(Encoding::UTF_8)
        invalid!("#{option} must contain valid UTF-8") unless text.valid_encoding?

        text
      end
      private_class_method :validated_utf8_text

      def valid_repository_id?(value)
        return false if value.count("/") > 1 || value.include?("--") || value.include?("..") || value.end_with?(".git")

        namespace, name = value.split("/", 2)
        name ||= namespace
        parts = value.include?("/") ? [namespace, name] : [name]
        return false unless name.length.between?(1, 96)

        parts.all? do |part|
          !part.empty? &&
            part.match?(/\A[\p{L}\p{N}_](?:[\p{L}\p{N}_.-]*[\p{L}\p{N}_])?\z/u)
        end
      end
      private_class_method :valid_repository_id?

      def positive_integer_if_set!(value, option)
        return if value.nil? || value.positive?

        invalid!("#{option} must be positive")
      end

      def positive_integer?(value)
        value.is_a?(Integer) && value.positive?
      end
      private_class_method :positive_integer?

      def positive_finite_if_set!(value, option)
        return if value.nil?

        positive_finite!(value, option)
      end

      def positive_finite!(value, option)
        invalid!("#{option} must be finite and positive") unless finite?(value) && value.positive?
      end

      def finite?(value)
        value.is_a?(Numeric) && !value.is_a?(Complex) && value.respond_to?(:finite?) && value.finite?
      end

      def invalid!(message)
        error_class = defined?(TranscriptionConfigurationError) ? TranscriptionConfigurationError : ArgumentError
        raise error_class, message
      end
    end
  end
end
