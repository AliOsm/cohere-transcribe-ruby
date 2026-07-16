# frozen_string_literal: true

module Cohere
  module Transcribe
    module TypesSupport
      DEFAULT_ASR_MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
      OUTPUT_FORMATS = %w[txt srt vtt json].freeze
      EXISTING_POLICIES = %w[error overwrite skip].freeze

      module KeywordOnlyConstructor
        def new(*arguments, **keywords, &)
          raise ArgumentError, "#{self} accepts keyword arguments only" unless arguments.empty?

          super(**keywords, &)
        end
      end

      module_function

      def immutable(value)
        case value
        when String
          value.dup.freeze
        when Pathname
          Pathname.new(value.to_s).freeze
        when Array
          value.map { |item| immutable(item) }.freeze
        when Hash
          value.to_h do |key, item|
            [immutable(key), immutable(item)]
          end.freeze
        else
          value
        end
      end

      def tuple(value, field:)
        array = if value.is_a?(String)
                  value.each_char.to_a
                elsif value.respond_to?(:to_ary)
                  value.to_ary
                elsif value.respond_to?(:each)
                  value.to_a
                else
                  raise TypeError, "#{field} must be enumerable"
                end
        immutable(array)
      end

      def reference(value)
        source = value.respond_to?(:to_path) ? value.to_path : value
        return immutable(value) unless source.is_a?(String)

        text = source.b.dup.force_encoding(Encoding::UTF_8)
        text.valid_encoding? ? text.freeze : immutable(source)
      end

      def path(value, field: "path")
        source = value.respond_to?(:to_path) ? value.to_path : value
        raise TypeError, "#{field} must resolve to a text path" unless source.is_a?(String)

        text = source.b.dup.force_encoding(Encoding::UTF_8)
        raise ArgumentError, "#{field} must contain valid UTF-8" unless text.valid_encoding?

        absolute = text.start_with?("/")
        prefix = if text.start_with?("//") && !text.start_with?("///")
                   "//"
                 elsif absolute
                   "/"
                 else
                   ""
                 end
        components = text.split("/").reject { |component| component.empty? || component == "." }
        normalized = prefix + components.join("/")
        normalized = "." if normalized.empty?
        Pathname.new(normalized).freeze
      end
    end
    private_constant :TypesSupport

    # Optional durable output settings for a transcription run.
    PublicationOptions = Data.define(:formats, :output_dir, :existing, :profile_json) do
      def initialize(formats: nil, output_dir: nil, existing: "error", profile_json: nil)
        unless formats.nil?
          formats = TypesSupport.tuple(formats, field: "formats").uniq.freeze
          raise ArgumentError, "formats must contain at least one output format" if formats.empty?

          unsupported = formats.reject { |format| TypesSupport::OUTPUT_FORMATS.include?(format) }
          unless unsupported.empty?
            rendered = unsupported.map(&:to_s).sort.join(", ")
            raise ArgumentError, "Unsupported output format(s): #{rendered}"
          end
        end
        raise ArgumentError, "existing must be 'error', 'overwrite', or 'skip'" unless TypesSupport::EXISTING_POLICIES.include?(existing)

        super(
          formats: formats,
          output_dir: output_dir.nil? ? nil : TypesSupport.path(output_dir, field: "output_dir"),
          existing: TypesSupport.immutable(existing),
          profile_json: profile_json.nil? ? nil : TypesSupport.path(profile_json, field: "profile_json")
        )
      end
    end

    # Complete transcription configuration shared with the command-line interface.
    TranscriptionOptions = Data.define(
      :model,
      :model_revision,
      :adapter,
      :adapter_revision,
      :language,
      :text_only,
      :recursive,
      :device,
      :dtype,
      :audio_backend,
      :audio_memory_gb,
      :preprocess_workers,
      :pipeline_preparation,
      :vad,
      :vad_engine,
      :vad_batch_size,
      :vad_block_frames,
      :vad_threads,
      :vad_merge,
      :min_dur,
      :max_dur,
      :max_silence,
      :energy_threshold,
      :vad_threshold,
      :min_silence_ms,
      :speech_pad_ms,
      :batch_size,
      :batch_max_size,
      :batch_audio_seconds,
      :batch_vram_target,
      :adaptive_batch,
      :pin_memory,
      :max_new_tokens,
      :max_retry_tokens,
      :truncation_policy,
      :stop_repetition_loops,
      :alignment,
      :align_batch_size,
      :align_dtype,
      :max_chars,
      :max_cue_dur,
      :max_gap,
      :publication
    ) do
      def initialize(
        model: TypesSupport::DEFAULT_ASR_MODEL_ID,
        model_revision: nil,
        adapter: nil,
        adapter_revision: nil,
        language: "ar",
        text_only: false,
        recursive: true,
        device: "auto",
        dtype: "auto",
        audio_backend: "auto",
        audio_memory_gb: 4.0,
        preprocess_workers: nil,
        pipeline_preparation: true,
        vad: "silero",
        vad_engine: "auto",
        vad_batch_size: 16,
        vad_block_frames: 512,
        vad_threads: nil,
        vad_merge: false,
        min_dur: 0.5,
        max_dur: 30.0,
        max_silence: 0.6,
        energy_threshold: 50.0,
        vad_threshold: 0.5,
        min_silence_ms: 300,
        speech_pad_ms: 60,
        batch_size: nil,
        batch_max_size: nil,
        batch_audio_seconds: nil,
        batch_vram_target: 0.9,
        adaptive_batch: false,
        pin_memory: false,
        max_new_tokens: 445,
        max_retry_tokens: 896,
        truncation_policy: "retry",
        stop_repetition_loops: true,
        alignment: "segment",
        align_batch_size: 4,
        align_dtype: "fp32",
        max_chars: 80,
        max_cue_dur: 6.0,
        max_gap: 0.6,
        publication: nil
      )
        super(
          model: TypesSupport.reference(model),
          model_revision: TypesSupport.reference(model_revision),
          adapter: TypesSupport.reference(adapter),
          adapter_revision: TypesSupport.reference(adapter_revision),
          language: TypesSupport.immutable(language),
          text_only: text_only,
          recursive: recursive,
          device: TypesSupport.immutable(device),
          dtype: TypesSupport.immutable(dtype),
          audio_backend: TypesSupport.immutable(audio_backend),
          audio_memory_gb: audio_memory_gb,
          preprocess_workers: preprocess_workers,
          pipeline_preparation: pipeline_preparation,
          vad: TypesSupport.immutable(vad),
          vad_engine: TypesSupport.immutable(vad_engine),
          vad_batch_size: vad_batch_size,
          vad_block_frames: vad_block_frames,
          vad_threads: vad_threads,
          vad_merge: vad_merge,
          min_dur: min_dur,
          max_dur: max_dur,
          max_silence: max_silence,
          energy_threshold: energy_threshold,
          vad_threshold: vad_threshold,
          min_silence_ms: min_silence_ms,
          speech_pad_ms: speech_pad_ms,
          batch_size: batch_size,
          batch_max_size: batch_max_size,
          batch_audio_seconds: batch_audio_seconds,
          batch_vram_target: batch_vram_target,
          adaptive_batch: adaptive_batch,
          pin_memory: pin_memory,
          max_new_tokens: max_new_tokens,
          max_retry_tokens: max_retry_tokens,
          truncation_policy: TypesSupport.immutable(truncation_policy),
          stop_repetition_loops: stop_repetition_loops,
          alignment: TypesSupport.immutable(alignment),
          align_batch_size: align_batch_size,
          align_dtype: TypesSupport.immutable(align_dtype),
          max_chars: max_chars,
          max_cue_dur: max_cue_dur,
          max_gap: max_gap,
          publication: publication
        )
      end
    end

    # Text generated for one ASR input segment.
    TranscriptionSegment = Data.define(:index, :start, :end, :text) do
      def initialize(index:, start:, end:, text:)
        super(index: index, start: start, end: binding.local_variable_get(:end), text: TypesSupport.immutable(text))
      end
    end

    # One word with either CTC or approximate timing.
    TranscriptionWord = Data.define(
      :start,
      :end,
      :text,
      :segment_index,
      :segment_word_index,
      :timing_source
    ) do
      def initialize(start:, end:, text:, segment_index:, segment_word_index:, timing_source:)
        super(
          start: start,
          end: binding.local_variable_get(:end),
          text: TypesSupport.immutable(text),
          segment_index: segment_index,
          segment_word_index: segment_word_index,
          timing_source: TypesSupport.immutable(timing_source)
        )
      end
    end

    # One rendered subtitle cue.
    SubtitleCue = Data.define(:start, :end, :text) do
      def initialize(start:, end:, text:)
        super(start: start, end: binding.local_variable_get(:end), text: TypesSupport.immutable(text))
      end
    end

    # Per-file decoder, VAD, generation, and alignment provenance.
    TranscriptionProvenance = Data.define(
      :model_id,
      :model_revision,
      :model_format,
      :adapter_id,
      :adapter_revision,
      :decode_backend,
      :decode_fallback_reason,
      :vad_engine_requested,
      :vad_engine_actual,
      :vad_provider,
      :vad_fallback_reason,
      :fallback_alignment_segments,
      :repetition_stopped_segments,
      :truncation_retried_segments,
      :token_limit_segments,
      :generated_tokens_by_segment,
      :resumed_from_asr_checkpoint,
      :published
    ) do
      def initialize(
        model_id: nil,
        model_revision: nil,
        model_format: nil,
        adapter_id: nil,
        adapter_revision: nil,
        decode_backend: nil,
        decode_fallback_reason: nil,
        vad_engine_requested: nil,
        vad_engine_actual: nil,
        vad_provider: nil,
        vad_fallback_reason: nil,
        fallback_alignment_segments: 0,
        repetition_stopped_segments: [],
        truncation_retried_segments: [],
        token_limit_segments: [],
        generated_tokens_by_segment: [],
        resumed_from_asr_checkpoint: false,
        published: false
      )
        super(
          model_id: TypesSupport.immutable(model_id),
          model_revision: TypesSupport.immutable(model_revision),
          model_format: TypesSupport.immutable(model_format),
          adapter_id: TypesSupport.immutable(adapter_id),
          adapter_revision: TypesSupport.immutable(adapter_revision),
          decode_backend: TypesSupport.immutable(decode_backend),
          decode_fallback_reason: TypesSupport.immutable(decode_fallback_reason),
          vad_engine_requested: TypesSupport.immutable(vad_engine_requested),
          vad_engine_actual: TypesSupport.immutable(vad_engine_actual),
          vad_provider: TypesSupport.immutable(vad_provider),
          vad_fallback_reason: TypesSupport.immutable(vad_fallback_reason),
          fallback_alignment_segments: fallback_alignment_segments,
          repetition_stopped_segments: TypesSupport.tuple(
            repetition_stopped_segments,
            field: "repetition_stopped_segments"
          ),
          truncation_retried_segments: TypesSupport.tuple(
            truncation_retried_segments,
            field: "truncation_retried_segments"
          ),
          token_limit_segments: TypesSupport.tuple(token_limit_segments, field: "token_limit_segments"),
          generated_tokens_by_segment: TypesSupport.tuple(
            generated_tokens_by_segment,
            field: "generated_tokens_by_segment"
          ),
          resumed_from_asr_checkpoint: resumed_from_asr_checkpoint,
          published: published
        )
      end
    end

    # Immutable result for one expanded input audio file.
    TranscriptionResult = Data.define(
      :path,
      :relative_path,
      :status,
      :text,
      :duration,
      :segments,
      :words,
      :cues,
      :outputs,
      :error,
      :provenance
    ) do
      def initialize(
        path:,
        relative_path:,
        status:,
        text:,
        duration:,
        segments: [],
        words: [],
        cues: [],
        outputs: [],
        error: nil,
        provenance: TranscriptionProvenance.new
      )
        super(
          path: TypesSupport.immutable(path),
          relative_path: TypesSupport.immutable(relative_path),
          status: TypesSupport.immutable(status),
          text: TypesSupport.immutable(text),
          duration: duration,
          segments: TypesSupport.tuple(segments, field: "segments"),
          words: TypesSupport.tuple(words, field: "words"),
          cues: TypesSupport.tuple(cues, field: "cues"),
          outputs: TypesSupport.tuple(outputs, field: "outputs"),
          error: TypesSupport.immutable(error),
          provenance: provenance
        )
      end
    end

    # Stable high-level performance and resource statistics for one run.
    TranscriptionStatistics = Data.define(
      :elapsed_seconds,
      :successful_audio_seconds,
      :real_time_factor_x,
      :runtime_import_seconds,
      :serialization_wait_seconds,
      :input_validation_seconds,
      :decode_seconds,
      :vad_seconds,
      :asr_load_seconds,
      :asr_seconds,
      :aligner_load_seconds,
      :emissions_seconds,
      :viterbi_seconds,
      :peak_cuda_allocated_gib,
      :peak_cuda_reserved_gib,
      :asr_batches,
      :asr_processor_rows,
      :generated_tokens,
      :oom_retries,
      :truncation_retries
    ) do
      def initialize(
        elapsed_seconds:,
        successful_audio_seconds:,
        real_time_factor_x:,
        runtime_import_seconds:,
        serialization_wait_seconds:,
        input_validation_seconds:,
        decode_seconds:,
        vad_seconds:,
        asr_load_seconds:,
        asr_seconds:,
        aligner_load_seconds:,
        emissions_seconds:,
        viterbi_seconds:,
        peak_cuda_allocated_gib:,
        peak_cuda_reserved_gib:,
        asr_batches:,
        asr_processor_rows:,
        generated_tokens:,
        oom_retries:,
        truncation_retries:
      )
        super
      end
    end

    # A serialized message or bounded progress update from a run.
    ProgressEvent = Data.define(:stage, :message, :current, :total) do
      def initialize(stage:, message: nil, current: nil, total: nil)
        super(
          stage: TypesSupport.immutable(stage),
          message: TypesSupport.immutable(message),
          current: current,
          total: total
        )
      end
    end

    # Immutable, sequence-like result of one API call.
    TranscriptionRun = Data.define(
      :results,
      :requested_options,
      :resolved_options,
      :statistics,
      :errors
    ) do
      include Enumerable

      def initialize(results:, requested_options:, resolved_options:, statistics:, errors: [])
        super(
          results: TypesSupport.tuple(results, field: "results"),
          requested_options: requested_options,
          resolved_options: resolved_options,
          statistics: statistics,
          errors: TypesSupport.tuple(errors, field: "errors")
        )
      end

      def each(&block)
        return results.each unless block

        results.each(&block)
      end

      # Enumerable#to_h otherwise shadows Data#to_h and interprets results as
      # key-value pairs instead of returning this value object's fields.
      def to_h
        {
          results: results,
          requested_options: requested_options,
          resolved_options: resolved_options,
          statistics: statistics,
          errors: errors
        }
      end

      def length
        results.length
      end
      alias_method :size, :length

      def [](*)
        selected = results[*]
        selected.is_a?(Array) ? selected.freeze : selected
      end

      def single
        unless results.length == 1
          raise ArgumentError,
                "Expected exactly one expanded audio file, found #{results.length}"
        end

        results.first
      end

      def successful
        results.select { |result| result.status == "completed" }.freeze
      end

      def failed
        results.select { |result| result.status == "failed" }.freeze
      end

      def skipped
        results.select { |result| result.status == "skipped" }.freeze
      end

      def ok?
        failed.empty? && errors.empty?
      end
      alias_method :ok, :ok?
    end

    [
      PublicationOptions,
      TranscriptionOptions,
      TranscriptionSegment,
      TranscriptionWord,
      SubtitleCue,
      TranscriptionProvenance,
      TranscriptionResult,
      TranscriptionStatistics,
      ProgressEvent,
      TranscriptionRun
    ].each do |value_class|
      value_class.singleton_class.prepend(TypesSupport::KeywordOnlyConstructor)
    end
  end
end
