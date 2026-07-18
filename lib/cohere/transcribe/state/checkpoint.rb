# frozen_string_literal: true

require "securerandom"

module Cohere
  module Transcribe
    module State
      Checkpoint = Data.define(
        :generation_id,
        :duration,
        :segment_times,
        :speech_spans,
        :segment_texts,
        :generated_tokens_by_segment,
        :repetition_stopped_segments,
        :truncation_retried_segments,
        :token_limit_segments,
        :decode_backend,
        :decode_fallback_reason,
        :vad_engine_actual,
        :vad_provider,
        :vad_provider_options,
        :vad_fallback_reason
      )
      CheckpointRestore = Data.define(:checkpoint, :reason) do
        def restored?
          !checkpoint.nil?
        end
      end

      module_function

      def write_asr_checkpoint(path:, result:, source_snapshot:, asr_contract_key:,
                               speech_spans:, vad_provider_options: nil, generation_id: nil,
                               directory_binding: nil, guard_bindings: nil, lock: nil)
        generation_id = generation_id.to_s
        generation_id = SecureRandom.hex(16) if generation_id.empty?
        payload = asr_checkpoint_payload(
          result: result,
          source_snapshot: source_snapshot,
          asr_contract_key: asr_contract_key,
          speech_spans: speech_spans,
          vad_provider_options: vad_provider_options,
          generation_id: generation_id
        )
        write_state_atomic(
          path,
          payload,
          source_snapshot: source_snapshot,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings,
          commit_guard: lock&.method(:verify!)
        )
        generation_id.freeze
      end

      def asr_checkpoint_payload(result:, source_snapshot:, asr_contract_key:, speech_spans:,
                                 vad_provider_options:, generation_id:)
        provenance = result.provenance
        {
          "kind" => "asr_complete",
          "generation_id" => generation_id,
          "asr_contract_key" => asr_contract_key,
          "source" => source_snapshot.payload,
          "updated_unix_seconds" => Time.now.to_f,
          "checkpoint" => {
            "duration" => result.duration,
            "segment_times" => result.segments.map { |segment| [segment.start, segment.end] },
            "speech_spans" => speech_spans.map { |start_time, end_time| [start_time, end_time] },
            "segment_texts" => result.segments.map(&:text),
            "generated_tokens" => provenance.generated_tokens_by_segment.sort,
            "repetition_stopped_segments" => provenance.repetition_stopped_segments.sort,
            "truncation_retried_segments" => provenance.truncation_retried_segments.sort,
            "token_limit_segments" => provenance.token_limit_segments.sort,
            "decode_backend" => provenance.decode_backend,
            "decode_fallback_reason" => provenance.decode_fallback_reason,
            "vad_engine_actual" => provenance.vad_engine_actual,
            "vad_provider" => provenance.vad_provider,
            "vad_provider_options" => vad_provider_options,
            "vad_fallback_reason" => provenance.vad_fallback_reason
          }
        }
      end

      def restore_asr_checkpoint(path:, source_snapshot:, asr_contract_key:,
                                 directory_binding: nil, guard_bindings: nil)
        payload, reason = decode_state(
          path,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        )
        return CheckpointRestore.new(checkpoint: nil, reason: reason.freeze) unless payload
        unless payload["kind"] == "asr_complete"
          return CheckpointRestore.new(
            checkpoint: nil,
            reason: "state is #{payload["kind"].inspect}, not an ASR checkpoint".freeze
          )
        end
        unless payload["asr_contract_key"] == asr_contract_key
          return CheckpointRestore.new(checkpoint: nil, reason: "ASR checkpoint contract does not match")
        end
        unless payload["source"] == source_snapshot.payload
          return CheckpointRestore.new(checkpoint: nil, reason: "state marker source snapshot does not match")
        end

        CheckpointRestore.new(checkpoint: validate_checkpoint(payload).freeze, reason: nil)
      rescue TypeError, ArgumentError => e
        CheckpointRestore.new(checkpoint: nil, reason: "ASR checkpoint is invalid (#{e.message})".freeze)
      end

      def validate_checkpoint(payload)
        generation_id = payload["generation_id"]
        validate!(generation_id.is_a?(String) && !generation_id.empty?, "generation ID is invalid")
        checkpoint = payload["checkpoint"]
        validate!(checkpoint.is_a?(Hash), "checkpoint is not an object")

        duration = finite_number(checkpoint["duration"], "duration")
        validate!(duration >= 0, "duration is invalid")
        segment_times = validated_spans(checkpoint["segment_times"], duration, "segment_times")
        speech_spans = validated_spans(checkpoint["speech_spans"], duration, "speech_spans")
        texts = checkpoint["segment_texts"]
        validate!(texts.is_a?(Array) && texts.all?(String), "segment_texts is invalid")
        validate!(texts.length == segment_times.length, "segment text/time counts differ")
        texts = texts.map { |text| text.dup.freeze }.freeze
        segment_count = segment_times.length

        generated_tokens = validated_generated_tokens(checkpoint.fetch("generated_tokens", []), segment_count)
        repetition = validated_indices(checkpoint.fetch("repetition_stopped_segments", []), segment_count,
                                       "repetition_stopped_segments")
        truncation = validated_indices(checkpoint.fetch("truncation_retried_segments", []), segment_count,
                                       "truncation_retried_segments")
        token_limit = validated_indices(checkpoint.fetch("token_limit_segments", []), segment_count,
                                        "token_limit_segments")
        optional_names = %w[
          decode_backend decode_fallback_reason vad_engine_actual vad_provider vad_fallback_reason
        ]
        optional_names.each do |name|
          value = checkpoint[name]
          validate!(value.nil? || value.is_a?(String), "checkpoint provenance strings are invalid")
        end
        provider_options = checkpoint["vad_provider_options"]
        validate!(provider_options.nil? || provider_options.is_a?(Hash), "vad_provider_options is invalid")

        Checkpoint.new(
          generation_id: generation_id.dup.freeze,
          duration: duration,
          segment_times: segment_times,
          speech_spans: speech_spans,
          segment_texts: texts,
          generated_tokens_by_segment: generated_tokens,
          repetition_stopped_segments: repetition,
          truncation_retried_segments: truncation,
          token_limit_segments: token_limit,
          decode_backend: immutable_optional_string(checkpoint["decode_backend"]),
          decode_fallback_reason: immutable_optional_string(checkpoint["decode_fallback_reason"]),
          vad_engine_actual: immutable_optional_string(checkpoint["vad_engine_actual"]),
          vad_provider: immutable_optional_string(checkpoint["vad_provider"]),
          vad_provider_options: deep_freeze_json(provider_options),
          vad_fallback_reason: immutable_optional_string(checkpoint["vad_fallback_reason"])
        )
      end

      def validated_spans(value, duration, name)
        validate!(value.is_a?(Array), "#{name} is not a list")
        previous_end = nil
        value.map do |row|
          validate!(row.is_a?(Array) && row.length == 2, "#{name} contains an invalid row")
          start_time = finite_number(row[0], name)
          end_time = finite_number(row[1], name)
          valid = start_time.between?(0, end_time) && end_time <= duration + 1e-6
          valid &&= previous_end.nil? || start_time >= previous_end
          validate!(valid, "#{name} contains an overlapping, out-of-order, or out-of-range row")
          previous_end = end_time
          [start_time, [end_time, duration].min].freeze
        end.freeze
      end

      def validated_generated_tokens(value, segment_count)
        validate!(value.is_a?(Array), "generated_tokens is invalid")
        seen = {}
        value.map do |row|
          valid = row.is_a?(Array) && row.length == 2 && row.all?(Integer)
          valid &&= row[0].between?(0, segment_count - 1) && row[1] >= 0
          valid &&= !seen.key?(row[0])
          validate!(valid, "generated_tokens contains an invalid or duplicate row")
          seen[row[0]] = true
          [row[0], row[1]].freeze
        end.freeze
      end

      def validated_indices(value, segment_count, name)
        validate!(value.is_a?(Array), "#{name} is invalid")
        valid = value.all? { |index| index.is_a?(Integer) && index.between?(0, segment_count - 1) }
        validate!(valid && value.uniq.length == value.length, "#{name} is invalid or contains duplicate indices")
        value.dup.freeze
      end

      def finite_number(value, name)
        validate!(value.is_a?(Numeric) && !value.is_a?(Complex) && value.finite?, "#{name} is not finite numeric data")
        value.to_f
      end

      def validate!(condition, message)
        raise ArgumentError, message unless condition
      end

      def immutable_optional_string(value)
        value&.dup&.freeze
      end

      def deep_freeze_json(value)
        case value
        when Hash
          value.to_h { |key, item| [key.to_s.freeze, deep_freeze_json(item)] }.freeze
        when Array
          value.map { |item| deep_freeze_json(item) }.freeze
        when String
          value.dup.freeze
        else
          value
        end
      end
    end
  end
end
