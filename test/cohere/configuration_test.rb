# frozen_string_literal: true

require "test_helper"
require "tmpdir"

module Cohere
  module Transcribe
    class ConfigurationTest < Minitest::Test
      def test_defaults_are_valid
        options = TranscriptionOptions.new

        assert_same options, Configuration.validate!(options)
      end

      def test_pin_memory_is_valid_but_resolves_off_for_the_native_ggml_runtime
        requested = TranscriptionOptions.new(pin_memory: true)
        resolved = Configuration.resolved(requested)

        assert requested.pin_memory
        refute resolved.pin_memory
      end

      def test_requires_transcription_options
        error = assert_raises(TypeError) { Configuration.validate!({}) }

        assert_match(/TranscriptionOptions/, error.message)
      end

      def test_choice_options_require_exact_strings_and_reject_symbols
        Configuration::CHOICES.each do |field, choices|
          choices.each do |choice|
            options = TranscriptionOptions.new.with(**{ field => choice })
            assert_same options, Configuration.validate!(options), "#{field}=#{choice.inspect}"
          end

          error = assert_raises(TranscriptionConfigurationError, field.to_s) do
            Configuration.validate!(TranscriptionOptions.new.with(**{ field => choices.first.to_sym }))
          end
          assert_match(/--#{field.to_s.tr("_", "-")} must be one of:/, error.message)
        end
      end

      def test_choice_options_reject_unknown_strings
        Configuration::CHOICES.each_key do |field|
          error = assert_raises(TranscriptionConfigurationError) do
            Configuration.validate!(TranscriptionOptions.new.with(**{ field => "unknown" }))
          end
          assert_match(/--#{field.to_s.tr("_", "-")} must be one of:/, error.message)
        end
      end

      def test_integer_options_reject_booleans_and_floats
        Configuration::INTEGER_OPTIONS.each do |field|
          [true, false, 1.5].each do |invalid|
            changes = { field => invalid }
            changes[:adaptive_batch] = true if field == :batch_max_size
            changes[:alignment] = "word" if field == :align_batch_size

            error = assert_raises(TranscriptionConfigurationError, "#{field}=#{invalid.inspect}") do
              Configuration.validate!(TranscriptionOptions.new.with(**changes))
            end
            assert_match(/--#{field.to_s.tr("_", "-")} must be an integer/, error.message)
          end
        end
      end

      def test_real_options_reject_booleans_strings_and_complex_values
        Configuration::REAL_OPTIONS.each do |field|
          [true, "1.0", Complex(1, 0)].each do |invalid|
            error = assert_raises(TranscriptionConfigurationError, "#{field}=#{invalid.inspect}") do
              Configuration.validate!(TranscriptionOptions.new.with(**{ field => invalid }))
            end
            assert_match(/--#{field.to_s.tr("_", "-")} must be a real number/, error.message)
          end
        end
      end

      def test_boolean_options_require_true_or_false
        Configuration::BOOLEAN_OPTIONS.each do |field|
          [nil, 0, "false"].each do |invalid|
            error = assert_raises(TranscriptionConfigurationError, "#{field}=#{invalid.inspect}") do
              Configuration.validate!(TranscriptionOptions.new.with(**{ field => invalid }))
            end
            assert_match(/--#{field.to_s.tr("_", "-")} must be a boolean/, error.message)
          end
        end
      end

      def test_packed_vad_limits_and_engine_specific_threads
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad_batch_size: 65, vad_block_frames: 512))
        end
        assert_same(
          options = TranscriptionOptions.new(vad_engine: "onnx", vad_batch_size: 0, vad_block_frames: 0),
          Configuration.validate!(options)
        )

        error = assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad_engine: "onnx", vad_threads: 1))
        end
        assert_match(/packed Torch/, error.message)
      end

      def test_vad_specific_validation_and_inactive_values
        assert_same(
          options = TranscriptionOptions.new(
            vad: "none", min_dur: -10.0, vad_threshold: -1.0,
            min_silence_ms: -1, speech_pad_ms: -1, max_dur: 1.0
          ),
          Configuration.validate!(options)
        )

        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad_threshold: 1.01))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad: "auditok", min_dur: 0.0))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad: "auditok", max_dur: 0.6, max_silence: 0.6))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(vad: "none", max_dur: 0.999))
        end
      end

      def test_batch_and_generation_relations
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(batch_size: 4, batch_max_size: 3, adaptive_batch: true))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(batch_max_size: 4))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(max_new_tokens: 500, max_retry_tokens: 499))
        end

        [0.50, 0.98].each do |target|
          options = TranscriptionOptions.new(batch_vram_target: target)
          assert_same options, Configuration.validate!(options)
        end
      end

      def test_contextually_required_integer_values_reject_nil_cleanly
        invalid_options = [
          TranscriptionOptions.new(vad_batch_size: nil),
          TranscriptionOptions.new(vad_block_frames: nil),
          TranscriptionOptions.new(min_silence_ms: nil),
          TranscriptionOptions.new(speech_pad_ms: nil),
          TranscriptionOptions.new(max_new_tokens: nil),
          TranscriptionOptions.new(max_retry_tokens: nil),
          TranscriptionOptions.new(alignment: "word", align_batch_size: nil),
          TranscriptionOptions.new(max_chars: nil)
        ]

        invalid_options.each do |options|
          assert_raises(TranscriptionConfigurationError, options.to_h.inspect) do
            Configuration.validate!(options)
          end
        end

        inactive = TranscriptionOptions.new(
          vad: "none",
          max_dur: 1.0,
          vad_batch_size: nil,
          vad_block_frames: nil,
          min_silence_ms: nil,
          speech_pad_ms: nil,
          alignment: "none",
          align_batch_size: nil,
          max_chars: nil
        )
        assert_same inactive, Configuration.validate!(inactive)
      end

      def test_non_finite_auditok_thresholds_use_the_shared_numeric_error
        {
          Float::NAN => /All numeric thresholds/,
          Float::INFINITY => /max-silence < --max-dur/,
          -Float::INFINITY => /Auditok silence/
        }.each do |invalid, message|
          error = assert_raises(TranscriptionConfigurationError) do
            Configuration.validate!(
              TranscriptionOptions.new(vad: "auditok", max_silence: invalid)
            )
          end
          assert_match(message, error.message)
        end
      end

      def test_plain_text_mode_and_publication_formats
        conflict = TranscriptionOptions.new(text_only: true, alignment: "word")
        assert_raises(TranscriptionConfigurationError) { Configuration.validate!(conflict) }

        invalid = TranscriptionOptions.new(
          alignment: "none",
          publication: PublicationOptions.new(formats: %w[txt json])
        )
        assert_raises(TranscriptionConfigurationError) { Configuration.validate!(invalid) }

        resolved = Configuration.resolved(TranscriptionOptions.new(publication: PublicationOptions.new))
        assert_equal %w[txt srt vtt], resolved.publication.formats

        text = Configuration.resolved(
          TranscriptionOptions.new(text_only: true, publication: PublicationOptions.new)
        )
        assert_equal "none", text.alignment
        assert_equal ["txt"], text.publication.formats
      end

      def test_model_references_distinguish_repository_ids_and_local_paths
        assert_same(
          options = TranscriptionOptions.new(model: "owner/model"),
          Configuration.validate!(options)
        )
        assert_same(
          options = TranscriptionOptions.new(model: "model-name"),
          Configuration.validate!(options)
        )
        assert_same(
          options = TranscriptionOptions.new(model: "مالك/نموذج²"),
          Configuration.validate!(options)
        )
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(model: "owner/model/extra"))
        end
        [".hidden", "owner/model--bad", "owner/model.git", "owner/e\u0301"].each do |reference|
          assert_raises(TranscriptionConfigurationError) do
            Configuration.validate!(TranscriptionOptions.new(model: reference))
          end
        end

        Dir.mktmpdir do |directory|
          options = TranscriptionOptions.new(model: Pathname(directory))
          assert_same options, Configuration.validate!(options)

          error = assert_raises(TranscriptionConfigurationError) do
            Configuration.validate!(options.with(model_revision: "main"))
          end
          assert_match(/local model directory/, error.message)
        end
      end

      def test_unresolvable_named_tilde_path_has_a_typed_configuration_message
        error = assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(
            TranscriptionOptions.new(model: "~definitely-no-such-user-ct/model")
          )
        end

        assert_match(/Cannot resolve model path/, error.message)
      end

      def test_path_like_model_must_resolve_to_text
        path_like = Object.new
        path_like.define_singleton_method(:to_path) { 123 }

        error = assert_raises(TypeError) do
          Configuration.validate!(TranscriptionOptions.new(model: path_like))
        end
        assert_match(/resolve to text/, error.message)
      end

      def test_false_is_not_treated_as_an_absent_optional_object
        assert_raises(TypeError) do
          Configuration.validate!(TranscriptionOptions.new(adapter: false))
        end
        assert_raises(TranscriptionConfigurationError) do
          Configuration.validate!(TranscriptionOptions.new(publication: false))
        end
      end

      def test_model_and_revision_validation_use_python_unicode_whitespace
        codepoints = [
          *(0x0009..0x000D), *(0x001C..0x0020), 0x0085, 0x00A0, 0x1680,
          *(0x2000..0x200A), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000
        ]
        codepoints.each do |codepoint|
          whitespace = codepoint.chr(Encoding::UTF_8)
          label = format("U+%04X", codepoint)

          error = assert_raises(TranscriptionConfigurationError, label) do
            Configuration.validate!(TranscriptionOptions.new(model: whitespace))
          end
          assert_match(/--model must be a non-empty/, error.message)

          error = assert_raises(TranscriptionConfigurationError, label) do
            Configuration.validate!(TranscriptionOptions.new(model_revision: whitespace))
          end
          assert_match(/--model-revision must be a non-empty revision/, error.message)

          error = assert_raises(TranscriptionConfigurationError, label) do
            Configuration.validate!(TranscriptionOptions.new(adapter: whitespace))
          end
          assert_match(/--adapter must be a non-empty/, error.message)

          error = assert_raises(TranscriptionConfigurationError, label) do
            Configuration.validate!(
              TranscriptionOptions.new(adapter: "owner/adapter", adapter_revision: whitespace)
            )
          end
          assert_match(/--adapter-revision must be a non-empty revision/, error.message)
        end

        zero_width_revision = "rev\u200B"
        options = TranscriptionOptions.new(model_revision: zero_width_revision)
        assert_same options, Configuration.validate!(options)
      end
    end
  end
end
