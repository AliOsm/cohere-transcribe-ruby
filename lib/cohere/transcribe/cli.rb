# frozen_string_literal: true

require "optparse"
require_relative "constants"
require_relative "errors"
require_relative "version"

module Cohere
  module Transcribe
    # Command-line adapter for the dependency-light public API.
    module CLI
      DEFAULT_MODEL_ID = DEFAULT_ASR_MODEL_ID
      OUTPUT_FORMATS = Cohere::Transcribe::OUTPUT_FORMATS
      OUTPUT_PATH_DISPLAY_LIMIT = 20
      FORMAT_ARGUMENT_SEPARATOR = "\u001F"
      private_constant :FORMAT_ARGUMENT_SEPARATOR
      VALUE_OPTIONS = %w[
        model model-revision adapter adapter-revision language output-dir existing
        device dtype audio-backend audio-memory-gb preprocess-workers vad vad-engine
        vad-batch-size vad-block-frames vad-threads min-dur max-dur max-silence
        energy-threshold vad-threshold min-silence-ms speech-pad-ms batch-size
        batch-max-size batch-audio-seconds batch-vram-target max-new-tokens
        max-retry-tokens truncation-policy alignment align-batch-size align-dtype
        max-chars max-cue-dur max-gap profile-json
      ].freeze
      REAL_VALUE_OPTIONS = %w[
        audio-memory-gb min-dur max-dur max-silence energy-threshold vad-threshold
        batch-audio-seconds batch-vram-target max-cue-dur max-gap
      ].freeze
      INTEGER_VALUE_OPTIONS = %w[
        preprocess-workers vad-batch-size vad-block-frames vad-threads min-silence-ms
        speech-pad-ms batch-size batch-max-size max-new-tokens max-retry-tokens
        align-batch-size max-chars
      ].freeze
      FLAG_OPTIONS = %w[
        recursive no-recursive pipeline-preparation no-pipeline-preparation vad-merge
        no-vad-merge adaptive-batch no-adaptive-batch pin-memory no-pin-memory
        stop-repetition-loops no-stop-repetition-loops text-only version help
      ].freeze
      private_constant :VALUE_OPTIONS, :REAL_VALUE_OPTIONS, :INTEGER_VALUE_OPTIONS, :FLAG_OPTIONS

      ParsedCommand = Data.define(:audio, :options)

      class EarlyExit < StandardError
        attr_reader :status

        def initialize(status = 0)
          @status = status
          super()
        end
      end

      module_function

      def parse_args(argv = ARGV, out: $stdout)
        values = default_values
        state = { alignment: false, text_only: false }
        raw_arguments = Array(argv).dup
        preflight_mutually_exclusive_options!(raw_arguments)
        preflight_reference_numeric_tokens!(raw_arguments)
        preflight_ambiguous_long_options!(raw_arguments)
        parser_arguments = defer_unknown_options_before_early_exit(raw_arguments)
        arguments = normalize_formats_arguments(parser_arguments)
        parser = option_parser(values, state, out: out)
        parser.parse!(arguments)
        enforce_reference_positional_order!(raw_arguments)
        raise OptionParser::MissingArgument, "audio" if arguments.empty?
        raise OptionParser::InvalidOption, "--text-only conflicts with --alignment" if state.values_at(:alignment, :text_only).all?

        load_configuration_contract
        values[:alignment] = "none" if values[:text_only]
        text_mode = values[:alignment] == "none"
        formats = values.delete(:formats)
        formats = text_mode ? ["txt"] : %w[txt srt vtt] if formats.nil?
        formats = formats.uniq
        raise TranscriptionConfigurationError, "Plain-text mode supports only --formats txt" if text_mode && formats != ["txt"]

        publication = PublicationOptions.new(
          formats: formats,
          output_dir: values.delete(:output_dir),
          existing: values.delete(:existing),
          profile_json: values.delete(:profile_json)
        )
        options = TranscriptionOptions.new(**values, publication: publication)
        Configuration.validate!(options)
        ParsedCommand.new(audio: arguments.map { |value| value.dup.freeze }.freeze, options: options)
      rescue ArgumentError => e
        raise if defined?(TranscriptionError) && e.is_a?(TranscriptionError)

        raise OptionParser::InvalidArgument, e.message
      end

      def main(argv = ARGV, out: $stdout, err: $stderr, transcriber: nil)
        command = parse_args(argv, out: out)
        load_public_api unless transcriber
        unless transcriber || Cohere::Transcribe.respond_to?(:transcribe)
          raise TranscriptionRuntimeError, "The public transcription runtime is unavailable"
        end

        callable = transcriber || Cohere::Transcribe.method(:transcribe)

        out.puts("\n[1/4] Validating inputs and outputs")
        stage_two_printed = false
        print_stage_two = lambda do
          next if stage_two_printed

          out.puts("\n[2/4] Loading ASR + preparing audio")
          stage_two_printed = true
        end
        run = callable.call(
          command.audio,
          options: command.options,
          progress: progress_reporter(out, before_first: print_stage_two),
          raise_on_error: false
        )
        if all_skipped?(run)
          run.skipped.each do |result|
            out.puts("    skipping #{result.path}: verified output generation is complete")
          end
          out.puts("    All inputs were skipped; no model was loaded.")
          run.errors.each { |error| out.puts("    [error] #{error}") }
          command.options.publication.profile_json&.then { |path| out.puts("    #{path}") }
          return run.ok? ? 0 : 1
        end

        print_stage_two.call
        out.puts(
          command.options.alignment == "word" ? "\n[3/4] Forced alignment + transactional outputs" : "\n[3/4] Transactional outputs"
        )
        out.puts("\n[4/4] Summary")
        print_summary(run, out: out)
        command.options.publication.profile_json&.then { |path| out.puts("    #{path}") }
        run.ok? ? 0 : 1
      rescue EarlyExit => e
        e.status
      rescue OptionParser::ParseError => e
        err.puts(option_parser(default_values, {}, out: out))
        err.puts("cohere-transcribe: error: #{e.message}")
        2
      rescue TranscriptionError => e
        err.puts(e.message)
        1
      rescue Interrupt
        out.puts(
          "\n    Interrupted; the active output commit was rolled back. " \
          "Files completed earlier remain published."
        )
        130
      rescue SignalException => e
        raise unless e.signm == "SIGTERM"

        out.puts(
          "\n    Termination requested; active work was cancelled and " \
          "completed files remain published."
        )
        143
      end

      # Console-entry alias matching the reference package's outer entry point.
      def cli(argv = ARGV, out: $stdout, err: $stderr, transcriber: nil)
        main(argv, out: out, err: err, transcriber: transcriber)
      end

      def option_parser(values, state, out:)
        OptionParser.new do |parser|
          parser.banner = "Usage: cohere-transcribe [options] audio [audio ...]"
          parser.separator("")
          parser.separator("Batch Arabic/English transcription with optional timestamp alignment.")
          parser.separator("")

          parser.on("--model MODEL", "Hugging Face repository or local native Cohere ASR directory.") do |value|
            values[:model] = value
          end
          parser.on("--model-revision REVISION", "Hub model commit, tag, or branch.") do |value|
            values[:model_revision] = value
          end
          parser.on("--adapter ADAPTER", "Optional Hub repository or local LoRA adapter directory.") do |value|
            values[:adapter] = value
          end
          parser.on("--adapter-revision REVISION", "Hub adapter commit, tag, or branch.") do |value|
            values[:adapter_revision] = value
          end
          parser.on("--language LANGUAGE", String, "Spoken language tag (ar or en).") do |value|
            values[:language] = exact_choice!("language", value, %w[ar en])
          end
          parser.on("--formats FORMAT [FORMAT ...]", String, "Outputs to write: txt, srt, vtt, json.") do |encoded|
            value = encoded.split(FORMAT_ARGUMENT_SEPARATOR, -1)
            unsupported = value - OUTPUT_FORMATS
            unless unsupported.empty?
              raise OptionParser::InvalidArgument,
                    "unsupported output format(s): #{unsupported.sort.join(", ")}"
            end
            values[:formats] = value
          end
          parser.on("--output-dir DIRECTORY", "Output root; preserve directory-relative structure.") do |value|
            values[:output_dir] = value
          end
          parser.on("--recursive", "Recurse into input directories (default).") { values[:recursive] = true }
          parser.on("--no-recursive", "Do not recurse into input directories.") { values[:recursive] = false }
          parser.on("--existing POLICY", String, "Existing outputs policy.") do |value|
            values[:existing] = exact_choice!("existing", value, %w[error overwrite skip])
          end
          parser.on("--device DEVICE", String, "Inference device.") do |value|
            values[:device] = exact_choice!("device", value, %w[auto mps cuda cpu])
          end
          parser.on("--dtype DTYPE", String, "ASR model precision.") do |value|
            values[:dtype] = exact_choice!("dtype", value, %w[auto bf16 fp16 fp32])
          end
          parser.on(
            "--audio-backend BACKEND",
            String,
            "Audio decoder configuration."
          ) { |value| values[:audio_backend] = exact_choice!("audio-backend", value, %w[auto torchcodec ffmpeg librosa]) }
          parser.on("--audio-memory-gb GIB", String, "Decoded-PCM memory limit per file/group.") do |value|
            values[:audio_memory_gb] = parse_reference_float(value)
          end
          parser.on("--preprocess-workers COUNT", String, "Concurrent audio decode workers.") do |value|
            values[:preprocess_workers] = parse_reference_integer(value)
          end
          parser.on("--pipeline-preparation", "Overlap bounded preparation with ASR (default).") do
            values[:pipeline_preparation] = true
          end
          parser.on("--no-pipeline-preparation", "Disable preparation/ASR overlap.") do
            values[:pipeline_preparation] = false
          end

          parser.separator("")
          parser.separator("Segmentation (VAD):")
          parser.on("--vad MODE", String, "Segmentation policy.") do |value|
            values[:vad] = exact_choice!("vad", value, %w[silero auditok none])
          end
          parser.on("--vad-engine ENGINE", String, "Silero runtime.") do |value|
            values[:vad_engine] = exact_choice!("vad-engine", value, %w[auto torch onnx jit])
          end
          parser.on("--vad-batch-size COUNT", String, "Maximum files per packed VAD call.") do |value|
            values[:vad_batch_size] = parse_reference_integer(value)
          end
          parser.on("--vad-block-frames COUNT", String, "Maximum frames per packed VAD file.") do |value|
            values[:vad_block_frames] = parse_reference_integer(value)
          end
          parser.on("--vad-threads COUNT", String, "Packed VAD CPU thread count.") do |value|
            values[:vad_threads] = parse_reference_integer(value)
          end
          parser.on("--vad-merge", "Merge adjacent Silero speech spans.") { values[:vad_merge] = true }
          parser.on("--no-vad-merge", "Do not merge adjacent Silero spans (default).") do
            values[:vad_merge] = false
          end
          parser.on("--min-dur SECONDS", String, "Minimum speech duration.") do |value|
            values[:min_dur] = parse_reference_float(value)
          end
          parser.on("--max-dur SECONDS", String, "Maximum segment/window duration.") do |value|
            values[:max_dur] = parse_reference_float(value)
          end
          parser.on("--max-silence SECONDS", String, "Maximum Auditok internal silence.") do |value|
            values[:max_silence] = parse_reference_float(value)
          end
          parser.on("--energy-threshold DB", String, "Auditok energy threshold.") do |value|
            values[:energy_threshold] = parse_reference_float(value)
          end
          parser.on("--vad-threshold PROBABILITY", String, "Silero speech threshold.") do |value|
            values[:vad_threshold] = parse_reference_float(value)
          end
          parser.on("--min-silence-ms MILLISECONDS", String, "Silero split silence.") do |value|
            values[:min_silence_ms] = parse_reference_integer(value)
          end
          parser.on("--speech-pad-ms MILLISECONDS", String, "Silero speech padding.") do |value|
            values[:speech_pad_ms] = parse_reference_integer(value)
          end

          parser.separator("")
          parser.separator("Transcription:")
          parser.on("--batch-size COUNT", String, "Initial ASR segment count.") do |value|
            values[:batch_size] = parse_reference_integer(value)
          end
          parser.on("--batch-max-size COUNT", String, "Adaptive batch upper row cap.") do |value|
            values[:batch_max_size] = parse_reference_integer(value)
          end
          parser.on("--batch-audio-seconds SECONDS", String, "Maximum padded audio per batch.") do |value|
            values[:batch_audio_seconds] = parse_reference_float(value)
          end
          parser.on("--batch-vram-target FRACTION", String, "Adaptive batching VRAM target.") do |value|
            values[:batch_vram_target] = parse_reference_float(value)
          end
          parser.on("--adaptive-batch", "Enable experimental adaptive batch growth.") do
            values[:adaptive_batch] = true
          end
          parser.on("--no-adaptive-batch", "Disable adaptive growth (default).") do
            values[:adaptive_batch] = false
          end
          parser.on("--pin-memory", "Compatibility flag; not applicable to native ggml.") do
            values[:pin_memory] = true
          end
          parser.on("--no-pin-memory", "Leave native ggml host buffers unpinned (default).") do
            values[:pin_memory] = false
          end
          parser.on("--max-new-tokens COUNT", String, "Initial decoder token limit.") do |value|
            values[:max_new_tokens] = parse_reference_integer(value)
          end
          parser.on("--max-retry-tokens COUNT", String, "Automatic retry token limit.") do |value|
            values[:max_retry_tokens] = parse_reference_integer(value)
          end
          parser.on("--truncation-policy POLICY", String, "Token-limit behavior.") do |value|
            values[:truncation_policy] = exact_choice!("truncation-policy", value, %w[retry warn])
          end
          parser.on("--stop-repetition-loops", "Stop conservative decoder repetition loops (default).") do
            values[:stop_repetition_loops] = true
          end
          parser.on("--no-stop-repetition-loops", "Disable the repetition-loop guard.") do
            values[:stop_repetition_loops] = false
          end

          parser.separator("")
          parser.separator("Alignment and subtitle cues:")
          parser.on("--alignment MODE", String, "Timestamp mode.") do |value|
            state[:alignment] = true
            values[:alignment] = exact_choice!("alignment", value, %w[word segment none])
          end
          parser.on("--text-only", "Alias for --alignment none.") do
            state[:text_only] = true
            values[:text_only] = true
          end
          parser.on("--align-batch-size COUNT", String, "Maximum alignment windows per batch.") do |value|
            values[:align_batch_size] = parse_reference_integer(value)
          end
          parser.on("--align-dtype DTYPE", String, "Alignment precision.") do |value|
            values[:align_dtype] = exact_choice!("align-dtype", value, %w[fp32 fp16])
          end
          parser.on("--max-chars COUNT", String, "Target subtitle cue length.") do |value|
            values[:max_chars] = parse_reference_integer(value)
          end
          parser.on("--max-cue-dur SECONDS", String, "Target subtitle cue duration.") do |value|
            values[:max_cue_dur] = parse_reference_float(value)
          end
          parser.on("--max-gap SECONDS", String, "Maximum inter-word cue gap.") do |value|
            values[:max_gap] = parse_reference_float(value)
          end
          parser.on("--profile-json PATH", "Write performance telemetry as JSON.") do |value|
            values[:profile_json] = value
          end
          parser.on("--version", "Show version and exit.") do
            out.puts("cohere-transcribe #{VERSION}")
            raise EarlyExit, 0
          end
          parser.on_tail("-h", "--help", "Show this help and exit.") do
            out.puts(parser)
            raise EarlyExit, 0
          end
        end
      end

      def default_values
        {
          model: DEFAULT_MODEL_ID,
          model_revision: nil,
          adapter: nil,
          adapter_revision: nil,
          language: "ar",
          formats: nil,
          output_dir: nil,
          recursive: true,
          existing: "error",
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
          text_only: false,
          align_batch_size: 4,
          align_dtype: "fp32",
          max_chars: 80,
          max_cue_dur: 6.0,
          max_gap: 0.6,
          profile_json: nil
        }
      end

      def normalize_formats_arguments(arguments)
        normalized = []
        index = 0
        while index < arguments.length
          argument = arguments[index]
          if argument == "--"
            normalized.concat(arguments[index..])
            break
          end
          option = canonical_long_option(argument)
          if argument == "-h" || %w[help version].include?(option)
            normalized.concat(arguments[index..])
            break
          end
          if option == "formats" && argument.include?("=")
            validate_raw_formats!([argument.split("=", 2).last])
            normalized << argument
            index += 1
            next
          end
          unless option == "formats" && !argument.include?("=")
            normalized << argument
            index += 1
            next
          end

          formats = []
          index += 1
          while index < arguments.length && (!arguments[index].start_with?("-") || arguments[index] == "-")
            formats << arguments[index]
            index += 1
          end
          validate_raw_formats!(formats) unless formats.empty?
          normalized << if formats.empty?
                          "--formats"
                        else
                          "--formats=#{formats.join(FORMAT_ARGUMENT_SEPARATOR)}"
                        end
        end
        normalized
      end
      private_class_method :normalize_formats_arguments

      def validate_raw_formats!(formats)
        unsupported = formats - OUTPUT_FORMATS
        return if unsupported.empty?

        raise OptionParser::InvalidArgument,
              "unsupported output formats: #{unsupported.sort.join(", ")}"
      end
      private_class_method :validate_raw_formats!

      def preflight_ambiguous_long_options!(arguments)
        arguments.each do |argument|
          break if argument == "--"
          next unless argument.start_with?("--") && argument.length > 2

          supplied = argument.delete_prefix("--").split("=", 2).first
          names = VALUE_OPTIONS + FLAG_OPTIONS + ["formats"]
          candidates = long_option_candidates(argument)
          canonical = if names.include?(supplied)
                        supplied
                      elsif candidates.one?
                        candidates.first
                      end
          break if %w[help version].include?(canonical)
          next if names.include?(supplied)

          raise OptionParser::AmbiguousOption, "--#{supplied}" if candidates.length > 1
        end
      end
      private_class_method :preflight_ambiguous_long_options!

      def preflight_reference_numeric_tokens!(arguments)
        index = 0
        while index < arguments.length
          argument = arguments[index]
          break if argument == "--"

          option = canonical_long_option(argument)
          return if argument == "-h" || %w[help version].include?(option)

          if VALUE_OPTIONS.include?(option) && !argument.include?("=")
            candidate = arguments[index + 1]
            negative_value = candidate&.match?(/\A-\p{Nd}+\z|\A-\p{Nd}*\.\p{Nd}+\z/)
            missing_value = candidate&.start_with?("-") && candidate != "-" && !negative_value
            raise OptionParser::MissingArgument, "--#{option}" if missing_value
          end

          if option == "formats" && !argument.include?("=")
            index += 1
            index += 1 while index < arguments.length && (!arguments[index].start_with?("-") || arguments[index] == "-")
            next
          end
          index += 1 if VALUE_OPTIONS.include?(option) && !argument.include?("=")
          index += 1
        end
      end
      private_class_method :preflight_reference_numeric_tokens!

      def preflight_mutually_exclusive_options!(arguments)
        alignment_seen = false
        text_only_seen = false
        index = 0
        while index < arguments.length
          argument = arguments[index]
          break if argument == "--"

          option = canonical_long_option(argument)
          return if argument == "-h" || %w[help version].include?(option)

          alignment_seen ||= option == "alignment"
          text_only_seen ||= option == "text-only"
          raise OptionParser::InvalidOption, "--text-only conflicts with --alignment" if alignment_seen && text_only_seen

          index += 1
        end
      end
      private_class_method :preflight_mutually_exclusive_options!

      def defer_unknown_options_before_early_exit(arguments)
        unknown_indices = []
        early_index = nil
        index = 0
        while index < arguments.length
          argument = arguments[index]
          break if argument == "--"

          option = canonical_long_option(argument)
          if argument == "-h" || %w[help version].include?(option)
            early_index = index
            break
          end
          if argument.start_with?("--") && long_option_candidates(argument).empty?
            unknown_indices << index
          elsif argument.start_with?("-") && argument != "-" && !argument.start_with?("--")
            unknown_indices << index
          end

          if option == "formats" && !argument.include?("=")
            index += 1
            index += 1 while index < arguments.length && (!arguments[index].start_with?("-") || arguments[index] == "-")
            next
          end
          index += 1 if VALUE_OPTIONS.include?(option) && !argument.include?("=")
          index += 1
        end
        return arguments.dup if early_index.nil? || unknown_indices.empty?

        deferred = unknown_indices.map { |unknown_index| arguments[unknown_index] }
        arguments.each_with_index.filter_map do |argument, argument_index|
          next if unknown_indices.include?(argument_index)

          argument_index == early_index ? [argument, *deferred] : argument
        end.flatten
      end
      private_class_method :defer_unknown_options_before_early_exit

      def exact_choice!(option, value, choices)
        return value if choices.include?(value)

        raise OptionParser::InvalidArgument,
              "--#{option}: invalid choice #{value.inspect}; choose from #{choices.join(", ")}"
      end
      private_class_method :exact_choice!

      def parse_reference_float(value)
        text = normalize_reference_numeric_text(value)
        special = text.match?(/\A[+-]?(?:inf(?:inity)?|nan)\z/i)
        digits = "[0-9](?:_?[0-9])*"
        decimal = /\A[+-]?(?:(?:#{digits})(?:\.(?:#{digits})?)?|\.(?:#{digits}))(?:[eE][+-]?#{digits})?\z/
        raise ArgumentError, "invalid float value: #{value.inspect}" unless special || text.match?(decimal)

        case text.downcase
        when "inf", "+inf", "infinity", "+infinity"
          Float::INFINITY
        when "-inf", "-infinity"
          -Float::INFINITY
        when "nan", "+nan", "-nan"
          Float::NAN
        else
          Float(text.delete("_"))
        end
      end
      private_class_method :parse_reference_float

      def parse_reference_integer(value)
        text = normalize_reference_numeric_text(value)
        digits = "[0-9](?:_?[0-9])*"
        raise ArgumentError, "invalid integer value: #{value.inspect}" unless text.match?(/\A[+-]?#{digits}\z/)

        Integer(text.delete("_"), 10)
      end
      private_class_method :parse_reference_integer

      def normalize_reference_numeric_text(value)
        text = value.gsub(/\A[[:space:]]+|[[:space:]]+\z/, "")
        text.each_char.map do |character|
          next character unless character.match?(/\p{Nd}/)

          codepoint = character.ord
          first = codepoint
          while first.positive?
            previous = (first - 1).chr(Encoding::UTF_8)
            break unless previous.match?(/\p{Nd}/)

            first -= 1
          end
          ((codepoint - first) % 10).to_s
        end.join
      end
      private_class_method :normalize_reference_numeric_text

      def enforce_reference_positional_order!(arguments)
        positional_started = false
        option_after_positional = false
        index = 0
        while index < arguments.length
          argument = arguments[index]
          if argument == "--"
            detail = arguments[index + 1] || "--"
            raise OptionParser::InvalidArgument, "unrecognized arguments: #{detail}" if option_after_positional

            break
          end

          option = canonical_long_option(argument)
          if option
            option_after_positional ||= positional_started
            if option == "formats" && !argument.include?("=")
              index += 1
              index += 1 while index < arguments.length && (!arguments[index].start_with?("-") || arguments[index] == "-")
              next
            end
            index += 1 if VALUE_OPTIONS.include?(option) && !argument.include?("=")
          elsif argument.start_with?("-") && argument != "-"
            # A successful OptionParser pass means this is either the short help
            # flag (which exits before here) or a literal positional dash.
          else
            raise OptionParser::InvalidArgument, "unrecognized arguments: #{argument}" if option_after_positional

            positional_started = true
          end
          index += 1
        end
      end
      private_class_method :enforce_reference_positional_order!

      def canonical_long_option(argument)
        return nil unless argument.start_with?("--") && argument.length > 2

        supplied = argument.delete_prefix("--").split("=", 2).first
        names = VALUE_OPTIONS + FLAG_OPTIONS + ["formats"]
        return supplied if names.include?(supplied)

        candidates = long_option_candidates(argument)
        candidates.one? ? candidates.first : nil
      end
      private_class_method :canonical_long_option

      def long_option_candidates(argument)
        return [] unless argument.start_with?("--") && argument.length > 2

        supplied = argument.delete_prefix("--").split("=", 2).first
        (VALUE_OPTIONS + FLAG_OPTIONS + ["formats"]).select { |name| name.start_with?(supplied) }
      end
      private_class_method :long_option_candidates

      def load_configuration_contract
        require_relative "constants"
        require_relative "errors"
        require_relative "types"
        require_relative "model_identity"
        require_relative "configuration"
      end
      private_class_method :load_configuration_contract

      def load_public_api
        return if Cohere::Transcribe.respond_to?(:transcribe)

        require "cohere/transcribe"
      end
      private_class_method :load_public_api

      def progress_reporter(out, before_first: nil)
        lambda do |event|
          before_first&.call
          if event.message
            out.puts(event.message)
          elsif !event.current.nil? && !event.total.nil? && event.current == event.total
            out.puts("    #{event.stage}: #{event.current}/#{event.total}")
          end
        end
      end
      private_class_method :progress_reporter

      def all_skipped?(run)
        !run.results.empty? && run.skipped.length == run.results.length
      end
      private_class_method :all_skipped?

      def print_summary(run, out:)
        attempted = run.successful.length + run.failed.length
        statistics = run.statistics
        out.puts(
          "    #{run.successful.length}/#{attempted} files finished in " \
          "#{format_duration(statistics.elapsed_seconds)} " \
          "(RTFx #{format("%.1f", statistics.real_time_factor_x)})"
        )
        written = run.flat_map(&:outputs)
        written.first(OUTPUT_PATH_DISPLAY_LIMIT).each { |path| out.puts("    #{path}") }
        if written.length > OUTPUT_PATH_DISPLAY_LIMIT
          out.puts("    ... and #{written.length - OUTPUT_PATH_DISPLAY_LIMIT} more output files")
        end
        unless run.failed.empty?
          out.puts("    Failures:")
          run.failed.each { |result| out.puts("    - #{result.path}: #{result.error}") }
        end
        run.errors.each { |error| out.puts("    [error] #{error}") }
      end
      private_class_method :print_summary

      def format_duration(seconds)
        rounded = [0, seconds.to_f.round].max
        hours, remainder = rounded.divmod(3600)
        minutes, remaining_seconds = remainder.divmod(60)
        return format("%dh%02dm%02ds", hours, minutes, remaining_seconds) if hours.positive?
        return format("%dm%02ds", minutes, remaining_seconds) if minutes.positive?

        "#{remaining_seconds}s"
      end
      private_class_method :format_duration
    end
  end
end
