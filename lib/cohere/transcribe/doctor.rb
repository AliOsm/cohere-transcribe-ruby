# frozen_string_literal: true

require "optparse"
require_relative "constants"
require_relative "version"

module Cohere
  module Transcribe
    # Installation and model-metadata diagnostics that never load ASR weights.
    module Doctor
      DEFAULT_MODEL_ID = DEFAULT_ASR_MODEL_ID
      EXPECTED_ONNX_SHA256 = "914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9"
      COHERE_PROCESSOR_CLASS = "CohereAsrProcessor"
      COHERE_FEATURE_EXTRACTOR_CLASS = "CohereAsrFeatureExtractor"
      DEFAULT_MAX_AUDIO_CLIP_SECONDS = 35.0
      REQUIRED_PROMPT_TOKENS = %w[
        ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
        <|ar|> <|en|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|>
        <|endoftext|>
      ].freeze
      VALUE_OPTIONS = %w[mode model model-revision adapter adapter-revision audio-backend].freeze
      FLAG_OPTIONS = %w[model-access help].freeze
      private_constant :VALUE_OPTIONS, :FLAG_OPTIONS

      Options = Data.define(
        :mode,
        :model_access,
        :model,
        :model_revision,
        :adapter,
        :adapter_revision,
        :audio_backend
      )

      class EarlyExit < StandardError
        attr_reader :status

        def initialize(status = 0)
          @status = status
          super()
        end
      end

      class Results
        attr_reader :failures, :warnings

        def initialize(out: $stdout)
          @out = out
          @failures = 0
          @warnings = 0
        end

        def ok(message)
          @out.puts("[OK]   #{message}")
        end

        def warn(message)
          @warnings += 1
          @out.puts("[WARN] #{message}")
        end

        def fail(message)
          @failures += 1
          @out.puts("[FAIL] #{message}")
        end
      end

      module_function

      def parse_args(argv = ARGV, out: $stdout)
        values = {
          mode: "segment",
          model_access: false,
          model: DEFAULT_MODEL_ID,
          model_revision: nil,
          adapter: nil,
          adapter_revision: nil,
          audio_backend: "auto"
        }
        raw_arguments = Array(argv).dup
        preflight_ambiguous_long_options!(raw_arguments)
        arguments = defer_unknown_options_before_early_exit(raw_arguments)
        parser = option_parser(values, out: out)
        parser.parse!(arguments)
        raise OptionParser::InvalidArgument, "unexpected positional argument: #{arguments.first}" unless arguments.empty?

        Options.new(**values)
      end

      def main(argv = ARGV, out: $stdout, err: $stderr, checks: nil)
        options = parse_args(argv, out: out)
        checker = checks || self
        results = Results.new(out: out)
        checker.validate_files(results)
        checker.validate_common_runtime(results)
        checker.validate_silero(results)
        checker.validate_word_alignment(results) if options.mode == "word"
        checker.report_optional_runtime(results, options.audio_backend)

        selected_model = options.model != DEFAULT_MODEL_ID ||
                         !options.model_revision.nil? ||
                         !options.adapter.nil? ||
                         !options.adapter_revision.nil?
        if options.model_access || selected_model
          checker.validate_model_access(
            results,
            include_aligner: options.mode == "word",
            model_id: options.model,
            model_revision: options.model_revision,
            adapter_id: options.adapter,
            adapter_revision: options.adapter_revision
          )
        end

        out.puts
        if results.failures.positive?
          out.puts("Validation failed: #{results.failures} failure(s), #{results.warnings} warning(s).")
          1
        else
          out.puts("Validation passed for #{options.mode} mode with #{results.warnings} warning(s).")
          0
        end
      rescue EarlyExit => e
        e.status
      rescue OptionParser::ParseError => e
        err.puts(option_parser({}, out: out))
        err.puts("cohere-transcribe-doctor: error: #{e.message}")
        2
      end

      def option_parser(values, out:)
        OptionParser.new do |parser|
          # OptionParser installs an implicit --version switch that aborts when
          # no program version is configured.  The reference doctor does not
          # expose that switch, so remove it and let normal unknown-option
          # handling report the command-line error.
          parser.base.long.delete("version")
          parser.banner = "Usage: cohere-transcribe-doctor [options]"
          parser.separator("")
          parser.separator("Validate the transcription package without loading model weights.")
          parser.separator("")
          parser.on("--mode MODE", String, %w[word segment], "Output mode to validate (default: segment).") do |value|
            values[:mode] = value
          end
          parser.on("--model-access", "Resolve and validate selected model metadata.") do
            values[:model_access] = true
          end
          parser.on("--model MODEL", "Hub repository or local model directory; implies model access.") do |value|
            values[:model] = value
          end
          parser.on("--model-revision REVISION", "Optional Hub model commit, tag, or branch.") do |value|
            values[:model_revision] = value
          end
          parser.on("--adapter ADAPTER", "Optional Hub repository or local LoRA adapter directory.") do |value|
            values[:adapter] = value
          end
          parser.on("--adapter-revision REVISION", "Optional Hub adapter commit, tag, or branch.") do |value|
            values[:adapter_revision] = value
          end
          parser.on(
            "--audio-backend BACKEND",
            String,
            %w[auto torchcodec ffmpeg librosa],
            "Decoder configuration to validate (default: auto)."
          ) { |value| values[:audio_backend] = value }
          parser.on_tail("-h", "--help", "Show this help and exit.") do
            out.puts(parser)
            raise EarlyExit, 0
          end
        end
      end

      def preflight_ambiguous_long_options!(arguments)
        arguments.each do |argument|
          break if argument == "--"
          next unless argument.start_with?("--") && argument.length > 2

          supplied = argument.delete_prefix("--").split("=", 2).first
          names = VALUE_OPTIONS + FLAG_OPTIONS
          candidates = names.select { |name| name.start_with?(supplied) }
          canonical = if names.include?(supplied)
                        supplied
                      elsif candidates.one?
                        candidates.first
                      end
          break if canonical == "help"
          next if names.include?(supplied)

          raise OptionParser::AmbiguousOption, "--#{supplied}" if candidates.length > 1
        end
      end
      private_class_method :preflight_ambiguous_long_options!

      def defer_unknown_options_before_early_exit(arguments)
        unknown_indices = []
        early_index = nil
        index = 0
        while index < arguments.length
          argument = arguments[index]
          break if argument == "--"

          option = canonical_long_option(argument)
          if argument == "-h" || option == "help"
            early_index = index
            break
          end
          if argument.start_with?("--") && long_option_candidates(argument).empty?
            unknown_indices << index
          elsif argument.start_with?("-") && argument != "-" && !argument.start_with?("--")
            unknown_indices << index
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

      def canonical_long_option(argument)
        return nil unless argument.start_with?("--") && argument.length > 2

        supplied = argument.delete_prefix("--").split("=", 2).first
        names = VALUE_OPTIONS + FLAG_OPTIONS
        return supplied if names.include?(supplied)

        candidates = long_option_candidates(argument)
        candidates.one? ? candidates.first : nil
      end
      private_class_method :canonical_long_option

      def long_option_candidates(argument)
        return [] unless argument.start_with?("--") && argument.length > 2

        supplied = argument.delete_prefix("--").split("=", 2).first
        (VALUE_OPTIONS + FLAG_OPTIONS).select { |name| name.start_with?(supplied) }
      end
      private_class_method :long_option_candidates

      def validate_files(results)
        require "digest"

        specification = Gem.loaded_specs["cohere-transcribe"]
        if specification.nil?
          results.warn("package metadata is unavailable; running from a source checkout")
        elsif specification.version.to_s == VERSION
          results.ok("package metadata version: #{VERSION}")
        else
          results.fail("package metadata is #{specification.version}, runtime is #{VERSION}")
        end

        asset = File.expand_path("vad/silero_vad_v6.onnx", __dir__)
        unless File.file?(asset)
          results.fail("missing Silero ONNX asset: #{asset}")
          return
        end
        digest = Digest::SHA256.file(asset).hexdigest
        if digest == EXPECTED_ONNX_SHA256
          results.ok("Silero ONNX asset integrity: #{digest}")
        else
          results.fail(
            "Silero ONNX checksum mismatch: expected #{EXPECTED_ONNX_SHA256}, found #{digest}"
          )
        end
      rescue SystemCallError => e
        results.fail("cannot validate packaged assets: #{e.class}: #{e.message}")
      end

      def validate_common_runtime(results, native_library: nil)
        results.ok("Ruby #{RUBY_VERSION}")
        import_required(results, "numo/narray", "numeric runtime")
        import_required(results, "onnxruntime", "ONNX inference runtime")
        require_relative "constants"
        require_relative "errors"
        require_relative "asr/native"
        library = native_library || ASR::NativeLibrary.load
        results.ok("native Cohere ASR runtime: #{library.path}")
        report_native_device_capabilities(results, library)
      rescue LoadError, StandardError => e
        results.fail("native Cohere ASR runtime: #{e.class}: #{e.message}")
      end

      def report_native_device_capabilities(results, library)
        available = ["cpu"]
        %w[cuda mps].each do |device|
          available << device if library.resolve_device(device) == device
        rescue TranscriptionRuntimeError
          next
        end
        resolved = library.resolve_device("auto")
        results.ok("native inference devices: #{available.join(", ")}; auto resolves to #{resolved}")
        case resolved
        when "cuda"
          bf16 = library.supports_bf16?(resolved) ? "supported" : "not supported"
          results.ok("accelerator: CUDA available through native runtime; BF16 operations #{bf16}")
        when "mps"
          bf16 = library.supports_bf16?(resolved) ? "supported" : "not supported"
          results.ok("accelerator: Apple Metal available through native runtime; BF16 operations #{bf16}")
        when "cpu"
          results.warn("accelerator: CPU only; the 2B model will be substantially slower")
        else
          results.fail("native runtime returned an unsupported automatic device: #{resolved.inspect}")
        end
      end

      def validate_silero(results)
        require_relative "vad/silero"

        probabilities = VAD::Silero.new.speech_probabilities(Array.new(1024, 0.0))
        valid = probabilities.length == 2 && probabilities.all? do |probability|
          probability.is_a?(Numeric) && probability.finite? && probability.between?(0.0, 1.0)
        end
        if valid
          results.ok("Silero ONNX smoke: 2 finite probability frames")
        else
          results.fail("Silero ONNX smoke returned invalid probabilities: #{probabilities.inspect}")
        end
      rescue LoadError, StandardError => e
        results.fail("Silero ONNX smoke: #{e.class}: #{e.message}")
      end

      def validate_word_alignment(results)
        require_relative "alignment/aligner"

        path = Alignment::CTC.forced_align(
          Numo::SFloat.cast([[4.0, 0.0], [0.0, 4.0]]),
          [1],
          blank: 0
        )
        raise "unexpected CTC path #{path.inspect}" unless path == [0, 1]

        tokens, = Alignment::Text.preprocess("مرحبا بكم في العالم", "ara")
        expected = ["<star>", "m r h b a", "<star>", "b k m", "<star>", "f y", "<star>", "a l ' a l m"]
        raise "unexpected Arabic romanization #{tokens.inspect}" unless tokens == expected

        fp32 = Alignment::ModelProvider::ARTIFACTS.fetch("fp32")
        results.ok(
          "word alignment: pure-Ruby MMS CTC Viterbi and Arabic romanization smokes execute"
        )
        results.ok(
          "word aligner: #{Alignment::ModelProvider::REPOSITORY}@#{Alignment::ModelProvider::REVISION}, " \
          "FP32 SHA-256 #{fp32.sha256}, bounded per-segment uniform fallback"
        )
      rescue LoadError, StandardError => e
        results.fail("word alignment: #{e.class}: #{e.message}")
      end

      def report_optional_runtime(results, audio_backend)
        case audio_backend
        when "auto", "ffmpeg", "torchcodec", "librosa"
          require_relative "audio/decoder"
          if Audio::FFmpegNative.available?
            results.ok("native FFmpeg decoder: #{Audio::FFmpegNative.diagnostic}")
            if %w[torchcodec librosa].include?(audio_backend)
              results.warn("#{audio_backend} compatibility mode uses FFmpeg through the native C ABI")
            end
          else
            sound_file = Audio.const_get(:SoundFileABI, false)
            if %w[auto librosa].include?(audio_backend) && sound_file.const_get(:AVAILABLE)
              label = audio_backend == "librosa" ? "librosa compatibility fallback" : "native audio decoder fallback"
              results.ok("#{label}: libsndfile ABI")
              results.warn("native FFmpeg decoder unavailable: #{Audio::FFmpegNative.diagnostic}")
            else
              results.fail("native FFmpeg decoder unavailable: #{Audio::FFmpegNative.diagnostic}")
            end
          end
        end
      rescue LoadError, StandardError => e
        results.fail("audio decoder configuration: #{e.class}: #{e.message}")
      end

      def validate_model_access(
        results,
        include_aligner:,
        model_id: DEFAULT_MODEL_ID,
        model_revision: nil,
        adapter_id: nil,
        adapter_revision: nil,
        hub: nil
      )
        require_relative "constants"
        require_relative "hub"
        require_relative "model_identity"

        hub ||= Hub.new
        identity = ModelIdentity.resolve(
          model_id,
          model_revision,
          adapter_id,
          adapter_revision,
          hub: hub
        )
        maximum = validate_asr_processor_metadata!(identity, hub)
        validate_aligner_metadata!(hub) if include_aligner
        model_reference = reference(identity.model_id, identity.model_revision)
        adapter = if identity.adapter_id
                    ", adapter #{reference(identity.adapter_id, identity.adapter_revision)}"
                  else
                    ""
                  end
        aligner = include_aligner ? " and pinned MMS ONNX aligner metadata" : ""
        results.ok(
          "ASR configuration #{model_reference} (#{identity.model_format})#{adapter}#{aligner} accessible; " \
          "#{COHERE_PROCESSOR_CLASS} one-row limit is #{maximum}s"
        )
        if identity.model_format.to_s != "dense"
          results.warn(
            "saved #{identity.model_format} checkpoint detected; quantized inference is not part of the core Ruby path"
          )
        end
        results.warn("PEFT/LoRA adapter inference is not part of the core Ruby Dense path") if identity.adapter_id
      rescue Hub::AuthenticationError => e
        results.fail(
          "Cannot access the gated ASR model #{model_id}. Accept its terms, then set HF_TOKEN. " \
          "(#{e.class}: #{e.message})"
        )
      rescue StandardError => e
        results.fail("ASR model or adapter validation: #{e.class}: #{e.message}")
      end

      def import_required(results, library, feature)
        require library
        gem_name = { "numo/narray" => "numo-narray" }.fetch(library, library.split("/").first)
        version = Gem.loaded_specs[gem_name]&.version
        suffix = version ? " #{version}" : ""
        results.ok("#{feature}: #{library}#{suffix}")
        true
      rescue LoadError => e
        results.fail("#{feature}: cannot require #{library.inspect}: #{e.message}")
        false
      end

      def find_executable(name)
        ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).filter_map do |directory|
          path = File.join(directory, name)
          path if File.file?(path) && File.executable?(path)
        end.first
      end
      private_class_method :find_executable

      def reference(identifier, revision)
        revision ? "#{identifier}@#{revision}" : identifier
      end
      private_class_method :reference

      def validate_asr_processor_metadata!(identity, hub)
        config = read_model_artifact(identity, hub, "config.json")
        preprocessor = read_model_artifact(identity, hub, "preprocessor_config.json")
        tokenizer = read_model_artifact(identity, hub, "tokenizer.json")

        unless config["model_type"] == "cohere_asr"
          raise ArgumentError,
                "config.json declares model_type=#{config["model_type"].inspect}; expected cohere_asr"
        end

        processor_class = preprocessor["processor_class"]
        unless processor_class == COHERE_PROCESSOR_CLASS
          raise ArgumentError,
                "preprocessor_config.json declares processor_class=#{processor_class.inspect}; " \
                "expected #{COHERE_PROCESSOR_CLASS}"
        end
        extractor_class = preprocessor["feature_extractor_type"]
        if extractor_class && extractor_class != COHERE_FEATURE_EXTRACTOR_CLASS
          raise ArgumentError,
                "preprocessor_config.json declares feature_extractor_type=#{extractor_class.inspect}; " \
                "expected #{COHERE_FEATURE_EXTRACTOR_CLASS}"
        end

        maximum = preprocessor.fetch("max_audio_clip_s", DEFAULT_MAX_AUDIO_CLIP_SECONDS)
        unless maximum.is_a?(Numeric) && !maximum.is_a?(Complex) && maximum.finite? && maximum.positive?
          raise ArgumentError,
                "Cohere processor reported an invalid max_audio_clip_s: #{maximum.inspect}"
        end
        validate_tokenizer_metadata!(tokenizer, config)
        Float(maximum)
      end
      private_class_method :validate_asr_processor_metadata!

      def read_model_artifact(identity, hub, filename)
        path = if identity.model_revision.nil?
                 Pathname(identity.model_id).join(filename)
               else
                 hub.download(identity.model_id, filename, revision: identity.model_revision)
               end
        ModelIdentity.read_json_object(path)
      end
      private_class_method :read_model_artifact

      def validate_tokenizer_metadata!(tokenizer, config)
        vocabulary = tokenizer.dig("model", "vocab")
        raise ArgumentError, "tokenizer.json has no model vocabulary" unless vocabulary.is_a?(Hash) && !vocabulary.empty?

        added_tokens = tokenizer.fetch("added_tokens", [])
        raise ArgumentError, "tokenizer.json added_tokens must be an array" unless added_tokens.is_a?(Array)

        tokens_by_id = {}
        ids_by_token = {}
        vocabulary.each do |token, id|
          add_tokenizer_entry!(tokens_by_id, ids_by_token, token, id)
        end
        added_tokens.each do |entry|
          raise ArgumentError, "tokenizer.json contains an invalid added token" unless entry.is_a?(Hash)

          add_tokenizer_entry!(tokens_by_id, ids_by_token, entry["content"], entry["id"])
        end

        expected_size = config["vocab_size"] || config.dig("head", "num_classes")
        if expected_size
          unless expected_size.is_a?(Integer) && expected_size.positive?
            raise ArgumentError, "model configuration has an invalid vocabulary size: #{expected_size.inspect}"
          end

          missing_ids = (0...expected_size).reject { |id| tokens_by_id.key?(id) }
          unless missing_ids.empty?
            raise ArgumentError,
                  "tokenizer.json vocabulary has missing token IDs: #{missing_ids.first(8).inspect}"
          end
          outside = tokens_by_id.keys.select { |id| id >= expected_size }
          unless outside.empty?
            raise ArgumentError,
                  "tokenizer.json token ID #{outside.min} is outside the checkpoint vocabulary size #{expected_size}"
          end
        end

        missing_prompt = REQUIRED_PROMPT_TOKENS.reject { |token| ids_by_token.key?(token) }
        return if missing_prompt.empty?

        raise ArgumentError,
              "tokenizer.json is missing Cohere prompt tokens: #{missing_prompt.join(", ")}"
      end
      private_class_method :validate_tokenizer_metadata!

      def add_tokenizer_entry!(tokens_by_id, ids_by_token, token, id)
        raise ArgumentError, "tokenizer.json contains an invalid token/id entry" unless token.is_a?(String) && id.is_a?(Integer) && id >= 0

        previous_token = tokens_by_id[id]
        if previous_token && previous_token != token
          raise ArgumentError,
                "tokenizer.json assigns ID #{id} to both #{previous_token.inspect} and #{token.inspect}"
        end
        previous_id = ids_by_token[token]
        if previous_id && previous_id != id
          raise ArgumentError,
                "tokenizer.json assigns token #{token.inspect} to IDs #{previous_id} and #{id}"
        end

        tokens_by_id[id] = token
        ids_by_token[token] = id
      end
      private_class_method :add_tokenizer_entry!

      def validate_aligner_metadata!(hub)
        require_relative "alignment/aligner"

        provider = Alignment::ModelProvider
        config_path = hub.download(provider::REPOSITORY, "config.json", revision: provider::REVISION)
        vocabulary_path = hub.download(provider::REPOSITORY, "vocab.json", revision: provider::REVISION)
        config = ModelIdentity.read_json_object(config_path)
        vocabulary = ModelIdentity.read_json_object(vocabulary_path)
        stride = Array(config["conv_stride"]).reduce(1) { |product, value| product * Integer(value) }

        raise "pinned aligner input stride changed" unless stride == Alignment::Aligner::INPUTS_TO_LOGITS_RATIO
        raise "pinned aligner vocabulary size changed" unless config["vocab_size"] == Alignment::Aligner::VOCABULARY.length
        raise "pinned aligner source checkpoint changed" unless config["_name_or_path"] == provider::SOURCE_REPOSITORY
        raise "pinned aligner vocabulary changed" unless vocabulary == Alignment::Aligner::VOCABULARY
      end
      private_class_method :validate_aligner_metadata!
    end
  end
end
