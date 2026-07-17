# frozen_string_literal: true

require "json"
require_relative "constants"
require_relative "internal/utf8"

module Cohere
  module Transcribe
    ResolvedModelIdentity = Data.define(
      :model_id,
      :model_revision,
      :model_format,
      :quantization_config,
      :adapter_id,
      :adapter_revision
    )

    module ModelIdentity
      module_function

      MODEL_WEIGHT_FILES = %w[
        model.safetensors model.safetensors.index.json
        pytorch_model.bin pytorch_model.bin.index.json
      ].freeze
      ADAPTER_WEIGHT_FILES = %w[adapter_model.safetensors adapter_model.bin].freeze

      def default_model_revision(model_id, revision)
        revision || (model_id == DEFAULT_ASR_MODEL_ID ? DEFAULT_ASR_MODEL_REVISION : nil)
      end

      def packaged_default_model?(model_id, revision)
        model_id == DEFAULT_ASR_MODEL_ID && revision == DEFAULT_ASR_MODEL_REVISION
      end
      private_class_method :packaged_default_model?

      def resolve_local_directory(reference, description: "Model")
        raise TypeError, "#{description} must be a string" unless reference.is_a?(String)

        source_path = Pathname(reference)
        path = begin
          expand_user_path(source_path)
        rescue ArgumentError, SystemCallError => e
          raise ArgumentError,
                "Cannot resolve #{description.downcase} path #{reference.inspect}: #{e.message}"
        end
        return utf8_resolved_path(path.realpath.to_s, description) if path.directory?
        if path.exist? || path.symlink?
          raise ArgumentError,
                "#{description} path #{reference.inspect} is not a directory"
        end

        explicit_reference = reference.start_with?("./", "../", "~") ||
                             [".", "..", "~"].include?(reference) ||
                             reference.count("/") > 1
        explicit = source_path.absolute? || explicit_reference
        raise ArgumentError, "#{description} directory #{reference.inspect} does not exist" if explicit

        nil
      rescue SystemCallError => e
        raise ArgumentError, "Cannot resolve #{description.downcase} path #{reference.inspect}: #{e.message}"
      end

      def utf8_resolved_path(value, description)
        text = Internal::UTF8.normalize(value)
        return text if text

        raise ArgumentError, "#{description} resolved path must contain valid UTF-8"
      end
      private_class_method :utf8_resolved_path

      # Path.expanduser() expands only the leading home-directory component;
      # unlike Pathname#expand_path, it does not lexically erase a missing
      # component followed by `..` before the filesystem probe.
      def expand_user_path(path)
        reference = path.to_s
        # pathlib drops relative `.` components while constructing Path, before
        # expanduser() examines the first component (so `./~` means the home
        # directory). Preserve every `..` component for the later OS probe.
        normalized_reference = reference.sub(%r{\A(?:\./+)+}, "")
        return path unless normalized_reference.start_with?("~")

        home_reference, separator, remainder = normalized_reference.partition("/")
        home = Pathname(File.expand_path(home_reference))
        separator.empty? ? home : home.join(remainder)
      end
      private_class_method :expand_user_path

      def classify_model_config(config, reference)
        unless config["model_type"] == "cohere_asr"
          raise ArgumentError,
                "#{reference} uses model_type=#{python_value_repr(config["model_type"])}; " \
                "expected a native Transformers Cohere ASR checkpoint"
        end
        architectures = config["architectures"]
        if python_truthy?(architectures) && !declares_cohere_asr_architecture?(architectures)
          raise ArgumentError, "#{reference} does not declare CohereAsrForConditionalGeneration"
        end

        quantization = config["quantization_config"]
        return [:dense, nil] if quantization.nil?
        raise ArgumentError, "#{reference} has an invalid quantization_config" unless quantization.is_a?(Hash)

        method = python_lower(python_string(quantization.fetch("quant_method", "")))
        four_bit = quantization.fetch("load_in_4bit", quantization.fetch("_load_in_4bit", false))
        eight_bit = quantization.fetch("load_in_8bit", quantization.fetch("_load_in_8bit", false))
        unless [four_bit, eight_bit].all? { |value| [true, false].include?(value) }
          raise ArgumentError,
                "#{reference} has an invalid quantization_config: " \
                "bitsandbytes load flags must be boolean"
        end
        unless method == "bitsandbytes" && four_bit != eight_bit
          rendered_method = method.empty? ? "unknown" : method
          raise ArgumentError,
                "#{reference} uses unsupported saved quantization configuration " \
                "#{python_string_repr(rendered_method)}"
        end

        [four_bit ? :"bitsandbytes-int4" : :"bitsandbytes-int8", quantization.dup.freeze]
      end

      def resolve(model_id, model_revision = nil, adapter_id = nil, adapter_revision = nil,
                  hub: Hub.new, verify_weight_artifacts: true)
        raise ArgumentError, "adapter_revision requires an adapter_id" if adapter_id.nil? && adapter_revision

        resolved_model_id, resolved_model_revision, model_dir = resolve_reference(
          model_id,
          model_revision,
          description: "Model",
          metadata: "config.json",
          hub: hub,
          default_revision: default_model_revision(model_id, model_revision)
        )
        if packaged_default_model?(resolved_model_id, resolved_model_revision)
          format = :dense
          quantization = nil
        else
          config_path = if model_dir
                          Pathname(model_dir).join("config.json")
                        else
                          hub.download(
                            resolved_model_id, "config.json", revision: resolved_model_revision
                          )
                        end
          config = read_json_object(config_path)
          reference = reference(resolved_model_id, resolved_model_revision)
          format, quantization = classify_model_config(config, reference)
        end

        resolved_adapter_id = nil
        resolved_adapter_revision = nil
        if adapter_id
          raise ArgumentError, "PEFT adapters are supported only with dense base models" unless format == :dense

          resolved_adapter_id, resolved_adapter_revision, adapter_dir = resolve_reference(
            adapter_id,
            adapter_revision,
            description: "Adapter",
            metadata: "adapter_config.json",
            hub: hub,
            default_revision: adapter_revision
          )
          adapter_path = if adapter_dir
                           Pathname(adapter_dir).join("adapter_config.json")
                         else
                           hub.download(
                             resolved_adapter_id, "adapter_config.json", revision: resolved_adapter_revision
                           )
                         end
          adapter_config = read_json_object(adapter_path)
          validate_adapter_config!(
            adapter_config,
            resolved_model_id,
            resolved_model_revision,
            adapter_id: resolved_adapter_id,
            adapter_revision: resolved_adapter_revision,
            hub: hub
          )
        end

        identity = ResolvedModelIdentity.new(
          model_id: resolved_model_id,
          model_revision: resolved_model_revision,
          model_format: format,
          quantization_config: quantization,
          adapter_id: resolved_adapter_id,
          adapter_revision: resolved_adapter_revision
        )
        verify_model_weight_artifacts(identity, hub: hub) if verify_weight_artifacts
        identity
      end

      # Identity resolution is also used while planning runs that may be fully
      # satisfied by checkpoints. Keep weight discovery as a separate operation
      # so those runs do not require inference artifacts they will never load.
      def verify_model_weight_artifacts(identity, hub: Hub.new)
        unless packaged_default_model?(identity.model_id, identity.model_revision)
          verify_weight_artifacts!(
            identity.model_id,
            identity.model_revision,
            MODEL_WEIGHT_FILES,
            "Transformers model weights",
            hub: hub
          )
        end
        return unless identity.adapter_id

        verify_weight_artifacts!(
          identity.adapter_id,
          identity.adapter_revision,
          ADAPTER_WEIGHT_FILES,
          "PEFT adapter weights",
          hub: hub
        )
      end

      def reference(model_id, revision)
        revision ? "#{model_id}@#{revision}" : model_id
      end

      def read_json_object(path)
        path = Pathname(path)
        raise ArgumentError, "Local artifact #{path} is missing or is not a file" unless path.file?

        payload = JSON.parse(path.read(encoding: "UTF-8"))
        raise ArgumentError, "#{path} is not a JSON object" unless payload.is_a?(Hash)

        payload
      rescue SystemCallError, JSON::ParserError, EncodingError => e
        raise ArgumentError, "Cannot read JSON object from #{path}: #{e.message}"
      end

      # JSON config data reaches the Python reference implementation as native
      # Python values. Preserve its truthiness and container-membership behavior
      # here, including TypeError for truthy scalar architecture declarations.
      def python_truthy?(value)
        return false if value.nil? || value == false
        return !value.zero? if value.is_a?(Numeric)
        return !value.empty? if value.is_a?(String) || value.is_a?(Array) || value.is_a?(Hash)

        true
      end
      private_class_method :python_truthy?

      def declares_cohere_asr_architecture?(architectures)
        expected = "CohereAsrForConditionalGeneration"
        case architectures
        when String
          architectures.include?(expected)
        when Array
          architectures.include?(expected)
        when Hash
          architectures.key?(expected)
        else
          type = case architectures
                 when true, false then "bool"
                 when Integer then "int"
                 when Float then "float"
                 else architectures.class.name
                 end
          raise TypeError, "argument of type '#{type}' is not iterable"
        end
      end
      private_class_method :declares_cohere_asr_architecture?

      # Python's classifier applies str(...).lower() to quant_method. JSON's
      # null/boolean/container spellings differ from Ruby's #to_s, so reproduce
      # them before classifying and rendering the public diagnostic.
      def python_string(value)
        case value
        when nil then "None"
        when true then "True"
        when false then "False"
        when String then value
        when Float then python_float_string(value)
        when Array then "[#{value.map { |item| python_value_repr(item) }.join(", ")}]"
        when Hash
          entries = value.map do |key, item|
            "#{python_value_repr(key)}: #{python_value_repr(item)}"
          end
          "{#{entries.join(", ")}}"
        else value.to_s
        end
      end
      private_class_method :python_string

      def python_float_string(value)
        return "nan" if value.nan?
        return value.negative? ? "-inf" : "inf" if value.infinite?

        rendered = value.to_s
        match = /\A(-?)(\d)(?:\.(\d+))?e([+-])(\d+)\z/.match(rendered)
        return rendered unless match

        exponent = match[5].to_i * (match[4] == "-" ? -1 : 1)
        return rendered.sub(/\.0(?=e[+-]\d+\z)/, "") unless exponent.between?(-4, 15)

        digits = match[2] + match[3].to_s
        decimal_index = exponent + 1
        body = if decimal_index <= 0
                 "0.#{"0" * -decimal_index}#{digits}"
               elsif decimal_index >= digits.length
                 "#{digits}#{"0" * (decimal_index - digits.length)}.0"
               else
                 "#{digits[0, decimal_index]}.#{digits[decimal_index..]}"
               end
        "#{match[1]}#{body}"
      end
      private_class_method :python_float_string

      def python_lower(value)
        characters = value.each_char.to_a
        characters.each_with_index.map do |character, index|
          if character == "Σ" && python_final_sigma?(characters, index)
            "ς"
          else
            character.downcase
          end
        end.join
      end
      private_class_method :python_lower

      def python_final_sigma?(characters, index)
        before = characters[0...index].rfind { |character| !character.match?(/\p{Case_Ignorable}/u) }
        return false unless before&.match?(/\p{Cased}/u)

        after = characters[(index + 1)..].find { |character| !character.match?(/\p{Case_Ignorable}/u) }
        !after&.match?(/\p{Cased}/u)
      end
      private_class_method :python_final_sigma?

      def python_value_repr(value)
        case value
        when nil then "None"
        when true then "True"
        when false then "False"
        when String then python_string_repr(value)
        else python_string(value)
        end
      end
      private_class_method :python_value_repr

      def python_string_repr(value)
        quote = value.include?("'") && !value.include?('"') ? '"' : "'"
        escaped = value.each_char.map do |character|
          case character
          when "\\" then "\\\\"
          when quote then "\\#{quote}"
          when "\t" then "\\t"
          when "\n" then "\\n"
          when "\r" then "\\r"
          else
            codepoint = character.ord
            non_printing = character != " " && character.match?(/[\p{C}\p{Z}]/u)
            if codepoint < 0x20 || codepoint == 0x7f || non_printing
              case codepoint
              when 0..0xff then format("\\x%02x", codepoint)
              when 0x100..0xffff then format("\\u%04x", codepoint)
              else format("\\U%08x", codepoint)
              end
            else
              character
            end
          end
        end.join
        "#{quote}#{escaped}#{quote}"
      end
      private_class_method :python_string_repr

      def resolve_reference(id, revision, description:, metadata:, hub:, default_revision:)
        raise TypeError, "#{description} must be a string" unless id.is_a?(String)
        raise ArgumentError, "#{description} must not be empty" if id.strip.empty?

        local = resolve_local_directory(id, description: description)
        if local
          raise ArgumentError, "A #{description.downcase} revision cannot be used with a local directory" if revision

          return [local, nil, local]
        end

        commit = hub.resolve_revision(id, default_revision, filename: metadata)
        [id, commit, nil]
      end
      private_class_method :resolve_reference

      def verify_weight_artifacts!(id, revision, candidates, description, hub:)
        if revision
          verify_remote_weights!(hub, id, revision, candidates, description)
        else
          verify_local_weights!(id, candidates, description)
        end
      end
      private_class_method :verify_weight_artifacts!

      def verify_local_weights!(directory, candidates, description)
        directory = Pathname(directory)
        return if candidates.any? { |name| directory.join(name).file? }

        expected = candidates.sort.join(", ")
        raise ArgumentError,
              "Local directory #{directory.to_s.inspect} does not contain supported #{description}; " \
              "expected one of: #{expected}"
      end
      private_class_method :verify_local_weights!

      def verify_remote_weights!(hub, repo_id, revision, candidates, description)
        return if candidates.any? { |name| hub.cached_file(repo_id, name, revision: revision) }
        return if hub.list_files(repo_id, revision: revision).intersect?(candidates)

        expected = candidates.sort.join(", ")
        raise ArgumentError,
              "#{repo_id}@#{revision} does not contain supported #{description}; expected one of: #{expected}"
      end
      private_class_method :verify_remote_weights!

      def validate_adapter_config!(config, model_id, model_revision, adapter_id:, adapter_revision:, hub:)
        raise ArgumentError, "Adapter peft_type must be LORA" unless config["peft_type"].to_s.upcase == "LORA"
        raise ArgumentError, "Adapter task_type must be SEQ_2_SEQ_LM" unless config["task_type"].to_s.upcase == "SEQ_2_SEQ_LM"

        # A local base has no stable Hub identity against which a training-time
        # repository string can be compared. Remote bases do, so require the
        # adapter to name that exact repository and compare immutable commits.
        return unless model_revision

        declared_base = config["base_model_name_or_path"]
        adapter_reference = reference(adapter_id, adapter_revision)
        if declared_base != model_id
          raise ArgumentError,
                "Adapter #{adapter_reference} requires base model " \
                "#{declared_base.inspect}, not #{model_id.inspect}"
        end

        declared_revision = config["revision"]
        return unless declared_revision

        expected_revision = hub.resolve_revision(
          model_id,
          declared_revision.to_s,
          filename: "config.json"
        )
        return if expected_revision == model_revision

        raise ArgumentError,
              "Adapter #{adapter_reference} requires base revision " \
              "#{expected_revision}, not #{model_revision}"
      end
      private_class_method :validate_adapter_config!
    end
  end
end
