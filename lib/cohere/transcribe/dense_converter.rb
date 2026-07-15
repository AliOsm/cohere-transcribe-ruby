# frozen_string_literal: true

require "json"

require_relative "gguf_writer"
require_relative "model_identity"
require_relative "pytorch_checkpoint"
require_relative "safetensors"

module Cohere
  module Transcribe
    # Converts a native Transformers Cohere ASR dense checkpoint to the GGUF
    # tensor contract consumed by CrispASR's Cohere backend.
    #
    # The tensor naming contract is derived from CrispASR, MIT-licensed by the
    # ggml authors. See licenses/crispasr.txt for the retained license notice.
    class DenseConverter
      class Error < StandardError; end

      Mapping = Data.define(:output_name, :source_name, :precision)
      PlannedTensor = Data.define(:output_name, :source_tensor, :output_dtype)
      Plan = Data.define(:tensors, :metadata, :vocabulary, :source_dtype_counts, :output_dtype_counts)
      Result = Data.define(:path, :tensor_count, :source_dtype_counts, :output_dtype_counts)

      FRONTEND_SOURCE_PATTERN = /\A(?:model\.)?preprocessor\.featurizer\.(?:fb|window)\z/
      COUNTER_SOURCE_PATTERN = /\A(?:model\.)?encoder\.layers\.\d+\.conv\.batch_norm\.num_batches_tracked\z/
      REQUIRED_PROMPT_TOKENS = %w[
        ▁ <|startofcontext|> <|startoftranscript|> <|emo:undefined|>
        <|ar|> <|pnc|> <|noitn|> <|notimestamp|> <|nodiarize|> <|endoftext|>
      ].freeze
      OUTPUT_TYPES = %i[f16 f32 bf16].freeze
      REQUIRED_ARTIFACT_FILENAMES = %w[config.json tokenizer.json].freeze
      WEIGHT_ARTIFACT_FILENAMES = %w[
        model.safetensors
        model.safetensors.index.json
        pytorch_model.bin
        pytorch_model.bin.index.json
      ].freeze
      SOURCE_ARTIFACT_FILENAMES = (
        REQUIRED_ARTIFACT_FILENAMES + WEIGHT_ARTIFACT_FILENAMES
      ).freeze
      SOURCE_ARTIFACT_REQUIREMENTS = {
        required: REQUIRED_ARTIFACT_FILENAMES,
        one_weight_file: WEIGHT_ARTIFACT_FILENAMES
      }.freeze

      attr_reader :model_directory, :output_path, :output_type, :dtype_converter, :chunk_bytes, :progress

      # Stable integration entry point for model stores and CLI adapters.
      def self.convert(model_dir:, output_path:, overwrite: false, fsync: true, **)
        new(model_dir, output_path: output_path, **).convert(overwrite: overwrite, fsync: fsync)
      end

      # Every filename a downloader may need to request. The safetensors file
      # and its sharded index are alternatives; see required_source_artifacts.
      def self.source_artifact_filenames
        SOURCE_ARTIFACT_FILENAMES
      end

      def self.required_source_artifacts
        SOURCE_ARTIFACT_REQUIREMENTS
      end

      def initialize(model_directory, output_path:, output_type: :f16,
                     dtype_converter: Safetensors::DTypeConverter.default,
                     chunk_bytes: Safetensors::DEFAULT_CHUNK_BYTES, progress: nil)
        @model_directory = Pathname(model_directory).expand_path
        @output_path = Pathname(output_path).expand_path
        @output_type = output_type.to_sym
        @dtype_converter = dtype_converter
        @chunk_bytes = Integer(chunk_bytes)
        @progress = progress
        unless OUTPUT_TYPES.include?(@output_type)
          raise ArgumentError,
                "Unsupported dense GGUF output type #{@output_type.inspect}"
        end
        raise ArgumentError, "chunk_bytes must be positive" unless @chunk_bytes.positive?
        raise ArgumentError, "progress must respond to call" if progress && !progress.respond_to?(:call)
      end

      def convert(overwrite: false, fsync: true)
        conversion_plan = plan
        writer = build_writer(conversion_plan)
        written_path = writer.write(output_path, overwrite: overwrite, fsync: fsync)
        Result.new(
          path: written_path,
          tensor_count: conversion_plan.tensors.length,
          source_dtype_counts: conversion_plan.source_dtype_counts,
          output_dtype_counts: conversion_plan.output_dtype_counts
        )
      rescue Safetensors::Error, PyTorchCheckpoint::Error, GGUF::Error => e
        raise Error, e.message
      end

      def plan
        @plan ||= build_plan
      rescue Safetensors::Error, PyTorchCheckpoint::Error => e
        raise Error, e.message
      end

      def self.tensor_mappings(encoder_layers:, decoder_layers:)
        mappings = []
        add_pair = lambda do |output, source, weight_precision = :f16|
          mappings << Mapping.new(output_name: "#{output}.weight", source_name: "#{source}.weight",
                                  precision: weight_precision)
          mappings << Mapping.new(output_name: "#{output}.bias", source_name: "#{source}.bias", precision: :f32)
        end

        [0, 2, 3, 5, 6].each do |index|
          add_pair.call("enc.pre.conv.#{index}", "encoder.pre_encode.conv.#{index}")
        end
        add_pair.call("enc.pre.out", "encoder.pre_encode.out")

        encoder_layers.times do |index|
          source = "encoder.layers.#{index}"
          output = "enc.blk.#{index}"
          add_pair.call("#{output}.ff1.norm", "#{source}.norm_feed_forward1", :f32)
          add_pair.call("#{output}.ff1.up", "#{source}.feed_forward1.linear1")
          add_pair.call("#{output}.ff1.down", "#{source}.feed_forward1.linear2")
          add_pair.call("#{output}.attn.norm", "#{source}.norm_self_att", :f32)
          add_pair.call("#{output}.attn.q", "#{source}.self_attn.linear_q")
          add_pair.call("#{output}.attn.k", "#{source}.self_attn.linear_k")
          add_pair.call("#{output}.attn.v", "#{source}.self_attn.linear_v")
          add_pair.call("#{output}.attn.out", "#{source}.self_attn.linear_out")
          mappings << Mapping.new(output_name: "#{output}.attn.pos.weight",
                                  source_name: "#{source}.self_attn.linear_pos.weight", precision: :f16)
          mappings << Mapping.new(output_name: "#{output}.attn.pos_bias_u",
                                  source_name: "#{source}.self_attn.pos_bias_u", precision: :f32)
          mappings << Mapping.new(output_name: "#{output}.attn.pos_bias_v",
                                  source_name: "#{source}.self_attn.pos_bias_v", precision: :f32)
          add_pair.call("#{output}.conv.norm", "#{source}.norm_conv", :f32)
          add_pair.call("#{output}.conv.pw1", "#{source}.conv.pointwise_conv1")
          add_pair.call("#{output}.conv.dw", "#{source}.conv.depthwise_conv")
          add_pair.call("#{output}.conv.bn", "#{source}.conv.batch_norm", :f32)
          mappings << Mapping.new(output_name: "#{output}.conv.bn.mean",
                                  source_name: "#{source}.conv.batch_norm.running_mean", precision: :f32)
          mappings << Mapping.new(output_name: "#{output}.conv.bn.var",
                                  source_name: "#{source}.conv.batch_norm.running_var", precision: :f32)
          add_pair.call("#{output}.conv.pw2", "#{source}.conv.pointwise_conv2")
          add_pair.call("#{output}.ff2.norm", "#{source}.norm_feed_forward2", :f32)
          add_pair.call("#{output}.ff2.up", "#{source}.feed_forward2.linear1")
          add_pair.call("#{output}.ff2.down", "#{source}.feed_forward2.linear2")
          add_pair.call("#{output}.out_norm", "#{source}.norm_out", :f32)
        end

        add_pair.call("enc.proj", "encoder_decoder_proj")
        mappings << Mapping.new(output_name: "dec.emb.weight",
                                source_name: "transf_decoder._embedding.token_embedding.weight", precision: :f16)
        mappings << Mapping.new(output_name: "dec.pos.weight",
                                source_name: "transf_decoder._embedding.position_embedding.pos_enc", precision: :f16)
        add_pair.call("dec.emb_ln", "transf_decoder._embedding.layer_norm", :f32)

        decoder_layers.times do |index|
          source = "transf_decoder._decoder.layers.#{index}"
          output = "dec.blk.#{index}"
          add_pair.call("#{output}.attn_ln", "#{source}.layer_norm_1", :f32)
          add_pair.call("#{output}.attn_q", "#{source}.first_sub_layer.query_net")
          add_pair.call("#{output}.attn_k", "#{source}.first_sub_layer.key_net")
          add_pair.call("#{output}.attn_v", "#{source}.first_sub_layer.value_net")
          add_pair.call("#{output}.attn_o", "#{source}.first_sub_layer.out_projection")
          add_pair.call("#{output}.cross_ln", "#{source}.layer_norm_2", :f32)
          add_pair.call("#{output}.cross_q", "#{source}.second_sub_layer.query_net")
          add_pair.call("#{output}.cross_k", "#{source}.second_sub_layer.key_net")
          add_pair.call("#{output}.cross_v", "#{source}.second_sub_layer.value_net")
          add_pair.call("#{output}.cross_o", "#{source}.second_sub_layer.out_projection")
          add_pair.call("#{output}.ffn_ln", "#{source}.layer_norm_3", :f32)
          add_pair.call("#{output}.ffn_up", "#{source}.third_sub_layer.dense_in")
          add_pair.call("#{output}.ffn_down", "#{source}.third_sub_layer.dense_out")
        end

        add_pair.call("dec.out_ln", "transf_decoder._decoder.final_layer_norm", :f32)
        add_pair.call("dec.head", "log_softmax.mlp.layer0")
        mappings.freeze
      end

      private

      def build_plan
        validate_directory!
        @config = read_json_object(model_directory.join("config.json"), "model configuration")
        validate_config!
        @tensor_set = checkpoint_tensor_set
        encoder_layers = detect_layer_count(:encoder)
        decoder_layers = detect_layer_count(:decoder)
        validate_configured_layer_count!(encoder_layers, decoder_layers)

        mappings = self.class.tensor_mappings(encoder_layers: encoder_layers, decoder_layers: decoder_layers)
        used_sources = Set.new
        planned = mappings.map do |mapping|
          tensor = fetch_source_tensor(mapping.source_name)
          used_sources << tensor.name
          dtype = case output_type
                  when :f32 then :f32
                  when :bf16 then mapping.precision == :f16 ? :bf16 : mapping.precision
                  else mapping.precision
                  end
          PlannedTensor.new(output_name: mapping.output_name, source_tensor: tensor, output_dtype: dtype)
        end
        validate_unmapped_tensors!(used_sources)

        vocabulary = load_vocabulary
        metadata = build_metadata(
          encoder_layers: encoder_layers,
          decoder_layers: decoder_layers,
          vocabulary: vocabulary
        )
        validate_shapes!(metadata, vocabulary)
        vocabulary.freeze
        metadata.each_value(&:freeze)

        Plan.new(
          tensors: planned.freeze,
          metadata: metadata.freeze,
          vocabulary: vocabulary,
          source_dtype_counts: tally(planned.map { |item| item.source_tensor.dtype }),
          output_dtype_counts: tally(planned.map(&:output_dtype))
        )
      end

      def validate_directory!
        raise Error, "Model directory #{model_directory} does not exist" unless model_directory.directory?
      end

      def validate_config!
        format, = ModelIdentity.classify_model_config(@config, model_directory.to_s)
        return if format == :dense

        raise Error, "Dense GGUF conversion does not accept a saved quantized checkpoint"
      rescue ArgumentError, TypeError => e
        raise Error, e.message
      end

      def detect_layer_count(kind)
        pattern = case kind
                  when :encoder
                    /\A(?:model\.)?encoder\.layers\.(\d+)\.norm_out\.weight\z/
                  when :decoder
                    /\A(?:model\.)?transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_1\.weight\z/
                  else
                    raise ArgumentError, "Unknown layer kind #{kind.inspect}"
                  end
        indices = @tensor_set.names.filter_map { |name| pattern.match(name)&.[](1)&.to_i }.uniq.sort
        raise Error, "Checkpoint contains no #{kind} layers" if indices.empty?

        expected = (0..indices.last).to_a
        raise Error, "Checkpoint #{kind} layer indices are not contiguous: #{indices.inspect}" unless indices == expected

        indices.length
      end

      def validate_configured_layer_count!(encoder_layers, decoder_layers)
        encoder = @config["encoder_config"] || @config["encoder"] || {}
        configured_encoder = encoder["num_hidden_layers"] || encoder["n_layers"]
        decoder = @config.dig("transf_decoder", "config_dict") || {}
        configured_decoder = decoder["num_layers"] || @config["num_hidden_layers"]
        validate_dimension!("encoder layer count", configured_encoder, encoder_layers) if configured_encoder
        validate_dimension!("decoder layer count", configured_decoder, decoder_layers) if configured_decoder
      end

      def fetch_source_tensor(base_name)
        @tensor_set.fetch_any([base_name, "model.#{base_name}"])
      rescue Safetensors::Error, PyTorchCheckpoint::Error => e
        raise Error, "Cohere dense checkpoint is missing #{base_name.inspect}: #{e.message}"
      end

      def checkpoint_tensor_set
        if model_directory.join("model.safetensors").file? ||
           model_directory.join("model.safetensors.index.json").file?
          Safetensors::TensorSet.from_directory(model_directory)
        elsif model_directory.join("pytorch_model.bin").file? ||
              model_directory.join("pytorch_model.bin.index.json").file?
          PyTorchCheckpoint::TensorSet.from_directory(model_directory)
        else
          raise Error,
                "#{model_directory} contains neither Safetensors nor PyTorch Dense weights"
        end
      end

      def validate_unmapped_tensors!(used_sources)
        unexpected = @tensor_set.tensors.values.reject do |tensor|
          used_sources.include?(tensor.name) ||
            (tensor.floating_point? && FRONTEND_SOURCE_PATTERN.match?(tensor.name)) ||
            (tensor.dtype == "I64" && COUNTER_SOURCE_PATTERN.match?(tensor.name))
        end
        return if unexpected.empty?

        examples = unexpected.first(8).map { |tensor| "#{tensor.name} (#{tensor.dtype})" }.join(", ")
        suffix = unexpected.length > 8 ? ", ..." : ""
        raise Error, "Checkpoint contains unmapped model tensors: #{examples}#{suffix}"
      end

      def load_vocabulary
        embedding = fetch_source_tensor("transf_decoder._embedding.token_embedding.weight")
        unless embedding.shape.length == 2 && embedding.shape[0].positive?
          raise Error, "Tensor #{embedding.name.inspect} has an invalid vocabulary shape #{embedding.shape.inspect}"
        end

        expected_size = embedding.shape[0]
        tokenizer = read_json_object(model_directory.join("tokenizer.json"), "tokenizer")
        vocab = tokenizer.dig("model", "vocab")
        raise Error, "#{model_directory.join("tokenizer.json")} has no model vocabulary" unless vocab.is_a?(Hash) && !vocab.empty?

        tokens_by_id = {}
        ids_by_token = {}
        vocab.each do |token, id|
          add_vocabulary_entry!(tokens_by_id, ids_by_token, token, id, expected_size: expected_size)
        end
        added = tokenizer["added_tokens"] || []
        raise Error, "tokenizer.json added_tokens must be an array" unless added.is_a?(Array)

        added.each do |entry|
          raise Error, "tokenizer.json contains an invalid added token" unless entry.is_a?(Hash)

          add_vocabulary_entry!(
            tokens_by_id,
            ids_by_token,
            entry["content"],
            entry["id"],
            expected_size: expected_size
          )
        end

        missing = (0...expected_size).reject { |id| tokens_by_id.key?(id) }
        raise Error, "tokenizer.json vocabulary has missing token IDs: #{missing.first(8).inspect}" unless missing.empty?

        missing_prompt = REQUIRED_PROMPT_TOKENS.reject { |token| ids_by_token.key?(token) }
        raise Error, "tokenizer.json is missing Cohere prompt tokens: #{missing_prompt.join(", ")}" unless missing_prompt.empty?

        (0...expected_size).map { |id| tokens_by_id.fetch(id) }
      end

      def add_vocabulary_entry!(tokens_by_id, ids_by_token, token, id, expected_size:)
        raise Error, "tokenizer.json contains an invalid token/id entry" unless token.is_a?(String) && id.is_a?(Integer) && id >= 0
        raise Error, "tokenizer.json token ID #{id} is outside the checkpoint vocabulary size #{expected_size}" if id >= expected_size

        previous_token = tokens_by_id[id]
        if previous_token && previous_token != token
          raise Error, "tokenizer.json assigns ID #{id} to both #{previous_token.inspect} and #{token.inspect}"
        end

        previous_id = ids_by_token[token]
        raise Error, "tokenizer.json assigns token #{token.inspect} to IDs #{previous_id} and #{id}" if previous_id && previous_id != id

        tokens_by_id[id] = token.freeze
        ids_by_token[token] = id
      end

      def build_metadata(encoder_layers:, decoder_layers:, vocabulary:)
        embedding = fetch_source_tensor("transf_decoder._embedding.token_embedding.weight")
        projection = fetch_source_tensor("encoder_decoder_proj.weight")
        encoder_ffn = fetch_source_tensor("encoder.layers.0.feed_forward1.linear1.weight")
        encoder_conv = fetch_source_tensor("encoder.layers.0.conv.depthwise_conv.weight")
        decoder_ffn = fetch_source_tensor("transf_decoder._decoder.layers.0.third_sub_layer.dense_in.weight")
        position = fetch_source_tensor("transf_decoder._embedding.position_embedding.pos_enc")
        pre_conv = fetch_source_tensor("encoder.pre_encode.conv.0.weight")

        require_rank!(embedding, 2)
        require_rank!(projection, 2)
        require_rank!(encoder_ffn, 2)
        require_rank!(encoder_conv, 3)
        require_rank!(decoder_ffn, 2)
        require_rank!(position, 2)
        require_rank!(pre_conv, 4)

        vocab_size, decoder_model = embedding.shape
        projection_decoder, encoder_model = projection.shape
        validate_dimension!("decoder model size", projection_decoder, decoder_model)
        validate_dimension!("tokenizer vocabulary size", vocabulary.length, vocab_size)

        encoder = @config["encoder_config"] || @config["encoder"] || {}
        decoder = @config.dig("transf_decoder", "config_dict") || {}
        encoder_heads = positive_integer(
          encoder["num_attention_heads"] || encoder["n_heads"], "encoder attention heads"
        )
        decoder_heads = positive_integer(
          decoder["num_attention_heads"] || @config["num_attention_heads"], "decoder attention heads"
        )
        unless (encoder_model % encoder_heads).zero? && (decoder_model % decoder_heads).zero?
          raise Error, "Model dimensions are not divisible by their attention head counts"
        end

        pre_conv_channels = pre_conv.shape[0]
        raise Error, "CrispASR Cohere backend requires 256 subsampling channels, got #{pre_conv_channels}" unless pre_conv_channels == 256

        subsampling = encoder["subsampling_factor"]
        validate_dimension!("encoder subsampling factor", subsampling, 8) if subsampling

        preprocessor = @config["preprocessor"] || {}
        sample_rate = positive_integer(@config["sample_rate"] || preprocessor["sample_rate"] || 16_000,
                                       "sample rate")
        n_fft = positive_integer(preprocessor["n_fft"] || 512, "FFT length")
        n_mels = positive_integer(
          encoder["num_mel_bins"] || encoder["feat_in"] || preprocessor["features"] || 128,
          "mel feature count"
        )
        hop_length = positive_integer(
          preprocessor["hop_length"] || seconds_to_samples(preprocessor["window_stride"], sample_rate) || 160,
          "hop length"
        )
        win_length = positive_integer(
          preprocessor["win_length"] || seconds_to_samples(preprocessor["window_size"], sample_rate) || 400,
          "window length"
        )
        max_context = positive_integer(
          decoder["max_sequence_length"] || @config["max_position_embeddings"] || position.shape[0],
          "decoder context length"
        )

        {
          "general.architecture" => [:string, "cohere-transcribe"],
          "general.name" => [:string, @config["_name_or_path"] || model_directory.basename.to_s],
          "general.file_type" => [:uint32, { f32: 0, f16: 1, bf16: 24 }.fetch(output_type)],
          "cohere_transcribe.frontend" => [:string, "runtime-generated-fp32"],
          "cohere_transcribe.vocab_size" => [:uint32, vocab_size],
          "cohere_transcribe.encoder.n_layers" => [:uint32, encoder_layers],
          "cohere_transcribe.encoder.d_model" => [:uint32, encoder_model],
          "cohere_transcribe.encoder.n_heads" => [:uint32, encoder_heads],
          "cohere_transcribe.encoder.head_dim" => [:uint32, encoder_model / encoder_heads],
          "cohere_transcribe.encoder.ffn_dim" => [:uint32, encoder_ffn.shape[0]],
          "cohere_transcribe.encoder.conv_kernel" => [:uint32, encoder_conv.shape[2]],
          "cohere_transcribe.decoder.n_layers" => [:uint32, decoder_layers],
          "cohere_transcribe.decoder.d_model" => [:uint32, decoder_model],
          "cohere_transcribe.decoder.n_heads" => [:uint32, decoder_heads],
          "cohere_transcribe.decoder.head_dim" => [:uint32, decoder_model / decoder_heads],
          "cohere_transcribe.decoder.ffn_dim" => [:uint32, decoder_ffn.shape[0]],
          "cohere_transcribe.decoder.max_ctx" => [:uint32, max_context],
          "cohere_transcribe.audio.sample_rate" => [:uint32, sample_rate],
          "cohere_transcribe.audio.n_mels" => [:uint32, n_mels],
          "cohere_transcribe.audio.n_fft" => [:uint32, n_fft],
          "cohere_transcribe.audio.hop_length" => [:uint32, hop_length],
          "cohere_transcribe.audio.win_length" => [:uint32, win_length],
          "tokenizer.ggml.tokens" => [:string_array, vocabulary],
          "tokenizer.ggml.model" => [:string, "llama"]
        }
      end

      def validate_shapes!(metadata, vocabulary)
        encoder_model = metadata.fetch("cohere_transcribe.encoder.d_model")[1]
        decoder_model = metadata.fetch("cohere_transcribe.decoder.d_model")[1]
        encoder_ffn = metadata.fetch("cohere_transcribe.encoder.ffn_dim")[1]
        decoder_ffn = metadata.fetch("cohere_transcribe.decoder.ffn_dim")[1]
        vocab_size = vocabulary.length

        validate_shape!("encoder_decoder_proj.weight", [decoder_model, encoder_model])
        validate_shape!("transf_decoder._embedding.token_embedding.weight", [vocab_size, decoder_model])
        validate_shape!("log_softmax.mlp.layer0.weight", [vocab_size, decoder_model])
        validate_shape!("encoder.layers.0.feed_forward1.linear1.weight", [encoder_ffn, encoder_model])
        validate_shape!("transf_decoder._decoder.layers.0.third_sub_layer.dense_in.weight",
                        [decoder_ffn, decoder_model])
      end

      def validate_shape!(source_name, expected)
        tensor = fetch_source_tensor(source_name)
        return if tensor.shape == expected

        raise Error, "Tensor #{tensor.name.inspect} has shape #{tensor.shape.inspect}; expected #{expected.inspect}"
      end

      def validate_dimension!(description, actual, expected)
        return if Integer(actual) == Integer(expected)

        raise Error, "Configured #{description} is #{actual}; checkpoint requires #{expected}"
      end

      def positive_integer(value, description)
        integer = Integer(value)
        raise Error, "#{description} must be positive" unless integer.positive?

        integer
      rescue ArgumentError, TypeError
        raise Error, "#{description} is missing or invalid"
      end

      def seconds_to_samples(value, sample_rate)
        return unless value

        (Float(value) * sample_rate).round
      rescue ArgumentError, TypeError
        raise Error, "Audio window configuration is invalid"
      end

      def require_rank!(tensor, rank)
        return if tensor.shape.length == rank

        raise Error, "Tensor #{tensor.name.inspect} must have rank #{rank}, got #{tensor.shape.inspect}"
      end

      def build_writer(conversion_plan)
        writer = GGUF::Writer.new
        conversion_plan.metadata.each do |key, (type, value)|
          case type
          when :string then writer.add_string(key, value)
          when :uint32 then writer.add_uint32(key, value)
          when :string_array then writer.add_string_array(key, value)
          else raise Error, "Unsupported converter metadata type #{type.inspect}"
          end
        end

        total = conversion_plan.tensors.length
        conversion_plan.tensors.each_with_index do |planned, index|
          writer.add_tensor(
            planned.output_name,
            shape: planned.source_tensor.shape,
            dtype: planned.output_dtype,
            &tensor_writer(planned, index, total)
          )
        end
        writer
      end

      def tensor_writer(planned, index, total)
        lambda do |output|
          tensor = planned.source_tensor
          tensor.reader.write_tensor(
            tensor,
            output,
            target_dtype: planned.output_dtype,
            converter: dtype_converter,
            chunk_bytes: chunk_bytes
          )
          progress&.call(
            index: index + 1,
            total: total,
            output_name: planned.output_name,
            source_name: tensor.name,
            source_dtype: tensor.dtype,
            output_dtype: planned.output_dtype
          )
        end
      end

      def read_json_object(path, description)
        value = JSON.parse(path.read(encoding: "UTF-8"))
        raise Error, "#{description.capitalize} #{path} is not a JSON object" unless value.is_a?(Hash)

        value
      rescue Errno::ENOENT
        raise Error, "#{model_directory} is missing #{path.basename}"
      rescue JSON::ParserError, EncodingError => e
        raise Error, "Cannot read #{description} #{path}: #{e.message}"
      end

      def tally(values)
        values.tally.transform_keys(&:to_s).freeze
      end
    end
  end
end
