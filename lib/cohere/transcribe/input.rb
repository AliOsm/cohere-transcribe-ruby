# frozen_string_literal: true

require "find"

require_relative "constants"
require_relative "errors"
require_relative "internal/utf8"
require_relative "python_text"

module Cohere
  module Transcribe
    InputEntry = Data.define(:path, :relative_path)

    module Input
      module_function

      def normalize(audio)
        sequence = audio.to_ary if !path_like?(audio) && audio.respond_to?(:to_ary)
        values = if path_like?(audio)
                   [audio]
                 elsif sequence.is_a?(Array)
                   raise TranscriptionInputError, "audio must contain at least one path" if sequence.empty?

                   sequence
                 else
                   raise TranscriptionInputError, "audio must be one path or an ordered sequence of paths"
                 end

        values.each_with_index.map do |value, index|
          label = values.length == 1 && path_like?(audio) ? "audio" : "audio[#{index}]"
          raise TranscriptionInputError, "#{label} must be a string or path-like object" unless path_like?(value)

          text = value.is_a?(String) ? value : value.to_path
          raise TranscriptionInputError, "#{label} must resolve to a text path" unless text.is_a?(String)

          text = utf8_path(text, label)
          raise TranscriptionInputError, "#{label} must not be empty" if PythonText.blank?(text)

          text.freeze
        rescue EncodingError, TypeError, ArgumentError, SystemCallError => e
          raise e if e.is_a?(TranscriptionInputError)

          raise TranscriptionInputError, "#{label} is not a valid text path: #{e.message}"
        end.freeze
      rescue EncodingError, TypeError, ArgumentError, SystemCallError => e
        raise e if e.is_a?(TranscriptionInputError)

        raise TranscriptionInputError, "audio is not a valid ordered path sequence: #{e.message}"
      end

      def expand(audio, recursive: true)
        inputs = normalize(audio)
        seen = {}
        entries = inputs.flat_map do |raw|
          source = strict_realpath(raw)
          candidates = if source.file?
                         [[source, Pathname(source.basename.to_s)]]
                       elsif source.directory?
                         directory_candidates(source, recursive)
                       else
                         raise TranscriptionInputError, "Input is not a regular file or directory: #{source}"
                       end
          candidates.filter_map do |path, relative_path|
            canonical = strict_realpath(path)
            next if seen.key?(canonical.to_s)

            seen[canonical.to_s] = true
            InputEntry.new(path: canonical.freeze, relative_path: relative_path.freeze)
          end
        end
        raise TranscriptionInputError, "No audio files found in the supplied inputs." if entries.empty?

        entries.freeze
      end

      def directory_candidates(source, recursive)
        paths = if recursive
                  found = []
                  Find.find(source.to_s) do |name|
                    path = Pathname(name)
                    found << path if path.file? && AUDIO_EXTENSIONS.include?(path.extname.downcase)
                  end
                  found
                else
                  source.children.select do |path|
                    path.file? && AUDIO_EXTENSIONS.include?(path.extname.downcase)
                  end
                end
        paths = paths.map { |path| Pathname(utf8_path(path.to_s, "discovered audio")) }
        paths.sort_by { |path| path.to_s.downcase(:fold) }.map do |path|
          relative_path = path.relative_path_from(source)
          relative_text = utf8_path(relative_path.to_s, "discovered audio")
          [strict_realpath(path), Pathname(relative_text)]
        end
      rescue EncodingError, SystemCallError, ArgumentError => e
        raise TranscriptionInputError, "Cannot inspect input #{source}: #{e.message}"
      end
      private_class_method :directory_candidates

      def strict_realpath(value)
        resolved = Pathname(value).expand_path.realpath
        Pathname(utf8_path(resolved.to_s, "resolved input"))
      rescue Errno::ENOENT
        raise TranscriptionInputError, "Input does not exist: #{Pathname(value).expand_path}"
      rescue EncodingError, SystemCallError, ArgumentError => e
        detail = e.is_a?(Errno::ELOOP) ? "Symlink loop" : e.message
        raise TranscriptionInputError, "Invalid input path #{value.inspect}: #{detail}"
      end
      private_class_method :strict_realpath

      def path_like?(value)
        value.is_a?(String) || value.respond_to?(:to_path)
      end
      private_class_method :path_like?

      def utf8_path(value, label)
        text = Internal::UTF8.normalize(value)
        return text if text

        raise TranscriptionInputError, "#{label} path must contain valid UTF-8"
      end
      private_class_method :utf8_path
    end
  end
end
