# frozen_string_literal: true

module Cohere
  module Transcribe
    # Python's str.strip()/str.split() recognize a slightly wider whitespace
    # set than Ruby's String#strip/String#split. Keep public validation and text
    # normalization stable across the two language implementations.
    module PythonText
      WHITESPACE_CLASS = "[[:space:]\u001C-\u001F]"
      LEADING_WHITESPACE = /\A#{WHITESPACE_CLASS}+/u
      TRAILING_WHITESPACE = /#{WHITESPACE_CLASS}+\z/u
      WHITESPACE_RUN = /#{WHITESPACE_CLASS}+/u
      BINARY_WHITESPACE_CLASS = "[\x09-\x0D\x1C-\x20]"
      BINARY_LEADING_WHITESPACE = /\A#{BINARY_WHITESPACE_CLASS}+/n
      BINARY_TRAILING_WHITESPACE = /#{BINARY_WHITESPACE_CLASS}+\z/n
      BINARY_WHITESPACE_RUN = /#{BINARY_WHITESPACE_CLASS}+/n
      # Python 3.12 uses Unicode 15.0. Ruby 4 recognizes a small Unicode 16
      # set whose newly assigned case/compatibility mappings would otherwise
      # make the same input normalize differently from the reference package.
      UNICODE_15_NORMALIZATION_BOUNDARIES = [
        0x1C89,
        0xA7CB, 0xA7CC, 0xA7CE, 0xA7D2, 0xA7D4, 0xA7DA, 0xA7DC, 0xA7F1,
        *(0x10D50..0x10D65),
        *(0x16EA0..0x16EB8),
        *(0x1CCD6..0x1CCF9)
      ].to_h { |codepoint| [codepoint, true] }.freeze
      private_constant :WHITESPACE_CLASS, :LEADING_WHITESPACE, :TRAILING_WHITESPACE, :WHITESPACE_RUN,
                       :BINARY_WHITESPACE_CLASS, :BINARY_LEADING_WHITESPACE, :BINARY_TRAILING_WHITESPACE,
                       :BINARY_WHITESPACE_RUN,
                       :UNICODE_15_NORMALIZATION_BOUNDARIES

      module_function

      def strip(value)
        leading, trailing, = whitespace_patterns(value)
        value.sub(leading, "").sub(trailing, "")
      end

      def blank?(value)
        strip(value).empty?
      end

      def split(value)
        stripped = strip(value)
        _, _, run = whitespace_patterns(stripped)
        stripped.empty? ? [] : stripped.split(run)
      end

      def collapse(value)
        _, _, run = whitespace_patterns(value)
        strip(value.gsub(run, " "))
      end

      def nfkc_lower(value)
        return value.unicode_normalize(:nfkc).downcase unless value.each_codepoint.any? do |codepoint|
          UNICODE_15_NORMALIZATION_BOUNDARIES.key?(codepoint)
        end

        normalized = +""
        ordinary = +""
        value.each_char do |character|
          if UNICODE_15_NORMALIZATION_BOUNDARIES.key?(character.ord)
            normalized << ordinary.unicode_normalize(:nfkc).downcase
            ordinary.clear
            normalized << character
          else
            ordinary << character
          end
        end
        normalized << ordinary.unicode_normalize(:nfkc).downcase
      end

      def whitespace_class
        WHITESPACE_CLASS
      end

      def whitespace_patterns(value)
        if value.encoding == Encoding::ASCII_8BIT
          [BINARY_LEADING_WHITESPACE, BINARY_TRAILING_WHITESPACE, BINARY_WHITESPACE_RUN]
        else
          [LEADING_WHITESPACE, TRAILING_WHITESPACE, WHITESPACE_RUN]
        end
      end
      private_class_method :whitespace_patterns
    end
    private_constant :PythonText
  end
end
