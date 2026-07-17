# frozen_string_literal: true

module Cohere
  module Transcribe
    module Internal
      # Normalizes path/reference bytes without applying locale-dependent
      # transcoding. Callers retain control over their public error type and
      # message when the bytes are not valid UTF-8.
      module UTF8
        module_function

        def normalize(value)
          text = value.b.force_encoding(Encoding::UTF_8)
          text if text.valid_encoding?
        end
      end
    end
  end
end
