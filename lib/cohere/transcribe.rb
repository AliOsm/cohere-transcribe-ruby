# frozen_string_literal: true

require_relative "transcribe/version"
require_relative "transcribe/constants"
require_relative "transcribe/errors"
require_relative "transcribe/types"
require_relative "transcribe/input"
require_relative "transcribe/loader"
require_relative "transcribe/api"

module Cohere
  module Transcribe
    # Ruby counterpart to the Python package's explicit root export contract.
    # Internal implementation constants remain reachable through autoloads but
    # are intentionally absent from this stable list.
    PUBLIC_API = %w[
      BatchTranscriptionError Error ProgressCallbackError ProgressEvent
      PublicationOptions SubtitleCue Transcriber TranscriberBusyError
      TranscriberClosedError TranscriptionConfigurationError TranscriptionError
      TranscriptionInputError TranscriptionOptions TranscriptionProvenance
      TranscriptionResult TranscriptionRun TranscriptionRuntimeError
      TranscriptionSegment TranscriptionStatistics TranscriptionWord VERSION
      transcribe
    ].freeze
  end
end
