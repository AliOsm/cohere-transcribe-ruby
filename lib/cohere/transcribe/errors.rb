# frozen_string_literal: true

module Cohere
  module Transcribe
    # Keep the generated gem's original base error compatible when this file is
    # loaded directly as well as through the eventual public entry point.
    class Error < StandardError; end unless const_defined?(:Error, false)

    # Base exception for programmatic transcription failures.
    class TranscriptionError < Error; end

    # Invalid or unsupported transcription configuration.
    class TranscriptionConfigurationError < TranscriptionError; end

    # Invalid audio input or output planning request.
    class TranscriptionInputError < TranscriptionError; end

    # Dependency, device, model, or execution initialization failure.
    class TranscriptionRuntimeError < TranscriptionError; end

    # A user-provided progress callback raised an exception.
    class ProgressCallbackError < TranscriptionError
      attr_reader :original

      def initialize(original)
        @original = original
        super("Progress callback failed: #{original.class}: #{original}")
      end
    end

    # Operation attempted after a transcriber was closed.
    class TranscriberClosedError < TranscriptionError; end

    # A reentrant or otherwise conflicting transcriber operation was attempted.
    class TranscriberBusyError < TranscriptionError; end

    # Aggregate failure that retains every completed per-file result.
    class BatchTranscriptionError < TranscriptionError
      attr_reader :run

      def initialize(run)
        @run = run
        failed_count = run.failed.length
        run_error_count = run.errors.length
        message = if failed_count.positive? && run_error_count.positive?
                    "#{failed_count} transcription file(s) failed; " \
                      "run errors: #{run_error_count}"
                  elsif failed_count.positive?
                    "#{failed_count} transcription file(s) failed"
                  else
                    "Transcription run failed with #{run_error_count} run error(s)"
                  end
        super(message)
      end
    end
  end
end
