# frozen_string_literal: true

require_relative "errors"
require_relative "types"
require_relative "input"
require_relative "loader"

module Cohere
  module Transcribe
    # Reusable, serialized transcription session with lazy model loading.
    class Transcriber
      OPTIONS_UNSET = Object.new.freeze
      private_constant :OPTIONS_UNSET

      attr_reader :options, :progress

      def initialize(positional_options = OPTIONS_UNSET, options: OPTIONS_UNSET, progress: nil, engine_factory: nil)
        positional_given = !positional_options.equal?(OPTIONS_UNSET)
        keyword_given = !options.equal?(OPTIONS_UNSET)
        raise ArgumentError, "options must be supplied either positionally or by keyword, not both" if positional_given && keyword_given

        selected_options = keyword_given ? options : positional_options
        selected_options = nil if selected_options.equal?(OPTIONS_UNSET)
        @options = selected_options.nil? ? TranscriptionOptions.new : selected_options
        raise TypeError, "options must be a TranscriptionOptions instance" unless @options.is_a?(TranscriptionOptions)
        raise TypeError, "progress must be callable or nil" if !progress.nil? && !progress.respond_to?(:call)

        @progress = progress
        @engine_factory = if engine_factory.nil?
                            lambda do
                              Loader.load_runtime!
                              Runtime::Engine.new(@options, progress: @progress)
                            end
                          else
                            engine_factory
                          end
        @implementation = nil
        @lock = Mutex.new
        @closed = false
        @closing = false
      end

      def transcribe(audio, raise_on_error: false)
        started = monotonic
        normalized = Input.normalize(audio)
        implementation, import_seconds, wait_seconds = implementation()
        run = implementation.transcribe(
          normalized,
          raise_on_error: raise_on_error,
          runtime_import_seconds: import_seconds,
          serialization_wait_seconds: wait_seconds
        )
        with_elapsed(run, monotonic - started)
      rescue BatchTranscriptionError => e
        raise BatchTranscriptionError.new(with_elapsed(e.run, monotonic - started))
      end

      def close
        implementation = @lock.synchronize do
          return if @closed
          raise TranscriberBusyError, "This Transcriber is already being closed" if @closing

          @closing = true
          @implementation
        end
        begin
          implementation&.close
        rescue Exception
          @lock.synchronize do
            @closing = false
          end
          raise
        end
        @lock.synchronize do
          @closed = true
          @closing = false
          @implementation = nil if @implementation.equal?(implementation)
        end
        nil
      end

      def closed?
        @closed
      end

      def self.open(...)
        transcriber = new(...)
        return transcriber unless block_given?

        begin
          yield transcriber
        ensure
          transcriber.close
        end
      end

      private

      def implementation
        started = monotonic
        @lock.synchronize do
          wait_seconds = monotonic - started
          raise TranscriberClosedError, "This Transcriber has been closed" if @closed
          raise TranscriberBusyError, "This Transcriber is being closed" if @closing

          import_seconds = 0.0
          unless @implementation
            import_started = monotonic
            @implementation = @engine_factory.call
            unless @implementation.respond_to?(:transcribe) && @implementation.respond_to?(:close)
              @implementation = nil
              raise TranscriptionRuntimeError, "engine_factory returned an invalid transcription engine"
            end
            import_seconds = monotonic - import_started
          end
          [@implementation, import_seconds, wait_seconds]
        end
      rescue TranscriptionError
        raise
      rescue SystemExit => e
        detail = e.message.to_s
        detail = "Transcription runtime initialization failed" if detail.empty? || detail == "SystemExit"
        raise TranscriptionRuntimeError, detail
      rescue ScriptError => e
        raise TranscriptionRuntimeError,
              "Cannot initialize the transcription runtime: #{e.class}: #{e.message}"
      rescue StandardError => e
        raise TranscriptionRuntimeError,
              "Cannot initialize the transcription runtime: #{e.class}: #{e.message}"
      end

      def with_elapsed(run, elapsed)
        statistics = run.statistics.with(
          elapsed_seconds: elapsed,
          real_time_factor_x: elapsed.positive? ? run.statistics.successful_audio_seconds / elapsed : 0.0
        )
        run.with(statistics: statistics)
      end

      def monotonic
        Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end
    end

    module_function

    def transcribe(audio, options: nil, progress: nil, raise_on_error: false)
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      transcriber = Transcriber.new(options, progress: progress)
      run = nil
      batch_error = nil
      begin
        begin
          run = transcriber.transcribe(audio, raise_on_error: raise_on_error)
        rescue BatchTranscriptionError => e
          batch_error = e
        end
      ensure
        transcriber.close
      end
      elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
      selected = batch_error ? batch_error.run : run
      statistics = selected.statistics.with(
        elapsed_seconds: elapsed,
        real_time_factor_x: elapsed.positive? ? selected.statistics.successful_audio_seconds / elapsed : 0.0
      )
      selected = selected.with(statistics: statistics)
      raise BatchTranscriptionError.new(selected) if batch_error

      selected
    end
  end
end
