# frozen_string_literal: true

require_relative "../errors"

module Cohere
  module Transcribe
    module Internal
      # Detached, thread-safe session state for explicit close and ObjectSpace
      # finalization. The close callable must not retain the owner whose
      # finalizer holds this object.
      class SessionOwnership
        CLOSE_RUBY_SESSION = :close.to_proc.freeze

        def self.finalizer_for(ownership)
          proc { |_object_id| ownership.finalize }
        end

        def initialize(close: CLOSE_RUBY_SESSION, installed_error: "Session ownership is already installed")
          raise ArgumentError, "close must respond to call" unless close.respond_to?(:call)

          @close = close
          @installed_error = installed_error.freeze
          @mutex = Mutex.new
          @session = nil
        end

        def session
          @mutex.synchronize { @session }
        end

        def install(session)
          @mutex.synchronize do
            raise TranscriptionRuntimeError, @installed_error if @session

            @session = session
          end
        end

        def close
          Thread.handle_interrupt(Object => :never) do
            session = @mutex.synchronize do
              current = @session
              @session = nil
              current
            end
            return unless session

            @close.call(session)
          end
          nil
        end

        def finalize
          close
        rescue Exception # rubocop:disable Lint/RescueException -- finalizers must not escape during GC or shutdown
          nil
        end
      end
    end
  end
end
