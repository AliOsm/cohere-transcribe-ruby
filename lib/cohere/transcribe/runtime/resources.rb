# frozen_string_literal: true

require "monitor"
require "weakref"

module Cohere
  module Transcribe
    module Runtime
      # Owns one reusable native ASR session. Across all ModelResources
      # instances, only one may retain ASR at a time; this bounds process memory
      # when applications create multiple reusable Transcribers.
      class ModelResources
        OWNER_GUARD = Monitor.new

        class << self
          def evict_current_asr_owner
            OWNER_GUARD.synchronize do
              owner = current_owner_locked
              owner&.send(:evict_asr_locked)
            end
            nil
          end

          def current_asr_owner
            OWNER_GUARD.synchronize { current_owner_locked }
          end

          private

          def current_owner_locked
            reference = @owner_reference
            return nil unless reference

            reference.__getobj__
          rescue WeakRef::RefError
            @owner_reference = nil
            nil
          end

          def claim_locked(owner)
            @owner_reference = WeakRef.new(owner)
          end

          def release_locked(owner)
            @owner_reference = nil if current_owner_locked.equal?(owner)
          end
        end

        attr_reader :asr_key, :batch_controller

        def initialize
          @asr_key = nil
          @asr_session = nil
          @batch_controller = nil
          @closed = false
        end

        # Returns [session, loaded]. A key change evicts this instance's prior
        # session, while acquisition also evicts a different process owner.
        def acquire_asr(key)
          raise ArgumentError, "an ASR loader block is required" unless block_given?

          self.class::OWNER_GUARD.synchronize do
            ensure_open!
            evict_asr_locked if @asr_session && @asr_key != key

            owner = self.class.send(:current_owner_locked)
            owner.send(:evict_asr_locked) if owner && !owner.equal?(self)
            self.class.send(:claim_locked, self)

            loaded = false
            unless @asr_session
              installed = false
              begin
                session = yield
                raise TranscriptionRuntimeError, "ASR loader returned no native session" if session.nil?

                @asr_session = session
                @asr_key = immutable_key(key)
                @batch_controller = nil
                loaded = true
                installed = true
              ensure
                self.class.send(:release_locked, self) unless installed
              end
            end
            [@asr_session, loaded].freeze
          end
        end

        def install_batch_controller(controller)
          self.class::OWNER_GUARD.synchronize do
            ensure_open!
            raise TranscriptionRuntimeError, "Cannot install an ASR controller before acquiring ASR" unless @asr_session

            @batch_controller ||= controller
          end
        end

        def asr_session
          self.class::OWNER_GUARD.synchronize { @asr_session }
        end

        def asr?
          self.class::OWNER_GUARD.synchronize { !@asr_session.nil? }
        end
        alias has_asr? asr?

        def asr_circuit_broken?
          self.class::OWNER_GUARD.synchronize do
            @batch_controller&.circuit_open?
          end || false
        end

        def evict_asr
          self.class::OWNER_GUARD.synchronize { evict_asr_locked }
          nil
        end

        def close
          self.class::OWNER_GUARD.synchronize do
            return if @closed

            evict_asr_locked
            @closed = true
          end
          nil
        end

        def closed?
          self.class::OWNER_GUARD.synchronize { @closed }
        end

        private

        def ensure_open!
          raise TranscriberClosedError, "Model resources have been closed" if @closed
        end

        def evict_asr_locked
          self.class.send(:release_locked, self)
          session = @asr_session
          @asr_session = nil
          @asr_key = nil
          @batch_controller = nil
          session&.close
          nil
        end

        def immutable_key(value)
          case value
          when Array
            value.map { |item| immutable_key(item) }.freeze
          when Hash
            value.to_h { |key, item| [immutable_key(key), immutable_key(item)] }.freeze
          when String
            value.dup.freeze
          else
            value
          end
        end
      end
    end
  end
end
