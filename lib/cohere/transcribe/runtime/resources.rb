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

        # The global owner record and the ObjectSpace finalizer retain this
        # state, never the ModelResources instance itself. It also lets a new
        # owner synchronously evict a collected predecessor before loading.
        class SessionOwnership
          def initialize
            @mutex = Mutex.new
            @session = nil
          end

          def session
            @mutex.synchronize { @session }
          end

          def install(session)
            @mutex.synchronize do
              raise TranscriptionRuntimeError, "ASR session ownership is already installed" if @session

              @session = session
            end
          end

          def close
            session = @mutex.synchronize do
              current = @session
              @session = nil
              current
            end
            return unless session

            session.close
            nil
          end

          def finalize
            close
          rescue Exception # rubocop:disable Lint/RescueException -- finalizers must not escape during GC or shutdown
            nil
          end
        end
        private_constant :SessionOwnership

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
            cleanup_abandoned_owner_locked
            nil
          end

          def claim_locked(owner)
            @owner_reference = WeakRef.new(owner)
            @owner_ownership = owner.send(:asr_ownership)
          end

          def release_locked(owner)
            return unless @owner_ownership.equal?(owner.send(:asr_ownership))

            @owner_reference = nil
            @owner_ownership = nil
          end

          def cleanup_abandoned_owner_locked
            ownership = @owner_ownership
            @owner_reference = nil
            @owner_ownership = nil
            ownership&.close
          end

          def finalizer_for(ownership)
            proc { |_object_id| finalize_ownership(ownership) }
          end

          def finalize_ownership(ownership)
            OWNER_GUARD.synchronize do
              if @owner_ownership.equal?(ownership)
                @owner_reference = nil
                @owner_ownership = nil
              end
              ownership.finalize
            end
          rescue Exception # rubocop:disable Lint/RescueException -- finalizers must not escape during GC or shutdown
            nil
          end
        end

        attr_reader :asr_key, :batch_controller

        def initialize
          @asr_key = nil
          @asr_ownership = SessionOwnership.new
          @batch_controller = nil
          @closed = false
          ObjectSpace.define_finalizer(
            self,
            self.class.send(:finalizer_for, @asr_ownership)
          )
        end

        # Returns [session, loaded]. A key change evicts this instance's prior
        # session, while acquisition also evicts a different process owner.
        def acquire_asr(key)
          raise ArgumentError, "an ASR loader block is required" unless block_given?

          self.class::OWNER_GUARD.synchronize do
            ensure_open!
            owned_key = immutable_key(key)
            evict_asr_locked if @asr_ownership.session && @asr_key != owned_key

            owner = self.class.send(:current_owner_locked)
            owner.send(:evict_asr_locked) if owner && !owner.equal?(self)
            self.class.send(:claim_locked, self)

            loaded = false
            unless @asr_ownership.session
              installed = false
              session = nil
              begin
                session = yield
                raise TranscriptionRuntimeError, "ASR loader returned no native session" if session.nil?

                Thread.handle_interrupt(Exception => :never) do
                  @asr_ownership.install(session)
                  @asr_key = owned_key
                  @batch_controller = nil
                  loaded = true
                  installed = true
                end
              rescue Exception # rubocop:disable Lint/RescueException -- roll back asynchronous loader interruption
                begin
                  if @asr_ownership.session.equal?(session)
                    evict_asr_locked
                  else
                    session&.close
                  end
                rescue Exception # rubocop:disable Lint/RescueException -- preserve the loader failure
                  nil
                end
                raise
              ensure
                self.class.send(:release_locked, self) unless installed
              end
            end
            [@asr_ownership.session, loaded].freeze
          end
        end

        def install_batch_controller(controller)
          self.class::OWNER_GUARD.synchronize do
            ensure_open!
            raise TranscriptionRuntimeError, "Cannot install an ASR controller before acquiring ASR" unless @asr_ownership.session

            @batch_controller ||= controller
          end
        end

        def asr_session
          self.class::OWNER_GUARD.synchronize { @asr_ownership.session }
        end

        def asr?
          self.class::OWNER_GUARD.synchronize { !@asr_ownership.session.nil? }
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
            ObjectSpace.undefine_finalizer(self)
          end
          nil
        end

        def closed?
          self.class::OWNER_GUARD.synchronize { @closed }
        end

        private

        attr_reader :asr_ownership

        def ensure_open!
          raise TranscriberClosedError, "Model resources have been closed" if @closed
        end

        def evict_asr_locked
          self.class.send(:release_locked, self)
          @asr_key = nil
          @batch_controller = nil
          @asr_ownership.close
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
