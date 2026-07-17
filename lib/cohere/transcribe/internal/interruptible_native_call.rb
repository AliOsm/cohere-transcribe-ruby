# frozen_string_literal: true

module Cohere
  module Transcribe
    module Internal
      # Runs a blocking foreign call on a dedicated worker so the caller remains
      # interruptible. Callbacks are retained only for this invocation; callers
      # remain responsible for supplying non-poisoning cancellation behavior.
      module InterruptibleNativeCall
        module_function

        def run(cancel:, join_interval:, missing_outcome:, thread_name: nil)
          raise ArgumentError, "an operation block is required" unless block_given?

          outcome = Queue.new
          worker = nil
          Thread.handle_interrupt(Exception => :on_blocking) do
            worker = Thread.new do
              outcome << [:returned, yield]
            rescue Exception => e # rubocop:disable Lint/RescueException -- transfer native cancellation intact
              outcome << [:raised, e]
            end
            worker.name = thread_name if thread_name && worker.respond_to?(:name=)
            worker.report_on_exception = false
            worker.join
          ensure
            if worker&.alive?
              Thread.handle_interrupt(Exception => :never) do
                cancel_and_hard_join(worker, cancel, join_interval)
              end
            end
          end

          status, value = begin
            outcome.pop(true)
          rescue ThreadError
            raise missing_outcome
          end
          raise value if status == :raised

          value
        end

        # Cancellation can arrive before the worker enters the foreign call.
        # Retry until it exits, suppressing every secondary exception so the
        # first caller exception remains intact.
        def cancel_and_hard_join(worker, cancel, join_interval)
          loop do
            begin
              return if worker.join(0)
            rescue Exception # rubocop:disable Lint/RescueException -- preserve the first caller exception
              nil
            end
            begin
              cancel.call
            rescue Exception # rubocop:disable Lint/RescueException -- preserve the first caller exception
              nil
            end
            begin
              return if worker.join(join_interval)
            rescue Exception # rubocop:disable Lint/RescueException -- preserve the first caller exception
              nil
            end
          end
        end
        private_class_method :cancel_and_hard_join
      end
    end
  end
end
