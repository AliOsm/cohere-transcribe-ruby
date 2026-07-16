# frozen_string_literal: true

require "test_helper"
require "cohere/transcribe/internal/session_ownership"

class Cohere::Transcribe::SessionOwnershipTest < Minitest::Test
  Ownership = Cohere::Transcribe::Internal::SessionOwnership

  class Session
    attr_reader :close_count

    def initialize
      @close_count = 0
    end

    def close
      @close_count += 1
    end
  end

  def test_default_close_detaches_and_closes_exactly_once
    ownership = Ownership.new
    session = Session.new

    ownership.install(session)
    assert_same session, ownership.session
    ownership.close
    ownership.close

    assert_nil ownership.session
    assert_equal 1, session.close_count
  end

  def test_custom_close_runs_after_detachment_and_remains_idempotent_after_failure
    observed = []
    ownership = nil
    close = lambda do |session|
      observed << [session, ownership.session]
      raise "close failed"
    end
    ownership = Ownership.new(close: close)
    session = Object.new
    ownership.install(session)

    error = assert_raises(RuntimeError) { ownership.close }
    assert_equal "close failed", error.message
    assert_equal [[session, nil]], observed
    assert_nil ownership.close
  end

  def test_finalize_suppresses_close_failures
    calls = 0
    ownership = Ownership.new(close: lambda { |_session|
      calls += 1
      raise Exception, "shutdown failure" # rubocop:disable Lint/RaiseException
    })
    ownership.install(Object.new)

    assert_nil ownership.finalize
    assert_nil ownership.finalize
    assert_equal 1, calls
  end

  def test_install_rejects_overwriting_a_live_session_with_the_configured_message
    ownership = Ownership.new(installed_error: "already retained")
    ownership.install(Session.new)

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      ownership.install(Session.new)
    end
    assert_equal "already retained", error.message
  ensure
    ownership&.close
  end
end
