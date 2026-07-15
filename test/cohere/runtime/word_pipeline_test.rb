# frozen_string_literal: true

require "test_helper"
require "cohere/transcribe/runtime/word_pipeline"
require "timeout"

class Cohere::Transcribe::RuntimeWordPipelineTest < Minitest::Test
  WordPipeline = Cohere::Transcribe::Runtime::WordPipeline
  Entry = Data.define(:path)
  Decoded = Data.define(:samples, :backend)

  class RecordingDecoder
    attr_reader :calls

    def initialize(samples: 16_000, actual_backend: nil)
      @samples = samples
      @actual_backend = actual_backend
      @calls = []
    end

    def decode(path, **options)
      @calls << [path, options]
      Decoded.new(
        samples: Array.new(@samples, 0.0),
        backend: @actual_backend || options.fetch(:backend)
      )
    end
  end

  def test_all_asr_finishes_before_one_eviction_and_ordered_alignment
    events = []
    completed = []
    coordinator = WordPipeline::Coordinator.new(
      total: 4,
      fixed_results: { 1 => :skipped },
      evict_asr: -> { events << :asr_evicted },
      align: lambda do |work|
        events << [:align, work.index]
        :"aligned-#{work.index}"
      end,
      completed: ->(index, current, total, result) { completed << [index, current, total, result] }
    )

    results = coordinator.run(%i[first third fourth]) do |prepared|
      index = { first: 0, third: 2, fourth: 3 }.fetch(prepared)
      events << [:asr, index]
      alignment_work(index)
    end

    assert_equal(
      [[:asr, 0], [:asr, 2], [:asr, 3], :asr_evicted, [:align, 0], [:align, 2], [:align, 3]],
      events
    )
    assert_equal %i[aligned-0 skipped aligned-2 aligned-3], results
    assert_equal [0, 1, 2, 3], completed.map(&:first)
    completion_counts = completed.map { |row| row.fetch(1) }
    assert_equal [1, 2, 3, 4], completion_counts
    assert(completed.all? { |row| row.fetch(2) == 4 })
  end

  def test_asr_failure_still_evicts_once_and_never_starts_alignment
    evictions = 0
    alignments = 0
    coordinator = WordPipeline::Coordinator.new(
      total: 2,
      evict_asr: -> { evictions += 1 },
      align: ->(_work) { alignments += 1 }
    )

    error = assert_raises(Interrupt) do
      coordinator.run(%i[first second]) do |prepared|
        raise Interrupt if prepared == :second

        alignment_work(0)
      end
    end
    assert_instance_of Interrupt, error
    assert_equal 1, evictions
    assert_equal 0, alignments
  end

  def test_final_failures_are_values_and_progress_remains_input_ordered
    completed = []
    coordinator = WordPipeline::Coordinator.new(
      total: 3,
      fixed_results: { 0 => :preflight_failure },
      evict_asr: -> {},
      align: ->(work) { work.index == 1 ? :alignment_failure : :completed },
      completed: ->(index, _current, _total, result) { completed << [index, result] }
    )

    results = coordinator.run(%i[one two]) do |prepared|
      alignment_work(prepared == :one ? 1 : 2)
    end

    assert_equal %i[preflight_failure alignment_failure completed], results
    assert_equal [[0, :preflight_failure], [1, :alignment_failure], [2, :completed]], completed
  end

  def test_checkpoint_alignment_work_can_enter_as_a_fixed_outcome_without_asr
    events = []
    fixed = alignment_work(0)
    coordinator = WordPipeline::Coordinator.new(
      total: 1,
      fixed_results: { 0 => fixed },
      evict_asr: -> { events << :evicted },
      align: lambda do |work|
        events << [:align, work.index]
        :resumed
      end
    )

    results = coordinator.run([]) { flunk "a fixed checkpoint must bypass ASR" }

    assert_equal [:resumed], results
    assert_equal [:evicted, [:align, 0]], events
  end

  def test_no_model_eviction_occurs_when_every_outcome_is_final_or_empty
    evictions = 0
    finals = WordPipeline::Coordinator.new(
      total: 1,
      fixed_results: { 0 => :skipped },
      evict_asr: -> { evictions += 1 },
      align: ->(_work) { flunk "a final result cannot be aligned" }
    )
    assert_equal [:skipped], finals.run([]) { flunk "a fixed result cannot run ASR" }

    empty = WordPipeline::Coordinator.new(
      total: 1,
      fixed_results: { 0 => alignment_work(0, audio_required: false) },
      evict_asr: -> { evictions += 1 },
      align: ->(_work) { :empty }
    )
    assert_equal [:empty], empty.run([]) { flunk "a fixed empty result cannot run ASR" }
    assert_equal 0, evictions
  end

  def test_scheduler_rejects_omitted_duplicate_and_out_of_range_inputs
    base = -> { WordPipeline::Coordinator.new(total: 2, evict_asr: -> {}, align: ->(_work) { :ok }) }

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      base.call.run([:only]) { alignment_work(0) }
    end
    assert_match(/omitted input index\(es\): 1/, error.message)

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      base.call.run(%i[first duplicate]) { alignment_work(0) }
    end
    assert_match(/input 0 more than once/, error.message)

    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      base.call.run(%i[first second]) { |item| alignment_work(item == :first ? 0 : 2) }
    end
    assert_match(/out-of-range input index 2/, error.message)
  end

  def test_alignment_work_is_deeply_immutable_and_cannot_retain_pcm
    segment_times = [[0, 1]]
    speech_spans = [[0.1, 0.9]]
    work = alignment_work(0, segment_times: segment_times, speech_spans: speech_spans)
    segment_times.first[0] = 9
    speech_spans << [1, 2]

    assert_equal [[0.0, 1.0]], work.segment_times
    assert_equal [[0.1, 0.9]], work.speech_spans
    assert work.segment_times.frozen?
    assert work.segment_times.first.frozen?
    refute_includes WordPipeline::AlignmentWork.members, :decoded
    refute_includes WordPipeline::AlignmentWork.members, :samples
  end

  def test_concrete_redecoder_skips_empty_transcripts_without_touching_audio
    decoder = RecordingDecoder.new
    redecoder = WordPipeline::ConcreteRedecoder.new(
      decoder: decoder, sample_rate: 16_000, memory_byte_limit: 1_000_000
    )
    work = alignment_work(0, audio_required: false, decode_backend: nil)

    assert_nil redecoder.call(work)
    assert_empty decoder.calls
  end

  def test_concrete_redecoder_uses_recorded_backend_and_full_per_file_budget
    decoder = RecordingDecoder.new
    redecoder = WordPipeline::ConcreteRedecoder.new(
      decoder: decoder,
      sample_rate: 16_000,
      memory_byte_limit: 123_456
    )

    decoded = redecoder.call(alignment_work(0, decode_backend: "ffmpeg"))

    assert_equal 16_000, decoded.samples.length
    path, options = decoder.calls.fetch(0)
    assert_equal Pathname("/tmp/input-0.wav"), path
    assert_equal "ffmpeg", options.fetch(:backend)
    assert_equal 16_000, options.fetch(:sample_rate)
    assert_equal 123_456, options.fetch(:max_decoded_bytes)
  end

  def test_concrete_redecoder_rejects_fallback_and_sample_drift
    fallback = RecordingDecoder.new(actual_backend: "libsndfile")
    redecoder = WordPipeline::ConcreteRedecoder.new(
      decoder: fallback, sample_rate: 16_000, memory_byte_limit: 1_000_000
    )
    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      redecoder.call(alignment_work(0, decode_backend: "ffmpeg"))
    end
    assert_match(/backend changed/, error.message)

    drift = RecordingDecoder.new(samples: 15_999)
    redecoder = WordPipeline::ConcreteRedecoder.new(
      decoder: drift, sample_rate: 16_000, memory_byte_limit: 1_000_000
    )
    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
      redecoder.call(alignment_work(0, decode_backend: "ffmpeg"))
    end
    assert_match(/15999 != 16000/, error.message)

    automatic = alignment_work(0, decode_backend: "auto")
    error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) { redecoder.call(automatic) }
    assert_match(/without a concrete ASR backend/, error.message)
  end

  def test_pair_budget_reload_overlaps_only_when_adjacent_pcm_fits
    next_started = Queue.new
    events = []
    reload = lambda do |work|
      events << [:decode, work.index]
      next_started << true if work.index == 1
      :decoded
    end
    works = [alignment_work(0), alignment_work(1)]
    pipeline = WordPipeline::PairBudgetReload.new(
      works,
      memory_byte_limit: 128_000,
      reload: reload
    )

    pipeline.each do |work, reload_result|
      assert reload_result.ok?
      assert_equal :decoded, reload_result.decoded
      if work.index.zero?
        Timeout.timeout(2) { next_started.pop }
        events << :first_alignment
      end
    end

    assert_operator events.index([:decode, 1]), :<, events.index(:first_alignment)
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-alignment-reload-") })
  end

  def test_pair_budget_reload_defers_oversized_pair_until_current_is_consumed
    events = []
    works = [alignment_work(0), alignment_work(1)]
    pipeline = WordPipeline::PairBudgetReload.new(
      works,
      memory_byte_limit: 100_000,
      reload: lambda do |work|
        events << [:decode, work.index]
        :decoded
      end
    )

    pipeline.each do |work, reload_result|
      assert reload_result.ok?
      events << [:align, work.index]
      refute_includes events, [:decode, 1] if work.index.zero?
    end

    assert_equal [[:decode, 0], [:align, 0], [:decode, 1], [:align, 1]], events
  end

  def test_pair_budget_reload_cancels_lookahead_when_alignment_aborts
    second_started = Queue.new
    pipeline = WordPipeline::PairBudgetReload.new(
      [alignment_work(0), alignment_work(1)],
      memory_byte_limit: 128_000,
      reload: lambda do |work|
        next :decoded if work.index.zero?

        second_started << true
        sleep
      end
    )

    assert_raises(Interrupt) do
      pipeline.each do |pair|
        work = pair.first
        next unless work.index.zero?

        Timeout.timeout(2) { second_started.pop }
        raise Interrupt
      end
    end
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-alignment-reload-") })
  end

  def test_reload_thread_is_killed_even_if_cooperative_cancel_hook_fails
    started = Queue.new
    reload = Object.new
    reload.define_singleton_method(:call) do |_work|
      started << true
      sleep
    end
    reload.define_singleton_method(:cancel) { raise "cancellation backend failed" }
    pipeline = WordPipeline::PairBudgetReload.new(
      [alignment_work(0)],
      memory_byte_limit: 128_000,
      reload: reload,
      started: lambda do
        Timeout.timeout(2) { started.pop }
        raise Interrupt
      end
    )

    assert_raises(Interrupt) { pipeline.each { flunk "alignment must not start" } }
    refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-alignment-reload-") })
  end

  def test_coordinator_can_feed_bounded_redecoded_pcm_to_alignment
    events = []
    coordinator = WordPipeline::Coordinator.new(
      total: 2,
      evict_asr: -> { events << :evicted },
      reload: lambda do |work|
        events << [:reload, work.index]
        "pcm-#{work.index}"
      end,
      memory_byte_limit: 128_000,
      align: lambda do |work, reload_result|
        events << [:align, work.index, reload_result.decoded]
        "done-#{work.index}"
      end
    )

    results = coordinator.run(%i[first second]) do |item|
      index = item == :first ? 0 : 1
      events << [:asr, index]
      alignment_work(index)
    end

    assert_equal %w[done-0 done-1], results
    eviction = events.index(:evicted)
    asr_indices = events.each_index.select { |index| events[index].is_a?(Array) && events[index].first == :asr }
    reload_indices = events.each_index.select { |index| events[index].is_a?(Array) && events[index].first == :reload }
    assert(asr_indices.all? { |index| index < eviction })
    assert(reload_indices.all? { |index| index > eviction })
  end

  def test_reload_errors_are_delivered_per_file_and_do_not_cancel_later_work
    coordinator = WordPipeline::Coordinator.new(
      total: 2,
      evict_asr: -> {},
      reload: lambda do |work|
        raise "bad decode" if work.index.zero?

        :pcm
      end,
      memory_byte_limit: 128_000,
      align: lambda do |_work, reload_result|
        reload_result.ok? ? :completed : reload_result.error.message
      end
    )

    results = coordinator.run(%i[first second]) do |item|
      alignment_work(item == :first ? 0 : 1)
    end

    assert_equal ["bad decode", :completed], results
  end

  private

  def alignment_work(index, segment_times: [[0.0, 1.0]], speech_spans: [[0.0, 1.0]],
                     decode_backend: "ffmpeg", audio_required: true)
    WordPipeline::AlignmentWork.new(
      index: index,
      entry: Entry.new(path: Pathname("/tmp/input-#{index}.wav")),
      plan: { publication: index.even? },
      result: :"asr-#{index}",
      segment_times: segment_times,
      speech_spans: speech_spans,
      generation_id: "generation-#{index}",
      decode_backend: decode_backend,
      expected_sample_count: 16_000,
      audio_required: audio_required,
      source_snapshot: [1, 2, 3]
    )
  end
end
