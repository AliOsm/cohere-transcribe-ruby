# frozen_string_literal: true

require "tmpdir"
require "test_helper"

class Cohere::Transcribe::RuntimeWordPipelineEngineTest < Minitest::Test
  class PhaseDecoder
    attr_reader :calls

    def initialize(events, drift_on_alignment: nil, empty: false)
      @events = events
      @drift_on_alignment = drift_on_alignment
      @empty = empty
      @calls = []
      @counts = Hash.new(0)
      @gate = Mutex.new
    end

    def decode(path, backend:, **)
      index = File.basename(path.to_s)[/\d+/].to_i
      count = @gate.synchronize do
        @counts[index] += 1
        @calls << [index, @counts[index], backend]
        @counts[index]
      end
      @events << [:decode, index, count, backend]
      length = count == 2 && index == @drift_on_alignment ? 15_999 : 16_000
      value = @empty ? 0.0 : index + 0.25
      Cohere::Transcribe::Audio::Decoded.new(
        samples: Array.new(length, value),
        sample_rate: 16_000,
        backend: "ffmpeg",
        fallback_reason: nil
      )
    end
  end

  class PhaseSession
    attr_reader :closed

    def initialize(events, blank: false)
      @events = events
      @blank = blank
      @closed = false
    end

    def transcribe(samples, language:, offset:, max_new_tokens:)
      index = samples.first.to_f.floor
      @events << [:asr, index]
      result(index, max_new_tokens)
    end

    def transcribe_batch(sample_batches, language:, offsets:, max_new_tokens:)
      sample_batches.map do |samples|
        index = samples.first.to_f.floor
        @events << [:asr, index]
        result(index, max_new_tokens)
      end
    end

    def close
      return if @closed

      @events << :asr_closed
      @closed = true
    end

    private

    def result(index, max_new_tokens)
      Cohere::Transcribe::ASR::NativeResult.new(
        text: @blank ? "   " : "file #{index}",
        segments: [],
        words: [],
        generated_tokens: @blank ? 0 : 2,
        generation_limit: max_new_tokens,
        generation_capacity: 1_000,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end
  end

  class PhaseProvider
    attr_reader :open_count, :session

    def initialize(events, blank: false, fail_if_opened: false)
      @events = events
      @session = PhaseSession.new(events, blank: blank)
      @fail_if_opened = fail_if_opened
      @open_count = 0
    end

    def resolve(_options)
      Cohere::Transcribe::ResolvedModelIdentity.new(
        model_id: "CohereLabs/cohere-transcribe-arabic-07-2026",
        model_revision: "a" * 40,
        model_format: :dense,
        quantization_config: nil,
        adapter_id: nil,
        adapter_revision: nil
      )
    end

    def open(_identity, _options)
      raise "checkpoint resume unexpectedly opened Dense ASR" if @fail_if_opened

      @open_count += 1
      @events << :asr_opened
      @session
    end
  end

  class PhaseAligner
    attr_reader :calls, :closed, :load_count, :load_seconds, :emissions_seconds, :viterbi_seconds

    def initialize(events, failure: nil, load_failure: nil)
      @events = events
      @failure = failure
      @load_failure = load_failure
      @calls = []
      @load_count = 0
      @load_seconds = 0.0
      @emissions_seconds = 0.0
      @viterbi_seconds = 0.0
      @closed = false
    end

    def load!
      @load_count += 1
      @events << :aligner_loaded
      raise @load_failure if @load_failure

      self
    end

    def align(audio, segment_times, segment_texts, language:)
      index = audio.first.to_f.floor
      @events << [:align, index]
      raise @failure if @failure

      @calls << [index, audio.length, segment_times, segment_texts, language]
      @load_seconds += 0.01
      @emissions_seconds += 0.02
      @viterbi_seconds += 0.03
      words = segment_texts.each_with_index.flat_map do |text, segment_index|
        text.split.map.with_index do |word, word_index|
          Cohere::Transcribe::TranscriptionWord.new(
            start: segment_times.fetch(segment_index).fetch(0),
            end: segment_times.fetch(segment_index).fetch(1),
            text: word,
            segment_index: segment_index,
            segment_word_index: word_index,
            timing_source: "ctc"
          )
        end
      end
      [words.freeze, 0]
    end

    def close
      return if @closed

      @events << :aligner_closed
      @closed = true
    end
  end

  def test_multi_file_word_mode_has_one_dense_phase_then_one_mms_phase
    with_audio_files(3) do |paths|
      events = []
      decoder = PhaseDecoder.new(events)
      provider = PhaseProvider.new(events)
      aligner = PhaseAligner.new(events)
      progress = []
      factory_calls = 0
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        progress: ->(event) { progress << event },
        model_provider: provider,
        decoder: decoder,
        aligner_factory: lambda do |**|
          factory_calls += 1
          assert provider.session.closed, "MMS was constructed before Dense ASR was evicted"
          events << :aligner_opened
          aligner
        end
      )

      run = engine.transcribe(paths)

      assert run.ok?
      assert_equal 1, provider.open_count
      assert_equal 1, factory_calls
      assert aligner.closed
      assert_equal [0, 1, 2], aligner.calls.map(&:first)
      assert_equal ["file 0", "file 1", "file 2"], run.map(&:text)
      result_paths = run.map { |result| result.path.to_s }
      assert_equal paths.map { |path| File.realpath(path) }, result_paths
      asr_closed = events.index(:asr_closed)
      asr_indices = events.each_index.select { |index| event_kind(events[index]) == :asr }
      alignment_indices = events.each_index.select { |index| event_kind(events[index]) == :align }
      assert(asr_indices.all? { |index| index < asr_closed })
      assert(alignment_indices.all? { |index| index > asr_closed })
      decode_counts = decoder.calls.group_by(&:first).sort.map { |_index, calls| calls.length }
      assert_equal Array.new(3, 2), decode_counts
      decoder.calls.group_by(&:first).each_value do |calls|
        backends = calls.sort_by { |call| call.fetch(1) }.map { |call| call.fetch(2) }
        assert_equal %w[auto ffmpeg], backends
      end
      file_progress = progress.select { |event| event.stage == "files" }
      assert_equal [1, 2, 3], file_progress.map(&:current)
      assert(file_progress.all? { |event| event.total == 3 })
    ensure
      engine&.close
    end
  end

  def test_alignment_redecode_failure_is_per_file_and_later_files_continue
    with_audio_files(3) do |paths|
      events = []
      aligner = PhaseAligner.new(events)
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: PhaseProvider.new(events),
        decoder: PhaseDecoder.new(events, drift_on_alignment: 1),
        aligner_factory: ->(**) { aligner }
      )

      run = engine.transcribe(paths)

      assert_equal %w[completed failed completed], run.map(&:status)
      assert_match(/Decoded sample count changed/, run[1].error)
      assert_equal [0, 2], aligner.calls.map(&:first)
      result_paths = run.map { |result| result.path.to_s }
      assert_equal paths.map { |path| File.realpath(path) }, result_paths
    ensure
      engine&.close
    end
  end

  def test_empty_asr_results_publish_without_alignment_redecode_or_mms_load
    with_audio_files(2) do |paths|
      events = []
      decoder = PhaseDecoder.new(events, empty: true)
      provider = PhaseProvider.new(events, blank: true)
      factory_calls = 0
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: provider,
        decoder: decoder,
        aligner_factory: lambda do |**|
          factory_calls += 1
          raise "blank transcripts must not load MMS"
        end
      )

      run = engine.transcribe(paths)

      assert run.ok?
      assert_equal ["", ""], run.map(&:text)
      assert_equal 0, factory_calls
      refute provider.session.closed
      decode_counts = decoder.calls.group_by(&:first).sort.map { |_index, calls| calls.length }
      assert_equal [1, 1], decode_counts
    ensure
      engine&.close
    end
  end

  def test_mms_load_failure_is_attempted_once_and_applied_to_every_completed_asr_file
    with_audio_files(3) do |paths|
      events = []
      decoder = PhaseDecoder.new(events)
      aligner = PhaseAligner.new(
        events,
        load_failure: Cohere::Transcribe::Alignment::BackendUnavailable.new("MMS is unavailable")
      )
      factory_calls = 0
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: PhaseProvider.new(events),
        decoder: decoder,
        aligner_factory: lambda do |**|
          factory_calls += 1
          aligner
        end
      )

      run = engine.transcribe(paths)

      assert_equal %w[failed failed failed], run.map(&:status)
      assert(run.all? { |result| result.error.include?("MMS is unavailable") })
      assert_equal 1, factory_calls
      assert_equal 1, aligner.load_count
      assert_empty aligner.calls
      alignment_decodes = decoder.calls.count { |_index, count, _backend| count == 2 }
      assert_operator alignment_decodes, :<=, 1
    ensure
      engine&.close
    end
  end

  def test_failed_alignment_checkpoint_resumes_without_reopening_dense_and_publishes
    with_audio_files(1) do |paths, directory|
      output_dir = File.join(directory, "out")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: ["txt"], output_dir: output_dir, existing: "overwrite"
      )
      run_options = options.with(publication: publication)
      first_events = []
      first = Cohere::Transcribe::Runtime::Engine.new(
        run_options,
        model_provider: PhaseProvider.new(first_events),
        decoder: PhaseDecoder.new(first_events),
        aligner_factory: lambda do |**|
          PhaseAligner.new(first_events, failure: RuntimeError.new("aligner unavailable"))
        end
      )

      failed = first.transcribe(paths.first).single
      first.close
      assert_equal "failed", failed.status
      assert_match(/aligner unavailable/, failed.error)
      checkpoint = Pathname(output_dir).join(".audio-0#{Cohere::Transcribe::State::CHECKPOINT_SUFFIX}")
      assert checkpoint.file?
      refute Pathname(output_dir).join("audio-0.txt").exist?

      resumed_events = []
      resumed_provider = PhaseProvider.new(resumed_events, fail_if_opened: true)
      second = Cohere::Transcribe::Runtime::Engine.new(
        run_options,
        model_provider: resumed_provider,
        decoder: PhaseDecoder.new(resumed_events),
        aligner_factory: ->(**) { PhaseAligner.new(resumed_events) }
      )
      result = second.transcribe(paths.first).single

      assert_equal "completed", result.status
      assert result.provenance.resumed_from_asr_checkpoint
      assert_equal 0, resumed_provider.open_count
      assert_equal "file 0\n", Pathname(output_dir).join("audio-0.txt").read
      resumed_decodes = resumed_events.count { |event| event_kind(event) == :decode }
      assert_equal 1, resumed_decodes
    ensure
      first&.close
      second&.close
    end
  end

  private

  def options
    Cohere::Transcribe::TranscriptionOptions.new(
      device: "cpu",
      dtype: "fp32",
      audio_backend: "auto",
      audio_memory_gb: 1.0,
      preprocess_workers: 1,
      pipeline_preparation: false,
      vad: "none",
      max_dur: 1.0,
      alignment: "word",
      align_dtype: "fp32"
    )
  end

  def event_kind(event)
    event.is_a?(Array) ? event.first : event
  end

  def with_audio_files(count)
    Dir.mktmpdir("cohere-word-engine") do |directory|
      paths = Array.new(count) do |index|
        path = File.join(directory, "audio-#{index}.wav")
        File.binwrite(path, "stable source #{index}")
        path
      end
      yield paths, directory
    end
  end
end
