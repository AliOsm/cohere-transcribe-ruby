# frozen_string_literal: true

require "tmpdir"
require "timeout"
require "test_helper"

class Cohere::Transcribe::RuntimeEngineTest < Minitest::Test
  class FakeDecoder
    def initialize(samples = Array.new(32_000, 0.1), failure: nil)
      @samples = samples
      @failure = failure
    end

    def decode(_path, **)
      raise @failure if @failure

      Cohere::Transcribe::Audio::Decoded.new(
        samples: @samples.dup,
        sample_rate: 16_000,
        backend: "fake",
        fallback_reason: nil
      )
    end
  end

  class FakeSession
    attr_reader :calls

    def initialize
      @calls = []
      @closed = false
    end

    def transcribe(samples, language:, offset:, max_new_tokens:)
      @calls << [samples.length, language, offset, max_new_tokens]
      index = @calls.length
      Cohere::Transcribe::ASR::NativeResult.new(
        text: "segment #{index}",
        segments: [],
        words: [],
        generated_tokens: 2,
        generation_limit: max_new_tokens,
        generation_capacity: 1_000,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end

    def close
      @closed = true
    end

    def closed?
      @closed
    end
  end

  class MemorySession < FakeSession
    GIB = 1024**3

    attr_reader :device

    def initialize
      super
      @device = "cuda"
      @memory_calls = 0
    end

    def memory
      free_gib = [8.0 - (@memory_calls * 0.25), 6.0].max
      @memory_calls += 1
      [(free_gib * GIB).to_i, 12 * GIB]
    end
  end

  class FakeProvider
    attr_reader :open_count, :session

    def initialize
      @session = FakeSession.new
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
      @open_count += 1
      @session
    end
  end

  class GenerationSession
    attr_reader :limits

    def initialize(results)
      @results = results.dup
      @limits = []
    end

    def transcribe(_samples, language:, offset:, max_new_tokens:)
      @limits << max_new_tokens
      metadata = @results.shift or raise "missing fake generation result"
      Cohere::Transcribe::ASR::NativeResult.new(
        text: metadata.fetch(:text, "generated"),
        segments: [],
        words: [],
        generated_tokens: metadata.fetch(:generated_tokens),
        generation_limit: metadata.fetch(:generation_limit, max_new_tokens),
        generation_capacity: metadata.fetch(:generation_capacity, 1_000),
        stopped_by_max_tokens: metadata.fetch(:stopped_by_max_tokens, false),
        repetition_stopped: metadata.fetch(:repetition_stopped, false)
      )
    end

    def close; end
  end

  class GenerationProvider < FakeProvider
    def initialize(session)
      @session = session
      @open_count = 0
    end
  end

  class BatchGenerationSession
    attr_reader :calls

    def initialize
      @calls = []
    end

    def transcribe_batch(sample_batches, language:, offsets:, max_new_tokens:)
      @calls << {
        rows: sample_batches.length,
        samples: sample_batches.map(&:length),
        language: language,
        offsets: offsets,
        max_new_tokens: max_new_tokens
      }
      offsets.map do |offset|
        Cohere::Transcribe::ASR::NativeResult.new(
          text: "at #{offset.to_i}",
          segments: [],
          words: [],
          generated_tokens: 3,
          generation_limit: max_new_tokens,
          generation_capacity: 1_000,
          stopped_by_max_tokens: false,
          repetition_stopped: false
        )
      end
    end

    def close; end
  end

  class LifecycleBatchGenerationSession < BatchGenerationSession
    attr_reader :events

    def initialize(events)
      super()
      @events = events
      @closed = false
    end

    def close
      @events << :asr_closed
      @closed = true
    end

    def closed?
      @closed
    end
  end

  class FakeAligner
    attr_reader :calls, :load_seconds, :emissions_seconds, :viterbi_seconds

    def initialize(words, fallback_count:, events:)
      @words = words
      @fallback_count = fallback_count
      @events = events
      @calls = []
      @load_seconds = 0.0
      @emissions_seconds = 0.0
      @viterbi_seconds = 0.0
      @closed = false
    end

    def align(audio, segment_times, segment_texts, language:)
      @events << :aligned
      @calls << {
        audio: audio,
        segment_times: segment_times,
        segment_texts: segment_texts,
        language: language
      }
      @load_seconds += 0.125
      @emissions_seconds += 0.25
      @viterbi_seconds += 0.375
      [@words, @fallback_count]
    end

    def close
      @events << :aligner_closed
      @closed = true
    end

    def closed?
      @closed
    end
  end

  class ScriptedBatchSession < BatchGenerationSession
    def initialize(&script)
      super()
      @script = script
    end

    def transcribe_batch(sample_batches, language:, offsets:, max_new_tokens:)
      @calls << {
        rows: sample_batches.length,
        samples: sample_batches.map(&:length),
        language: language,
        offsets: offsets,
        max_new_tokens: max_new_tokens
      }
      @script.call(@calls.length, offsets, max_new_tokens)
    end
  end

  class ResilienceSession < BatchGenerationSession
    attr_reader :closed, :single_calls

    def initialize(mode)
      super()
      @mode = mode
      @failed_batch = false
      @isolating_data_error = false
      @single_calls = []
      @closed = false
    end

    def transcribe_batch(sample_batches, language:, offsets:, max_new_tokens:)
      @calls << {
        rows: sample_batches.length,
        samples: sample_batches.map(&:length),
        language: language,
        offsets: offsets,
        max_new_tokens: max_new_tokens
      }
      unless @failed_batch
        @failed_batch = true
        if @mode == :fatal
          raise Cohere::Transcribe::ASR::ExecutionError.new(
            "CUDA error: illegal memory access",
            failure_kind: :fatal
          )
        end
        if @mode == :data_local
          @isolating_data_error = true
          raise "malformed sample payload"
        end
      end

      offsets.map { |offset| native_result(offset, max_new_tokens) }
    end

    def transcribe(_samples, language:, offset:, max_new_tokens:)
      @single_calls << { language: language, offset: offset, max_new_tokens: max_new_tokens }
      if @isolating_data_error && offset.zero?
        @isolating_data_error = false
        raise "malformed sample payload"
      end

      native_result(offset, max_new_tokens)
    end

    def close
      @closed = true
    end

    private

    def native_result(offset, max_new_tokens)
      Cohere::Transcribe::ASR::NativeResult.new(
        text: "at #{offset.to_i}",
        segments: [],
        words: [],
        generated_tokens: 3,
        generation_limit: max_new_tokens,
        generation_capacity: 1_000,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end
  end

  class CoordinatedDecoder
    attr_reader :calls, :completion_order, :maximum_active

    def initialize(group_size:, next_group_started: nil, failure_index: nil, block_next_group: false)
      @group_size = group_size
      @next_group_started = next_group_started
      @failure_index = failure_index
      @block_next_group = block_next_group
      @calls = []
      @completion_order = []
      @maximum_active = 0
      @active = 0
      @group_counts = Hash.new(0)
      @mutex = Mutex.new
      @condition = ConditionVariable.new
    end

    def decode(path, max_decoded_bytes:, **)
      index = File.basename(path.to_s)[/\d+/].to_i
      group = index / @group_size
      @mutex.synchronize do
        @calls << [index, max_decoded_bytes]
        @active += 1
        @maximum_active = [@maximum_active, @active].max
        @group_counts[group] += 1
        @condition.broadcast
        @condition.wait(@mutex) until @group_counts[group] >= @group_size
      end
      @next_group_started << index if index >= @group_size && @next_group_started
      sleep if index >= @group_size && @block_next_group
      sleep 0.02 if index.even?
      raise "bad media #{index}" if index == @failure_index

      Cohere::Transcribe::Audio::Decoded.new(
        samples: Array.new(16_000, 0.1),
        sample_rate: 16_000,
        backend: "coordinated",
        fallback_reason: nil
      )
    ensure
      @mutex.synchronize do
        @completion_order << index if defined?(index)
        @active -= 1 if defined?(index)
        @condition.broadcast
      end
    end
  end

  class OverlapSession < FakeSession
    def initialize(next_group_started)
      super()
      @next_group_started = next_group_started
      @checked_overlap = false
    end

    def transcribe(...)
      unless @checked_overlap
        Timeout.timeout(2) { @next_group_started.pop }
        @checked_overlap = true
      end
      super
    end
  end

  class BlockingDecoder
    def initialize(entered, release)
      @entered = entered
      @release = release
    end

    def decode(_path, **)
      @entered << true
      @release.pop
      Cohere::Transcribe::Audio::Decoded.new(
        samples: Array.new(16_000, 0.1),
        sample_rate: 16_000,
        backend: "blocking",
        fallback_reason: nil
      )
    end
  end

  class ThreadConfinedSilero
    attr_reader :calls, :keyword_calls

    def initialize
      @calls = []
      @keyword_calls = []
      @active = false
      @mutex = Mutex.new
    end

    def provider
      "fake-onnx"
    end

    def speech_timestamps(samples, **keywords)
      @mutex.synchronize do
        raise "Silero session used concurrently" if @active

        @active = true
        @calls << Thread.current.object_id
        @keyword_calls << keywords
      end
      sleep 0.005
      [{ start: 0, end: samples.length }]
    ensure
      @mutex.synchronize { @active = false }
    end
  end

  class TelemetrySilero
    attr_reader :block_frames, :intra_op_threads, :last_execution

    def initialize(block_frames:, threads:)
      @block_frames = block_frames
      @intra_op_threads = threads
      @last_execution = nil
    end

    def provider
      "CPUExecutionProvider"
    end

    def provider_options
      { "CPUExecutionProvider" => {} }
    end

    def speech_timestamps(samples, **)
      frames = (samples.length + 511) / 512
      calls = (frames + block_frames - 1) / block_frames
      @last_execution = Cohere::Transcribe::VAD::Silero::Execution.new(
        model_calls: calls,
        valid_frames: frames,
        padded_frames: frames,
        max_files_per_call: frames.positive? ? 1 : 0,
        effective_block_frames: block_frames
      )
      [{ start: 0, end: samples.length }]
    end
  end

  class SessionProvider < FakeProvider
    def initialize(session)
      @session = session
      @open_count = 0
    end
  end

  def test_complete_pipeline_is_ordered_reusable_and_timed
    with_audio_files(1) do |paths|
      provider = FakeProvider.new
      events = []
      options = base_options
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        progress: ->(event) { events << event },
        model_provider: provider,
        decoder: FakeDecoder.new
      )

      run = engine.transcribe(paths.first)

      assert run.ok?
      assert_equal 1, run.length
      assert_equal "completed", run.single.status
      assert_equal "segment 1\nsegment 2", run.single.text
      assert_equal([[0.0, 1.0], [1.0, 2.0]], run.single.segments.map { |item| [item.start, item.end] })
      assert_equal %w[segment 1 segment 2], run.single.words.map(&:text)
      assert_equal [0, 0, 1, 1], run.single.words.map(&:segment_index)
      assert_equal 2.0, run.statistics.successful_audio_seconds
      assert_equal 2, run.statistics.asr_batches
      assert_equal "fake", run.single.provenance.decode_backend
      assert_equal "dense", run.single.provenance.model_format
      assert_equal [0, 1, 2], events.select { |event| event.stage == "ASR" }.map(&:current)
      assert_equal 1, provider.open_count

      second = engine.transcribe(paths.first)
      assert second.ok?
      assert_equal 1, provider.open_count
      engine.close
      assert provider.session.closed?
      assert_raises(Cohere::Transcribe::TranscriberClosedError) { engine.transcribe(paths.first) }
    end
  end

  def test_public_results_filter_blank_segments_after_checkpoint_publication_and_profile
    with_audio_files(1) do |paths, directory|
      output_dir = Pathname(directory).join("out")
      profile = Pathname(directory).join("profile.json")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: ["json"],
        output_dir: output_dir,
        existing: "overwrite",
        profile_json: profile
      )
      session = GenerationSession.new(
        [
          { text: "   ", generated_tokens: 1 },
          { text: "spoken", generated_tokens: 2 }
        ]
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new
      )

      run = engine.transcribe(paths.first)

      assert_equal [1], run.single.segments.map(&:index)
      assert_equal "spoken", run.single.text
      output = JSON.parse(output_dir.join("audio-0.json").read)
      assert_equal([1], output.fetch("segments").map { |segment| segment.fetch("segment_index") })
      telemetry = JSON.parse(profile.read)
      assert_equal 2, telemetry.fetch("files").first.fetch("segment_count")
      asr_telemetry = telemetry.fetch("asr")
      assert_equal 1, asr_telemetry.fetch("effective_batch_min")
      assert_equal 1, asr_telemetry.fetch("effective_batch_max")
      assert_equal 1, asr_telemetry.fetch("final_batch_size")
      assert_equal 1, asr_telemetry.fetch("final_batch_cap")
      assert_equal 2, asr_telemetry.fetch("batch_history").length
      assert_equal 1, asr_telemetry.fetch("checkpoint_written_files")
      asr_telemetry.fetch("batch_history").each do |batch|
        assert_equal 1, batch.fetch("processor_rows")
        assert_equal 1, batch.fetch("generated_tokens_by_row").length
        assert_operator batch.fetch("generation_call_wall_seconds"), :>, 0.0
        assert_equal 1.0, batch.fetch("padded_audio_seconds")
      end
      timings = telemetry.fetch("timings")
      assert_equal 0.0, timings.fetch("vad_model_load_seconds")
      assert_equal 0.0, timings.fetch("vad_inference_seconds")
      assert_equal 0.0, timings.fetch("vad_postprocess_seconds")
      assert_equal 0.0, timings.fetch("preparation_wait_seconds")
      assert_operator timings.fetch("asr_generation_call_wall_seconds"), :>, 0.0
      assert_equal 0.0, timings.fetch("post_asr_seconds")
      assert_operator timings.fetch("checkpoint_seconds"), :>, 0.0
      assert_operator timings.fetch("progressive_output_seconds"), :>, 0.0
      assert_equal 0.0, timings.fetch("asr_discarded_feature_seconds")
      assert_equal 0.0, timings.fetch("asr_feature_wait_seconds")
      checkpoint, reason = Cohere::Transcribe::State.decode_state(
        output_dir.join(".audio-0.cohere-transcribe.asr.json")
      )
      assert_nil reason
      assert_equal ["", "spoken"], checkpoint.dig("checkpoint", "segment_texts")
    ensure
      engine&.close
    end
  end

  def test_profile_uses_native_cuda_memory_snapshots_without_allocator_estimates
    with_audio_files(1) do |paths, directory|
      profile = Pathname(directory).join("profile.json")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: ["json"],
        output_dir: Pathname(directory).join("out"),
        existing: "overwrite",
        profile_json: profile
      )
      session = MemorySession.new
      options = base_options.with(device: "cuda", dtype: "bf16", publication: publication)
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: SessionProvider.new(session),
        decoder: FakeDecoder.new
      )

      precision = Cohere::Transcribe::Runtime::Precision
      original_resolve = precision.method(:resolve)
      precision.define_singleton_method(:resolve) { |value| value }
      begin
        assert engine.transcribe(paths.first).ok?
      ensure
        precision.define_singleton_method(:resolve) do |*args, **keywords|
          original_resolve.call(*args, **keywords)
        end
      end

      telemetry = JSON.parse(profile.read)
      memory = telemetry.fetch("cuda_memory")
      assert_equal 12.0, memory.fetch("total_gib")
      assert_equal 8.0, memory.fetch("free_start_gib")
      assert_operator memory.fetch("free_end_gib"), :<, memory.fetch("free_start_gib")
      assert_nil memory.fetch("peak_allocated_gib")
      assert_nil memory.fetch("peak_reserved_gib")
      assert_equal 12.0, telemetry.dig("environment", "cuda", "total_memory_gib")
      assert_equal memory.fetch("free_end_gib"),
                   telemetry.dig("environment", "cuda", "free_memory_at_profile_gib")
    ensure
      engine&.close
    end
  end

  def test_word_alignment_uses_full_audio_and_evicts_asr_before_loading_mms
    with_audio_files(1) do |paths|
      events = []
      native = LifecycleBatchGenerationSession.new(events)
      provider = SessionProvider.new(native)
      aligned_words = [
        Cohere::Transcribe::TranscriptionWord.new(
          start: 0.1,
          end: 0.8,
          text: "MMS",
          segment_index: 0,
          segment_word_index: 0,
          timing_source: "ctc"
        )
      ].freeze
      aligner = FakeAligner.new(aligned_words, fallback_count: 1, events: events)
      factory_calls = []
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(alignment: "word", align_batch_size: 7, align_dtype: "fp32"),
        model_provider: provider,
        decoder: FakeDecoder.new,
        aligner_factory: lambda do |**keywords|
          factory_calls << keywords.merge(asr_closed: native.closed?)
          events << :aligner_constructed
          aligner
        end
      )

      run = engine.transcribe(paths.first)

      assert run.ok?
      assert_equal 1, native.calls.length
      assert_equal 2, native.calls.first.fetch(:rows)
      assert_equal %i[asr_closed aligner_constructed aligned aligner_closed], events
      assert_equal(
        [{ dtype: "fp32", device: "cpu", batch_size: 7, asr_closed: true }],
        factory_calls
      )
      call = aligner.calls.fetch(0)
      assert_equal 32_000, call.fetch(:audio).length
      assert_equal [[0.0, 1.0], [1.0, 2.0]], call.fetch(:segment_times)
      assert_equal ["at 0", "at 1"], call.fetch(:segment_texts)
      assert_equal "ar", call.fetch(:language)
      assert_equal aligned_words, run.single.words
      assert_equal 1, run.single.provenance.fallback_alignment_segments
      assert_in_delta 0.125, run.statistics.aligner_load_seconds
      assert_in_delta 0.25, run.statistics.emissions_seconds
      assert_in_delta 0.375, run.statistics.viterbi_seconds

      engine.close
      assert aligner.closed?
      assert_equal :aligner_closed, events.last
    end
  end

  def test_file_decode_failure_is_returned_and_raise_on_error_preserves_run
    with_audio_files(1) do |paths|
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options,
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new(failure: RuntimeError.new("bad media"))
      )

      run = engine.transcribe(paths.first)
      refute run.ok?
      assert_equal "failed", run.single.status
      assert_match(/bad media/, run.single.error)

      error = assert_raises(Cohere::Transcribe::BatchTranscriptionError) do
        engine.transcribe(paths.first, raise_on_error: true)
      end
      assert_equal "failed", error.run.single.status
    ensure
      engine&.close
    end
  end

  def test_progress_exceptions_are_typed_and_abort_the_run
    with_audio_files(1) do |paths|
      original = RuntimeError.new("stop reporting")
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options,
        progress: ->(_event) { raise original },
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )

      error = assert_raises(Cohere::Transcribe::ProgressCallbackError) do
        engine.transcribe(paths.first)
      end
      assert_same original, error.original
      assert_same original, error.cause
    ensure
      engine&.close
    end
  end

  def test_transactional_publication_returns_all_requested_outputs
    with_audio_files(1) do |paths, directory|
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt srt vtt json],
        output_dir: File.join(directory, "out"),
        existing: "error"
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )

      result = engine.transcribe(paths.first).single

      assert_equal 4, result.outputs.length
      assert result.outputs.all?(&:file?)
      assert_equal "segment 1\nsegment 2\n", result.outputs.find { |path| path.extname == ".txt" }.read
      assert_match(/WEBVTT/, result.outputs.find { |path| path.extname == ".vtt" }.read)
      json = JSON.parse(result.outputs.find { |path| path.extname == ".json" }.read)
      assert_equal 8, json.fetch("schema_version")
      assert_equal Cohere::Transcribe::VERSION, json.dig("implementation", "package_version")
      assert result.provenance.published
      assert File.file?(File.join(directory, "out", ".audio-0.cohere-transcribe.manifest.json"))
    ensure
      engine&.close
    end
  end

  def test_model_resolution_cannot_redirect_a_preplanned_output_or_profile_parent
    with_audio_files(1) do |paths, directory|
      root = Pathname(directory)
      output_root = root.join("out")
      outside = root.join("outside")
      parked = root.join("parked")
      outside.mkdir
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json],
        output_dir: output_root,
        existing: "overwrite",
        profile_json: output_root.join("profile.json")
      )
      provider = FakeProvider.new
      provider.define_singleton_method(:resolve) do |requested|
        output_root.rename(parked)
        output_root.make_symlink(outside)
        super(requested)
      end
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: provider,
        decoder: FakeDecoder.new(failure: "redirected output decoded audio")
      )

      error = assert_raises(Cohere::Transcribe::TranscriptionRuntimeError) do
        engine.transcribe(paths.first)
      end

      assert_match(/Publication parent changed/, error.message)
      assert_equal 0, provider.open_count
      assert_empty outside.children
      assert_empty parked.children
    ensure
      engine&.close
    end
  end

  def test_skip_requires_a_verified_manifest_and_rebuilds_tampered_output
    with_audio_files(1) do |paths, directory|
      output_dir = File.join(directory, "out")
      overwrite = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      first_provider = FakeProvider.new
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: overwrite),
        model_provider: first_provider,
        decoder: FakeDecoder.new
      )
      first_result = first.transcribe(paths.first).single
      first.close
      assert_equal "completed", first_result.status
      assert_equal 1, first_provider.open_count

      skip = overwrite.with(existing: "skip")
      skipped_provider = FakeProvider.new
      skip_events = []
      second = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: skip),
        progress: ->(event) { skip_events << event },
        model_provider: skipped_provider,
        decoder: FakeDecoder.new(failure: "verified skip must not decode")
      )
      skipped = second.transcribe(paths.first).single
      second.close
      assert_equal "skipped", skipped.status
      assert_equal 0, skipped_provider.open_count
      assert_empty skip_events

      File.write(File.join(output_dir, "audio-0.txt"), "tampered\n")
      rebuild_provider = FakeProvider.new
      third = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: skip),
        model_provider: rebuild_provider,
        decoder: FakeDecoder.new
      )
      rebuilt = third.transcribe(paths.first).single
      third.close
      assert_equal "completed", rebuilt.status
      assert_equal 0, rebuild_provider.open_count
      assert rebuilt.provenance.resumed_from_asr_checkpoint
      assert_equal "segment 1\nsegment 2\n", File.read(File.join(output_dir, "audio-0.txt"))
    end
  end

  def test_verified_skip_does_not_need_to_create_a_lock_in_a_read_only_output_directory
    skip "read-only mode bits are unavailable" if Gem.win_platform?

    with_audio_files(1) do |paths, directory|
      output_dir = Pathname(directory).join("out")
      overwrite = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: overwrite),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )
      assert_equal "completed", first.transcribe(paths.first).single.status
      first.close
      FileUtils.rm_rf(output_dir.join(".cohere-transcribe-locks"))
      output_dir.chmod(0o555)

      provider = FakeProvider.new
      skipped_engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: overwrite.with(existing: "skip")),
        model_provider: provider,
        decoder: FakeDecoder.new(failure: "verified skip must not decode")
      )
      skipped = skipped_engine.transcribe(paths.first).single
      skipped_engine.close

      assert_equal "skipped", skipped.status
      assert_equal 0, provider.open_count
      refute output_dir.join(".cohere-transcribe-locks").exist?
    ensure
      output_dir&.chmod(0o755) if output_dir&.exist?
      first&.close
      skipped_engine&.close
    end
  end

  def test_skip_plan_is_revalidated_when_an_output_changes_before_preflight
    with_audio_files(1) do |paths, directory|
      output_dir = Pathname(directory).join("out")
      overwrite = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: overwrite),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )
      first_result = first.transcribe(paths.first).single
      first.close
      changed_output = first_result.outputs.first

      publication = Cohere::Transcribe::Output::Publication
      original_plan = publication.method(:plan)
      plan_calls = 0
      publication.define_singleton_method(:plan) do |*arguments|
        plans = original_plan.call(*arguments)
        plan_calls += 1
        changed_output.binwrite("changed after planning") if plan_calls == 2
        plans
      end

      skip_options = overwrite.with(existing: "skip")
      second = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: skip_options),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )
      result = second.transcribe(paths.first).single
      second.close

      assert_equal "completed", result.status
      refute_equal "changed after planning", changed_output.binread
    ensure
      publication&.define_singleton_method(:plan, original_plan) if original_plan
      first&.close
      second&.close
    end
  end

  def test_render_only_change_reuses_checkpoint_but_asr_change_does_not
    with_audio_files(1) do |paths, directory|
      output_dir = File.join(directory, "out")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      first_provider = FakeProvider.new
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: first_provider,
        decoder: FakeDecoder.new
      )
      initial = first.transcribe(paths.first).single
      first.close
      assert_equal "completed", initial.status
      assert_equal 1, first_provider.open_count

      render_provider = FakeProvider.new
      render_events = []
      render_only = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(max_chars: 37, publication: publication),
        progress: ->(event) { render_events << event },
        model_provider: render_provider,
        decoder: FakeDecoder.new(failure: "render-only checkpoint resume must not decode")
      )
      rendered = render_only.transcribe(paths.first).single
      render_only.close
      assert_equal "completed", rendered.status
      assert rendered.provenance.resumed_from_asr_checkpoint
      assert_equal 0, render_provider.open_count
      assert_equal "segment 1\nsegment 2\n", File.read(File.join(output_dir, "audio-0.txt"))
      assert_equal ["files"], render_events.map(&:stage)

      asr_provider = FakeProvider.new
      asr_change = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(language: "en", publication: publication),
        model_provider: asr_provider,
        decoder: FakeDecoder.new
      )
      retranscribed = asr_change.transcribe(paths.first).single
      asr_change.close
      assert_equal "completed", retranscribed.status
      refute retranscribed.provenance.resumed_from_asr_checkpoint
      assert_equal 1, asr_provider.open_count
    end
  end

  def test_semantically_corrupt_checkpoint_is_ignored_without_stale_metadata
    with_audio_files(1) do |paths, directory|
      source = Pathname(paths.first).realpath
      output_dir = Pathname(directory).join("out")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new
      )
      assert_equal "completed", first.transcribe(source).single.status
      first.close

      checkpoint_path = output_dir.join(".audio-0.cohere-transcribe.asr.json")
      payload, reason = Cohere::Transcribe::State.decode_state(checkpoint_path)
      assert_nil reason
      payload.fetch("checkpoint")["token_limit_segments"] = [99]
      Cohere::Transcribe::State.write_state_atomic(
        checkpoint_path,
        payload,
        source_snapshot: Cohere::Transcribe::State::SourceSnapshot.capture(source)
      )
      output_dir.join("audio-0.txt").binwrite("tampered\n")

      provider = FakeProvider.new
      skip_publication = publication.with(existing: "skip")
      rebuilt_engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: skip_publication),
        model_provider: provider,
        decoder: FakeDecoder.new
      )
      rebuilt = rebuilt_engine.transcribe(source).single
      rebuilt_engine.close
      assert_equal "completed", rebuilt.status
      refute rebuilt.provenance.resumed_from_asr_checkpoint
      assert_equal 1, provider.open_count
      assert_equal [], rebuilt.provenance.token_limit_segments
      assert_equal "segment 1\nsegment 2\n", output_dir.join("audio-0.txt").read
    end
  end

  def test_alignment_failure_retains_checkpoint_for_render_only_retry
    with_audio_files(1) do |paths, directory|
      output_dir = Pathname(directory).join("out")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json], output_dir: output_dir, existing: "overwrite"
      )
      provider = FakeProvider.new
      failing = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(alignment: "word", publication: publication),
        model_provider: provider,
        decoder: FakeDecoder.new,
        aligner_factory: ->(**) { raise "simulated alignment failure" }
      )
      failed = failing.transcribe(paths.first).single
      failing.close
      assert_equal "failed", failed.status
      assert_match(/alignment failure/, failed.error)
      assert_equal 1, provider.open_count
      checkpoint = output_dir.join(".audio-0.cohere-transcribe.asr.json")
      assert checkpoint.file?
      refute output_dir.join(".audio-0.cohere-transcribe.manifest.json").exist?

      retry_provider = FakeProvider.new
      retry_engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(alignment: "segment", publication: publication),
        model_provider: retry_provider,
        decoder: FakeDecoder.new
      )
      retried = retry_engine.transcribe(paths.first).single
      retry_engine.close
      assert_equal "completed", retried.status
      assert retried.provenance.resumed_from_asr_checkpoint
      assert_equal 0, retry_provider.open_count
      assert output_dir.join(".audio-0.cohere-transcribe.manifest.json").file?
    end
  end

  def test_retries_only_truncated_segments_and_records_decoder_metadata
    with_audio_files(1) do |paths|
      session = GenerationSession.new([
                                        { generated_tokens: 100, stopped_by_max_tokens: true },
                                        { generated_tokens: 228, stopped_by_max_tokens: true },
                                        { text: "complete", generated_tokens: 301 }
                                      ])
      options = base_options.with(max_new_tokens: 100, max_retry_tokens: 500)
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )

      run = engine.transcribe(paths.first)
      result = run.single

      assert_equal [100, 228, 500], session.limits
      assert_equal "complete", result.text
      assert_equal [0], result.provenance.truncation_retried_segments
      assert_empty result.provenance.token_limit_segments
      assert_equal [[0, 301]], result.provenance.generated_tokens_by_segment
      assert_equal 3, run.statistics.asr_batches
      assert_equal 629, run.statistics.generated_tokens
      assert_equal 2, run.statistics.truncation_retries
    ensure
      engine&.close
    end
  end

  def test_warn_policy_records_an_unretried_token_limit_and_repetition_stops
    with_audio_files(1) do |paths|
      truncated = GenerationSession.new([
                                          { text: "partial", generated_tokens: 100, stopped_by_max_tokens: true }
                                        ])
      warn_engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(max_new_tokens: 100, max_retry_tokens: 500, truncation_policy: "warn"),
        model_provider: GenerationProvider.new(truncated),
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )
      warned = warn_engine.transcribe(paths.first).single
      warn_engine.close

      assert_equal [100], truncated.limits
      assert_empty warned.provenance.truncation_retried_segments
      assert_equal [0], warned.provenance.token_limit_segments

      repeated = GenerationSession.new([
                                         { text: "loop", generated_tokens: 96, repetition_stopped: true }
                                       ])
      repeat_engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options,
        model_provider: GenerationProvider.new(repeated),
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )
      repetition = repeat_engine.transcribe(paths.first).single

      assert_equal [0], repetition.provenance.repetition_stopped_segments
      assert_empty repetition.provenance.token_limit_segments
    ensure
      warn_engine&.close
      repeat_engine&.close
    end
  end

  def test_token_retry_is_capped_by_native_decoder_positions
    with_audio_files(1) do |paths|
      session = GenerationSession.new([
                                        {
                                          generated_tokens: 70,
                                          generation_limit: 70,
                                          generation_capacity: 90,
                                          stopped_by_max_tokens: true
                                        },
                                        { text: "complete", generated_tokens: 81, generation_capacity: 90 }
                                      ])
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(max_new_tokens: 70, max_retry_tokens: 200),
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )

      result = engine.transcribe(paths.first).single

      assert_equal [70, 90], session.limits
      assert_equal [0], result.provenance.truncation_retried_segments
      assert_empty result.provenance.token_limit_segments
    ensure
      engine&.close
    end
  end

  def test_native_batches_honor_row_and_padded_audio_caps_and_restore_segment_order
    with_audio_files(1) do |paths|
      session = BatchGenerationSession.new
      options = base_options.with(
        batch_size: 4,
        batch_audio_seconds: 2.1,
        max_new_tokens: 200,
        max_retry_tokens: 200
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(64_000, 0.1))
      )

      run = engine.transcribe(paths.first)

      assert_equal([2, 2], session.calls.map { |call| call.fetch(:rows) })
      assert(session.calls.all? { |call| call.fetch(:max_new_tokens) == 200 })
      assert_equal "at 0\nat 1\nat 2\nat 3", run.single.text
      assert_equal 2, run.statistics.asr_batches
      assert_equal 4, run.statistics.asr_processor_rows
      assert_equal 12, run.statistics.generated_tokens
    ensure
      engine&.close
    end
  end

  def test_native_batch_controller_uses_the_capacity_reported_by_the_session
    with_audio_files(1) do |paths|
      session = BatchGenerationSession.new
      session.define_singleton_method(:batch_capacity) { 12 }
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(batch_size: 12, batch_audio_seconds: 20.0),
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(192_000, 0.1))
      )

      run = engine.transcribe(paths.first)

      assert run.ok?
      assert_equal([12], session.calls.map { |call| call.fetch(:rows) })
      assert_equal 12, run.statistics.asr_processor_rows
    ensure
      engine&.close
    end
  end

  def test_adaptive_native_batch_does_not_guess_growth_without_backend_memory
    with_audio_files(1) do |paths|
      session = BatchGenerationSession.new
      options = base_options.with(
        batch_size: 2,
        batch_max_size: 4,
        batch_audio_seconds: 20.0,
        adaptive_batch: true
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(112_000, 0.1))
      )

      engine.transcribe(paths.first)

      # The Python controller grows only from measured accelerator memory;
      # CPU/no-telemetry sessions retain the configured starting size.
      assert_equal([2, 2, 2], session.calls.map { |call| call.fetch(:rows) })
    ensure
      engine&.close
    end
  end

  def test_batch_retry_regenerates_only_the_truncated_lanes
    with_audio_files(1) do |paths|
      session = ScriptedBatchSession.new do |call, offsets, limit|
        offsets.map do |offset|
          truncated = call == 1 && offset != 1.0
          Cohere::Transcribe::ASR::NativeResult.new(
            text: truncated ? "partial" : "done #{offset.to_i}",
            segments: [],
            words: [],
            generated_tokens: truncated ? limit : 7,
            generation_limit: limit,
            generation_capacity: 1_000,
            stopped_by_max_tokens: truncated,
            repetition_stopped: false
          )
        end
      end
      options = base_options.with(
        batch_size: 3,
        batch_audio_seconds: 10.0,
        max_new_tokens: 100,
        max_retry_tokens: 500
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        options,
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(48_000, 0.1))
      )

      run = engine.transcribe(paths.first)

      assert_equal([[0.0, 1.0, 2.0], [0.0, 2.0]], session.calls.map { |call| call.fetch(:offsets) })
      assert_equal([100, 228], session.calls.map { |call| call.fetch(:max_new_tokens) })
      assert_equal [0, 2], run.single.provenance.truncation_retried_segments
      assert_empty run.single.provenance.token_limit_segments
      assert_equal 2, run.statistics.truncation_retries
      assert_equal 2, run.statistics.asr_batches
    ensure
      engine&.close
    end
  end

  def test_failed_native_batches_split_and_learn_a_smaller_cap
    with_audio_files(1) do |paths|
      session = ScriptedBatchSession.new do |_call, offsets, limit|
        if offsets.length > 2
          raise Cohere::Transcribe::ASR::ExecutionError.new(
            "simulated native out of memory",
            failure_kind: :oom
          )
        end

        offsets.map do |offset|
          Cohere::Transcribe::ASR::NativeResult.new(
            text: "done #{offset.to_i}",
            segments: [],
            words: [],
            generated_tokens: 4,
            generation_limit: limit,
            generation_capacity: 1_000,
            stopped_by_max_tokens: false,
            repetition_stopped: false
          )
        end
      end
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(batch_size: 4, batch_audio_seconds: 20.0),
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(96_000, 0.1))
      )

      run = engine.transcribe(paths.first)

      assert_equal([4, 2, 2, 2], session.calls.map { |call| call.fetch(:rows) })
      assert run.ok?
      assert_equal 3, run.statistics.asr_batches
      assert_equal 6, run.statistics.asr_processor_rows
      assert_equal 1, run.statistics.oom_retries

      second = engine.transcribe(paths.first)
      assert second.ok?
      assert_equal([2, 2, 2], session.calls.drop(4).map { |call| call.fetch(:rows) })
      assert_equal 0, second.statistics.oom_retries
    ensure
      engine&.close
    end
  end

  def test_fatal_native_failure_opens_circuit_without_bisection_and_evicts_poisoned_session
    with_audio_files(2) do |paths|
      session = ResilienceSession.new(:fatal)
      provider = GenerationProvider.new(session)
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(batch_size: 2, batch_audio_seconds: 20.0),
        model_provider: provider,
        decoder: FakeDecoder.new(Array.new(32_000, 0.1))
      )

      run = engine.transcribe(paths)

      refute run.ok?
      assert_equal %w[failed failed], run.map(&:status)
      assert_equal 1, session.calls.length
      assert_match(/illegal memory access/, run[0].error)
      assert_match(/circuit breaker is open/, run[1].error)
      assert_equal 0, run.statistics.asr_batches
      assert_equal 0, run.statistics.oom_retries
      assert session.closed
    ensure
      engine&.close
    end
  end

  def test_data_local_native_failure_does_not_poison_later_files
    with_audio_files(2) do |paths|
      session = ResilienceSession.new(:data_local)
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(batch_size: 2, batch_audio_seconds: 20.0),
        model_provider: GenerationProvider.new(session),
        decoder: FakeDecoder.new(Array.new(32_000, 0.1))
      )

      run = engine.transcribe(paths)

      assert_equal %w[failed completed], run.map(&:status)
      assert_match(/malformed sample payload/, run[0].error)
      assert_equal "at 0\nat 1", run[1].text
      assert_equal([2, 2], session.calls.map { |call| call.fetch(:rows) })
      assert_equal([0.0, 1.0], session.single_calls.map { |call| call.fetch(:offset) })
      assert_equal 2, run.statistics.asr_batches
      assert_equal 0, run.statistics.oom_retries
      refute session.closed
    ensure
      engine&.close
    end
  end

  def test_second_engine_evicts_the_first_engines_retained_native_session
    with_audio_files(2) do |paths|
      first_provider = FakeProvider.new
      second_provider = FakeProvider.new
      first = Cohere::Transcribe::Runtime::Engine.new(
        base_options,
        model_provider: first_provider,
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )
      second = Cohere::Transcribe::Runtime::Engine.new(
        base_options,
        model_provider: second_provider,
        decoder: FakeDecoder.new(Array.new(16_000, 0.1))
      )

      assert first.transcribe(paths.first).ok?
      refute first_provider.session.closed?
      assert second.transcribe(paths.last).ok?

      assert first_provider.session.closed?
      refute second_provider.session.closed?
      assert_equal 1, first_provider.open_count
      assert_equal 1, second_provider.open_count
    ensure
      second&.close
      first&.close
    end
  end

  def test_checkpoint_only_word_alignment_evicts_another_engines_asr_owner
    with_audio_files(1) do |paths, directory|
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: %w[txt json],
        output_dir: File.join(directory, "out"),
        existing: "overwrite"
      )
      owner_provider = FakeProvider.new
      owner = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(publication: publication),
        model_provider: owner_provider,
        decoder: FakeDecoder.new
      )
      assert owner.transcribe(paths.first).ok?
      refute owner_provider.session.closed?

      events = []
      aligner = FakeAligner.new([], fallback_count: 0, events: events)
      checkpoint_provider = FakeProvider.new
      checkpoint_only = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(alignment: "word", publication: publication),
        model_provider: checkpoint_provider,
        decoder: FakeDecoder.new,
        aligner_factory: lambda do |**|
          events << :aligner_constructed
          assert owner_provider.session.closed?
          aligner
        end
      )

      result = checkpoint_only.transcribe(paths.first).single

      assert_equal "completed", result.status
      assert result.provenance.resumed_from_asr_checkpoint
      assert_equal 0, checkpoint_provider.open_count
      assert_equal %i[aligner_constructed aligned aligner_closed], events
      assert owner_provider.session.closed?
    ensure
      checkpoint_only&.close
      owner&.close
    end
  end

  def test_preparation_overlaps_asr_but_results_failures_and_progress_remain_ordered
    with_audio_files(4) do |paths|
      workers = [2, Etc.nprocessors].min
      next_group_started = Queue.new
      decoder = CoordinatedDecoder.new(
        group_size: workers,
        next_group_started: next_group_started,
        failure_index: 1
      )
      session = OverlapSession.new(next_group_started)
      events = []
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          audio_memory_gb: 1.0,
          preprocess_workers: 2,
          pipeline_preparation: true
        ),
        progress: ->(event) { events << event },
        model_provider: SessionProvider.new(session),
        decoder: decoder
      )

      run = engine.transcribe(paths)

      assert_equal %w[completed failed completed completed], run.map(&:status)
      assert_match(/bad media 1/, run[1].error)
      actual_paths = run.map { |result| result.path.to_s }
      assert_equal paths.map { |path| File.realpath(path) }, actual_paths
      expected_first_completion = workers == 1 ? [0, 1] : [1, 0]
      assert_equal expected_first_completion, decoder.completion_order.first(2)
      assert_equal workers, decoder.maximum_active
      expected_limit = (512 * (1024**2)) / workers
      assert_equal Array.new(4, expected_limit), decoder.calls.sort.map(&:last)
      assert_equal [1, 2, 3, 4], events.select { |event| event.stage == "files" }.map(&:current)
      assert_equal 3, session.calls.length
    ensure
      engine&.close
    end
  end

  def test_disabling_pipeline_decodes_sequentially_with_the_full_pcm_budget
    with_audio_files(3) do |paths|
      decoder = FakeDecoder.new(Array.new(16_000, 0.1))
      limits = []
      instrumented = Object.new
      instrumented.define_singleton_method(:decode) do |path, max_decoded_bytes:, **keywords|
        limits << [File.basename(path.to_s), max_decoded_bytes, Thread.current]
        decoder.decode(path, max_decoded_bytes: max_decoded_bytes, **keywords)
      end
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          audio_memory_gb: 1.0,
          preprocess_workers: 8,
          pipeline_preparation: false
        ),
        model_provider: FakeProvider.new,
        decoder: instrumented
      )

      run = engine.transcribe(paths)

      assert run.ok?
      actual_limits = limits.map { |row| row[1] }
      assert_equal Array.new(3, 1024**3), actual_limits
      assert(limits.all? { |row| row[2].equal?(Thread.current) })
    ensure
      engine&.close
    end
  end

  def test_underestimated_decodes_retry_sequentially_with_the_full_pcm_budget
    with_audio_files(2) do |paths|
      calls = []
      decoder = Object.new
      decoder.define_singleton_method(:estimate_decoded_bytes) { |_path, **| 230 * (1024**2) }
      decoder.define_singleton_method(:decode) do |path, max_decoded_bytes:, **|
        calls << [File.basename(path.to_s), max_decoded_bytes, Thread.current]
        if max_decoded_bytes < 300 * (1024**2)
          raise Cohere::Transcribe::Audio::DecodedAudioLimitError, "metadata underestimated decoded audio"
        end

        Cohere::Transcribe::Audio::Decoded.new(
          samples: Array.new(16_000, 0.1),
          sample_rate: 16_000,
          backend: "fake",
          fallback_reason: nil
        )
      end
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(audio_memory_gb: 1.0, preprocess_workers: 2, pipeline_preparation: true),
        model_provider: FakeProvider.new,
        decoder: decoder
      )

      run = engine.transcribe(paths)

      assert run.ok?
      calls_by_path = calls.group_by(&:first)
      paths.each do |path|
        limits = calls_by_path.fetch(File.basename(path)).map { |call| call.fetch(1) }
        assert_equal [256 * (1024**2), 1024**3], limits
      end
      retry_threads = calls.select { |call| call.fetch(1) == 1024**3 }.map { |call| call.fetch(2) }
      refute(retry_threads.any? { |thread| thread.equal?(Thread.current) })
      assert_equal 1, retry_threads.uniq.length
    ensure
      engine&.close
    end
  end

  def test_progress_abort_cancels_next_group_workers_and_evicts_the_session
    with_audio_files(4) do |paths|
      workers = [2, Etc.nprocessors].min
      next_group_started = Queue.new
      decoder = CoordinatedDecoder.new(
        group_size: workers,
        next_group_started: next_group_started,
        block_next_group: true
      )
      provider = FakeProvider.new
      progress = lambda do |event|
        next unless event.stage == "files" && event.current == 1

        Timeout.timeout(2) { next_group_started.pop }
        raise "stop after first file"
      end
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(preprocess_workers: 2, pipeline_preparation: true),
        progress: progress,
        model_provider: provider,
        decoder: decoder
      )

      error = assert_raises(Cohere::Transcribe::ProgressCallbackError) do
        engine.transcribe(paths)
      end
      assert_match(/stop after first file/, error.message)
      refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-audio-") })
      assert provider.session.closed?
    ensure
      engine&.close
    end
  end

  def test_close_waits_for_preparation_workers_and_no_worker_outlives_the_engine
    with_audio_files(2) do |paths|
      workers = [2, Etc.nprocessors].min
      entered = Queue.new
      release = Queue.new
      provider = FakeProvider.new
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(preprocess_workers: 2, pipeline_preparation: true),
        model_provider: provider,
        decoder: BlockingDecoder.new(entered, release)
      )
      outcome = Queue.new
      transcription = Thread.new do
        outcome << engine.transcribe(paths)
      rescue Exception => e # rubocop:disable Lint/RescueException -- the test must surface thread failures
        outcome << e
      end
      Timeout.timeout(2) { workers.times { entered.pop } }

      closer = Thread.new { engine.close }
      assert_nil closer.join(0.02), "close returned while preparation was active"
      2.times { release << true }

      Timeout.timeout(2) do
        transcription.join
        closer.join
      end
      run = outcome.pop
      raise run if run.is_a?(Exception)

      assert run.ok?
      assert provider.session.closed?
      refute(Thread.list.any? { |thread| thread.name&.start_with?("cohere-audio-") })
    ensure
      2.times { release << true } if defined?(release)
      transcription&.kill
      transcription&.join
      closer&.kill
      closer&.join
      engine&.close
    end
  end

  def test_each_preprocess_worker_reuses_a_thread_confined_silero_session
    with_audio_files(4) do |paths|
      workers = [2, Etc.nprocessors].min
      instances = []
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          vad: "silero",
          vad_engine: "onnx",
          preprocess_workers: 2,
          pipeline_preparation: true
        ),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new(Array.new(16_000, 0.1)),
        silero_factory: lambda do
          ThreadConfinedSilero.new.tap { |instance| instances << instance }
        end
      )

      run = engine.transcribe(paths)

      assert run.ok?
      assert_equal workers, instances.length
      call_count = instances.sum { |instance| instance.calls.length }
      assert_equal 4, call_count
      assert(instances.all? { |instance| instance.calls.uniq.length == 1 })
    ensure
      engine&.close
    end
  end

  def test_packed_compatible_vad_options_are_forwarded_and_file_concurrency_is_capped
    with_audio_files(4) do |paths|
      instances = []
      received = []
      gate = Mutex.new
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          vad: "silero",
          vad_engine: "torch",
          vad_batch_size: 1,
          vad_block_frames: 3,
          vad_threads: 4,
          preprocess_workers: 4,
          pipeline_preparation: true
        ),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new(Array.new(16_000, 0.1)),
        silero_factory: lambda do |**keywords|
          ThreadConfinedSilero.new.tap do |instance|
            gate.synchronize do
              received << keywords
              instances << instance
            end
          end
        end
      )

      run = engine.transcribe(paths)

      assert run.ok?
      assert_equal [{ block_frames: 3, threads: 4 }], received
      assert_equal 1, instances.length
      assert_equal 4, instances.fetch(0).calls.length
    ensure
      engine&.close
    end
  end

  def test_zero_arity_injected_silero_factory_remains_compatible_with_auto_tuning
    with_audio_files(1) do |paths|
      instance = ThreadConfinedSilero.new
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          vad: "silero",
          vad_engine: "auto",
          vad_batch_size: 1,
          vad_block_frames: 2,
          vad_threads: 2
        ),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new(Array.new(16_000, 0.1)),
        silero_factory: -> { instance }
      )

      run = engine.transcribe(paths)

      assert run.ok?
      assert_equal 1, instance.calls.length
    ensure
      engine&.close
    end
  end

  def test_silero_minimum_duration_uses_python_half_even_millisecond_rounding
    instance = ThreadConfinedSilero.new
    requested = base_options.with(
      vad: "silero",
      vad_engine: "onnx",
      min_dur: 0.0025,
      max_dur: 1.0
    )
    engine = Cohere::Transcribe::Runtime::Engine.new(
      requested,
      model_provider: FakeProvider.new
    )
    resolved = engine.send(:resolve_configuration)

    engine.send(:segment, Array.new(16_000, 0.0), resolved, silero: instance)

    assert_equal 2, instance.keyword_calls.fetch(0).fetch(:min_speech_duration_ms)
  ensure
    engine&.close
  end

  def test_native_segment_slices_use_python_half_even_sample_rounding
    captured = []
    session = Object.new
    session.define_singleton_method(:transcribe) do |samples, language:, offset:, max_new_tokens:|
      captured << [samples, language, offset, max_new_tokens]
      Cohere::Transcribe::ASR::NativeResult.new(
        text: "rounded",
        segments: [],
        words: [],
        generated_tokens: 1,
        generation_limit: max_new_tokens,
        generation_capacity: 10,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end
    engine = Cohere::Transcribe::Runtime::Engine.new(base_options, model_provider: FakeProvider.new)
    measurements_class = Cohere::Transcribe::Runtime.const_get(:Measurements, false)

    engine.send(
      :call_native_batch,
      session,
      (0...10).to_a,
      [{ start: 0.5 / 16_000, end: 2.5 / 16_000 }],
      language: "ar",
      measurements: measurements_class.new,
      max_new_tokens: 10
    )

    assert_equal [0, 1], captured.fetch(0).fetch(0)
  ensure
    engine&.close
  end

  def test_direct_onnx_and_jit_requests_ignore_packed_only_batch_and_block_values
    with_audio_files(2) do |paths|
      %w[onnx jit].each do |requested_engine|
        instances = []
        received = []
        engine = Cohere::Transcribe::Runtime::Engine.new(
          base_options.with(
            vad: "silero",
            vad_engine: requested_engine,
            vad_batch_size: 0,
            vad_block_frames: 0,
            preprocess_workers: 2,
            pipeline_preparation: true
          ),
          model_provider: FakeProvider.new,
          decoder: FakeDecoder.new(Array.new(16_000, 0.1)),
          silero_factory: lambda do |**keywords|
            received << keywords
            ThreadConfinedSilero.new.tap { |instance| instances << instance }
          end
        )

        run = engine.transcribe(paths)

        assert run.ok?, requested_engine
        assert_equal [{}, {}], received, requested_engine
        assert_equal 2, instances.length, requested_engine
      ensure
        engine&.close
      end
    end
  end

  def test_durable_contract_retains_requested_vad_engine_after_executor_resolution
    requested = base_options.with(
      vad: "silero",
      vad_engine: "torch",
      vad_batch_size: 1,
      vad_block_frames: 1
    )
    engine = Cohere::Transcribe::Runtime::Engine.new(requested, model_provider: FakeProvider.new)
    resolved = engine.send(:resolve_configuration)
    contract_options = engine.send(:state_contract_options, resolved)

    assert_equal "onnx", resolved.vad_engine
    assert_equal "torch", contract_options.vad_engine
    refute_equal(
      Cohere::Transcribe::State.asr_contract_key(resolved),
      Cohere::Transcribe::State.asr_contract_key(contract_options)
    )
  ensure
    engine&.close
  end

  def test_engine_profile_and_json_publish_effective_onnx_substitution_tuning
    with_audio_files(1) do |paths, directory|
      output_dir = Pathname(directory).join("out")
      profile = Pathname(directory).join("profile.json")
      publication = Cohere::Transcribe::PublicationOptions.new(
        formats: ["json"],
        output_dir: output_dir,
        existing: "overwrite",
        profile_json: profile
      )
      engine = Cohere::Transcribe::Runtime::Engine.new(
        base_options.with(
          vad: "silero",
          vad_engine: "torch",
          vad_batch_size: 2,
          vad_block_frames: 300,
          vad_threads: 4,
          max_dur: 30.0,
          publication: publication
        ),
        model_provider: FakeProvider.new,
        decoder: FakeDecoder.new(Array.new(601 * 512, 0.1)),
        silero_factory: ->(**keywords) { TelemetrySilero.new(**keywords) }
      )

      run = engine.transcribe(paths.first)

      assert run.ok?
      telemetry = JSON.parse(profile.read)
      assert_equal 4, telemetry.dig("vad", "torch_intraop_threads")
      assert_equal 300, telemetry.dig("vad", "effective_block_frames")
      assert_equal 3, telemetry.dig("vad", "model_calls")
      assert_equal 601, telemetry.dig("vad", "valid_frames")
      assert_equal 601, telemetry.dig("vad", "padded_frames")
      assert_equal 1, telemetry.dig("vad", "max_files_per_call")

      output = JSON.parse(output_dir.join("audio-0.json").read)
      assert_equal "torch", output.dig("segmentation_details", "requested_engine")
      assert_equal "onnx", output.dig("segmentation_details", "actual_engine")
      assert_equal(
        { "CPUExecutionProvider" => {} },
        output.dig("segmentation_details", "provider_options")
      )
    ensure
      engine&.close
    end
  end

  private

  def base_options
    Cohere::Transcribe::TranscriptionOptions.new(
      device: "cpu",
      dtype: "fp32",
      vad: "none",
      max_dur: 1.0,
      alignment: "segment"
    )
  end

  def with_audio_files(count)
    Dir.mktmpdir("cohere-engine-test") do |directory|
      paths = Array.new(count) do |index|
        path = File.join(directory, "audio-#{index}.wav")
        File.binwrite(path, "unchanged source")
        path
      end
      yield paths, directory
    end
  end
end
