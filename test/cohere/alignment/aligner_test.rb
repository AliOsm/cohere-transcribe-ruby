# frozen_string_literal: true

require "digest"
require "tmpdir"
require "test_helper"
require "cohere/transcribe/alignment/aligner"

class Cohere::Transcribe::Alignment::AlignerTest < Minitest::Test
  Alignment = Cohere::Transcribe::Alignment
  Aligner = Alignment::Aligner

  class SyntheticSession
    attr_reader :calls

    def initialize(oom_above: nil, oom_class: RuntimeError)
      @oom_above = oom_above
      @oom_class = oom_class
      @calls = []
    end

    def provider
      "CPUExecutionProvider"
    end

    def load_seconds
      0.0
    end

    def run(input_values)
      calls << input_values.dup
      raise @oom_class, "out of memory" if @oom_above && input_values.shape[0] > @oom_above

      Numo::SFloat.zeros(input_values.shape[0], 1_600, 31)
    end
  end

  def test_ctc_kernel_matches_torchaudio_tie_breaking_and_repeated_targets
    log_probs = Numo::SFloat.cast(
      [
        [-0.1, -1.0, -3.0],
        [-2.0, -0.1, -3.0],
        [-0.1, -2.0, -3.0],
        [-3.0, -0.1, -2.0],
        [-0.1, -2.0, -3.0]
      ]
    )
    assert_equal [0, 1, 0, 1, 0], Alignment::CTC.forced_align(log_probs, [1, 1], blank: 0)

    error = assert_raises(ArgumentError) do
      Alignment::CTC.forced_align(log_probs[0...2, true], [1, 1], blank: 0)
    end
    assert_match(/too long for CTC/, error.message)
  end

  def test_arabic_and_english_preprocessing_matches_pinned_uroman_reference
    cases = {
      "مرحبا" => "m r h b a",
      "عَلَيْكُمْ" => "' l y k m",
      "الذكاء" => "a l t h k a '",
      "الاصطناعي" => "a l a s t n a ' y",
      "café" => "c a f e",
      "don’t" => "d o n ' t"
    }
    cases.each do |text, expected|
      assert_equal expected, Alignment::Text.romanize(text, "ara")
    end

    tokens, original = Alignment::Text.preprocess("foo (123) bar", "eng")
    assert_equal ["<star>", "f o o", "<star>", "", "<star>", "b a r"], tokens
    assert_equal ["<star>", "foo", "<star>", "(123)", "<star>", "bar"], original
    assert_equal "foo bar", Alignment::Text.normalize("foo (123) bar")
    assert_equal "foo ١٢٣ bar", Alignment::Text.normalize("foo ١٢٣ bar")
    assert_equal "foo bar", Alignment::Text.normalize("foo ꤀꤁ bar")
  end

  def test_extended_uroman_targets_and_python_whitespace_are_preserved
    cases = {
      "Ä" => "a e",
      "Å" => "a a",
      "Ç" => "s",
      "µ" => "m",
      "ڒ" => "r",
      "ڱ" => "n g",
      "ۍ" => "y e",
      "ؕ" => "t a h",
      "か" => "k a",
      "国" => "g u o",
      "한" => "h a n"
    }
    cases.each do |text, expected|
      assert_equal expected, Alignment::Text.romanize(text, "ara")
      assert_equal expected, Alignment::Text.romanize(text, "eng")
    end

    assert_equal "foo bar baz", Alignment::Text.normalize(" foo\u001Cbar\u0085baz\u202F ")
    assert_equal "\0", Alignment::Text.normalize("\0")
    assert_equal "Ɤ", Alignment::Text.normalize("Ɤ")
    assert_equal "", Alignment::Text.romanize("Ɤ", "eng")
    tokens, originals = Alignment::Text.preprocess("foo\u001Cbar\u0085baz", "eng")
    assert_equal ["<star>", "f o o", "<star>", "b a r", "<star>", "b a z"], tokens
    assert_equal ["<star>", "foo", "<star>", "bar", "<star>", "baz"], originals
  end

  def test_pinned_tokenizer_uroman_sequence_and_context_rules
    cases = {
      "ου" => "o u",
      "μπ" => "b",
      "εμπ" => "e m b",
      "γκ" => "g",
      "εγκ" => "e n g",
      "αντι" => "a n d i",
      "かった" => "k a t t a",
      "じゃ" => "j a",
      "ちょっと" => "c h o t t o",
      "きょ" => "k y o",
      "ショ" => "s h o",
      "っちゃ" => "t c h a"
    }
    cases.each do |text, expected|
      assert_equal expected, Alignment::Text.romanize(text, "ara")
      assert_equal expected, Alignment::Text.romanize(text, "eng")
    end
  end

  def test_streaming_windows_context_crop_wildcard_and_tail_geometry
    length = Aligner::WINDOW_SAMPLES + 160
    audio = Numo::SFloat.new(length).seq
    session = SyntheticSession.new
    emissions, stride = Aligner.new(session: session, batch_size: 4).compute_emissions(audio)

    assert_equal 20.0, stride
    assert_equal [1_501, 32], emissions.shape
    assert_equal 1, session.calls.length
    input = session.calls.first
    assert_equal [2, Aligner::INPUT_SAMPLES], input.shape
    assert_equal Array.new(Aligner::CONTEXT_SAMPLES, 0.0), input[0, 0...Aligner::CONTEXT_SAMPLES].to_a
    assert_equal audio[0], input[0, Aligner::CONTEXT_SAMPLES]
    assert_equal audio[448_000], input[1, 0]
    assert_equal audio[-1], input[1, length - 448_000 - 1]
    assert_equal 0.0, input[1, length - 448_000]
    assert_in_delta(-Math.log(31), emissions[0, 0], 1e-6)
    assert_equal 0.0, emissions[0, 31]
  end

  def test_emission_batch_halves_on_memory_error_without_losing_windows
    audio = Numo::SFloat.zeros(Aligner::WINDOW_SAMPLES + 1)
    session = SyntheticSession.new(oom_above: 1)
    emissions, = Aligner.new(session: session, batch_size: 4).compute_emissions(audio)

    # The learned ceiling starts at four, so the first retry halves that
    # ceiling to two before the second retry reaches a physical batch of one.
    assert_equal([2, 2, 1, 1], session.calls.map { |call| call.shape[0] })
    assert_equal [1_501, 32], emissions.shape
  end

  def test_emission_batch_halves_on_host_no_memory_error
    audio = Numo::SFloat.zeros(Aligner::WINDOW_SAMPLES + 1)
    session = SyntheticSession.new(oom_above: 1, oom_class: NoMemoryError)

    emissions, = Aligner.new(session: session, batch_size: 2).compute_emissions(audio)

    assert_equal([2, 1, 1], session.calls.map { |call| call.shape[0] })
    assert_equal [1_501, 32], emissions.shape
  end

  def test_ctc_alignment_returns_complete_transcript_and_bounds_each_word
    emissions = Numo::SFloat.zeros(20, 32).fill(-20.0)
    emissions[true, 0] = 0.0
    emissions[1..3, 31] = 5.0
    emissions[5..7, 4] = 5.0
    emissions[10..12, 31] = 5.0
    emissions[14..16, 20] = 5.0

    words, fallback_count = Aligner.new(session: SyntheticSession.new).align_emissions(
      emissions,
      20.0,
      [[0.0, 0.4]],
      ["a b"],
      language: "en"
    )
    assert_equal 0, fallback_count
    assert_equal %w[a b], words.map(&:text)
    assert_equal ["ctc"], words.map(&:timing_source).uniq
    assert(words.each_cons(2).all? { |left, right| left.end <= right.start })
    assert(words.all? { |word| word.start.between?(0.0, 0.4) && word.end.between?(word.start, 0.4) })
  end

  def test_unalignable_segment_falls_back_without_dropping_asr_words
    emissions = Numo::SFloat.zeros(10, 32)
    words, fallback_count = Aligner.new(session: SyntheticSession.new).align_emissions(
      emissions,
      20.0,
      [[0.0, 0.2]],
      ["aaaa aaaa"],
      language: "en"
    )

    assert_equal 1, fallback_count
    assert_equal %w[aaaa aaaa], words.map(&:text)
    assert_equal ["uniform_fallback"], words.map(&:timing_source).uniq
  end

  def test_model_artifacts_are_immutable_full_precision_exports_of_the_pinned_model
    provider = Alignment::ModelProvider
    assert_equal "MahmoudAshraf/mms-300m-1130-forced-aligner", provider::SOURCE_REPOSITORY
    assert_equal "49402e9577b1158620820667c218cd494cc44486", provider::SOURCE_REVISION
    assert_equal "CC-BY-NC-4.0", provider::LICENSE
    assert_equal 1_262_529_881, provider::ARTIFACTS.fetch("fp32").size
    assert_equal(
      "429e5d05c62acc8a9264db874a1b131e359fc626e40c253ac7b1fe52b11149b4",
      provider::ARTIFACTS.fetch("fp32").sha256
    )
  end

  def test_model_provider_replaces_a_corrupt_cached_download_before_use
    Dir.mktmpdir("cohere-aligner") do |directory|
      path = Pathname(directory).join("model.onnx")
      valid_bytes = "valid-model".b
      artifact = Alignment::ModelArtifact.new(
        filename: "model.onnx",
        size: valid_bytes.bytesize,
        sha256: Digest::SHA256.hexdigest(valid_bytes)
      )
      hub = Object.new
      calls = 0
      hub.define_singleton_method(:download) do |_repository, _filename, revision:|
        raise "revision was not pinned" unless revision == Alignment::ModelProvider::REVISION

        calls += 1
        File.binwrite(path, calls == 1 ? "corrupt" : valid_bytes)
        path
      end

      actual = Alignment::ModelProvider.new(hub: hub, artifacts: { "fp32" => artifact }).fetch("fp32")
      assert_equal path, actual
      assert_equal 2, calls
      assert_equal valid_bytes, File.binread(path)
    end
  end

  def test_model_provider_rejects_an_integrity_lock_symlink
    skip "File::NOFOLLOW is unavailable on this platform" unless defined?(File::NOFOLLOW)

    Dir.mktmpdir("cohere-aligner") do |directory|
      path = Pathname(directory).join("model.onnx")
      path.binwrite("model")
      victim = Pathname(directory).join("victim")
      victim.write("untouched")
      File.symlink(victim, "#{path}.integrity.lock")
      artifact = Alignment::ModelArtifact.new(
        filename: "model.onnx",
        size: path.size,
        sha256: Digest::SHA256.file(path).hexdigest
      )
      hub = Object.new
      hub.define_singleton_method(:download) { |*| path }

      error = assert_raises(Alignment::BackendUnavailable) do
        Alignment::ModelProvider.new(hub: hub, artifacts: { "fp32" => artifact }).fetch("fp32")
      end

      assert_match(/lock is not a regular file/, error.message)
      assert_equal "untouched", victim.read
    end
  end

  def test_fp16_session_refuses_to_silently_run_on_cpu
    provider = Object.new
    provider.define_singleton_method(:fetch) { |_dtype| Pathname("/tmp/not-opened.onnx") }
    session = Alignment::Session.new(
      dtype: "fp16",
      device: "cuda",
      model_provider: provider,
      available_providers: ["CPUExecutionProvider"],
      session_factory: ->(*) { flunk "session factory must not run" }
    )

    error = assert_raises(Alignment::BackendUnavailable) do
      session.run(Numo::SFloat.zeros(1, 400))
    end
    assert_match(/CUDAExecutionProvider is unavailable/, error.message)
  end

  def test_real_fp32_onnx_logits_match_the_pinned_transformers_checkpoint
    hub = Cohere::Transcribe::Hub.new
    model_path = hub.cached_file(
      Alignment::ModelProvider::REPOSITORY,
      Alignment::ModelProvider::ARTIFACTS.fetch("fp32").filename,
      revision: Alignment::ModelProvider::REVISION
    )
    skip "full MMS ONNX model is not already cached" unless model_path
    skip "cached MMS ONNX model has the wrong size" unless model_path.size == 1_262_529_881

    require "onnxruntime"
    runtime = OnnxRuntime::InferenceSession.new(
      model_path.to_s,
      providers: ["CPUExecutionProvider"],
      graph_optimization_level: :all,
      intra_op_num_threads: 4,
      log_severity_level: 4
    )
    session = Alignment::Session.new(session: runtime)
    input = Numo::SFloat.zeros(1, 32_000)
    32_000.times do |index|
      input[0, index] = (0.2 * Math.sin(2 * Math::PI * 440 * index / 16_000.0)) +
                        (0.05 * Math.sin(2 * Math::PI * 113 * index / 16_000.0))
    end
    logits = session.run(input)

    assert_equal [1, 99, 31], logits.shape
    assert_in_delta 3.899554, logits[0, 0, 0], 4e-4
    assert_in_delta(-3.8989575, logits[0, -1, -1], 4e-4)
    assert_in_delta(-20.549452, logits.min, 4e-4)
    assert_in_delta 4.2698402, logits.max, 4e-4
  rescue LoadError
    skip "onnxruntime gem is unavailable"
  end
end
