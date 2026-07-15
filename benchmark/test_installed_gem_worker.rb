# frozen_string_literal: true

require "minitest/autorun"
require "tempfile"
require "tmpdir"
require_relative "installed_gem_worker"

class InstalledGemWERWorkerTest < Minitest::Test
  DecodedFixture = Data.define(:samples, :backend, :fallback_reason)
  ResultFixture = Data.define(
    :text,
    :generated_tokens,
    :generation_limit,
    :generation_capacity,
    :stopped_by_max_tokens,
    :repetition_stopped
  )

  def test_json_value_preserves_nil_as_json_null
    assert_nil InstalledGemWERWorker.json_value(nil)
    assert_equal({ "value" => nil }, InstalledGemWERWorker.json_value({ value: nil }))
  end

  def test_native_batch_plans_keep_long_files_single_and_length_order_batches
    samples = [
      { "id" => "short", "duration" => 10.0 },
      { "id" => "longest", "duration" => 50.0 },
      { "id" => "middle", "duration" => 20.0 },
      { "id" => "long", "duration" => 40.0 }
    ]

    plans = InstalledGemWERWorker.native_batch_plans(samples, 2)

    assert_equal([%w[longest], %w[long], %w[middle short]], plans.map { |plan| plan.map { |sample| sample.fetch("id") } })
  end

  def test_native_batch_plan_order_is_stable_for_equal_durations
    samples = %w[first second third].map { |id| { "id" => id, "duration" => 2.0 } }

    plans = InstalledGemWERWorker.native_batch_plans(samples, 2)

    assert_equal([%w[first second], %w[third]], plans.map { |plan| plan.map { |sample| sample.fetch("id") } })
  end

  def test_native_operation_uses_cross_file_batch_for_short_rows
    Tempfile.create(["installed-gem-worker", ".wav"]) do |first|
      Tempfile.create(["installed-gem-worker", ".wav"]) do |second|
        samples = [first, second].map.with_index do |file, index|
          [{ "id" => "sample-#{index}", "audio_path" => file.path },
           DecodedFixture.new(samples: Array.new(10), backend: "ffmpeg", fallback_reason: nil)]
        end
        session = batch_session(%w[first-text second-text])
        options = Struct.new(:language, :max_new_tokens).new("ar", 445)

        rows, mode, metrics = InstalledGemWERWorker.native_operation(session, samples, options, "f" * 64, 3)

        assert_equal "batch", mode
        assert_equal(%w[first-text second-text], rows.map { |row| row.fetch("hypothesis") })
        assert_equal(["native_cross_file_batch"] * 2, rows.map { |row| row.dig("provenance", "execution_mode") })
        assert_equal({ "generation_steps" => 4 }, metrics)
      end
    end
  end

  def test_failure_summary_is_preserved_when_worker_raises
    Dir.mktmpdir("installed-gem-worker") do |directory|
      root = Pathname(directory)
      config_path = root.join("config.json")
      summary_path = root.join("summary.json")
      config_path.write(
        JSON.generate(
          "fingerprint" => "f" * 64,
          "lane" => "native_batch",
          "worker_rows_path" => root.join("rows.jsonl").to_s,
          "worker_summary_path" => summary_path.to_s
        )
      )

      error = assert_raises(RuntimeError) do
        InstalledGemWERWorker.run_with_failure_summary(
          config_path,
          runner: ->(_path) { raise "fixture failure" }
        )
      end
      summary = JSON.parse(summary_path.read)

      assert_equal "fixture failure", error.message
      assert_equal "native_batch", summary.fetch("lane")
      assert_equal "RuntimeError", summary.dig("worker_error", "class")
      assert_equal 0, summary.fetch("sample_count")
    end
  end

  def test_aggregate_statistics_recomputes_ratio_and_takes_peak_maximum
    chunks = [
      {
        "statistics" => {
          "elapsed_seconds" => 2.0,
          "successful_audio_seconds" => 10.0,
          "real_time_factor_x" => 5.0,
          "peak_cuda_allocated_gib" => 3.0,
          "generated_tokens" => 4
        }
      },
      {
        "statistics" => {
          "elapsed_seconds" => 8.0,
          "successful_audio_seconds" => 20.0,
          "real_time_factor_x" => 2.5,
          "peak_cuda_allocated_gib" => 2.0,
          "generated_tokens" => 6
        }
      }
    ]

    aggregate = InstalledGemWERWorker.aggregate_statistics(chunks)

    assert_equal 3, aggregate.fetch("real_time_factor_x")
    assert_equal 3, aggregate.fetch("peak_cuda_allocated_gib")
    assert_equal 10, aggregate.fetch("generated_tokens")
  end

  private

  def batch_session(texts)
    results = texts.map do |text|
      ResultFixture.new(
        text: text,
        generated_tokens: 2,
        generation_limit: 445,
        generation_capacity: 896,
        stopped_by_max_tokens: false,
        repetition_stopped: false
      )
    end
    Class.new do
      define_method(:transcribe_batch) do |sample_batches, language:, offsets:, max_new_tokens:|
        raise "wrong batch size" unless sample_batches.length == results.length
        raise "wrong options" unless language == "ar" && offsets == [0.0, 0.0] && max_new_tokens == 445

        results
      end

      define_method(:last_batch_metrics) { { generation_steps: 4 } }
    end.new
  end
end
