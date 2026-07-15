# frozen_string_literal: true

require "json"

module InstalledGemWERWorker
  SAMPLE_RATE = 16_000
  NATIVE_BATCH_MAX_SAMPLES = 35 * SAMPLE_RATE

  module_function

  def monotonic
    Process.clock_gettime(Process::CLOCK_MONOTONIC)
  end

  def canonical(path)
    File.realpath(path)
  end

  def atomic_json(path, payload)
    destination = Pathname(path)
    temporary = destination.dirname.join(".#{destination.basename}.#{Process.pid}.tmp")
    File.open(temporary, "w", 0o600) do |handle|
      handle.write(JSON.pretty_generate(payload))
      handle.write("\n")
      handle.flush
      handle.fsync
    end
    File.rename(temporary, destination)
  ensure
    File.unlink(temporary) if defined?(temporary) && temporary&.exist?
  end

  def append_rows(path, rows)
    File.open(path, "a", 0o600) do |handle|
      rows.each do |row|
        handle.write(JSON.generate(row))
        handle.write("\n")
      end
      handle.flush
      handle.fsync
    end
  end

  def json_value(value)
    case value
    when nil, true, false, String, Numeric
      value
    when Pathname
      value.to_s
    when Hash
      value.to_h { |key, item| [key.to_s, json_value(item)] }
    when Array
      value.map { |item| json_value(item) }
    else
      value.respond_to?(:to_h) && !value.is_a?(Struct) ? json_value(value.to_h) : value
    end
  end

  def peak_rss_kib
    status = Pathname("/proc/self/status")
    return nil unless status.file?

    match = status.read.match(/^VmHWM:\s+(\d+)\s+kB$/)
    match && Integer(match[1])
  rescue SystemCallError
    nil
  end

  def verify_installed_gem!(expected_root, expected_version)
    require "cohere/transcribe"

    specification = Gem.loaded_specs["cohere-transcribe"]
    raise "cohere-transcribe was loaded without an installed gem specification" unless specification

    actual_root = canonical(specification.full_gem_path)
    expected_root = canonical(expected_root)
    raise "loaded gem root #{actual_root.inspect}, expected #{expected_root.inspect}" unless actual_root == expected_root
    raise "loaded gem version #{specification.version}, expected #{expected_version}" unless specification.version.to_s == expected_version

    feature = $LOADED_FEATURES.find { |path| path.end_with?("/cohere/transcribe.rb") }
    raise "cohere/transcribe was not loaded from the installed gem root" unless feature && canonical(feature).start_with?("#{actual_root}/")

    specification
  end

  def progress_callback
    lambda do |event|
      if event.message
        warn event.message
      elsif event.total && (event.current == event.total || (event.current % 100).zero?)
        warn "#{event.stage}: #{event.current}/#{event.total}"
      end
    end
  end

  def row_for(sample, result, fingerprint, chunk_index)
    status = if result.status == "completed"
               result.text.to_s.empty? ? "empty" : "ok"
             else
               result.status.to_s
             end
    {
      "id" => sample.fetch("id"),
      "fingerprint" => fingerprint,
      "recognition_status" => status,
      "hypothesis" => result.text.to_s,
      "result_path" => result.path.to_s,
      "duration" => result.duration,
      "error" => result.error,
      "chunk_index" => chunk_index,
      "provenance" => json_value(result.provenance)
    }
  end

  def native_row_for(sample, result, fingerprint, chunk_index, decoded, mode)
    {
      "id" => sample.fetch("id"),
      "fingerprint" => fingerprint,
      "recognition_status" => result.text.to_s.empty? ? "empty" : "ok",
      "hypothesis" => result.text.to_s,
      "result_path" => canonical(sample.fetch("audio_path")),
      "duration" => decoded.samples.size.fdiv(SAMPLE_RATE),
      "error" => nil,
      "chunk_index" => chunk_index,
      "provenance" => {
        "execution_mode" => mode,
        "decode_backend" => decoded.backend,
        "decode_fallback_reason" => decoded.fallback_reason,
        "generated_tokens" => result.generated_tokens,
        "generation_limit" => result.generation_limit,
        "generation_capacity" => result.generation_capacity,
        "stopped_by_max_tokens" => result.stopped_by_max_tokens,
        "repetition_stopped" => result.repetition_stopped
      }
    }
  end

  def native_failure_row(sample, fingerprint, chunk_index, error)
    {
      "id" => sample.fetch("id"),
      "fingerprint" => fingerprint,
      "recognition_status" => "failed",
      "hypothesis" => "",
      "result_path" => canonical(sample.fetch("audio_path")),
      "duration" => nil,
      "error" => "#{error.class}: #{error.message}",
      "chunk_index" => chunk_index,
      "provenance" => { "execution_mode" => "decode_failed" }
    }
  end

  def native_batch_plans(samples, batch_size)
    ordered = samples.each_with_index.sort_by do |(sample, index)|
      [-Float(sample["duration"] || 0.0), index]
    end.map(&:first)
    long, eligible = ordered.partition { |sample| Float(sample["duration"] || 0.0) > 35.0 }
    long.map { |sample| [sample] } + eligible.each_slice(batch_size).to_a
  end

  def decode_native_plan(plan, options, fingerprint, chunk_index)
    decoded = []
    failures = []
    byte_limit = (Float(options.audio_memory_gb) * (1024**3)).to_i
    plan.each do |sample|
      audio = Cohere::Transcribe::Audio::Decoder.decode(
        sample.fetch("audio_path"),
        backend: options.audio_backend,
        sample_rate: SAMPLE_RATE,
        max_decoded_bytes: byte_limit
      )
      decoded << [sample, audio]
    rescue StandardError => e
      failures << native_failure_row(sample, fingerprint, chunk_index, e)
    end
    [decoded, failures]
  end

  def native_operation(session, decoded, options, fingerprint, chunk_index)
    if decoded.length == 1 && decoded.first.last.samples.size > NATIVE_BATCH_MAX_SAMPLES
      sample, audio = decoded.first
      result = session.transcribe(
        audio.samples,
        language: options.language,
        max_new_tokens: options.max_new_tokens
      )
      rows = [native_row_for(sample, result, fingerprint, chunk_index, audio, "native_single_processor_chunks")]
      return [rows, "single", nil]
    end

    oversized, batchable = decoded.partition { |_sample, audio| audio.samples.size > NATIVE_BATCH_MAX_SAMPLES }
    rows = []
    oversized.each do |sample, audio|
      result = session.transcribe(
        audio.samples,
        language: options.language,
        max_new_tokens: options.max_new_tokens
      )
      rows << native_row_for(
        sample,
        result,
        fingerprint,
        chunk_index,
        audio,
        "native_single_processor_chunks"
      )
    end
    metrics = nil
    unless batchable.empty?
      results = session.transcribe_batch(
        batchable.map { |_sample, audio| audio.samples },
        language: options.language,
        offsets: Array.new(batchable.length, 0.0),
        max_new_tokens: options.max_new_tokens
      )
      metrics = json_value(session.last_batch_metrics) if session.respond_to?(:last_batch_metrics)
      rows.concat(batchable.zip(results).map do |(sample, audio), result|
        native_row_for(sample, result, fingerprint, chunk_index, audio, "native_cross_file_batch")
      end)
    end
    mode = if oversized.empty?
             "batch"
           else
             (batchable.empty? ? "single" : "mixed")
           end
    [rows, mode, metrics]
  end

  def run_native_batch(config, specification, options, samples, output_path, summary_path, started)
    provider = Cohere::Transcribe::Runtime::ModelProvider.new
    setup_started = monotonic
    identity = provider.resolve(options)
    resolved_options = Cohere::Transcribe::Runtime::Precision.resolve(
      Cohere::Transcribe::Configuration.resolved(options, model_identity: identity)
    )
    session = provider.open(identity, resolved_options)
    setup_seconds = monotonic - setup_started
    requested_batch_size = Integer(resolved_options.batch_size)
    if requested_batch_size > session.batch_capacity
      raise "requested native batch size #{requested_batch_size} exceeds session capacity #{session.batch_capacity}"
    end

    chunk_summaries = []
    row_count = 0
    batch_calls = 0
    single_calls = 0
    failed_decodes = 0
    decode_seconds = 0.0
    inference_seconds = 0.0
    generated_tokens = 0
    begin
      native_batch_plans(samples, requested_batch_size).each_with_index do |plan, chunk_index|
        chunk_started = monotonic
        decode_started = monotonic
        decoded, failures = decode_native_plan(
          plan,
          resolved_options,
          config.fetch("fingerprint"),
          chunk_index
        )
        chunk_decode_seconds = monotonic - decode_started
        decode_seconds += chunk_decode_seconds

        inference_started = monotonic
        rows, mode, native_metrics = if decoded.empty?
                                       [[], "decode_failed", nil]
                                     else
                                       native_operation(
                                         session,
                                         decoded,
                                         resolved_options,
                                         config.fetch("fingerprint"),
                                         chunk_index
                                       )
                                     end
        chunk_inference_seconds = monotonic - inference_started
        inference_seconds += chunk_inference_seconds
        rows.concat(failures)
        append_rows(output_path, rows)
        row_count += rows.length
        failed_decodes += failures.length
        batch_calls += 1 if %w[batch mixed].include?(mode)
        single_calls += decoded.count { |_sample, audio| audio.samples.size > NATIVE_BATCH_MAX_SAMPLES }
        generated_tokens += rows.sum do |row|
          Integer(row.dig("provenance", "generated_tokens") || 0)
        end
        chunk_summaries << {
          "index" => chunk_index,
          "mode" => mode,
          "samples" => rows.length,
          "audio_seconds" => decoded.sum { |_sample, audio| audio.samples.size.fdiv(SAMPLE_RATE) },
          "decode_seconds" => chunk_decode_seconds,
          "inference_seconds" => chunk_inference_seconds,
          "wall_seconds" => monotonic - chunk_started,
          "native_batch_metrics" => native_metrics
        }
        warn "native batches: #{row_count}/#{samples.length}" if row_count == samples.length || (row_count % 100).zero?
      end
    ensure
      session.close
    end
    wall_seconds = monotonic - started
    summary = {
      "worker_version" => 3,
      "lane" => "native_batch",
      "fingerprint" => config.fetch("fingerprint"),
      "gem" => {
        "name" => specification.name,
        "version" => specification.version.to_s,
        "root" => canonical(specification.full_gem_path)
      },
      "ruby" => {
        "version" => RUBY_VERSION,
        "description" => RUBY_DESCRIPTION,
        "platform" => RUBY_PLATFORM
      },
      "process" => {
        "pid" => Process.pid,
        "wall_seconds" => wall_seconds,
        "peak_rss_kib" => peak_rss_kib
      },
      "sample_count" => row_count,
      "batch_size" => requested_batch_size,
      "session" => {
        "backend" => session.backend,
        "compute_backend" => session.compute_backend,
        "device" => session.device,
        "batch_capacity" => session.batch_capacity,
        "model_path" => session.model_path.to_s
      },
      "resolved_options" => json_value(resolved_options),
      "model_identity" => json_value(identity),
      "chunks" => chunk_summaries,
      "aggregate_statistics" => {
        "model_setup_seconds" => setup_seconds,
        "decode_seconds" => decode_seconds,
        "inference_seconds" => inference_seconds,
        "batch_calls" => batch_calls,
        "single_calls" => single_calls,
        "failed_decodes" => failed_decodes,
        "generated_tokens" => generated_tokens
      }
    }
    atomic_json(summary_path, summary)
    warn "installed-gem native-batch worker: #{row_count} samples in #{format("%.3f", wall_seconds)} seconds"
    0
  end

  def aggregate_statistics(chunks)
    numeric = Hash.new(0.0)
    peaks = Hash.new(0.0)
    chunks.each do |chunk|
      chunk.fetch("statistics").each do |key, value|
        next unless value.is_a?(Numeric)
        next if key == "real_time_factor_x"

        if %w[peak_cuda_allocated_gib peak_cuda_reserved_gib].include?(key)
          peaks[key] = [peaks[key], value].max
        else
          numeric[key] += value
        end
      end
    end
    numeric.merge!(peaks)
    elapsed = numeric["elapsed_seconds"]
    numeric["real_time_factor_x"] = numeric.fetch("successful_audio_seconds", 0.0).fdiv(elapsed) if elapsed&.positive?
    numeric.transform_values do |value|
      value.finite? && value == value.to_i ? value.to_i : value
    end
  end

  def run(config_path)
    config = JSON.parse(Pathname(config_path).read(encoding: "UTF-8"))
    output_path = Pathname(config.fetch("worker_rows_path"))
    summary_path = Pathname(config.fetch("worker_summary_path"))
    raise "worker rows already exist: #{output_path}" if output_path.exist?

    specification = verify_installed_gem!(
      config.fetch("expected_gem_root"),
      config.fetch("expected_gem_version")
    )
    option_keywords = config.fetch("options").to_h { |key, value| [key.to_sym, value] }
    options = Cohere::Transcribe::TranscriptionOptions.new(**option_keywords)
    samples = config.fetch("samples")

    paths = samples.map { |sample| canonical(sample.fetch("audio_path")) }
    raise "manifest contains duplicate canonical audio paths" unless paths.uniq.length == paths.length

    started = monotonic
    lane = config.fetch("lane", "public_api")
    if lane == "native_batch"
      return run_native_batch(
        config,
        specification,
        options,
        samples,
        output_path,
        summary_path,
        started
      )
    end
    raise "unsupported benchmark lane: #{lane.inspect}" unless lane == "public_api"

    chunk_size = Integer(config.fetch("chunk_size"))
    raise "chunk_size must be positive" unless chunk_size.positive?

    chunk_summaries = []
    row_count = 0
    transcriber = Cohere::Transcribe::Transcriber.new(options, progress: progress_callback)
    begin
      samples.each_slice(chunk_size).with_index do |chunk, chunk_index|
        chunk_started = monotonic
        run = transcriber.transcribe(chunk.map { |sample| sample.fetch("audio_path") })
        raise "gem returned #{run.results.length} results for #{chunk.length} inputs" if run.results.length != chunk.length

        rows = chunk.zip(run.results).map do |sample, result|
          expected_path = canonical(sample.fetch("audio_path"))
          actual_path = canonical(result.path.to_s)
          raise "result path #{actual_path.inspect} does not match #{expected_path.inspect}" unless actual_path == expected_path

          row_for(sample, result, config.fetch("fingerprint"), chunk_index)
        end
        append_rows(output_path, rows)
        row_count += rows.length
        chunk_summaries << {
          "index" => chunk_index,
          "samples" => rows.length,
          "wall_seconds" => monotonic - chunk_started,
          "statistics" => json_value(run.statistics),
          "resolved_options" => json_value(run.resolved_options),
          "errors" => json_value(run.errors)
        }
      end
    ensure
      transcriber.close
    end
    wall_seconds = monotonic - started

    summary = {
      "worker_version" => 3,
      "lane" => "public_api",
      "fingerprint" => config.fetch("fingerprint"),
      "gem" => {
        "name" => specification.name,
        "version" => specification.version.to_s,
        "root" => canonical(specification.full_gem_path)
      },
      "ruby" => {
        "version" => RUBY_VERSION,
        "description" => RUBY_DESCRIPTION,
        "platform" => RUBY_PLATFORM
      },
      "process" => {
        "pid" => Process.pid,
        "wall_seconds" => wall_seconds,
        "peak_rss_kib" => peak_rss_kib
      },
      "sample_count" => row_count,
      "chunk_size" => chunk_size,
      "chunks" => chunk_summaries,
      "aggregate_statistics" => aggregate_statistics(chunk_summaries)
    }
    atomic_json(summary_path, summary)
    warn "installed-gem worker: #{row_count} samples in #{format("%.3f", wall_seconds)} seconds"
    0
  end

  def run_with_failure_summary(config_path, runner: nil)
    started = monotonic
    (runner || method(:run)).call(config_path)
  rescue Exception => e # rubocop:disable Lint/RescueException -- retain diagnostics for interrupted standalone runs
    begin
      config = JSON.parse(Pathname(config_path).read(encoding: "UTF-8"))
      rows_path = Pathname(config.fetch("worker_rows_path"))
      summary_path = Pathname(config.fetch("worker_summary_path"))
      row_count = if rows_path.file?
                    File.foreach(rows_path).count { |line| !line.strip.empty? }
                  else
                    0
                  end
      unless summary_path.exist?
        atomic_json(
          summary_path,
          {
            "worker_version" => 3,
            "lane" => config.fetch("lane", "public_api"),
            "fingerprint" => config["fingerprint"],
            "process" => {
              "pid" => Process.pid,
              "wall_seconds" => monotonic - started,
              "peak_rss_kib" => peak_rss_kib
            },
            "sample_count" => row_count,
            "worker_error" => {
              "class" => e.class.name,
              "message" => e.message,
              "backtrace" => Array(e.backtrace).first(40)
            }
          }
        )
      end
    rescue StandardError => artifact_error
      warn "cannot preserve worker failure summary: #{artifact_error.class}: #{artifact_error.message}"
    end
    raise
  end
end

if $PROGRAM_NAME == __FILE__
  unless ARGV.length == 1
    warn "usage: ruby installed_gem_worker.rb CONFIG.json"
    exit 2
  end

  exit InstalledGemWERWorker.run_with_failure_summary(ARGV.fetch(0))
end
