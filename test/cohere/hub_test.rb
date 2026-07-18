# frozen_string_literal: true

require "test_helper"
require "json"
require "io/wait"
require "net/http"
require "socket"
require "timeout"

module Cohere
  module Transcribe
    class HubTest < Minitest::Test
      def test_connection_errors_are_part_of_the_public_transient_category
        assert_operator Hub::ConnectionError, :<, Hub::TransientError
      end

      def test_repository_validation_matches_the_public_model_reference_contract
        hub = Hub.new
        validator = ->(value) { hub.send(:validate_repo_id!, value) }

        %w[model-name owner/model owner_name/model.name].each do |reference|
          assert_nil validator.call(reference), reference
        end
        ["", ".hidden", "owner/model/extra", "owner/model--bad", "owner/model.git"].each do |reference|
          assert_raises(ArgumentError, reference) { validator.call(reference) }
        end
      end

      def test_hub_token_is_never_forwarded_to_a_redirected_cdn_origin
        hub = Hub.new(endpoint: "https://huggingface.co", token: "secret")
        trusted = ->(url) { hub.send(:trusted_authorization_origin?, URI(url)) }

        assert trusted.call("https://huggingface.co/private/model")
        refute trusted.call("https://cdn-lfs.huggingface.co/private/model")
        refute trusted.call("http://huggingface.co/private/model")
        refute trusted.call("https://huggingface.co:444/private/model")
      end

      def test_revision_paths_cannot_escape_the_repository_cache
        Dir.mktmpdir("cohere-hub") do |directory|
          hub = Hub.new(cache_dir: directory)
          validator = ->(value) { hub.send(:validate_revision!, value) }

          %w[main feature/audio refs/pr/123].each do |revision|
            assert_nil validator.call(revision), revision
          end
          ["", "/tmp/escape", "../escape", "feature/../escape", "./main",
           "main/", " main", "main\0evil"].each do |revision|
            assert_raises(ArgumentError, revision.inspect) { validator.call(revision) }
          end

          repository = Pathname(directory).join("models--owner--model")
          resolved = hub.send(:revision_path, repository, "feature/audio")
          assert_equal repository.join("refs/feature/audio").expand_path, resolved
          assert_raises(ArgumentError) { hub.snapshot_path("owner/model", "../../escape") }
        end
      end

      def test_hub_filenames_reject_dot_components_and_nul_bytes
        hub = Hub.new
        validator = ->(value) { hub.send(:validate_filename!, value) }

        %w[config.json onnx/model.onnx].each { |name| assert_nil validator.call(name) }
        ["../config.json", "a/../config.json", "./config.json", "/tmp/config.json",
         "config.json/", "config\0.json", nil].each do |name|
          assert_raises(ArgumentError, name.inspect) { validator.call(name) }
        end
      end

      def test_concurrent_ref_writers_publish_one_complete_immutable_commit
        Dir.mktmpdir("cohere-hub-ref") do |directory|
          hub = Hub.new(cache_dir: directory)
          commits = Array.new(32) { |index| format("%040x", index + 1) }
          ready = Queue.new
          release = Queue.new
          errors = Queue.new
          writers = commits.map do |commit|
            Thread.new do
              ready << true
              release.pop
              hub.send(:write_ref, "owner/model", "feature/audio", commit)
            rescue Exception => e # rubocop:disable Lint/RescueException -- surface every writer failure
              errors << e
            end
          end
          writers.length.times { ready.pop }
          writers.length.times { release << true }
          writers.each(&:join)

          assert errors.empty?, errors.empty? ? nil : errors.pop.full_message
          ref = Pathname(directory).join("models--owner--model/refs/feature/audio")
          assert_includes commits, ref.read.strip
          assert_equal 41, ref.size
          assert_empty Dir.glob(File.join(ref.dirname, ".cohere-transcribe-ref-*.tmp"))
        end
      end

      def test_truncated_http_body_is_not_published_to_the_cache
        server = TCPServer.new("127.0.0.1", 0)
        server_thread = Thread.new do
          socket = server.accept
          while (line = socket.gets)
            break if line == "\r\n"
          end
          socket.write(
            "HTTP/1.1 200 OK\r\n" \
            "Content-Length: 64\r\n" \
            "Connection: close\r\n\r\n" \
            "truncated"
          )
        ensure
          socket&.close
        end

        Dir.mktmpdir("cohere-hub-truncated") do |directory|
          commit = "a" * 40
          hub = Hub.new(
            cache_dir: directory,
            endpoint: "http://127.0.0.1:#{server.addr[1]}"
          )
          destination = hub.snapshot_path("owner/model", commit).join("weights.bin")

          error = assert_raises(Hub::TransientError) do
            hub.download("owner/model", "weights.bin", revision: commit)
          end

          assert_match(/incomplete response body/, error.message)
          refute destination.exist?
          assert_empty Dir.glob(File.join(destination.dirname, "cohere-transcribe-*.download"))
        end
      ensure
        server&.close
        server_thread&.join
      end

      def test_deterministically_invalid_success_metadata_and_redirects_are_definitive
        hub = Hub.new(endpoint: "https://example.invalid")
        response = http_response(302)

        response["content-length"] = "invalid"
        error = assert_raises(Hub::Error) do
          hub.send(:response_content_length, response, URI("https://example.invalid/model"))
        end
        assert_instance_of Hub::Error, error

        error = assert_raises(Hub::Error) do
          hub.send(:resolve_redirect, URI("https://example.invalid/model"), http_response(302))
        end
        assert_instance_of Hub::Error, error

        ["http://[", "file:///tmp/model"].each do |location|
          redirect = http_response(302)
          redirect["location"] = location
          error = assert_raises(Hub::Error, location) do
            hub.send(:resolve_redirect, URI("https://example.invalid/model"), redirect)
          end
          assert_instance_of Hub::Error, error
        end

        error = assert_raises(Hub::Error) do
          hub.send(:request, URI("https://example.invalid/model"), redirects: 9)
        end
        assert_instance_of Hub::Error, error
      end

      def test_transient_http_status_with_non_object_json_remains_typed
        [429, 503].each do |status|
          response = http_response(status, body: "[]")

          assert_raises(Hub::TransientError, status.to_s) do
            Hub.new.send(:raise_http_error!, response, URI("https://example.invalid/model"))
          end
        end
      end

      def test_cached_snapshot_symlink_cannot_escape_its_repository
        Dir.mktmpdir("cohere-hub-symlink") do |directory|
          commit = "b" * 40
          repository = Pathname(directory).join("models--owner--model")
          snapshot = repository.join("snapshots", commit).tap(&:mkpath)
          victim = Pathname(directory).join("victim").tap { |path| path.write("private") }
          destination = snapshot.join("config.json")
          File.symlink(victim, destination)
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            requests += 1
            stream.write("original")
          end

          result = hub.download("owner/model", "config.json", revision: commit)

          assert_equal destination, result
          assert_equal 1, requests
          refute destination.symlink?
          assert_equal "original", destination.read
          assert_equal "private", victim.read
        end
      end

      def test_download_rejects_a_snapshot_directory_symlink_that_escapes_the_cache
        Dir.mktmpdir("cohere-hub-directory-symlink") do |directory|
          commit = "c" * 40
          snapshot = Pathname(directory).join(
            "models--owner--model", "snapshots", commit
          ).tap(&:mkpath)
          outside = Pathname(directory).join("outside").tap(&:mkpath)
          File.symlink(outside, snapshot.join("nested"))
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            stream.write("must not escape")
          end

          error = assert_raises(Hub::Error) do
            hub.download("owner/model", "nested/weights.bin", revision: commit)
          end

          assert_match(/escapes its repository cache/, error.message)
          refute outside.join("weights.bin").exist?
        end
      end

      def test_standard_internal_blob_symlink_is_reused_without_network_access
        Dir.mktmpdir("cohere-hub-blob-symlink") do |directory|
          commit = "d" * 40
          repository = Pathname(directory).join("models--owner--model")
          snapshot = repository.join("snapshots", commit).tap(&:mkpath)
          blob = repository.join("blobs", "fixture").tap do |path|
            path.dirname.mkpath
            path.write("cached")
          end
          destination = snapshot.join("config.json")
          File.symlink(Pathname("../../blobs/fixture"), destination)
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |*_args, **_options|
            raise "cached internal symlink must not contact the network"
          end

          result = hub.download("owner/model", "config.json", revision: commit)

          assert_equal destination, result
          assert destination.symlink?
          assert_equal blob.realpath, destination.realpath
          assert_equal "cached", result.read
        end
      end

      def test_symbolic_revision_reuses_an_internal_snapshot_offline
        Dir.mktmpdir("cohere-hub-offline") do |directory|
          commit = "8" * 40
          repository = Pathname(directory).join("models--owner--model")
          repository.join("refs").tap(&:mkpath).join("main").write("#{commit}\n")
          snapshot = repository.join("snapshots", commit).tap(&:mkpath)
          blob = repository.join("blobs", "config").tap do |path|
            path.dirname.mkpath
            path.write("cached config")
          end
          File.symlink(Pathname("../../blobs/config"), snapshot.join("config.json"))
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid", offline: true)
          hub.define_singleton_method(:request) do |*_args, **_options|
            raise "offline cache reuse must not contact the network"
          end

          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal blob.realpath,
                       hub.download("owner/model", "config.json", revision: "main").realpath
        end
      end

      def test_symbolic_revision_is_revalidated_online_and_updates_the_cached_ref
        Dir.mktmpdir("cohere-hub-refresh") do |directory|
          old_commit = "8" * 40
          new_commit = "9" * 40
          repository = Pathname(directory).join("models--owner--model")
          repository.join("refs").tap(&:mkpath).join("main").write("#{old_commit}\n")
          snapshot = repository.join("snapshots", old_commit).tap(&:mkpath)
          snapshot.join("config.json").write("cached config")
          requests = []
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |uri, **_options|
            requests << uri
            Struct.new(:body).new(JSON.generate("sha" => new_commit))
          end

          assert_equal new_commit, hub.resolve_revision("owner/model", "main")
          assert_equal 1, requests.length
          assert_equal new_commit, repository.join("refs/main").read.strip
        end
      end

      def test_successful_revision_lookups_are_memoized_briefly_then_revalidated
        Dir.mktmpdir("cohere-hub-memo") do |directory|
          commits = ["a" * 40, "b" * 40]
          requests = 0
          clock = 100.0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:monotonic) { clock }
          hub.define_singleton_method(:request) do |_uri, **_options|
            commit = commits.fetch(requests)
            requests += 1
            Struct.new(:body).new(JSON.generate("sha" => commit))
          end

          assert_equal commits[0], hub.resolve_revision("owner/model", "main")
          assert_equal commits[0], hub.resolve_revision("owner/model", "main")
          assert_equal 1, requests

          clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)

          assert_equal commits[1], hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests
        end
      end

      def test_revision_memo_has_a_fixed_entry_limit
        limit = Hub.const_get(:RESOLUTION_MEMO_LIMIT)
        commit = "c" * 40
        requests = 0
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:write_ref) { |_repo_id, _revision, commit, **_options| commit }
        hub.define_singleton_method(:request) do |_uri, **_options|
          requests += 1
          Struct.new(:body).new(JSON.generate("sha" => commit))
        end

        (limit + 1).times do |index|
          assert_equal commit, hub.resolve_revision("owner/model#{index}", "main")
        end
        assert_equal commit, hub.resolve_revision("owner/model0", "main")

        assert_equal limit + 2, requests
        assert_equal limit, hub.instance_variable_get(:@resolution_memo).length
      end

      def test_slow_concurrent_revision_lookup_cannot_replace_a_cached_success
        Dir.mktmpdir("cohere-hub-concurrent-resolution") do |directory|
          late_commit = "d" * 40
          current_commit = "e" * 40
          entered_late_request = Queue.new
          release_late_request = Queue.new
          request_guard = Mutex.new
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            index = request_guard.synchronize do
              current = requests
              requests += 1
              current
            end
            if index.zero?
              entered_late_request << true
              release_late_request.pop
              Struct.new(:body).new(JSON.generate("sha" => late_commit))
            else
              Struct.new(:body).new(JSON.generate("sha" => current_commit))
            end
          end

          late = Thread.new { hub.resolve_revision("owner/model", "main") }
          entered_late_request.pop
          current = Thread.new { hub.resolve_revision("owner/model", "main") }

          assert_equal current_commit, Timeout.timeout(2) { current.value }
          assert_equal current_commit, hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests

          release_late_request << true
          assert_equal current_commit, late.value
          assert_equal current_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
          assert_equal current_commit, hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests
        ensure
          release_late_request << true
          late&.kill
          late&.join
          current&.kill
          current&.join
        end
      end

      def test_late_revision_lookup_cannot_replace_a_newer_ref_after_two_memo_expirations
        Dir.mktmpdir("cohere-hub-expired-concurrent-resolution") do |directory|
          stale_commit = "1" * 40
          current_commit = "2" * 40
          entered_stale_request = Queue.new
          release_stale_request = Queue.new
          request_guard = Mutex.new
          requests = 0
          clock = 100.0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:monotonic) { clock }
          hub.define_singleton_method(:request) do |_uri, **_options|
            index = request_guard.synchronize do
              current = requests
              requests += 1
              current
            end
            if index.zero?
              entered_stale_request << true
              release_stale_request.pop
              Struct.new(:body).new(JSON.generate("sha" => stale_commit))
            else
              Struct.new(:body).new(JSON.generate("sha" => current_commit))
            end
          end

          stale = Thread.new { hub.resolve_revision("owner/model", "main") }
          entered_stale_request.pop
          clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)

          assert_equal current_commit, hub.resolve_revision("owner/model", "main")
          clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)
          release_stale_request << true

          assert_equal current_commit, stale.value
          assert_equal current_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
          assert_equal 2, requests
          assert_empty hub.instance_variable_get(:@resolution_states)
        ensure
          release_stale_request << true
          stale&.kill
          stale&.join
        end
      end

      def test_late_revision_lookup_cannot_replace_a_ref_published_by_another_hub_instance
        Dir.mktmpdir("cohere-hub-cross-instance-resolution") do |directory|
          stale_commit = "3" * 40
          current_commit = "4" * 40
          entered_stale_request = Queue.new
          release_stale_request = Queue.new
          stale_hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          current_hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          stale_hub.define_singleton_method(:request) do |_uri, **_options|
            entered_stale_request << true
            release_stale_request.pop
            Struct.new(:body).new(JSON.generate("sha" => stale_commit))
          end
          current_hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(JSON.generate("sha" => current_commit))
          end

          stale = Thread.new { stale_hub.resolve_revision("owner/model", "main") }
          entered_stale_request.pop
          assert_equal current_commit, current_hub.resolve_revision("owner/model", "main")
          release_stale_request << true

          assert_equal current_commit, stale.value
          assert_equal current_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
        ensure
          release_stale_request << true
          stale&.kill
          stale&.join
        end
      end

      def test_separate_process_ref_publishers_serialize_the_observed_ref_check
        skip "fork is unavailable" unless Process.respond_to?(:fork)

        current_pid = nil
        stale_pid = nil
        current_waited = false
        stale_waited = false
        Dir.mktmpdir("cohere-hub-process-resolution") do |directory|
          stale_commit = "9" * 40
          current_commit = "a" * 40
          hub = Hub.new(cache_dir: directory)
          observed = hub.send(:capture_ref_snapshot, "owner/model", "main")
          current_ready_read, current_ready_write = IO.pipe
          release_current_read, release_current_write = IO.pipe
          current_pid = Process.fork do
            current_ready_read.close
            release_current_write.close
            current_hub = Hub.new(cache_dir: directory)
            original_verify = current_hub.method(:verify_cache_lock_identity!)
            verify_calls = 0
            current_hub.define_singleton_method(:verify_cache_lock_identity!) do |*arguments, **keywords|
              original_verify.call(*arguments, **keywords)
              verify_calls += 1
              next unless verify_calls == 2

              current_ready_write.write("1")
              release_current_read.read(1)
            end
            value = current_hub.send(
              :write_ref,
              "owner/model",
              "main",
              current_commit,
              expected_snapshot: nil
            )
            exit!(value == current_commit ? 0 : 41)
          end
          current_ready_write.close
          release_current_read.close
          assert_equal "1", current_ready_read.read(1)
          current_ready_read.close

          stale_started_read, stale_started_write = IO.pipe
          stale_result_read, stale_result_write = IO.pipe
          stale_pid = Process.fork do
            release_current_write.close
            stale_started_read.close
            stale_result_read.close
            stale_started_write.write("1")
            stale_started_write.close
            value = hub.send(
              :write_ref,
              "owner/model",
              "main",
              stale_commit,
              expected_snapshot: observed
            )
            stale_result_write.write(value)
            stale_result_write.close
            exit!(value == current_commit ? 0 : 42)
          end
          stale_started_write.close
          stale_result_write.close
          assert_equal "1", stale_started_read.read(1)
          stale_started_read.close
          assert_nil stale_result_read.wait_readable(0.1),
                     "stale publisher passed the held reference lock"

          release_current_write.write("1")
          release_current_write.close
          _, current_status = Process.wait2(current_pid)
          current_waited = true
          assert current_status.success?, "current reference publisher failed"
          assert_equal current_commit, stale_result_read.read
          stale_result_read.close
          _, stale_status = Process.wait2(stale_pid)
          stale_waited = true
          assert stale_status.success?, "stale reference publisher was not suppressed"
          assert_equal current_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
        end
      ensure
        [[current_pid, current_waited], [stale_pid, stale_waited]].each do |pid, waited|
          next unless pid && !waited

          Process.kill("KILL", pid)
          Process.wait(pid)
        rescue Errno::ESRCH, Errno::ECHILD
          nil
        end
      end

      def test_post_rename_sync_failure_cannot_let_an_older_response_restore_the_previous_ref
        Dir.mktmpdir("cohere-hub-post-rename-failure") do |directory|
          stale_commit = "5" * 40
          current_commit = "6" * 40
          entered_stale_request = Queue.new
          release_stale_request = Queue.new
          request_guard = Mutex.new
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            index = request_guard.synchronize do
              current = requests
              requests += 1
              current
            end
            if index.zero?
              entered_stale_request << true
              release_stale_request.pop
              Struct.new(:body).new(JSON.generate("sha" => stale_commit))
            else
              Struct.new(:body).new(JSON.generate("sha" => current_commit))
            end
          end
          failed_sync = false
          original_sync = hub.method(:sync_directory)
          hub.define_singleton_method(:sync_directory) do |path|
            unless failed_sync
              failed_sync = true
              raise Errno::EIO, "simulated ref directory sync failure"
            end

            original_sync.call(path)
          end

          stale = Thread.new { hub.resolve_revision("owner/model", "main") }
          entered_stale_request.pop
          assert_raises(Errno::EIO) { hub.resolve_revision("owner/model", "main") }
          release_stale_request << true

          assert_equal current_commit, stale.value
          assert_equal current_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
          assert_equal 2, requests
        ensure
          release_stale_request << true
          stale&.kill
          stale&.join
        end
      end

      def test_late_transient_lookup_returns_a_newer_published_result_after_memo_expiration
        current_commit = "7" * 40
        entered_stale_request = Queue.new
        release_stale_request = Queue.new
        request_guard = Mutex.new
        requests = 0
        clock = 100.0
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:monotonic) { clock }
        hub.define_singleton_method(:write_ref) { |_repo_id, _revision, commit, **_options| commit }
        hub.define_singleton_method(:request) do |_uri, **_options|
          index = request_guard.synchronize do
            current = requests
            requests += 1
            current
          end
          if index.zero?
            entered_stale_request << true
            release_stale_request.pop
            raise Hub::ConnectionError, "stale request failed"
          end

          Struct.new(:body).new(JSON.generate("sha" => current_commit))
        end

        stale = Thread.new { hub.resolve_revision("owner/model", "main") }
        entered_stale_request.pop
        clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)
        assert_equal current_commit, hub.resolve_revision("owner/model", "main")
        clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)
        release_stale_request << true

        assert_equal current_commit, stale.value
        assert_equal 2, requests
        assert_empty hub.instance_variable_get(:@resolution_states)
      ensure
        release_stale_request << true
        stale&.kill
        stale&.join
      end

      def test_unrelated_memo_hit_is_not_blocked_by_ref_fsync
        ready_commit = "3" * 40
        slow_commit = "4" * 40
        write_started = Queue.new
        finish_write = Queue.new
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |uri, **_options|
          commit = uri.path.include?("slow-model") ? slow_commit : ready_commit
          Struct.new(:body).new(JSON.generate("sha" => commit))
        end
        hub.define_singleton_method(:write_ref) do |repo_id, _revision, commit, **_options|
          if repo_id == "owner/slow-model"
            write_started << true
            finish_write.pop
          end
          commit
        end

        assert_equal ready_commit, hub.resolve_revision("owner/ready-model", "main")
        slow = Thread.new { hub.resolve_revision("owner/slow-model", "main") }
        write_started.pop

        assert_equal ready_commit,
                     Timeout.timeout(0.5) { hub.resolve_revision("owner/ready-model", "main") }

        finish_write << true
        assert_equal slow_commit, slow.value
      ensure
        finish_write << true
        slow&.kill
        slow&.join
      end

      def test_first_same_reference_writer_remains_authoritative_while_ref_fsync_runs
        first_commit = "5" * 40
        second_commit = "6" * 40
        write_started = Queue.new
        finish_write = Queue.new
        request_guard = Mutex.new
        requests = 0
        writes = []
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          index = request_guard.synchronize do
            current = requests
            requests += 1
            current
          end
          commit = index.zero? ? first_commit : second_commit
          Struct.new(:body).new(JSON.generate("sha" => commit))
        end
        hub.define_singleton_method(:write_ref) do |_repo_id, _revision, commit, **_options|
          writes << commit
          write_started << true
          finish_write.pop
          commit
        end

        first = Thread.new { hub.resolve_revision("owner/model", "main") }
        write_started.pop
        second = Thread.new { hub.resolve_revision("owner/model", "main") }
        Timeout.timeout(2) { Thread.pass until request_guard.synchronize { requests == 2 } }
        finish_write << true

        assert_equal first_commit, first.value
        assert_equal first_commit, second.value
        assert_equal [first_commit], writes
        assert_equal 2, requests
      ensure
        finish_write << true
        first&.kill
        first&.join
        second&.kill
        second&.join
      end

      def test_killed_revision_request_retires_its_sequence_state
        request_started = Queue.new
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          request_started << true
          sleep
        end
        worker = Thread.new { hub.resolve_revision("owner/model", "main") }
        worker.report_on_exception = false
        request_started.pop

        worker.kill

        assert worker.join(2), "killed Hub request did not finish"
        assert_empty hub.instance_variable_get(:@resolution_states)
      ensure
        worker&.kill
        worker&.join
      end

      def test_kill_during_ref_observation_is_prompt_and_registers_no_ticket
        observation_started = Queue.new
        requests = 0
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:capture_ref_snapshot) do |_repo_id, _revision|
          observation_started << true
          sleep
        end
        hub.define_singleton_method(:request) do |_uri, **_options|
          requests += 1
          raise "request must not start before ref observation finishes"
        end
        worker = Thread.new { hub.resolve_revision("owner/model", "main") }
        worker.report_on_exception = false
        observation_started.pop

        worker.kill

        assert worker.join(2), "Hub resolver did not terminate during ref observation"
        assert_nil worker.value
        assert_equal 0, requests
        assert_empty hub.instance_variable_get(:@resolution_states)
      ensure
        worker&.kill
        worker&.join
      end

      def test_kill_during_ref_sync_is_prompt_and_leaves_the_visible_ref_authoritative
        Dir.mktmpdir("cohere-hub-ref-sync-kill") do |directory|
          commit = "8" * 40
          sync_started = Queue.new
          requests = 0
          block_sync = true
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            requests += 1
            Struct.new(:body).new(JSON.generate("sha" => commit))
          end
          original_sync = hub.method(:sync_directory)
          hub.define_singleton_method(:sync_directory) do |path|
            if block_sync
              block_sync = false
              sync_started << true
              sleep
            end
            original_sync.call(path)
          end
          worker = Thread.new { hub.resolve_revision("owner/model", "main") }
          worker.report_on_exception = false
          sync_started.pop

          worker.kill

          assert worker.join(2), "Hub resolver did not terminate during ref sync"
          assert_nil worker.value
          assert_empty hub.instance_variable_get(:@resolution_states)
          assert_equal commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests
        ensure
          worker&.kill
          worker&.join
        end
      end

      def test_transient_error_is_public_and_used_for_malformed_revision_json
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new("{")
        end

        error = assert_raises(Hub::TransientError) do
          hub.resolve_revision("owner/model", "main")
        end

        assert_match(/Invalid Hub response/, error.message)
      end

      def test_invalid_revision_schema_is_definitive
        [JSON.generate([]), JSON.generate("sha" => "short")].each do |body|
          hub = Hub.new(endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(body)
          end

          error = assert_raises(Hub::Error, body) do
            hub.resolve_revision("owner/model", "main")
          end
          assert_instance_of Hub::Error, error
        end
      end

      def test_malformed_json_file_list_raises_a_transient_error
        commit = "f" * 40
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new("{")
        end

        assert_raises(Hub::TransientError) do
          hub.list_files("owner/model", revision: commit)
        end
      end

      def test_invalid_file_list_schema_is_definitive
        commit = "f" * 40
        [JSON.generate([]), JSON.generate("siblings" => "missing")].each do |body|
          hub = Hub.new(endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(body)
          end

          error = assert_raises(Hub::Error, body) do
            hub.list_files("owner/model", revision: commit)
          end
          assert_instance_of Hub::Error, error
        end
      end

      def test_file_list_ignores_malformed_sibling_entries
        commit = "f" * 40
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new(
            JSON.generate(
              "siblings" => [
                {},
                nil,
                { "rfilename" => 123 },
                { "rfilename" => "config.json" },
                { "rfilename" => "weights.bin" }
              ]
            )
          )
        end

        files = hub.list_files("owner/model", revision: commit)

        assert_equal %w[config.json weights.bin], files
        assert_predicate files, :frozen?
      end

      def test_valid_file_list_is_returned_frozen
        commit = "f" * 40
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new(
            JSON.generate("siblings" => [{ "rfilename" => "config.json" }, { "rfilename" => "weights.bin" }])
          )
        end

        files = hub.list_files("owner/model", revision: commit)

        assert_equal %w[config.json weights.bin], files
        assert_predicate files, :frozen?
      end

      def test_file_list_keeps_authentication_and_not_found_errors_distinct
        commit = "f" * 40
        [[401, Hub::AuthenticationError], [403, Hub::AuthenticationError], [404, Hub::NotFoundError]].each do |status, error_class|
          hub = Hub.new(endpoint: "https://example.invalid")
          response = http_response(status)
          hub.define_singleton_method(:request) do |uri, **_options|
            send(:raise_http_error!, response, uri)
          end

          assert_raises(error_class, status.to_s) do
            hub.list_files("owner/model", revision: commit)
          end
        end
      end

      def test_warm_snapshot_is_reused_for_connection_failures_without_repeated_requests
        Dir.mktmpdir("cohere-hub-connection-fallback") do |directory|
          commit = cache_snapshot(directory)
          clock = 100.0
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:monotonic) { clock }
          hub.define_singleton_method(:request) do |_uri, **_options|
            requests += 1
            raise Hub::ConnectionError, "fixture connection failure"
          end

          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal 1, requests

          clock += Hub.const_get(:RESOLUTION_MEMO_TTL_SECONDS)

          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests
        end
      end

      def test_transient_http_failures_reuse_a_warm_snapshot
        [429, 503].each do |status|
          Dir.mktmpdir("cohere-hub-http-fallback") do |directory|
            commit = cache_snapshot(directory)
            hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
            response = http_response(status)
            hub.define_singleton_method(:request) do |uri, **_options|
              send(:raise_http_error!, response, uri)
            end

            assert_equal commit, hub.resolve_revision("owner/model", "main"), status.to_s
          end
        end
      end

      def test_malformed_json_response_reuses_a_warm_snapshot
        Dir.mktmpdir("cohere-hub-response-fallback") do |directory|
          commit = cache_snapshot(directory)
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new("{")
          end

          assert_equal commit, hub.resolve_revision("owner/model", "main")
        end
      end

      def test_invalid_revision_schema_does_not_reuse_a_warm_snapshot
        [JSON.generate([]), JSON.generate("sha" => "short")].each do |body|
          Dir.mktmpdir("cohere-hub-schema-failure") do |directory|
            cache_snapshot(directory)
            hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
            hub.define_singleton_method(:request) do |_uri, **_options|
              Struct.new(:body).new(body)
            end

            error = assert_raises(Hub::Error, body) do
              hub.resolve_revision("owner/model", "main")
            end
            assert_instance_of Hub::Error, error
          end
        end
      end

      def test_definitive_http_failures_do_not_reuse_a_warm_snapshot
        [[401, Hub::AuthenticationError], [403, Hub::AuthenticationError], [404, Hub::NotFoundError]].each do |status, error_class|
          Dir.mktmpdir("cohere-hub-definitive-failure") do |directory|
            cache_snapshot(directory)
            hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
            response = http_response(status)
            hub.define_singleton_method(:request) do |uri, **_options|
              send(:raise_http_error!, response, uri)
            end

            assert_raises(error_class, status.to_s) do
              hub.resolve_revision("owner/model", "main")
            end
          end
        end
      end

      def test_transient_lookup_memo_still_requires_the_requested_cached_file
        Dir.mktmpdir("cohere-hub-fallback-file") do |directory|
          commit = cache_snapshot(directory)
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            requests += 1
            raise Hub::ConnectionError, "fixture connection failure"
          end

          assert_equal commit, hub.resolve_revision("owner/model", "main", filename: "config.json")
          assert_raises(Hub::ConnectionError) do
            hub.resolve_revision("owner/model", "main", filename: "weights.bin")
          end
          assert_equal 1, requests
        end
      end

      def test_offline_symbolic_revision_without_a_cached_snapshot_fails_without_a_request
        Dir.mktmpdir("cohere-hub-offline-missing") do |directory|
          hub = Hub.new(cache_dir: directory, offline: true)

          error = assert_raises(Hub::Error) do
            hub.resolve_revision("owner/model", "main")
          end
          assert_match(/offline mode has no cached/, error.message)
        end
      end

      def test_concurrent_downloads_publish_only_complete_files_and_clean_temporaries
        Dir.mktmpdir("cohere-hub-downloads") do |directory|
          commit = "7" * 40
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          ready = Queue.new
          release = Queue.new
          requests = 0
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            requests += 1
            ready << true
            release.pop
            stream.write("complete payload")
          end
          threads = Array.new(16) do
            Thread.new { hub.download("owner/model", "weights.bin", revision: commit) }
          end
          ready.pop
          release << true

          paths = threads.map(&:value)

          assert_equal 1, requests
          assert_equal 1, paths.uniq.length
          assert_equal "complete payload", paths.first.read
          assert_empty Dir.glob(File.join(paths.first.dirname, "cohere-transcribe-*.download"))
        end
      end

      def test_stale_download_lock_and_temporary_are_recovered
        Dir.mktmpdir("cohere-hub-stale-download") do |directory|
          commit = "6" * 40
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          destination = hub.snapshot_path("owner/model", commit).join("weights.bin")
          destination.dirname.mkpath
          lock_path = hub.send(:download_lock_path, destination)
          lock_path.write("stale metadata")
          stale = destination.dirname.join(
            "#{hub.send(:download_temporary_prefix, destination)}crashed.download"
          )
          stale.write("partial")
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            stream.write("complete")
          end

          result = hub.download("owner/model", "weights.bin", revision: commit)

          assert_equal "complete", result.read
          refute stale.exist?
          assert lock_path.file?
        end
      end

      def test_download_lock_symlink_is_rejected_without_touching_its_target
        skip "File::NOFOLLOW is unavailable on this platform" unless defined?(File::NOFOLLOW)

        Dir.mktmpdir("cohere-hub-lock-symlink") do |directory|
          commit = "5" * 40
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          destination = hub.snapshot_path("owner/model", commit).join("weights.bin")
          destination.dirname.mkpath
          victim = Pathname(directory).join("victim").tap { |path| path.write("private") }
          File.symlink(victim, hub.send(:download_lock_path, destination))
          hub.define_singleton_method(:request) do |*_args, **_options|
            raise "invalid lock must be rejected before the request"
          end

          error = assert_raises(Hub::Error) do
            hub.download("owner/model", "weights.bin", revision: commit)
          end

          assert_match(/download lock is not a regular file/, error.message)
          assert_equal "private", victim.read
          refute destination.exist?
        end
      end

      private

      def cache_snapshot(directory, repo_id: "owner/model", revision: "main", filename: "config.json")
        commit = "8" * 40
        repository = Pathname(directory).join("models--#{repo_id.gsub("/", "--")}")
        repository.join("refs").tap(&:mkpath).join(revision).tap do |path|
          path.dirname.mkpath
          path.write("#{commit}\n")
        end
        repository.join("snapshots", commit, filename).tap do |path|
          path.dirname.mkpath
          path.write("cached")
        end
        commit
      end

      def http_response(status, body: JSON.generate("error" => "fixture failure"))
        response_class = Net::HTTPResponse::CODE_TO_OBJ.fetch(status.to_s)
        response = response_class.new("1.1", status.to_s, "fixture response")
        response.instance_variable_set(:@body, body)
        response.instance_variable_set(:@read, true)
        response
      end
    end
  end
end
