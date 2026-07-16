# frozen_string_literal: true

require "test_helper"
require "json"
require "socket"
require "timeout"

module Cohere
  module Transcribe
    class HubTest < Minitest::Test
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

          error = assert_raises(Hub::Error) do
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
        hub.define_singleton_method(:write_ref) { |_repo_id, _revision, _commit| nil }
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

      def test_concurrent_revision_lookups_share_one_request
        Dir.mktmpdir("cohere-hub-coalescing") do |directory|
          commit = "d" * 40
          entered_request = Queue.new
          release_request = Queue.new
          waiter_entered = Queue.new
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            requests += 1
            entered_request << true
            release_request.pop
            Struct.new(:body).new(JSON.generate("sha" => commit))
          end

          first = Thread.new { hub.resolve_revision("owner/model", "main") }
          entered_request.pop
          flight = hub.instance_variable_get(:@resolution_flights).values.fetch(0)
          original_wait = flight.condition.method(:wait)
          flight.condition.define_singleton_method(:wait) do |*arguments|
            waiter_entered << true
            original_wait.call(*arguments)
          end
          second = Thread.new { hub.resolve_revision("owner/model", "main") }
          Timeout.timeout(2) { waiter_entered.pop }
          release_request << true

          assert_equal [commit, commit], [first.value, second.value]
          assert_equal 1, requests
        ensure
          release_request << true
          first&.join
          second&.join
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

      def test_malformed_successful_responses_reuse_a_warm_snapshot
        ["{", JSON.generate([]), JSON.generate("sha" => "short")].each do |body|
          Dir.mktmpdir("cohere-hub-response-fallback") do |directory|
            commit = cache_snapshot(directory)
            hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
            hub.define_singleton_method(:request) do |_uri, **_options|
              Struct.new(:body).new(body)
            end

            assert_equal commit, hub.resolve_revision("owner/model", "main"), body
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

      def http_response(status)
        response_class = Net::HTTPResponse::CODE_TO_OBJ.fetch(status.to_s)
        response = response_class.new("1.1", status.to_s, "fixture response")
        response.instance_variable_set(:@body, JSON.generate("error" => "fixture failure"))
        response.instance_variable_set(:@read, true)
        response
      end
    end
  end
end
