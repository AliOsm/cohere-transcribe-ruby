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
          assert_includes commits, ref.read
          assert_equal 40, ref.size
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

      def test_concurrent_misses_return_their_own_complete_online_responses
        Dir.mktmpdir("cohere-hub-concurrent-resolution") do |directory|
          late_commit = "d" * 40
          current_commit = "e" * 40
          entered_late_request = Queue.new
          release_late_request = Queue.new
          request_guard = Mutex.new
          requests = 0
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            index = request_guard.synchronize { (requests += 1) - 1 }
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
          release_late_request << true
          assert_equal late_commit, late.value
          assert_equal late_commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
          assert_equal late_commit, hub.resolve_revision("owner/model", "main")
          assert_equal 2, requests
        ensure
          release_late_request << true
          late&.kill
          late&.join
          current&.kill
          current&.join
        end
      end

      def test_deleted_ref_during_resolution_is_recreated_from_the_valid_response
        Dir.mktmpdir("cohere-hub-deleted-ref") do |directory|
          old_commit = cache_snapshot(directory)
          new_commit = "2" * 40
          ref = Pathname(directory).join("models--owner--model/refs/main")
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            ref.delete
            Struct.new(:body).new(JSON.generate("sha" => new_commit))
          end

          assert_equal new_commit, hub.resolve_revision("owner/model", "main")
          assert_equal new_commit, ref.read.strip
          refute_equal old_commit, ref.read.strip
        end
      end

      def test_unreadable_ref_does_not_block_online_resolution_and_is_repaired
        Dir.mktmpdir("cohere-hub-unreadable-ref") do |directory|
          new_commit = "4" * 40
          cache_snapshot(directory)
          ref = Pathname(directory).join("models--owner--model/refs/main")
          ref.chmod(0o000)
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(JSON.generate("sha" => new_commit))
          end

          assert_equal new_commit, hub.resolve_revision("owner/model", "main")
          assert_equal new_commit, ref.read.strip
          assert_equal 0o644, ref.stat.mode & 0o777
        ensure
          ref&.chmod(0o600) if ref&.exist?
        end
      end

      def test_shared_cache_ref_mode_follows_the_refs_directory_and_creates_no_lock
        Dir.mktmpdir("cohere-hub-shared-ref") do |directory|
          commit = "5" * 40
          repository = Pathname(directory).join("models--owner--model")
          refs = repository.join("refs").tap(&:mkpath)
          refs.chmod(0o770)
          hub = Hub.new(cache_dir: directory)

          hub.send(:write_ref, "owner/model", "main", commit)

          ref = refs.join("main")
          assert_equal commit, ref.read.strip
          assert_equal 0o640, ref.stat.mode & 0o777
          assert_empty Dir.glob(repository.join("*.ref.lock").to_s)
        end
      end

      def test_nested_ref_directories_inherit_shared_cache_mode_despite_private_umask
        Dir.mktmpdir("cohere-hub-shared-nested-ref") do |directory|
          cache = Pathname(directory).join("hub").tap(&:mkpath)
          cache.chmod(0o2770)
          commit = "6" * 40
          previous_umask = File.umask(0o077)
          begin
            Hub.new(cache_dir: cache).send(:write_ref, "owner/model", "feature/audio", commit)
          ensure
            File.umask(previous_umask)
          end

          repository = cache.join("models--owner--model")
          refs = repository.join("refs")
          nested = refs.join("feature")
          ref = nested.join("audio")
          [repository, refs, nested].each do |path|
            assert_equal 0o2770, path.stat.mode & 0o2777, path.to_s
          end
          assert_equal commit, ref.read
          assert_equal 0o640, ref.stat.mode & 0o777
        end
      end

      def test_download_payload_and_lock_inherit_shared_cache_mode_despite_private_umask
        Dir.mktmpdir("cohere-hub-shared-download") do |directory|
          cache = Pathname(directory).join("hub").tap(&:mkpath)
          cache.chmod(0o2770)
          commit = "7" * 40
          hub = Hub.new(cache_dir: cache, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            stream.write("shared payload")
          end
          previous_umask = File.umask(0o077)
          begin
            destination = hub.download("owner/model", "nested/weights.bin", revision: commit)
          ensure
            File.umask(previous_umask)
          end

          lock = hub.send(:download_lock_path, destination)
          assert_equal "shared payload", destination.read
          assert_equal 0o640, destination.stat.mode & 0o777
          assert_equal 0o660, lock.stat.mode & 0o777
          assert_equal 0o3770, destination.dirname.stat.mode & 0o3777
        end
      end

      def test_world_writable_cache_uses_role_specific_modes
        Dir.mktmpdir("cohere-hub-world-cache") do |directory|
          cache = Pathname(directory).join("hub").tap(&:mkpath)
          cache.chmod(0o777)
          commit = "7" * 40
          hub = Hub.new(cache_dir: cache, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, stream: nil, **_options|
            stream.write("immutable payload")
          end

          destination = hub.download("owner/model", "weights.bin", revision: commit)
          hub.send(:write_ref, "owner/model", "main", commit)

          repository = cache.join("models--owner--model")
          refs = repository.join("refs")
          assert_equal 0o1777, repository.stat.mode & 0o3777
          assert_equal 0o1777, destination.dirname.stat.mode & 0o3777
          assert_equal 0o777, refs.stat.mode & 0o3777
          assert_equal 0o644, destination.stat.mode & 0o777
          assert_equal 0o644, refs.join("main").stat.mode & 0o777
          assert_equal 0o666, hub.send(:download_lock_path, destination).stat.mode & 0o777
        end
      end

      def test_cache_modes_do_not_cross_a_directory_without_execute_access
        Dir.mktmpdir("cohere-hub-mode-gating") do |directory|
          parent = Pathname(directory).join("parent").tap(&:mkpath)
          parent.chmod(0o760)
          hub = Hub.new(cache_dir: directory)

          assert_equal 0o700, hub.send(:cache_directory_mode, parent)
          assert_equal 0o600, hub.send(:cache_payload_mode, parent)
          assert_equal 0o600, hub.send(:cache_lock_mode, parent)
        end
      end

      def test_cache_mode_updates_tolerate_filesystems_without_mode_changes
        hub = Hub.new
        errors = %i[EPERM EROFS EOPNOTSUPP ENOTSUP].filter_map do |name|
          Errno.const_get(name) if Errno.const_defined?(name)
        end

        errors.each do |error_class|
          target = Object.new
          target.define_singleton_method(:chmod) { |_mode| raise error_class, "unsupported mode change" }
          assert_nil hub.send(:apply_cache_mode_if_supported, target, 0o660), error_class.name
        end

        target = Object.new
        target.define_singleton_method(:chmod) { |_mode| raise Errno::EIO, "broken mount" }
        assert_raises(Errno::EIO) { hub.send(:apply_cache_mode_if_supported, target, 0o660) }
      end

      def test_current_owned_cache_files_are_upgraded_for_shared_use
        Dir.mktmpdir("cohere-hub-owned-modes") do |directory|
          commit = "7" * 40
          repository = Pathname(directory).join("models--owner--model").tap(&:mkpath)
          repository.chmod(0o770)
          snapshot = repository.join("snapshots", commit).tap(&:mkpath)
          refs = repository.join("refs").tap(&:mkpath)
          snapshot.chmod(0o770)
          refs.chmod(0o770)
          payload = snapshot.join("weights.bin").tap { |path| path.write("cached") }
          ref = refs.join("main").tap { |path| path.write(commit) }
          lock = snapshot.join("download.lock").tap { |path| path.write("existing") }
          [payload, ref, lock].each { |path| path.chmod(0o600) }
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) { |*_args, **_options| flunk "cached payload must not be fetched" }

          assert_equal payload, hub.download("owner/model", "weights.bin", revision: commit)
          assert_equal commit, hub.send(:write_ref, "owner/model", "main", commit)
          hub.send(:open_cache_lock, lock, purpose: "Hub download") { |_handle| nil }

          assert_equal 0o640, payload.stat.mode & 0o777
          assert_equal 0o640, ref.stat.mode & 0o777
          assert_equal 0o660, lock.stat.mode & 0o777
          assert_equal 0o1770, repository.stat.mode & 0o3777
        end
      end

      def test_cached_download_upgrades_current_owned_shared_directories_without_a_write
        Dir.mktmpdir("cohere-hub-owned-cached-path") do |directory|
          commit = "7" * 40
          repository = Pathname(directory).join("models--owner--model")
          snapshot_root = repository.join("snapshots")
          snapshot = snapshot_root.join(commit).tap(&:mkpath)
          [repository, snapshot_root, snapshot].each { |path| path.chmod(0o770) }
          payload = snapshot.join("weights.bin").tap { |path| path.write("cached") }
          payload.chmod(0o600)
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) { |*_args, **_options| flunk "cached payload must not be fetched" }

          assert_equal payload, hub.download("owner/model", "weights.bin", revision: commit)

          [repository, snapshot_root, snapshot].each do |path|
            assert_equal 0o1770, path.stat.mode & 0o3777, path.to_s
          end
          assert_equal 0o640, payload.stat.mode & 0o777
        end
      end

      def test_cached_download_does_not_mutate_an_unowned_read_only_cache
        Dir.mktmpdir("cohere-hub-unowned-cached-path") do |directory|
          commit = "7" * 40
          repository = Pathname(directory).join("models--owner--model")
          snapshot_root = repository.join("snapshots")
          snapshot = snapshot_root.join(commit).tap(&:mkpath)
          payload = snapshot.join("weights.bin").tap { |path| path.write("cached") }
          [repository, snapshot_root, snapshot].each { |path| path.chmod(0o550) }
          payload.chmod(0o440)
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:cache_file_owned_by_current_process?) { |_stat| false }
          hub.define_singleton_method(:request) { |*_args, **_options| flunk "cached payload must not be fetched" }

          assert_equal payload, hub.download("owner/model", "weights.bin", revision: commit)

          [repository, snapshot_root, snapshot].each do |path|
            assert_equal 0o550, path.stat.mode & 0o3777, path.to_s
          end
          assert_equal 0o440, payload.stat.mode & 0o777
        ensure
          [repository, snapshot_root, snapshot].compact.each do |path|
            path.chmod(0o700) if path.exist?
          end
          payload&.chmod(0o600) if payload&.exist?
        end
      end

      def test_readable_shared_download_lock_can_be_used_read_only
        Dir.mktmpdir("cohere-hub-readonly-lock") do |directory|
          path = Pathname(directory).join("download.lock").tap { |lock| lock.write("existing") }
          path.chmod(0o440)
          hub = Hub.new(cache_dir: directory)
          hub.define_singleton_method(:cache_file_owned_by_current_process?) { |_stat| false }

          hub.send(:open_cache_lock, path, purpose: "Hub download") do |lock|
            assert_raises(IOError) { lock.write("not writable") }
            assert lock.flock(File::LOCK_EX | File::LOCK_NB)
            lock.flock(File::LOCK_UN)
          end
        ensure
          path&.chmod(0o600) if path&.exist?
        end
      end

      def test_writable_lock_reopen_closes_the_descriptor_after_an_identity_race
        Dir.mktmpdir("cohere-hub-lock-reopen-race") do |directory|
          path = Pathname(directory).join("download.lock").tap { |lock| lock.write("existing") }
          hub = Hub.new(cache_dir: directory)
          captured = nil
          closes = 0
          hub.define_singleton_method(:verify_cache_lock_identity!) do |_path, handle, purpose:|
            captured = handle
            original_close = handle.method(:close)
            handle.define_singleton_method(:close) do
              closes += 1
              original_close.call
            end
            raise Hub::Error, "#{purpose} lock changed during the fixture race"
          end

          assert_raises(Hub::Error) do
            hub.send(:reopen_cache_lock_writable, path, purpose: "Hub download")
          end
          assert_predicate captured, :closed?
          assert_equal 1, closes
        end
      end

      def test_readonly_to_writable_lock_handoff_closes_both_handles_when_killed
        Dir.mktmpdir("cohere-hub-lock-reopen-kill") do |directory|
          path = Pathname(directory).join("download.lock").tap { |lock| lock.write("existing") }
          path.chmod(0o400)
          hub = Hub.new(cache_dir: directory)
          close_started = Queue.new
          readonly = nil
          replacement = nil
          readonly_close_calls = 0
          replacement_close_calls = 0
          original_verify = hub.method(:verify_cache_lock_identity!)
          hub.define_singleton_method(:verify_cache_lock_identity!) do |candidate, handle, purpose:|
            result = original_verify.call(candidate, handle, purpose: purpose)
            if readonly.nil?
              readonly = handle
              original_close = handle.method(:close)
              handle.define_singleton_method(:close) do
                readonly_close_calls += 1
                if readonly_close_calls == 1
                  close_started << true
                  sleep
                end
                original_close.call unless closed?
              end
            else
              replacement = handle
              original_close = handle.method(:close)
              handle.define_singleton_method(:close) do
                replacement_close_calls += 1
                original_close.call unless closed?
              end
            end
            result
          end
          worker = Thread.new do
            hub.send(:open_cache_lock_handle, path, purpose: "Hub download", mode: 0o600)
          end
          worker.report_on_exception = false
          Timeout.timeout(2) { close_started.pop }

          worker.kill

          assert worker.join(2), "cache-lock handoff did not finish after termination"
          assert_predicate readonly, :closed?
          assert_predicate replacement, :closed?
          assert_equal 2, readonly_close_calls
          assert_equal 1, replacement_close_calls
        ensure
          worker&.kill
          worker&.join
          path&.chmod(0o600) if path&.exist?
        end
      end

      def test_unreadable_existing_download_lock_fails_with_remediation
        Dir.mktmpdir("cohere-hub-unreadable-lock") do |directory|
          path = Pathname(directory).join("download.lock").tap { |lock| lock.write("existing") }
          path.chmod(0o000)
          hub = Hub.new(cache_dir: directory)

          error = assert_raises(Hub::Error) do
            hub.send(:open_cache_lock, path, purpose: "Hub download") { flunk "lock must not open" }
          end

          assert_match(/not readable/, error.message)
          assert_match(%r{grant shared read/write access}, error.message)
          assert path.file?
          path.chmod(0o600)
          assert_equal "existing", File.binread(path)
        ensure
          path&.chmod(0o600) if path&.exist?
        end
      end

      def test_readonly_lock_flock_failures_are_typed
        hub = Hub.new
        errors = [Errno::EBADF, Errno::ENOLCK]
        errors << Errno::EOPNOTSUPP if Errno.const_defined?(:EOPNOTSUPP)

        errors.each do |error_class|
          lock = Object.new
          lock.define_singleton_method(:flock) { |_operation| raise error_class, "fixture lock failure" }
          error = assert_raises(Hub::Error, error_class.name) do
            hub.send(:acquire_download_lock!, lock, Pathname("download.lock"), Pathname("weights.bin"))
          end
          assert_instance_of error_class, error.cause
        end
      end

      def test_oversized_ref_is_rejected_and_rewritten_without_reuse
        Dir.mktmpdir("cohere-hub-oversized-ref") do |directory|
          cached_commit = cache_snapshot(directory)
          replacement = "9" * 40
          ref = Pathname(directory).join("models--owner--model/refs/main")
          ref.open("ab") { |file| file.truncate(1024 * 1024) }
          hub = Hub.new(cache_dir: directory, offline: true)

          assert_nil hub.send(:read_ref_commit, ref)
          assert_raises(Hub::Error) { hub.resolve_revision("owner/model", "main") }

          hub.send(:write_ref, "owner/model", "main", replacement)
          assert_equal replacement, ref.read
          assert_equal 40, ref.size
          refute_equal cached_commit, ref.read
        end
      end

      def test_download_lock_filesystem_failures_remain_typed
        errors = [Errno::ENOLCK]
        errors << Errno::EOPNOTSUPP if Errno.const_defined?(:EOPNOTSUPP)
        errors.each do |error_class|
          Dir.mktmpdir("cohere-hub-download-lock-error") do |directory|
            hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
            hub.define_singleton_method(:open_download_lock) do |_path|
              raise error_class, "simulated cache mount lock failure"
            end

            error = assert_raises(Hub::Error, error_class.name) do
              hub.download("owner/model", "weights.bin", revision: "7" * 40)
            end
            assert_instance_of Hub::Error, error
            assert_match(%r{Cannot cache owner/model/weights\.bin}, error.message)
          end
        end
      end

      def test_ref_publication_failure_does_not_invalidate_a_valid_online_resolution
        commit = "6" * 40
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new(JSON.generate("sha" => commit))
        end
        hub.define_singleton_method(:write_ref) do |*_arguments|
          raise Errno::EACCES, "shared cache is read-only"
        end

        assert_equal commit, hub.resolve_revision("owner/model", "main")
        assert_equal commit, hub.resolve_revision("owner/model", "main")
      end

      def test_ref_sync_failure_does_not_invalidate_a_valid_online_resolution
        Dir.mktmpdir("cohere-hub-ref-sync-failure") do |directory|
          commit = "7" * 40
          hub = Hub.new(cache_dir: directory, endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(JSON.generate("sha" => commit))
          end
          hub.define_singleton_method(:sync_directory) do |_path|
            raise Errno::EIO, "simulated ref directory sync failure"
          end

          assert_equal commit, hub.resolve_revision("owner/model", "main")
          assert_equal commit,
                       Pathname(directory).join("models--owner--model/refs/main").read.strip
        end
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

      def test_killed_revision_request_leaves_no_memo_entry
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
        assert_empty hub.instance_variable_get(:@resolution_memo)
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
          assert_empty hub.instance_variable_get(:@resolution_memo)
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

      def test_invalid_revision_schema_is_transient
        [JSON.generate([]), JSON.generate("sha" => "short")].each do |body|
          hub = Hub.new(endpoint: "https://example.invalid")
          hub.define_singleton_method(:request) do |_uri, **_options|
            Struct.new(:body).new(body)
          end

          error = assert_raises(Hub::TransientError, body) do
            hub.resolve_revision("owner/model", "main")
          end
          assert_instance_of Hub::TransientError, error
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

      def test_file_list_rejects_an_all_malformed_sibling_list_accurately
        commit = "f" * 40
        hub = Hub.new(endpoint: "https://example.invalid")
        hub.define_singleton_method(:request) do |_uri, **_options|
          Struct.new(:body).new(
            JSON.generate("siblings" => [{}, nil, { "rfilename" => 123 }, { "rfilename" => "" }])
          )
        end

        error = assert_raises(Hub::Error) do
          hub.list_files("owner/model", revision: commit)
        end

        assert_instance_of Hub::Error, error
        assert_match(/no valid repository file entries/, error.message)
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

      def test_invalid_revision_schema_reuses_a_warm_snapshot
        [JSON.generate([]), JSON.generate("sha" => "short")].each do |body|
          Dir.mktmpdir("cohere-hub-schema-failure") do |directory|
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
