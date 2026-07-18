# frozen_string_literal: true

require "fileutils"
require "digest"
require "json"
require "net/http"
require "tempfile"
require "uri"

module Cohere
  module Transcribe
    # Minimal Hugging Face Hub client used by the native Ruby runtime. It uses
    # the standard Hub cache layout, so artifacts already fetched by other Hub
    # clients are reused without copying multi-gigabyte model weights.
    class Hub
      class Error < StandardError; end
      class TransientError < Error; end
      class ConnectionError < TransientError; end
      class AuthenticationError < Error; end
      class NotFoundError < Error; end

      COMMIT_PATTERN = /\A[0-9a-f]{40}\z/i
      DEFAULT_ENDPOINT = "https://huggingface.co"
      RESOLUTION_MEMO_TTL_SECONDS = 5.0
      RESOLUTION_MEMO_LIMIT = 64
      ResolutionMemo = Struct.new(:value, :error, :expires_at, keyword_init: true)
      ResolutionState = Struct.new(
        :publication_guard,
        :next_sequence,
        :active,
        :published_sequence,
        :published_value,
        keyword_init: true
      )
      RefSnapshot = Data.define(:commit, :device, :inode, :size, :mtime_ns, :ctime_ns)
      ResolutionTicket = Struct.new(:key, :state, :sequence, :observed_ref, keyword_init: true)
      UNCONDITIONAL_REF_WRITE = Object.new.freeze

      private_constant :RESOLUTION_MEMO_TTL_SECONDS, :RESOLUTION_MEMO_LIMIT, :ResolutionMemo,
                       :ResolutionState, :RefSnapshot, :ResolutionTicket, :UNCONDITIONAL_REF_WRITE

      attr_reader :cache_dir, :endpoint

      def initialize(cache_dir: nil, endpoint: nil, token: nil, offline: nil)
        hf_home = ENV.fetch("HF_HOME", File.expand_path("~/.cache/huggingface"))
        @cache_dir = Pathname(cache_dir || ENV.fetch("HF_HUB_CACHE", File.join(hf_home, "hub"))).expand_path
        @endpoint = (endpoint || ENV.fetch("HF_ENDPOINT", DEFAULT_ENDPOINT)).sub(%r{/+\z}, "")
        @endpoint_uri = URI(@endpoint)
        @token = token || ENV["HF_TOKEN"] || cached_token(hf_home)
        @offline = if offline.nil?
                     truthy_environment?(ENV.fetch("HF_HUB_OFFLINE", nil))
                   else
                     offline != false
                   end
        @resolution_guard = Mutex.new
        @resolution_memo = {}
        @resolution_states = {}
      end

      def offline?
        @offline
      end

      def resolve_revision(repo_id, revision = nil, filename: "config.json")
        validate_repo_id!(repo_id)
        requested = revision || "main"
        validate_revision!(requested)
        return requested.downcase if COMMIT_PATTERN.match?(requested)

        if offline?
          cached = cached_revision(repo_id, requested, filename: filename)
          return cached if cached

          raise Error,
                "Hub offline mode has no cached #{filename} snapshot for #{repo_id.inspect} at #{requested.inspect}"
        end

        begin
          resolve_online_revision(repo_id, requested)
        rescue TransientError
          cached = cached_revision(repo_id, requested, filename: filename)
          raise unless cached

          cached
        end
      end

      def download(repo_id, filename, revision: nil)
        validate_filename!(filename)
        commit = resolve_revision(repo_id, revision, filename: filename)
        repository = cache_dir.join(cache_repo_name(repo_id))
        destination = repository.join("snapshots", commit, filename)
        return destination if safe_cached_file(repository, destination)

        prepare_cache_directory!(repository, destination.dirname)
        encoded_repo = repo_id.split("/").map { |part| URI.encode_www_form_component(part) }.join("/")
        encoded_filename = filename.split("/").map { |part| URI.encode_www_form_component(part) }.join("/")
        uri = URI("#{endpoint}/#{encoded_repo}/resolve/#{commit}/#{encoded_filename}")

        open_download_lock(download_lock_path(destination)) do |lock|
          raise Error, "Cannot acquire Hub download lock for #{destination}" unless lock.flock(File::LOCK_EX)

          verify_cache_lock_identity!(
            download_lock_path(destination),
            lock,
            purpose: "Hub download"
          )
          return destination if safe_cached_file(repository, destination)

          cleanup_download_temporaries(destination)
          Tempfile.create(
            [download_temporary_prefix(destination), ".download"],
            destination.dirname.to_s,
            binmode: true
          ) do |temporary|
            request(uri, stream: temporary)
            temporary.flush
            temporary.fsync
            temporary.close
            File.rename(temporary.path, destination)
          end
          sync_directory(destination.dirname)
        end
        destination
      rescue Errno::EACCES, Errno::ENOSPC, Errno::EROFS => e
        raise Error, "Cannot cache #{repo_id}/#{filename}: #{e.message}"
      end

      def list_files(repo_id, revision: nil)
        commit = resolve_revision(repo_id, revision)
        encoded_repo = repo_id.split("/").map { |part| URI.encode_www_form_component(part) }.join("/")
        response = request(URI("#{endpoint}/api/models/#{encoded_repo}/revision/#{commit}"))
        payload = JSON.parse(response.body.to_s)
        siblings = payload.is_a?(Hash) ? payload["siblings"] : nil
        raise Error, "Hub returned no repository file list for #{repo_id}@#{commit}" unless siblings.is_a?(Array)

        siblings.filter_map do |item|
          name = item.is_a?(Hash) ? item["rfilename"] : nil
          name if name.is_a?(String)
        end.freeze
      rescue JSON::ParserError => e
        raise TransientError, "Invalid Hub response while listing #{repo_id.inspect}: #{e.message}"
      end

      def snapshot_path(repo_id, commit)
        validate_repo_id!(repo_id)
        raise ArgumentError, "Invalid immutable Hub commit: #{commit.inspect}" unless commit.is_a?(String) && COMMIT_PATTERN.match?(commit)

        cache_dir.join(cache_repo_name(repo_id), "snapshots", commit)
      end

      def cached_file(repo_id, filename, revision: nil)
        validate_repo_id!(repo_id)
        validate_filename!(filename)
        validate_revision!(revision) if revision
        commit = if revision && COMMIT_PATTERN.match?(revision)
                   revision.downcase
                 else
                   cached_revision(repo_id, revision || "main", filename: filename)
                 end
        return unless commit

        repository = cache_dir.join(cache_repo_name(repo_id))
        path = repository.join("snapshots", commit, filename)
        safe_cached_file(repository, path)
      end

      private

      def resolve_online_revision(repo_id, revision)
        key = resolution_key(repo_id, revision)
        Thread.handle_interrupt(Object => :never) do
          memo, ticket = claim_resolution(key, repo_id, revision)
          return resolution_value(memo.value, memo.error) if memo

          # Concurrent misses deliberately perform independent requests. The
          # first successful publication becomes the shared result for this
          # client's brief memo window. Per-key publication sequencing prevents
          # a response from an older request replacing a newer published ref
          # even when that response arrives after the memo has expired.
          begin
            value = Thread.handle_interrupt(Object => :immediate) do
              fetch_online_revision(repo_id, revision)
            end
            Thread.handle_interrupt(Object => :immediate) do
              publish_resolution(ticket, repo_id, revision, value)
            end
          rescue TransientError => e
            publish_resolution_error(ticket, e)
          ensure
            finish_resolution(ticket)
          end
        end
      end

      def fetch_online_revision(repo_id, revision)
        encoded_repo = repo_id.split("/").map { |part| URI.encode_www_form_component(part) }.join("/")
        encoded_revision = URI.encode_www_form_component(revision)
        response = request(URI("#{endpoint}/api/models/#{encoded_repo}/revision/#{encoded_revision}"))
        payload = JSON.parse(response.body.to_s)
        commit = payload.is_a?(Hash) ? payload["sha"] : nil
        unless commit.is_a?(String) && COMMIT_PATTERN.match?(commit)
          raise Error, "Hub returned no immutable commit for #{repo_id.inspect} at #{revision.inspect}"
        end

        commit.downcase
      rescue JSON::ParserError => e
        raise TransientError, "Invalid Hub response while resolving #{repo_id.inspect}: #{e.message}"
      end

      def claim_resolution(key, repo_id, revision)
        memo = @resolution_guard.synchronize do
          prune_resolution_memo
          @resolution_memo[key]
        end
        return [memo, nil] if memo

        observed_ref = Thread.handle_interrupt(Object => :immediate) do
          capture_ref_snapshot(repo_id, revision)
        end
        Thread.handle_interrupt(Object => :never) do
          @resolution_guard.synchronize do
            prune_resolution_memo
            memo = @resolution_memo[key]
            return [memo, nil] if memo

            state = @resolution_states[key] ||= ResolutionState.new(
              publication_guard: Mutex.new,
              next_sequence: 0,
              active: 0,
              published_sequence: nil,
              published_value: nil
            )
            state.next_sequence += 1
            state.active += 1
            ticket = ResolutionTicket.new(
              key: key,
              state: state,
              sequence: state.next_sequence,
              observed_ref: observed_ref
            )
            [nil, ticket]
          end
        end
      end

      def publish_resolution(ticket, repo_id, revision, value)
        ticket.state.publication_guard.synchronize do
          published = @resolution_guard.synchronize do
            prune_resolution_memo
            memo = @resolution_memo[ticket.key]
            if memo && !memo.error
              memo.value
            elsif ticket.state.published_sequence && ticket.sequence < ticket.state.published_sequence
              ticket.state.published_value
            end
          end
          return published if published

          # Ref preparation and synchronization stay interruptible and outside
          # the instance-wide memo mutex. The observed-ref check remains the
          # authority if termination lands after rename but before this
          # instance records the completed value.
          authoritative = write_ref(
            repo_id,
            revision,
            value,
            expected_snapshot: ticket.observed_ref
          )
          Thread.handle_interrupt(Object => :never) do
            @resolution_guard.synchronize do
              ticket.state.published_sequence = ticket.sequence
              ticket.state.published_value = authoritative
              memoize_resolution(ticket.key, authoritative, nil)
            end
            value = authoritative
          end
        end
        value
      end

      def publish_resolution_error(ticket, error)
        @resolution_guard.synchronize do
          prune_resolution_memo
          memo = @resolution_memo[ticket.key]
          return resolution_value(memo.value, memo.error) if memo
          return ticket.state.published_value if ticket.state.published_sequence &&
                                                 ticket.sequence < ticket.state.published_sequence

          memoize_resolution(ticket.key, nil, error)
        end
        raise error
      end

      def finish_resolution(ticket)
        Thread.handle_interrupt(Object => :never) do
          @resolution_guard.synchronize do
            ticket.state.active -= 1
            return unless ticket.state.active.zero?
            return unless @resolution_states[ticket.key].equal?(ticket.state)

            @resolution_states.delete(ticket.key)
          end
        end
      end

      def memoize_resolution(key, value, error)
        @resolution_memo.delete(key)
        @resolution_memo[key] = ResolutionMemo.new(
          value: value,
          error: error,
          expires_at: monotonic + RESOLUTION_MEMO_TTL_SECONDS
        )
        @resolution_memo.shift while @resolution_memo.length > RESOLUTION_MEMO_LIMIT
      end

      def resolution_value(value, error)
        raise error if error

        value
      end

      def prune_resolution_memo
        now = monotonic
        @resolution_memo.delete_if { |_key, entry| entry.expires_at <= now }
      end

      def resolution_key(repo_id, revision)
        [repo_id.dup.freeze, revision.dup.freeze].freeze
      end

      def monotonic
        Process.clock_gettime(Process::CLOCK_MONOTONIC)
      end

      def request(uri, stream: nil, redirects: 0)
        raise Error, "Hub offline mode prevents a request to #{uri}" if offline?
        raise Error, "Too many redirects while fetching #{uri}" if redirects > 8

        request = Net::HTTP::Get.new(uri)
        request["Authorization"] = "Bearer #{@token}" if @token && !@token.empty? && trusted_authorization_origin?(uri)
        # Keep Content-Length comparable to the bytes written below. Net::HTTP
        # otherwise permits transparent content decoding.
        request["Accept-Encoding"] = "identity"
        request["User-Agent"] = "cohere-transcribe-ruby/#{Cohere::Transcribe::VERSION}"
        response = Net::HTTP.start(
          uri.host,
          uri.port,
          use_ssl: uri.scheme == "https",
          open_timeout: 30,
          read_timeout: 300
        ) do |http|
          if stream
            http.request(request) do |incoming|
              case incoming
              when Net::HTTPSuccess
                written = 0
                incoming.read_body do |chunk|
                  count = stream.write(chunk)
                  unless count == chunk.bytesize
                    raise Error,
                          "Cannot write the complete Hub response body for #{uri}: " \
                          "wrote #{count.inspect} of #{chunk.bytesize} bytes"
                  end
                  written += count
                end
                expected = response_content_length(incoming, uri)
                if expected && written != expected
                  raise TransientError,
                        "Hub returned an incomplete response body for #{uri}: " \
                        "received #{written} bytes, expected #{expected}"
                end
              when Net::HTTPRedirection
                return request(resolve_redirect(uri, incoming), stream: stream, redirects: redirects + 1)
              else
                raise_http_error!(incoming, uri)
              end
            end
          else
            http.request(request)
          end
        end

        return response if stream
        return response if response.is_a?(Net::HTTPSuccess)
        return request(resolve_redirect(uri, response), redirects: redirects + 1) if response.is_a?(Net::HTTPRedirection)

        raise_http_error!(response, uri)
      rescue Timeout::Error, SocketError, SystemCallError, EOFError,
             Net::HTTPBadResponse, Net::HTTPHeaderSyntaxError, OpenSSL::SSL::SSLError => e
        raise ConnectionError, "Cannot reach Hugging Face Hub at #{uri}: #{e.class}: #{e.message}"
      end

      def response_content_length(response, uri)
        value = response["content-length"]
        return unless value
        return Integer(value, 10) if value.match?(/\A[0-9]+\z/)

        raise Error, "Hub returned an invalid Content-Length for #{uri}: #{value.inspect}"
      end

      def resolve_redirect(base, response)
        location = response["location"]
        raise Error, "Hub redirect from #{base} did not include a location" unless location

        resolved = URI.join(base.to_s, location)
        unless resolved.is_a?(URI::HTTP) && resolved.host
          raise Error, "Hub redirect from #{base} returned an invalid location: #{location.inspect}"
        end

        resolved
      rescue URI::Error => e
        raise Error,
              "Hub redirect from #{base} returned an invalid location: #{location.inspect}",
              cause: e
      end

      def trusted_authorization_origin?(uri)
        uri.scheme == @endpoint_uri.scheme && uri.host == @endpoint_uri.host &&
          uri.port == @endpoint_uri.port
      end

      def raise_http_error!(response, uri)
        detail = begin
          parsed = JSON.parse(response.body.to_s)
          parsed.is_a?(Hash) ? parsed["error"] || parsed["message"] : nil
        rescue JSON::ParserError
          nil
        end
        detail ||= response.message
        message = "Hugging Face request failed (#{response.code}) for #{uri}: #{detail}"
        case response.code.to_i
        when 401, 403 then raise AuthenticationError, message
        when 404 then raise NotFoundError, message
        when 429, 500..599 then raise TransientError, message
        else raise Error, message
        end
      end

      def cached_revision(repo_id, revision, filename:)
        repository = cache_dir.join(cache_repo_name(repo_id))
        candidates = []
        ref = revision_path(repository, revision)
        cached_ref = safe_cached_file(repository, ref)
        candidates << cached_ref.read.strip if cached_ref
        candidates << revision if COMMIT_PATTERN.match?(revision)
        candidates.find do |candidate|
          COMMIT_PATTERN.match?(candidate) &&
            safe_cached_file(repository, repository.join("snapshots", candidate, filename))
        end
      rescue SystemCallError
        nil
      end

      def write_ref(repo_id, revision, commit, expected_snapshot: UNCONDITIONAL_REF_WRITE)
        repository = cache_dir.join(cache_repo_name(repo_id))
        path = revision_path(repository, revision)
        prepare_cache_directory!(repository, repository)
        lock_path = ref_publication_lock_path(repository, revision)
        open_cache_lock(lock_path, purpose: "Hub reference publication") do |lock|
          raise Error, "Cannot acquire Hub reference publication lock for #{path}" unless lock.flock(File::LOCK_EX)

          verify_cache_lock_identity!(lock_path, lock, purpose: "Hub reference publication")
          current = capture_ref_snapshot(repo_id, revision)
          unless expected_snapshot.equal?(UNCONDITIONAL_REF_WRITE) || current == expected_snapshot
            return current.commit if current&.commit

            raise TransientError,
                  "Hub reference changed while resolving #{repo_id.inspect} at #{revision.inspect}"
          end
          return commit if current&.commit == commit

          prepare_cache_directory!(repository, path.dirname)
          # Each writer publishes a complete immutable commit from its own
          # temporary file. The retained lock and observed-ref comparison keep
          # a response that started from an older cache view from replacing a
          # ref already published by another Hub instance or process.
          Tempfile.create([".cohere-transcribe-ref-", ".tmp"], path.dirname.to_s) do |temporary|
            temporary.write("#{commit}\n")
            temporary.flush
            temporary.fsync
            temporary.close
            verify_cache_lock_identity!(lock_path, lock, purpose: "Hub reference publication")
            File.rename(temporary.path, path)
            verify_cache_lock_identity!(lock_path, lock, purpose: "Hub reference publication")
          end
          sync_directory(path.dirname)
          verify_cache_lock_identity!(lock_path, lock, purpose: "Hub reference publication")
        end
        commit
      end

      def capture_ref_snapshot(repo_id, revision)
        repository = cache_dir.join(cache_repo_name(repo_id))
        path = revision_path(repository, revision)
        path = safe_cached_file(repository, path)
        return nil unless path

        3.times do
          handle = nil
          begin
            flags = File::RDONLY
            flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
            flags |= File::CLOEXEC if defined?(File::CLOEXEC)
            descriptor = ::IO.sysopen(path.to_s, flags)
            handle = File.new(descriptor, "r", autoclose: true)
            descriptor = nil
            opened = handle.stat
            body = handle.read(256)
            current = path.lstat
            next unless opened.file? && !current.symlink? && opened.dev == current.dev && opened.ino == current.ino

            text = body.to_s.strip
            commit = text.downcase if COMMIT_PATTERN.match?(text)
            return RefSnapshot.new(
              commit: commit,
              device: opened.dev,
              inode: opened.ino,
              size: opened.size,
              mtime_ns: stat_nanoseconds(opened.mtime),
              ctime_ns: stat_nanoseconds(opened.ctime)
            )
          rescue Errno::ENOENT
            return nil
          rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO
            begin
              current = path.lstat
            rescue Errno::ENOENT
              next
            end
            return RefSnapshot.new(
              commit: nil,
              device: current.dev,
              inode: current.ino,
              size: current.size,
              mtime_ns: stat_nanoseconds(current.mtime),
              ctime_ns: stat_nanoseconds(current.ctime)
            )
          ensure
            handle&.close
            ::IO.new(descriptor).close if descriptor
          end
        end

        raise TransientError,
              "Hub reference changed repeatedly while resolving #{repo_id.inspect} at #{revision.inspect}"
      rescue SystemCallError => e
        raise Error, "Cannot inspect cached Hub reference #{path}: #{e.message}", cause: e
      end

      def ref_publication_lock_path(repository, revision)
        digest = Digest::SHA256.hexdigest(revision)[0, 24]
        repository.join(".cohere-transcribe-#{digest}.ref.lock")
      end

      def stat_nanoseconds(time)
        (time.to_i * 1_000_000_000) + time.nsec
      end

      def safe_cached_file(repository, path)
        return unless path.file?

        cache_root = cache_dir.realpath
        repository_stat = repository.lstat
        return if repository_stat.symlink? || !repository_stat.directory?
        return unless repository.parent.realpath == cache_root

        repository_root = repository.realpath
        resolved = path.realpath
        return unless resolved.file? && within_directory?(resolved, repository_root)

        path
      rescue SystemCallError
        nil
      end

      def download_cache_key(destination)
        Digest::SHA256.hexdigest(destination.basename.to_s)[0, 24]
      end

      def download_lock_path(destination)
        destination.dirname.join(".cohere-transcribe-#{download_cache_key(destination)}.download.lock")
      end

      def download_temporary_prefix(destination)
        "cohere-transcribe-#{download_cache_key(destination)}-"
      end

      def cleanup_download_temporaries(destination)
        pattern = destination.dirname.join("#{download_temporary_prefix(destination)}*.download")
        Dir.glob(pattern.to_s).each do |path|
          File.unlink(path)
        rescue Errno::ENOENT
          nil
        end
      end

      def open_download_lock(path, &)
        open_cache_lock(path, purpose: "Hub download", &)
      end

      def open_cache_lock(path, purpose:)
        flags = File::RDWR | File::CREAT
        flags |= File::NOFOLLOW if defined?(File::NOFOLLOW)
        flags |= File::CLOEXEC if defined?(File::CLOEXEC)
        descriptor = ::IO.sysopen(path.to_s, flags, 0o600)
        lock = File.new(descriptor, "r+", autoclose: true)
        descriptor = nil
        verify_cache_lock_identity!(path, lock, purpose: purpose)

        yield lock
      rescue Errno::ELOOP, Errno::EISDIR, Errno::ENXIO => e
        raise Error, "#{purpose} lock is not a regular file: #{path}", cause: e
      ensure
        lock&.close
        ::IO.new(descriptor).close if descriptor
      end

      def verify_cache_lock_identity!(path, lock, purpose:)
        opened = lock.stat
        current = path.lstat
        return if opened.file? && !current.symlink? && opened.dev == current.dev && opened.ino == current.ino

        raise Error, "#{purpose} lock changed while held or is not a regular file: #{path}"
      rescue SystemCallError => e
        raise Error, "#{purpose} lock changed while held or is not a regular file: #{path}", cause: e
      end

      # Build each repository subdirectory only after proving that its parent is
      # still inside this repository. This prevents a poisoned snapshots/refs
      # symlink from redirecting a Tempfile and rename outside the cache.
      def prepare_cache_directory!(repository, directory)
        FileUtils.mkdir_p(cache_dir)
        cache_root = cache_dir.realpath
        begin
          Dir.mkdir(repository, 0o755)
        rescue Errno::EEXIST
          nil
        end

        repository_stat = repository.lstat
        unless repository_stat.directory? && !repository_stat.symlink? &&
               repository.parent.realpath == cache_root
          raise Error, "Hub cache repository is not a regular directory: #{repository}"
        end

        repository_root = repository.realpath
        relative = directory.relative_path_from(repository)
        if relative.absolute? || relative.each_filename.any? { |part| part == ".." }
          raise Error, "Hub cache path #{directory} escapes its repository cache #{repository}"
        end

        current = repository
        relative.each_filename do |part|
          next if part == "."

          current = current.join(part)
          begin
            Dir.mkdir(current, 0o755)
          rescue Errno::EEXIST
            nil
          end
          resolved = current.realpath
          unless resolved.directory? && within_directory?(resolved, repository_root)
            raise Error, "Hub cache path #{current} escapes its repository cache #{repository}"
          end
        end
        directory
      rescue Error
        raise
      rescue SystemCallError => e
        raise Error, "Cannot prepare Hub cache directory #{directory}: #{e.message}"
      end

      def within_directory?(path, directory)
        path_text = path.to_s
        directory_text = directory.to_s
        path_text == directory_text || path_text.start_with?("#{directory_text}#{File::SEPARATOR}")
      end

      def sync_directory(path)
        File.open(path, File::RDONLY, &:fsync)
      rescue Errno::EINVAL, Errno::ENOTSUP, Errno::EISDIR
        nil
      end

      def cached_token(hf_home)
        token_path = File.join(hf_home, "token")
        File.file?(token_path) ? File.read(token_path).strip : nil
      rescue Errno::EACCES
        nil
      end

      def truthy_environment?(value)
        %w[1 ON YES TRUE].include?(value.to_s.upcase)
      end

      def cache_repo_name(repo_id)
        "models--#{repo_id.gsub("/", "--")}"
      end

      def validate_repo_id!(repo_id)
        valid = if repo_id.is_a?(String) && repo_id.length.between?(1, 96) && repo_id.count("/") <= 1 &&
                   !repo_id.include?("..") && !repo_id.include?("--") && !repo_id.end_with?(".git")
                  repo_id.split("/").all? do |part|
                    part.match?(/\A[[:alnum:]_](?:[[:alnum:]_.-]*[[:alnum:]_])?\z/)
                  end
                else
                  false
                end
        raise ArgumentError, "Invalid Hugging Face model repository ID: #{repo_id.inspect}" unless valid
      end

      def validate_filename!(filename)
        raise ArgumentError, "Invalid Hub filename: #{filename.inspect}" unless filename.is_a?(String)

        path = Pathname(filename)
        safe_components = path.each_filename.none? do |part|
          part == "." || part == ".." || part.empty?
        end
        valid = !filename.empty? && !filename.include?("\0") && !path.absolute? &&
                !filename.start_with?("/") && !filename.end_with?("/") &&
                safe_components
        raise ArgumentError, "Invalid Hub filename: #{filename.inspect}" unless valid
      end

      def validate_revision!(revision)
        safe_components = revision.is_a?(String) && Pathname(revision).each_filename.none? do |part|
          part == "." || part == ".." || part.empty?
        end
        valid = revision.is_a?(String) && !revision.empty? && !revision.include?("\0") &&
                revision == revision.strip && !Pathname(revision).absolute? &&
                !revision.start_with?("/") && !revision.end_with?("/") &&
                safe_components
        raise ArgumentError, "Invalid Hugging Face revision: #{revision.inspect}" unless valid
      end

      def revision_path(repository, revision)
        validate_revision!(revision)
        refs = repository.join("refs").expand_path
        path = refs.join(revision).expand_path
        prefix = "#{refs}#{File::SEPARATOR}"
        raise ArgumentError, "Invalid Hugging Face revision: #{revision.inspect}" unless path.to_s.start_with?(prefix)

        path
      end
    end
  end
end
