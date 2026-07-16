# frozen_string_literal: true

require "test_helper"
require "tmpdir"

module Cohere
  module Transcribe
    class InputTest < Minitest::Test
      def test_normalize_accepts_one_path_and_ordered_sequences
        path_like = Object.new
        path_like.define_singleton_method(:to_path) { "second.mp3" }

        assert_equal ["first.wav"], Input.normalize("first.wav")
        assert_equal ["first.wav"], Input.normalize(Pathname("first.wav"))
        assert_equal ["first.wav", "second.mp3"], Input.normalize(["first.wav", path_like])

        sequence = Object.new
        sequence.define_singleton_method(:to_ary) { [Pathname("one.wav"), "two.ogg"] }
        assert_equal ["one.wav", "two.ogg"], Input.normalize(sequence)

        path_and_sequence = Object.new
        path_and_sequence.define_singleton_method(:to_path) { "path-wins.wav" }
        path_and_sequence.define_singleton_method(:to_ary) { raise "sequence protocol must not be called" }
        assert_equal ["path-wins.wav"], Input.normalize(path_and_sequence)
      end

      def test_normalize_returns_detached_frozen_text_and_array
        source = +"audio.wav"
        normalized = Input.normalize([source])
        source.replace("changed.wav")

        assert_equal ["audio.wav"], normalized
        assert normalized.frozen?
        assert normalized.first.frozen?
      end

      def test_binary_encoded_utf8_paths_are_preserved_as_utf8_text
        path = "/tmp/caf\xC3\xA9.wav".b
        normalized = Input.normalize(path)

        assert_equal path.bytes, normalized.first.bytes
        assert_equal Encoding::UTF_8, normalized.first.encoding
        assert normalized.first.valid_encoding?
      end

      def test_non_utf8_byte_paths_are_rejected_before_file_access
        path = "/tmp/caf\xE9.wav".b

        error = assert_raises(TranscriptionInputError) { Input.normalize(path) }

        assert_equal "audio path must contain valid UTF-8", error.message
      end

      def test_normalize_rejects_ambiguous_and_invalid_values
        ["", "   ", [], Set["audio.wav"], { audio: "audio.wav" }, 42, :audio, Object.new].each do |audio|
          assert_raises(TranscriptionInputError, audio.inspect) { Input.normalize(audio) }
        end

        [["audio.wav", ""], ["audio.wav", Object.new], ["audio.wav", :other]].each do |audio|
          assert_raises(TranscriptionInputError, audio.inspect) { Input.normalize(audio) }
        end
      end

      def test_normalize_uses_python_unicode_whitespace_semantics
        python_whitespace_codepoints.each do |codepoint|
          whitespace = codepoint.chr(Encoding::UTF_8)
          error = assert_raises(TranscriptionInputError, format("U+%04X", codepoint)) do
            Input.normalize(whitespace)
          end
          assert_equal "audio must not be empty", error.message
        end

        assert_equal ["\u200B"], Input.normalize("\u200B")
        assert_equal ["\uFEFF"], Input.normalize("\uFEFF")
      end

      def test_normalize_wraps_invalid_path_protocol_results
        bad_path = Object.new
        bad_path.define_singleton_method(:to_path) { 123 }
        error = assert_raises(TranscriptionInputError) { Input.normalize(bad_path) }
        assert_match(/resolve to a text path/, error.message)

        bad_sequence = Object.new
        bad_sequence.define_singleton_method(:to_ary) { raise TypeError, "broken sequence" }
        error = assert_raises(TranscriptionInputError) { Input.normalize(bad_sequence) }
        assert_match(/ordered path sequence/, error.message)
      end

      def test_directory_expansion_recurses_only_when_requested
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          root.join("nested").mkdir
          File.binwrite(root.join("direct.wav"), "direct")
          File.binwrite(root.join("nested/episode.mp3"), "nested")
          File.binwrite(root.join("nested/ignored.txt"), "ignored")

          direct = Input.expand(root, recursive: false)
          assert_equal(["direct.wav"], direct.map { |entry| entry.relative_path.to_s })

          recursive = Input.expand(root, recursive: true)
          assert_equal(["direct.wav", "nested/episode.mp3"], recursive.map { |entry| entry.relative_path.to_s })
          assert recursive.frozen?
          assert(recursive.all? { |entry| entry.path.absolute? && entry.path.frozen? && entry.relative_path.frozen? })
        end
      end

      def test_directory_expansion_accepts_all_extensions_case_insensitively
        Dir.mktmpdir do |directory|
          AUDIO_EXTENSIONS.sort.each_with_index do |extension, index|
            File.binwrite(File.join(directory, format("audio-%02d%s", index, extension.upcase)), "audio")
          end
          File.binwrite(File.join(directory, "audio.raw"), "ignored")

          entries = Input.expand(directory)
          assert_equal AUDIO_EXTENSIONS.length, entries.length
          assert_equal AUDIO_EXTENSIONS.sort, entries.map { |entry| entry.path.extname.downcase }.sort
        end
      end

      def test_unsupported_files_are_ignored_in_directories_but_allowed_explicitly
        Dir.mktmpdir do |directory|
          audio = File.join(directory, "clip.wav")
          metadata = File.join(directory, "clip.transcript")
          File.binwrite(audio, "audio")
          File.binwrite(metadata, "metadata")

          assert_equal(["clip.wav"], Input.expand(directory).map { |entry| entry.relative_path.to_s })
          assert_equal(["clip.transcript"], Input.expand(metadata).map { |entry| entry.relative_path.to_s })
        end
      end

      def test_expansion_preserves_group_order_and_casefolded_directory_order
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          explicit = root.join("outside.opus")
          collection = root.join("collection")
          collection.mkdir
          File.binwrite(explicit, "audio")
          ["z last.wav", "A first.mp3", "archive.tar.WAV", "مرحبا.WEBM", ".hidden.ogg"].each do |name|
            File.binwrite(collection.join(name), "audio")
          end

          entries = Input.expand([explicit, collection])
          assert_equal(
            ["outside.opus", ".hidden.ogg", "A first.mp3", "archive.tar.WAV", "z last.wav", "مرحبا.WEBM"],
            entries.map { |entry| entry.relative_path.to_s }
          )
        end
      end

      def test_directory_sort_uses_unicode_casefolding
        Dir.mktmpdir do |directory|
          File.binwrite(File.join(directory, "z.wav"), "audio")
          File.binwrite(File.join(directory, "ß.wav"), "audio")

          assert_equal(["ß.wav", "z.wav"], Input.expand(directory).map { |entry| entry.relative_path.to_s })
        end
      end

      def test_expansion_deduplicates_symlinks_by_canonical_path
        Dir.mktmpdir do |directory|
          root = Pathname(directory)
          source = root.join("real.wav")
          alias_path = root.join("00 alias.wav")
          File.binwrite(source, "audio")
          begin
            alias_path.make_symlink(source.basename)
          rescue NotImplementedError, Errno::EPERM
            skip "symbolic links are unavailable"
          end

          entries = Input.expand(root)
          assert_equal 1, entries.length
          assert_equal source.realpath, entries.first.path
          assert_equal "00 alias.wav", entries.first.relative_path.to_s

          entries = Input.expand([source, alias_path, root])
          assert_equal(["real.wav"], entries.map { |entry| entry.relative_path.to_s })
        end
      end

      def test_expansion_rejects_a_valid_alias_to_a_non_utf8_target
        Dir.mktmpdir do |directory|
          target = File.join(directory.b, "target-\xE9.wav".b)
          alias_path = File.join(directory, "alias.wav")
          begin
            File.binwrite(target, "audio")
            File.symlink(File.basename(target), alias_path)
          rescue NotImplementedError, Errno::EILSEQ, Errno::EPERM
            skip "this filesystem cannot create the fixture path"
          end

          error = assert_raises(TranscriptionInputError) { Input.expand(alias_path) }

          assert_match(/resolved input path must contain valid UTF-8/, error.message)
        end
      end

      def test_directory_expansion_rejects_non_utf8_audio_entries
        Dir.mktmpdir do |directory|
          path = File.join(directory.b, "clip-\xE9.wav".b)
          begin
            File.binwrite(path, "audio")
          rescue Errno::EILSEQ
            skip "this filesystem cannot create the fixture path"
          end

          error = assert_raises(TranscriptionInputError) { Input.expand(directory) }

          assert_match(/valid UTF-8/, error.message)
        end
      end

      def test_missing_broken_empty_and_invalid_paths_are_typed
        Dir.mktmpdir do |directory|
          missing = File.join(directory, "missing.wav")
          error = assert_raises(TranscriptionInputError) { Input.expand(missing) }
          assert_match(/does not exist/, error.message)

          broken = Pathname(directory).join("broken.wav")
          begin
            broken.make_symlink("absent.wav")
            error = assert_raises(TranscriptionInputError) { Input.expand(broken) }
            assert_match(/does not exist/, error.message)
          rescue NotImplementedError, Errno::EPERM
            # Other error cases still provide coverage where symlinks are absent.
          end

          empty = Pathname(directory).join("empty")
          empty.mkdir
          error = assert_raises(TranscriptionInputError) { Input.expand(empty) }
          assert_match(/No audio files found/, error.message)

          error = assert_raises(TranscriptionInputError) { Input.expand("invalid\0path") }
          assert_match(/Invalid input path/, error.message)
        end
      end

      def test_explicit_fifo_is_rejected_as_nonregular
        skip "FIFO creation is unavailable" unless File.respond_to?(:mkfifo) || system("command -v mkfifo >/dev/null 2>&1")

        Dir.mktmpdir do |directory|
          fifo = File.join(directory, "audio.wav")
          system("mkfifo", fifo, exception: true)
          error = assert_raises(TranscriptionInputError) { Input.expand(fifo) }
          assert_match(/not a regular file or directory/, error.message)
        end
      end

      private

      def python_whitespace_codepoints
        [
          *(0x0009..0x000D), *(0x001C..0x0020), 0x0085, 0x00A0, 0x1680,
          *(0x2000..0x200A), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000
        ]
      end
    end
  end
end
