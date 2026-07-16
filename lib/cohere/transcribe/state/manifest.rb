# frozen_string_literal: true

require "digest"

module Cohere
  module Transcribe
    module State
      PublicationVerification = Data.define(:verified, :generation_id, :reason) do
        def verified?
          verified
        end
      end

      module_function

      def published_manifest_content(source_snapshot:, output_paths:, contents:, asr_contract_key:,
                                     render_contract_key:, generation_id:)
        payload = published_payload(
          source_snapshot: source_snapshot,
          output_paths: output_paths,
          contents: contents,
          asr_contract_key: asr_contract_key,
          render_contract_key: render_contract_key,
          generation_id: generation_id
        )
        JSON.pretty_generate(envelope(payload)) << "\n"
      end

      def published_payload(source_snapshot:, output_paths:, contents:, asr_contract_key:,
                            render_contract_key:, generation_id:)
        raise ArgumentError, "generation ID is invalid" unless generation_id.is_a?(String) && !generation_id.empty?

        {
          "kind" => "published",
          "generation_id" => generation_id,
          "asr_contract_key" => asr_contract_key,
          "render_contract_key" => render_contract_key,
          "source" => source_snapshot.payload,
          "updated_unix_seconds" => Time.now.to_f,
          "outputs" => output_paths.keys.sort.to_h do |format|
            content = contents.fetch(format)
            [
              format,
              {
                "name" => Pathname(output_paths.fetch(format)).basename.to_s,
                "size" => content.bytesize,
                "sha256" => Digest::SHA256.hexdigest(content)
              }
            ]
          end
        }
      end

      def verify_published_outputs(source_snapshot:, output_paths:, state_path:, asr_contract_key:,
                                   render_contract_key:, directory_binding: nil,
                                   guard_bindings: nil)
        payload, reason = decode_state(
          state_path,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        )
        return PublicationVerification.new(verified: false, generation_id: nil, reason: reason.freeze) unless payload

        mismatch = publication_mismatch_reason(
          payload,
          source_snapshot: source_snapshot,
          output_paths: output_paths,
          asr_contract_key: asr_contract_key,
          render_contract_key: render_contract_key,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        )
        return PublicationVerification.new(verified: false, generation_id: nil, reason: mismatch.freeze) if mismatch

        verify_publication_bindings!(directory_binding, guard_bindings)
        PublicationVerification.new(
          verified: true,
          generation_id: payload.fetch("generation_id").dup.freeze,
          reason: nil
        )
      rescue PublicationDirectoryChangedError => e
        PublicationVerification.new(
          verified: false,
          generation_id: nil,
          reason: "publication parent changed during verification (#{e.message})".freeze
        )
      rescue EncodingError, SystemCallError, TypeError, ArgumentError => e
        PublicationVerification.new(
          verified: false,
          generation_id: nil,
          reason: "cannot verify output generation (#{e.class}: #{e.message})".freeze
        )
      end

      def publication_mismatch_reason(payload, source_snapshot:, output_paths:, asr_contract_key:,
                                      render_contract_key:, directory_binding: nil,
                                      guard_bindings: nil)
        return "state is #{payload["kind"].inspect}, not published" unless payload["kind"] == "published"
        return "state marker ASR contract does not match" unless payload["asr_contract_key"] == asr_contract_key
        return "state marker render contract does not match" unless payload["render_contract_key"] == render_contract_key
        return "state marker source snapshot does not match" unless payload["source"] == source_snapshot.payload

        generation_id = payload["generation_id"]
        return "state marker generation ID is invalid" unless generation_id.is_a?(String) && !generation_id.empty?

        records = payload["outputs"]
        return "state marker output formats do not match" unless records.is_a?(Hash) && records.keys.sort == output_paths.keys.sort

        output_paths.each do |format, path_value|
          path = Pathname(path_value)
          record = records[format]
          return "state marker path for #{format} does not match" unless record.is_a?(Hash) && record["name"] == path.basename.to_s

          begin
            output_record = if directory_binding
                              bound_output_record(
                                path,
                                directory_binding: directory_binding,
                                guard_bindings: guard_bindings
                              )
                            else
                              stat = path.lstat
                              return "#{format} output is missing or not regular" unless stat.file? && !stat.symlink?

                              [stat.size, Digest::SHA256.file(path).hexdigest]
                            end
            return "#{format} output changed or is not regular" unless output_record

            size, sha256 = output_record
            return "#{format} output does not match its state marker" if record["size"] != size || record["sha256"] != sha256
          rescue Errno::ENOENT
            return "#{format} output is missing or not regular"
          end
        end
        nil
      end

      def bound_output_record(path, directory_binding:, guard_bindings:)
        with_bound_parent(
          path,
          directory_binding: directory_binding,
          guard_bindings: guard_bindings
        ) do |bound, basename|
          bound.verify!
          handle = nil
          begin
            Thread.handle_interrupt(DEFERRED_PUBLICATION_EXCEPTIONS) do
              handle = bound.open_regular(basename)
            end
          rescue PublicationEntryError
            bound.verify!
            return nil
          end
          opened = handle.stat
          digest = Digest::SHA256.new
          digest << handle.read(1024 * 1024) until handle.eof?
          unchanged = bound.same_regular_entry?(basename, opened)
          unless unchanged
            bound.verify!
            return nil
          end

          bound.verify!
          [opened.size, digest.hexdigest]
        ensure
          handle&.close
        end
      end
      private_class_method :bound_output_record

      def verify_publication_bindings!(directory_binding, guard_bindings)
        (Array(guard_bindings).compact + [directory_binding].compact).uniq.each(&:verify!)
      end
      private_class_method :verify_publication_bindings!
    end
  end
end
