# frozen_string_literal: true

require_relative "lib/cohere/transcribe/version"

Gem::Specification.new do |spec|
  spec.name = "cohere-transcribe"
  spec.version = Cohere::Transcribe::VERSION
  spec.authors = ["Ali Hamdi Ali Fadel"]
  spec.email = ["aliosm1997@gmail.com"]

  spec.summary = "Pure Ruby and native Cohere Arabic/English speech transcription"
  spec.description =
    "A Ruby 4 API and CLI for Dense Cohere ASR checkpoints, with native audio decoding, " \
    "Silero or energy VAD, MMS CTC word alignment, subtitle rendering, and a source-built " \
    "C/C++ inference backend."
  spec.homepage = "https://github.com/AliOsm/cohere-transcribe-ruby"
  spec.licenses = ["Apache-2.0", "BSD-2-Clause", "CC-BY-NC-4.0", "MIT", "LicenseRef-Uroman"]
  spec.required_ruby_version = ">= 4.0", "< 5.0"
  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "#{spec.homepage}/tree/main"
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"
  spec.metadata["allowed_push_host"] = "https://rubygems.org"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir.chdir(__dir__) do
    Dir[
      "CHANGELOG.md", "LICENSE.txt", "NOTICE", "README.md", "THIRD_PARTY_NOTICES.md",
      "exe/*", "ext/**/*", "lib/**/*", "licenses/**/*", "sig/**/*", "vendor/**/*"
    ].select { |path| File.file?(path) }
     .reject do |path|
       path == "ext/cohere_transcribe_native/Makefile" ||
         path.start_with?("lib/cohere/transcribe/native/") ||
         path.end_with?(".py", ".pyc")
     end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.extensions = ["ext/cohere_transcribe_native/extconf.rb"]
  spec.require_paths = ["lib"]

  spec.add_dependency "fiddle", ">= 1.1", "< 2.0"
  spec.add_dependency "numo-narray", ">= 0.9.2", "< 1.0"
  spec.add_dependency "onnxruntime", ">= 0.11", "< 1.0"
end
