# frozen_string_literal: true

require "open3"
require "fileutils"
require "tmpdir"
require "test_helper"

class Cohere::Transcribe::NativeBuildConfigurationTest < Minitest::Test
  def test_cmake_314_is_rejected_before_a_native_makefile_is_generated
    Dir.mktmpdir("cohere-old-cmake") do |directory|
      cmake = File.join(directory, "cmake")
      File.write(cmake, "#!/bin/sh\nprintf '%s\\n' 'cmake version 3.14.9'\n")
      File.chmod(0o755, cmake)

      _output, error, status = Open3.capture3(
        { "CMAKE" => cmake },
        "ruby",
        "ext/cohere_transcribe_native/extconf.rb",
        chdir: project_root
      )

      refute_predicate status, :success?
      assert_includes error, "cmake 3.15 or newer is required"
    end
  end

  def test_cmake_project_declares_the_same_minimum
    declaration = File.foreach(
      File.join(project_root, "ext/cohere_transcribe_native/CMakeLists.txt")
    ).first

    assert_equal "cmake_minimum_required(VERSION 3.15...3.31)\n", declaration
  end

  def test_cuda_install_keeps_the_discovered_toolkit_library_path
    cmake = File.read(
      File.join(project_root, "ext/cohere_transcribe_native/CMakeLists.txt")
    )

    assert_includes cmake, 'BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"'
    assert_includes cmake, 'INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}"'
  end

  def test_gemspec_omits_untracked_native_build_artifacts
    skip "repository metadata is unavailable" unless File.exist?(File.join(project_root, ".git"))

    artifact = File.join(
      project_root, "ext/cohere_transcribe_native/build/untracked/native-object.o"
    )
    FileUtils.mkdir_p(File.dirname(artifact))
    File.binwrite(artifact, "not packaged")

    specification = Gem::Specification.load(File.join(project_root, "cohere-transcribe.gemspec"))

    refute_includes specification.files,
                    "ext/cohere_transcribe_native/build/untracked/native-object.o"
  ensure
    FileUtils.rm_f(artifact) if defined?(artifact)
    FileUtils.rm_rf(File.dirname(artifact)) if defined?(artifact)
    build_directory = File.join(project_root, "ext/cohere_transcribe_native/build")
    FileUtils.rm_rf(build_directory) if File.directory?(build_directory) && Dir.empty?(build_directory)
  end

  def test_gemspec_fallback_omits_every_native_staging_directory
    Dir.mktmpdir("cohere-gemspec-fallback") do |directory|
      version_directory = File.join(directory, "lib/cohere/transcribe")
      extension_directory = File.join(directory, "ext/cohere_transcribe_native")
      FileUtils.mkdir_p(version_directory)
      FileUtils.mkdir_p(extension_directory)
      FileUtils.cp(File.join(project_root, "cohere-transcribe.gemspec"), directory)
      File.binwrite(File.join(version_directory, "version.rb"), "# test fixture\n")
      File.binwrite(File.join(extension_directory, "extconf.rb"), "fixture")
      artifact_paths = %w[build cuda-build stage cuda-stage].map do |name|
        File.join(extension_directory, name, "native-object.o").tap do |path|
          FileUtils.mkdir_p(File.dirname(path))
          File.binwrite(path, "not packaged")
        end
      end

      specification = Gem::Specification.load(File.join(directory, "cohere-transcribe.gemspec"))

      assert_includes specification.files, "ext/cohere_transcribe_native/extconf.rb"
      artifact_paths.each do |path|
        refute_includes specification.files, Pathname(path).relative_path_from(Pathname(directory)).to_s
      end
    end
  end

  private

  def project_root
    File.expand_path("../..", __dir__)
  end
end
