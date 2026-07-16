# frozen_string_literal: true

require "open3"
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

  private

  def project_root
    File.expand_path("../..", __dir__)
  end
end
