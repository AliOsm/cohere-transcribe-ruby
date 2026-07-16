# frozen_string_literal: true

require "fileutils"
require "rbconfig"
require "rubygems/version"
require "shellwords"
require "open3"

EXTENSION_DIR = File.expand_path(__dir__)
GEM_ROOT = File.expand_path("../..", EXTENSION_DIR)

def enabled?(name)
  value = ENV.fetch(name, nil)
  value && !value.empty? && !%w[0 false no off].include?(value.downcase)
end

host_os = RbConfig::CONFIG.fetch("host_os")
abort "cohere-transcribe's vendored native runtime currently supports Linux and macOS source builds" unless host_os.match?(/linux|darwin/)

cuda = enabled?("COHERE_TRANSCRIBE_CUDA")
metal = enabled?("COHERE_TRANSCRIBE_METAL")
abort "COHERE_TRANSCRIBE_METAL=1 is only supported on macOS" if metal && !host_os.match?(/darwin/)
abort "COHERE_TRANSCRIBE_CUDA=1 is only supported on Linux" if cuda && !host_os.match?(/linux/)
abort "enable either CUDA or Metal, not both" if cuda && metal

cmake = ENV.fetch("CMAKE", "cmake")
begin
  cmake_output, cmake_status = Open3.capture2e(cmake, "--version")
rescue SystemCallError
  cmake_output = nil
  cmake_status = nil
end
cmake_version_match = cmake_output&.match(/cmake version (\d+(?:\.\d+)+)/i)
cmake_version = cmake_version_match && cmake_version_match[1]
unless cmake_status&.success? && cmake_version && Gem::Version.new(cmake_version) >= Gem::Version.new("3.15")
  abort "cmake 3.15 or newer is required to build cohere-transcribe's native runtime"
end

build_dir = File.expand_path(
  ENV.fetch("COHERE_TRANSCRIBE_NATIVE_BUILD_DIR", File.join(GEM_ROOT, "tmp", "cohere_transcribe_native"))
)
output_dir = File.expand_path(
  ENV.fetch(
    "COHERE_TRANSCRIBE_NATIVE_OUTPUT",
    File.join(GEM_ROOT, "lib", "cohere", "transcribe", "native")
  )
)
build_type = ENV.fetch("COHERE_TRANSCRIBE_BUILD_TYPE", "Release")

cmake_options = {
  "COHERE_TRANSCRIBE_CUDA" => cuda,
  "COHERE_TRANSCRIBE_METAL" => metal,
  "COHERE_TRANSCRIBE_NATIVE" => enabled?("COHERE_TRANSCRIBE_NATIVE"),
  "COHERE_TRANSCRIBE_OPENMP" => enabled?("COHERE_TRANSCRIBE_OPENMP")
}

configure = [
  cmake,
  "-S", EXTENSION_DIR,
  "-B", build_dir,
  "-DCMAKE_BUILD_TYPE=#{build_type}",
  *cmake_options.map { |name, value| "-D#{name}=#{value ? "ON" : "OFF"}" },
  *Shellwords.split(ENV.fetch("COHERE_TRANSCRIBE_CMAKE_ARGS", ""))
]
build = [cmake, "--build", build_dir, "--config", build_type, "--target", "crispasr", "cohere_audio"]
jobs = ENV.fetch("COHERE_TRANSCRIBE_NATIVE_JOBS", nil)
build.push("--parallel", Integer(jobs, 10).to_s) if jobs && !jobs.empty?
install = [
  cmake,
  "--install", build_dir,
  "--config", build_type,
  "--prefix", output_dir,
  "--component", "cohere_transcribe_runtime"
]
smoke = [RbConfig.ruby, File.join(EXTENSION_DIR, "test", "abi_smoke.rb"), output_dir]

FileUtils.mkdir_p(build_dir)

makefile = <<~MAKEFILE
  .PHONY: all install check clean distclean

  all:
  \t#{Shellwords.join(configure)}
  \t+#{Shellwords.join(build)}
  \t#{Shellwords.join(install)}

  install: all

  check: all
  \t#{Shellwords.join(smoke)}

  clean:
  \t#{Shellwords.join([cmake, "-E", "remove_directory", build_dir])}

  distclean: clean
MAKEFILE

File.write(File.join(EXTENSION_DIR, "Makefile"), makefile)
puts "Native runtime will be installed in #{output_dir}"
