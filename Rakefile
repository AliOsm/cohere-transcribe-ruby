# frozen_string_literal: true

require "bundler/gem_tasks"
require "minitest/test_task"

Minitest::TestTask.create

require "rubocop/rake_task"

RuboCop::RakeTask.new

desc "Validate the packaged RBS signatures"
task :rbs do
  sh "bundle", "exec", "rbs", "validate"
end

task default: %i[test rubocop rbs]
