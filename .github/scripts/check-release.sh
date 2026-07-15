#!/bin/sh

set -eu

current_tag=${GITHUB_REF_NAME#v}
file_tag=$(ruby -Ilib -rcohere/transcribe/version -e 'print Cohere::Transcribe::VERSION')

if [ "$current_tag" != "$file_tag" ]; then
  echo "The release tag does not match the gem version."
  echo "$current_tag vs $file_tag"
  exit 1
fi

echo "Release tag v${file_tag} matches the gem version."
