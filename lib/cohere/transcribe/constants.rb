# frozen_string_literal: true

module Cohere
  module Transcribe
    DEFAULT_ASR_MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
    DEFAULT_ASR_MODEL_REVISION = "0a8193caa4f3f92131471ab08824e488141cb392"
    SAMPLE_RATE = 16_000
    AUDIO_EXTENSIONS = %w[
      .aac .aif .aiff .alac .flac .m4a .mp3 .mp4 .oga .ogg .opus .wav .wave
      .webm .wma
    ].freeze
    OUTPUT_FORMATS = %w[txt srt vtt json].freeze
  end
end
