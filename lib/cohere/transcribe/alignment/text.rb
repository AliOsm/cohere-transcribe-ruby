# frozen_string_literal: true

require_relative "../python_text"
require_relative "uroman_data"

module Cohere
  module Transcribe
    module Alignment
      # Arabic/English normalization used by the pinned MMS aligner. The
      # punctuation inventory mirrors the retained Python reference, including
      # its corpus-derived private-use and invalid-codepoint entries.
      module Text
        mms_punctuation_hex = <<~HEX
          0008 001A 0020 0023-0026 0028-002B 002F 0033 0035 0037 003D
          0040-0043 0045-0049 004B-004D 0051-0055 0057 005B 005D-005F
          007C 007E 0081 008D 008F-0090 0092-0094 009D 00A3 00A9 00AC
          00AF-00B0 00B6-00B7 00BC-00BD 00D7 02C7 02DC 037E 0387 05BE
          061E 06D4 06D6-06D7 06DA 0965 09F7 0E46 0EC6 0F04-0F05
          0F0C-0F0E 104A-104F 109F 1361-1366 166D-166E 17D4-17D7 17DA
          2006 2015 2020-2022 2026 202F-2030 2038 203B 2044 2060 206F
          2071 207F 20AC 2153 215F 2192 2212 226A-226B 230A-230B
          231E-231F 2D70 3005 3010-3011 3014-3015 30FB-30FC A4FE-A4FF
          A92E-A92F E003 E013 E01D E040-E041 E043 E06B E06D E070 E203
          E231-E234 E236 E290-E296 E2D9 E2DB E2DE E2F9-E2FC E2FE E313
          E328-E32E E492-E493 E514 E516-E518 EA01 EA7B F171-F173 F218
          F21D F50E-F50F F511 F513 F518-F519 F521-F523 F529 F52D F62D
          F7D0 F7E8 FD3E-FD3F FE50 FE55-FE57 FF0E FF1C FF1E FF65 FFFD
        HEX
        MMS_PUNCTUATION_HEX = mms_punctuation_hex.tr("\n", " ").freeze

        PREFIX_CODEPOINTS = [
          *".?,:!{}“”«»‹›„‚‟‛’‘><¿。;՚՛՜՝՞՟։¡،、।\"؛؟，！？；：（）﹏⋯、‧／～─＿".codepoints,
          *(0x300C..0x300F), *(0xFF41..0xFF44), *(0x3008..0x300B)
        ].freeze

        PUNCTUATION_CODEPOINTS = begin
          values = MMS_PUNCTUATION_HEX.split.flat_map do |token|
            first, last = token.split("-", 2).map { |hex| Integer(hex, 16) }
            last ? (first..last).to_a : [first]
          end
          (values + PREFIX_CODEPOINTS).to_h { |codepoint| [codepoint, true] }.freeze
        end

        DELETION_CODEPOINTS = [
          0x200E, 0x200C, *(0x0656..0x0657), 0x200B, *(0x064B..0x0652),
          0x202C, 0x200F, 0x202A
        ].to_h { |codepoint| [codepoint, true] }.freeze

        ARABIC_ROMANIZATION = {
          "ء" => "'", "آ" => "a", "أ" => "a", "ؤ" => "w", "إ" => "i", "ئ" => "ye",
          "ا" => "a", "ب" => "b", "ة" => "a", "ت" => "t", "ث" => "th", "ج" => "j",
          "ح" => "h", "خ" => "kh", "د" => "d", "ذ" => "th", "ر" => "r", "ز" => "z",
          "س" => "s", "ش" => "sh", "ص" => "s", "ض" => "d", "ط" => "t", "ظ" => "z",
          "ع" => "'", "غ" => "gh", "ف" => "f", "ق" => "q", "ك" => "k", "ل" => "l",
          "م" => "m", "ن" => "n", "ه" => "h", "و" => "w", "ى" => "a", "ي" => "y",
          "ٮ" => "b", "ٯ" => "q", "ٹ" => "tt", "ٺ" => "tt", "ٻ" => "b", "ټ" => "t",
          "ٽ" => "t", "پ" => "p", "ٿ" => "t", "ڀ" => "b", "ځ" => "z", "ڂ" => "h",
          "ڃ" => "ny", "ڄ" => "dy", "څ" => "ts", "چ" => "tch", "ڇ" => "tch",
          "ڈ" => "dd", "ډ" => "d", "ڊ" => "d", "ڋ" => "d", "ڌ" => "d", "ڍ" => "dd",
          "ڎ" => "d", "ڏ" => "d", "ڐ" => "d", "ڑ" => "rr", "ژ" => "j", "ښ" => "kh",
          "ڤ" => "v", "ک" => "k", "ڪ" => "k", "ګ" => "g", "ڭ" => "ng", "گ" => "g",
          "ں" => "n", "ھ" => "h", "ہ" => "h", "ۃ" => "a", "ۇ" => "u", "ۈ" => "yu",
          "ۋ" => "v", "ی" => "i", "ۍ" => "y", "ێ" => "y", "ې" => "e", "ے" => "y",
          "ە" => "ae", "ٴ" => "h"
        }.freeze

        LATIN_ROMANIZATION = {
          "æ" => "ae", "œ" => "oe", "ø" => "oe", "ß" => "ss", "ð" => "d", "þ" => "th",
          "ł" => "l", "đ" => "d", "ħ" => "h", "ı" => "i", "ə" => "e"
        }.freeze

        # Multi-codepoint rules from Uroman 1.3.1.1 that are reachable from the
        # pinned ASR tokenizers. Uroman chooses these before its single-character
        # candidates; keeping them here preserves that longest-match behavior.
        UROMAN_SEQUENCE_ROMANIZATION = {
          "λμπ" => "lb", "νμπ" => "nb", "ρμπ" => "rb",
          "γγ" => "ng", "ει" => "ei", "ευ" => "eu", "αυ" => "au", "ου" => "ou",
          "ηυ" => "eu", "υι" => "ui", "ωυ" => "ou", "ντ" => "nd",
          "シャ" => "sha", "シュ" => "shu", "ショ" => "sho",
          "チャ" => "cha", "チェ" => "che", "チュ" => "chu", "チョ" => "cho",
          "ジャ" => "ja", "ジュ" => "ju", "ジョ" => "jo", "ジェ" => "je",
          "ヂャ" => "ja", "ヂュ" => "ju", "ヂョ" => "jo",
          "フェ" => "fe", "ヴェ" => "ve", "フィ" => "fi", "ウィ" => "wi",
          "ヴィ" => "vi", "ティ" => "ti", "ディ" => "di",
          "しゃ" => "sha", "しゅ" => "shu", "しょ" => "sho",
          "ちゃ" => "cha", "ちゅ" => "chu", "ちょ" => "cho",
          "じゃ" => "ja", "じゅ" => "ju", "じょ" => "jo",
          "ぢゃ" => "ja", "ぢゅ" => "ju", "ぢょ" => "jo"
        }.freeze
        UROMAN_SEQUENCE_LENGTHS = UROMAN_SEQUENCE_ROMANIZATION.keys.map(&:length).uniq.sort.reverse.freeze
        JAPANESE_SMALL_Y = "ゃゅょャュョ".chars.freeze
        JAPANESE_SMALL_TSU = "っッ".chars.freeze
        JAPANESE_SCRIPT = {
          "ゃ" => :hiragana, "ゅ" => :hiragana, "ょ" => :hiragana,
          "ャ" => :katakana, "ュ" => :katakana, "ョ" => :katakana
        }.freeze

        DIGIT_PATTERN = "0-9০-৯០-៩०-९୦-୯۰-۹꤀-꤉０-９൦-൯၀-၉ⅰ-ⅹ⁯"

        module_function

        def normalize(text)
          normalized = PythonText.nfkc_lower(text.to_s)
          normalized = normalized.gsub(/\([^)]*\p{Nd}[^)]*\)/, " ")
          normalized = normalized.gsub("&lt;", "").gsub("&gt;", "").gsub("&nbsp", "")
          normalized = normalized.gsub(/(\S+)[\u201B\u2019\u2018](\S+)/, "\\1'\\2")
          normalized = normalized.tr("ٱٰۥۦ", "ااوي")
          normalized = normalized.delete("ـٓ")
          normalized = normalized.tr("ٕٔ", "ءء")
          normalized = normalized.each_char.map do |character|
            PUNCTUATION_CODEPOINTS.key?(character.ord) ? " " : character
          end.join
          normalized = normalized.each_char.reject do |character|
            DELETION_CODEPOINTS.key?(character.ord)
          end.join
          whitespace = PythonText.whitespace_class
          normalized = normalized.gsub(/\A[#{DIGIT_PATTERN}]+(?=#{whitespace})/, " ")
          normalized = normalized.gsub(/(?<=#{whitespace})[#{DIGIT_PATTERN}]+(?=#{whitespace}|\z)/, " ")
          PythonText.collapse(normalized)
        end

        def romanize(text, language)
          normalized = normalize(text)
          transliterated = romanize_normalized(normalized, language)
          spaced = PythonText.strip(transliterated).each_char.to_a.join(" ").downcase
          PythonText.strip(spaced.gsub(/[^a-z' ]/, " ").gsub(/ +/, " "))
        end

        def preprocess(text, language)
          words = PythonText.split(text.to_s)
          tokens = words.map { |word| romanize(word, language) }
          token_stream = []
          text_stream = []
          tokens.zip(words).each do |token, word|
            token_stream.push("<star>", token)
            text_stream.push("<star>", word)
          end
          [token_stream.freeze, text_stream.freeze]
        end

        def postprocess(text_starred, spans, stride_ms)
          results = []
          text_starred.each_with_index do |text, index|
            next if text == "<star>"

            span = spans.fetch(index)
            results << {
              start: span.first.start * stride_ms / 1_000.0,
              end: (span.last.end + 1) * stride_ms / 1_000.0,
              text: text
            }
          end
          results.each_cons(2) do |current, following|
            following[:start] = current[:end] if following[:start] < current[:end]
          end
          results.freeze
        end

        def romanize_character(character, _language)
          mapped = UROMAN_SINGLE_CODEPOINT[character.ord]
          return mapped unless mapped.nil?

          return character if /[a-z']/i.match?(character)
          return ARABIC_ROMANIZATION.fetch(character) if ARABIC_ROMANIZATION.key?(character)
          return LATIN_ROMANIZATION.fetch(character) if LATIN_ROMANIZATION.key?(character)

          decomposed = character.unicode_normalize(:nfkd).gsub(/\p{Mn}/, "")
          decomposed.match?(/\A[a-z]+\z/i) ? decomposed : ""
        end
        private_class_method :romanize_character

        def romanize_normalized(normalized, language)
          characters = normalized.each_char.to_a
          output = +""
          index = 0
          while index < characters.length
            if JAPANESE_SMALL_TSU.include?(characters[index]) && index + 1 < characters.length
              following, = romanize_unit(characters, index + 1, language)
              consonant = following.match(/\A(ch|[bcdfghjklmnpqrstwz])/)&.[](1)
              if consonant
                output << (consonant == "ch" ? "t" : consonant)
                index += 1
                next
              end
            end

            romanized, consumed = romanize_unit(characters, index, language)
            output << romanized
            index += consumed
          end
          output
        end
        private_class_method :romanize_normalized

        def romanize_unit(characters, index, language)
          UROMAN_SEQUENCE_LENGTHS.each do |length|
            sequence = characters[index, length]&.join
            mapped = UROMAN_SEQUENCE_ROMANIZATION[sequence]
            return [mapped, length] unless mapped.nil?
          end

          pair = characters[index, 2]&.join
          if pair == "μπ"
            return [word_start?(characters, index) ? "b" : "mb", 2]
          elsif pair == "γκ"
            return [word_start?(characters, index) ? "g" : "ng", 2]
          end

          character = characters.fetch(index)
          mapped = romanize_character(character, language)
          following = characters[index + 1]
          if JAPANESE_SMALL_Y.include?(following) && same_japanese_script?(character, following) &&
             mapped.match?(/[bcdfghjklmnpqrstvwxyz]i\z/)
            return [mapped.delete_suffix("i") + romanize_character(following, language), 2]
          end

          [mapped, 1]
        end
        private_class_method :romanize_unit

        def word_start?(characters, index)
          index.zero? || !characters[index - 1].match?(/\p{L}/)
        end
        private_class_method :word_start?

        def same_japanese_script?(character, following)
          script = JAPANESE_SCRIPT[following]
          return false if script.nil?

          codepoint = character.ord
          script == :hiragana ? codepoint.between?(0x3040, 0x309F) : codepoint.between?(0x30A0, 0x30FF)
        end
        private_class_method :same_japanese_script?
      end
    end
  end
end
