#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace cohere_token_renderer {

constexpr int speaker_token_count = 16;

struct speaker_token_ids {
    int change = -1;
    int first_speaker = -1;
};

struct token_record {
    int id = -1;
    float probability = 0.0f;
    int generated_index = -1;
    std::string text;
};

enum class render_error {
    none,
    probability_count_mismatch,
};

struct render_result {
    render_error error = render_error::none;
    std::string full_text;
    std::vector<token_record> tokens;

    bool ok() const {
        return error == render_error::none;
    }
};

namespace detail {

inline int hex_nibble(char value) {
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'A' && value <= 'F')
        return 10 + value - 'A';
    if (value >= 'a' && value <= 'f')
        return 10 + value - 'a';
    return -1;
}

inline bool byte_fallback_value(const std::string& piece, unsigned char& byte) {
    if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>')
        return false;
    const int high = hex_nibble(piece[3]);
    const int low = hex_nibble(piece[4]);
    if (high < 0 || low < 0)
        return false;
    byte = (unsigned char)((high << 4) | low);
    return true;
}

inline int utf8_expected_length(unsigned char lead) {
    if ((lead & 0x80) == 0)
        return 1;
    if ((lead & 0xE0) == 0xC0)
        return 2;
    if ((lead & 0xF0) == 0xE0)
        return 3;
    if ((lead & 0xF8) == 0xF0)
        return 4;
    return 0;
}

inline bool is_complete_utf8(const std::string& bytes) {
    if (bytes.empty())
        return false;
    const int expected = utf8_expected_length((unsigned char)bytes[0]);
    if (expected <= 0 || (int)bytes.size() != expected)
        return false;
    for (int index = 1; index < expected; ++index) {
        if ((((unsigned char)bytes[(size_t)index]) & 0xC0) != 0x80)
            return false;
    }
    return true;
}

inline bool is_speaker_id(int id, int first_speaker) {
    if (first_speaker < 0 || id < first_speaker)
        return false;
    return (long long)id - (long long)first_speaker < speaker_token_count;
}

} // namespace detail

// Decode generated Cohere IDs without depending on a model context. Byte
// fallback pieces are collapsed into one rendered record per complete UTF-8
// sequence. The record keeps the first byte's ID, the minimum probability,
// and the last byte's generated index, matching the production decoder.
inline render_result render(const std::vector<int>& generated_ids, const std::vector<float>& probabilities,
                            const std::vector<std::string>& vocabulary_pieces,
                            const speaker_token_ids& speaker_tokens) {
    render_result result;
    if (generated_ids.size() != probabilities.size()) {
        result.error = render_error::probability_count_mismatch;
        return result;
    }

    auto emit = [&](int id, float probability, int generated_index, const std::string& text) {
        result.tokens.push_back({id, probability, generated_index, text});
        result.full_text += text;
    };

    std::string pending_bytes;
    int pending_id = -1;
    int pending_generated_index = -1;
    float pending_probability = 1.0f;

    auto flush_pending = [&]() {
        if (pending_bytes.empty())
            return;
        if (detail::is_complete_utf8(pending_bytes))
            emit(pending_id, pending_probability, pending_generated_index, pending_bytes);
        pending_bytes.clear();
        pending_id = -1;
        pending_generated_index = -1;
        pending_probability = 1.0f;
    };

    for (size_t generated_index = 0; generated_index < generated_ids.size(); ++generated_index) {
        const int id = generated_ids[generated_index];
        if (id < 0 || (size_t)id >= vocabulary_pieces.size())
            continue;

        const std::string& piece = vocabulary_pieces[(size_t)id];
        unsigned char byte = 0;
        if (detail::byte_fallback_value(piece, byte)) {
            if (pending_bytes.empty()) {
                pending_id = id;
                pending_generated_index = (int)generated_index;
                pending_probability = probabilities[generated_index];
            } else {
                pending_probability = std::min(pending_probability, probabilities[generated_index]);
                pending_generated_index = (int)generated_index;
            }
            pending_bytes.push_back((char)byte);
            if (detail::is_complete_utf8(pending_bytes))
                flush_pending();
            continue;
        }

        flush_pending();
        if (piece.empty())
            continue;

        std::string text;
        if (piece.front() == '<' && piece.back() == '>') {
            if (id == speaker_tokens.change) {
                text = " [SPEAKER_TURN]";
            } else if (detail::is_speaker_id(id, speaker_tokens.first_speaker)) {
                text = " [Speaker " + std::to_string(id - speaker_tokens.first_speaker) + "]";
            } else {
                continue;
            }
        } else {
            text = piece;
            const std::string boundary = "\xE2\x96\x81";
            size_t position = 0;
            while ((position = text.find(boundary)) != std::string::npos)
                text.replace(position, boundary.size(), " ");
        }

        emit(id, probabilities[generated_index], (int)generated_index, text);
    }

    flush_pending();
    if (!result.full_text.empty() && result.full_text.front() == ' ')
        result.full_text.erase(result.full_text.begin());
    return result;
}

} // namespace cohere_token_renderer
