#pragma once

#include "core/audio_chunking.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace cohere_chunking {

// The processor enters its chunk-metadata path above 30 seconds, but each
// model row may contain up to 35 seconds. Boundaries are selected from the
// final five seconds in non-overlapping 100 ms energy windows.
constexpr int processor_direct_seconds = 30;
constexpr int max_clip_seconds = 35;
constexpr int boundary_search_seconds = 5;
constexpr int energy_window_ms = 100;

inline std::vector<std::pair<size_t, size_t>> plan(const float* samples, size_t n_samples, int sample_rate) {
    if (!samples || n_samples == 0 || sample_rate <= 0)
        return {};
    const size_t max_clip = (size_t)max_clip_seconds * (size_t)sample_rate;
    const size_t search = (size_t)boundary_search_seconds * (size_t)sample_rate;
    const size_t energy_window = std::max<size_t>(1, (size_t)sample_rate * energy_window_ms / 1000);
    return audio_chunking::split_at_energy_minima(samples, n_samples, max_clip, search, energy_window);
}

inline bool is_ascii_space(unsigned char value) {
    return std::isspace(value) != 0;
}

inline std::string trim_right(std::string_view value) {
    size_t end = value.size();
    while (end > 0 && is_ascii_space((unsigned char)value[end - 1]))
        --end;
    return std::string(value.substr(0, end));
}

inline std::string trim(std::string_view value) {
    size_t begin = 0;
    while (begin < value.size() && is_ascii_space((unsigned char)value[begin]))
        ++begin;
    return trim_right(value.substr(begin));
}

// Match CohereAsrProcessor._reconstruct_transcripts(): discard empty rows,
// preserve leading whitespace on the first row, strip later rows, and join
// them with exactly one Arabic word separator.
inline void append_text(std::string& combined, std::string_view next) {
    if (trim(next).empty())
        return;
    std::string part = combined.empty() ? trim_right(next) : trim(next);
    if (part.empty())
        return;
    if (!combined.empty())
        combined.push_back(' ');
    combined += part;
}

} // namespace cohere_chunking
