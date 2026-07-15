#pragma once

#include <cstddef>

namespace cohere_encoder_padded {

constexpr int max_batch = 8;
constexpr int max_mel_frames = 4096;

constexpr int stride2_length(int length) {
    return (length + 1) / 2;
}

constexpr int encoder_length(int mel_length) {
    return stride2_length(stride2_length(stride2_length(mel_length)));
}

constexpr std::size_t lane_time_index(int lane, int time, int padded_time) {
    return (std::size_t)lane * (std::size_t)padded_time + (std::size_t)time;
}

constexpr std::size_t lane_value_index(int lane, int time, int feature, int padded_time, int n_features) {
    return ((std::size_t)lane * (std::size_t)padded_time + (std::size_t)time) * (std::size_t)n_features +
           (std::size_t)feature;
}

} // namespace cohere_encoder_padded
