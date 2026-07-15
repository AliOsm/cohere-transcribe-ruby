#pragma once

#include <cstddef>

namespace cohere_decoder_batch {

constexpr int max_batch = 24;
constexpr int max_encoder_frames = 512;
constexpr int max_teacher_steps = 64;

constexpr std::size_t cache_slot(int layer, int lane, int n_batch) {
    return (std::size_t)layer * (std::size_t)n_batch + (std::size_t)lane;
}

constexpr std::size_t cache_element_offset(int head_dim, int max_ctx, int n_heads, int layer, int lane,
                                           int n_batch, int position, int head, int component) {
    const std::size_t slot_stride = (std::size_t)head_dim * (std::size_t)max_ctx * (std::size_t)n_heads;
    const std::size_t head_stride = (std::size_t)head_dim * (std::size_t)max_ctx;
    return cache_slot(layer, lane, n_batch) * slot_stride + (std::size_t)head * head_stride +
           (std::size_t)position * (std::size_t)head_dim + (std::size_t)component;
}

constexpr std::size_t graph_token_index(int lane, int token, int n_tokens) {
    return (std::size_t)lane * (std::size_t)n_tokens + (std::size_t)token;
}

constexpr std::size_t sequence_token_index(int step, int lane, int n_batch) {
    return (std::size_t)step * (std::size_t)n_batch + (std::size_t)lane;
}

constexpr std::size_t output_logit_index(int output, int lane, int token, int n_batch, int vocab_size) {
    return ((std::size_t)output * (std::size_t)n_batch + (std::size_t)lane) * (std::size_t)vocab_size +
           (std::size_t)token;
}

constexpr bool valid_cross_length(int valid_T_enc, int T_enc_max) {
    return T_enc_max >= 1 && T_enc_max <= max_encoder_frames && valid_T_enc >= 1 && valid_T_enc <= T_enc_max;
}

// Host full-logit generation intentionally uses strict greater-than so exact
// ties resolve to the lowest vocabulary ID. Device argmax kernels are not
// required to preserve this tie policy.
inline int host_argmax_lowest(const float* values, int count) {
    if (!values || count < 1)
        return -1;
    int best = 0;
    for (int index = 1; index < count; ++index) {
        if (values[index] > values[best])
            best = index;
    }
    return best;
}

// GGML flash-attention mask layout: [T_enc_max, n_queries, 1, n_batch].
constexpr std::size_t cross_mask_index(int lane, int query, int key, int n_queries, int T_enc_max) {
    return ((std::size_t)lane * (std::size_t)n_queries + (std::size_t)query) * (std::size_t)T_enc_max +
           (std::size_t)key;
}

} // namespace cohere_decoder_batch
