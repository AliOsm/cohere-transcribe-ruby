#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace cohere_batch_planner {

// Round a graph dimension up to a stable bucket so repeated batches can reuse
// the same backend graph. A zero bucket preserves the exact input length.
// Zero is also the failure sentinel for invalid or overflowing dimensions.
inline int round_up_to_bucket(int length, int bucket, int maximum) {
    if (length <= 0 || bucket < 0 || maximum <= 0 || length > maximum || bucket > maximum)
        return 0;
    if (bucket == 0)
        return length;
    const std::int64_t rounded = ((std::int64_t)length + bucket - 1) / bucket * bucket;
    return rounded <= maximum ? (int)rounded : 0;
}

struct batch {
    std::vector<std::size_t> lanes;
    int padded_length = 0;
    std::size_t valid_length_sum = 0;

    double padding_ratio() const {
        if (lanes.empty() || valid_length_sum == 0)
            return 0.0;
        return (double)((std::size_t)padded_length * lanes.size()) / (double)valid_length_sum;
    }
};

// Stable longest-first ordering primes GGML's CUDA scratch pool with the
// largest shapes and keeps equal-length inputs in caller order.
inline std::vector<std::size_t> descending_order(const std::vector<int>& lengths) {
    if (std::any_of(lengths.begin(), lengths.end(), [](int length) { return length <= 0; }))
        return {};

    std::vector<std::size_t> order(lengths.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return lengths[lhs] > lengths[rhs];
    });
    return order;
}

// Build padded microbatches without dropping or reordering equal-length work.
// A candidate starts a new batch when it would exceed max_batch or the
// padded/valid work ratio. The first item always fits with ratio 1.
inline std::vector<batch> make_batches(const std::vector<int>& lengths, std::size_t max_batch,
                                       double max_padding_ratio) {
    if (max_batch == 0 || !std::isfinite(max_padding_ratio) || max_padding_ratio < 1.0)
        return {};
    const std::vector<std::size_t> order = descending_order(lengths);
    if (!lengths.empty() && order.empty())
        return {};

    std::vector<batch> result;
    for (std::size_t lane : order) {
        const int length = lengths[lane];
        bool start_new = result.empty() || result.back().lanes.size() >= max_batch;
        if (!start_new) {
            const batch& current = result.back();
            const std::size_t next_sum = current.valid_length_sum + (std::size_t)length;
            const double next_ratio =
                (double)((std::size_t)current.padded_length * (current.lanes.size() + 1)) / (double)next_sum;
            start_new = next_ratio > max_padding_ratio;
        }
        if (start_new) {
            result.push_back({});
            result.back().padded_length = length;
        }
        result.back().lanes.push_back(lane);
        result.back().valid_length_sum += (std::size_t)length;
    }
    return result;
}

} // namespace cohere_batch_planner
