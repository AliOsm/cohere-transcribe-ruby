#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace core_repetition_loop {

constexpr std::size_t min_generated_tokens = 96;
constexpr std::size_t repeats = 4;
constexpr std::size_t min_period = 8;
constexpr std::size_t max_period = 32;

// Match cohere-transcribe's conservative Transformers stopping criterion:
// after at least 96 generated tokens, stop when the current suffix contains
// four immediately repeated blocks of any length from 8 through 32 tokens.
inline bool triggered(const std::vector<int>& generated) {
    if (generated.size() < min_generated_tokens)
        return false;

    const std::size_t largest_period = std::min(max_period, generated.size() / repeats);
    for (std::size_t period = min_period; period <= largest_period; ++period) {
        const std::size_t first = generated.size() - repeats * period;
        bool matches = true;
        for (std::size_t repeat = 1; repeat < repeats && matches; ++repeat) {
            const std::size_t offset = first + repeat * period;
            for (std::size_t token = 0; token < period; ++token) {
                if (generated[first + token] != generated[offset + token]) {
                    matches = false;
                    break;
                }
            }
        }
        if (matches)
            return true;
    }
    return false;
}

// Return the first generated-prefix length that would trigger the guard, or
// zero when no prefix triggers it. Batched generation uses this to reproduce
// row-wise stopping after the ragged decoder has completed the physical batch.
inline std::size_t first_trigger_length(const std::vector<int>& generated) {
    std::vector<int> prefix;
    prefix.reserve(generated.size());
    for (int token : generated) {
        prefix.push_back(token);
        if (triggered(prefix))
            return prefix.size();
    }
    return 0;
}

} // namespace core_repetition_loop
