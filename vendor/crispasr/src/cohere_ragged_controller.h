#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace cohere_ragged {

enum class stop_reason {
    active,
    eos,
    max_tokens,
};

enum class step_result {
    advanced,
    complete,
    invalid_input,
    already_complete,
};

struct generated_token {
    int token_id = 0;
    float probability = 0.0f;

    generated_token() = default;
    generated_token(int token_id_value, float probability_value)
        : token_id(token_id_value), probability(probability_value) {
    }
};

struct lane_result {
    stop_reason reason = stop_reason::active;
    std::size_t max_tokens = 0;
    std::vector<generated_token> generated;
};

// Host-side greedy-generation state for a fixed physical decoder batch. Lanes
// remain at their original indices for the controller's lifetime; completed
// lanes feed placeholder_token_id while active lanes feed their latest token.
class controller {
  public:
    controller(int eos_token_id, int placeholder_token_id, const std::vector<std::size_t>& max_tokens)
        : eos_token_id_(eos_token_id), placeholder_token_id_(placeholder_token_id),
          valid_(eos_token_id >= 0 && placeholder_token_id >= 0 && !max_tokens.empty()) {
        if (!valid_)
            return;

        feed_tokens_.assign(max_tokens.size(), placeholder_token_id_);
        results_.reserve(max_tokens.size());
        for (std::size_t limit : max_tokens) {
            lane_result lane;
            lane.max_tokens = limit;
            if (limit == 0)
                lane.reason = stop_reason::max_tokens;
            else
                ++active_count_;
            results_.push_back(lane);
        }
    }

    bool valid() const {
        return valid_;
    }

    bool done() const {
        return valid_ && active_count_ == 0;
    }

    std::size_t lane_count() const {
        return results_.size();
    }

    std::size_t active_count() const {
        return active_count_;
    }

    int eos_token_id() const {
        return eos_token_id_;
    }

    int placeholder_token_id() const {
        return placeholder_token_id_;
    }

    const std::vector<int>& feed_tokens() const {
        return feed_tokens_;
    }

    const std::vector<lane_result>& results() const {
        return results_;
    }

    step_result apply_step(const std::vector<int>& token_ids, const std::vector<float>& probabilities) {
        if (token_ids.size() != probabilities.size())
            return step_result::invalid_input;
        return apply_step(token_ids.data(), probabilities.data(), token_ids.size());
    }

    step_result apply_step(const int* token_ids, const float* probabilities, std::size_t count) {
        if (!valid_ || !token_ids || !probabilities || count != results_.size())
            return step_result::invalid_input;
        if (done())
            return step_result::already_complete;

        // Validate the whole physical batch before mutating any lane.
        for (std::size_t lane = 0; lane < count; ++lane) {
            const float probability = probabilities[lane];
            if (token_ids[lane] < 0 || !std::isfinite((double)probability) || probability < 0.0f ||
                probability > 1.0f)
                return step_result::invalid_input;
        }

        for (std::size_t lane = 0; lane < count; ++lane) {
            lane_result& result = results_[lane];
            if (result.reason != stop_reason::active)
                continue;

            const int token_id = token_ids[lane];
            if (token_id == eos_token_id_) {
                result.reason = stop_reason::eos;
                feed_tokens_[lane] = placeholder_token_id_;
                --active_count_;
                continue;
            }

            result.generated.push_back(generated_token(token_id, probabilities[lane]));
            if (result.generated.size() >= result.max_tokens) {
                result.reason = stop_reason::max_tokens;
                feed_tokens_[lane] = placeholder_token_id_;
                --active_count_;
            } else {
                feed_tokens_[lane] = token_id;
            }
        }

        return done() ? step_result::complete : step_result::advanced;
    }

  private:
    int eos_token_id_ = 0;
    int placeholder_token_id_ = 0;
    bool valid_ = false;
    std::size_t active_count_ = 0;
    std::vector<int> feed_tokens_;
    std::vector<lane_result> results_;
};

} // namespace cohere_ragged
