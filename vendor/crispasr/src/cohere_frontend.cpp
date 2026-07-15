#include "cohere_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

namespace cohere_frontend {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

std::vector<float> make_nonperiodic_hann(int length) {
    if (length <= 0)
        return {};
    if (length == 1)
        return {1.0f};

    // torch.hann_window(..., periodic=false, dtype=float32) evaluates the
    // phase in FP32. Keeping every operand float reproduces that behavior,
    // including its tiny endpoint-asymmetry from phase rounding.
    constexpr float pi_f = (float)kPi;
    std::vector<float> window((size_t)length);
    for (int i = 0; i < length; ++i) {
        const float phase = 2.0f * pi_f * (float)i / (float)(length - 1);
        window[(size_t)i] = 0.5f * (1.0f - std::cos(phase));
    }
    return window;
}

std::vector<float> make_librosa_slaney_filters(int sample_rate, int n_fft, int n_mels) {
    if (sample_rate <= 0 || n_fft <= 0 || n_mels <= 0)
        return {};

    const int n_freqs = n_fft / 2 + 1;
    const double fmin = 0.0;
    const double fmax = (double)sample_rate / 2.0;
    const double f_sp = 200.0 / 3.0;
    const double min_log_hz = 1000.0;
    const double min_log_mel = min_log_hz / f_sp;
    const double logstep = std::log(6.4) / 27.0;

    auto hz_to_mel = [&](double hz) {
        return hz >= min_log_hz ? min_log_mel + std::log(hz / min_log_hz) / logstep : hz / f_sp;
    };
    auto mel_to_hz = [&](double mel) {
        return mel >= min_log_mel ? min_log_hz * std::exp(logstep * (mel - min_log_mel)) : f_sp * mel;
    };

    // librosa computes its frequency grid and mel centers in float64, writes
    // each unnormalized triangle to a float32 output row, then applies the
    // float64 Slaney area scale in-place (casting the result back to FP32).
    const double mel_min = hz_to_mel(fmin);
    const double mel_max = hz_to_mel(fmax);
    const double mel_step = (mel_max - mel_min) / (double)(n_mels + 1);
    std::vector<double> centers((size_t)n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i)
        centers[(size_t)i] = mel_to_hz(mel_min + mel_step * (double)i);
    centers.back() = fmax; // np.linspace includes the stop value exactly.

    std::vector<float> filters((size_t)n_mels * n_freqs, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        const double lo = centers[(size_t)m];
        const double mid = centers[(size_t)m + 1];
        const double hi = centers[(size_t)m + 2];
        const double area_scale = 2.0 / (hi - lo);
        for (int k = 0; k < n_freqs; ++k) {
            const double hz = (double)k * (double)sample_rate / (double)n_fft;
            const double lower = (hz - lo) / (mid - lo);
            const double upper = (hi - hz) / (hi - mid);
            const float triangle = (float)std::max(0.0, std::min(lower, upper));
            filters[(size_t)m * n_freqs + k] = (float)((double)triangle * area_scale);
        }
    }
    return filters;
}

inline float torch_uniform_float(std::mt19937& generator) {
    // at::uniform_real_distribution<float>: keep the lower 24 bits and
    // scale by 2^-24, yielding [0, 1).
    constexpr uint32_t mask = (1u << 24) - 1u;
    constexpr float scale = 1.0f / (float)(1u << 24);
    return (float)(generator() & mask) * scale;
}

inline double torch_uniform_double(std::mt19937& generator) {
    // CPUGeneratorImpl::random64 concatenates two consecutive MT outputs,
    // then uniform_real_distribution<double> keeps the lower 53 bits.
    constexpr uint64_t mask = (1ull << 53) - 1ull;
    constexpr double scale = 1.0 / (double)(1ull << 53);
    const uint64_t value = ((uint64_t)generator() << 32) | generator();
    return (double)(value & mask) * scale;
}

void box_muller_16(float* values) {
    constexpr float two_pi = (float)(2.0 * kPi);
    for (int j = 0; j < 8; ++j) {
        const float u1 = 1.0f - values[j]; // [0,1) -> (0,1] before log
        const float u2 = values[j + 8];
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = two_pi * u2;
        values[j] = radius * std::cos(theta);
        values[j + 8] = radius * std::sin(theta);
    }
}

std::vector<float> length_seeded_normal(int valid_samples) {
    std::vector<float> values((size_t)std::max(valid_samples, 0));
    if (valid_samples <= 0)
        return values;

    // CPUGeneratorImpl's mt19937 sequence matches std::mt19937 for a 32-bit
    // seed. torch.randn first fills the entire contiguous tensor with uniform
    // values, transforms complete groups of 16, then regenerates/recomputes
    // the final 16 values when the length has a remainder.
    std::mt19937 generator((uint32_t)valid_samples);
    for (float& value : values)
        value = torch_uniform_float(generator);

    if (valid_samples < 16) {
        // PyTorch dispatches short contiguous tensors through its scalar
        // double-precision Box-Muller path and caches the second sample.
        generator.seed((uint32_t)valid_samples);
        bool have_cached = false;
        double cached = 0.0;
        for (float& value : values) {
            if (have_cached) {
                value = (float)cached;
                have_cached = false;
                continue;
            }
            const double u1 = torch_uniform_double(generator);
            const double u2 = torch_uniform_double(generator);
            const double radius = std::sqrt(-2.0 * std::log1p(-u2));
            const double theta = 2.0 * kPi * u1;
            value = (float)(radius * std::cos(theta));
            cached = radius * std::sin(theta);
            have_cached = true;
        }
        return values;
    }

    for (int i = 0; i < valid_samples - 15; i += 16)
        box_muller_16(values.data() + i);

    if (valid_samples % 16 != 0) {
        float* tail = values.data() + valid_samples - 16;
        for (int i = 0; i < 16; ++i)
            tail[i] = torch_uniform_float(generator);
        box_muller_16(tail);
    }
    return values;
}

} // namespace

Constants make_constants(int sample_rate, int n_fft, int win_length, int n_mels) {
    Constants constants;
    constants.window = make_nonperiodic_hann(win_length);
    constants.mel_filters = make_librosa_slaney_filters(sample_rate, n_fft, n_mels);
    return constants;
}

std::vector<float> dither_and_preemphasize(const float* samples, int valid_samples, float dither,
                                           float preemphasis) {
    if (!samples || valid_samples <= 0)
        return {};

    std::vector<float> waveform(samples, samples + valid_samples);
    if (dither > 0.0f) {
        auto noise = length_seeded_normal(valid_samples);
        for (int i = 0; i < valid_samples; ++i) {
            const float scaled_noise = dither * noise[(size_t)i];
            waveform[(size_t)i] += scaled_noise;
        }
    }

    // Work backwards so the source sample at i-1 is still the original
    // dithered waveform, matching torch.cat([x[:1], x[1:]-a*x[:-1]]).
    if (preemphasis != 0.0f) {
        for (int i = valid_samples - 1; i >= 1; --i) {
            const float previous = preemphasis * waveform[(size_t)i - 1];
            waveform[(size_t)i] -= previous;
        }
    }
    return waveform;
}

} // namespace cohere_frontend
