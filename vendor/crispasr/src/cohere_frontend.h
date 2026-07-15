// Cohere ASR frontend constants and waveform preparation.
//
// These values are generated at runtime because the checkpoint stores the
// filterbank/window in reduced precision. The implementation mirrors the HF
// CohereAsrFeatureExtractor contract without depending on PyTorch or librosa.

#pragma once

#include <vector>

namespace cohere_frontend {

struct Constants {
    // FP32 non-periodic Hann window, length win_length.
    std::vector<float> window;

    // FP32 librosa-compatible Slaney filterbank in [n_mels, n_fft/2+1]
    // row-major layout.
    std::vector<float> mel_filters;
};

Constants make_constants(int sample_rate, int n_fft, int win_length, int n_mels);

// Add deterministic N(0, dither^2) noise using valid_samples as the seed,
// then apply y[0]=x[0], y[i]=x[i]-preemphasis*x[i-1]. PyTorch's CPU RNG
// stream and 16-value Box-Muller grouping are reproduced; portable scalar
// libm replaces PyTorch's architecture-specific vector math.
std::vector<float> dither_and_preemphasize(const float* samples, int valid_samples, float dither,
                                           float preemphasis);

} // namespace cohere_frontend
