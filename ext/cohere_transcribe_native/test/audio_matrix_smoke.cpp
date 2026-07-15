#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename Function>
Function load(void* library, const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (!function)
        throw std::runtime_error(std::string("missing symbol ") + name);
    return function;
}

template <typename Integer>
void little_endian(std::ofstream& output, Integer value) {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        output.put(static_cast<char>((static_cast<std::uint64_t>(value) >> (8 * index)) & 0xff));
    }
}

void write_fixture(const char* path) {
    constexpr std::uint32_t frames = 400;
    constexpr std::uint32_t rate = 16'000;
    constexpr std::uint16_t channels = 2;
    constexpr std::uint32_t pcm_bytes = frames * channels * sizeof(std::int16_t);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write("RIFF", 4);
    little_endian(output, 36U + pcm_bytes);
    output.write("WAVEfmt ", 8);
    little_endian(output, 16U);
    little_endian(output, static_cast<std::uint16_t>(1));
    little_endian(output, channels);
    little_endian(output, rate);
    little_endian(output, static_cast<std::uint32_t>(rate * channels * sizeof(std::int16_t)));
    little_endian(output, static_cast<std::uint16_t>(channels * sizeof(std::int16_t)));
    little_endian(output, static_cast<std::uint16_t>(16));
    output.write("data", 4);
    little_endian(output, pcm_bytes);
    for (std::uint32_t index = 0; index < frames; ++index) {
        little_endian(output, static_cast<std::uint16_t>(16'384));
        little_endian(output, static_cast<std::uint16_t>(-8'192));
    }
    if (!output)
        throw std::runtime_error("could not write WAV fixture");
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: audio_matrix_smoke ADAPTER FIXTURE");
    write_fixture(argv[2]);
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error(dlerror());

    using Probe = int (*)(char*, std::size_t);
    using Duration = int (*)(const char*, double*, char*, std::size_t);
    using Decode = int (*)(const char*, int, std::uint64_t, float**, std::int64_t*, char*, std::size_t);
    using Free = void (*)(void*);
    const auto probe = load<Probe>(library, "cohere_audio_ffmpeg_probe");
    const auto duration = load<Duration>(library, "cohere_audio_ffmpeg_duration");
    const auto decode = load<Decode>(library, "cohere_audio_ffmpeg_decode");
    const auto release = load<Free>(library, "cohere_audio_ffmpeg_free");

    char diagnostic[1024] = {};
    if (probe(diagnostic, sizeof(diagnostic)) != 0)
        throw std::runtime_error(diagnostic);

    float* invalid_samples = reinterpret_cast<float*>(static_cast<std::uintptr_t>(1));
    std::int64_t invalid_count = 17;
    if (decode(nullptr, 16'000, 0, &invalid_samples, &invalid_count,
               diagnostic, sizeof(diagnostic)) != 2 || invalid_samples || invalid_count != 0) {
        throw std::runtime_error("invalid decode did not clear output ownership metadata");
    }
    double invalid_duration = 17.0;
    if (duration(nullptr, &invalid_duration, diagnostic, sizeof(diagnostic)) != 2 ||
        invalid_duration != -1.0) {
        throw std::runtime_error("invalid duration probe did not clear output metadata");
    }
    release(nullptr);

    double seconds = -1.0;
    std::int64_t count = 0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        if (duration(argv[2], &seconds, diagnostic, sizeof(diagnostic)) != 0)
            throw std::runtime_error(diagnostic);
        if (std::abs(seconds - (400.0 / 16'000.0)) > 1e-9)
            throw std::runtime_error("duration mismatch");

        float* samples = nullptr;
        count = 0;
        const int status = decode(argv[2], 16'000, 400 * sizeof(float), &samples, &count,
                                  diagnostic, sizeof(diagnostic));
        if (status != 0) {
            release(samples);
            throw std::runtime_error(diagnostic);
        }
        if (!samples || count != 400 || !std::isfinite(samples[200])) {
            release(samples);
            throw std::runtime_error("decode output mismatch");
        }
        release(samples);
    }
    std::cout << diagnostic << " duration=" << seconds << " samples=" << count << '\n';
    dlclose(library);
    std::remove(argv[2]);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
