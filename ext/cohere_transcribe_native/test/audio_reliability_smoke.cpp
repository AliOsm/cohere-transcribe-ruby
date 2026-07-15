#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Probe = int (*)(char*, std::size_t);
using Duration = int (*)(const char*, double*, char*, std::size_t);
using Decode = int (*)(const char*, int, std::uint64_t, float**, std::int64_t*, char*, std::size_t);
using Cancel = void (*)();
using Free = void (*)(void*);

template <typename Function>
Function load(void* library, const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (!function)
        throw std::runtime_error(std::string("missing symbol ") + name);
    return function;
}

template <typename Integer>
void little_endian(std::vector<std::uint8_t>& bytes, Integer value) {
    for (std::size_t index = 0; index < sizeof(value); ++index)
        bytes.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(value) >> (8 * index)) & 0xff));
}

std::vector<std::uint8_t> wav_fixture() {
    constexpr std::uint32_t frames = 400;
    constexpr std::uint32_t rate = 16'000;
    constexpr std::uint16_t channels = 2;
    constexpr std::uint32_t pcm_bytes = frames * channels * sizeof(std::int16_t);
    std::vector<std::uint8_t> bytes;
    const auto append = [&](const char* text, std::size_t length) {
        bytes.insert(bytes.end(), text, text + length);
    };
    append("RIFF", 4);
    little_endian(bytes, 36U + pcm_bytes);
    append("WAVEfmt ", 8);
    little_endian(bytes, 16U);
    little_endian(bytes, static_cast<std::uint16_t>(1));
    little_endian(bytes, channels);
    little_endian(bytes, rate);
    little_endian(bytes, static_cast<std::uint32_t>(rate * channels * sizeof(std::int16_t)));
    little_endian(bytes, static_cast<std::uint16_t>(channels * sizeof(std::int16_t)));
    little_endian(bytes, static_cast<std::uint16_t>(16));
    append("data", 4);
    little_endian(bytes, pcm_bytes);
    for (std::uint32_t index = 0; index < frames; ++index) {
        little_endian(bytes, static_cast<std::uint16_t>(16'384));
        little_endian(bytes, static_cast<std::uint16_t>(-8'192));
    }
    return bytes;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("could not write audio reliability fixture");
}

void check_decode_result(int status, float* samples, std::int64_t count,
                         std::uint64_t maximum_bytes, Free release) {
    if (status == 0) {
        if (count < 0 || (count > 0 && !samples) ||
            static_cast<std::uint64_t>(count) > maximum_bytes / sizeof(float)) {
            release(samples);
            throw std::runtime_error("successful decode returned invalid ownership metadata");
        }
        for (std::int64_t index = 0; index < count; ++index) {
            if (!std::isfinite(samples[index])) {
                release(samples);
                throw std::runtime_error("successful decode returned a non-finite sample");
            }
        }
    } else if (samples || count != 0) {
        release(samples);
        throw std::runtime_error("failed decode retained output ownership metadata");
    }
    release(samples);
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: audio_reliability_smoke ADAPTER PATH_PREFIX");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error(dlerror());

    const auto probe = load<Probe>(library, "cohere_audio_ffmpeg_probe");
    const auto duration = load<Duration>(library, "cohere_audio_ffmpeg_duration");
    const auto decode = load<Decode>(library, "cohere_audio_ffmpeg_decode");
    const auto cancel = load<Cancel>(library, "cohere_audio_ffmpeg_cancel");
    const auto release = load<Free>(library, "cohere_audio_ffmpeg_free");
    char diagnostic[1024] = {};
    if (probe(diagnostic, sizeof(diagnostic)) != 0)
        throw std::runtime_error(diagnostic);
    probe(nullptr, 0);

    const std::string valid_path = std::string(argv[2]) + "-valid.wav";
    const std::string fuzz_path = std::string(argv[2]) + "-fuzz.bin";
    const auto valid = wav_fixture();
    write_bytes(valid_path, valid);

    std::atomic<bool> concurrency_failed{false};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&] {
            for (int iteration = 0; iteration < 50; ++iteration) {
                float* samples = nullptr;
                std::int64_t count = 0;
                char message[256] = {};
                const int status = decode(valid_path.c_str(), 16'000, 4096, &samples, &count,
                                          message, sizeof(message));
                if (status != 0 || !samples || count != 400)
                    concurrency_failed.store(true, std::memory_order_release);
                release(samples);
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    if (concurrency_failed.load(std::memory_order_acquire))
        throw std::runtime_error("concurrent valid decodes were not stable");

    float* limited_samples = nullptr;
    std::int64_t limited_count = 0;
    if (decode(valid_path.c_str(), 16'000, 4, &limited_samples, &limited_count,
               diagnostic, sizeof(diagnostic)) != 4 || limited_samples || limited_count != 0) {
        release(limited_samples);
        throw std::runtime_error("decoded-audio memory limit did not fail closed");
    }

    std::mt19937_64 random(0xC0E4EULL);
    for (int iteration = 0; iteration < 1'000; ++iteration) {
        std::vector<std::uint8_t> bytes;
        if (iteration < 500) {
            bytes = valid;
            const std::size_t mutations = 1 + (random() % 32);
            for (std::size_t mutation = 0; mutation < mutations; ++mutation)
                bytes[random() % bytes.size()] = static_cast<std::uint8_t>(random());
            if ((iteration % 3) == 0)
                bytes.resize(random() % (bytes.size() + 1));
        } else {
            bytes.resize(random() % 65'536);
            for (auto& byte : bytes)
                byte = static_cast<std::uint8_t>(random());
        }
        write_bytes(fuzz_path, bytes);

        double seconds = 99.0;
        const int duration_status = duration(fuzz_path.c_str(), &seconds, diagnostic, sizeof(diagnostic));
        if ((duration_status == 0 && (!std::isfinite(seconds) || seconds < 0.0)) ||
            (duration_status != 0 && seconds != -1.0)) {
            throw std::runtime_error("duration fuzz returned invalid output metadata");
        }

        float* samples = nullptr;
        std::int64_t count = 0;
        const int status = decode(fuzz_path.c_str(), 16'000, 64 * 1024, &samples, &count,
                                  diagnostic, sizeof(diagnostic));
        check_decode_result(status, samples, count, 64 * 1024, release);
        if ((iteration % 97) == 0)
            cancel();
    }

    float* recovered = nullptr;
    std::int64_t recovered_count = 0;
    const int recovered_status = decode(valid_path.c_str(), 16'000, 4096, &recovered,
                                        &recovered_count, diagnostic, sizeof(diagnostic));
    if (recovered_status != 0 || !recovered || recovered_count != 400) {
        release(recovered);
        throw std::runtime_error("cancellation or malformed input poisoned later decoding");
    }
    release(recovered);

    std::remove(valid_path.c_str());
    std::remove(fuzz_path.c_str());
    dlclose(library);
    std::cout << "audio ABI reliability smoke: 400 concurrent valid + 1000 malformed cases\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
