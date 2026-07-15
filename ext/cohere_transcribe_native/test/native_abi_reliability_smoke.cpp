// Standalone ownership and invalid-argument reliability test for instrumented builds.
//
//   c++ -std=c++20 -O1 -g native_abi_reliability_smoke.cpp -ldl -pthread -o smoke
//   ./smoke /path/to/libcrispasr.so /path/to/model.gguf

#include <atomic>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

template <typename Function>
Function load(void* library, const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (!function)
        throw std::runtime_error(std::string("missing symbol ") + name);
    return function;
}

struct open_params {
    int abi_version;
    int n_threads;
    int use_gpu;
    int verbosity;
    int flash_attn;
    int n_gpu_layers;
    int reserved[6];
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: native_abi_reliability_smoke LIBCRISPASR MODEL_GGUF");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error(dlerror());

    using Open = void* (*)(const char*, const char*, const open_params*);
    using Close = void (*)(void*);
    using ErrorKind = int (*)();
    using ErrorMessage = const char* (*)();
    using ResolveDevice = const char* (*)(const char*);
    using Memory = int (*)(void*, std::uint64_t*, std::uint64_t*);
    using Capacity = int (*)(void*);
    using Cancel = int (*)(void*);
    using Transcribe = void* (*)(void*, const float*, int, const char*);
    using TranscribeBatch = void* (*)(void*, const float* const*, const int*, int, const char*);
    using SetInteger = int (*)(void*, int);
    using FreeResult = void (*)(void*);
    using Count = int (*)(void*);
    using IndexedCount = int (*)(void*, int);
    using IndexedText = const char* (*)(void*, int);
    using WordText = const char* (*)(void*, int, int);
    using BatchAt = void* (*)(void*, int);

    const auto open = load<Open>(library, "crispasr_session_open_with_params");
    const auto close = load<Close>(library, "crispasr_session_close");
    const auto error_kind = load<ErrorKind>(library, "crispasr_last_error_kind");
    const auto error_message = load<ErrorMessage>(library, "crispasr_last_error_message");
    const auto resolve_device = load<ResolveDevice>(library, "crispasr_runtime_resolve_device");
    const auto memory = load<Memory>(library, "crispasr_session_memory");
    const auto capacity = load<Capacity>(library, "crispasr_session_batch_capacity");
    const auto cancel = load<Cancel>(library, "crispasr_session_cancel");
    const auto transcribe = load<Transcribe>(library, "crispasr_session_transcribe_lang");
    const auto transcribe_batch =
        load<TranscribeBatch>(library, "crispasr_session_transcribe_batch_lang");
    const auto set_tokens = load<SetInteger>(library, "crispasr_session_set_max_new_tokens");
    const auto set_beam = load<SetInteger>(library, "crispasr_session_set_beam_size");
    const auto set_guard = load<SetInteger>(library, "crispasr_session_set_repetition_loop_guard");
    const auto free_result = load<FreeResult>(library, "crispasr_session_result_free");
    const auto free_batch = load<FreeResult>(library, "crispasr_session_batch_result_free");
    const auto segment_count = load<Count>(library, "crispasr_session_result_n_segments");
    const auto segment_words = load<IndexedCount>(library, "crispasr_session_result_n_words");
    const auto segment_text = load<IndexedText>(library, "crispasr_session_result_segment_text");
    const auto word_text = load<WordText>(library, "crispasr_session_result_word_text");
    const auto batch_count = load<Count>(library, "crispasr_session_batch_result_count");
    const auto batch_at = load<BatchAt>(library, "crispasr_session_batch_result_at");

    require(std::strcmp(resolve_device(nullptr), "cpu") == 0 ||
                std::strcmp(resolve_device(nullptr), "cuda") == 0 ||
                std::strcmp(resolve_device(nullptr), "mps") == 0,
            "automatic device resolution returned an invalid value");
    require(std::strcmp(resolve_device("invalid"), "") == 0,
            "invalid device resolution did not fail closed");

    for (int iteration = 0; iteration < 32; ++iteration) {
        require(!open(nullptr, "cohere", nullptr), "null model path unexpectedly opened");
        require(error_kind() == 1, "null model path returned the wrong error kind");
        require(!open("", "cohere", nullptr), "empty model path unexpectedly opened");
        require(!open("/definitely/missing/cohere.gguf", "other", nullptr),
                "unsupported backend unexpectedly opened");
    }

    // Error state belongs to the calling thread and concurrent callers must not
    // overwrite one another's diagnostics.
    require(!open(nullptr, "cohere", nullptr), "main-thread error fixture failed");
    const std::string main_message = error_message();
    std::atomic<bool> thread_failed{false};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&] {
            for (int iteration = 0; iteration < 200; ++iteration) {
                if (transcribe(nullptr, nullptr, 0, nullptr) || error_kind() != 1 ||
                    std::strstr(error_message(), "invalid") == nullptr) {
                    thread_failed.store(true, std::memory_order_release);
                }
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    require(!thread_failed.load(std::memory_order_acquire),
            "thread-local invalid-argument diagnostics were unstable");
    require(error_kind() == 1 && main_message == error_message(),
            "another thread overwrote the main-thread error state");

    std::uint64_t free_bytes = 99;
    std::uint64_t total_bytes = 99;
    require(memory(nullptr, &free_bytes, &total_bytes) == -1 && free_bytes == 0 && total_bytes == 0,
            "invalid memory query did not clear outputs");
    require(memory(nullptr, nullptr, nullptr) == -1, "null memory outputs were accepted");
    require(capacity(nullptr) == 0, "null session reported batch capacity");
    require(cancel(nullptr) == -1, "null cancellation returned the wrong status");
    require(set_tokens(nullptr, 1) == -1 && set_beam(nullptr, 1) == -1 && set_guard(nullptr, 1) == -1,
            "null session setter was accepted");
    require(segment_count(nullptr) == 0 && segment_words(nullptr, -1) == 0,
            "null result reported members");
    require(std::strcmp(segment_text(nullptr, -1), "") == 0 &&
                std::strcmp(word_text(nullptr, -1, -1), "") == 0,
            "null result returned text");
    require(batch_count(nullptr) == 0 && !batch_at(nullptr, 0),
            "null batch result reported a lane");
    free_result(nullptr);
    free_batch(nullptr);
    close(nullptr);

    const open_params params{2, 4, 0, 0, 1, -1, {}};
    void* session = open(argv[2], "cohere", &params);
    if (!session)
        throw std::runtime_error(std::string("session open failed: ") + error_message());
    try {
        require(capacity(session) >= 1, "real session reported no batch capacity");
        const float sample = 0.0f;
        const float* rows[] = {&sample};
        const int zero[] = {0};
        const int too_long[] = {35 * 16'000 + 1};
        require(!transcribe(session, nullptr, 1, "en") && error_kind() == 1,
                "null single-row PCM was accepted");
        require(!transcribe(session, &sample, 0, "en") && error_kind() == 1,
                "empty single-row PCM was accepted");
        require(!transcribe_batch(session, nullptr, zero, 1, "en") && error_kind() == 1,
                "null batch PCM table was accepted");
        require(!transcribe_batch(session, rows, nullptr, 1, "en") && error_kind() == 1,
                "null batch size table was accepted");
        require(!transcribe_batch(session, rows, zero, 1, "en") && error_kind() == 1,
                "empty batch row was accepted");
        require(!transcribe_batch(session, rows, too_long, 1, "en") && error_kind() == 1,
                "oversized batch row was accepted");
        require(!transcribe_batch(session, rows, zero, capacity(session) + 1, "en") &&
                    error_kind() == 1,
                "over-capacity batch was accepted");
        require(cancel(session) == 0, "invalid calls poisoned the idle cancellation state");
    } catch (...) {
        close(session);
        throw;
    }
    close(session);
    dlclose(library);
    std::cout << "native ABI reliability smoke: 32 opens + 1600 threaded failures + real-session bounds\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
