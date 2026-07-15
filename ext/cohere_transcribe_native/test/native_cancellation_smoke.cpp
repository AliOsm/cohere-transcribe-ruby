// Standalone real-model cancellation/lifetime smoke test for sanitizer builds.
//
//   c++ -std=c++20 -O1 -g -pthread native_cancellation_smoke.cpp -ldl -o smoke
//   ASAN_OPTIONS=detect_leaks=1 ./smoke /path/to/libcrispasr.so /path/to/model.gguf

// Keeping this process independent from Ruby avoids false positives from
// ASan's interaction with Ruby's separately mapped fiber stacks.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

template <typename Function>
Function load(void* library, const char* name) {
    dlerror();
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    const char* error = dlerror();
    if (error || !function)
        throw std::runtime_error(std::string("missing symbol ") + name + ": " +
                                 (error ? error : "unknown loader error"));
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

struct worker_outcome {
    void* result = nullptr;
    int error_kind = -1;
    std::string error_message;
};

} // namespace

int main(int argc, char** argv) try {
    if (argc != 3)
        throw std::runtime_error("usage: native_cancellation_smoke LIBCRISPASR MODEL_GGUF");

    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library)
        throw std::runtime_error(dlerror());

    using Open = void* (*)(const char*, const char*, const open_params*);
    using Close = void (*)(void*);
    using Cancel = int (*)(void*);
    using Transcribe = void* (*)(void*, const float*, int, const char*);
    using TranscribeBatch = void* (*)(void*, const float* const*, const int*, int, const char*);
    using FreeResult = void (*)(void*);
    using BatchCount = int (*)(void*);
    using FreeBatchResult = void (*)(void*);
    using ErrorKind = int (*)();
    using ErrorMessage = const char* (*)();
    using SetInteger = int (*)(void*, int);

    const auto open = load<Open>(library, "crispasr_session_open_with_params");
    const auto close = load<Close>(library, "crispasr_session_close");
    const auto cancel = load<Cancel>(library, "crispasr_session_cancel");
    const auto transcribe = load<Transcribe>(library, "crispasr_session_transcribe_lang");
    const auto transcribe_batch =
        load<TranscribeBatch>(library, "crispasr_session_transcribe_batch_lang");
    const auto free_result = load<FreeResult>(library, "crispasr_session_result_free");
    const auto batch_count = load<BatchCount>(library, "crispasr_session_batch_result_count");
    const auto free_batch_result =
        load<FreeBatchResult>(library, "crispasr_session_batch_result_free");
    const auto error_kind = load<ErrorKind>(library, "crispasr_last_error_kind");
    const auto error_message = load<ErrorMessage>(library, "crispasr_last_error_message");
    const auto set_max_tokens = load<SetInteger>(library, "crispasr_session_set_max_new_tokens");

    const open_params params{2, 4, 0, 0, 1, -1, {}};
    void* session = open(argv[2], "cohere", &params);
    if (!session)
        throw std::runtime_error(std::string("session open failed: ") + error_message());

    try {
        if (cancel(session) != 0)
            throw std::runtime_error("idle cancellation was not a no-op");
        if (set_max_tokens(session, 64) != 0)
            throw std::runtime_error("could not set the generation limit");

        // A maximum-length row leaves ample time to observe the running state,
        // even on a fast host. Repeating exercises reset-to-idle and retained
        // graph lifetime after several abort points.
        const std::vector<float> long_audio(35 * 16'000, 0.0f);
        for (int iteration = 0; iteration < 5; ++iteration) {
            std::atomic<bool> entered{false};
            worker_outcome outcome;
            std::thread worker([&] {
                entered.store(true, std::memory_order_release);
                outcome.result = transcribe(
                    session,
                    long_audio.data(),
                    static_cast<int>(long_audio.size()),
                    "en");
                outcome.error_kind = error_kind();
                const char* message = error_message();
                outcome.error_message = message ? message : "";
            });

            while (!entered.load(std::memory_order_acquire))
                std::this_thread::yield();

            bool requested = false;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (std::chrono::steady_clock::now() < deadline) {
                if (cancel(session) == 1) {
                    requested = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (!requested) {
                worker.join();
                free_result(outcome.result);
                throw std::runtime_error("never observed an active inference operation");
            }

            worker.join();
            if (outcome.result) {
                free_result(outcome.result);
                throw std::runtime_error("cancelled inference returned a result");
            }
            if (outcome.error_kind != 5 || outcome.error_message.find("cancel") == std::string::npos)
                throw std::runtime_error("cancelled inference returned the wrong typed diagnostic");
            if (cancel(session) != 0)
                throw std::runtime_error("completed cancellation poisoned the idle session");
        }

        // Prove the retained session remains usable after repeated aborts.
        if (set_max_tokens(session, 4) != 0)
            throw std::runtime_error("could not set the reuse generation limit");
        const std::vector<float> short_audio(16'000, 0.0f);
        void* reused = transcribe(
            session,
            short_audio.data(),
            static_cast<int>(short_audio.size()),
            "en");
        if (!reused)
            throw std::runtime_error(std::string("post-cancellation reuse failed: ") + error_message());
        free_result(reused);
        if (cancel(session) != 0)
            throw std::runtime_error("post-reuse idle cancellation was not a no-op");

        // Exercise the independent padded-encoder/ragged-decoder ownership
        // path after a cancelled single-row graph has been replaced.
        const float* rows[] = {short_audio.data(), short_audio.data()};
        const int row_sizes[] = {
            static_cast<int>(short_audio.size()),
            static_cast<int>(short_audio.size()),
        };
        void* batch = transcribe_batch(session, rows, row_sizes, 2, "en");
        if (!batch)
            throw std::runtime_error(std::string("post-cancellation batch failed: ") + error_message());
        if (batch_count(batch) != 2) {
            free_batch_result(batch);
            throw std::runtime_error("post-cancellation batch returned the wrong row count");
        }
        free_batch_result(batch);
        if (cancel(session) != 0)
            throw std::runtime_error("post-batch idle cancellation was not a no-op");
    } catch (...) {
        close(session);
#if !defined(COHERE_SANITIZER_KEEP_LIBRARY_LOADED)
        dlclose(library);
#endif
        throw;
    }

    close(session);
#if !defined(COHERE_SANITIZER_KEEP_LIBRARY_LOADED)
    dlclose(library);
#endif
    std::cout << "verified five cancellations plus single-row and batch reuse\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
