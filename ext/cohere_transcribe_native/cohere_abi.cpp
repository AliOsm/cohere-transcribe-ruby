// Lean Cohere-only implementation of the CrispASR session C ABI consumed by
// lib/cohere/transcribe/asr/native.rb. The model implementation and grouping
// behavior are copied from the pinned CrispASR sources; unrelated backends are
// intentionally omitted from this private gem runtime.

#include "cohere.h"
#include "cohere_decoder_batch_layout.h"
#include "cohere_encoder_padded_layout.h"
#include "cohere_token_renderer.h"
#include "core/gpu_backend_pref.h"
#include "core/repetition_loop_guard.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define CRISPASR_GEM_EXPORT extern "C" __declspec(dllexport)
#else
#define CRISPASR_GEM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct crispasr_open_params_v1 {
    int abi_version;
    int n_threads;
    int use_gpu;
    int verbosity;
    int flash_attn;
    int n_gpu_layers;
    int reserved[6];
};

struct crispasr_session_word {
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    float p = 1.0f;
};

struct crispasr_session_segment {
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    std::vector<crispasr_session_word> words;
};

struct crispasr_session_result {
    std::vector<crispasr_session_segment> segments;
    int generated_tokens = 0;
    int generation_limit = 0;
    int generation_capacity = 0;
    bool stopped_by_max_tokens = false;
    bool repetition_stopped = false;
};

struct crispasr_session_batch_stats {
    int64_t total_us = 0;
    int64_t feature_wall_us = 0;
    int64_t feature_worker_us = 0;
    int64_t mel_pack_us = 0;
    int64_t encoder_graph_build_us = 0;
    int64_t encoder_graph_alloc_us = 0;
    int64_t encoder_input_us = 0;
    int64_t encoder_compute_us = 0;
    int64_t encoder_readback_us = 0;
    int64_t encoder_repack_us = 0;
    int64_t decoder_total_us = 0;
    int64_t decoder_cross_kv_us = 0;
    int64_t decoder_reserve_us = 0;
    int64_t decoder_decode_us = 0;
    int64_t render_us = 0;
    int64_t decoder_calls = 0;
    int64_t generation_steps = 0;
    int64_t encoder_microbatches = 0;
    int64_t token_id_readback_bytes = 0;
};

struct crispasr_session_batch_result {
    std::vector<std::unique_ptr<crispasr_session_result>> lanes;
    crispasr_session_batch_stats stats;
};

enum class crispasr_operation_state : unsigned char {
    idle = 0,
    running = 1,
    cancel_requested = 2,
};

struct crispasr_session {
    std::string backend = "cohere";
    cohere_context* cohere_ctx = nullptr;
    int n_threads = 4;
    int max_new_tokens = 0;
    int beam_size = 1;
    bool repetition_loop_guard = true;
    std::atomic<crispasr_operation_state> operation_state{crispasr_operation_state::idle};
};

// One logical session batch may span several padded-encoder microbatches.
// CrispASR deliberately gives those two stages different structural limits:
// the activation-heavy encoder is bounded independently while the ragged
// decoder owns collision-tested cache layouts for up to twenty-four lanes.
static constexpr int crispasr_session_batch_capacity_value() {
    static_assert(cohere_encoder_padded::max_batch > 0);
    static_assert(cohere_decoder_batch::max_batch >= cohere_encoder_padded::max_batch);
    return cohere_decoder_batch::max_batch;
}

enum class crispasr_error_kind : int {
    none = 0,
    invalid_argument = 1,
    out_of_memory = 2,
    invariant = 3,
    runtime = 4,
    cancelled = 5,
};

struct crispasr_error_state {
    crispasr_error_kind kind = crispasr_error_kind::none;
    char message[512] = {};
};

static thread_local crispasr_error_state crispasr_thread_error;

static void crispasr_clear_error() noexcept {
    crispasr_thread_error = {};
}

static void crispasr_set_error(crispasr_error_kind kind, const char* message) noexcept {
    crispasr_thread_error.kind = kind;
    const char* source = message ? message : "";
    const size_t length = std::min(std::strlen(source), sizeof(crispasr_thread_error.message) - 1);
    std::memcpy(crispasr_thread_error.message, source, length);
    crispasr_thread_error.message[length] = '\0';
}

template <typename T>
static T* crispasr_fail(crispasr_error_kind kind, const char* message) noexcept {
    crispasr_set_error(kind, message);
    return nullptr;
}

static void crispasr_import_cohere_error(crispasr_error_kind fallback_kind, const char* fallback_message) noexcept {
    crispasr_error_kind kind = fallback_kind;
    switch (cohere_last_error_kind()) {
    case COHERE_ERROR_INVALID_ARGUMENT: kind = crispasr_error_kind::invalid_argument; break;
    case COHERE_ERROR_OUT_OF_MEMORY: kind = crispasr_error_kind::out_of_memory; break;
    case COHERE_ERROR_INVARIANT: kind = crispasr_error_kind::invariant; break;
    case COHERE_ERROR_RUNTIME: kind = crispasr_error_kind::runtime; break;
    case COHERE_ERROR_CANCELLED: kind = crispasr_error_kind::cancelled; break;
    case COHERE_ERROR_NONE: break;
    }
    const char* message = cohere_last_error_message();
    crispasr_set_error(kind, message && message[0] != '\0' ? message : fallback_message);
}

static bool crispasr_session_abort_requested(void* opaque) noexcept {
    const auto* session = static_cast<const crispasr_session*>(opaque);
    return session && session->operation_state.load(std::memory_order_acquire) ==
                          crispasr_operation_state::cancel_requested;
}

class crispasr_operation_guard {
  public:
    explicit crispasr_operation_guard(crispasr_session* session) noexcept : session_(session) {
        if (!session_)
            return;
        auto expected = crispasr_operation_state::idle;
        acquired_ = session_->operation_state.compare_exchange_strong(
            expected,
            crispasr_operation_state::running,
            std::memory_order_acq_rel,
            std::memory_order_acquire);
    }

    ~crispasr_operation_guard() {
        if (acquired_)
            session_->operation_state.store(crispasr_operation_state::idle, std::memory_order_release);
    }

    explicit operator bool() const noexcept { return acquired_; }

    crispasr_operation_guard(const crispasr_operation_guard&) = delete;
    crispasr_operation_guard& operator=(const crispasr_operation_guard&) = delete;

  private:
    crispasr_session* session_ = nullptr;
    bool acquired_ = false;
};

template <typename T>
static T* crispasr_cancelled() noexcept {
    return crispasr_fail<T>(crispasr_error_kind::cancelled, "Cohere inference was cancelled");
}

struct token_record {
    std::string text;
    int64_t t0 = 0;
    int64_t t1 = 0;
    float p = 1.0f;
};

static bool backend_name_starts_with(const char* name, const char* prefix) {
    if (!name || !prefix)
        return false;
    while (*prefix) {
        if (!*name)
            return false;
        const auto left = static_cast<unsigned char>(*name++);
        const auto right = static_cast<unsigned char>(*prefix++);
        if (std::tolower(left) != std::tolower(right))
            return false;
    }
    return true;
}

static const char* canonical_device_for_backend(const char* name) {
    if (backend_name_starts_with(name, "cuda"))
        return "cuda";
    if (backend_name_starts_with(name, "metal"))
        return "mps";
    if (backend_name_starts_with(name, "cpu"))
        return "cpu";
    return "";
}

CRISPASR_GEM_EXPORT int crispasr_last_error_kind() {
    return static_cast<int>(crispasr_thread_error.kind);
}

CRISPASR_GEM_EXPORT const char* crispasr_last_error_message() {
    return crispasr_thread_error.message;
}

// Resolve only the public devices supported by the Ruby/Python contract.
// Merely compiling a CUDA or Metal backend is insufficient: the corresponding
// ggml device must be registered on this machine. Returning an empty string for
// an unavailable explicit accelerator prevents CrispASR's useful command-line
// fallback from becoming a silent API contract violation in the gem.
CRISPASR_GEM_EXPORT const char* crispasr_runtime_resolve_device(const char* requested) {
    const std::string request = requested ? requested : "auto";
    if (request == "cpu")
        return "cpu";
    if (request != "auto" && request != "cuda" && request != "mps")
        return "";

    ggml_backend_load_all();
    bool cuda_available = false;
    bool metal_available = false;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const auto type = ggml_backend_dev_type(device);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU)
            continue;
        const char* canonical = canonical_device_for_backend(ggml_backend_dev_name(device));
        cuda_available = cuda_available || std::strcmp(canonical, "cuda") == 0;
        metal_available = metal_available || std::strcmp(canonical, "mps") == 0;
    }

    if (request == "cuda")
        return cuda_available ? "cuda" : "";
    if (request == "mps")
        return metal_available ? "mps" : "";
    if (cuda_available)
        return "cuda";
    if (metal_available)
        return "mps";
    return "cpu";
}

CRISPASR_GEM_EXPORT int crispasr_runtime_supports_bf16(const char* requested) {
    const char* resolved = crispasr_runtime_resolve_device(requested);
    if (!resolved || resolved[0] == '\0' || std::strcmp(resolved, "cpu") == 0)
        return 0;

    ggml_backend_dev_t selected = nullptr;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(index);
        const char* canonical = canonical_device_for_backend(ggml_backend_dev_name(candidate));
        if (std::strcmp(canonical, resolved) == 0) {
            selected = candidate;
            break;
        }
    }
    if (!selected)
        return 0;

    // Ask the selected backend about an actual BF16 graph node. This follows
    // Metal's per-device `has_bfloat` capability and CUDA's operation support
    // without linking the embedding ABI to either platform SDK.
    ggml_init_params params = {};
    params.mem_size = 16 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context* context = ggml_init(params);
    if (!context)
        return 0;
    ggml_tensor* input = ggml_new_tensor_1d(context, GGML_TYPE_BF16, 16);
    ggml_tensor* operation = input ? ggml_dup(context, input) : nullptr;
    const bool supported = operation && ggml_backend_dev_supports_op(selected, operation);
    ggml_free(context);
    return supported ? 1 : 0;
}

// Keep Dense checkpoint conversion vectorized without exposing ggml's broad
// internal ABI. These three wrappers are the only numeric helpers consumed by
// the Ruby Safetensors streamer.
CRISPASR_GEM_EXPORT void crispasr_bf16_to_fp32_row(const void* source, void* destination, int64_t count) {
    ggml_bf16_to_fp32_row(
        static_cast<const ggml_bf16_t*>(source),
        static_cast<float*>(destination),
        count);
}

CRISPASR_GEM_EXPORT void crispasr_fp16_to_fp32_row(const void* source, void* destination, int64_t count) {
    ggml_fp16_to_fp32_row(
        static_cast<const ggml_fp16_t*>(source),
        static_cast<float*>(destination),
        count);
}

CRISPASR_GEM_EXPORT void crispasr_fp32_to_fp16_row(const void* source, void* destination, int64_t count) {
    ggml_fp32_to_fp16_row(
        static_cast<const float*>(source),
        static_cast<ggml_fp16_t*>(destination),
        count);
}

CRISPASR_GEM_EXPORT void crispasr_fp32_to_bf16_row(const void* source, void* destination, int64_t count) {
    ggml_fp32_to_bf16_row(
        static_cast<const float*>(source),
        static_cast<ggml_bf16_t*>(destination),
        count);
}

static bool punctuation_only(const std::string& value) {
    if (value.empty())
        return false;
    for (char character : value) {
        const unsigned char byte = static_cast<unsigned char>(character);
        if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z') ||
            (byte >= '0' && byte <= '9') || byte >= 0x80) {
            return false;
        }
    }
    return true;
}

// Preserve CrispASR's SentencePiece word grouping: a leading space starts a
// word, continuation pieces append, and punctuation attaches to the previous
// word. Confidence is the mean probability of the contributing tokens.
static std::vector<crispasr_session_word> group_words(const std::vector<token_record>& tokens) {
    std::vector<crispasr_session_word> output;
    crispasr_session_word current;
    bool has_current = false;
    float probability_sum = 0.0f;
    int probability_count = 0;

    auto flush = [&]() {
        current.p = probability_count > 0 ? probability_sum / static_cast<float>(probability_count) : 1.0f;
        output.push_back(std::move(current));
        current = {};
        has_current = false;
        probability_sum = 0.0f;
        probability_count = 0;
    };

    for (const auto& token : tokens) {
        if (token.text.empty())
            continue;
        if (token.text == " ") {
            if (has_current)
                flush();
            continue;
        }

        const bool leading_space = token.text.front() == ' ';
        if (leading_space && !punctuation_only(token.text) && has_current)
            flush();

        if (!has_current) {
            current.t0 = token.t0;
            has_current = true;
        }
        current.t1 = token.t1;
        probability_sum += token.p;
        probability_count += 1;
        current.text += leading_space ? token.text.substr(1) : token.text;
    }

    if (has_current)
        flush();
    return output;
}

static std::vector<token_record> timed_token_records(
    const cohere_token_renderer::render_result& rendered,
    int n_samples) {
    size_t total_bytes = 0;
    for (const auto& token : rendered.tokens)
        total_bytes += token.text.size();

    const int64_t duration_cs = static_cast<int64_t>(static_cast<double>(n_samples) * 100.0 / 16000.0);
    size_t byte_offset = 0;
    std::vector<token_record> records;
    records.reserve(rendered.tokens.size());
    for (const auto& token : rendered.tokens) {
        const int64_t t0 = total_bytes == 0
                               ? 0
                               : static_cast<int64_t>(static_cast<double>(byte_offset) / total_bytes * duration_cs);
        byte_offset += token.text.size();
        const int64_t t1 = total_bytes == 0
                               ? duration_cs
                               : static_cast<int64_t>(static_cast<double>(byte_offset) / total_bytes * duration_cs);
        records.push_back({token.text, t0, t1, token.probability});
    }
    return records;
}

CRISPASR_GEM_EXPORT void crispasr_set_gpu_backend(const char* name) {
    crispasr_set_gpu_backend_pref(name);
}

CRISPASR_GEM_EXPORT crispasr_session* crispasr_session_open_with_params(
    const char* model_path,
    const char* backend_name,
    const crispasr_open_params_v1* params) {
    crispasr_clear_error();
    if (!model_path || model_path[0] == '\0')
        return crispasr_fail<crispasr_session>(
            crispasr_error_kind::invalid_argument,
            "model path must not be empty");
    if (backend_name && backend_name[0] != '\0' && std::strcmp(backend_name, "cohere") != 0)
        return crispasr_fail<crispasr_session>(
            crispasr_error_kind::invalid_argument,
            "only the Cohere backend is supported");

    int n_threads = 4;
    bool use_gpu = true;
    bool flash_attn = true;
    int verbosity = 0;
    if (params && params->abi_version >= 1) {
        n_threads = params->n_threads > 0 ? params->n_threads : 4;
        use_gpu = params->use_gpu != 0;
        verbosity = params->verbosity;
        if (params->abi_version >= 2)
            flash_attn = params->flash_attn != 0;
    }

    try {
        auto session = std::make_unique<crispasr_session>();
        session->n_threads = n_threads;
        cohere_context_params cohere_params = cohere_context_default_params();
        cohere_params.n_threads = n_threads;
        cohere_params.use_gpu = use_gpu;
        cohere_params.use_flash = flash_attn;
        cohere_params.verbosity = verbosity;
        session->cohere_ctx = cohere_init_from_file(model_path, cohere_params);
        if (!session->cohere_ctx) {
            crispasr_import_cohere_error(
                crispasr_error_kind::runtime,
                "could not initialize the Cohere model");
            return nullptr;
        }
        cohere_set_abort_callback(
            session->cohere_ctx,
            crispasr_session_abort_requested,
            session.get());
        return session.release();
    } catch (const std::bad_alloc&) {
        return crispasr_fail<crispasr_session>(
            crispasr_error_kind::out_of_memory,
            "could not allocate the Cohere session");
    } catch (const std::exception& error) {
        return crispasr_fail<crispasr_session>(crispasr_error_kind::invariant, error.what());
    } catch (...) {
        return crispasr_fail<crispasr_session>(
            crispasr_error_kind::invariant,
            "unknown exception while opening the Cohere session");
    }
}

CRISPASR_GEM_EXPORT const char* crispasr_session_backend(crispasr_session* session) {
    return session ? session->backend.c_str() : "";
}

CRISPASR_GEM_EXPORT const char* crispasr_session_compute_backend(crispasr_session* session) {
    return session && session->cohere_ctx ? cohere_backend_name(session->cohere_ctx) : "";
}

CRISPASR_GEM_EXPORT int crispasr_session_memory(
    crispasr_session* session,
    uint64_t* free_bytes,
    uint64_t* total_bytes) {
    if (free_bytes)
        *free_bytes = 0;
    if (total_bytes)
        *total_bytes = 0;
    if (!session || !session->cohere_ctx || !free_bytes || !total_bytes)
        return -1;
    size_t available = 0;
    size_t total = 0;
    cohere_backend_memory(session->cohere_ctx, &available, &total);
    *free_bytes = static_cast<uint64_t>(available);
    *total_bytes = static_cast<uint64_t>(total);
    return total > 0 ? 0 : -1;
}

CRISPASR_GEM_EXPORT int crispasr_session_batch_capacity(crispasr_session* session) {
    return session && session->cohere_ctx ? crispasr_session_batch_capacity_value() : 0;
}

// Request cancellation only for the operation that is active at this instant.
// An idle request is deliberately a no-op, so it cannot poison the next call.
CRISPASR_GEM_EXPORT int crispasr_session_cancel(crispasr_session* session) {
    if (!session)
        return -1;
    auto expected = crispasr_operation_state::running;
    if (session->operation_state.compare_exchange_strong(
            expected,
            crispasr_operation_state::cancel_requested,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return 1;
    }
    return expected == crispasr_operation_state::cancel_requested ? 1 : 0;
}

CRISPASR_GEM_EXPORT crispasr_session_result* crispasr_session_transcribe_lang(
    crispasr_session* session,
    const float* pcm,
    int n_samples,
    const char* language) {
    crispasr_clear_error();
    if (!session || !session->cohere_ctx || !pcm || n_samples <= 0)
        return crispasr_fail<crispasr_session_result>(
            crispasr_error_kind::invalid_argument,
            "invalid single-row transcription arguments");
    crispasr_operation_guard operation(session);
    if (!operation)
        return crispasr_fail<crispasr_session_result>(
            crispasr_error_kind::invalid_argument,
            "the Cohere session already has an active inference operation");
    if (crispasr_session_abort_requested(session))
        return crispasr_cancelled<crispasr_session_result>();

    try {
        cohere_set_max_new_tokens(session->cohere_ctx, session->max_new_tokens);
        cohere_set_beam_size(session->cohere_ctx, session->beam_size);
        cohere_set_repetition_loop_guard(session->cohere_ctx, session->repetition_loop_guard);
        const char* source_language = language && language[0] != '\0' ? language : "en";
        std::unique_ptr<cohere_result, decltype(&cohere_result_free)> native_result(
            cohere_transcribe_ex(session->cohere_ctx, pcm, n_samples, source_language, 0),
            &cohere_result_free);
        if (!native_result) {
            if (crispasr_session_abort_requested(session))
                return crispasr_cancelled<crispasr_session_result>();
            crispasr_import_cohere_error(
                crispasr_error_kind::runtime,
                "single-row Cohere inference failed");
            return nullptr;
        }

        auto result = std::make_unique<crispasr_session_result>();
        crispasr_session_segment segment;
        segment.text = native_result->text ? native_result->text : "";
        segment.t0 = 0;
        segment.t1 = static_cast<int64_t>(static_cast<double>(n_samples) * 100.0 / 16000.0);

        std::vector<token_record> tokens;
        if (native_result->tokens && native_result->n_tokens > 0) {
            tokens.reserve(static_cast<size_t>(native_result->n_tokens));
            for (int i = 0; i < native_result->n_tokens; ++i) {
                const auto& token = native_result->tokens[i];
                tokens.push_back({token.text, token.t0, token.t1, token.p});
            }
        }
        segment.words = group_words(tokens);
        result->segments.push_back(std::move(segment));
        result->generated_tokens = native_result->generated_tokens;
        result->generation_limit = native_result->generation_limit;
        result->generation_capacity = native_result->generation_capacity;
        result->stopped_by_max_tokens = native_result->stopped_by_max_tokens;
        result->repetition_stopped = native_result->repetition_stopped;
        if (crispasr_session_abort_requested(session))
            return crispasr_cancelled<crispasr_session_result>();
        return result.release();
    } catch (const std::bad_alloc&) {
        return crispasr_fail<crispasr_session_result>(
            crispasr_error_kind::out_of_memory,
            "could not allocate the single-row transcription result");
    } catch (const std::exception& error) {
        return crispasr_fail<crispasr_session_result>(crispasr_error_kind::invariant, error.what());
    } catch (...) {
        return crispasr_fail<crispasr_session_result>(
            crispasr_error_kind::invariant,
            "unknown exception during single-row transcription");
    }
}

CRISPASR_GEM_EXPORT crispasr_session_batch_result* crispasr_session_transcribe_batch_lang(
    crispasr_session* session,
    const float* const* pcm,
    const int* n_samples,
    int n_batch,
    const char* language) {
    crispasr_clear_error();
    // The encoder is microbatched below at its independent eight-lane limit;
    // the logical call is therefore bounded by the decoder's 24-lane cache
    // layout rather than by one encoder graph.
    if (!session || !session->cohere_ctx || !pcm || !n_samples || n_batch < 1 ||
        n_batch > crispasr_session_batch_capacity_value())
        return crispasr_fail<crispasr_session_batch_result>(
            crispasr_error_kind::invalid_argument,
            "invalid batch transcription arguments");
    for (int lane = 0; lane < n_batch; ++lane) {
        if (!pcm[lane] || n_samples[lane] <= 0 || n_samples[lane] > 35 * 16000)
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invalid_argument,
                "batch audio row is empty or exceeds 35 seconds");
    }
    crispasr_operation_guard operation(session);
    if (!operation)
        return crispasr_fail<crispasr_session_batch_result>(
            crispasr_error_kind::invalid_argument,
            "the Cohere session already has an active inference operation");
    if (crispasr_session_abort_requested(session))
        return crispasr_cancelled<crispasr_session_batch_result>();
    const int64_t total_start_us = ggml_time_us();
    crispasr_session_batch_stats batch_stats{};

    struct alignment_restore {
        cohere_context* context;
        ~alignment_restore() { cohere_set_alignment_collection(context, true); }
    } restore{session->cohere_ctx};

    try {
        cohere_set_max_new_tokens(session->cohere_ctx, session->max_new_tokens);
        cohere_set_beam_size(session->cohere_ctx, 1);
        // The independent batch decoder intentionally omits cross-attention
        // collection. Word-aligned calls stay on the single-row path.
        cohere_set_alignment_collection(session->cohere_ctx, false);

        struct feature_lane {
            int n_mels = 0;
            int T_mel = 0;
            std::vector<float> values;
        };
        std::vector<feature_lane> features(static_cast<size_t>(n_batch));
        enum class feature_failure : int {
            none = 0,
            cancelled,
            compute,
            dimensions,
            overflow,
        };
        std::atomic<int> next_feature_lane{0};
        std::atomic<int64_t> feature_worker_us{0};
        std::atomic<feature_failure> feature_failure_kind{feature_failure::none};
        std::exception_ptr feature_exception;
        std::mutex feature_exception_mutex;
        auto record_feature_failure = [&](feature_failure failure) {
            feature_failure expected = feature_failure::none;
            feature_failure_kind.compare_exchange_strong(expected, failure);
        };
        auto compute_feature_lanes = [&]() {
            try {
                for (;;) {
                    if (feature_failure_kind.load() != feature_failure::none)
                        return;
                    if (crispasr_session_abort_requested(session)) {
                        record_feature_failure(feature_failure::cancelled);
                        return;
                    }
                    const int lane = next_feature_lane.fetch_add(1);
                    if (lane >= n_batch)
                        return;

                    const int64_t feature_lane_start_us = ggml_time_us();
                    feature_lane feature;
                    std::unique_ptr<float, decltype(&std::free)> raw(
                        cohere_compute_mel(
                            session->cohere_ctx,
                            pcm[lane],
                            n_samples[lane],
                            &feature.n_mels,
                            &feature.T_mel),
                        &std::free);
                    if (!raw) {
                        record_feature_failure(
                            crispasr_session_abort_requested(session)
                                ? feature_failure::cancelled
                                : feature_failure::compute);
                        return;
                    }
                    if (feature.n_mels <= 0 || feature.T_mel <= 0) {
                        record_feature_failure(feature_failure::dimensions);
                        return;
                    }
                    if (static_cast<size_t>(feature.T_mel) >
                        SIZE_MAX / static_cast<size_t>(feature.n_mels)) {
                        record_feature_failure(feature_failure::overflow);
                        return;
                    }
                    const size_t value_count =
                        static_cast<size_t>(feature.n_mels) * feature.T_mel;
                    feature.values.assign(raw.get(), raw.get() + value_count);
                    features[static_cast<size_t>(lane)] = std::move(feature);
                    feature_worker_us.fetch_add(
                        ggml_time_us() - feature_lane_start_us,
                        std::memory_order_relaxed);
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(feature_exception_mutex);
                    if (!feature_exception)
                        feature_exception = std::current_exception();
                }
                record_feature_failure(feature_failure::compute);
            }
        };

        int feature_worker_count = 1;
#if !defined(_OPENMP)
        // The packaged build does not use OpenMP, so each frontend call is
        // serial and independent rows can share the session's bounded CPU
        // budget. OpenMP builds retain one outer worker because their frontend
        // already distributes frames across the configured thread count.
        feature_worker_count = std::min(n_batch, std::max(1, session->n_threads));
#endif
        std::vector<std::jthread> feature_workers;
        const int64_t feature_wall_start_us = ggml_time_us();
        feature_workers.reserve(static_cast<size_t>(feature_worker_count - 1));
        for (int worker = 1; worker < feature_worker_count; ++worker)
            feature_workers.emplace_back(compute_feature_lanes);
        compute_feature_lanes();
        for (auto& worker : feature_workers)
            worker.join();
        batch_stats.feature_wall_us = ggml_time_us() - feature_wall_start_us;
        batch_stats.feature_worker_us = feature_worker_us.load(std::memory_order_relaxed);

        if (feature_exception)
            std::rethrow_exception(feature_exception);
        if (crispasr_session_abort_requested(session))
            return crispasr_cancelled<crispasr_session_batch_result>();
        switch (feature_failure_kind.load()) {
        case feature_failure::none:
            break;
        case feature_failure::cancelled:
            return crispasr_cancelled<crispasr_session_batch_result>();
        case feature_failure::compute:
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::runtime,
                "mel feature extraction failed");
        case feature_failure::dimensions:
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "mel feature extraction returned invalid dimensions");
        case feature_failure::overflow:
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "mel feature dimensions overflow size_t");
        }

        const int n_mels = features.front().n_mels;
        int T_mel_max = 0;
        for (const auto& feature : features) {
            if (feature.n_mels != n_mels)
                return crispasr_fail<crispasr_session_batch_result>(
                    crispasr_error_kind::invariant,
                    "mel feature dimensions differ across batch rows");
            T_mel_max = std::max(T_mel_max, feature.T_mel);
        }

        const int logical_T_enc_max = cohere_encoder_padded::encoder_length(T_mel_max);
        if (logical_T_enc_max < 1 || logical_T_enc_max > cohere_decoder_batch::max_encoder_frames)
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "encoder length is outside the decoder batch layout");
        std::vector<int> valid_T_enc(static_cast<size_t>(n_batch));
        int d_model = 0;
        std::vector<float> encoded_states;
        for (int first_lane = 0; first_lane < n_batch; first_lane += cohere_encoder_padded::max_batch) {
            if (crispasr_session_abort_requested(session))
                return crispasr_cancelled<crispasr_session_batch_result>();
            const int physical_batch = std::min(cohere_encoder_padded::max_batch, n_batch - first_lane);
            int physical_T_mel_max = 0;
            for (int physical_lane = 0; physical_lane < physical_batch; ++physical_lane) {
                const auto& feature = features[static_cast<size_t>(first_lane + physical_lane)];
                physical_T_mel_max = std::max(physical_T_mel_max, feature.T_mel);
            }
            if (physical_T_mel_max < 1 ||
                static_cast<size_t>(physical_T_mel_max) >
                    SIZE_MAX / static_cast<size_t>(n_mels) / static_cast<size_t>(physical_batch) / sizeof(float)) {
                return crispasr_fail<crispasr_session_batch_result>(
                    crispasr_error_kind::invariant,
                    "encoder microbatch dimensions overflow size_t");
            }

            const size_t mel_lane_stride = static_cast<size_t>(physical_T_mel_max) * n_mels;
            const int64_t mel_pack_start_us = ggml_time_us();
            std::vector<float> packed_mel(static_cast<size_t>(physical_batch) * mel_lane_stride, 0.0f);
            std::vector<int> physical_T_mel(static_cast<size_t>(physical_batch));
            for (int physical_lane = 0; physical_lane < physical_batch; ++physical_lane) {
                const auto& feature = features[static_cast<size_t>(first_lane + physical_lane)];
                physical_T_mel[static_cast<size_t>(physical_lane)] = feature.T_mel;
                std::copy(
                    feature.values.begin(),
                    feature.values.end(),
                    packed_mel.begin() + static_cast<size_t>(physical_lane) * mel_lane_stride);
            }
            batch_stats.mel_pack_us += ggml_time_us() - mel_pack_start_us;

            std::vector<int> physical_T_enc(static_cast<size_t>(physical_batch));
            int physical_T_enc_max = 0;
            int physical_d_model = 0;
            cohere_encoder_batch_stats encoder_stats{};
            std::unique_ptr<float, decltype(&std::free)> encoded(
                cohere_run_encoder_batch_padded(
                    session->cohere_ctx,
                    packed_mel.data(),
                    physical_T_mel.data(),
                    physical_batch,
                    n_mels,
                    physical_T_mel_max,
                    physical_T_enc.data(),
                    &physical_T_enc_max,
                    &physical_d_model,
                    &encoder_stats),
                &std::free);
            batch_stats.encoder_microbatches++;
            batch_stats.encoder_graph_build_us += encoder_stats.graph_build_us;
            batch_stats.encoder_graph_alloc_us += encoder_stats.graph_alloc_us;
            batch_stats.encoder_input_us += encoder_stats.input_us;
            batch_stats.encoder_compute_us += encoder_stats.compute_us;
            batch_stats.encoder_readback_us += encoder_stats.readback_us;
            if (!encoded) {
                if (crispasr_session_abort_requested(session))
                    return crispasr_cancelled<crispasr_session_batch_result>();
                crispasr_import_cohere_error(
                    crispasr_error_kind::runtime,
                    "padded encoder microbatch failed");
                return nullptr;
            }
            if (physical_T_enc_max <= 0 || physical_T_enc_max > logical_T_enc_max ||
                physical_d_model <= 0 || (d_model > 0 && physical_d_model != d_model))
                return crispasr_fail<crispasr_session_batch_result>(
                    crispasr_error_kind::invariant,
                    "padded encoder returned inconsistent dimensions");

            const int64_t encoder_repack_start_us = ggml_time_us();
            if (d_model == 0) {
                d_model = physical_d_model;
                if (static_cast<size_t>(n_batch) >
                    SIZE_MAX / static_cast<size_t>(logical_T_enc_max) /
                        static_cast<size_t>(d_model) / sizeof(float)) {
                    return crispasr_fail<crispasr_session_batch_result>(
                        crispasr_error_kind::invariant,
                        "logical encoder batch dimensions overflow size_t");
                }
                encoded_states.assign(
                    static_cast<size_t>(n_batch) * logical_T_enc_max * d_model,
                    0.0f);
            }

            const size_t source_lane_stride = static_cast<size_t>(physical_T_enc_max) * d_model;
            const size_t destination_lane_stride = static_cast<size_t>(logical_T_enc_max) * d_model;
            for (int physical_lane = 0; physical_lane < physical_batch; ++physical_lane) {
                const int valid = physical_T_enc[static_cast<size_t>(physical_lane)];
                if (valid < 1 || valid > physical_T_enc_max)
                    return crispasr_fail<crispasr_session_batch_result>(
                        crispasr_error_kind::invariant,
                        "padded encoder returned an invalid row length");
                const int logical_lane = first_lane + physical_lane;
                valid_T_enc[static_cast<size_t>(logical_lane)] = valid;
                std::copy_n(
                    encoded.get() + static_cast<size_t>(physical_lane) * source_lane_stride,
                    static_cast<size_t>(valid) * d_model,
                    encoded_states.begin() + static_cast<size_t>(logical_lane) * destination_lane_stride);
            }
            batch_stats.encoder_repack_us += ggml_time_us() - encoder_repack_start_us;
            if (crispasr_session_abort_requested(session))
                return crispasr_cancelled<crispasr_session_batch_result>();
        }
        if (d_model <= 0 || encoded_states.empty())
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "padded encoder produced no logical batch state");

        std::unique_ptr<cohere_decoder_batch_generate_result,
                        decltype(&cohere_decoder_batch_generate_result_free)>
            generated(
                cohere_run_decoder_batch_generate_ex(
                    session->cohere_ctx,
                    encoded_states.data(),
                    valid_T_enc.data(),
                    n_batch,
                    logical_T_enc_max,
                    d_model,
                    language && language[0] != '\0' ? language : "en",
                    session->max_new_tokens,
                    // Ruby publishes the selected token stream, not the decoder's
                    // per-token softmax values. Reduce logits on the compute backend
                    // so each step transfers one token ID per lane instead of the
                    // complete batch-by-vocabulary matrix.
                    COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX),
                &cohere_decoder_batch_generate_result_free);
        if (!generated) {
            if (crispasr_session_abort_requested(session))
                return crispasr_cancelled<crispasr_session_batch_result>();
            crispasr_import_cohere_error(
                crispasr_error_kind::runtime,
                "ragged decoder batch failed");
            return nullptr;
        }
        if (generated->n_batch != n_batch || !generated->sequences)
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "ragged decoder returned an inconsistent batch result");
        batch_stats.decoder_total_us = generated->stats.total_us;
        batch_stats.decoder_cross_kv_us = generated->stats.cross_kv_us;
        batch_stats.decoder_reserve_us = generated->stats.reserve_us;
        batch_stats.decoder_decode_us = generated->stats.decode_us;
        batch_stats.decoder_calls = generated->stats.decode_calls;
        batch_stats.generation_steps = generated->stats.generation_steps;
        batch_stats.token_id_readback_bytes = static_cast<int64_t>(generated->stats.token_id_readback_bytes);

        const int64_t render_start_us = ggml_time_us();
        const int vocabulary_size = cohere_n_vocab(session->cohere_ctx);
        if (vocabulary_size <= 0)
            return crispasr_fail<crispasr_session_batch_result>(
                crispasr_error_kind::invariant,
                "model vocabulary is empty");
        std::vector<std::string> vocabulary(static_cast<size_t>(vocabulary_size));
        for (int token = 0; token < vocabulary_size; ++token) {
            const char* piece = cohere_token_to_str(session->cohere_ctx, token);
            if (piece)
                vocabulary[static_cast<size_t>(token)] = piece;
        }
        const cohere_token_renderer::speaker_token_ids speaker_tokens = {
            cohere_str_to_token(session->cohere_ctx, "<|spkchange|>"),
            cohere_str_to_token(session->cohere_ctx, "<|spk0|>"),
        };

        auto output = std::make_unique<crispasr_session_batch_result>();
        output->lanes.reserve(static_cast<size_t>(n_batch));
        for (int lane = 0; lane < n_batch; ++lane) {
            if (crispasr_session_abort_requested(session))
                return crispasr_cancelled<crispasr_session_batch_result>();
            const auto& sequence = generated->sequences[lane];
            if (sequence.n_tokens < 0 ||
                (sequence.n_tokens > 0 && (!sequence.token_ids || !sequence.probabilities))) {
                return crispasr_fail<crispasr_session_batch_result>(
                    crispasr_error_kind::invariant,
                    "ragged decoder returned an invalid token sequence");
            }

            std::vector<int> token_ids;
            std::vector<float> probabilities;
            if (sequence.n_tokens > 0) {
                token_ids.assign(sequence.token_ids, sequence.token_ids + sequence.n_tokens);
                probabilities.assign(sequence.probabilities, sequence.probabilities + sequence.n_tokens);
            }

            bool repetition_stopped = false;
            if (session->repetition_loop_guard) {
                const size_t stop_length = core_repetition_loop::first_trigger_length(token_ids);
                repetition_stopped = stop_length > 0;
                if (repetition_stopped) {
                    token_ids.resize(stop_length);
                    probabilities.resize(stop_length);
                }
            }

            const auto rendered = cohere_token_renderer::render(
                token_ids, probabilities, vocabulary, speaker_tokens);
            if (!rendered.ok())
                return crispasr_fail<crispasr_session_batch_result>(
                    crispasr_error_kind::invariant,
                    "token renderer rejected the ragged decoder output");

            auto result = std::make_unique<crispasr_session_result>();
            crispasr_session_segment segment;
            segment.text = rendered.full_text;
            segment.t0 = 0;
            segment.t1 = static_cast<int64_t>(static_cast<double>(n_samples[lane]) * 100.0 / 16000.0);
            segment.words = group_words(timed_token_records(rendered, n_samples[lane]));
            result->segments.push_back(std::move(segment));
            const bool saw_eos = sequence.stop_reason == COHERE_GENERATION_STOP_EOS;
            result->generated_tokens = static_cast<int>(token_ids.size()) + (saw_eos && !repetition_stopped ? 1 : 0);
            result->generation_limit = sequence.stop_reason == COHERE_GENERATION_STOP_MAX_TOKENS
                                           ? sequence.n_tokens
                                           : session->max_new_tokens;
            result->generation_capacity = generated->generation_capacity;
            result->stopped_by_max_tokens =
                !repetition_stopped && sequence.stop_reason == COHERE_GENERATION_STOP_MAX_TOKENS;
            result->repetition_stopped = repetition_stopped;
            output->lanes.push_back(std::move(result));
        }
        if (crispasr_session_abort_requested(session))
            return crispasr_cancelled<crispasr_session_batch_result>();
        batch_stats.render_us = ggml_time_us() - render_start_us;
        batch_stats.total_us = ggml_time_us() - total_start_us;
        output->stats = batch_stats;
        return output.release();
    } catch (const std::bad_alloc&) {
        return crispasr_fail<crispasr_session_batch_result>(
            crispasr_error_kind::out_of_memory,
            "could not allocate the batch transcription result");
    } catch (const std::exception& error) {
        return crispasr_fail<crispasr_session_batch_result>(crispasr_error_kind::invariant, error.what());
    } catch (...) {
        return crispasr_fail<crispasr_session_batch_result>(
            crispasr_error_kind::invariant,
            "unknown exception during batch transcription");
    }
}

CRISPASR_GEM_EXPORT int crispasr_session_batch_result_count(crispasr_session_batch_result* result) {
    return result ? static_cast<int>(result->lanes.size()) : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_batch_result_stats_v1(
    crispasr_session_batch_result* result,
    int64_t* values,
    int capacity) {
    constexpr int64_t abi_version = 1;
    constexpr int64_t field_count = 21;
    if (!result || !values || capacity <= 0)
        return field_count;
    const auto& stats = result->stats;
    const int64_t fields[field_count] = {
        abi_version,
        field_count,
        stats.total_us,
        stats.feature_wall_us,
        stats.feature_worker_us,
        stats.mel_pack_us,
        stats.encoder_graph_build_us,
        stats.encoder_graph_alloc_us,
        stats.encoder_input_us,
        stats.encoder_compute_us,
        stats.encoder_readback_us,
        stats.encoder_repack_us,
        stats.decoder_total_us,
        stats.decoder_cross_kv_us,
        stats.decoder_reserve_us,
        stats.decoder_decode_us,
        stats.render_us,
        stats.decoder_calls,
        stats.generation_steps,
        stats.encoder_microbatches,
        stats.token_id_readback_bytes,
    };
    std::copy_n(fields, std::min(capacity, static_cast<int>(field_count)), values);
    return field_count;
}

CRISPASR_GEM_EXPORT crispasr_session_result* crispasr_session_batch_result_at(
    crispasr_session_batch_result* result,
    int lane) {
    return result && lane >= 0 && lane < static_cast<int>(result->lanes.size())
               ? result->lanes[static_cast<size_t>(lane)].get()
               : nullptr;
}

CRISPASR_GEM_EXPORT void crispasr_session_batch_result_free(crispasr_session_batch_result* result) {
    delete result;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_n_segments(crispasr_session_result* result) {
    return result ? static_cast<int>(result->segments.size()) : 0;
}

CRISPASR_GEM_EXPORT const char* crispasr_session_result_segment_text(crispasr_session_result* result, int index) {
    return result && index >= 0 && index < static_cast<int>(result->segments.size())
               ? result->segments[static_cast<size_t>(index)].text.c_str()
               : "";
}

CRISPASR_GEM_EXPORT int64_t crispasr_session_result_segment_t0(crispasr_session_result* result, int index) {
    return result && index >= 0 && index < static_cast<int>(result->segments.size())
               ? result->segments[static_cast<size_t>(index)].t0
               : 0;
}

CRISPASR_GEM_EXPORT int64_t crispasr_session_result_segment_t1(crispasr_session_result* result, int index) {
    return result && index >= 0 && index < static_cast<int>(result->segments.size())
               ? result->segments[static_cast<size_t>(index)].t1
               : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_n_words(crispasr_session_result* result, int segment_index) {
    if (!result || segment_index < 0 || segment_index >= static_cast<int>(result->segments.size()))
        return 0;
    return static_cast<int>(result->segments[static_cast<size_t>(segment_index)].words.size());
}

static const crispasr_session_word* result_word(
    crispasr_session_result* result,
    int segment_index,
    int word_index) {
    if (!result || segment_index < 0 || segment_index >= static_cast<int>(result->segments.size()))
        return nullptr;
    const auto& words = result->segments[static_cast<size_t>(segment_index)].words;
    if (word_index < 0 || word_index >= static_cast<int>(words.size()))
        return nullptr;
    return &words[static_cast<size_t>(word_index)];
}

CRISPASR_GEM_EXPORT const char* crispasr_session_result_word_text(
    crispasr_session_result* result,
    int segment_index,
    int word_index) {
    const auto* word = result_word(result, segment_index, word_index);
    return word ? word->text.c_str() : "";
}

CRISPASR_GEM_EXPORT int64_t crispasr_session_result_word_t0(
    crispasr_session_result* result,
    int segment_index,
    int word_index) {
    const auto* word = result_word(result, segment_index, word_index);
    return word ? word->t0 : 0;
}

CRISPASR_GEM_EXPORT int64_t crispasr_session_result_word_t1(
    crispasr_session_result* result,
    int segment_index,
    int word_index) {
    const auto* word = result_word(result, segment_index, word_index);
    return word ? word->t1 : 0;
}

CRISPASR_GEM_EXPORT float crispasr_session_result_word_p(
    crispasr_session_result* result,
    int segment_index,
    int word_index) {
    const auto* word = result_word(result, segment_index, word_index);
    return word ? word->p : -1.0f;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_generated_tokens(crispasr_session_result* result) {
    return result ? result->generated_tokens : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_generation_limit(crispasr_session_result* result) {
    return result ? result->generation_limit : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_generation_capacity(crispasr_session_result* result) {
    return result ? result->generation_capacity : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_stopped_by_max_tokens(crispasr_session_result* result) {
    return result && result->stopped_by_max_tokens ? 1 : 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_result_repetition_stopped(crispasr_session_result* result) {
    return result && result->repetition_stopped ? 1 : 0;
}

CRISPASR_GEM_EXPORT void crispasr_session_result_free(crispasr_session_result* result) {
    delete result;
}

CRISPASR_GEM_EXPORT void crispasr_session_close(crispasr_session* session) {
    if (!session)
        return;
    cohere_free(session->cohere_ctx);
    delete session;
}

CRISPASR_GEM_EXPORT int crispasr_session_set_max_new_tokens(crispasr_session* session, int count) {
    if (!session)
        return -1;
    session->max_new_tokens = count > 0 ? count : 0;
    cohere_set_max_new_tokens(session->cohere_ctx, session->max_new_tokens);
    return 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_set_beam_size(crispasr_session* session, int width) {
    if (!session)
        return -1;
    session->beam_size = width > 0 ? width : 1;
    cohere_set_beam_size(session->cohere_ctx, session->beam_size);
    return 0;
}

CRISPASR_GEM_EXPORT int crispasr_session_set_repetition_loop_guard(crispasr_session* session, int enabled) {
    if (!session)
        return -1;
    session->repetition_loop_guard = enabled != 0;
    cohere_set_repetition_loop_guard(session->cohere_ctx, session->repetition_loop_guard);
    return 0;
}
