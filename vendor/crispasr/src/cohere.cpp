// cohere.cpp — Cohere Transcribe inference via ggml
//
// Architecture:
//   Encoder: Conv2D subsampling (×8) + 48-layer Conformer (Transformer-XL rel-pos attention)
//   Decoder: 8-layer causal transformer with cross-attention + KV cache
//   Features: deterministic dither → preemphasis → STFT → mel filterbank → log → per-feature norm
//
// Tensor naming follows export_gguf.py / cohere-arch.h.

#include "cohere.h"
#include "cohere-arch.h"
#include "cohere_decoder_batch_layout.h"
#include "cohere_encoder_padded_layout.h"
#include "cohere_frontend.h"
#include "cohere_ragged_controller.h"
#include "core/repetition_loop_guard.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "crispasr_imatrix.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdarg>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __APPLE__
#include <Accelerate/Accelerate.h> // cblas + vDSP, no external deps needed
#else
#endif
#if defined(__F16C__) && defined(__AVX2__)
#include <immintrin.h>
#define CT_HAVE_F16C 1
#endif

// ---------------------------------------------------------------------------
// Logging & Debugging
// ---------------------------------------------------------------------------

struct cohere_error_state {
    enum cohere_error_kind kind = COHERE_ERROR_NONE;
    const char* message = "";
};

static thread_local cohere_error_state cohere_thread_error;

static void cohere_clear_error() noexcept {
    cohere_thread_error = {};
}

static void cohere_set_error(enum cohere_error_kind kind, const char* message) noexcept {
    cohere_thread_error.kind = kind;
    cohere_thread_error.message = message ? message : "";
}

enum cohere_error_kind cohere_last_error_kind(void) {
    return cohere_thread_error.kind;
}

const char* cohere_last_error_message(void) {
    return cohere_thread_error.message;
}

static bool cohere_debug_enabled(void) {
    static const bool enabled = getenv("COHERE_DEBUG") != nullptr;
    return enabled;
}

static bool cohere_bench_enabled(void) {
    static const bool enabled = getenv("COHERE_BENCH") != nullptr;
    return enabled;
}

// ===========================================================================
// Bench instrumentation — `COHERE_BENCH=1` for per-stage timings.
// ===========================================================================

struct cohere_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit cohere_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~cohere_bench_stage() {
        if (!cohere_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  cohere_bench: %-22s %.2f ms\n", name, ms);
    }
};

static void cohere_debug(const char* fmt, ...) {
    if (!cohere_debug_enabled())
        return;
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    fflush(stdout);
}

static void cohere_log_tensor(const char* name, const struct ggml_tensor* t) {
    if (!cohere_debug_enabled())
        return;
    if (!t) {
        cohere_debug("%-25s: NULL\n", name);
        return;
    }
    cohere_debug("%-25s: shape [%4ld, %4ld, %4ld, %4ld] type %d\n", name, (long)t->ne[0], (long)t->ne[1],
                 (long)t->ne[2], (long)t->ne[3], (int)t->type);
}

// Verbosity helpers — use these everywhere instead of bare fprintf(stderr).
// vlog: shown at verbosity >= 1 (model loading info)
// vlog2: shown at verbosity >= 2 (per-inference timing, steps, perf report)
// Always print errors/warnings regardless of verbosity.
// MSVC < C++20 doesn't support __VA_OPT__. Use the classic ##__VA_ARGS__
// extension (supported by GCC, Clang, and MSVC) which swallows the
// trailing comma when __VA_ARGS__ is empty.
#define COHERE_VLOG(v, fmt, ...)                                                                                       \
    do {                                                                                                               \
        if ((v) >= 1)                                                                                                  \
            fprintf(stderr, fmt, ##__VA_ARGS__);                                                                       \
    } while (0)
#define COHERE_VLOG2(v, fmt, ...)                                                                                      \
    do {                                                                                                               \
        if ((v) >= 2)                                                                                                  \
            fprintf(stderr, fmt, ##__VA_ARGS__);                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Performance counters — accumulated per cohere_transcribe() call
// ---------------------------------------------------------------------------

struct cohere_perf {
    int64_t t_features_us = 0;     // STFT + mel filterbank
    int64_t t_enc_build_us = 0;    // encoder graph construction
    int64_t t_enc_alloc_us = 0;    // encoder ggml_backend_sched_alloc_graph
    int64_t t_enc_compute_us = 0;  // encoder ggml_backend_sched_graph_compute
    int64_t t_cross_kv_us = 0;     // copying cross-KV tensors from encoder output
    int64_t t_crosskv_read_us = 0; // GPU->CPU readback of per-chunk cross-KV (#161 probe)
    int64_t t_reserve_us = 0;      // one-time sched reserve of max-ctx decoder graph (#161 probe)
    int64_t t_dec_build_us = 0;    // decoder graph build (all steps summed)
    int64_t t_dec_alloc_us = 0;    // decoder sched alloc (all steps summed)
    int64_t t_dec_compute_us = 0;  // decoder compute (all steps summed)
    int64_t t_dec_logits_us = 0;   // decoder ggml_backend_tensor_get logits (all steps)
    int64_t t_dec_step_min_us = INT64_MAX;
    int64_t t_dec_step_max_us = 0;
    int n_dec_steps = 0;    // total autoregressive steps (prompt + generated)
    int64_t t_total_us = 0; // wall time for entire cohere_transcribe()
    // Graph sizes
    int enc_n_nodes = 0;
    int dec_n_nodes_prompt = 0; // first decode call (full prompt batch)
    int dec_n_nodes_step = 0;   // subsequent calls (n_tok=1)
    // Memory (bytes, snapshot after init / after cross-kv alloc)
    size_t mem_model_buf = 0;
    size_t mem_kv_buf = 0;
    size_t mem_cross_kv_buf = 0;
    size_t mem_sched_buf = 0;
    size_t mem_compute_meta = 0;
};

static void cohere_perf_reset(cohere_perf& p) {
    p = cohere_perf{};
}

static void cohere_perf_print(const cohere_perf& p, int n_samples, int sample_rate, int verbosity) {
    if (verbosity < 2)
        return;
    double audio_sec = (double)n_samples / sample_rate;
    double total_sec = p.t_total_us / 1e6;
    double rtf = (total_sec > 0) ? (audio_sec / total_sec) : 0.0;

    fprintf(stderr, "\n");
    fprintf(stderr, "cohere: =========== performance report ===========\n");
    fprintf(stderr, "cohere:  audio            %.2f s\n", audio_sec);
    fprintf(stderr, "cohere:  total wall       %7.1f ms   (%.3f× real-time)\n", p.t_total_us / 1e3, rtf);
    fprintf(stderr, "cohere: ----- feature extraction -----\n");
    fprintf(stderr, "cohere:  features         %7.1f ms\n", p.t_features_us / 1e3);
    fprintf(stderr, "cohere: ----- encoder -----\n");
    fprintf(stderr, "cohere:  enc graph build  %7.1f ms   nodes=%d\n", p.t_enc_build_us / 1e3, p.enc_n_nodes);
    fprintf(stderr, "cohere:  enc sched alloc  %7.1f ms\n", p.t_enc_alloc_us / 1e3);
    fprintf(stderr, "cohere:  enc compute      %7.1f ms\n", p.t_enc_compute_us / 1e3);
    fprintf(stderr, "cohere:  cross-kv copy    %7.1f ms\n", p.t_cross_kv_us / 1e3);
    fprintf(stderr, "cohere:  enc total        %7.1f ms\n",
            (p.t_enc_build_us + p.t_enc_alloc_us + p.t_enc_compute_us + p.t_cross_kv_us) / 1e3);
    fprintf(stderr, "cohere: ----- decoder (%d steps) -----\n", p.n_dec_steps);
    if (p.n_dec_steps > 0) {
        double dprompt = p.dec_n_nodes_prompt;
        double dstep = p.dec_n_nodes_step;
        fprintf(stderr, "cohere:  dec graph build  %7.1f ms   %.2f ms/step  (nodes: prompt=%d step=%d)\n",
                p.t_dec_build_us / 1e3, p.t_dec_build_us / 1e3 / p.n_dec_steps, p.dec_n_nodes_prompt,
                p.dec_n_nodes_step);
        (void)dprompt;
        (void)dstep;
        fprintf(stderr, "cohere:  dec sched alloc  %7.1f ms   %.2f ms/step\n", p.t_dec_alloc_us / 1e3,
                p.t_dec_alloc_us / 1e3 / p.n_dec_steps);
        fprintf(stderr, "cohere:  dec compute      %7.1f ms   %.2f ms/step  min=%.2f max=%.2f\n",
                p.t_dec_compute_us / 1e3, p.t_dec_compute_us / 1e3 / p.n_dec_steps,
                (p.t_dec_step_min_us == INT64_MAX ? 0 : p.t_dec_step_min_us) / 1e3, p.t_dec_step_max_us / 1e3);
        fprintf(stderr, "cohere:  dec logits get   %7.1f ms   %.2f ms/step\n", p.t_dec_logits_us / 1e3,
                p.t_dec_logits_us / 1e3 / p.n_dec_steps);
        fprintf(stderr, "cohere:  dec total        %7.1f ms\n",
                (p.t_dec_build_us + p.t_dec_alloc_us + p.t_dec_compute_us + p.t_dec_logits_us) / 1e3);
    }
    // #161 probes: host-side work that lives in the gaps between the timed
    // stages above (cross-KV GPU->CPU readback, one-time max-ctx sched
    // reserve, and a UNACCOUNTED residual that catches any remaining gap —
    // e.g. the beam-search KV snapshot that drove the #161 regression).
    // Opt-in via COHERE_GAPS=1 (or COHERE_BENCH=1) to keep the default
    // report compact.
    if (std::getenv("COHERE_GAPS") || std::getenv("COHERE_BENCH")) {
        const int64_t accounted = p.t_features_us + p.t_enc_build_us + p.t_enc_alloc_us + p.t_enc_compute_us +
                                  p.t_cross_kv_us + p.t_crosskv_read_us + p.t_reserve_us + p.t_dec_build_us +
                                  p.t_dec_alloc_us + p.t_dec_compute_us + p.t_dec_logits_us;
        fprintf(stderr, "cohere: ----- untimed gaps (#161) -----\n");
        fprintf(stderr, "cohere:  cross-kv readback%7.1f ms\n", p.t_crosskv_read_us / 1e3);
        fprintf(stderr, "cohere:  sched reserve    %7.1f ms\n", p.t_reserve_us / 1e3);
        fprintf(stderr, "cohere:  UNACCOUNTED     %7.1f ms   <- residual = total - all stages\n",
                (p.t_total_us - accounted) / 1e3);
    }
    fprintf(stderr, "cohere: ----- memory -----\n");
    fprintf(stderr, "cohere:  model weights    %7.1f MiB\n", p.mem_model_buf / 1048576.0);
    fprintf(stderr, "cohere:  kv cache         %7.1f MiB\n", p.mem_kv_buf / 1048576.0);
    fprintf(stderr, "cohere:  cross-kv cache   %7.1f MiB\n", p.mem_cross_kv_buf / 1048576.0);
    fprintf(stderr, "cohere:  sched buf        %7.1f MiB\n", p.mem_sched_buf / 1048576.0);
    fprintf(stderr, "cohere:  compute_meta     %7.1f KiB\n", p.mem_compute_meta / 1024.0);
    fprintf(stderr, "cohere: ================================================\n\n");
}

// ---------------------------------------------------------------------------
// Per-op profiling (COHERE_PROF=1)
// ---------------------------------------------------------------------------

struct cohere_op_prof {
    int64_t t_us = 0;
    int count = 0;
};

struct cohere_prof_state {
    // coarse buckets
    cohere_op_prof mul_mat;   // GGML_OP_MUL_MAT
    cohere_op_prof soft_max;  // GGML_OP_SOFT_MAX
    cohere_op_prof norm;      // GGML_OP_NORM / RMS_NORM
    cohere_op_prof cont;      // GGML_OP_CONT
    cohere_op_prof im2col;    // GGML_OP_IM2COL (part of conv1d_dw expansion)
    cohere_op_prof conv_dw;   // GGML_OP_CONV_2D_DW (direct depthwise conv)
    cohere_op_prof add;       // GGML_OP_ADD / GGML_OP_MUL / GGML_OP_SCALE
    cohere_op_prof other;     // everything else (view, reshape, permute, silu, etc.)
    int64_t t_node_start = 0; // wall time before node execute (ask=true)
};

static bool cohere_prof_eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    auto* ps = (cohere_prof_state*)user_data;
    if (ask) {
        ps->t_node_start = ggml_time_us();
        return true;
    }
    int64_t dt = ggml_time_us() - ps->t_node_start;
    switch (t->op) {
    case GGML_OP_MUL_MAT:
        ps->mul_mat.t_us += dt;
        ps->mul_mat.count++;
        break;
    case GGML_OP_SOFT_MAX:
        ps->soft_max.t_us += dt;
        ps->soft_max.count++;
        break;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
        ps->norm.t_us += dt;
        ps->norm.count++;
        break;
    case GGML_OP_CONT:
        ps->cont.t_us += dt;
        ps->cont.count++;
        break;
    case GGML_OP_IM2COL:
    case GGML_OP_IM2COL_3D:
        ps->im2col.t_us += dt;
        ps->im2col.count++;
        break;
    case GGML_OP_CONV_2D_DW:
        ps->conv_dw.t_us += dt;
        ps->conv_dw.count++;
        break;
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_SCALE:
        ps->add.t_us += dt;
        ps->add.count++;
        break;
    default:
        ps->other.t_us += dt;
        ps->other.count++;
        break;
    }
    return true;
}

static void cohere_prof_print(const cohere_prof_state& ps) {
    auto pct = [&](int64_t v, int64_t total) { return total > 0 ? 100.0 * v / total : 0.0; };
    int64_t total = ps.mul_mat.t_us + ps.soft_max.t_us + ps.norm.t_us + ps.cont.t_us + ps.im2col.t_us +
                    ps.conv_dw.t_us + ps.add.t_us + ps.other.t_us;
    fprintf(stderr, "cohere: -------- encoder op profile (COHERE_PROF) --------\n");
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "mul_mat", ps.mul_mat.t_us / 1e3,
            pct(ps.mul_mat.t_us, total), ps.mul_mat.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "soft_max", ps.soft_max.t_us / 1e3,
            pct(ps.soft_max.t_us, total), ps.soft_max.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "norm", ps.norm.t_us / 1e3, pct(ps.norm.t_us, total),
            ps.norm.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "cont", ps.cont.t_us / 1e3, pct(ps.cont.t_us, total),
            ps.cont.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "im2col", ps.im2col.t_us / 1e3,
            pct(ps.im2col.t_us, total), ps.im2col.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "conv_2d_dw", ps.conv_dw.t_us / 1e3,
            pct(ps.conv_dw.t_us, total), ps.conv_dw.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "add/mul/sc", ps.add.t_us / 1e3,
            pct(ps.add.t_us, total), ps.add.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  %5.1f%%  n=%d\n", "other", ps.other.t_us / 1e3,
            pct(ps.other.t_us, total), ps.other.count);
    fprintf(stderr, "cohere:  %-12s  %7.1f ms  (measured sum; real enc may differ due to overhead)\n", "TOTAL",
            total / 1e3);
    fprintf(stderr, "cohere: -------------------------------------------------------\n");
}

// ---------------------------------------------------------------------------
// Helpers

// Like crispasr's ggml_graph_compute_helper: set n_threads on every backend
// in the scheduler (via registry proc address) before each compute call.
// This ensures the thread count is applied correctly even after sched resets.
static bool cohere_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph* gf, int n_threads) {
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); i++) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto* fn =
                (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn)
                fn(backend, n_threads);
        }
    }
    return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Model hyperparams
// ---------------------------------------------------------------------------

struct cohere_hparams {
    int vocab_size = 16384;
    // encoder
    int enc_n_layers = 48;
    int enc_d_model = 1280;
    int enc_n_heads = 8;
    int enc_head_dim = 160;
    int enc_ffn_dim = 5120;
    int enc_conv_k = 9;
    // decoder
    int dec_n_layers = 8;
    int dec_d_model = 1024;
    int dec_n_heads = 8;
    int dec_head_dim = 128;
    int dec_ffn_dim = 4096;
    int dec_max_ctx = 1024;
    // audio
    int sample_rate = 16000;
    int n_mels = 128;
    int n_fft = 512;
    int hop_length = 160;
    int win_length = 400;
    // derived
    int n_freqs() const { return n_fft / 2 + 1; } // 257
    int pre_conv_ch = 256;
    int pre_sub_fac = 8; // 3 × stride-2 → ×8 downsampling
};

// ---------------------------------------------------------------------------
// Conformer layer weights
// ---------------------------------------------------------------------------

struct cohere_enc_layer {
    // FF1
    ggml_tensor *ff1_norm_w, *ff1_norm_b;
    ggml_tensor *ff1_up_w, *ff1_up_b;
    ggml_tensor *ff1_dn_w, *ff1_dn_b;
    // Self-attention (relative pos)
    ggml_tensor *attn_norm_w, *attn_norm_b;
    ggml_tensor *attn_q_w, *attn_q_b;
    ggml_tensor *attn_k_w, *attn_k_b;
    ggml_tensor *attn_v_w, *attn_v_b;
    ggml_tensor *attn_qkv_w, *attn_qkv_b; // combined
    ggml_tensor *attn_out_w, *attn_out_b;

    ggml_tensor* attn_pos_w;      // linear_pos  [d,d]
    ggml_tensor* attn_pos_bias_u; // [heads, head_dim]
    ggml_tensor* attn_pos_bias_v;
    // Convolution module
    ggml_tensor *conv_norm_w, *conv_norm_b;
    ggml_tensor *conv_pw1_w, *conv_pw1_b; // pointwise1: [2d, d, 1]
    ggml_tensor *conv_dw_w, *conv_dw_b;   // depthwise:  [d, 1, k]
    ggml_tensor *conv_bn_w, *conv_bn_b;   // batch-norm scale/bias
    ggml_tensor *conv_bn_mean, *conv_bn_var;
    ggml_tensor *conv_pw2_w, *conv_pw2_b; // pointwise2: [d, d, 1]
    // FF2
    ggml_tensor *ff2_norm_w, *ff2_norm_b;
    ggml_tensor *ff2_up_w, *ff2_up_b;
    ggml_tensor *ff2_dn_w, *ff2_dn_b;
    // Output norm
    ggml_tensor *out_norm_w, *out_norm_b;

    // BatchNorm-folded depthwise conv weights (precomputed at model load time).
    // w_fused[d] = dw_w[d] * bn_gamma[d] / sqrt(bn_var[d] + 1e-5)
    // b_fused[d] = (dw_b[d] - bn_mean[d]) * bn_gamma[d] / sqrt(bn_var[d] + 1e-5) + bn_beta[d]
    // These are F32 host vectors used by both the legacy and ggml-graph paths.
    std::vector<float> conv_dw_w_fused; // [d * conv_k]
    std::vector<float> conv_dw_b_fused; // [d]
};

// ---------------------------------------------------------------------------
// Decoder layer weights
// ---------------------------------------------------------------------------

struct cohere_dec_layer {
    ggml_tensor *attn_ln_w, *attn_ln_b;
    ggml_tensor *attn_q_w, *attn_q_b;
    ggml_tensor *attn_k_w, *attn_k_b;
    ggml_tensor *attn_v_w, *attn_v_b;
    ggml_tensor *attn_o_w, *attn_o_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b;
    ggml_tensor *cross_k_w, *cross_k_b;
    ggml_tensor *cross_v_w, *cross_v_b;
    ggml_tensor *cross_o_w, *cross_o_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b;
    ggml_tensor *ffn_up_w, *ffn_up_b;
    ggml_tensor *ffn_dn_w, *ffn_dn_b;
};

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

struct cohere_model {
    cohere_hparams hparams;

    // Pre-encode subsampling
    ggml_tensor *pre_conv0_w, *pre_conv0_b;
    ggml_tensor *pre_conv2_w, *pre_conv2_b;
    ggml_tensor *pre_conv3_w, *pre_conv3_b;
    ggml_tensor *pre_conv5_w, *pre_conv5_b;
    ggml_tensor *pre_conv6_w, *pre_conv6_b;
    ggml_tensor *pre_out_w, *pre_out_b;

    // Encoder layers
    std::vector<cohere_enc_layer> enc_layers;

    // Encoder→decoder projection
    ggml_tensor *enc_proj_w, *enc_proj_b;

    // Decoder
    ggml_tensor* dec_emb_w;
    ggml_tensor* dec_pos_w;
    ggml_tensor *dec_emb_ln_w, *dec_emb_ln_b;
    std::vector<cohere_dec_layer> dec_layers;
    ggml_tensor *dec_out_ln_w, *dec_out_ln_b;
    ggml_tensor *dec_head_w, *dec_head_b;

    // ggml bookkeeping
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

struct cohere_vocab {
    std::vector<std::string> id_to_token;
    std::map<std::string, int> token_to_id;

    int n_vocab() const { return (int)id_to_token.size(); }

    int token_id(const std::string& s) const {
        auto it = token_to_id.find(s);
        return it == token_to_id.end() ? -1 : it->second;
    }
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct cohere_context {
    cohere_model model;
    cohere_vocab vocab;
    cohere_context_params params;

    // Persistent KV cache tensors and their context
    struct ggml_context* kv_ctx = nullptr;
    struct ggml_tensor* kv_k = nullptr;
    struct ggml_tensor* kv_v = nullptr;

    // Self-attention KV cache buffer (decoder).
    struct ggml_backend_buffer* kv_buf = nullptr;

    // Cross-attention KV cache: computed once per utterance from encoder output.
    // Shape per layer: (dec_d_model, T_enc)
    struct ggml_context* cross_kv_ctx = nullptr;
    struct ggml_backend_buffer* cross_kv_buf = nullptr;
    std::vector<struct ggml_tensor*> cross_kv_k;
    std::vector<struct ggml_tensor*> cross_kv_v;

    // ggml backends: primary (GPU if available, else CPU) + CPU fallback for unsupported ops
    ggml_backend_t ggml_backend = nullptr;     // primary backend (Metal/CUDA/CPU)
    ggml_backend_t ggml_backend_cpu = nullptr; // always CPU; used as sched fallback
    ggml_backend_sched_t ggml_alloc = nullptr;

    // Metadata context for graph node descriptors (no_alloc=true; actual buffers via gallocr)
    // Sized generously; only holds ggml_tensor_overhead() * N_NODES bytes.
    std::vector<uint8_t> compute_meta;
    struct ggml_context* compute_ctx = nullptr;
    const void* compute_meta_base = nullptr;
    size_t compute_meta_size = 0;

    // Cached T_enc from last encode call, needed by decode graph builder.
    int cached_T_enc = 0;

    // Mel spectrogram buffer
    std::vector<float> mel_buf;

    // Full-precision frontend constants generated from the HF contract. The
    // GGUF copies originated as BF16/F16 and are deliberately not consumed.
    cohere_frontend::Constants frontend;

    // Constants
    struct ggml_context* ctx_const = nullptr;
    struct ggml_tensor* eps = nullptr;

    // Cached decoder graph for n_tokens=1
    struct ggml_cgraph* gf_decode_1 = nullptr;
    int cached_offset_1 = -1;

    // Cross-attention alignment weights for token-level timestamps.
    // Populated by cohere_decode_step when params.collect_alignment=true and n_tok==1.
    // step_attn[i] = cross-attn weights for generated token i (last decoder layer, all heads).
    // Layout per step: [T_enc * n_heads] in row-major (head h: w[h*T_enc .. h*T_enc+T_enc-1]).
    int attn_T_enc = 0;
    int attn_n_heads = 0;
    std::vector<std::vector<float>> step_attn;

    // Performance counters (reset at start of each cohere_transcribe())
    cohere_perf perf;

    // Sticky decode-time sampling. temperature == 0 (default) keeps
    // the bit-identical greedy path; > 0 switches the transformer
    // decoder over to numerically-stable softmax sampling.
    float decode_temperature = 0.0f;
    float frequency_penalty = 0.0f;
    int max_new_tokens = 0;
    uint64_t decode_seed = 0;
    bool repetition_loop_guard = false;

    // Installed by embedding runtimes after construction. The callback data
    // outlives every inference and is never accessed after cohere_free().
    cohere_abort_callback abort_callback = nullptr;
    void* abort_callback_data = nullptr;

    // §90 beam-search width. 1 = greedy (default).
    int beam_size = 1;
};

static bool cohere_abort_requested(struct cohere_context* ctx) {
    return ctx && ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data);
}

static bool cohere_abort_if_requested(struct cohere_context* ctx) {
    if (!cohere_abort_requested(ctx))
        return false;
    cohere_set_error(COHERE_ERROR_CANCELLED, "Cohere inference was cancelled");
    return true;
}

// All ephemeral production graphs share compute_meta and never coexist: a new
// builder already overwrites that byte arena. Reuse one ggml_context header as
// well, resetting its object list for each graph. ggml_init allocates the
// header even when the tensor arena is caller-owned, so allocating a fresh
// context per encoder/decoder step leaked 40 bytes per graph. The pointer and
// size check also safely recreates the header when a staged debug path grows
// compute_meta and the vector moves.
static struct ggml_context* cohere_begin_compute_graph(struct cohere_context* ctx) {
    if (!ctx || ctx->compute_meta.empty())
        return nullptr;

    const void* base = ctx->compute_meta.data();
    const size_t size = ctx->compute_meta.size();
    if (ctx->compute_ctx &&
        (ctx->compute_meta_base != base || ctx->compute_meta_size != size)) {
        ggml_free(ctx->compute_ctx);
        ctx->compute_ctx = nullptr;
    }

    if (!ctx->compute_ctx) {
        ctx->compute_ctx = ggml_init({
            .mem_size = size,
            .mem_buffer = ctx->compute_meta.data(),
            .no_alloc = true,
        });
        if (!ctx->compute_ctx) {
            ctx->compute_meta_base = nullptr;
            ctx->compute_meta_size = 0;
            return nullptr;
        }
        ctx->compute_meta_base = base;
        ctx->compute_meta_size = size;
    } else {
        ggml_reset(ctx->compute_ctx);
    }
    return ctx->compute_ctx;
}

void cohere_set_abort_callback(
    struct cohere_context* ctx,
    cohere_abort_callback callback,
    void* user_data) {
    if (!ctx)
        return;
    ctx->abort_callback = callback;
    ctx->abort_callback_data = user_data;

    if (ctx->ggml_backend && ggml_backend_is_cpu(ctx->ggml_backend))
        ggml_backend_cpu_set_abort_callback(ctx->ggml_backend, callback, user_data);
    if (ctx->ggml_backend_cpu && ctx->ggml_backend_cpu != ctx->ggml_backend)
        ggml_backend_cpu_set_abort_callback(ctx->ggml_backend_cpu, callback, user_data);
#if defined(GGML_USE_METAL)
    if (ctx->ggml_backend && ggml_backend_is_metal(ctx->ggml_backend))
        ggml_backend_metal_set_abort_callback(ctx->ggml_backend, callback, user_data);
#endif
}

static void cohere_log_tensor(const char* name, const struct ggml_tensor* t);
static struct ggml_cgraph* cohere_build_graph_encoder(struct cohere_context* ctx, int T_mel, bool build_cross_kv);
static struct ggml_cgraph* cohere_build_graph_encoder_batch(struct cohere_context* ctx, int T_mel, int n_batch,
                                                            bool capture_stages,
                                                            enum cohere_encoder_batch_precision precision);
static struct ggml_cgraph* cohere_build_graph_encoder_batch_padded(struct cohere_context* ctx, int T_mel,
                                                                   int n_batch, bool capture_stages);
static struct ggml_cgraph* cohere_build_graph_encoder_staged(struct cohere_context* ctx, int T_mel);
static struct ggml_cgraph* cohere_build_graph_decoder(struct cohere_context* ctx, const int* /*tokens*/, int n_tokens,
                                                      int offset);
static void cohere_fold_batchnorm(cohere_model& model, int verbosity = 1);

// ---------------------------------------------------------------------------
// Encoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_tensor* cohere_rel_shift(struct ggml_context* ctx, struct ggml_tensor* a) {
    const int T = a->ne[1];
    const int n_heads = a->ne[2];
    struct ggml_tensor* out = ggml_view_3d(ctx, a, T, T, n_heads, a->nb[1] - a->nb[0], a->nb[2], (T - 1) * a->nb[0]);
    return out;
}

static struct ggml_tensor* cohere_rel_shift_batch(struct ggml_context* ctx, struct ggml_tensor* a) {
    const int T = (int)a->ne[1];
    const int n_heads = (int)a->ne[2];
    const int n_batch = (int)a->ne[3];
    return ggml_view_4d(ctx, a, T, T, n_heads, n_batch, a->nb[1] - a->nb[0], a->nb[2], a->nb[3],
                        (T - 1) * a->nb[0]);
}

static bool cohere_batch_precision_valid(enum cohere_encoder_batch_precision precision) {
    return precision == COHERE_ENCODER_BATCH_PRECISE || precision == COHERE_ENCODER_BATCH_FAST ||
           precision == COHERE_ENCODER_BATCH_F32_ACCUM;
}

// Flatten logical lanes into the token-column dimension for learned F16
// projections, then restore [features, tokens, lanes]. Keeping B in ggml's
// batch dimension makes a T=1 decoder step look like B independent matrix-
// vector products to the CUDA dispatcher; the flattened layout performs one
// matrix-matrix product and reuses each weight row across all lanes. Attention
// score products and the encoder's shared relative-position projection
// deliberately bypass this helper.
static struct ggml_tensor* cohere_precise_batch_projection(struct ggml_context* ctx,
                                                           struct ggml_tensor* weight,
                                                           struct ggml_tensor* input) {
    if (weight->type != GGML_TYPE_F16 || input->type != GGML_TYPE_F32) {
        return ggml_mul_mat(ctx, weight, input);
    }

    GGML_ASSERT(input->ne[3] == 1);
    GGML_ASSERT(ggml_is_contiguous(input));
    const int64_t T = input->ne[1];
    const int64_t B = input->ne[2];
    GGML_ASSERT(T > 0 && B > 0 && T <= INT64_MAX / B);
    struct ggml_tensor* flat = ggml_reshape_2d(ctx, input, input->ne[0], T * B);
    struct ggml_tensor* projected = ggml_mul_mat(ctx, weight, flat);
    return ggml_reshape_3d(ctx, projected, projected->ne[0], T, B);
}

static struct ggml_tensor* cohere_batch_projection(struct ggml_context* ctx, struct ggml_tensor* weight,
                                                   struct ggml_tensor* input,
                                                   enum cohere_encoder_batch_precision precision) {
    if (precision == COHERE_ENCODER_BATCH_PRECISE) {
        return cohere_precise_batch_projection(ctx, weight, input);
    }

    if (weight->type != GGML_TYPE_F16 || input->type != GGML_TYPE_F32) {
        return ggml_mul_mat(ctx, weight, input);
    }

    struct ggml_tensor* projected = ggml_mul_mat(ctx, weight, input);
    if (precision == COHERE_ENCODER_BATCH_F32_ACCUM) {
        ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    }
    return projected;
}

static struct ggml_cgraph* cohere_build_graph_encoder(struct cohere_context* ctx, int T_mel, bool build_cross_kv) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels = hp.n_mels;
    // Gated per-stage encoder snapshots (mel + per-block + pre-proj final) for
    // the crispasr-diff / transcribe.cpp comparison. No overhead when unset.
    const bool dump_stages = std::getenv("CRISPASR_COHERE_DUMP_STAGES") != nullptr;

    struct ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0)
        return nullptr;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    // mel: [n_mels, T_mel]
    struct ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // --- Subsampling ---
    // conv.0: [n_mels, T_mel, 1] -> [64, T/2, 256]
    struct ggml_tensor* cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.2: depthwise [64, T/2, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    // conv.3: pointwise [32, T/4, 256] -> [32, T/4, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // conv.5: depthwise [32, T/4, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    // conv.6: pointwise [16, T/8, 256] -> [16, T/8, 256]
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Flatten: [W3, H3, 256] -> [W3, 256, H3] -> [W3*256, H3]
    int H3 = cur->ne[1];
    int W3 = cur->ne[0];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3)); // [W3, C, H3]
    cur = ggml_reshape_2d(ctx0, cur, W3 * 256, H3);
    cohere_log_tensor("after_flatten", cur);

    // pre_out: [4096, H3] -> [d, H3]
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.pre_out_w, cur), model.pre_out_b);

    const int T = H3;

    // Positional encodings sinusoidal: [d, 2*T-1]
    struct ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& layer = model.enc_layers[il];
        struct ggml_tensor* inpL = cur;

        // FF1
        struct ggml_tensor* ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_mul_inplace(ctx0, ff1, layer.ff1_norm_w);
        ff1 = ggml_add_inplace(ctx0, ff1, layer.ff1_norm_b);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_up_w, ff1), layer.ff1_up_b);
        ff1 = ggml_silu_inplace(ctx0, ff1);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_dn_w, ff1), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));

        struct ggml_tensor* inpAtn = cur;

        // Self-Attention
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_norm_b);

        struct ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);

        struct ggml_tensor* R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc); // [d, 2T-1]

        struct ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);     // [hd, T, H]
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);     // [hd, T, H]
        K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, head_dim, n_heads, T), 0, 2, 1, 3);         // [hd, T, H]
        R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3); // [hd, 2T-1, H]

        // Q_u / Q_v are src1 of mul_mat — CPU kernel handles non-contiguous src1 natively.
        // K and R are src0 and must be contiguous (ggml_mul_mat asserts nb[0]==element_size and !transposed).
        struct ggml_tensor* AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), Q_u);     // [T, T, H]
        struct ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v); // [2T-1, T, H]

        struct ggml_tensor* BD = cohere_rel_shift(ctx0, BD_raw); // [T, T, H]

        struct ggml_tensor* scores = ggml_add_inplace(ctx0, AC, BD);
        scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max_inplace(ctx0, scores);

        struct ggml_tensor* V_reshaped = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T);

        // V_trans is transposed (nb[0] > nb[1]) so ggml_cont is required before mul_mat.
        struct ggml_tensor* V_trans = ggml_permute(ctx0, V_reshaped, 1, 2, 0, 3);            // [T, hd, H]
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores); // [hd, T, H]

        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);

        attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_out_w, attn_out), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);

        struct ggml_tensor* inpCnv = cur;

        // Convolution module
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.conv_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.conv_norm_b);

        struct ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        struct ggml_tensor* cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw1_w, cur), layer.conv_pw1_b);

        cnv = ggml_siglu_swapped(ctx0, cnv);

        // Depthwise conv via ggml_conv_2d_dw_direct (no im2col intermediate buffer).
        // Kernel needs F32 [k, 1, 1, d] layout; input needs contiguous [T, 1, d, 1] (WHCN).
        // Cast F16 kernel to F32 and reshape from [k, 1, d] to [k, 1, 1, d].
        struct ggml_tensor* dw_w_f32 = ggml_cast(ctx0, layer.conv_dw_w, GGML_TYPE_F32);        // [k, 1, d]
        struct ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, hp.enc_conv_k, 1, 1, d); // [k, 1, 1, d]
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));                                      // [d, T] -> [T, d]
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1); // [T, 1, d, 1] WHCN for direct conv
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (hp.enc_conv_k - 1) / 2, 0, 1, 1);
        // Output: [T, 1, d, 1] — same permute as conv_1d_dw path restores [d, T, 1, 1].
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));

        // Add folded bias (BN folded into conv_dw_w/b at init by cohere_fold_batchnorm)
        cnv = ggml_add(ctx0, cnv, ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1));

        cnv = ggml_silu_inplace(ctx0, cnv); // swish

        struct ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw2_w, cnv), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);

        struct ggml_tensor* inpFF2 = cur;

        // FF2
        struct ggml_tensor* ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_mul_inplace(ctx0, ff2, layer.ff2_norm_w);
        ff2 = ggml_add_inplace(ctx0, ff2, layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_up_w, ff2), layer.ff2_up_b);
        ff2 = ggml_silu_inplace(ctx0, ff2);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_dn_w, ff2), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));

        // Final output norm for the layer
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.out_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.out_norm_b);

        // Per-block encoder dump (gated). ggml_cont escapes the in-place norm
        // aliasing so the snapshot is not overwritten by buffer reuse. Used to
        // bisect encoder divergence against transcribe.cpp (see §231: a corrupt
        // GGUF zeroed layers 20/24 → block output collapsed to 0 from there).
        if (dump_stages) {
            struct ggml_tensor* bdbg = ggml_cont(ctx0, cur);
            char bn[32];
            snprintf(bn, sizeof(bn), "enc_block_%d", il);
            ggml_set_name(bdbg, bn);
            ggml_set_output(bdbg);
            ggml_build_forward_expand(gf, bdbg);
        }
    }

    // Full-T pre-projection encoder output (gated), the direct analog of
    // transcribe.cpp's enc.final [T,1280] for numeric comparison.
    if (dump_stages) {
        struct ggml_tensor* enc_final_dbg = ggml_cont(ctx0, cur);
        ggml_set_name(enc_final_dbg, "enc_final_preproj");
        ggml_set_output(enc_final_dbg);
        ggml_build_forward_expand(gf, enc_final_dbg);
    }

    // Encoder-decoder projection
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.enc_proj_w, cur), model.enc_proj_b);
    ggml_set_name(cur, "enc_out");
    if (!build_cross_kv) {
        // cohere_run_encoder() reads this terminal encoder-only tensor.
        ggml_set_output(cur);
    }

    // Pre-compute cross KV for all decoder layers
    if (build_cross_kv) {
        for (int il = 0; il < hp.dec_n_layers; il++) {
            const auto& layer = model.dec_layers[il];

            struct ggml_tensor* ck = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_k_w, cur), layer.cross_k_b);
            struct ggml_tensor* cv = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_v_w, cur), layer.cross_v_b);

            // CK: [d, T] -> [hd, H, T] -> [hd, T, H]
            struct ggml_tensor* ck_ready = ggml_cont(
                ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, ck, hp.dec_head_dim, hp.dec_n_heads, T), 0, 2, 1, 3));

            // CV: [d, T] -> [hd, H, T] -> [hd, T, H] (matches CK layout for ggml_flash_attn_ext)
            struct ggml_tensor* cv_ready = ggml_cont(
                ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, cv, hp.dec_head_dim, hp.dec_n_heads, T), 0, 2, 1, 3));

            char ck_name[32], cv_name[32];
            snprintf(ck_name, sizeof(ck_name), "ck_%d", il);
            snprintf(cv_name, sizeof(cv_name), "cv_%d", il);
            ggml_set_name(ck_ready, ck_name);
            ggml_set_name(cv_ready, cv_name);

            ggml_build_forward_expand(gf, ck_ready);
            ggml_build_forward_expand(gf, cv_ready);
        }
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Equal-length batched encoder used by cohere_run_encoder_batch(). Production
// B1 inference deliberately keeps using cohere_build_graph_encoder() above.
// The persistent activation layout is [d, T, B]; attention and direct
// depthwise convolution temporarily place B in ne[3] where ggml kernels expect
// it. Decoder cross-KV is intentionally omitted.
static struct ggml_cgraph* cohere_build_graph_encoder_batch(struct cohere_context* ctx, int T_mel, int n_batch,
                                                            bool capture_stages,
                                                            enum cohere_encoder_batch_precision precision) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels = hp.n_mels;

    struct ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0)
        return nullptr;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, capture_stages ? 16384 : 8192, false);
    auto snapshot = [&](struct ggml_tensor* value, const char* name) {
        if (!capture_stages)
            return;
        struct ggml_tensor* copy = ggml_dup(ctx0, value);
        ggml_set_name(copy, name);
        ggml_build_forward_expand(gf, copy);
    };

    // Contiguous storage is lane-major: [B, T_mel, n_mels].
    struct ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_mels, T_mel, 1, n_batch);
    ggml_set_name(mel, "mel_batch");
    ggml_set_input(mel);

    // Conv2D subsampling preserves batch in ne[3].
    struct ggml_tensor* cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "batch_pre_conv0");

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "batch_pre_conv3");

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "batch_pre_conv6");

    const int H3 = (int)cur->ne[1];
    const int W3 = (int)cur->ne[0];
    const int C3 = (int)cur->ne[2];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, W3 * C3, H3, n_batch);
    cur = ggml_add(ctx0, cohere_batch_projection(ctx0, model.pre_out_w, cur, precision), model.pre_out_b);
    snapshot(cur, "batch_pre_enc_out");

    const int T = H3;
    struct ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc_batch");
    ggml_set_input(pos_enc);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& layer = model.enc_layers[il];
        struct ggml_tensor* inpL = cur;

        struct ggml_tensor* ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_mul_inplace(ctx0, ff1, layer.ff1_norm_w);
        ff1 = ggml_add_inplace(ctx0, ff1, layer.ff1_norm_b);
        ff1 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff1_up_w, ff1, precision), layer.ff1_up_b);
        ff1 = ggml_silu_inplace(ctx0, ff1);
        ff1 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff1_dn_w, ff1, precision), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));
        if (il == 0)
            snapshot(cur, "batch_L0_ff1");

        struct ggml_tensor* inpAtn = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_norm_b);

        struct ggml_tensor* Q =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_q_w, cur, precision), layer.attn_q_b);
        struct ggml_tensor* K =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_k_w, cur, precision), layer.attn_k_b);
        struct ggml_tensor* V =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_v_w, cur, precision), layer.attn_v_b);
        struct ggml_tensor* R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc);

        struct ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Q_u, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Q_v, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_4d(ctx0, K, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

        struct ggml_tensor* AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), Q_u);
        struct ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        struct ggml_tensor* BD = cohere_rel_shift_batch(ctx0, BD_raw);

        struct ggml_tensor* scores = ggml_add_inplace(ctx0, AC, BD);
        scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max_inplace(ctx0, scores);

        struct ggml_tensor* V_reshaped = ggml_reshape_4d(ctx0, V, head_dim, n_heads, T, n_batch);
        struct ggml_tensor* V_trans = ggml_permute(ctx0, V_reshaped, 1, 2, 0, 3);
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores);
        attn_out = ggml_reshape_3d(
            ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T, n_batch);
        attn_out =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_out_w, attn_out, precision), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);
        if (il == 0)
            snapshot(cur, "batch_L0_attn");

        struct ggml_tensor* inpCnv = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.conv_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.conv_norm_b);

        struct ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        struct ggml_tensor* cnv =
            ggml_add(ctx0, cohere_batch_projection(ctx0, pw1_w, cur, precision), layer.conv_pw1_b);
        cnv = ggml_siglu_swapped(ctx0, cnv);

        struct ggml_tensor* dw_w_f32 = ggml_cast(ctx0, layer.conv_dw_w, GGML_TYPE_F32);
        struct ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, hp.enc_conv_k, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, n_batch);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (hp.enc_conv_k - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        cnv = ggml_add(ctx0, cnv, ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1));
        cnv = ggml_silu_inplace(ctx0, cnv);
        cnv = ggml_reshape_3d(ctx0, cnv, d, T, n_batch);

        struct ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, cohere_batch_projection(ctx0, pw2_w, cnv, precision), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);
        if (il == 0)
            snapshot(cur, "batch_L0_conv");

        struct ggml_tensor* inpFF2 = cur;
        struct ggml_tensor* ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_mul_inplace(ctx0, ff2, layer.ff2_norm_w);
        ff2 = ggml_add_inplace(ctx0, ff2, layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff2_up_w, ff2, precision), layer.ff2_up_b);
        ff2 = ggml_silu_inplace(ctx0, ff2);
        ff2 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff2_dn_w, ff2, precision), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));
        if (il == 0)
            snapshot(cur, "batch_L0_ff2");

        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.out_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.out_norm_b);
        if (il == 0)
            snapshot(cur, "batch_enc_L00");
    }

    cur = ggml_add(ctx0, cohere_batch_projection(ctx0, model.enc_proj_w, cur, precision), model.enc_proj_b);
    ggml_set_name(cur, "enc_out_batch");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Variable-length research graph. Production B1 and the equal-length graph
// above remain separate so their allocation and numeric contracts stay fixed.
static struct ggml_cgraph* cohere_build_graph_encoder_batch_padded(struct cohere_context* ctx, int T_mel,
                                                                   int n_batch, bool capture_stages) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels = hp.n_mels;
    constexpr auto precision = COHERE_ENCODER_BATCH_FAST;

    struct ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0)
        return nullptr;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, capture_stages ? 16384 : 8192, false);
    auto snapshot = [&](struct ggml_tensor* value, const char* name) {
        if (!capture_stages)
            return;
        struct ggml_tensor* copy = ggml_dup(ctx0, value);
        ggml_set_name(copy, name);
        ggml_build_forward_expand(gf, copy);
    };
    auto apply_subsampling_mask = [&](struct ggml_tensor* value, const char* name) {
        struct ggml_tensor* mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, value->ne[1], 1, n_batch);
        ggml_set_name(mask, name);
        ggml_set_input(mask);
        return ggml_mul(ctx0, value, mask);
    };

    struct ggml_tensor* mel = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_mels, T_mel, 1, n_batch);
    ggml_set_name(mel, "padded_mel_batch");
    ggml_set_input(mel);

    struct ggml_tensor* cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = apply_subsampling_mask(cur, "padded_sub_mask_0");
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "padded_pre_conv0");

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    cur = apply_subsampling_mask(cur, "padded_sub_mask_2");
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = apply_subsampling_mask(cur, "padded_sub_mask_3");
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "padded_pre_conv3");

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    cur = apply_subsampling_mask(cur, "padded_sub_mask_5");
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = apply_subsampling_mask(cur, "padded_sub_mask_6");
    cur = ggml_relu(ctx0, cur);
    snapshot(cur, "padded_pre_conv6");

    const int T = (int)cur->ne[1];
    const int W = (int)cur->ne[0];
    const int C = (int)cur->ne[2];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_3d(ctx0, cur, W * C, T, n_batch);
    cur = ggml_add(ctx0, cohere_batch_projection(ctx0, model.pre_out_w, cur, precision), model.pre_out_b);
    snapshot(cur, "padded_pre_enc_out");

    struct ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "padded_pos_enc");
    ggml_set_input(pos_enc);
    struct ggml_tensor* valid_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, T, n_batch);
    ggml_set_name(valid_mask, "padded_enc_valid_mask");
    ggml_set_input(valid_mask);
    struct ggml_tensor* key_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T, T, 1, n_batch);
    ggml_set_name(key_mask, "padded_enc_key_mask");
    ggml_set_input(key_mask);

    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& layer = model.enc_layers[il];
        struct ggml_tensor* inpL = cur;

        struct ggml_tensor* ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_mul_inplace(ctx0, ff1, layer.ff1_norm_w);
        ff1 = ggml_add_inplace(ctx0, ff1, layer.ff1_norm_b);
        ff1 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff1_up_w, ff1, precision), layer.ff1_up_b);
        ff1 = ggml_silu_inplace(ctx0, ff1);
        ff1 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff1_dn_w, ff1, precision), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));
        if (il == 0)
            snapshot(cur, "padded_L0_ff1");

        struct ggml_tensor* inpAtn = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_norm_b);

        struct ggml_tensor* Q =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_q_w, cur, precision), layer.attn_q_b);
        struct ggml_tensor* K =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_k_w, cur, precision), layer.attn_k_b);
        struct ggml_tensor* V =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_v_w, cur, precision), layer.attn_v_b);
        struct ggml_tensor* R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc);
        struct ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Q_u, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Q_v, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_4d(ctx0, K, head_dim, n_heads, T, n_batch), 0, 2, 1, 3);
        R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

        struct ggml_tensor* AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), Q_u);
        struct ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        struct ggml_tensor* BD = cohere_rel_shift_batch(ctx0, BD_raw);
        struct ggml_tensor* scores = ggml_add_inplace(ctx0, AC, BD);
        scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max_ext_inplace(ctx0, scores, key_mask, 1.0f, 0.0f);

        struct ggml_tensor* V_reshaped = ggml_reshape_4d(ctx0, V, head_dim, n_heads, T, n_batch);
        struct ggml_tensor* V_trans = ggml_permute(ctx0, V_reshaped, 1, 2, 0, 3);
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores);
        attn_out = ggml_reshape_3d(
            ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T, n_batch);
        attn_out =
            ggml_add(ctx0, cohere_batch_projection(ctx0, layer.attn_out_w, attn_out, precision), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);
        if (il == 0)
            snapshot(cur, "padded_L0_attn");

        struct ggml_tensor* inpCnv = cur;
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.conv_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.conv_norm_b);
        struct ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        struct ggml_tensor* cnv =
            ggml_add(ctx0, cohere_batch_projection(ctx0, pw1_w, cur, precision), layer.conv_pw1_b);
        cnv = ggml_siglu_swapped(ctx0, cnv);
        cnv = ggml_mul(ctx0, cnv, valid_mask);

        struct ggml_tensor* dw_w_f32 = ggml_cast(ctx0, layer.conv_dw_w, GGML_TYPE_F32);
        struct ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, hp.enc_conv_k, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, n_batch);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (hp.enc_conv_k - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        cnv = ggml_add(ctx0, cnv, ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1));
        cnv = ggml_silu_inplace(ctx0, cnv);
        cnv = ggml_reshape_3d(ctx0, cnv, d, T, n_batch);
        struct ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, cohere_batch_projection(ctx0, pw2_w, cnv, precision), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);
        if (il == 0)
            snapshot(cur, "padded_L0_conv");

        struct ggml_tensor* inpFF2 = cur;
        struct ggml_tensor* ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_mul_inplace(ctx0, ff2, layer.ff2_norm_w);
        ff2 = ggml_add_inplace(ctx0, ff2, layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff2_up_w, ff2, precision), layer.ff2_up_b);
        ff2 = ggml_silu_inplace(ctx0, ff2);
        ff2 = ggml_add(ctx0, cohere_batch_projection(ctx0, layer.ff2_dn_w, ff2, precision), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));
        if (il == 0)
            snapshot(cur, "padded_L0_ff2");

        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.out_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.out_norm_b);
        if (il == 0)
            snapshot(cur, "padded_enc_L00");
    }

    cur = ggml_add(ctx0, cohere_batch_projection(ctx0, model.enc_proj_w, cur, precision), model.enc_proj_b);
    cur = ggml_mul(ctx0, cur, valid_mask);
    ggml_set_name(cur, "padded_enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Staged encoder: same as above but snapshots per-layer output via ggml_dup.
// Names: "enc_L00".."enc_L47", "enc_out". Skips cross-KV computation.
static struct ggml_cgraph* cohere_build_graph_encoder_staged(struct cohere_context* ctx, int T_mel) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.enc_d_model;
    const int n_heads = hp.enc_n_heads;
    const int head_dim = hp.enc_head_dim;
    const int n_mels = hp.n_mels;

    struct ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0)
        return nullptr;
    // Larger graph for snapshots
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 16384, false);

    struct ggml_tensor* mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, T_mel);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // --- Subsampling (identical to non-staged) ---
    struct ggml_tensor* cur = ggml_conv_2d(ctx0, model.pre_conv0_w, mel, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv0_b, 1, 1, model.pre_conv0_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Snapshot after conv0+relu
    {
        struct ggml_tensor* s = ggml_dup(ctx0, cur);
        ggml_set_name(s, "pre_conv0");
        ggml_build_forward_expand(gf, s);
    }

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv2_b, 1, 1, model.pre_conv2_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_conv_2d(ctx0, model.pre_conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv3_b, 1, 1, model.pre_conv3_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Snapshot after conv2+3+relu
    {
        struct ggml_tensor* s = ggml_dup(ctx0, cur);
        ggml_set_name(s, "pre_conv3");
        ggml_build_forward_expand(gf, s);
    }

    cur = ggml_conv_2d_dw(ctx0, model.pre_conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv5_b, 1, 1, model.pre_conv5_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_conv_2d(ctx0, model.pre_conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(
        ctx0, cur,
        ggml_cast(ctx0, ggml_reshape_4d(ctx0, model.pre_conv6_b, 1, 1, model.pre_conv6_b->ne[0], 1), GGML_TYPE_F32));
    cur = ggml_relu(ctx0, cur);

    // Snapshot after conv5+6+relu (final conv output before flatten+linear)
    {
        struct ggml_tensor* s = ggml_dup(ctx0, cur);
        ggml_set_name(s, "pre_conv6");
        ggml_build_forward_expand(gf, s);
    }

    int H3 = cur->ne[1];
    int W3 = cur->ne[0];
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, W3 * 256, H3);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.pre_out_w, cur), model.pre_out_b);

    const int T = H3;

    // Snapshot pre_encode output
    {
        struct ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, "pre_enc_out");
        ggml_build_forward_expand(gf, snap);
    }

    struct ggml_tensor* pos_enc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d, 2 * T - 1);
    ggml_set_name(pos_enc, "pos_enc");
    ggml_set_input(pos_enc);

    // --- Conformer blocks with per-layer snapshots ---
    char lbuf[32];
    for (int il = 0; il < hp.enc_n_layers; il++) {
        const auto& layer = model.enc_layers[il];
        struct ggml_tensor* inpL = cur;

        // FF1
        struct ggml_tensor* ff1 = ggml_norm(ctx0, cur, 1e-5f);
        ff1 = ggml_mul_inplace(ctx0, ff1, layer.ff1_norm_w);
        ff1 = ggml_add_inplace(ctx0, ff1, layer.ff1_norm_b);

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, ff1);
            ggml_set_name(s, "L0_ff1_ln");
            ggml_build_forward_expand(gf, s);
        }

        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_up_w, ff1), layer.ff1_up_b);

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, ff1);
            ggml_set_name(s, "L0_ff1_up");
            ggml_build_forward_expand(gf, s);
        }

        ff1 = ggml_silu_inplace(ctx0, ff1);
        ff1 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff1_dn_w, ff1), layer.ff1_dn_b);
        cur = ggml_add(ctx0, inpL, ggml_scale(ctx0, ff1, 0.5f));

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, cur);
            ggml_set_name(s, "L0_ff1");
            ggml_build_forward_expand(gf, s);
        }

        struct ggml_tensor* inpAtn = cur;

        // Self-Attention
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_norm_b);

        struct ggml_tensor* Q = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor* K = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor* V = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);
        struct ggml_tensor* R = ggml_mul_mat(ctx0, layer.attn_pos_w, pos_enc);

        struct ggml_tensor* Q_u = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_u, d));
        struct ggml_tensor* Q_v = ggml_add(ctx0, Q, ggml_reshape_1d(ctx0, layer.attn_pos_bias_v, d));

        Q_u = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_u, head_dim, n_heads, T), 0, 2, 1, 3);
        Q_v = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Q_v, head_dim, n_heads, T), 0, 2, 1, 3);
        K = ggml_permute(ctx0, ggml_reshape_3d(ctx0, K, head_dim, n_heads, T), 0, 2, 1, 3);
        R = ggml_permute(ctx0, ggml_reshape_3d(ctx0, R, head_dim, n_heads, 2 * T - 1), 0, 2, 1, 3);

        struct ggml_tensor* AC = ggml_mul_mat(ctx0, ggml_cont(ctx0, K), Q_u);
        struct ggml_tensor* BD_raw = ggml_mul_mat(ctx0, ggml_cont(ctx0, R), Q_v);
        struct ggml_tensor* BD = cohere_rel_shift(ctx0, BD_raw);

        struct ggml_tensor* scores = ggml_add_inplace(ctx0, AC, BD);
        scores = ggml_scale_inplace(ctx0, scores, 1.0f / sqrtf((float)head_dim));
        scores = ggml_soft_max_inplace(ctx0, scores);

        struct ggml_tensor* V_reshaped = ggml_reshape_3d(ctx0, V, head_dim, n_heads, T);
        struct ggml_tensor* V_trans = ggml_permute(ctx0, V_reshaped, 1, 2, 0, 3);
        struct ggml_tensor* attn_out = ggml_mul_mat(ctx0, ggml_cont(ctx0, V_trans), scores);

        attn_out = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3)), d, T);
        attn_out = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_out_w, attn_out), layer.attn_out_b);
        cur = ggml_add(ctx0, inpAtn, attn_out);

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, cur);
            ggml_set_name(s, "L0_attn");
            ggml_build_forward_expand(gf, s);
        }

        struct ggml_tensor* inpCnv = cur;

        // Conv module
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.conv_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.conv_norm_b);

        struct ggml_tensor* pw1_w = ggml_reshape_2d(ctx0, layer.conv_pw1_w, d, 2 * d);
        struct ggml_tensor* cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw1_w, cur), layer.conv_pw1_b);
        cnv = ggml_siglu_swapped(ctx0, cnv);

        struct ggml_tensor* dw_w_f32 = ggml_cast(ctx0, layer.conv_dw_w, GGML_TYPE_F32);
        struct ggml_tensor* dw_w_4d = ggml_reshape_4d(ctx0, dw_w_f32, hp.enc_conv_k, 1, 1, d);
        cnv = ggml_cont(ctx0, ggml_transpose(ctx0, cnv));
        cnv = ggml_reshape_4d(ctx0, cnv, T, 1, d, 1);
        cnv = ggml_conv_2d_dw_direct(ctx0, dw_w_4d, cnv, 1, 1, (hp.enc_conv_k - 1) / 2, 0, 1, 1);
        cnv = ggml_cont(ctx0, ggml_permute(ctx0, cnv, 1, 2, 0, 3));
        cnv = ggml_add(ctx0, cnv, ggml_reshape_4d(ctx0, layer.conv_dw_b, d, 1, 1, 1));
        cnv = ggml_silu_inplace(ctx0, cnv);

        struct ggml_tensor* pw2_w = ggml_reshape_2d(ctx0, layer.conv_pw2_w, d, d);
        cnv = ggml_add(ctx0, ggml_mul_mat(ctx0, pw2_w, cnv), layer.conv_pw2_b);
        cur = ggml_add(ctx0, inpCnv, cnv);

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, cur);
            ggml_set_name(s, "L0_conv");
            ggml_build_forward_expand(gf, s);
        }

        struct ggml_tensor* inpFF2 = cur;

        // FF2
        struct ggml_tensor* ff2 = ggml_norm(ctx0, cur, 1e-5f);
        ff2 = ggml_mul_inplace(ctx0, ff2, layer.ff2_norm_w);
        ff2 = ggml_add_inplace(ctx0, ff2, layer.ff2_norm_b);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_up_w, ff2), layer.ff2_up_b);
        ff2 = ggml_silu_inplace(ctx0, ff2);
        ff2 = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ff2_dn_w, ff2), layer.ff2_dn_b);
        cur = ggml_add(ctx0, inpFF2, ggml_scale(ctx0, ff2, 0.5f));

        if (il == 0) {
            struct ggml_tensor* s = ggml_dup(ctx0, cur);
            ggml_set_name(s, "L0_ff2");
            ggml_build_forward_expand(gf, s);
        }

        // Final output norm
        cur = ggml_norm(ctx0, cur, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.out_norm_w);
        cur = ggml_add_inplace(ctx0, cur, layer.out_norm_b);

        // Snapshot this layer's output
        snprintf(lbuf, sizeof(lbuf), "enc_L%02d", il);
        struct ggml_tensor* snap = ggml_dup(ctx0, cur);
        ggml_set_name(snap, lbuf);
        ggml_build_forward_expand(gf, snap);
    }

    // Encoder-decoder projection (skip cross-KV for staged path)
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.enc_proj_w, cur), model.enc_proj_b);
    ggml_set_name(cur, "enc_out");
    ggml_build_forward_expand(gf, cur);

    return gf;
}


static struct ggml_cgraph* cohere_build_graph_decoder(struct cohere_context* ctx, const int* /*tokens*/, int n_tokens,
                                                      int offset);

#define CT_CHECK(x)                                                                                                    \
    do {                                                                                                               \
        if (!(x)) {                                                                                                    \
            fprintf(stderr, "CT_CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__);                                  \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static float swish(float x) {
    return x * sigmoid(x);
}

// ---------------------------------------------------------------------------
// GGUF loading helpers
// ---------------------------------------------------------------------------

#include "gguf.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static ggml_tensor* ct_get_tensor(cohere_model& model, const std::string& name) {
    auto it = model.tensors.find(name);
    if (it == model.tensors.end()) {
        fprintf(stderr, "cohere: tensor '%s' not found in GGUF\n", name.c_str());
        return nullptr;
    }
    return it->second;
}

static ggml_tensor* ct_get_tensor_fmt(cohere_model& model, const char* fmt, int idx) {
    char buf[128];
    snprintf(buf, sizeof(buf), fmt, idx);
    return ct_get_tensor(model, buf);
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

#include "core/attention.h"
#include "core/cpu_ops.h" // core_cpu::to_f32 (quantized-safe weight read)
#include "core/beam_decode.h"
#include "cohere_chunking.h"
#include "core/gguf_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static bool cohere_load_model(cohere_model& model, cohere_vocab& vocab, const char* path, ggml_backend_t backend) {
    // First pass: read metadata
    gguf_context* gguf_ctx = core_gguf::open_metadata(path);
    if (!gguf_ctx)
        return false;

    auto& hp = model.hparams;

    hp.vocab_size = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_VOCAB_SIZE, 0);
    hp.enc_n_layers = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_N_LAYERS, 0);
    hp.enc_d_model = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_D_MODEL, 0);
    hp.enc_n_heads = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_N_HEADS, 0);
    hp.enc_head_dim = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_HEAD_DIM, 0);
    hp.enc_ffn_dim = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_FFN_DIM, 0);
    hp.enc_conv_k = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_ENC_CONV_KERNEL, 0);
    hp.dec_n_layers = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_N_LAYERS, 0);
    hp.dec_d_model = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_D_MODEL, 0);
    hp.dec_n_heads = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_N_HEADS, 0);
    hp.dec_head_dim = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_HEAD_DIM, 0);
    hp.dec_ffn_dim = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_FFN_DIM, 0);
    hp.dec_max_ctx = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_DEC_MAX_CTX, 0);
    hp.n_mels = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_AUDIO_N_MELS, 0);
    hp.n_fft = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_AUDIO_N_FFT, 0);
    hp.hop_length = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_AUDIO_HOP, 0);
    hp.win_length = (int)core_gguf::kv_u32(gguf_ctx, CT_KEY_AUDIO_WIN, 0);

    // Load vocabulary
    {
        auto tokens = core_gguf::kv_str_array(gguf_ctx, "tokenizer.ggml.tokens");
        if (!tokens.empty()) {
            cohere_debug("cohere: loading %d tokens from GGUF\n", (int)tokens.size());
            vocab.id_to_token = std::move(tokens);
            for (int i = 0; i < (int)vocab.id_to_token.size(); i++) {
                vocab.token_to_id[vocab.id_to_token[i]] = i;
            }
            const char* specials[] = {"<|startoftranscript|>", "<|en|>", "<|endoftext|>"};
            for (const char* s : specials) {
                cohere_debug("cohere: token '%s' -> ID %d\n", s, vocab.token_id(s));
            }
        } else {
            cohere_debug("cohere: WARNING: tokenizer.ggml.tokens not found in GGUF\n");
        }
    }

    core_gguf::free_metadata(gguf_ctx);

    // Second pass: tensor data via shared helper
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path, backend, "cohere", wl)) {
            return false;
        }
        model.ctx = wl.ctx;
        model.buf = wl.buf;
        model.tensors = std::move(wl.tensors);

        struct ggml_tensor* tp = model.tensors["enc.proj.weight"];
        if (tp) {
            std::vector<uint8_t> data(ggml_nbytes(tp));
            ggml_backend_tensor_get(tp, data.data(), 0, data.size());
            if (tp->type == GGML_TYPE_F16) {
                ggml_fp16_t h;
                memcpy(&h, data.data(), sizeof(h));
                float v0 = ggml_fp16_to_fp32(h);
                cohere_debug("cohere: enc.proj.weight [0] (F16): %.4f\n", v0);
            } else if (tp->type == GGML_TYPE_BF16) {
                ggml_bf16_t h;
                memcpy(&h, data.data(), sizeof(h));
                float v0 = ggml_bf16_to_fp32(h);
                cohere_debug("cohere: enc.proj.weight [0] (BF16): %.4f\n", v0);
            } else if (tp->type == GGML_TYPE_F32) {
                float v0;
                memcpy(&v0, data.data(), sizeof(v0));
                cohere_debug("cohere: enc.proj.weight [0] (F32): %.4f\n", v0);
            }
        }
    }

    // Wire up model fields
    auto& m = model;
    auto T = [&](const char* name) { return ct_get_tensor(m, name); };
    auto TF = [&](const char* fmt, int i) { return ct_get_tensor_fmt(m, fmt, i); };

    m.pre_conv0_w = T(CT_PRE_CONV0_W);
    m.pre_conv0_b = T(CT_PRE_CONV0_B);
    m.pre_conv2_w = T(CT_PRE_CONV2_W);
    m.pre_conv2_b = T(CT_PRE_CONV2_B);
    m.pre_conv3_w = T(CT_PRE_CONV3_W);
    m.pre_conv3_b = T(CT_PRE_CONV3_B);
    m.pre_conv5_w = T(CT_PRE_CONV5_W);
    m.pre_conv5_b = T(CT_PRE_CONV5_B);
    m.pre_conv6_w = T(CT_PRE_CONV6_W);
    m.pre_conv6_b = T(CT_PRE_CONV6_B);
    m.pre_out_w = T(CT_PRE_OUT_W);
    m.pre_out_b = T(CT_PRE_OUT_B);

    m.enc_layers.resize(hp.enc_n_layers);
    for (int i = 0; i < hp.enc_n_layers; i++) {
        auto& l = m.enc_layers[i];
        l.ff1_norm_w = TF(CT_ENC_FF1_NORM_W, i);
        l.ff1_norm_b = TF(CT_ENC_FF1_NORM_B, i);
        l.ff1_up_w = TF(CT_ENC_FF1_UP_W, i);
        l.ff1_up_b = TF(CT_ENC_FF1_UP_B, i);
        l.ff1_dn_w = TF(CT_ENC_FF1_DN_W, i);
        l.ff1_dn_b = TF(CT_ENC_FF1_DN_B, i);
        l.attn_norm_w = TF(CT_ENC_ATN_NORM_W, i);
        l.attn_norm_b = TF(CT_ENC_ATN_NORM_B, i);
        l.attn_q_w = TF(CT_ENC_ATN_Q_W, i);
        l.attn_q_b = TF(CT_ENC_ATN_Q_B, i);
        l.attn_k_w = TF(CT_ENC_ATN_K_W, i);
        l.attn_k_b = TF(CT_ENC_ATN_K_B, i);
        l.attn_v_w = TF(CT_ENC_ATN_V_W, i);
        l.attn_v_b = TF(CT_ENC_ATN_V_B, i);
        l.attn_out_w = TF(CT_ENC_ATN_OUT_W, i);
        l.attn_out_b = TF(CT_ENC_ATN_OUT_B, i);
        l.attn_pos_w = TF(CT_ENC_ATN_POS_W, i);
        l.attn_pos_bias_u = TF(CT_ENC_ATN_POS_U, i);
        l.attn_pos_bias_v = TF(CT_ENC_ATN_POS_V, i);
        l.conv_norm_w = TF(CT_ENC_CNV_NORM_W, i);
        l.conv_norm_b = TF(CT_ENC_CNV_NORM_B, i);
        l.conv_pw1_w = TF(CT_ENC_CNV_PW1_W, i);
        l.conv_pw1_b = TF(CT_ENC_CNV_PW1_B, i);
        l.conv_dw_w = TF(CT_ENC_CNV_DW_W, i);
        l.conv_dw_b = TF(CT_ENC_CNV_DW_B, i);
        l.conv_bn_w = TF(CT_ENC_CNV_BN_W, i);
        l.conv_bn_b = TF(CT_ENC_CNV_BN_B, i);
        l.conv_bn_mean = TF(CT_ENC_CNV_BN_MEAN, i);
        l.conv_bn_var = TF(CT_ENC_CNV_BN_VAR, i);
        l.conv_pw2_w = TF(CT_ENC_CNV_PW2_W, i);
        l.conv_pw2_b = TF(CT_ENC_CNV_PW2_B, i);
        l.ff2_norm_w = TF(CT_ENC_FF2_NORM_W, i);
        l.ff2_norm_b = TF(CT_ENC_FF2_NORM_B, i);
        l.ff2_up_w = TF(CT_ENC_FF2_UP_W, i);
        l.ff2_up_b = TF(CT_ENC_FF2_UP_B, i);
        l.ff2_dn_w = TF(CT_ENC_FF2_DN_W, i);
        l.ff2_dn_b = TF(CT_ENC_FF2_DN_B, i);
        l.out_norm_w = TF(CT_ENC_OUT_NORM_W, i);
        l.out_norm_b = TF(CT_ENC_OUT_NORM_B, i);
    }

    m.enc_proj_w = T(CT_ENC_PROJ_W);
    m.enc_proj_b = T(CT_ENC_PROJ_B);

    m.dec_emb_w = T(CT_DEC_EMB_W);
    m.dec_pos_w = T(CT_DEC_POS_W);
    m.dec_emb_ln_w = T(CT_DEC_EMB_LN_W);
    m.dec_emb_ln_b = T(CT_DEC_EMB_LN_B);

    m.dec_layers.resize(hp.dec_n_layers);
    for (int i = 0; i < hp.dec_n_layers; i++) {
        auto& l = m.dec_layers[i];
        l.attn_ln_w = TF(CT_DEC_ATTN_LN_W, i);
        l.attn_ln_b = TF(CT_DEC_ATTN_LN_B, i);
        l.attn_q_w = TF(CT_DEC_ATTN_Q_W, i);
        l.attn_q_b = TF(CT_DEC_ATTN_Q_B, i);
        l.attn_k_w = TF(CT_DEC_ATTN_K_W, i);
        l.attn_k_b = TF(CT_DEC_ATTN_K_B, i);
        l.attn_v_w = TF(CT_DEC_ATTN_V_W, i);
        l.attn_v_b = TF(CT_DEC_ATTN_V_B, i);
        l.attn_o_w = TF(CT_DEC_ATTN_O_W, i);
        l.attn_o_b = TF(CT_DEC_ATTN_O_B, i);
        l.cross_ln_w = TF(CT_DEC_XATTN_LN_W, i);
        l.cross_ln_b = TF(CT_DEC_XATTN_LN_B, i);
        l.cross_q_w = TF(CT_DEC_XATTN_Q_W, i);
        l.cross_q_b = TF(CT_DEC_XATTN_Q_B, i);
        l.cross_k_w = TF(CT_DEC_XATTN_K_W, i);
        l.cross_k_b = TF(CT_DEC_XATTN_K_B, i);
        l.cross_v_w = TF(CT_DEC_XATTN_V_W, i);
        l.cross_v_b = TF(CT_DEC_XATTN_V_B, i);
        l.cross_o_w = TF(CT_DEC_XATTN_O_W, i);
        l.cross_o_b = TF(CT_DEC_XATTN_O_B, i);
        l.ffn_ln_w = TF(CT_DEC_FFN_LN_W, i);
        l.ffn_ln_b = TF(CT_DEC_FFN_LN_B, i);
        l.ffn_up_w = TF(CT_DEC_FFN_UP_W, i);
        l.ffn_up_b = TF(CT_DEC_FFN_UP_B, i);
        l.ffn_dn_w = TF(CT_DEC_FFN_DN_W, i);
        l.ffn_dn_b = TF(CT_DEC_FFN_DN_B, i);
    }

    m.dec_out_ln_w = T(CT_DEC_OUT_LN_W);
    m.dec_out_ln_b = T(CT_DEC_OUT_LN_B);
    m.dec_head_w = T(CT_DEC_HEAD_W);
    m.dec_head_b = T(CT_DEC_HEAD_B);

    return true;
}

// ---------------------------------------------------------------------------
// Utility: get F32 value from a ggml tensor (any dtype)
// ---------------------------------------------------------------------------

static float ct_f32(const ggml_tensor* t, int i0, int i1 = 0, int i2 = 0, int i3 = 0) {
    return ggml_get_f32_nd(const_cast<ggml_tensor*>(t), i0, i1, i2, i3);
}

// ---------------------------------------------------------------------------
// Layer norm (on float buffer, in-place)
// ---------------------------------------------------------------------------

static void ct_layer_norm(float* out, const float* in, int n, const float* w, const float* b, float eps = 1e-5f) {
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++)
        mean += in[i];
    mean /= n;
    for (int i = 0; i < n; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var /= n;
    float inv = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < n; i++)
        out[i] = (in[i] - (float)mean) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// SILU / Swish activation
// ---------------------------------------------------------------------------

static void ct_swish_inplace(float* x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = swish(x[i]);
}

// ---------------------------------------------------------------------------
// Fast F16→F32 conversion using AVX2/F16C hardware instructions.
// Writes n floats into dst. Falls back to scalar bit-manipulation if no F16C.
// ---------------------------------------------------------------------------
static void ct_f16_to_f32(const uint16_t* src, float* dst, int n) {
#ifdef CT_HAVE_F16C
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if (exp == 0 && mant == 0)
            r = sign;
        else if (exp == 31)
            r = sign | 0x7F800000u | (mant << 13);
        else
            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#else
    for (int i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t r;
        if (exp == 0 && mant == 0)
            r = sign;
        else if (exp == 31)
            r = sign | 0x7F800000u | (mant << 13);
        else
            r = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        memcpy(dst + i, &r, sizeof(float));
    }
#endif
}

// Thread-local F32 buffer for on-the-fly F16→F32 weight conversion.
// Grows to max weight tensor size (~26 MB for enc ffn); never freed.
static thread_local std::vector<float> tl_w_buf;

// Convert an F16 ggml_tensor's data to F32 in tl_w_buf; return pointer.
// Valid until the next call to ct_tensor_f32() on the same thread.
static const float* ct_tensor_f32(const ggml_tensor* t) {
    int n = (int)ggml_nelements(t);
    if ((int)tl_w_buf.size() < n)
        tl_w_buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        ct_f16_to_f32((const uint16_t*)t->data, tl_w_buf.data(), n);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(tl_w_buf.data(), t->data, (size_t)n * sizeof(float));
    } else {
        for (int i = 0; i < n; i++)
            tl_w_buf[i] = ggml_get_f32_1d(const_cast<ggml_tensor*>(t), i);
    }
    return tl_w_buf.data();
}

// ---------------------------------------------------------------------------
// Linear layer: out[m, T] = w[m, n] × in[n, T]  (weight in row-major out×in)
// Returns newly allocated buffer (caller frees).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Iterative Cooley-Tukey FFT — real input, complex interleaved output.
// N must be a power of 2. Output has N complex values (N*2 floats);
// only the first N/2+1 are unique for real input.
// ---------------------------------------------------------------------------
static void cohere_fft_r2c(const float* in, int N, float* out) {
    // Count bits for bit-reversal
    int bits = 0;
    for (int n = N; n > 1; n >>= 1)
        bits++;

    // Bit-reversal copy (real → complex, imaginary = 0)
    for (int i = 0; i < N; i++) {
        int rev = 0;
        for (int b = 0; b < bits; b++)
            rev = (rev << 1) | ((i >> b) & 1);
        out[2 * rev] = in[i];
        out[2 * rev + 1] = 0.0f;
    }

    // Cooley-Tukey butterflies
    for (int len = 2; len <= N; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wre = cosf(ang), wim = sinf(ang);
        for (int i = 0; i < N; i += len) {
            float ure = 1.0f, uim = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                int a = i + j, b = i + j + len / 2;
                float are = out[2 * a], aim = out[2 * a + 1];
                float bre = out[2 * b], bim = out[2 * b + 1];
                float tre = ure * bre - uim * bim, tim = ure * bim + uim * bre;
                out[2 * a] = are + tre;
                out[2 * a + 1] = aim + tim;
                out[2 * b] = are - tre;
                out[2 * b + 1] = aim - tim;
                float new_ure = ure * wre - uim * wim;
                uim = ure * wim + uim * wre;
                ure = new_ure;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Feature extraction: raw PCM → log-mel spectrogram
// Returns float array of shape (n_mels, T_mel), row-major.
// ---------------------------------------------------------------------------

#include "core/mel.h"
#include "core/gpu_backend_pref.h" // crispasr_init_gpu_backend (#214)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static std::vector<float> cohere_compute_features(const cohere_hparams& hp,
                                                  const cohere_frontend::Constants& frontend, const float* samples,
                                                  int n_samples, int n_threads, int& T_out) {
    const int n_fft = hp.n_fft;
    const int hop = hp.hop_length;
    const int win = hp.win_length;
    const int n_freqs = hp.n_freqs();
    const int n_mels = hp.n_mels;
    // HF seeds each waveform's dither from its valid sample count so results
    // are independent of batch composition. Dither precedes pre-emphasis.
    auto prepared = cohere_frontend::dither_and_preemphasize(samples, n_samples, 1e-5f, 0.97f);

    // Configure the shared helper for the NeMo cluster. Note: the original
    // cohere path used cblas_sgemm for the power->mel matmul, while
    // core_mel::compute() does a manual O(n_mels*n_freqs*T) loop. This
    // changes the floating-point accumulation order and can shift the mel
    // output by a handful of ULPs. The resulting transcript is still
    // correct; the regression guard measures transcript-level equality,
    // not bit-exact matmul output.
    core_mel::Params p;
    p.n_fft = n_fft;
    p.hop_length = hop;
    p.win_length = win;
    p.n_mels = n_mels;
    p.log_base = core_mel::LogBase::Ln;
    p.norm = core_mel::Normalization::PerFeatureZ;
    p.layout = core_mel::Layout::TimeMels;
    p.log_eps = (float)(1.0 / (1 << 24));
    p.center_pad = true;
    // Centered STFT physically produces floor(n/hop)+1 frames, while HF marks
    // only floor(n/hop) valid and excludes the tail from normalization and
    // encoder attention. Cropping first gives the same statistics without a
    // padding-mask path and removes the masked encoder tail position.
    p.drop_last_frame = true;
    // Cohere's FFT is re-entrant, and projection frames plus normalization
    // bands are independent. Their inner reductions retain scalar order.
    p.allow_parallel = true;
    p.parallel_threads = n_threads;

    auto mel = core_mel::compute(prepared.data(), n_samples, frontend.window.data(), win,
                                 frontend.mel_filters.data(), n_freqs, cohere_fft_r2c, p, T_out);

    if (cohere_debug_enabled() && mel.size() >= (size_t)5 * n_mels) {
        cohere_debug("cohere: mel m=0, t=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", mel[0 * n_mels + 0],
                     mel[1 * n_mels + 0], mel[2 * n_mels + 0], mel[3 * n_mels + 0], mel[4 * n_mels + 0]);
        cohere_debug("cohere: mel t=0, m=0..4: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", mel[0 * n_mels + 0],
                     mel[0 * n_mels + 1], mel[0 * n_mels + 2], mel[0 * n_mels + 3], mel[0 * n_mels + 4]);
    }

    return mel; // shape: [T, n_mels], time-major
}

// ---------------------------------------------------------------------------
// Conformer: sinusoidal relative positional encoding
// Returns (2T-1, d_model) array, positions from T-1 to -(T-1)
// ---------------------------------------------------------------------------

static std::vector<float> ct_rel_pos_enc(int T, int d_model) {
    int n_pos = 2 * T - 1;
    std::vector<float> pe(n_pos * d_model, 0.0f);
    for (int i = 0; i < n_pos; i++) {
        float pos = (float)(T - 1 - i); // T-1 down to -(T-1)
        for (int j = 0; j < d_model / 2; j++) {
            float div = powf(10000.0f, 2.0f * j / d_model);
            pe[i * d_model + 2 * j] = sinf(pos / div);
            pe[i * d_model + 2 * j + 1] = cosf(pos / div);
        }
    }
    return pe;
}

// ---------------------------------------------------------------------------
// Pre-compute cross-attention K and V for all decoder layers.
// Call once per utterance after encoding; results stored in cross_kv_k/v.
// Layout: cross_kv_k[li] has shape (T_enc, dec_d_model), row-major.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Decoder Graph Builder
// ---------------------------------------------------------------------------

static struct ggml_cgraph* cohere_build_graph_decoder(struct cohere_context* ctx, const int* /*tokens*/, int n_tokens,
                                                      int offset) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.dec_d_model;
    const int n_heads = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;

    cohere_debug("\n--- cohere_build_graph_decoder: n_tokens=%d, offset=%d ---\n", n_tokens, offset);

    struct ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0)
        return nullptr;
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    struct ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(embd, "embd");
    ggml_set_input(embd);

    struct ggml_tensor* position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // PLAN #73: causal mask input for ggml_flash_attn_ext on the
    // self-attention path. Built only for prefill (n_tokens > 1);
    // decode steps pass nullptr.
    const int sa_L = offset + n_tokens;
    struct ggml_tensor* sa_mask = nullptr;
    if (n_tokens > 1) {
        sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, sa_L, n_tokens);
        ggml_set_name(sa_mask, "sa_mask");
        ggml_set_input(sa_mask);
    }

    // [d, n_tokens]
    struct ggml_tensor* cur =
        ggml_add(ctx0, ggml_get_rows(ctx0, model.dec_emb_w, embd), ggml_get_rows(ctx0, model.dec_pos_w, position));
    cohere_log_tensor("input_embd", cur);

    // emb_ln
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul_inplace(ctx0, cur, model.dec_emb_ln_w);
    cur = ggml_add_inplace(ctx0, cur, model.dec_emb_ln_b);
    cohere_log_tensor("emb_ln", cur);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& layer = model.dec_layers[il];
        struct ggml_tensor* inpL = cur;

        cur = ggml_norm(ctx0, inpL, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_ln_b);

        // self-attention Q, K, V
        struct ggml_tensor* Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        struct ggml_tensor* Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        struct ggml_tensor* Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_v_w, cur), layer.attn_v_b);

        if (il == 0) {
            cohere_log_tensor("Qcur", Qcur);
            cohere_log_tensor("Kcur", Kcur);
            cohere_log_tensor("Vcur", Vcur);
        }

        // Store current K, V into cache.
        // PLAN #73: cache write via core_attn helper. F16/F32 caches go
        // the legacy ggml_cpy(view) path (bit-identical); Q8_0/Q4_0
        // caches go ggml_set_rows(position). Cohere already has the
        // `position` I32 graph input populated with [offset, offset+1,
        // …, offset+n_tokens-1] for the dec_pos_w lookup, so it
        // doubles as the row-index input.
        {
            struct ggml_tensor* K_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            struct ggml_tensor* V_perm =
                ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_heads, n_tokens), 0, 2, 1, 3);
            core_attn::kv_cache_write(ctx0, gf, K_perm, V_perm, ctx->kv_k, ctx->kv_v, il, offset, n_tokens, position);
        }

        struct ggml_tensor* Q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_heads, n_tokens), 0, 2, 1,
                                             3); // [hd, n_tok, n_heads]

        // Self-attention: KV cache views for this layer.
        struct ggml_tensor* K =
            ggml_view_3d(ctx0, ctx->kv_k, head_dim, sa_L, n_heads, ctx->kv_k->nb[1], ctx->kv_k->nb[2],
                         il * ctx->kv_k->nb[3]); // [hd, L, n_heads]
        struct ggml_tensor* V =
            ggml_view_3d(ctx0, ctx->kv_v, head_dim, sa_L, n_heads, ctx->kv_v->nb[1], ctx->kv_v->nb[2],
                         il * ctx->kv_v->nb[3]); // [hd, L, n_heads]

        // CRISPASR_COHERE_LEGACY_SA=1: fall back to the pre-v0.7 manual
        // mul_mat self-attention path. The flash_attn_ext path (PLAN #73)
        // fuses Q·K + softmax + V into a single op, but caused a ~10×
        // CUDA regression on some Windows setups (#161). This env var
        // lets users bisect whether flash_attn_ext is the culprit.
        static const bool legacy_sa = (getenv("CRISPASR_COHERE_LEGACY_SA") != nullptr);

        if (legacy_sa) {
            // Legacy path: explicit mul_mat attention (pre-v0.7).
            struct ggml_tensor* K_c = ggml_cont(ctx0, K);
            struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K_c, Q); // [L, n_tok, n_heads]
            KQ = ggml_scale_inplace(ctx0, KQ, 1.0f / sqrtf((float)head_dim));
            if (n_tokens > 1) {
                KQ = ggml_diag_mask_inf_inplace(ctx0, KQ, offset);
            }
            KQ = ggml_soft_max_inplace(ctx0, KQ);

            struct ggml_tensor* V_trans = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 0, 2, 3)); // [L, hd, n_heads]
            struct ggml_tensor* sa_out = ggml_mul_mat(ctx0, V_trans, KQ);                     // [hd, n_tok, n_heads]

            sa_out = ggml_permute(ctx0, sa_out, 0, 2, 1, 3); // [hd, n_heads, n_tok]
            sa_out = ggml_cont(ctx0, sa_out);
            cur = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);
        } else {
            // PLAN #73: ggml_flash_attn_ext fuses K-mul-Q + softmax + V-mul
            // into one op and natively handles quant K/V (no cast tax).
            struct ggml_tensor* sa_out =
                ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, Q), K, V, sa_mask, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);
            // flash_attn_ext output is [hd, n_heads, n_tokens] — same layout
            // as the legacy ggml_permute(sa_out, 0,2,1,3), so the additional
            // permute+cont is gone. reshape_2d packs the inner two dims into
            // d = hd * n_heads.
            cur = ggml_reshape_2d(ctx0, sa_out, d, n_tokens);
        }

        // out projection
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.attn_o_w, cur), layer.attn_o_b);
        cur = ggml_add(ctx0, cur, inpL);
        if (il == 0)
            ggml_set_name(cur, "h0_after_sa");

        struct ggml_tensor* inpCA = cur;

        // cross-attention
        cur = ggml_norm(ctx0, inpCA, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.cross_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.cross_ln_b);

        struct ggml_tensor* CQ = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_q_w, cur), layer.cross_q_b);
        CQ = ggml_reshape_3d(ctx0, CQ, head_dim, n_heads, n_tokens);
        CQ = ggml_permute(ctx0, CQ, 0, 2, 1, 3); // [hd, n_tok, n_heads]

        struct ggml_tensor* CK = ctx->cross_kv_k[il]; // [hd, T_enc, n_heads]
        struct ggml_tensor* CV = ctx->cross_kv_v[il]; // [hd, T_enc, n_heads]

        // cross-attention: no causal mask (encoder output is fully visible)
        struct ggml_tensor* ca_out =
            ggml_flash_attn_ext(ctx0, CQ, CK, CV, nullptr, 1.0f / sqrtf((float)head_dim), 0.0f, 0.0f);

        // For single-token generation steps: also compute explicit cross-attn weights for
        // the last decoder layer so we can use them for token timestamp alignment.
        // CQ: [head_dim, 1, n_heads]   CK: [head_dim, T_enc, n_heads]
        // ca_w = softmax(CQ^T @ CK / sqrt(hd)) : [T_enc, 1, n_heads]
        if (ctx->params.collect_alignment && il == hp.dec_n_layers - 1 && n_tokens == 1) {
            struct ggml_tensor* ca_w = ggml_mul_mat(ctx0, CK, CQ); // [T_enc, 1, n_heads]
            ca_w = ggml_scale_inplace(ctx0, ca_w, 1.0f / sqrtf((float)head_dim));
            ca_w = ggml_soft_max_inplace(ctx0, ca_w);
            ggml_set_name(ca_w, "ca_attn_w");
            ggml_build_forward_expand(gf, ca_w);
        }

        cur = ggml_reshape_2d(ctx0, ca_out, d, n_tokens);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.cross_o_w, cur), layer.cross_o_b);
        cur = ggml_add(ctx0, cur, inpCA);
        if (il == 0)
            ggml_set_name(cur, "h0_after_ca");

        struct ggml_tensor* inpFFN = cur;

        // FFN
        cur = ggml_norm(ctx0, inpFFN, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.ffn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.ffn_ln_b);

        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up_w, cur), layer.ffn_up_b);
        cur = ggml_relu_inplace(ctx0, cur);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_dn_w, cur), layer.ffn_dn_b);


        cur = ggml_add(ctx0, cur, inpFFN);

        if (il == 0) {
            cohere_log_tensor("layer_0_out", cur);
        }
    }

    // final output norm
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul_inplace(ctx0, cur, model.dec_out_ln_w);
    cur = ggml_add_inplace(ctx0, cur, model.dec_out_ln_b);


    // logits
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.dec_head_w, cur), model.dec_head_b);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ---------------------------------------------------------------------------
// Decoder: one step (auto-regressive)
// tokens:   [offset .. offset+n_tok-1]
// Returns logits: (n_tok, vocab_size)
// ---------------------------------------------------------------------------

static std::vector<float> cohere_decode_step(struct cohere_context* ctx, int T_enc, const int* tokens, int n_tok,
                                             int offset) {
    const auto& hp = ctx->model.hparams;
    const int vocab_size = hp.vocab_size;
    auto& perf = ctx->perf;
    int64_t t0, t1;

    ctx->cached_T_enc = T_enc;

    t0 = ggml_time_us();
    struct ggml_cgraph* gf = cohere_build_graph_decoder(ctx, tokens, n_tok, offset);
    t1 = ggml_time_us();
    perf.t_dec_build_us += (t1 - t0);
    if (perf.n_dec_steps == 0)
        perf.dec_n_nodes_prompt = ggml_graph_n_nodes(gf);
    else if (perf.n_dec_steps == 1)
        perf.dec_n_nodes_step = ggml_graph_n_nodes(gf);

    t0 = ggml_time_us();
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate decoder graph\n");
        return {};
    }
    t1 = ggml_time_us();
    perf.t_dec_alloc_us += (t1 - t0);

    // Set inputs
    struct ggml_tensor* embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, tokens, 0, n_tok * sizeof(int));

    struct ggml_tensor* position = ggml_graph_get_tensor(gf, "position");
    std::vector<int> pos_data(n_tok);
    for (int i = 0; i < n_tok; i++)
        pos_data[i] = offset + i;
    ggml_backend_tensor_set(position, pos_data.data(), 0, n_tok * sizeof(int));

    // PLAN #73: populate causal mask for ggml_flash_attn_ext. Built only
    // for prefill (n_tok > 1); decode steps pass nullptr mask.
    if (n_tok > 1) {
        const int L = offset + n_tok;
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask((size_t)n_tok * L, zero);
        for (int q = 0; q < n_tok; q++) {
            for (int k = offset + q + 1; k < L; k++)
                mask[(size_t)q * L + k] = neg_inf;
        }
        struct ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "sa_mask");
        if (mask_in)
            ggml_backend_tensor_set(mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    t0 = ggml_time_us();
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: failed to compute decoder graph\n");
        return {};
    }
    t1 = ggml_time_us();
    {
        int64_t dt = t1 - t0;
        perf.t_dec_compute_us += dt;
        if (dt < perf.t_dec_step_min_us)
            perf.t_dec_step_min_us = dt;
        if (dt > perf.t_dec_step_max_us)
            perf.t_dec_step_max_us = dt;
    }

    // Extract logits (last node in graph)
    struct ggml_tensor* logits_t = ggml_graph_node(gf, -1);
    std::vector<float> logits(n_tok * vocab_size);
    t0 = ggml_time_us();
    ggml_backend_tensor_get(logits_t, logits.data(), 0, logits.size() * sizeof(float));
    t1 = ggml_time_us();
    perf.t_dec_logits_us += (t1 - t0);

    if (offset == 0) {
        // Log some interesting hidden states for the first step (prompt)
        struct ggml_tensor* emb_ln = ggml_graph_get_tensor(gf, "emb_ln");
        if (emb_ln) {
            std::vector<float> h(ggml_nelements(emb_ln));
            ggml_backend_tensor_get(emb_ln, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after emb_ln: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", h[0], h[1],
                         h[2], h[3], h[4]);
        }

        struct ggml_tensor* h0_sa = ggml_graph_get_tensor(gf, "h0_after_sa");
        if (h0_sa) {
            std::vector<float> h(ggml_nelements(h0_sa));
            ggml_backend_tensor_get(h0_sa, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after sa_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", h[0], h[1],
                         h[2], h[3], h[4]);
        }

        struct ggml_tensor* h0_ca = ggml_graph_get_tensor(gf, "h0_after_ca");
        if (h0_ca) {
            std::vector<float> h(ggml_nelements(h0_ca));
            ggml_backend_tensor_get(h0_ca, h.data(), 0, h.size() * sizeof(float));
            cohere_debug("cohere: step 0 (prompt) h[0] after ca_li0: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", h[0], h[1],
                         h[2], h[3], h[4]);
        }

        // Log top logits for last token of prompt
        const float* last_logits = logits.data() + (n_tok - 1) * vocab_size;
        int top_id = 0;
        for (int i = 1; i < vocab_size; i++)
            if (last_logits[i] > last_logits[top_id])
                top_id = i;
        cohere_debug("cohere: step 0 (prompt) top_tok: %d, logit: %.4f, tok_749_logit: %.4f\n", top_id,
                     last_logits[top_id], last_logits[749]);
    }

    // Extract cross-attention alignment weights for token timestamp estimation.
    // Only collected for single-token generation steps (n_tok==1); the graph only
    // contains "ca_attn_w" when alignment collection is enabled and n_tok==1.
    if (ctx->params.collect_alignment && n_tok == 1) {
        struct ggml_tensor* ca_w_t = ggml_graph_get_tensor(gf, "ca_attn_w");
        if (ca_w_t) {
            int T_enc = (int)ca_w_t->ne[0];
            int n_heads = (int)ca_w_t->ne[2];
            ctx->attn_T_enc = T_enc;
            ctx->attn_n_heads = n_heads;
            std::vector<float> w(T_enc * n_heads);
            ggml_backend_tensor_get(ca_w_t, w.data(), 0, w.size() * sizeof(float));
            ctx->step_attn.push_back(std::move(w));
        }
    }

    return logits;
}

// ---------------------------------------------------------------------------
// Research batched decoder. This path owns separate caches and does not alter
// the production B1 decoder above.
// ---------------------------------------------------------------------------

struct cohere_decoder_persistent_graph {
    int self_context = 0;
    std::vector<uint8_t> meta;
    ggml_context* ctx = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
};

struct cohere_decoder_batch_state {
    int n_batch = 0;
    int T_enc = 0;
    int max_ctx = 0;
    bool use_cross_mask = false;
    std::vector<int> valid_T_enc;

    ggml_context* self_ctx = nullptr;
    ggml_tensor* self_k = nullptr;
    ggml_tensor* self_v = nullptr;
    ggml_backend_buffer_t self_buf = nullptr;

    ggml_context* cross_ctx = nullptr;
    std::vector<ggml_tensor*> cross_k;
    std::vector<ggml_tensor*> cross_v;
    ggml_backend_buffer_t cross_buf = nullptr;

    // Shape-stable T=1 graphs. Fixed mode uses one entry; progressive mode
    // lazily retains 64/128/256/final-context entries so CUDA graph keys and
    // allocations remain distinct across bucket switches.
    std::array<cohere_decoder_persistent_graph, 4> persistent_graphs;
    int n_persistent_graphs = 0;
    enum cohere_decoder_batch_output_mode persistent_output_mode = COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS;

    cohere_decoder_batch_state() = default;
    cohere_decoder_batch_state(const cohere_decoder_batch_state&) = delete;
    cohere_decoder_batch_state& operator=(const cohere_decoder_batch_state&) = delete;

    void clear() {
        for (auto& persistent : persistent_graphs) {
            if (persistent.galloc)
                ggml_gallocr_free(persistent.galloc);
            if (persistent.ctx)
                ggml_free(persistent.ctx);
            persistent = {};
        }
        if (cross_buf)
            ggml_backend_buffer_free(cross_buf);
        if (self_buf)
            ggml_backend_buffer_free(self_buf);
        if (cross_ctx)
            ggml_free(cross_ctx);
        if (self_ctx)
            ggml_free(self_ctx);
        cross_buf = nullptr;
        self_buf = nullptr;
        n_persistent_graphs = 0;
        cross_ctx = nullptr;
        self_ctx = nullptr;
        cross_k.clear();
        cross_v.clear();
    }

    ~cohere_decoder_batch_state() { clear(); }
};

static bool cohere_decoder_batch_state_init(struct cohere_context* ctx, cohere_decoder_batch_state& state,
                                            int n_batch, int T_enc, int max_ctx = 0) {
    const auto& hp = ctx->model.hparams;
    if (max_ctx <= 0)
        max_ctx = hp.dec_max_ctx;
    if (max_ctx > hp.dec_max_ctx) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "decoder cache context exceeds the model limit");
        return false;
    }
    state.n_batch = n_batch;
    state.T_enc = T_enc;
    state.max_ctx = max_ctx;

    state.self_ctx = ggml_init({
        .mem_size = ggml_tensor_overhead() * 2 + 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    });
    if (!state.self_ctx) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder self-cache metadata");
        return false;
    }

    const int cache_slots = n_batch * hp.dec_n_layers;
    state.self_k = ggml_new_tensor_4d(state.self_ctx, GGML_TYPE_F16, hp.dec_head_dim, max_ctx,
                                      hp.dec_n_heads, cache_slots);
    state.self_v = ggml_new_tensor_4d(state.self_ctx, GGML_TYPE_F16, hp.dec_head_dim, max_ctx,
                                      hp.dec_n_heads, cache_slots);
    ggml_set_name(state.self_k, "batch_self_k");
    ggml_set_name(state.self_v, "batch_self_v");
    state.self_buf = ggml_backend_alloc_ctx_tensors(state.self_ctx, ctx->ggml_backend);
    if (!state.self_buf) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder self-cache buffer");
        return false;
    }

    state.cross_ctx = ggml_init({
        .mem_size = (size_t)(hp.dec_n_layers * 2 + 4) * ggml_tensor_overhead() + 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    });
    if (!state.cross_ctx) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder cross-cache metadata");
        return false;
    }

    state.cross_k.resize(hp.dec_n_layers);
    state.cross_v.resize(hp.dec_n_layers);
    for (int il = 0; il < hp.dec_n_layers; il++) {
        state.cross_k[il] = ggml_new_tensor_4d(state.cross_ctx, GGML_TYPE_F16, hp.dec_head_dim, T_enc,
                                               hp.dec_n_heads, n_batch);
        state.cross_v[il] = ggml_new_tensor_4d(state.cross_ctx, GGML_TYPE_F16, hp.dec_head_dim, T_enc,
                                               hp.dec_n_heads, n_batch);
        char name[32];
        std::snprintf(name, sizeof(name), "batch_cross_k_%d", il);
        ggml_set_name(state.cross_k[il], name);
        std::snprintf(name, sizeof(name), "batch_cross_v_%d", il);
        ggml_set_name(state.cross_v[il], name);
    }
    state.cross_buf = ggml_backend_alloc_ctx_tensors(state.cross_ctx, ctx->ggml_backend);
    if (!state.cross_buf) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder cross-cache buffer");
        return false;
    }

    ggml_backend_buffer_clear(state.self_buf, 0);
    ggml_backend_buffer_clear(state.cross_buf, 0);
    return true;
}

static bool cohere_decoder_batch_prepare_cross_kv(struct cohere_context* ctx, cohere_decoder_batch_state& state,
                                                  const float* encoder_states) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.dec_d_model;
    const int head_dim = hp.dec_head_dim;
    const int n_heads = hp.dec_n_heads;
    const int T_enc = state.T_enc;
    const int n_batch = state.n_batch;

    ggml_context* ctx0 = cohere_begin_compute_graph(ctx);
    if (!ctx0) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder cross-KV graph metadata");
        return false;
    }
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* enc = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d, T_enc, n_batch);
    ggml_set_name(enc, "batch_encoder_states");
    ggml_set_input(enc);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& layer = model.dec_layers[il];
        ggml_tensor* ck =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.cross_k_w, enc), layer.cross_k_b);
        ggml_tensor* cv =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.cross_v_w, enc), layer.cross_v_b);
        ck = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, ck, head_dim, n_heads, T_enc, n_batch),
                                          0, 2, 1, 3));
        cv = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, cv, head_dim, n_heads, T_enc, n_batch),
                                          0, 2, 1, 3));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ck, state.cross_k[il]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, cv, state.cross_v[il]));
    }

    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate decoder cross-KV graph");
        std::fprintf(stderr, "cohere: failed to allocate batched cross-KV graph\n");
        return false;
    }
    ggml_tensor* enc_input = ggml_graph_get_tensor(gf, "batch_encoder_states");
    const size_t input_bytes = (size_t)n_batch * (size_t)T_enc * (size_t)d * sizeof(float);
    ggml_backend_tensor_set(enc_input, encoder_states, 0, input_bytes);
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        cohere_set_error(COHERE_ERROR_RUNTIME, "decoder cross-KV graph compute failed");
        std::fprintf(stderr, "cohere: failed to compute batched cross-KV graph\n");
        return false;
    }
    return true;
}

struct cohere_decoder_batch_cache_write_result {
    ggml_tensor* k = nullptr;
    ggml_tensor* v = nullptr;
};

static cohere_decoder_batch_cache_write_result cohere_decoder_batch_cache_write(
    ggml_context* ctx0, ggml_cgraph* gf, ggml_tensor* src_k, ggml_tensor* src_v,
    cohere_decoder_batch_state& state, int layer, int offset, int n_tokens,
    ggml_tensor* runtime_positions = nullptr) {
    const int head_dim = (int)state.self_k->ne[0];
    const int n_heads = (int)state.self_k->ne[2];
    const int n_batch = state.n_batch;
    const size_t layer_offset = cohere_decoder_batch::cache_slot(layer, 0, n_batch) * state.self_k->nb[3];

    if (runtime_positions) {
        GGML_ASSERT(src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32);
        GGML_ASSERT(src_k->ne[0] == head_dim && src_k->ne[1] == n_tokens && src_k->ne[2] == n_heads &&
                    src_k->ne[3] == n_batch);
        GGML_ASSERT(ggml_are_same_shape(src_k, src_v));
        GGML_ASSERT(runtime_positions->type == GGML_TYPE_I32 && runtime_positions->ne[0] == n_tokens &&
                    runtime_positions->ne[1] == 1 && runtime_positions->ne[2] == n_batch &&
                    runtime_positions->ne[3] == 1);
        // set_rows treats ne[1] as the cache-row axis. Indices [T,1,B]
        // broadcast across heads while selecting one independent row per lane.
        ggml_tensor* layer_k = ggml_view_4d(ctx0, state.self_k, head_dim, state.max_ctx, n_heads, n_batch,
                                            state.self_k->nb[1], state.self_k->nb[2], state.self_k->nb[3],
                                            layer_offset);
        ggml_tensor* layer_v = ggml_view_4d(ctx0, state.self_v, head_dim, state.max_ctx, n_heads, n_batch,
                                            state.self_v->nb[1], state.self_v->nb[2], state.self_v->nb[3],
                                            layer_offset);
        ggml_tensor* written_k = ggml_set_rows(ctx0, layer_k, src_k, runtime_positions);
        ggml_tensor* written_v = ggml_set_rows(ctx0, layer_v, src_v, runtime_positions);
        ggml_build_forward_expand(gf, written_k);
        ggml_build_forward_expand(gf, written_v);
        return {written_k, written_v};
    }

    ggml_tensor* dst_k = ggml_view_4d(ctx0, state.self_k, head_dim, n_tokens, n_heads, n_batch,
                                      state.self_k->nb[1], state.self_k->nb[2], state.self_k->nb[3],
                                      layer_offset + (size_t)offset * state.self_k->nb[1]);
    ggml_tensor* dst_v = ggml_view_4d(ctx0, state.self_v, head_dim, n_tokens, n_heads, n_batch,
                                      state.self_v->nb[1], state.self_v->nb[2], state.self_v->nb[3],
                                      layer_offset + (size_t)offset * state.self_v->nb[1]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src_k, dst_k));
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, src_v, dst_v));
    return {state.self_k, state.self_v};
}

static bool cohere_decoder_batch_output_mode_valid(enum cohere_decoder_batch_output_mode output_mode) {
    return output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS ||
           output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX;
}

static ggml_cgraph* cohere_build_graph_decoder_batch(
    struct cohere_context* ctx, cohere_decoder_batch_state& state, int n_tokens, int offset,
    enum cohere_decoder_batch_output_mode output_mode = COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS,
    ggml_context* persistent_ctx = nullptr, int fixed_self_context = 0) {
    const auto& model = ctx->model;
    const auto& hp = model.hparams;
    const int d = hp.dec_d_model;
    const int n_heads = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;
    const int n_batch = state.n_batch;
    const int sa_L = fixed_self_context > 0 ? fixed_self_context : offset + n_tokens;
    if (n_tokens < 1 || offset < 0 || sa_L > state.max_ctx ||
        (fixed_self_context > 0 && n_tokens != 1) ||
        !cohere_decoder_batch_output_mode_valid(output_mode))
        return nullptr;

    ggml_context* ctx0 = persistent_ctx;
    if (!ctx0) {
        ctx0 = cohere_begin_compute_graph(ctx);
    }
    if (!ctx0)
        return nullptr;
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 4096, false);

    ggml_tensor* embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_batch);
    ggml_set_name(embd, "batch_embd");
    ggml_set_input(embd);
    ggml_tensor* position = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * n_batch);
    ggml_set_name(position, "batch_position");
    ggml_set_input(position);

    ggml_tensor* cache_position = nullptr;
    if (fixed_self_context > 0) {
        cache_position = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, n_tokens, 1, n_batch);
        ggml_set_name(cache_position, "batch_cache_position");
        ggml_set_input(cache_position);
    }

    ggml_tensor* sa_mask = nullptr;
    if (n_tokens > 1 || fixed_self_context > 0) {
        sa_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, sa_L, n_tokens);
        ggml_set_name(sa_mask, "batch_sa_mask");
        ggml_set_input(sa_mask);
    }

    ggml_tensor* cross_mask = nullptr;
    if (state.use_cross_mask) {
        cross_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, state.T_enc, n_tokens, 1, n_batch);
        ggml_set_name(cross_mask, "batch_cross_mask");
        ggml_set_input(cross_mask);
    }

    ggml_tensor* token_embedding = ggml_reshape_3d(
        ctx0, ggml_get_rows(ctx0, model.dec_emb_w, embd), d, n_tokens, n_batch);
    ggml_tensor* position_embedding = ggml_reshape_3d(
        ctx0, ggml_get_rows(ctx0, model.dec_pos_w, position), d, n_tokens, n_batch);
    ggml_tensor* cur = ggml_add(ctx0, token_embedding, position_embedding);
    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul_inplace(ctx0, cur, model.dec_emb_ln_w);
    cur = ggml_add_inplace(ctx0, cur, model.dec_emb_ln_b);

    for (int il = 0; il < hp.dec_n_layers; il++) {
        const auto& layer = model.dec_layers[il];
        ggml_tensor* residual = cur;

        cur = ggml_norm(ctx0, residual, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.attn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.attn_ln_b);

        ggml_tensor* q_cur =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.attn_q_w, cur), layer.attn_q_b);
        ggml_tensor* k_cur =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.attn_k_w, cur), layer.attn_k_b);
        ggml_tensor* v_cur =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.attn_v_w, cur), layer.attn_v_b);
        ggml_tensor* q = ggml_permute(ctx0, ggml_reshape_4d(ctx0, q_cur, head_dim, n_heads, n_tokens, n_batch),
                                      0, 2, 1, 3);
        ggml_tensor* k_new = ggml_permute(
            ctx0, ggml_reshape_4d(ctx0, k_cur, head_dim, n_heads, n_tokens, n_batch), 0, 2, 1, 3);
        ggml_tensor* v_new = ggml_permute(
            ctx0, ggml_reshape_4d(ctx0, v_cur, head_dim, n_heads, n_tokens, n_batch), 0, 2, 1, 3);
        const cohere_decoder_batch_cache_write_result written = cohere_decoder_batch_cache_write(
            ctx0, gf, k_new, v_new, state, il, offset, n_tokens, cache_position);

        const size_t layer_offset = cohere_decoder_batch::cache_slot(il, 0, n_batch) * state.self_k->nb[3];
        ggml_tensor* k = ggml_view_4d(ctx0, written.k, head_dim, sa_L, n_heads, n_batch,
                                      state.self_k->nb[1], state.self_k->nb[2], state.self_k->nb[3],
                                      cache_position ? 0 : layer_offset);
        ggml_tensor* v = ggml_view_4d(ctx0, written.v, head_dim, sa_L, n_heads, n_batch,
                                      state.self_v->nb[1], state.self_v->nb[2], state.self_v->nb[3],
                                      cache_position ? 0 : layer_offset);
        ggml_tensor* sa = ggml_flash_attn_ext(ctx0, ggml_cont(ctx0, q), k, v, sa_mask,
                                              1.0f / std::sqrt((float)head_dim), 0.0f, 0.0f);
        cur = ggml_reshape_3d(ctx0, sa, d, n_tokens, n_batch);
        cur = ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.attn_o_w, cur), layer.attn_o_b);
        cur = ggml_add(ctx0, cur, residual);

        residual = cur;
        cur = ggml_norm(ctx0, residual, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.cross_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.cross_ln_b);
        ggml_tensor* cq =
            ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.cross_q_w, cur), layer.cross_q_b);
        cq = ggml_permute(ctx0, ggml_reshape_4d(ctx0, cq, head_dim, n_heads, n_tokens, n_batch), 0, 2, 1, 3);
        ggml_tensor* ca = ggml_flash_attn_ext(ctx0, cq, state.cross_k[il], state.cross_v[il], cross_mask,
                                              1.0f / std::sqrt((float)head_dim), 0.0f, 0.0f);
        cur = ggml_reshape_3d(ctx0, ca, d, n_tokens, n_batch);
        cur = ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.cross_o_w, cur), layer.cross_o_b);
        cur = ggml_add(ctx0, cur, residual);

        residual = cur;
        cur = ggml_norm(ctx0, residual, 1e-5f);
        cur = ggml_mul_inplace(ctx0, cur, layer.ffn_ln_w);
        cur = ggml_add_inplace(ctx0, cur, layer.ffn_ln_b);
        cur = ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.ffn_up_w, cur), layer.ffn_up_b);
        cur = ggml_relu_inplace(ctx0, cur);
        cur = ggml_add(ctx0, cohere_precise_batch_projection(ctx0, layer.ffn_dn_w, cur), layer.ffn_dn_b);
        cur = ggml_add(ctx0, cur, residual);
        if (persistent_ctx) {
            // Keep the layer boundary live across raw-gallocr graph replays.
            // Without this anchor the in-place normalization path aliases a
            // later layer over `cur`, making the second replay non-finite.
            char name[48];
            std::snprintf(name, sizeof(name), "persistent_replay_anchor_%d", il);
            ggml_set_name(cur, name);
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }
    }

    cur = ggml_norm(ctx0, cur, 1e-5f);
    cur = ggml_mul_inplace(ctx0, cur, model.dec_out_ln_w);
    cur = ggml_add_inplace(ctx0, cur, model.dec_out_ln_b);
    cur = ggml_add(ctx0, cohere_precise_batch_projection(ctx0, model.dec_head_w, cur), model.dec_head_b);

    ggml_tensor* last = ggml_view_3d(ctx0, cur, hp.vocab_size, 1, n_batch, cur->nb[1], cur->nb[2],
                                     (size_t)(n_tokens - 1) * cur->nb[1]);
    last = ggml_cont(ctx0, last);
    ggml_set_name(last, "batch_logits");
    if (output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS) {
        ggml_set_output(last);
        ggml_build_forward_expand(gf, last);
    } else {
        // ggml_argmax reduces ne[0] of a matrix. The last-token view is
        // physically [vocab,1,batch], so flatten its non-vocab axes to [V,B].
        if (last->type != GGML_TYPE_F32 || last->ne[0] != hp.vocab_size || last->ne[1] != 1 ||
            last->ne[2] != n_batch || last->ne[3] != 1 ||
            ggml_nelements(last) != (int64_t)hp.vocab_size * n_batch || !ggml_is_contiguous(last)) {
            return nullptr;
        }
        ggml_tensor* rows = ggml_reshape_2d(ctx0, last, hp.vocab_size, n_batch);
        if (!rows || !ggml_is_matrix(rows))
            return nullptr;
        ggml_tensor* argmax = ggml_argmax(ctx0, rows);
        ggml_set_name(argmax, "batch_argmax");
        ggml_set_output(argmax);
        ggml_build_forward_expand(gf, argmax);
    }
    return gf;
}

static bool cohere_decoder_batch_set_positions(ggml_cgraph* gf, const int* tokens, int n_tokens,
                                               int n_batch, int offset) {
    ggml_tensor* embd = ggml_graph_get_tensor(gf, "batch_embd");
    ggml_tensor* position = ggml_graph_get_tensor(gf, "batch_position");
    if (!embd || !position)
        return false;
    ggml_backend_tensor_set(embd, tokens, 0, (size_t)n_tokens * (size_t)n_batch * sizeof(int));
    std::vector<int> positions((size_t)n_tokens * (size_t)n_batch);
    for (int lane = 0; lane < n_batch; lane++) {
        for (int token = 0; token < n_tokens; token++) {
            positions[cohere_decoder_batch::graph_token_index(lane, token, n_tokens)] = offset + token;
        }
    }
    ggml_backend_tensor_set(position, positions.data(), 0, positions.size() * sizeof(int));
    if (ggml_tensor* cache_position = ggml_graph_get_tensor(gf, "batch_cache_position")) {
        ggml_backend_tensor_set(cache_position, positions.data(), 0, positions.size() * sizeof(int));
    }
    return true;
}

static bool cohere_decoder_batch_set_self_mask(ggml_cgraph* gf, int n_tokens, int offset) {
    ggml_tensor* mask_input = ggml_graph_get_tensor(gf, "batch_sa_mask");
    if (!mask_input)
        return n_tokens == 1;
    if (mask_input->type != GGML_TYPE_F16 || mask_input->ne[1] != n_tokens)
        return false;

    const int mask_length = (int)mask_input->ne[0];
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask((size_t)n_tokens * (size_t)mask_length, neg_inf);
    for (int query = 0; query < n_tokens; query++) {
        const int visible = std::min(mask_length, offset + query + 1);
        std::fill_n(mask.data() + (size_t)query * (size_t)mask_length, visible, zero);
    }
    ggml_backend_tensor_set(mask_input, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool cohere_decoder_batch_set_cross_mask(ggml_cgraph* gf, const cohere_decoder_batch_state& state,
                                                int n_tokens) {
    if (!state.use_cross_mask)
        return true;
    ggml_tensor* mask_input = ggml_graph_get_tensor(gf, "batch_cross_mask");
    if (!mask_input)
        return false;

    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t padded = ggml_fp32_to_fp16(-10000.0f);
    std::vector<ggml_fp16_t> mask((size_t)state.n_batch * (size_t)n_tokens * (size_t)state.T_enc, padded);
    for (int lane = 0; lane < state.n_batch; lane++) {
        for (int query = 0; query < n_tokens; query++) {
            for (int key = 0; key < state.valid_T_enc[lane]; key++) {
                mask[cohere_decoder_batch::cross_mask_index(lane, query, key, n_tokens, state.T_enc)] = zero;
            }
        }
    }
    ggml_backend_tensor_set(mask_input, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool cohere_decoder_batch_graph_supported(ggml_backend_t backend, ggml_cgraph* gf) {
    if (!backend || !gf)
        return false;
    for (int node = 0; node < ggml_graph_n_nodes(gf); node++) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, node)))
            return false;
    }
    return true;
}

static cohere_decoder_persistent_graph* cohere_decoder_batch_persistent_get(
    struct cohere_context* ctx, cohere_decoder_batch_state& state,
    enum cohere_decoder_batch_output_mode output_mode, int self_context) {
    if (self_context < 1 || self_context > state.max_ctx)
        return nullptr;
    for (int index = 0; index < state.n_persistent_graphs; index++) {
        auto& persistent = state.persistent_graphs[(size_t)index];
        if (persistent.self_context == self_context)
            return state.persistent_output_mode == output_mode ? &persistent : nullptr;
    }
    if (state.n_persistent_graphs >= (int)state.persistent_graphs.size() ||
        (state.n_persistent_graphs > 0 && state.persistent_output_mode != output_mode)) {
        return nullptr;
    }

    auto& persistent = state.persistent_graphs[(size_t)state.n_persistent_graphs];
    persistent.self_context = self_context;
    persistent.meta.assign(ctx->compute_meta.size(), 0);
    persistent.ctx = ggml_init({
        .mem_size = persistent.meta.size(),
        .mem_buffer = persistent.meta.data(),
        .no_alloc = true,
    });
    if (!persistent.ctx)
        return nullptr;
    persistent.graph = cohere_build_graph_decoder_batch(
        ctx, state, 1, 0, output_mode, persistent.ctx, self_context);
    if (!persistent.graph ||
        !cohere_decoder_batch_graph_supported(ctx->ggml_backend, persistent.graph)) {
        return nullptr;
    }
    persistent.galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->ggml_backend));
    if (!persistent.galloc ||
        !ggml_gallocr_alloc_graph(persistent.galloc, persistent.graph)) {
        return nullptr;
    }
    if (!cohere_decoder_batch_set_cross_mask(persistent.graph, state, 1))
        return nullptr;
    state.persistent_output_mode = output_mode;
    state.n_persistent_graphs++;
    return &persistent;
}

static bool cohere_decode_step_batch_persistent_ex(
    struct cohere_context* ctx, cohere_decoder_batch_state& state,
    cohere_decoder_persistent_graph& persistent, const int* tokens, int offset,
    enum cohere_decoder_batch_output_mode output_mode, std::vector<float>* logits,
    std::vector<int32_t>* token_ids) {
    const auto& hp = ctx->model.hparams;
    if (!tokens || !persistent.graph || state.persistent_output_mode != output_mode ||
        offset < 0 || offset >= persistent.self_context ||
        (output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS && (!logits || token_ids)) ||
        (output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX && (!token_ids || logits))) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "invalid persistent ragged decoder step arguments");
        std::fprintf(stderr, "cohere: invalid persistent decoder step at offset %d (context=%d)\n",
                     offset, persistent.self_context);
        return false;
    }
    if (cohere_abort_if_requested(ctx))
        return false;
    ggml_cgraph* gf = persistent.graph;
    if (!cohere_decoder_batch_set_positions(gf, tokens, 1, state.n_batch, offset) ||
        !cohere_decoder_batch_set_self_mask(gf, 1, offset)) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "could not populate persistent decoder graph inputs");
        std::fprintf(stderr, "cohere: failed to set persistent decoder inputs at offset %d\n", offset);
        return false;
    }
    if (ggml_backend_graph_compute(ctx->ggml_backend, gf) != GGML_STATUS_SUCCESS) {
        if (cohere_abort_if_requested(ctx))
            return false;
        cohere_set_error(COHERE_ERROR_RUNTIME, "persistent ragged decoder graph compute failed");
        std::fprintf(stderr, "cohere: persistent batched decoder graph compute failed\n");
        return false;
    }
    if (output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS) {
        ggml_tensor* output = ggml_graph_get_tensor(gf, "batch_logits");
        if (!output || output->type != GGML_TYPE_F32 ||
            ggml_nelements(output) != (int64_t)state.n_batch * hp.vocab_size) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "persistent decoder returned an invalid logits tensor");
            std::fprintf(stderr, "cohere: invalid persistent full-logit output at offset %d\n", offset);
            return false;
        }
        logits->resize((size_t)state.n_batch * (size_t)hp.vocab_size);
        ggml_backend_tensor_get(output, logits->data(), 0, logits->size() * sizeof(float));
    } else {
        ggml_tensor* output = ggml_graph_get_tensor(gf, "batch_argmax");
        if (!output || output->type != GGML_TYPE_I32 || output->ne[0] != state.n_batch ||
            ggml_nelements(output) != state.n_batch) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "persistent decoder returned an invalid token tensor");
            std::fprintf(stderr, "cohere: invalid persistent argmax output at offset %d\n", offset);
            return false;
        }
        token_ids->resize((size_t)state.n_batch);
        ggml_backend_tensor_get(output, token_ids->data(), 0, token_ids->size() * sizeof(int32_t));
        for (int32_t token_id : *token_ids) {
            if (token_id < 0 || token_id >= hp.vocab_size) {
                cohere_set_error(COHERE_ERROR_INVARIANT, "persistent decoder returned an out-of-range token");
                std::fprintf(stderr, "cohere: invalid persistent token %d at offset %d\n", token_id, offset);
                return false;
            }
        }
    }
    return true;
}

static bool cohere_decode_step_batch_ex(struct cohere_context* ctx, cohere_decoder_batch_state& state,
                                        const int* tokens, int n_tokens, int offset,
                                        enum cohere_decoder_batch_output_mode output_mode,
                                        std::vector<float>* logits, std::vector<int32_t>* token_ids) {
    const auto& hp = ctx->model.hparams;
    if (!tokens || !cohere_decoder_batch_output_mode_valid(output_mode) ||
        (output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS && (!logits || token_ids)) ||
        (output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX && (!token_ids || logits))) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "invalid ragged decoder step arguments");
        return false;
    }
    if (cohere_abort_if_requested(ctx))
        return false;
    ggml_cgraph* gf = cohere_build_graph_decoder_batch(ctx, state, n_tokens, offset, output_mode);
    if (!gf) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "could not build the ragged decoder graph");
        return false;
    }

    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate the ragged decoder graph");
        std::fprintf(stderr, "cohere: failed to allocate batched decoder graph\n");
        return false;
    }

    if (!cohere_decoder_batch_set_positions(gf, tokens, n_tokens, state.n_batch, offset) ||
        !cohere_decoder_batch_set_self_mask(gf, n_tokens, offset) ||
        !cohere_decoder_batch_set_cross_mask(gf, state, n_tokens)) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "could not populate ragged decoder graph inputs");
        return false;
    }

    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        if (cohere_abort_if_requested(ctx))
            return false;
        cohere_set_error(COHERE_ERROR_RUNTIME, "ragged decoder graph compute failed");
        std::fprintf(stderr, "cohere: failed to compute batched decoder graph\n");
        return false;
    }

    if (output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS) {
        ggml_tensor* output = ggml_graph_get_tensor(gf, "batch_logits");
        if (!output || output->type != GGML_TYPE_F32 ||
            ggml_nelements(output) != (int64_t)state.n_batch * hp.vocab_size) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "ragged decoder returned an invalid logits tensor");
            return false;
        }
        logits->resize((size_t)state.n_batch * (size_t)hp.vocab_size);
        ggml_backend_tensor_get(output, logits->data(), 0, logits->size() * sizeof(float));
    } else {
        ggml_tensor* output = ggml_graph_get_tensor(gf, "batch_argmax");
        if (!output || output->type != GGML_TYPE_I32 || output->ne[0] != state.n_batch ||
            ggml_nelements(output) != state.n_batch) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "ragged decoder returned an invalid token tensor");
            return false;
        }
        token_ids->resize((size_t)state.n_batch);
        ggml_backend_tensor_get(output, token_ids->data(), 0, token_ids->size() * sizeof(int32_t));
        for (int32_t token_id : *token_ids) {
            if (token_id < 0 || token_id >= hp.vocab_size) {
                cohere_set_error(COHERE_ERROR_INVARIANT, "ragged decoder returned an out-of-range token");
                return false;
            }
        }
    }
    return true;
}

static bool cohere_decode_step_batch(struct cohere_context* ctx, cohere_decoder_batch_state& state,
                                     const int* tokens, int n_tokens, int offset, std::vector<float>& logits) {
    return cohere_decode_step_batch_ex(ctx, state, tokens, n_tokens, offset,
                                       COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS, &logits, nullptr);
}

static bool cohere_decode_step_batch_argmax(struct cohere_context* ctx, cohere_decoder_batch_state& state,
                                            const int* tokens, int n_tokens, int offset,
                                            std::vector<int32_t>& token_ids) {
    return cohere_decode_step_batch_ex(ctx, state, tokens, n_tokens, offset,
                                       COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX, nullptr, &token_ids);
}

void cohere_decoder_batch_probe_result_free(struct cohere_decoder_batch_probe_result* result) {
    if (!result)
        return;
    std::free(result->logits);
    std::free(result->argmax_ids);
    std::free(result);
}

static struct cohere_decoder_batch_probe_result* cohere_run_decoder_batch_probe_impl(
    struct cohere_context* ctx, const float* encoder_states, int n_batch, int T_enc, int d_model,
    const int* teacher_tokens, int n_teacher_steps, const int* valid_T_enc) {
    if (!ctx || !encoder_states || n_batch < 1 || n_batch > cohere_decoder_batch::max_batch || T_enc < 1 ||
        T_enc > cohere_decoder_batch::max_encoder_frames || d_model != ctx->model.hparams.dec_d_model ||
        n_teacher_steps < 0 || n_teacher_steps > cohere_decoder_batch::max_teacher_steps ||
        (n_teacher_steps > 0 && !teacher_tokens)) {
        return nullptr;
    }

    if (valid_T_enc) {
        for (int lane = 0; lane < n_batch; lane++) {
            if (!cohere_decoder_batch::valid_cross_length(valid_T_enc[lane], T_enc))
                return nullptr;
        }
    }

    const auto& hp = ctx->model.hparams;
    const auto& voc = ctx->vocab;
    for (int i = 0; i < n_teacher_steps * n_batch; i++) {
        if (teacher_tokens[i] < 0 || teacher_tokens[i] >= hp.vocab_size)
            return nullptr;
    }

    auto token_id = [&](const char* text) { return voc.token_id(text); };
    const char* punctuation = ctx->params.no_punctuation ? "<|nopnc|>" : "<|pnc|>";
    std::vector<int> prompt = {
        token_id("▁"),
        token_id("<|startofcontext|>"),
        token_id("<|startoftranscript|>"),
        token_id("<|emo:undefined|>"),
        token_id("<|ar|>"),
        token_id("<|ar|>"),
        token_id(punctuation),
        token_id("<|noitn|>"),
        token_id("<|notimestamp|>"),
        token_id(ctx->params.diarize ? "<|diarize|>" : "<|nodiarize|>"),
    };
    if (std::any_of(prompt.begin(), prompt.end(), [](int token) { return token < 0; }) ||
        (int)prompt.size() + n_teacher_steps > hp.dec_max_ctx) {
        return nullptr;
    }

    auto* result = (cohere_decoder_batch_probe_result*)std::calloc(1, sizeof(cohere_decoder_batch_probe_result));
    if (!result)
        return nullptr;
    result->n_outputs = n_teacher_steps + 1;
    result->n_batch = n_batch;
    result->vocab_size = hp.vocab_size;
    const size_t n_output_rows = (size_t)result->n_outputs * (size_t)n_batch;
    result->logits = (float*)std::malloc(n_output_rows * (size_t)hp.vocab_size * sizeof(float));
    result->argmax_ids = (int*)std::malloc(n_output_rows * sizeof(int));
    if (!result->logits || !result->argmax_ids) {
        cohere_decoder_batch_probe_result_free(result);
        return nullptr;
    }

    cohere_decoder_batch_state state;
    if (valid_T_enc) {
        state.use_cross_mask = true;
        state.valid_T_enc.assign(valid_T_enc, valid_T_enc + n_batch);
    }
    if (!cohere_decoder_batch_state_init(ctx, state, n_batch, T_enc) ||
        !cohere_decoder_batch_prepare_cross_kv(ctx, state, encoder_states)) {
        cohere_decoder_batch_probe_result_free(result);
        return nullptr;
    }
    result->self_cache_bytes = ggml_backend_buffer_get_size(state.self_buf);
    result->cross_cache_bytes = ggml_backend_buffer_get_size(state.cross_buf);

    std::vector<int> packed_prompt((size_t)n_batch * prompt.size());
    for (int lane = 0; lane < n_batch; lane++) {
        for (int token = 0; token < (int)prompt.size(); token++) {
            packed_prompt[cohere_decoder_batch::graph_token_index(lane, token, (int)prompt.size())] = prompt[token];
        }
    }

    std::vector<float> step_logits;
    auto store_output = [&](int output) {
        float* destination = result->logits + (size_t)output * (size_t)n_batch * (size_t)hp.vocab_size;
        std::memcpy(destination, step_logits.data(), step_logits.size() * sizeof(float));
        for (int lane = 0; lane < n_batch; lane++) {
            const float* lane_logits = step_logits.data() + (size_t)lane * (size_t)hp.vocab_size;
            result->argmax_ids[(size_t)output * (size_t)n_batch + (size_t)lane] =
                (int)(std::max_element(lane_logits, lane_logits + hp.vocab_size) - lane_logits);
        }
    };

    if (!cohere_decode_step_batch(ctx, state, packed_prompt.data(), (int)prompt.size(), 0, step_logits)) {
        cohere_decoder_batch_probe_result_free(result);
        return nullptr;
    }
    store_output(0);

    for (int step = 0; step < n_teacher_steps; step++) {
        const int* step_tokens = teacher_tokens + cohere_decoder_batch::sequence_token_index(step, 0, n_batch);
        if (!cohere_decode_step_batch(ctx, state, step_tokens, 1, (int)prompt.size() + step, step_logits)) {
            cohere_decoder_batch_probe_result_free(result);
            return nullptr;
        }
        store_output(step + 1);
    }
    return result;
}

struct cohere_decoder_batch_probe_result* cohere_run_decoder_batch_probe(
    struct cohere_context* ctx, const float* encoder_states, int n_batch, int T_enc, int d_model,
    const int* teacher_tokens, int n_teacher_steps) {
    return cohere_run_decoder_batch_probe_impl(ctx, encoder_states, n_batch, T_enc, d_model, teacher_tokens,
                                               n_teacher_steps, nullptr);
}

struct cohere_decoder_batch_probe_result* cohere_run_decoder_batch_masked_probe(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const int* teacher_tokens, int n_teacher_steps) {
    if (!valid_T_enc)
        return nullptr;
    return cohere_run_decoder_batch_probe_impl(ctx, encoder_states, n_batch, T_enc_max, d_model, teacher_tokens,
                                               n_teacher_steps, valid_T_enc);
}

void cohere_decoder_batch_generate_result_free(struct cohere_decoder_batch_generate_result* result) {
    if (!result)
        return;
    if (result->sequences) {
        for (int lane = 0; lane < result->n_batch; lane++) {
            std::free(result->sequences[lane].token_ids);
            std::free(result->sequences[lane].probabilities);
        }
    }
    std::free(result->sequences);
    std::free(result);
}

enum class cohere_decoder_persistent_policy {
    disabled,
    fixed,
    progressive_64,
    progressive_128,
    progressive_256,
};

static cohere_decoder_persistent_policy cohere_decoder_batch_persistent_policy_from_env() {
    const char* value = std::getenv("COHERE_BATCH_DECODER_PERSISTENT");
    if (!value || !*value || std::strcmp(value, "0") == 0)
        return cohere_decoder_persistent_policy::disabled;
    if (std::strcmp(value, "progressive") == 0 || std::strcmp(value, "progressive64") == 0 ||
        std::strcmp(value, "buckets") == 0)
        return cohere_decoder_persistent_policy::progressive_64;
    if (std::strcmp(value, "progressive128") == 0)
        return cohere_decoder_persistent_policy::progressive_128;
    if (std::strcmp(value, "progressive256") == 0)
        return cohere_decoder_persistent_policy::progressive_256;
    return cohere_decoder_persistent_policy::fixed;
}

static int cohere_decoder_batch_persistent_context_for_offset(
    cohere_decoder_persistent_policy policy, int offset, int max_context) {
    if (policy == cohere_decoder_persistent_policy::disabled || offset < 0 || offset >= max_context)
        return 0;
    if (policy == cohere_decoder_persistent_policy::fixed)
        return max_context;
    constexpr int progressive_contexts[] = {64, 128, 256};
    const int minimum_context = policy == cohere_decoder_persistent_policy::progressive_256
                                    ? 256
                                    : policy == cohere_decoder_persistent_policy::progressive_128 ? 128 : 64;
    const int required = offset + 1;
    for (int context : progressive_contexts) {
        if (context >= minimum_context && required <= context && context < max_context)
            return context;
    }
    return max_context;
}

struct cohere_decoder_batch_generate_result* cohere_run_decoder_batch_generate_ex(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const char* lang, int max_new_tokens, enum cohere_decoder_batch_output_mode output_mode) {
    cohere_clear_error();
    if (!ctx || !encoder_states || !valid_T_enc || n_batch < 1 ||
        n_batch > cohere_decoder_batch::max_batch || T_enc_max < 1 ||
        T_enc_max > cohere_decoder_batch::max_encoder_frames || d_model != ctx->model.hparams.dec_d_model ||
        max_new_tokens < 0 || ctx->beam_size != 1 || ctx->decode_temperature != 0.0f ||
        ctx->frequency_penalty != 0.0f || ctx->params.collect_alignment ||
        !cohere_decoder_batch_output_mode_valid(output_mode)) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "invalid ragged decoder generation arguments");
        return nullptr;
    }
    if (cohere_abort_if_requested(ctx))
        return nullptr;
    for (int lane = 0; lane < n_batch; lane++) {
        if (!cohere_decoder_batch::valid_cross_length(valid_T_enc[lane], T_enc_max)) {
            cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "ragged decoder encoder length is out of range");
            return nullptr;
        }
    }

    const int64_t total_start_us = ggml_time_us();
    const auto& hp = ctx->model.hparams;
    const auto& voc = ctx->vocab;
    auto token_id = [&](const std::string& text) { return voc.token_id(text); };
    const char* language = lang ? lang : "en";
    char language_token[32];
    const int language_token_size = std::snprintf(language_token, sizeof(language_token), "<|%s|>", language);
    if (language_token_size < 0 || language_token_size >= (int)sizeof(language_token)) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "decoder language token is too long");
        return nullptr;
    }
    const char* punctuation = ctx->params.no_punctuation ? "<|nopnc|>" : "<|pnc|>";
    std::vector<int> prompt = {
        token_id("▁"),
        token_id("<|startofcontext|>"),
        token_id("<|startoftranscript|>"),
        token_id("<|emo:undefined|>"),
        token_id(language_token),
        token_id(language_token),
        token_id(punctuation),
        token_id("<|noitn|>"),
        token_id("<|notimestamp|>"),
        token_id(ctx->params.diarize ? "<|diarize|>" : "<|nodiarize|>"),
    };
    const int eos_id = token_id("<|endoftext|>");
    if (std::any_of(prompt.begin(), prompt.end(), [](int token) { return token < 0; }) || eos_id < 0) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "model vocabulary is missing a required decoder token");
        return nullptr;
    }

    const int generation_capacity = hp.dec_max_ctx - (int)prompt.size() - 4;
    int generation_limit = generation_capacity;
    if (ctx->max_new_tokens > 0)
        generation_limit = std::min(generation_limit, ctx->max_new_tokens);
    if (max_new_tokens > 0)
        generation_limit = std::min(generation_limit, max_new_tokens);
    if (generation_limit <= 0) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "decoder generation limit is not positive");
        return nullptr;
    }

    cohere_decoder_batch_state state;
    state.use_cross_mask = true;
    state.valid_T_enc.assign(valid_T_enc, valid_T_enc + n_batch);
    const int cache_max_ctx = (int)prompt.size() + generation_limit;
    if (!cohere_decoder_batch_state_init(ctx, state, n_batch, T_enc_max, cache_max_ctx))
        return nullptr;

    const int64_t cross_start_us = ggml_time_us();
    if (!cohere_decoder_batch_prepare_cross_kv(ctx, state, encoder_states))
        return nullptr;
    if (cohere_abort_if_requested(ctx))
        return nullptr;
    const int64_t cross_kv_us = ggml_time_us() - cross_start_us;

    std::vector<int> packed_prompt((size_t)n_batch * prompt.size());
    for (int lane = 0; lane < n_batch; lane++) {
        for (int token = 0; token < (int)prompt.size(); token++) {
            packed_prompt[cohere_decoder_batch::graph_token_index(lane, token, (int)prompt.size())] = prompt[token];
        }
    }

    std::vector<float> logits;
    std::vector<int32_t> device_token_ids;
    int64_t decode_us = 0;
    int decode_calls = 0;
    {
        const int64_t start_us = ggml_time_us();
        const bool decoded = output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS
                                 ? cohere_decode_step_batch(ctx, state, packed_prompt.data(), (int)prompt.size(), 0,
                                                            logits)
                                 : cohere_decode_step_batch_argmax(ctx, state, packed_prompt.data(),
                                                                   (int)prompt.size(), 0, device_token_ids);
        if (!decoded)
            return nullptr;
        if (cohere_abort_if_requested(ctx))
            return nullptr;
        decode_us += ggml_time_us() - start_us;
        decode_calls++;
    }

    const cohere_decoder_persistent_policy persistent_policy =
        cohere_decoder_batch_persistent_policy_from_env();
    cohere_decoder_persistent_graph* persistent_graph = nullptr;
    bool persistent_active = false;
    bool persistent_used = false;
    int64_t persistent_init_us = 0;
    int persistent_decode_calls = 0;
    int64_t persistent_decode_us = 0;
    int64_t reserve_us = 0;
    bool dynamic_reserved = false;
    auto reserve_dynamic = [&]() {
        if (dynamic_reserved)
            return true;
        const int64_t reserve_start_us = ggml_time_us();
        ggml_cgraph* reserve_graph =
            cohere_build_graph_decoder_batch(ctx, state, 1, cache_max_ctx - 1, output_mode);
        if (!reserve_graph) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "could not build the reserved ragged decoder graph");
            return false;
        }
        if (!ggml_backend_sched_reserve(ctx->ggml_alloc, reserve_graph)) {
            cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not reserve the ragged decoder graph");
            return false;
        }
        reserve_us += ggml_time_us() - reserve_start_us;
        dynamic_reserved = true;
        return true;
    };
    if (persistent_policy != cohere_decoder_persistent_policy::disabled) {
        const int64_t persistent_init_start_us = ggml_time_us();
        const int initial_context = cohere_decoder_batch_persistent_context_for_offset(
            persistent_policy, (int)prompt.size(), cache_max_ctx);
        persistent_graph = cohere_decoder_batch_persistent_get(
            ctx, state, output_mode, initial_context);
        persistent_active = persistent_graph != nullptr;
        persistent_init_us = ggml_time_us() - persistent_init_start_us;
        if (!persistent_active && ctx->params.verbosity >= 1) {
            std::fprintf(stderr,
                         "cohere: persistent batched decoder unavailable; using dynamic scheduler path\n");
        }
    }
    if (!persistent_active && !reserve_dynamic())
        return nullptr;

    const std::vector<size_t> lane_limits((size_t)n_batch, (size_t)generation_limit);
    cohere_ragged::controller controller(eos_id, eos_id, lane_limits);
    if (!controller.valid()) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "ragged generation controller initialization failed");
        return nullptr;
    }

    std::vector<int> chosen((size_t)n_batch, eos_id);
    std::vector<float> probabilities((size_t)n_batch, 1.0f);
    int generation_steps = 0;
    int offset = (int)prompt.size();
    while (!controller.done()) {
        if (cohere_abort_if_requested(ctx))
            return nullptr;
        for (int lane = 0; lane < n_batch; lane++) {
            if (controller.results()[(size_t)lane].reason != cohere_ragged::stop_reason::active) {
                chosen[(size_t)lane] = eos_id;
                probabilities[(size_t)lane] = 1.0f;
                continue;
            }
            if (output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX) {
                if ((int)device_token_ids.size() != n_batch) {
                    cohere_set_error(COHERE_ERROR_INVARIANT, "ragged decoder returned the wrong token row count");
                    return nullptr;
                }
                chosen[(size_t)lane] = device_token_ids[(size_t)lane];
                probabilities[(size_t)lane] = 1.0f;
            } else {
                const float* row = logits.data() + (size_t)lane * (size_t)hp.vocab_size;
                const int next = cohere_decoder_batch::host_argmax_lowest(row, hp.vocab_size);
                if (next < 0) {
                    cohere_set_error(COHERE_ERROR_INVARIANT, "ragged decoder logits could not be reduced");
                    return nullptr;
                }
                double partition = 0.0;
                for (int token = 0; token < hp.vocab_size; token++)
                    partition += std::exp((double)(row[token] - row[next]));
                chosen[(size_t)lane] = next;
                probabilities[(size_t)lane] = (float)(1.0 / partition);
            }
        }

        const cohere_ragged::step_result step = controller.apply_step(chosen, probabilities);
        if (step == cohere_ragged::step_result::invalid_input ||
            step == cohere_ragged::step_result::already_complete) {
            cohere_set_error(COHERE_ERROR_INVARIANT, "ragged generation controller rejected a decoder step");
            std::fprintf(stderr, "cohere: ragged controller rejected generation step %d\n", generation_steps + 1);
            return nullptr;
        }
        generation_steps++;
        if (controller.done())
            break;

        const int64_t start_us = ggml_time_us();
        bool decoded = false;
        if (persistent_active) {
            const int required_context = cohere_decoder_batch_persistent_context_for_offset(
                persistent_policy, offset, cache_max_ctx);
            if (!persistent_graph || persistent_graph->self_context != required_context) {
                const int64_t switch_start_us = ggml_time_us();
                persistent_graph = cohere_decoder_batch_persistent_get(
                    ctx, state, output_mode, required_context);
                persistent_init_us += ggml_time_us() - switch_start_us;
                if (!persistent_graph) {
                    persistent_active = false;
                    if (!reserve_dynamic())
                        return nullptr;
                }
            }
        }
        if (persistent_active) {
            decoded = cohere_decode_step_batch_persistent_ex(
                ctx, state, *persistent_graph, controller.feed_tokens().data(), offset, output_mode,
                output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS ? &logits : nullptr,
                output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX ? &device_token_ids : nullptr);
            persistent_used = persistent_used || decoded;
        } else {
            decoded = output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS
                          ? cohere_decode_step_batch(ctx, state, controller.feed_tokens().data(), 1, offset, logits)
                          : cohere_decode_step_batch_argmax(ctx, state, controller.feed_tokens().data(), 1, offset,
                                                           device_token_ids);
        }
        if (!decoded) {
            if (cohere_abort_if_requested(ctx))
                return nullptr;
            if (cohere_last_error_kind() == COHERE_ERROR_NONE)
                cohere_set_error(COHERE_ERROR_RUNTIME, "ragged decoder step failed");
            std::fprintf(stderr, "cohere: batched generation decode failed at offset %d\n", offset);
            return nullptr;
        }
        if (cohere_abort_if_requested(ctx))
            return nullptr;
        const int64_t step_us = ggml_time_us() - start_us;
        decode_us += step_us;
        if (persistent_active) {
            persistent_decode_us += step_us;
            persistent_decode_calls++;
        }
        decode_calls++;
        offset++;
    }

    auto* result = (cohere_decoder_batch_generate_result*)std::calloc(1, sizeof(cohere_decoder_batch_generate_result));
    if (!result) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate the ragged decoder result");
        return nullptr;
    }
    result->n_batch = n_batch;
    result->generation_limit = generation_limit;
    result->generation_capacity = generation_capacity;
    result->sequences = (cohere_generated_sequence*)std::calloc((size_t)n_batch, sizeof(cohere_generated_sequence));
    if (!result->sequences) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate ragged decoder result rows");
        cohere_decoder_batch_generate_result_free(result);
        return nullptr;
    }

    for (int lane = 0; lane < n_batch; lane++) {
        const auto& source = controller.results()[(size_t)lane];
        auto& destination = result->sequences[lane];
        destination.n_tokens = (int)source.generated.size();
        destination.stop_reason = source.reason == cohere_ragged::stop_reason::eos
                                      ? COHERE_GENERATION_STOP_EOS
                                      : COHERE_GENERATION_STOP_MAX_TOKENS;
        if (destination.n_tokens == 0)
            continue;
        destination.token_ids = (int*)std::malloc((size_t)destination.n_tokens * sizeof(int));
        destination.probabilities = (float*)std::malloc((size_t)destination.n_tokens * sizeof(float));
        if (!destination.token_ids || !destination.probabilities) {
            cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate ragged decoder token rows");
            cohere_decoder_batch_generate_result_free(result);
            return nullptr;
        }
        for (int token = 0; token < destination.n_tokens; token++) {
            destination.token_ids[token] = source.generated[(size_t)token].token_id;
            destination.probabilities[token] = source.generated[(size_t)token].probability;
        }
    }

    result->stats.decode_calls = decode_calls;
    result->stats.generation_steps = generation_steps;
    result->stats.logits_readback_bytes = output_mode == COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS
                                              ? (size_t)decode_calls * (size_t)n_batch *
                                                    (size_t)hp.vocab_size * sizeof(float)
                                              : 0;
    result->stats.self_cache_bytes = ggml_backend_buffer_get_size(state.self_buf);
    result->stats.cross_cache_bytes = ggml_backend_buffer_get_size(state.cross_buf);
    result->stats.cross_kv_us = cross_kv_us;
    result->stats.reserve_us = reserve_us;
    result->stats.decode_us = decode_us;
    result->stats.token_id_readback_bytes = output_mode == COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX
                                                ? (size_t)decode_calls * (size_t)n_batch * sizeof(int32_t)
                                                : 0;
    result->stats.output_mode = output_mode;
    result->stats.persistent_graph = persistent_used || persistent_active;
    result->stats.persistent_graph_count = state.n_persistent_graphs;
    result->stats.persistent_decode_calls = persistent_decode_calls;
    result->stats.persistent_init_us = persistent_init_us;
    result->stats.persistent_decode_us = persistent_decode_us;
    state.clear();
    result->stats.total_us = ggml_time_us() - total_start_us;
    return result;
}

struct cohere_decoder_batch_generate_result* cohere_run_decoder_batch_generate(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const char* lang, int max_new_tokens) {
    return cohere_run_decoder_batch_generate_ex(ctx, encoder_states, valid_T_enc, n_batch, T_enc_max, d_model, lang,
                                                max_new_tokens, COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

// Pre-populate the ct_to_f32_ref cache for every model weight tensor.
// Called once at init so all inferences pay zero conversion cost.
static void cohere_model_warm_cache(const cohere_model& m) {
    (void)m;
}

struct cohere_context_params cohere_context_default_params(void) {
    return {
        .n_threads = 4,
        .use_flash = false,
        .use_gpu = true,
        .no_punctuation = false,
        .diarize = false,
        .collect_alignment = true,
        .verbosity = 1,
    };
}

struct cohere_context* cohere_init_from_file(const char* path_model, struct cohere_context_params params) {
    ggml_time_init();
    auto* ctx = new cohere_context;
    ctx->params = params;

    // Thread count: overridable via COHERE_THREADS env var.
    // NOTE: ggml_backend_cpu_set_n_threads is NOT called by default —
    // profiling showed it regressed perf for our small matrix sizes.
    {
        const char* env = getenv("COHERE_THREADS");
        if (env) {
            int n = atoi(env);
            if (n > 0)
                params.n_threads = n;
        }
    }

    // ---------------------------------------------------------------------------
    // Backend selection: GPU first (Metal/CUDA), CPU fallback.
    // Set COHERE_DEVICE=cpu  to force CPU.
    // Set COHERE_DEVICE=metal (or cuda) to request a specific backend.
    // ---------------------------------------------------------------------------
    if (params.use_gpu) {
        ggml_backend_load_all(); // registers Metal (macOS), CUDA, Vulkan, etc. if compiled in
    }

    {
        const char* dev_env = getenv("COHERE_DEVICE");
        if (dev_env && strlen(dev_env) > 0) {
            ctx->ggml_backend = ggml_backend_init_by_name(dev_env, nullptr);
            if (!ctx->ggml_backend) {
                fprintf(stderr, "cohere: WARNING: COHERE_DEVICE='%s' not available, falling back\n", dev_env);
            }
        }
        if (!ctx->ggml_backend) {
            ctx->ggml_backend = params.use_gpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
        }
        if (!ctx->ggml_backend) {
            fprintf(stderr, "cohere: failed to initialize any ggml backend\n");
            delete ctx;
            return nullptr;
        }
    }

    // Always have a CPU backend available as scheduler fallback for unsupported ops
    ctx->ggml_backend_cpu = ggml_backend_cpu_init();
    bool using_gpu = !ggml_backend_is_cpu(ctx->ggml_backend);
    const int vb = params.verbosity;
    COHERE_VLOG(vb, "cohere: backend: %s%s\n", ggml_backend_name(ctx->ggml_backend), using_gpu ? "" : " (CPU-only)");

    // Apply thread count only when explicitly requested via env var
    if (getenv("COHERE_THREADS")) {
        COHERE_VLOG(vb, "cohere: applying n_threads=%d to CPU backend [COHERE_THREADS override]\n", params.n_threads);
        ggml_backend_cpu_set_n_threads(ctx->ggml_backend_cpu, params.n_threads);
        if (!using_gpu) {
            ggml_backend_cpu_set_n_threads(ctx->ggml_backend, params.n_threads);
        }
    } else {
        COHERE_VLOG(vb, "cohere: n_threads param=%d (use COHERE_THREADS=N to override)\n", params.n_threads);
    }

    if (!cohere_load_model(ctx->model, ctx->vocab, path_model, ctx->ggml_backend)) {
        cohere_free(ctx);
        return nullptr;
    }

    // Fold inference BN into depthwise conv weights — removes 6 graph nodes × 48 layers
    cohere_fold_batchnorm(ctx->model, params.verbosity);

    const auto& hp = ctx->model.hparams;
    ctx->frontend = cohere_frontend::make_constants(hp.sample_rate, hp.n_fft, hp.win_length, hp.n_mels);
    if (ctx->frontend.window.size() != (size_t)hp.win_length ||
        ctx->frontend.mel_filters.size() != (size_t)hp.n_mels * hp.n_freqs()) {
        fprintf(stderr, "cohere: failed to initialize FP32 frontend constants\n");
        cohere_free(ctx);
        return nullptr;
    }

    // Allocate persistent KV cache
    {
        struct ggml_init_params kv_params = {
            .mem_size = ggml_tensor_overhead() * 2 + 1024,
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        ctx->kv_ctx = ggml_init(kv_params);
        // PLAN #60e + #69e: per-half KV dtype. Cohere's per-step
        // write goes through core_attn::kv_cache_write (PLAN #73);
        // the read path adds a ggml_cast(F32) before the permute+cont
        // chain when the cache is quant.
        const auto kv_pair = core_attn::kv_dtype_pair_from_env("cohere");
        ctx->kv_k = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.k, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads,
                                       hp.dec_n_layers);
        ctx->kv_v = ggml_new_tensor_4d(ctx->kv_ctx, kv_pair.v, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads,
                                       hp.dec_n_layers);
        // PLAN #69b: optional KV-on-CPU spill for VRAM-tight users.
        ggml_backend_t kv_backend = core_attn::kv_backend_from_env(ctx->ggml_backend, ctx->ggml_backend_cpu, "cohere");
        ctx->kv_buf = ggml_backend_alloc_buffer(kv_backend, ggml_nbytes(ctx->kv_k) + ggml_nbytes(ctx->kv_v));
        ggml_backend_buffer_t kv_buf = ctx->kv_buf;
        char* base = (char*)ggml_backend_buffer_get_base(kv_buf);
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_k, (void*)(base));
        ggml_backend_tensor_alloc(kv_buf, ctx->kv_v, (void*)(base + ggml_nbytes(ctx->kv_k)));

        COHERE_VLOG(vb, "cohere: kv cache     = %.1f MiB  (dec_head_dim=%d max_ctx=%d n_heads=%d n_layers=%d)\n",
                    ggml_backend_buffer_get_size(kv_buf) / 1048576.0, hp.dec_head_dim, hp.dec_max_ctx, hp.dec_n_heads,
                    hp.dec_n_layers);
    }

    // Initialize scheduler — GPU backend first (highest priority), CPU as fallback
    if (using_gpu) {
        ggml_backend_t backends[] = {ctx->ggml_backend, ctx->ggml_backend_cpu};
        ctx->ggml_alloc = ggml_backend_sched_new(backends, nullptr, 2, 16384, false, false);
        crispasr_imatrix_install(ctx->ggml_alloc); // no-op unless CRISPASR_IMATRIX_OUT is set
    } else {
        ggml_backend_t backends[] = {ctx->ggml_backend};
        ctx->ggml_alloc = ggml_backend_sched_new(backends, nullptr, 1, 16384, false, false);
        crispasr_imatrix_install(ctx->ggml_alloc); // no-op unless CRISPASR_IMATRIX_OUT is set
    }

    ctx->compute_meta.resize(ggml_tensor_overhead() * 16384 + 1024);

    // Create constants context
    ctx->ctx_const = ggml_init({.mem_size = 1024, .mem_buffer = nullptr, .no_alloc = false});
    ctx->eps = ggml_new_tensor_1d(ctx->ctx_const, GGML_TYPE_F32, 1);
    float eps_val = 1e-5f;
    memcpy(ctx->eps->data, &eps_val, sizeof(float));

    ctx->cross_kv_ctx = nullptr;
    ctx->cross_kv_buf = nullptr;

    // Log static memory layout
    if (ctx->model.buf) {
        COHERE_VLOG(vb, "cohere: model weights = %.1f MiB\n", ggml_backend_buffer_get_size(ctx->model.buf) / 1048576.0);
    }
    COHERE_VLOG(vb, "cohere: compute_meta  = %.1f KiB\n", ctx->compute_meta.size() / 1024.0);

    return ctx;
}

void cohere_free(struct cohere_context* ctx) {
    if (!ctx)
        return;
    // Free scheduler first (releases gallocr-owned buffer views), then backend buffers, then backends.
    if (ctx->ggml_alloc)
        ggml_backend_sched_free(ctx->ggml_alloc);
    if (ctx->cross_kv_buf)
        ggml_backend_buffer_free(ctx->cross_kv_buf);
    if (ctx->kv_buf)
        ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->model.buf)
        ggml_backend_buffer_free(ctx->model.buf);
    if (ctx->ggml_backend)
        ggml_backend_free(ctx->ggml_backend);
    if (ctx->ggml_backend_cpu && ctx->ggml_backend_cpu != ctx->ggml_backend)
        ggml_backend_free(ctx->ggml_backend_cpu);
    if (ctx->kv_ctx)
        ggml_free(ctx->kv_ctx);
    if (ctx->ctx_const)
        ggml_free(ctx->ctx_const);
    if (ctx->cross_kv_ctx)
        ggml_free(ctx->cross_kv_ctx);
    if (ctx->model.ctx)
        ggml_free(ctx->model.ctx);
    if (ctx->compute_ctx)
        ggml_free(ctx->compute_ctx);
    delete ctx;
}

void cohere_backend_memory(struct cohere_context* ctx, size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes)
        *free_bytes = 0;
    if (total_bytes)
        *total_bytes = 0;
    if (!ctx || !ctx->ggml_backend)
        return;
    ggml_backend_dev_t device = ggml_backend_get_device(ctx->ggml_backend);
    if (!device)
        return;
    size_t available = 0;
    size_t total = 0;
    ggml_backend_dev_memory(device, &available, &total);
    if (free_bytes)
        *free_bytes = available;
    if (total_bytes)
        *total_bytes = total;
}

const char* cohere_backend_name(struct cohere_context* ctx) {
    if (!ctx || !ctx->ggml_backend)
        return "";
    return ggml_backend_name(ctx->ggml_backend);
}

int cohere_n_vocab(struct cohere_context* ctx) {
    return ctx->vocab.n_vocab();
}

const char* cohere_token_to_str(struct cohere_context* ctx, int id) {
    if (id < 0 || id >= (int)ctx->vocab.id_to_token.size())
        return "<unk>";
    return ctx->vocab.id_to_token[id].c_str();
}

int cohere_str_to_token(struct cohere_context* ctx, const char* s) {
    return ctx->vocab.token_id(s);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

static std::vector<float> ct_get_f32(const ggml_tensor* t) {
    return core_cpu::to_f32(t); // F32/F16/quantized-safe
}

// ---------------------------------------------------------------------------
// BatchNorm folding — called once after model load.
//
// Inference-time BN is: y = (x - mean) / sqrt(var + eps) * scale + bn_b
// Equivalently:         y = x * s + (bn_b - mean * s)   where s = scale/sqrt(var+eps)
//
// Since mean/var are fixed constants after training, we fold s into the
// depthwise conv kernel weights and absorb the full bias shift into conv_dw_b:
//
//   conv_dw_w[:, c] *= s[c]
//   conv_dw_b[c]     = (conv_dw_b[c] - mean[c]) * s[c] + bn_b[c]
//
// After this the encoder graph drops the 6-node BN block (288 nodes total for
// 48 layers) and replaces it with nothing — the folded conv already applies it.
// ---------------------------------------------------------------------------
static void cohere_fold_batchnorm(cohere_model& model, int verbosity) {
    const int d = model.hparams.enc_d_model; // 1280
    const int k = model.hparams.enc_conv_k;  // 9
    const float eps = 1e-5f;

    for (int il = 0; il < model.hparams.enc_n_layers; il++) {
        auto& layer = model.enc_layers[il];

        // Read all F32 BN parameters (all are F32 in the GGUF)
        std::vector<float> bn_mean(d), bn_var(d), bn_scale(d), bn_bias(d);
        ggml_backend_tensor_get(layer.conv_bn_mean, bn_mean.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_var, bn_var.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_w, bn_scale.data(), 0, d * sizeof(float));
        ggml_backend_tensor_get(layer.conv_bn_b, bn_bias.data(), 0, d * sizeof(float));

        // Per-channel fold factor: s[c] = bn_scale[c] / sqrt(bn_var[c] + eps)
        std::vector<float> s(d);
        for (int c = 0; c < d; c++)
            s[c] = bn_scale[c] / sqrtf(bn_var[c] + eps);

        // Fold s into conv_dw_w (ggml shape ne[0]=k, ne[1]=1, ne[2]=d)
        // Linear index for kernel pos ki, channel c: ki + c*k
        {
            std::vector<float> w_f32 = ct_get_f32(layer.conv_dw_w); // k*d elements
            for (int c = 0; c < d; c++)
                for (int ki = 0; ki < k; ki++)
                    w_f32[ki + c * k] *= s[c];
            // Write back in the tensor's native type
            if (layer.conv_dw_w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(layer.conv_dw_w, w_f32.data(), 0, k * d * sizeof(float));
            } else if (layer.conv_dw_w->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> w_f16(k * d);
                ggml_fp32_to_fp16_row(w_f32.data(), w_f16.data(), k * d);
                ggml_backend_tensor_set(layer.conv_dw_w, w_f16.data(), 0, k * d * sizeof(ggml_fp16_t));
            } else if (layer.conv_dw_w->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> w_bf16(k * d);
                ggml_fp32_to_bf16_row(w_f32.data(), w_bf16.data(), k * d);
                ggml_backend_tensor_set(layer.conv_dw_w, w_bf16.data(), 0, k * d * sizeof(ggml_bf16_t));
            } else {
                COHERE_VLOG(
                    verbosity,
                    "cohere: cannot fold BN into unsupported depthwise weight type %s\n",
                    ggml_type_name(layer.conv_dw_w->type));
                return;
            }
        }

        // Fold into conv_dw_b (F32 tensor, shape [d]):
        // b_folded[c] = (dw_b[c] - mean[c]) * s[c] + bn_bias[c]
        {
            std::vector<float> dw_b(d);
            ggml_backend_tensor_get(layer.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
            for (int c = 0; c < d; c++)
                dw_b[c] = (dw_b[c] - bn_mean[c]) * s[c] + bn_bias[c];
            ggml_backend_tensor_set(layer.conv_dw_b, dw_b.data(), 0, d * sizeof(float));
        }
    }
    COHERE_VLOG(verbosity, "cohere: BN folded into conv_dw weights for %d layers\n", model.hparams.enc_n_layers);
}

struct cohere_result* cohere_transcribe_ex(struct cohere_context* ctx, const float* samples, int n_samples,
                                           const char* lang, int64_t t_offset_cs) {
    cohere_clear_error();
    if (!ctx || !samples || n_samples <= 0) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "invalid single-row transcription arguments");
        return nullptr;
    }
    if (cohere_abort_if_requested(ctx))
        return nullptr;
    const auto& hp = ctx->model.hparams;
    const auto& voc = ctx->vocab;
    const int n_heads = hp.dec_n_heads;
    const int head_dim = hp.dec_head_dim;
    auto& perf = ctx->perf;
    int64_t t0, t1;

    cohere_perf_reset(perf);
    int64_t t_total_start = ggml_time_us();
    const int vb = ctx->params.verbosity;

    COHERE_VLOG2(vb, "cohere: transcribe started   n_samples=%d  audio=%.2fs\n", n_samples,
                 (double)n_samples / hp.sample_rate);

    // --- Long-audio chunking: encode AND decode each <=35s model row independently ---
    //
    // The previous approach assembled a single giant cross-KV for all chunks and ran
    // the decoder once. This is out-of-distribution for a bounded short-form model
    // and causes the decoder to predict EOS far too early (e.g. ~70 words for a 2-min clip).
    //
    // Correct approach: each Cohere processor row is fully encoded and decoded on its own.
    // Results are concatenated with per-chunk t_offset_cs so timestamps remain absolute.
    //
    // The pinned processor accepts rows up to 35 s. For longer input it chooses
    // an RMS minimum in [30 s, 35 s), yielding contiguous, non-overlapping rows.
    {
        const int CHUNK_S = cohere_chunking::max_clip_seconds * hp.sample_rate;
        if (n_samples > CHUNK_S) {
            auto chunks = cohere_chunking::plan(samples, (size_t)n_samples, hp.sample_rate);
            if (chunks.empty())
                return nullptr;
            // Merge helper: append src into dst
            auto merge_results = [](cohere_result* dst, cohere_result* src) -> bool {
                if (!src)
                    return true;
                // Append text
                if (src->text && src->text[0]) {
                    std::string combined = dst->text ? std::string(dst->text) : "";
                    cohere_chunking::append_text(combined, src->text);
                    char* new_text = (char*)malloc(combined.size() + 1);
                    if (!new_text) {
                        cohere_result_free(src);
                        return false;
                    }
                    memcpy(new_text, combined.c_str(), combined.size() + 1);
                    free(dst->text);
                    dst->text = new_text;
                }
                // Append tokens
                if (src->n_tokens > 0) {
                    int new_n = (dst->n_tokens) + src->n_tokens;
                    cohere_token_data* new_toks = (cohere_token_data*)malloc(new_n * sizeof(cohere_token_data));
                    if (!new_toks) {
                        cohere_result_free(src);
                        return false;
                    }
                    if (dst->n_tokens > 0)
                        memcpy(new_toks, dst->tokens, dst->n_tokens * sizeof(cohere_token_data));
                    memcpy(new_toks + dst->n_tokens, src->tokens, src->n_tokens * sizeof(cohere_token_data));
                    free(dst->tokens);
                    dst->tokens = new_toks;
                    dst->n_tokens = new_n;
                }
                dst->generated_tokens += src->generated_tokens;
                dst->generation_limit += src->generation_limit;
                dst->generation_capacity += src->generation_capacity;
                dst->stopped_by_max_tokens = dst->stopped_by_max_tokens || src->stopped_by_max_tokens;
                dst->repetition_stopped = dst->repetition_stopped || src->repetition_stopped;
                cohere_result_free(src);
                return true;
            };

            cohere_result* full = (cohere_result*)calloc(1, sizeof(cohere_result));
            if (!full)
                return nullptr;
            full->text = (char*)calloc(1, 1); // empty string
            if (!full->text) {
                cohere_result_free(full);
                return nullptr;
            }

            for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
                if (cohere_abort_if_requested(ctx)) {
                    cohere_result_free(full);
                    return nullptr;
                }
                const int offset = (int)chunks[chunk_idx].first;
                const int chunk_end = (int)chunks[chunk_idx].second;
                int64_t chunk_t0_cs = t_offset_cs + (int64_t)((double)offset / hp.sample_rate * 100.0);
                COHERE_VLOG(vb, "cohere: chunk %zu/%zu  samples [%d, %d)  t0=%.1fs  dur=%.2fs\n", chunk_idx + 1,
                            chunks.size(), offset, chunk_end, chunk_t0_cs / 100.0,
                            (double)(chunk_end - offset) / hp.sample_rate);
                cohere_result* chunk_r =
                    cohere_transcribe_ex(ctx, samples + offset, chunk_end - offset, lang, chunk_t0_cs);
                if (!chunk_r && cohere_abort_if_requested(ctx)) {
                    cohere_result_free(full);
                    return nullptr;
                }
                if (!merge_results(full, chunk_r)) {
                    cohere_result_free(full);
                    return nullptr;
                }
            }
            return full;
        }
    }

    // --- Feature extraction (single processor row ≤ 35s) ---
    cohere_bench_stage _b_total("total");
    // --- Feature extraction + Encoder ---
    // The recursive reference planner above guarantees this path sees one
    // processor row, capped at 35 seconds to bound O(T²) attention cost.
    // Cross-KV (K/V projections of encoder output) is extracted from each chunk's
    // encoder graph and scatter-copied into the final K[head_dim,T_total,n_heads] /
    // V[T_total,head_dim,n_heads] layout on the CPU, then uploaded to the backend.
    const int CHUNK_SAMPLES = cohere_chunking::max_clip_seconds * hp.sample_rate;
    const bool do_chunked = (n_samples > CHUNK_SAMPLES);

    int T_enc_total = 0;

    // Per-chunk K/V CPU storage: partial_k[il][chunk] and partial_v[il][chunk]
    // K chunk: [head_dim, T_c, n_heads] layout (F32; converted to F16 on upload)
    // V chunk: [T_c, head_dim, n_heads] layout (F32; converted to F16 on upload)
    std::vector<std::vector<std::vector<float>>> partial_k(hp.dec_n_layers);
    std::vector<std::vector<std::vector<float>>> partial_v(hp.dec_n_layers);
    std::vector<int> T_enc_chunks;
    for (int il = 0; il < hp.dec_n_layers; il++) {
        partial_k[il].reserve(4);
        partial_v[il].reserve(4);
    }

    // Optional per-op profiling (COHERE_PROF=1, single-chunk only)
    cohere_prof_state prof_state;
    bool do_prof = !do_chunked && (getenv("COHERE_PROF") != nullptr);

    {
        cohere_bench_stage _b_enc("encoder (all chunks)");
        int n_chunks = 0;
        for (int sample_offset = 0; sample_offset < n_samples; sample_offset += CHUNK_SAMPLES) {
            if (cohere_abort_if_requested(ctx))
                return nullptr;
            int chunk_n = std::min(CHUNK_SAMPLES, n_samples - sample_offset);
            n_chunks++;

            // Feature extraction for this chunk
            int T_mel_c = 0;
            t0 = ggml_time_us();
            auto mel_c = cohere_compute_features(hp, ctx->frontend, samples + sample_offset, chunk_n,
                                                 ctx->params.n_threads, T_mel_c);
            t1 = ggml_time_us();
            perf.t_features_us += (t1 - t0);
            if (cohere_abort_if_requested(ctx))
                return nullptr;

            if (do_chunked) {
                COHERE_VLOG2(vb, "cohere: chunk %d/%d  n=%d  T_mel=%d\n", n_chunks,
                             (n_samples + CHUNK_SAMPLES - 1) / CHUNK_SAMPLES, chunk_n, T_mel_c);
            } else {
                COHERE_VLOG2(vb, "cohere: features done         T_mel=%d  %.1f ms\n", T_mel_c,
                             perf.t_features_us / 1e3);
            }

            // Build encoder graph for this chunk
            t0 = ggml_time_us();
            struct ggml_cgraph* gf_enc = cohere_build_graph_encoder(ctx, T_mel_c, true);
            t1 = ggml_time_us();
            perf.t_enc_build_us += (t1 - t0);
            perf.enc_n_nodes = ggml_graph_n_nodes(gf_enc);

            if (!do_chunked) {
                COHERE_VLOG2(vb, "cohere: enc graph built       nodes=%d  %.1f ms\n", perf.enc_n_nodes,
                             perf.t_enc_build_us / 1e3);
            }

            // Allocate
            t0 = ggml_time_us();
            ggml_backend_sched_reset(ctx->ggml_alloc);
            if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf_enc)) {
                fprintf(stderr, "cohere: failed to allocate encoder graph (chunk %d)\n", n_chunks);
                return nullptr;
            }
            t1 = ggml_time_us();
            perf.t_enc_alloc_us += (t1 - t0);
            perf.mem_sched_buf = ggml_backend_sched_get_buffer_size(ctx->ggml_alloc, ctx->ggml_backend);
            if (!do_chunked) {
                COHERE_VLOG2(vb, "cohere: enc sched alloc       %.1f ms   sched_buf=%.1f MiB\n",
                             perf.t_enc_alloc_us / 1e3, perf.mem_sched_buf / 1048576.0);
            }

            // Set inputs
            struct ggml_tensor* mel_t = ggml_graph_get_tensor(gf_enc, "mel");
            if (!mel_t) {
                fprintf(stderr, "error: mel tensor not found\n");
                return nullptr;
            }
            ggml_backend_tensor_set(mel_t, mel_c.data(), 0, mel_c.size() * sizeof(float));

            int H1c = (T_mel_c + 2 - 3) / 2 + 1;
            int H2c = (H1c + 2 - 3) / 2 + 1;
            int H3c = (H2c + 2 - 3) / 2 + 1;
            auto pos_enc_c = ct_rel_pos_enc(H3c, hp.enc_d_model);
            struct ggml_tensor* pos_enc_t = ggml_graph_get_tensor(gf_enc, "pos_enc");
            if (!pos_enc_t) {
                fprintf(stderr, "error: pos_enc tensor not found\n");
                return nullptr;
            }
            ggml_backend_tensor_set(pos_enc_t, pos_enc_c.data(), 0, pos_enc_c.size() * sizeof(float));

            if (do_prof) {
                ggml_backend_sched_set_eval_callback(ctx->ggml_alloc, cohere_prof_eval_cb, &prof_state);
            }

            // Compute
            t0 = ggml_time_us();
            if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf_enc, ctx->params.n_threads)) {
                if (cohere_abort_if_requested(ctx))
                    return nullptr;
                fprintf(stderr, "cohere: failed to compute encoder graph (chunk %d)\n", n_chunks);
                return nullptr;
            }
            if (cohere_abort_if_requested(ctx))
                return nullptr;
            t1 = ggml_time_us();
            perf.t_enc_compute_us += (t1 - t0);

            if (do_chunked) {
                COHERE_VLOG2(vb, "cohere: chunk %d enc done     %.1f ms\n", n_chunks, (t1 - t0) / 1e3);
            } else {
                COHERE_VLOG2(vb, "cohere: enc compute done      %.1f ms\n", perf.t_enc_compute_us / 1e3);
            }

            if (do_prof) {
                ggml_backend_sched_set_eval_callback(ctx->ggml_alloc, nullptr, nullptr);
            }

            // Per-stage encoder snapshots for crispasr-diff / transcribe.cpp
            // comparison (CRISPASR_COHERE_DUMP_STAGES=<dir>). Writes raw
            // [ne0,ne1] f32 (2 int32 dims + data): crisp.mel.bin, crisp.enc_final.bin,
            // crisp.block<N>.bin — matching transcribe.cpp's TRANSCRIBE_DUMP_DIR.
            if (const char* sd = std::getenv("CRISPASR_COHERE_DUMP_STAGES")) {
                auto dump_named = [&](const char* tname, const char* fname) {
                    struct ggml_tensor* t = ggml_graph_get_tensor(gf_enc, tname);
                    if (!t)
                        return;
                    std::vector<float> v((size_t)t->ne[0] * t->ne[1]);
                    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
                    char path[1024];
                    std::snprintf(path, sizeof(path), "%s/%s", sd, fname);
                    if (FILE* f = std::fopen(path, "wb")) {
                        int32_t d0 = (int32_t)t->ne[0], d1 = (int32_t)t->ne[1];
                        std::fwrite(&d0, 4, 1, f);
                        std::fwrite(&d1, 4, 1, f);
                        std::fwrite(v.data(), sizeof(float), v.size(), f);
                        std::fclose(f);
                    }
                };
                char mpath[1024];
                std::snprintf(mpath, sizeof(mpath), "%s/crisp.mel.bin", sd);
                if (FILE* f = std::fopen(mpath, "wb")) {
                    int32_t d0 = (int32_t)hp.n_mels, d1 = (int32_t)T_mel_c;
                    std::fwrite(&d0, 4, 1, f);
                    std::fwrite(&d1, 4, 1, f);
                    std::fwrite(mel_c.data(), sizeof(float), mel_c.size(), f);
                    std::fclose(f);
                }
                dump_named("enc_final_preproj", "crisp.enc_final.bin");
                for (int bi = 0; bi < hp.enc_n_layers; bi++) {
                    char tn[32], fn[48];
                    std::snprintf(tn, sizeof(tn), "enc_block_%d", bi);
                    std::snprintf(fn, sizeof(fn), "crisp.block%d.bin", bi);
                    dump_named(tn, fn);
                }
            }

            // Extract T_enc for this chunk
            struct ggml_tensor* enc_out_t = ggml_graph_get_tensor(gf_enc, "enc_out");
            if (!enc_out_t) {
                fprintf(stderr, "error: enc_out tensor not found\n");
                return nullptr;
            }
            int T_enc_c = enc_out_t->ne[1];
            T_enc_total += T_enc_c;
            T_enc_chunks.push_back(T_enc_c);

            // Extract cross-KV from this chunk's encoder graph into CPU vectors.
            // K shape: [head_dim, T_enc_c, n_heads] (raw F32 from encoder graph)
            // V shape: [T_enc_c, head_dim, n_heads] (raw F32 from encoder graph)
            const int64_t t_ckr0 = ggml_time_us();
            for (int il = 0; il < hp.dec_n_layers; il++) {
                if (cohere_abort_if_requested(ctx))
                    return nullptr;
                char ck_name[32], cv_name[32];
                snprintf(ck_name, sizeof(ck_name), "ck_%d", il);
                snprintf(cv_name, sizeof(cv_name), "cv_%d", il);
                struct ggml_tensor* ck_src = ggml_graph_get_tensor(gf_enc, ck_name);
                struct ggml_tensor* cv_src = ggml_graph_get_tensor(gf_enc, cv_name);
                if (!ck_src || !cv_src) {
                    fprintf(stderr, "error: cross-KV tensor %s or %s not found\n", ck_name, cv_name);
                    return nullptr;
                }
                partial_k[il].emplace_back(ggml_nelements(ck_src));
                partial_v[il].emplace_back(ggml_nelements(cv_src));
                ggml_backend_tensor_get(ck_src, partial_k[il].back().data(), 0, ggml_nbytes(ck_src));
                ggml_backend_tensor_get(cv_src, partial_v[il].back().data(), 0, ggml_nbytes(cv_src));
            }
            perf.t_crosskv_read_us += (ggml_time_us() - t_ckr0);
        } // end chunk loop

        if (do_chunked) {
            COHERE_VLOG2(vb, "cohere: enc compute done      %.1f ms  (chunked %d×35s)\n", perf.t_enc_compute_us / 1e3,
                         n_chunks);
        }

        if (do_prof) {
            cohere_prof_print(prof_state);
        }
    }

    // Assemble cross-KV from per-chunk CPU data and upload to backend buffer.
    {
        cohere_bench_stage _b_ckv("cross-kv assembly");
        t0 = ggml_time_us();

        if (ctx->cross_kv_ctx)
            ggml_free(ctx->cross_kv_ctx);
        if (ctx->cross_kv_buf)
            ggml_backend_buffer_free(ctx->cross_kv_buf);

        ctx->cross_kv_k.resize(hp.dec_n_layers);
        ctx->cross_kv_v.resize(hp.dec_n_layers);

        struct ggml_init_params ckv_params = {
            .mem_size = (hp.dec_n_layers * 2 + 10) * ggml_tensor_overhead(),
            .mem_buffer = nullptr,
            .no_alloc = true,
        };
        ctx->cross_kv_ctx = ggml_init(ckv_params);

        for (int il = 0; il < hp.dec_n_layers; il++) {
            ctx->cross_kv_k[il] = ggml_new_tensor_3d(ctx->cross_kv_ctx, GGML_TYPE_F16, head_dim, T_enc_total, n_heads);
            // V same layout as K: [head_dim, T_enc, n_heads] for ggml_flash_attn_ext
            ctx->cross_kv_v[il] = ggml_new_tensor_3d(ctx->cross_kv_ctx, GGML_TYPE_F16, head_dim, T_enc_total, n_heads);
        }

        const size_t k_size = ggml_nbytes(ctx->cross_kv_k[0]); // head_dim × T_enc_total × n_heads × 2 (F16)
        const size_t v_size = ggml_nbytes(ctx->cross_kv_v[0]); // T_enc_total × head_dim × n_heads × 2 (F16)
        ctx->cross_kv_buf = ggml_backend_alloc_buffer(ctx->ggml_backend, (k_size + v_size) * hp.dec_n_layers);
        char* base = (char*)ggml_backend_buffer_get_base(ctx->cross_kv_buf);

        for (int il = 0; il < hp.dec_n_layers; il++) {
            ggml_backend_tensor_alloc(ctx->cross_kv_buf, ctx->cross_kv_k[il], (void*)(base + il * (k_size + v_size)));
            ggml_backend_tensor_alloc(ctx->cross_kv_buf, ctx->cross_kv_v[il],
                                      (void*)(base + il * (k_size + v_size) + k_size));
        }

        const int n_chunks = (int)T_enc_chunks.size();
        const int64_t kv_nelems = (int64_t)head_dim * T_enc_total * n_heads;
        // Temporary F16 staging buffer (reused for K and V across all layers)
        std::vector<ggml_fp16_t> fp16_staging(kv_nelems);

        if (n_chunks == 1) {
            // Fast path: single chunk — convert F32→F16 then upload
            for (int il = 0; il < hp.dec_n_layers; il++) {
                if (cohere_abort_if_requested(ctx))
                    return nullptr;
                ggml_fp32_to_fp16_row(partial_k[il][0].data(), fp16_staging.data(), partial_k[il][0].size());
                ggml_backend_tensor_set(ctx->cross_kv_k[il], fp16_staging.data(), 0,
                                        partial_k[il][0].size() * sizeof(ggml_fp16_t));
                ggml_fp32_to_fp16_row(partial_v[il][0].data(), fp16_staging.data(), partial_v[il][0].size());
                ggml_backend_tensor_set(ctx->cross_kv_v[il], fp16_staging.data(), 0,
                                        partial_v[il][0].size() * sizeof(ggml_fp16_t));
            }
        } else {
            // Multi-chunk: scatter-copy into F32, then convert F32→F16 before upload.
            //
            // K layout [head_dim, T_enc_total, n_heads]: for head h, all T frames are at
            //   offset h × T_enc_total × head_dim (a contiguous block of T×head_dim floats).
            //   Each chunk's head-h block: chunk_k[h × T_c × head_dim .. (h+1) × T_c × head_dim)
            //
            // V layout [head_dim, T_enc_total, n_heads]: identical to K layout.
            //   Each chunk's head-h block: chunk_v[h × T_c × head_dim .. (h+1) × T_c × head_dim)
            std::vector<float> k_full(head_dim * T_enc_total * n_heads);
            std::vector<float> v_full(head_dim * T_enc_total * n_heads);

            for (int il = 0; il < hp.dec_n_layers; il++) {
                if (cohere_abort_if_requested(ctx))
                    return nullptr;
                int T_so_far = 0;
                for (int c = 0; c < n_chunks; c++) {
                    int T_c = T_enc_chunks[c];
                    const float* ck = partial_k[il][c].data();
                    const float* cv = partial_v[il][c].data();
                    for (int h = 0; h < n_heads; h++) {
                        // K and V: head h block in chunk is contiguous (T_c × head_dim floats)
                        const float* ks = ck + h * T_c * head_dim;
                        float* kd = k_full.data() + h * T_enc_total * head_dim + T_so_far * head_dim;
                        memcpy(kd, ks, T_c * head_dim * sizeof(float));
                        const float* vs = cv + h * T_c * head_dim;
                        float* vd = v_full.data() + h * T_enc_total * head_dim + T_so_far * head_dim;
                        memcpy(vd, vs, T_c * head_dim * sizeof(float));
                    }
                    T_so_far += T_c;
                }
                ggml_fp32_to_fp16_row(k_full.data(), fp16_staging.data(), k_full.size());
                ggml_backend_tensor_set(ctx->cross_kv_k[il], fp16_staging.data(), 0,
                                        k_full.size() * sizeof(ggml_fp16_t));
                ggml_fp32_to_fp16_row(v_full.data(), fp16_staging.data(), v_full.size());
                ggml_backend_tensor_set(ctx->cross_kv_v[il], fp16_staging.data(), 0,
                                        v_full.size() * sizeof(ggml_fp16_t));
            }
        }

        t1 = ggml_time_us();
        perf.t_cross_kv_us = t1 - t0;
        perf.mem_cross_kv_buf = ggml_backend_buffer_get_size(ctx->cross_kv_buf);
    }

    const int T_enc = T_enc_total;

    // --- Decoder prompt ---
    cohere_bench_stage _b_dec("decoder (total)");
    auto tid = [&](const std::string& s) { return voc.token_id(s); };
    const char* lang_tok = lang ? lang : "en";
    char lang_tok_str[32];
    snprintf(lang_tok_str, sizeof(lang_tok_str), "<|%s|>", lang_tok);

    const char* pnc_tok = ctx->params.no_punctuation ? "<|nopnc|>" : "<|pnc|>";
    std::vector<int> prompt = {
        // decoder_start_token_id = 13764 ("▁"). The reference processor prepends
        // it to decoder_input_ids; omitting it shifts every decoder position by
        // one and gives every prompt token the wrong positional embedding.
        tid("▁"),
        tid("<|startofcontext|>"),
        tid("<|startoftranscript|>"),
        tid("<|emo:undefined|>"),
        tid(lang_tok_str),
        tid(lang_tok_str),
        tid(pnc_tok),
        tid("<|noitn|>"),
        tid("<|notimestamp|>"),
        tid(ctx->params.diarize ? "<|diarize|>" : "<|nodiarize|>"),
    };
    prompt.erase(std::remove_if(prompt.begin(), prompt.end(), [](int t) { return t == -1; }), prompt.end());
    COHERE_VLOG2(vb, "cohere: prompt               n=%d\n", (int)prompt.size());
    cohere_debug("cohere: prompt IDs: ");
    for (int id : prompt)
        cohere_debug("%d ", id);
    cohere_debug("\n");

    ggml_backend_buffer_clear(ctx->kv_k->buffer, 0);
    ggml_backend_buffer_clear(ctx->kv_v->buffer, 0);

    // Reset per-utterance alignment state. params.collect_alignment controls
    // whether decode graphs produce and copy explicit cross-attention weights.
    ctx->step_attn.clear();
    ctx->attn_T_enc = T_enc;
    ctx->attn_n_heads = hp.dec_n_heads;

    const int eos_id = tid("<|endoftext|>");
    const int generation_capacity = hp.dec_max_ctx - (int)prompt.size() - 4;
    int max_gen = generation_capacity;
    if (ctx->max_new_tokens > 0)
        max_gen = std::min(max_gen, ctx->max_new_tokens);
    if (cohere_abort_if_requested(ctx))
        return nullptr;

    // Prompt pass
    auto logits = cohere_decode_step(ctx, T_enc, prompt.data(), (int)prompt.size(), 0);
    if (logits.empty()) {
        if (cohere_abort_if_requested(ctx))
            return nullptr;
        cohere_set_error(COHERE_ERROR_RUNTIME, "decoder prompt graph failed");
        return nullptr;
    }
    perf.n_dec_steps++;
    int offset = (int)prompt.size();
    COHERE_VLOG2(vb, "cohere: prompt pass done      nodes=%d  build=%.1f alloc=%.1f compute=%.1f ms\n",
                 perf.dec_n_nodes_prompt, perf.t_dec_build_us / 1e3, perf.t_dec_alloc_us / 1e3,
                 perf.t_dec_compute_us / 1e3);

    // Pre-reserve the scheduler with the maximum-size single-step decoder graph.
    // The prompt pass leaves the gallocr sized for the prompt graph (different structure).
    // By reserving with n_tokens=1, offset=max_ctx-1 (largest possible K/V views),
    // the gallocr's size_max covers all future autoregressive steps, so
    // ggml_gallocr_needs_realloc returns false for every step and the plan is reused.
    {
        const int64_t t_rsv0 = ggml_time_us();
        const int dummy_tok = 0;
        struct ggml_cgraph* gf_max = cohere_build_graph_decoder(ctx, &dummy_tok, 1, hp.dec_max_ctx - 1);
        ggml_backend_sched_reserve(ctx->ggml_alloc, gf_max);
        perf.t_reserve_us += (ggml_time_us() - t_rsv0);
    }

    std::vector<int> generated;
    std::vector<float> gen_probs;
    bool generation_saw_eos = false;
    bool repetition_stopped = false;

    // §90 beam search — run_with_probs_branched when beam_size > 1.
    // Cross-attention KV (cross_kv_k/v) is shared across beams; only self-attention
    // KV (kv_k/kv_v) is snapshotted per beam.
    if (ctx->beam_size > 1) {
        // GH #161: snapshot/restore self-attention KV on-device via a recycled
        // buffer pool (no PCIe round-trip + sync per beam per step). The pool
        // outlives every snapshot produced by the beam search below.
        core_attn::kv_snapshot_pool kv_pool(ctx->kv_k, ctx->kv_v);

        auto save_fn = [&kv_pool](cohere_context*) -> core_attn::kv_snapshot* { return kv_pool.save(); };

        auto restore_fn = [&kv_pool](cohere_context*, core_attn::kv_snapshot* s) { kv_pool.restore(s); };

        auto snap_free_fn = [&kv_pool](core_attn::kv_snapshot* s) { kv_pool.release(s); };

        // Capture T_enc so step_fn can pass it to cohere_decode_step.
        const int beam_T_enc = T_enc;
        auto step_fn = [beam_T_enc](cohere_context* c, int32_t tok, int n_past) -> float* {
            if (cohere_abort_if_requested(c))
                return nullptr;
            auto lg = cohere_decode_step(c, beam_T_enc, &tok, 1, n_past);
            if (lg.empty())
                return nullptr;
            float* out = (float*)std::malloc(lg.size() * sizeof(float));
            std::memcpy(out, lg.data(), lg.size() * sizeof(float));
            return out;
        };

        const int vocab = hp.vocab_size;
        // Use only the last-position logits from the prefill
        const float* last_logits = logits.data() + ((int)prompt.size() - 1) * vocab;

        core_beam_decode::Config bcfg;
        bcfg.max_new_tokens = max_gen;
        bcfg.eos_id = eos_id;
        bcfg.vocab_size = vocab;
        bcfg.beam_size = ctx->beam_size;
        bcfg.prompt_len = (int)prompt.size();

        auto br = core_beam_decode::run_with_probs_branched(ctx, last_logits, save_fn, restore_fn, snap_free_fn,
                                                            step_fn, bcfg);

        // Strip EOS if present at the end
        for (int i = 0; i < (int)br.tokens.size(); i++) {
            if (br.tokens[i] == eos_id) {
                generation_saw_eos = true;
                break;
            }
            generated.push_back(br.tokens[i]);
            gen_probs.push_back(br.probs[i]);
        }
    } else {
        // Greedy / temperature-sampled decode
        const bool sampling = ctx->decode_temperature > 0.0f;
        std::mt19937_64 rng(ctx->decode_seed != 0 ? ctx->decode_seed : (uint64_t)std::random_device{}());
        std::vector<int> token_counts(ctx->frequency_penalty > 0.0f ? (size_t)hp.vocab_size : 0);
        std::vector<float> adjusted_logits;

        for (int step = 0; step < max_gen; step++) {
            if (cohere_abort_if_requested(ctx))
                return nullptr;
            const int vocab = hp.vocab_size;
            const float* last_logits = (step == 0) ? logits.data() + ((int)prompt.size() - 1) * vocab : logits.data();
            const float* pick_logits = last_logits;
            if (ctx->frequency_penalty > 0.0f && !token_counts.empty()) {
                adjusted_logits.assign(last_logits, last_logits + vocab);
                for (int v = 0; v < vocab; v++) {
                    if (token_counts[(size_t)v] > 0)
                        adjusted_logits[(size_t)v] -= ctx->frequency_penalty * (float)token_counts[(size_t)v];
                }
                pick_logits = adjusted_logits.data();
            }

            int next_tok = (int)(std::max_element(pick_logits, pick_logits + vocab) - pick_logits);

            // Numerically-stable softmax probability of (initially) the
            // argmax token. We compute the partition function once per
            // step regardless of which path we take, because we need
            // the per-token probability for the gen_probs vector either
            // way.
            float max_l = pick_logits[next_tok];
            double sum_e = 0.0;
            for (int v = 0; v < vocab; v++)
                sum_e += std::exp((double)(pick_logits[v] - max_l));
            float tok_p = (float)(1.0 / sum_e);

            if (sampling) {
                // Re-do the partition with logits/T instead of logits.
                const float inv_t = 1.0f / ctx->decode_temperature;
                float max_lt = pick_logits[0] * inv_t;
                for (int v = 1; v < vocab; v++) {
                    const float s = pick_logits[v] * inv_t;
                    if (s > max_lt)
                        max_lt = s;
                }
                std::vector<double> pr((size_t)vocab);
                double sum_t = 0.0;
                for (int v = 0; v < vocab; v++) {
                    const double e = std::exp((double)(pick_logits[v] * inv_t - max_lt));
                    pr[(size_t)v] = e;
                    sum_t += e;
                }
                if (sum_t > 0.0) {
                    std::uniform_real_distribution<double> unif(0.0, sum_t);
                    const double rr = unif(rng);
                    double acc = 0.0;
                    for (int v = 0; v < vocab; v++) {
                        acc += pr[(size_t)v];
                        if (rr <= acc) {
                            next_tok = v;
                            break;
                        }
                    }
                    // Recompute the unsoftened probability of the
                    // newly-picked token so the JSON-full output reflects
                    // the model's actual confidence in it, not the
                    // temperature-warped value.
                    max_l = pick_logits[next_tok];
                    sum_e = 0.0;
                    for (int v = 0; v < vocab; v++) {
                        sum_e += std::exp((double)(pick_logits[v] - max_l));
                    }
                    tok_p = (float)(1.0 / sum_e);
                }
            }

            COHERE_VLOG2(vb, "cohere: step %3d  tok=%5d  p=%.3f  %s\n", step, next_tok, tok_p,
                         (next_tok >= 0 && next_tok < (int)voc.id_to_token.size()) ? voc.id_to_token[next_tok].c_str()
                                                                                   : "?");
            if (next_tok == eos_id) {
                generation_saw_eos = true;
                break;
            }
            if (next_tok < 0)
                break;

            generated.push_back(next_tok);
            gen_probs.push_back(tok_p);
            if (next_tok >= 0 && next_tok < (int)token_counts.size())
                token_counts[(size_t)next_tok]++;
            offset++;

            logits = cohere_decode_step(ctx, T_enc, &next_tok, 1, offset - 1);
            perf.n_dec_steps++;
            if (logits.empty()) {
                if (cohere_abort_if_requested(ctx))
                    return nullptr;
                cohere_set_error(COHERE_ERROR_RUNTIME, "decoder generation graph failed");
                return nullptr;
            }
            if (ctx->repetition_loop_guard && core_repetition_loop::triggered(generated)) {
                repetition_stopped = true;
                break;
            }
        }
    } // end else (greedy path)

    // Snapshot memory
    perf.mem_model_buf = ctx->model.buf ? ggml_backend_buffer_get_size(ctx->model.buf) : 0;
    perf.mem_kv_buf = (ctx->kv_k && ctx->kv_k->buffer) ? ggml_backend_buffer_get_size(ctx->kv_k->buffer) : 0;
    perf.mem_compute_meta = ctx->compute_meta.size();
    perf.t_total_us = ggml_time_us() - t_total_start;

    cohere_perf_print(perf, n_samples, hp.sample_rate, vb);

    // --- Decode tokens to text + build per-token data ---
    struct cohere_result* res = (struct cohere_result*)calloc(1, sizeof(struct cohere_result));
    if (!res)
        return nullptr;
    std::string full_text;
    std::vector<cohere_token_data> tok_data;
    // Maps tok_data index → generated[] index (needed for cross-attn lookup)
    std::vector<int> tok_to_gen_idx;

    // Diarization token IDs (from vocabulary scan)
    const int tok_spkchange = voc.token_id("<|spkchange|>"); // 14
    // <|spk0|>..<|spk15|> are 205..220
    auto is_spk_id = [&](int id) {
        int spk0 = voc.token_id("<|spk0|>");
        return (spk0 >= 0) && (id >= spk0) && (id < spk0 + 16);
    };

    auto hex_nibble = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'A' && c <= 'F')
            return 10 + c - 'A';
        if (c >= 'a' && c <= 'f')
            return 10 + c - 'a';
        return -1;
    };

    auto byte_fallback_value = [&](const std::string& tok, unsigned char& byte) -> bool {
        if (tok.size() != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>')
            return false;
        int hi = hex_nibble(tok[3]), lo = hex_nibble(tok[4]);
        if (hi < 0 || lo < 0)
            return false;
        byte = (unsigned char)((hi << 4) | lo);
        return true;
    };

    auto utf8_expected_len = [](unsigned char lead) -> int {
        if ((lead & 0x80) == 0)
            return 1;
        if ((lead & 0xE0) == 0xC0)
            return 2;
        if ((lead & 0xF0) == 0xE0)
            return 3;
        if ((lead & 0xF8) == 0xF0)
            return 4;
        return 0;
    };

    auto utf8_complete = [&](const std::string& bytes) -> bool {
        if (bytes.empty())
            return false;
        const int expected = utf8_expected_len((unsigned char)bytes[0]);
        if (expected <= 0 || (int)bytes.size() != expected)
            return false;
        for (int j = 1; j < expected; j++) {
            if ((((unsigned char)bytes[j]) & 0xC0) != 0x80)
                return false;
        }
        return true;
    };

    auto emit_token = [&](int id, float p, int gen_idx, const std::string& t) {
        cohere_token_data td;
        td.id = id;
        td.p = p;
        td.t0 = 0;
        td.t1 = 0;
        snprintf(td.text, sizeof(td.text), "%s", t.c_str());
        // Ensure truncation doesn't break a multi-byte UTF-8 codepoint.
        // Walk back from the end if the last byte is an incomplete lead/continuation.
        {
            size_t len = strlen(td.text);
            while (len > 0 && (td.text[len - 1] & 0xC0) == 0x80) {
                len--; // skip continuation bytes
            }
            if (len > 0) {
                unsigned char lead = (unsigned char)td.text[len - 1];
                int expected = (lead >= 0xF0) ? 4 : (lead >= 0xE0) ? 3 : (lead >= 0xC0) ? 2 : 1;
                if ((int)(strlen(td.text) - (len - 1)) < expected) {
                    td.text[len - 1] = '\0'; // remove incomplete codepoint
                }
            }
        }
        tok_data.push_back(td);
        tok_to_gen_idx.push_back(gen_idx);
        full_text += t;
    };

    std::string pending_bytes;
    int pending_id = -1;
    int pending_gen_idx = -1;
    float pending_p = 1.0f;

    auto flush_pending_bytes = [&]() {
        if (pending_bytes.empty())
            return;
        // If the decoder produced an incomplete byte-fallback sequence
        // (for example a lone UTF-8 lead byte before a normal token), do
        // not surface raw invalid UTF-8 into transcript/token text.
        if (utf8_complete(pending_bytes))
            emit_token(pending_id, pending_p, pending_gen_idx, pending_bytes);
        pending_bytes.clear();
        pending_id = -1;
        pending_gen_idx = -1;
        pending_p = 1.0f;
    };

    for (int i = 0; i < (int)generated.size(); i++) {
        int id = generated[i];
        if (id < 0 || id >= (int)voc.id_to_token.size())
            continue;
        const std::string& tok = voc.id_to_token[id];

        std::string t;
        unsigned char byte = 0;
        if (byte_fallback_value(tok, byte)) {
            if (pending_bytes.empty()) {
                pending_id = id;
                pending_gen_idx = i;
                pending_p = gen_probs[i];
            } else {
                pending_p = std::min(pending_p, gen_probs[i]);
                pending_gen_idx = i;
            }
            pending_bytes.push_back((char)byte);
            if (utf8_complete(pending_bytes)) {
                flush_pending_bytes();
            }
            continue;
        }

        flush_pending_bytes();

        if (tok.front() == '<' && tok.back() == '>') {
            // SentencePiece byte-fallback pieces: "<0xE6>" → raw byte 0xE6.
            // These appear for any codepoint that wasn't in the unigram
            // vocab — most kanji compounds in Japanese (e.g. 漫画 → 漫
            // is byte-fallback as <0xE6><0xBC><0xAB>, but 画 is a single
            // piece). Without decoding, the byte sequence is dropped
            // entirely and only the kanji that survived as single pieces
            // remain in the transcript (issue #67).
            // Render diarization tokens; drop all other special tokens
            if (id == tok_spkchange) {
                t = " [SPEAKER_TURN]";
            } else if (is_spk_id(id)) {
                int spk0 = voc.token_id("<|spk0|>");
                char buf[32];
                snprintf(buf, sizeof(buf), " [Speaker %d]", id - spk0);
                t = buf;
            } else {
                continue;
            }
        } else {
            t = tok;
            size_t pos;
            while ((pos = t.find("\xe2\x96\x81")) != std::string::npos)
                t.replace(pos, 3, " ");
        }

        emit_token(id, gen_probs[i], i, t);
    }
    flush_pending_bytes();
    if (!full_text.empty() && full_text[0] == ' ')
        full_text = full_text.substr(1);
    // Do NOT strip the leading space from tok_data[0].text.
    // That space is the SentencePiece word-start marker (▁ → ' ').
    // make_disp_segments uses it as a word-boundary signal; flush() strips it on output.
    // Stripping here causes cross-chunk merges like "ofAmerican." and "point.Representative."

    // Token timestamps via cross-attention alignment.
    //
    // For each generated token we have step_attn[gen_idx] = cross-attn weights from
    // the last decoder layer, layout [T_enc * n_heads] row-major (head h occupies
    // w[h*T_enc .. h*T_enc+T_enc-1]).  Average across heads, then argmax gives the
    // encoder frame the decoder was attending to most strongly → convert to centiseconds.
    //
    // Encoder frame duration:
    //   hop_length * subsampling / sample_rate = 160 * 8 / 16000 = 0.08 s = 8 cs/frame
    //
    // Fallback: linear interpolation proportional to character length (as before).
    const int64_t seg_end_cs = t_offset_cs + (int64_t)((double)n_samples / hp.sample_rate * 100.0);
    constexpr int64_t CS_PER_FRAME = 8; // 160 * 8 / 16000 * 100

    const bool have_attn = !ctx->step_attn.empty() && ctx->attn_T_enc > 0 && ctx->step_attn.size() == generated.size();

    if (have_attn && !tok_data.empty()) {
        const int n_tok = (int)tok_data.size();
        const int T_enc = ctx->attn_T_enc;
        const int n_heads = ctx->attn_n_heads;

        // --- DTW-based token timestamp alignment ---
        //
        // We find the monotone path through the [n_tok × T_enc] cross-attention
        // matrix that maximises cumulative attention weight, using dynamic programming
        // with prefix-max predecessors.  This is the same approach as Whisper's
        // experimental --dtw timestamps, but simplified to work with any model
        // (no pre-selected "timing heads" needed).
        //
        // DP recurrence (prefix-max allows any forward jump per token step):
        //   D[i][j] = A[i][j] + max_{k ≤ j}( D[i-1][k] )
        //
        // Each D[i][j] is backed by P[i][j] = argmax_{k ≤ j}( D[i-1][k] ),
        // which is stored cheaply as the running argmax while scanning row i-1.
        // Traceback is O(n_tok): path[i] = P[i+1][ path[i+1] ].
        //
        // Memory: two flat arrays of (n_tok × T_enc) floats/ints.
        //   Worst case (35 s chunk): ~230 tokens × 438 frames × 8 bytes ≈ 787 KiB.

        // Step 1: head-averaged attention matrix A[n_tok * T_enc]
        //
        // Temporal offset correction:
        // step_attn[i] is collected while the decoder processes generated[i] as input
        // to produce the NEXT token's logits.  The cross-attention at that step reflects
        // which encoder frames are relevant for predicting generated[i+1], not generated[i].
        // Using step_attn[gi-1] (one step back) corrects this one-token forward bias,
        // bringing timestamps to the start rather than the end of each word.
        // For the very first generated token (gi==0) there is no previous step, so we
        // use step_attn[0] (same as before — one-step error for the first token only).
        std::vector<float> A(n_tok * T_enc, 0.0f);
        for (int i = 0; i < n_tok; i++) {
            int gi = tok_to_gen_idx[i];
            int gi_src = (gi > 0) ? gi - 1 : 0; // one-step back to get start-of-token position
            const float* w = ctx->step_attn[gi_src].data();
            float* Ai = A.data() + i * T_enc;
            for (int h = 0; h < n_heads; h++)
                for (int t = 0; t < T_enc; t++)
                    Ai[t] += w[h * T_enc + t];
            for (int t = 0; t < T_enc; t++)
                Ai[t] /= n_heads;
        }

        // Column normalization: subtract per-frame mean across all tokens.
        // Without this, "hot" encoder frames that many tokens attend to (often
        // early frames or silence) dominate the prefix-max DP, causing the entire
        // path to collapse to a single frame.  After normalization, the path
        // is forced to spread — it only benefits from frames where a token attends
        // *more than average*, not from frames that are globally popular.
        {
            std::vector<float> col_mean(T_enc, 0.f);
            for (int i = 0; i < n_tok; i++)
                for (int t = 0; t < T_enc; t++)
                    col_mean[t] += A[i * T_enc + t];
            for (int t = 0; t < T_enc; t++)
                col_mean[t] /= n_tok;
            for (int i = 0; i < n_tok; i++)
                for (int t = 0; t < T_enc; t++)
                    A[i * T_enc + t] -= col_mean[t];
        }

        // Step 2: forward DP
        std::vector<float> D(n_tok * T_enc);
        std::vector<int> P(n_tok * T_enc); // back-pointer: P[i][j] = best predecessor frame

        // Row 0: no predecessors
        for (int j = 0; j < T_enc; j++)
            D[j] = A[j];

        for (int i = 1; i < n_tok; i++) {
            const float* Di_1 = D.data() + (i - 1) * T_enc;
            const float* Ai = A.data() + i * T_enc;
            float* Di = D.data() + i * T_enc;
            int* Pi = P.data() + i * T_enc;

            // Running prefix-max scan over row i-1
            float pm_val = Di_1[0];
            int pm_idx = 0;
            for (int j = 0; j < T_enc; j++) {
                if (Di_1[j] > pm_val) {
                    pm_val = Di_1[j];
                    pm_idx = j;
                }
                Di[j] = Ai[j] + pm_val;
                Pi[j] = pm_idx;
            }
        }

        // Step 3: traceback — recover optimal monotone path
        std::vector<int> path(n_tok);
        {
            const float* Dlast = D.data() + (n_tok - 1) * T_enc;
            path[n_tok - 1] = (int)(std::max_element(Dlast, Dlast + T_enc) - Dlast);
        }
        for (int i = n_tok - 2; i >= 0; i--)
            path[i] = P[(i + 1) * T_enc + path[i + 1]];

        // Step 4: path[i] is the encoder frame for token i → centiseconds
        for (int i = 0; i < n_tok; i++) {
            tok_data[i].t0 = t_offset_cs + (int64_t)path[i] * CS_PER_FRAME;
            tok_data[i].t1 = (i + 1 < n_tok) ? t_offset_cs + (int64_t)path[i + 1] * CS_PER_FRAME : seg_end_cs;
        }
        COHERE_VLOG2(vb, "cohere: timestamps via DTW alignment (%d tokens, T_enc=%d)\n", n_tok, T_enc);

        // Optional: dump raw per-head attention for head-selection analysis.
        // Set COHERE_DUMP_ATTN=/path/to/file.bin to activate.
        // Format: int32 n_tok, int32 n_heads, int32 T_enc, then
        //         n_tok × n_heads × T_enc float32 row-major.
        if (const char* dump_path = getenv("COHERE_DUMP_ATTN")) {
            if (FILE* fp = fopen(dump_path, "wb")) {
                int32_t hdr[3] = {n_tok, n_heads, T_enc};
                fwrite(hdr, sizeof(hdr), 1, fp);
                // Write raw per-head weights (not averaged), using same gi_src shift
                for (int i = 0; i < n_tok; i++) {
                    int gi = tok_to_gen_idx[i];
                    int gi_src = (gi > 0) ? gi - 1 : 0;
                    fwrite(ctx->step_attn[gi_src].data(), sizeof(float), T_enc * n_heads, fp);
                }
                fclose(fp);
                fprintf(stderr,
                        "cohere: attention dump written to %s "
                        "(%d tok × %d heads × %d frames)\n",
                        dump_path, n_tok, n_heads, T_enc);
            }
        }
    } else {
        // Fallback: distribute segment duration proportional to character length
        int total_chars = 0;
        for (const auto& td : tok_data)
            total_chars += (int)strlen(td.text);
        if (total_chars > 0) {
            const int64_t seg_dur_cs = seg_end_cs - t_offset_cs;
            int char_pos = 0;
            for (auto& td : tok_data) {
                int len = (int)strlen(td.text);
                td.t0 = t_offset_cs + (int64_t)((double)char_pos / total_chars * seg_dur_cs);
                char_pos += len;
                td.t1 = t_offset_cs + (int64_t)((double)char_pos / total_chars * seg_dur_cs);
            }
        }
        if (!have_attn)
            COHERE_VLOG2(vb, "cohere: timestamps via linear interpolation (no attn weights)\n");
    }

    res->n_tokens = (int)tok_data.size();
    res->generated_tokens = (int)generated.size() + (generation_saw_eos ? 1 : 0);
    res->generation_limit = max_gen;
    res->generation_capacity = generation_capacity;
    res->stopped_by_max_tokens = !generation_saw_eos && !repetition_stopped && (int)generated.size() >= max_gen;
    res->repetition_stopped = repetition_stopped;
    if (res->n_tokens > 0) {
        res->tokens = (cohere_token_data*)malloc(res->n_tokens * sizeof(cohere_token_data));
        if (!res->tokens) {
            cohere_result_free(res);
            return nullptr;
        }
        memcpy(res->tokens, tok_data.data(), res->n_tokens * sizeof(cohere_token_data));
    }
    res->text = (char*)malloc(full_text.size() + 1);
    if (!res->text) {
        cohere_result_free(res);
        return nullptr;
    }
    memcpy(res->text, full_text.c_str(), full_text.size() + 1);
    return res;
}

void cohere_result_free(struct cohere_result* r) {
    if (!r)
        return;
    free(r->text);
    free(r->tokens);
    free(r);
}

void cohere_set_temperature(struct cohere_context* ctx, float temperature, uint64_t seed) {
    if (!ctx)
        return;
    ctx->decode_temperature = temperature;
    ctx->decode_seed = seed;
}

void cohere_set_max_new_tokens(struct cohere_context* ctx, int max_new_tokens) {
    if (!ctx)
        return;
    ctx->max_new_tokens = max_new_tokens > 0 ? max_new_tokens : 0;
}

void cohere_set_frequency_penalty(struct cohere_context* ctx, float frequency_penalty) {
    if (!ctx)
        return;
    ctx->frequency_penalty = frequency_penalty > 0.0f ? frequency_penalty : 0.0f;
}

void cohere_set_repetition_loop_guard(struct cohere_context* ctx, bool enabled) {
    if (!ctx)
        return;
    ctx->repetition_loop_guard = enabled;
}

void cohere_set_alignment_collection(struct cohere_context* ctx, bool enabled) {
    if (!ctx)
        return;
    ctx->params.collect_alignment = enabled;
}

void cohere_set_beam_size(struct cohere_context* ctx, int n) {
    if (!ctx)
        return;
    ctx->beam_size = n > 0 ? n : 1;
}

char* cohere_transcribe(struct cohere_context* ctx, const float* samples, int n_samples, const char* lang) {
    struct cohere_result* r = cohere_transcribe_ex(ctx, samples, n_samples, lang, 0);
    if (!r)
        return nullptr;
    char* text = r->text;
    r->text = nullptr;
    cohere_result_free(r);
    return text;
}

// ---- Stage-level entry points for crispasr-diff ----

float* cohere_compute_mel(struct cohere_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel) {
    if (!ctx || !samples || n_samples <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;

    int T_mel = 0;
    auto mel = cohere_compute_features(hp, ctx->frontend, samples, n_samples, ctx->params.n_threads, T_mel);
    if (mel.empty())
        return nullptr;

    if (out_n_mels)
        *out_n_mels = hp.n_mels;
    if (out_T_mel)
        *out_T_mel = T_mel;

    float* r = (float*)malloc(mel.size() * sizeof(float));
    if (!r)
        return nullptr;
    std::memcpy(r, mel.data(), mel.size() * sizeof(float));
    return r;
}

float* cohere_run_encoder(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model) {
    if (!ctx || !mel || T_mel <= 0)
        return nullptr;
    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels) {
        fprintf(stderr, "cohere: mel feature mismatch (%d vs %d)\n", n_mels, hp.n_mels);
        return nullptr;
    }

    // Build + allocate + run the encoder graph on the supplied mel.
    // This mirrors the per-chunk encoder pass in cohere_transcribe_ex
    // but strips out the performance counters, decoder cross-KV copy,
    // and multi-chunk accumulation.
    struct ggml_cgraph* gf_enc = cohere_build_graph_encoder(ctx, T_mel, false);
    if (!gf_enc) {
        fprintf(stderr, "cohere: failed to build encoder graph\n");
        return nullptr;
    }

    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf_enc)) {
        fprintf(stderr, "cohere: failed to allocate encoder graph\n");
        return nullptr;
    }

    // Set mel input.
    struct ggml_tensor* mel_t = ggml_graph_get_tensor(gf_enc, "mel");
    if (!mel_t)
        return nullptr;
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    // Positional encoding. H3 = floor((T_mel / 8)) after the three
    // stride-2 conv layers; matches the arithmetic in
    // cohere_transcribe_ex.
    const int H1 = (T_mel + 2 - 3) / 2 + 1;
    const int H2 = (H1 + 2 - 3) / 2 + 1;
    const int H3 = (H2 + 2 - 3) / 2 + 1;
    auto pos_enc = ct_rel_pos_enc(H3, hp.enc_d_model);
    struct ggml_tensor* pos_enc_t = ggml_graph_get_tensor(gf_enc, "pos_enc");
    if (!pos_enc_t)
        return nullptr;
    ggml_backend_tensor_set(pos_enc_t, pos_enc.data(), 0, pos_enc.size() * sizeof(float));

    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf_enc, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: failed to compute encoder graph\n");
        return nullptr;
    }

    struct ggml_tensor* enc_out = ggml_graph_get_tensor(gf_enc, "enc_out");
    if (!enc_out) {
        fprintf(stderr, "cohere: enc_out tensor not found\n");
        return nullptr;
    }
    const int d = (int)enc_out->ne[0];
    const int T_enc = (int)enc_out->ne[1];

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;

    float* r = (float*)malloc((size_t)d * T_enc * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(enc_out, r, 0, (size_t)d * T_enc * sizeof(float));
    return r;
}

float* cohere_run_encoder_batch_ex(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                   enum cohere_encoder_batch_precision precision, int* out_T_enc, int* out_d_model,
                                   struct cohere_encoder_batch_stats* out_stats) {
    if (out_T_enc)
        *out_T_enc = 0;
    if (out_d_model)
        *out_d_model = 0;
    if (out_stats)
        std::memset(out_stats, 0, sizeof(*out_stats));
    if (!ctx || !mel || n_batch <= 0 || T_mel <= 0 || !cohere_batch_precision_valid(precision))
        return nullptr;

    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels) {
        fprintf(stderr, "cohere: mel feature mismatch (%d vs %d)\n", n_mels, hp.n_mels);
        return nullptr;
    }
    if ((size_t)n_batch > SIZE_MAX / (size_t)n_mels / (size_t)T_mel / sizeof(float)) {
        fprintf(stderr, "cohere: encoder batch input size overflow\n");
        return nullptr;
    }

    int64_t t0 = ggml_time_us();
    struct ggml_cgraph* gf_enc = cohere_build_graph_encoder_batch(ctx, T_mel, n_batch, false, precision);
    int64_t t1 = ggml_time_us();
    if (out_stats)
        out_stats->graph_build_us = t1 - t0;
    if (!gf_enc) {
        fprintf(stderr, "cohere: failed to build batched encoder graph\n");
        return nullptr;
    }

    t0 = ggml_time_us();
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf_enc)) {
        fprintf(stderr, "cohere: failed to allocate batched encoder graph (batch=%d)\n", n_batch);
        return nullptr;
    }
    t1 = ggml_time_us();
    if (out_stats) {
        out_stats->graph_alloc_us = t1 - t0;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->ggml_alloc); i++) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(ctx->ggml_alloc, i);
            out_stats->scheduler_bytes += ggml_backend_sched_get_buffer_size(ctx->ggml_alloc, backend);
        }
    }

    struct ggml_tensor* mel_t = ggml_graph_get_tensor(gf_enc, "mel_batch");
    if (!mel_t)
        return nullptr;
    t0 = ggml_time_us();
    const size_t mel_bytes = (size_t)n_batch * (size_t)T_mel * (size_t)n_mels * sizeof(float);
    ggml_backend_tensor_set(mel_t, mel, 0, mel_bytes);

    const int H1 = (T_mel + 2 - 3) / 2 + 1;
    const int H2 = (H1 + 2 - 3) / 2 + 1;
    const int H3 = (H2 + 2 - 3) / 2 + 1;
    auto pos_enc = ct_rel_pos_enc(H3, hp.enc_d_model);
    struct ggml_tensor* pos_enc_t = ggml_graph_get_tensor(gf_enc, "pos_enc_batch");
    if (!pos_enc_t)
        return nullptr;
    ggml_backend_tensor_set(pos_enc_t, pos_enc.data(), 0, pos_enc.size() * sizeof(float));
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->input_us = t1 - t0;

    t0 = ggml_time_us();
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf_enc, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: failed to compute batched encoder graph\n");
        return nullptr;
    }
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->compute_us = t1 - t0;

    struct ggml_tensor* enc_out = ggml_graph_get_tensor(gf_enc, "enc_out_batch");
    if (!enc_out) {
        fprintf(stderr, "cohere: batched enc_out tensor not found\n");
        return nullptr;
    }
    const int d = (int)enc_out->ne[0];
    const int T_enc = (int)enc_out->ne[1];
    if (enc_out->ne[2] != n_batch || enc_out->ne[3] != 1) {
        fprintf(stderr, "cohere: unexpected batched encoder output shape [%lld,%lld,%lld,%lld]\n",
                (long long)enc_out->ne[0], (long long)enc_out->ne[1], (long long)enc_out->ne[2],
                (long long)enc_out->ne[3]);
        return nullptr;
    }

    if (out_T_enc)
        *out_T_enc = T_enc;
    if (out_d_model)
        *out_d_model = d;

    const size_t n_values = (size_t)n_batch * (size_t)T_enc * (size_t)d;
    float* r = (float*)malloc(n_values * sizeof(float));
    if (!r)
        return nullptr;
    t0 = ggml_time_us();
    ggml_backend_tensor_get(enc_out, r, 0, n_values * sizeof(float));
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->readback_us = t1 - t0;
    return r;
}

static bool cohere_set_encoder_batch_padded_inputs(struct cohere_context* ctx, struct ggml_cgraph* gf,
                                                   const float* mel, const int* valid_T_mel, int n_batch, int n_mels,
                                                   int T_mel_max) {
    const auto& hp = ctx->model.hparams;
    std::vector<int> lengths_0(n_batch);
    std::vector<int> lengths_2(n_batch);
    std::vector<int> lengths_5(n_batch);
    for (int lane = 0; lane < n_batch; lane++) {
        lengths_0[lane] = cohere_encoder_padded::stride2_length(valid_T_mel[lane]);
        lengths_2[lane] = cohere_encoder_padded::stride2_length(lengths_0[lane]);
        lengths_5[lane] = cohere_encoder_padded::stride2_length(lengths_2[lane]);
    }

    const size_t mel_values = (size_t)n_batch * (size_t)T_mel_max * (size_t)n_mels;
    std::vector<float> sanitized_mel(mel, mel + mel_values);
    for (int lane = 0; lane < n_batch; lane++) {
        const size_t first = cohere_encoder_padded::lane_value_index(lane, valid_T_mel[lane], 0, T_mel_max, n_mels);
        const size_t last = cohere_encoder_padded::lane_value_index(lane, T_mel_max, 0, T_mel_max, n_mels);
        std::fill(sanitized_mel.begin() + first, sanitized_mel.begin() + last, 0.0f);
    }
    struct ggml_tensor* mel_input = ggml_graph_get_tensor(gf, "padded_mel_batch");
    if (!mel_input)
        return false;
    ggml_backend_tensor_set(mel_input, sanitized_mel.data(), 0, mel_values * sizeof(float));

    auto set_mask = [&](const char* name, int padded_length, const std::vector<int>& valid_lengths) {
        struct ggml_tensor* tensor = ggml_graph_get_tensor(gf, name);
        if (!tensor)
            return false;
        std::vector<float> mask((size_t)n_batch * (size_t)padded_length, 0.0f);
        for (int lane = 0; lane < n_batch; lane++) {
            std::fill(mask.begin() + (size_t)lane * padded_length,
                      mask.begin() + (size_t)lane * padded_length + valid_lengths[lane], 1.0f);
        }
        ggml_backend_tensor_set(tensor, mask.data(), 0, mask.size() * sizeof(float));
        return true;
    };

    const int H0 = cohere_encoder_padded::stride2_length(T_mel_max);
    const int H2 = cohere_encoder_padded::stride2_length(H0);
    const int T_enc_max = cohere_encoder_padded::stride2_length(H2);
    if (!set_mask("padded_sub_mask_0", H0, lengths_0) ||
        !set_mask("padded_sub_mask_2", H2, lengths_2) ||
        !set_mask("padded_sub_mask_3", H2, lengths_2) ||
        !set_mask("padded_sub_mask_5", T_enc_max, lengths_5) ||
        !set_mask("padded_sub_mask_6", T_enc_max, lengths_5) ||
        !set_mask("padded_enc_valid_mask", T_enc_max, lengths_5)) {
        return false;
    }

    struct ggml_tensor* key_mask_input = ggml_graph_get_tensor(gf, "padded_enc_key_mask");
    if (!key_mask_input)
        return false;
    std::vector<float> key_mask((size_t)n_batch * (size_t)T_enc_max * (size_t)T_enc_max, -10000.0f);
    for (int lane = 0; lane < n_batch; lane++) {
        for (int query = 0; query < T_enc_max; query++) {
            const size_t row = ((size_t)lane * (size_t)T_enc_max + (size_t)query) * (size_t)T_enc_max;
            std::fill(key_mask.begin() + row, key_mask.begin() + row + lengths_5[lane], 0.0f);
        }
    }
    ggml_backend_tensor_set(key_mask_input, key_mask.data(), 0, key_mask.size() * sizeof(float));

    auto pos_enc = ct_rel_pos_enc(T_enc_max, hp.enc_d_model);
    struct ggml_tensor* pos_enc_input = ggml_graph_get_tensor(gf, "padded_pos_enc");
    if (!pos_enc_input)
        return false;
    ggml_backend_tensor_set(pos_enc_input, pos_enc.data(), 0, pos_enc.size() * sizeof(float));
    return true;
}

float* cohere_run_encoder_batch_padded(struct cohere_context* ctx, const float* mel, const int* valid_T_mel,
                                       int n_batch, int n_mels, int T_mel_max, int* out_valid_T_enc,
                                       int* out_T_enc_max, int* out_d_model,
                                       struct cohere_encoder_batch_stats* out_stats) {
    cohere_clear_error();
    if (out_T_enc_max)
        *out_T_enc_max = 0;
    if (out_d_model)
        *out_d_model = 0;
    if (out_valid_T_enc && n_batch > 0 && n_batch <= cohere_encoder_padded::max_batch)
        std::fill(out_valid_T_enc, out_valid_T_enc + n_batch, 0);
    if (out_stats)
        std::memset(out_stats, 0, sizeof(*out_stats));
    if (!ctx || !mel || !valid_T_mel || n_batch < 1 || n_batch > cohere_encoder_padded::max_batch ||
        T_mel_max < 1 || T_mel_max > cohere_encoder_padded::max_mel_frames) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "invalid padded encoder arguments");
        return nullptr;
    }
    if (cohere_abort_if_requested(ctx))
        return nullptr;

    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "padded encoder feature dimension does not match the model");
        fprintf(stderr, "cohere: padded mel feature mismatch (%d vs %d)\n", n_mels, hp.n_mels);
        return nullptr;
    }
    if ((size_t)n_batch > SIZE_MAX / (size_t)n_mels / (size_t)T_mel_max / sizeof(float)) {
        cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "padded encoder input dimensions overflow size_t");
        fprintf(stderr, "cohere: padded encoder batch input size overflow\n");
        return nullptr;
    }

    std::vector<int> lengths_0(n_batch);
    std::vector<int> lengths_2(n_batch);
    std::vector<int> lengths_5(n_batch);
    for (int lane = 0; lane < n_batch; lane++) {
        if (valid_T_mel[lane] < 1 || valid_T_mel[lane] > T_mel_max) {
            cohere_set_error(COHERE_ERROR_INVALID_ARGUMENT, "padded encoder lane length is out of range");
            return nullptr;
        }
        lengths_0[lane] = cohere_encoder_padded::stride2_length(valid_T_mel[lane]);
        lengths_2[lane] = cohere_encoder_padded::stride2_length(lengths_0[lane]);
        lengths_5[lane] = cohere_encoder_padded::stride2_length(lengths_2[lane]);
    }
    int64_t t0 = ggml_time_us();
    struct ggml_cgraph* gf = cohere_build_graph_encoder_batch_padded(ctx, T_mel_max, n_batch, false);
    int64_t t1 = ggml_time_us();
    if (out_stats)
        out_stats->graph_build_us = t1 - t0;
    if (!gf) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "could not build the padded encoder graph");
        fprintf(stderr, "cohere: failed to build padded encoder graph\n");
        return nullptr;
    }

    t0 = ggml_time_us();
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate the padded encoder graph");
        fprintf(stderr, "cohere: failed to allocate padded encoder graph (batch=%d)\n", n_batch);
        return nullptr;
    }
    t1 = ggml_time_us();
    if (out_stats) {
        out_stats->graph_alloc_us = t1 - t0;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->ggml_alloc); i++) {
            ggml_backend_t backend = ggml_backend_sched_get_backend(ctx->ggml_alloc, i);
            out_stats->scheduler_bytes += ggml_backend_sched_get_buffer_size(ctx->ggml_alloc, backend);
        }
    }

    const int H0 = cohere_encoder_padded::stride2_length(T_mel_max);
    const int H2 = cohere_encoder_padded::stride2_length(H0);
    const int T_enc_max = cohere_encoder_padded::stride2_length(H2);
    t0 = ggml_time_us();
    if (!cohere_set_encoder_batch_padded_inputs(ctx, gf, mel, valid_T_mel, n_batch, n_mels, T_mel_max)) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "could not populate padded encoder graph inputs");
        return nullptr;
    }
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->input_us = t1 - t0;

    t0 = ggml_time_us();
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        if (cohere_abort_if_requested(ctx))
            return nullptr;
        cohere_set_error(COHERE_ERROR_RUNTIME, "padded encoder graph compute failed");
        fprintf(stderr, "cohere: failed to compute padded encoder graph\n");
        return nullptr;
    }
    if (cohere_abort_if_requested(ctx))
        return nullptr;
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->compute_us = t1 - t0;

    struct ggml_tensor* output = ggml_graph_get_tensor(gf, "padded_enc_out");
    if (!output || output->ne[0] != hp.dec_d_model || output->ne[1] != T_enc_max || output->ne[2] != n_batch ||
        output->ne[3] != 1) {
        cohere_set_error(COHERE_ERROR_INVARIANT, "padded encoder returned an invalid output tensor");
        fprintf(stderr, "cohere: unexpected padded encoder output shape\n");
        return nullptr;
    }
    const size_t output_values = (size_t)n_batch * (size_t)T_enc_max * (size_t)hp.dec_d_model;
    float* result = (float*)malloc(output_values * sizeof(float));
    if (!result) {
        cohere_set_error(COHERE_ERROR_OUT_OF_MEMORY, "could not allocate padded encoder readback memory");
        return nullptr;
    }
    t0 = ggml_time_us();
    ggml_backend_tensor_get(output, result, 0, output_values * sizeof(float));
    t1 = ggml_time_us();
    if (out_stats)
        out_stats->readback_us = t1 - t0;
    if (out_valid_T_enc)
        std::copy(lengths_5.begin(), lengths_5.end(), out_valid_T_enc);
    if (out_T_enc_max)
        *out_T_enc_max = T_enc_max;
    if (out_d_model)
        *out_d_model = hp.dec_d_model;
    return result;
}

int cohere_run_encoder_batch_padded_staged(struct cohere_context* ctx, const float* mel, const int* valid_T_mel,
                                           int n_batch, int n_mels, int T_mel_max, cohere_padded_stage_cb cb,
                                           void* userdata) {
    if (!ctx || !mel || !valid_T_mel || !cb || n_batch < 1 || n_batch > cohere_encoder_padded::max_batch ||
        T_mel_max < 1 || T_mel_max > cohere_encoder_padded::max_mel_frames) {
        return -1;
    }
    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels)
        return -1;
    for (int lane = 0; lane < n_batch; lane++) {
        if (valid_T_mel[lane] < 1 || valid_T_mel[lane] > T_mel_max)
            return -1;
    }

    const size_t staged_meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    if (ctx->compute_meta.size() < staged_meta)
        ctx->compute_meta.resize(staged_meta);
    struct ggml_cgraph* gf = cohere_build_graph_encoder_batch_padded(ctx, T_mel_max, n_batch, true);
    if (!gf)
        return -1;
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate staged padded encoder graph\n");
        return -1;
    }
    if (!cohere_set_encoder_batch_padded_inputs(ctx, gf, mel, valid_T_mel, n_batch, n_mels, T_mel_max))
        return -1;
    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: staged padded encoder compute failed\n");
        return -1;
    }

    struct stage_name {
        const char* tensor;
        const char* callback;
    };
    const stage_name stages[] = {
        {"padded_pre_conv0", "pre_conv0"}, {"padded_pre_conv3", "pre_conv3"},
        {"padded_pre_conv6", "pre_conv6"}, {"padded_pre_enc_out", "pre_enc_out"},
        {"padded_L0_ff1", "L0_ff1"},       {"padded_L0_attn", "L0_attn"},
        {"padded_L0_conv", "L0_conv"},     {"padded_L0_ff2", "L0_ff2"},
        {"padded_enc_L00", "enc_L00"},     {"padded_enc_out", "enc_out"},
    };
    for (const auto& stage : stages) {
        struct ggml_tensor* value = ggml_graph_get_tensor(gf, stage.tensor);
        if (!value)
            return -1;
        const size_t total = ggml_nelements(value);
        std::vector<float> data(total);
        ggml_backend_tensor_get(value, data.data(), 0, total * sizeof(float));
        cb(stage.callback, data.data(), (int)value->ne[0], (int)value->ne[1], (int)value->ne[2],
           (int)value->ne[3], userdata);
    }
    return 0;
}

float* cohere_run_encoder_batch(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                int* out_T_enc, int* out_d_model, struct cohere_encoder_batch_stats* out_stats) {
    return cohere_run_encoder_batch_ex(ctx, mel, n_batch, n_mels, T_mel, COHERE_ENCODER_BATCH_PRECISE, out_T_enc,
                                       out_d_model, out_stats);
}

int cohere_run_encoder_batch_staged_ex(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels,
                                       int T_mel, enum cohere_encoder_batch_precision precision,
                                       cohere_batch_stage_cb cb, void* userdata) {
    if (!ctx || !mel || n_batch <= 0 || T_mel <= 0 || !cb || !cohere_batch_precision_valid(precision))
        return -1;
    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels) {
        fprintf(stderr, "cohere: mel feature mismatch (%d vs %d)\n", n_mels, hp.n_mels);
        return -1;
    }

    const size_t staged_meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    if (ctx->compute_meta.size() < staged_meta)
        ctx->compute_meta.resize(staged_meta);

    struct ggml_cgraph* gf = cohere_build_graph_encoder_batch(ctx, T_mel, n_batch, true, precision);
    if (!gf)
        return -1;
    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate staged batched encoder graph\n");
        return -1;
    }

    struct ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel_batch");
    if (!mel_t)
        return -1;
    const size_t mel_values = (size_t)n_batch * (size_t)n_mels * (size_t)T_mel;
    ggml_backend_tensor_set(mel_t, mel, 0, mel_values * sizeof(float));

    const int H1 = (T_mel + 2 - 3) / 2 + 1;
    const int H2 = (H1 + 2 - 3) / 2 + 1;
    const int H3 = (H2 + 2 - 3) / 2 + 1;
    auto pos_enc = ct_rel_pos_enc(H3, hp.enc_d_model);
    struct ggml_tensor* pos_enc_t = ggml_graph_get_tensor(gf, "pos_enc_batch");
    if (!pos_enc_t)
        return -1;
    ggml_backend_tensor_set(pos_enc_t, pos_enc.data(), 0, pos_enc.size() * sizeof(float));

    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: staged batched encoder compute failed\n");
        return -1;
    }

    struct stage_name {
        const char* tensor;
        const char* callback;
    };
    const stage_name stages[] = {
        {"batch_pre_conv0", "pre_conv0"}, {"batch_pre_conv3", "pre_conv3"},
        {"batch_pre_conv6", "pre_conv6"}, {"batch_pre_enc_out", "pre_enc_out"},
        {"batch_L0_ff1", "L0_ff1"},       {"batch_L0_attn", "L0_attn"},
        {"batch_L0_conv", "L0_conv"},     {"batch_L0_ff2", "L0_ff2"},
        {"batch_enc_L00", "enc_L00"},     {"enc_out_batch", "enc_out"},
    };
    for (const auto& stage : stages) {
        struct ggml_tensor* value = ggml_graph_get_tensor(gf, stage.tensor);
        if (!value)
            continue;
        const size_t total = ggml_nelements(value);
        if (total % (size_t)n_batch != 0) {
            fprintf(stderr, "cohere: staged tensor %s cannot be split across %d lanes\n", stage.tensor, n_batch);
            return -1;
        }
        std::vector<float> data(total);
        ggml_backend_tensor_get(value, data.data(), 0, total * sizeof(float));
        cb(stage.callback, data.data(), n_batch, (int)(total / (size_t)n_batch), userdata);
    }
    return 0;
}

int cohere_run_encoder_batch_staged(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                    cohere_batch_stage_cb cb, void* userdata) {
    return cohere_run_encoder_batch_staged_ex(ctx, mel, n_batch, n_mels, T_mel, COHERE_ENCODER_BATCH_PRECISE, cb,
                                              userdata);
}

int cohere_debug_dump_encoder_b1_graphs(struct cohere_context* ctx, int T_mel, const char* production_path,
                                        const char* batch_path) {
    if (!ctx || T_mel <= 0 || !production_path || !batch_path)
        return -1;

    auto dump = [](struct ggml_cgraph* graph, const char* path) -> bool {
        FILE* file = std::fopen(path, "wb");
        if (!file) {
            std::fprintf(stderr, "cohere: cannot create graph metadata dump %s\n", path);
            return false;
        }
        std::fprintf(file,
                     "index\top\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tsrc0_op\tsrc1_op\tname\n");
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++) {
            const struct ggml_tensor* node = ggml_graph_node(graph, i);
            const char* src0 = node->src[0] ? ggml_op_name(node->src[0]->op) : "-";
            const char* src1 = node->src[1] ? ggml_op_name(node->src[1]->op) : "-";
            std::fprintf(file, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%zu\t%zu\t%zu\t%zu\t%s\t%s\t%s\n", i,
                         ggml_op_name(node->op), ggml_type_name(node->type), (long long)node->ne[0],
                         (long long)node->ne[1], (long long)node->ne[2], (long long)node->ne[3], node->nb[0],
                         node->nb[1], node->nb[2], node->nb[3], src0, src1, ggml_get_name(node));
        }
        std::fclose(file);
        return true;
    };

    struct ggml_cgraph* production = cohere_build_graph_encoder(ctx, T_mel, true);
    if (!production || !dump(production, production_path))
        return -1;
    // Both builders use ctx->compute_meta, so the first graph must be fully
    // consumed before constructing the second one.
    struct ggml_cgraph* batch =
        cohere_build_graph_encoder_batch(ctx, T_mel, 1, false, COHERE_ENCODER_BATCH_PRECISE);
    if (!batch || !dump(batch, batch_path))
        return -1;
    return 0;
}

int cohere_run_encoder_staged(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, cohere_stage_cb cb,
                              void* userdata) {
    if (!ctx || !mel || T_mel <= 0 || !cb)
        return -1;
    const auto& hp = ctx->model.hparams;
    if (n_mels != hp.n_mels) {
        fprintf(stderr, "cohere: mel feature mismatch (%d vs %d)\n", n_mels, hp.n_mels);
        return -1;
    }

    // Ensure compute_meta is large enough for the staged graph
    const size_t staged_meta = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    if (ctx->compute_meta.size() < staged_meta)
        ctx->compute_meta.resize(staged_meta);

    struct ggml_cgraph* gf = cohere_build_graph_encoder_staged(ctx, T_mel);
    if (!gf)
        return -1;

    ggml_backend_sched_reset(ctx->ggml_alloc);
    if (!ggml_backend_sched_alloc_graph(ctx->ggml_alloc, gf)) {
        fprintf(stderr, "cohere: failed to allocate staged encoder graph\n");
        return -1;
    }

    struct ggml_tensor* mel_t = ggml_graph_get_tensor(gf, "mel");
    if (!mel_t)
        return -1;
    ggml_backend_tensor_set(mel_t, mel, 0, (size_t)n_mels * T_mel * sizeof(float));

    const int H1 = (T_mel + 2 - 3) / 2 + 1;
    const int H2 = (H1 + 2 - 3) / 2 + 1;
    const int H3 = (H2 + 2 - 3) / 2 + 1;
    auto pos_enc = ct_rel_pos_enc(H3, hp.enc_d_model);
    struct ggml_tensor* pos_enc_t = ggml_graph_get_tensor(gf, "pos_enc");
    if (!pos_enc_t)
        return -1;
    ggml_backend_tensor_set(pos_enc_t, pos_enc.data(), 0, pos_enc.size() * sizeof(float));

    if (!cohere_sched_graph_compute(ctx->ggml_alloc, gf, ctx->params.n_threads)) {
        fprintf(stderr, "cohere: staged encoder compute failed\n");
        return -1;
    }

    const int d = hp.enc_d_model;

    // Deliver pre-conv snapshots (4D tensors: [W, H, C, N])
    {
        const char* conv_snaps[] = {"pre_conv0", "pre_conv3", "pre_conv6"};
        for (const char* sn : conv_snaps) {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, sn);
            if (t) {
                const size_t total = ggml_nelements(t);
                std::vector<float> buf(total);
                ggml_backend_tensor_get(t, buf.data(), 0, total * sizeof(float));
                // Report as (total_elements, 1) — just for debug printing
                cb(sn, buf.data(), (int)total, 1, userdata);
            }
        }
    }

    // Deliver pre_encode output
    {
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, "pre_enc_out");
        if (t) {
            const int t_steps = (int)t->ne[1];
            std::vector<float> buf((size_t)d * t_steps);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
            cb("pre_enc_out", buf.data(), t_steps, d, userdata);
        }
    }

    // Deliver layer-0 sub-stage snapshots
    {
        const char* sub_names[] = {"L0_ff1_ln", "L0_ff1_up", "L0_ff1", "L0_attn", "L0_conv", "L0_ff2"};
        for (const char* sn : sub_names) {
            struct ggml_tensor* t = ggml_graph_get_tensor(gf, sn);
            if (t) {
                const int t_steps = (int)t->ne[1];
                std::vector<float> buf((size_t)d * t_steps);
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                cb(sn, buf.data(), t_steps, d, userdata);
            }
        }
    }

    // Deliver per-layer snapshots
    char lbuf[32];
    for (int il = 0; il < hp.enc_n_layers; il++) {
        snprintf(lbuf, sizeof(lbuf), "enc_L%02d", il);
        struct ggml_tensor* t = ggml_graph_get_tensor(gf, lbuf);
        if (!t)
            continue;
        const int t_steps = (int)t->ne[1];
        std::vector<float> buf((size_t)d * t_steps);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        cb(lbuf, buf.data(), t_steps, d, userdata);
    }

    // Deliver enc_out
    struct ggml_tensor* enc_out = ggml_graph_get_tensor(gf, "enc_out");
    if (enc_out) {
        const int t_steps = (int)enc_out->ne[1];
        const int d_out = (int)enc_out->ne[0];
        std::vector<float> buf((size_t)d_out * t_steps);
        ggml_backend_tensor_get(enc_out, buf.data(), 0, buf.size() * sizeof(float));
        cb("enc_out", buf.data(), t_steps, d_out, userdata);
    }

    return 0;
}
