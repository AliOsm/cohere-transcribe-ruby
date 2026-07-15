#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cohere_context;

// Thread-local diagnostic for C entry points that otherwise return NULL. The
// message remains valid until the next Cohere call on the same thread.
enum cohere_error_kind {
    COHERE_ERROR_NONE = 0,
    COHERE_ERROR_INVALID_ARGUMENT = 1,
    COHERE_ERROR_OUT_OF_MEMORY = 2,
    COHERE_ERROR_INVARIANT = 3,
    COHERE_ERROR_RUNTIME = 4,
    COHERE_ERROR_CANCELLED = 5,
};
enum cohere_error_kind cohere_last_error_kind(void);
const char* cohere_last_error_message(void);

struct cohere_context_params {
    int n_threads;       // default: number of physical cores
    bool use_flash;      // flash attention in decoder (default: false for now)
    bool use_gpu;        // false => force CPU backend
    bool no_punctuation; // use <|nopnc|> instead of <|pnc|> in prompt (default: false)
    bool diarize;        // use <|diarize|> instead of <|nodiarize|>; model may emit
                         // <|spkchange|> and <|spk0|>..<|spk15|> tokens (experimental)
    bool collect_alignment; // collect decoder cross-attention for native token timestamps
                            // (default: true; disable for text-only inference)
    // Output verbosity:
    //   0 = silent  — only hard errors (failed/cannot) go to stderr
    //   1 = normal  — model loading info printed (default)
    //   2 = verbose — per-inference timing, per-step tokens, performance report
    int verbosity;
};

struct cohere_context_params cohere_context_default_params(void);

// Load model from GGUF file produced by export_gguf.py
struct cohere_context* cohere_init_from_file(const char* path_model, struct cohere_context_params params);

void cohere_free(struct cohere_context* ctx);

// Cooperative, context-local inference cancellation. The callback may run on
// ggml worker threads and therefore must be thread-safe, non-blocking, and must
// not call back into Cohere/ggml. CPU and Metal graph execution use their
// native abort hooks; every backend is also checked between bounded inference
// stages and autoregressive decoder steps. Passing NULL removes the callback.
typedef bool (*cohere_abort_callback)(void* user_data);
void cohere_set_abort_callback(struct cohere_context* ctx, cohere_abort_callback callback, void* user_data);

// Report currently available and total memory for the selected ggml backend
// device. Values are zero when a backend cannot provide memory telemetry.
void cohere_backend_memory(struct cohere_context* ctx, size_t* free_bytes, size_t* total_bytes);

// Name of the ggml backend that owns the model weights (for example CPU,
// CUDA0, or Metal). The returned pointer is owned by ggml and remains valid
// for the lifetime of the context. This lets embedding runtimes report and
// enforce the device that was actually selected instead of trusting a request
// that may have fallen back during backend initialization.
const char* cohere_backend_name(struct cohere_context* ctx);

// Transcribe raw 16 kHz mono PCM.
// Returns a newly allocated UTF-8 string (caller must free()).
// lang: ISO-639-1 code e.g. "en", "fr", "de" (NULL → autodetect, not implemented yet)
char* cohere_transcribe(struct cohere_context* ctx, const float* samples, int n_samples, const char* lang);

// Vocabulary helpers
int cohere_n_vocab(struct cohere_context* ctx);
const char* cohere_token_to_str(struct cohere_context* ctx, int token_id);
int cohere_str_to_token(struct cohere_context* ctx, const char* str);

// Sampling: temperature > 0 enables stable softmax sampling in the
// transformer decoder. Default 0 keeps the bit-identical greedy path.
// Sticky on the context until the next call.
void cohere_set_temperature(struct cohere_context* ctx, float temperature, uint64_t seed);
void cohere_set_max_new_tokens(struct cohere_context* ctx, int max_new_tokens);
void cohere_set_frequency_penalty(struct cohere_context* ctx, float frequency_penalty);
// Stop greedy generation when the generated suffix contains four immediately
// repeated 8..32-token blocks after at least 96 generated tokens. This matches
// the conservative Transformers stopping criterion used by cohere-transcribe.
void cohere_set_repetition_loop_guard(struct cohere_context* ctx, bool enabled);

// Enable or disable native token-timestamp alignment. When disabled, the
// decoder skips the extra cross-attention graph output and device-to-host
// copies; returned token times fall back to linear interpolation. Sticky on
// the context until changed. Enabled by default.
void cohere_set_alignment_collection(struct cohere_context* ctx, bool enabled);

// §90 beam-search width. n > 1 activates beam search; n <= 0 clamped to 1 (greedy).
void cohere_set_beam_size(struct cohere_context* ctx, int n);

// ---- Extended API: per-token confidence and timing ----

// Per-token data returned by cohere_transcribe_ex().
struct cohere_token_data {
    int id;        // vocabulary token ID
    char text[64]; // decoded text (SentencePiece '▁' already converted to ' ')
    float p;       // softmax probability [0, 1]
    int64_t t0;    // start time, centiseconds (absolute, includes t_offset_cs)
    int64_t t1;    // end time, centiseconds
};

// Result from cohere_transcribe_ex() — free with cohere_result_free().
struct cohere_result {
    char* text;                       // full transcript (malloc'd)
    struct cohere_token_data* tokens; // per-token data (malloc'd)
    int n_tokens;
    // Decoder-level metadata. n_tokens counts rendered tokens and can be lower
    // than generated_tokens when special or byte-fallback pieces are consumed.
    int generated_tokens;
    int generation_limit;
    int generation_capacity;
    bool stopped_by_max_tokens;
    bool repetition_stopped;
};

void cohere_result_free(struct cohere_result* r);

// Like cohere_transcribe() but also returns per-token probability and timing.
//
// t_offset_cs: absolute start time of this audio slice, in centiseconds.
//   Token t0/t1 values equal (t_offset_cs + interpolated_offset_within_segment).
//   Pass 0 when processing a single file without VAD segmentation.
//   With VAD, pass (vad_segment_t0_seconds * 100).
//
// Token times are linearly interpolated across the segment duration,
// proportional to each token's decoded text length (best approximation
// without model-native timestamp tokens).
//
// Returns NULL on failure. Free result with cohere_result_free().
struct cohere_result* cohere_transcribe_ex(struct cohere_context* ctx, const float* samples, int n_samples,
                                           const char* lang, int64_t t_offset_cs);

// ---- Stage-level entry points (for crispasr-diff testing) ----
// Returns malloc'd F32 buffers the caller must free(). NULL on failure.

// Log-mel spectrogram of raw 16 kHz mono PCM, row-major (n_mels, T_mel).
// Applies deterministic length-seeded dither, pre-emphasis (0.97), and
// valid-frame-only per-feature log-mel normalization exactly as the live path.
float* cohere_compute_mel(struct cohere_context* ctx, const float* samples, int n_samples, int* out_n_mels,
                          int* out_T_mel);

// Run just the audio encoder on a mel spectrogram. Takes (n_mels, T_mel)
// row-major mel as produced by cohere_compute_mel() and returns the
// encoder hidden state in row-major (T_enc, d_model) where T_enc is the
// mel frame count after the 8x conv subsampling.
float* cohere_run_encoder(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, int* out_T_enc,
                          int* out_d_model);

// Timing and scheduler-memory measurements from cohere_run_encoder_batch().
// Times cover only the named stage and exclude model loading / mel creation.
struct cohere_encoder_batch_stats {
    int64_t graph_build_us;
    int64_t graph_alloc_us;
    int64_t input_us;
    int64_t compute_us;
    int64_t readback_us;
    size_t scheduler_bytes;
};

enum cohere_encoder_batch_precision {
    // Fold [in,T,B] to [in,T*B] so CUDA uses the existing F32 SGEMM path.
    COHERE_ENCODER_BATCH_PRECISE = 0,
    // Preserve native 3D batching and the fastest available tensor-core path.
    COHERE_ENCODER_BATCH_FAST = 1,
    // Native 3D batching with GGML_PREC_F32 accumulation/output.
    COHERE_ENCODER_BATCH_F32_ACCUM = 2,
};

// Run one encoder graph over an equal-length batch. Input and output are
// contiguous, lane-major row-major buffers:
//   mel:     [n_batch, T_mel, n_mels]
//   return:  [n_batch, T_enc, d_model]
//
// No padding mask is applied, so every lane must have exactly T_mel frames.
// Unlike the production encoder path, this test/runtime primitive does not
// compute decoder cross-KV tensors. The returned F32 buffer must be free()'d.
// This compatibility entry point uses COHERE_ENCODER_BATCH_PRECISE.
float* cohere_run_encoder_batch(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                int* out_T_enc, int* out_d_model, struct cohere_encoder_batch_stats* out_stats);
float* cohere_run_encoder_batch_ex(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                   enum cohere_encoder_batch_precision precision, int* out_T_enc, int* out_d_model,
                                   struct cohere_encoder_batch_stats* out_stats);

// Variable-length research batch. mel is already-normalized lane-major
// [n_batch, T_mel_max, n_mels]; valid_T_mel supplies each unpadded length.
// Returns zero-padded [n_batch, T_enc_max, 1024] and writes n_batch valid
// encoder lengths to out_valid_T_enc. Uses FAST projections.
float* cohere_run_encoder_batch_padded(struct cohere_context* ctx, const float* mel, const int* valid_T_mel,
                                       int n_batch, int n_mels, int T_mel_max, int* out_valid_T_enc,
                                       int* out_T_enc_max, int* out_d_model,
                                       struct cohere_encoder_batch_stats* out_stats);
typedef void (*cohere_padded_stage_cb)(const char* name, const float* data, int ne0, int ne1, int ne2, int ne3,
                                       void* userdata);
// CPU-oriented localization hook. Snapshot tensors retain large intermediate
// activations and are not part of the production batching path.
int cohere_run_encoder_batch_padded_staged(struct cohere_context* ctx, const float* mel, const int* valid_T_mel,
                                           int n_batch, int n_mels, int T_mel_max, cohere_padded_stage_cb cb,
                                           void* userdata);

// Diagnostic variant used to localize batch-vs-B1 divergence. Each callback
// receives one packed lane-major F32 snapshot. n_values_per_lane describes a
// flattened lane because convolution snapshots are 4D while encoder states
// are [T, d_model]. Returns 0 on success.
typedef void (*cohere_batch_stage_cb)(const char* name, const float* data, int n_batch, int n_values_per_lane,
                                      void* userdata);
int cohere_run_encoder_batch_staged(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels, int T_mel,
                                    cohere_batch_stage_cb cb, void* userdata);
int cohere_run_encoder_batch_staged_ex(struct cohere_context* ctx, const float* mel, int n_batch, int n_mels,
                                       int T_mel, enum cohere_encoder_batch_precision precision,
                                       cohere_batch_stage_cb cb, void* userdata);

// Dump unallocated production-B1 and batched-B1 graph metadata as TSV. Each
// row contains the op, type, full shape, full byte strides, and tensor name.
// This is a graph-construction diagnostic and performs no model computation.
int cohere_debug_dump_encoder_b1_graphs(struct cohere_context* ctx, int T_mel, const char* production_path,
                                        const char* batch_path);

// Bounded research API for validating the independent batched decoder path
// (B <= 24, T_enc <= 512, n_teacher_steps <= 64).
// encoder_states is lane-major [n_batch, T_enc, 1024]. teacher_tokens is
// step-major [n_teacher_steps, n_batch]. Output 0 is the last Arabic-prompt
// logit row; output i+1 follows teacher step i.
struct cohere_decoder_batch_probe_result {
    float* logits;   // [n_outputs, n_batch, vocab_size]
    int* argmax_ids; // [n_outputs, n_batch]
    int n_outputs;
    int n_batch;
    int vocab_size;
    size_t self_cache_bytes;
    size_t cross_cache_bytes;
};

struct cohere_decoder_batch_probe_result* cohere_run_decoder_batch_probe(
    struct cohere_context* ctx, const float* encoder_states, int n_batch, int T_enc, int d_model,
    const int* teacher_tokens, int n_teacher_steps);
// Variable-length diagnostic variant. encoder_states is padded lane-major
// [n_batch, T_enc_max, 1024], and valid_T_enc supplies each lane's true length.
// Every decoder cross-attention layer masks padded keys with a finite F16 bias.
struct cohere_decoder_batch_probe_result* cohere_run_decoder_batch_masked_probe(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const int* teacher_tokens, int n_teacher_steps);
void cohere_decoder_batch_probe_result_free(struct cohere_decoder_batch_probe_result* result);

enum cohere_generation_stop_reason {
    COHERE_GENERATION_STOP_EOS = 0,
    COHERE_GENERATION_STOP_MAX_TOKENS = 1,
};

struct cohere_generated_sequence {
    int* token_ids;
    // Device-argmax generation returns 1.0 placeholders because full logits
    // are intentionally not copied to the host in that mode.
    float* probabilities;
    int n_tokens;
    enum cohere_generation_stop_reason stop_reason;
};

enum cohere_decoder_batch_output_mode {
    COHERE_DECODER_BATCH_OUTPUT_FULL_LOGITS = 0,
    COHERE_DECODER_BATCH_OUTPUT_DEVICE_ARGMAX = 1,
};

struct cohere_decoder_batch_generate_stats {
    int decode_calls;
    int generation_steps;
    size_t logits_readback_bytes;
    size_t self_cache_bytes;
    size_t cross_cache_bytes;
    int64_t cross_kv_us;
    int64_t reserve_us;
    int64_t decode_us;
    int64_t total_us;
    size_t token_id_readback_bytes;
    enum cohere_decoder_batch_output_mode output_mode;
    // True when COHERE_BATCH_DECODER_PERSISTENT selected an opt-in
    // sched-free single-token replay path.
    bool persistent_graph;
    int persistent_graph_count;
    int persistent_decode_calls;
    int64_t persistent_init_us;
    int64_t persistent_decode_us;
};

struct cohere_decoder_batch_generate_result {
    struct cohere_generated_sequence* sequences;
    int n_batch;
    int generation_limit;
    int generation_capacity;
    struct cohere_decoder_batch_generate_stats stats;
};

// Reliable text-only greedy generation over a padded encoder batch. This
// compatibility-first path reads full logits back to the host and uses the
// same ordered double-precision softmax as the production B1 decoder.
// Alignment, sampling, frequency penalty, and beam search are rejected.
struct cohere_decoder_batch_generate_result* cohere_run_decoder_batch_generate(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const char* lang, int max_new_tokens);
// Optional device-argmax variant. FULL_LOGITS preserves the exact API above.
// DEVICE_ARGMAX copies only one I32 token ID per physical lane and reports
// probability placeholders of 1.0. Exact logit ties can be backend-dependent
// in DEVICE_ARGMAX; FULL_LOGITS uses the lowest vocabulary ID. Set
// COHERE_BATCH_DECODER_PERSISTENT=1 to test one fixed-context T=1 graph, or
// =progressive for lazy 64/128/256/final-context graphs. The research variants
// =progressive128 and =progressive256 skip smaller buckets. Unsupported graphs
// and allocation failures fall back to the dynamic path.
struct cohere_decoder_batch_generate_result* cohere_run_decoder_batch_generate_ex(
    struct cohere_context* ctx, const float* encoder_states, const int* valid_T_enc, int n_batch, int T_enc_max,
    int d_model, const char* lang, int max_new_tokens, enum cohere_decoder_batch_output_mode output_mode);
void cohere_decoder_batch_generate_result_free(struct cohere_decoder_batch_generate_result* result);

// Staged encoder: runs the encoder with per-layer snapshots for crispasr-diff.
// Callback receives each snapshot: name, data, T_enc, d_model.
typedef void (*cohere_stage_cb)(const char* name, const float* data, int T_enc, int d_model, void* userdata);
int cohere_run_encoder_staged(struct cohere_context* ctx, const float* mel, int n_mels, int T_mel, cohere_stage_cb cb,
                              void* userdata);

#ifdef __cplusplus
}
#endif
