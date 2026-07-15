// Subprocess-free audio decoding through FFmpeg's public shared-library ABI.
//
// The source gem intentionally does not require FFmpeg development headers:
// Linux distributions commonly install the runtime libraries with the ffmpeg
// executable but omit the -dev packages.  This adapter uses only stable public
// functions and the small, documented prefixes of AVFormatContext, AVPacket,
// AVFrame, AVStream, and AVCodecParameters needed for decoding.  Public field
// offsets that changed at an FFmpeg major boundary are selected only after all
// four runtime library versions have been verified as a compatible tuple.

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
#define COHERE_AUDIO_EXPORT extern "C" __declspec(dllexport)
#else
#define COHERE_AUDIO_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

struct AVFormatContext;
struct AVIOContext;
struct AVCodecContext;
struct AVCodec;
struct AVCodecParameters;
struct AVPacket;
struct AVFrame;
struct SwrContext;

struct AVIOInterruptCB {
    int (*callback)(void*);
    void* opaque;
};

struct AVChannelLayout {
    int order;
    int nb_channels;
    union {
        std::uint64_t mask;
        void* map;
    } details;
    void* opaque;
};

// This prefix is unchanged across the supported libavformat major versions.
struct FormatContextPrefix {
    const void* av_class;
    const void* input_format;
    const void* output_format;
    void* private_data;
    void* io_context;
    int context_flags;
    unsigned int stream_count;
    void** streams;
};

// The fields through stream_index retain these offsets in FFmpeg 4 through 8.
struct PacketPrefix {
    void* buffer;
    std::int64_t pts;
    std::int64_t dts;
    std::uint8_t* data;
    int size;
    int stream_index;
};

// The fields through format retain these offsets in FFmpeg 4 through 8.
struct FramePrefix {
    std::uint8_t* data[8];
    int line_size[8];
    std::uint8_t** extended_data;
    int width;
    int height;
    int sample_count;
    int format;
};

struct Rational {
    int numerator;
    int denominator;
};

static_assert(sizeof(void*) == 8, "cohere_audio currently targets 64-bit Ruby runtimes");
static_assert(offsetof(FormatContextPrefix, streams) == 48);
static_assert(offsetof(PacketPrefix, stream_index) == 36);
static_assert(offsetof(FramePrefix, extended_data) == 96);
static_assert(offsetof(FramePrefix, sample_count) == 112);
static_assert(offsetof(FramePrefix, format) == 116);
static_assert(sizeof(AVChannelLayout) == 24);
static_assert(sizeof(Rational) == 8);

constexpr int kAudioMediaType = 1;
constexpr int kFloatSampleFormat = 3;
constexpr int kAvioFlagRead = 1;
constexpr std::int64_t kMonoChannelMask = 0x4;
constexpr int kAvErrorEof = -541478725;
constexpr int kCancelledStatus = 6;
constexpr int kTimeoutStatus = 7;
constexpr std::uint64_t kMaximumScratchSamples = 16ULL * 1024ULL * 1024ULL;
constexpr std::chrono::seconds kDecodeTimeout(3600);
constexpr std::chrono::seconds kDurationProbeTimeout(30);

std::atomic<std::uint64_t> cancellation_generation{0};

class OperationDeadline {
public:
    OperationDeadline(std::chrono::seconds timeout, const char* operation)
        : generation_(cancellation_generation.load(std::memory_order_acquire)),
          deadline_(std::chrono::steady_clock::now() + timeout),
          timeout_(timeout),
          operation_(operation) {}

    int check(std::string& error) const {
        if (cancellation_generation.load(std::memory_order_acquire) != generation_) {
            error = std::string(operation_) + " was cancelled";
            return kCancelledStatus;
        }
        if (std::chrono::steady_clock::now() >= deadline_) {
            error = std::string(operation_) + " exceeded the " +
                std::to_string(timeout_.count()) + "s timeout";
            return kTimeoutStatus;
        }
        return 0;
    }

    bool interrupted() const noexcept {
        return cancellation_generation.load(std::memory_order_acquire) != generation_ ||
            std::chrono::steady_clock::now() >= deadline_;
    }

private:
    std::uint64_t generation_;
    std::chrono::steady_clock::time_point deadline_;
    std::chrono::seconds timeout_;
    const char* operation_;
};

using AvformatVersion = unsigned int (*)();
using AvcodecVersion = unsigned int (*)();
using AvutilVersion = unsigned int (*)();
using SwresampleVersion = unsigned int (*)();
using AvformatAllocContext = AVFormatContext* (*)();
using AvformatFreeContext = void (*)(AVFormatContext*);
using AvformatOpenInput = int (*)(AVFormatContext**, const char*, const void*, void**);
using AvformatFindStreamInfo = int (*)(AVFormatContext*, void**);
using AvReadFrame = int (*)(AVFormatContext*, AVPacket*);
using AvformatCloseInput = void (*)(AVFormatContext**);
using AvioOpen2 = int (*)(AVIOContext**, const char*, int, const AVIOInterruptCB*, void**);
using AvioClosep = int (*)(AVIOContext**);
using AvcodecFindDecoder = const AVCodec* (*)(int);
using AvcodecAllocContext3 = AVCodecContext* (*)(const AVCodec*);
using AvcodecParametersToContext = int (*)(AVCodecContext*, const AVCodecParameters*);
using AvcodecOpen2 = int (*)(AVCodecContext*, const AVCodec*, void**);
using AvcodecSendPacket = int (*)(AVCodecContext*, const AVPacket*);
using AvcodecReceiveFrame = int (*)(AVCodecContext*, AVFrame*);
using AvcodecFreeContext = void (*)(AVCodecContext**);
using AvPacketAlloc = AVPacket* (*)();
using AvPacketUnref = void (*)(AVPacket*);
using AvPacketFree = void (*)(AVPacket**);
using AvFrameAlloc = AVFrame* (*)();
using AvFrameUnref = void (*)(AVFrame*);
using AvFrameFree = void (*)(AVFrame**);
using AvStrerror = int (*)(int, char*, std::size_t);
using AvLogGetLevel = int (*)();
using AvLogSetLevel = void (*)(int);
using AvChannelLayoutDefault = void (*)(AVChannelLayout*, int);
using AvChannelLayoutCheck = int (*)(const AVChannelLayout*);
using AvChannelLayoutCompare = int (*)(const AVChannelLayout*, const AVChannelLayout*);
using AvChannelLayoutCopy = int (*)(AVChannelLayout*, const AVChannelLayout*);
using AvChannelLayoutUninit = void (*)(AVChannelLayout*);
using AvGetDefaultChannelLayout = std::int64_t (*)(int);
using SwrAllocSetOpts2 = int (*)(SwrContext**, const AVChannelLayout*, int, int,
                                const AVChannelLayout*, int, int, int, void*);
using SwrAllocSetOpts = SwrContext* (*)(SwrContext*, std::int64_t, int, int,
                                      std::int64_t, int, int, int, void*);
using SwrInit = int (*)(SwrContext*);
using SwrConvert = int (*)(SwrContext*, std::uint8_t**, int,
                          const std::uint8_t**, int);
using SwrGetDelay = std::int64_t (*)(SwrContext*, std::int64_t);
using SwrFree = void (*)(SwrContext**);

struct LibraryTuple {
    int format_major;
    int codec_major;
    int util_major;
    int resample_major;
    const char* format_name;
    const char* codec_name;
    const char* util_name;
    const char* resample_name;
};

#if defined(__APPLE__)
constexpr LibraryTuple kLibraryTuples[] = {
    {62, 62, 60, 6, "libavformat.62.dylib", "libavcodec.62.dylib", "libavutil.60.dylib", "libswresample.6.dylib"},
    {61, 61, 59, 5, "libavformat.61.dylib", "libavcodec.61.dylib", "libavutil.59.dylib", "libswresample.5.dylib"},
    {60, 60, 58, 4, "libavformat.60.dylib", "libavcodec.60.dylib", "libavutil.58.dylib", "libswresample.4.dylib"},
    {59, 59, 57, 4, "libavformat.59.dylib", "libavcodec.59.dylib", "libavutil.57.dylib", "libswresample.4.dylib"},
    {58, 58, 56, 3, "libavformat.58.dylib", "libavcodec.58.dylib", "libavutil.56.dylib", "libswresample.3.dylib"},
    {0, 0, 0, 0, "libavformat.dylib", "libavcodec.dylib", "libavutil.dylib", "libswresample.dylib"},
};
#else
constexpr LibraryTuple kLibraryTuples[] = {
    {62, 62, 60, 6, "libavformat.so.62", "libavcodec.so.62", "libavutil.so.60", "libswresample.so.6"},
    {61, 61, 59, 5, "libavformat.so.61", "libavcodec.so.61", "libavutil.so.59", "libswresample.so.5"},
    {60, 60, 58, 4, "libavformat.so.60", "libavcodec.so.60", "libavutil.so.58", "libswresample.so.4"},
    {59, 59, 57, 4, "libavformat.so.59", "libavcodec.so.59", "libavutil.so.57", "libswresample.so.4"},
    {58, 58, 56, 3, "libavformat.so.58", "libavcodec.so.58", "libavutil.so.56", "libswresample.so.3"},
    {0, 0, 0, 0, "libavformat.so", "libavcodec.so", "libavutil.so", "libswresample.so"},
};
#endif

void set_message(char* destination, std::size_t capacity, const std::string& message) {
    if (!destination || capacity == 0)
        return;
    const std::size_t length = std::min(capacity - 1, message.size());
    if (length > 0)
        std::memcpy(destination, message.data(), length);
    destination[length] = '\0';
}

unsigned int major_version(unsigned int packed) {
    return packed >> 16;
}

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle open_library(const char* name) {
    return LoadLibraryA(name);
}

void close_library(LibraryHandle handle) {
    if (handle)
        FreeLibrary(handle);
}

void* load_symbol(LibraryHandle handle, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(handle, name));
}

#else
using LibraryHandle = void*;

LibraryHandle open_library(const char* name) {
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

void close_library(LibraryHandle handle) {
    if (handle)
        dlclose(handle);
}

void* load_symbol(LibraryHandle handle, const char* name) {
    dlerror();
    return dlsym(handle, name);
}

#endif

template <typename Function>
Function symbol(LibraryHandle handle, const char* name) {
    return reinterpret_cast<Function>(load_symbol(handle, name));
}

class Runtime {
public:
    Runtime() {
        load();
    }

    ~Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    bool available = false;
    std::string diagnostic;
    int format_major = 0;
    int codec_major = 0;
    int util_major = 0;
    int resample_major = 0;

    AvformatOpenInput avformat_open_input = nullptr;
    AvformatAllocContext avformat_alloc_context = nullptr;
    AvformatFreeContext avformat_free_context = nullptr;
    AvformatFindStreamInfo avformat_find_stream_info = nullptr;
    AvReadFrame av_read_frame = nullptr;
    AvformatCloseInput avformat_close_input = nullptr;
    AvioOpen2 avio_open2 = nullptr;
    AvioClosep avio_closep = nullptr;
    AvcodecFindDecoder avcodec_find_decoder = nullptr;
    AvcodecAllocContext3 avcodec_alloc_context3 = nullptr;
    AvcodecParametersToContext avcodec_parameters_to_context = nullptr;
    AvcodecOpen2 avcodec_open2 = nullptr;
    AvcodecSendPacket avcodec_send_packet = nullptr;
    AvcodecReceiveFrame avcodec_receive_frame = nullptr;
    AvcodecFreeContext avcodec_free_context = nullptr;
    AvPacketAlloc av_packet_alloc = nullptr;
    AvPacketUnref av_packet_unref = nullptr;
    AvPacketFree av_packet_free = nullptr;
    AvFrameAlloc av_frame_alloc = nullptr;
    AvFrameUnref av_frame_unref = nullptr;
    AvFrameFree av_frame_free = nullptr;
    AvStrerror av_strerror = nullptr;
    AvLogGetLevel av_log_get_level = nullptr;
    AvLogSetLevel av_log_set_level = nullptr;
    AvChannelLayoutDefault av_channel_layout_default = nullptr;
    AvChannelLayoutCheck av_channel_layout_check = nullptr;
    AvChannelLayoutCompare av_channel_layout_compare = nullptr;
    AvChannelLayoutCopy av_channel_layout_copy = nullptr;
    AvChannelLayoutUninit av_channel_layout_uninit = nullptr;
    AvGetDefaultChannelLayout av_get_default_channel_layout = nullptr;
    SwrAllocSetOpts2 swr_alloc_set_opts2 = nullptr;
    SwrAllocSetOpts swr_alloc_set_opts = nullptr;
    SwrInit swr_init = nullptr;
    SwrConvert swr_convert = nullptr;
    SwrGetDelay swr_get_delay = nullptr;
    SwrFree swr_free = nullptr;

    std::string error_text(int code) const {
        char buffer[256] = {};
        if (av_strerror && av_strerror(code, buffer, sizeof(buffer)) == 0)
            return buffer;
        return "FFmpeg error " + std::to_string(code);
    }

private:
    std::vector<LibraryHandle> handles_;

    bool supported_versions(int format, int codec, int util, int resample) const {
        if (format < 58 || format > 62 || codec != format)
            return false;
        const int expected_util = format - 2;
        const int expected_resample = format == 58 ? 3 : (format <= 60 ? 4 : format - 56);
        return util == expected_util && resample == expected_resample;
    }

    bool bind_required() {
#define BIND_FROM(member, handle, name)                                                     \
        do {                                                                                \
            member = symbol<decltype(member)>(handle, name);                                \
            if (!member) {                                                                  \
                diagnostic = std::string("FFmpeg runtime is missing ") + name;             \
                return false;                                                               \
            }                                                                               \
        } while (false)

        const auto format = handles_[0];
        const auto codec = handles_[1];
        const auto util = handles_[2];
        const auto resample = handles_[3];
        BIND_FROM(avformat_alloc_context, format, "avformat_alloc_context");
        BIND_FROM(avformat_free_context, format, "avformat_free_context");
        BIND_FROM(avformat_open_input, format, "avformat_open_input");
        BIND_FROM(avformat_find_stream_info, format, "avformat_find_stream_info");
        BIND_FROM(av_read_frame, format, "av_read_frame");
        BIND_FROM(avformat_close_input, format, "avformat_close_input");
        BIND_FROM(avio_open2, format, "avio_open2");
        BIND_FROM(avio_closep, format, "avio_closep");
        BIND_FROM(avcodec_find_decoder, codec, "avcodec_find_decoder");
        BIND_FROM(avcodec_alloc_context3, codec, "avcodec_alloc_context3");
        BIND_FROM(avcodec_parameters_to_context, codec, "avcodec_parameters_to_context");
        BIND_FROM(avcodec_open2, codec, "avcodec_open2");
        BIND_FROM(avcodec_send_packet, codec, "avcodec_send_packet");
        BIND_FROM(avcodec_receive_frame, codec, "avcodec_receive_frame");
        BIND_FROM(avcodec_free_context, codec, "avcodec_free_context");
        BIND_FROM(av_packet_alloc, codec, "av_packet_alloc");
        BIND_FROM(av_packet_unref, codec, "av_packet_unref");
        BIND_FROM(av_packet_free, codec, "av_packet_free");
        BIND_FROM(av_frame_alloc, util, "av_frame_alloc");
        BIND_FROM(av_frame_unref, util, "av_frame_unref");
        BIND_FROM(av_frame_free, util, "av_frame_free");
        BIND_FROM(av_strerror, util, "av_strerror");
        BIND_FROM(av_log_get_level, util, "av_log_get_level");
        BIND_FROM(av_log_set_level, util, "av_log_set_level");
        BIND_FROM(swr_init, resample, "swr_init");
        BIND_FROM(swr_convert, resample, "swr_convert");
        BIND_FROM(swr_get_delay, resample, "swr_get_delay");
        BIND_FROM(swr_free, resample, "swr_free");

        if (util_major >= 57) {
            BIND_FROM(av_channel_layout_default, util, "av_channel_layout_default");
            BIND_FROM(av_channel_layout_check, util, "av_channel_layout_check");
            BIND_FROM(av_channel_layout_compare, util, "av_channel_layout_compare");
            BIND_FROM(av_channel_layout_copy, util, "av_channel_layout_copy");
            BIND_FROM(av_channel_layout_uninit, util, "av_channel_layout_uninit");
            BIND_FROM(swr_alloc_set_opts2, resample, "swr_alloc_set_opts2");
        } else {
            BIND_FROM(av_get_default_channel_layout, util, "av_get_default_channel_layout");
            BIND_FROM(swr_alloc_set_opts, resample, "swr_alloc_set_opts");
        }
#undef BIND_FROM
        return true;
    }

    bool try_tuple(const LibraryTuple& tuple) {
        std::vector<LibraryHandle> opened;
        for (const char* name : {tuple.format_name, tuple.codec_name, tuple.util_name, tuple.resample_name}) {
            LibraryHandle handle = open_library(name);
            if (!handle) {
                for (auto iterator = opened.rbegin(); iterator != opened.rend(); ++iterator)
                    close_library(*iterator);
                return false;
            }
            opened.push_back(handle);
        }

        const auto format_version = symbol<AvformatVersion>(opened[0], "avformat_version");
        const auto codec_version = symbol<AvcodecVersion>(opened[1], "avcodec_version");
        const auto util_version = symbol<AvutilVersion>(opened[2], "avutil_version");
        const auto resample_version = symbol<SwresampleVersion>(opened[3], "swresample_version");
        if (!format_version || !codec_version || !util_version || !resample_version) {
            for (auto iterator = opened.rbegin(); iterator != opened.rend(); ++iterator)
                close_library(*iterator);
            return false;
        }

        const int actual_format = static_cast<int>(major_version(format_version()));
        const int actual_codec = static_cast<int>(major_version(codec_version()));
        const int actual_util = static_cast<int>(major_version(util_version()));
        const int actual_resample = static_cast<int>(major_version(resample_version()));
        const bool expected = tuple.format_major == 0 ||
            (actual_format == tuple.format_major && actual_codec == tuple.codec_major &&
             actual_util == tuple.util_major && actual_resample == tuple.resample_major);
        if (!expected || !supported_versions(actual_format, actual_codec, actual_util, actual_resample)) {
            for (auto iterator = opened.rbegin(); iterator != opened.rend(); ++iterator)
                close_library(*iterator);
            return false;
        }

        handles_ = std::move(opened);
        format_major = actual_format;
        codec_major = actual_codec;
        util_major = actual_util;
        resample_major = actual_resample;
        if (!bind_required()) {
            for (auto iterator = handles_.rbegin(); iterator != handles_.rend(); ++iterator)
                close_library(*iterator);
            handles_.clear();
            return false;
        }
        return true;
    }

    void load() {
        const char* explicit_format = std::getenv("COHERE_TRANSCRIBE_AVFORMAT_LIBRARY");
        const char* explicit_codec = std::getenv("COHERE_TRANSCRIBE_AVCODEC_LIBRARY");
        const char* explicit_util = std::getenv("COHERE_TRANSCRIBE_AVUTIL_LIBRARY");
        const char* explicit_resample = std::getenv("COHERE_TRANSCRIBE_SWRESAMPLE_LIBRARY");
        const bool any_explicit = explicit_format || explicit_codec || explicit_util || explicit_resample;
        if (any_explicit) {
            if (!explicit_format || !explicit_codec || !explicit_util || !explicit_resample) {
                diagnostic = "all four COHERE_TRANSCRIBE_AV* library overrides must be set together";
                return;
            }
            const LibraryTuple explicit_tuple = {
                0, 0, 0, 0, explicit_format, explicit_codec, explicit_util, explicit_resample
            };
            if (!try_tuple(explicit_tuple)) {
                diagnostic = "the explicit FFmpeg shared-library tuple is unavailable or incompatible";
                return;
            }
        } else {
            bool loaded = false;
            for (const auto& tuple : kLibraryTuples) {
                if (try_tuple(tuple)) {
                    loaded = true;
                    break;
                }
            }
            if (!loaded) {
                diagnostic = "no compatible FFmpeg 4-8 shared-library tuple was found";
                return;
            }
        }

        available = true;
        diagnostic = "FFmpeg ABI avformat " + std::to_string(format_major) +
            ", avcodec " + std::to_string(codec_major) +
            ", avutil " + std::to_string(util_major) +
            ", swresample " + std::to_string(resample_major);
    }
};

Runtime& runtime() {
    static Runtime instance;
    return instance;
}

// FFmpeg logging is process-global.  A reference-counted guard temporarily
// matches the reference CLI's captured stderr behavior without serializing
// parallel decodes or permanently changing the embedding application's level.
class ErrorLogGuard {
public:
    explicit ErrorLogGuard(Runtime& runtime) : api_(runtime) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_++ == 0) {
            saved_level_ = api_.av_log_get_level();
            if (saved_level_ > -8)
                api_.av_log_set_level(-8);
        }
    }

    ~ErrorLogGuard() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--active_ == 0 && saved_level_ > -8)
            api_.av_log_set_level(saved_level_);
    }

    ErrorLogGuard(const ErrorLogGuard&) = delete;
    ErrorLogGuard& operator=(const ErrorLogGuard&) = delete;

private:
    Runtime& api_;
    static std::mutex mutex_;
    static int active_;
    static int saved_level_;
};

std::mutex ErrorLogGuard::mutex_;
int ErrorLogGuard::active_ = 0;
int ErrorLogGuard::saved_level_ = -8;

std::size_t stream_parameters_offset(int format_major) {
    return format_major >= 60 ? 16 : 208;
}

std::size_t stream_time_base_offset(int format_major) {
    // The public AVStream layouts place time_base at byte 24 in 58-59 and
    // byte 32 after codecpar moved to the prefix in 60-62.
    return format_major >= 60 ? 32 : 24;
}

std::size_t format_start_time_offset(int format_major) {
    // AVFormatContext retained its public prefix but changed the fields between
    // streams and url at the FFmpeg 5 and 7 major boundaries.
    if (format_major == 58)
        return 1088; // legacy filename[1024], then url
    if (format_major <= 60)
        return 64;   // url
    return 96;       // stream groups, chapters, then url
}

std::size_t frame_sample_rate_offset(int util_major) {
    switch (util_major) {
    case 56: return 272;
    case 57:
    case 58: return 208;
    case 59: return 192;
    case 60: return 180;
    default: return 0;
    }
}

std::size_t frame_channel_layout_offset(int util_major) {
    switch (util_major) {
    case 57:
    case 58: return 448;
    case 59: return 408;
    case 60: return 384;
    default: return 0;
    }
}

template <typename Value>
Value field_at(const void* object, std::size_t offset) {
    Value value{};
    std::memcpy(&value, static_cast<const std::uint8_t*>(object) + offset, sizeof(value));
    return value;
}

AVCodecParameters* parameters_for_stream(Runtime& api, AVFormatContext* context, int stream_index) {
    const auto* prefix = reinterpret_cast<const FormatContextPrefix*>(context);
    if (!prefix || stream_index < 0 || static_cast<unsigned int>(stream_index) >= prefix->stream_count ||
        !prefix->streams || !prefix->streams[stream_index]) {
        return nullptr;
    }
    return field_at<AVCodecParameters*>(
        prefix->streams[stream_index], stream_parameters_offset(api.format_major));
}

int first_audio_stream(Runtime& api, AVFormatContext* context, AVCodecParameters** parameters) {
    const auto* prefix = reinterpret_cast<const FormatContextPrefix*>(context);
    if (!prefix || !prefix->streams || prefix->stream_count > 1'000'000)
        return -1;
    for (unsigned int index = 0; index < prefix->stream_count; ++index) {
        AVCodecParameters* candidate = parameters_for_stream(api, context, static_cast<int>(index));
        if (!candidate)
            continue;
        const int media_type = field_at<int>(candidate, 0);
        if (media_type == kAudioMediaType) {
            *parameters = candidate;
            return static_cast<int>(index);
        }
    }
    return -1;
}

double duration_for_stream(Runtime& api, AVFormatContext* context, int stream_index) {
    const auto* prefix = reinterpret_cast<const FormatContextPrefix*>(context);
    if (!prefix || !prefix->streams || stream_index < 0 ||
        static_cast<unsigned int>(stream_index) >= prefix->stream_count ||
        !prefix->streams[stream_index]) {
        return -1.0;
    }
    const void* stream = prefix->streams[stream_index];
    const std::size_t time_base_offset = stream_time_base_offset(api.format_major);
    const Rational time_base = field_at<Rational>(stream, time_base_offset);
    const std::int64_t raw_duration = field_at<std::int64_t>(stream, time_base_offset + 16);
    if (raw_duration < 0 || time_base.numerator <= 0 || time_base.denominator <= 0)
        return -1.0;
    const double seconds = static_cast<double>(raw_duration) *
        static_cast<double>(time_base.numerator) / static_cast<double>(time_base.denominator);
    return std::isfinite(seconds) && seconds >= 0.0 ? seconds : -1.0;
}

double duration_for_format(Runtime& api, AVFormatContext* context) {
    const std::size_t start_offset = format_start_time_offset(api.format_major);
    const std::int64_t raw_start = field_at<std::int64_t>(context, start_offset);
    const std::int64_t raw_duration = field_at<std::int64_t>(context, start_offset + 8);
    if (raw_duration < 0)
        return -1.0;

    double duration = static_cast<double>(raw_duration) / 1'000'000.0;
    const double start = raw_start < 0 ? -1.0 : static_cast<double>(raw_start) / 1'000'000.0;
    // Match the reference ffprobe fallback: some containers expose an
    // absolute end timestamp as format duration when start_time is positive.
    if (start > 0.0 && duration > start)
        duration -= start;
    return std::isfinite(duration) && duration >= 0.0 ? duration : -1.0;
}

int interrupt_io(void* opaque) noexcept {
    const auto* deadline = static_cast<const OperationDeadline*>(opaque);
    return deadline && deadline->interrupted() ? 1 : 0;
}

// Pre-open the input through avio_open2 so FFmpeg receives a stable public
// AVIOInterruptCB without writing the version-dependent interrupt_callback
// field deep inside AVFormatContext. avformat_open_input detects the supplied
// pb and marks it as custom I/O; consequently we retain and close it ourselves.
struct DemuxState {
    explicit DemuxState(Runtime& runtime) : api(runtime) {}

    ~DemuxState() {
        if (format_opened && format)
            api.avformat_close_input(&format);
        else if (format)
            api.avformat_free_context(format);
        if (io)
            api.avio_closep(&io);
    }

    DemuxState(const DemuxState&) = delete;
    DemuxState& operator=(const DemuxState&) = delete;

    Runtime& api;
    AVFormatContext* format = nullptr;
    AVIOContext* io = nullptr;
    bool format_opened = false;
};

int open_input(DemuxState& state, const char* path,
               const OperationDeadline& deadline, std::string& error) {
    int interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;

    state.format = state.api.avformat_alloc_context();
    if (!state.format) {
        error = "could not allocate the FFmpeg input context";
        return 5;
    }

    const AVIOInterruptCB interrupt_callback = {interrupt_io, const_cast<OperationDeadline*>(&deadline)};
    int result = state.api.avio_open2(
        &state.io, path, kAvioFlagRead, &interrupt_callback, nullptr);
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "FFmpeg could not open the input: " + state.api.error_text(result);
        return 3;
    }

    std::memcpy(
        static_cast<std::uint8_t*>(static_cast<void*>(state.format)) +
            offsetof(FormatContextPrefix, io_context),
        &state.io,
        sizeof(state.io));
    result = state.api.avformat_open_input(&state.format, path, nullptr, nullptr);
    state.format_opened = result >= 0;
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "FFmpeg could not open the input: " + state.api.error_text(result);
        return 3;
    }
    return 0;
}

using DurationProbeState = DemuxState;

int probe_duration_file(const char* path, double* output_duration, std::string& error) {
    Runtime& api = runtime();
    if (!api.available) {
        error = api.diagnostic;
        return 1;
    }
    if (!path || path[0] == '\0' || !output_duration) {
        error = "invalid native FFmpeg duration-probe arguments";
        return 2;
    }
    *output_duration = -1.0;

    OperationDeadline deadline(kDurationProbeTimeout, "FFmpeg duration probe");
    ErrorLogGuard log_guard(api);
    DurationProbeState state(api);
    int interrupted = open_input(state, path, deadline, error);
    if (interrupted != 0)
        return interrupted;
    const int result = api.avformat_find_stream_info(state.format, nullptr);
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "FFmpeg could not inspect the input streams: " + api.error_text(result);
        return 3;
    }

    AVCodecParameters* parameters = nullptr;
    const int stream_index = first_audio_stream(api, state.format, &parameters);
    if (stream_index < 0 || !parameters) {
        error = "FFmpeg input has no audio stream";
        return 3;
    }
    double duration = duration_for_stream(api, state.format, stream_index);
    if (duration < 0.0)
        duration = duration_for_format(api, state.format);
    *output_duration = duration;
    return 0;
}

class OutputBuffer {
public:
    explicit OutputBuffer(std::uint64_t maximum_bytes)
        : limited_(maximum_bytes != 0), maximum_samples_(maximum_bytes / sizeof(float)) {}

    ~OutputBuffer() {
        std::free(data_);
    }

    OutputBuffer(const OutputBuffer&) = delete;
    OutputBuffer& operator=(const OutputBuffer&) = delete;

    int append(const float* samples, std::uint64_t count, std::string& error) {
        if (count == 0)
            return 0;
        if (!samples || count > std::numeric_limits<std::uint64_t>::max() - size_) {
            error = "decoded audio sample count overflowed";
            return 3;
        }
        const std::uint64_t wanted = size_ + count;
        if (limited_ && wanted > maximum_samples_) {
            error = "FFmpeg output exceeded the configured decoded-audio memory limit";
            return 4;
        }
        if (wanted > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            error = "decoded audio is too large for this process";
            return 5;
        }
        if (wanted > capacity_) {
            const std::uint64_t maximum_capacity =
                std::numeric_limits<std::size_t>::max() / sizeof(float);
            std::uint64_t next = capacity_ == 0 ? 4096 : capacity_;
            while (next < wanted && next <= maximum_capacity / 2)
                next *= 2;
            if (next < wanted)
                next = wanted;
            if (limited_)
                next = std::min(next, maximum_samples_);
            next = std::min(next, maximum_capacity);
            void* replacement = std::realloc(data_, static_cast<std::size_t>(next) * sizeof(float));
            if (!replacement) {
                error = "could not allocate decoded audio";
                return 5;
            }
            data_ = static_cast<float*>(replacement);
            capacity_ = next;
        }
        std::memcpy(data_ + size_, samples, static_cast<std::size_t>(count) * sizeof(float));
        size_ = wanted;
        return 0;
    }

    float* release() {
        float* result = data_;
        data_ = nullptr;
        capacity_ = 0;
        size_ = 0;
        return result;
    }

    std::uint64_t size() const {
        return size_;
    }

private:
    float* data_ = nullptr;
    std::uint64_t size_ = 0;
    std::uint64_t capacity_ = 0;
    bool limited_ = false;
    std::uint64_t maximum_samples_ = 0;
};

struct DecodeState : DemuxState {
    explicit DecodeState(Runtime& runtime) : DemuxState(runtime) {}

    ~DecodeState() {
        if (input_layout_owned && api.av_channel_layout_uninit)
            api.av_channel_layout_uninit(&input_layout);
        if (resampler)
            api.swr_free(&resampler);
        if (frame)
            api.av_frame_free(&frame);
        if (packet)
            api.av_packet_free(&packet);
        if (codec)
            api.avcodec_free_context(&codec);
    }

    AVCodecContext* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwrContext* resampler = nullptr;
    int stream_index = -1;
    int target_rate = 0;
    int input_rate = 0;
    int input_format = -1;
    int input_channels = 0;
    std::int64_t old_input_layout = 0;
    AVChannelLayout input_layout{};
    bool input_layout_owned = false;
    const OperationDeadline* deadline = nullptr;
};

int frame_sample_rate(const DecodeState& state) {
    const std::size_t offset = frame_sample_rate_offset(state.api.util_major);
    return offset == 0 ? 0 : field_at<int>(state.frame, offset);
}

const AVChannelLayout* new_frame_layout(const DecodeState& state) {
    const std::size_t offset = frame_channel_layout_offset(state.api.util_major);
    if (offset == 0)
        return nullptr;
    return reinterpret_cast<const AVChannelLayout*>(
        static_cast<const std::uint8_t*>(static_cast<const void*>(state.frame)) + offset);
}

int initialize_resampler(DecodeState& state, std::string& error) {
    const auto* prefix = reinterpret_cast<const FramePrefix*>(state.frame);
    state.input_format = prefix->format;
    state.input_rate = frame_sample_rate(state);
    if (state.input_format < 0 || state.input_rate <= 0 || state.input_rate > 1'000'000) {
        error = "FFmpeg returned invalid decoded audio format metadata";
        return 3;
    }

    int result = 0;
    if (state.api.util_major >= 57) {
        const AVChannelLayout* source_layout = new_frame_layout(state);
        AVChannelLayout fallback{};
        bool fallback_initialized = false;
        if (!source_layout || state.api.av_channel_layout_check(source_layout) != 1 ||
            source_layout->nb_channels <= 0 || source_layout->nb_channels > 256) {
            const int channels = source_layout ? source_layout->nb_channels : 0;
            if (channels <= 0 || channels > 256) {
                error = "FFmpeg returned an invalid decoded channel layout";
                return 3;
            }
            state.api.av_channel_layout_default(&fallback, channels);
            source_layout = &fallback;
            fallback_initialized = true;
        }
        state.input_channels = source_layout->nb_channels;
        if (state.api.av_channel_layout_copy(&state.input_layout, source_layout) < 0) {
            if (fallback_initialized)
                state.api.av_channel_layout_uninit(&fallback);
            error = "could not retain the FFmpeg input channel layout";
            return 5;
        }
        state.input_layout_owned = true;

        AVChannelLayout output_layout{};
        state.api.av_channel_layout_default(&output_layout, 1);
        result = state.api.swr_alloc_set_opts2(
            &state.resampler,
            &output_layout,
            kFloatSampleFormat,
            state.target_rate,
            source_layout,
            state.input_format,
            state.input_rate,
            0,
            nullptr);
        state.api.av_channel_layout_uninit(&output_layout);
        if (fallback_initialized)
            state.api.av_channel_layout_uninit(&fallback);
    } else {
        state.old_input_layout = field_at<std::int64_t>(state.frame, 280);
        state.input_channels = field_at<int>(state.frame, 444);
        if (state.input_channels <= 0 || state.input_channels > 256) {
            error = "FFmpeg returned an invalid decoded channel count";
            return 3;
        }
        if (state.old_input_layout == 0)
            state.old_input_layout = state.api.av_get_default_channel_layout(state.input_channels);
        if (state.old_input_layout == 0) {
            error = "FFmpeg could not determine the decoded channel layout";
            return 3;
        }
        state.resampler = state.api.swr_alloc_set_opts(
            nullptr,
            kMonoChannelMask,
            kFloatSampleFormat,
            state.target_rate,
            state.old_input_layout,
            state.input_format,
            state.input_rate,
            0,
            nullptr);
        result = state.resampler ? 0 : -1;
    }

    if (result < 0 || !state.resampler) {
        error = "could not configure FFmpeg audio resampling";
        return 3;
    }
    result = state.api.swr_init(state.resampler);
    if (result < 0) {
        error = "could not initialize FFmpeg audio resampling: " + state.api.error_text(result);
        return 3;
    }
    return 0;
}

bool frame_matches_resampler(const DecodeState& state) {
    const auto* prefix = reinterpret_cast<const FramePrefix*>(state.frame);
    if (prefix->format != state.input_format || frame_sample_rate(state) != state.input_rate)
        return false;
    if (state.api.util_major >= 57) {
        const AVChannelLayout* layout = new_frame_layout(state);
        if (layout && state.api.av_channel_layout_check(layout) == 1)
            return state.api.av_channel_layout_compare(layout, &state.input_layout) == 0;
        return layout && layout->nb_channels == state.input_channels;
    }
    const int channels = field_at<int>(state.frame, 444);
    std::int64_t layout = field_at<std::int64_t>(state.frame, 280);
    if (layout == 0 && channels > 0)
        layout = state.api.av_get_default_channel_layout(channels);
    return channels == state.input_channels && layout == state.old_input_layout;
}

int flush_resampler(DecodeState& state, OutputBuffer& output, std::string& error);

int check_decode_deadline(const DecodeState& state, std::string& error) {
    return state.deadline ? state.deadline->check(error) : 0;
}

void release_resampler_configuration(DecodeState& state) {
    if (state.input_layout_owned) {
        state.api.av_channel_layout_uninit(&state.input_layout);
        state.input_layout = {};
        state.input_layout_owned = false;
    }
    if (state.resampler)
        state.api.swr_free(&state.resampler);
    state.input_rate = 0;
    state.input_format = -1;
    state.input_channels = 0;
    state.old_input_layout = 0;
}

std::uint64_t output_capacity(std::int64_t delayed, int input_samples,
                              int input_rate, int target_rate) {
    if (delayed < 0 || input_samples < 0 || input_rate <= 0 || target_rate <= 0)
        return 0;
    const auto total = static_cast<std::uint64_t>(delayed) +
        static_cast<unsigned int>(input_samples);
    const auto denominator = static_cast<unsigned int>(input_rate);
    const auto multiplier = static_cast<unsigned int>(target_rate);
    const std::uint64_t quotient = total / denominator;
    const std::uint64_t remainder = total % denominator;
    if (quotient > std::numeric_limits<std::uint64_t>::max() / multiplier)
        return std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t whole = quotient * multiplier;
    const std::uint64_t remainder_numerator = remainder * multiplier;
    const std::uint64_t fractional =
        (remainder_numerator + denominator - 1) / denominator;
    if (whole > std::numeric_limits<std::uint64_t>::max() - fractional)
        return std::numeric_limits<std::uint64_t>::max();
    return whole + fractional;
}

int append_decoded_frame(DecodeState& state, OutputBuffer& output, std::string& error) {
    int interrupted = check_decode_deadline(state, error);
    if (interrupted != 0)
        return interrupted;
    const auto* frame = reinterpret_cast<const FramePrefix*>(state.frame);
    if (frame->sample_count < 0 || !frame->extended_data) {
        error = "FFmpeg returned an invalid decoded audio frame";
        return 3;
    }
    if (!state.resampler) {
        const int status = initialize_resampler(state, error);
        if (status != 0)
            return status;
    } else if (!frame_matches_resampler(state)) {
        int status = flush_resampler(state, output, error);
        if (status != 0)
            return status;
        release_resampler_configuration(state);
        status = initialize_resampler(state, error);
        if (status != 0)
            return status;
    }

    const std::int64_t delayed = state.api.swr_get_delay(state.resampler, state.input_rate);
    const std::uint64_t capacity = output_capacity(
        delayed, frame->sample_count, state.input_rate, state.target_rate);
    if (capacity == 0 && frame->sample_count > 0) {
        error = "FFmpeg returned an invalid resampling delay";
        return 3;
    }
    if (capacity > kMaximumScratchSamples || capacity > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        error = "FFmpeg returned an unreasonably large decoded audio frame";
        return 3;
    }
    std::vector<float> scratch(static_cast<std::size_t>(capacity));
    std::uint8_t* planes[] = {
        capacity == 0 ? nullptr : reinterpret_cast<std::uint8_t*>(scratch.data())
    };
    const int converted = state.api.swr_convert(
        state.resampler,
        planes,
        static_cast<int>(capacity),
        const_cast<const std::uint8_t**>(frame->extended_data),
        frame->sample_count);
    interrupted = check_decode_deadline(state, error);
    if (interrupted != 0)
        return interrupted;
    if (converted < 0) {
        error = "FFmpeg audio resampling failed: " + state.api.error_text(converted);
        return 3;
    }
    return output.append(scratch.data(), static_cast<std::uint64_t>(converted), error);
}

int flush_resampler(DecodeState& state, OutputBuffer& output, std::string& error) {
    if (!state.resampler)
        return 0;
    for (int iteration = 0; iteration < 1024; ++iteration) {
        int interrupted = check_decode_deadline(state, error);
        if (interrupted != 0)
            return interrupted;
        const std::int64_t delayed = state.api.swr_get_delay(state.resampler, state.input_rate);
        if (delayed <= 0)
            return 0;
        const std::uint64_t capacity = output_capacity(
            delayed, 0, state.input_rate, state.target_rate);
        if (capacity == 0 || capacity > kMaximumScratchSamples ||
            capacity > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            error = "FFmpeg returned an invalid final resampling delay";
            return 3;
        }
        std::vector<float> scratch(static_cast<std::size_t>(capacity));
        std::uint8_t* planes[] = {reinterpret_cast<std::uint8_t*>(scratch.data())};
        const int converted = state.api.swr_convert(
            state.resampler, planes, static_cast<int>(capacity), nullptr, 0);
        interrupted = check_decode_deadline(state, error);
        if (interrupted != 0)
            return interrupted;
        if (converted < 0) {
            error = "FFmpeg final audio resampling failed: " + state.api.error_text(converted);
            return 3;
        }
        if (converted == 0)
            return 0;
        const int status = output.append(scratch.data(), static_cast<std::uint64_t>(converted), error);
        if (status != 0)
            return status;
    }
    error = "FFmpeg audio resampling did not finish";
    return 3;
}

int decode_file(const char* path, int target_rate, std::uint64_t maximum_bytes,
                float** output_samples, std::int64_t* output_count, std::string& error) {
    Runtime& api = runtime();
    if (!api.available) {
        error = api.diagnostic;
        return 1;
    }
    if (!path || path[0] == '\0' || !output_samples || !output_count ||
        target_rate <= 0 || target_rate > 1'000'000) {
        error = "invalid native FFmpeg decode arguments";
        return 2;
    }
    *output_samples = nullptr;
    *output_count = 0;

    OperationDeadline deadline(kDecodeTimeout, "FFmpeg audio decode");
    ErrorLogGuard log_guard(api);
    DecodeState state(api);
    state.target_rate = target_rate;
    state.deadline = &deadline;
    int interrupted = open_input(state, path, deadline, error);
    if (interrupted != 0)
        return interrupted;
    int result = api.avformat_find_stream_info(state.format, nullptr);
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "FFmpeg could not inspect the input streams: " + api.error_text(result);
        return 3;
    }

    AVCodecParameters* parameters = nullptr;
    state.stream_index = first_audio_stream(api, state.format, &parameters);
    if (state.stream_index < 0 || !parameters) {
        error = "FFmpeg input has no audio stream";
        return 3;
    }
    const int codec_identifier = field_at<int>(parameters, sizeof(int));
    const AVCodec* decoder = api.avcodec_find_decoder(codec_identifier);
    if (!decoder) {
        error = "FFmpeg has no decoder for the first audio stream";
        return 3;
    }
    state.codec = api.avcodec_alloc_context3(decoder);
    if (!state.codec) {
        error = "could not allocate the FFmpeg decoder context";
        return 5;
    }
    result = api.avcodec_parameters_to_context(state.codec, parameters);
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "could not configure the FFmpeg decoder: " + api.error_text(result);
        return 3;
    }
    result = api.avcodec_open2(state.codec, decoder, nullptr);
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result < 0) {
        error = "could not open the FFmpeg audio decoder: " + api.error_text(result);
        return 3;
    }
    state.packet = api.av_packet_alloc();
    state.frame = api.av_frame_alloc();
    if (!state.packet || !state.frame) {
        error = "could not allocate FFmpeg packet/frame state";
        return 5;
    }

    OutputBuffer output(maximum_bytes);
    auto receive_frames = [&]() -> int {
        while (true) {
            const int cancellation_status = deadline.check(error);
            if (cancellation_status != 0)
                return cancellation_status;
            const int receive_result = api.avcodec_receive_frame(state.codec, state.frame);
            const int post_receive_status = deadline.check(error);
            if (post_receive_status != 0) {
                if (receive_result >= 0)
                    api.av_frame_unref(state.frame);
                return post_receive_status;
            }
            if (receive_result == -EAGAIN || receive_result == kAvErrorEof)
                return 0;
            if (receive_result < 0) {
                error = "FFmpeg audio decoding failed: " + api.error_text(receive_result);
                return 3;
            }
            const int append_result = append_decoded_frame(state, output, error);
            api.av_frame_unref(state.frame);
            if (append_result != 0)
                return append_result;
        }
    };

    while ((result = api.av_read_frame(state.format, state.packet)) >= 0) {
        interrupted = deadline.check(error);
        if (interrupted != 0) {
            api.av_packet_unref(state.packet);
            return interrupted;
        }
        const auto* packet = reinterpret_cast<const PacketPrefix*>(state.packet);
        int packet_status = 0;
        if (packet->stream_index == state.stream_index) {
            while (true) {
                interrupted = deadline.check(error);
                if (interrupted != 0) {
                    packet_status = interrupted;
                    break;
                }
                const int send_result = api.avcodec_send_packet(state.codec, state.packet);
                interrupted = deadline.check(error);
                if (interrupted != 0) {
                    packet_status = interrupted;
                    break;
                }
                if (send_result == -EAGAIN) {
                    packet_status = receive_frames();
                    if (packet_status != 0)
                        break;
                    continue;
                }
                if (send_result < 0) {
                    error = "FFmpeg rejected an audio packet: " + api.error_text(send_result);
                    packet_status = 3;
                } else {
                    packet_status = receive_frames();
                }
                break;
            }
        }
        api.av_packet_unref(state.packet);
        if (packet_status != 0)
            return packet_status;
    }
    interrupted = deadline.check(error);
    if (interrupted != 0)
        return interrupted;
    if (result != kAvErrorEof) {
        error = "FFmpeg failed while reading the input: " + api.error_text(result);
        return 3;
    }

    while (true) {
        result = api.avcodec_send_packet(state.codec, nullptr);
        interrupted = deadline.check(error);
        if (interrupted != 0)
            return interrupted;
        if (result != -EAGAIN)
            break;
        result = receive_frames();
        if (result != 0)
            return result;
    }
    if (result < 0 && result != kAvErrorEof) {
        error = "FFmpeg could not flush the audio decoder: " + api.error_text(result);
        return 3;
    }
    result = receive_frames();
    if (result != 0)
        return result;
    result = flush_resampler(state, output, error);
    if (result != 0)
        return result;
    if (output.size() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        error = "decoded audio is too large for the Ruby ABI";
        return 5;
    }

    *output_count = static_cast<std::int64_t>(output.size());
    *output_samples = output.release();
    return 0;
}

} // namespace

COHERE_AUDIO_EXPORT int cohere_audio_ffmpeg_probe(char* diagnostic, std::size_t capacity) {
    try {
        Runtime& api = runtime();
        set_message(diagnostic, capacity, api.diagnostic);
        return api.available ? 0 : 1;
    } catch (const std::exception& exception) {
        set_message(diagnostic, capacity, exception.what());
        return 1;
    } catch (...) {
        set_message(diagnostic, capacity, "unknown error while loading the FFmpeg ABI");
        return 1;
    }
}

COHERE_AUDIO_EXPORT int cohere_audio_ffmpeg_decode(
    const char* path,
    int target_rate,
    std::uint64_t maximum_bytes,
    float** output_samples,
    std::int64_t* output_count,
    char* diagnostic,
    std::size_t capacity) {
    // Make the ownership contract deterministic on every return path,
    // including argument validation and runtime-loader failures.
    if (output_samples)
        *output_samples = nullptr;
    if (output_count)
        *output_count = 0;
    try {
        std::string error;
        const int result = decode_file(
            path, target_rate, maximum_bytes, output_samples, output_count, error);
        set_message(diagnostic, capacity, result == 0 ? runtime().diagnostic : error);
        return result;
    } catch (const std::bad_alloc&) {
        if (output_samples)
            *output_samples = nullptr;
        if (output_count)
            *output_count = 0;
        set_message(diagnostic, capacity, "could not allocate native audio decode memory");
        return 5;
    } catch (const std::exception& exception) {
        if (output_samples)
            *output_samples = nullptr;
        if (output_count)
            *output_count = 0;
        set_message(diagnostic, capacity, exception.what());
        return 3;
    } catch (...) {
        if (output_samples)
            *output_samples = nullptr;
        if (output_count)
            *output_count = 0;
        set_message(diagnostic, capacity, "unknown native FFmpeg decode failure");
        return 3;
    }
}

COHERE_AUDIO_EXPORT int cohere_audio_ffmpeg_duration(
    const char* path,
    double* output_duration,
    char* diagnostic,
    std::size_t capacity) {
    if (output_duration)
        *output_duration = -1.0;
    try {
        std::string error;
        const int result = probe_duration_file(path, output_duration, error);
        set_message(diagnostic, capacity, result == 0 ? runtime().diagnostic : error);
        return result;
    } catch (const std::exception& exception) {
        if (output_duration)
            *output_duration = -1.0;
        set_message(diagnostic, capacity, exception.what());
        return 3;
    } catch (...) {
        if (output_duration)
            *output_duration = -1.0;
        set_message(diagnostic, capacity, "unknown native FFmpeg duration-probe failure");
        return 3;
    }
}

COHERE_AUDIO_EXPORT void cohere_audio_ffmpeg_cancel() {
    cancellation_generation.fetch_add(1, std::memory_order_acq_rel);
}

COHERE_AUDIO_EXPORT void cohere_audio_ffmpeg_free(void* samples) {
    std::free(samples);
}
