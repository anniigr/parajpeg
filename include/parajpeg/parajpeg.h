// ParaJPEG — a parallel, bit-exact JPEG decoder with scalar, SIMD and CUDA backends.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parajpeg {

enum class StatusCode : int {
    Ok = 0,
    InvalidArgument,
    NotJpeg,          // missing SOI / not a JPEG stream
    Unsupported,      // valid JPEG feature we do not implement (arithmetic coding, 12-bit, lossless, hierarchical)
    Corrupt,          // malformed stream that cannot be recovered
    Truncated,        // stream ended early (output is best-effort, like libjpeg)
    BackendUnavailable,
    Internal,
};

struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message;
    bool ok() const { return code == StatusCode::Ok; }
    explicit operator bool() const { return ok(); }
    static Status success() { return {}; }
    static Status error(StatusCode c, std::string msg) { return {c, std::move(msg)}; }
};

enum class PixelFormat : uint8_t { RGB, BGR, RGBA, BGRA, Gray };

enum class Backend : uint8_t {
    Auto,       // best available: CUDA if compiled in and a device exists, else CPU
    Scalar,     // single-threaded portable C++ — the reference implementation
    Cpu,        // multi-threaded + SIMD (NEON / AVX2) when available
    Cuda,       // GPU pipeline (requires PARAJPEG_WITH_CUDA)
};

enum class Subsampling : uint8_t { Gray, S444, S422, S420, S440, S411, Other };

struct ImageInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    int components = 0;          // 1 (grayscale), 3 (YCbCr/RGB), 4 (CMYK/YCCK)
    Subsampling subsampling = Subsampling::Other;
    bool progressive = false;
    bool arithmetic = false;     // reported so callers can route to a fallback decoder
    uint16_t restart_interval = 0;
    int precision = 8;
    bool jfif = false;
    bool adobe = false;
    int adobe_transform = -1;
};

struct DecodeOptions {
    PixelFormat format = PixelFormat::RGB;
    Backend backend = Backend::Auto;
    int threads = 0;                 // 0 = hardware concurrency
    bool fancy_upsampling = true;    // libjpeg "do_fancy_upsampling" (triangle filter) — default on
    // Entropy decoding strategy for baseline scans without restart markers:
    //   0 = sequential, >0 = speculative parallel decoding with this subsequence size in bytes.
    uint32_t speculative_subsequence_bytes = 0;  // 0 = backend default
    bool collect_stats = false;      // fill DecodeStats
};

struct DecodeStats {
    double parse_ms = 0, entropy_ms = 0, idct_ms = 0, color_ms = 0, total_ms = 0;
    uint32_t restart_segments = 0;
    uint32_t subsequences = 0;
    uint32_t sync_iterations = 0;    // passes needed until all speculative decoders synchronised
    uint32_t threads_used = 0;
    std::string backend;
    std::vector<std::string> warnings;
};

struct Image {
    uint32_t width = 0;
    uint32_t height = 0;
    int channels = 0;
    size_t stride = 0;               // bytes per row
    PixelFormat format = PixelFormat::RGB;
    std::vector<uint8_t> pixels;     // stride * height bytes
    const uint8_t* row(uint32_t y) const { return pixels.data() + y * stride; }
    uint8_t* row(uint32_t y) { return pixels.data() + y * stride; }
};

// Parse headers only. Fast; does not touch entropy-coded data.
Status read_header(const uint8_t* data, size_t size, ImageInfo& info);

// Decode a full image into `out` (reallocated as needed).
Status decode(const uint8_t* data, size_t size, const DecodeOptions& opts, Image& out,
              DecodeStats* stats = nullptr);

// Decode into caller-provided memory. `stride` must be >= width * channels(format).
Status decode_into(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst,
                   size_t stride, DecodeStats* stats = nullptr);

// Capability queries.
bool backend_available(Backend b);
const char* backend_name(Backend b);
const char* version();
int channels_of(PixelFormat f);

}  // namespace parajpeg
