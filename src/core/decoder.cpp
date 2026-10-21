// Decoder orchestration: parse -> entropy decode -> IDCT -> upsample/colour convert.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>

#include "color.h"
#include "entropy.h"
#include "idct.h"
#include "parser.h"
#include "thread_pool.h"

namespace parajpeg {

int channels_of(PixelFormat f) {
    switch (f) {
        case PixelFormat::RGB: case PixelFormat::BGR: return 3;
        case PixelFormat::RGBA: case PixelFormat::BGRA: return 4;
        case PixelFormat::Gray: return 1;
    }
    return 3;
}

const char* version() { return "0.1.0"; }

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::Auto: return "auto";
        case Backend::Scalar: return "scalar";
        case Backend::Cpu: return "cpu";
        case Backend::Cuda: return "cuda";
    }
    return "?";
}

bool backend_available(Backend b) {
    switch (b) {
        case Backend::Auto: case Backend::Scalar: case Backend::Cpu: return true;
        case Backend::Cuda:
#ifdef PARAJPEG_WITH_CUDA
            return detail::cuda_available();
#else
            return false;
#endif
    }
    return false;
}

namespace detail {

// A process-wide pool, created on first use, so that decoding many small images does not pay for
// thread creation each time. Requests for fewer threads use a subset of it.
ThreadPool& shared_pool() {
    static ThreadPool pool(int(std::max(1u, std::thread::hardware_concurrency())));
    return pool;
}

namespace {

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Coefficient buffers must start zeroed (sparse writes; progressive accumulates across scans):
// calloc hands out lazily-faulted zero pages, so the cost is paid in parallel by the decode threads
// instead of up-front on one thread. Sample planes are fully overwritten and stay uninitialised.
struct Planes {
    void* coefs[kMaxComponents] = {};
    void* planes[kMaxComponents] = {};
    void* masks[kMaxComponents] = {};
    ~Planes() {
        for (int i = 0; i < kMaxComponents; ++i) { std::free(coefs[i]); std::free(planes[i]); std::free(masks[i]); }
    }
};

Status allocate_planes(Frame& f, Planes& P, bool need_coefs) {
    for (int i = 0; i < f.ncomp; ++i) {
        Component& c = f.comps[i];
        size_t nblocks = size_t(c.padded_blocks_w) * c.padded_blocks_h;
        if (need_coefs) {
            P.coefs[i] = std::calloc(nblocks * kDctSize2, sizeof(int16_t));
            if (!P.coefs[i]) return Status::error(StatusCode::Internal, "out of memory");
            c.coefs = static_cast<int16_t*>(P.coefs[i]);
            if (f.progressive) {
                P.masks[i] = std::calloc(nblocks, sizeof(uint64_t));
                if (!P.masks[i]) return Status::error(StatusCode::Internal, "out of memory");
                c.nzmask = static_cast<uint64_t*>(P.masks[i]);
            }
        } else {
            c.coefs = nullptr;
        }
        c.plane_stride = size_t(c.padded_blocks_w) * 8;
        P.planes[i] = std::malloc(c.plane_stride * size_t(c.padded_blocks_h) * 8 + 64);
        if (!P.planes[i]) return Status::error(StatusCode::Internal, "out of memory");
        c.plane = static_cast<uint8_t*>(P.planes[i]);
    }
    return Status::success();
}

// IDCT all real blocks of a component (rows [by0, by1)).
void idct_rows(const Frame& f, const Component& c, uint32_t by0, uint32_t by1, bool simd) {
    const QuantTable& q = c.qt_latched;
    for (uint32_t by = by0; by < by1; ++by) {
        for (uint32_t bx = 0; bx < c.blocks_w; ++bx) {
            const int16_t* blk = c.coefs + (size_t(by) * c.padded_blocks_w + bx) * kDctSize2;
            uint8_t* out = c.plane + size_t(by) * 8 * c.plane_stride + size_t(bx) * 8;
#ifdef PARAJPEG_HAVE_NEON
            if (simd) idct_islow_neon(blk, q.q.data(), out, c.plane_stride);
            else
#endif
                idct_islow_scalar(blk, q.q.data(), out, c.plane_stride);
        }
    }
}

// Colour stage for output rows [y0, y1).
void color_rows(const Frame& f, const DecodeOptions& opts, uint32_t y0, uint32_t y1, uint8_t* dst, size_t stride,
                std::vector<uint8_t>& scratch) {
    const uint32_t W = f.width;
    const int n = f.ncomp;
    const size_t rowbuf = size_t(W) + 64;
    if (scratch.size() < rowbuf * size_t(n)) scratch.resize(rowbuf * size_t(n));
    const uint8_t* rows[kMaxComponents] = {nullptr, nullptr, nullptr, nullptr};
    for (uint32_t y = y0; y < y1; ++y) {
        for (int i = 0; i < n; ++i) {
            const Component& c = f.comps[i];
            int hexp = f.hmax / c.h, vexp = f.vmax / c.v;
            if (hexp == 1 && vexp == 1) {
                rows[i] = c.plane + size_t(y) * c.plane_stride;
            } else {
                uint8_t* buf = scratch.data() + size_t(i) * rowbuf;
                upsample_row(c, hexp, vexp, opts.fancy_upsampling, y, W, buf);
                rows[i] = buf;
            }
        }
        convert_row(rows, n, f.jpeg_color_space, opts.format, W, dst + size_t(y) * stride);
    }
}

Status decode_cpu(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride,
                  DecodeStats* stats, bool scalar_only) {
    auto t0 = Clock::now();
    ParseResult pr;
    Status s = parse_stream(data, size, pr, false);
    if (!s) return s;
    Frame& f = pr.frame;
    if (f.arithmetic) return Status::error(StatusCode::Unsupported, "arithmetic coding");
    if (f.ncomp == 4 && channels_of(opts.format) != 4)
        return Status::error(StatusCode::Unsupported, "CMYK/YCCK JPEG requires a 4-channel output format");
    if (stride < size_t(f.width) * channels_of(opts.format)) return Status::error(StatusCode::InvalidArgument, "stride too small");

    int threads = scalar_only ? 1 : (opts.threads > 0 ? opts.threads : int(std::thread::hardware_concurrency()));
    if (threads < 1) threads = 1;
    ThreadPool* pool = threads > 1 ? &shared_pool() : nullptr;
    if (pool) threads = std::min(threads, pool->size());
    const bool simd = !scalar_only;

    // Baseline: entropy decoding and IDCT are fused (blocks are reconstructed while still in cache);
    // progressive images need the full coefficient buffer because scans refine it incrementally.
    const bool fused = !f.progressive;
    IdctFn idct_fn = idct_islow_scalar;
#ifdef PARAJPEG_HAVE_NEON
    if (simd) idct_fn = idct_islow_neon;
#endif
    Planes P;
    s = allocate_planes(f, P, !fused);
    if (!s) return s;
    if (fused) {
        // A component that no scan covers keeps all-zero coefficients: its samples are 128.
        bool covered[kMaxComponents] = {false, false, false, false};
        for (const ScanHeader& sh : f.scans)
            for (int i = 0; i < sh.ncomp; ++i) covered[sh.comp_index[i]] = true;
        for (int i = 0; i < f.ncomp; ++i)
            if (!covered[i]) std::memset(f.comps[i].plane, 128, f.comps[i].plane_stride * size_t(f.comps[i].padded_blocks_h) * 8);
    }
    double parse_ms = ms_since(t0);

    // Progressive and multi-scan images touch every coefficient page in an early (often sequential)
    // scan; fault the zeroed pages in parallel first so that cost is spread over all threads.
    if (pool && f.progressive) {
        for (int i = 0; i < f.ncomp; ++i) {
            Component& c = f.comps[i];
            size_t bytes = size_t(c.padded_blocks_w) * c.padded_blocks_h * kDctSize2 * sizeof(int16_t);
            const size_t chunk = 1 << 20;
            uint32_t nch = uint32_t((bytes + chunk - 1) / chunk);
            uint8_t* base = reinterpret_cast<uint8_t*>(c.coefs);
            pool->parallel_for(nch, [&](uint32_t k, int) {
                size_t off = size_t(k) * chunk;
                std::memset(base + off, 0, std::min(chunk, bytes - off));
            }, threads);
        }
    }

    auto t1 = Clock::now();
    EntropyStats es;
    s = decode_scans(data, size, f, pr.events, pool, threads, opts.speculative_subsequence_bytes, es, fused ? idct_fn : nullptr, opts.min_parallel_scan_bytes);
    if (!s) return s;
    double entropy_ms = ms_since(t1);

    auto t2 = Clock::now();
    for (int i = 0; i < f.ncomp && !fused; ++i) {
        const Component& c = f.comps[i];
        if (pool) {
            pool->parallel_for(c.blocks_h, [&](uint32_t by, int) { idct_rows(f, c, by, by + 1, simd); }, threads);
        } else {
            idct_rows(f, c, 0, c.blocks_h, simd);
        }
    }
    double idct_ms = ms_since(t2);

    auto t3 = Clock::now();
    const uint32_t band = f.mcu_h;  // rows per task
    const uint32_t nbands = (f.height + band - 1) / band;
    if (pool) {
        std::vector<std::vector<uint8_t>> scratch(size_t(pool->size()));
        pool->parallel_for(nbands, [&](uint32_t b, int tid) {
            uint32_t y0 = b * band, y1 = std::min(f.height, y0 + band);
            color_rows(f, opts, y0, y1, dst, stride, scratch[size_t(tid)]);
        }, threads);
    } else {
        std::vector<uint8_t> scratch;
        color_rows(f, opts, 0, f.height, dst, stride, scratch);
    }
    double color_ms = ms_since(t3);

    if (stats) {
        stats->parse_ms = parse_ms;
        stats->entropy_ms = entropy_ms;
        stats->idct_ms = idct_ms;
        stats->color_ms = color_ms;
        stats->total_ms = ms_since(t0);
        stats->restart_segments = es.restart_segments;
        stats->subsequences = es.subsequences;
        stats->sync_iterations = es.sync_iterations;
        stats->waves = es.waves;
        stats->scans.clear();
        for (const ScanTrace& t : es.scans) {
            ScanStat ss;
            ss.index = t.index; ss.wave = t.wave; ss.ncomp = t.ncomp; ss.first_component = t.first_comp;
            ss.ss = t.ss; ss.se = t.se; ss.ah = t.ah; ss.al = t.al; ss.bytes = t.bytes; ss.blocks = t.blocks;
            ss.strategy = t.strategy == ScanStrategy::Sequential ? "sequential" : t.strategy == ScanStrategy::RestartParallel ? "restart-parallel"
                        : t.strategy == ScanStrategy::BitIndexed ? "bit-indexed" : "speculative";
            ss.subsequences = t.subsequences; ss.sync_iterations = t.sync_iterations; ss.ms = t.ms;
            stats->scans.push_back(ss);
        }
        stats->threads_used = uint32_t(threads);
        stats->backend = scalar_only ? "scalar" : "cpu";
        stats->warnings = f.warnings;
        if (es.bad_code) stats->warnings.push_back("corrupt JPEG data: bad Huffman code");
        if (es.premature_end) stats->warnings.push_back("premature end of data segment");
        if (es.resync) stats->warnings.push_back("restart marker resynchronisation performed");
        if (es.speculative_fallback) stats->warnings.push_back("restart markers inconsistent; sequential fallback");
    }
    if (f.truncated) return Status::error(StatusCode::Truncated, "stream truncated; output is best-effort");
    return Status::success();
}

}  // namespace

#ifdef PARAJPEG_WITH_CUDA
bool cuda_available();
Status decode_cuda(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats);
#endif

}  // namespace detail

Status read_header(const uint8_t* data, size_t size, ImageInfo& info) {
    detail::ParseResult pr;
    Status s = detail::parse_stream(data, size, pr, true);
    if (!s) return s;
    const detail::Frame& f = pr.frame;
    info.width = f.width;
    info.height = f.height;
    info.components = f.ncomp;
    info.subsampling = f.subsampling();
    info.progressive = f.progressive;
    info.arithmetic = f.arithmetic;
    info.restart_interval = f.restart_interval;
    info.precision = f.precision;
    info.jfif = f.jfif;
    info.adobe = f.adobe;
    info.adobe_transform = f.adobe_transform;
    return Status::success();
}

Status decode_into(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats) {
    if (!data || !dst) return Status::error(StatusCode::InvalidArgument, "null pointer");
    Backend b = opts.backend;
    if (b == Backend::Auto) b = backend_available(Backend::Cuda) ? Backend::Cuda : Backend::Cpu;
    switch (b) {
        case Backend::Scalar: return detail::decode_cpu(data, size, opts, dst, stride, stats, true);
        case Backend::Cpu: return detail::decode_cpu(data, size, opts, dst, stride, stats, false);
        case Backend::Cuda:
#ifdef PARAJPEG_WITH_CUDA
            return detail::decode_cuda(data, size, opts, dst, stride, stats);
#else
            return Status::error(StatusCode::BackendUnavailable, "built without CUDA");
#endif
        default: return Status::error(StatusCode::InvalidArgument, "bad backend");
    }
}

Status decode(const uint8_t* data, size_t size, const DecodeOptions& opts, Image& out, DecodeStats* stats) {
    ImageInfo info;
    Status s = read_header(data, size, info);
    if (!s) return s;
    out.width = info.width;
    out.height = info.height;
    out.channels = channels_of(opts.format);
    out.format = opts.format;
    out.stride = size_t(out.width) * size_t(out.channels);
    out.pixels.assign(out.stride * out.height, 0);
    return decode_into(data, size, opts, out.pixels.data(), out.stride, stats);
}

}  // namespace parajpeg
