// CUDA backend.
//
// Baseline scans: fully on the GPU, in the massively parallel formulation of speculative Huffman
// decoding (Weißenberger & Schmidt, HiPC 2021), extended with interleaved-MCU phase tracking:
//   k_spec_init    one thread per K-byte subsequence decodes from an arbitrary byte, records the
//                  state it reaches at the subsequence end;
//   k_spec_iter    Jacobi iterations: thread i re-decodes subsequence i from the state recorded for
//                  i-1; repeat until no state changes (usually 1-2 rounds). The converged round also
//                  yields per-subsequence block counts and DC-difference sums;
//   k_seg_*        exclusive prefix sums -> absolute block offsets and DC predictors;
//   k_spec_write   final decode from exact states, writing coefficients (absolute DC);
//   k_restart      (images with restart markers) one thread per restart interval instead.
// Progressive scans: entropy-decoded by the parallel CPU engine, reconstructed on the GPU.
// Then k_idct (one thread per 8x8 block, libjpeg islow, shared with the CPU) and k_color (one thread
// per output pixel: libjpeg fancy upsampling + fixed-point YCbCr->RGB).
//
// No kernel uses shared memory or __syncthreads, so the identical source runs in the CPU grid
// emulator (PJ_CUDA_EMULATION) and the bit-exactness tests cover the GPU algorithm on any machine.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "../common/idct_core.h"
#include "../core/bitreader.h"
#include "../core/entropy.h"
#include "../core/parser.h"
#include "../core/thread_pool.h"
#include "gpu_runtime.h"
#include "parajpeg/parajpeg.h"

namespace parajpeg::detail {
namespace gpuk {

constexpr int kLookup = HuffTable::kLookupBits;

struct DevHuff {
    uint16_t lookup[1 << kLookup];
    int32_t maxcode[18];
    int32_t valoffset[17];
    uint8_t vals[256];
};

struct DevComp {
    uint32_t h, v, padded_bw, padded_bh, blocks_w, blocks_h, width, height;
    uint64_t coef_off;   // in int16 elements
    uint64_t plane_off;  // in bytes
    uint32_t plane_stride;
    uint16_t quant[64];
};

struct DevFrame {
    int ncomp, hmax, vmax, cs;
    uint32_t W, H, mcus_x, mcus_y;
    DevComp comps[kMaxComponents];
};

struct DevScan {
    int interleaved, bpm, comp;
    uint8_t sc[kMaxBlocksPerMcu], sbx[kMaxBlocksPerMcu], sby[kMaxBlocksPerMcu];
    uint32_t total_blocks, bw;
    int dc_tab[kMaxComponents], ac_tab[kMaxComponents];  // indices into the DevHuff array
    uint64_t begin, end;
    uint32_t K;  // subsequence bytes
};

struct DState {
    uint64_t bitpos;
    uint32_t blocks;
    uint8_t k, slot;
    int32_t dc[kMaxComponents];
};

PJ_CONSTANT const uint8_t c_natural[80] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,
    6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
    39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

PJ_HD PJ_FORCEINLINE bool same_key(const DState& a, const DState& b) {
    return a.bitpos == b.bitpos && a.k == b.k && a.slot == b.slot;
}

struct DCursor {
    uint32_t ordinal, mx, my;
    int slot;
    PJ_HD void seek(const DevScan& S, const DevFrame& F, uint32_t o) {
        ordinal = o;
        if (S.interleaved) {
            uint32_t mcu = o / uint32_t(S.bpm);
            slot = int(o % uint32_t(S.bpm));
            mx = mcu % F.mcus_x;
            my = mcu / F.mcus_x;
        } else {
            slot = 0;
            mx = S.bw ? o % S.bw : 0;
            my = S.bw ? o / S.bw : 0;
        }
    }
    PJ_HD PJ_FORCEINLINE int comp(const DevScan& S) const { return S.interleaved ? S.sc[slot] : S.comp; }
    PJ_HD PJ_FORCEINLINE int16_t* ptr(const DevScan& S, const DevFrame& F, int16_t* coefs) const {
        if (S.interleaved) {
            const DevComp& c = F.comps[S.sc[slot]];
            uint32_t bx = mx * c.h + S.sbx[slot], by = my * c.v + S.sby[slot];
            return coefs + c.coef_off + (uint64_t(by) * c.padded_bw + bx) * 64;
        }
        const DevComp& c = F.comps[S.comp];
        return coefs + c.coef_off + (uint64_t(my) * c.padded_bw + mx) * 64;
    }
    PJ_HD PJ_FORCEINLINE void advance(const DevScan& S, const DevFrame& F) {
        ++ordinal;
        if (S.interleaved) {
            if (++slot == S.bpm) {
                slot = 0;
                if (++mx == F.mcus_x) { mx = 0; ++my; }
            }
        } else if (++mx == S.bw) {
            mx = 0;
            ++my;
        }
    }
};

PJ_HD PJ_FORCEINLINE int dhuff(BitReader& br, const DevHuff& t, bool& bad) {
    uint32_t p = br.peek(kLookup);
    uint16_t e = t.lookup[p];
    if (e) {
        br.consume(e >> 8);
        return e & 0xFF;
    }
    int l = kLookup + 1;
    int code = int(br.peek(l));
    while (code > t.maxcode[l]) {
        if (++l > 16) {
            bad = true;
            br.consume(17);
            return 0;
        }
        code = int(br.peek(l));
    }
    br.consume(l);
    return t.vals[(code + t.valoffset[l]) & 0xFF];
}

PJ_HD PJ_FORCEINLINE int dextend(uint32_t v, int s) { return int(v) < (1 << (s - 1)) ? int(v) - (1 << s) + 1 : int(v); }
PJ_HD PJ_FORCEINLINE int dwadd(int a, int b) { return int(unsigned(a) + unsigned(b)); }

// Baseline symbol decoding until `until` bits (or, when kCheckEnd, until `stop_ordinal` blocks have
// been completed). Identical stopping rule in every kernel, which the synchronisation relies on.
template <bool kWrite, bool kCheckEnd>
PJ_HD void decode_until(const DevScan& S, const DevFrame& F, const DevHuff* tabs, BitReader& br, DCursor& cur,
                        DState& st, uint64_t until, uint32_t stop_ordinal, int16_t* coefs, bool& bad) {
    int comp = 0;
    int16_t* blk = nullptr;
    if (st.k != 0) {
        if (kCheckEnd && cur.ordinal >= stop_ordinal) return;
        comp = cur.comp(S);
        if (kWrite) blk = cur.ptr(S, F, coefs);
    }
    for (;;) {
        if (br.approx_bitpos() + 64 >= until && br.canonical_bitpos() >= until) break;
        if (st.k == 0) {
            if (kCheckEnd && cur.ordinal >= stop_ordinal) break;
            comp = cur.comp(S);
            if (kWrite) blk = cur.ptr(S, F, coefs);
            st.blocks++;
            if (br.bits_available() < 32) br.refill();
            int s = dhuff(br, tabs[S.dc_tab[comp]], bad);
            int diff = 0;
            if (s) {
                if (s > 16) { bad = true; s = 16; }
                diff = dextend(br.get_bits_nocheck(s), s);
            }
            st.dc[comp] = dwadd(st.dc[comp], diff);
            if (kWrite) blk[0] = int16_t(st.dc[comp]);
            st.k = 1;
            continue;
        }
        if (br.bits_available() < 32) br.refill();
        int rs = dhuff(br, tabs[S.ac_tab[comp]], bad);
        int r = rs >> 4, s = rs & 15;
        int k = st.k;
        if (s) {
            k += r;
            int v = dextend(br.get_bits_nocheck(s), s);
            if (kWrite) blk[c_natural[k]] = int16_t(v);
            ++k;
        } else if (r == 15) {
            k += 16;
        } else {
            k = 64;
        }
        if (k > 63) {
            st.k = 0;
            cur.advance(S, F);
        } else {
            st.k = uint8_t(k);
        }
    }
}

PJ_HD PJ_FORCEINLINE uint64_t sub_end_bit(const DevScan& S, uint32_t i) {
    uint64_t e = S.begin + uint64_t(i + 1) * S.K;
    return (e < S.end ? e : S.end) * 8;
}

PJ_HD PJ_FORCEINLINE void position(BitReader& br, uint64_t bitpos) {
    br.set_pos(size_t(bitpos / 8));
    if (bitpos % 8) { br.refill(); br.consume(int(bitpos % 8)); }
}

PJ_HD PJ_FORCEINLINE DState exact_start(const DevScan& S) {
    DState s{};
    s.bitpos = S.begin * 8;
    return s;
}

PJ_GLOBAL void k_spec_init(DevScan S, DevFrame F, const DevHuff* tabs, const uint8_t* data, uint32_t nsub, DState* E) {
    const uint32_t i = PJ_TID;
    if (i >= nsub) return;
    BitReader br(data, size_t(S.begin), size_t(S.end), size_t(S.begin + uint64_t(i) * S.K));
    DCursor cur;
    // Guess the MCU phase from the byte offset (exact for i == 0).
    uint64_t nbytes = S.end - S.begin;
    uint32_t guess = i == 0 ? 0 : uint32_t((uint64_t(i) * S.K * S.total_blocks) / (nbytes ? nbytes : 1));
    cur.seek(S, F, guess);
    DState st{};
    bool bad = false;
    decode_until<false, false>(S, F, tabs, br, cur, st, sub_end_bit(S, i), 0, nullptr, bad);
    st.bitpos = br.canonical_bitpos();
    st.slot = uint8_t(cur.slot);
    E[i] = st;
}

PJ_GLOBAL void k_spec_iter(DevScan S, DevFrame F, const DevHuff* tabs, const uint8_t* data, uint32_t nsub, const DState* Ein,
                           DState* Eout, uint32_t* counts, int32_t* dcd, int* changed) {
    const uint32_t i = PJ_TID;
    if (i >= nsub) return;
    const DState st0 = i == 0 ? exact_start(S) : Ein[i - 1];
    BitReader br(data, size_t(S.begin), size_t(S.end), size_t(S.begin));
    position(br, st0.bitpos);
    DCursor cur;
    cur.seek(S, F, 0);
    cur.slot = st0.slot;
    DState st{};
    st.k = st0.k;
    bool bad = false;
    decode_until<false, false>(S, F, tabs, br, cur, st, sub_end_bit(S, i), 0, nullptr, bad);
    st.bitpos = br.canonical_bitpos();
    st.slot = uint8_t(cur.slot);
    counts[i] = st.blocks;
    for (int c = 0; c < kMaxComponents; ++c) dcd[i * kMaxComponents + c] = st.dc[c];
    Eout[i] = st;
    if (!same_key(st, Ein[i])) *changed = 1;
}

// Exclusive prefix sums of (blocks, dc[0..3]) over subsequences, in three sync-free kernels.
constexpr uint32_t kSeg = 256;
PJ_GLOBAL void k_seg_sum(uint32_t n, const uint32_t* counts, const int32_t* dcd, int64_t* seg) {
    const uint32_t t = PJ_TID;
    const uint32_t a = t * kSeg;
    if (a >= n) return;
    const uint32_t b = a + kSeg < n ? a + kSeg : n;
    int64_t s[5] = {0, 0, 0, 0, 0};
    for (uint32_t j = a; j < b; ++j) {
        s[0] += counts[j];
        for (int c = 0; c < 4; ++c) s[1 + c] = int64_t(uint32_t(int32_t(s[1 + c]) + dcd[j * 4 + c]));
    }
    for (int c = 0; c < 5; ++c) seg[t * 5 + c] = s[c];
}
PJ_GLOBAL void k_seg_scan(uint32_t nseg, int64_t* seg) {
    if (PJ_TID != 0) return;
    int64_t run[5] = {0, 0, 0, 0, 0};
    for (uint32_t t = 0; t < nseg; ++t)
        for (int c = 0; c < 5; ++c) {
            int64_t v = seg[t * 5 + c];
            seg[t * 5 + c] = run[c];
            run[c] = c == 0 ? run[c] + v : int64_t(uint32_t(int32_t(run[c]) + int32_t(v)));
        }
}
PJ_GLOBAL void k_seg_apply(uint32_t n, const uint32_t* counts, const int32_t* dcd, const int64_t* seg, uint32_t* off, int32_t* dcs) {
    const uint32_t t = PJ_TID;
    const uint32_t a = t * kSeg;
    if (a >= n) return;
    const uint32_t b = a + kSeg < n ? a + kSeg : n;
    uint32_t blocks = uint32_t(seg[t * 5]);
    int32_t dc[4] = {int32_t(seg[t * 5 + 1]), int32_t(seg[t * 5 + 2]), int32_t(seg[t * 5 + 3]), int32_t(seg[t * 5 + 4])};
    for (uint32_t j = a; j < b; ++j) {
        off[j] = blocks;
        for (int c = 0; c < 4; ++c) dcs[j * 4 + c] = dc[c];
        blocks += counts[j];
        for (int c = 0; c < 4; ++c) dc[c] = dwadd(dc[c], dcd[j * 4 + c]);
    }
}

PJ_GLOBAL void k_spec_write(DevScan S, DevFrame F, const DevHuff* tabs, const uint8_t* data, uint32_t nsub, const DState* E,
                            const uint32_t* off, const int32_t* dcs, int16_t* coefs, int* bad_flag) {
    const uint32_t i = PJ_TID;
    if (i >= nsub) return;
    DState st = i == 0 ? exact_start(S) : E[i - 1];
    st.blocks = off[i];
    for (int c = 0; c < kMaxComponents; ++c) st.dc[c] = dcs[i * kMaxComponents + c];
    if (st.k == 0 && st.blocks >= S.total_blocks) return;
    BitReader br(data, size_t(S.begin), size_t(S.end), size_t(S.begin));
    position(br, st.bitpos);
    DCursor cur;
    cur.seek(S, F, st.k ? st.blocks - 1 : st.blocks);
    bool bad = false;
    const uint64_t until = (i + 1 == nsub) ? ~uint64_t(0) : sub_end_bit(S, i);
    decode_until<true, true>(S, F, tabs, br, cur, st, until, S.total_blocks, coefs, bad);
    if (bad) *bad_flag = 1;
}

PJ_GLOBAL void k_restart(DevScan S, DevFrame F, const DevHuff* tabs, const uint8_t* data, uint32_t nseg, const uint64_t* segs,
                         uint32_t per, int16_t* coefs, int* bad_flag) {
    const uint32_t i = PJ_TID;
    if (i >= nseg) return;
    BitReader br(data, size_t(segs[2 * i]), size_t(segs[2 * i + 1]), size_t(segs[2 * i]));
    DCursor cur;
    cur.seek(S, F, i * per);
    DState st{};
    bool bad = false;
    uint32_t stop = (i + 1) * per < S.total_blocks ? (i + 1) * per : S.total_blocks;
    decode_until<true, true>(S, F, tabs, br, cur, st, ~uint64_t(0), stop, coefs, bad);
    if (bad) *bad_flag = 1;
}

PJ_GLOBAL void k_idct(DevFrame F, const int16_t* coefs, uint8_t* planes, uint64_t total_blocks) {
    const uint64_t b = PJ_TID;
    if (b >= total_blocks) return;
    uint64_t local = b;
    int ci = 0;
    while (ci < F.ncomp - 1 && local >= uint64_t(F.comps[ci].padded_bw) * F.comps[ci].padded_bh) {
        local -= uint64_t(F.comps[ci].padded_bw) * F.comps[ci].padded_bh;
        ++ci;
    }
    const DevComp& c = F.comps[ci];
    const uint32_t bx = uint32_t(local % c.padded_bw), by = uint32_t(local / c.padded_bw);
    if (bx >= c.blocks_w || by >= c.blocks_h) return;
    idct_islow_core(coefs + c.coef_off + local * 64, c.quant, planes + c.plane_off + uint64_t(by) * 8 * c.plane_stride + bx * 8,
                    c.plane_stride);
}

// One upsampled sample of component `c` at output pixel (x, y): libjpeg-turbo jdsample.c semantics.
PJ_HD PJ_FORCEINLINE int sample(const DevFrame& F, const DevComp& c, const uint8_t* plane, uint32_t x, uint32_t y, bool fancy) {
    const int hexp = F.hmax / int(c.h), vexp = F.vmax / int(c.v);
    const uint32_t in_w = c.width, in_h = c.height;
    uint32_t iy = y / uint32_t(vexp);
    if (iy > in_h - 1) iy = in_h - 1;
    const uint8_t* row = plane + uint64_t(iy) * c.plane_stride;
    if (hexp == 1 && vexp == 1) return row[x];
    const uint32_t v = y & 1;
    const uint32_t ny = v == 0 ? (iy == 0 ? 0 : iy - 1) : (iy + 1 < in_h ? iy + 1 : in_h - 1);
    const uint8_t* nrow = plane + uint64_t(ny) * c.plane_stride;
    const uint32_t cx = x >> 1;
    if (fancy && in_w > 2 && hexp == 2 && vexp == 1) {
        if ((x & 1) == 0) return cx == 0 ? row[0] : (3 * row[cx] + row[cx - 1] + 1) >> 2;
        return cx == in_w - 1 ? row[cx] : (3 * row[cx] + row[cx + 1] + 2) >> 2;
    }
    if (fancy && in_w > 2 && hexp == 2 && vexp == 2) {
        const int cs = 3 * row[cx] + nrow[cx];
        if ((x & 1) == 0) return cx == 0 ? (cs * 4 + 8) >> 4 : (cs * 3 + (3 * row[cx - 1] + nrow[cx - 1]) + 8) >> 4;
        return cx == in_w - 1 ? (cs * 4 + 7) >> 4 : (cs * 3 + (3 * row[cx + 1] + nrow[cx + 1]) + 7) >> 4;
    }
    if (fancy && hexp == 1 && vexp == 2) return (3 * row[x] + nrow[x] + (v == 0 ? 1 : 2)) >> 2;
    return row[x / uint32_t(hexp)];
}

PJ_HD PJ_FORCEINLINE uint8_t clamp255(int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); }

PJ_GLOBAL void k_color(DevFrame F, const uint8_t* planes, uint8_t* out, uint64_t out_stride, int fmt, int fancy) {
    const uint64_t t = PJ_TID;
    if (t >= uint64_t(F.W) * F.H) return;
    const uint32_t x = uint32_t(t % F.W), y = uint32_t(t / F.W);
    int s[4] = {0, 0, 0, 0};
    for (int i = 0; i < F.ncomp; ++i) s[i] = sample(F, F.comps[i], planes + F.comps[i].plane_off, x, y, fancy != 0);
    uint8_t* o = out + uint64_t(y) * out_stride;
    // fmt: 0 RGB, 1 BGR, 2 RGBA, 3 BGRA, 4 Gray ; cs: 0 Gray, 1 YCbCr, 2 RGB, 3 CMYK, 4 YCCK
    const int ch = fmt == 4 ? 1 : (fmt >= 2 ? 4 : 3);
    const bool bgr = fmt == 1 || fmt == 3;
    if (fmt == 4) {
        if (F.ncomp == 1 || F.cs == 1) o[x] = uint8_t(s[0]);
        else o[x] = uint8_t((s[0] * 19595 + s[1] * 38470 + s[2] * 7471 + 32768) >> 16);
        return;
    }
    uint8_t* p = o + uint64_t(x) * ch;
    int r, g, b;
    if (F.ncomp == 1) {
        r = g = b = s[0];
    } else if (F.ncomp == 3 && F.cs == 2) {
        r = s[0]; g = s[1]; b = s[2];
    } else if (F.ncomp == 3) {
        const int cb = s[1] - 128, cr = s[2] - 128;
        r = clamp255(s[0] + ((91881 * cr + 32768) >> 16));
        g = clamp255(s[0] + ((-22554 * cb + 32768 - 46802 * cr) >> 16));
        b = clamp255(s[0] + ((116130 * cb + 32768) >> 16));
    } else {
        int cc = s[0], m = s[1], yy = s[2];
        if (F.cs == 4) {
            const int cb = s[1] - 128, cr = s[2] - 128;
            cc = 255 - clamp255(s[0] + ((91881 * cr + 32768) >> 16));
            m = 255 - clamp255(s[0] + ((-22554 * cb + 32768 - 46802 * cr) >> 16));
            yy = 255 - clamp255(s[0] + ((116130 * cb + 32768) >> 16));
        }
        p[0] = uint8_t(cc); p[1] = uint8_t(m); p[2] = uint8_t(yy); p[3] = uint8_t(s[3]);
        return;
    }
    p[0] = uint8_t(bgr ? b : r);
    p[1] = uint8_t(g);
    p[2] = uint8_t(bgr ? r : b);
    if (ch == 4) p[3] = 255;
}

}  // namespace gpuk

// ------------------------------------------------------------------------------------------------
// Host orchestration.
// ------------------------------------------------------------------------------------------------
namespace {

using namespace gpuk;
using Clock = std::chrono::steady_clock;

struct DevBuf {
    void* p = nullptr;
    ~DevBuf() { if (p) gpu::dfree(p); }
    bool alloc(size_t n) { return gpu::dmalloc(&p, n); }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

DevHuff to_dev(const HuffTable& t) {
    DevHuff d{};
    for (int i = 0; i < (1 << kLookup); ++i) d.lookup[i] = t.lookup[size_t(i)];
    for (int i = 0; i < 18; ++i) d.maxcode[i] = t.maxcode[size_t(i)];
    for (int i = 0; i < 17; ++i) d.valoffset[i] = t.valoffset[size_t(i)];
    for (int i = 0; i < 256; ++i) d.vals[i] = t.vals[size_t(i)];
    return d;
}

int fmt_code(PixelFormat f) {
    switch (f) {
        case PixelFormat::RGB: return 0;
        case PixelFormat::BGR: return 1;
        case PixelFormat::RGBA: return 2;
        case PixelFormat::BGRA: return 3;
        case PixelFormat::Gray: return 4;
    }
    return 0;
}
int cs_code(Frame::ColorSpace c) {
    switch (c) {
        case Frame::ColorSpace::Gray: return 0;
        case Frame::ColorSpace::YCbCr: return 1;
        case Frame::ColorSpace::RGB: return 2;
        case Frame::ColorSpace::CMYK: return 3;
        case Frame::ColorSpace::YCCK: return 4;
        default: return 1;
    }
}

inline unsigned blocks_for(uint64_t n, unsigned bs) { return unsigned((n + bs - 1) / bs); }

bool find_segments(const uint8_t* d, size_t begin, size_t end, uint32_t expected, std::vector<uint64_t>& segs) {
    segs.clear();
    size_t start = begin, p = begin;
    int expect_n = 0;
    while (p < end) {
        const void* hit = std::memchr(d + p, 0xFF, end - p);
        if (!hit) break;
        p = size_t(static_cast<const uint8_t*>(hit) - d);
        size_t q = p;
        while (q < end && d[q] == 0xFF) ++q;
        if (q >= end) break;
        uint8_t m = d[q];
        if (m == 0x00) { p = q + 1; continue; }
        if (m >= 0xD0 && m <= 0xD7) {
            if (int(m - 0xD0) != expect_n) return false;
            segs.push_back(start);
            segs.push_back(p);
            expect_n = (expect_n + 1) & 7;
            start = q + 1;
            p = q + 1;
            continue;
        }
        break;
    }
    segs.push_back(start);
    segs.push_back(end);
    return segs.size() / 2 == expected;
}

}  // namespace

#if defined(PJ_CUDA_EMULATION)
#define PJ_GPU_ENTRY decode_cuda_emulated
#define PJ_GPU_AVAILABLE cuda_emulated_available
#else
#define PJ_GPU_ENTRY decode_cuda
#define PJ_GPU_AVAILABLE cuda_available
#endif

ThreadPool& shared_pool();

bool PJ_GPU_AVAILABLE() { return gpu::runtime_available(); }

Status PJ_GPU_ENTRY(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats) {
    auto t0 = Clock::now();
    ParseResult pr;
    Status s = parse_stream(data, size, pr, false);
    if (!s) return s;
    Frame& f = pr.frame;
    if (f.arithmetic) return Status::error(StatusCode::Unsupported, "arithmetic coding");
    if (f.ncomp == 4 && channels_of(opts.format) != 4)
        return Status::error(StatusCode::Unsupported, "CMYK/YCCK JPEG requires a 4-channel output format");
    if (stride < size_t(f.width) * channels_of(opts.format)) return Status::error(StatusCode::InvalidArgument, "stride too small");

    gpu::Stream st;
    if (!gpu::stream_create(&st)) return Status::error(StatusCode::BackendUnavailable, "cannot create CUDA stream");
    struct StreamGuard { gpu::Stream s; ~StreamGuard() { gpu::stream_destroy(s); } } sg{st};

    // Geometry and buffers.
    DevFrame F{};
    F.ncomp = f.ncomp; F.hmax = f.hmax; F.vmax = f.vmax; F.cs = cs_code(f.jpeg_color_space);
    F.W = f.width; F.H = f.height; F.mcus_x = f.mcus_x; F.mcus_y = f.mcus_y;
    uint64_t coef_elems = 0, plane_bytes = 0, total_blocks = 0;
    for (int i = 0; i < f.ncomp; ++i) {
        const Component& c = f.comps[i];
        DevComp& d = F.comps[i];
        d.h = c.h; d.v = c.v; d.padded_bw = c.padded_blocks_w; d.padded_bh = c.padded_blocks_h;
        d.blocks_w = c.blocks_w; d.blocks_h = c.blocks_h; d.width = c.width; d.height = c.height;
        d.coef_off = coef_elems;
        d.plane_off = plane_bytes;
        d.plane_stride = c.padded_blocks_w * 8;
        uint64_t nb = uint64_t(c.padded_blocks_w) * c.padded_blocks_h;
        coef_elems += nb * 64;
        plane_bytes += uint64_t(d.plane_stride) * c.padded_blocks_h * 8;
        total_blocks += nb;
    }
    const int ch = channels_of(opts.format);
    DevBuf d_data, d_coefs, d_planes, d_out, d_bad;
    if (!d_data.alloc(size) || !d_coefs.alloc(coef_elems * 2) || !d_planes.alloc(plane_bytes) ||
        !d_out.alloc(size_t(f.width) * f.height * ch) || !d_bad.alloc(sizeof(int)))
        return Status::error(StatusCode::Internal, "GPU out of memory");
    gpu::h2d(d_data.p, data, size, st);
    gpu::dmemset(d_coefs.p, 0, coef_elems * 2, st);
    gpu::dmemset(d_bad.p, 0, sizeof(int), st);
    double parse_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    auto t1 = Clock::now();
    EntropyStats es;
    uint32_t total_sub = 0, total_iters = 0;
    if (f.progressive) {
        // Entropy-decode on the CPU engine (parallel scan DAG), then reconstruct on the GPU.
        std::vector<int16_t> coefs(coef_elems, 0);
        std::vector<uint64_t> masks(total_blocks, 0);
        uint64_t moff = 0;
        for (int i = 0; i < f.ncomp; ++i) {
            f.comps[i].coefs = coefs.data() + F.comps[i].coef_off;
            f.comps[i].nzmask = masks.data() + moff;
            moff += uint64_t(F.comps[i].padded_bw) * F.comps[i].padded_bh;
        }
        int threads = opts.threads > 0 ? opts.threads : shared_pool().size();
        s = decode_scans(data, size, f, pr.events, threads > 1 ? &shared_pool() : nullptr, threads, opts.speculative_subsequence_bytes, es);
        if (!s) return s;
        gpu::h2d(d_coefs.p, coefs.data(), coef_elems * 2, st);
        for (int i = 0; i < f.ncomp; ++i) std::memcpy(F.comps[i].quant, f.comps[i].qt_latched.q.data(), sizeof(F.comps[i].quant));
        gpu::sync(st);
    } else {
        // Replay table events per scan; latch quantisation tables at each component's first scan.
        size_t ev = 0;
        while (ev < pr.events.size() && pr.events[ev].before_scan == 0) ++ev;
        bool latched[kMaxComponents] = {false, false, false, false};
        const uint32_t K = opts.speculative_subsequence_bytes ? std::max<uint32_t>(opts.speculative_subsequence_bytes, 16) : 256;
        for (size_t si = 0; si < f.scans.size(); ++si) {
            while (ev < pr.events.size() && pr.events[ev].before_scan == int(si)) {
                s = apply_table_event(data, size, pr.events[ev], f);
                if (!s) return s;
                ++ev;
            }
            const ScanHeader& sh = f.scans[si];
            ScanLayout L;
            L.init(f, sh);
            DevScan S{};
            S.interleaved = L.interleaved ? 1 : 0;
            S.bpm = L.blocks_per_mcu;
            S.comp = L.comp;
            for (int k = 0; k < L.blocks_per_mcu; ++k) {
                S.sc[k] = uint8_t(L.slots[k].comp);
                S.sbx[k] = uint8_t(L.slots[k].bx);
                S.sby[k] = uint8_t(L.slots[k].by);
            }
            S.total_blocks = L.total_blocks;
            S.bw = L.bw;
            S.begin = sh.data_offset;
            S.end = sh.data_offset + sh.data_size;
            S.K = K;
            std::vector<DevHuff> tabs;
            for (int i = 0; i < sh.ncomp; ++i) {
                const int ci = sh.comp_index[i];
                const HuffTable& dct = f.dc_tables[sh.td[i]];
                const HuffTable& act = f.ac_tables[sh.ta[i]];
                if (!dct.defined) return Status::error(StatusCode::Corrupt, "scan references undefined DC Huffman table");
                if (!act.defined) return Status::error(StatusCode::Corrupt, "scan references undefined AC Huffman table");
                S.dc_tab[ci] = int(tabs.size());
                tabs.push_back(to_dev(dct));
                S.ac_tab[ci] = int(tabs.size());
                tabs.push_back(to_dev(act));
                if (!latched[ci]) {
                    if (!f.qt[f.comps[ci].tq].defined) return Status::error(StatusCode::Corrupt, "component references undefined quant table");
                    std::memcpy(F.comps[ci].quant, f.qt[f.comps[ci].tq].q.data(), sizeof(F.comps[ci].quant));
                    latched[ci] = true;
                }
            }
            DevBuf d_tabs;
            if (!d_tabs.alloc(tabs.size() * sizeof(DevHuff))) return Status::error(StatusCode::Internal, "GPU out of memory");
            gpu::h2d(d_tabs.p, tabs.data(), tabs.size() * sizeof(DevHuff), st);
            const unsigned BS = 128;

            std::vector<uint64_t> segs;
            const uint32_t ri = f.restart_interval;
            const uint32_t nseg = ri ? (L.total_mcus + ri - 1) / ri : 0;
            if (ri && find_segments(data, size_t(S.begin), size_t(S.end), nseg, segs)) {
                DevBuf d_segs;
                if (!d_segs.alloc(segs.size() * 8)) return Status::error(StatusCode::Internal, "GPU out of memory");
                gpu::h2d(d_segs.p, segs.data(), segs.size() * 8, st);
                PJ_LAUNCH(k_restart, blocks_for(nseg, BS), BS, st, S, F, d_tabs.as<DevHuff>(), d_data.as<uint8_t>(), nseg,
                          d_segs.as<uint64_t>(), ri * uint32_t(L.blocks_per_mcu), d_coefs.as<int16_t>(), d_bad.as<int>());
                gpu::sync(st);
                es.restart_segments += nseg - 1;
                continue;
            }
            if (ri) {
                // Inconsistent restart markers: the sequential CPU decoder applies libjpeg's resync.
                return Status::error(StatusCode::Unsupported, "corrupt restart markers: use the CPU backend");
            }
            const uint32_t nsub = uint32_t((S.end - S.begin + K - 1) / K);
            if (nsub == 0) continue;
            DevBuf dE0, dE1, dCounts, dDcd, dOff, dDcs, dSeg, dChanged;
            const uint32_t nsegs = (nsub + kSeg - 1) / kSeg;
            if (!dE0.alloc(nsub * sizeof(DState)) || !dE1.alloc(nsub * sizeof(DState)) || !dCounts.alloc(nsub * 4) ||
                !dDcd.alloc(nsub * 16) || !dOff.alloc(nsub * 4) || !dDcs.alloc(nsub * 16) || !dSeg.alloc(nsegs * 5 * 8) ||
                !dChanged.alloc(sizeof(int)))
                return Status::error(StatusCode::Internal, "GPU out of memory");
            PJ_LAUNCH(k_spec_init, blocks_for(nsub, BS), BS, st, S, F, d_tabs.as<DevHuff>(), d_data.as<uint8_t>(), nsub, dE0.as<DState>());
            DState* Ein = dE0.as<DState>();
            DState* Eout = dE1.as<DState>();
            uint32_t iters = 0;
            for (;;) {
                int changed = 0;
                gpu::h2d(dChanged.p, &changed, sizeof(int), st);
                PJ_LAUNCH(k_spec_iter, blocks_for(nsub, BS), BS, st, S, F, d_tabs.as<DevHuff>(), d_data.as<uint8_t>(), nsub, Ein, Eout,
                          dCounts.as<uint32_t>(), dDcd.as<int32_t>(), dChanged.as<int>());
                gpu::d2h(&changed, dChanged.p, sizeof(int), st);
                gpu::sync(st);
                ++iters;
                std::swap(Ein, Eout);
                if (!changed || iters > nsub + 1) break;
            }
            // Ein now holds the converged end states.
            PJ_LAUNCH(k_seg_sum, blocks_for(nsegs, BS), BS, st, nsub, dCounts.as<uint32_t>(), dDcd.as<int32_t>(), dSeg.as<int64_t>());
            PJ_LAUNCH(k_seg_scan, 1u, 1u, st, nsegs, dSeg.as<int64_t>());
            PJ_LAUNCH(k_seg_apply, blocks_for(nsegs, BS), BS, st, nsub, dCounts.as<uint32_t>(), dDcd.as<int32_t>(), dSeg.as<int64_t>(),
                      dOff.as<uint32_t>(), dDcs.as<int32_t>());
            PJ_LAUNCH(k_spec_write, blocks_for(nsub, BS), BS, st, S, F, d_tabs.as<DevHuff>(), d_data.as<uint8_t>(), nsub, Ein,
                      dOff.as<uint32_t>(), dDcs.as<int32_t>(), d_coefs.as<int16_t>(), d_bad.as<int>());
            gpu::sync(st);
            total_sub += nsub;
            total_iters += iters;
        }
        // Components never covered by a scan: libjpeg leaves zero coefficients and a latched table
        // is still needed for the IDCT.
        for (int i = 0; i < f.ncomp; ++i)
            if (!latched[i]) {
                if (!f.qt[f.comps[i].tq].defined) return Status::error(StatusCode::Corrupt, "component references undefined quant table");
                std::memcpy(F.comps[i].quant, f.qt[f.comps[i].tq].q.data(), sizeof(F.comps[i].quant));
            }
    }
    double entropy_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

    auto t2 = Clock::now();
    PJ_LAUNCH(k_idct, blocks_for(total_blocks, 128), 128u, st, F, d_coefs.as<int16_t>(), d_planes.as<uint8_t>(), total_blocks);
    gpu::sync(st);
    double idct_ms = std::chrono::duration<double, std::milli>(Clock::now() - t2).count();
    auto t3 = Clock::now();
    const uint64_t npix = uint64_t(f.width) * f.height;
    PJ_LAUNCH(k_color, blocks_for(npix, 256), 256u, st, F, d_planes.as<uint8_t>(), d_out.as<uint8_t>(), uint64_t(f.width) * ch,
              fmt_code(opts.format), opts.fancy_upsampling ? 1 : 0);
    gpu::d2h_2d(dst, stride, d_out.p, size_t(f.width) * ch, size_t(f.width) * ch, f.height, st);
    int bad = 0;
    gpu::d2h(&bad, d_bad.p, sizeof(int), st);
    gpu::sync(st);
    double color_ms = std::chrono::duration<double, std::milli>(Clock::now() - t3).count();
    std::string err = gpu::last_error();
    if (!err.empty()) return Status::error(StatusCode::Internal, "CUDA error: " + err);

    if (stats) {
        stats->parse_ms = parse_ms;
        stats->entropy_ms = entropy_ms;
        stats->idct_ms = idct_ms;
        stats->color_ms = color_ms;
        stats->total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        stats->subsequences = total_sub;
        stats->sync_iterations = total_iters;
        stats->restart_segments = es.restart_segments;
        stats->threads_used = 0;
        stats->backend = gpu::runtime_name();
        stats->warnings = f.warnings;
        if (bad || es.bad_code) stats->warnings.push_back("corrupt JPEG data: bad Huffman code");
    }
    if (f.truncated) return Status::error(StatusCode::Truncated, "stream truncated; output is best-effort");
    return Status::success();
}

}  // namespace parajpeg::detail
