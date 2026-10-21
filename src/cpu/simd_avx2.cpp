// AVX2 kernels (x86-64). This file is compiled with -mavx2 and only called after a runtime CPUID
// check (see simd_dispatch). Bit-exact with the scalar reference for all conforming input: the IDCT
// keeps libjpeg's 32-bit arithmetic in 8 lanes (one lane per column, then per row), and blocks whose
// outputs leave the range-limit window fall back to the scalar code.
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include <cstring>

#include "../core/color.h"
#include "../core/idct.h"

namespace parajpeg::detail {

namespace {
constexpr int CB = 13, P1 = 2;

__attribute__((target("avx2"))) inline __m256i mul(__m256i a, int c) { return _mm256_mullo_epi32(a, _mm256_set1_epi32(c)); }

template <int SHIFT>
__attribute__((target("avx2"))) inline __m256i descale(__m256i v) {
    return _mm256_srai_epi32(_mm256_add_epi32(v, _mm256_set1_epi32(1 << (SHIFT - 1))), SHIFT);
}

// True if every lane of the 8 vectors satisfies |x| <= 2^14 (32-bit IDCT arithmetic is then exact).
__attribute__((target("avx2"))) inline bool within_limit(const __m256i* v) {
    __m256i m = _mm256_setzero_si256();
    for (int i = 0; i < 8; ++i) m = _mm256_max_epu32(m, _mm256_abs_epi32(v[i]));
    __m256i over = _mm256_cmpgt_epi32(m, _mm256_set1_epi32(1 << 14));  // m < 2^31 as unsigned abs of int32 except INT_MIN
    __m256i neg = _mm256_cmpgt_epi32(_mm256_setzero_si256(), m);        // INT_MIN: abs stays negative
    __m256i bad = _mm256_or_si256(over, neg);
    return _mm256_testz_si256(bad, bad);
}

template <int SHIFT>
__attribute__((target("avx2"))) inline void idct8(const __m256i* x, __m256i* o) {
    __m256i z2 = x[2], z3 = x[6];
    __m256i z1 = mul(_mm256_add_epi32(z2, z3), 4433);
    __m256i tmp2 = _mm256_add_epi32(z1, mul(z3, -15137));
    __m256i tmp3 = _mm256_add_epi32(z1, mul(z2, 6270));
    __m256i tmp0 = _mm256_slli_epi32(_mm256_add_epi32(x[0], x[4]), CB);
    __m256i tmp1 = _mm256_slli_epi32(_mm256_sub_epi32(x[0], x[4]), CB);
    __m256i t10 = _mm256_add_epi32(tmp0, tmp3), t13 = _mm256_sub_epi32(tmp0, tmp3);
    __m256i t11 = _mm256_add_epi32(tmp1, tmp2), t12 = _mm256_sub_epi32(tmp1, tmp2);
    __m256i a0 = x[7], a1 = x[5], a2 = x[3], a3 = x[1];
    z1 = _mm256_add_epi32(a0, a3);
    z2 = _mm256_add_epi32(a1, a2);
    z3 = _mm256_add_epi32(a0, a2);
    __m256i z4 = _mm256_add_epi32(a1, a3);
    __m256i z5 = mul(_mm256_add_epi32(z3, z4), 9633);
    a0 = mul(a0, 2446); a1 = mul(a1, 16819); a2 = mul(a2, 25172); a3 = mul(a3, 12299);
    z1 = mul(z1, -7373); z2 = mul(z2, -20995);
    z3 = _mm256_add_epi32(mul(z3, -16069), z5);
    z4 = _mm256_add_epi32(mul(z4, -3196), z5);
    a0 = _mm256_add_epi32(a0, _mm256_add_epi32(z1, z3));
    a1 = _mm256_add_epi32(a1, _mm256_add_epi32(z2, z4));
    a2 = _mm256_add_epi32(a2, _mm256_add_epi32(z2, z3));
    a3 = _mm256_add_epi32(a3, _mm256_add_epi32(z1, z4));
    o[0] = descale<SHIFT>(_mm256_add_epi32(t10, a3)); o[7] = descale<SHIFT>(_mm256_sub_epi32(t10, a3));
    o[1] = descale<SHIFT>(_mm256_add_epi32(t11, a2)); o[6] = descale<SHIFT>(_mm256_sub_epi32(t11, a2));
    o[2] = descale<SHIFT>(_mm256_add_epi32(t12, a1)); o[5] = descale<SHIFT>(_mm256_sub_epi32(t12, a1));
    o[3] = descale<SHIFT>(_mm256_add_epi32(t13, a0)); o[4] = descale<SHIFT>(_mm256_sub_epi32(t13, a0));
}

__attribute__((target("avx2"))) inline void transpose8(__m256i* r) {
    __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20); r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20); r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20); r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20); r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}
}  // namespace

__attribute__((target("avx2"))) void idct_islow_avx2(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    __m256i x[8], w[8];
    for (int r = 0; r < 8; ++r) {
        __m256i c = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(coef + r * 8)));
        __m256i q = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(quant + r * 8)));
        x[r] = _mm256_mullo_epi32(c, q);
    }
    // libjpeg's all-zero-AC column shortcut yields exactly dc << PASS1, which the full computation
    // also produces, so it needs no special case. Out-of-range input takes the 64-bit scalar path.
    if (!within_limit(x)) {
        idct_islow_scalar(coef, quant, out, stride);
        return;
    }
    idct8<CB - P1>(x, w);   // lanes = columns, w[r] = row r
    if (!within_limit(w)) {
        idct_islow_scalar(coef, quant, out, stride);
        return;
    }
    transpose8(w);          // w[c] = column c, lanes = rows
    __m256i o[8];
    idct8<CB + P1 + 3>(w, o);  // o[c] = output column c, lanes = rows
    transpose8(o);          // o[r] = output row r
    // Range-limit window check: all values in [-512, 511] => table == saturate(v + 128).
    __m256i chk = _mm256_setzero_si256();
    for (int r = 0; r < 8; ++r)
        chk = _mm256_or_si256(chk, _mm256_and_si256(_mm256_add_epi32(o[r], _mm256_set1_epi32(512)), _mm256_set1_epi32(~1023)));
    if (!_mm256_testz_si256(chk, chk)) {
        idct_islow_scalar(coef, quant, out, stride);
        return;
    }
    const __m256i c128 = _mm256_set1_epi32(128);
    for (int r = 0; r < 8; r += 2) {
        __m256i a = _mm256_add_epi32(o[r], c128), b = _mm256_add_epi32(o[r + 1], c128);
        __m256i p16 = _mm256_packs_epi32(a, b);  // [a0..3 b0..3 | a4..7 b4..7]
        p16 = _mm256_permute4x64_epi64(p16, 0xD8);  // [a0..7 | b0..7]
        __m128i p8 = _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + r * stride), p8);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out + (r + 1) * stride), _mm_srli_si128(p8, 8));
    }
}

// Colour conversion, 8 pixels per step in 32-bit lanes (libjpeg jdcolor.c fixed point).
__attribute__((target("avx2"))) void ycc_to_rgb_row_avx2(const uint8_t* yp, const uint8_t* cbp, const uint8_t* crp, uint8_t* dst,
                                                         uint32_t width, int ch, bool bgr) {
    uint32_t i = 0;
    alignas(32) int32_t R[8], G[8], B[8];
    const __m256i half = _mm256_set1_epi32(32768), c128 = _mm256_set1_epi32(128);
    for (; i + 8 <= width; i += 8) {
        __m256i y = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(yp + i)));
        __m256i cb = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(cbp + i))), c128);
        __m256i cr = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(crp + i))), c128);
        __m256i r = _mm256_add_epi32(y, _mm256_srai_epi32(_mm256_add_epi32(mul(cr, 91881), half), 16));
        __m256i b = _mm256_add_epi32(y, _mm256_srai_epi32(_mm256_add_epi32(mul(cb, 116130), half), 16));
        __m256i g = _mm256_add_epi32(y, _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(mul(cb, -22554), mul(cr, -46802)), half), 16));
        const __m256i z = _mm256_setzero_si256(), m = _mm256_set1_epi32(255);
        _mm256_store_si256(reinterpret_cast<__m256i*>(R), _mm256_min_epi32(_mm256_max_epi32(r, z), m));
        _mm256_store_si256(reinterpret_cast<__m256i*>(G), _mm256_min_epi32(_mm256_max_epi32(g, z), m));
        _mm256_store_si256(reinterpret_cast<__m256i*>(B), _mm256_min_epi32(_mm256_max_epi32(b, z), m));
        uint8_t* o = dst + size_t(i) * ch;
        for (int k = 0; k < 8; ++k) {
            o[k * ch] = uint8_t(bgr ? B[k] : R[k]);
            o[k * ch + 1] = uint8_t(G[k]);
            o[k * ch + 2] = uint8_t(bgr ? R[k] : B[k]);
            if (ch == 4) o[k * ch + 3] = 255;
        }
    }
    if (i < width) ycc_to_rgb_row_scalar(yp + i, cbp + i, crp + i, dst + size_t(i) * ch, width - i, ch, bgr);
}

// Fancy upsampling, 16 input samples per step in 16-bit lanes; output interleaved with unpack.
__attribute__((target("avx2"))) void h2v1_fancy_row_avx2(const uint8_t* in, uint32_t in_w, uint8_t* out) {
    if (in_w < 34) { h2v1_fancy_scalar(in, in_w, out); return; }
    out[0] = in[0];
    out[1] = uint8_t((in[0] * 3 + in[1] + 2) >> 2);
    uint32_t c = 1;
    const __m256i one = _mm256_set1_epi16(1), two = _mm256_set1_epi16(2);
    for (; c + 16 <= in_w - 1; c += 16) {
        __m256i cur = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + c)));
        __m256i prev = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + c - 1)));
        __m256i next = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + c + 1)));
        __m256i t = _mm256_add_epi16(_mm256_add_epi16(cur, cur), cur);
        __m256i e = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(t, prev), one), 2);
        __m256i o = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(t, next), two), 2);
        // interleave e/o (16-bit, in-lane) then pack to bytes
        __m256i lo = _mm256_unpacklo_epi16(e, o), hi = _mm256_unpackhi_epi16(e, o);
        __m256i a = _mm256_permute2x128_si256(lo, hi, 0x20), b = _mm256_permute2x128_si256(lo, hi, 0x31);
        __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 2 * c), p);
    }
    for (; c < in_w - 1; ++c) {
        int v = in[c] * 3;
        out[2 * c] = uint8_t((v + in[c - 1] + 1) >> 2);
        out[2 * c + 1] = uint8_t((v + in[c + 1] + 2) >> 2);
    }
    out[2 * (in_w - 1)] = uint8_t((in[in_w - 1] * 3 + in[in_w - 2] + 1) >> 2);
    out[2 * (in_w - 1) + 1] = in[in_w - 1];
}

__attribute__((target("avx2"))) static inline __m256i colsum16(const uint8_t* p0, const uint8_t* p1) {
    __m256i a = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0)));
    __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p1)));
    return _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(a, a), a), b);
}

__attribute__((target("avx2"))) void h2v2_fancy_row_avx2(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out) {
    if (in_w < 34) { h2v2_fancy_scalar(in0, in1, in_w, out); return; }
    auto colsum = [&](uint32_t c) { return in0[c] * 3 + in1[c]; };
    int cs0 = colsum(0), cs1 = colsum(1);
    out[0] = uint8_t((cs0 * 4 + 8) >> 4);
    out[1] = uint8_t((cs0 * 3 + cs1 + 7) >> 4);
    uint32_t c = 1;
    const __m256i seven = _mm256_set1_epi16(7), eight = _mm256_set1_epi16(8);
    for (; c + 16 <= in_w - 1; c += 16) {
        __m256i cur = colsum16(in0 + c, in1 + c), prev = colsum16(in0 + c - 1, in1 + c - 1), next = colsum16(in0 + c + 1, in1 + c + 1);
        __m256i t = _mm256_add_epi16(_mm256_add_epi16(cur, cur), cur);
        __m256i e = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(t, prev), eight), 4);
        __m256i o = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(t, next), seven), 4);
        __m256i lo = _mm256_unpacklo_epi16(e, o), hi = _mm256_unpackhi_epi16(e, o);
        __m256i a = _mm256_permute2x128_si256(lo, hi, 0x20), b = _mm256_permute2x128_si256(lo, hi, 0x31);
        __m256i p = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 2 * c), p);
    }
    for (; c < in_w - 1; ++c) {
        int cur = colsum(c);
        out[2 * c] = uint8_t((cur * 3 + colsum(c - 1) + 8) >> 4);
        out[2 * c + 1] = uint8_t((cur * 3 + colsum(c + 1) + 7) >> 4);
    }
    int last = colsum(in_w - 1), before = colsum(in_w - 2);
    out[2 * (in_w - 1)] = uint8_t((last * 3 + before + 8) >> 4);
    out[2 * (in_w - 1) + 1] = uint8_t((last * 4 + 7) >> 4);
}

}  // namespace parajpeg::detail
#endif
