// NEON implementations, bit-exact with the scalar reference by construction: the IDCT uses the same
// 32-bit integer arithmetic as jidctint.c (rounding shifts are exact), colour conversion evaluates
// libjpeg's fixed-point formulas in 32-bit lanes, and the fancy upsamplers use the same 16-bit sums.
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

#include "../core/color.h"
#include "../core/idct.h"
#include "../common/idct_core.h"

namespace parajpeg::detail {

namespace {
constexpr int CONST_BITS = 13;
constexpr int PASS1_BITS = 2;
constexpr int16_t F0_298 = 2446, F0_390 = 3196, F0_541 = 4433, F0_765 = 6270, F0_899 = 7373, F1_175 = 9633,
                  F1_501 = 12299, F1_847 = 15137, F1_961 = 16069, F2_053 = 16819, F2_562 = 20995, F3_072 = 25172;

template <int SHIFT>
inline int16x4_t descale_narrow(int32x4_t v) {
    if constexpr (SHIFT <= 16) return vqrshrn_n_s32(v, SHIFT);
    else return vqmovn_s32(vrshrq_n_s32(v, SHIFT));
}

// One 1-D 8-point islow IDCT on four 16-bit lanes (x[0..7] are the 8 inputs of four independent
// transforms), 32-bit products via widening multiplies, result rounded, shifted and narrowed with
// saturation. Arithmetic is libjpeg's, in 32 bits; the callers guarantee |x| <= 2^14, for which no
// 32-bit intermediate can overflow (worst case < 1.8e9), so the result equals the 64-bit reference.
template <int SHIFT, bool kSparse>
inline void idct8_s16(const int16x4_t* x, int16x4_t* o) {
    // Even part.
    int32x4_t z1 = kSparse ? vmull_n_s16(x[2], F0_541) : vmull_n_s16(vadd_s16(x[2], x[6]), F0_541);
    // x2 + x6 may exceed int16 for |x| = 2^14: widen instead.
    if (!kSparse) z1 = vmulq_n_s32(vaddl_s16(x[2], x[6]), F0_541);
    int32x4_t tmp2 = kSparse ? z1 : vmlal_n_s16(z1, x[6], int16_t(-F1_847));
    int32x4_t tmp3 = vmlal_n_s16(z1, x[2], F0_765);
    int32x4_t tmp0 = kSparse ? vshll_n_s16(x[0], CONST_BITS) : vshlq_n_s32(vaddl_s16(x[0], x[4]), CONST_BITS);
    int32x4_t tmp1 = kSparse ? tmp0 : vshlq_n_s32(vsubl_s16(x[0], x[4]), CONST_BITS);
    int32x4_t t10 = vaddq_s32(tmp0, tmp3), t13 = vsubq_s32(tmp0, tmp3);
    int32x4_t t11 = vaddq_s32(tmp1, tmp2), t12 = vsubq_s32(tmp1, tmp2);
    // Odd part: tmp0..3 = x7, x5, x3, x1.
    int32x4_t a0, a1, a2, a3, zz1, zz2, zz3, zz4;
    if (kSparse) {
        // x5 = x7 = 0: z1 = x1, z2 = z3 = x3, z4 = x1.
        int32x4_t z5 = vmulq_n_s32(vaddl_s16(x[3], x[1]), F1_175);
        a2 = vmull_n_s16(x[3], F3_072);
        a3 = vmull_n_s16(x[1], F1_501);
        zz1 = vmull_n_s16(x[1], int16_t(-F0_899));
        zz2 = vmull_n_s16(x[3], int16_t(-F2_562));
        zz3 = vmlal_n_s16(z5, x[3], int16_t(-F1_961));
        zz4 = vmlal_n_s16(z5, x[1], int16_t(-F0_390));
        a0 = vaddq_s32(zz1, zz3);
        a1 = vaddq_s32(zz2, zz4);
    } else {
        int32x4_t s1 = vaddl_s16(x[7], x[1]), s2 = vaddl_s16(x[5], x[3]), s3 = vaddl_s16(x[7], x[3]), s4 = vaddl_s16(x[5], x[1]);
        int32x4_t z5 = vmulq_n_s32(vaddq_s32(s3, s4), F1_175);
        a0 = vmull_n_s16(x[7], F0_298);
        a1 = vmull_n_s16(x[5], F2_053);
        a2 = vmull_n_s16(x[3], F3_072);
        a3 = vmull_n_s16(x[1], F1_501);
        zz1 = vmulq_n_s32(s1, -F0_899);
        zz2 = vmulq_n_s32(s2, -F2_562);
        zz3 = vmlaq_n_s32(z5, s3, -F1_961);
        zz4 = vmlaq_n_s32(z5, s4, -F0_390);
        a0 = vaddq_s32(a0, vaddq_s32(zz1, zz3));
        a1 = vaddq_s32(a1, vaddq_s32(zz2, zz4));
    }
    a2 = vaddq_s32(a2, vaddq_s32(zz2, zz3));
    a3 = vaddq_s32(a3, vaddq_s32(zz1, zz4));
    o[0] = descale_narrow<SHIFT>(vaddq_s32(t10, a3));
    o[7] = descale_narrow<SHIFT>(vsubq_s32(t10, a3));
    o[1] = descale_narrow<SHIFT>(vaddq_s32(t11, a2));
    o[6] = descale_narrow<SHIFT>(vsubq_s32(t11, a2));
    o[2] = descale_narrow<SHIFT>(vaddq_s32(t12, a1));
    o[5] = descale_narrow<SHIFT>(vsubq_s32(t12, a1));
    o[3] = descale_narrow<SHIFT>(vaddq_s32(t13, a0));
    o[4] = descale_narrow<SHIFT>(vsubq_s32(t13, a0));
}

// 8x8 transpose of int16 values held as rows r[i] = (lo half, hi half).
inline void transpose8x8(int16x8_t* r) {
    int16x8x2_t a0 = vtrnq_s16(r[0], r[1]), a1 = vtrnq_s16(r[2], r[3]), a2 = vtrnq_s16(r[4], r[5]), a3 = vtrnq_s16(r[6], r[7]);
    int32x4x2_t b0 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[0]), vreinterpretq_s32_s16(a1.val[0]));
    int32x4x2_t b1 = vtrnq_s32(vreinterpretq_s32_s16(a0.val[1]), vreinterpretq_s32_s16(a1.val[1]));
    int32x4x2_t b2 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[0]), vreinterpretq_s32_s16(a3.val[0]));
    int32x4x2_t b3 = vtrnq_s32(vreinterpretq_s32_s16(a2.val[1]), vreinterpretq_s32_s16(a3.val[1]));
    r[0] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b0.val[0]), vget_low_s32(b2.val[0])));
    r[4] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b0.val[0]), vget_high_s32(b2.val[0])));
    r[1] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b1.val[0]), vget_low_s32(b3.val[0])));
    r[5] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.val[0]), vget_high_s32(b3.val[0])));
    r[2] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b0.val[1]), vget_low_s32(b2.val[1])));
    r[6] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b0.val[1]), vget_high_s32(b2.val[1])));
    r[3] = vreinterpretq_s16_s32(vcombine_s32(vget_low_s32(b1.val[1]), vget_low_s32(b3.val[1])));
    r[7] = vreinterpretq_s16_s32(vcombine_s32(vget_high_s32(b1.val[1]), vget_high_s32(b3.val[1])));
}

// One pass over 8 rows of 8 int16 values: lanes are independent transforms.
template <int SHIFT>
inline void pass(int16x8_t* v, bool sparse_hi_rows, bool hi_lanes_zero) {
    int16x4_t lo[8], hi[8], olo[8], ohi[8];
    for (int i = 0; i < 8; ++i) { lo[i] = vget_low_s16(v[i]); hi[i] = vget_high_s16(v[i]); }
    if (sparse_hi_rows) idct8_s16<SHIFT, true>(lo, olo);
    else idct8_s16<SHIFT, false>(lo, olo);
    if (hi_lanes_zero) {
        // All-zero inputs give all-zero outputs (every term is a multiple of an input).
        for (int i = 0; i < 8; ++i) ohi[i] = vdup_n_s16(0);
    } else if (sparse_hi_rows) {
        idct8_s16<SHIFT, true>(hi, ohi);
    } else {
        idct8_s16<SHIFT, false>(hi, ohi);
    }
    for (int i = 0; i < 8; ++i) v[i] = vcombine_s16(olo[i], ohi[i]);
}

inline uint16_t max_abs(const int16x8_t* v) {
    uint16x8_t m = vreinterpretq_u16_s16(vabsq_s16(v[0]));
    for (int i = 1; i < 8; ++i) m = vmaxq_u16(m, vreinterpretq_u16_s16(vabsq_s16(v[i])));
    return vmaxvq_u16(m);  // INT16_MIN maps to 0x8000 (as unsigned) and fails any limit check
}

inline bool rows_zero(const int16x8_t* v, int from) {
    int16x8_t acc = v[from];
    for (int i = from + 1; i < 8; ++i) acc = vorrq_s16(acc, v[i]);
    return vmaxvq_u16(vreinterpretq_u16_s16(acc)) == 0;
}
}  // namespace

void idct_islow_neon(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    int16x8_t c[8];
    for (int r = 0; r < 8; ++r) c[r] = vld1q_s16(coef + r * 8);
    // DC-only block (raw coefficients, as libjpeg tests them).
    {
        int16x8_t ac = vorrq_s16(vsetq_lane_s16(0, c[0], 0), c[1]);
        for (int r = 2; r < 8; ++r) ac = vorrq_s16(ac, c[r]);
        if (vmaxvq_u16(vreinterpretq_u16_s16(ac)) == 0) {
            int32_t dc = int32_t(coef[0]) * int32_t(quant[0]);
            int32_t v = ((dc * (1 << PASS1_BITS)) + (1 << (PASS1_BITS + 2))) >> (PASS1_BITS + 3);
            uint8x8_t row = vdup_n_u8(idct_range_limit_table()[v & 1023]);
            for (int r = 0; r < 8; ++r) vst1_u8(out + r * stride, row);
            return;
        }
    }
    // Dequantise in 16 bits, exact when every |coef * q| <= 2^14 (bounded by max|coef| * max q).
    uint16x8_t qv[8];
    for (int r = 0; r < 8; ++r) qv[r] = vld1q_u16(quant + r * 8);
    uint16x8_t qm = vmaxq_u16(vmaxq_u16(vmaxq_u16(qv[0], qv[1]), vmaxq_u16(qv[2], qv[3])), vmaxq_u16(vmaxq_u16(qv[4], qv[5]), vmaxq_u16(qv[6], qv[7])));
    const uint32_t bound = uint32_t(max_abs(c)) * vmaxvq_u16(qm);
    if (bound > (1u << 14)) {
        idct_islow_core(coef, quant, out, stride);
        return;
    }
    const bool sparse = rows_zero(c, 4);
    bool hi_cols_zero;
    {
        int16x8_t acc = c[0];
        for (int r = 1; r < 8; ++r) acc = vorrq_s16(acc, c[r]);
        hi_cols_zero = vmaxv_u16(vreinterpret_u16_s16(vget_high_s16(acc))) == 0;
    }
    for (int r = 0; r < 8; ++r) c[r] = vmulq_s16(c[r], vreinterpretq_s16_u16(qv[r]));
    // Pass 1 (columns: lanes are columns, inputs are rows).
    pass<CONST_BITS - PASS1_BITS>(c, sparse, hi_cols_zero);
    if (max_abs(c) > (1u << 14)) {  // saturated narrowing is caught here too
        idct_islow_core(coef, quant, out, stride);
        return;
    }
    transpose8x8(c);  // c[i] = column i of the workspace, lanes = rows
    const bool sparse2 = rows_zero(c, 4);
    // Pass 2 (rows: lanes are rows, inputs are columns).
    pass<CONST_BITS + PASS1_BITS + 3>(c, sparse2, false);
    // Range limit: exact fast path when every value lies in [-512, 511].
    if (max_abs(c) > 511) {
        idct_islow_core(coef, quant, out, stride);
        return;
    }
    transpose8x8(c);  // back to rows
    const int16x8_t c128 = vdupq_n_s16(128);
    for (int r = 0; r < 8; ++r) vst1_u8(out + r * stride, vqmovun_s16(vaddq_s16(c[r], c128)));
}

// ---- Colour conversion -----------------------------------------------------------------------------
// libjpeg: R = y + ((FIX(1.402)*cr + ONE_HALF) >> 16), B = y + ((FIX(1.772)*cb + ONE_HALF) >> 16),
//          G = y + ((-FIX(0.34414)*cb - FIX(0.71414)*cr + ONE_HALF) >> 16), cb/cr centred at 0.
// With FIX(1.402) = 91881 = 65536 + 26345 and floor((x*65536 + y) / 65536) = x + floor(y / 65536):
//   R = y + cr + rshr16(26345*cr);  B = y + 2*cb + rshr16(-14942*cb)  (116130 = 2*65536 - 14942);
//   G = y - cr + rshr16(18734*cr - 22554*cb)  (-46802 = -65536 + 18734).
// rshr16(v) = (v + 32768) >> 16 is the NEON rounding narrowing shift, and all products fit int32.
namespace {
inline uint8x8_t ycc8_u8(uint8x8_t y8, uint8x8_t cb8, uint8x8_t cr8, uint8x8_t& R, uint8x8_t& G, uint8x8_t& B) {
    int16x8_t y = vreinterpretq_s16_u16(vmovl_u8(y8));
    int16x8_t cb = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cb8)), vdupq_n_s16(128));
    int16x8_t cr = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(cr8)), vdupq_n_s16(128));
    int16x4_t cbl = vget_low_s16(cb), cbh = vget_high_s16(cb), crl = vget_low_s16(cr), crh = vget_high_s16(cr);
    int16x8_t rr = vcombine_s16(vrshrn_n_s32(vmull_n_s16(crl, 26345), 16), vrshrn_n_s32(vmull_n_s16(crh, 26345), 16));
    int16x8_t bb = vcombine_s16(vrshrn_n_s32(vmull_n_s16(cbl, -14942), 16), vrshrn_n_s32(vmull_n_s16(cbh, -14942), 16));
    int16x8_t gg = vcombine_s16(vrshrn_n_s32(vmlal_n_s16(vmull_n_s16(crl, 18734), cbl, -22554), 16),
                                vrshrn_n_s32(vmlal_n_s16(vmull_n_s16(crh, 18734), cbh, -22554), 16));
    int16x8_t r = vaddq_s16(vaddq_s16(y, cr), rr);
    int16x8_t b = vaddq_s16(vaddq_s16(y, vshlq_n_s16(cb, 1)), bb);
    int16x8_t g = vaddq_s16(vsubq_s16(y, cr), gg);
    R = vqmovun_s16(r);
    G = vqmovun_s16(g);
    B = vqmovun_s16(b);
    return R;
}
}  // namespace

void ycc_to_rgb_row_neon(const uint8_t* yp, const uint8_t* cbp, const uint8_t* crp, uint8_t* dst, uint32_t width, int ch, bool bgr) {
    uint32_t i = 0;
    for (; i + 16 <= width; i += 16) {
        uint8x16_t y = vld1q_u8(yp + i), cb = vld1q_u8(cbp + i), cr = vld1q_u8(crp + i);
        uint8x8_t R0, G0, B0, R1, G1, B1;
        ycc8_u8(vget_low_u8(y), vget_low_u8(cb), vget_low_u8(cr), R0, G0, B0);
        ycc8_u8(vget_high_u8(y), vget_high_u8(cb), vget_high_u8(cr), R1, G1, B1);
        uint8x16_t R = vcombine_u8(R0, R1), G = vcombine_u8(G0, G1), B = vcombine_u8(B0, B1);
        if (ch == 3) {
            uint8x16x3_t v = {{bgr ? B : R, G, bgr ? R : B}};
            vst3q_u8(dst + i * 3, v);
        } else {
            uint8x16x4_t v = {{bgr ? B : R, G, bgr ? R : B, vdupq_n_u8(255)}};
            vst4q_u8(dst + i * 4, v);
        }
    }
    if (i < width) ycc_to_rgb_row_scalar(yp + i, cbp + i, crp + i, dst + size_t(i) * ch, width - i, ch, bgr);
}

// ---- Fancy upsampling ------------------------------------------------------------------------------

void h2v1_fancy_row_neon(const uint8_t* in, uint32_t in_w, uint8_t* out) {
    if (in_w < 18) { h2v1_fancy_scalar(in, in_w, out); return; }
    // First output pair (scalar, libjpeg edge rule)
    out[0] = in[0];
    out[1] = uint8_t((in[0] * 3 + in[1] + 2) >> 2);
    // Middle: for input column c (1..in_w-2): out[2c] = (3*in[c] + in[c-1] + 1) >> 2 ; out[2c+1] = (3*in[c] + in[c+1] + 2) >> 2
    uint32_t c = 1;
    for (; c + 8 <= in_w - 1; c += 8) {
        uint16x8_t cur = vmovl_u8(vld1_u8(in + c));
        uint16x8_t prev = vmovl_u8(vld1_u8(in + c - 1));
        uint16x8_t next = vmovl_u8(vld1_u8(in + c + 1));
        uint16x8_t t = vmulq_n_u16(cur, 3);
        uint8x8_t e = vshrn_n_u16(vaddq_u16(vaddq_u16(t, prev), vdupq_n_u16(1)), 2);
        uint8x8_t o = vshrn_n_u16(vaddq_u16(vaddq_u16(t, next), vdupq_n_u16(2)), 2);
        uint8x8x2_t z = {{e, o}};
        vst2_u8(out + 2 * c, z);
    }
    for (; c < in_w - 1; ++c) {
        int v = in[c] * 3;
        out[2 * c] = uint8_t((v + in[c - 1] + 1) >> 2);
        out[2 * c + 1] = uint8_t((v + in[c + 1] + 2) >> 2);
    }
    out[2 * (in_w - 1)] = uint8_t((in[in_w - 1] * 3 + in[in_w - 2] + 1) >> 2);
    out[2 * (in_w - 1) + 1] = in[in_w - 1];
}

void h2v2_fancy_row_neon(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out) {
    if (in_w < 18) { h2v2_fancy_scalar(in0, in1, in_w, out); return; }
    // colsum[c] = 3*in0[c] + in1[c]. out[2c] = (3*colsum[c] + colsum[c-1] + 8) >> 4 ; out[2c+1] = (3*colsum[c] + colsum[c+1] + 7) >> 4
    auto colsum = [&](uint32_t c) { return in0[c] * 3 + in1[c]; };
    int cs0 = colsum(0), cs1 = colsum(1);
    out[0] = uint8_t((cs0 * 4 + 8) >> 4);
    out[1] = uint8_t((cs0 * 3 + cs1 + 7) >> 4);
    uint32_t c = 1;
    for (; c + 8 <= in_w - 1; c += 8) {
        uint16x8_t cur = vmlaq_n_u16(vmovl_u8(vld1_u8(in1 + c)), vmovl_u8(vld1_u8(in0 + c)), 3);
        uint16x8_t prev = vmlaq_n_u16(vmovl_u8(vld1_u8(in1 + c - 1)), vmovl_u8(vld1_u8(in0 + c - 1)), 3);
        uint16x8_t next = vmlaq_n_u16(vmovl_u8(vld1_u8(in1 + c + 1)), vmovl_u8(vld1_u8(in0 + c + 1)), 3);
        uint16x8_t t = vmulq_n_u16(cur, 3);
        uint8x8_t e = vshrn_n_u16(vaddq_u16(vaddq_u16(t, prev), vdupq_n_u16(8)), 4);
        uint8x8_t o = vshrn_n_u16(vaddq_u16(vaddq_u16(t, next), vdupq_n_u16(7)), 4);
        uint8x8x2_t z = {{e, o}};
        vst2_u8(out + 2 * c, z);
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
