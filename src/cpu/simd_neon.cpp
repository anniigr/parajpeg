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
constexpr int32_t F0_298 = 2446, F0_390 = 3196, F0_541 = 4433, F0_765 = 6270, F0_899 = 7373, F1_175 = 9633,
                  F1_501 = 12299, F1_847 = 15137, F1_961 = 16069, F2_053 = 16819, F2_562 = 20995, F3_072 = 25172;

// One 1-D 8-point islow IDCT over 4 lanes (inputs x0..x7 are 4 independent columns/rows).
// Output is DESCALEd by `shift` using rounding shift (identical to libjpeg DESCALE).
template <int SHIFT>
inline void idct8_lanes(int32x4_t x0, int32x4_t x1, int32x4_t x2, int32x4_t x3, int32x4_t x4, int32x4_t x5,
                        int32x4_t x6, int32x4_t x7, int32x4_t* o) {
    // Even part
    int32x4_t z2 = x2, z3 = x6;
    int32x4_t z1 = vmulq_n_s32(vaddq_s32(z2, z3), F0_541);
    int32x4_t tmp2 = vmlaq_n_s32(z1, z3, -F1_847);
    int32x4_t tmp3 = vmlaq_n_s32(z1, z2, F0_765);
    int32x4_t tmp0 = vshlq_n_s32(vaddq_s32(x0, x4), CONST_BITS);
    int32x4_t tmp1 = vshlq_n_s32(vsubq_s32(x0, x4), CONST_BITS);
    int32x4_t tmp10 = vaddq_s32(tmp0, tmp3), tmp13 = vsubq_s32(tmp0, tmp3);
    int32x4_t tmp11 = vaddq_s32(tmp1, tmp2), tmp12 = vsubq_s32(tmp1, tmp2);
    // Odd part
    int32x4_t t0 = x7, t1 = x5, t2 = x3, t3 = x1;
    z1 = vaddq_s32(t0, t3);
    z2 = vaddq_s32(t1, t2);
    z3 = vaddq_s32(t0, t2);
    int32x4_t z4 = vaddq_s32(t1, t3);
    int32x4_t z5 = vmulq_n_s32(vaddq_s32(z3, z4), F1_175);
    t0 = vmulq_n_s32(t0, F0_298);
    t1 = vmulq_n_s32(t1, F2_053);
    t2 = vmulq_n_s32(t2, F3_072);
    t3 = vmulq_n_s32(t3, F1_501);
    z1 = vmulq_n_s32(z1, -F0_899);
    z2 = vmulq_n_s32(z2, -F2_562);
    z3 = vmlaq_n_s32(z5, z3, -F1_961);
    z4 = vmlaq_n_s32(z5, z4, -F0_390);
    t0 = vaddq_s32(t0, vaddq_s32(z1, z3));
    t1 = vaddq_s32(t1, vaddq_s32(z2, z4));
    t2 = vaddq_s32(t2, vaddq_s32(z2, z3));
    t3 = vaddq_s32(t3, vaddq_s32(z1, z4));
    o[0] = vrshrq_n_s32(vaddq_s32(tmp10, t3), SHIFT);
    o[7] = vrshrq_n_s32(vsubq_s32(tmp10, t3), SHIFT);
    o[1] = vrshrq_n_s32(vaddq_s32(tmp11, t2), SHIFT);
    o[6] = vrshrq_n_s32(vsubq_s32(tmp11, t2), SHIFT);
    o[2] = vrshrq_n_s32(vaddq_s32(tmp12, t1), SHIFT);
    o[5] = vrshrq_n_s32(vsubq_s32(tmp12, t1), SHIFT);
    o[3] = vrshrq_n_s32(vaddq_s32(tmp13, t0), SHIFT);
    o[4] = vrshrq_n_s32(vsubq_s32(tmp13, t0), SHIFT);
}

inline bool within_limit(const int32x4_t* a, const int32x4_t* b) {
    uint32x4_t m = vdupq_n_u32(0);
    for (int i = 0; i < 8; ++i) m = vmaxq_u32(m, vmaxq_u32(vreinterpretq_u32_s32(vabsq_s32(a[i])), vreinterpretq_u32_s32(vabsq_s32(b[i]))));
    return vmaxvq_u32(m) <= (1u << 14);
}

// 4x4 transpose of int32 lanes.
inline void transpose4(int32x4_t& a, int32x4_t& b, int32x4_t& c, int32x4_t& d) {
    int32x4x2_t ab = vtrnq_s32(a, b);
    int32x4x2_t cd = vtrnq_s32(c, d);
    a = vcombine_s32(vget_low_s32(ab.val[0]), vget_low_s32(cd.val[0]));
    b = vcombine_s32(vget_low_s32(ab.val[1]), vget_low_s32(cd.val[1]));
    c = vcombine_s32(vget_high_s32(ab.val[0]), vget_high_s32(cd.val[0]));
    d = vcombine_s32(vget_high_s32(ab.val[1]), vget_high_s32(cd.val[1]));
}

// libjpeg range limit on (v & 1023): v in [-512, 511] saturates to [0,255] after +128; outside
// that window the table wraps. Valid streams stay inside; corrupt ones must still match, so apply
// the table semantics exactly: idx = v & 1023; idx<128 -> 128+idx; <512 -> 255; <896 -> 0; else idx-896.
inline uint8x8_t range_limit_exact(int32x4_t lo, int32x4_t hi) {
    int32x4_t mask = vdupq_n_s32(1023);
    lo = vandq_s32(lo, mask);
    hi = vandq_s32(hi, mask);
    uint16x8_t idx = vcombine_u16(vmovn_u32(vreinterpretq_u32_s32(lo)), vmovn_u32(vreinterpretq_u32_s32(hi)));
    uint16x8_t r1 = vaddq_u16(idx, vdupq_n_u16(128));
    uint16x8_t r4 = vsubq_u16(idx, vdupq_n_u16(896));
    uint16x8_t lt128 = vcltq_u16(idx, vdupq_n_u16(128));
    uint16x8_t lt512 = vcltq_u16(idx, vdupq_n_u16(512));
    uint16x8_t lt896 = vcltq_u16(idx, vdupq_n_u16(896));
    uint16x8_t r = vbslq_u16(lt128, r1, vbslq_u16(lt512, vdupq_n_u16(255), vbslq_u16(lt896, vdupq_n_u16(0), r4)));
    return vmovn_u16(r);
}
// Fast path valid when every value is within [-512, 511]: table == saturate(v + 128).
inline uint8x8_t range_limit_fast(int32x4_t lo, int32x4_t hi) {
    int16x8_t v = vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi));
    return vqmovun_s16(vaddq_s16(v, vdupq_n_s16(128)));
}
// OR-accumulate (v + 512) & ~1023 to detect any out-of-window lane.
inline int32x4_t window_check(int32x4_t acc, int32x4_t v) {
    return vorrq_s32(acc, vandq_s32(vaddq_s32(v, vdupq_n_s32(512)), vdupq_n_s32(~1023)));
}
}  // namespace

void idct_islow_neon(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    // Zero-AC detection (raw coefficients, as libjpeg does): whole block, and the cols 4-7 half.
    int16x8_t b0 = vld1q_s16(coef), b1 = vld1q_s16(coef + 8), b2 = vld1q_s16(coef + 16), b3 = vld1q_s16(coef + 24);
    int16x8_t b4 = vld1q_s16(coef + 32), b5 = vld1q_s16(coef + 40), b6 = vld1q_s16(coef + 48), b7 = vld1q_s16(coef + 56);
    int16x8_t ac_rows = vorrq_s16(vorrq_s16(vorrq_s16(b1, b2), vorrq_s16(b3, b4)), vorrq_s16(vorrq_s16(b5, b6), b7));
    int16x8_t all = vorrq_s16(ac_rows, vcombine_s16(vand_s16(vget_low_s16(b0), vcreate_s16(0xFFFFFFFFFFFF0000ull)), vget_high_s16(b0)));
    if (vmaxvq_u16(vreinterpretq_u16_s16(all)) == 0) {
        // DC-only block: every output sample equals range_limit(DESCALE(dc * q << PASS1_BITS, PASS1_BITS + 3)).
        int32_t dc = int32_t(coef[0]) * int32_t(quant[0]);
        int32_t v = ((dc * (1 << PASS1_BITS)) + (1 << (PASS1_BITS + 2))) >> (PASS1_BITS + 3);
        uint8_t px = idct_range_limit_table()[v & 1023];
        uint8x8_t row = vdup_n_u8(px);
        for (int r = 0; r < 8; ++r) vst1_u8(out + r * stride, row);
        return;
    }
    const bool hi_ac_zero = vmaxv_u16(vreinterpret_u16_s16(vget_high_s16(ac_rows))) == 0;
    // Dequantise rows into int32 lanes: row r -> (lo: cols 0-3, hi: cols 4-7)
    int32x4_t lo[8], hi[8];
    const int16x8_t cs[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
    for (int r = 0; r < 8; ++r) {
        uint16x8_t q = vld1q_u16(quant + r * 8);
        int32x4_t ql = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(q)));
        int32x4_t qh = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(q)));
        lo[r] = vmulq_s32(vmovl_s16(vget_low_s16(cs[r])), ql);
        hi[r] = vmulq_s32(vmovl_s16(vget_high_s16(cs[r])), qh);
    }
    // 32-bit lanes are exact while every pass input satisfies |x| <= 2^14 (worst-case intermediate
    // < 1.8e9 < 2^31). Conforming 8-bit JPEGs stay far below; anything else takes the 64-bit path.
    if (!within_limit(lo, hi)) {
        idct_islow_core(coef, quant, out, stride);
        return;
    }
    // Pass 1: columns. Lanes are columns; the 8 inputs are the 8 rows.
    int32x4_t wl[8], wh[8];
    idct8_lanes<CONST_BITS - PASS1_BITS>(lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7], wl);
    if (hi_ac_zero) {
        // Columns 4-7 have no AC terms: pass-1 output is dc << PASS1_BITS (exactly what the full
        // computation yields, see the C shortcut).
        int32x4_t dcv = vshlq_n_s32(hi[0], PASS1_BITS);
        for (int r = 0; r < 8; ++r) wh[r] = dcv;
    } else {
        idct8_lanes<CONST_BITS - PASS1_BITS>(hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7], wh);
    }
    if (!within_limit(wl, wh)) {
        idct_islow_core(coef, quant, out, stride);
        return;
    }
    // wl[r] = row r, cols 0-3 ; wh[r] = row r, cols 4-7. Transpose so lanes become rows.
    // Rows 0-3: a = cols0-3 of rows 0..3 -> after transpose, lane = row, vector index = col
    int32x4_t c0 = wl[0], c1 = wl[1], c2 = wl[2], c3 = wl[3];
    transpose4(c0, c1, c2, c3);  // c0 = col0 rows0-3, c1 = col1 rows0-3, ...
    int32x4_t c4 = wh[0], c5 = wh[1], c6 = wh[2], c7 = wh[3];
    transpose4(c4, c5, c6, c7);  // c4 = col4 rows0-3 ...
    int32x4_t d0 = wl[4], d1 = wl[5], d2 = wl[6], d3 = wl[7];
    transpose4(d0, d1, d2, d3);  // col0..3 rows4-7
    int32x4_t d4 = wh[4], d5 = wh[5], d6 = wh[6], d7 = wh[7];
    transpose4(d4, d5, d6, d7);  // col4..7 rows4-7
    // Pass 2: rows. For rows 0-3 inputs are c0..c7 (lane = row), for rows 4-7 d0..d7.
    int32x4_t o03[8], o47[8];
    idct8_lanes<CONST_BITS + PASS1_BITS + 3>(c0, c1, c2, c3, c4, c5, c6, c7, o03);
    idct8_lanes<CONST_BITS + PASS1_BITS + 3>(d0, d1, d2, d3, d4, d5, d6, d7, o47);
    // o03[j] = output col j for rows 0-3 (lanes). Transpose back to rows.
    int32x4_t r0 = o03[0], r1 = o03[1], r2 = o03[2], r3 = o03[3];
    transpose4(r0, r1, r2, r3);  // r0 = row0 cols0-3, r1 = row1 cols0-3, ...
    int32x4_t r4 = o03[4], r5 = o03[5], r6 = o03[6], r7 = o03[7];
    transpose4(r4, r5, r6, r7);  // r4 = row0 cols4-7, r5 = row1 cols4-7 ...
    int32x4_t s0 = o47[0], s1 = o47[1], s2 = o47[2], s3 = o47[3];
    transpose4(s0, s1, s2, s3);
    int32x4_t s4 = o47[4], s5 = o47[5], s6 = o47[6], s7 = o47[7];
    transpose4(s4, s5, s6, s7);
    int32x4_t chk = vdupq_n_s32(0);
    chk = window_check(chk, r0); chk = window_check(chk, r1); chk = window_check(chk, r2); chk = window_check(chk, r3);
    chk = window_check(chk, r4); chk = window_check(chk, r5); chk = window_check(chk, r6); chk = window_check(chk, r7);
    chk = window_check(chk, s0); chk = window_check(chk, s1); chk = window_check(chk, s2); chk = window_check(chk, s3);
    chk = window_check(chk, s4); chk = window_check(chk, s5); chk = window_check(chk, s6); chk = window_check(chk, s7);
    if (__builtin_expect(vmaxvq_u32(vreinterpretq_u32_s32(chk)) == 0, 1)) {
        vst1_u8(out + 0 * stride, range_limit_fast(r0, r4));
        vst1_u8(out + 1 * stride, range_limit_fast(r1, r5));
        vst1_u8(out + 2 * stride, range_limit_fast(r2, r6));
        vst1_u8(out + 3 * stride, range_limit_fast(r3, r7));
        vst1_u8(out + 4 * stride, range_limit_fast(s0, s4));
        vst1_u8(out + 5 * stride, range_limit_fast(s1, s5));
        vst1_u8(out + 6 * stride, range_limit_fast(s2, s6));
        vst1_u8(out + 7 * stride, range_limit_fast(s3, s7));
    } else {
        vst1_u8(out + 0 * stride, range_limit_exact(r0, r4));
        vst1_u8(out + 1 * stride, range_limit_exact(r1, r5));
        vst1_u8(out + 2 * stride, range_limit_exact(r2, r6));
        vst1_u8(out + 3 * stride, range_limit_exact(r3, r7));
        vst1_u8(out + 4 * stride, range_limit_exact(s0, s4));
        vst1_u8(out + 5 * stride, range_limit_exact(s1, s5));
        vst1_u8(out + 6 * stride, range_limit_exact(s2, s6));
        vst1_u8(out + 7 * stride, range_limit_exact(s3, s7));
    }
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
