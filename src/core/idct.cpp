#include "idct.h"

#include <cstddef>

namespace parajpeg::detail {

namespace {
constexpr int CONST_BITS = 13;
constexpr int PASS1_BITS = 2;
constexpr int32_t FIX_0_298631336 = 2446;
constexpr int32_t FIX_0_390180644 = 3196;
constexpr int32_t FIX_0_541196100 = 4433;
constexpr int32_t FIX_0_765366865 = 6270;
constexpr int32_t FIX_0_899976223 = 7373;
constexpr int32_t FIX_1_175875602 = 9633;
constexpr int32_t FIX_1_501321110 = 12299;
constexpr int32_t FIX_1_847759065 = 15137;
constexpr int32_t FIX_1_961570560 = 16069;
constexpr int32_t FIX_2_053119869 = 16819;
constexpr int32_t FIX_2_562915447 = 20995;
constexpr int32_t FIX_3_072711026 = 25172;

inline int32_t DESCALE(int32_t x, int n) { return (x + (1 << (n - 1))) >> n; }

struct RangeTable {
    uint8_t t[1024];
    RangeTable() {
        // IDCT_range_limit semantics (see jdmaster.c prepare_range_limit_table):
        // idx 0..127 -> 128..255 ; 128..511 -> 255 ; 512..895 -> 0 ; 896..1023 -> 0..127
        for (int i = 0; i < 128; ++i) t[i] = uint8_t(128 + i);
        for (int i = 128; i < 512; ++i) t[i] = 255;
        for (int i = 512; i < 896; ++i) t[i] = 0;
        for (int i = 896; i < 1024; ++i) t[i] = uint8_t(i - 896);
    }
};
const RangeTable g_range;
}  // namespace

const uint8_t* idct_range_limit_table() { return g_range.t; }

void idct_islow_scalar(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    int32_t ws[64];
    const uint8_t* rl = g_range.t;
    // Pass 1: columns.
    for (int c = 0; c < 8; ++c) {
        const int16_t* in = coef + c;
        const uint16_t* q = quant + c;
        int32_t* w = ws + c;
        if (in[8] == 0 && in[16] == 0 && in[24] == 0 && in[32] == 0 && in[40] == 0 && in[48] == 0 && in[56] == 0) {
            int32_t dc = int32_t(in[0] * q[0]) * (1 << PASS1_BITS);
            for (int r = 0; r < 8; ++r) w[r * 8] = dc;
            continue;
        }
        int32_t z2 = in[16] * q[16], z3 = in[48] * q[48];
        int32_t z1 = (z2 + z3) * FIX_0_541196100;
        int32_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int32_t tmp3 = z1 + z2 * FIX_0_765366865;
        z2 = in[0] * q[0];
        z3 = in[32] * q[32];
        int32_t tmp0 = (z2 + z3) * (1 << CONST_BITS);
        int32_t tmp1 = (z2 - z3) * (1 << CONST_BITS);
        int32_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = in[56] * q[56];
        tmp1 = in[40] * q[40];
        tmp2 = in[24] * q[24];
        tmp3 = in[8] * q[8];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int32_t z4 = tmp1 + tmp3;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        w[0 * 8] = DESCALE(tmp10 + tmp3, CONST_BITS - PASS1_BITS);
        w[7 * 8] = DESCALE(tmp10 - tmp3, CONST_BITS - PASS1_BITS);
        w[1 * 8] = DESCALE(tmp11 + tmp2, CONST_BITS - PASS1_BITS);
        w[6 * 8] = DESCALE(tmp11 - tmp2, CONST_BITS - PASS1_BITS);
        w[2 * 8] = DESCALE(tmp12 + tmp1, CONST_BITS - PASS1_BITS);
        w[5 * 8] = DESCALE(tmp12 - tmp1, CONST_BITS - PASS1_BITS);
        w[3 * 8] = DESCALE(tmp13 + tmp0, CONST_BITS - PASS1_BITS);
        w[4 * 8] = DESCALE(tmp13 - tmp0, CONST_BITS - PASS1_BITS);
    }
    // Pass 2: rows.
    for (int r = 0; r < 8; ++r) {
        const int32_t* w = ws + r * 8;
        uint8_t* o = out + r * stride;
        if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0) {
            uint8_t v = rl[DESCALE(w[0], PASS1_BITS + 3) & 1023];
            for (int i = 0; i < 8; ++i) o[i] = v;
            continue;
        }
        int32_t z2 = w[2], z3 = w[6];
        int32_t z1 = (z2 + z3) * FIX_0_541196100;
        int32_t tmp2 = z1 + z3 * (-FIX_1_847759065);
        int32_t tmp3 = z1 + z2 * FIX_0_765366865;
        int32_t tmp0 = (w[0] + w[4]) * (1 << CONST_BITS);
        int32_t tmp1 = (w[0] - w[4]) * (1 << CONST_BITS);
        int32_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = w[7];
        tmp1 = w[5];
        tmp2 = w[3];
        tmp3 = w[1];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int32_t z4 = tmp1 + tmp3;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;
        tmp0 *= FIX_0_298631336;
        tmp1 *= FIX_2_053119869;
        tmp2 *= FIX_3_072711026;
        tmp3 *= FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;
        z3 += z5;
        z4 += z5;
        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;
        constexpr int SH = CONST_BITS + PASS1_BITS + 3;
        o[0] = rl[DESCALE(tmp10 + tmp3, SH) & 1023];
        o[7] = rl[DESCALE(tmp10 - tmp3, SH) & 1023];
        o[1] = rl[DESCALE(tmp11 + tmp2, SH) & 1023];
        o[6] = rl[DESCALE(tmp11 - tmp2, SH) & 1023];
        o[2] = rl[DESCALE(tmp12 + tmp1, SH) & 1023];
        o[5] = rl[DESCALE(tmp12 - tmp1, SH) & 1023];
        o[3] = rl[DESCALE(tmp13 + tmp0, SH) & 1023];
        o[4] = rl[DESCALE(tmp13 - tmp0, SH) & 1023];
    }
}

}  // namespace parajpeg::detail
