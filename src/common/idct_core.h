// Bit-exact libjpeg "islow" inverse DCT (jidctint.c), shared by the scalar CPU backend and the CUDA
// kernels. 64-bit intermediates mirror libjpeg's JLONG, so even adversarial coefficients are defined.
#pragma once
#include <cstddef>
#include <cstdint>

#include "hd.h"

namespace parajpeg::detail {

// libjpeg's post-IDCT range-limit table, as a function of (value & 1023).
PJ_HD PJ_FORCEINLINE uint8_t idct_range_limit(int64_t v) {
    int idx = int(v & 1023);
    if (idx < 128) return uint8_t(128 + idx);
    if (idx < 512) return 255;
    if (idx < 896) return 0;
    return uint8_t(idx - 896);
}

PJ_HD inline void idct_islow_core(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    constexpr int CB = 13, P1 = 2;
    constexpr int64_t F0_298 = 2446, F0_390 = 3196, F0_541 = 4433, F0_765 = 6270, F0_899 = 7373, F1_175 = 9633,
                      F1_501 = 12299, F1_847 = 15137, F1_961 = 16069, F2_053 = 16819, F2_562 = 20995, F3_072 = 25172;
    auto descale = [](int64_t x, int n) -> int64_t { return (x + (int64_t(1) << (n - 1))) >> n; };
    int32_t ws[64];
    for (int c = 0; c < 8; ++c) {
        const int16_t* in = coef + c;
        const uint16_t* q = quant + c;
        int32_t* w = ws + c;
        if (in[8] == 0 && in[16] == 0 && in[24] == 0 && in[32] == 0 && in[40] == 0 && in[48] == 0 && in[56] == 0) {
            int32_t dc = int32_t(uint32_t((int64_t(in[0]) * q[0]) * (1 << P1)));
            for (int r = 0; r < 8; ++r) w[r * 8] = dc;
            continue;
        }
        int64_t z2 = int64_t(in[16]) * q[16], z3 = int64_t(in[48]) * q[48];
        int64_t z1 = (z2 + z3) * F0_541;
        int64_t tmp2 = z1 + z3 * (-F1_847);
        int64_t tmp3 = z1 + z2 * F0_765;
        z2 = int64_t(in[0]) * q[0];
        z3 = int64_t(in[32]) * q[32];
        int64_t tmp0 = (z2 + z3) * (1 << CB);
        int64_t tmp1 = (z2 - z3) * (1 << CB);
        int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = int64_t(in[56]) * q[56];
        tmp1 = int64_t(in[40]) * q[40];
        tmp2 = int64_t(in[24]) * q[24];
        tmp3 = int64_t(in[8]) * q[8];
        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        int64_t z4 = tmp1 + tmp3;
        int64_t z5 = (z3 + z4) * F1_175;
        tmp0 *= F0_298; tmp1 *= F2_053; tmp2 *= F3_072; tmp3 *= F1_501;
        z1 *= -F0_899; z2 *= -F2_562; z3 *= -F1_961; z4 *= -F0_390;
        z3 += z5; z4 += z5;
        tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4;
        w[0] = int32_t(uint32_t(descale(tmp10 + tmp3, CB - P1)));
        w[56] = int32_t(uint32_t(descale(tmp10 - tmp3, CB - P1)));
        w[8] = int32_t(uint32_t(descale(tmp11 + tmp2, CB - P1)));
        w[48] = int32_t(uint32_t(descale(tmp11 - tmp2, CB - P1)));
        w[16] = int32_t(uint32_t(descale(tmp12 + tmp1, CB - P1)));
        w[40] = int32_t(uint32_t(descale(tmp12 - tmp1, CB - P1)));
        w[24] = int32_t(uint32_t(descale(tmp13 + tmp0, CB - P1)));
        w[32] = int32_t(uint32_t(descale(tmp13 - tmp0, CB - P1)));
    }
    for (int r = 0; r < 8; ++r) {
        const int32_t* w = ws + r * 8;
        uint8_t* o = out + r * stride;
        if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0) {
            uint8_t v = idct_range_limit(descale(int64_t(w[0]), P1 + 3));
            for (int i = 0; i < 8; ++i) o[i] = v;
            continue;
        }
        int64_t z2 = w[2], z3 = w[6];
        int64_t z1 = (z2 + z3) * F0_541;
        int64_t tmp2 = z1 + z3 * (-F1_847);
        int64_t tmp3 = z1 + z2 * F0_765;
        int64_t tmp0 = (int64_t(w[0]) + w[4]) * (1 << CB);
        int64_t tmp1 = (int64_t(w[0]) - w[4]) * (1 << CB);
        int64_t tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
        tmp0 = w[7]; tmp1 = w[5]; tmp2 = w[3]; tmp3 = w[1];
        z1 = tmp0 + tmp3; z2 = tmp1 + tmp2; z3 = tmp0 + tmp2;
        int64_t z4 = tmp1 + tmp3;
        int64_t z5 = (z3 + z4) * F1_175;
        tmp0 *= F0_298; tmp1 *= F2_053; tmp2 *= F3_072; tmp3 *= F1_501;
        z1 *= -F0_899; z2 *= -F2_562; z3 *= -F1_961; z4 *= -F0_390;
        z3 += z5; z4 += z5;
        tmp0 += z1 + z3; tmp1 += z2 + z4; tmp2 += z2 + z3; tmp3 += z1 + z4;
        constexpr int SH = CB + P1 + 3;
        o[0] = idct_range_limit(descale(tmp10 + tmp3, SH));
        o[7] = idct_range_limit(descale(tmp10 - tmp3, SH));
        o[1] = idct_range_limit(descale(tmp11 + tmp2, SH));
        o[6] = idct_range_limit(descale(tmp11 - tmp2, SH));
        o[2] = idct_range_limit(descale(tmp12 + tmp1, SH));
        o[5] = idct_range_limit(descale(tmp12 - tmp1, SH));
        o[3] = idct_range_limit(descale(tmp13 + tmp0, SH));
        o[4] = idct_range_limit(descale(tmp13 - tmp0, SH));
    }
}

}  // namespace parajpeg::detail
