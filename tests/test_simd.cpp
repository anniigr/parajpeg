// The SIMD kernels selected for this CPU (NEON on AArch64, AVX2 on x86-64 when available) must equal
// the scalar reference bit for bit, on random and on adversarial inputs.
#include <cstring>
#include <string>
#include <random>

#include "doctest.h"
#include "cpu/simd_dispatch.h"

using namespace parajpeg::detail;

TEST_CASE("SIMD IDCT equals the scalar libjpeg islow IDCT") {
    const SimdKernels& K = best_kernels();
    const SimdKernels& S = scalar_kernels();
    MESSAGE("kernels: " << std::string(K.name));
    std::mt19937 rng(42);
    int16_t coef[64];
    uint16_t q[64];
    uint8_t a[8 * 16], b[8 * 16];
    for (int iter = 0; iter < 200000; ++iter) {
        const int mode = iter % 6;
        for (int i = 0; i < 64; ++i) {
            q[i] = uint16_t(1 + rng() % (mode >= 4 ? (mode == 5 ? 65535 : 255) : 64));
            int v = 0;
            switch (mode) {
                case 0: v = (i == 0) ? int(rng() % 2048) - 1024 : 0; break;                    // DC only
                case 1: v = (i < 8 || i % 8 == 0) ? int(rng() % 200) - 100 : 0; break;         // sparse
                case 2: v = (rng() % 4 == 0) ? int(rng() % 64) - 32 : 0; break;                // typical
                case 3: v = int(rng() % 2048) - 1024; break;                                   // dense, out of window
                case 4: v = int(int16_t(rng())) / 64; break;                                   // large
                case 5: v = int(int16_t(rng())); break;                                        // adversarial
            }
            coef[i] = int16_t(v);
        }
        std::memset(a, 0xAA, sizeof a);
        std::memset(b, 0xAA, sizeof b);
        K.idct(coef, q, a, 16);
        S.idct(coef, q, b, 16);
        if (std::memcmp(a, b, sizeof a) != 0) {
            FAIL_CHECK("IDCT mismatch at iteration " << iter << " mode " << mode);
            break;
        }
    }
}

TEST_CASE("SIMD colour conversion and fancy upsampling equal the scalar code") {
    const SimdKernels& K = best_kernels();
    const SimdKernels& S = scalar_kernels();
    std::mt19937 rng(7);
    for (uint32_t w : {1u, 3u, 7u, 8u, 15u, 16u, 17u, 31u, 33u, 34u, 35u, 64u, 100u, 257u, 1000u}) {
        std::vector<uint8_t> y(w), cb(w), cr(w), r0(w), r1(w);
        for (auto* v : {&y, &cb, &cr, &r0, &r1})
            for (auto& x : *v) x = uint8_t(rng());
        for (int ch : {3, 4})
            for (bool bgr : {false, true}) {
                std::vector<uint8_t> a(size_t(w) * ch, 1), b(size_t(w) * ch, 2);
                K.ycc(y.data(), cb.data(), cr.data(), a.data(), w, ch, bgr);
                S.ycc(y.data(), cb.data(), cr.data(), b.data(), w, ch, bgr);
                INFO("ycc w=" << w << " ch=" << ch << " bgr=" << bgr);
                CHECK(a == b);
            }
        if (w > 2) {
            std::vector<uint8_t> a(2 * size_t(w), 1), b(2 * size_t(w), 2);
            K.h2v1(r0.data(), w, a.data());
            S.h2v1(r0.data(), w, b.data());
            INFO("h2v1 w=" << w);
            CHECK(a == b);
            K.h2v2(r0.data(), r1.data(), w, a.data());
            S.h2v2(r0.data(), r1.data(), w, b.data());
            INFO("h2v2 w=" << w);
            CHECK(a == b);
        }
    }
}
