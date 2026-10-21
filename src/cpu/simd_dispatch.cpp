#include "simd_dispatch.h"

#include <cstdlib>
#include <cstring>

#include "../core/color.h"
#include "../core/idct.h"

namespace parajpeg::detail {

#if defined(__x86_64__) || defined(_M_X64)
void idct_islow_avx2(const int16_t*, const uint16_t*, uint8_t*, size_t);
void ycc_to_rgb_row_avx2(const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*, uint32_t, int, bool);
void h2v1_fancy_row_avx2(const uint8_t*, uint32_t, uint8_t*);
void h2v2_fancy_row_avx2(const uint8_t*, const uint8_t*, uint32_t, uint8_t*);
#endif

const SimdKernels& scalar_kernels() {
    static const SimdKernels k{"scalar", idct_islow_scalar, ycc_to_rgb_row_scalar, h2v1_fancy_scalar, h2v2_fancy_scalar};
    return k;
}

const SimdKernels& best_kernels() {
#if defined(__ARM_NEON) || defined(__aarch64__)
    static const SimdKernels k{"neon", idct_islow_neon, ycc_to_rgb_row_neon, h2v1_fancy_row_neon, h2v2_fancy_row_neon};
    return k;
#elif defined(__x86_64__) || defined(_M_X64)
    // PARAJPEG_FORCE_SIMD=avx2 bypasses CPUID (testing only: Rosetta 2 executes AVX2 without
    // advertising it). PARAJPEG_FORCE_SIMD=scalar disables SIMD.
    static const char* force = std::getenv("PARAJPEG_FORCE_SIMD");
    static const bool use_avx2 = force ? std::strcmp(force, "avx2") == 0 : __builtin_cpu_supports("avx2");
    static const SimdKernels k = use_avx2
                                     ? SimdKernels{"avx2", idct_islow_avx2, ycc_to_rgb_row_avx2, h2v1_fancy_row_avx2, h2v2_fancy_row_avx2}
                                     : scalar_kernels();
    return k;
#else
    return scalar_kernels();
#endif
}

}  // namespace parajpeg::detail
