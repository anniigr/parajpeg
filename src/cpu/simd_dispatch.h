// Runtime selection of the SIMD kernels: NEON is architectural on AArch64; AVX2 is chosen by CPUID.
#pragma once
#include <cstddef>
#include <cstdint>

namespace parajpeg::detail {

using IdctKernel = void (*)(const int16_t*, const uint16_t*, uint8_t*, size_t);
using YccKernel = void (*)(const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*, uint32_t, int, bool);
using H2V1Kernel = void (*)(const uint8_t*, uint32_t, uint8_t*);
using H2V2Kernel = void (*)(const uint8_t*, const uint8_t*, uint32_t, uint8_t*);

struct SimdKernels {
    const char* name;
    IdctKernel idct;
    YccKernel ycc;
    H2V1Kernel h2v1;
    H2V2Kernel h2v2;
};

const SimdKernels& scalar_kernels();
const SimdKernels& best_kernels();  // best available on this CPU

}  // namespace parajpeg::detail
