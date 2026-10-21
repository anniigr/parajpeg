// AVX2 implementations (x86-64 only). Compiled with -mavx2 on x86; empty elsewhere.
#if defined(__AVX2__)
#include <immintrin.h>
#include "../core/idct.h"
namespace parajpeg::detail {
void idct_islow_avx2(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    idct_islow_scalar(coef, quant, out, stride);
}
}  // namespace parajpeg::detail
#endif
