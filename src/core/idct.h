// Dequantisation + inverse DCT, bit-exact with libjpeg's jidctint.c ("islow").
#pragma once
#include <cstddef>
#include <cstdint>

namespace parajpeg::detail {

// out: 8 rows of 8 samples, written with `stride` bytes between rows.
void idct_islow_scalar(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride);

#if defined(__ARM_NEON) || defined(__aarch64__)
#define PARAJPEG_HAVE_NEON 1
void idct_islow_neon(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride);
#endif

// libjpeg's post-IDCT range-limit table semantics: index = (value & 1023), maps to a clamped sample.
const uint8_t* idct_range_limit_table();

}  // namespace parajpeg::detail
