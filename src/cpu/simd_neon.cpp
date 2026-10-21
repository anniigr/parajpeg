// NEON implementations. First version delegates to the scalar reference; the vectorised versions
// replace these one by one with bit-exactness tests guarding each step.
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>

#include "../core/color.h"
#include "../core/idct.h"

namespace parajpeg::detail {

void h2v1_fancy_row_neon(const uint8_t* in, uint32_t in_w, uint8_t* out) { h2v1_fancy_scalar(in, in_w, out); }
void h2v2_fancy_row_neon(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out) { h2v2_fancy_scalar(in0, in1, in_w, out); }
void ycc_to_rgb_row_neon(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* dst, uint32_t width, int channels, bool bgr) {
    ycc_to_rgb_row_scalar(y, cb, cr, dst, width, channels, bgr);
}
void idct_islow_neon(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    idct_islow_scalar(coef, quant, out, stride);
}

}  // namespace parajpeg::detail
#endif
