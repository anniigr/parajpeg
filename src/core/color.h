// Upsampling (libjpeg "fancy" triangle filters) and colour conversion, bit-exact with jdsample.c /
// jdcolor.c. Operates on one output row at a time from component sample planes.
#pragma once
#include <cstddef>
#include <cstdint>

#include "jpeg_types.h"
#include "../cpu/simd_dispatch.h"

namespace parajpeg::detail {

struct ColorTables {
    int32_t cr_r[256], cb_b[256], cr_g[256], cb_g[256];
    uint8_t range[1024 + 512];  // sample_range_limit: index (x + 256), valid for x in [-256, 1279]
    ColorTables();
    inline uint8_t clamp(int x) const { return range[x + 256]; }
};
const ColorTables& color_tables();

// Produces an upsampled chroma row (width = comp.width * hexp, written up to `out_w` samples) for
// output row `y` of the image, honouring libjpeg's fancy-upsampling rules. `ctx` is a scratch row
// buffer of at least out_w + 32 bytes.
void upsample_row(const Component& c, int hexp, int vexp, bool fancy, uint32_t y, uint32_t out_w,
                  uint8_t* out, const SimdKernels& K);

// Converts one row. Inputs are full-resolution rows per component (1, 3 or 4). Writes to dst in the
// requested format. `cs` is the JPEG colour space (decides YCbCr->RGB vs pass-through vs YCCK->CMYK).
void convert_row(const uint8_t* const* rows, int ncomp, Frame::ColorSpace cs, PixelFormat fmt,
                 uint32_t width, uint8_t* dst, const SimdKernels& K);

void h2v1_fancy_scalar(const uint8_t* in, uint32_t in_w, uint8_t* out);
void h2v2_fancy_scalar(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out);
void ycc_to_rgb_row_scalar(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* dst, uint32_t width, int channels, bool bgr);

#if defined(__ARM_NEON) || defined(__aarch64__)
void ycc_to_rgb_row_neon(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* dst, uint32_t width, int channels, bool bgr);
void h2v1_fancy_row_neon(const uint8_t* in, uint32_t in_w, uint8_t* out);
void h2v2_fancy_row_neon(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out);
#endif

}  // namespace parajpeg::detail
