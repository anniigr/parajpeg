#include "color.h"

#include <algorithm>
#include <cstring>

namespace parajpeg::detail {

namespace {
constexpr int SCALEBITS = 16;
constexpr int32_t ONE_HALF = 1 << (SCALEBITS - 1);
constexpr int32_t FIX(double x) { return int32_t(x * (1 << SCALEBITS) + 0.5); }
}  // namespace

ColorTables::ColorTables() {
    for (int i = 0; i < 256; ++i) {
        int x = i - 128;
        cr_r[i] = (FIX(1.40200) * x + ONE_HALF) >> SCALEBITS;
        cb_b[i] = (FIX(1.77200) * x + ONE_HALF) >> SCALEBITS;
        cr_g[i] = (-FIX(0.71414)) * x;
        cb_g[i] = (-FIX(0.34414)) * x + ONE_HALF;
    }
    // sample_range_limit: [-256,-1] -> 0 ; [0,255] -> x ; [256,639] -> 255 ; [640,1023] -> 0 ;
    // [1024,1279] -> wrap like libjpeg (copy of 0..255 region)... only [-256, 511] is reachable here.
    for (int x = -256; x < 1280; ++x) {
        uint8_t v;
        if (x < 0) v = 0;
        else if (x < 256) v = uint8_t(x);
        else if (x < 640) v = 255;
        else if (x < 1024) v = 0;
        else v = uint8_t(x - 1024);
        range[x + 256] = v;
    }
}

const ColorTables& color_tables() {
    static const ColorTables t;
    return t;
}

// ---- Upsampling ---------------------------------------------------------------------------------

void h2v1_fancy_scalar(const uint8_t* in, uint32_t in_w, uint8_t* out) {
    // jdsample.c h2v1_fancy_upsample
    const uint8_t* p = in;
    uint8_t* o = out;
    int invalue = *p++;
    *o++ = uint8_t(invalue);
    *o++ = uint8_t((invalue * 3 + p[0] + 2) >> 2);
    for (uint32_t col = in_w - 2; col > 0; --col) {
        invalue = (*p++) * 3;
        *o++ = uint8_t((invalue + p[-2] + 1) >> 2);
        *o++ = uint8_t((invalue + p[0] + 2) >> 2);
    }
    invalue = *p;
    *o++ = uint8_t((invalue * 3 + p[-1] + 1) >> 2);
    *o++ = uint8_t(invalue);
}

void h2v2_fancy_scalar(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, uint8_t* out) {
    // jdsample.c h2v2_fancy_upsample (one output row; in1 = nearest neighbour row)
    const uint8_t* p0 = in0;
    const uint8_t* p1 = in1;
    uint8_t* o = out;
    int thiscolsum = (*p0++) * 3 + (*p1++);
    int nextcolsum = (*p0++) * 3 + (*p1++);
    *o++ = uint8_t((thiscolsum * 4 + 8) >> 4);
    *o++ = uint8_t((thiscolsum * 3 + nextcolsum + 7) >> 4);
    int lastcolsum = thiscolsum;
    thiscolsum = nextcolsum;
    for (uint32_t col = in_w - 2; col > 0; --col) {
        nextcolsum = (*p0++) * 3 + (*p1++);
        *o++ = uint8_t((thiscolsum * 3 + lastcolsum + 8) >> 4);
        *o++ = uint8_t((thiscolsum * 3 + nextcolsum + 7) >> 4);
        lastcolsum = thiscolsum;
        thiscolsum = nextcolsum;
    }
    *o++ = uint8_t((thiscolsum * 3 + lastcolsum + 8) >> 4);
    *o++ = uint8_t((thiscolsum * 4 + 7) >> 4);
}

static void h1v2_fancy(const uint8_t* in0, const uint8_t* in1, uint32_t in_w, int bias, uint8_t* out) {
    // libjpeg-turbo jdsample.c h1v2_fancy_upsample: bias 1 for the upper row (v==0), 2 for lower.
    for (uint32_t i = 0; i < in_w; ++i) out[i] = uint8_t((in0[i] * 3 + in1[i] + bias) >> 2);
}

void upsample_row(const Component& c, int hexp, int vexp, bool fancy, uint32_t y, uint32_t out_w, uint8_t* out, const SimdKernels& K) {
    const uint32_t in_w = c.width;
    const uint32_t in_h = c.height;
    const uint32_t iy = std::min(y / uint32_t(vexp), in_h - 1);
    const uint8_t* row = c.plane + size_t(iy) * c.plane_stride;
    if (hexp == 1 && vexp == 1) {
        std::memcpy(out, row, out_w);
        return;
    }
    const bool do_fancy = fancy && in_w > 2;
    if (hexp == 2 && vexp == 1 && do_fancy) {
        K.h2v1(row, in_w, out);
        return;
    }
    if (hexp == 2 && vexp == 2 && do_fancy) {
        // v == 0 (even output row): neighbour is the row above; v == 1: the row below. Edges replicate.
        const uint32_t v = y & 1;
        uint32_t ny = v == 0 ? (iy == 0 ? 0 : iy - 1) : std::min(iy + 1, in_h - 1);
        const uint8_t* nrow = c.plane + size_t(ny) * c.plane_stride;
        K.h2v2(row, nrow, in_w, out);
        return;
    }
    if (hexp == 1 && vexp == 2 && fancy) {
        const uint32_t v = y & 1;
        uint32_t ny = v == 0 ? (iy == 0 ? 0 : iy - 1) : std::min(iy + 1, in_h - 1);
        const uint8_t* nrow = c.plane + size_t(ny) * c.plane_stride;
        h1v2_fancy(row, nrow, in_w, v == 0 ? 1 : 2, out);
        return;
    }
    // Generic box replication (int_upsample / h2v1_upsample / h2v2_upsample).
    for (uint32_t i = 0; i < in_w; ++i) {
        uint8_t s = row[i];
        for (int k = 0; k < hexp; ++k) out[i * hexp + k] = s;
    }
}

// ---- Colour conversion ----------------------------------------------------------------------------

void convert_row(const uint8_t* const* rows, int ncomp, Frame::ColorSpace cs, PixelFormat fmt, uint32_t width, uint8_t* dst, const SimdKernels& K) {
    const ColorTables& T = color_tables();
    const int ch = channels_of(fmt);
    if (fmt == PixelFormat::Gray) {
        if (ncomp == 1 || cs == Frame::ColorSpace::YCbCr) {
            std::memcpy(dst, rows[0], width);
        } else {
            // RGB -> gray (libjpeg rgb_gray_convert): Y = 0.299R + 0.587G + 0.114B with FIX rounding
            constexpr int32_t R_Y = FIX(0.29900), G_Y = FIX(0.58700), B_Y = FIX(0.11400);
            for (uint32_t i = 0; i < width; ++i)
                dst[i] = uint8_t((rows[0][i] * R_Y + rows[1][i] * G_Y + rows[2][i] * B_Y + ONE_HALF) >> SCALEBITS);
        }
        return;
    }
    const bool bgr = (fmt == PixelFormat::BGR || fmt == PixelFormat::BGRA);
    const bool alpha = (fmt == PixelFormat::RGBA || fmt == PixelFormat::BGRA);
    if (ncomp == 1) {
        for (uint32_t i = 0; i < width; ++i) {
            uint8_t v = rows[0][i];
            dst[i * ch] = v; dst[i * ch + 1] = v; dst[i * ch + 2] = v;
            if (alpha) dst[i * ch + 3] = 255;
        }
        return;
    }
    if (ncomp == 3) {
        if (cs == Frame::ColorSpace::RGB) {
            for (uint32_t i = 0; i < width; ++i) {
                uint8_t r = rows[0][i], g = rows[1][i], b = rows[2][i];
                dst[i * ch] = bgr ? b : r; dst[i * ch + 1] = g; dst[i * ch + 2] = bgr ? r : b;
                if (alpha) dst[i * ch + 3] = 255;
            }
            return;
        }
        K.ycc(rows[0], rows[1], rows[2], dst, width, ch, bgr);
        return;
    }
    // ncomp == 4: CMYK / YCCK. Output CMYK (4 channels) when fmt has 4 channels, else drop K.
    for (uint32_t i = 0; i < width; ++i) {
        int c0 = rows[0][i], c1 = rows[1][i], c2 = rows[2][i], k = rows[3][i];
        uint8_t c, m, yy;
        if (cs == Frame::ColorSpace::YCCK) {
            // jdcolor.c ycck_cmyk_convert: CMY = 255 - (YCbCr->RGB)
            c = uint8_t(255 - T.clamp(c0 + T.cr_r[c2]));
            m = uint8_t(255 - T.clamp(c0 + ((T.cb_g[c1] + T.cr_g[c2]) >> SCALEBITS)));
            yy = uint8_t(255 - T.clamp(c0 + T.cb_b[c1]));
        } else {
            c = uint8_t(c0); m = uint8_t(c1); yy = uint8_t(c2);
        }
        if (ch == 4) {
            dst[i * 4] = c; dst[i * 4 + 1] = m; dst[i * 4 + 2] = yy; dst[i * 4 + 3] = uint8_t(k);
        } else {
            // Adobe-inverted CMYK -> RGB approximation (as TurboJPEG does not support it, document it).
            dst[i * ch] = uint8_t((c * k) / 255); dst[i * ch + 1] = uint8_t((m * k) / 255); dst[i * ch + 2] = uint8_t((yy * k) / 255);
        }
    }
}

}  // namespace parajpeg::detail

namespace parajpeg::detail {
void ycc_to_rgb_row_scalar(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* dst, uint32_t width, int ch, bool bgr) {
    const ColorTables& T = color_tables();
    const bool alpha = ch == 4;
    for (uint32_t i = 0; i < width; ++i) {
        int Y = y[i], Cb = cb[i], Cr = cr[i];
        uint8_t r = T.clamp(Y + T.cr_r[Cr]);
        uint8_t g = T.clamp(Y + ((T.cb_g[Cb] + T.cr_g[Cr]) >> SCALEBITS));
        uint8_t b = T.clamp(Y + T.cb_b[Cb]);
        dst[i * ch] = bgr ? b : r; dst[i * ch + 1] = g; dst[i * ch + 2] = bgr ? r : b;
        if (alpha) dst[i * ch + 3] = 255;
    }
}
}  // namespace parajpeg::detail
