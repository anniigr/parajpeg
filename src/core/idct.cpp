#include "idct.h"

#include "../common/idct_core.h"

namespace parajpeg::detail {

namespace {
struct RangeTable {
    uint8_t t[1024];
    RangeTable() {
        for (int i = 0; i < 1024; ++i) t[i] = idct_range_limit(i);
    }
};
const RangeTable g_range;
}  // namespace

const uint8_t* idct_range_limit_table() { return g_range.t; }

void idct_islow_scalar(const int16_t* coef, const uint16_t* quant, uint8_t* out, size_t stride) {
    idct_islow_core(coef, quant, out, stride);
}

}  // namespace parajpeg::detail
