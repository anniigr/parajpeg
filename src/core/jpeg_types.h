// Internal data model shared by all backends.
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "parajpeg/parajpeg.h"

namespace parajpeg::detail {

constexpr int kDctSize = 8;
constexpr int kDctSize2 = 64;
constexpr int kMaxComponents = 4;
constexpr int kMaxBlocksPerMcu = 10;  // JPEG spec limit

// Zig-zag index -> natural (row-major) index. Entries 64..79 clamp to 63 for corrupt streams,
// mirroring libjpeg's jpeg_natural_order safety padding.
extern const uint8_t kNaturalOrder[kDctSize2 + 16];

struct QuantTable {
    bool defined = false;
    std::array<uint16_t, kDctSize2> q{};  // in natural order
};

// Canonical Huffman table with an 11-bit first-level lookup (fast path) and
// the JPEG-standard maxcode/valptr slow path for codes longer than 9 bits.
struct HuffTable {
    static constexpr int kLookupBits = 11;
    bool defined = false;
    std::array<uint8_t, 17> bits{};    // bits[1..16]: number of codes of each length
    std::array<uint8_t, 256> vals{};
    // Derived:
    // lookup[peek] = (length << 8) | symbol, or 0 if the code is longer than kLookupBits.
    std::array<uint16_t, 1 << kLookupBits> lookup{};
    std::array<int32_t, 18> maxcode{};  // maxcode[l] = largest code of length l, -1 if none; [17] = sentinel
    std::array<int32_t, 17> valoffset{};  // vals index = code + valoffset[l]
    int total_codes = 0;
};

struct Component {
    uint8_t id = 0;
    uint8_t h = 1, v = 1;           // sampling factors
    uint8_t tq = 0;                 // quant table index
    uint8_t td = 0, ta = 0;         // DC/AC Huffman table ids (per scan)
    // Quantisation table latched at the start of the component's first scan (libjpeg semantics:
    // a DQT redefining the table later does not affect an already-started component).
    QuantTable qt_latched{};
    bool qt_latched_set = false;
    // Geometry
    uint32_t width = 0, height = 0;            // actual component dims (ceil(image * h / hmax))
    uint32_t blocks_w = 0, blocks_h = 0;       // ceil(width/8), ceil(height/8) — used for non-interleaved scans
    uint32_t padded_blocks_w = 0, padded_blocks_h = 0;  // MCU-aligned block dims — allocation size
    // Coefficient storage: padded_blocks_w * padded_blocks_h blocks of 64 int16 (natural order).
    int16_t* coefs = nullptr;
    // Progressive only: per block, bit k set <=> coefficient at zig-zag index k is nonzero (k >= 1).
    // Lets AC refinement visit only coefficients with history instead of all 63 positions.
    uint64_t* nzmask = nullptr;
    // Sample plane after IDCT: padded_blocks_w*8 x padded_blocks_h*8, row stride = plane_stride.
    uint8_t* plane = nullptr;
    size_t plane_stride = 0;
};

struct ScanHeader {
    int ncomp = 0;
    std::array<int, kMaxComponents> comp_index{};  // indices into Frame::comps
    std::array<uint8_t, kMaxComponents> td{}, ta{};
    uint8_t ss = 0, se = 63, ah = 0, al = 0;
    // Entropy-coded data span (after SOS header) up to (not including) the terminating marker.
    size_t data_offset = 0;
    size_t data_size = 0;
};

struct Frame {
    bool progressive = false;
    bool arithmetic = false;
    bool valid = false;
    int precision = 8;
    uint32_t width = 0, height = 0;
    int ncomp = 0;
    std::array<Component, kMaxComponents> comps{};
    int hmax = 1, vmax = 1;
    uint32_t mcu_w = 8, mcu_h = 8;     // pixels
    uint32_t mcus_x = 0, mcus_y = 0;   // MCUs across / down (interleaved geometry)
    uint16_t restart_interval = 0;
    std::array<QuantTable, 4> qt{};
    std::array<HuffTable, 4> dc_tables{}, ac_tables{};
    bool jfif = false, adobe = false;
    int adobe_transform = -1;
    std::vector<ScanHeader> scans;
    std::vector<std::string> warnings;
    // Whether the data was truncated (missing EOI) — libjpeg warns and pads with zeros.
    bool truncated = false;
    // Subsampling classification for ImageInfo.
    Subsampling subsampling() const;
    // Color transform decision (libjpeg jdapimin.c default_decompress_parms semantics).
    enum class ColorSpace { Gray, YCbCr, RGB, CMYK, YCCK, Unknown } jpeg_color_space = ColorSpace::Unknown;
};

}  // namespace parajpeg::detail
