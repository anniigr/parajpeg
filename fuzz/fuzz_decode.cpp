// libFuzzer target: parse + decode arbitrary bytes with every CPU strategy (sequential, forced
// parallel with tiny subsequences) and the emulated GPU pipeline. Any crash, sanitizer report or
// hang is a bug; decode errors are expected and fine.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "parajpeg/parajpeg.h"

namespace parajpeg::detail {
Status decode_cuda_emulated(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    parajpeg::ImageInfo info;
    if (!parajpeg::read_header(data, size, info).ok()) return 0;
    if (uint64_t(info.width) * info.height > (uint64_t(1) << 22)) return 0;  // keep iterations fast
    const auto fmt = info.components == 4 ? parajpeg::PixelFormat::RGBA : parajpeg::PixelFormat::RGB;
    std::vector<uint8_t> out(size_t(info.width) * info.height * 4);
    for (int variant = 0; variant < 3; ++variant) {
        parajpeg::DecodeOptions o;
        o.format = fmt;
        o.backend = variant == 0 ? parajpeg::Backend::Scalar : parajpeg::Backend::Cpu;
        o.threads = variant == 0 ? 1 : 4;
        if (variant == 2) { o.min_parallel_scan_bytes = 1; o.speculative_subsequence_bytes = 64; }
        (void)parajpeg::decode_into(data, size, o, out.data(), size_t(info.width) * 4, nullptr);
    }
    parajpeg::DecodeOptions g;
    g.format = fmt;
    g.speculative_subsequence_bytes = 16;
    (void)parajpeg::detail::decode_cuda_emulated(data, size, g, out.data(), size_t(info.width) * 4, nullptr);
    return 0;
}
