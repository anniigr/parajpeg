// Truncated streams: libjpeg-turbo recovers (zero-fills the current MCU, skips the rest of the
// restart interval, warns). parajpeg must reproduce that recovery output exactly, on every backend
// and strategy — including the emulated GPU pipeline.
#include <cstdlib>
#include <string>

#include "doctest.h"
#include "parajpeg/parajpeg.h"
#include "test_util.h"

namespace parajpeg::detail {
Status decode_cuda_emulated(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats);
}

#ifdef HAVE_TURBOJPEG
#include <turbojpeg.h>

// Damaged data can produce coefficients whose IDCT output overflows; there libjpeg-turbo's SIMD
// kernels saturate while its C reference (jidctint.c, which parajpeg follows) wraps through the
// range-limit table. The comparison therefore runs against the C reference: ctest executes this
// suite a second time with JSIMD_FORCENONE=1.
static bool c_reference() {
    const char* e = std::getenv("JSIMD_FORCENONE");
    return e && std::string(e) == "1";
}

TEST_CASE("truncated streams: output equals libjpeg-turbo's recovery output") {
    if (!c_reference()) {
        MESSAGE("skipped: needs JSIMD_FORCENONE=1 (libjpeg-turbo C reference); see ctest parajpeg_tests_cref");
        return;
    }
    auto files = corpus_files();
    int compared = 0, skipped = 0;
    for (auto& f : files) {
        auto full = read_file(f);
        parajpeg::ImageInfo info;
        REQUIRE(parajpeg::read_header(full.data(), full.size(), info).ok());
        if (info.arithmetic || uint64_t(info.width) * info.height > 600000) continue;
#ifndef TJ_NUMINIT  // TurboJPEG 3.x API marker: libjpeg-turbo < 3.0
        // Recovery of damaged progressive files (block smoothing) changed in libjpeg-turbo 3.0;
        // parajpeg reproduces the 3.x behaviour.
        if (info.progressive) continue;
#endif
        const bool gray = info.components == 1;
        for (double frac : {0.97, 0.8, 0.55, 0.3}) {
            std::vector<uint8_t> d(full.begin(), full.begin() + long(full.size() * frac));
            tjhandle h = tjInitDecompress();
            int w, hh, sub, cs;
            if (tjDecompressHeader3(h, d.data(), (unsigned long)d.size(), &w, &hh, &sub, &cs) != 0) { tjDestroy(h); ++skipped; continue; }
            const int pf = gray ? TJPF_GRAY : TJPF_RGB;
            // TurboJPEG's error code is unreliable after an earlier warning, so detect whether it
            // actually produced an image: decode into buffers pre-filled with two patterns.
            std::vector<uint8_t> ref(size_t(w) * hh * tjPixelSize[pf], 0x00), ref2(ref.size(), 0xFF);
            tjDecompress2(h, d.data(), (unsigned long)d.size(), ref.data(), w, 0, hh, pf, TJFLAG_ACCURATEDCT);
            tjDecompress2(h, d.data(), (unsigned long)d.size(), ref2.data(), w, 0, hh, pf, TJFLAG_ACCURATEDCT);
            tjDestroy(h);
            if (ref != ref2) { ++skipped; continue; }  // libjpeg gave up: no reference output
            const size_t stride = size_t(w) * tjPixelSize[pf];
            for (int variant = 0; variant < 5; ++variant) {
                parajpeg::DecodeOptions o;
                o.format = gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB;
                o.backend = variant == 0 ? parajpeg::Backend::Scalar : parajpeg::Backend::Cpu;
                o.threads = variant == 0 ? 1 : 8;
                if (variant >= 2) { o.min_parallel_scan_bytes = 1; o.speculative_subsequence_bytes = variant == 2 ? 64 : 300; }
                std::vector<uint8_t> out(ref.size(), 0x5A);
                parajpeg::Status s = variant == 4 ? parajpeg::detail::decode_cuda_emulated(d.data(), d.size(), o, out.data(), stride, nullptr)
                                                  : parajpeg::decode_into(d.data(), d.size(), o, out.data(), stride, nullptr);
                INFO(f << " frac=" << frac << " variant=" << variant << " status=" << s.message);
                REQUIRE((s.ok() || s.code == parajpeg::StatusCode::Truncated));
                size_t diff = 0;
                for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
                CHECK(diff == 0);
            }
            ++compared;
        }
    }
    MESSAGE("compared " << compared << " truncated streams, skipped " << skipped);
    CHECK(compared > 100);
}

TEST_CASE("fuzzer-found regressions decode like libjpeg-turbo and terminate") {
    for (const char* name : {"fuzz_nosync_640x480.jpg"}) {
        auto d = read_file(std::string(PARAJPEG_CORPUS_DIR) + "/../regress/" + name);
        REQUIRE(!d.empty());
        tjhandle h = tjInitDecompress();
        int w, hh, sub, cs;
        REQUIRE(tjDecompressHeader3(h, d.data(), (unsigned long)d.size(), &w, &hh, &sub, &cs) == 0);
        std::vector<uint8_t> ref(size_t(w) * hh * 3);
        tjDecompress2(h, d.data(), (unsigned long)d.size(), ref.data(), w, 0, hh, TJPF_RGB, TJFLAG_ACCURATEDCT);
        tjDestroy(h);
        for (int variant = 0; variant < 3; ++variant) {
            parajpeg::DecodeOptions o;
            o.threads = 8;
            o.speculative_subsequence_bytes = 16;
            if (variant == 1) o.min_parallel_scan_bytes = 1;
            std::vector<uint8_t> out(ref.size());
            if (variant == 2) parajpeg::detail::decode_cuda_emulated(d.data(), d.size(), o, out.data(), size_t(w) * 3, nullptr);
            else parajpeg::decode_into(d.data(), d.size(), o, out.data(), size_t(w) * 3, nullptr);
            INFO(name << " variant " << variant);
            CHECK(out == ref);
        }
    }
}
#endif
