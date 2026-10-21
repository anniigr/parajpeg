// The CUDA pipeline (speculative GPU Huffman decoding, restart-parallel kernel, IDCT and colour
// kernels) executed by the CPU grid emulator must match libjpeg-turbo exactly.
#include "doctest.h"
#include "parajpeg/parajpeg.h"
#include "test_util.h"

namespace parajpeg::detail {
Status decode_cuda_emulated(const uint8_t* data, size_t size, const DecodeOptions& opts, uint8_t* dst, size_t stride, DecodeStats* stats);
}

#ifdef HAVE_TURBOJPEG
#include <turbojpeg.h>

TEST_CASE("emulated CUDA pipeline is bit-exact with libjpeg-turbo") {
    auto files = corpus_files();
    REQUIRE(!files.empty());
    int tested = 0;
    for (auto& f : files) {
        auto d = read_file(f);
        parajpeg::ImageInfo info;
        REQUIRE(parajpeg::read_header(d.data(), d.size(), info).ok());
        if (info.arithmetic) continue;
        const bool gray = info.components == 1;
        for (int variant = 0; variant < 3; ++variant) {
            const int pf = variant == 2 ? TJPF_BGRA : (gray ? TJPF_GRAY : TJPF_RGB);
            if (variant == 2 && gray) continue;
            tjhandle h = tjInitDecompress();
            int w, hh, sub, cs;
            REQUIRE(tjDecompressHeader3(h, d.data(), (unsigned long)d.size(), &w, &hh, &sub, &cs) == 0);
            std::vector<uint8_t> ref(size_t(w) * hh * tjPixelSize[pf]);
            REQUIRE(tjDecompress2(h, d.data(), (unsigned long)d.size(), ref.data(), w, 0, hh, pf, TJFLAG_ACCURATEDCT) == 0);
            tjDestroy(h);
            parajpeg::DecodeOptions o;
            o.format = variant == 2 ? parajpeg::PixelFormat::BGRA : (gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB);
            o.speculative_subsequence_bytes = variant == 1 ? 32 : 0;  // tiny subsequences stress synchronisation
            o.threads = 4;
            std::vector<uint8_t> out(ref.size(), 0);
            parajpeg::DecodeStats st;
            auto s = parajpeg::detail::decode_cuda_emulated(d.data(), d.size(), o, out.data(), size_t(w) * tjPixelSize[pf], &st);
            INFO(f << " variant " << variant << " status " << s.message << " iters " << st.sync_iterations << " subseq " << st.subsequences);
            REQUIRE(s.ok());
            size_t diff = 0;
            for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
            CHECK(diff == 0);
            if (!info.progressive && info.restart_interval == 0) {
                CHECK(st.subsequences > 0);        // the speculative kernels really ran
                CHECK(st.sync_iterations >= 1);
            }
            if (!info.progressive && info.restart_interval > 0) CHECK(st.restart_segments > 0);
        }
        ++tested;
    }
    CHECK(tested >= 50);
}
#endif
