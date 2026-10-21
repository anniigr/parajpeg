// Bit-exactness against libjpeg-turbo (TurboJPEG) for every corpus image and every backend/strategy.
#include <cstring>
#include "doctest.h"
#include "parajpeg/parajpeg.h"
#include "test_util.h"

#ifdef HAVE_TURBOJPEG
#include <turbojpeg.h>

static bool tj_decode(const std::vector<uint8_t>& d, int pf, std::vector<uint8_t>& out, int& w, int& h) {
    tjhandle h_ = tjInitDecompress();
    int sub, cs;
    if (tjDecompressHeader3(h_, d.data(), (unsigned long)d.size(), &w, &h, &sub, &cs) != 0) { tjDestroy(h_); return false; }
    int ch = tjPixelSize[pf];
    out.assign(size_t(w) * h * ch, 0);
    int rc = tjDecompress2(h_, d.data(), (unsigned long)d.size(), out.data(), w, 0, h, pf, TJFLAG_ACCURATEDCT);
    tjDestroy(h_);
    return rc == 0;
}

static size_t count_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, int& maxd) {
    size_t n = 0; maxd = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        int d = std::abs(int(a[i]) - int(b[i]));
        if (d) { ++n; if (d > maxd) maxd = d; }
    }
    return n + (a.size() > b.size() ? a.size() - b.size() : b.size() - a.size());
}

static void check_backend(const std::string& f, const std::vector<uint8_t>& d, const std::vector<uint8_t>& ref, int w, int h,
                          parajpeg::Backend be, int threads, uint32_t subseq, parajpeg::PixelFormat fmt, bool force_parallel = false) {
    parajpeg::DecodeOptions o;
    o.backend = be; o.threads = threads; o.speculative_subsequence_bytes = subseq; o.format = fmt; o.collect_stats = true;
    if (force_parallel) o.min_parallel_scan_bytes = 1;
    parajpeg::Image img;
    parajpeg::DecodeStats st;
    auto s = parajpeg::decode(d.data(), d.size(), o, img, &st);
    INFO(f << " backend=" << parajpeg::backend_name(be) << " threads=" << threads << " subseq=" << subseq);
    REQUIRE(s.ok());
    REQUIRE(int(img.width) == w);
    REQUIRE(int(img.height) == h);
    int maxd = 0;
    size_t nd = count_diff(img.pixels, ref, maxd);
    INFO("differing bytes=" << nd << " max abs diff=" << maxd << " sync_iters=" << st.sync_iterations << " subseqs=" << st.subsequences);
    CHECK(nd == 0);
    if (force_parallel && threads > 1) {
        // Every scan that can be parallel must actually have been decoded in parallel.
        for (auto& sc : st.scans) {
            bool refine_ac = sc.ah != 0 && sc.ss != 0;
            INFO("scan " << sc.index << " strategy " << sc.strategy);
            if (sc.blocks > 1 && sc.bytes > 0 && !refine_ac) CHECK(sc.strategy != "sequential");
        }
    }
}

TEST_CASE("decoded pixels are identical to libjpeg-turbo for the whole corpus") {
    auto files = corpus_files();
    REQUIRE(!files.empty());
    int tested = 0;
    for (auto& f : files) {
        auto d = read_file(f);
        parajpeg::ImageInfo info;
        REQUIRE(parajpeg::read_header(d.data(), d.size(), info).ok());
        if (info.arithmetic) {
            parajpeg::Image img;
            CHECK(parajpeg::decode(d.data(), d.size(), {}, img).code == parajpeg::StatusCode::Unsupported);
            continue;
        }
        bool gray = info.components == 1;
        std::vector<uint8_t> ref; int w, h;
        REQUIRE(tj_decode(d, gray ? TJPF_GRAY : TJPF_RGB, ref, w, h));
        auto fmt = gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB;
        check_backend(f, d, ref, w, h, parajpeg::Backend::Scalar, 1, 0, fmt);
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 1, 0, fmt);
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 8, 0, fmt);
        // Force every scan through its parallel strategy (speculative with tiny subsequences, or
        // restart-parallel), with several thread counts.
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 8, 64, fmt, true);
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 3, 256, fmt, true);
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 2, 97, fmt, true);
        if (!gray) {
            std::vector<uint8_t> refbgr; int w2, h2;
            REQUIRE(tj_decode(d, TJPF_BGRA, refbgr, w2, h2));
            check_backend(f, d, refbgr, w2, h2, parajpeg::Backend::Cpu, 4, 0, parajpeg::PixelFormat::BGRA);
        }
        ++tested;
    }
    CHECK(tested >= 50);
}

TEST_CASE("grayscale output of a colour JPEG matches libjpeg-turbo") {
    auto files = corpus_files();
    for (auto& f : files) {
        if (f.find("plasma_512_q75_420.jpg") == std::string::npos) continue;
        auto d = read_file(f);
        std::vector<uint8_t> ref; int w, h;
        REQUIRE(tj_decode(d, TJPF_GRAY, ref, w, h));
        check_backend(f, d, ref, w, h, parajpeg::Backend::Cpu, 4, 0, parajpeg::PixelFormat::Gray);
    }
}
#else
TEST_CASE("bit-exactness tests require TurboJPEG" * doctest::skip(true)) {}
#endif
