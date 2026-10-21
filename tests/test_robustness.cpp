// Truncated / corrupted streams must never crash and must behave predictably.
#include <random>
#include "doctest.h"
#include "parajpeg/parajpeg.h"
#include "test_util.h"

TEST_CASE("truncated streams decode best-effort without crashing") {
    auto files = corpus_files();
    REQUIRE(!files.empty());
    int n = 0;
    for (auto& f : files) {
        if (f.find("4096") != std::string::npos || f.find("1920") != std::string::npos) continue;
        auto d = read_file(f);
        parajpeg::ImageInfo info;
        REQUIRE(parajpeg::read_header(d.data(), d.size(), info).ok());
        if (info.arithmetic) continue;
        for (double frac : {0.95, 0.7, 0.4, 0.1}) {
            size_t len = size_t(d.size() * frac);
            for (auto be : {parajpeg::Backend::Scalar, parajpeg::Backend::Cpu}) {
                parajpeg::DecodeOptions o; o.backend = be; o.threads = 4; o.speculative_subsequence_bytes = 128;
                parajpeg::Image img;
                auto s = parajpeg::decode(d.data(), len, o, img);
                INFO(f << " frac=" << frac);
                CHECK((s.ok() || s.code == parajpeg::StatusCode::Truncated || s.code == parajpeg::StatusCode::Corrupt || s.code == parajpeg::StatusCode::NotJpeg));
            }
        }
        ++n;
    }
    CHECK(n > 10);
}

TEST_CASE("random byte corruption never crashes") {
    auto files = corpus_files();
    std::mt19937 rng(1234);
    int n = 0;
    for (auto& f : files) {
        if (f.find("4096") != std::string::npos || f.find("1920") != std::string::npos) continue;
        auto d = read_file(f);
        for (int iter = 0; iter < 6; ++iter) {
            auto c = d;
            int flips = 1 + int(rng() % 8);
            for (int k = 0; k < flips; ++k) c[rng() % c.size()] = uint8_t(rng());
            for (auto be : {parajpeg::Backend::Scalar, parajpeg::Backend::Cpu}) {
                parajpeg::DecodeOptions o; o.backend = be; o.threads = 4; o.speculative_subsequence_bytes = 128;
                parajpeg::Image img;
                (void)parajpeg::decode(c.data(), c.size(), o, img);
            }
        }
        ++n;
    }
    CHECK(n > 10);
}
