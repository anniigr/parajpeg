#include <algorithm>
#include "doctest.h"
#include "parajpeg/parajpeg.h"
#include "test_util.h"

TEST_CASE("read_header reports geometry and features for every corpus file") {
    auto files = corpus_files();
    REQUIRE(!files.empty());
    for (auto& f : files) {
        auto d = read_file(f);
        parajpeg::ImageInfo info;
        auto s = parajpeg::read_header(d.data(), d.size(), info);
        INFO(f);
        CHECK(s.ok());
        CHECK(info.width > 0);
        CHECK(info.height > 0);
        CHECK((info.components == 1 || info.components == 3));
        if (f.find("_prog") != std::string::npos) CHECK(info.progressive);
        if (f.find("_arith") != std::string::npos) CHECK(info.arithmetic);
        if (f.find("_rst") != std::string::npos) CHECK(info.restart_interval > 0);
        if (f.find("_gray") != std::string::npos) CHECK(info.subsampling == parajpeg::Subsampling::Gray);
        if (f.find("_420") != std::string::npos && f.find("gray") == std::string::npos) CHECK(info.subsampling == parajpeg::Subsampling::S420);
        if (f.find("_444") != std::string::npos && f.find("gray") == std::string::npos) CHECK(info.subsampling == parajpeg::Subsampling::S444);
    }
}

TEST_CASE("non-JPEG input is rejected") {
    uint8_t junk[16] = {0};
    parajpeg::ImageInfo info;
    CHECK(parajpeg::read_header(junk, sizeof junk, info).code == parajpeg::StatusCode::NotJpeg);
    parajpeg::Image img;
    CHECK(!parajpeg::decode(junk, sizeof junk, {}, img).ok());
}
