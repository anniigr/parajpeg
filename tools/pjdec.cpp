// pjdec: decode a JPEG to PPM/PGM with a chosen backend, printing per-stage timings.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "parajpeg/parajpeg.h"

static std::vector<uint8_t> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: pjdec in.jpg [out.ppm] [--backend scalar|cpu|cuda] [--threads N] [--subseq BYTES] [--gray] [--nofancy]\n");
        return 2;
    }
    const char* in = argv[1];
    const char* out = nullptr;
    parajpeg::DecodeOptions o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            std::string b = argv[++i];
            o.backend = b == "scalar" ? parajpeg::Backend::Scalar : b == "cuda" ? parajpeg::Backend::Cuda : parajpeg::Backend::Cpu;
        } else if (a == "--threads" && i + 1 < argc) o.threads = std::atoi(argv[++i]);
        else if (a == "--subseq" && i + 1 < argc) o.speculative_subsequence_bytes = uint32_t(std::atoi(argv[++i]));
        else if (a == "--gray") o.format = parajpeg::PixelFormat::Gray;
        else if (a == "--nofancy") o.fancy_upsampling = false;
        else out = argv[i];
    }
    auto data = read_file(in);
    if (data.empty()) { std::fprintf(stderr, "cannot read %s\n", in); return 1; }
    parajpeg::Image img;
    parajpeg::DecodeStats st;
    o.collect_stats = true;
    parajpeg::Status s = parajpeg::decode(data.data(), data.size(), o, img, &st);
    std::fprintf(stderr, "%s: %ux%u ch=%d backend=%s threads=%u | parse %.3f ms, setup %.3f ms, entropy %.3f ms, idct %.3f ms, color %.3f ms, total %.3f ms | restart=%u subseq=%u sync_iters=%u\n",
                 in, img.width, img.height, img.channels, st.backend.c_str(), st.threads_used, st.parse_ms, st.setup_ms, st.entropy_ms, st.idct_ms, st.color_ms, st.total_ms,
                 st.restart_segments, st.subsequences, st.sync_iterations);
    if (std::getenv("PJ_SCANS"))
        for (auto& sc : st.scans)
            std::fprintf(stderr, "  scan %2d wave %d comps=%d(c%d) Ss=%2d Se=%2d Ah=%d Al=%d %8zu B %7u blk %-16s subseq=%u sync=%u %.3f ms\n", sc.index, sc.wave, sc.ncomp,
                         sc.first_component, sc.ss, sc.se, sc.ah, sc.al, sc.bytes, sc.blocks, sc.strategy.c_str(), sc.subsequences, sc.sync_iterations, sc.ms);
    for (auto& w : st.warnings) std::fprintf(stderr, "  warning: %s\n", w.c_str());
    if (!s.ok()) std::fprintf(stderr, "  status %d: %s\n", int(s.code), s.message.c_str());
    if (!s.ok() && s.code != parajpeg::StatusCode::Truncated) return 1;
    if (out) {
        FILE* f = std::fopen(out, "wb");
        if (!f) return 1;
        std::fprintf(f, "%s\n%u %u\n255\n", img.channels == 1 ? "P5" : "P6", img.width, img.height);
        for (uint32_t y = 0; y < img.height; ++y) std::fwrite(img.row(y), 1, size_t(img.width) * img.channels, f);
        std::fclose(f);
    }
    return 0;
}
