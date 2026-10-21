// pjbench: throughput of parajpeg backends vs libjpeg-turbo on a set of files. Prints a markdown table
// and optionally JSON. Metrics: megapixels/s and compressed MB/s (median of N runs).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "parajpeg/parajpeg.h"
#ifdef HAVE_TURBOJPEG
#include <turbojpeg.h>
#endif

using Clock = std::chrono::steady_clock;

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class F>
static double median_ms(int runs, F&& fn) {
    std::vector<double> t;
    for (int i = 0; i < runs; ++i) {
        auto t0 = Clock::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char** argv) {
    int runs = 7, threads = 0;
    std::vector<std::string> files;
    std::string json;
    uint32_t subseq = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--subseq" && i + 1 < argc) subseq = uint32_t(std::atoi(argv[++i]));
        else if (a == "--json" && i + 1 < argc) json = argv[++i];
        else files.push_back(a);
    }
    if (files.empty()) { std::fprintf(stderr, "usage: pjbench [--runs N] [--threads N] [--subseq B] [--json out.json] files...\n"); return 2; }
    std::printf("| file | size (KB) | Mpix | turbo 1T ms | scalar ms | cpu 1T ms | cpu NT ms | cpu NT Mpix/s | cpu NT MB/s | speedup vs turbo | sync iters |\n");
    std::printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    std::string js = "[\n";
    for (auto& f : files) {
        auto d = read_file(f);
        parajpeg::ImageInfo info;
        if (!parajpeg::read_header(d.data(), d.size(), info).ok() || info.arithmetic) continue;
        bool gray = info.components == 1;
        auto fmt = gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB;
        double mpix = double(info.width) * info.height / 1e6;
        double turbo = -1;
#ifdef HAVE_TURBOJPEG
        {
            tjhandle h = tjInitDecompress();
            std::vector<uint8_t> out(size_t(info.width) * info.height * (gray ? 1 : 3));
            turbo = median_ms(runs, [&] { tjDecompress2(h, d.data(), (unsigned long)d.size(), out.data(), info.width, 0, info.height, gray ? TJPF_GRAY : TJPF_RGB, TJFLAG_ACCURATEDCT); });
            tjDestroy(h);
        }
#endif
        parajpeg::Image img;
        parajpeg::DecodeStats st;
        auto run = [&](parajpeg::Backend be, int th) {
            parajpeg::DecodeOptions o; o.backend = be; o.threads = th; o.format = fmt; o.speculative_subsequence_bytes = subseq; o.collect_stats = true;
            return median_ms(runs, [&] { parajpeg::decode(d.data(), d.size(), o, img, &st); });
        };
        double scalar = run(parajpeg::Backend::Scalar, 1);
        double cpu1 = run(parajpeg::Backend::Cpu, 1);
        double cpuN = run(parajpeg::Backend::Cpu, threads);
        std::string base = f.substr(f.find_last_of('/') + 1);
        std::printf("| %s | %.0f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f | %.1f | %.2fx | %u |\n", base.c_str(), d.size() / 1024.0, mpix, turbo, scalar, cpu1, cpuN,
                    mpix / (cpuN / 1000.0), d.size() / 1e6 / (cpuN / 1000.0), turbo > 0 ? turbo / cpuN : 0.0, st.sync_iterations);
        char buf[512];
        std::snprintf(buf, sizeof buf, "  {\"file\":\"%s\",\"bytes\":%zu,\"width\":%u,\"height\":%u,\"turbo_ms\":%.4f,\"scalar_ms\":%.4f,\"cpu1_ms\":%.4f,\"cpuN_ms\":%.4f,\"threads\":%u,\"sync_iters\":%u,\"subsequences\":%u},\n",
                      base.c_str(), d.size(), info.width, info.height, turbo, scalar, cpu1, cpuN, st.threads_used, st.sync_iterations, st.subsequences);
        js += buf;
    }
    js += "]\n";
    if (!json.empty()) { std::ofstream(json) << js; }
    return 0;
}
