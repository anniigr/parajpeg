// pjbench — decode throughput of parajpeg vs libjpeg-turbo.
//
//   pjbench [--runs N] [--threads 1,2,4,8] [--json out.json] files...        latency mode
//   pjbench --batch W [--runs N] [--json out.json] files...                  batch mode
//
// Latency mode: per image, median wall time of one decode for libjpeg-turbo (TurboJPEG API, accurate
// DCT, one thread — it has no intra-image parallelism), parajpeg scalar, and parajpeg CPU backend for
// each thread count; plus the per-stage breakdown of the fastest configuration.
// Batch mode: W worker threads decode the whole file list (each image single-threaded) — the
// data-loader scenario, where parallelism across images is available.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
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
    fn();  // warm-up
    std::vector<double> t;
    for (int i = 0; i < runs; ++i) {
        auto t0 = Clock::now();
        fn();
        t.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static std::string base(const std::string& f) { return f.substr(f.find_last_of('/') + 1); }

static std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
    return v;
}

#ifdef HAVE_TURBOJPEG
static bool turbo_decode(tjhandle h, const std::vector<uint8_t>& d, int w, int hgt, bool gray, std::vector<uint8_t>& out) {
    return tjDecompress2(h, d.data(), (unsigned long)d.size(), out.data(), w, 0, hgt, gray ? TJPF_GRAY : TJPF_RGB, TJFLAG_ACCURATEDCT) == 0;
}
#endif

int main(int argc, char** argv) {
    int runs = 7, batch = 0;
    std::vector<int> threads = {1, 2, 4, int(std::thread::hardware_concurrency())};
    std::vector<std::string> files;
    std::string json;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--runs" && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = parse_list(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = std::atoi(argv[++i]);
        else if (a == "--json" && i + 1 < argc) json = argv[++i];
        else files.push_back(a);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: pjbench [--runs N] [--threads 1,2,4,8] [--batch W] [--json out.json] files...\n");
        return 2;
    }
    std::string js;
    if (batch > 0) {
        std::vector<std::vector<uint8_t>> data;
        double mpix = 0;
        for (auto& f : files) {
            data.push_back(read_file(f));
            parajpeg::ImageInfo info;
            parajpeg::read_header(data.back().data(), data.back().size(), info);
            mpix += double(info.width) * info.height / 1e6;
        }
        auto run_batch = [&](bool turbo) {
            return median_ms(runs, [&] {
                std::atomic<size_t> next{0};
                std::vector<std::thread> ws;
                for (int w = 0; w < batch; ++w)
                    ws.emplace_back([&, turbo] {
#ifdef HAVE_TURBOJPEG
                        tjhandle h = turbo ? tjInitDecompress() : nullptr;
#endif
                        std::vector<uint8_t> out;
                        for (size_t i; (i = next.fetch_add(1)) < data.size();) {
                            parajpeg::ImageInfo info;
                            parajpeg::read_header(data[i].data(), data[i].size(), info);
                            const bool gray = info.components == 1;
                            out.resize(size_t(info.width) * info.height * (gray ? 1 : 3));
#ifdef HAVE_TURBOJPEG
                            if (turbo) { turbo_decode(h, data[i], int(info.width), int(info.height), gray, out); continue; }
#endif
                            parajpeg::DecodeOptions o;
                            o.threads = 1;
                            o.format = gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB;
                            parajpeg::decode_into(data[i].data(), data[i].size(), o, out.data(), size_t(info.width) * (gray ? 1 : 3));
                        }
#ifdef HAVE_TURBOJPEG
                        if (h) tjDestroy(h);
#endif
                    });
                for (auto& t : ws) t.join();
            });
        };
        double tt = -1;
#ifdef HAVE_TURBOJPEG
        tt = run_batch(true);
#endif
        double tp = run_batch(false);
        std::printf("| workers | images | Mpix | turbo ms | parajpeg ms | turbo img/s | parajpeg img/s | ratio |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        std::printf("| %d | %zu | %.1f | %.1f | %.1f | %.0f | %.0f | %.2fx |\n", batch, files.size(), mpix, tt, tp, tt > 0 ? files.size() / (tt / 1e3) : 0,
                    files.size() / (tp / 1e3), tt > 0 ? tt / tp : 0);
        char buf[256];
        std::snprintf(buf, sizeof buf, "{\"mode\":\"batch\",\"workers\":%d,\"images\":%zu,\"mpix\":%.3f,\"turbo_ms\":%.4f,\"parajpeg_ms\":%.4f}\n", batch, files.size(), mpix, tt, tp);
        js = buf;
    } else {
        std::printf("| image | size | MP | KB | turbo 1T ms | scalar ms |");
        for (int t : threads) std::printf(" %dT ms |", t);
        std::printf(" best vs turbo | stages at %dT (setup/entropy/idct/color ms) |\n|---|---|---:|---:|---:|---:|", threads.back());
        for (size_t i = 0; i < threads.size(); ++i) std::printf("---:|");
        std::printf("---:|---|\n");
        js = "[\n";
        for (size_t fi = 0; fi < files.size(); ++fi) {
            const std::string& f = files[fi];
            auto d = read_file(f);
            parajpeg::ImageInfo info;
            if (!parajpeg::read_header(d.data(), d.size(), info).ok() || info.arithmetic) continue;
            const bool gray = info.components == 1;
            const auto fmt = gray ? parajpeg::PixelFormat::Gray : parajpeg::PixelFormat::RGB;
            const double mp = double(info.width) * info.height / 1e6;
            std::vector<uint8_t> out(size_t(info.width) * info.height * (gray ? 1 : 3));
            double turbo = -1;
#ifdef HAVE_TURBOJPEG
            {
                tjhandle h = tjInitDecompress();
                turbo = median_ms(runs, [&] { turbo_decode(h, d, int(info.width), int(info.height), gray, out); });
                tjDestroy(h);
            }
#endif
            auto run = [&](parajpeg::Backend be, int th, parajpeg::DecodeStats* st) {
                parajpeg::DecodeOptions o;
                o.backend = be;
                o.threads = th;
                o.format = fmt;
                return median_ms(runs, [&] { parajpeg::decode_into(d.data(), d.size(), o, out.data(), size_t(info.width) * (gray ? 1 : 3), st); });
            };
            parajpeg::DecodeStats st;
            double scalar = run(parajpeg::Backend::Scalar, 1, nullptr);
            std::vector<double> tms;
            for (int t : threads) tms.push_back(run(parajpeg::Backend::Cpu, t, t == threads.back() ? &st : nullptr));
            double best = *std::min_element(tms.begin(), tms.end());
            char sub[16];
            const char* s = info.components == 1 ? "gray" : info.subsampling == parajpeg::Subsampling::S420 ? "420" : info.subsampling == parajpeg::Subsampling::S444 ? "444" : info.subsampling == parajpeg::Subsampling::S422 ? "422" : "other";
            std::snprintf(sub, sizeof sub, "%s%s%s", s, info.progressive ? " prog" : "", info.restart_interval ? " rst" : "");
            std::printf("| %s | %s | %.1f | %.0f | %.2f | %.2f |", base(f).c_str(), sub, mp, d.size() / 1024.0, turbo, scalar);
            for (double t : tms) std::printf(" %.2f |", t);
            std::printf(" %.2fx | %.1f / %.1f / %.1f / %.1f |\n", turbo > 0 ? turbo / best : 0, st.setup_ms, st.entropy_ms, st.idct_ms, st.color_ms);
            std::fflush(stdout);
            std::ostringstream o;
            o << "  {\"file\":\"" << base(f) << "\",\"kind\":\"" << sub << "\",\"width\":" << info.width << ",\"height\":" << info.height << ",\"bytes\":" << d.size()
              << ",\"turbo_ms\":" << turbo << ",\"scalar_ms\":" << scalar << ",\"threads\":[";
            for (size_t i = 0; i < threads.size(); ++i) o << (i ? "," : "") << threads[i];
            o << "],\"cpu_ms\":[";
            for (size_t i = 0; i < tms.size(); ++i) o << (i ? "," : "") << tms[i];
            o << "],\"stages\":{\"setup\":" << st.setup_ms << ",\"entropy\":" << st.entropy_ms << ",\"idct\":" << st.idct_ms << ",\"color\":" << st.color_ms
              << "},\"subsequences\":" << st.subsequences << ",\"sync_iterations\":" << st.sync_iterations << ",\"restart_segments\":" << st.restart_segments
              << ",\"waves\":" << st.waves << "}" << (fi + 1 < files.size() ? "," : "") << "\n";
            js += o.str();
        }
        js += "]\n";
    }
    if (!json.empty()) std::ofstream(json) << js;
    return 0;
}
