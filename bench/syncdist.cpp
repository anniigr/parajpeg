// pjsyncdist: measures how quickly a Huffman decoder started at an arbitrary byte of a baseline scan
// falls into step with the true symbol stream (the premise of speculative decoding).
//
//   pjsyncdist [--samples N] [--csv out.csv] files...
//
// For every file, the scan is first decoded sequentially, recording the decoder state at every block
// boundary. Then N start offsets, evenly spaced over the scan, are decoded speculatively, exactly as
// pass 1 of the parallel decoder does: empty state, MCU phase guessed from the byte offset. For each
// start, two distances are reported, in bytes from the start offset to the first block boundary where
//   * bit: the canonical bit position coincides with a true block boundary;
//   * state: bit position AND MCU phase (block-in-MCU slot) coincide, i.e. the decoder state matches
//     and every later symbol is decoded identically.
// Detection happens at block boundaries, so both distances are upper bounds within one block.
// Baseline scans without restart markers only (the ones the speculative decoder handles).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/bitreader.h"
#include "core/entropy.h"
#include "core/parser.h"

using namespace parajpeg::detail;

namespace {

std::vector<uint8_t> read_file(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Consumes one baseline block (DC difference + AC run/size symbols) without producing output.
void skip_block(BitReader& br, const HuffTable& dct, const HuffTable& act) {
    bool bad = false;
    int s = huff_decode(br, dct, bad);
    if (s) br.get_bits(s > 16 ? 16 : s);
    for (int k = 1; k < 64;) {
        int rs = huff_decode(br, act, bad);
        int r = rs >> 4, sz = rs & 15;
        if (sz) {
            k += r + 1;
            br.get_bits(sz);
        } else if (r == 15) {
            k += 16;
        } else {
            break;
        }
    }
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(p * double(v.size() - 1) + 0.5))];
}

}  // namespace

int main(int argc, char** argv) {
    int samples = 2000;
    const char* csv_path = nullptr;
    std::vector<const char*> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--samples" && i + 1 < argc) samples = std::atoi(argv[++i]);
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else files.push_back(argv[i]);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: pjsyncdist [--samples N] [--csv out.csv] files...\n");
        return 2;
    }
    FILE* csv = csv_path ? std::fopen(csv_path, "w") : nullptr;
    if (csv) std::fprintf(csv, "file,offset,bit_bytes,state_bytes\n");
    std::printf("| image | scan KB | starts | bit sync: median / p99 / max B | state sync: median / p99 / max B | not synced |\n");
    std::printf("|---|---:|---:|---|---|---:|\n");
    for (const char* path : files) {
        auto data = read_file(path);
        ParseResult pr;
        if (!parse_stream(data.data(), data.size(), pr, false)) continue;
        Frame& f = pr.frame;
        if (f.progressive || f.arithmetic || f.restart_interval || f.scans.size() != 1) {
            std::fprintf(stderr, "%s: skipped (needs a single-scan baseline image without restart markers)\n", path);
            continue;
        }
        const ScanHeader& sh = f.scans[0];
        ScanLayout L;
        L.init(f, sh);
        const HuffTable* dc[kMaxComponents] = {};
        const HuffTable* ac[kMaxComponents] = {};
        for (int i = 0; i < sh.ncomp; ++i) {
            dc[sh.comp_index[i]] = &f.dc_tables[sh.td[i]];
            ac[sh.comp_index[i]] = &f.ac_tables[sh.ta[i]];
        }
        const size_t begin = sh.data_offset, end = sh.data_offset + sh.data_size, nbytes = end - begin;

        // Ground truth: block-in-MCU slot at every true block boundary, keyed by canonical bit position.
        std::unordered_map<size_t, uint8_t> truth;
        truth.reserve(L.total_blocks * 2);
        {
            BitReader br(data.data(), begin, end, begin);
            BlockCursor cur;
            cur.seek(L, 0);
            while (!cur.done()) {
                truth.emplace(br.canonical_bitpos(), uint8_t(cur.slot));
                const int c = cur.comp();
                skip_block(br, *dc[c], *ac[c]);
                cur.advance();
            }
        }

        std::vector<double> bit_d, state_d;
        int unsynced = 0;
        for (int s = 0; s < samples; ++s) {
            // Evenly spaced starts, avoiding the first and last 1% of the scan.
            const size_t off = begin + nbytes / 100 + size_t((double(s) + 0.5) / samples * double(nbytes) * 0.98);
            BitReader br(data.data(), begin, end, off);
            BlockCursor cur;
            const uint32_t guess = uint32_t(uint64_t(off - begin) * L.total_blocks / nbytes);
            cur.seek(L, std::min(guess, L.total_blocks - 1));
            cur.ordinal = 0;  // only the slot phase matters; keep done() from firing
            double db = -1, ds = -1;
            for (;;) {
                const size_t bp = br.canonical_bitpos();
                if (bp >= end * 8) break;
                auto it = truth.find(bp);
                if (it != truth.end()) {
                    const double d = double(bp - off * 8) / 8.0;
                    if (db < 0) db = d;
                    if (it->second == cur.slot) { ds = d; break; }
                }
                const int c = cur.comp();
                skip_block(br, *dc[c], *ac[c]);
                cur.advance();
            }
            if (ds < 0) { ++unsynced; continue; }
            bit_d.push_back(db);
            state_d.push_back(ds);
            if (csv) std::fprintf(csv, "%s,%zu,%.3f,%.3f\n", path, off - begin, db, ds);
        }
        std::string name = path;
        name = name.substr(name.find_last_of('/') + 1);
        std::printf("| %s | %.0f | %d | %.0f / %.0f / %.0f | %.0f / %.0f / %.0f | %d |\n", name.c_str(), double(nbytes) / 1024.0, samples,
                    pct(bit_d, 0.5), pct(bit_d, 0.99), pct(bit_d, 1.0), pct(state_d, 0.5), pct(state_d, 0.99), pct(state_d, 1.0), unsynced);
        std::fflush(stdout);
    }
    if (csv) std::fclose(csv);
    return 0;
}
