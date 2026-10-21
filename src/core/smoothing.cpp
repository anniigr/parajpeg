#include "smoothing.h"

#include <algorithm>
#include <cstdint>

#include "thread_pool.h"

namespace parajpeg::detail {

namespace {
constexpr int kSaved = 10;  // libjpeg SAVED_COEFS: DC + first 9 AC in zig-zag order
// Natural positions of zig-zag 0..9.
constexpr int kPos[kSaved] = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24};

inline int predict(int64_t num, int64_t q, int al) {
    int pred;
    if (num >= 0) {
        pred = int(((q << 7) + num) / (q << 8));
        if (al > 0 && pred >= (1 << al)) pred = (1 << al) - 1;
    } else {
        pred = int(((q << 7) - num) / (q << 8));
        if (al > 0 && pred >= (1 << al)) pred = (1 << al) - 1;
        pred = -pred;
    }
    return pred;
}
}  // namespace

bool apply_block_smoothing(Frame& f, const std::vector<ScanTrace>& scans, ThreadPool* pool, int threads) {
    if (!f.progressive || scans.empty()) return false;
    const int nc = f.ncomp;
    // Replay jdphuff.c start_pass_phuff_decoder's progression bookkeeping.
    int cur[kMaxComponents][64], prev[kMaxComponents][64];
    for (int c = 0; c < nc; ++c)
        for (int k = 0; k < 64; ++k) cur[c][k] = prev[c][k] = -1;
    for (size_t si = 0; si < f.scans.size(); ++si) {
        const ScanHeader& sh = f.scans[si];
        for (int i = 0; i < sh.ncomp; ++i) {
            const int c = sh.comp_index[i];
            for (int k = std::min<int>(sh.ss, 1); k <= std::max<int>(sh.se, 9); ++k) prev[c][k] = si > 0 ? cur[c][k] : 0;
            for (int k = sh.ss; k <= sh.se; ++k) cur[c][k] = sh.al;
        }
    }
    // smoothing_ok()
    bool useful = false;
    int latch[kMaxComponents][kSaved], prev_latch[kMaxComponents][kSaved];
    for (int c = 0; c < nc; ++c) {
        const auto& q = f.comps[c].qt_latched.q;
        for (int k = 0; k < kSaved; ++k)
            if (q[size_t(kPos[k])] == 0) return false;
        if (cur[c][0] < 0) return false;
        latch[c][0] = cur[c][0];
        for (int k = 1; k < kSaved; ++k) {
            prev_latch[c][k] = f.scans.size() > 1 ? prev[c][k] : -1;
            latch[c][k] = cur[c][k];
            if (cur[c][k] != 0) useful = true;
        }
    }
    if (!useful) return false;

    // libjpeg's last_good_iMCU_row: iMCU row of the last MCU of the final scan decoded before the
    // data ran out (or the last row).
    const uint32_t total_rows = f.mcus_y;
    uint32_t last_good = total_rows - 1;
    {
        const ScanHeader& sh = f.scans.back();
        const ScanTrace& tr = scans.back();
        if (tr.first_skip != 0xFFFFFFFFu) {
            ScanLayout L;
            L.init(f, sh);
            const uint32_t mcu = tr.first_skip / uint32_t(L.blocks_per_mcu);
            const uint32_t good_mcu = mcu ? mcu - 1 : 0;
            if (L.interleaved) last_good = good_mcu / f.mcus_x;
            else last_good = (good_mcu / std::max<uint32_t>(L.bw, 1)) / f.comps[L.comp].v;
        }
    }

    for (int c = 0; c < nc; ++c) {
        Component& comp = f.comps[c];
        const uint32_t pw = comp.padded_blocks_w, ph = comp.padded_blocks_h;
        const uint32_t v = comp.v;
        // Snapshot of DC values (neighbours must see the unsmoothed values).
        std::vector<int> dc(size_t(pw) * ph);
        for (size_t b = 0; b < dc.size(); ++b) dc[b] = comp.coefs[b * kDctSize2];
        const auto& qv = comp.qt_latched.q;
        const int64_t Q00 = qv[0], Q01 = qv[1], Q10 = qv[8], Q20 = qv[16], Q11 = qv[9], Q02 = qv[2], Q03 = qv[3], Q12 = qv[10], Q21 = qv[17], Q30 = qv[24];
        const uint32_t height_in_blocks = comp.blocks_h, width_in_blocks = comp.blocks_w;
        const uint32_t last_block_column = width_in_blocks - 1;

        auto do_row = [&](uint32_t r) {
            const int* cb = (r > last_good) ? prev_latch[c] : latch[c];
            int bits[kSaved];
            bits[0] = latch[c][0];
            for (int k = 1; k < kSaved; ++k) bits[k] = cb[k];
            const bool change_dc = bits[1] == -1 && bits[2] == -1 && bits[3] == -1 && bits[4] == -1 && bits[5] == -1 &&
                                   bits[6] == -1 && bits[7] == -1 && bits[8] == -1 && bits[9] == -1;
            uint32_t block_rows = v;
            if (r == total_rows - 1) {
                block_rows = height_in_blocks % v;
                if (block_rows == 0) block_rows = v;
            }
            const uint32_t image_block_rows = block_rows * total_rows;  // sic (libjpeg)
            for (uint32_t br = 0; br < block_rows; ++br) {
                const uint32_t by = r * v + br;
                const uint32_t ibr = r * block_rows + br;  // image_block_row, sic
                const uint32_t row_cur = by;
                const uint32_t row_prev = ibr > 0 ? by - 1 : row_cur;
                const uint32_t row_pp = ibr > 1 ? by - 2 : row_prev;
                const uint32_t row_next = ibr + 1 < image_block_rows ? by + 1 : row_cur;
                const uint32_t row_nn = ibr + 2 < image_block_rows ? by + 2 : row_next;
                auto D = [&](uint32_t row, uint32_t col) { return dc[size_t(row) * pw + col]; };
                int DC01, DC02, DC03, DC04, DC05, DC06, DC07, DC08, DC09, DC10, DC11, DC12, DC13, DC14, DC15, DC16, DC17, DC18, DC19,
                    DC20, DC21, DC22, DC23, DC24, DC25;
                DC01 = DC02 = DC03 = DC04 = DC05 = D(row_pp, 0);
                DC06 = DC07 = DC08 = DC09 = DC10 = D(row_prev, 0);
                DC11 = DC12 = DC13 = DC14 = DC15 = D(row_cur, 0);
                DC16 = DC17 = DC18 = DC19 = DC20 = D(row_next, 0);
                DC21 = DC22 = DC23 = DC24 = DC25 = D(row_nn, 0);
                for (uint32_t bn = 0; bn <= last_block_column; ++bn) {
                    if (bn == 0 && bn < last_block_column) {
                        DC04 = DC05 = D(row_pp, bn + 1);
                        DC09 = DC10 = D(row_prev, bn + 1);
                        DC14 = DC15 = D(row_cur, bn + 1);
                        DC19 = DC20 = D(row_next, bn + 1);
                        DC24 = DC25 = D(row_nn, bn + 1);
                    }
                    if (bn + 1 < last_block_column) {
                        DC05 = D(row_pp, bn + 2);
                        DC10 = D(row_prev, bn + 2);
                        DC15 = D(row_cur, bn + 2);
                        DC20 = D(row_next, bn + 2);
                        DC25 = D(row_nn, bn + 2);
                    }
                    int16_t* w = comp.coefs + (size_t(by) * pw + bn) * kDctSize2;
                    int al;
                    if ((al = bits[1]) != 0 && w[1] == 0) {
                        int64_t num = Q00 * (change_dc ? (-DC01 - DC02 + DC04 + DC05 - 3 * DC06 + 13 * DC07 - 13 * DC09 + 3 * DC10 - 3 * DC11 +
                                                          38 * DC12 - 38 * DC14 + 3 * DC15 - 3 * DC16 + 13 * DC17 - 13 * DC19 + 3 * DC20 - DC21 -
                                                          DC22 + DC24 + DC25)
                                                       : (-7 * DC11 + 50 * DC12 - 50 * DC14 + 7 * DC15));
                        w[1] = int16_t(predict(num, Q01, al));
                    }
                    if ((al = bits[2]) != 0 && w[8] == 0) {
                        int64_t num = Q00 * (change_dc ? (-DC01 - 3 * DC02 - 3 * DC03 - 3 * DC04 - DC05 - DC06 + 13 * DC07 + 38 * DC08 + 13 * DC09 -
                                                          DC10 + DC16 - 13 * DC17 - 38 * DC18 - 13 * DC19 + DC20 + DC21 + 3 * DC22 + 3 * DC23 +
                                                          3 * DC24 + DC25)
                                                       : (-7 * DC03 + 50 * DC08 - 50 * DC18 + 7 * DC23));
                        w[8] = int16_t(predict(num, Q10, al));
                    }
                    if ((al = bits[3]) != 0 && w[16] == 0) {
                        int64_t num = Q00 * (change_dc ? (DC03 + 2 * DC07 + 7 * DC08 + 2 * DC09 - 5 * DC12 - 14 * DC13 - 5 * DC14 + 2 * DC17 +
                                                          7 * DC18 + 2 * DC19 + DC23)
                                                       : (-DC03 + 13 * DC08 - 24 * DC13 + 13 * DC18 - DC23));
                        w[16] = int16_t(predict(num, Q20, al));
                    }
                    if ((al = bits[4]) != 0 && w[9] == 0) {
                        int64_t num = Q00 * (change_dc ? (-DC01 + DC05 + 9 * DC07 - 9 * DC09 - 9 * DC17 + 9 * DC19 + DC21 - DC25)
                                                       : (DC10 + DC16 - 10 * DC17 + 10 * DC19 - DC02 - DC20 + DC22 - DC24 + DC04 - DC06 +
                                                          10 * DC07 - 10 * DC09));
                        w[9] = int16_t(predict(num, Q11, al));
                    }
                    if ((al = bits[5]) != 0 && w[2] == 0) {
                        int64_t num = Q00 * (change_dc ? (2 * DC07 - 5 * DC08 + 2 * DC09 + DC11 + 7 * DC12 - 14 * DC13 + 7 * DC14 + DC15 +
                                                          2 * DC17 - 5 * DC18 + 2 * DC19)
                                                       : (-DC11 + 13 * DC12 - 24 * DC13 + 13 * DC14 - DC15));
                        w[2] = int16_t(predict(num, Q02, al));
                    }
                    if (change_dc) {
                        if ((al = bits[6]) != 0 && w[3] == 0)
                            w[3] = int16_t(predict(Q00 * (DC07 - DC09 + 2 * DC12 - 2 * DC14 + DC17 - DC19), Q03, al));
                        if ((al = bits[7]) != 0 && w[10] == 0)
                            w[10] = int16_t(predict(Q00 * (DC07 - 3 * DC08 + DC09 - DC17 + 3 * DC18 - DC19), Q12, al));
                        if ((al = bits[8]) != 0 && w[17] == 0)
                            w[17] = int16_t(predict(Q00 * (DC07 - DC09 - 3 * DC12 + 3 * DC14 + DC17 - DC19), Q21, al));
                        if ((al = bits[9]) != 0 && w[24] == 0)
                            w[24] = int16_t(predict(Q00 * (DC07 + 2 * DC08 + DC09 - DC17 - 2 * DC18 - DC19), Q30, al));
                        int64_t num = Q00 * (-2 * DC01 - 6 * DC02 - 8 * DC03 - 6 * DC04 - 2 * DC05 - 6 * DC06 + 6 * DC07 + 42 * DC08 + 6 * DC09 -
                                             6 * DC10 - 8 * DC11 + 42 * DC12 + 152 * DC13 + 42 * DC14 - 8 * DC15 - 6 * DC16 + 6 * DC17 +
                                             42 * DC18 + 6 * DC19 - 6 * DC20 - 2 * DC21 - 6 * DC22 - 8 * DC23 - 6 * DC24 - 2 * DC25);
                        int pred = num >= 0 ? int(((Q00 << 7) + num) / (Q00 << 8)) : -int(((Q00 << 7) - num) / (Q00 << 8));
                        w[0] = int16_t(pred);
                    }
                    DC01 = DC02; DC02 = DC03; DC03 = DC04; DC04 = DC05;
                    DC06 = DC07; DC07 = DC08; DC08 = DC09; DC09 = DC10;
                    DC11 = DC12; DC12 = DC13; DC13 = DC14; DC14 = DC15;
                    DC16 = DC17; DC17 = DC18; DC18 = DC19; DC19 = DC20;
                    DC21 = DC22; DC22 = DC23; DC23 = DC24; DC24 = DC25;
                }
            }
        };
        if (pool && threads > 1) pool->parallel_for(total_rows, [&](uint32_t r, int) { do_row(r); }, threads);
        else for (uint32_t r = 0; r < total_rows; ++r) do_row(r);
    }
    return true;
}

}  // namespace parajpeg::detail
