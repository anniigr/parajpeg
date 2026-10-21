// Entropy decoding engine.
//
// Strategies, chosen per scan:
//   * Sequential           — reference decoder with libjpeg's restart-marker resynchronisation.
//   * RestartParallel      — restart intervals are independent; decoded concurrently.
//   * Speculative          — no restart markers: self-synchronising subsequence decoding (baseline,
//                            progressive DC-first and AC-first scans). See decode_speculative().
// Scans form a dependency DAG over (component, spectral band); independent scans of a progressive
// (or multi-scan baseline) image are decoded concurrently, wave by wave.
#include "entropy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include "parser.h"
#include "thread_pool.h"

namespace parajpeg::detail {

void ScanLayout::init(const Frame& f, const ScanHeader& s) {
    frame = &f;
    scan = &s;
    interleaved = s.ncomp > 1;
    if (interleaved) {
        blocks_per_mcu = 0;
        for (int i = 0; i < s.ncomp; ++i) {
            const Component& c = f.comps[s.comp_index[i]];
            for (int by = 0; by < c.v; ++by)
                for (int bx = 0; bx < c.h; ++bx) slots[blocks_per_mcu++] = Slot{s.comp_index[i], bx, by};
        }
        total_mcus = f.mcus_x * f.mcus_y;
        total_blocks = total_mcus * uint32_t(blocks_per_mcu);
    } else {
        comp = s.comp_index[0];
        const Component& c = f.comps[comp];
        bw = c.blocks_w;
        bh = c.blocks_h;
        blocks_per_mcu = 1;
        slots[0] = Slot{comp, 0, 0};
        total_mcus = bw * bh;
        total_blocks = total_mcus;
    }
}

namespace {

using Clock = std::chrono::steady_clock;

enum class Mode : uint8_t { Baseline, DcFirst, DcRefine, AcFirst, AcRefine };

// Tables are snapshotted per scan because DHT segments may redefine them between scans and scans
// may run concurrently.
struct ScanTables {
    HuffTable dc[4], ac[4];
};

struct Ctx {
    const uint8_t* data = nullptr;
    const Frame* frame = nullptr;
    const ScanHeader* sh = nullptr;
    ScanLayout L;
    const HuffTable* dc[kMaxComponents]{};
    const HuffTable* ac[kMaxComponents]{};
    Mode mode = Mode::Baseline;
    int ss = 0, se = 63, al = 0;
    size_t begin = 0, end = 0;  // entropy-coded bytes
    IdctFn fused = nullptr;     // baseline: reconstruct blocks immediately into sample planes
};

// Fused reconstruction of a completed baseline block held in `scratch`; leaves scratch zeroed.
inline void emit_block(const Ctx& C, const BlockCursor& cur, int16_t* scratch) {
    int c;
    uint32_t bx, by;
    cur.coords(c, bx, by);
    const Component& comp = C.frame->comps[c];
    C.fused(scratch, comp.qt_latched.q.data(), comp.plane + size_t(by) * 8 * comp.plane_stride + size_t(bx) * 8, comp.plane_stride);
    std::memset(scratch, 0, kDctSize2 * sizeof(int16_t));
}

// Error/warning flags accumulated by one decoding task.
struct Flags {
    bool bad = false, premature = false, resync = false;
};

inline int16_t shl16(int v, int al) { return int16_t(unsigned(v) << al); }

// ------------------------------------------------------------------------------------------------
// Block decoders. Each consumes exactly one block (or one EOB-run step) and updates `eobrun`.
// They mirror libjpeg-turbo's jdhuff.c / jdphuff.c bit for bit, including corrupt-data behaviour.
// ------------------------------------------------------------------------------------------------

inline int decode_dc_diff(BitReader& br, const HuffTable& t, Flags& fl) {
    ensure32(br);
    int s = huff_decode_nocheck(br, t, fl.bad);
    if (!s) return 0;
    if (s > 16) { fl.bad = true; s = 16; }
    return huff_extend(br.get_bits_nocheck(s), s);
}

inline void baseline_block(BitReader& br, const HuffTable& dct, const HuffTable& act, int16_t* blk, int& pred, Flags& fl) {
    pred += decode_dc_diff(br, dct, fl);
    blk[0] = int16_t(pred);
    for (int k = 1; k < 64;) {
        ensure32(br);
        int rs = huff_decode_nocheck(br, act, fl.bad);
        int r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            blk[kNaturalOrder[k]] = int16_t(huff_extend(br.get_bits_nocheck(s), s));
            ++k;
        } else if (r == 15) {
            k += 16;
        } else {
            break;
        }
    }
}

inline void ac_first_block(BitReader& br, const HuffTable& act, int16_t* blk, uint32_t& eobrun, int ss, int se, int al, Flags& fl) {
    if (eobrun > 0) { --eobrun; return; }
    for (int k = ss; k <= se; ++k) {
        ensure32(br);
        int rs = huff_decode_nocheck(br, act, fl.bad);
        int r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            blk[kNaturalOrder[k]] = shl16(huff_extend(br.get_bits_nocheck(s), s), al);
        } else {
            if (r == 15) { k += 15; continue; }
            eobrun = 1u << r;
            if (r) eobrun += br.get_bits(r);
            --eobrun;
            break;
        }
    }
}

inline void ac_refine_block(BitReader& br, const HuffTable& act, int16_t* blk, uint32_t& eobrun, int ss, int se, int al, Flags& fl) {
    const int p1 = 1 << al, m1 = -(1 << al);
    int k = ss;
    if (eobrun == 0) {
        for (; k <= se; ++k) {
            ensure32(br);
            int rs = huff_decode_nocheck(br, act, fl.bad);
            int r = rs >> 4, s = rs & 15;
            if (s) {
                if (s != 1) fl.bad = true;
                s = br.get_bit() ? p1 : m1;
            } else if (r != 15) {
                eobrun = 1u << r;
                if (r) eobrun += br.get_bits(r);
                break;
            }
            do {
                int16_t* c = blk + kNaturalOrder[k];
                if (*c != 0) {
                    if (br.get_bit() && (*c & p1) == 0) *c = int16_t(*c >= 0 ? *c + p1 : *c + m1);
                } else if (--r < 0) {
                    break;
                }
                ++k;
            } while (k <= se);
            if (s) blk[kNaturalOrder[k]] = int16_t(s);
        }
    }
    if (eobrun > 0) {
        for (; k <= se; ++k) {
            int16_t* c = blk + kNaturalOrder[k];
            if (*c != 0 && br.get_bit() && (*c & p1) == 0) *c = int16_t(*c >= 0 ? *c + p1 : *c + m1);
        }
        --eobrun;
    }
}

// Decode `nblocks` blocks starting at the cursor from a fresh (restart) state.
template <Mode M>
void decode_blocks(const Ctx& C, BitReader& br, BlockCursor& cur, uint32_t nblocks, Flags& fl) {
    int pred[kMaxComponents] = {0, 0, 0, 0};
    uint32_t eobrun = 0;
    const uint32_t end = std::min<uint32_t>(cur.ordinal + nblocks, C.L.total_blocks);
    alignas(64) int16_t scratch[kDctSize2] = {};
    while (cur.ordinal < end) {
        const int comp = cur.comp();
        if constexpr (M == Mode::Baseline) {
            if (C.fused) {
                baseline_block(br, *C.dc[comp], *C.ac[comp], scratch, pred[comp], fl);
                emit_block(C, cur, scratch);
            } else {
                baseline_block(br, *C.dc[comp], *C.ac[comp], cur.ptr(), pred[comp], fl);
            }
            cur.advance();
            continue;
        }
        int16_t* blk = cur.ptr();
        if constexpr (M == Mode::DcFirst) {
            pred[comp] += decode_dc_diff(br, *C.dc[comp], fl);
            blk[0] = shl16(pred[comp], C.al);
        } else if constexpr (M == Mode::DcRefine) {
            if (br.get_bit()) blk[0] = int16_t(blk[0] | (1 << C.al));
        } else if constexpr (M == Mode::AcFirst) {
            if (eobrun > 0) {
                // Skip the whole run in O(1) when it lies inside this range (no output, no bits).
                uint32_t n = std::min(eobrun, end - cur.ordinal);
                eobrun -= n;
                cur.seek(C.L, cur.ordinal + n);
                continue;
            }
            ac_first_block(br, *C.ac[comp], blk, eobrun, C.ss, C.se, C.al, fl);
        } else {
            ac_refine_block(br, *C.ac[comp], blk, eobrun, C.ss, C.se, C.al, fl);
        }
        cur.advance();
    }
}

// libjpeg jpeg_resync_to_restart. `pos` is the reader's byte position after byte alignment.
size_t resync_restart(const uint8_t* d, size_t pos, size_t end, int desired, bool& leave_unread, bool& warn) {
    leave_unread = false;
    for (;;) {
        while (pos < end && d[pos] != 0xFF) { ++pos; warn = true; }
        size_t p = pos;
        while (p < end && d[p] == 0xFF) ++p;
        if (p >= end) return end;
        int marker = d[p];
        if (marker == 0x00) { pos = p + 1; warn = true; continue; }
        int action;
        if (marker < 0xC0) action = 2;
        else if (marker < 0xD0 || marker > 0xD7) action = 3;
        else {
            int m = marker - 0xD0;
            if (m == ((desired + 1) & 7) || m == ((desired + 2) & 7)) action = 3;
            else if (m == ((desired - 1) & 7) || m == ((desired - 2) & 7)) action = 2;
            else action = 1;
        }
        if (action == 1) return p + 1;
        if (action == 3) { leave_unread = true; return pos; }
        pos = p + 1;
        warn = true;
    }
}

template <Mode M>
void decode_sequential(const Ctx& C, Flags& fl, uint32_t& restarts) {
    BitReader br(C.data, C.begin, C.end, C.begin);
    BlockCursor cur;
    cur.seek(C.L, 0);
    const uint32_t ri = C.frame->restart_interval;
    if (ri == 0) {
        decode_blocks<M>(C, br, cur, C.L.total_blocks, fl);
    } else {
        const uint32_t per = ri * uint32_t(C.L.blocks_per_mcu);
        int next_marker = 0;
        for (;;) {
            decode_blocks<M>(C, br, cur, per, fl);
            if (cur.done()) break;
            br.align_to_byte_boundary();
            bool leave = false, warn = false;
            size_t np = resync_restart(C.data, br.byte_pos(), C.end, next_marker, leave, warn);
            if (warn) fl.resync = true;
            br.set_pos(np);
            if (leave) { br.force_zero_fill(); fl.resync = true; }
            next_marker = (next_marker + 1) & 7;
            ++restarts;
        }
    }
    if (br.hit_marker() && !cur.done()) fl.premature = true;
}

void decode_sequential_any(const Ctx& C, Flags& fl, uint32_t& restarts) {
    switch (C.mode) {
        case Mode::Baseline: decode_sequential<Mode::Baseline>(C, fl, restarts); break;
        case Mode::DcFirst: decode_sequential<Mode::DcFirst>(C, fl, restarts); break;
        case Mode::DcRefine: decode_sequential<Mode::DcRefine>(C, fl, restarts); break;
        case Mode::AcFirst: decode_sequential<Mode::AcFirst>(C, fl, restarts); break;
        case Mode::AcRefine: decode_sequential<Mode::AcRefine>(C, fl, restarts); break;
    }
}

// ------------------------------------------------------------------------------------------------
// Restart-interval parallelism (all modes).
// ------------------------------------------------------------------------------------------------
struct Segment { size_t begin, end; };

bool find_restart_segments(const uint8_t* d, size_t begin, size_t end, uint32_t expected, std::vector<Segment>& segs) {
    segs.clear();
    segs.reserve(expected);
    size_t start = begin, p = begin;
    int expect_n = 0;
    while (p < end) {
        const void* hit = std::memchr(d + p, 0xFF, end - p);
        if (!hit) break;
        p = size_t(static_cast<const uint8_t*>(hit) - d);
        size_t q = p;
        while (q < end && d[q] == 0xFF) ++q;
        if (q >= end) break;
        uint8_t m = d[q];
        if (m == 0x00) { p = q + 1; continue; }
        if (m >= 0xD0 && m <= 0xD7) {
            if (int(m - 0xD0) != expect_n) return false;
            segs.push_back({start, p});
            expect_n = (expect_n + 1) & 7;
            start = q + 1;
            p = q + 1;
            continue;
        }
        break;
    }
    segs.push_back({start, end});
    return segs.size() == expected;
}

template <Mode M>
bool decode_restart_parallel(const Ctx& C, ThreadPool& pool, int threads, Flags& fl, uint32_t& nsegs) {
    const uint32_t ri = C.frame->restart_interval;
    const uint32_t per = ri * uint32_t(C.L.blocks_per_mcu);
    const uint32_t n = (C.L.total_mcus + ri - 1) / ri;
    std::vector<Segment> segs;
    if (!find_restart_segments(C.data, C.begin, C.end, n, segs)) return false;
    std::atomic<bool> bad{false};
    pool.parallel_for(n, [&](uint32_t i, int) {
        BitReader br(C.data, segs[i].begin, segs[i].end, segs[i].begin);
        BlockCursor cur;
        cur.seek(C.L, i * per);
        Flags f;
        decode_blocks<M>(C, br, cur, per, f);
        if (f.bad) bad = true;
    }, threads);
    if (bad) fl.bad = true;
    nsegs = n;
    return true;
}

bool decode_restart_parallel_any(const Ctx& C, ThreadPool& pool, int threads, Flags& fl, uint32_t& nsegs) {
    switch (C.mode) {
        case Mode::Baseline: return decode_restart_parallel<Mode::Baseline>(C, pool, threads, fl, nsegs);
        case Mode::DcFirst: return decode_restart_parallel<Mode::DcFirst>(C, pool, threads, fl, nsegs);
        case Mode::DcRefine: return decode_restart_parallel<Mode::DcRefine>(C, pool, threads, fl, nsegs);
        case Mode::AcFirst: return decode_restart_parallel<Mode::AcFirst>(C, pool, threads, fl, nsegs);
        case Mode::AcRefine: return decode_restart_parallel<Mode::AcRefine>(C, pool, threads, fl, nsegs);
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
// Speculative self-synchronising decoding (Baseline, DcFirst, AcFirst).
//
// A decoder started at an arbitrary byte quickly falls into step with the true symbol stream
// (Huffman codes self-synchronise; Klein & Wiseman 2003). The scan is cut into subsequences of S
// bytes, grouped contiguously into G groups:
//   pass 1 (parallel)   each group decodes from an empty state, recording its state at every
//                       subsequence boundary;
//   pass 2 (sequential) starting from the exact state at the end of the previous group, re-decode
//                       until a recorded boundary state is reproduced — from there on the recorded
//                       states are exact and only need their absolute block counts;
//   pass 3 (parallel)   every subsequence is decoded from its exact start state, writing output.
// Values that depend on earlier blocks (DC prediction) are stored as differences and resolved
// afterwards with a prefix sum. AC refinement scans cannot use this: how many bits a block consumes
// depends on that block's earlier coefficients, i.e. on its absolute index, which a speculative
// decoder does not know.
// ------------------------------------------------------------------------------------------------

// Decode until the canonical bit position reaches `until_bit` (and, with kFinishBlock, the current
// block is complete) or the scan ends. DC predictors are carried in st.dc, so values written in the
// final pass are absolute.
template <Mode M, bool kWrite, bool kFinishBlock>
void decode_until(const Ctx& C, BitReader& br, BlockCursor& cur, DecoderState& st, size_t until_bit, Flags& fl,
                  int16_t* scratch) {
    int16_t* blk = nullptr;
    int comp = 0;
    const bool fused = M == Mode::Baseline && C.fused != nullptr;
    auto bind = [&]() {
        comp = cur.comp();
        if (kWrite) blk = fused ? scratch : cur.ptr();
    };
    if (st.k != 0) {
        if (cur.done()) return;
        bind();
    }
    for (;;) {
        if ((!kFinishBlock || st.k == 0) && br.approx_bitpos() + 64 >= until_bit && br.canonical_bitpos() >= until_bit) break;
        if (st.k == 0) {
            if (cur.done()) break;
            if constexpr (M == Mode::AcFirst) {
                if (st.eobrun > 0) {
                    --st.eobrun;
                    st.blocks++;
                    cur.advance();
                    continue;
                }
            }
            bind();
            st.blocks++;
            if constexpr (M == Mode::DcFirst) {
                st.dc[comp] += decode_dc_diff(br, *C.dc[comp], fl);
                if (kWrite) blk[0] = shl16(st.dc[comp], C.al);
                cur.advance();
                continue;
            } else if constexpr (M == Mode::Baseline) {
                st.dc[comp] += decode_dc_diff(br, *C.dc[comp], fl);
                if (kWrite) blk[0] = int16_t(st.dc[comp]);
                st.k = 1;
                continue;
            } else {
                st.k = uint8_t(C.ss);
            }
        }
        ensure32(br);
        int rs = huff_decode_nocheck(br, *C.ac[comp], fl.bad);
        int r = rs >> 4, s = rs & 15;
        int k = st.k;
        const int last = M == Mode::Baseline ? 63 : C.se;
        if (s) {
            k += r;
            int v = huff_extend(br.get_bits_nocheck(s), s);
            if (kWrite) blk[kNaturalOrder[k]] = (M == Mode::AcFirst) ? shl16(v, C.al) : int16_t(v);
            ++k;
        } else if (r == 15) {
            k += 16;
        } else {
            if constexpr (M == Mode::AcFirst) {
                uint32_t run = 1u << r;
                if (r) run += br.get_bits(r);
                st.eobrun = run - 1;
            }
            k = last + 1;
        }
        if (k > last) {
            st.k = 0;
            if (kWrite && fused) emit_block(C, cur, scratch);
            cur.advance();
        } else {
            st.k = uint8_t(k);
        }
    }
}

// Speculative self-synchronising decoding.
//   K  checkpoint spacing in bytes (fine: bounds the sequential correction work),
//   G  speculative groups (pass 1 tasks; each starts decoding at an arbitrary byte),
//   U  pass-3 work units (contiguous runs of checkpoints, aligned to block boundaries).
template <Mode M>
void decode_speculative(const Ctx& C, ThreadPool& pool, int threads, uint32_t K, Flags& fl, ScanTrace& tr) {
    const size_t begin = C.begin, end = C.end;
    const size_t nbytes = end - begin;
    const ScanLayout& L = C.L;
    if (K < 64) K = 64;
    const uint32_t nck = uint32_t((nbytes + K - 1) / K);  // checkpoints = subsequences
    const int G = std::max(1, std::min<int>(threads * 4, int(nck)));
    const uint32_t per_group = (nck + G - 1) / G;
    tr.subsequences = nck;

    // endst[i]: state at the end of subsequence i as seen by its group's speculative decoder.
    // startst[i]: exact state at the start of subsequence i (after pass 2).
    std::vector<DecoderState> endst(nck), startst(nck);
    std::atomic<bool> bad{false};
    auto ck_bit = [&](uint32_t i) -> size_t { return std::min(end, begin + size_t(i + 1) * K) * 8; };
    auto position = [&](BitReader& br, size_t bitpos) {
        br.set_pos(bitpos / 8);
        if (bitpos % 8) { br.refill(); br.consume(int(bitpos % 8)); }
    };
    auto t0 = Clock::now();

    // Pass 1 (parallel): discover. The MCU phase at a group start is unknown; guess it from the offset.
    pool.parallel_for(uint32_t(G), [&](uint32_t g, int) {
        uint32_t s0 = g * per_group, s1 = std::min(nck, s0 + per_group);
        if (s0 >= s1) return;
        BitReader br(C.data, begin, end, begin + size_t(s0) * K);
        uint32_t guess = g == 0 ? 0 : uint32_t((uint64_t(s0) * K * L.total_blocks) / std::max<size_t>(nbytes, 1));
        BlockCursor cur;
        cur.seek(L, std::min(guess, L.total_blocks ? L.total_blocks - 1 : 0));
        cur.ordinal = 0;  // only the slot phase matters when not writing; keep done() from firing early
        DecoderState st;
        Flags f;
        for (uint32_t i = s0; i < s1; ++i) {
            decode_until<M, false, false>(C, br, cur, st, ck_bit(i), f, nullptr);
            st.bitpos = br.canonical_bitpos();
            st.slot = uint8_t(cur.slot);
            endst[i] = st;
        }
        if (f.bad) bad = true;
    }, threads);
    auto t1 = Clock::now();

    // Pass 2 (sequential, short): from the exact state at a group start, re-decode until one of the
    // group's recorded checkpoint states is reproduced; later checkpoints are then exact up to their
    // block counts and DC predictors, which are rebased.
    startst[0] = DecoderState{};
    startst[0].bitpos = begin * 8;
    for (uint32_t i = 0; i + 1 < nck && i < per_group; ++i) startst[i + 1] = endst[i];
    uint32_t iterations = 0;
    for (int g = 1; g < G; ++g) {
        uint32_t s0 = g * per_group, s1 = std::min(nck, s0 + per_group);
        if (s0 >= s1) break;
        const DecoderState st0 = startst[s0];
        BitReader br(C.data, begin, end, begin);
        position(br, st0.bitpos);
        BlockCursor cur;
        cur.seek(L, st0.k ? st0.blocks - 1 : st0.blocks);
        DecoderState st = st0;  // absolute
        Flags f;
        for (uint32_t i = s0; i < s1; ++i) {
            ++iterations;
            decode_until<M, false, false>(C, br, cur, st, ck_bit(i), f, nullptr);
            st.bitpos = br.canonical_bitpos();
            st.slot = uint8_t(cur.slot);
            if (endst[i].same_sync_key(st)) {
                const DecoderState rec = endst[i];
                for (uint32_t j = i; j < s1 && j + 1 < nck; ++j) {
                    DecoderState a = endst[j];
                    a.blocks = st.blocks + (endst[j].blocks - rec.blocks);
                    for (int c = 0; c < kMaxComponents; ++c) a.dc[c] = st.dc[c] + (endst[j].dc[c] - rec.dc[c]);
                    startst[j + 1] = a;
                }
                break;
            }
            if (i + 1 < nck) startst[i + 1] = st;
        }
        if (f.bad) bad = true;
    }
    tr.sync_iterations = iterations;
    auto t2 = Clock::now();

    // Pass 3 (parallel): decode with output from exact states. Units are runs of ~16 KiB; a unit that
    // starts mid-block first skips to the next block boundary (the previous unit finishes that block).
    const uint32_t ck_per_unit = std::max<uint32_t>(1, (16384 + K - 1) / K);
    const uint32_t nunits = (nck + ck_per_unit - 1) / ck_per_unit;
    pool.parallel_for(nunits, [&](uint32_t u, int) {
        const uint32_t i0 = u * ck_per_unit, i1 = std::min(nck, i0 + ck_per_unit);
        DecoderState st = startst[i0];
        if (st.blocks >= L.total_blocks && st.k == 0) return;
        BitReader br(C.data, begin, end, begin);
        position(br, st.bitpos);
        BlockCursor cur;
        cur.seek(L, st.k ? st.blocks - 1 : st.blocks);
        Flags f;
        alignas(64) int16_t scratch[kDctSize2] = {};
        if (st.k != 0) decode_until<M, false, true>(C, br, cur, st, 0, f, nullptr);  // skip partial block
        const size_t until = (i1 == nck) ? ~size_t(0) : ck_bit(i1 - 1);
        decode_until<M, true, true>(C, br, cur, st, until, f, scratch);
        if (f.bad) bad = true;
    }, threads);
    auto t3 = Clock::now();
    if (bad) fl.bad = true;
    tr.pass_ms[0] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    tr.pass_ms[1] = std::chrono::duration<double, std::milli>(t2 - t1).count();
    tr.pass_ms[2] = std::chrono::duration<double, std::milli>(t3 - t2).count();
}

// ------------------------------------------------------------------------------------------------
// Scheduling.
// ------------------------------------------------------------------------------------------------
Mode mode_of(const Frame& f, const ScanHeader& sh) {
    if (!f.progressive) return Mode::Baseline;
    if (sh.ss == 0) return sh.ah == 0 ? Mode::DcFirst : Mode::DcRefine;
    return sh.ah == 0 ? Mode::AcFirst : Mode::AcRefine;
}

bool speculative_capable(Mode m) { return m == Mode::Baseline || m == Mode::DcFirst || m == Mode::AcFirst; }

// Do scans a (earlier) and b touch a common coefficient?
bool scans_conflict(const Frame& f, const ScanHeader& a, const ScanHeader& b) {
    int ass = f.progressive ? a.ss : 0, ase = f.progressive ? a.se : 63;
    int bss = f.progressive ? b.ss : 0, bse = f.progressive ? b.se : 63;
    if (ase < bss || bse < ass) return false;
    for (int i = 0; i < a.ncomp; ++i)
        for (int j = 0; j < b.ncomp; ++j)
            if (a.comp_index[i] == b.comp_index[j]) return true;
    return false;
}

struct ScanJob {
    Ctx ctx;
    ScanTrace trace;
    Flags flags;
    uint32_t restarts = 0;
    bool intra_parallel = false;
};

void run_job(ScanJob& J, ThreadPool* pool, int threads, uint32_t S, EntropyStats& stats) {
    auto t0 = Clock::now();
    const Ctx& C = J.ctx;
    bool done = false;
    if (J.intra_parallel && pool) {
        if (C.frame->restart_interval) {
            uint32_t nseg = 0;
            if (decode_restart_parallel_any(C, *pool, threads, J.flags, nseg)) {
                J.trace.strategy = ScanStrategy::RestartParallel;
                J.restarts = nseg ? nseg - 1 : 0;
                done = true;
            } else {
                stats.speculative_fallback = true;
            }
        } else {
            switch (C.mode) {
                case Mode::Baseline: decode_speculative<Mode::Baseline>(C, *pool, threads, S, J.flags, J.trace); break;
                case Mode::DcFirst: decode_speculative<Mode::DcFirst>(C, *pool, threads, S, J.flags, J.trace); break;
                case Mode::AcFirst: decode_speculative<Mode::AcFirst>(C, *pool, threads, S, J.flags, J.trace); break;
                default: break;
            }
            J.trace.strategy = ScanStrategy::Speculative;
            done = true;
        }
    }
    if (!done) {
        J.trace.strategy = ScanStrategy::Sequential;
        decode_sequential_any(C, J.flags, J.restarts);
    }
    J.trace.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

Status decode_scans(const uint8_t* data, size_t size, Frame& frame, const std::vector<TableEvent>& events,
                    ThreadPool* pool, int threads, uint32_t subsequence_bytes, EntropyStats& stats, IdctFn fused_idct,
                    uint32_t min_parallel_scan_bytes) {
    const size_t min_bytes = min_parallel_scan_bytes ? min_parallel_scan_bytes : 32768;
    const uint32_t min_blocks = min_parallel_scan_bytes ? 1 : 256;
    const size_t nscans = frame.scans.size();
    const uint32_t S = subsequence_bytes ? subsequence_bytes : 1024;
    std::vector<ScanTables> tables(nscans);
    std::vector<ScanJob> jobs(nscans);

    // Replay table events in stream order, snapshot tables per scan, latch quant tables.
    size_t ev = 0;
    while (ev < events.size() && events[ev].before_scan == 0) ++ev;  // applied by the parser
    for (size_t si = 0; si < nscans; ++si) {
        while (ev < events.size() && events[ev].before_scan == int(si)) {
            Status s = apply_table_event(data, size, events[ev], frame);
            if (!s) return s;
            ++ev;
        }
        const ScanHeader& sh = frame.scans[si];
        ScanJob& J = jobs[si];
        Ctx& C = J.ctx;
        C.data = data;
        C.frame = &frame;
        C.sh = &sh;
        C.mode = mode_of(frame, sh);
        C.ss = sh.ss;
        C.se = sh.se;
        C.al = sh.al;
        C.begin = sh.data_offset;
        C.end = sh.data_offset + sh.data_size;
        C.fused = frame.progressive ? nullptr : fused_idct;
        for (int i = 0; i < sh.ncomp; ++i) {
            const int ci = sh.comp_index[i];
            const bool need_dc = C.mode == Mode::Baseline || C.mode == Mode::DcFirst;
            const bool need_ac = C.mode == Mode::Baseline || C.mode == Mode::AcFirst || C.mode == Mode::AcRefine;
            if (need_dc && !frame.dc_tables[sh.td[i]].defined)
                return Status::error(StatusCode::Corrupt, "scan references undefined DC Huffman table");
            if (need_ac && !frame.ac_tables[sh.ta[i]].defined)
                return Status::error(StatusCode::Corrupt, "scan references undefined AC Huffman table");
            if (need_dc) { tables[si].dc[sh.td[i]] = frame.dc_tables[sh.td[i]]; C.dc[ci] = &tables[si].dc[sh.td[i]]; }
            if (need_ac) { tables[si].ac[sh.ta[i]] = frame.ac_tables[sh.ta[i]]; C.ac[ci] = &tables[si].ac[sh.ta[i]]; }
            Component& comp = frame.comps[ci];
            if (!comp.qt_latched_set) {
                if (!frame.qt[comp.tq].defined) return Status::error(StatusCode::Corrupt, "component references undefined quant table");
                comp.qt_latched = frame.qt[comp.tq];
                comp.qt_latched_set = true;
            }
        }
        C.L.init(frame, sh);
        ScanTrace& T = J.trace;
        T.index = int(si);
        T.ncomp = sh.ncomp;
        T.first_comp = sh.comp_index[0];
        T.ss = C.ss; T.se = C.se; T.ah = sh.ah; T.al = sh.al;
        T.bytes = sh.data_size;
        T.blocks = C.L.total_blocks;
    }
    // Components never referenced by any scan still need a quant table for the IDCT (their
    // coefficients stay zero, like libjpeg).
    for (int i = 0; i < frame.ncomp; ++i) {
        Component& comp = frame.comps[i];
        if (!comp.qt_latched_set) {
            if (!frame.qt[comp.tq].defined) return Status::error(StatusCode::Corrupt, "component references undefined quant table");
            comp.qt_latched = frame.qt[comp.tq];
            comp.qt_latched_set = true;
        }
    }

    // Dependency waves.
    std::vector<int> wave(nscans, 0);
    int nwaves = 0;
    for (size_t j = 0; j < nscans; ++j) {
        for (size_t i = 0; i < j; ++i)
            if (scans_conflict(frame, frame.scans[i], frame.scans[j])) wave[j] = std::max(wave[j], wave[i] + 1);
        jobs[j].trace.wave = wave[j];
        nwaves = std::max(nwaves, wave[j] + 1);
    }
    stats.waves = uint32_t(nwaves);

    const bool par = pool && threads > 1;
    for (int w = 0; w < nwaves; ++w) {
        std::vector<ScanJob*> intra, seq;
        for (size_t j = 0; j < nscans; ++j) {
            if (wave[j] != w) continue;
            ScanJob& J = jobs[j];
            const Ctx& C = J.ctx;
            const bool big = C.L.total_blocks > min_blocks && C.end - C.begin >= min_bytes;
            J.intra_parallel = par && ((frame.restart_interval && C.L.total_blocks > min_blocks) || (speculative_capable(C.mode) && big));
            (J.intra_parallel ? intra : seq).push_back(&J);
        }
        // Sequential-only scans of this wave run on their own threads while the pool works through
        // the intra-parallel scans. Small ones (or a lone one with nothing to overlap) run inline.
        std::vector<std::thread> helpers;
        std::vector<ScanJob*> inline_jobs;
        for (size_t k = 0; k < seq.size(); ++k) {
            ScanJob* J = seq[k];
            const bool worth_thread = par && J->ctx.end - J->ctx.begin > std::min<size_t>(16384, min_bytes) && (seq.size() > 1 || !intra.empty());
            if (worth_thread && helpers.size() + 1 < seq.size() + intra.size())
                helpers.emplace_back([J, &stats, S] { prefer_performance_cores(); run_job(*J, nullptr, 1, S, stats); });
            else
                inline_jobs.push_back(J);
        }
        for (ScanJob* J : intra) run_job(*J, pool, threads, S, stats);
        for (ScanJob* J : inline_jobs) run_job(*J, nullptr, 1, S, stats);
        for (auto& t : helpers) t.join();
    }

    for (ScanJob& J : jobs) {
        stats.bad_code |= J.flags.bad;
        stats.premature_end |= J.flags.premature;
        stats.resync |= J.flags.resync;
        stats.restart_segments += J.restarts;
        stats.subsequences += J.trace.subsequences;
        stats.sync_iterations += J.trace.sync_iterations;
        stats.scans.push_back(J.trace);
    }
    return Status::success();
}

}  // namespace parajpeg::detail
