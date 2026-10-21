// Entropy decoding: sequential baseline, restart-interval-parallel baseline, speculative
// self-synchronising parallel baseline, and sequential progressive.
#include "entropy.h"

#include <algorithm>
#include <cstring>
#include <memory>

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

// ------------------------------------------------------------------------------------------------
// Baseline block decoding core shared by all strategies.
// ------------------------------------------------------------------------------------------------

struct Tables {
    const HuffTable* dc[kMaxComponents]{};
    const HuffTable* ac[kMaxComponents]{};
};

// Decode symbols from `br` until the canonical bit position reaches `until_bit` (or all blocks are
// consumed). State is carried in/out through `st`; coefficients are written through the cursor if
// `write` is true. DC is stored as a *difference* (resolved later) when kStoreDiff, else resolved
// with dcpred[].
template <bool kWrite, bool kStoreDiff>
static void decode_until(BitReader& br, const Tables& T, const ScanLayout& L, BlockCursor& cur,
                         DecoderState& st, size_t until_bit, int* dcpred, bool& bad_code) {
    int16_t scratch[kDctSize2];
    int16_t* blk = nullptr;
    int comp = 0;
    const HuffTable* dct = nullptr;
    const HuffTable* act = nullptr;
    auto bind_block = [&]() {
        if (cur.done()) return false;
        comp = cur.comp();
        dct = T.dc[comp];
        act = T.ac[comp];
        blk = kWrite ? cur.ptr() : scratch;
        return true;
    };
    if (st.k != 0) {
        // Resuming mid-block: the current block is the one most recently started.
        if (!bind_block()) return;
    }
    for (;;) {
        // Boundary check (cheap approximate test, exact only when close).
        if (br.approx_bitpos() + 64 >= until_bit) {
            if (br.canonical_bitpos() >= until_bit) break;
        }
        if (st.k == 0) {
            if (!bind_block()) break;
            // DC symbol
            int s = huff_decode(br, *dct, bad_code);
            int diff = 0;
            if (s) {
                if (s > 16) { bad_code = true; s = 16; }
                diff = huff_extend(br.get_bits(s), s);
            }
            if (kWrite) {
                if (kStoreDiff) blk[0] = int16_t(diff);
                else { dcpred[comp] += diff; blk[0] = int16_t(dcpred[comp]); }
            }
            st.blocks++;
            st.k = 1;
            continue;
        }
        // AC symbol
        int rs = huff_decode(br, *act, bad_code);
        int r = rs >> 4, s = rs & 15;
        int k = st.k;
        if (s) {
            k += r;
            int v = huff_extend(br.get_bits(s), s);
            if (kWrite) blk[kNaturalOrder[k]] = int16_t(v);
            ++k;
        } else if (r == 15) {
            k += 16;
        } else {
            k = 64;  // EOB
        }
        if (k >= 64) {
            st.k = 0;
            cur.advance();
        } else {
            st.k = uint8_t(k);
        }
    }
}

// Convenience: decode exactly `nblocks` blocks sequentially (used for restart intervals).
template <bool kStoreDiff>
static void decode_blocks(BitReader& br, const Tables& T, const ScanLayout& L, BlockCursor& cur,
                          uint32_t nblocks, int* dcpred, bool& bad_code) {
    int16_t* blk;
    uint32_t end = std::min<uint32_t>(cur.ordinal + nblocks, L.total_blocks);
    while (cur.ordinal < end) {
        int comp = cur.comp();
        const HuffTable& dct = *T.dc[comp];
        const HuffTable& act = *T.ac[comp];
        blk = cur.ptr();
        int s = huff_decode(br, dct, bad_code);
        int diff = 0;
        if (s) {
            if (s > 16) { bad_code = true; s = 16; }
            diff = huff_extend(br.get_bits(s), s);
        }
        if (kStoreDiff) blk[0] = int16_t(diff);
        else { dcpred[comp] += diff; blk[0] = int16_t(dcpred[comp]); }
        for (int k = 1; k < 64;) {
            int rs = huff_decode(br, act, bad_code);
            int r = rs >> 4, sz = rs & 15;
            if (sz) {
                k += r;
                blk[kNaturalOrder[k]] = int16_t(huff_extend(br.get_bits(sz), sz));
                ++k;
            } else if (r == 15) {
                k += 16;
            } else {
                break;
            }
        }
        cur.advance();
    }
}

// Resolve DC differences into absolute values in decode order, resetting at restart intervals.
static void resolve_dc(const ScanLayout& L, uint32_t restart_mcus) {
    int pred[kMaxComponents] = {0, 0, 0, 0};
    BlockCursor cur;
    cur.seek(L, 0);
    uint32_t mcu_in_interval = 0;
    uint32_t last_mcu = 0;
    while (!cur.done()) {
        if (restart_mcus && cur.mcu != last_mcu) {
            last_mcu = cur.mcu;
            if (++mcu_in_interval == restart_mcus) {
                mcu_in_interval = 0;
                pred[0] = pred[1] = pred[2] = pred[3] = 0;
            }
        }
        int16_t* b = cur.ptr();
        int c = cur.comp();
        pred[c] += b[0];
        b[0] = int16_t(pred[c]);
        cur.advance();
    }
}

// libjpeg jpeg_resync_to_restart semantics. `pos` is at a marker 0xFF xx (after skipping fill).
// Returns the byte position from which decoding should continue, and sets `leave_unread` when the
// marker must be left for the next interval (action 3).
static size_t resync_restart(const uint8_t* d, size_t pos, size_t end, int desired, bool& leave_unread,
                             bool& warn) {
    leave_unread = false;
    for (;;) {
        // find marker at/after pos
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
        pos = p + 1;  // action 2: skip and look again
        warn = true;
    }
}

struct RestartSegment {
    size_t begin, end;  // entropy data bytes
};

// Locate restart intervals by scanning for RSTn markers. Returns false if the marker count does not
// match the expected number (corrupt stream => caller falls back to the sequential resync decoder).
static bool find_restart_segments(const uint8_t* d, size_t begin, size_t end, uint32_t expected,
                                  std::vector<RestartSegment>& segs) {
    segs.clear();
    size_t start = begin;
    size_t p = begin;
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
        break;  // other marker: end of scan data
    }
    segs.push_back({start, end});
    return segs.size() == expected;
}

// ------------------------------------------------------------------------------------------------
// Baseline: sequential (with libjpeg restart resync) — the reference strategy.
// ------------------------------------------------------------------------------------------------
static void decode_baseline_sequential(const uint8_t* d, const ScanHeader& sh, const Frame& f,
                                       const Tables& T, const ScanLayout& L, EntropyStats& stats) {
    size_t begin = sh.data_offset, end = sh.data_offset + sh.data_size;
    BitReader br(d, begin, end, begin);
    BlockCursor cur;
    cur.seek(L, 0);
    int dcpred[kMaxComponents] = {0, 0, 0, 0};
    uint32_t ri = f.restart_interval;
    bool bad = false;
    if (ri == 0) {
        decode_blocks<false>(br, T, L, cur, L.total_blocks, dcpred, bad);
    } else {
        uint32_t blocks_per_interval = ri * uint32_t(L.blocks_per_mcu);
        int next_marker = 0;
        while (!cur.done()) {
            decode_blocks<false>(br, T, L, cur, blocks_per_interval, dcpred, bad);
            if (cur.done()) break;
            // Process restart: byte-align, find marker.
            br.align_to_byte_boundary();
            bool leave = false, warn = false;
            size_t np = resync_restart(d, br.byte_pos(), end, next_marker, leave, warn);
            if (warn) stats.resync = true;
            br.set_pos(np);
            if (leave) { br.force_zero_fill(); stats.resync = true; }
            next_marker = (next_marker + 1) & 7;
            dcpred[0] = dcpred[1] = dcpred[2] = dcpred[3] = 0;
            ++stats.restart_segments;
        }
    }
    if (br.hit_marker() && !cur.done()) stats.premature_end = true;
    if (bad) stats.bad_code = true;
}

// ------------------------------------------------------------------------------------------------
// Baseline: restart-interval parallel.
// ------------------------------------------------------------------------------------------------
static bool decode_baseline_restart_parallel(const uint8_t* d, const ScanHeader& sh, const Frame& f,
                                             const Tables& T, const ScanLayout& L, ThreadPool& pool,
                                             EntropyStats& stats) {
    size_t begin = sh.data_offset, end = sh.data_offset + sh.data_size;
    uint32_t ri = f.restart_interval;
    uint32_t blocks_per_interval = ri * uint32_t(L.blocks_per_mcu);
    uint32_t nseg = (L.total_mcus + ri - 1) / ri;
    std::vector<RestartSegment> segs;
    if (!find_restart_segments(d, begin, end, nseg, segs)) return false;
    std::atomic<bool> bad{false}, premature{false};
    pool.parallel_for(nseg, [&](uint32_t i, int) {
        BitReader br(d, segs[i].begin, segs[i].end, segs[i].begin);
        BlockCursor cur;
        cur.seek(L, i * blocks_per_interval);
        int dcpred[kMaxComponents] = {0, 0, 0, 0};
        bool b = false;
        decode_blocks<false>(br, T, L, cur, blocks_per_interval, dcpred, b);
        if (b) bad = true;
    });
    stats.restart_segments = nseg;
    if (bad) stats.bad_code = true;
    (void)premature;
    return true;
}

// ------------------------------------------------------------------------------------------------
// Baseline: speculative self-synchronising parallel decoding (no restart markers).
//
// The scan is split into subsequences of S bytes, grouped contiguously per worker. Pass 1: every
// worker decodes its group from an empty state, recording the decoder state at each subsequence
// boundary. Pass 2 (sequential over groups, cheap): starting from the corrected state at the end of
// the previous group, re-decode until the recorded state at a subsequence boundary matches; from
// there on the recorded states are correct and get their absolute block index. Pass 3: parallel
// final decode of every subsequence from its (now correct) start state, writing coefficients.
// DC differences are resolved afterwards with a prefix pass.
// ------------------------------------------------------------------------------------------------
static void decode_baseline_speculative(const uint8_t* d, const ScanHeader& sh, const Frame& f,
                                        const Tables& T, const ScanLayout& L, ThreadPool& pool,
                                        uint32_t S, EntropyStats& stats) {
    const size_t begin = sh.data_offset, end = sh.data_offset + sh.data_size;
    const size_t nbytes = end - begin;
    if (S < 64) S = 64;
    const uint32_t nsub = uint32_t((nbytes + S - 1) / S);
    const int G = std::max(1, std::min<int>(pool.size() * 4, int(nsub)));  // worker groups
    const uint32_t per_group = (nsub + G - 1) / G;
    stats.subsequences = nsub;

    // endst[i]: state recorded at the end of subsequence i in pass 1 (blocks relative to the group
    // start). startst[i]: corrected absolute start state of subsequence i (filled in pass 2).
    std::vector<DecoderState> endst(nsub);
    std::vector<DecoderState> startst(nsub);
    std::atomic<bool> bad{false};

    auto sub_end_bit = [&](uint32_t i) -> size_t { return std::min(end, begin + size_t(i + 1) * S) * 8; };
    auto position_reader = [&](BitReader& br, size_t bitpos) {
        br.set_pos(bitpos / 8);
        if (bitpos % 8) { br.refill(); br.consume(int(bitpos % 8)); }
    };

    // Pass 1: every group decodes from an empty state. The MCU phase at the group start is unknown;
    // we guess it proportionally to the byte offset (blocks are roughly uniform in the byte stream).
    pool.parallel_for(uint32_t(G), [&](uint32_t g, int) {
        uint32_t s0 = g * per_group, s1 = std::min(nsub, s0 + per_group);
        if (s0 >= s1) return;
        BitReader br(d, begin, end, begin + size_t(s0) * S);
        uint32_t guess = uint32_t((uint64_t(s0) * S * L.total_blocks) / std::max<size_t>(nbytes, 1));
        guess -= guess % uint32_t(L.blocks_per_mcu);  // keep slot phase 0 for group 0 exactness
        if (g == 0) guess = 0;
        BlockCursor cur;
        cur.seek(L, std::min(guess, L.total_blocks ? L.total_blocks - 1 : 0));
        // The cursor's ordinal is a guess; make sure done() cannot fire early: rebase ordinal to 0
        // while keeping the slot/coordinates (coordinates are irrelevant when not writing).
        cur.ordinal = 0;
        DecoderState st;
        bool b = false;
        for (uint32_t i = s0; i < s1; ++i) {
            decode_until<false, true>(br, T, L, cur, st, sub_end_bit(i), nullptr, b);
            st.bitpos = br.canonical_bitpos();
            st.slot = uint8_t(cur.slot);
            endst[i] = st;
        }
        if (b) bad = true;
    });

    // Pass 2: group 0's recorded states are exact (true start state). For every later group,
    // re-decode from the corrected state at the group start until a recorded boundary state matches.
    startst[0] = DecoderState{begin * 8, 0, 0, 0};
    for (uint32_t i = 0; i + 1 < nsub && i < per_group; ++i) startst[i + 1] = endst[i];
    uint32_t iterations = 0;
    for (int g = 1; g < G; ++g) {
        uint32_t s0 = g * per_group, s1 = std::min(nsub, s0 + per_group);
        if (s0 >= s1) break;
        DecoderState st = startst[s0];  // absolute
        const uint32_t abs_blocks = st.blocks;
        BitReader br(d, begin, end, begin);
        position_reader(br, st.bitpos);
        BlockCursor cur;
        cur.seek(L, st.k ? abs_blocks - 1 : abs_blocks);
        DecoderState rel;
        rel.k = st.k;
        bool b = false;
        for (uint32_t i = s0; i < s1; ++i) {
            ++iterations;
            decode_until<false, true>(br, T, L, cur, rel, sub_end_bit(i), nullptr, b);
            rel.bitpos = br.canonical_bitpos();
            rel.slot = uint8_t(cur.slot);
            DecoderState corrected{rel.bitpos, abs_blocks + rel.blocks, rel.k, rel.slot};
            if (endst[i].same_sync_key(rel)) {
                // Synchronised: every later recorded state in this group is valid; make it absolute.
                const uint32_t rec_at_sync = endst[i].blocks;
                for (uint32_t j = i; j < s1 && j + 1 < nsub; ++j) {
                    DecoderState a = endst[j];
                    a.blocks = corrected.blocks + (endst[j].blocks - rec_at_sync);
                    startst[j + 1] = a;
                }
                break;
            }
            if (i + 1 < nsub) startst[i + 1] = corrected;
        }
        if (b) bad = true;
    }
    stats.sync_iterations = iterations;

    // Pass 3: parallel final decode from exact start states, writing coefficients. The last
    // subsequence decodes to completion (zero-fill semantics for truncated streams, as libjpeg).
    pool.parallel_for(nsub, [&](uint32_t i, int) {
        DecoderState st = startst[i];
        if (st.blocks >= L.total_blocks && st.k == 0) return;
        BitReader br(d, begin, end, begin);
        position_reader(br, st.bitpos);
        BlockCursor cur;
        cur.seek(L, st.k ? st.blocks - 1 : st.blocks);
        DecoderState rel;
        rel.k = st.k;
        bool b = false;
        size_t until = (i + 1 == nsub) ? ~size_t(0) : sub_end_bit(i);
        decode_until<true, true>(br, T, L, cur, rel, until, nullptr, b);
        if (b) bad = true;
    });
    if (bad) stats.bad_code = true;
    resolve_dc(L, 0);
}

// ------------------------------------------------------------------------------------------------
// Progressive (sequential), mirroring libjpeg jdphuff.c.
// ------------------------------------------------------------------------------------------------
struct ProgState {
    uint32_t eobrun = 0;
    int dcpred[kMaxComponents] = {0, 0, 0, 0};
};

static void prog_dc_first(BitReader& br, const Tables& T, int comp, int16_t* blk, ProgState& ps, int al, bool& bad) {
    int s = huff_decode(br, *T.dc[comp], bad);
    int diff = 0;
    if (s) {
        if (s > 16) { bad = true; s = 16; }
        diff = huff_extend(br.get_bits(s), s);
    }
    ps.dcpred[comp] += diff;
    blk[0] = int16_t(ps.dcpred[comp] * (1 << al));
}

static void prog_dc_refine(BitReader& br, int16_t* blk, int al) {
    if (br.get_bit()) blk[0] |= int16_t(1 << al);
}

static void prog_ac_first(BitReader& br, const Tables& T, int comp, int16_t* blk, ProgState& ps, int ss, int se, int al, bool& bad) {
    if (ps.eobrun > 0) { --ps.eobrun; return; }
    const HuffTable& act = *T.ac[comp];
    for (int k = ss; k <= se; ++k) {
        int rs = huff_decode(br, act, bad);
        int r = rs >> 4, s = rs & 15;
        if (s) {
            k += r;
            int v = huff_extend(br.get_bits(s), s);
            blk[kNaturalOrder[k]] = int16_t(v * (1 << al));
        } else {
            if (r == 15) { k += 15; continue; }
            ps.eobrun = 1u << r;
            if (r) ps.eobrun += br.get_bits(r);
            --ps.eobrun;
            break;
        }
    }
}

static void prog_ac_refine(BitReader& br, const Tables& T, int comp, int16_t* blk, ProgState& ps, int ss, int se, int al, bool& bad) {
    const HuffTable& act = *T.ac[comp];
    int p1 = 1 << al, m1 = -(1 << al);
    int k = ss;
    if (ps.eobrun == 0) {
        for (; k <= se; ++k) {
            int rs = huff_decode(br, act, bad);
            int r = rs >> 4, s = rs & 15;
            if (s) {
                if (s != 1) bad = true;
                s = br.get_bit() ? p1 : m1;
            } else {
                if (r != 15) {
                    ps.eobrun = 1u << r;
                    if (r) ps.eobrun += br.get_bits(r);
                    break;
                }
            }
            do {
                int16_t* c = blk + kNaturalOrder[k];
                if (*c != 0) {
                    if (br.get_bit()) {
                        if ((*c & p1) == 0) *c = int16_t(*c >= 0 ? *c + p1 : *c + m1);
                    }
                } else {
                    if (--r < 0) break;
                }
                ++k;
            } while (k <= se);
            if (s) blk[kNaturalOrder[k]] = int16_t(s);
        }
    }
    if (ps.eobrun > 0) {
        for (; k <= se; ++k) {
            int16_t* c = blk + kNaturalOrder[k];
            if (*c != 0) {
                if (br.get_bit()) {
                    if ((*c & p1) == 0) *c = int16_t(*c >= 0 ? *c + p1 : *c + m1);
                }
            }
        }
        --ps.eobrun;
    }
}

static void decode_progressive_scan(const uint8_t* d, const ScanHeader& sh, const Frame& f, const Tables& T,
                                    const ScanLayout& L, EntropyStats& stats) {
    size_t begin = sh.data_offset, end = sh.data_offset + sh.data_size;
    BitReader br(d, begin, end, begin);
    BlockCursor cur;
    cur.seek(L, 0);
    ProgState ps;
    bool bad = false;
    uint32_t ri = f.restart_interval;
    uint32_t blocks_per_interval = ri * uint32_t(L.blocks_per_mcu);
    uint32_t in_interval = 0;
    int next_marker = 0;
    const bool dc = sh.ss == 0;
    const bool first = sh.ah == 0;
    while (!cur.done()) {
        if (ri && in_interval == blocks_per_interval) {
            br.align_to_byte_boundary();
            bool leave = false, warn = false;
            size_t np = resync_restart(d, br.byte_pos(), end, next_marker, leave, warn);
            if (warn) stats.resync = true;
            br.set_pos(np);
            if (leave) { br.force_zero_fill(); stats.resync = true; }
            next_marker = (next_marker + 1) & 7;
            ps.eobrun = 0;
            ps.dcpred[0] = ps.dcpred[1] = ps.dcpred[2] = ps.dcpred[3] = 0;
            in_interval = 0;
            ++stats.restart_segments;
        }
        int16_t* blk = cur.ptr();
        int comp = cur.comp();
        if (dc) {
            if (first) prog_dc_first(br, T, comp, blk, ps, sh.al, bad);
            else prog_dc_refine(br, blk, sh.al);
        } else {
            if (first) prog_ac_first(br, T, comp, blk, ps, sh.ss, sh.se, sh.al, bad);
            else prog_ac_refine(br, T, comp, blk, ps, sh.ss, sh.se, sh.al, bad);
        }
        cur.advance();
        ++in_interval;
    }
    if (bad) stats.bad_code = true;
    if (br.hit_marker() && br.bits_available() < 0) stats.premature_end = true;
}

}  // namespace

Status decode_scans(const uint8_t* data, size_t size, Frame& frame, const std::vector<TableEvent>& events,
                    ThreadPool* pool, int threads, uint32_t subsequence_bytes, EntropyStats& stats) {
    // Reset running tables to the state before scan 0 by replaying events in order as scans progress.
    // (Events before scan 0 were already applied by the parser; we re-apply everything for scans >= 1.)
    size_t ev = 0;
    while (ev < events.size() && events[ev].before_scan == 0) ++ev;
    for (size_t si = 0; si < frame.scans.size(); ++si) {
        while (ev < events.size() && events[ev].before_scan == int(si)) {
            Status s = apply_table_event(data, size, events[ev], frame);
            if (!s) return s;
            ++ev;
        }
        const ScanHeader& sh = frame.scans[si];
        Tables T;
        for (int i = 0; i < sh.ncomp; ++i) {
            int ci = sh.comp_index[i];
            const HuffTable& dct = frame.dc_tables[sh.td[i]];
            const HuffTable& act = frame.ac_tables[sh.ta[i]];
            bool need_dc = !frame.progressive || sh.ss == 0;
            bool need_ac = !frame.progressive || sh.ss != 0;
            if (frame.progressive && sh.ss == 0 && sh.ah != 0) need_dc = false;  // DC refine uses no table
            if (need_dc && !dct.defined) return Status::error(StatusCode::Corrupt, "scan references undefined DC Huffman table");
            if (need_ac && !act.defined) return Status::error(StatusCode::Corrupt, "scan references undefined AC Huffman table");
            T.dc[ci] = &dct;
            T.ac[ci] = &act;
        }
        ScanLayout L;
        L.init(frame, sh);
        if (frame.progressive) {
            decode_progressive_scan(data, sh, frame, T, L, stats);
            continue;
        }
        bool parallel = pool && threads > 1 && L.total_blocks > 256;
        if (parallel && frame.restart_interval) {
            if (decode_baseline_restart_parallel(data, sh, frame, T, L, *pool, stats)) continue;
            stats.speculative_fallback = true;  // marker mismatch -> sequential with resync
        } else if (parallel && sh.data_size >= 4 * std::max<uint32_t>(subsequence_bytes, 64)) {
            decode_baseline_speculative(data, sh, frame, T, L, *pool, subsequence_bytes ? subsequence_bytes : 4096, stats);
            continue;
        }
        decode_baseline_sequential(data, sh, frame, T, L, stats);
    }
    return Status::success();
}

}  // namespace parajpeg::detail
