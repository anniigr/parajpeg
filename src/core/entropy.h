// Entropy decoding (Huffman) for baseline-sequential and progressive JPEG.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "bitreader.h"
#include "jpeg_types.h"

namespace parajpeg::detail {

class ThreadPool;

inline int huff_extend(uint32_t v, int s) {
    return int(v) < (1 << (s - 1)) ? int(v) - (1 << s) + 1 : int(v);
}

// Decode one Huffman symbol. Mirrors libjpeg's jpeg_huff_decode including its behaviour on invalid
// codes (consume 17 bits, return 0).
inline int huff_decode(BitReader& br, const HuffTable& t, bool& bad_code) {
    if (br.bits_available() < 16) br.refill();
    uint32_t p = br.peek(HuffTable::kLookupBits);
    uint16_t e = t.lookup[p];
    if (e) {
        br.consume(e >> 8);
        return e & 0xFF;
    }
    int l = HuffTable::kLookupBits + 1;
    int code = int(br.peek(l));
    while (code > t.maxcode[l]) {
        ++l;
        if (l > 16) {
            bad_code = true;
            br.consume(17 > br.bits_available() ? br.bits_available() : 17);
            return 0;
        }
        code = int(br.peek(l));
    }
    br.consume(l);
    return t.vals[(code + t.valoffset[l]) & 0xFF];
}

// Describes the sequence of blocks in a scan and maps the n-th block to its coefficient storage.
struct ScanLayout {
    const Frame* frame = nullptr;
    const ScanHeader* scan = nullptr;
    bool interleaved = false;
    int blocks_per_mcu = 0;                    // interleaved
    uint32_t total_mcus = 0;                   // interleaved: mcus_x*mcus_y; non-interleaved: blocks_w*blocks_h
    uint32_t total_blocks = 0;
    // per block-in-MCU slot
    struct Slot { int comp; int bx, by; };     // offset within the MCU in blocks
    Slot slots[kMaxBlocksPerMcu];
    // non-interleaved
    int comp = 0;
    uint32_t bw = 0, bh = 0;

    void init(const Frame& f, const ScanHeader& s);
    // Returns the coefficient block for MCU index m and slot i (interleaved) / block raster index.
    inline int16_t* block_ptr(uint32_t mcu, int slot) const {
        const Component& c = frame->comps[slots[slot].comp];
        uint32_t mx = mcu % frame->mcus_x, my = mcu / frame->mcus_x;
        uint32_t bx = mx * c.h + slots[slot].bx, by = my * c.v + slots[slot].by;
        return c.coefs + (size_t(by) * c.padded_blocks_w + bx) * kDctSize2;
    }
    inline int16_t* block_ptr_noninterleaved(uint32_t n) const {
        const Component& c = frame->comps[comp];
        uint32_t by = n / bw, bx = n % bw;
        return c.coefs + (size_t(by) * c.padded_blocks_w + bx) * kDctSize2;
    }
};

// Iterates blocks in decode order with division-free O(1) advance; used by all baseline decoders.
struct BlockCursor {
    const ScanLayout* L = nullptr;
    uint32_t mcu = 0;      // interleaved: MCU index; non-interleaved: block index
    uint32_t mx = 0, my = 0;  // MCU coordinates (interleaved) or block coordinates (non-interleaved)
    int slot = 0;
    uint32_t ordinal = 0;  // global block ordinal in decode order
    void seek(const ScanLayout& layout, uint32_t block_ordinal) {
        L = &layout;
        ordinal = block_ordinal;
        if (L->interleaved) {
            mcu = block_ordinal / uint32_t(L->blocks_per_mcu);
            slot = int(block_ordinal % uint32_t(L->blocks_per_mcu));
            mx = mcu % L->frame->mcus_x;
            my = mcu / L->frame->mcus_x;
        } else {
            mcu = block_ordinal;
            slot = 0;
            mx = L->bw ? mcu % L->bw : 0;
            my = L->bw ? mcu / L->bw : 0;
        }
    }
    inline bool done() const { return ordinal >= L->total_blocks; }
    inline int16_t* ptr() const {
        if (L->interleaved) {
            const ScanLayout::Slot& s = L->slots[slot];
            const Component& c = L->frame->comps[s.comp];
            uint32_t bx = mx * c.h + s.bx, by = my * c.v + s.by;
            return c.coefs + (size_t(by) * c.padded_blocks_w + bx) * kDctSize2;
        }
        const Component& c = L->frame->comps[L->comp];
        return c.coefs + (size_t(my) * c.padded_blocks_w + mx) * kDctSize2;
    }
    inline int comp() const { return L->interleaved ? L->slots[slot].comp : L->comp; }
    inline void advance() {
        ++ordinal;
        if (L->interleaved) {
            if (++slot == L->blocks_per_mcu) {
                slot = 0;
                ++mcu;
                if (++mx == L->frame->mcus_x) { mx = 0; ++my; }
            }
        } else {
            ++mcu;
            if (++mx == L->bw) { mx = 0; ++my; }
        }
    }
};

// State of a baseline decoder at a symbol boundary. Two decoders in the same state decode the
// remainder of the stream identically, so (bitpos, k, slot) is the synchronisation key.
struct DecoderState {
    size_t bitpos = 0;        // canonical bit position
    uint32_t blocks = 0;      // blocks started (DC symbol decoded) since the reference point
    uint8_t k = 0;            // next coefficient index in the current block (0 = DC)
    uint8_t slot = 0;         // block index within the MCU of the *next* block to start (k==0) or current block
    bool same_sync_key(const DecoderState& o) const { return bitpos == o.bitpos && k == o.k && slot == o.slot; }
};

struct EntropyStats {
    uint32_t restart_segments = 0;
    uint32_t subsequences = 0;
    uint32_t sync_iterations = 0;
    bool bad_code = false;
    bool premature_end = false;
    bool resync = false;
    bool speculative_fallback = false;
};

// Decode all scans of a frame into the components' coefficient buffers.
//   threads: 1 => sequential only. >1 => restart-interval parallelism / speculative decoding.
//   subsequence_bytes: size of speculative subsequences (0 => default 4096 on CPU).
Status decode_scans(const uint8_t* data, size_t size, Frame& frame,
                    const std::vector<struct TableEvent>& events, ThreadPool* pool, int threads,
                    uint32_t subsequence_bytes, EntropyStats& stats);

}  // namespace parajpeg::detail
