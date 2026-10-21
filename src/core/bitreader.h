// Bit reader for JPEG entropy-coded segments.
//
// Properties required by the parallel decoders:
//  * can start at ANY byte offset of a segment (speculative decoding),
//  * handles 0xFF00 byte stuffing and stops at markers, feeding zero bits afterwards (libjpeg semantics),
//  * exposes a *canonical* bit position (absolute bit index in the raw byte stream of the next unread
//    bit) that is independent of refill history, so two decoders that reached the same point in the
//    symbol stream report identical positions even if they started at different offsets.
#pragma once
#include <cstddef>
#include <cstdint>

#include "../common/hd.h"

namespace parajpeg::detail {

class BitReader {
public:
    PJ_HD BitReader() = default;
    // [begin, end) is the entropy-coded data; `start` is the first byte to read (begin <= start <= end).
    PJ_HD BitReader(const uint8_t* base, size_t begin, size_t end, size_t start) { reset(base, begin, end, start); }

    PJ_HD void reset(const uint8_t* base, size_t begin, size_t end, size_t start) {
        base_ = base;
        begin_ = begin;
        end_ = end;
        pos_ = start;
        // Never start on the stuffed 0x00 of an FF00 pair — step onto the next real byte.
        if (pos_ > begin_ && pos_ < end_ && base_[pos_] == 0x00 && base_[pos_ - 1] == 0xFF) ++pos_;
        buf_ = 0;
        bits_ = 0;
        virtual_loaded_ = 0;
        hit_marker_ = false;
    }

    // Ensure at least 25 bits are available (enough for a 16-bit code + up to 16 extra bits is NOT
    // guaranteed by one call; callers needing >25 bits call refill() again — see get_bits()).
    PJ_HD inline void refill() {
#if !defined(__CUDA_ARCH__)
        if (!hit_marker_ && pos_ + 8 <= end_) {
            uint64_t v;
            __builtin_memcpy(&v, base_ + pos_, 8);
            v = __builtin_bswap64(v);
            // Any 0xFF byte? (zero-byte detection on v ^ 0xFF..)
            uint64_t x = ~v;
            if (((x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull) == 0) {
                int nbytes = (64 - bits_) >> 3;  // >= 1 whenever bits_ <= 56
                int nbits = nbytes * 8;
                buf_ |= (v >> (64 - nbits)) << (64 - bits_ - nbits);
                bits_ += nbits;
                pos_ += size_t(nbytes);
                return;
            }
        }
#endif
        while (bits_ <= 56) load_byte();
    }
    // Callers that already ensured enough bits (see huff_decode) use these to skip the check.
    PJ_HD inline uint32_t get_bits_nocheck(int n) {
        uint32_t v = peek(n);
        consume(n);
        return v;
    }
    PJ_HD inline uint32_t peek(int n) const { return uint32_t(buf_ >> (64 - n)); }
    PJ_HD inline void consume(int n) {
        buf_ <<= n;
        bits_ -= n;
    }
    PJ_HD inline uint32_t get_bits(int n) {  // n in [0,16]
        if (n == 0) return 0;
        if (bits_ < n) refill();
        uint32_t v = peek(n);
        consume(n);
        return v;
    }
    PJ_HD inline int get_bit() {
        if (bits_ < 1) refill();
        int v = int(buf_ >> 63);
        consume(1);
        return v;
    }
    PJ_HD inline int bits_available() const { return bits_; }
    PJ_HD bool hit_marker() const { return hit_marker_; }
    // True once a zero bit fed after the end of the data (marker or end of input) has been consumed:
    // libjpeg's "insufficient data" condition (jdhuff.c), after which the remaining MCUs of the
    // restart interval are not decoded.
    PJ_HD bool insufficient() const {
        const size_t unconsumed = size_t(bits_) < virtual_loaded_ ? size_t(bits_) : virtual_loaded_;
        return virtual_loaded_ > unconsumed;
    }
    PJ_HD size_t byte_pos() const { return pos_; }
    PJ_HD size_t end() const { return end_; }

    // Cheap estimate of canonical_bitpos(), within 64 bits of it in either direction: buffered real
    // bits may straddle stuffed bytes (estimate >= exact), and after a marker the reader feeds
    // virtual zero bits without advancing pos_ (estimate <= exact). Callers use it as a pre-check:
    // canonical >= x implies approx_bitpos() + 64 >= x.
    PJ_HD inline size_t approx_bitpos() const { return pos_ * 8 + virtual_loaded_ - size_t(bits_); }

    // Exact absolute bit index of the next unread bit. Beyond-the-end virtual zero bits are counted
    // from end_*8 onward.
    PJ_HD size_t canonical_bitpos() const {
        const size_t unconsumed_virtual = size_t(bits_) < virtual_loaded_ ? size_t(bits_) : virtual_loaded_;
        int real = bits_ - int(unconsumed_virtual);
        if (real == 0 && virtual_loaded_ > 0) {
            // All real bits consumed: we are at the marker / end (pos_ stops advancing there).
            size_t consumed_virtual = virtual_loaded_ - unconsumed_virtual;
            return pos_ * 8 + consumed_virtual;
        }
        size_t idx = pos_;
        int r = real;
        while (r > 0) {
            --idx;
            if (base_[idx] == 0x00 && idx > begin_ && base_[idx - 1] == 0xFF) continue;  // stuffed byte: 0 bits
            r -= 8;
        }
        return idx * 8 + size_t(-r);
    }

    // Discard buffered bits (restart marker handling) and position the reader on the next byte.
    // After this call pos_ points at the first byte not yet consumed in full.
    PJ_HD void align_to_byte_boundary() {
        // Compute canonical position and rewind pos_ to the byte containing the next unread bit, then
        // drop any partially consumed byte (JPEG restart intervals end on a byte boundary).
        size_t bp = canonical_bitpos();
        size_t byte = bp / 8;
        if (bp % 8) ++byte;
        if (byte > end_) byte = end_;
        pos_ = byte;
        buf_ = 0;
        bits_ = 0;
        virtual_loaded_ = 0;
        hit_marker_ = false;
    }

    // Force the reader to feed zeros without advancing (used when a restart marker is deliberately
    // left unread, see jpeg_resync_to_restart action 3).
    PJ_HD void force_zero_fill() { hit_marker_ = true; }
    // Enter the "insufficient data" state directly (a restart that left a marker unread keeps
    // libjpeg's out-of-data flag set, so the next interval is not decoded at all).
    PJ_HD void force_insufficient() { hit_marker_ = true; buf_ = 0; bits_ = 0; virtual_loaded_ = 8; }
    PJ_HD void set_pos(size_t p) { pos_ = p; buf_ = 0; bits_ = 0; virtual_loaded_ = 0; hit_marker_ = false; }

private:
    PJ_HD inline void load_byte() {
        uint32_t b;
        if (hit_marker_ || pos_ >= end_) {
            b = 0;
            virtual_loaded_ += 8;
            hit_marker_ = true;
        } else {
            b = base_[pos_];
            if (b == 0xFF) {
                size_t nxt = pos_ + 1;
                if (nxt < end_ && base_[nxt] == 0x00) {
                    pos_ += 2;  // stuffed FF
                } else {
                    // Marker (or truncated FF at the end): stop here and feed zeros.
                    hit_marker_ = true;
                    b = 0;
                    virtual_loaded_ += 8;
                }
            } else {
                ++pos_;
            }
        }
        buf_ |= uint64_t(b) << (56 - bits_);
        bits_ += 8;
    }

    const uint8_t* base_ = nullptr;
    size_t begin_ = 0, end_ = 0, pos_ = 0;
    uint64_t buf_ = 0;
    int bits_ = 0;
    size_t virtual_loaded_ = 0;
    bool hit_marker_ = false;
};

}  // namespace parajpeg::detail
