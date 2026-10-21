#include "parser.h"

#include <algorithm>
#include <cstring>

namespace parajpeg::detail {

const uint8_t kNaturalOrder[kDctSize2 + 16] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63};

Subsampling Frame::subsampling() const {
    if (ncomp == 1) return Subsampling::Gray;
    if (ncomp != 3) return Subsampling::Other;
    const auto& y = comps[0];
    const auto& cb = comps[1];
    const auto& cr = comps[2];
    if (cb.h != cr.h || cb.v != cr.v || cb.h != 1 || cb.v != 1) return Subsampling::Other;
    if (y.h == 1 && y.v == 1) return Subsampling::S444;
    if (y.h == 2 && y.v == 1) return Subsampling::S422;
    if (y.h == 2 && y.v == 2) return Subsampling::S420;
    if (y.h == 1 && y.v == 2) return Subsampling::S440;
    if (y.h == 4 && y.v == 1) return Subsampling::S411;
    return Subsampling::Other;
}

namespace {

enum Marker : uint8_t {
    SOF0 = 0xC0, SOF1 = 0xC1, SOF2 = 0xC2, SOF3 = 0xC3, SOF5 = 0xC5, SOF6 = 0xC6, SOF7 = 0xC7,
    JPG = 0xC8, SOF9 = 0xC9, SOF10 = 0xCA, SOF11 = 0xCB, SOF13 = 0xCD, SOF14 = 0xCE, SOF15 = 0xCF,
    DHT = 0xC4, DAC = 0xCC, RST0 = 0xD0, RST7 = 0xD7, SOI = 0xD8, EOI = 0xD9, SOS = 0xDA,
    DQT = 0xDB, DNL = 0xDC, DRI = 0xDD, DHP = 0xDE, EXP = 0xDF, APP0 = 0xE0, APP14 = 0xEE,
    APP15 = 0xEF, COM = 0xFE, TEM = 0x01,
};

inline uint16_t be16(const uint8_t* p) { return uint16_t((p[0] << 8) | p[1]); }

Status err(StatusCode c, const char* m) { return Status::error(c, m); }

// Compute geometry once SOF is known.
void compute_geometry(Frame& f) {
    f.hmax = 1;
    f.vmax = 1;
    for (int i = 0; i < f.ncomp; ++i) {
        f.hmax = std::max<int>(f.hmax, f.comps[i].h);
        f.vmax = std::max<int>(f.vmax, f.comps[i].v);
    }
    f.mcu_w = 8u * f.hmax;
    f.mcu_h = 8u * f.vmax;
    f.mcus_x = (f.width + f.mcu_w - 1) / f.mcu_w;
    f.mcus_y = (f.height + f.mcu_h - 1) / f.mcu_h;
    for (int i = 0; i < f.ncomp; ++i) {
        Component& c = f.comps[i];
        // libjpeg: downsampled_width = ceil(image_width * h / hmax)
        c.width = uint32_t((uint64_t(f.width) * c.h + f.hmax - 1) / f.hmax);
        c.height = uint32_t((uint64_t(f.height) * c.v + f.vmax - 1) / f.vmax);
        c.blocks_w = (c.width + 7) / 8;
        c.blocks_h = (c.height + 7) / 8;
        c.padded_blocks_w = f.mcus_x * c.h;
        c.padded_blocks_h = f.mcus_y * c.v;
    }
}

void decide_color_space(Frame& f) {
    using CS = Frame::ColorSpace;
    // libjpeg jdapimin.c default_decompress_parms
    switch (f.ncomp) {
        case 1: f.jpeg_color_space = CS::Gray; break;
        case 3:
            if (f.jfif) f.jpeg_color_space = CS::YCbCr;
            else if (f.adobe) {
                f.jpeg_color_space = (f.adobe_transform == 0) ? CS::RGB : CS::YCbCr;
            } else {
                int c0 = f.comps[0].id, c1 = f.comps[1].id, c2 = f.comps[2].id;
                if (c0 == 1 && c1 == 2 && c2 == 3) f.jpeg_color_space = CS::YCbCr;
                else if (c0 == 82 && c1 == 71 && c2 == 66) f.jpeg_color_space = CS::RGB;  // 'R','G','B'
                else f.jpeg_color_space = CS::YCbCr;
            }
            break;
        case 4:
            if (f.adobe) f.jpeg_color_space = (f.adobe_transform == 2) ? CS::YCCK : CS::CMYK;
            else f.jpeg_color_space = CS::CMYK;
            break;
        default: f.jpeg_color_space = CS::Unknown;
    }
}

}  // namespace

bool build_huffman_table(HuffTable& t) {
    // Generate code lengths & codes (JPEG Annex C) then lookup tables (libjpeg jdhuff.c style).
    int code = 0, k = 0;
    int huffcode[257];
    for (int l = 1; l <= 16; ++l) {
        int n = t.bits[l];
        if (k + n > 256) return false;
        for (int i = 0; i < n; ++i) {
            huffcode[k] = code++;
            ++k;
        }
        // Code must fit in l bits.
        if (code > (1 << l)) return false;
        code <<= 1;
    }
    t.total_codes = k;
    // maxcode / valoffset
    int p = 0;
    for (int l = 1; l <= 16; ++l) {
        if (t.bits[l]) {
            t.valoffset[l] = p - huffcode[p];
            p += t.bits[l];
            t.maxcode[l] = huffcode[p - 1];
        } else {
            t.maxcode[l] = -1;
        }
    }
    t.valoffset[0] = 0;
    t.maxcode[17] = 0x7FFFFFFF;
    // Lookup table
    t.lookup.fill(0);
    p = 0;
    for (int l = 1; l <= HuffTable::kLookupBits; ++l) {
        for (int i = 0; i < t.bits[l]; ++i, ++p) {
            int shift = HuffTable::kLookupBits - l;
            int base = huffcode[p] << shift;
            for (int j = 0; j < (1 << shift); ++j) {
                t.lookup[base + j] = uint16_t((l << 8) | t.vals[p]);
            }
        }
    }
    // AC fast table (symbols with a nonzero magnitude category only; EOB/ZRL take the general path).
    t.fast_ac.fill(0);
    for (int peek = 0; peek < (1 << HuffTable::kLookupBits); ++peek) {
        uint16_t e = t.lookup[size_t(peek)];
        if (!e) continue;
        int len = e >> 8, sym = e & 0xFF;
        int run = sym >> 4, s = sym & 15;
        if (s == 0 || len + s > HuffTable::kLookupBits) continue;
        int bits = (peek >> (HuffTable::kLookupBits - len - s)) & ((1 << s) - 1);
        int v = bits < (1 << (s - 1)) ? bits - (1 << s) + 1 : bits;
        t.fast_ac[size_t(peek)] = int32_t(unsigned(v) << 9) | (run << 5) | (len + s);
    }
    t.defined = true;
    return true;
}

Status apply_table_event(const uint8_t* data, size_t size, const TableEvent& ev, Frame& f) {
    if (ev.offset + ev.length > size || ev.length < 2) return err(StatusCode::Corrupt, "table segment out of bounds");
    const uint8_t* p = data + ev.offset + 2;
    const uint8_t* end = data + ev.offset + ev.length;
    switch (ev.kind) {
        case TableEvent::DQT:
            while (p < end) {
                int pq = p[0] >> 4, tq = p[0] & 15;
                if (tq > 3) return err(StatusCode::Corrupt, "bad DQT table id");
                ++p;
                size_t need = pq ? 128 : 64;
                if (size_t(end - p) < need) return err(StatusCode::Corrupt, "DQT segment too short");
                QuantTable& q = f.qt[tq];
                for (int i = 0; i < 64; ++i) {
                    uint16_t v = pq ? be16(p + 2 * i) : p[i];
                    q.q[kNaturalOrder[i]] = v;
                }
                q.defined = true;
                p += need;
                if (pq && f.precision == 8) f.warnings.push_back("16-bit quant table in 8-bit image");
            }
            return Status::success();
        case TableEvent::DHT:
            while (p < end) {
                if (end - p < 17) return err(StatusCode::Corrupt, "DHT segment too short");
                int tc = p[0] >> 4, th = p[0] & 15;
                if (tc > 1 || th > 3) return err(StatusCode::Corrupt, "bad DHT table class/id");
                HuffTable& t = tc ? f.ac_tables[th] : f.dc_tables[th];
                int count = 0;
                t.bits[0] = 0;
                for (int i = 1; i <= 16; ++i) {
                    t.bits[i] = p[i];
                    count += p[i];
                }
                p += 17;
                if (count > 256 || size_t(end - p) < size_t(count)) return err(StatusCode::Corrupt, "bad DHT counts");
                std::memcpy(t.vals.data(), p, count);
                p += count;
                if (!build_huffman_table(t)) return err(StatusCode::Corrupt, "invalid Huffman table");
            }
            return Status::success();
        case TableEvent::DRI:
            if (end - p < 2) return err(StatusCode::Corrupt, "DRI too short");
            f.restart_interval = be16(p);
            return Status::success();
    }
    return err(StatusCode::Internal, "unknown table event");
}

Status parse_stream(const uint8_t* data, size_t size, ParseResult& out, bool header_only) {
    Frame& f = out.frame;
    f = Frame{};
    out.events.clear();
    if (size < 4 || data[0] != 0xFF || data[1] != SOI) return err(StatusCode::NotJpeg, "missing SOI marker");
    size_t pos = 2;
    bool seen_sof = false;
    uint16_t cur_ri = 0;  // restart interval in effect (DRI may appear anywhere before a scan)
    for (;;) {
        // Find next marker: skip fill bytes 0xFF.
        if (pos >= size) {
            if (!f.scans.empty() || header_only) {
                f.truncated = true;
                f.warnings.push_back("premature end of stream (missing EOI)");
                break;
            }
            return err(StatusCode::Truncated, "stream ended before SOS");
        }
        if (data[pos] != 0xFF) {
            // libjpeg: "Corrupt JPEG data: N extraneous bytes before marker" — skip to next 0xFF.
            size_t skip = pos;
            while (skip < size && data[skip] != 0xFF) ++skip;
            f.warnings.push_back("extraneous bytes before marker");
            pos = skip;
            continue;
        }
        while (pos < size && data[pos] == 0xFF) ++pos;
        if (pos >= size) { f.truncated = true; break; }
        uint8_t m = data[pos++];
        if (m == 0x00) continue;  // stuffed byte outside scan — ignore
        if (m == EOI) break;
        if ((m >= RST0 && m <= RST7) || m == TEM) continue;  // standalone markers
        if (pos + 2 > size) return err(StatusCode::Truncated, "marker length truncated");
        size_t len = be16(data + pos);
        if (len < 2 || pos + len > size) {
            // A segment cut off by the end of the file: libjpeg decodes the scans it already has.
            if ((header_only && seen_sof) || !f.scans.empty()) {
                f.truncated = true;
                f.warnings.push_back("premature end of stream inside a marker segment");
                break;
            }
            return err(StatusCode::Truncated, "marker segment exceeds stream");
        }
        const uint8_t* seg = data + pos + 2;
        size_t seglen = len - 2;
        switch (m) {
            case SOF0: case SOF1: case SOF2: {
                if (seen_sof) return err(StatusCode::Corrupt, "duplicate SOF");
                seen_sof = true;
                f.progressive = (m == SOF2);
                if (seglen < 6) return err(StatusCode::Corrupt, "SOF too short");
                f.precision = seg[0];
                f.height = be16(seg + 1);
                f.width = be16(seg + 3);
                f.ncomp = seg[5];
                if (f.precision != 8) return err(StatusCode::Unsupported, "only 8-bit precision is supported");
                if (f.width == 0) return err(StatusCode::Corrupt, "zero image width");
                if (f.height == 0) return err(StatusCode::Unsupported, "DNL-defined height is not supported");
                if (f.ncomp < 1 || f.ncomp > kMaxComponents) return err(StatusCode::Unsupported, "unsupported component count");
                if (seglen < size_t(6 + 3 * f.ncomp)) return err(StatusCode::Corrupt, "SOF component list truncated");
                for (int i = 0; i < f.ncomp; ++i) {
                    Component& c = f.comps[i];
                    c.id = seg[6 + 3 * i];
                    c.h = seg[7 + 3 * i] >> 4;
                    c.v = seg[7 + 3 * i] & 15;
                    c.tq = seg[8 + 3 * i];
                    if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3)
                        return err(StatusCode::Corrupt, "bad component sampling factors");
                }
                // Spec: blocks per MCU <= 10
                int blocks = 0;
                for (int i = 0; i < f.ncomp; ++i) blocks += f.comps[i].h * f.comps[i].v;
                if (f.ncomp > 1 && blocks > kMaxBlocksPerMcu) return err(StatusCode::Corrupt, "too many blocks per MCU");
                compute_geometry(f);
                f.valid = true;
                break;
            }
            case SOF3: case SOF5: case SOF6: case SOF7: case SOF11: case SOF13: case SOF14: case SOF15:
            case JPG: case DHP: case EXP:
                return err(StatusCode::Unsupported, "lossless/hierarchical/12-bit JPEG is not supported");
            case SOF9: case SOF10:
                f.arithmetic = true;
                // Still parse the header so read_header can report it.
                if (seglen >= 6) {
                    f.precision = seg[0]; f.height = be16(seg + 1); f.width = be16(seg + 3); f.ncomp = seg[5];
                    f.progressive = (m == SOF10);
                    if (f.ncomp >= 1 && f.ncomp <= kMaxComponents && seglen >= size_t(6 + 3 * f.ncomp)) {
                        for (int i = 0; i < f.ncomp; ++i) {
                            f.comps[i].id = seg[6 + 3 * i]; f.comps[i].h = seg[7 + 3 * i] >> 4;
                            f.comps[i].v = seg[7 + 3 * i] & 15; f.comps[i].tq = seg[8 + 3 * i];
                        }
                        compute_geometry(f);
                        f.valid = true;
                    }
                }
                if (header_only) { decide_color_space(f); return Status::success(); }
                return err(StatusCode::Unsupported, "arithmetic-coded JPEG is not supported");
            case DAC:
                return err(StatusCode::Unsupported, "arithmetic-coded JPEG is not supported");
            case DQT:
            case DHT:
            case DRI: {
                TableEvent ev{m == DQT ? TableEvent::DQT : m == DHT ? TableEvent::DHT : TableEvent::DRI,
                              pos, len, int(f.scans.size())};
                if (m == DRI && seglen >= 2) cur_ri = be16(seg);
                out.events.push_back(ev);
                // Apply immediately as well so that header-only parsing has tables for info purposes
                // and so the decoder's initial state equals the state before the first scan.
                if (int(f.scans.size()) == 0) {
                    Status s = apply_table_event(data, size, ev, f);
                    if (!s) return s;
                }
                break;
            }
            case APP0:
                if (seglen >= 5 && std::memcmp(seg, "JFIF\0", 5) == 0) f.jfif = true;
                break;
            case APP14:
                if (seglen >= 12 && std::memcmp(seg, "Adobe", 5) == 0) {
                    f.adobe = true;
                    f.adobe_transform = seg[11];
                }
                break;
            case SOS: {
                if (!seen_sof) return err(StatusCode::Corrupt, "SOS before SOF");
                if (header_only) { decide_color_space(f); return Status::success(); }
                if (seglen < 1) return err(StatusCode::Corrupt, "SOS too short");
                ScanHeader sh;
                sh.ncomp = seg[0];
                if (sh.ncomp < 1 || sh.ncomp > 4 || seglen < size_t(1 + 2 * sh.ncomp + 3))
                    return err(StatusCode::Corrupt, "bad SOS component count");
                for (int i = 0; i < sh.ncomp; ++i) {
                    uint8_t cid = seg[1 + 2 * i];
                    int ci = -1;
                    for (int j = 0; j < f.ncomp; ++j)
                        if (f.comps[j].id == cid) { ci = j; break; }
                    if (ci < 0) return err(StatusCode::Corrupt, "SOS references unknown component");
                    sh.comp_index[i] = ci;
                    sh.td[i] = seg[2 + 2 * i] >> 4;
                    sh.ta[i] = seg[2 + 2 * i] & 15;
                    if (sh.td[i] > 3 || sh.ta[i] > 3) return err(StatusCode::Corrupt, "bad Huffman table selector");
                }
                const uint8_t* q = seg + 1 + 2 * sh.ncomp;
                sh.ss = q[0];
                sh.se = q[1];
                sh.ah = q[2] >> 4;
                sh.al = q[2] & 15;
                if (f.progressive) {
                    if (sh.ss > 63 || sh.se > 63 || sh.ss > sh.se || sh.ah > 13 || sh.al > 13)
                        return err(StatusCode::Corrupt, "bad progressive scan parameters");
                    if (sh.ss == 0 && sh.se != 0) return err(StatusCode::Corrupt, "DC scan with AC range");
                    if (sh.ss != 0 && sh.ncomp != 1) return err(StatusCode::Corrupt, "interleaved AC scan");
                } else {
                    // Baseline: libjpeg tolerates nonsense Ss/Se/Ah/Al with a warning.
                    if (sh.ss != 0 || sh.se != 63 || sh.ah != 0 || sh.al != 0)
                        f.warnings.push_back("non-standard Ss/Se/Ah/Al in sequential scan (ignored)");
                }
                // Entropy-coded data begins after the segment; scan forward to the next marker that is
                // not RSTn or a stuffed 0xFF00.
                size_t start = pos + len;
                size_t p2 = start;
                for (;;) {
                    // Fast skip to next 0xFF
                    const void* hit = p2 < size ? std::memchr(data + p2, 0xFF, size - p2) : nullptr;
                    if (!hit) { p2 = size; break; }
                    p2 = size_t(static_cast<const uint8_t*>(hit) - data);
                    // Look past fill bytes
                    size_t p3 = p2;
                    while (p3 < size && data[p3] == 0xFF) ++p3;
                    if (p3 >= size) { p2 = size; break; }
                    uint8_t mm = data[p3];
                    // RSTn belongs to the scan only when a restart interval is in effect; otherwise
                    // libjpeg's entropy decoder stops at it like at any other marker, so the
                    // entropy-coded data ends there (and must not be seen by speculative decoders).
                    if (mm == 0x00 || (cur_ri && mm >= RST0 && mm <= RST7)) { p2 = p3 + 1; continue; }
                    break;  // real marker at p2
                }
                sh.data_offset = start;
                sh.data_size = p2 - start;
                f.scans.push_back(sh);
                pos = p2;
                continue;  // do not advance by len again
            }
            case DNL:
                f.warnings.push_back("DNL marker ignored");
                break;
            default:
                break;  // APPn, COM, etc.
        }
        pos += len;
    }
    if (!seen_sof) return err(StatusCode::NotJpeg, "no SOF marker found");
    if (f.scans.empty()) return err(StatusCode::Corrupt, "no scans found");
    decide_color_space(f);
    return Status::success();
}

}  // namespace parajpeg::detail
