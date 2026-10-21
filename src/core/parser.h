#pragma once
#include <cstddef>
#include <cstdint>

#include "jpeg_types.h"

namespace parajpeg::detail {

// Parses all markers of a JPEG stream (SOI..EOI), filling `frame` including the list of scans with
// their entropy-coded data spans. Huffman/quant tables are recorded at the point of each scan via
// Frame tables being updated as DHT/DQT markers appear; because tables may legally be redefined
// between scans, each ScanHeader snapshots table *ids* while the decoder keeps a running table set
// (see ScanContext in decoder). To keep this simple and correct, the parser records the byte offset
// of every DHT/DQT/DRI marker so the decoder replays them in order (see `TableEvent`).
struct TableEvent {
    enum Kind : uint8_t { DQT, DHT, DRI } kind;
    size_t offset;   // offset of marker payload length field
    size_t length;   // payload length including the 2 length bytes
    int before_scan; // index of the scan this event precedes
};

struct ParseResult {
    Frame frame;
    std::vector<TableEvent> events;
};

Status parse_stream(const uint8_t* data, size_t size, ParseResult& out, bool header_only);

// Apply a DQT / DHT / DRI segment to the frame's running table state.
Status apply_table_event(const uint8_t* data, size_t size, const TableEvent& ev, Frame& frame);

// Build derived lookup structures for a Huffman table from bits/vals. Returns false if invalid.
bool build_huffman_table(HuffTable& t);

}  // namespace parajpeg::detail
