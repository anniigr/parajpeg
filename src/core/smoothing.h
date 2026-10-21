// libjpeg-turbo interblock smoothing for progressive JPEGs whose low-frequency coefficients are not
// known to full precision (truncated files, or encoders that stop before the final refinement
// scans). Reproduces jdcoefct.c decompress_smooth_data() exactly, as an in-place coefficient
// transform applied before the IDCT, so every reconstruction backend (CPU, GPU) shares it.
#pragma once
#include <vector>

#include "entropy.h"
#include "jpeg_types.h"

namespace parajpeg::detail {

class ThreadPool;

// Returns true if smoothing was applied. `scans` must be the per-scan traces of decode_scans (used
// for libjpeg's "last good iMCU row" of the final scan).
bool apply_block_smoothing(Frame& f, const std::vector<ScanTrace>& scans, ThreadPool* pool, int threads);

}  // namespace parajpeg::detail
