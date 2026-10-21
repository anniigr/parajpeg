#!/usr/bin/env bash
# The command sequence shown in docs/media/demo.gif. Run from the repository root after building
# (cmake -S . -B build && cmake --build build -j) and creating the benchmark corpus (bench/make_corpus.sh).
set -euo pipefail
IMG=bench/data/jpg/iss062e001866   # 5568x3712 NASA photograph, re-encoded several ways
ctest --test-dir build                                         # bit-exactness, SIMD, emulated GPU
./build/pjdec ${IMG}_q90_420.jpg /tmp/out.ppm                    # baseline, no restart markers
PJ_SCANS=1 ./build/pjdec ${IMG}_q85_prog.jpg                     # progressive: per-scan strategy
bench/check_bitexact.sh ${IMG}_*.jpg                             # byte-for-byte vs djpeg
./build/pjbench --runs 5 --threads 1,8 ${IMG}_q90_420.jpg ${IMG}_q85_prog.jpg
