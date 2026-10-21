# Correctness and validation

parajpeg's contract is that its output is **byte-identical to libjpeg-turbo's**. The reference
configuration is TurboJPEG's `tjDecompress2` with `TJFLAG_ACCURATEDCT`: the islow integer IDCT,
fancy (triangle) upsampling and fixed-point YCbCr→RGB. The contract covers every backend, every
thread count and every parallel strategy. This document lists how that is checked, and where the
checks stop.

How exactness is achieved (IDCT bounds, colour conversion, libjpeg's error recovery) is described in
[design.md §6](design.md#6-exactness).

## Verification matrix

| what | how | where | runs in CI |
|---|---|---|---|
| Bit-exactness, CPU | 70 corpus images × 7 configurations (below), byte-for-byte comparison with TurboJPEG | `tests/test_bitexact.cpp` | Linux x86-64 (GCC, AVX2), macOS arm64 (NEON), sanitizer jobs |
| Parallel strategies are really used | with `min_parallel_scan_bytes = 1`, every scan that can be parallel must report a parallel strategy | `tests/test_bitexact.cpp` | same |
| Bit-exactness, CUDA algorithm | the CUDA source compiled for the CPU grid emulator; every corpus image in RGB, BGRA, and with 32-byte subsequences | `tests/test_gpu_emulated.cpp` | same |
| Damaged data | 252 truncated streams compared byte-for-byte with libjpeg-turbo's recovery output, 5 decoder variants each | `tests/test_truncation.cpp` | Linux and macOS (`parajpeg_tests_cref`) |
| Robustness | random byte corruption and truncation must not crash; non-JPEG input is rejected | `tests/test_robustness.cpp` | all test jobs |
| Fuzzer regressions | inputs found by libFuzzer decode exactly like libjpeg-turbo, on CPU and emulated GPU | `tests/test_truncation.cpp`, `tests/regress/` | Linux and macOS |
| SIMD kernels | 200,000 random and adversarial blocks; colour conversion and upsampling for 15 widths | `tests/test_simd.cpp` | NEON (macOS), AVX2 (Linux) |
| Scalar fallback | the full suite with `PARAJPEG_FORCE_SIMD=scalar` | CI step | Linux |
| Memory and UB | AddressSanitizer + UndefinedBehaviorSanitizer (`-fno-sanitize-recover=undefined`) over the full suite | `sanitizers` job | yes |
| Data races | ThreadSanitizer over the full suite, which exercises the pipelined scan scheduler | `sanitizers` job | yes |
| Fuzzing | libFuzzer + ASan + UBSan on the decode entry points, 3 minutes per CI run | `fuzz/fuzz_decode.cpp` | yes |
| Header parsing | `read_header` reports geometry and features for every corpus file | `tests/test_parser.cpp` | all test jobs |

## Test corpus

`tests/corpus/` holds 71 small JPEGs encoded with libjpeg-turbo's `cjpeg` from nine source images
(gradients, noise, text, plasma fractals, and ImageMagick's built-in `rose`, `wizard` and `logo`),
sized from 61×47 to 1920×1080 so that edge
blocks, partial MCUs and odd dimensions occur:

| encoding | files |
|---|---:|
| baseline 4:2:0 | 9 |
| baseline 4:2:0, optimised Huffman tables | 8 |
| baseline 4:2:2 | 8 |
| baseline 4:4:4 | 9 |
| baseline 4:1:1, 4:4:0 | 2 |
| grayscale | 8 |
| restart markers every 64 MCUs (4:2:0, q50) | 9 |
| restart marker after every MCU (4:4:4, q95) | 8 |
| progressive | 9 |
| arithmetic coding (must be rejected as `Unsupported`) | 1 |

The bit-exactness test decodes each of the 70 Huffman-coded files in these configurations:

| backend | threads | subsequence | forced parallel | output |
|---|---:|---:|---|---|
| Scalar | 1 | default | no | RGB / Gray |
| Cpu (SIMD) | 1 | default | no | RGB / Gray |
| Cpu | 8 | default | no | RGB / Gray |
| Cpu | 8 | 64 B | yes | RGB / Gray |
| Cpu | 3 | 256 B | yes | RGB / Gray |
| Cpu | 2 | 97 B | yes | RGB / Gray |
| Cpu | 4 | default | no | BGRA (colour images) |

The 97-byte subsequence size is deliberately not a power of two, so that checkpoints fall at
arbitrary positions relative to bytes, MCUs and stuffed bytes.

Large real files are checked outside the unit tests by
[`bench/check_bitexact.sh`](../bench/check_bitexact.sh), which compares `pjdec` with `djpeg -dct int`
byte for byte at 1 and 8 threads. On the 48-file benchmark corpus (below), the latest run reported
`96 identical, 0 different`. Besides re-encodings, that corpus contains the files as NASA distributes them:
three camera originals with EXIF data and restart markers, and one 29 MB progressive file with 7
scans.

## Damaged and truncated data

Each eligible corpus file (Huffman-coded, at most 0.6 MP) is cut at 97%, 80%, 55% and 30% of its
length. libjpeg-turbo decodes the truncated stream. If it produces an image, parajpeg must produce
the same bytes with each of five variants: scalar; 8 threads; 8 threads with forced parallel
strategies and 64-byte or 300-byte subsequences; and the emulated CUDA pipeline.

Latest local run (Apple M3, libjpeg-turbo 3.2.0): **252 streams compared, 12 skipped** because
libjpeg-turbo itself produced no output.

Two details of this comparison are deliberate:

* **C reference.** On damaged data the coefficients can be large enough for the IDCT output to
  overflow. Then libjpeg-turbo's SIMD kernels saturate, while its C code (`jidctint.c`) wraps
  through the range-limit table. parajpeg follows the C code, so this test runs only under
  `JSIMD_FORCENONE=1`: ctest runs the suite a second time as `parajpeg_tests_cref`.
* **libjpeg-turbo 3.x.** The recovery of damaged *progressive* files (interblock smoothing) changed
  in libjpeg-turbo 3.0, and parajpeg reproduces the 3.x behaviour. When the test is built against an
  older TurboJPEG, as on Ubuntu 24.04's `libturbojpeg0-dev`, it skips progressive files.

## Where the guarantee stops

* **Reference configuration only.** The comparison is with the accurate integer IDCT and fancy
  upsampling. libjpeg-turbo's fast or float IDCT and its scaled decoding produce different output,
  and parajpeg implements neither.
* **SIMD on damaged data.** For valid streams, libjpeg-turbo's SIMD and C paths agree, and
  parajpeg matches both. For overflowing coefficients parajpeg matches the C path only (see
  above).
* **CMYK/YCCK** decoding is implemented (4-channel output only) but no CMYK or YCCK file is in the
  corpus, so it is not covered by the bit-exactness tests.
* **CUDA on hardware.** The emulator verifies the algorithm, not the execution on a device. See
  [gpu.md §4](gpu.md#4-the-cpu-grid-emulator).
* **No formal proof.** The claims are established by testing: a corpus, systematic configurations,
  damaged data and fuzzing. The SIMD overflow bounds are derived analytically in the source
  comments and checked by randomized tests, but they are not machine-checked.

## Running the checks

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure          # parajpeg_tests, parajpeg_tests_cref, parajpeg_simd_tests

# Sanitizers (as in CI)
cmake -S . -B build-asan -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=undefined" -DPARAJPEG_BUILD_BENCH=OFF
cmake --build build-asan -j && ./build-asan/parajpeg_tests

# Fuzzing (needs a clang with libFuzzer, e.g. Homebrew LLVM on macOS)
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DPARAJPEG_BUILD_FUZZERS=ON
cmake --build build-fuzz --target fuzz_decode && mkdir -p corpus && cp tests/corpus/*.jpg corpus/
./build-fuzz/fuzz_decode corpus -max_total_time=600 -max_len=65536
```

The local run on which the numbers in this document are based:

```
$ ctest --test-dir build
1/3 Test #1: parajpeg_tests ...................   Passed
2/3 Test #2: parajpeg_tests_cref ..............   Passed
3/3 Test #3: parajpeg_simd_tests ..............   Passed
100% tests passed, 0 tests failed out of 3

parajpeg_tests        9 test cases, 4548 assertions passed
parajpeg_tests_cref   9 test cases, 7140 assertions passed (adds the 252 truncated-stream comparisons)
parajpeg_simd_tests   2 test cases, 88 assertions passed (kernels: neon)
```
