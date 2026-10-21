# CUDA backend

> **Status.** The CUDA backend compiles with CUDA 13 for sm_75, sm_80, sm_86, sm_89, sm_90, sm_100
> and sm_120. Its algorithm is verified bit-exact against libjpeg-turbo by running the same kernel
> source in a CPU grid emulator. It has **not been executed on NVIDIA hardware**, so there are no GPU
> performance numbers, and correctness on a real device has not been tested. This document
> describes what exists and what the first hardware session must check.

Source: [`src/cuda/gpu_decoder.cu`](../src/cuda/gpu_decoder.cu) (kernels and host pipeline) and
[`src/cuda/gpu_runtime.h`](../src/cuda/gpu_runtime.h) (CUDA runtime or emulator shim).

## 1. What runs on the GPU

| image | entropy decoding | IDCT | upsampling + colour |
|---|---|---|---|
| baseline, no restart markers | GPU: speculative decoding (`k_spec_*`, `k_seg_*`) | GPU | GPU |
| baseline, restart markers consistent with DRI | GPU: one thread per restart interval (`k_restart`) | GPU | GPU |
| baseline, restart markers inconsistent (damaged data) | CPU engine (needs libjpeg's sequential resynchronisation) | GPU | GPU |
| progressive | CPU engine (parallel scan DAG, see [design.md §4](design.md#4-progressive-jpeg)) | GPU | GPU |

For progressive images the coefficients are decoded on the CPU, interblock smoothing is applied on
the CPU, and the coefficient buffer is then copied to the device.

## 2. Kernels

All kernels are one-dimensional grids. None uses shared memory, `__syncthreads` or atomics. The
only cross-thread writes inside a launch are two global flags ("state changed" and "bad Huffman
code"), which threads may only set to 1. Threads otherwise communicate only between kernel
launches. This constraint is what allows the emulator to run the kernels thread by thread (§4).

| kernel | one thread per | work |
|---|---|---|
| `k_spec_init` | subsequence of K bytes (default 256) | decode subsequence i from an empty state, MCU phase guessed from the byte offset; record the end state |
| `k_spec_iter` | subsequence | re-decode subsequence i starting from the end state recorded for i−1; set a flag if the end state changed; also emit the subsequence's block count and DC-difference sums |
| `k_spec_chain` | (single thread) | fallback after 16 non-converging rounds: walk all subsequences in order from the exact start |
| `k_seg_sum`, `k_seg_scan`, `k_seg_apply` | segment of 256 subsequences / (single thread) / segment | exclusive prefix sums of block counts and the four DC predictors |
| `k_spec_write` | subsequence | final decode from the exact start state, writing coefficients with absolute DC values |
| `k_restart` | restart interval | independent decode of one interval |
| `k_idct` | 8×8 block | libjpeg islow IDCT (shared source with the CPU: [`src/common/idct_core.h`](../src/common/idct_core.h)) |
| `k_color` | output pixel | fancy upsampling and fixed-point YCbCr→RGB for one pixel, recomputed from the sample planes |

Threads per block: 128 for the entropy and IDCT kernels, 256 for `k_color`.

### Synchronisation by Jacobi iteration

The speculative kernels follow Weißenberger and Schmidt (§5): instead of the CPU decoder's
three-pass scheme with a short sequential correction pass, every subsequence repeatedly re-decodes
from its left neighbour's latest end state. The state compared is
`(bit position, coefficient index k, block-in-MCU slot)`; the MCU slot is this implementation's
extension for interleaved scans, where the slot selects the Huffman tables.

After round *t*, the end states of the first *t* subsequences are exact (by induction). How many
rounds are needed in practice depends on how far a speculative decoder runs before its state is
exact. The bit position aligns within a few dozen bytes, but the MCU phase of a 4:2:0 scan
often takes a few hundred bytes ([measurement](benchmarks.md#synchronisation-distance)). In the
emulator, on the benchmark images, this gives:

| subsequence size | 4:4:4 images | 4:2:0 images |
|---|---|---|
| 256 B (default) | 1–2 rounds | 3–15 rounds |
| 1 KiB | 1 round | 1–4 rounds |

One 4:2:0 image needed 15 rounds, one below the fallback limit described next. The default
subsequence size is therefore the first parameter to revisit on hardware. Each round is a full
pass over the scan, and more rounds trade directly against the parallelism that small
subsequences provide.

On data that never synchronises (adversarial or heavily corrupted), each round fixes only one more
subsequence, which would cost O(n²). After 16 rounds the host therefore switches to `k_spec_chain`,
which finishes the work linearly on one thread. The fuzzer found the input that motivated this
fallback; it is now the regression test `tests/regress/fuzz_nosync_640x480.jpg`.

Each round copies a 4-byte "changed" flag back to the host, so every round costs one
device-to-host synchronisation.

## 3. Compilation evidence

CI (`cuda` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) installs `nvcc` from the
`nvidia-cuda-nvcc` pip wheel (CUDA 13) and compiles `gpu_decoder.cu` once per architecture with
`-Xptxas -v`. It also compiles the host decoder with `PARAJPEG_WITH_CUDA=1` to check the wiring.
No GPU is attached to the runner, so nothing is executed.

ptxas resource usage for sm_90, taken from the CI log:

| kernel | registers | stack |
|---|---:|---:|
| `k_spec_init`, `k_spec_iter` | 32 | 144 B |
| `k_spec_write` | 40 | 144 B |
| `k_restart` | 38 | 144 B |
| `k_spec_chain` | 31 | 144 B |
| `k_seg_sum`, `k_seg_scan`, `k_seg_apply` | 32 | 0 |
| `k_idct` | 92 | 0 |
| `k_color` | 31 | 0 |

Two of these numbers are likely to matter on hardware. The 144-byte stack of the entropy kernels is
per-thread local memory; which variables ptxas placed there has not been examined yet. The 92 registers of
`k_idct` come from the 64-bit intermediates that reproduce libjpeg's `JLONG` arithmetic, and will
limit occupancy. The CPU SIMD kernels avoid that cost with proven 32-bit bounds and a guarded
fallback ([design.md §6](design.md#6-exactness)); the same technique applies here.

The CMake build (`-DPARAJPEG_WITH_CUDA=ON`) defaults to `CUDA_ARCHITECTURES 75;80;86;89;90`. For
Blackwell, pass `-DCMAKE_CUDA_ARCHITECTURES=100` or `120`.

## 4. The CPU grid emulator

When compiled with `PJ_CUDA_EMULATION` ([`src/cuda/gpu_decoder_emu.cpp`](../src/cuda/gpu_decoder_emu.cpp)),
`gpu_runtime.h` maps device allocations to `malloc`, copies to `memcpy`, and a launch to a loop over
every (block, thread) pair that runs the kernel body to completion. The host pipeline, the kernels
and the launch sequence are exactly the CUDA source.

The emulated pipeline is part of the test suite:

* every corpus image in RGB, in BGRA, and with 32-byte subsequences (which forces many more
  synchronisation rounds), compared byte for byte with libjpeg-turbo;
* the 252 truncated streams of the damaged-data test;
* the fuzzer regression input that does not synchronise.

**What the emulator proves:** the parallel algorithm (speculation, Jacobi convergence, prefix sums,
kernel decomposition, indexing, and the numerics of the IDCT and colour kernels) produces exactly
libjpeg-turbo's output.

**What it cannot show:**

* performance of any kind;
* behaviour that depends on concurrent execution, such as memory ordering between threads (by
  construction the kernels do not rely on it, but this has not been observed on hardware);
* device limits: launch configuration errors, out-of-memory on large images, and the integer width
  of grid indices;
* differences between nvcc and the host compiler in code generation.

## 5. Plan for the first hardware session

Target: one L4 (sm_89) or A100 (sm_80) instance, about 10 hours.

1. Build with `-DPARAJPEG_WITH_CUDA=ON` and run `pjdec --backend cuda` over the test corpus and the
   benchmark corpus, comparing output byte for byte with `djpeg` (libjpeg-turbo, `-dct int`). Add a
   `Backend::Cuda` variant to `tests/test_bitexact.cpp` and `tests/test_truncation.cpp` that is
   enabled when a device is present.
2. Run `compute-sanitizer --tool memcheck` and `--tool racecheck` on the corpus.
3. Measure end-to-end latency (including host↔device copies) and kernel time separately, with
   Nsight Systems. Compare against nvJPEG on the same images, for both its GPU-hybrid backend and,
   on A100, the hardware JPEG decoder. Compare the baseline entropy throughput with the figures
   reported by Weißenberger and Schmidt.
4. Profile the entropy and IDCT kernels with Nsight Compute. Tune K and the block size, and decide
   whether shared-memory Huffman tables and a 32-bit IDCT with fallback are worth their complexity.

Until step 1 has been done, every statement about the CUDA backend in this repository is about the
algorithm and the source code, not about execution on a GPU.

## 6. References

* A. Weißenberger, B. Schmidt. *Massively Parallel Huffman Decoding on GPUs.* ICPP 2018.
  [doi:10.1145/3225058.3225076](https://doi.org/10.1145/3225058.3225076)
* A. Weißenberger, B. Schmidt. *Accelerating JPEG Decompression on GPUs.* HiPC 2021.
  [arXiv:2111.09219](https://arxiv.org/abs/2111.09219). The authors report, on an A100, up to 51×
  over libjpeg-turbo and up to 8.0× over nvJPEG; these are their measurements, not parajpeg's.
