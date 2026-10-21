# parajpeg — design notes

This document explains how parajpeg decodes a JPEG in parallel while producing output that is
byte-identical to libjpeg-turbo, and why each design decision was made. It assumes familiarity with
the JPEG decoding pipeline (entropy decoding, dequantisation + IDCT, upsampling, colour conversion).

## 1. The problem

Three of the four stages of a JPEG decoder are embarrassingly parallel: every 8×8 block can be
inverse-transformed independently, and every output pixel can be upsampled and colour-converted
independently. Entropy decoding is not. The scan is one Huffman-coded bitstream in which code
boundaries are unknown until everything before them has been decoded, DC coefficients are coded as
differences from the previous block of the same component, and in progressive JPEGs a coefficient
is assembled over several scans.

After the SIMD stages are fast, entropy decoding is what limits single-image latency (Amdahl's law).
JPEG offers one standard escape hatch, restart markers, but most images in the wild have none
(libjpeg's `cjpeg` default, most web images), and progressive JPEGs bring their own dependencies.

The decoder's three goals, in priority order:

1. **Exactness.** Output is byte-identical to libjpeg-turbo with `TJFLAG_ACCURATEDCT`, using the
   islow integer IDCT, fancy (triangle) upsampling and fixed-point YCbCr→RGB. This holds for every
   conforming stream, on every backend and with every thread count. Corrupt streams must never
   crash or hit undefined behaviour.
2. **Parallelism inside one image**, whatever its encoding: baseline or progressive, with or without
   restart markers.
3. **Per-core efficiency** close to libjpeg-turbo, so parallel speedups are not bought with a slow
   baseline.

## 2. Pipeline overview

```
 bytes ─► marker parser ─► scan DAG scheduler ─┬─► restart-parallel scans
                                               ├─► speculative scans (self-synchronising)
                                               ├─► bit-indexed DC refinement
                                               └─► pipelined sequential scans (AC refinement)
                 baseline: entropy + IDCT fused per block ─┐
                 progressive: coefficient buffer ─► IDCT ──┴─► upsample + colour (row bands, parallel)
```

## 3. Speculative self-synchronising entropy decoding

**The observation.** Huffman codes self-synchronise (Klein & Wiseman, 2003). A decoder started at an
arbitrary bit decodes garbage for a while and then, with high probability, falls onto a true code
boundary, after which it emits exactly the same symbols as a decoder that started at the beginning.
For JPEG the decoder state is more than a bit position. Two decoders are in step only when they also
agree on:

* `k`, the coefficient index inside the current block (DC tables vs AC tables, end of block);
* `slot`, the block's position inside an interleaved MCU (which component, hence which tables);
* `eobrun`, the remaining end-of-band run (progressive AC first scans).

`DecoderState{bitpos, k, slot, eobrun}` is the synchronisation key. The block count and DC
predictors are bookkeeping that can be rebased once the true starting point is known.

**Canonical bit positions.** Byte stuffing (`FF 00`) makes "bit position" ambiguous: two decoders
that started at different bytes can hold different refill histories at the same symbol boundary.
`BitReader::canonical_bitpos()` reconstructs the absolute position of the next unread bit in the raw
byte stream, independently of buffering, so state comparison is exact.

**CPU algorithm (three passes).** The scan is cut into checkpoints every K bytes (default 1 KiB),
grouped into G = 4 × threads groups.

1. *Discover (parallel).* Each group decodes from its first byte with an empty state, guessing the
   MCU phase from the byte offset, and records its state at every checkpoint. No output is written.
2. *Correct (sequential, short).* Group 0's states are exact. For every later group, re-decode from
   the exact state at its start until one of its recorded checkpoint states is reproduced. From that
   checkpoint on, its recorded states are exact; only their block counts and DC predictors need
   rebasing by a constant. The cost is G × (distance to synchronisation), typically one checkpoint per
   group, so sequential work stays at about G × K bytes. A first version compared states only at
   group boundaries, which made this pass as long as the whole scan.
3. *Decode (parallel).* Work units of about 16 KiB, aligned to block boundaries, decode from their
   exact start states and write absolute values. A unit that starts mid-block skips to the next
   block boundary, because the previous unit finishes that block.

Speculation does about twice the Huffman work (discover plus decode), so it pays only from three
threads up. Below that the scheduler decodes sequentially. This threshold was measured: with two
threads, speculative decoding was no faster than one.

**GPU algorithm.** On a GPU the right trade-off is the opposite one: thousands of threads, each owning
a tiny subsequence (256 bytes by default). The CUDA backend follows Weißenberger & Schmidt (HiPC
2021), extended with the MCU-phase component of the state:

* `k_spec_init`: thread i decodes subsequence i from an arbitrary start.
* `k_spec_iter`: Jacobi iterations. Thread i re-decodes subsequence i from the end state recorded
  for i−1, until no state changes. By induction, after round t the first t states are exact. In
  practice the loop converges in one or two rounds. The converged round also yields per-subsequence
  block counts and DC sums.
* `k_seg_*`: exclusive prefix sums turn those into absolute block offsets and DC predictors.
* `k_spec_write`: the final decode writes coefficients.

## 4. Progressive JPEG

A progressive image is a sequence of scans: DC first, AC first (spectral selection), then
successive-approximation refinements of DC and AC. parajpeg treats the scans as a dependency DAG and
gives each scan type its own parallel strategy.

| scan type | strategy | why |
|---|---|---|
| DC first | speculative | consumption depends only on the MCU phase |
| AC first | speculative | state is `(bitpos, k, eobrun)`; independent of absolute position |
| DC refinement | bit-indexed | exactly one bit per block: after destuffing, block n's bit is bit n |
| AC refinement | sequential, **pipelined** | see below |
| any, with restart markers | restart-parallel | intervals are independent |

**Why AC refinement cannot be speculative.** How many correction bits a block consumes depends on
how many of its coefficients are already nonzero, which is a property of that particular block's
history. A decoder that starts at an unknown position does not know which block it is in, so it
cannot even decode; no amount of resynchronisation helps. This is the one place where JPEG's
format genuinely serialises decoding.

**What is done instead.**

* *Nonzero masks.* Every block carries a 64-bit mask of nonzero coefficients in zig-zag order,
  maintained by the AC first and refinement decoders. Refinement then finds "the (r+1)-th
  zero-history position" with popcount and clear-lowest-bit tricks, and reads correction bits in
  batches of up to 16 only for coefficients with history, instead of walking all 63 positions as
  libjpeg does. This alone makes single-threaded progressive decoding faster than libjpeg-turbo.
* *Scan DAG.* Two scans conflict if they share a component and their spectral bands overlap. Two
  AC scans of the same component also always conflict, because they share the masks. A failing
  bit-exactness test exposed the need for that second rule: two scans on disjoint bands of one
  component raced on the shared mask word.
  Non-conflicting scans run concurrently, for example chroma refinements alongside luma.
* *Pipelining.* Dependent sequential scans with the same block order run at the same time. Each
  publishes its progress in blocks with a release store. A successor may decode block b once every
  predecessor has published progress greater than b, read with an acquire load. Two luma
  refinement scans then overlap almost completely instead of running back to back. Deadlock freedom
  holds because dependencies always point to earlier scans and the main thread runs its own scans
  in index order.

## 5. Fused baseline reconstruction

For baseline images every block is complete after one scan, so each block is inverse-transformed into
the component plane as soon as its last coefficient is decoded, while it is still in L1. No
coefficient buffer exists at all: for a 20 MP 4:4:4 image that saves allocating and touching
about 120 MB. The speculative decoder supports this because pass 2 also yields exact DC predictors at
every checkpoint, so pass 3 writes final DC values directly.

## 6. Exactness

* **IDCT.** `src/common/idct_core.h` is a transcription of libjpeg's `jidctint.c` with 64-bit
  intermediates (libjpeg's `JLONG`), including the range-limit table's wrap-around semantics. The
  scalar backend and the CUDA kernel share this one implementation.
* **SIMD IDCTs (NEON, AVX2)** use 32-bit lanes. With every input of a 1-D pass bounded by
  |x| ≤ 2¹⁴, no intermediate can exceed about 1.8·10⁹ < 2³¹. Each pass checks the bound (dequantised
  input, then pass-1 output). The final stage checks the range-limit window [−512, 511], where the
  table equals `saturate(v + 128)`. Blocks that fail a check take the 64-bit path. The SIMD
  kernels are therefore exact for all inputs, adversarial ones included. A randomized test with
  200,000 blocks, including full-range int16 coefficients and 16-bit quantisers, checks this for
  NEON natively and for AVX2 under Rosetta 2.
* **Colour conversion** reproduces `jdcolor.c`'s fixed-point formulas. The NEON version splits
  `FIX(1.402) = 65536 + 26345` and similar constants so that 16×16→32-bit products with rounding
  narrowing shifts produce identical results.
* **Upsampling** reproduces `jdsample.c`'s fancy filters, including libjpeg-turbo's `h1v2` bias and
  the "width > 2" condition. The GPU computes the same formulas per output pixel.
* **Corrupt data** follows libjpeg: invalid codes consume 17 bits and yield 0, markers inside data
  feed zero bits, and restart markers resynchronise with `jpeg_resync_to_restart`'s action table.
  DC predictors use wrapping arithmetic; only their low 16 bits are ever stored, so this matches.

## 7. Testing

| layer | how |
|---|---|
| bit-exactness | every corpus image × backend × thread count × forced-parallel strategy, compared byte-for-byte to TurboJPEG (RGB, BGRA, gray) |
| strategy coverage | `min_parallel_scan_bytes = 1` forces every scan through its parallel strategy; the test asserts the strategy was actually used |
| GPU algorithm | the CUDA source compiled for a CPU grid emulator, run on the whole corpus with 32-byte subsequences |
| SIMD | randomized and adversarial equivalence tests vs scalar (NEON native, AVX2 via Rosetta, GCC on Linux x86-64) |
| robustness | truncation at 10–95 %, random byte corruption, libFuzzer with ASan + UBSan |
| concurrency | ThreadSanitizer over the full suite (which forces the pipelined path) |
| real data | NASA camera originals (Nikon, restart markers, EXIF) and a 30 MB, 7-scan progressive original |

## 8. What did not work, or is left

* Speculative decoding of AC refinement scans: impossible in general (section 4).
* Speculation with two threads: no gain (section 3); disabled.
* GPU progressive entropy decoding: the CPU engine decodes progressive scans for the CUDA
  backend, and reconstruction runs on the GPU. Porting the speculative first-scan decoders is
  straightforward; refinement scans have the same limitation on a GPU.
* Deferred placement: pass 1 of the CPU speculative decoder could emit sparse
  (relative block, position, value) records so that pass 3 becomes a cheap scatter instead of a second
  Huffman decode. That would raise the ceiling from about P/2 to about P on P cores, at the cost of
  roughly 3 bytes per nonzero coefficient.
