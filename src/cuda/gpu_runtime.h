// Thin runtime layer so the same kernel source runs on a CUDA device or, for testing on machines
// without one, in a sequential CPU "grid emulator". Kernels must not use shared memory or
// __syncthreads (each thread is run to completion in emulation); every kernel in this backend is
// written in that style: inter-thread communication happens only between kernel launches.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(PJ_CUDA_EMULATION)

#include "../common/hd.h"
#define PJ_GLOBAL
#define PJ_DEVICE
#define PJ_CONSTANT

namespace parajpeg::gpu {
struct Dim3 { unsigned x = 1, y = 1, z = 1; };
inline thread_local Dim3 g_threadIdx, g_blockIdx, g_blockDim, g_gridDim;
using Stream = void*;

inline bool runtime_available() { return true; }
inline const char* runtime_name() { return "cuda-emulated"; }
inline bool dmalloc(void** p, size_t n) { *p = std::malloc(n ? n : 1); return *p != nullptr; }
inline void dfree(void* p) { std::free(p); }
inline void h2d(void* d, const void* h, size_t n, Stream) { std::memcpy(d, h, n); }
inline void d2h(void* h, const void* d, size_t n, Stream) { std::memcpy(h, d, n); }
inline void d2h_2d(void* h, size_t hpitch, const void* d, size_t dpitch, size_t width, size_t rows, Stream) {
    for (size_t r = 0; r < rows; ++r) std::memcpy(static_cast<uint8_t*>(h) + r * hpitch, static_cast<const uint8_t*>(d) + r * dpitch, width);
}
inline void dmemset(void* d, int v, size_t n, Stream) { std::memset(d, v, n); }
inline void sync(Stream) {}
inline bool stream_create(Stream* s) { *s = nullptr; return true; }
inline void stream_destroy(Stream) {}
inline std::string last_error() { return {}; }

template <class K, class... A>
void launch(unsigned grid, unsigned block, Stream, K kernel, A... args) {
    g_gridDim = Dim3{grid, 1, 1};
    g_blockDim = Dim3{block, 1, 1};
    for (unsigned b = 0; b < grid; ++b)
        for (unsigned t = 0; t < block; ++t) {
            g_blockIdx = Dim3{b, 1, 1};
            g_threadIdx = Dim3{t, 1, 1};
            kernel(args...);
        }
}
}  // namespace parajpeg::gpu

#define PJ_TID (parajpeg::gpu::g_blockIdx.x * parajpeg::gpu::g_blockDim.x + parajpeg::gpu::g_threadIdx.x)
#define PJ_LAUNCH(kernel, grid, block, stream, ...) parajpeg::gpu::launch(grid, block, stream, kernel, __VA_ARGS__)

#else  // real CUDA

#include <cuda_runtime.h>

#include "../common/hd.h"
#define PJ_GLOBAL __global__
#define PJ_DEVICE __device__
#define PJ_CONSTANT __constant__

namespace parajpeg::gpu {
using Stream = cudaStream_t;
inline bool runtime_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
inline const char* runtime_name() { return "cuda"; }
inline bool dmalloc(void** p, size_t n) { return cudaMalloc(p, n ? n : 1) == cudaSuccess; }
inline void dfree(void* p) { cudaFree(p); }
inline void h2d(void* d, const void* h, size_t n, Stream s) { cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, s); }
inline void d2h(void* h, const void* d, size_t n, Stream s) { cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, s); }
inline void d2h_2d(void* h, size_t hpitch, const void* d, size_t dpitch, size_t width, size_t rows, Stream s) {
    cudaMemcpy2DAsync(h, hpitch, d, dpitch, width, rows, cudaMemcpyDeviceToHost, s);
}
inline void dmemset(void* d, int v, size_t n, Stream s) { cudaMemsetAsync(d, v, n, s); }
inline void sync(Stream s) { cudaStreamSynchronize(s); }
inline bool stream_create(Stream* s) { return cudaStreamCreate(s) == cudaSuccess; }
inline void stream_destroy(Stream s) { cudaStreamDestroy(s); }
inline std::string last_error() {
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? std::string() : std::string(cudaGetErrorString(e));
}
}  // namespace parajpeg::gpu

#define PJ_TID (blockIdx.x * blockDim.x + threadIdx.x)
#define PJ_LAUNCH(kernel, grid, block, stream, ...) kernel<<<(grid), (block), 0, (stream)>>>(__VA_ARGS__)

#endif
