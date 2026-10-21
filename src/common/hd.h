// Host/device annotations. Code marked PJ_HD is compiled for the CPU, for CUDA devices, and for the
// CPU-side CUDA grid emulator used by the tests.
#pragma once
#if defined(__CUDACC__)
#define PJ_HD __host__ __device__
#define PJ_FORCEINLINE __forceinline__
#else
#define PJ_HD
#define PJ_FORCEINLINE inline __attribute__((always_inline))
#endif
