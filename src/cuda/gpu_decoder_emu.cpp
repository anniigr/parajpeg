// The CUDA backend compiled for the CPU grid emulator: same kernels, run thread by thread. Used by
// the tests to check the GPU algorithm bit-exactly on machines without an NVIDIA GPU.
#define PJ_CUDA_EMULATION 1
#include "gpu_decoder.cu"
