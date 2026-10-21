cmake_minimum_required(VERSION 3.21)
project(parajpeg VERSION 0.1.0 LANGUAGES CXX)

option(PARAJPEG_WITH_CUDA "Build the CUDA backend" OFF)
option(PARAJPEG_BUILD_TESTS "Build tests" ON)
option(PARAJPEG_BUILD_BENCH "Build benchmarks" ON)
option(PARAJPEG_BUILD_TOOLS "Build CLI tools" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(parajpeg
  src/core/parser.cpp
  src/core/entropy.cpp
  src/core/idct.cpp
  src/core/color.cpp
  src/core/decoder.cpp
  src/cpu/simd_neon.cpp
  src/cpu/simd_avx2.cpp
  src/cpu/simd_dispatch.cpp)
target_include_directories(parajpeg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_options(parajpeg PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -O3>)
# AVX2 kernels use function-level target attributes and are selected at runtime by CPUID, so the
# library itself stays runnable on any x86-64 CPU.
find_package(Threads REQUIRED)
target_link_libraries(parajpeg PUBLIC Threads::Threads)

if(PARAJPEG_WITH_CUDA)
  enable_language(CUDA)
  target_sources(parajpeg PRIVATE src/cuda/gpu_decoder.cu)
  target_compile_options(parajpeg PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)
  target_compile_definitions(parajpeg PUBLIC PARAJPEG_WITH_CUDA=1)
  set_target_properties(parajpeg PROPERTIES CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES "75;80;86;89;90")
endif()

# libjpeg-turbo (TurboJPEG) is the test oracle and benchmark baseline. Optional.
find_path(TURBOJPEG_INCLUDE turbojpeg.h HINTS /opt/homebrew/opt/jpeg-turbo/include /usr/include /usr/local/include)
find_library(TURBOJPEG_LIB NAMES turbojpeg HINTS /opt/homebrew/opt/jpeg-turbo/lib /usr/lib /usr/local/lib)
if(TURBOJPEG_INCLUDE AND TURBOJPEG_LIB)
  set(HAVE_TURBOJPEG ON)
  message(STATUS "TurboJPEG found: ${TURBOJPEG_LIB}")
else()
  set(HAVE_TURBOJPEG OFF)
  message(WARNING "TurboJPEG not found: bit-exactness tests and baseline benchmarks disabled")
endif()

if(PARAJPEG_BUILD_TOOLS)
  add_executable(pjdec tools/pjdec.cpp)
  target_link_libraries(pjdec parajpeg)
endif()

if(PARAJPEG_BUILD_TESTS)
  enable_testing()
  # The CUDA backend's kernels, compiled for the CPU grid emulator, are tested bit-exactly too.
  add_executable(parajpeg_tests tests/test_main.cpp tests/test_parser.cpp tests/test_bitexact.cpp tests/test_robustness.cpp
                 tests/test_gpu_emulated.cpp src/cuda/gpu_decoder_emu.cpp)
  target_include_directories(parajpeg_tests PRIVATE third_party src)
  target_link_libraries(parajpeg_tests parajpeg)
  if(HAVE_TURBOJPEG)
    target_include_directories(parajpeg_tests PRIVATE ${TURBOJPEG_INCLUDE})
    target_link_libraries(parajpeg_tests ${TURBOJPEG_LIB})
    target_compile_definitions(parajpeg_tests PRIVATE HAVE_TURBOJPEG=1)
  endif()
  target_compile_definitions(parajpeg_tests PRIVATE PARAJPEG_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus")
  add_test(NAME parajpeg_tests COMMAND parajpeg_tests)
  # SIMD-vs-scalar equivalence on random data. No external dependencies, so it can also be built for
  # another architecture (e.g. x86-64 under Rosetta on Apple silicon to exercise the AVX2 kernels).
  add_executable(parajpeg_simd_tests tests/test_main.cpp tests/test_simd.cpp)
  target_include_directories(parajpeg_simd_tests PRIVATE third_party src)
  target_link_libraries(parajpeg_simd_tests parajpeg)
  add_test(NAME parajpeg_simd_tests COMMAND parajpeg_simd_tests)
endif()

if(PARAJPEG_BUILD_BENCH)
  add_executable(pjbench bench/pjbench.cpp)
  target_link_libraries(pjbench parajpeg)
  if(HAVE_TURBOJPEG)
    target_include_directories(pjbench PRIVATE ${TURBOJPEG_INCLUDE})
    target_link_libraries(pjbench ${TURBOJPEG_LIB})
    target_compile_definitions(pjbench PRIVATE HAVE_TURBOJPEG=1)
  endif()
endif()
