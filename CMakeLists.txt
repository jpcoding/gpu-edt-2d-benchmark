cmake_minimum_required(VERSION 3.20)
project(gpu_edt_2d_benchmark LANGUAGES CXX CUDA)

# Modern, target-based build for the NUS PBA+ reference and the benchmark.
# (The classic Makefile still works; this adds a portable cross-platform path.)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# GPU architecture. Default uses nvcc's own -arch=native, which reliably detects the
# present GPU. (CMake's CMAKE_CUDA_ARCHITECTURES=native can mis-detect and fall back to
# an old arch whose kernel image won't run here — project() already sets such a default,
# so we override it unconditionally.) To pin arch(es) instead: -DEDT_CUDA_ARCH="89;120".
set(EDT_CUDA_ARCH "native" CACHE STRING "CUDA arch: 'native' (nvcc autodetect) or a CMake list e.g. 89;120")
if(EDT_CUDA_ARCH STREQUAL "native")
  set(CMAKE_CUDA_ARCHITECTURES OFF)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-arch=native>)
else()
  set(CMAKE_CUDA_ARCHITECTURES ${EDT_CUDA_ARCH})
endif()

find_package(CUDAToolkit REQUIRED)   # provides CUDA::nppif / nppc / nppisu targets

# NUS PBA+ reference, built as its own library target (the "modernized" packaging:
# no deprecated headers, isolated translation unit, consumable by any target).
add_library(nus_pba2d STATIC third_party/nus/pba2DHost.cu)
target_include_directories(nus_pba2d PUBLIC third_party/nus)
target_compile_options(nus_pba2d PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)

# Common include dirs for the harness/tools.
add_library(edt_headers INTERFACE)
target_include_directories(edt_headers INTERFACE ours baselines third_party/nus)

add_executable(bench bench.cu)
target_link_libraries(bench PRIVATE nus_pba2d edt_headers
                       CUDA::nppif CUDA::nppc CUDA::nppisu)
target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)

add_executable(npp_prof prof/npp_prof.cu)
target_link_libraries(npp_prof PRIVATE CUDA::nppif CUDA::nppc CUDA::nppisu)
target_compile_options(npp_prof PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)

add_executable(tune prof/tune.cu)
target_link_libraries(tune PRIVATE nus_pba2d edt_headers)
target_compile_options(tune PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-w>)
