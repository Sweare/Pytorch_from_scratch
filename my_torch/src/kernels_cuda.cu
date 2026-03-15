// ============================================================
//  kernels_cuda.cu  –  CUDA math kernels  (TODO)
//
//  Compiled by nvcc only when USE_CUDA is set in CMake:
//    cmake -DUSE_CUDA=ON ..
//
//  Same signatures as kernels_cpu.cpp — ops.hpp routes here
//  automatically at compile time via dispatch_matmul.
// ============================================================

#include <cstdint>

void matmul_cuda_f32(const float*, const float*, float*,
                     int64_t, int64_t, int64_t) {
    // TODO
}

void matmul_cuda_f64(const double*, const double*, double*,
                     int64_t, int64_t, int64_t) {
    // TODO
}
