#include <cstdint>

// ============================================================
//  kernels_cpu.cpp  –  CPU math kernels
//
//  Plain typed C-style functions (no templates).
//  Declared in ops.hpp, called via dispatch_matmul.
// ============================================================

void matmul_cpu_f32(const float* A, const float* B, float* C,
                    int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M * N; i++) C[i] = 0.0f;
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < K; j++) {
            float av = A[i * K + j];
            for (int64_t k = 0; k < N; k++)
                C[i * N + k] += av * B[j * N + k];
        }
}

void matmul_cpu_f64(const double* A, const double* B, double* C,
                    int64_t M, int64_t K, int64_t N) {
    for (int64_t i = 0; i < M * N; i++) C[i] = 0.0;
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < K; j++) {
            double av = A[i * K + j];
            for (int64_t k = 0; k < N; k++)
                C[i * N + k] += av * B[j * N + k];
        }
}
