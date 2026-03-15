#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include "include/tensor.hpp"
#include "include/ops.hpp"

int main() {
    int M = 2048, K = 2048, N = 2048;
    std::cout << "Allocating " << M << "x" << N << " matrices...\n";

    Tensor<float> A(std::vector<int64_t>{M, K});
    Tensor<float> B(std::vector<int64_t>{K, N});

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < (size_t)(M * K); i++) A.data->at(i) = dis(gen);
    for (size_t i = 0; i < (size_t)(K * N); i++) B.data->at(i) = dis(gen);

    // --- BENCHMARK 1: single-threaded ---
    std::cout << "Starting single-threaded benchmark...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    Tensor<float> C_single = matmul2d(A, B);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur_single = t1 - t0;
    std::cout << "Single-threaded: " << dur_single.count() << " s\n";

    // --- BENCHMARK 2: multi-threaded ---
    std::cout << "Starting multi-threaded benchmark...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    Tensor<float> C_multi = pgemm(A, B);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur_multi = t3 - t2;
    std::cout << "Multi-threaded:  " << dur_multi.count() << " s\n";

    // --- VERIFICATION ---
    std::cout << "Verifying correctness...\n";
    bool ok = true;
    for (size_t i = 0; i < (size_t)(M * N); i++) {
        float diff = std::abs(C_single.data->at(i) - C_multi.data->at(i));
        if (diff > 1e-4f) {
            std::cout << "MISMATCH at " << i << ": single=" << C_single.data->at(i)
                      << " multi=" << C_multi.data->at(i) << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "\nSUCCESS! Speedup: "
                  << (dur_single.count() / dur_multi.count()) << "x\n";
    } else {
        std::cout << "\nFAILURE: race condition or logic error.\n";
    }

    return 0;
}
