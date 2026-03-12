#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include "Tensor.hpp"
int main() {
    // 1. Setup massive dimensions (2048x2048 is a great stress test)
    int M = 2048;
    int K = 2048;
    int N = 2048;

    std::cout << "Allocating " << M << "x" << N << " matrices...\n";
    
    // Create tensors (Assuming your constructor allocates the shared_ptr<vector>)
    Tensor<float> A(std::vector<int64_t>{M, K});
    Tensor<float> B(std::vector<int64_t>{K, N});
    
    // Fill with random floats between 0.0 and 1.0
    std::mt19937 gen(42); // Fixed seed for reproducible tests
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for(size_t i = 0; i < M * K; i++) A.data->at(i) = dis(gen);
    for(size_t i = 0; i < K * N; i++) B.data->at(i) = dis(gen);

    std::cout << "Starting Single-Threaded Benchmark...\n";
    
    // --- BENCHMARK 1: SINGLE THREAD ---
    auto start_single = std::chrono::high_resolution_clock::now();
    
    Tensor<float> C_single = A.matMul2d(B);
    
    auto end_single = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_single = end_single - start_single;
    std::cout << "Single-Threaded Time: " << duration_single.count() << " seconds\n";


    std::cout << "Starting Multi-Threaded Benchmark...\n";

    // --- BENCHMARK 2: MULTI-THREAD ---
    auto start_multi = std::chrono::high_resolution_clock::now();
    
    Tensor<float> C_multi = A.PGEMM(B);
    
    auto end_multi = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_multi = end_multi - start_multi;
    std::cout << "Multi-Threaded Time:  " << duration_multi.count() << " seconds\n";

    
    // --- VERIFICATION ---
    // Floating point math can have tiny rounding differences, so we check if 
    // the difference is smaller than a tiny epsilon (1e-4).
    std::cout << "Verifying correctness...\n";
    bool passed = true;
    for(size_t i = 0; i < M * N; i++) {
        float diff = std::abs(C_single.data->at(i) - C_multi.data->at(i));
        if (diff > 1e-4) {
            std::cout << "MISMATCH at index " << i << "! Single: " 
                      << C_single.data->at(i) << ", Multi: " << C_multi.data->at(i) << "\n";
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "\nSUCCESS! The multi-threaded output exactly matches the single-threaded output.\n";
        std::cout << "Speedup Factor: " << (duration_single.count() / duration_multi.count()) << "x faster!\n";
    } else {
        std::cout << "\nFAILURE! Race condition or logic error detected.\n";
    }

    return 0;
}