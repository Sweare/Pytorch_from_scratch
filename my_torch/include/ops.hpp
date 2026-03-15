#pragma once

#include "tensor.hpp"
#include <thread>
#include <vector>
#include <limits>
#include <cstdint>

// ============================================================
//  ops.hpp  –  Operation dispatch layer
//
//  This file has TWO jobs:
//
//  1. DECLARE raw kernel signatures
//     The actual implementations live in:
//       src/kernels_cpu.cpp   (compiled always)
//       src/kernels_cuda.cu   (compiled when USE_CUDA is defined)
//
//  2. TEMPLATE WRAPPERS that users call (matmul, sum, mean, max…)
//     These extract raw pointers from Tensors, then call the
//     right kernel via dispatch_matmul.
//
//  Adding CUDA support = fill in kernels_cuda.cu + flip USE_CUDA in
//  CMake.  Nothing else in this file changes.
// ============================================================


// ============================================================
//  1.  Raw kernel declarations
//      (no templates — plain typed C-style functions)
// ============================================================

// --- CPU ---
void matmul_cpu_f32(const float*,  const float*,  float*,
                    int64_t M, int64_t K, int64_t N);
void matmul_cpu_f64(const double*, const double*, double*,
                    int64_t M, int64_t K, int64_t N);

// --- CUDA (only linked when USE_CUDA is set in CMake) ---
#ifdef USE_CUDA
void matmul_cuda_f32(const float*,  const float*,  float*,
                     int64_t M, int64_t K, int64_t N);
void matmul_cuda_f64(const double*, const double*, double*,
                     int64_t M, int64_t K, int64_t N);
#endif


// ============================================================
//  2.  Compile-time dispatch helpers
//      Pick CPU or CUDA backend via if constexpr + #ifdef
// ============================================================

template<typename T>
void dispatch_matmul(const T* A, const T* B, T* C,
                     int64_t M, int64_t K, int64_t N) {
#ifdef USE_CUDA
    if constexpr      (std::is_same_v<T, float>)  matmul_cuda_f32(A, B, C, M, K, N);
    else if constexpr (std::is_same_v<T, double>) matmul_cuda_f64(A, B, C, M, K, N);
    else static_assert(!sizeof(T), "matmul: unsupported type for CUDA backend");
#else
    if constexpr      (std::is_same_v<T, float>)  matmul_cpu_f32(A, B, C, M, K, N);
    else if constexpr (std::is_same_v<T, double>) matmul_cpu_f64(A, B, C, M, K, N);
    else static_assert(!sizeof(T), "matmul: unsupported type for CPU backend");
#endif
}


// ============================================================
//  3.  Template wrappers (public API)
// ============================================================

// ---- internal broadcast helpers ----------------------------

template<typename T>
static bool checkBroadcastable(const std::vector<int64_t>& A,
                                const std::vector<int64_t>& B) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size() - 2; i++)
        if (A[i] != B[i] && A[i] != 1 && B[i] != 1) return false;
    return true;
}

template<typename T>
static bool checkExactBatch(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.shape.size() != b.shape.size()) return false;
    for (size_t i = 0; i < a.shape.size() - 2; i++)
        if (a.shape[i] != b.shape[i]) return false;
    return true;
}

template<typename T>
static std::vector<int64_t> matmulResultShape(const Tensor<T>& a, const Tensor<T>& b) {
    std::vector<int64_t> s = a.shape;
    s.back() = b.shape.back();
    return s;
}

template<typename T>
static std::vector<int64_t> matmulResultShapePadded(const std::vector<int64_t>& A,
                                                     const std::vector<int64_t>& B) {
    std::vector<int64_t> s = A;
    for (size_t i = 0; i < A.size() - 2; i++)
        if (s[i] < B[i]) s[i] = B[i];
    s.back() = B.back();
    return s;
}

// ---- 2-D matmul core  (called by the N-D loop) -------------

template<typename T>
void matmul2d_direct(const Tensor<T>& a, const Tensor<T>& b, T* dest) {
    int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
    dispatch_matmul<T>(a.data->data() + a.offset,
                       b.data->data() + b.offset,
                       dest, M, K, N);
}

template<typename T>
Tensor<T> matmul2d(const Tensor<T>& a, const Tensor<T>& b) {
    Tensor<T> res(std::vector<int64_t>{a.shape[0], b.shape[1]});
    matmul2d_direct(a, b, res.data->data());
    return res;
}

// ---- N-D matmul  (broadcast aware) -------------------------

template<typename T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
    size_t da = a.shape.size(), db = b.shape.size();
    if (a.shape[da - 1] != b.shape[db - 2])
        throw std::invalid_argument("matmul: last dims must satisfy K×N * N×R");

    if (da == 2 && db == 2) return matmul2d(a, b);

    std::vector<int64_t> pa = a.shape, pb = b.shape;
    while (pa.size() < pb.size()) pa.insert(pa.begin(), 1);
    while (pb.size() < pa.size()) pb.insert(pb.begin(), 1);

    bool exact = checkExactBatch(a, b);
    bool bcast = !exact && checkBroadcastable<T>(pa, pb);
    if (!exact && !bcast)
        throw std::invalid_argument("matmul: batch dims must match or be broadcastable");

    std::vector<int64_t> res_shape = exact
        ? matmulResultShape(a, b)
        : matmulResultShapePadded<T>(pa, pb);

    Tensor<T> result(res_shape);
    T*      dst     = result.data->data() + result.offset;
    int64_t M       = res_shape[res_shape.size() - 2];
    int64_t N       = res_shape[res_shape.size() - 1];
    int64_t sz      = M * N;
    int64_t batches = result.countBatches();

    for (int64_t i = 0; i < batches; i++) {
        Tensor<T> mA = exact ? a.getMatrix(i) : a.getMatrix(i, res_shape, pa);
        Tensor<T> mB = exact ? b.getMatrix(i) : b.getMatrix(i, res_shape, pb);
        matmul2d_direct(mA, mB, dst + i * sz);
    }
    return result;
}

// operator* delegates to matmul
template<typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    return matmul(a, b);
}

// ---- Parallel GEMM (threading sits here, math goes to kernel) -----

template<typename T>
static void worker_gemm(Tensor<T> C, const Tensor<T> A, const Tensor<T> B) {
    int64_t M = A.shape[0], K = A.shape[1], N = B.shape[1];
    dispatch_matmul<T>(A.data->data() + A.offset,
                       B.data->data() + B.offset,
                       C.data->data() + C.offset,
                       M, K, N);
}

template<typename T>
Tensor<T> pgemm(const Tensor<T>& a, const Tensor<T>& b) {
    unsigned int nthreads = std::thread::hardware_concurrency();
    int64_t total_rows    = a.shape[0];
    int64_t chunk         = total_rows / nthreads;

    Tensor<T> C(std::vector<int64_t>{total_rows, b.shape[1]});
    std::vector<std::thread> threads;

    for (unsigned int i = 0; i < nthreads; i++) {
        int64_t start = i * chunk;
        int64_t end   = (i == nthreads - 1) ? total_rows : start + chunk;
        threads.emplace_back(worker_gemm<T>,
                             C.getRows(start, end),
                             a.getRows(start, end), b);
    }
    for (auto& t : threads) t.join();
    return C;
}

// ---- Reduction ops  (shape logic is template — stays here) ---------

template<typename T>
Tensor<T> sum(const Tensor<T>& t, int dim) {
    if (dim > (int)t.shape.size() - 1)
        throw std::invalid_argument("sum: dimension out of range");

    std::vector<int64_t> newShape;
    for (size_t i = 0; i < t.shape.size(); i++)
        if ((int)i != dim) newShape.push_back(t.shape[i]);

    Tensor<T> res(newShape);
    T*       out = res.data->data() + res.offset;
    const T* in  = t.data->data()   + t.offset;

    if (dim == (int)t.shape.size() - 1) {        // last dim — contiguous fast path
        int64_t chunk = t.shape[dim];
        size_t  oi    = 0;
        for (size_t i = 0; i < (size_t)t.getSize(); i += chunk) {
            T s = T(0);
            for (int64_t k = 0; k < chunk; k++) s += in[i + k];
            out[oi++] = s;
        }
        return res;
    }

    for (size_t i = 0; i < (size_t)res.getSize(); i++) out[i] = T(0);
    for (size_t i = 0; i < (size_t)t.getSize(); i++) {
        size_t out_id = 0, out_dim = 0, rem = i;
        for (size_t d = 0; d < t.shape.size(); d++) {
            size_t coord = rem / t.strides[d];
            rem         %= t.strides[d];
            if ((int)d != dim) {
                out_id += coord * res.strides[out_dim];
                out_dim++;
            }
        }
        out[out_id] += in[i];
    }
    return res;
}

template<typename T>
Tensor<T> mean(const Tensor<T>& t, int dim) {
    Tensor<T> result = sum(t, dim);
    T divisor = static_cast<T>(t.shape[dim]);
    for (auto& v : *result.data) v /= divisor;
    return result;
}

template<typename T>
Tensor<T> max(const Tensor<T>& t, int dim) {
    if (dim > (int)t.shape.size() - 1)
        throw std::out_of_range("max: dimension out of range");

    std::vector<int64_t> newShape;
    for (size_t i = 0; i < t.shape.size(); i++)
        if ((int)i != dim) newShape.push_back(t.shape[i]);

    Tensor<T> res(newShape);
    T*       out = res.data->data() + res.offset;
    const T* in  = t.data->data()   + t.offset;

    for (size_t i = 0; i < (size_t)res.getSize(); i++)
        out[i] = std::numeric_limits<T>::lowest();

    for (size_t i = 0; i < (size_t)t.getSize(); i++) {
        size_t out_id = 0, out_dim = 0, rem = i;
        for (size_t d = 0; d < t.shape.size(); d++) {
            size_t coord = rem / t.strides[d];
            rem         %= t.strides[d];
            if ((int)d != dim) {
                out_id += coord * res.strides[out_dim];
                out_dim++;
            }
        }
        if (in[i] > out[out_id]) out[out_id] = in[i];
    }
    return res;
}

// ---- Activations (TODO: add relu, sigmoid, softmax… as you implement them)
