# PyTorch From Scratch

A neural network framework built from scratch in C++20, following the same core design principles as PyTorch. The goal is to understand every layer of the stack — from raw memory layout to autograd to CUDA kernels — by building it yourself.

---

## Project Structure

```
my_torch/
├── CMakeLists.txt
├── include/
│   ├── tensor.hpp      # Core Tensor class: memory, shape, indexing
│   ├── ops.hpp         # Math operations + backend dispatch layer
│   ├── autograd.hpp    # Computation graph + backward pass  [STUB]
│   └── nn.hpp          # Linear, Loss, Optimizers            [STUB]
├── src/
│   ├── kernels_cpu.cpp # Typed CPU kernels (the actual math)
│   ├── kernels_cuda.cu # CUDA kernels                        [STUB]
│   ├── autograd.cpp    # Non-template autograd impl          [STUB]
│   └── nn.cpp          # Non-template nn impl                [STUB]
└── main.cpp            # Test / benchmark
```

### Why this layout?

The key architectural decision is **separating the dispatch layer from the kernels**:

- `tensor.hpp` and `ops.hpp` are C++ templates — they must stay in headers because the compiler needs to see the full definition for every type (`float`, `double`) at compile time.
- `kernels_cpu.cpp` and `kernels_cuda.cu` are **plain typed C functions** (no templates). This is where the actual math lives — loops, SIMD, CUDA thread blocks.
- `ops.hpp` bridges the two: it receives a `Tensor<T>`, extracts the raw pointer, and calls the right typed kernel via `dispatch_matmul<T>()` using `if constexpr`.

The payoff: when you add CUDA support, you only touch `kernels_cuda.cu` and flip one CMake flag. Nothing in `ops.hpp` or `tensor.hpp` changes.

---

## Build

### CPU (default)
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### CUDA (when you're ready)
```bash
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
cmake --build . --config Release
```

**Requirements:** C++20 compiler (GCC 11+ / Clang 13+ / MSVC 2022), CMake 3.20+. For CUDA: CUDA Toolkit 11.0+.

---

## What Is Already Implemented

### `tensor.hpp` — Core data structure

| Feature | Notes |
|---|---|
| `Tensor<T>` | Templated N-D array. Owns memory via `shared_ptr<vector<T>>` |
| Shape + strides | Row-major, computed automatically on construction |
| Offset | Enables zero-copy views into shared memory |
| `view(shape)` | Reinterpret shape, no copy (requires contiguous) |
| `reshape(shape)` | Like view but copies first if non-contiguous |
| `transpose(i, j)` | Swaps two axes by swapping strides — no copy |
| `isContiguous()` | Checks if strides match a fresh row-major layout |
| `makeContiguous()` | Physical copy into a new contiguous buffer |
| `getElement(i,j,…)` | Checked variadic access with rank validation |
| `operator()(i,j,…)` | Fast unchecked access, offset-aware (works on views) |
| `operator[](i)` | Drops outermost dimension, returns a view |
| `operator=` | Shape-checked assignment into a view's memory |
| `operator+`, `+=` | Element-wise add (tensor + tensor, tensor + scalar) |
| `operator-`, `-=` | Element-wise subtract |
| `getRows(start, end)` | Row-slice view (used by parallel matmul) |
| `getMatrix(batch_i)` | 2-D view of one matrix in a batched tensor |

### `ops.hpp` — Math operations

| Feature | Notes |
|---|---|
| `matmul(a, b)` | N-D matrix multiply with full broadcast support |
| `matmul2d(a, b)` | 2-D only, returns a new Tensor |
| `matmul2d_direct(a, b, ptr)` | Writes directly into pre-allocated memory — no allocation per batch |
| `pgemm(a, b)` | Parallel matmul using `std::thread`, splits rows across all CPU cores |
| `operator*(a, b)` | Delegates to `matmul` |
| `sum(t, dim)` | Reduce along a dimension. Has fast path for last (contiguous) dim |
| `mean(t, dim)` | `sum / shape[dim]` |
| `max(t, dim)` | Max reduction along a dimension |

### `src/kernels_cpu.cpp` — CPU kernels

| Function | Notes |
|---|---|
| `matmul_cpu_f32` | Cache-friendly i-j-k loop for `float` |
| `matmul_cpu_f64` | Same for `double` |

---

## What Still Needs To Be Built

Progress toward a working, trainable neural network.

### Phase 1 — Missing tensor ops `(ops.hpp + kernels_cpu.cpp)`

These are all blockers for Phase 2 and 3.

| Op | Why you need it |
|---|---|
| `zeros(shape)`, `ones(shape)`, `fill(val)` | Can't initialize weights or gradients without them |
| `hadamard(a, b)` | Element-wise multiply — `*` is matmul. Needed for gradient computations |
| `exp(t)`, `log(t)` | Required for softmax and cross-entropy loss |
| `relu(t)` | First activation function |
| `sigmoid(t)` | Needed for binary classification |
| `softmax(t, dim)` | Needed for multi-class classification |
| Broadcast `+` and `-` | Right now add/sub require exact shape match. `Linear` needs `output + bias` where bias is `{1, N}` |

### Phase 2 — Autograd `(autograd.hpp + autograd.cpp)`

The most important and hardest part. Without this, you can't train anything.

The core idea: every operation that produces a Tensor records *how it was computed*. When you call `backward()`, it walks that record in reverse and accumulates gradients.

| Component | What it does |
|---|---|
| `Variable` | Wraps a `Tensor`, holds a `grad` Tensor and a `grad_fn` |
| `grad_fn` | A callable that knows the backward rule for one specific op |
| `MatMulBackward` | `dA = dC * B.T`, `dB = A.T * dC` |
| `AddBackward` | `dA = dC`, `dB = dC` (sum over broadcast dims if needed) |
| `ReLUBackward` | `dX = dC * (X > 0)` — the gradient mask |
| `MSEBackward` | `dX = 2 * (pred - target) / N` |
| `backward()` | Topological sort of the graph, calls each `grad_fn` in reverse order |

### Phase 3 — Neural network modules `(nn.hpp + nn.cpp)`

Depends on Phase 2 (autograd) being done.

| Module | What it does |
|---|---|
| `Linear(in, out)` | Holds `weight {out, in}` and `bias {1, out}`. `forward(x)` = `x * W.T + bias` |
| `MSELoss` | `mean((pred - target)^2)` — regression loss |
| `CrossEntropyLoss` | `softmax` + `log` + `nll` — classification loss |
| `Sequential` | Chains modules: `forward(x)` calls each layer in order |

### Phase 4 — Optimizers `(nn.hpp)`

Depends on Phase 2 (gradients must exist before you can step).

| Optimizer | Notes |
|---|---|
| `SGD` | `weight -= lr * weight.grad` — simplest possible optimizer |
| `SGD + momentum` | Accumulates a velocity term to smooth updates |
| `Adam` | Adaptive per-parameter learning rate. The standard choice for most networks |

### Phase 5 — CUDA backend `(src/kernels_cuda.cu)`

The dispatch infrastructure is already in place. Adding CUDA means filling in the `TODO` stubs.

| Kernel | Notes |
|---|---|
| `matmul_cuda_f32/f64` | Start with naive one-thread-per-element. Then add tiled shared-memory version for 10-20x more throughput |
| `relu_cuda_f32/f64` | Embarrassingly parallel — one thread per element |
| `hadamard_cuda_f32/f64` | Same as relu, trivial |
| Memory management | Will need `cudaMalloc` / `cudaMemcpy` wrappers, or a unified memory strategy |

---

## The Critical Path (minimum to train something)

If you want to train a single Linear layer on a toy problem (XOR, sine regression) as fast as possible, the order is:

```
1. zeros() / fill()         ← need this to init weights
2. hadamard(a, b)           ← need this for backward passes
3. broadcast add            ← need this for bias in Linear
4. relu()                   ← first activation
5. Variable + grad_fn       ← the autograd engine
6. MatMulBackward           ← backward through Linear
7. AddBackward              ← backward through bias
8. ReLUBackward             ← backward through activation
9. MSELoss + MSEBackward    ← loss and its gradient
10. SGD optimizer           ← update the weights
11. Linear module           ← wire it all together
```

At step 11 you can train your first network.

---

## Design Principles

- **No dependencies.** Pure C++20 standard library only (until CUDA).
- **Understand before abstracting.** Each component is built at the level where the learning happens — no wrapping `cblas_sgemm` until you've written the loop yourself.
- **Templates at the boundary, typed kernels at the core.** `Tensor<T>` and op wrappers are templates for flexibility. The actual compute kernels are plain typed functions so they can be swapped for BLAS, SIMD, or CUDA without touching the API.
- **Shared memory by default.** `view()`, `transpose()`, `getRows()`, `getMatrix()` all share the underlying buffer. No copies unless you explicitly call `makeContiguous()` or `reshape()` on a non-contiguous tensor.
