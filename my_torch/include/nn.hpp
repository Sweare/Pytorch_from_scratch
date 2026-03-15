#pragma once

#include "tensor.hpp"
#include "ops.hpp"
// #include "autograd.hpp"   // uncomment once autograd is ready

// ============================================================
//  nn.hpp  –  Neural-network modules
//
//  Planned responsibilities:
//    - Linear (fully-connected layer): weight, bias, forward()
//    - MSELoss : compute loss, kick off backward()
//    - SGD / Adam optimizers: step(), zero_grad()
//
//  Status: STUB – to be implemented after autograd.
// ============================================================

// template<typename T>
// struct Linear {
//     Tensor<T> weight;   // shape {out_features, in_features}
//     Tensor<T> bias;     // shape {1, out_features}
//
//     Linear(int64_t in, int64_t out)
//         : weight({out, in}), bias({1, out}) {}
//
//     Tensor<T> forward(const Tensor<T>& x) const {
//         return matmul(x, weight.transpose()) + bias;
//     }
// };

// template<typename T>
// T mse_loss(const Tensor<T>& pred, const Tensor<T>& target) {
//     Tensor<T> diff = pred - target;
//     // element-wise square + mean
//     ...
// }
