#pragma once

#include "tensor.hpp"
#include <functional>
#include <memory>
#include <vector>

// ============================================================
//  autograd.hpp  –  Automatic differentiation (computation graph)
//
//  Planned responsibilities:
//    - Node / Variable  : wraps a Tensor + its gradient + grad_fn
//    - grad_fn          : functor that knows how to back-prop through
//                         a specific operation (AddBackward, MulBackward…)
//    - backward()       : walk the graph and accumulate gradients
//
//  Status: STUB – to be implemented next.
// ============================================================

// template<typename T>
// struct Node {
//     Tensor<T>              data;       // forward value
//     Tensor<T>              grad;       // accumulated gradient
//     std::function<void()>  grad_fn;   // backward function
//     std::vector<std::shared_ptr<Node<T>>> inputs; // parents in the graph
//
//     Node(Tensor<T> d) : data(d), grad(d.shape) {}
// };
