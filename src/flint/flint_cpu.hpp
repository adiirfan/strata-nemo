// src/flint/flint_cpu.hpp - hybrid mode's CPU side (see flint_cpu.cpp)
#pragma once
#include <cstdint>
#include <functional>

namespace flint_cpu {

// one routed expert's three Q4_1 matrices in host memory (GGUF layout, rows of 32-value blocks of 20 bytes)
struct Expert { const void* gate; const void* up; const void* down; };

// the worker threads (0: one per physical core); once
void init(int threads);
// workers spin while a writing step is under way, sleep otherwise
void set_active(bool active);
// y (n_in) = sum_k w[k] * down_k(silu(gate_k x) * up_k x), x = rmsnorm(h) * nw; gate / up are n_ff x n_in, down
// n_in x n_ff
void moe(const float* h, const float* nw, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y);

}  // namespace flint_cpu
