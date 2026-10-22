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
// GGUF type codes moe_q takes
constexpr int kQ4_0 = 2, kQ4_1 = 3;
// as moe(), with each matrix's format (gate / up share one) and the swiglu clamp (gate <= c, |up| <= c; 0: none)
void moe_q(const float* h, const float* nw, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y, int fmt_gu,
           int fmt_d, float clamp);

}  // namespace flint_cpu
