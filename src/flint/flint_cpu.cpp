// src/flint/flint_cpu.cpp - hybrid mode: the routed experts the card does not hold, computed on the CPU from host
// memory while the card computes the ones it holds. Q4_1 rows (GGUF block: fp16 d, fp16 m, 16 bytes of nibbles - the
// low nibbles are values 0-15, the high ones 16-31) times fp32 inputs, as the card's writing path does (no rounding of
// the inputs: Flint is sensitive to it). AVX-512 where the CPU has it, AVX2 otherwise.
#include "flint_cpu.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <fstream>
#include <set>
#include <pthread.h>
#include <sched.h>

namespace flint_cpu {
namespace {

struct B { uint16_t d, m; uint8_t q[16]; };
static_assert(sizeof(B) == 20, "Q4_1 block");

// one row (nb blocks) . x, with sx[b] = the sum of x over block b: d * sum(q x) + m * sum(x) a block
__attribute__((target("avx512f,avx512bw,f16c,fma"))) float dot512(const B* w, const float* x, const float* sx, int nb) {
    __m512 acc = _mm512_setzero_ps();
    float mt = 0.f;
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < nb; ++b) {
        const __m128i v = _mm_loadu_si128((const __m128i*) w[b].q);
        const __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v, mask)));
        const __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v, 4), mask)));
        __m512 t = _mm512_mul_ps(lo, _mm512_loadu_ps(x + b * 32));
        t = _mm512_fmadd_ps(hi, _mm512_loadu_ps(x + b * 32 + 16), t);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(_cvtsh_ss(w[b].d)), t, acc);
        mt += _cvtsh_ss(w[b].m) * sx[b];
    }
    return _mm512_reduce_add_ps(acc) + mt;
}
__attribute__((target("avx2,f16c,fma"))) float dot256(const B* w, const float* x, const float* sx, int nb) {
    __m256 acc = _mm256_setzero_ps();
    float mt = 0.f;
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < nb; ++b) {
        const __m128i v = _mm_loadu_si128((const __m128i*) w[b].q);
        const __m128i lo = _mm_and_si128(v, mask), hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        const float* xb = x + b * 32;
        __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), _mm256_loadu_ps(xb));
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(xb + 8), t);
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), _mm256_loadu_ps(xb + 16), t);
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(xb + 24), t);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(_cvtsh_ss(w[b].d)), t, acc);
        mt += _cvtsh_ss(w[b].m) * sx[b];
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s) + mt;
}
// Q4_0 (fp16 d, 16 bytes of nibbles; value d * (q - 8)): d * (sum q x - 8 sum x) a block
struct B0 { uint16_t d; uint8_t q[16]; };
static_assert(sizeof(B0) == 18, "Q4_0 block");
__attribute__((target("avx512f,avx512bw,f16c,fma"))) float dot0_512(const B0* w, const float* x, const float* sx, int nb) {
    __m512 acc = _mm512_setzero_ps();
    float mt = 0.f;
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < nb; ++b) {
        const __m128i v = _mm_loadu_si128((const __m128i*) w[b].q);
        const __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(v, mask)));
        const __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v, 4), mask)));
        __m512 t = _mm512_mul_ps(lo, _mm512_loadu_ps(x + b * 32));
        t = _mm512_fmadd_ps(hi, _mm512_loadu_ps(x + b * 32 + 16), t);
        const float d = _cvtsh_ss(w[b].d);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), t, acc);
        mt -= 8.f * d * sx[b];
    }
    return _mm512_reduce_add_ps(acc) + mt;
}
__attribute__((target("avx2,f16c,fma"))) float dot0_256(const B0* w, const float* x, const float* sx, int nb) {
    __m256 acc = _mm256_setzero_ps();
    float mt = 0.f;
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int b = 0; b < nb; ++b) {
        const __m128i v = _mm_loadu_si128((const __m128i*) w[b].q);
        const __m128i lo = _mm_and_si128(v, mask), hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        const float* xb = x + b * 32;
        __m256 t = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)), _mm256_loadu_ps(xb));
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(xb + 8), t);
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)), _mm256_loadu_ps(xb + 16), t);
        t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(xb + 24), t);
        const float d = _cvtsh_ss(w[b].d);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), t, acc);
        mt -= 8.f * d * sx[b];
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s) + mt;
}
using Dot0Fn = float (*)(const B0*, const float*, const float*, int);
const Dot0Fn dot0 = [] { __builtin_cpu_init(); return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") ? dot0_512 : dot0_256; }();
using DotFn = float (*)(const B*, const float*, const float*, int);
DotFn pick_dot() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") ? dot512 : dot256;
}
const DotFn dot = pick_dot();

void pin_to(int cpu);

// A pool of n - 1 workers plus the calling thread. While active (a writing step under way) the workers spin, since
// a layer's CPU share is ~100 us and waking sleeping threads twice a layer would cost a good part of that; otherwise
// they sleep on a condition variable.
class Pool {
public:
    explicit Pool(int n, const std::vector<int>& cores) : n_(n) {
        for (int i = 1; i < n_; ++i)
            th_.emplace_back([this, i, cores] {
                if (i < (int) cores.size()) pin_to(cores[i]);
                work(i);
            });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(mu_); quit_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    int size() const { return n_; }
    void set_active(bool a) {
        active_.store(a, std::memory_order_relaxed);
        if (a) cv_.notify_all();
    }
    template <class F> void run(F&& f) {
        job_ = [&f](int id) { f(id); };
        left_.store(n_ - 1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            gen_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        job_(0);
        while (left_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

private:
    void work(int id) {
        uint64_t seen = 0;
        for (;;) {
            while (gen_.load(std::memory_order_acquire) == seen) {
                if (quit_) return;
                if (active_.load(std::memory_order_relaxed)) { _mm_pause(); continue; }
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_ || gen_.load(std::memory_order_acquire) != seen || active_.load(std::memory_order_relaxed); });
            }
            if (quit_) return;
            seen = gen_.load(std::memory_order_acquire);
            job_(id);
            left_.fetch_sub(1, std::memory_order_release);
        }
    }
    int n_;
    std::vector<std::thread> th_;
    std::function<void(int)> job_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> left_{0};
    std::atomic<bool> active_{false};
    bool quit_ = false;
    std::mutex mu_;
    std::condition_variable cv_;
};

Pool* g_pool = nullptr;

// one logical processor per physical core (the first of its SMT siblings), in order: two workers on one core's
// siblings would share its memory bandwidth (Strata's pool measured that as the CPU path running far below its model)
std::vector<int> physical_cores() {
    std::vector<int> out;
    std::set<std::string> seen;
    for (int c = 0; c < 1024; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!f) break;
        std::string sib;
        std::getline(f, sib);
        if (seen.insert(sib).second) out.push_back(c);
    }
    return out;
}
void pin_to(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}
std::vector<int> g_cores;
cpu_set_t g_host_saved;
bool g_host_pinned = false;

inline float silu(float v) { return v / (1.f + std::exp(-v)); }

}  // namespace

void init(int threads) {
    if (g_pool) return;
    g_cores = physical_cores();
    if (threads <= 0) threads = g_cores.empty() ? int(std::max(1u, std::thread::hardware_concurrency() / 2)) : int(g_cores.size());
    g_pool = new Pool(threads, g_cores);
}
void set_active(bool a) {
    if (!g_pool) return;
    g_pool->set_active(a);
    // the calling thread (worker 0, which also watches the card's flags) on the first physical core while active
    if (a && !g_cores.empty() && !g_host_pinned) {
        g_host_pinned = pthread_getaffinity_np(pthread_self(), sizeof g_host_saved, &g_host_saved) == 0;
        if (g_host_pinned) pin_to(g_cores[0]);
    } else if (!a && g_host_pinned) {
        pthread_setaffinity_np(pthread_self(), sizeof g_host_saved, &g_host_saved);
        g_host_pinned = false;
    }
}

void moe(const float* h, const float* nw, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y) {
    // x = rmsnorm(h) * nw, and its block sums
    // (kept between calls: no allocation per layer; the workers get plain pointers - the names below are the CALLING
    // thread's buffers, which a worker naming them would not see)
    static thread_local std::vector<float> xv, sxv, actv, sav;
    xv.resize(n_in); sxv.resize(n_in / 32); actv.resize(size_t(n) * n_ff); sav.resize(size_t(n) * (n_ff / 32));
    float *x = xv.data(), *sx = sxv.data(), *act = actv.data(), *sa = sav.data();
    double ss = 0;
    for (int i = 0; i < n_in; ++i) ss += double(h[i]) * h[i];
    const float r = 1.f / std::sqrt(float(ss / n_in) + 1e-6f);
    for (int i = 0; i < n_in; ++i) x[i] = h[i] * r * nw[i];
    for (int b = 0; b < n_in / 32; ++b) { float s = 0; for (int i = 0; i < 32; ++i) s += x[b * 32 + i]; sx[b] = s; }
    const int P = g_pool->size(), nbi = n_in / 32, nbf = n_ff / 32;
    // gate / up: n * n_ff rows, split evenly
    const int rows1 = n * n_ff;
    g_pool->run([&](int id) {
        const int a = rows1 * id / P, z = rows1 * (id + 1) / P;
        for (int rr = a; rr < z; ++rr) {
            const int k = rr / n_ff, row = rr % n_ff;
            const float gv = dot((const B*) ex[k].gate + size_t(row) * nbi, x, sx, nbi);
            const float uv = dot((const B*) ex[k].up + size_t(row) * nbi, x, sx, nbi);
            act[rr] = silu(gv) * uv;
        }
    });
    for (int k = 0; k < n; ++k)
        for (int b = 0; b < nbf; ++b) { float s = 0; for (int i = 0; i < 32; ++i) s += act[size_t(k) * n_ff + b * 32 + i]; sa[size_t(k) * nbf + b] = s; }
    // down: n_in output rows, each the weighted sum over the n experts
    g_pool->run([&](int id) {
        const int a = n_in * id / P, z = n_in * (id + 1) / P;
        for (int row = a; row < z; ++row) {
            float t = 0;
            for (int k = 0; k < n; ++k)
                t += w[k] * dot((const B*) ex[k].down + size_t(row) * nbf, act + size_t(k) * n_ff, sa + size_t(k) * nbf, nbf);
            y[row] = t;
        }
    });
}

// a row (nb blocks) of format fmt (kQ4_0 or kQ4_1) . x
static inline float row_dot(int fmt, const void* row, const float* x, const float* sx, int nb) {
    return fmt == kQ4_0 ? dot0((const B0*) row, x, sx, nb) : dot((const B*) row, x, sx, nb);
}
static inline size_t row_bytes(int fmt, int n) { return size_t(n / 32) * (fmt == kQ4_0 ? 18 : 20); }

void moe_q(const float* h, const float* nw, int n_in, int n_ff, int n, const Expert* ex, const float* w, float* y, int fmt_gu,
           int fmt_d, float clamp) {
    static thread_local std::vector<float> xv, sxv, actv, sav;   // (see moe(): the workers get plain pointers)
    xv.resize(n_in); sxv.resize(n_in / 32); actv.resize(size_t(n) * n_ff); sav.resize(size_t(n) * (n_ff / 32));
    float *x = xv.data(), *sx = sxv.data(), *act = actv.data(), *sa = sav.data();
    double ss = 0;
    for (int i = 0; i < n_in; ++i) ss += double(h[i]) * h[i];
    const float r = 1.f / std::sqrt(float(ss / n_in) + 1e-6f);
    for (int i = 0; i < n_in; ++i) x[i] = h[i] * r * nw[i];
    for (int b = 0; b < n_in / 32; ++b) { float s = 0; for (int i = 0; i < 32; ++i) s += x[b * 32 + i]; sx[b] = s; }
    const int P = g_pool->size(), nbi = n_in / 32, nbf = n_ff / 32;
    const size_t rb_gu = row_bytes(fmt_gu, n_in), rb_d = row_bytes(fmt_d, n_ff);
    const int rows1 = n * n_ff;
    g_pool->run([&](int id) {
        const int a = rows1 * id / P, z = rows1 * (id + 1) / P;
        for (int rr = a; rr < z; ++rr) {
            const int k = rr / n_ff, row = rr % n_ff;
            float gv = row_dot(fmt_gu, (const uint8_t*) ex[k].gate + row * rb_gu, x, sx, nbi);
            float uv = row_dot(fmt_gu, (const uint8_t*) ex[k].up + row * rb_gu, x, sx, nbi);
            if (clamp > 0.f) { gv = std::min(gv, clamp); uv = std::max(-clamp, std::min(uv, clamp)); }
            act[rr] = silu(gv) * uv;
        }
    });
    for (int k = 0; k < n; ++k)
        for (int b = 0; b < nbf; ++b) { float s = 0; for (int i = 0; i < 32; ++i) s += act[size_t(k) * n_ff + b * 32 + i]; sa[size_t(k) * nbf + b] = s; }
    g_pool->run([&](int id) {
        const int a = n_in * id / P, z = n_in * (id + 1) / P;
        for (int row = a; row < z; ++row) {
            float t = 0;
            for (int k = 0; k < n; ++k)
                t += w[k] * row_dot(fmt_d, (const uint8_t*) ex[k].down + row * rb_d, act + size_t(k) * n_ff, sa + size_t(k) * nbf, nbf);
            y[row] = t;
        }
    });
}

}  // namespace flint_cpu
