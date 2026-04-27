// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_benchmark.cpp — wall-clock benchmark harness for the XVM transition
// substrate.
//
// Per backend (CPU reference, Metal, CUDA-if-built, WGSL-if-built):
//   * 3 warm-up iterations + 10 measured iterations
//   * mean / p50 / p95 / p99 / min / max wall-clock per round
//   * throughput in UTXOs/sec and txs/sec
//   * speedup vs CPU
//
// Workload shape (matches xvm_determinism_test): N seed UTXOs (LUX) feeding M
// transfer txs, each tx consumes one seed and emits one new UTXO. Deterministic
// SplitMix64 utxo_id generator so every backend sees the same input bytes.
//
// Workload sizes are bounded by the GPU driver's per-round buffers:
//   kMaxTxs = 4096, kDefaultUtxoSlots = 16384, kMaxInputs = 65536. The
//   "xlarge" tier is CPU-only — exceeds the GPU per-round arenas — to show
//   reference scaling. -O3 -DNDEBUG release build, std::chrono::steady_clock.

#include "lux/xvm/xvm_gpu_engine.hpp"
#include "lux/xvm/xvm_cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace xvm::gpu;

#if defined(LUX_XVM_TEST_WGPU)
namespace xvm::gpu {
std::unique_ptr<XVMGPUEngine> create_xvm_gpu_engine_wgpu();
}
#endif

namespace {

constexpr int kWarmup     = 3;
constexpr int kMeasured   = 10;
// WGSL on M1 with 1x1x1 dispatches and synchronous map-read drains seconds
// per round even for tiny inputs. Run a smaller iteration set so the harness
// completes in finite time but still produces honest median + range.
constexpr int kWgpuWarmup   = 1;
constexpr int kWgpuMeasured = 3;

struct WorkloadShape {
    const char* name;
    uint32_t    seed_n;
    uint32_t    tx_n;
    bool        gpu_eligible;   // false for tiers that exceed GPU per-round arenas
    int         gpu_warmup;     // override for slow backends
    int         gpu_measured;
};

// Static workload table. seed_n + tx_n must fit kDefaultUtxoSlots (16384) on
// the GPU path because each tx inserts one new UTXO into the same arena that
// holds the seeds. xlarge exceeds the GPU per-round arenas and runs CPU-only.
//
// GPU iter counts shrink at higher tiers because Metal's 1x1x1 dispatches scale
// linearly with seed_n + tx_n — a single 12K-seed round can take 30+ s. We
// keep CPU at the full 3+10 budget so its percentiles stay tight; GPU uses a
// smaller-but-honest sample.
const WorkloadShape kWorkloads[] = {
    {"small",   1024u,    256u, true,  3, 10},
    {"medium",  4096u,   1024u, true,  2,  5},
    {"large",  12288u,   4096u, true,  1,  3},   // seed + tx == 16384 = arena cap
    {"xlarge", 32768u,   8192u, false, 0,  0},   // CPU only — pre-sized arena
};

void fill32(uint8_t out[32], uint8_t v) {
    for (uint32_t i = 0; i < 32; ++i) out[i] = v;
}

XVMRoundDescriptor make_desc(uint64_t round, uint32_t mode = static_cast<uint32_t>(XVMTransitionMode::FullRound)) {
    XVMRoundDescriptor d{};
    d.chain_id     = 1u;
    d.round        = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.height       = 100u + round;
    d.mode         = mode;
    d.closing_flag = 1u;
    return d;
}

struct UtxoId32 { uint8_t v[32]{}; };

struct Workload {
    uint8_t lux_id[32]{};
    uint8_t mint_auth[32]{};
    uint8_t alice[32]{};
    uint8_t bob[32]{};

    std::vector<UtxoId32>    seed_utxo_ids;
    std::vector<uint64_t>    seed_amounts;

    std::vector<XvmTx>       txs;
    std::vector<InputBatch>  ibs;
    std::vector<OutputBatch> obs;
    std::vector<uint8_t>     inputs;        // 32-byte utxo_ids concatenated
    std::vector<UTXO>        outputs;

    uint32_t seed_count = 0;
    uint32_t tx_count   = 0;
};

void build_workload(Workload& w, uint32_t seed_n, uint32_t tx_n) {
    fill32(w.lux_id, 0);
    fill32(w.mint_auth, 0xAA);
    fill32(w.alice, 0x11);
    fill32(w.bob, 0x22);

    w.seed_count = seed_n;
    w.tx_count   = tx_n;

    w.seed_utxo_ids.resize(seed_n);
    w.seed_amounts.resize(seed_n);

    // SplitMix64 — same generator the determinism harness uses, so utxo_ids
    // distribute across all 32 bytes (avoids cuckoo clustering in the low
    // 16 bits and keeps the Bloom filter honest).
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto smx = [&]() -> uint64_t {
        st += 0x9E3779B97F4A7C15ULL;
        uint64_t z = st;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (uint32_t i = 0; i < seed_n; ++i) {
        for (uint32_t k = 0; k < 32u; k += 8u) {
            uint64_t r = smx();
            for (uint32_t b = 0; b < 8u && k + b < 32u; ++b) {
                w.seed_utxo_ids[i].v[k + b] = uint8_t((r >> (b * 8u)) & 0xFFu);
            }
        }
        w.seed_amounts[i] = 100u + i;
    }

    w.txs.resize(tx_n);
    w.ibs.resize(tx_n);
    w.obs.resize(tx_n);
    w.inputs.assign(static_cast<uint64_t>(tx_n) * 32u, 0);
    w.outputs.resize(tx_n);

    for (uint32_t i = 0; i < tx_n; ++i) {
        XvmTx& tx = w.txs[i];
        tx.tx_id[0] = 0xC0;
        tx.tx_id[1] = uint8_t(i & 0xFF);
        tx.tx_id[2] = uint8_t((i >> 8) & 0xFF);
        tx.tx_id[3] = uint8_t((i >> 16) & 0xFF);
        tx.kind                = static_cast<uint32_t>(XvmTxKind::Transfer);
        tx.input_batch_offset  = i;
        tx.output_batch_offset = i;

        InputBatch& ib = w.ibs[i];
        std::memcpy(ib.tx_id, tx.tx_id, 32);
        ib.input_offset  = i * 32u;
        ib.input_count   = 1;
        ib.witness_count = 1;

        OutputBatch& ob = w.obs[i];
        std::memcpy(ob.tx_id, tx.tx_id, 32);
        ob.output_offset = i;
        ob.output_count  = 1;

        // Each tx consumes a distinct seed UTXO (i % seed_n). When tx_n <=
        // seed_n the inputs are unique; otherwise multiple txs would consume
        // the same UTXO and the duplicate-input rule would reject them. We
        // size workloads with tx_n <= seed_n so all txs accept and the
        // throughput numbers reflect the accept path.
        std::memcpy(w.inputs.data() + static_cast<uint64_t>(i) * 32u,
                    w.seed_utxo_ids[i % seed_n].v, 32);

        UTXO& nu = w.outputs[i];
        nu.utxo_id[0] = 0x20;
        nu.utxo_id[1] = uint8_t(i & 0xFF);
        nu.utxo_id[2] = uint8_t((i >> 8) & 0xFF);
        nu.utxo_id[3] = uint8_t((i >> 16) & 0xFF);
        std::memcpy(nu.asset_id, w.lux_id, 32);
        nu.amount_lo = w.seed_amounts[i % seed_n];
        std::memcpy(nu.owner_root, w.bob, 32);
        nu.threshold = 1;
    }
}

// =============================================================================
// CPU reference runner
// =============================================================================

struct RunStats {
    uint32_t tx_accepted   = 0;
    uint32_t tx_rejected   = 0;
    uint32_t inputs_consumed = 0;
    uint32_t outputs_created = 0;
    uint8_t  execution_root[32]{};
};

RunStats run_cpu_full(const Workload& w, uint64_t round_idx) {
    auto state = ref::XVMReferenceState::empty();
    // Pre-size for xlarge (>kDefaultUtxoSlots) so the seed inserts succeed.
    if (w.seed_count > state.utxos.size()) {
        state.utxos.assign(w.seed_count * 2u, UTXO{});
    }
    state.seed_asset(w.lux_id, 1ull << 50, 0, w.mint_auth, 9);
    for (uint32_t i = 0; i < w.seed_count; ++i) {
        state.seed_utxo(w.seed_utxo_ids[i].v, w.lux_id,
                        w.seed_amounts[i], 0, w.alice, 0, 1);
    }
    auto desc = make_desc(round_idx);
    auto r = ref::run_reference(state, desc, w.txs, w.ibs, w.obs,
                                w.inputs, w.outputs, {});
    RunStats s;
    s.tx_accepted     = r.tx_accepted;
    s.tx_rejected     = r.tx_rejected;
    s.inputs_consumed = r.inputs_consumed;
    s.outputs_created = r.outputs_created;
    std::memcpy(s.execution_root, r.execution_root, 32);
    return s;
}

// CPU per-kernel sub-run timings. The reference's mode dispatch lets us
// isolate each kernel by running a sub-mode round on a freshly-seeded state.
struct CpuKernelTimes {
    double input_apply_ns = 0.0;
    double asset_ns       = 0.0;
    double roots_ns       = 0.0;
};

CpuKernelTimes run_cpu_per_kernel(const Workload& w, uint64_t round_idx) {
    using clk = std::chrono::steady_clock;

    auto seed_state = [&]() {
        auto state = ref::XVMReferenceState::empty();
        if (w.seed_count > state.utxos.size()) {
            state.utxos.assign(w.seed_count * 2u, UTXO{});
        }
        state.seed_asset(w.lux_id, 1ull << 50, 0, w.mint_auth, 9);
        for (uint32_t i = 0; i < w.seed_count; ++i) {
            state.seed_utxo(w.seed_utxo_ids[i].v, w.lux_id,
                            w.seed_amounts[i], 0, w.alice, 0, 1);
        }
        return state;
    };

    CpuKernelTimes t{};

    // 1+2. UTXOInputCheck + UTXOTransitionApply (fused in the reference).
    {
        auto state = seed_state();
        auto desc  = make_desc(round_idx,
                               static_cast<uint32_t>(XVMTransitionMode::TransitionApply));
        auto t0 = clk::now();
        (void)ref::run_reference(state, desc, w.txs, w.ibs, w.obs,
                                 w.inputs, w.outputs, {});
        auto t1 = clk::now();
        t.input_apply_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    // 3. XAssetTransition (no asset ops in this workload — measures empty pass).
    {
        auto state = seed_state();
        auto desc  = make_desc(round_idx,
                               static_cast<uint32_t>(XVMTransitionMode::AssetTransition));
        auto t0 = clk::now();
        (void)ref::run_reference(state, desc, w.txs, {}, {}, {}, {}, {});
        auto t1 = clk::now();
        t.asset_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    // 4. XRootUpdate (always runs in run_reference; isolate by handing in a
    //    seeded state with no per-round work — RootUpdate mode).
    {
        auto state = seed_state();
        auto desc  = make_desc(round_idx,
                               static_cast<uint32_t>(XVMTransitionMode::RootUpdate));
        auto t0 = clk::now();
        (void)ref::run_reference(state, desc, {}, {}, {}, {}, {}, {});
        auto t1 = clk::now();
        t.roots_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    return t;
}

// =============================================================================
// GPU runner — generic over any XVMGPUEngine subclass
// =============================================================================

RunStats run_gpu_full(XVMGPUEngine* engine, const Workload& w, uint64_t round_idx) {
    auto desc = make_desc(round_idx);
    auto h = engine->begin_round(desc);
    engine->seed_asset(h, w.lux_id, 1ull << 50, 0, w.mint_auth, 9);
    for (uint32_t i = 0; i < w.seed_count; ++i) {
        engine->seed_utxo(h, w.seed_utxo_ids[i].v, w.lux_id,
                          w.seed_amounts[i], 0, w.alice, 0, 1);
    }
    engine->push_txs(h, w.txs);
    engine->push_input_batches(h, w.ibs);
    engine->push_output_batches(h, w.obs);
    engine->push_inputs(h, w.inputs);
    engine->push_outputs(h, w.outputs);
    auto r = engine->run_until_done(h);
    engine->end_round(h);

    RunStats s;
    s.tx_accepted     = r.tx_accepted;
    s.tx_rejected     = r.tx_rejected;
    s.inputs_consumed = r.inputs_consumed;
    s.outputs_created = r.outputs_created;
    std::memcpy(s.execution_root, r.execution_root, 32);
    return s;
}

// =============================================================================
// Stats reduction
// =============================================================================

struct LatencySummary {
    double mean_ns = 0.0;
    double p50_ns  = 0.0;
    double p95_ns  = 0.0;
    double p99_ns  = 0.0;
    double min_ns  = 0.0;
    double max_ns  = 0.0;
};

LatencySummary summarize(std::vector<double> samples_ns) {
    LatencySummary s{};
    if (samples_ns.empty()) return s;
    std::sort(samples_ns.begin(), samples_ns.end());
    double sum = 0.0;
    for (double v : samples_ns) sum += v;
    s.mean_ns = sum / static_cast<double>(samples_ns.size());
    auto pick = [&](double q) -> double {
        if (samples_ns.size() == 1) return samples_ns[0];
        double idx = q * static_cast<double>(samples_ns.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(idx));
        size_t hi = static_cast<size_t>(std::ceil(idx));
        double frac = idx - static_cast<double>(lo);
        return samples_ns[lo] + frac * (samples_ns[hi] - samples_ns[lo]);
    };
    s.p50_ns = pick(0.50);
    s.p95_ns = pick(0.95);
    s.p99_ns = pick(0.99);
    s.min_ns = samples_ns.front();
    s.max_ns = samples_ns.back();
    return s;
}

const char* fmt_time(double ns) {
    static thread_local char buf[64];
    if (ns < 1.0e3) {
        std::snprintf(buf, sizeof(buf), "%9.3f ns", ns);
    } else if (ns < 1.0e6) {
        std::snprintf(buf, sizeof(buf), "%9.3f us", ns / 1.0e3);
    } else if (ns < 1.0e9) {
        std::snprintf(buf, sizeof(buf), "%9.3f ms", ns / 1.0e6);
    } else {
        std::snprintf(buf, sizeof(buf), "%9.3f  s", ns / 1.0e9);
    }
    return buf;
}

const char* fmt_throughput(double per_sec) {
    static thread_local char buf[64];
    if (per_sec >= 1.0e6) {
        std::snprintf(buf, sizeof(buf), "%7.2fM/s", per_sec / 1.0e6);
    } else if (per_sec >= 1.0e3) {
        std::snprintf(buf, sizeof(buf), "%7.2fK/s", per_sec / 1.0e3);
    } else {
        std::snprintf(buf, sizeof(buf), "%7.2f /s", per_sec);
    }
    return buf;
}

// =============================================================================
// Per-backend driver
// =============================================================================

struct BackendResult {
    std::string    label;
    bool           ran = false;
    LatencySummary lat{};
    double         utxo_per_sec = 0.0;
    double         tx_per_sec   = 0.0;
    uint8_t        execution_root[32]{};
    uint32_t       tx_accepted = 0;
    uint32_t       tx_rejected = 0;
};

template <typename RunFn>
BackendResult bench_backend(const char* label, const Workload& w, RunFn&& run_one,
                            int warmup = kWarmup, int measured = kMeasured) {
    using clk = std::chrono::steady_clock;
    BackendResult br;
    br.label = label;

    // Warm-ups (results discarded).
    for (int i = 0; i < warmup; ++i) {
        (void)run_one(static_cast<uint64_t>(i + 1));
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(measured));
    RunStats last{};
    for (int i = 0; i < measured; ++i) {
        auto t0 = clk::now();
        last = run_one(static_cast<uint64_t>(warmup + i + 1));
        auto t1 = clk::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        samples.push_back(ns);
    }
    br.lat          = summarize(samples);
    double mean_sec = br.lat.mean_ns / 1.0e9;
    br.utxo_per_sec = static_cast<double>(w.seed_count) / mean_sec;
    br.tx_per_sec   = static_cast<double>(w.tx_count)   / mean_sec;
    br.tx_accepted  = last.tx_accepted;
    br.tx_rejected  = last.tx_rejected;
    std::memcpy(br.execution_root, last.execution_root, 32);
    br.ran          = true;
    return br;
}

void print_workload_header(const WorkloadShape& s) {
    std::printf("\n=== %s : %u UTXOs / %u txs ===\n", s.name, s.seed_n, s.tx_n);
    std::fflush(stdout);
}

void print_backend_row(const BackendResult& br, const BackendResult* baseline) {
    if (!br.ran) {
        std::printf("  %-22s  (skipped)\n", br.label.c_str());
        return;
    }
    double speedup = (baseline && baseline->ran && br.lat.mean_ns > 0.0)
                       ? (baseline->lat.mean_ns / br.lat.mean_ns)
                       : 1.0;
    std::printf("  %-22s  mean=%s  p50=%s  p95=%s  p99=%s  min=%s  max=%s  | %s UTXOs/s  %s txs/s  | %6.2fx vs CPU  | accept=%u reject=%u\n",
                br.label.c_str(),
                fmt_time(br.lat.mean_ns),
                fmt_time(br.lat.p50_ns),
                fmt_time(br.lat.p95_ns),
                fmt_time(br.lat.p99_ns),
                fmt_time(br.lat.min_ns),
                fmt_time(br.lat.max_ns),
                fmt_throughput(br.utxo_per_sec),
                fmt_throughput(br.tx_per_sec),
                speedup,
                br.tx_accepted, br.tx_rejected);
    std::fflush(stdout);
}

void print_root(const char* label, const uint8_t r[32]) {
    std::printf("    %s.execution_root = ", label);
    for (int i = 0; i < 32; ++i) std::printf("%02x", r[i]);
    std::printf("\n");
}

bool roots_equal(const uint8_t a[32], const uint8_t b[32]) {
    return std::memcmp(a, b, 32) == 0;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    std::printf("=== xvm-benchmark — XVM transition substrate wall-clock ===\n");
    std::printf("warmup=%d  measured=%d  build=Release -O3 -DNDEBUG\n", kWarmup, kMeasured);

    // Probe backends.
    auto default_engine = XVMGPUEngine::create();   // Metal on Apple, CUDA on Linux+CUDA, else nullptr
#if defined(LUX_XVM_TEST_WGPU)
    auto wgpu_engine    = create_xvm_gpu_engine_wgpu();
#else
    std::unique_ptr<XVMGPUEngine> wgpu_engine;
#endif

    std::printf("backends:\n");
    std::printf("  CPU reference : enabled\n");
    std::printf("  default GPU   : %s\n",
                default_engine ? default_engine->device_name() : "(unavailable)");
#if defined(LUX_XVM_TEST_WGPU)
    std::printf("  WGSL/WebGPU   : %s\n",
                wgpu_engine ? wgpu_engine->device_name() : "(runtime unavailable)");
#else
    std::printf("  WGSL/WebGPU   : (not built)\n");
#endif

    for (const auto& shape : kWorkloads) {
        Workload w;
        build_workload(w, shape.seed_n, shape.tx_n);
        print_workload_header(shape);

        // -------------------------------------------------------------------
        // Determinism oracle — single canonical run at a fixed descriptor on
        // every backend. Without this, the per-backend wall-clock loops
        // (which iterate round_idx for cache freshness) produce a LAST root
        // tied to a different `height` per backend, which falsely looks like
        // kernel divergence. The timed loop below uses iterating round_idx
        // for honest wall-clock numbers; this block alone is the determinism
        // check, and is what gets reported as "execution_root" per backend.
        // -------------------------------------------------------------------
        constexpr uint64_t kCanonRound = 1u;  // matches xvm_determinism_test
        uint8_t canon_cpu_root[32]{};
        {
            auto s = run_cpu_full(w, kCanonRound);
            std::memcpy(canon_cpu_root, s.execution_root, 32);
        }
        std::printf("  canonical execution_root @ round=%llu : ",
                    (unsigned long long)kCanonRound);
        for (int i = 0; i < 32; ++i) std::printf("%02x", canon_cpu_root[i]);
        std::printf("  (CPU oracle)\n");

        bool det_ok = true;
        if (shape.gpu_eligible && default_engine) {
            auto s = run_gpu_full(default_engine.get(), w, kCanonRound);
            if (!roots_equal(canon_cpu_root, s.execution_root)) {
                det_ok = false;
                std::printf("  DETERMINISM FAIL: %s @ round=%llu\n",
                            default_engine->device_name(),
                            (unsigned long long)kCanonRound);
                print_root("CPU", canon_cpu_root);
                print_root(default_engine->device_name(), s.execution_root);
            }
        }
#if defined(LUX_XVM_TEST_WGPU)
        if (shape.gpu_eligible && wgpu_engine) {
            auto s = run_gpu_full(wgpu_engine.get(), w, kCanonRound);
            if (!roots_equal(canon_cpu_root, s.execution_root)) {
                det_ok = false;
                std::printf("  DETERMINISM FAIL: %s (WGSL) @ round=%llu\n",
                            wgpu_engine->device_name(),
                            (unsigned long long)kCanonRound);
                print_root("CPU",                          canon_cpu_root);
                std::string lbl = std::string(wgpu_engine->device_name()) + " (WGSL)";
                print_root(lbl.c_str(),                    s.execution_root);
            }
        }
#endif
        if (det_ok)
            std::printf("  determinism: CPU == all GPU backends @ round=%llu (byte-equal)\n",
                        (unsigned long long)kCanonRound);

        // -------------------------------------------------------------------
        // Wall-clock loops — each iteration uses a distinct round_idx so the
        // workload's `height` field varies and per-tx tx_id does not collide
        // across iterations. We measure mean/p50/p95/p99/min/max latency.
        // The per-iteration root is NOT compared cross-backend (different
        // round_idx per backend would falsely flag a mismatch); the canonical
        // check above is the determinism oracle.
        // -------------------------------------------------------------------
        BackendResult cpu = bench_backend("CPU", w, [&](uint64_t round) {
            return run_cpu_full(w, round);
        });
        print_backend_row(cpu, nullptr);

        BackendResult gpu_default;
        if (shape.gpu_eligible && default_engine) {
            std::string default_label =
                std::string(default_engine->device_name())
#if defined(__APPLE__)
                + " (Metal)";
#else
                + " (CUDA)";
#endif
            gpu_default = bench_backend(default_label.c_str(), w,
                                        [&](uint64_t round) {
                return run_gpu_full(default_engine.get(), w, round);
            }, shape.gpu_warmup, shape.gpu_measured);
            print_backend_row(gpu_default, &cpu);
        }

#if defined(LUX_XVM_TEST_WGPU)
        BackendResult gpu_wgpu;
        if (shape.gpu_eligible && wgpu_engine) {
            std::string wgpu_label = std::string(wgpu_engine->device_name()) + " (WGSL)";
            // WGSL is slower per iter than Metal; reuse the per-tier GPU
            // budget but never run more than the WGSL ceiling.
            int wgpu_warmup   = std::min(shape.gpu_warmup,   kWgpuWarmup);
            int wgpu_measured = std::min(shape.gpu_measured, kWgpuMeasured);
            gpu_wgpu = bench_backend(wgpu_label.c_str(), w,
                                     [&](uint64_t round) {
                return run_gpu_full(wgpu_engine.get(), w, round);
            }, wgpu_warmup, wgpu_measured);
            print_backend_row(gpu_wgpu, &cpu);
        }
#endif

        // Per-kernel CPU breakdown. Each sub-mode run also computes roots, so
        // we subtract the dedicated XRootUpdate timing to isolate the kernel.
        std::printf("  per-kernel (CPU, avg of %d):\n", kMeasured);
        double sum_ia = 0.0, sum_a = 0.0, sum_r = 0.0;
        for (int i = 0; i < kMeasured; ++i) {
            auto kt = run_cpu_per_kernel(w, static_cast<uint64_t>(1000 + i));
            sum_ia += kt.input_apply_ns;
            sum_a  += kt.asset_ns;
            sum_r  += kt.roots_ns;
        }
        double avg_ia_total = sum_ia / static_cast<double>(kMeasured);
        double avg_a_total  = sum_a  / static_cast<double>(kMeasured);
        double avg_r        = sum_r  / static_cast<double>(kMeasured);
        double avg_ia       = std::max(avg_ia_total - avg_r, 0.0);
        double avg_a        = std::max(avg_a_total  - avg_r, 0.0);
        std::printf("    UTXOInputCheck+TransitionApply : %s\n", fmt_time(avg_ia));
        std::printf("    XAssetTransition (empty pass)  : %s\n", fmt_time(avg_a));
        std::printf("    XRootUpdate                    : %s\n", fmt_time(avg_r));
    }

    std::printf("\n=== xvm-benchmark complete ===\n");
    return 0;
}
