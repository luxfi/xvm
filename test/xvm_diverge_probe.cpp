// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_diverge_probe.cpp — root-level divergence probe.
//
// Reproduces the size-dependent root mismatch seen in the v0.55.2 benchmark.
// For each canonical workload size, runs CPU + Metal + WGSL and prints which
// root component (utxo / asset / tx / execution) is the first to diverge.

#include "lux/xvm/xvm_gpu_engine.hpp"
#include "lux/xvm/xvm_cpu_reference.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace xvm::gpu;

#if defined(LUX_XVM_TEST_WGPU)
namespace xvm::gpu {
std::unique_ptr<XVMGPUEngine> create_xvm_gpu_engine_wgpu();
}
#endif

namespace {

void fill32(uint8_t out[32], uint8_t v) { for (uint32_t i = 0; i < 32; ++i) out[i] = v; }

void print_root(const char* label, const uint8_t r[32]) {
    std::printf("    %s = ", label);
    for (int i = 0; i < 32; ++i) std::printf("%02x", r[i]);
    std::printf("\n");
}

XVMRoundDescriptor make_desc(uint64_t round) {
    XVMRoundDescriptor d{};
    d.chain_id = 1u; d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.height = 100u + round;
    d.mode = static_cast<uint32_t>(XVMTransitionMode::FullRound);
    d.closing_flag = 1u;
    return d;
}

struct UtxoId32 { uint8_t v[32]{}; };

struct Workload {
    uint8_t lux_id[32]{};
    uint8_t mint_auth[32]{};
    uint8_t alice[32]{};
    uint8_t bob[32]{};
    std::vector<UtxoId32>   seed_utxo_ids;
    std::vector<uint64_t>   seed_amounts;
    std::vector<XvmTx>      txs;
    std::vector<InputBatch> ibs;
    std::vector<OutputBatch> obs;
    std::vector<uint8_t>    inputs;
    std::vector<UTXO>       outputs;
    uint32_t seed_count = 0;
    uint32_t tx_count   = 0;
};

void build_workload(Workload& w, uint32_t seed_n, uint32_t tx_n) {
    fill32(w.lux_id, 0);
    fill32(w.mint_auth, 0xAA);
    fill32(w.alice, 0x11);
    fill32(w.bob, 0x22);
    w.seed_count = seed_n;
    w.tx_count = tx_n;
    w.seed_utxo_ids.resize(seed_n);
    w.seed_amounts.resize(seed_n);
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    auto smx = [&]() -> uint64_t {
        st += 0x9E3779B97F4A7C15ULL;
        uint64_t z = st;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };
    for (uint32_t i = 0; i < seed_n; ++i) {
        for (uint32_t k = 0; k < 32u; k += 8) {
            uint64_t r = smx();
            for (uint32_t b = 0; b < 8u && k + b < 32u; ++b)
                w.seed_utxo_ids[i].v[k + b] = uint8_t((r >> (b*8)) & 0xFF);
        }
        w.seed_amounts[i] = 100u + i;
    }
    w.txs.resize(tx_n);
    w.ibs.resize(tx_n);
    w.obs.resize(tx_n);
    w.inputs.assign(uint64_t(tx_n) * 32u, 0);
    w.outputs.resize(tx_n);
    for (uint32_t i = 0; i < tx_n; ++i) {
        XvmTx& tx = w.txs[i];
        tx.tx_id[0] = 0xC0; tx.tx_id[1] = uint8_t(i & 0xFF);
        tx.tx_id[2] = uint8_t((i >> 8) & 0xFF);
        tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
        tx.input_batch_offset = i;
        tx.output_batch_offset = i;
        InputBatch& ib = w.ibs[i];
        std::memcpy(ib.tx_id, tx.tx_id, 32);
        ib.input_offset = i * 32u;
        ib.input_count = 1;
        ib.witness_count = 1;
        OutputBatch& ob = w.obs[i];
        std::memcpy(ob.tx_id, tx.tx_id, 32);
        ob.output_offset = i;
        ob.output_count = 1;
        std::memcpy(w.inputs.data() + uint64_t(i) * 32u,
                    w.seed_utxo_ids[i % seed_n].v, 32);
        UTXO& nu = w.outputs[i];
        nu.utxo_id[0] = 0x20; nu.utxo_id[1] = uint8_t(i & 0xFF);
        nu.utxo_id[2] = uint8_t((i >> 8) & 0xFF);
        std::memcpy(nu.asset_id, w.lux_id, 32);
        nu.amount_lo = w.seed_amounts[i % seed_n];
        std::memcpy(nu.owner_root, w.bob, 32);
        nu.threshold = 1;
    }
}

XVMTransitionResult run_cpu(const Workload& w, const XVMRoundDescriptor& desc) {
    auto state = ref::XVMReferenceState::empty();
    if (w.seed_count > state.utxos.size()) state.utxos.assign(w.seed_count * 2u, UTXO{});
    state.seed_asset(w.lux_id, 1ull << 50, 0, w.mint_auth, 9);
    for (uint32_t i = 0; i < w.seed_count; ++i)
        state.seed_utxo(w.seed_utxo_ids[i].v, w.lux_id, w.seed_amounts[i],
                        0, w.alice, 0, 1);
    return ref::run_reference(state, desc, w.txs, w.ibs, w.obs, w.inputs, w.outputs, {});
}

XVMTransitionResult run_gpu(XVMGPUEngine* eng, const Workload& w, const XVMRoundDescriptor& desc) {
    auto h = eng->begin_round(desc);
    eng->seed_asset(h, w.lux_id, 1ull << 50, 0, w.mint_auth, 9);
    for (uint32_t i = 0; i < w.seed_count; ++i)
        eng->seed_utxo(h, w.seed_utxo_ids[i].v, w.lux_id, w.seed_amounts[i],
                       0, w.alice, 0, 1);
    eng->push_txs(h, w.txs);
    eng->push_input_batches(h, w.ibs);
    eng->push_output_batches(h, w.obs);
    eng->push_inputs(h, w.inputs);
    eng->push_outputs(h, w.outputs);
    auto r = eng->run_until_done(h);
    eng->end_round(h);
    return r;
}

void compare(const char* tier, const XVMTransitionResult& cpu,
             const XVMTransitionResult& gpu, const char* gpu_name) {
    std::printf("[%s] CPU vs %s\n", tier, gpu_name);
    bool um = std::memcmp(cpu.utxo_root,      gpu.utxo_root,      32) == 0;
    bool am = std::memcmp(cpu.asset_root,     gpu.asset_root,     32) == 0;
    bool tm = std::memcmp(cpu.tx_root,        gpu.tx_root,        32) == 0;
    bool em = std::memcmp(cpu.execution_root, gpu.execution_root, 32) == 0;
    std::printf("  utxo_root      : %s   acc(cpu)=%u acc(gpu)=%u  consumed cpu=%u gpu=%u  created cpu=%u gpu=%u\n",
                um ? "MATCH" : "DIFF",
                cpu.tx_accepted, gpu.tx_accepted,
                cpu.inputs_consumed, gpu.inputs_consumed,
                cpu.outputs_created, gpu.outputs_created);
    if (!um) { print_root("cpu.utxo", cpu.utxo_root); print_root("gpu.utxo", gpu.utxo_root); }
    std::printf("  asset_root     : %s\n", am ? "MATCH" : "DIFF");
    if (!am) { print_root("cpu.asset", cpu.asset_root); print_root("gpu.asset", gpu.asset_root); }
    std::printf("  tx_root        : %s\n", tm ? "MATCH" : "DIFF");
    if (!tm) { print_root("cpu.tx", cpu.tx_root); print_root("gpu.tx", gpu.tx_root); }
    std::printf("  execution_root : %s\n", em ? "MATCH" : "DIFF");
    if (!em) { print_root("cpu.exec", cpu.execution_root); print_root("gpu.exec", gpu.execution_root); }
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    bool wgsl_only = (argc > 1 && std::strcmp(argv[1], "--wgsl-only") == 0);
    bool metal_only = (argc > 1 && std::strcmp(argv[1], "--metal-only") == 0);
    const char* filter = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tier") == 0 && i + 1 < argc) filter = argv[++i];
    }
    auto metal = (wgsl_only ? nullptr : XVMGPUEngine::create());
#if defined(LUX_XVM_TEST_WGPU)
    auto wgpu = (metal_only ? nullptr : create_xvm_gpu_engine_wgpu());
#else
    std::unique_ptr<XVMGPUEngine> wgpu;
#endif

    struct Tier { const char* name; uint32_t seed; uint32_t tx; };
    Tier tiers[] = {
        {"small",   1024u,    256u},
        {"medium",  4096u,   1024u},
        {"large",  12288u,   4096u},
        {"brief",  10000u,   1000u},  // existing pinned case
    };
    bool repeat = false;
    bool bench_repro = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--repeat") == 0) repeat = true;
        if (std::strcmp(argv[i], "--bench-repro") == 0) bench_repro = true;
    }
    // Reproduce the FULL bench order: small, medium, large in sequence so
    // the engines accumulate prior rounds the way the actual bench does.
    if (bench_repro) {
        struct ReproT { const char* n; uint32_t s; uint32_t t; int gw; int gm; int ww; int wm; };
        ReproT all_tiers[] = {
            {"small",  1024u,   256u,  3, 10, 1, 3},
            {"medium", 4096u,  1024u,  2,  5, 1, 3},
            {"large", 12288u,  4096u,  1,  3, 1, 3},
        };
        for (auto rt : all_tiers) {
            std::printf("\n=== BENCH-REPRO : %s %u/%u ===\n", rt.n, rt.s, rt.t);
            Workload w; build_workload(w, rt.s, rt.t);
            // Canonical run on each backend.
            auto desc1 = make_desc(1u);
            auto cpu_r = run_cpu(w, desc1);
            if (metal) {
                auto m1 = run_gpu(metal.get(), w, desc1);
                std::printf("  canonical Metal: acc=%u match=%s\n",
                            m1.tx_accepted,
                            std::memcmp(cpu_r.execution_root, m1.execution_root, 32) == 0 ? "Y" : "N");
            }
            if (wgpu) {
                auto w1 = run_gpu(wgpu.get(), w, desc1);
                std::printf("  canonical WGSL : acc=%u match=%s\n",
                            w1.tx_accepted,
                            std::memcmp(cpu_r.execution_root, w1.execution_root, 32) == 0 ? "Y" : "N");
            }
            // CPU bench loop x 13.
            for (int i = 0; i < 13; ++i) {
                auto desc_n = make_desc(uint64_t(i + 1));
                auto c_n = run_cpu(w, desc_n);
                (void)c_n;
            }
            int g_total = rt.gw + rt.gm;
            int wg_total = rt.ww + rt.wm;
            if (metal) {
                std::printf("  Metal bench loop x%d:\n", g_total);
                for (int i = 0; i < g_total; ++i) {
                    auto desc_n = make_desc(uint64_t(i + 2));
                    auto m_n = run_gpu(metal.get(), w, desc_n);
                    std::printf("    iter[%d] round=%llu acc=%u\n", i,
                                (unsigned long long)desc_n.round, m_n.tx_accepted);
                }
            }
            if (wgpu) {
                std::printf("  WGSL bench loop x%d:\n", wg_total);
                for (int i = 0; i < wg_total; ++i) {
                    auto desc_n = make_desc(uint64_t(i + 2));
                    auto w_n = run_gpu(wgpu.get(), w, desc_n);
                    std::printf("    iter[%d] round=%llu acc=%u  exec=",
                                i, (unsigned long long)desc_n.round, w_n.tx_accepted);
                    for (int b = 0; b < 16; ++b) std::printf("%02x", w_n.execution_root[b]);
                    std::printf("...\n");
                }
            }
        }
        return 0;
    }
    // Repro of the benchmark mismatch: each backend runs a different round_idx
    // because of the warmup+measured mismatch in the bench. Verify that the
    // SAME (state, descriptor) gives the same root across all backends.
    for (auto t : tiers) {
        if (filter && std::strcmp(filter, t.name) != 0) continue;
        Workload w; build_workload(w, t.seed, t.tx);

        // Test 1: same descriptor (round=1, height=101) — must match.
        auto desc1 = make_desc(1u);
        std::printf("\n--- %s : SAME descriptor (round=1, height=%lu) ---\n",
                    t.name, (unsigned long)desc1.height);
        auto cpu_r1 = run_cpu(w, desc1);
        if (metal) {
            auto m_r = run_gpu(metal.get(), w, desc1);
            compare(t.name, cpu_r1, m_r, metal->device_name());
        }
        if (wgpu) {
            auto w_r = run_gpu(wgpu.get(), w, desc1);
            std::string label = std::string(wgpu->device_name()) + " (WGSL)";
            compare(t.name, cpu_r1, w_r, label.c_str());
            if (repeat) {
                // Run the same workload 3 more times on the same engine to
                // check whether subsequent rounds diverge (bench symptom).
                for (int i = 0; i < 3; ++i) {
                    auto desc_n = make_desc(uint64_t(2 + i));
                    auto cpu_n = run_cpu(w, desc_n);
                    auto wgs_n = run_gpu(wgpu.get(), w, desc_n);
                    std::printf("  repeat[%d] @ round=%llu height=%llu\n",
                                i, (unsigned long long)desc_n.round,
                                (unsigned long long)desc_n.height);
                    compare(t.name, cpu_n, wgs_n, label.c_str());
                }
            }
        }

        // Test 2: simulate the benchmark — each backend's LAST round_idx
        // depends on per-tier warmup+measured. Roots SHOULD differ because
        // heights differ. This is the source of the v0.55.2 "divergence".
        std::printf("\n--- %s : last round_idx per bench shape (bench repro) ---\n", t.name);
        // Compute last round_idx per tier per backend; mirrors xvm_benchmark.
        struct Shape { int cpu_w=3, cpu_m=10; int g_w; int g_m; int wg_w; int wg_m; };
        Shape sh{};
        if (std::strcmp(t.name, "small") == 0)  { sh.g_w = 3; sh.g_m = 10; sh.wg_w = 1; sh.wg_m = 3; }
        else if (std::strcmp(t.name, "medium") == 0) { sh.g_w = 2; sh.g_m = 5; sh.wg_w = 1; sh.wg_m = 3; }
        else if (std::strcmp(t.name, "large") == 0) { sh.g_w = 1; sh.g_m = 3; sh.wg_w = 1; sh.wg_m = 3; }
        else { sh.g_w = 3; sh.g_m = 10; sh.wg_w = 1; sh.wg_m = 3; }
        uint64_t cpu_last = uint64_t(sh.cpu_w + sh.cpu_m);    // 13
        uint64_t g_last   = uint64_t(sh.g_w   + sh.g_m);
        uint64_t wg_last  = uint64_t(sh.wg_w  + sh.wg_m);
        auto desc_cpu  = make_desc(cpu_last);
        auto desc_g    = make_desc(g_last);
        auto desc_wg   = make_desc(wg_last);
        auto cpu_rB = run_cpu(w, desc_cpu);
        std::printf("  cpu  @ round=%llu height=%llu\n",
                    (unsigned long long)cpu_last, (unsigned long long)desc_cpu.height);
        if (metal) {
            auto m_r = run_gpu(metal.get(), w, desc_g);
            std::printf("  metal @ round=%llu height=%llu\n",
                        (unsigned long long)g_last, (unsigned long long)desc_g.height);
            compare(t.name, cpu_rB, m_r, metal->device_name());
        }
        if (wgpu) {
            auto w_r = run_gpu(wgpu.get(), w, desc_wg);
            std::printf("  wgpu @ round=%llu height=%llu\n",
                        (unsigned long long)wg_last, (unsigned long long)desc_wg.height);
            std::string label = std::string(wgpu->device_name()) + " (WGSL)";
            compare(t.name, cpu_rB, w_r, label.c_str());
        }
    }
    return 0;
}
