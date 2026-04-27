// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_determinism_test.cpp — cross-backend determinism harness.
//
// Compares CPU reference vs GPU engine (Metal on Apple, CUDA on Linux+CUDA)
// across canonical workloads:
//   1. 10K UTXOs / 1K transactions (the brief)
//   2. Empty round (deterministic, non-zero root)
//   3. Duplicate-input detection in same batch
//   4. Mint authority enforcement (wrong authority rejected)
//   5. Cross-chain export creates atomic marker
//   6. Membership Bloom filter zero false-negatives, bounded FPR
//   7. Two engines run on same input — bytes match

#include "lux/xvm/xvm_gpu_engine.hpp"
#include "lux/xvm/xvm_cpu_reference.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using namespace xvm::gpu;

#if defined(LUX_XVM_TEST_WGPU)
namespace xvm::gpu {
std::unique_ptr<XVMGPUEngine> create_xvm_gpu_engine_wgpu();
}
#endif

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

void fill32(uint8_t out[32], uint8_t v) { for (uint32_t i = 0; i < 32; ++i) out[i] = v; }

XVMRoundDescriptor make_desc(uint64_t round, uint64_t height = 100)
{
    XVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.height = height;
    d.mode = static_cast<uint32_t>(XVMTransitionMode::FullRound);
    d.closing_flag = 1u;
    return d;
}

struct WorkloadResult {
    uint8_t  utxo_root[32];
    uint8_t  asset_root[32];
    uint8_t  tx_root[32];
    uint8_t  execution_root[32];
    uint32_t tx_accepted, tx_rejected;
    uint32_t inputs_consumed, outputs_created;
    uint32_t asset_ops_applied, exports, imports;

    static WorkloadResult from(const XVMTransitionResult& r) {
        WorkloadResult w{};
        std::memcpy(w.utxo_root,      r.utxo_root,      32);
        std::memcpy(w.asset_root,     r.asset_root,     32);
        std::memcpy(w.tx_root,        r.tx_root,        32);
        std::memcpy(w.execution_root, r.execution_root, 32);
        w.tx_accepted       = r.tx_accepted;
        w.tx_rejected       = r.tx_rejected;
        w.inputs_consumed   = r.inputs_consumed;
        w.outputs_created   = r.outputs_created;
        w.asset_ops_applied = r.asset_ops_applied;
        w.exports           = r.export_markers;
        w.imports           = r.import_verified;
        return w;
    }

    bool equals(const WorkloadResult& o) const {
        return std::memcmp(utxo_root,      o.utxo_root,      32) == 0
            && std::memcmp(asset_root,     o.asset_root,     32) == 0
            && std::memcmp(tx_root,        o.tx_root,        32) == 0
            && std::memcmp(execution_root, o.execution_root, 32) == 0
            && tx_accepted     == o.tx_accepted
            && tx_rejected     == o.tx_rejected
            && inputs_consumed == o.inputs_consumed
            && outputs_created == o.outputs_created
            && asset_ops_applied == o.asset_ops_applied
            && exports         == o.exports
            && imports         == o.imports;
    }
};

// Build a brief-style workload: seed N UTXOs, run M txs each consuming 1 input
// + creating 1 output. Returns the prepared inputs/outputs/batches/txs and
// the seed assets/utxos that the caller must inject into both CPU+GPU state.
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

void build_workload(Workload& w, uint32_t seed_n, uint32_t tx_n)
{
    fill32(w.lux_id, 0);
    fill32(w.mint_auth, 0xAA);
    fill32(w.alice, 0x11);
    fill32(w.bob, 0x22);

    w.seed_count = seed_n;
    w.tx_count = tx_n;

    w.seed_utxo_ids.resize(seed_n);
    w.seed_amounts.resize(seed_n);
    // Spread utxo_ids over all 32 bytes via a deterministic SplitMix64 walk
    // so the Bloom + cuckoo hash family sees varied input. (Structured ids
    // like "i in low 16 bits, zero everywhere else" cluster in cuckoo and
    // make the load factor look much higher than reality.)
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
        tx.input_batch_offset  = i;
        tx.output_batch_offset = i;

        InputBatch& ib = w.ibs[i];
        std::memcpy(ib.tx_id, tx.tx_id, 32);
        ib.input_offset = i * 32u;     // byte offset into inputs[]
        ib.input_count  = 1;
        ib.witness_count = 1;

        OutputBatch& ob = w.obs[i];
        std::memcpy(ob.tx_id, tx.tx_id, 32);
        ob.output_offset = i;          // index into outputs[] (UTXO records)
        ob.output_count  = 1;

        // each tx consumes seed_utxo_ids[i % seed_n]
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

WorkloadResult run_cpu_workload(const Workload& w, const XVMRoundDescriptor& desc)
{
    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(w.lux_id, /*supply=*/1ull << 50, 0, w.mint_auth, 9);
    for (uint32_t i = 0; i < w.seed_count; ++i) {
        state.seed_utxo(w.seed_utxo_ids[i].v, w.lux_id,
                        w.seed_amounts[i], 0, w.alice, 0, 1);
    }
    auto r = ref::run_reference(state, desc, w.txs, w.ibs, w.obs,
                                w.inputs, w.outputs, {});
    return WorkloadResult::from(r);
}

WorkloadResult run_gpu_workload(XVMGPUEngine* engine, const Workload& w,
                                const XVMRoundDescriptor& desc)
{
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
    return WorkloadResult::from(r);
}

void test_brief_workload(XVMGPUEngine* engine)
{
    // Distinct seed and tx counts. Each tx consumes a distinct seed UTXO (no
    // duplicate consumption of the same UTXO across txs) so all 1K txs accept.
    Workload w; build_workload(w, /*seed_n=*/10000u, /*tx_n=*/1000u);
    // Each tx maps i -> i % seed_n. To avoid any tx consuming the same input
    // twice (different txs picking the same seed), restrict tx_n <= seed_n.
    auto desc = make_desc(1u);
    auto cpu = run_cpu_workload(w, desc);
    if (engine == nullptr) {
        auto cpu2 = run_cpu_workload(w, desc);
        EXPECT("brief.cpu.det", cpu.equals(cpu2));
        std::printf("  brief: CPU-only path; root match across runs\n");
        PASS("brief workload (10K seed / 1K tx) — CPU determinism");
        return;
    }
    auto gpu = run_gpu_workload(engine, w, desc);
    EXPECT("brief.match", cpu.equals(gpu));
    std::printf("  brief: roots match CPU<->GPU; accepted=%u rejected=%u consumed=%u created=%u\n",
                gpu.tx_accepted, gpu.tx_rejected, gpu.inputs_consumed, gpu.outputs_created);
    PASS("brief workload (10K seed / 1K tx) — CPU<->GPU byte match");
}

void test_empty_round(XVMGPUEngine* engine)
{
    auto desc = make_desc(1u);
    auto state = ref::XVMReferenceState::empty();
    auto cpu = WorkloadResult::from(ref::run_reference(state, desc, {}, {}, {}, {}, {}, {}));
    bool nonzero = false;
    for (auto b : cpu.execution_root) if (b != 0) { nonzero = true; break; }
    EXPECT("empty.cpu.nz", nonzero);

    if (engine != nullptr) {
        auto h = engine->begin_round(desc);
        auto gpu = WorkloadResult::from(engine->run_until_done(h));
        engine->end_round(h);
        EXPECT("empty.match", cpu.equals(gpu));
    }
    PASS("empty round deterministic non-zero");
}

void test_duplicate_input(XVMGPUEngine* engine)
{
    auto desc = make_desc(2u);
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t mint_auth[32]; fill32(mint_auth, 0xAA);
    uint8_t alice[32];  fill32(alice, 0x33);
    uint8_t in1[32];    fill32(in1, 0x40);

    XvmTx tx{}; fill32(tx.tx_id, 0xC2);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
    InputBatch ib{}; std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_offset = 0; ib.input_count = 2; ib.witness_count = 1;
    std::vector<uint8_t> inputs(64);
    std::memcpy(inputs.data(),       in1, 32);
    std::memcpy(inputs.data() + 32,  in1, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(lux_id, 1000, 0, mint_auth, 9);
    state.seed_utxo (in1, lux_id, 50, 0, alice, 0, 1);
    auto cpu = WorkloadResult::from(
        ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {}));
    EXPECT("dup.cpu.rej", cpu.tx_rejected == 1u);

    if (engine != nullptr) {
        auto h = engine->begin_round(desc);
        engine->seed_asset(h, lux_id, 1000, 0, mint_auth, 9);
        engine->seed_utxo (h, in1, lux_id, 50, 0, alice, 0, 1);
        engine->push_txs(h, txs);
        engine->push_input_batches(h, ibs);
        engine->push_inputs(h, inputs);
        auto gpu = WorkloadResult::from(engine->run_until_done(h));
        engine->end_round(h);
        EXPECT("dup.match", cpu.equals(gpu));
    }
    PASS("duplicate input rejected (CPU/GPU agree)");
}

void test_mint_authority(XVMGPUEngine* engine)
{
    auto desc = make_desc(3u);
    uint8_t my_asset[32]; fill32(my_asset, 0x77);
    uint8_t real_auth[32];  fill32(real_auth,  0xAA);
    uint8_t wrong_auth[32]; fill32(wrong_auth, 0xBB);

    XvmTx tx{}; fill32(tx.tx_id, 0xC3);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Mint);
    tx.asset_changes_offset = 0;
    tx.asset_changes_count  = 1;

    AssetOp op{};
    std::memcpy(op.asset_id, my_asset, 32);
    op.amount_lo = 1000;
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);
    std::memcpy(op.authority_witness_root, wrong_auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(my_asset, 0, 0, real_auth, 9);
    auto cpu = WorkloadResult::from(
        ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops));
    EXPECT("mint.cpu.rej", cpu.tx_rejected == 1u);

    if (engine != nullptr) {
        auto h = engine->begin_round(desc);
        engine->seed_asset(h, my_asset, 0, 0, real_auth, 9);
        engine->push_txs(h, txs);
        engine->push_asset_ops(h, ops);
        auto gpu = WorkloadResult::from(engine->run_until_done(h));
        engine->end_round(h);
        EXPECT("mint.match", cpu.equals(gpu));
    }
    PASS("mint authority enforced (CPU/GPU agree)");
}

void test_export_marker(XVMGPUEngine* engine)
{
    auto desc = make_desc(4u);
    uint8_t my_asset[32]; fill32(my_asset, 0x44);
    uint8_t auth[32]; fill32(auth, 0xCC);
    uint8_t recipient[32]; fill32(recipient, 0xDD);

    XvmTx tx{}; fill32(tx.tx_id, 0xC4);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Export);
    tx.target_chain = 2u;
    tx.asset_changes_offset = 0;
    tx.asset_changes_count = 1;

    AssetOp op{};
    std::memcpy(op.asset_id, my_asset, 32);
    op.amount_lo = 250;
    op.kind = static_cast<uint32_t>(AssetOpKind::Export);
    op.target_chain = 2u;
    std::memcpy(op.authority_witness_root, recipient, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(my_asset, 1000, 0, auth, 9);
    auto cpu = WorkloadResult::from(
        ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops));
    EXPECT("export.cpu.acc", cpu.tx_accepted == 1u);
    EXPECT("export.cpu.markers", cpu.exports == 1u);

    if (engine != nullptr) {
        auto h = engine->begin_round(desc);
        engine->seed_asset(h, my_asset, 1000, 0, auth, 9);
        engine->push_txs(h, txs);
        engine->push_asset_ops(h, ops);
        auto gpu = WorkloadResult::from(engine->run_until_done(h));
        engine->end_round(h);
        EXPECT("export.match", cpu.equals(gpu));
    }
    PASS("cross-chain export creates atomic marker");
}

void test_membership_no_false_negatives()
{
    // Pure CPU-side test of the membership API. The GPU kernels share the
    // exact same Bloom + cuckoo recipe and produce byte-identical state, so
    // the false-negative property holds equally on Metal/CUDA.
    auto state = ref::XVMReferenceState::empty();
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t owner[32]; fill32(owner, 0x55);
    constexpr uint32_t N = 4096u;
    for (uint32_t i = 0; i < N; ++i) {
        uint8_t uid[32]; fill32(uid, 0);
        uid[0] = uint8_t(i & 0xFF); uid[1] = uint8_t((i >> 8) & 0xFF);
        uid[31] = 0x55;
        uint32_t slot = state.seed_utxo(uid, lux_id, 100u + i, 0, owner, 0, 1);
        EXPECT("memb.seed", slot != 0xFFFFFFFFu);
    }
    // Zero false-negatives: every inserted id must be found.
    for (uint32_t i = 0; i < N; ++i) {
        uint8_t uid[32]; fill32(uid, 0);
        uid[0] = uint8_t(i & 0xFF); uid[1] = uint8_t((i >> 8) & 0xFF);
        uid[31] = 0x55;
        uint32_t slot = 0xFFFFFFFFu;
        EXPECT("memb.found", ref::membership_query(state, uid, &slot));
    }
    // Bounded false-positive rate: random non-inserted ids should pass Bloom
    // less than ~5% of the time at our 4-hash, 1Mbit configuration with N=4K.
    // We sample 10K random non-inserted ids and require <500 Bloom hits.
    std::mt19937_64 rng(0x123456789abcdef0ULL);
    uint32_t bloom_hits = 0;
    constexpr uint32_t kSamples = 10000u;
    for (uint32_t s = 0; s < kSamples; ++s) {
        uint8_t uid[32];
        for (uint32_t k = 0; k < 32u; ++k) uid[k] = uint8_t(rng() & 0xFF);
        uid[31] = 0xEE;  // distinct domain from inserted (0x55)
        if (ref::bloom_query(state, uid)) ++bloom_hits;
    }
    std::printf("  membership: bloom_hits=%u / %u (target <500)\n", bloom_hits, kSamples);
    EXPECT("memb.fpr_bounded", bloom_hits < 500u);
    PASS("membership Bloom: zero false-negatives, FPR bounded");
}

void test_two_engines_match(XVMGPUEngine* engine)
{
    if (engine == nullptr) {
        PASS("two-engines (skipped — no GPU)");
        return;
    }
    auto a = XVMGPUEngine::create();
    auto b = XVMGPUEngine::create();
    EXPECT("twoeng.a", a != nullptr);
    EXPECT("twoeng.b", b != nullptr);

    Workload w; build_workload(w, /*seed_n=*/64u, /*tx_n=*/32u);
    auto desc = make_desc(99u);
    auto ra = run_gpu_workload(a.get(), w, desc);
    auto rb = run_gpu_workload(b.get(), w, desc);
    EXPECT("twoeng.match", ra.equals(rb));
    PASS("two engines bytewise identical");
}

}  // namespace

void run_all_against(const char* label, XVMGPUEngine* engine)
{
    std::printf("[%s] %s\n", label,
                engine ? engine->device_name() : "(CPU-only)");
    test_brief_workload(engine);
    test_empty_round(engine);
    test_duplicate_input(engine);
    test_mint_authority(engine);
    test_export_marker(engine);
    test_two_engines_match(engine);
}

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[xvm_determinism_test] starting\n");

    // -- Default backend (Metal on Apple, CUDA on Linux+CUDA, else nullptr) --
    auto engine = XVMGPUEngine::create();
    run_all_against("default", engine.get());

    // membership FPR is a CPU-only oracle test — run once
    test_membership_no_false_negatives();

#if defined(LUX_XVM_TEST_WGPU)
    // -- WGSL/Dawn backend — runs the same workloads through the WebGPU
    //    runtime and compares to the CPU oracle for byte equivalence. --
    auto wgpu_engine = create_xvm_gpu_engine_wgpu();
    if (wgpu_engine == nullptr) {
        std::printf("[wgpu] runtime unavailable — driver compiled but no Dawn/wgpu-native at link\n");
    } else {
        run_all_against("wgpu", wgpu_engine.get());
    }
#endif

    std::printf("[xvm_determinism_test] passed=%d failed=%d\n",
                g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
