// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_gpu_engine_test.mm — Metal-side correctness for XVMGPUEngine.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/xvm/xvm_gpu_engine.hpp"
#include "lux/xvm/xvm_cpu_reference.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace xvm::gpu;

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

void test_engine_creates()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("engine.create", engine != nullptr);
    std::printf("  engine.device: %s\n", engine->device_name());
    PASS("engine creates");
}

void test_engine_simple_transfer_matches_cpu()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("xfer.engine", engine != nullptr);

    auto desc = make_desc(1u);

    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t mint_auth[32]; fill32(mint_auth, 0xAA);
    uint8_t alice[32];  fill32(alice, 0x11);
    uint8_t bob[32];    fill32(bob, 0x22);
    uint8_t in1[32];    fill32(in1, 0x10);
    uint8_t out1[32];   fill32(out1, 0x20);

    auto h = engine->begin_round(desc);
    EXPECT("xfer.handle", h.valid());
    engine->seed_asset(h, lux_id, 1'000'000ull, 0, mint_auth, 9);
    engine->seed_utxo (h, in1, lux_id, 100, 0, alice, 0, 1);

    XvmTx tx{}; fill32(tx.tx_id, 0xC1);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
    InputBatch ib{}; std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_offset = 0; ib.input_count = 1; ib.witness_count = 1;
    OutputBatch ob{}; std::memcpy(ob.tx_id, tx.tx_id, 32);
    ob.output_offset = 0; ob.output_count = 1;
    UTXO new_utxo{};
    std::memcpy(new_utxo.utxo_id, out1, 32);
    std::memcpy(new_utxo.asset_id, lux_id, 32);
    new_utxo.amount_lo = 100;
    std::memcpy(new_utxo.owner_root, bob, 32);
    new_utxo.threshold = 1;

    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), in1, 32);

    std::vector<XvmTx>       txs = {tx};
    std::vector<InputBatch>  ibs = {ib};
    std::vector<OutputBatch> obs = {ob};
    std::vector<UTXO>        outs = {new_utxo};
    engine->push_txs(h, txs);
    engine->push_input_batches(h, ibs);
    engine->push_output_batches(h, obs);
    engine->push_inputs(h, inputs);
    engine->push_outputs(h, outs);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(lux_id, 1'000'000ull, 0, mint_auth, 9);
    state.seed_utxo (in1, lux_id, 100, 0, alice, 0, 1);
    auto cpu_r = ref::run_reference(state, desc, txs, ibs, obs, inputs, outs, {});

    EXPECT("xfer.match.accepted",  gpu_r.tx_accepted     == cpu_r.tx_accepted);
    EXPECT("xfer.match.rejected",  gpu_r.tx_rejected     == cpu_r.tx_rejected);
    EXPECT("xfer.match.consumed",  gpu_r.inputs_consumed == cpu_r.inputs_consumed);
    EXPECT("xfer.match.created",   gpu_r.outputs_created == cpu_r.outputs_created);
    EXPECT("xfer.match.utxo",      std::memcmp(gpu_r.utxo_root,      cpu_r.utxo_root,      32) == 0);
    EXPECT("xfer.match.asset",     std::memcmp(gpu_r.asset_root,     cpu_r.asset_root,     32) == 0);
    EXPECT("xfer.match.tx",        std::memcmp(gpu_r.tx_root,        cpu_r.tx_root,        32) == 0);
    EXPECT("xfer.match.exec",      std::memcmp(gpu_r.execution_root, cpu_r.execution_root, 32) == 0);
    PASS("simple transfer matches CPU");
}

void test_engine_empty_round_deterministic()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("empty.engine", engine != nullptr);

    auto desc = make_desc(1u);
    auto h1 = engine->begin_round(desc);
    auto r1 = engine->run_until_done(h1);
    engine->end_round(h1);
    auto h2 = engine->begin_round(desc);
    auto r2 = engine->run_until_done(h2);
    engine->end_round(h2);

    EXPECT("empty.same.exec", std::memcmp(r1.execution_root, r2.execution_root, 32) == 0);
    bool any_nonzero = false;
    for (auto b : r1.execution_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("empty.exec.nonzero", any_nonzero);

    auto state = ref::XVMReferenceState::empty();
    auto cpu_r = ref::run_reference(state, desc, {}, {}, {}, {}, {}, {});
    EXPECT("empty.match.cpu", std::memcmp(r1.execution_root, cpu_r.execution_root, 32) == 0);
    PASS("empty round deterministic and matches CPU");
}

void test_engine_duplicate_input_rejected()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("dup.engine", engine != nullptr);

    auto desc = make_desc(2u);
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t alice[32];  fill32(alice, 0x33);
    uint8_t in1[32];    fill32(in1, 0x40);

    auto h = engine->begin_round(desc);
    uint8_t mint_auth[32]; fill32(mint_auth, 0xAA);
    engine->seed_asset(h, lux_id, 1000, 0, mint_auth, 9);
    engine->seed_utxo (h, in1, lux_id, 50, 0, alice, 0, 1);

    XvmTx tx{}; fill32(tx.tx_id, 0xC2);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
    InputBatch ib{}; std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_offset = 0; ib.input_count = 2; ib.witness_count = 1;
    std::vector<uint8_t> inputs(64);
    std::memcpy(inputs.data(),       in1, 32);
    std::memcpy(inputs.data() + 32,  in1, 32);
    std::vector<XvmTx>      txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    engine->push_txs(h, txs);
    engine->push_input_batches(h, ibs);
    engine->push_inputs(h, inputs);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(lux_id, 1000, 0, mint_auth, 9);
    state.seed_utxo (in1, lux_id, 50, 0, alice, 0, 1);
    auto cpu_r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});

    EXPECT("dup.match.rejected", gpu_r.tx_rejected == cpu_r.tx_rejected);
    EXPECT("dup.rejected.eq.1",  gpu_r.tx_rejected == 1u);
    EXPECT("dup.match.exec",     std::memcmp(gpu_r.execution_root, cpu_r.execution_root, 32) == 0);
    PASS("duplicate input rejected");
}

void test_engine_mint_authority_enforced()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("mint.engine", engine != nullptr);

    auto desc = make_desc(3u);
    uint8_t my_asset[32]; fill32(my_asset, 0x77);
    uint8_t real_auth[32];  fill32(real_auth,  0xAA);
    uint8_t wrong_auth[32]; fill32(wrong_auth, 0xBB);

    auto h = engine->begin_round(desc);
    engine->seed_asset(h, my_asset, 0, 0, real_auth, 9);

    // Tx with a Mint op signed by wrong_auth — must be rejected with kRejectMintAuthority.
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
    engine->push_txs(h, txs);
    engine->push_asset_ops(h, ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(my_asset, 0, 0, real_auth, 9);
    auto cpu_r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);

    EXPECT("mint.match.rejected", gpu_r.tx_rejected == cpu_r.tx_rejected);
    EXPECT("mint.rejected.eq.1",  gpu_r.tx_rejected == 1u);
    EXPECT("mint.match.exec",     std::memcmp(gpu_r.execution_root, cpu_r.execution_root, 32) == 0);
    PASS("mint authority enforced");
}

void test_engine_export_creates_marker()
{
    auto engine = XVMGPUEngine::create();
    EXPECT("export.engine", engine != nullptr);

    auto desc = make_desc(4u);
    uint8_t my_asset[32]; fill32(my_asset, 0x44);
    uint8_t auth[32]; fill32(auth, 0xCC);
    uint8_t recipient[32]; fill32(recipient, 0xDD);

    auto h = engine->begin_round(desc);
    engine->seed_asset(h, my_asset, 1000, 0, auth, 9);

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
    engine->push_txs(h, txs);
    engine->push_asset_ops(h, ops);
    auto gpu_r = engine->run_until_done(h);
    engine->end_round(h);

    auto state = ref::XVMReferenceState::empty();
    state.seed_asset(my_asset, 1000, 0, auth, 9);
    auto cpu_r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);

    EXPECT("export.match.accepted", gpu_r.tx_accepted == cpu_r.tx_accepted);
    EXPECT("export.match.markers",  gpu_r.export_markers == cpu_r.export_markers);
    EXPECT("export.markers.eq.1",   gpu_r.export_markers == 1u);
    EXPECT("export.match.exec",     std::memcmp(gpu_r.execution_root, cpu_r.execution_root, 32) == 0);
    PASS("export creates atomic marker");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[xvm_gpu_engine_test] starting\n");

        test_engine_creates();
        test_engine_simple_transfer_matches_cpu();
        test_engine_empty_round_deterministic();
        test_engine_duplicate_input_rejected();
        test_engine_mint_authority_enforced();
        test_engine_export_creates_marker();

        std::printf("[xvm_gpu_engine_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
