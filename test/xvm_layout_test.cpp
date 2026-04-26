// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file xvm_layout_test.cpp
/// XVM v0.54 — layout invariants for cross-backend determinism.
///
/// Validates struct sizes, alignment, member offsets, and basic CPU-reference
/// determinism on a small canonical workload. Anything that drifts in the
/// host header without a parallel update in the GPU kernels would break
/// CPU/Metal/CUDA equivalence — these checks are the first line of defense.

#include "lux/xvm/xvm_gpu_layout.hpp"
#include "lux/xvm/xvm_cpu_reference.hpp"

#include <cstddef>
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

void test_utxo_layout()
{
    EXPECT("UTXO.size",  sizeof(UTXO) == 144);
    EXPECT("UTXO.align", alignof(UTXO) == 16);
    EXPECT("UTXO.utxo_id.off",        offsetof(UTXO, utxo_id)          == 0);
    EXPECT("UTXO.asset_id.off",       offsetof(UTXO, asset_id)         == 32);
    EXPECT("UTXO.amount_lo.off",      offsetof(UTXO, amount_lo)        == 64);
    EXPECT("UTXO.amount_hi.off",      offsetof(UTXO, amount_hi)        == 72);
    EXPECT("UTXO.owner_root.off",     offsetof(UTXO, owner_root)       == 80);
    EXPECT("UTXO.locktime.off",       offsetof(UTXO, locktime)         == 112);
    EXPECT("UTXO.threshold.off",      offsetof(UTXO, threshold)        == 120);
    EXPECT("UTXO.status.off",         offsetof(UTXO, status)           == 124);
    EXPECT("UTXO.addresses_offset.off", offsetof(UTXO, addresses_offset) == 128);
    EXPECT("UTXO.addresses_count.off", offsetof(UTXO, addresses_count) == 132);
    PASS("UTXO layout");
}

void test_input_batch_layout()
{
    EXPECT("InputBatch.size",  sizeof(InputBatch) == 64);
    EXPECT("InputBatch.align", alignof(InputBatch) == 16);
    EXPECT("InputBatch.tx_id.off",          offsetof(InputBatch, tx_id)          == 0);
    EXPECT("InputBatch.input_offset.off",   offsetof(InputBatch, input_offset)   == 32);
    EXPECT("InputBatch.input_count.off",    offsetof(InputBatch, input_count)    == 36);
    EXPECT("InputBatch.witness_offset.off", offsetof(InputBatch, witness_offset) == 40);
    EXPECT("InputBatch.witness_count.off",  offsetof(InputBatch, witness_count)  == 44);
    PASS("InputBatch layout");
}

void test_output_batch_layout()
{
    EXPECT("OutputBatch.size",  sizeof(OutputBatch) == 64);
    EXPECT("OutputBatch.align", alignof(OutputBatch) == 16);
    EXPECT("OutputBatch.tx_id.off",         offsetof(OutputBatch, tx_id)         == 0);
    EXPECT("OutputBatch.output_offset.off", offsetof(OutputBatch, output_offset) == 32);
    EXPECT("OutputBatch.output_count.off",  offsetof(OutputBatch, output_count)  == 36);
    PASS("OutputBatch layout");
}

void test_asset_layout()
{
    EXPECT("Asset.size",  sizeof(Asset) == 112);
    EXPECT("Asset.align", alignof(Asset) == 16);
    EXPECT("Asset.asset_id.off",            offsetof(Asset, asset_id)            == 0);
    EXPECT("Asset.total_supply_lo.off",     offsetof(Asset, total_supply_lo)     == 32);
    EXPECT("Asset.total_supply_hi.off",     offsetof(Asset, total_supply_hi)     == 40);
    EXPECT("Asset.mint_authority_root.off", offsetof(Asset, mint_authority_root) == 48);
    EXPECT("Asset.freeze_flag.off",         offsetof(Asset, freeze_flag)         == 80);
    EXPECT("Asset.denomination.off",        offsetof(Asset, denomination)        == 84);
    EXPECT("Asset.name_offset.off",         offsetof(Asset, name_offset)         == 88);
    EXPECT("Asset.name_length.off",         offsetof(Asset, name_length)         == 92);
    EXPECT("Asset.occupied.off",            offsetof(Asset, occupied)            == 96);
    PASS("Asset layout");
}

void test_xvm_tx_layout()
{
    EXPECT("XvmTx.size",  sizeof(XvmTx) == 112);
    EXPECT("XvmTx.align", alignof(XvmTx) == 16);
    EXPECT("XvmTx.tx_id.off",                offsetof(XvmTx, tx_id)                == 0);
    EXPECT("XvmTx.kind.off",                 offsetof(XvmTx, kind)                 == 32);
    EXPECT("XvmTx.input_batch_offset.off",   offsetof(XvmTx, input_batch_offset)   == 36);
    EXPECT("XvmTx.output_batch_offset.off",  offsetof(XvmTx, output_batch_offset)  == 40);
    EXPECT("XvmTx.asset_changes_offset.off", offsetof(XvmTx, asset_changes_offset) == 44);
    EXPECT("XvmTx.asset_changes_count.off",  offsetof(XvmTx, asset_changes_count)  == 48);
    EXPECT("XvmTx.target_chain.off",         offsetof(XvmTx, target_chain)         == 52);
    EXPECT("XvmTx.status.off",               offsetof(XvmTx, status)               == 56);
    EXPECT("XvmTx.reject_reason.off",        offsetof(XvmTx, reject_reason)        == 60);
    EXPECT("XvmTx.proof_digest.off",         offsetof(XvmTx, proof_digest)         == 64);
    PASS("XvmTx layout");
}

void test_asset_op_layout()
{
    EXPECT("AssetOp.size",  sizeof(AssetOp) == 112);
    EXPECT("AssetOp.align", alignof(AssetOp) == 16);
    EXPECT("AssetOp.asset_id.off",               offsetof(AssetOp, asset_id)               == 0);
    EXPECT("AssetOp.amount_lo.off",              offsetof(AssetOp, amount_lo)              == 32);
    EXPECT("AssetOp.amount_hi.off",              offsetof(AssetOp, amount_hi)              == 40);
    EXPECT("AssetOp.authority_witness_root.off", offsetof(AssetOp, authority_witness_root) == 48);
    EXPECT("AssetOp.kind.off",                   offsetof(AssetOp, kind)                   == 80);
    EXPECT("AssetOp.target_chain.off",           offsetof(AssetOp, target_chain)           == 84);
    PASS("AssetOp layout");
}

void test_export_marker_layout()
{
    EXPECT("AtomicExportMarker.size",  sizeof(AtomicExportMarker) == 144);
    EXPECT("AtomicExportMarker.align", alignof(AtomicExportMarker) == 16);
    EXPECT("AEM.marker_id.off",      offsetof(AtomicExportMarker, marker_id)      == 0);
    EXPECT("AEM.asset_id.off",       offsetof(AtomicExportMarker, asset_id)       == 32);
    EXPECT("AEM.amount_lo.off",      offsetof(AtomicExportMarker, amount_lo)      == 64);
    EXPECT("AEM.amount_hi.off",      offsetof(AtomicExportMarker, amount_hi)      == 72);
    EXPECT("AEM.source_chain.off",   offsetof(AtomicExportMarker, source_chain)   == 80);
    EXPECT("AEM.target_chain.off",   offsetof(AtomicExportMarker, target_chain)   == 84);
    EXPECT("AEM.status.off",         offsetof(AtomicExportMarker, status)         == 88);
    EXPECT("AEM.occupied.off",       offsetof(AtomicExportMarker, occupied)       == 92);
    EXPECT("AEM.recipient_root.off", offsetof(AtomicExportMarker, recipient_root) == 96);
    PASS("AtomicExportMarker layout");
}

void test_round_descriptor_layout()
{
    EXPECT("XVMRoundDescriptor.size",  sizeof(XVMRoundDescriptor) == 112);
    EXPECT("XVMRoundDescriptor.align", alignof(XVMRoundDescriptor) == 16);
    EXPECT("Desc.chain_id.off",            offsetof(XVMRoundDescriptor, chain_id)            == 0);
    EXPECT("Desc.round.off",               offsetof(XVMRoundDescriptor, round)               == 8);
    EXPECT("Desc.height.off",              offsetof(XVMRoundDescriptor, height)              == 24);
    EXPECT("Desc.mode.off",                offsetof(XVMRoundDescriptor, mode)                == 32);
    EXPECT("Desc.tx_count.off",            offsetof(XVMRoundDescriptor, tx_count)            == 36);
    EXPECT("Desc.parent_execution_root.off", offsetof(XVMRoundDescriptor, parent_execution_root) == 64);
    PASS("XVMRoundDescriptor layout");
}

void test_transition_result_layout()
{
    EXPECT("XVMTransitionResult.size",  sizeof(XVMTransitionResult) == 208);
    EXPECT("XVMTransitionResult.align", alignof(XVMTransitionResult) == 16);
    EXPECT("Result.utxo_root.off",      offsetof(XVMTransitionResult, utxo_root)      == 80);
    EXPECT("Result.asset_root.off",     offsetof(XVMTransitionResult, asset_root)     == 112);
    EXPECT("Result.tx_root.off",        offsetof(XVMTransitionResult, tx_root)        == 144);
    EXPECT("Result.execution_root.off", offsetof(XVMTransitionResult, execution_root) == 176);
    PASS("XVMTransitionResult layout");
}

// -----------------------------------------------------------------------------
// CPU reference behaviour — small canonical workloads
// -----------------------------------------------------------------------------

XVMRoundDescriptor make_desc(uint64_t round, uint64_t height = 100,
                             XVMTransitionMode mode = XVMTransitionMode::FullRound)
{
    XVMRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 1700000000000000000ULL;
    d.height = height;
    d.mode = static_cast<uint32_t>(mode);
    d.closing_flag = 1u;
    return d;
}

void fill32(uint8_t out[32], uint8_t v) { for (uint32_t i = 0; i < 32; ++i) out[i] = v; }

void test_cpu_reference_determinism()
{
    auto desc = make_desc(1u);

    auto state1 = ref::XVMReferenceState::empty();
    auto state2 = ref::XVMReferenceState::empty();

    auto r1 = ref::run_reference(state1, desc, {}, {}, {}, {}, {}, {});
    auto r2 = ref::run_reference(state2, desc, {}, {}, {}, {}, {}, {});

    EXPECT("ref.det.status",   r1.status == 1u);
    EXPECT("ref.det.utxo",     std::memcmp(r1.utxo_root,      r2.utxo_root,      32) == 0);
    EXPECT("ref.det.asset",    std::memcmp(r1.asset_root,     r2.asset_root,     32) == 0);
    EXPECT("ref.det.tx",       std::memcmp(r1.tx_root,        r2.tx_root,        32) == 0);
    EXPECT("ref.det.exec",     std::memcmp(r1.execution_root, r2.execution_root, 32) == 0);

    // Empty round must produce non-zero execution_root (composes parent || ...
    // through keccak; even an all-zero pre-image yields a nontrivial digest).
    bool any_nonzero = false;
    for (auto b : r1.execution_root) if (b != 0) { any_nonzero = true; break; }
    EXPECT("ref.det.exec.nonzero", any_nonzero);
    PASS("CPU reference determinism (empty round)");
}

void test_cpu_reference_simple_transfer()
{
    auto desc = make_desc(2u);
    auto state = ref::XVMReferenceState::empty();

    // Seed an asset (LUX, asset_id all-zero).
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t mint_auth[32]; fill32(mint_auth, 0xAA);
    state.seed_asset(lux_id, /*supply_lo=*/1'000'000'000ull, 0, mint_auth, 9);

    // Seed a starting UTXO owned by Alice.
    uint8_t alice_owner[32]; fill32(alice_owner, 0x11);
    uint8_t bob_owner[32];   fill32(bob_owner,   0x22);
    uint8_t utxo_in_id[32];  fill32(utxo_in_id,  0x10);
    state.seed_utxo(utxo_in_id, lux_id, /*amount_lo=*/100, 0, alice_owner,
                    /*locktime=*/0, /*threshold=*/1);

    // Tx that consumes utxo_in and creates a new utxo for Bob.
    XvmTx tx{};
    fill32(tx.tx_id, 0xC1);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
    tx.input_batch_offset = 0;
    tx.output_batch_offset = 0;

    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_offset = 0; ib.input_count = 1;
    ib.witness_offset = 0; ib.witness_count = 1;  // any witness presence satisfies the threshold predicate

    OutputBatch ob{};
    std::memcpy(ob.tx_id, tx.tx_id, 32);
    ob.output_offset = 0; ob.output_count = 1;

    UTXO new_utxo{};
    fill32(new_utxo.utxo_id, 0x20);
    std::memcpy(new_utxo.asset_id, lux_id, 32);
    new_utxo.amount_lo = 100;
    std::memcpy(new_utxo.owner_root, bob_owner, 32);
    new_utxo.threshold = 1;

    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), utxo_in_id, 32);

    std::vector<XvmTx> txs       = {tx};
    std::vector<InputBatch> ibs  = {ib};
    std::vector<OutputBatch> obs = {ob};
    std::vector<UTXO> outputs    = {new_utxo};

    auto r = ref::run_reference(state, desc, txs, ibs, obs, inputs, outputs, {});

    EXPECT("xfer.status",    r.status == 1u);
    EXPECT("xfer.accepted",  r.tx_accepted == 1u);
    EXPECT("xfer.rejected",  r.tx_rejected == 0u);
    EXPECT("xfer.consumed",  r.inputs_consumed == 1u);
    EXPECT("xfer.created",   r.outputs_created == 1u);

    // Determinism: re-run on a fresh state and require byte-equal roots.
    auto state2 = ref::XVMReferenceState::empty();
    state2.seed_asset(lux_id, 1'000'000'000ull, 0, mint_auth, 9);
    state2.seed_utxo(utxo_in_id, lux_id, 100, 0, alice_owner, 0, 1);
    auto r2 = ref::run_reference(state2, desc, txs, ibs, obs, inputs, outputs, {});
    EXPECT("xfer.det.utxo", std::memcmp(r.utxo_root,      r2.utxo_root,      32) == 0);
    EXPECT("xfer.det.exec", std::memcmp(r.execution_root, r2.execution_root, 32) == 0);

    PASS("CPU reference simple transfer");
}

void test_cpu_reference_duplicate_input_rejected()
{
    auto desc = make_desc(3u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t mint_auth[32]; fill32(mint_auth, 0xAA);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(lux_id, 1'000'000'000ull, 0, mint_auth, 9);

    uint8_t in1[32]; fill32(in1, 0x30);
    state.seed_utxo(in1, lux_id, 50, 0, alice, 0, 1);

    XvmTx tx{}; fill32(tx.tx_id, 0xC2);
    tx.kind = static_cast<uint32_t>(XvmTxKind::Transfer);
    InputBatch ib{}; std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_offset = 0; ib.input_count = 2; ib.witness_count = 1;

    // Two copies of the same input.
    std::vector<uint8_t> inputs(64);
    std::memcpy(inputs.data(),       in1, 32);
    std::memcpy(inputs.data() + 32,  in1, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};

    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("dup.rejected", r.tx_rejected == 1u);
    EXPECT("dup.accepted", r.tx_accepted == 0u);
    PASS("Duplicate input rejected within batch");
}

void test_cpu_reference_membership_no_false_negative()
{
    auto state = ref::XVMReferenceState::empty();
    uint8_t lux_id[32]; fill32(lux_id, 0);
    uint8_t owner[32]; fill32(owner, 0x33);

    // Insert 256 distinct utxos.
    for (uint32_t i = 0; i < 256u; ++i) {
        uint8_t uid[32]; fill32(uid, 0); uid[0] = uint8_t(i & 0xFF); uid[1] = uint8_t(i >> 8);
        uint32_t slot = state.seed_utxo(uid, lux_id, 100u + i, 0, owner, 0, 1);
        EXPECT("memb.seed", slot != 0xFFFFFFFFu);
    }
    // Every inserted id must hit Bloom AND cuckoo.
    for (uint32_t i = 0; i < 256u; ++i) {
        uint8_t uid[32]; fill32(uid, 0); uid[0] = uint8_t(i & 0xFF); uid[1] = uint8_t(i >> 8);
        EXPECT("memb.bloom",   ref::bloom_query(state, uid));
        uint32_t slot = 0xFFFFFFFFu;
        EXPECT("memb.cuckoo", ref::membership_query(state, uid, &slot));
    }
    // Random non-inserted id should not produce a cuckoo hit (Bloom may
    // false-positive but cuckoo is exact).
    uint8_t miss[32]; fill32(miss, 0xEE);
    uint32_t miss_slot = 0xFFFFFFFFu;
    EXPECT("memb.no.fn", !ref::membership_query(state, miss, &miss_slot));
    PASS("Membership zero false-negatives");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[xvm_layout_test] starting\n");

    test_utxo_layout();
    test_input_batch_layout();
    test_output_batch_layout();
    test_asset_layout();
    test_xvm_tx_layout();
    test_asset_op_layout();
    test_export_marker_layout();
    test_round_descriptor_layout();
    test_transition_result_layout();

    test_cpu_reference_determinism();
    test_cpu_reference_simple_transfer();
    test_cpu_reference_duplicate_input_rejected();
    test_cpu_reference_membership_no_false_negative();

    std::printf("[xvm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
