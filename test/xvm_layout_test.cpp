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

void test_membership_insert_remove_idempotent()
{
    auto state = ref::XVMReferenceState::empty();
    uint8_t uid[32]; fill32(uid, 0xA5);
    ref::membership_insert(state, uid, /*slot_index=*/7);
    uint32_t slot = 0u;
    EXPECT("rm.before", ref::membership_query(state, uid, &slot));
    EXPECT("rm.slot",   slot == 7u);

    ref::membership_remove(state, uid);
    EXPECT("rm.after", !ref::membership_query(state, uid, &slot));

    // Bloom is monotonic — bloom_query may still report true (false positive).
    // That's the acceptable Bloom contract; we only care cuckoo is exact.
    PASS("membership_insert_remove_idempotent");
}

// =============================================================================
// Asset op every kind: Mint / Burn / Transfer / Export / Import
// =============================================================================

XvmTx make_tx(uint8_t fill, XvmTxKind kind = XvmTxKind::Transfer)
{
    XvmTx tx{};
    fill32(tx.tx_id, fill);
    tx.kind = static_cast<uint32_t>(kind);
    return tx;
}

void test_asset_mint_with_authority_succeeds()
{
    auto desc = make_desc(10u);
    auto state = ref::XVMReferenceState::empty();

    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, /*supply_lo=*/0, 0, auth, 9);

    XvmTx tx = make_tx(0xC1, XvmTxKind::Mint);
    tx.asset_changes_offset = 0;
    tx.asset_changes_count = 1;

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 1000;
    std::memcpy(op.authority_witness_root, auth, 32);
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("mint.ok.applied", r.asset_ops_applied == 1u);
    EXPECT("mint.ok.minted",  r.total_minted_lo == 1000u);
    EXPECT("mint.ok.accepted",r.tx_accepted == 1u);
    PASS("asset_mint_with_authority_succeeds");
}

void test_asset_mint_wrong_authority_rejects()
{
    auto desc = make_desc(11u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t bad[32]; fill32(bad, 0x99);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xC2, XvmTxKind::Mint);
    tx.asset_changes_count = 1;

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 1000;
    std::memcpy(op.authority_witness_root, bad, 32); // wrong
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("mint.bad.applied", r.asset_ops_applied == 0u);
    EXPECT("mint.bad.rejected", r.tx_rejected == 1u);
    PASS("asset_mint_wrong_authority_rejects");
}

void test_asset_mint_implicit_create_for_unknown_asset()
{
    auto desc = make_desc(12u);
    auto state = ref::XVMReferenceState::empty();

    uint8_t asset_id[32]; fill32(asset_id, 0xCD);
    uint8_t auth[32]; fill32(auth, 0x77);

    XvmTx tx = make_tx(0xC3, XvmTxKind::Mint);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 500;
    std::memcpy(op.authority_witness_root, auth, 32);
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("imp.applied",  r.asset_ops_applied == 1u);
    EXPECT("imp.minted",   r.total_minted_lo == 500u);
    PASS("asset_mint_implicit_create_for_unknown_asset");
}

void test_asset_burn_succeeds_with_supply()
{
    auto desc = make_desc(13u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, /*supply_lo=*/10000, 0, auth, 9);

    XvmTx tx = make_tx(0xC4, XvmTxKind::Burn);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 200;
    op.kind = static_cast<uint32_t>(AssetOpKind::Burn);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("burn.applied", r.asset_ops_applied == 1u);
    EXPECT("burn.burned",  r.total_burned_lo == 200u);
    PASS("asset_burn_succeeds_with_supply");
}

void test_asset_burn_overflow_rejects()
{
    auto desc = make_desc(14u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 100, 0, auth, 9);

    XvmTx tx = make_tx(0xC5, XvmTxKind::Burn);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 999; // > 100 supply
    op.kind = static_cast<uint32_t>(AssetOpKind::Burn);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("burn.over.applied", r.asset_ops_applied == 0u);
    EXPECT("burn.over.rejected", r.tx_rejected == 1u);
    PASS("asset_burn_overflow_rejects");
}

void test_asset_transfer_op_counts_only()
{
    auto desc = make_desc(15u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 5000, 0, auth, 9);

    XvmTx tx = make_tx(0xC6, XvmTxKind::Transfer);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 100;
    op.kind = static_cast<uint32_t>(AssetOpKind::Transfer);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("xfer.asset.applied", r.asset_ops_applied == 1u);
    EXPECT("xfer.asset.minted", r.total_minted_lo == 0u);
    EXPECT("xfer.asset.burned", r.total_burned_lo == 0u);
    PASS("asset_transfer_op_counts_only");
}

void test_asset_export_marker_emitted()
{
    auto desc = make_desc(16u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, /*supply_lo=*/10000, 0, auth, 9);

    XvmTx tx = make_tx(0xC7, XvmTxKind::Export);
    tx.asset_changes_count = 1;
    tx.target_chain = 2u; // some other chain id
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 1000;
    std::memcpy(op.authority_witness_root, auth, 32);
    op.kind = static_cast<uint32_t>(AssetOpKind::Export);
    op.target_chain = 2u;

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("exp.applied", r.asset_ops_applied == 1u);
    EXPECT("exp.markers", r.export_markers == 1u);
    PASS("asset_export_marker_emitted");
}

void test_asset_export_overflow_rejects()
{
    auto desc = make_desc(17u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 100, 0, auth, 9);

    XvmTx tx = make_tx(0xC8, XvmTxKind::Export);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 999;   // > supply
    op.kind = static_cast<uint32_t>(AssetOpKind::Export);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("exp.over.applied", r.asset_ops_applied == 0u);
    EXPECT("exp.over.rejected", r.tx_rejected == 1u);
    PASS("asset_export_overflow_rejects");
}

void test_asset_import_with_marker_accepted()
{
    auto desc = make_desc(18u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, /*supply_lo=*/10000, 0, auth, 9);

    // Round 1: Export creates a marker
    XvmTx tx_e = make_tx(0xE1, XvmTxKind::Export);
    tx_e.asset_changes_count = 1;
    tx_e.target_chain = 2u;
    AssetOp op_e{};
    std::memcpy(op_e.asset_id, asset_id, 32);
    op_e.amount_lo = 500;
    std::memcpy(op_e.authority_witness_root, auth, 32);
    op_e.kind = static_cast<uint32_t>(AssetOpKind::Export);
    op_e.target_chain = 2u;

    std::vector<XvmTx> txs1 = {tx_e};
    std::vector<AssetOp> ops1 = {op_e};
    auto r1 = ref::run_reference(state, desc, txs1, {}, {}, {}, {}, ops1);
    EXPECT("imp.exp.markers", r1.export_markers == 1u);

    // Compute the marker_id the same way the implementation does
    // (keccak(tx_id || target_chain || amount_lo || amount_hi)).
    // The actual marker_id is in state.export_markers[i].marker_id.
    uint8_t marker_id[32];
    bool found = false;
    for (auto& m : state.export_markers) {
        if (m.target_chain == 2u && m.amount_lo == 500u) {
            std::memcpy(marker_id, m.marker_id, 32);
            found = true;
            break;
        }
    }
    EXPECT("imp.marker.found", found);

    // Round 2: Import using the marker_id
    auto desc2 = make_desc(19u);
    XvmTx tx_i = make_tx(0xE2, XvmTxKind::Import);
    tx_i.asset_changes_count = 1;
    std::memcpy(tx_i.proof_digest, marker_id, 32);
    AssetOp op_i{};
    std::memcpy(op_i.asset_id, asset_id, 32);
    op_i.amount_lo = 500;
    op_i.kind = static_cast<uint32_t>(AssetOpKind::Import);

    std::vector<XvmTx> txs2 = {tx_i};
    std::vector<AssetOp> ops2 = {op_i};
    auto r2 = ref::run_reference(state, desc2, txs2, {}, {}, {}, {}, ops2);
    EXPECT("imp.applied", r2.asset_ops_applied == 1u);
    EXPECT("imp.verified", r2.import_verified == 1u);

    // Re-import the same marker should fail (consumed).
    auto desc3 = make_desc(20u);
    std::vector<XvmTx> txs3 = {tx_i};
    auto r3 = ref::run_reference(state, desc3, txs3, {}, {}, {}, {}, ops2);
    EXPECT("imp.replay.rej", r3.tx_rejected == 1u);
    PASS("asset_import_with_marker_accepted");
}

void test_asset_import_no_marker_rejects()
{
    auto desc = make_desc(21u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xBE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xC9, XvmTxKind::Import);
    tx.asset_changes_count = 1;
    fill32(tx.proof_digest, 0xFE); // doesn't match any marker

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 100;
    op.kind = static_cast<uint32_t>(AssetOpKind::Import);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("imp.no.applied", r.asset_ops_applied == 0u);
    EXPECT("imp.no.rejected", r.tx_rejected == 1u);
    PASS("asset_import_no_marker_rejects");
}

void test_asset_burn_unknown_asset_rejects()
{
    auto desc = make_desc(22u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xCC); // not seeded

    XvmTx tx = make_tx(0xCA, XvmTxKind::Burn);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 100;
    op.kind = static_cast<uint32_t>(AssetOpKind::Burn);

    std::vector<XvmTx> txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("burn.unk.applied", r.asset_ops_applied == 0u);
    EXPECT("burn.unk.rejected", r.tx_rejected == 1u);
    PASS("asset_burn_unknown_asset_rejects");
}

// =============================================================================
// Input-check edge cases
// =============================================================================

void test_input_missing_utxo_rejected()
{
    auto desc = make_desc(30u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    XvmTx tx = make_tx(0xD1);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1;
    ib.witness_count = 1;

    uint8_t bogus[32]; fill32(bogus, 0xBA);
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), bogus, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("miss.rejected", r.tx_rejected == 1u);
    PASS("input_missing_utxo_rejected");
}

void test_input_locktime_in_future_rejected()
{
    auto desc = make_desc(31u);
    auto state = ref::XVMReferenceState::empty();
    state.height = 100;
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    uint8_t uid[32]; fill32(uid, 0x40);
    state.seed_utxo(uid, asset_id, 50, 0, alice, /*locktime=*/200, /*threshold=*/1);

    XvmTx tx = make_tx(0xD2);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;

    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);
    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("lt.rejected", r.tx_rejected == 1u);
    PASS("input_locktime_in_future_rejected");
}

void test_input_threshold_unmet_rejected()
{
    auto desc = make_desc(32u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);
    uint8_t uid[32]; fill32(uid, 0x41);
    state.seed_utxo(uid, asset_id, 50, 0, alice, 0, /*threshold=*/1);

    XvmTx tx = make_tx(0xD3);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 0;  // missing witness

    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);
    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("thr.rejected", r.tx_rejected == 1u);
    PASS("input_threshold_unmet_rejected");
}

void test_input_double_spend_within_round_rejected()
{
    auto desc = make_desc(33u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);
    uint8_t uid[32]; fill32(uid, 0x42);
    state.seed_utxo(uid, asset_id, 50, 0, alice, 0, 1);

    XvmTx tx_a = make_tx(0xA1);
    XvmTx tx_b = make_tx(0xA2);
    InputBatch ib_a{}, ib_b{};
    std::memcpy(ib_a.tx_id, tx_a.tx_id, 32);
    std::memcpy(ib_b.tx_id, tx_b.tx_id, 32);
    ib_a.input_offset = 0;  ib_a.input_count = 1; ib_a.witness_count = 1;
    ib_b.input_offset = 32; ib_b.input_count = 1; ib_b.witness_count = 1;

    tx_a.input_batch_offset = 0;
    tx_b.input_batch_offset = 1;

    std::vector<uint8_t> inputs(64);
    std::memcpy(inputs.data(),       uid, 32);
    std::memcpy(inputs.data() + 32,  uid, 32); // double-spend across batches

    std::vector<XvmTx> txs = {tx_a, tx_b};
    std::vector<InputBatch> ibs = {ib_a, ib_b};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    // First tx accepts, second sees the cuckoo-removed (or spent) state and rejects.
    EXPECT("ds.accepted.1", r.tx_accepted == 1u);
    EXPECT("ds.rejected.1", r.tx_rejected == 1u);
    PASS("input_double_spend_within_round_rejected");
}

// =============================================================================
// Edge cases — branch coverage for cpu reference
// =============================================================================

// Cover input_count > 1 with no-duplicate path through tx_inputs_have_duplicates.
void test_tx_no_duplicates_multi_input()
{
    auto desc = make_desc(40u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    uint8_t uid_a[32]; fill32(uid_a, 0x60);
    uint8_t uid_b[32]; fill32(uid_b, 0x61);
    uint8_t uid_c[32]; fill32(uid_c, 0x62);
    state.seed_utxo(uid_a, asset_id, 50, 0, alice, 0, 1);
    state.seed_utxo(uid_b, asset_id, 50, 0, alice, 0, 1);
    state.seed_utxo(uid_c, asset_id, 50, 0, alice, 0, 1);

    XvmTx tx = make_tx(0xD4);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 3; ib.witness_count = 1;

    std::vector<uint8_t> inputs(96);
    std::memcpy(inputs.data(),       uid_a, 32);
    std::memcpy(inputs.data() + 32,  uid_b, 32);
    std::memcpy(inputs.data() + 64,  uid_c, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("multi.in.acc", r.tx_accepted == 1u);
    EXPECT("multi.in.con", r.inputs_consumed == 3u);
    PASS("tx_no_duplicates_multi_input");
}

// Cover the `kRejectMissingInput` path where Bloom passes but cuckoo misses.
// We inject a utxo_id whose hash collides with a present id's Bloom bits but
// is not actually inserted into the cuckoo arena — bloom_test() returns true,
// cuckoo_query() returns false.
void test_input_bloom_pass_cuckoo_miss()
{
    auto desc = make_desc(41u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    // Seed many UTXOs to densify Bloom; then ask about a non-inserted id with
    // bits that happen to be set. Practically, we just hand-paint the entire
    // bloom filter to all-ones — bloom_test always passes, cuckoo doesn't.
    uint8_t uid_present[32]; fill32(uid_present, 0x70);
    state.seed_utxo(uid_present, asset_id, 50, 0, alice, 0, 1);
    std::fill(state.bloom_bits.begin(), state.bloom_bits.end(), 0xFFu);

    uint8_t uid_absent[32]; fill32(uid_absent, 0x71);  // not in cuckoo
    XvmTx tx = make_tx(0xD5);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid_absent, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("bloom-pass.cuckoo-miss.rej", r.tx_rejected == 1u);
    PASS("input_bloom_pass_cuckoo_miss");
}

// Cover the `kRejectAlreadySpent` branch — feed the same UTXO twice across
// two txs in the same round; second sees status |= kUtxoSpent on the first
// pass (after it removed from cuckoo), but if we also pre-set status before
// the round, we exercise the AlreadySpent reject path inside input check.
void test_input_already_spent_rejected()
{
    auto desc = make_desc(42u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    uint8_t uid[32]; fill32(uid, 0x80);
    uint32_t slot = state.seed_utxo(uid, asset_id, 50, 0, alice, 0, 1);
    EXPECT("spent.seed", slot != 0xFFFFFFFFu);
    // Mark the seeded UTXO spent without removing from cuckoo. The input check
    // path will Bloom-pass + cuckoo-find a slot pointing at a status with
    // kUtxoSpent set, hitting the AlreadySpent branch.
    state.utxos[slot].status |= kUtxoSpent;

    XvmTx tx = make_tx(0xD6);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("spent.rej", r.tx_rejected == 1u);
    PASS("input_already_spent_rejected");
}

// Cover the `(u.status & kUtxoOccupied) == 0u` branch — point cuckoo at an
// unoccupied slot.
void test_input_cuckoo_points_unoccupied_slot()
{
    auto desc = make_desc(43u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    uint8_t uid[32]; fill32(uid, 0x90);
    uint32_t slot = state.seed_utxo(uid, asset_id,
                                    50, 0, asset_id /*owner*/, 0, 1);
    EXPECT("unocc.seed", slot != 0xFFFFFFFFu);
    // Clear the occupied bit but leave membership in place. The cuckoo entry
    // still points here; input check sees Occupied=0 and rejects.
    state.utxos[slot].status = 0u;

    XvmTx tx = make_tx(0xD7);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("unocc.rej", r.tx_rejected == 1u);
    PASS("input_cuckoo_points_unoccupied_slot");
}

// Cover the `slot >= state.utxos.size()` branch — handcraft a cuckoo entry
// pointing past the end of the utxos arena.
void test_input_cuckoo_slot_out_of_range()
{
    auto desc = make_desc(44u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 1000, 0, auth, 9);

    uint8_t uid[32]; fill32(uid, 0xA0);
    state.seed_utxo(uid, asset_id, 50, 0, asset_id /*owner*/, 0, 1);
    // Walk the cuckoo arena and rewrite the slot_index of our entry so it
    // points past state.utxos.size().
    for (auto& e : state.cuckoo) {
        if (e.occupied != 0u) {
            bool match = true;
            for (uint32_t i = 0; i < 32u; ++i)
                if (e.utxo_id[i] != uid[i]) { match = false; break; }
            if (match) { e.slot_index = uint32_t(state.utxos.size()) + 7u; break; }
        }
    }

    XvmTx tx = make_tx(0xD8);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("oor.rej", r.tx_rejected == 1u);
    PASS("input_cuckoo_slot_out_of_range");
}

// Cover the asset_changes_offset >= asset_ops.size() guard.
void test_asset_op_offset_out_of_range()
{
    auto desc = make_desc(45u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xCC);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xD9, XvmTxKind::Mint);
    tx.asset_changes_count = 1;
    tx.asset_changes_offset = 99;  // beyond ops.size()

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 100;
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);
    std::memcpy(op.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    // Out-of-range offset means no op is applied; tx still accepts (no inputs).
    EXPECT("oor.ops.acc", r.tx_accepted == 1u);
    EXPECT("oor.ops.applied.0", r.asset_ops_applied == 0u);
    PASS("asset_op_offset_out_of_range");
}

// Cover the inner `off >= asset_ops.size()` break (count > 1 but later ops
// would exceed the bound).
void test_asset_op_count_overruns_arena()
{
    auto desc = make_desc(46u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xDD);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xDA, XvmTxKind::Mint);
    tx.asset_changes_count = 4;     // claims 4 ops
    tx.asset_changes_offset = 0;

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 100;
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);
    std::memcpy(op.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};        // only 1 op given
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("overrun.applied.1", r.asset_ops_applied == 1u);
    EXPECT("overrun.acc", r.tx_accepted == 1u);
    PASS("asset_op_count_overruns_arena");
}

// Cover an unrecognized AssetOpKind reaching the switch default (no case
// matches → all ops paths skipped, tx still accepts as no-op).
void test_asset_op_kind_unknown_no_op()
{
    auto desc = make_desc(47u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xEE);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xDB, XvmTxKind::Transfer);
    tx.asset_changes_count = 1;
    tx.asset_changes_offset = 0;

    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.kind = 99u;  // not a recognized AssetOpKind
    std::memcpy(op.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("unkkind.acc", r.tx_accepted == 1u);
    EXPECT("unkkind.applied.0", r.asset_ops_applied == 0u);
    PASS("asset_op_kind_unknown_no_op");
}

// Cover u128_add carry and u128_sub borrow: mint at amount_lo near max, then
// burn an amount where supply.lo < amount_lo (forcing the borrow path).
void test_u128_add_carry_and_sub_borrow()
{
    auto desc = make_desc(48u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xCA);
    uint8_t auth[32]; fill32(auth, 0x42);
    // Seed supply low byte at 0x0F and hi at 1 so a burn of 0x10 in low
    // exercises the borrow path: hi > sub_hi || (hi == sub_hi && lo < sub_lo).
    state.seed_asset(asset_id, /*lo=*/0x0FULL, /*hi=*/1ULL, auth, 9);

    // First mint with carry: Mint twice with amount_lo near 2^64.
    XvmTx tx_m1 = make_tx(0xDC, XvmTxKind::Mint);
    tx_m1.asset_changes_count = 1;
    AssetOp op_m1{};
    std::memcpy(op_m1.asset_id, asset_id, 32);
    op_m1.amount_lo = 0xFFFFFFFFFFFFFF00ULL;
    op_m1.kind = static_cast<uint32_t>(AssetOpKind::Mint);
    std::memcpy(op_m1.authority_witness_root, auth, 32);

    XvmTx tx_m2 = make_tx(0xDF, XvmTxKind::Mint);
    tx_m2.asset_changes_offset = 1;
    tx_m2.asset_changes_count = 1;
    AssetOp op_m2 = op_m1;
    op_m2.amount_lo = 0x200ULL;   // Sum exceeds 2^64 → carry into hi.

    std::vector<XvmTx>   txs1 = {tx_m1, tx_m2};
    std::vector<AssetOp> ops1 = {op_m1, op_m2};
    auto r_m = ref::run_reference(state, desc, txs1, {}, {}, {}, {}, ops1);
    EXPECT("u128.mint.acc", r_m.tx_accepted == 2u);
    // total_minted_hi must reflect carry from the cumulative low overflow.
    EXPECT("u128.mint.hi", r_m.total_minted_hi == 1u);

    // Burn that exercises the borrow inside u128_sub (lo < sub_lo, hi >= sub_hi).
    auto desc2 = make_desc(49u);
    XvmTx tx_b = make_tx(0xDD, XvmTxKind::Burn);
    tx_b.asset_changes_count = 1;
    AssetOp op_b{};
    std::memcpy(op_b.asset_id, asset_id, 32);
    op_b.amount_lo = 0xFFFFFFFFFFFFFF00ULL;  // larger than current supply.lo
    op_b.kind = static_cast<uint32_t>(AssetOpKind::Burn);
    std::memcpy(op_b.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs2 = {tx_b};
    std::vector<AssetOp> ops2 = {op_b};
    auto r_b = ref::run_reference(state, desc2, txs2, {}, {}, {}, {}, ops2);
    EXPECT("u128.burn.acc", r_b.tx_accepted == 1u);
    EXPECT("u128.burn.applied", r_b.asset_ops_applied == 1u);
    PASS("u128_add_carry_and_sub_borrow");
}

// Cover the cuckoo_insert "update if utxo_id already present" path: insert
// a UTXO, then replay the same insert through membership_insert. The second
// invocation finds the existing entry in either bucket and updates slot_index.
void test_cuckoo_insert_update_existing()
{
    auto state = ref::XVMReferenceState::empty();
    uint8_t uid[32]; fill32(uid, 0xBB);
    ref::membership_insert(state, uid, /*slot=*/123u);
    // Repeat insert with a different slot — must succeed (returns true) and
    // overwrite slot_index in place rather than burning a new slot.
    ref::membership_insert(state, uid, /*slot=*/456u);
    uint32_t out = 0xFFFFFFFFu;
    EXPECT("cuck.upd.found", ref::membership_query(state, uid, &out));
    EXPECT("cuck.upd.slot", out == 456u);
    PASS("cuckoo_insert_update_existing");
}

// Cover InputCheck-only mode (no asset transition pass).
void test_mode_input_check_only()
{
    auto desc = make_desc(60u, 100, XVMTransitionMode::InputCheck);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0);
    uint8_t auth[32]; fill32(auth, 0x42);
    uint8_t alice[32]; fill32(alice, 0x11);
    state.seed_asset(asset_id, 1000, 0, auth, 9);
    uint8_t uid[32]; fill32(uid, 0xB1);
    state.seed_utxo(uid, asset_id, 50, 0, alice, 0, 1);

    XvmTx tx = make_tx(0xE0);
    InputBatch ib{};
    std::memcpy(ib.tx_id, tx.tx_id, 32);
    ib.input_count = 1; ib.witness_count = 1;
    std::vector<uint8_t> inputs(32);
    std::memcpy(inputs.data(), uid, 32);

    std::vector<XvmTx> txs = {tx};
    std::vector<InputBatch> ibs = {ib};
    auto r = ref::run_reference(state, desc, txs, ibs, {}, inputs, {}, {});
    EXPECT("ic.acc", r.tx_accepted == 1u);
    PASS("mode_input_check_only");
}

// Cover AssetTransition-only mode (no input/output transition).
void test_mode_asset_transition_only()
{
    auto desc = make_desc(61u, 100, XVMTransitionMode::AssetTransition);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xAC);
    uint8_t auth[32]; fill32(auth, 0x42);
    state.seed_asset(asset_id, 0, 0, auth, 9);

    XvmTx tx = make_tx(0xE1, XvmTxKind::Mint);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 1000;
    op.kind = static_cast<uint32_t>(AssetOpKind::Mint);
    std::memcpy(op.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("at.applied", r.asset_ops_applied == 1u);
    PASS("mode_asset_transition_only");
}

// Cover the borrow-fails branch inside u128_sub: burn an amount strictly
// greater than the asset's supply (across both lo and hi). The supply check
// fires returning false, taking the rejection path.
void test_burn_greater_than_supply_rejects()
{
    auto desc = make_desc(62u);
    auto state = ref::XVMReferenceState::empty();
    uint8_t asset_id[32]; fill32(asset_id, 0xCD);
    uint8_t auth[32]; fill32(auth, 0x42);
    // supply = (lo=10, hi=0); burn 20 → hi==sub_hi && lo<sub_lo → rejects.
    state.seed_asset(asset_id, 10, 0, auth, 9);

    XvmTx tx = make_tx(0xE2, XvmTxKind::Burn);
    tx.asset_changes_count = 1;
    AssetOp op{};
    std::memcpy(op.asset_id, asset_id, 32);
    op.amount_lo = 20;            // > 10
    op.kind = static_cast<uint32_t>(AssetOpKind::Burn);
    std::memcpy(op.authority_witness_root, auth, 32);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("burn.over.rej", r.tx_rejected == 1u);
    PASS("burn_greater_than_supply_rejects");
}

// Cover asset_locate on an empty assets table (size 0 → returns max).
// run_reference auto-grows arenas, so we have to call run_reference with a
// state whose assets vector is empty, then assert the asset op rejects.
void test_asset_locate_empty_table_for_burn()
{
    auto desc = make_desc(50u);
    ref::XVMReferenceState state;  // not via empty(); arenas size=0.

    XvmTx tx = make_tx(0xDE, XvmTxKind::Burn);
    tx.asset_changes_count = 1;
    AssetOp op{};
    fill32(op.asset_id, 0xAB);
    op.amount_lo = 1;
    op.kind = static_cast<uint32_t>(AssetOpKind::Burn);

    std::vector<XvmTx>   txs = {tx};
    std::vector<AssetOp> ops = {op};
    // run_reference grows the arenas; the burn must reject (asset missing).
    auto r = ref::run_reference(state, desc, txs, {}, {}, {}, {}, ops);
    EXPECT("alc.empty.rej", r.tx_rejected == 1u);
    PASS("asset_locate_empty_table_for_burn");
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

    test_membership_insert_remove_idempotent();

    test_asset_mint_with_authority_succeeds();
    test_asset_mint_wrong_authority_rejects();
    test_asset_mint_implicit_create_for_unknown_asset();
    test_asset_burn_succeeds_with_supply();
    test_asset_burn_overflow_rejects();
    test_asset_transfer_op_counts_only();
    test_asset_export_marker_emitted();
    test_asset_export_overflow_rejects();
    test_asset_import_with_marker_accepted();
    test_asset_import_no_marker_rejects();
    test_asset_burn_unknown_asset_rejects();

    test_input_missing_utxo_rejected();
    test_input_locktime_in_future_rejected();
    test_input_threshold_unmet_rejected();
    test_input_double_spend_within_round_rejected();

    test_tx_no_duplicates_multi_input();
    test_input_bloom_pass_cuckoo_miss();
    test_input_already_spent_rejected();
    test_input_cuckoo_points_unoccupied_slot();
    test_input_cuckoo_slot_out_of_range();
    test_asset_op_offset_out_of_range();
    test_asset_op_count_overruns_arena();
    test_asset_op_kind_unknown_no_op();
    test_u128_add_carry_and_sub_borrow();
    test_cuckoo_insert_update_existing();
    test_mode_input_check_only();
    test_mode_asset_transition_only();
    test_burn_greater_than_supply_rejects();
    test_asset_locate_empty_table_for_burn();

    std::printf("[xvm_layout_test] passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
