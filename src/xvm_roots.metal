// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_roots.metal — XRootUpdate kernel.
//
// Recomputes utxo_root, asset_root, tx_root and the composed execution_root
// in canonical order. Tallies tx_accepted / tx_rejected from the per-tx
// status field. Other counters (inputs_consumed, outputs_created, asset
// counters) are stamped onto the result by the host driver from the atomics
// the prior kernels populated.
//
// Single-threadgroup so the keccak fold sees writes from the prior kernels
// in canonical order.

#include "xvm_kernels_common.h.metal"

kernel void xvm_root_update(
    device const XVMRoundDescriptor* desc        [[buffer(0)]],
    device const XvmTx*              txs         [[buffer(1)]],
    device const UTXO*               utxos       [[buffer(2)]],
    device const Asset*              assets      [[buffer(3)]],
    device XVMTransitionResult*      result      [[buffer(4)]],
    constant uint&                   tx_count    [[buffer(5)]],
    constant uint&                   utxo_count  [[buffer(6)]],
    constant uint&                   asset_count [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint accepted = 0, rejected = 0;
    for (uint i = 0; i < tx_count; ++i) {
        device const XvmTx& tx = txs[i];
        if (tx.status == kTxStatusAccepted) ++accepted;
        else if (tx.status == kTxStatusRejected) ++rejected;
    }

    // -- utxo_root --
    uchar acc[32]; for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < utxo_count; ++i) {
        device const UTXO& u = utxos[i];
        if ((u.status & kUtxoOccupied) == 0u) continue;
        uchar leaf[32 + 32 + 8 + 8 + 32 + 8 + 4 + 4 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = u.utxo_id[k];    o += 32;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = u.asset_id[k];   o += 32;
        absorb_u64(leaf, o, u.amount_lo);                             o += 8;
        absorb_u64(leaf, o, u.amount_hi);                             o += 8;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = u.owner_root[k]; o += 32;
        absorb_u64(leaf, o, u.locktime);                              o += 8;
        absorb_u32(leaf, o, u.threshold);                             o += 4;
        absorb_u32(leaf, o, u.status);                                o += 4;
        absorb_u32(leaf, o, i);                                       o += 4;
        uchar lh[32];
        keccak256(leaf, o, lh);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uchar utxo_root[32]; for (uint k = 0; k < 32u; ++k) utxo_root[k] = acc[k];

    // -- asset_root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < asset_count; ++i) {
        device const Asset& a = assets[i];
        if (a.occupied == 0u) continue;
        uchar leaf[32 + 8 + 8 + 32 + 4 + 4 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.asset_id[k];           o += 32;
        absorb_u64(leaf, o, a.total_supply_lo);                               o += 8;
        absorb_u64(leaf, o, a.total_supply_hi);                               o += 8;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = a.mint_authority_root[k]; o += 32;
        absorb_u32(leaf, o, a.freeze_flag);                                   o += 4;
        absorb_u32(leaf, o, a.denomination);                                  o += 4;
        absorb_u32(leaf, o, i);                                               o += 4;
        uchar lh[32];
        keccak256(leaf, o, lh);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uchar asset_root[32]; for (uint k = 0; k < 32u; ++k) asset_root[k] = acc[k];

    // -- tx_root --
    for (uint k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint i = 0; i < tx_count; ++i) {
        device const XvmTx& tx = txs[i];
        uchar leaf[32 + 4 + 4 + 4 + 32 + 4];
        uint o = 0;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = tx.tx_id[k];        o += 32;
        absorb_u32(leaf, o, tx.kind);                                    o += 4;
        absorb_u32(leaf, o, tx.status);                                  o += 4;
        absorb_u32(leaf, o, tx.reject_reason);                           o += 4;
        for (uint k = 0; k < 32u; ++k) leaf[o + k] = tx.proof_digest[k]; o += 32;
        absorb_u32(leaf, o, i);                                          o += 4;
        uchar lh[32];
        keccak256(leaf, o, lh);
        uchar buf[64];
        for (uint k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uchar tx_root[32]; for (uint k = 0; k < 32u; ++k) tx_root[k] = acc[k];

    // -- composed execution_root --
    uchar composed[32 + 32 + 32 + 32 + 8];
    uint o = 0;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = desc->parent_execution_root[k]; o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = utxo_root[k];                   o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = asset_root[k];                  o += 32;
    for (uint k = 0; k < 32u; ++k) composed[o + k] = tx_root[k];                     o += 32;
    absorb_u64(composed, o, desc->height);                                           o += 8;
    uchar exec_root[32];
    keccak256(composed, o, exec_root);

    // -- write result --
    for (uint k = 0; k < 32u; ++k) result->utxo_root[k]      = utxo_root[k];
    for (uint k = 0; k < 32u; ++k) result->asset_root[k]     = asset_root[k];
    for (uint k = 0; k < 32u; ++k) result->tx_root[k]        = tx_root[k];
    for (uint k = 0; k < 32u; ++k) result->execution_root[k] = exec_root[k];
    result->status      = 1u;
    result->tx_accepted = accepted;
    result->tx_rejected = rejected;
    result->height      = desc->height;
    // Other counters are populated by the host from the atomics the prior
    // kernels write — see xvm_gpu_engine.mm.
}
