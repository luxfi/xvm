// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_roots.cu — CUDA peer of xvm_roots.metal.

#include "xvm_kernels_common.cuh"

namespace xvm::cuda {

extern "C" __global__ void xvm_root_update(
    const XVMRoundDescriptor* desc,
    const XvmTx*              txs,
    const UTXO*               utxos,
    const Asset*              assets,
    XVMTransitionResult*      result,
    uint32_t                  tx_count,
    uint32_t                  utxo_count,
    uint32_t                  asset_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t accepted = 0, rejected = 0;
    for (uint32_t i = 0; i < tx_count; ++i) {
        const XvmTx& tx = txs[i];
        if (tx.status == kTxStatusAccepted) ++accepted;
        else if (tx.status == kTxStatusRejected) ++rejected;
    }

    // utxo_root
    uint8_t acc[32]; for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < utxo_count; ++i) {
        const UTXO& u = utxos[i];
        if ((u.status & kUtxoOccupied) == 0u) continue;
        uint8_t leaf[32 + 32 + 8 + 8 + 32 + 8 + 4 + 4 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = u.utxo_id[k];    o += 32;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = u.asset_id[k];   o += 32;
        absorb_u64(leaf, o, u.amount_lo);                                 o += 8;
        absorb_u64(leaf, o, u.amount_hi);                                 o += 8;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = u.owner_root[k]; o += 32;
        absorb_u64(leaf, o, u.locktime);                                  o += 8;
        absorb_u32(leaf, o, u.threshold);                                 o += 4;
        absorb_u32(leaf, o, u.status);                                    o += 4;
        absorb_u32(leaf, o, i);                                           o += 4;
        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uint8_t utxo_root[32]; for (uint32_t k = 0; k < 32u; ++k) utxo_root[k] = acc[k];

    // asset_root
    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < asset_count; ++i) {
        const Asset& a = assets[i];
        if (a.occupied == 0u) continue;
        uint8_t leaf[32 + 8 + 8 + 32 + 4 + 4 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.asset_id[k];           o += 32;
        absorb_u64(leaf, o, a.total_supply_lo);                                   o += 8;
        absorb_u64(leaf, o, a.total_supply_hi);                                   o += 8;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = a.mint_authority_root[k]; o += 32;
        absorb_u32(leaf, o, a.freeze_flag);                                       o += 4;
        absorb_u32(leaf, o, a.denomination);                                      o += 4;
        absorb_u32(leaf, o, i);                                                   o += 4;
        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uint8_t asset_root[32]; for (uint32_t k = 0; k < 32u; ++k) asset_root[k] = acc[k];

    // tx_root
    for (uint32_t k = 0; k < 32u; ++k) acc[k] = 0;
    for (uint32_t i = 0; i < tx_count; ++i) {
        const XvmTx& tx = txs[i];
        uint8_t leaf[32 + 4 + 4 + 4 + 32 + 4];
        uint32_t o = 0;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = tx.tx_id[k];        o += 32;
        absorb_u32(leaf, o, tx.kind);                                        o += 4;
        absorb_u32(leaf, o, tx.status);                                      o += 4;
        absorb_u32(leaf, o, tx.reject_reason);                               o += 4;
        for (uint32_t k = 0; k < 32u; ++k) leaf[o + k] = tx.proof_digest[k]; o += 32;
        absorb_u32(leaf, o, i);                                              o += 4;
        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        for (uint32_t k = 0; k < 32u; ++k) buf[k]      = acc[k];
        for (uint32_t k = 0; k < 32u; ++k) buf[32 + k] = lh[k];
        keccak256(buf, 64, acc);
    }
    uint8_t tx_root[32]; for (uint32_t k = 0; k < 32u; ++k) tx_root[k] = acc[k];

    uint8_t composed[32 + 32 + 32 + 32 + 8];
    uint32_t o = 0;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = desc->parent_execution_root[k]; o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = utxo_root[k];                   o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = asset_root[k];                  o += 32;
    for (uint32_t k = 0; k < 32u; ++k) composed[o + k] = tx_root[k];                     o += 32;
    absorb_u64(composed, o, desc->height);                                               o += 8;
    uint8_t exec_root[32];
    keccak256(composed, o, exec_root);

    for (uint32_t k = 0; k < 32u; ++k) result->utxo_root[k]      = utxo_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->asset_root[k]     = asset_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->tx_root[k]        = tx_root[k];
    for (uint32_t k = 0; k < 32u; ++k) result->execution_root[k] = exec_root[k];
    result->status      = 1u;
    result->tx_accepted = accepted;
    result->tx_rejected = rejected;
    result->height      = desc->height;
}

}  // namespace xvm::cuda
