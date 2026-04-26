// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_asset.metal — XAssetTransition kernel.
//
// Walks AssetOps in canonical order (per-tx) and applies mint/burn/transfer
// /export/import to the asset arena and export-marker arena. Mirrors the
// CPU reference exactly.

#include "xvm_kernels_common.h.metal"

kernel void xvm_asset_transition(
    device const XVMRoundDescriptor* desc        [[buffer(0)]],
    device XvmTx*                    txs         [[buffer(1)]],
    device const AssetOp*            asset_ops   [[buffer(2)]],
    device Asset*                    assets      [[buffer(3)]],
    device AtomicExportMarker*       markers     [[buffer(4)]],
    device atomic_uint*              applied_out [[buffer(5)]],
    device atomic_uint*              exports_out [[buffer(6)]],
    device atomic_uint*              imports_out [[buffer(7)]],
    device atomic_uint*              minted_lo_out [[buffer(8)]],
    device atomic_uint*              minted_hi_out [[buffer(9)]],
    device atomic_uint*              burned_lo_out [[buffer(10)]],
    device atomic_uint*              burned_hi_out [[buffer(11)]],
    constant uint&                   asset_count [[buffer(12)]],
    constant uint&                   asset_op_count [[buffer(13)]],
    constant uint&                   marker_count [[buffer(14)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    uint applied = 0, exports_n = 0, imports_n = 0;
    ulong minted_lo = 0, minted_hi = 0;
    ulong burned_lo = 0, burned_hi = 0;

    uint tx_count = desc->tx_count;
    for (uint ti = 0; ti < tx_count; ++ti) {
        device XvmTx& tx = txs[ti];
        if (tx.status == kTxStatusRejected) continue;
        if (tx.asset_changes_count == 0u) continue;
        if (tx.asset_changes_offset >= asset_op_count) continue;

        bool tx_done = false;
        for (uint k = 0; k < tx.asset_changes_count && !tx_done; ++k) {
            uint off = tx.asset_changes_offset + k;
            if (off >= asset_op_count) break;
            device const AssetOp& op = asset_ops[off];

            uchar asset_id[32];
            for (uint i = 0; i < 32u; ++i) asset_id[i] = op.asset_id[i];

            uint a_idx = asset_locate(assets, asset_count, asset_id, false);
            if (a_idx == 0xFFFFFFFFu) {
                if (op.kind == kAssetOpMint) {
                    a_idx = asset_locate(assets, asset_count, asset_id, true);
                    if (a_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        tx_done = true;
                        break;
                    }
                    for (uint i = 0; i < 32u; ++i)
                        assets[a_idx].mint_authority_root[i] = op.authority_witness_root[i];
                } else {
                    tx.status = kTxStatusRejected;
                    tx.reject_reason = kRejectAssetMissing;
                    tx_done = true;
                    break;
                }
            }
            device Asset& a = assets[a_idx];

            switch (op.kind) {
                case kAssetOpMint: {
                    uchar witness[32];
                    for (uint i = 0; i < 32u; ++i) witness[i] = op.authority_witness_root[i];
                    if (!memeq32_dt(a.mint_authority_root, witness)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectMintAuthority;
                        tx_done = true;
                        break;
                    }
                    u128_add_d(a.total_supply_lo, a.total_supply_hi,
                               op.amount_lo, op.amount_hi);
                    u128_add(minted_lo, minted_hi, op.amount_lo, op.amount_hi);
                    ++applied;
                    break;
                }
                case kAssetOpBurn: {
                    if (!u128_sub_d(a.total_supply_lo, a.total_supply_hi,
                                    op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        tx_done = true;
                        break;
                    }
                    u128_add(burned_lo, burned_hi, op.amount_lo, op.amount_hi);
                    ++applied;
                    break;
                }
                case kAssetOpTransfer: {
                    ++applied;
                    break;
                }
                case kAssetOpExport: {
                    if (!u128_sub_d(a.total_supply_lo, a.total_supply_hi,
                                    op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        tx_done = true;
                        break;
                    }
                    uchar tx_id_local[32];
                    for (uint i = 0; i < 32u; ++i) tx_id_local[i] = tx.tx_id[i];
                    uchar marker_id[32];
                    compose_marker_id(tx_id_local, op.target_chain,
                                      op.amount_lo, op.amount_hi, marker_id);
                    uint m_idx = export_marker_locate(markers, marker_count, marker_id, true);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        tx_done = true;
                        break;
                    }
                    device AtomicExportMarker& m = markers[m_idx];
                    for (uint i = 0; i < 32u; ++i) m.asset_id[i] = op.asset_id[i];
                    m.amount_lo = op.amount_lo;
                    m.amount_hi = op.amount_hi;
                    m.source_chain = 0u;
                    m.target_chain = op.target_chain;
                    for (uint i = 0; i < 32u; ++i) m.recipient_root[i] = op.authority_witness_root[i];
                    ++exports_n;
                    ++applied;
                    break;
                }
                case kAssetOpImport: {
                    uchar proof[32];
                    for (uint i = 0; i < 32u; ++i) proof[i] = tx.proof_digest[i];
                    uint m_idx = export_marker_locate(markers, marker_count, proof, false);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        tx_done = true;
                        break;
                    }
                    device AtomicExportMarker& m = markers[m_idx];
                    if (m.status == kExportConsumed) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        tx_done = true;
                        break;
                    }
                    m.status = kExportConsumed;
                    u128_add_d(a.total_supply_lo, a.total_supply_hi,
                               op.amount_lo, op.amount_hi);
                    u128_add(minted_lo, minted_hi, op.amount_lo, op.amount_hi);
                    ++imports_n;
                    ++applied;
                    break;
                }
                default: break;
            }
        }
    }

    atomic_store_explicit(applied_out,   applied,   memory_order_relaxed);
    atomic_store_explicit(exports_out,   exports_n, memory_order_relaxed);
    atomic_store_explicit(imports_out,   imports_n, memory_order_relaxed);
    // Split 64-bit totals into two uint32 atomics (Metal has no 64-bit atomics).
    atomic_store_explicit(minted_lo_out, (uint)(minted_lo & 0xFFFFFFFFu), memory_order_relaxed);
    atomic_store_explicit(minted_hi_out, (uint)((minted_lo >> 32) & 0xFFFFFFFFu), memory_order_relaxed);
    atomic_store_explicit(burned_lo_out, (uint)(burned_lo & 0xFFFFFFFFu), memory_order_relaxed);
    atomic_store_explicit(burned_hi_out, (uint)((burned_lo >> 32) & 0xFFFFFFFFu), memory_order_relaxed);
    (void)minted_hi; (void)burned_hi;
}
