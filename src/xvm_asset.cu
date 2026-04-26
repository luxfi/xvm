// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_asset.cu — CUDA peer of xvm_asset.metal.

#include "xvm_kernels_common.cuh"

namespace xvm::cuda {

extern "C" __global__ void xvm_asset_transition(
    const XVMRoundDescriptor* desc,
    XvmTx*                    txs,
    const AssetOp*            asset_ops,
    Asset*                    assets,
    AtomicExportMarker*       markers,
    uint32_t*                 applied_out,
    uint32_t*                 exports_out,
    uint32_t*                 imports_out,
    uint64_t*                 minted_out,
    uint64_t*                 burned_out,
    uint32_t                  asset_count,
    uint32_t                  asset_op_count,
    uint32_t                  marker_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t applied = 0, exports_n = 0, imports_n = 0;
    uint64_t minted_lo = 0, minted_hi = 0;
    uint64_t burned_lo = 0, burned_hi = 0;

    uint32_t tx_count = desc->tx_count;
    for (uint32_t ti = 0; ti < tx_count; ++ti) {
        XvmTx& tx = txs[ti];
        if (tx.status == kTxStatusRejected) continue;
        if (tx.asset_changes_count == 0u) continue;
        if (tx.asset_changes_offset >= asset_op_count) continue;

        for (uint32_t k = 0; k < tx.asset_changes_count; ++k) {
            uint32_t off = tx.asset_changes_offset + k;
            if (off >= asset_op_count) break;
            const AssetOp& op = asset_ops[off];

            uint32_t a_idx = asset_locate(assets, asset_count, op.asset_id, false);
            if (a_idx == 0xFFFFFFFFu) {
                if (op.kind == kAssetOpMint) {
                    a_idx = asset_locate(assets, asset_count, op.asset_id, true);
                    if (a_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        break;
                    }
                    for (uint32_t i = 0; i < 32u; ++i)
                        assets[a_idx].mint_authority_root[i] = op.authority_witness_root[i];
                } else {
                    tx.status = kTxStatusRejected;
                    tx.reject_reason = kRejectAssetMissing;
                    break;
                }
            }
            Asset& a = assets[a_idx];

            switch (op.kind) {
                case kAssetOpMint: {
                    if (!memeq32(a.mint_authority_root, op.authority_witness_root)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectMintAuthority;
                        goto next_tx;
                    }
                    u128_add(a.total_supply_lo, a.total_supply_hi, op.amount_lo, op.amount_hi);
                    u128_add(minted_lo, minted_hi, op.amount_lo, op.amount_hi);
                    ++applied;
                    break;
                }
                case kAssetOpBurn: {
                    if (!u128_sub(a.total_supply_lo, a.total_supply_hi, op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        goto next_tx;
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
                    if (!u128_sub(a.total_supply_lo, a.total_supply_hi, op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        goto next_tx;
                    }
                    uint8_t marker_id[32];
                    compose_marker_id(tx.tx_id, op.target_chain,
                                      op.amount_lo, op.amount_hi, marker_id);
                    uint32_t m_idx = export_marker_locate(markers, marker_count, marker_id, true);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        goto next_tx;
                    }
                    AtomicExportMarker& m = markers[m_idx];
                    for (uint32_t i = 0; i < 32u; ++i) m.asset_id[i] = op.asset_id[i];
                    m.amount_lo = op.amount_lo;
                    m.amount_hi = op.amount_hi;
                    m.source_chain = 0u;
                    m.target_chain = op.target_chain;
                    for (uint32_t i = 0; i < 32u; ++i) m.recipient_root[i] = op.authority_witness_root[i];
                    ++exports_n;
                    ++applied;
                    break;
                }
                case kAssetOpImport: {
                    uint32_t m_idx = export_marker_locate(markers, marker_count, tx.proof_digest, false);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        goto next_tx;
                    }
                    AtomicExportMarker& m = markers[m_idx];
                    if (m.status == kExportConsumed) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        goto next_tx;
                    }
                    m.status = kExportConsumed;
                    u128_add(a.total_supply_lo, a.total_supply_hi, op.amount_lo, op.amount_hi);
                    u128_add(minted_lo, minted_hi, op.amount_lo, op.amount_hi);
                    ++imports_n;
                    ++applied;
                    break;
                }
                default: break;
            }
        }
next_tx:
        continue;
    }

    *applied_out = applied;
    *exports_out = exports_n;
    *imports_out = imports_n;
    *minted_out  = minted_lo;
    *burned_out  = burned_lo;
    (void)minted_hi; (void)burned_hi;
}

}  // namespace xvm::cuda
