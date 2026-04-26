// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_utxo.metal — fused UTXOInputCheck + UTXOTransitionApply kernel.
//
// Single-threadgroup canonical-order traversal of XvmTx records. Per tx:
//   * walk the input batch:
//       - duplicate-input scan (O(n^2))
//       - Bloom + cuckoo membership lookup
//       - locktime + threshold predicate
//       - reject the tx (set status, reject_reason) on any failure
//   * on accept: mark inputs spent, remove from cuckoo (Bloom is monotonic),
//                bump-allocate outputs into the UTXO arena, set Bloom +
//                cuckoo for the new utxo_id
//
// The kernel writes per-tx status into the txs[] buffer; the host pulls
// these back as part of the result. Counts (tx_accepted, tx_rejected,
// inputs_consumed, outputs_created) are aggregated into result by the
// roots kernel.

#include "xvm_kernels_common.h.metal"

inline bool tx_inputs_have_duplicates_dev(uint input_offset, uint input_count,
                                          device const uchar* inputs)
{
    if (input_count < 2u) return false;
    for (uint i = 0; i + 1u < input_count; ++i) {
        for (uint j = i + 1u; j < input_count; ++j) {
            if (memeq32_dd(inputs + input_offset + i*32u,
                           inputs + input_offset + j*32u))
                return true;
        }
    }
    return false;
}

kernel void xvm_utxo_transition(
    device const XVMRoundDescriptor* desc          [[buffer(0)]],
    device XvmTx*                    txs           [[buffer(1)]],
    device const InputBatch*         input_batches [[buffer(2)]],
    device const OutputBatch*        output_batches [[buffer(3)]],
    device const uchar*              inputs        [[buffer(4)]],
    device const UTXO*               outputs       [[buffer(5)]],
    device UTXO*                     utxos         [[buffer(6)]],
    device uchar*                    bloom_bits    [[buffer(7)]],
    device CuckooEntry*              cuckoo        [[buffer(8)]],
    device atomic_uint*              inputs_consumed_out [[buffer(9)]],
    device atomic_uint*              outputs_created_out [[buffer(10)]],
    constant uint&                   utxo_count    [[buffer(11)]],
    constant uint&                   bloom_bit_count [[buffer(12)]],
    constant uint&                   cuckoo_bucket_count [[buffer(13)]],
    constant uint&                   input_batch_count [[buffer(14)]],
    constant uint&                   output_batch_count [[buffer(15)]],
    constant uint&                   outputs_count [[buffer(16)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    ulong height = desc->height;
    uint inputs_consumed = 0;
    uint outputs_created = 0;

    uint count = desc->tx_count;
    for (uint ti = 0; ti < count; ++ti) {
        device XvmTx& tx = txs[ti];

        bool reject = false;
        uint reject_reason = 0;

        // Locate input batch (if any).
        device const InputBatch* ib = nullptr;
        if (tx.input_batch_offset < input_batch_count) {
            ib = &input_batches[tx.input_batch_offset];
        }

        // Duplicate scan within batch.
        if (ib != nullptr) {
            if (tx_inputs_have_duplicates_dev(ib->input_offset, ib->input_count, inputs)) {
                reject = true; reject_reason = kRejectDuplicateInput;
            }
        }

        // Walk inputs.
        uint consumed_slots[256];
        uint consumed_n = 0;
        if (!reject && ib != nullptr) {
            for (uint i = 0; i < ib->input_count; ++i) {
                uchar uid[32];
                for (uint k = 0; k < 32u; ++k)
                    uid[k] = inputs[ib->input_offset + i*32u + k];
                if (!bloom_test(bloom_bits, bloom_bit_count, uid)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                uint slot = 0xFFFFFFFFu;
                if (!cuckoo_query(cuckoo, cuckoo_bucket_count, uid, &slot)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if (slot >= utxo_count) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                device UTXO& u = utxos[slot];
                if ((u.status & kUtxoOccupied) == 0u) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if ((u.status & kUtxoSpent) != 0u) {
                    reject = true; reject_reason = kRejectAlreadySpent; break;
                }
                if (u.locktime > height) {
                    reject = true; reject_reason = kRejectLocktime; break;
                }
                if (u.threshold > 0u && ib->witness_count == 0u) {
                    reject = true; reject_reason = kRejectAuth; break;
                }
                if (consumed_n < 256u) consumed_slots[consumed_n++] = slot;
            }
        }

        if (reject) {
            tx.status = kTxStatusRejected;
            tx.reject_reason = reject_reason;
            continue;
        }

        // Apply: mark inputs spent + remove from cuckoo.
        for (uint i = 0; i < consumed_n; ++i) {
            uint slot = consumed_slots[i];
            device UTXO& u = utxos[slot];
            uchar uid[32];
            for (uint k = 0; k < 32u; ++k) uid[k] = u.utxo_id[k];
            u.status |= kUtxoSpent;
            cuckoo_remove(cuckoo, cuckoo_bucket_count, uid);
            ++inputs_consumed;
        }

        // Apply: insert outputs.
        bool arena_full = false;
        if (tx.output_batch_offset < output_batch_count) {
            device const OutputBatch& ob = output_batches[tx.output_batch_offset];
            for (uint j = 0; j < ob.output_count; ++j) {
                uint off = ob.output_offset + j;
                if (off >= outputs_count) break;
                UTXO src;
                load_struct_utxo(&outputs[off], &src);
                src.status = kUtxoOccupied;
                uint new_slot = utxo_arena_insert_local(utxos, utxo_count, &src);
                if (new_slot == 0xFFFFFFFFu) { arena_full = true; break; }
                bloom_set(bloom_bits, bloom_bit_count, src.utxo_id);
                if (!cuckoo_insert(cuckoo, cuckoo_bucket_count, src.utxo_id, new_slot)) {
                    arena_full = true; break;
                }
                ++outputs_created;
            }
        }
        if (arena_full) {
            tx.status = kTxStatusRejected;
            tx.reject_reason = kRejectArenaFull;
            continue;
        }

        tx.status = kTxStatusAccepted;
        tx.reject_reason = 0;
    }

    atomic_store_explicit(inputs_consumed_out, inputs_consumed, memory_order_relaxed);
    atomic_store_explicit(outputs_created_out, outputs_created, memory_order_relaxed);
}
