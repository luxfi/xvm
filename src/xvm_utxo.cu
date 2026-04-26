// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_utxo.cu — CUDA peer of xvm_utxo.metal.

#include "xvm_kernels_common.cuh"

namespace xvm::cuda {

__device__ inline bool tx_inputs_have_duplicates(uint32_t input_offset, uint32_t input_count,
                                                 const uint8_t* inputs)
{
    if (input_count < 2u) return false;
    for (uint32_t i = 0; i + 1u < input_count; ++i) {
        for (uint32_t j = i + 1u; j < input_count; ++j) {
            if (memeq32(inputs + input_offset + i*32u,
                        inputs + input_offset + j*32u))
                return true;
        }
    }
    return false;
}

extern "C" __global__ void xvm_utxo_transition(
    const XVMRoundDescriptor* desc,
    XvmTx*                    txs,
    const InputBatch*         input_batches,
    const OutputBatch*        output_batches,
    const uint8_t*            inputs,
    const UTXO*               outputs,
    UTXO*                     utxos,
    uint8_t*                  bloom_bits,
    CuckooEntry*              cuckoo,
    uint32_t*                 inputs_consumed_out,
    uint32_t*                 outputs_created_out,
    uint32_t                  utxo_count,
    uint32_t                  bloom_bit_count,
    uint32_t                  cuckoo_bucket_count,
    uint32_t                  input_batch_count,
    uint32_t                  output_batch_count,
    uint32_t                  outputs_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint64_t height = desc->height;
    uint32_t inputs_consumed = 0;
    uint32_t outputs_created = 0;

    uint32_t count = desc->tx_count;
    for (uint32_t ti = 0; ti < count; ++ti) {
        XvmTx& tx = txs[ti];

        bool reject = false;
        uint32_t reject_reason = 0;

        const InputBatch* ib = nullptr;
        if (tx.input_batch_offset < input_batch_count) {
            ib = &input_batches[tx.input_batch_offset];
        }

        if (ib != nullptr) {
            if (tx_inputs_have_duplicates(ib->input_offset, ib->input_count, inputs)) {
                reject = true; reject_reason = kRejectDuplicateInput;
            }
        }

        uint32_t consumed_slots[256];
        uint32_t consumed_n = 0;
        if (!reject && ib != nullptr) {
            for (uint32_t i = 0; i < ib->input_count; ++i) {
                const uint8_t* uid = inputs + ib->input_offset + i*32u;
                if (!bloom_test(bloom_bits, bloom_bit_count, uid)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                uint32_t slot = 0xFFFFFFFFu;
                if (!cuckoo_query(cuckoo, cuckoo_bucket_count, uid, &slot)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if (slot >= utxo_count) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                UTXO& u = utxos[slot];
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

        for (uint32_t i = 0; i < consumed_n; ++i) {
            uint32_t slot = consumed_slots[i];
            UTXO& u = utxos[slot];
            uint8_t uid[32];
            for (uint32_t k = 0; k < 32u; ++k) uid[k] = u.utxo_id[k];
            u.status |= kUtxoSpent;
            cuckoo_remove(cuckoo, cuckoo_bucket_count, uid);
            ++inputs_consumed;
        }

        bool arena_full = false;
        if (tx.output_batch_offset < output_batch_count) {
            const OutputBatch& ob = output_batches[tx.output_batch_offset];
            for (uint32_t j = 0; j < ob.output_count; ++j) {
                uint32_t off = ob.output_offset + j;
                if (off >= outputs_count) break;
                UTXO src = outputs[off];
                src.status = kUtxoOccupied;
                uint32_t new_slot = utxo_arena_insert(utxos, utxo_count, src);
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

    *inputs_consumed_out = inputs_consumed;
    *outputs_created_out = outputs_created;
}

}  // namespace xvm::cuda
