// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_membership.cu — CUDA peer of xvm_membership.metal.

#include "xvm_kernels_common.cuh"

namespace xvm::cuda {

extern "C" __global__ void xvm_membership_rebuild(
    UTXO*               utxos,
    uint8_t*            bloom_bits,
    CuckooEntry*        cuckoo,
    uint32_t            utxo_count,
    uint32_t            bloom_bit_count,
    uint32_t            cuckoo_bucket_count)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    uint32_t total = cuckoo_bucket_count * kCuckooSlotsPerBucket;
    for (uint32_t i = 0; i < total; ++i) {
        CuckooEntry& e = cuckoo[i];
        for (uint32_t k = 0; k < 32u; ++k) e.utxo_id[k] = 0;
        e.slot_index = 0;
        e.occupied = 0;
        e._pad0 = 0;
    }

    for (uint32_t i = 0; i < utxo_count; ++i) {
        const UTXO& u = utxos[i];
        if ((u.status & kUtxoOccupied) == 0u) continue;
        if ((u.status & kUtxoSpent)    != 0u) continue;
        bloom_set(bloom_bits, bloom_bit_count, u.utxo_id);
        (void)cuckoo_insert(cuckoo, cuckoo_bucket_count, u.utxo_id, i);
    }
}

}  // namespace xvm::cuda
