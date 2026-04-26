// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_membership.metal — Bloom + cuckoo membership maintenance.
//
// v0.55 single-thread kernel: re-seed membership from the live UTXO arena.
// This is the deterministic equivalent of the CPU reference's membership
// API and is used to bring the membership table into sync after the host
// pre-populates the UTXO arena via seed_utxo().

#include "xvm_kernels_common.h.metal"

kernel void xvm_membership_rebuild(
    device UTXO*         utxos          [[buffer(0)]],
    device uchar*        bloom_bits     [[buffer(1)]],
    device CuckooEntry*  cuckoo         [[buffer(2)]],
    constant uint&       utxo_count     [[buffer(3)]],
    constant uint&       bloom_bit_count [[buffer(4)]],
    constant uint&       cuckoo_bucket_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0u) return;

    // Clear the cuckoo arena (Bloom is monotonic — never cleared on re-seed).
    uint total = cuckoo_bucket_count * kCuckooSlotsPerBucket;
    for (uint i = 0; i < total; ++i) {
        device CuckooEntry& e = cuckoo[i];
        for (uint k = 0; k < 32u; ++k) e.utxo_id[k] = 0;
        e.slot_index = 0;
        e.occupied = 0;
        e._pad0 = 0;
    }

    for (uint i = 0; i < utxo_count; ++i) {
        device const UTXO& u = utxos[i];
        if ((u.status & kUtxoOccupied) == 0u) continue;
        if ((u.status & kUtxoSpent)    != 0u) continue;
        uchar uid[32];
        for (uint k = 0; k < 32u; ++k) uid[k] = u.utxo_id[k];
        bloom_set(bloom_bits, bloom_bit_count, uid);
        (void)cuckoo_insert(cuckoo, cuckoo_bucket_count, uid, i);
    }
}
