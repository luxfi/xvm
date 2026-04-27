// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_kernels_common.cuh — shared device code for the four CUDA XVM kernels.
// Layout MUST match xvm_gpu_layout.hpp byte-for-byte.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace xvm::cuda {

struct alignas(16) UTXO {
    uint8_t  utxo_id[32];
    uint8_t  asset_id[32];
    uint64_t amount_lo;
    uint64_t amount_hi;
    uint8_t  owner_root[32];
    uint64_t locktime;
    uint32_t threshold;
    uint32_t status;
    uint32_t addresses_offset;
    uint32_t addresses_count;
    uint64_t _pad0;
};

struct alignas(16) InputBatch {
    uint8_t  tx_id[32];
    uint32_t input_offset;
    uint32_t input_count;
    uint32_t witness_offset;
    uint32_t witness_count;
    uint64_t _pad0;
};

struct alignas(16) OutputBatch {
    uint8_t  tx_id[32];
    uint32_t output_offset;
    uint32_t output_count;
    uint64_t _pad0;
    uint64_t _pad1;
    uint64_t _pad2;
};

struct alignas(16) Asset {
    uint8_t  asset_id[32];
    uint64_t total_supply_lo;
    uint64_t total_supply_hi;
    uint8_t  mint_authority_root[32];
    uint32_t freeze_flag;
    uint32_t denomination;
    uint32_t name_offset;
    uint32_t name_length;
    uint32_t occupied;
    uint32_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) CuckooEntry {
    uint8_t  utxo_id[32];
    uint32_t slot_index;
    uint32_t occupied;
    uint64_t _pad0;
};

struct alignas(16) AtomicExportMarker {
    uint8_t  marker_id[32];
    uint8_t  asset_id[32];
    uint64_t amount_lo;
    uint64_t amount_hi;
    uint32_t source_chain;
    uint32_t target_chain;
    uint32_t status;
    uint32_t occupied;
    uint8_t  recipient_root[32];
    uint64_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) XvmTx {
    uint8_t  tx_id[32];
    uint32_t kind;
    uint32_t input_batch_offset;
    uint32_t output_batch_offset;
    uint32_t asset_changes_offset;
    uint32_t asset_changes_count;
    uint32_t target_chain;
    uint32_t status;
    uint32_t reject_reason;
    uint8_t  proof_digest[32];
    uint64_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) AssetOp {
    uint8_t  asset_id[32];
    uint64_t amount_lo;
    uint64_t amount_hi;
    uint8_t  authority_witness_root[32];
    uint32_t kind;
    uint32_t target_chain;
    uint64_t _pad0;
    uint64_t _pad1;
    uint64_t _pad2;
};

struct alignas(16) XVMRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t height;
    uint32_t mode;
    uint32_t tx_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t asset_op_count;
    uint32_t input_batch_count;
    uint32_t output_batch_count;
    uint32_t closing_flag;
    uint8_t  parent_execution_root[32];
    uint64_t _pad0;
    uint64_t _pad1;
};

struct alignas(16) XVMTransitionResult {
    uint32_t status;
    uint32_t tx_accepted;
    uint32_t tx_rejected;
    uint32_t inputs_consumed;
    uint32_t outputs_created;
    uint32_t asset_ops_applied;
    uint32_t export_markers;
    uint32_t import_verified;
    uint64_t total_burned_lo;
    uint64_t total_burned_hi;
    uint64_t total_minted_lo;
    uint64_t total_minted_hi;
    uint64_t height;
    uint64_t _pad0;
    uint8_t  utxo_root[32];
    uint8_t  asset_root[32];
    uint8_t  tx_root[32];
    uint8_t  execution_root[32];
};

constexpr uint32_t kUtxoOccupied = 0x1u;
constexpr uint32_t kUtxoSpent    = 0x2u;
constexpr uint32_t kAssetActive  = 0x1u;
constexpr uint32_t kAssetFrozen  = 0x2u;
constexpr uint32_t kExportPending  = 0u;
constexpr uint32_t kExportConsumed = 1u;
constexpr uint32_t kTxStatusPending  = 0u;
constexpr uint32_t kTxStatusAccepted = 1u;
constexpr uint32_t kTxStatusRejected = 2u;

constexpr uint32_t kRejectMissingInput   = 1u;
constexpr uint32_t kRejectDuplicateInput = 2u;
constexpr uint32_t kRejectAlreadySpent   = 3u;
constexpr uint32_t kRejectLocktime       = 4u;
constexpr uint32_t kRejectAuth           = 5u;
constexpr uint32_t kRejectMintAuthority  = 6u;
constexpr uint32_t kRejectAssetMissing   = 7u;
constexpr uint32_t kRejectImportNoMarker = 8u;
constexpr uint32_t kRejectArenaFull      = 9u;
constexpr uint32_t kRejectAmountOverflow = 10u;

constexpr uint32_t kAssetOpMint     = 0u;
constexpr uint32_t kAssetOpBurn     = 1u;
constexpr uint32_t kAssetOpTransfer = 2u;
constexpr uint32_t kAssetOpExport   = 3u;
constexpr uint32_t kAssetOpImport   = 4u;

constexpr uint32_t kBloomHashes        = 4u;
constexpr uint32_t kCuckooSlotsPerBucket = 4u;

__constant__ static const uint64_t kKeccakRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__constant__ static const uint32_t kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Mask the shift to [0,63]. kKeccakRot[0] is 0; an unmasked rotate by 0 is
// `(x << 0) | (x >> 64)` which is UB. Masking is a no-op for non-zero n and
// yields x for n == 0.
__device__ inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << (n & 63u)) | (x >> ((64u - n) & 63u));
}

__device__ inline void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

__device__ inline void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {0};
    constexpr uint32_t rate = 136u;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((uint64_t)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[136] = {0};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((uint64_t)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uint8_t)((s[lane] >> sh) & 0xFFu);
    }
}

__device__ inline void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}
__device__ inline void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = (uint8_t)((v >> (k*8u)) & 0xFFu);
}

__device__ inline bool memeq32(const uint8_t* a, const uint8_t* b) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

__device__ inline void hash_quad(const uint8_t* utxo_id, uint64_t out[4]) {
    uint8_t h[32];
    keccak256(utxo_id, 32u, h);
    for (uint32_t i = 0; i < 4u; ++i) {
        uint64_t v = 0;
        for (uint32_t k = 0; k < 8u; ++k) v |= ((uint64_t)h[i*8u + k]) << (k*8u);
        out[i] = v;
    }
}

__device__ inline void bloom_set(uint8_t* bits, uint32_t bit_count, const uint8_t* utxo_id) {
    if (bit_count == 0u) return;
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t i = 0; i < kBloomHashes; ++i) {
        uint64_t bit = hashes[i] % (uint64_t)bit_count;
        bits[bit >> 3] |= (uint8_t)(1u << (bit & 7u));
    }
}

__device__ inline bool bloom_test(const uint8_t* bits, uint32_t bit_count, const uint8_t* utxo_id) {
    if (bit_count == 0u) return true;
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t i = 0; i < kBloomHashes; ++i) {
        uint64_t bit = hashes[i] % (uint64_t)bit_count;
        if ((bits[bit >> 3] & (uint8_t)(1u << (bit & 7u))) == 0u) return false;
    }
    return true;
}

__device__ inline uint32_t cuckoo_bucket(uint64_t h, uint32_t bucket_count) {
    return (uint32_t)h & (bucket_count - 1u);
}

__device__ inline bool cuckoo_insert(CuckooEntry* arena, uint32_t bucket_count,
                                     const uint8_t* utxo_id, uint32_t slot_index)
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                e.slot_index = slot_index;
                return true;
            }
        }
    }
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied == 0u) {
                for (uint32_t i = 0; i < 32u; ++i) e.utxo_id[i] = utxo_id[i];
                e.slot_index = slot_index;
                e.occupied = 1u;
                return true;
            }
        }
    }
    return false;
}

__device__ inline void cuckoo_remove(CuckooEntry* arena, uint32_t bucket_count,
                                     const uint8_t* utxo_id)
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                for (uint32_t i = 0; i < 32u; ++i) e.utxo_id[i] = 0;
                e.slot_index = 0;
                e.occupied = 0;
                e._pad0 = 0;
                return;
            }
        }
    }
}

__device__ inline bool cuckoo_query(const CuckooEntry* arena, uint32_t bucket_count,
                                    const uint8_t* utxo_id, uint32_t* out_slot)
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kCuckooSlotsPerBucket; ++s) {
            const CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                if (out_slot) *out_slot = e.slot_index;
                return true;
            }
        }
    }
    return false;
}

__device__ inline uint32_t asset_locate(Asset* tab, uint32_t count,
                                        const uint8_t* asset_id, bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < 32u; ++i) {
        h ^= (uint64_t)asset_id[i];
        h *= 0x100000001b3ULL;
    }
    uint32_t idx = (uint32_t)h & mask;
    for (uint32_t probe = 0; probe < count; ++probe) {
        Asset& a = tab[idx];
        if (a.occupied == 0u) {
            if (insert_if_missing) {
                for (uint32_t i = 0; i < 32u; ++i) a.asset_id[i] = asset_id[i];
                a.total_supply_lo = 0; a.total_supply_hi = 0;
                for (uint32_t i = 0; i < 32u; ++i) a.mint_authority_root[i] = 0;
                a.freeze_flag = kAssetActive;
                a.denomination = 0;
                a.name_offset = 0;
                a.name_length = 0;
                a.occupied = 1u;
                a._pad0 = 0; a._pad1 = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32(a.asset_id, asset_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline uint32_t export_marker_locate(AtomicExportMarker* arena, uint32_t count,
                                                const uint8_t* marker_id, bool insert_if_missing)
{
    uint32_t mask = count - 1u;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < 32u; ++i) { h ^= (uint64_t)marker_id[i]; h *= 0x100000001b3ULL; }
    uint32_t idx = (uint32_t)h & mask;
    for (uint32_t probe = 0; probe < count; ++probe) {
        AtomicExportMarker& m = arena[idx];
        if (m.occupied == 0u) {
            if (insert_if_missing) {
                for (uint32_t i = 0; i < 32u; ++i) m.marker_id[i] = marker_id[i];
                for (uint32_t i = 0; i < 32u; ++i) m.asset_id[i] = 0;
                m.amount_lo = 0; m.amount_hi = 0;
                m.source_chain = 0; m.target_chain = 0;
                m.status = kExportPending;
                m.occupied = 1u;
                for (uint32_t i = 0; i < 32u; ++i) m.recipient_root[i] = 0;
                m._pad0 = 0; m._pad1 = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32(m.marker_id, marker_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

__device__ inline void u128_add(uint64_t& lo, uint64_t& hi, uint64_t add_lo, uint64_t add_hi) {
    uint64_t new_lo = lo + add_lo;
    uint64_t carry = (new_lo < lo) ? 1u : 0u;
    lo = new_lo;
    hi = hi + add_hi + carry;
}
__device__ inline bool u128_sub(uint64_t& lo, uint64_t& hi, uint64_t sub_lo, uint64_t sub_hi) {
    if (hi < sub_hi || (hi == sub_hi && lo < sub_lo)) return false;
    uint64_t new_lo = lo - sub_lo;
    uint64_t borrow = (lo < sub_lo) ? 1u : 0u;
    lo = new_lo;
    hi = hi - sub_hi - borrow;
    return true;
}

__device__ inline uint32_t utxo_arena_insert(UTXO* arena, uint32_t count, const UTXO& src) {
    for (uint32_t i = 0; i < count; ++i) {
        UTXO& s = arena[i];
        if ((s.status & kUtxoOccupied) == 0u) {
            for (uint32_t k = 0; k < 32u; ++k) s.utxo_id[k]    = src.utxo_id[k];
            for (uint32_t k = 0; k < 32u; ++k) s.asset_id[k]   = src.asset_id[k];
            s.amount_lo = src.amount_lo;
            s.amount_hi = src.amount_hi;
            for (uint32_t k = 0; k < 32u; ++k) s.owner_root[k] = src.owner_root[k];
            s.locktime = src.locktime;
            s.threshold = src.threshold;
            s.addresses_offset = src.addresses_offset;
            s.addresses_count  = src.addresses_count;
            s.status = kUtxoOccupied;
            s._pad0 = 0;
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

__device__ inline void compose_marker_id(const uint8_t* tx_id, uint32_t target_chain,
                                         uint64_t amount_lo, uint64_t amount_hi,
                                         uint8_t* out)
{
    uint8_t buf[32 + 4 + 8 + 8];
    uint32_t o = 0;
    for (uint32_t i = 0; i < 32u; ++i) buf[o + i] = tx_id[i]; o += 32;
    absorb_u32(buf, o, target_chain); o += 4;
    absorb_u64(buf, o, amount_lo);    o += 8;
    absorb_u64(buf, o, amount_hi);    o += 8;
    keccak256(buf, o, out);
}

}  // namespace xvm::cuda
