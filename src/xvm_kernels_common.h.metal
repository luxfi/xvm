// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_kernels_common.h.metal — shared device code for the XVM kernels.
//
// Layout structs MUST match xvm_gpu_layout.hpp byte-for-byte. keccak256 is
// the same Keccak-f[1600] / 0x01 / 0x80 padding used by the CPU reference
// (xvm_cpu_reference.cpp), pvm, and cevm/quasar. Determinism across CPU /
// Metal / CUDA hinges on this exact byte-for-byte recipe.

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layout structs — must match xvm_gpu_layout.hpp byte-for-byte.
// =============================================================================

struct alignas(16) UTXO {
    uchar  utxo_id[32];
    uchar  asset_id[32];
    ulong  amount_lo;
    ulong  amount_hi;
    uchar  owner_root[32];
    ulong  locktime;
    uint   threshold;
    uint   status;
    uint   addresses_offset;
    uint   addresses_count;
    ulong  _pad0;
};

struct alignas(16) InputBatch {
    uchar  tx_id[32];
    uint   input_offset;
    uint   input_count;
    uint   witness_offset;
    uint   witness_count;
    ulong  _pad0;
};

struct alignas(16) OutputBatch {
    uchar  tx_id[32];
    uint   output_offset;
    uint   output_count;
    ulong  _pad0;
    ulong  _pad1;
    ulong  _pad2;
};

struct alignas(16) Asset {
    uchar  asset_id[32];
    ulong  total_supply_lo;
    ulong  total_supply_hi;
    uchar  mint_authority_root[32];
    uint   freeze_flag;
    uint   denomination;
    uint   name_offset;
    uint   name_length;
    uint   occupied;
    uint   _pad0;
    ulong  _pad1;
};

struct alignas(16) CuckooEntry {
    uchar  utxo_id[32];
    uint   slot_index;
    uint   occupied;
    ulong  _pad0;
};

struct alignas(16) AtomicExportMarker {
    uchar  marker_id[32];
    uchar  asset_id[32];
    ulong  amount_lo;
    ulong  amount_hi;
    uint   source_chain;
    uint   target_chain;
    uint   status;
    uint   occupied;
    uchar  recipient_root[32];
    ulong  _pad0;
    ulong  _pad1;
};

struct alignas(16) XvmTx {
    uchar  tx_id[32];
    uint   kind;
    uint   input_batch_offset;
    uint   output_batch_offset;
    uint   asset_changes_offset;
    uint   asset_changes_count;
    uint   target_chain;
    uint   status;
    uint   reject_reason;
    uchar  proof_digest[32];
    ulong  _pad0;
    ulong  _pad1;
};

struct alignas(16) AssetOp {
    uchar  asset_id[32];
    ulong  amount_lo;
    ulong  amount_hi;
    uchar  authority_witness_root[32];
    uint   kind;
    uint   target_chain;
    ulong  _pad0;
    ulong  _pad1;
    ulong  _pad2;
};

struct alignas(16) XVMRoundDescriptor {
    ulong  chain_id;
    ulong  round;
    ulong  timestamp_ns;
    ulong  height;
    uint   mode;
    uint   tx_count;
    uint   input_count;
    uint   output_count;
    uint   asset_op_count;
    uint   input_batch_count;
    uint   output_batch_count;
    uint   closing_flag;
    uchar  parent_execution_root[32];
    ulong  _pad0;
    ulong  _pad1;
};

struct alignas(16) XVMTransitionResult {
    uint   status;
    uint   tx_accepted;
    uint   tx_rejected;
    uint   inputs_consumed;
    uint   outputs_created;
    uint   asset_ops_applied;
    uint   export_markers;
    uint   import_verified;
    ulong  total_burned_lo;
    ulong  total_burned_hi;
    ulong  total_minted_lo;
    ulong  total_minted_hi;
    ulong  height;
    ulong  _pad0;
    uchar  utxo_root[32];
    uchar  asset_root[32];
    uchar  tx_root[32];
    uchar  execution_root[32];
};

// =============================================================================
// Constants — match xvm_gpu_layout.hpp
// =============================================================================

constant uint kUtxoOccupied = 0x1u;
constant uint kUtxoSpent    = 0x2u;

constant uint kAssetActive = 0x1u;
constant uint kAssetFrozen = 0x2u;

constant uint kExportPending  = 0u;
constant uint kExportConsumed = 1u;

constant uint kTxStatusPending  = 0u;
constant uint kTxStatusAccepted = 1u;
constant uint kTxStatusRejected = 2u;

constant uint kRejectMissingInput   = 1u;
constant uint kRejectDuplicateInput = 2u;
constant uint kRejectAlreadySpent   = 3u;
constant uint kRejectLocktime       = 4u;
constant uint kRejectAuth           = 5u;
constant uint kRejectMintAuthority  = 6u;
constant uint kRejectAssetMissing   = 7u;
constant uint kRejectImportNoMarker = 8u;
constant uint kRejectArenaFull      = 9u;
constant uint kRejectAmountOverflow = 10u;

constant uint kXvmTxTransfer = 0u;
constant uint kXvmTxMint     = 1u;
constant uint kXvmTxBurn     = 2u;
constant uint kXvmTxExport   = 3u;
constant uint kXvmTxImport   = 4u;

constant uint kAssetOpMint     = 0u;
constant uint kAssetOpBurn     = 1u;
constant uint kAssetOpTransfer = 2u;
constant uint kAssetOpExport   = 3u;
constant uint kAssetOpImport   = 4u;

constant uint kBloomHashes        = 4u;
constant uint kCuckooSlotsPerBucket = 4u;

constant uint kModeInputCheck      = 0u;
constant uint kModeTransitionApply = 1u;
constant uint kModeAssetTransition = 2u;
constant uint kModeRootUpdate      = 3u;
constant uint kModeFullRound       = 4u;

// =============================================================================
// keccak256
// =============================================================================

constant ulong kKeccakRC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant uint kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64u - n));
}

inline void keccak_f1600(thread ulong* s) {
    for (uint round = 0; round < 24u; ++round) {
        ulong c[5];
        for (uint x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        ulong d[5];
        for (uint x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        for (uint y = 0; y < 25u; y += 5u)
            for (uint x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        ulong b[25];
        for (uint y = 0; y < 5u; ++y)
            for (uint x = 0; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint y = 0; y < 25u; y += 5u) {
            ulong t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

inline void keccak256(thread const uchar* data, ulong len, thread uchar* out) {
    ulong s[25] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    const uint rate = 136u;
    ulong off = 0;
    while (len - off >= rate) {
        for (uint i = 0; i < rate; ++i) {
            uint lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= ((ulong)data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uchar block[136] = {};
    ulong rem = len - off;
    for (ulong i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint i = 0; i < rate; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= ((ulong)block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint i = 0; i < 32u; ++i) {
        uint lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = (uchar)((s[lane] >> sh) & 0xFFu);
    }
}

inline void absorb_u32(thread uchar* dst, uint off, uint v) {
    for (uint k = 0; k < 4u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

inline void absorb_u64(thread uchar* dst, uint off, ulong v) {
    for (uint k = 0; k < 8u; ++k) dst[off + k] = (uchar)((v >> (k*8u)) & 0xFFu);
}

inline bool memeq32(thread const uchar* a, thread const uchar* b) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

inline bool memeq32_dd(device const uchar* a, device const uchar* b) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

inline bool memeq32_dt(device const uchar* a, thread const uchar* b) {
    for (uint i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

// =============================================================================
// Membership: Bloom + cuckoo (matches CPU reference byte-for-byte)
// =============================================================================

inline void hash_quad_local(thread const uchar* utxo_id, thread ulong* out) {
    uchar h[32];
    keccak256(utxo_id, 32u, h);
    for (uint i = 0; i < 4u; ++i) {
        ulong v = 0;
        for (uint k = 0; k < 8u; ++k) v |= ((ulong)h[i*8u + k]) << (k*8u);
        out[i] = v;
    }
}

inline void load_utxo_id_to_thread(device const uchar* src, thread uchar* dst) {
    for (uint i = 0; i < 32u; ++i) dst[i] = src[i];
}

inline void bloom_set(device uchar* bits, uint bit_count,
                      thread const uchar* utxo_id)
{
    if (bit_count == 0u) return;
    ulong hashes[4]; hash_quad_local(utxo_id, hashes);
    for (uint i = 0; i < kBloomHashes; ++i) {
        ulong bit = hashes[i] % (ulong)bit_count;
        uint byte_off = (uint)(bit >> 3);
        uchar mask = (uchar)(1u << ((uint)bit & 7u));
        bits[byte_off] |= mask;
    }
}

inline bool bloom_test(device const uchar* bits, uint bit_count,
                       thread const uchar* utxo_id)
{
    if (bit_count == 0u) return true;
    ulong hashes[4]; hash_quad_local(utxo_id, hashes);
    for (uint i = 0; i < kBloomHashes; ++i) {
        ulong bit = hashes[i] % (ulong)bit_count;
        uint byte_off = (uint)(bit >> 3);
        uchar mask = (uchar)(1u << ((uint)bit & 7u));
        if ((bits[byte_off] & mask) == 0u) return false;
    }
    return true;
}

inline uint cuckoo_bucket(ulong h, uint bucket_count) {
    return (uint)h & (bucket_count - 1u);
}

inline bool cuckoo_insert(device CuckooEntry* arena, uint bucket_count,
                          thread const uchar* utxo_id, uint slot_index)
{
    ulong hashes[4]; hash_quad_local(utxo_id, hashes);
    for (uint k = 0; k < 2u; ++k) {
        uint b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint s = 0; s < kCuckooSlotsPerBucket; ++s) {
            device CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32_dt(e.utxo_id, utxo_id)) {
                e.slot_index = slot_index;
                return true;
            }
        }
    }
    for (uint k = 0; k < 2u; ++k) {
        uint b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint s = 0; s < kCuckooSlotsPerBucket; ++s) {
            device CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied == 0u) {
                for (uint i = 0; i < 32u; ++i) e.utxo_id[i] = utxo_id[i];
                e.slot_index = slot_index;
                e.occupied = 1u;
                return true;
            }
        }
    }
    return false;
}

inline void cuckoo_remove(device CuckooEntry* arena, uint bucket_count,
                          thread const uchar* utxo_id)
{
    ulong hashes[4]; hash_quad_local(utxo_id, hashes);
    for (uint k = 0; k < 2u; ++k) {
        uint b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint s = 0; s < kCuckooSlotsPerBucket; ++s) {
            device CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32_dt(e.utxo_id, utxo_id)) {
                for (uint i = 0; i < 32u; ++i) e.utxo_id[i] = 0;
                e.slot_index = 0;
                e.occupied = 0;
                e._pad0 = 0;
                return;
            }
        }
    }
}

inline bool cuckoo_query(device const CuckooEntry* arena, uint bucket_count,
                         thread const uchar* utxo_id, thread uint* out_slot)
{
    ulong hashes[4]; hash_quad_local(utxo_id, hashes);
    for (uint k = 0; k < 2u; ++k) {
        uint b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint s = 0; s < kCuckooSlotsPerBucket; ++s) {
            device const CuckooEntry& e = arena[b * kCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32_dt(e.utxo_id, utxo_id)) {
                if (out_slot) *out_slot = e.slot_index;
                return true;
            }
        }
    }
    return false;
}

// =============================================================================
// Asset locator (open-addressing, FNV-1a on the 32-byte asset_id)
// =============================================================================

inline uint asset_index_hash_thread(thread const uchar* asset_id, uint mask) {
    ulong h = 0xcbf29ce484222325UL;
    for (uint i = 0; i < 32u; ++i) {
        h ^= (ulong)asset_id[i];
        h *= 0x100000001b3UL;
    }
    return (uint)h & mask;
}

inline uint asset_locate(device Asset* tab, uint count,
                         thread const uchar* asset_id, bool insert_if_missing)
{
    uint mask = count - 1u;
    uint idx = asset_index_hash_thread(asset_id, mask);
    for (uint probe = 0; probe < count; ++probe) {
        device Asset& a = tab[idx];
        if (a.occupied == 0u) {
            if (insert_if_missing) {
                for (uint i = 0; i < 32u; ++i) a.asset_id[i] = asset_id[i];
                a.total_supply_lo = 0;
                a.total_supply_hi = 0;
                for (uint i = 0; i < 32u; ++i) a.mint_authority_root[i] = 0;
                a.freeze_flag = kAssetActive;
                a.denomination = 0;
                a.name_offset = 0;
                a.name_length = 0;
                a.occupied = 1u;
                a._pad0 = 0;
                a._pad1 = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32_dt(a.asset_id, asset_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline uint export_marker_locate(device AtomicExportMarker* arena, uint count,
                                 thread const uchar* marker_id, bool insert_if_missing)
{
    uint mask = count - 1u;
    ulong h = 0xcbf29ce484222325UL;
    for (uint i = 0; i < 32u; ++i) { h ^= (ulong)marker_id[i]; h *= 0x100000001b3UL; }
    uint idx = (uint)h & mask;
    for (uint probe = 0; probe < count; ++probe) {
        device AtomicExportMarker& m = arena[idx];
        if (m.occupied == 0u) {
            if (insert_if_missing) {
                for (uint i = 0; i < 32u; ++i) m.marker_id[i] = marker_id[i];
                for (uint i = 0; i < 32u; ++i) m.asset_id[i] = 0;
                m.amount_lo = 0;
                m.amount_hi = 0;
                m.source_chain = 0;
                m.target_chain = 0;
                m.status = kExportPending;
                m.occupied = 1u;
                for (uint i = 0; i < 32u; ++i) m.recipient_root[i] = 0;
                m._pad0 = 0;
                m._pad1 = 0;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32_dt(m.marker_id, marker_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// =============================================================================
// 128-bit saturating arithmetic
// =============================================================================

inline void u128_add(thread ulong& lo, thread ulong& hi, ulong add_lo, ulong add_hi) {
    ulong new_lo = lo + add_lo;
    ulong carry = (new_lo < lo) ? 1u : 0u;
    lo = new_lo;
    hi = hi + add_hi + carry;
}

inline void u128_add_d(device ulong& lo, device ulong& hi, ulong add_lo, ulong add_hi) {
    ulong cur_lo = lo;
    ulong cur_hi = hi;
    ulong new_lo = cur_lo + add_lo;
    ulong carry = (new_lo < cur_lo) ? 1u : 0u;
    lo = new_lo;
    hi = cur_hi + add_hi + carry;
}

inline bool u128_sub_d(device ulong& lo, device ulong& hi, ulong sub_lo, ulong sub_hi) {
    ulong cur_lo = lo;
    ulong cur_hi = hi;
    if (cur_hi < sub_hi || (cur_hi == sub_hi && cur_lo < sub_lo)) return false;
    ulong borrow = (cur_lo < sub_lo) ? 1u : 0u;
    lo = cur_lo - sub_lo;
    hi = cur_hi - sub_hi - borrow;
    return true;
}

// =============================================================================
// UTXO arena bump-insert (linear scan for first free slot)
// =============================================================================

inline uint utxo_arena_insert_local(device UTXO* arena, uint count,
                                    thread const UTXO* src)
{
    for (uint i = 0; i < count; ++i) {
        device UTXO& s = arena[i];
        if ((s.status & kUtxoOccupied) == 0u) {
            for (uint k = 0; k < 32u; ++k) s.utxo_id[k]    = src->utxo_id[k];
            for (uint k = 0; k < 32u; ++k) s.asset_id[k]   = src->asset_id[k];
            s.amount_lo = src->amount_lo;
            s.amount_hi = src->amount_hi;
            for (uint k = 0; k < 32u; ++k) s.owner_root[k] = src->owner_root[k];
            s.locktime = src->locktime;
            s.threshold = src->threshold;
            s.addresses_offset = src->addresses_offset;
            s.addresses_count  = src->addresses_count;
            s.status = kUtxoOccupied;
            s._pad0 = 0;
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

inline void load_struct_utxo(device const UTXO* src, thread UTXO* dst) {
    for (uint i = 0; i < 32u; ++i) dst->utxo_id[i]    = src->utxo_id[i];
    for (uint i = 0; i < 32u; ++i) dst->asset_id[i]   = src->asset_id[i];
    dst->amount_lo = src->amount_lo;
    dst->amount_hi = src->amount_hi;
    for (uint i = 0; i < 32u; ++i) dst->owner_root[i] = src->owner_root[i];
    dst->locktime = src->locktime;
    dst->threshold = src->threshold;
    dst->addresses_offset = src->addresses_offset;
    dst->addresses_count = src->addresses_count;
    dst->status = src->status;
    dst->_pad0 = 0;
}

inline void compose_marker_id(thread const uchar* tx_id, uint target_chain,
                              ulong amount_lo, ulong amount_hi,
                              thread uchar* out)
{
    uchar buf[32 + 4 + 8 + 8];
    uint o = 0;
    for (uint i = 0; i < 32u; ++i) buf[o + i] = tx_id[i]; o += 32;
    absorb_u32(buf, o, target_chain); o += 4;
    absorb_u64(buf, o, amount_lo);    o += 8;
    absorb_u64(buf, o, amount_hi);    o += 8;
    keccak256(buf, o, out);
}
