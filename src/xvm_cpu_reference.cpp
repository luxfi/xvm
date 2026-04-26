// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file xvm_cpu_reference.cpp
/// XVM CPU reference — deterministic oracle for cross-backend determinism.
///
/// Mirrors what xvm_*.metal / xvm_*.cu must produce byte-for-byte:
///   * UTXOInputCheck      : Bloom + cuckoo lookup, ownership predicate,
///                           duplicate-input scan within batch, locktime
///   * UTXOTransitionApply : mark inputs spent, append outputs, update
///                           membership table, accumulate asset balances
///   * XAssetTransition    : mint authority check, supply mutation, export
///                           marker emit, import marker verification
///   * XRootUpdate         : utxo_root, asset_root, tx_root, execution_root
///
/// Determinism contract: identical (state, ops) -> identical roots and
/// identical (status, reject_reason) per tx. Failures within a tx are
/// recorded into the tx's status/reject_reason — the tx is silently
/// rejected (not throwing) — same as the GPU kernels will.

#include "lux/xvm/xvm_cpu_reference.hpp"

#include <algorithm>
#include <array>
#include <cstring>

namespace xvm::gpu::ref {

namespace {

// =============================================================================
// keccak256 (matches pvm/quasar/cevm — bit-identical)
// =============================================================================

constexpr std::array<uint64_t, 24> kKeccakRC = {
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

constexpr std::array<uint32_t, 25> kKeccakRot = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

__attribute__((optnone))
void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1);
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

void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {};
    constexpr uint32_t rate = 136;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[rate] = {};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> sh) & 0xFFu);
    }
}

void absorb_u32(uint8_t* dst, uint32_t off, uint32_t v) {
    for (uint32_t k = 0; k < 4u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}
void absorb_u64(uint8_t* dst, uint32_t off, uint64_t v) {
    for (uint32_t k = 0; k < 8u; ++k) dst[off + k] = uint8_t((v >> (k*8)) & 0xFFu);
}

bool memeq32(const uint8_t* a, const uint8_t* b) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != b[i]) return false;
    return true;
}

bool is_zero32(const uint8_t* a) {
    for (uint32_t i = 0; i < 32u; ++i) if (a[i] != 0) return false;
    return true;
}

// =============================================================================
// Membership: Bloom + cuckoo hash
// =============================================================================
//
// Hash family: we derive 4 independent uint64s from keccak256(utxo_id) and
// use them as both Bloom hashes (mod kBloomBits) and cuckoo bucket indices
// (mod kBloomBuckets). The cuckoo arena is `kBuckets * 4` slots; each entry
// stores the full 32-byte utxo_id plus a back-pointer into the UTXO arena.

void hash_quad(const uint8_t utxo_id[32], uint64_t out[4]) {
    uint8_t h[32];
    keccak256(utxo_id, 32, h);
    for (uint32_t i = 0; i < 4u; ++i) {
        uint64_t v = 0;
        for (uint32_t k = 0; k < 8u; ++k) v |= uint64_t(h[i*8 + k]) << (k*8);
        out[i] = v;
    }
}

void bloom_set(std::vector<uint8_t>& bits, const uint8_t utxo_id[32]) {
    if (bits.empty()) return;
    uint32_t bit_count = uint32_t(bits.size()) * 8u;
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t i = 0; i < kDefaultBloomHashes; ++i) {
        uint64_t bit = hashes[i] % bit_count;
        bits[bit >> 3] |= uint8_t(1u << (bit & 7u));
    }
}

bool bloom_test(const std::vector<uint8_t>& bits, const uint8_t utxo_id[32]) {
    if (bits.empty()) return true;  // unsized => assume present (caller falls through to cuckoo)
    uint32_t bit_count = uint32_t(bits.size()) * 8u;
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t i = 0; i < kDefaultBloomHashes; ++i) {
        uint64_t bit = hashes[i] % bit_count;
        if ((bits[bit >> 3] & uint8_t(1u << (bit & 7u))) == 0u) return false;
    }
    return true;
}

uint32_t cuckoo_bucket(uint64_t h, uint32_t bucket_count) {
    return uint32_t(h) & (bucket_count - 1u);
}

bool cuckoo_insert(std::vector<CuckooEntry>& arena,
                   uint32_t bucket_count,
                   const uint8_t utxo_id[32], uint32_t slot_index)
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    // Check both candidate buckets; overwrite if id matches; insert into
    // the first free slot in either bucket.
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kDefaultCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kDefaultCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                e.slot_index = slot_index;
                return true;
            }
        }
    }
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kDefaultCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kDefaultCuckooSlotsPerBucket + s];
            if (e.occupied == 0u) {
                std::memcpy(e.utxo_id, utxo_id, 32);
                e.slot_index = slot_index;
                e.occupied = 1u;
                return true;
            }
        }
    }
    return false;  // both buckets full — caller treats as arena overflow
}

void cuckoo_remove(std::vector<CuckooEntry>& arena,
                   uint32_t bucket_count,
                   const uint8_t utxo_id[32])
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kDefaultCuckooSlotsPerBucket; ++s) {
            CuckooEntry& e = arena[b * kDefaultCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                std::memset(&e, 0, sizeof(e));
                return;
            }
        }
    }
}

bool cuckoo_query(const std::vector<CuckooEntry>& arena,
                  uint32_t bucket_count,
                  const uint8_t utxo_id[32], uint32_t* out_slot)
{
    uint64_t hashes[4]; hash_quad(utxo_id, hashes);
    for (uint32_t k = 0; k < 2u; ++k) {
        uint32_t b = cuckoo_bucket(hashes[k], bucket_count);
        for (uint32_t s = 0; s < kDefaultCuckooSlotsPerBucket; ++s) {
            const CuckooEntry& e = arena[b * kDefaultCuckooSlotsPerBucket + s];
            if (e.occupied != 0u && memeq32(e.utxo_id, utxo_id)) {
                if (out_slot) *out_slot = e.slot_index;
                return true;
            }
        }
    }
    return false;
}

// =============================================================================
// Asset open-addressing locator (keyed by full asset_id)
// =============================================================================

uint32_t asset_locate(std::vector<Asset>& tab, const uint8_t asset_id[32],
                      bool insert_if_missing)
{
    if (tab.empty()) return 0xFFFFFFFFu;
    uint32_t mask = uint32_t(tab.size()) - 1u;
    // FNV-1a over the asset_id (deterministic, simple).
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < 32u; ++i) {
        h ^= uint64_t(asset_id[i]);
        h *= 0x100000001b3ULL;
    }
    uint32_t idx = uint32_t(h) & mask;
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        Asset& a = tab[idx];
        if (a.occupied == 0u) {
            if (insert_if_missing) {
                std::memset(&a, 0, sizeof(a));
                std::memcpy(a.asset_id, asset_id, 32);
                a.occupied = 1u;
                a.freeze_flag = kAssetActive;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32(a.asset_id, asset_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// =============================================================================
// UTXO arena append (bump-allocate)
// =============================================================================

uint32_t utxo_arena_insert(std::vector<UTXO>& arena, const UTXO& src) {
    for (uint32_t i = 0; i < arena.size(); ++i) {
        UTXO& s = arena[i];
        if ((s.status & kUtxoOccupied) == 0u) {
            s = src;
            s.status |= kUtxoOccupied;
            s.status &= ~kUtxoSpent;
            return i;
        }
    }
    return 0xFFFFFFFFu;  // arena full
}

// =============================================================================
// Export marker arena
// =============================================================================

uint32_t export_marker_locate(std::vector<AtomicExportMarker>& arena,
                              const uint8_t marker_id[32], bool insert_if_missing)
{
    if (arena.empty()) return 0xFFFFFFFFu;
    uint32_t mask = uint32_t(arena.size()) - 1u;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < 32u; ++i) {
        h ^= uint64_t(marker_id[i]);
        h *= 0x100000001b3ULL;
    }
    uint32_t idx = uint32_t(h) & mask;
    for (uint32_t probe = 0; probe < arena.size(); ++probe) {
        AtomicExportMarker& m = arena[idx];
        if (m.occupied == 0u) {
            if (insert_if_missing) {
                std::memset(&m, 0, sizeof(m));
                std::memcpy(m.marker_id, marker_id, 32);
                m.occupied = 1u;
                m.status = kExportPending;
                return idx;
            }
            return 0xFFFFFFFFu;
        }
        if (memeq32(m.marker_id, marker_id)) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// =============================================================================
// Per-tx helpers
// =============================================================================

bool tx_inputs_have_duplicates(const InputBatch& ib, std::span<const uint8_t> inputs)
{
    // Linear scan inside the batch: O(n^2). Acceptable for n in the
    // hundreds (a single tx). A canonical-order sort + adjacent-equal scan
    // would yield O(n log n) but identical determinism; keep the simpler
    // form here so the GPU mirror is trivial.
    if (ib.input_count < 2u) return false;
    const uint8_t* base = inputs.data() + ib.input_offset;
    for (uint32_t i = 0; i + 1u < ib.input_count; ++i) {
        for (uint32_t j = i + 1u; j < ib.input_count; ++j) {
            if (memeq32(base + i*32u, base + j*32u)) return true;
        }
    }
    return false;
}

void compose_marker_id(const uint8_t tx_id[32], uint32_t target_chain,
                       uint64_t amount_lo, uint64_t amount_hi,
                       uint8_t out[32])
{
    uint8_t buf[32 + 4 + 8 + 8];
    uint32_t o = 0;
    std::memcpy(buf + o, tx_id, 32); o += 32;
    absorb_u32(buf, o, target_chain); o += 4;
    absorb_u64(buf, o, amount_lo);    o += 8;
    absorb_u64(buf, o, amount_hi);    o += 8;
    keccak256(buf, o, out);
}

// =============================================================================
// Kernels
// =============================================================================

void kernel_input_check_and_apply(XVMReferenceState& state,
                                  std::span<XvmTx> txs,
                                  std::span<const InputBatch> input_batches,
                                  std::span<const OutputBatch> output_batches,
                                  std::span<const uint8_t>    inputs,
                                  std::span<const UTXO>       outputs,
                                  XVMTransitionResult& r)
{
    uint32_t bucket_count = uint32_t(state.cuckoo.size()) / kDefaultCuckooSlotsPerBucket;

    for (uint32_t ti = 0; ti < txs.size(); ++ti) {
        XvmTx& tx = txs[ti];

        // -- locate input batch --
        const InputBatch* ib = nullptr;
        if (tx.input_batch_offset < input_batches.size())
            ib = &input_batches[tx.input_batch_offset];

        // -- input check --
        bool reject = false;
        uint32_t reject_reason = 0;

        if (ib != nullptr) {
            if (tx_inputs_have_duplicates(*ib, inputs)) {
                reject = true; reject_reason = kRejectDuplicateInput;
            }
        }

        // Each input must (a) pass Bloom+cuckoo lookup, (b) be unspent,
        // (c) satisfy locktime. v0.55 ownership/auth check is a
        // simplified threshold predicate: we require witness presence
        // (witness_count >= 1) when threshold > 0; full signature
        // verification is delegated to a sibling cevm-style verifier in a
        // future revision. The contract here is that CPU/Metal/CUDA agree
        // byte-for-byte on the same predicate.
        std::vector<uint32_t> consumed_slots;
        if (!reject && ib != nullptr) {
            for (uint32_t i = 0; i < ib->input_count; ++i) {
                const uint8_t* uid = inputs.data() + ib->input_offset + i*32u;
                if (!bloom_test(state.bloom_bits, uid)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                uint32_t slot = 0xFFFFFFFFu;
                if (!cuckoo_query(state.cuckoo, bucket_count, uid, &slot)) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if (slot >= state.utxos.size()) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                UTXO& u = state.utxos[slot];
                if ((u.status & kUtxoOccupied) == 0u) {
                    reject = true; reject_reason = kRejectMissingInput; break;
                }
                if ((u.status & kUtxoSpent) != 0u) {
                    reject = true; reject_reason = kRejectAlreadySpent; break;
                }
                if (u.locktime > state.height) {
                    reject = true; reject_reason = kRejectLocktime; break;
                }
                if (u.threshold > 0u && ib->witness_count == 0u) {
                    reject = true; reject_reason = kRejectAuth; break;
                }
                consumed_slots.push_back(slot);
            }
        }

        if (reject) {
            tx.status = kTxStatusRejected;
            tx.reject_reason = reject_reason;
            ++r.tx_rejected;
            continue;
        }

        // -- apply: mark inputs spent, remove from cuckoo (Bloom is monotonic) --
        for (uint32_t slot : consumed_slots) {
            UTXO& u = state.utxos[slot];
            u.status |= kUtxoSpent;
            cuckoo_remove(state.cuckoo, bucket_count, u.utxo_id);
            ++r.inputs_consumed;
        }

        // -- apply: insert outputs --
        bool arena_full = false;
        if (tx.output_batch_offset < output_batches.size()) {
            const OutputBatch& ob = output_batches[tx.output_batch_offset];
            for (uint32_t j = 0; j < ob.output_count; ++j) {
                if (ob.output_offset + j >= outputs.size()) break;
                const UTXO& src = outputs[ob.output_offset + j];
                uint32_t new_slot = utxo_arena_insert(state.utxos, src);
                if (new_slot == 0xFFFFFFFFu) {
                    arena_full = true; break;
                }
                bloom_set(state.bloom_bits, src.utxo_id);
                if (!cuckoo_insert(state.cuckoo, bucket_count, src.utxo_id, new_slot)) {
                    arena_full = true; break;
                }
                ++r.outputs_created;
            }
        }
        if (arena_full) {
            tx.status = kTxStatusRejected;
            tx.reject_reason = kRejectArenaFull;
            ++r.tx_rejected;
            continue;
        }

        tx.status = kTxStatusAccepted;
        ++r.tx_accepted;
    }
}

void kernel_asset_transition(XVMReferenceState& state,
                             std::span<XvmTx> txs,
                             std::span<const AssetOp> asset_ops,
                             XVMTransitionResult& r)
{
    auto u128_add = [](uint64_t& lo, uint64_t& hi, uint64_t add_lo, uint64_t add_hi) {
        uint64_t new_lo = lo + add_lo;
        uint64_t carry = (new_lo < lo) ? 1u : 0u;
        lo = new_lo;
        hi = hi + add_hi + carry;
    };
    auto u128_sub = [](uint64_t& lo, uint64_t& hi, uint64_t sub_lo, uint64_t sub_hi) -> bool {
        if (hi < sub_hi || (hi == sub_hi && lo < sub_lo)) return false;
        uint64_t new_lo = lo - sub_lo;
        uint64_t borrow = (lo < sub_lo) ? 1u : 0u;
        lo = new_lo;
        hi = hi - sub_hi - borrow;
        return true;
    };

    for (uint32_t ti = 0; ti < txs.size(); ++ti) {
        XvmTx& tx = txs[ti];
        if (tx.status == kTxStatusRejected) continue;
        if (tx.asset_changes_count == 0u) continue;
        if (tx.asset_changes_offset >= asset_ops.size()) continue;
        // Range: asset_ops[asset_changes_offset .. + asset_changes_count]
        for (uint32_t k = 0; k < tx.asset_changes_count; ++k) {
            uint32_t off = tx.asset_changes_offset + k;
            if (off >= asset_ops.size()) break;
            const AssetOp& op = asset_ops[off];

            uint32_t a_idx = asset_locate(state.assets, op.asset_id, false);
            if (a_idx == 0xFFFFFFFFu) {
                if (op.kind == uint32_t(AssetOpKind::Mint)) {
                    // Implicit-create on first Mint allowed: caller seeds
                    // mint authority by also passing it via authority_witness.
                    a_idx = asset_locate(state.assets, op.asset_id, true);
                    if (a_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        --r.tx_accepted; ++r.tx_rejected;
                        break;
                    }
                    std::memcpy(state.assets[a_idx].mint_authority_root,
                                op.authority_witness_root, 32);
                } else {
                    tx.status = kTxStatusRejected;
                    tx.reject_reason = kRejectAssetMissing;
                    --r.tx_accepted; ++r.tx_rejected;
                    break;
                }
            }
            Asset& a = state.assets[a_idx];

            switch (static_cast<AssetOpKind>(op.kind)) {
                case AssetOpKind::Mint: {
                    if (!memeq32(a.mint_authority_root, op.authority_witness_root)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectMintAuthority;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    u128_add(a.total_supply_lo, a.total_supply_hi,
                             op.amount_lo, op.amount_hi);
                    u128_add(r.total_minted_lo, r.total_minted_hi,
                             op.amount_lo, op.amount_hi);
                    ++r.asset_ops_applied;
                    break;
                }
                case AssetOpKind::Burn: {
                    if (!u128_sub(a.total_supply_lo, a.total_supply_hi,
                                  op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    u128_add(r.total_burned_lo, r.total_burned_hi,
                             op.amount_lo, op.amount_hi);
                    ++r.asset_ops_applied;
                    break;
                }
                case AssetOpKind::Transfer: {
                    // Fungible balance is recorded at the UTXO level, so
                    // this is a no-op at the asset arena. We still count
                    // the op so determinism counters track.
                    ++r.asset_ops_applied;
                    break;
                }
                case AssetOpKind::Export: {
                    if (!u128_sub(a.total_supply_lo, a.total_supply_hi,
                                  op.amount_lo, op.amount_hi)) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectAmountOverflow;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    uint8_t marker_id[32];
                    compose_marker_id(tx.tx_id, op.target_chain,
                                      op.amount_lo, op.amount_hi, marker_id);
                    uint32_t m_idx = export_marker_locate(state.export_markers,
                                                         marker_id, true);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectArenaFull;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    AtomicExportMarker& m = state.export_markers[m_idx];
                    std::memcpy(m.asset_id, op.asset_id, 32);
                    m.amount_lo = op.amount_lo;
                    m.amount_hi = op.amount_hi;
                    m.source_chain = uint32_t(0);  // X-Chain id; canonical 0 for now
                    m.target_chain = op.target_chain;
                    std::memcpy(m.recipient_root, op.authority_witness_root, 32);
                    ++r.export_markers;
                    ++r.asset_ops_applied;
                    break;
                }
                case AssetOpKind::Import: {
                    // Verify proof_digest matches a pending export marker.
                    uint32_t m_idx = export_marker_locate(state.export_markers,
                                                         tx.proof_digest, false);
                    if (m_idx == 0xFFFFFFFFu) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    AtomicExportMarker& m = state.export_markers[m_idx];
                    if (m.status == kExportConsumed) {
                        tx.status = kTxStatusRejected;
                        tx.reject_reason = kRejectImportNoMarker;
                        --r.tx_accepted; ++r.tx_rejected;
                        goto next_tx;
                    }
                    m.status = kExportConsumed;
                    u128_add(a.total_supply_lo, a.total_supply_hi,
                             op.amount_lo, op.amount_hi);
                    u128_add(r.total_minted_lo, r.total_minted_hi,
                             op.amount_lo, op.amount_hi);
                    ++r.import_verified;
                    ++r.asset_ops_applied;
                    break;
                }
            }
        }
next_tx:
        continue;
    }
}

void compute_utxo_root(const std::vector<UTXO>& utxos, uint8_t out[32]) {
    std::array<uint8_t, 32> acc{};
    for (uint32_t i = 0; i < utxos.size(); ++i) {
        const UTXO& u = utxos[i];
        if ((u.status & kUtxoOccupied) == 0u) continue;
        // leaf = keccak(utxo_id || asset_id || amount_lo || amount_hi
        //              || owner_root || locktime || threshold || status || index)
        uint8_t leaf[32 + 32 + 8 + 8 + 32 + 8 + 4 + 4 + 4];
        uint32_t o = 0;
        std::memcpy(leaf + o, u.utxo_id, 32);    o += 32;
        std::memcpy(leaf + o, u.asset_id, 32);   o += 32;
        absorb_u64(leaf, o, u.amount_lo);        o += 8;
        absorb_u64(leaf, o, u.amount_hi);        o += 8;
        std::memcpy(leaf + o, u.owner_root, 32); o += 32;
        absorb_u64(leaf, o, u.locktime);         o += 8;
        absorb_u32(leaf, o, u.threshold);        o += 4;
        absorb_u32(leaf, o, u.status);           o += 4;
        absorb_u32(leaf, o, i);                  o += 4;

        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, lh, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_asset_root(const std::vector<Asset>& assets, uint8_t out[32]) {
    std::array<uint8_t, 32> acc{};
    for (uint32_t i = 0; i < assets.size(); ++i) {
        const Asset& a = assets[i];
        if (a.occupied == 0u) continue;
        uint8_t leaf[32 + 8 + 8 + 32 + 4 + 4 + 4];
        uint32_t o = 0;
        std::memcpy(leaf + o, a.asset_id, 32);    o += 32;
        absorb_u64(leaf, o, a.total_supply_lo);   o += 8;
        absorb_u64(leaf, o, a.total_supply_hi);   o += 8;
        std::memcpy(leaf + o, a.mint_authority_root, 32); o += 32;
        absorb_u32(leaf, o, a.freeze_flag);       o += 4;
        absorb_u32(leaf, o, a.denomination);      o += 4;
        absorb_u32(leaf, o, i);                   o += 4;

        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, lh, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compute_tx_root(std::span<const XvmTx> txs, uint8_t out[32]) {
    std::array<uint8_t, 32> acc{};
    for (uint32_t i = 0; i < txs.size(); ++i) {
        const XvmTx& tx = txs[i];
        uint8_t leaf[32 + 4 + 4 + 4 + 32 + 4];
        uint32_t o = 0;
        std::memcpy(leaf + o, tx.tx_id, 32);          o += 32;
        absorb_u32(leaf, o, tx.kind);                 o += 4;
        absorb_u32(leaf, o, tx.status);               o += 4;
        absorb_u32(leaf, o, tx.reject_reason);        o += 4;
        std::memcpy(leaf + o, tx.proof_digest, 32);   o += 32;
        absorb_u32(leaf, o, i);                       o += 4;

        uint8_t lh[32];
        keccak256(leaf, o, lh);
        uint8_t buf[64];
        std::memcpy(buf, acc.data(), 32);
        std::memcpy(buf + 32, lh, 32);
        keccak256(buf, 64, acc.data());
    }
    std::memcpy(out, acc.data(), 32);
}

void compose_execution_root(const XVMRoundDescriptor& desc,
                            const uint8_t utxo_root[32],
                            const uint8_t asset_root[32],
                            const uint8_t tx_root[32],
                            uint64_t height,
                            uint8_t out[32])
{
    uint8_t composed[32 + 32 + 32 + 32 + 8];
    uint32_t o = 0;
    std::memcpy(composed + o, desc.parent_execution_root, 32); o += 32;
    std::memcpy(composed + o, utxo_root, 32);                  o += 32;
    std::memcpy(composed + o, asset_root, 32);                 o += 32;
    std::memcpy(composed + o, tx_root, 32);                    o += 32;
    absorb_u64(composed, o, height);                           o += 8;
    keccak256(composed, o, out);
}

}  // anonymous namespace

// =============================================================================
// Membership API (exposed for tests)
// =============================================================================

void membership_insert(XVMReferenceState& state, const uint8_t utxo_id[32],
                       uint32_t slot_index)
{
    bloom_set(state.bloom_bits, utxo_id);
    uint32_t bucket_count = uint32_t(state.cuckoo.size()) / kDefaultCuckooSlotsPerBucket;
    cuckoo_insert(state.cuckoo, bucket_count, utxo_id, slot_index);
}

void membership_remove(XVMReferenceState& state, const uint8_t utxo_id[32])
{
    uint32_t bucket_count = uint32_t(state.cuckoo.size()) / kDefaultCuckooSlotsPerBucket;
    cuckoo_remove(state.cuckoo, bucket_count, utxo_id);
}

bool membership_query(const XVMReferenceState& state,
                      const uint8_t utxo_id[32], uint32_t* out_slot)
{
    if (!bloom_test(state.bloom_bits, utxo_id)) return false;
    uint32_t bucket_count = uint32_t(state.cuckoo.size()) / kDefaultCuckooSlotsPerBucket;
    return cuckoo_query(state.cuckoo, bucket_count, utxo_id, out_slot);
}

bool bloom_query(const XVMReferenceState& state, const uint8_t utxo_id[32])
{
    return bloom_test(state.bloom_bits, utxo_id);
}

// =============================================================================
// XVMReferenceState
// =============================================================================

XVMReferenceState XVMReferenceState::empty() {
    XVMReferenceState s;
    s.utxos.assign(kDefaultUtxoSlots, UTXO{});
    s.addresses.assign(kDefaultAddressesArenaWords, 0);
    s.assets.assign(kDefaultAssetSlots, Asset{});
    s.asset_names.assign(kDefaultAssetNameArenaBytes, 0);
    s.export_markers.assign(kDefaultExportMarkerSlots, AtomicExportMarker{});
    s.bloom_bits.assign(kDefaultBloomBits / 8u, 0);
    s.cuckoo.assign(kDefaultCuckooBuckets * kDefaultCuckooSlotsPerBucket,
                    CuckooEntry{});
    s.height = 0;
    std::memset(s.execution_root, 0, 32);
    return s;
}

void XVMReferenceState::seed_asset(const uint8_t asset_id[32],
                                   uint64_t total_supply_lo,
                                   uint64_t total_supply_hi,
                                   const uint8_t mint_authority_root[32],
                                   uint32_t denomination)
{
    uint32_t idx = asset_locate(assets, asset_id, true);
    if (idx == 0xFFFFFFFFu) return;
    Asset& a = assets[idx];
    a.total_supply_lo = total_supply_lo;
    a.total_supply_hi = total_supply_hi;
    std::memcpy(a.mint_authority_root, mint_authority_root, 32);
    a.denomination = denomination;
    a.freeze_flag = kAssetActive;
}

uint32_t XVMReferenceState::seed_utxo(const uint8_t utxo_id[32],
                                      const uint8_t asset_id[32],
                                      uint64_t amount_lo,
                                      uint64_t amount_hi,
                                      const uint8_t owner_root[32],
                                      uint64_t locktime,
                                      uint32_t threshold)
{
    UTXO u{};
    std::memcpy(u.utxo_id, utxo_id, 32);
    std::memcpy(u.asset_id, asset_id, 32);
    u.amount_lo = amount_lo;
    u.amount_hi = amount_hi;
    std::memcpy(u.owner_root, owner_root, 32);
    u.locktime = locktime;
    u.threshold = threshold;
    u.status = kUtxoOccupied;
    uint32_t slot = utxo_arena_insert(utxos, u);
    if (slot == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    membership_insert(*this, utxo_id, slot);
    return slot;
}

// =============================================================================
// Top-level run_reference
// =============================================================================

XVMTransitionResult run_reference(XVMReferenceState& state,
                                  const XVMRoundDescriptor& desc,
                                  std::span<const XvmTx>      txs_in,
                                  std::span<const InputBatch> input_batches,
                                  std::span<const OutputBatch> output_batches,
                                  std::span<const uint8_t>    inputs,
                                  std::span<const UTXO>       outputs,
                                  std::span<const AssetOp>    asset_ops)
{
    if (state.utxos.empty())          state.utxos.assign(kDefaultUtxoSlots, UTXO{});
    if (state.assets.empty())         state.assets.assign(kDefaultAssetSlots, Asset{});
    if (state.export_markers.empty()) state.export_markers.assign(kDefaultExportMarkerSlots, AtomicExportMarker{});
    if (state.bloom_bits.empty())     state.bloom_bits.assign(kDefaultBloomBits / 8u, 0);
    if (state.cuckoo.empty())         state.cuckoo.assign(kDefaultCuckooBuckets * kDefaultCuckooSlotsPerBucket, CuckooEntry{});

    // Advance height if descriptor's height is later (forwards-only).
    if (desc.height > state.height) state.height = desc.height;

    // We mutate per-tx status/reject — keep a writable copy.
    std::vector<XvmTx> txs(txs_in.begin(), txs_in.end());

    XVMTransitionResult r{};
    r.height = state.height;

    auto mode = static_cast<XVMTransitionMode>(desc.mode);

    if (mode == XVMTransitionMode::InputCheck ||
        mode == XVMTransitionMode::TransitionApply ||
        mode == XVMTransitionMode::FullRound) {
        kernel_input_check_and_apply(state, txs, input_batches, output_batches,
                                     inputs, outputs, r);
    }
    if (mode == XVMTransitionMode::AssetTransition ||
        mode == XVMTransitionMode::FullRound) {
        kernel_asset_transition(state, txs, asset_ops, r);
    }

    // Roots — always recomputed so callers get a fresh execution_root.
    compute_utxo_root(state.utxos, r.utxo_root);
    compute_asset_root(state.assets, r.asset_root);
    compute_tx_root(txs, r.tx_root);
    compose_execution_root(desc, r.utxo_root, r.asset_root, r.tx_root,
                           state.height, r.execution_root);
    std::memcpy(state.execution_root, r.execution_root, 32);

    r.status = 1u;
    return r;
}

}  // namespace xvm::gpu::ref
