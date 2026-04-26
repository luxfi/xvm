// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file xvm_gpu_layout.hpp
/// Shared host/GPU memory layouts for the XVM (X-Chain Asset VM) GPU substrate.
///
/// **Scope** — XVM is the **GPU-native UTXO transition substrate** for the
/// X-Chain. State (UTXO arena, asset arena, hot-set membership table) lives
/// on the GPU; the canonical transition logic also runs on the GPU. This is
/// what closes the LP-137 gap from "GPU-resident" to "GPU-native": a small
/// dedicated set of transition kernels that match a deterministic CPU
/// reference byte-for-byte.
///
///   xvm/                            the substrate this header describes
///   xvm/src/xvm_cpu_reference.cpp   deterministic CPU oracle (this file's twin)
///   xvm/src/xvm_*.cu                CUDA transition kernels (v0.55)
///   xvm/src/xvm_*.metal             Metal transition kernels (v0.55)
///
/// Round model — one XVM round covers:
///   * UTXOInputCheck      : per-input existence + ownership/auth + duplicate
///                           detection within batch
///   * UTXOTransitionApply : consume inputs (mark spent), insert outputs,
///                           update membership table, update asset balances
///   * XAssetTransition    : mint / burn / transfer / export / import asset ops
///   * XRootUpdate         : utxo_root, asset_root, tx_root, execution_root
///
/// All offsets here MUST match xvm_*.metal and xvm_*.cu byte-for-byte. The
/// CPU reference implementation in xvm_cpu_reference.cpp consumes the same
/// arenas in the same order and produces identical roots — it is the
/// equivalence oracle for cross-backend determinism (CPU vs Metal vs CUDA).

#pragma once

#include <cstdint>

namespace xvm::gpu {

// =============================================================================
// Residency class tags
// =============================================================================
//
// XVM splits its working set into two residency classes:
//
//   * DeviceHot — UTXO arena, asset arena, membership table, current-round
//                 input/output batches. Pinned in GPU memory; transitions
//                 read and write here.
//   * HostCold  — archival UTXOs (already spent, beyond a retention window),
//                 historical export markers. Resident on the host; pulled
//                 into the device only on faulted access (out of scope for
//                 v0.54 / v0.55 — DeviceHot path only).

enum class ResidencyClass : uint32_t {
    DeviceHot = 0,
    HostCold  = 1,
};

// =============================================================================
// UTXO arena (DeviceHot)
// =============================================================================
//
// Bump-allocated arena keyed by utxo_id. Each slot carries the canonical
// UTXO payload plus a small MVCC slot for spend-tracking.
//
// status bits:
//   0x1  occupied  — the slot holds a valid UTXO record
//   0x2  spent     — the UTXO has been consumed (MVCC tombstone within window)
//
// addresses_offset / addresses_count point into a separate addresses arena
// (uint64 array). v0.54 ships the arena header; the addresses arena is a
// flat concatenation referenced from each UTXO record.

struct alignas(16) UTXO {
    uint8_t  utxo_id[32];           ///<  0  keccak(tx_id || output_index || asset_id)
    uint8_t  asset_id[32];          ///< 32  asset identifier (LUX = canonical id)
    uint64_t amount_lo;             ///< 64  amount (low 64 bits)
    uint64_t amount_hi;             ///< 72  amount (high 64 bits) — for >2^64 supplies
    uint8_t  owner_root[32];        ///< 80  keccak over sorted owner address set + threshold
    uint64_t locktime;              ///< 112 unix seconds; spend gated until >= locktime
    uint32_t threshold;             ///< 120 N-of-M threshold for the owner set
    uint32_t status;                ///< 124 0x1 occupied | 0x2 spent
    uint32_t addresses_offset;      ///< 128 byte offset into addresses arena
    uint32_t addresses_count;       ///< 132 count of uint64 addresses
    uint64_t _pad0;                 ///< 136 -> 144
};
static_assert(sizeof(UTXO) == 144, "UTXO layout drift");
static_assert(alignof(UTXO) == 16, "UTXO alignment drift");

inline constexpr uint32_t kDefaultUtxoSlots = 16384u;
inline constexpr uint32_t kDefaultAddressesArenaWords = 65536u;  // uint64 entries

// =============================================================================
// Input / output batches (DeviceHot, host-supplied per round)
// =============================================================================
//
// One InputBatch per tx names its slice within a flat `inputs` arena
// (uint8_t[32] utxo_ids) and its witnesses slice within a flat `witnesses`
// arena (signature blobs). Per-tx OutputBatch points into the per-round
// `outputs` arena (UTXO records the kernel will create).

struct alignas(16) InputBatch {
    uint8_t  tx_id[32];
    uint32_t input_offset;     ///< into round inputs[]
    uint32_t input_count;
    uint32_t witness_offset;   ///< into round witnesses[] (bytes)
    uint32_t witness_count;    ///< bytes
    uint64_t _pad0;            ///< pad to 64
};
static_assert(sizeof(InputBatch) == 64, "InputBatch layout drift");
static_assert(alignof(InputBatch) == 16, "InputBatch alignment drift");

struct alignas(16) OutputBatch {
    uint8_t  tx_id[32];
    uint32_t output_offset;    ///< into round outputs[] (UTXO records)
    uint32_t output_count;
    uint64_t _pad0;            ///< pad to 48
    uint64_t _pad1;            ///< pad to 56
    uint64_t _pad2;            ///< pad to 64
};
static_assert(sizeof(OutputBatch) == 64, "OutputBatch layout drift");
static_assert(alignof(OutputBatch) == 16, "OutputBatch alignment drift");

// =============================================================================
// Asset arena (DeviceHot)
// =============================================================================
//
// Open-addressing table keyed by asset_id (truncated to 64 bits for the
// hash; full asset_id stored for tie-breaking). LUX is asset_id = all-zero
// (canonical native asset).
//
// freeze_flag bits:
//   0x1  active  — asset is mintable / transferrable per its mint authority
//   0x2  frozen  — admin freeze (set via export to a frozen-asset partition)

struct alignas(16) Asset {
    uint8_t  asset_id[32];          ///<  0
    uint64_t total_supply_lo;       ///< 32
    uint64_t total_supply_hi;       ///< 40
    uint8_t  mint_authority_root[32]; ///< 48
    uint32_t freeze_flag;           ///< 80
    uint32_t denomination;          ///< 84  number of decimal digits
    uint32_t name_offset;           ///< 88  into asset_names arena (byte offset)
    uint32_t name_length;           ///< 92  bytes
    uint32_t occupied;              ///< 96  0=free, 1=occupied
    uint32_t _pad0;                 ///< 100
    uint64_t _pad1;                 ///< 104 -> 112
};
static_assert(sizeof(Asset) == 112, "Asset layout drift");
static_assert(alignof(Asset) == 16, "Asset alignment drift");

inline constexpr uint32_t kDefaultAssetSlots = 1024u;
inline constexpr uint32_t kDefaultAssetNameArenaBytes = 16384u;

// =============================================================================
// Membership table (DeviceHot)
// =============================================================================
//
// Hybrid Bloom filter + cuckoo hash for the hot UTXO set:
//   * Bloom filter — bit array sized to the working-set with target FPR.
//                    Zero false-negatives (UTXO present => bit set).
//   * Cuckoo hash  — exact membership check after a Bloom hit, sized to
//                    the working-set + load factor.
//
// kBloomBits / kCuckooBuckets are powers of two so masks replace mods.

struct alignas(16) CuckooEntry {
    uint8_t  utxo_id[32];
    uint32_t slot_index;         ///< pointer back into UTXO arena
    uint32_t occupied;           ///< 0=free, 1=occupied
    uint64_t _pad0;
};
static_assert(sizeof(CuckooEntry) == 48, "CuckooEntry layout drift");
static_assert(alignof(CuckooEntry) == 16, "CuckooEntry alignment drift");

inline constexpr uint32_t kDefaultBloomBits     = 1u << 20;   // 1 Mbit  -> 128 KiB
inline constexpr uint32_t kDefaultBloomHashes   = 4u;         // 4 hash functions
inline constexpr uint32_t kDefaultCuckooBuckets = 1u << 15;   // 32K buckets
inline constexpr uint32_t kDefaultCuckooSlotsPerBucket = 4u;  // 128K cuckoo slots

// =============================================================================
// Transaction arena (DeviceHot, per round)
// =============================================================================

enum class XvmTxKind : uint32_t {
    Transfer = 0,
    Mint     = 1,
    Burn     = 2,
    Export   = 3,   ///< cross-chain export (writes atomic-export marker)
    Import   = 4,   ///< cross-chain import (verifies atomic-export proof)
};

struct alignas(16) XvmTx {
    uint8_t  tx_id[32];                ///<  0
    uint32_t kind;                     ///< 32  XvmTxKind
    uint32_t input_batch_offset;       ///< 36  into round input_batches[]
    uint32_t output_batch_offset;      ///< 40  into round output_batches[]
    uint32_t asset_changes_offset;     ///< 44  into round asset_ops[]
    uint32_t asset_changes_count;      ///< 48
    uint32_t target_chain;             ///< 52  for Export/Import; 0 for Transfer/Mint/Burn
    uint32_t status;                   ///< 56  0=pending, 1=accepted, 2=rejected
    uint32_t reject_reason;            ///< 60  set when status == 2
    uint8_t  proof_digest[32];         ///< 64  for Import: keccak of source export marker
    uint64_t _pad0;                    ///< 96
    uint64_t _pad1;                    ///< 104 -> 112
};
static_assert(sizeof(XvmTx) == 112, "XvmTx layout drift");
static_assert(alignof(XvmTx) == 16, "XvmTx alignment drift");

// =============================================================================
// Asset op (per-tx asset-state change directives)
// =============================================================================

enum class AssetOpKind : uint32_t {
    Mint     = 0,   ///< mint_authority_root must match
    Burn     = 1,
    Transfer = 2,   ///< no balance change at the asset level for fungibles
    Export   = 3,   ///< total_supply -= amount, write export marker
    Import   = 4,   ///< total_supply += amount, mint from import proof
};

struct alignas(16) AssetOp {
    uint8_t  asset_id[32];               ///<  0
    uint64_t amount_lo;                  ///< 32
    uint64_t amount_hi;                  ///< 40
    uint8_t  authority_witness_root[32]; ///< 48 must equal asset.mint_authority_root for Mint
    uint32_t kind;                       ///< 80 AssetOpKind
    uint32_t target_chain;               ///< 84
    uint64_t _pad0;                      ///< 88
    uint64_t _pad1;                      ///< 96 -> 104
    uint64_t _pad2;                      ///< 104 -> 112
};
static_assert(sizeof(AssetOp) == 112, "AssetOp layout drift");
static_assert(alignof(AssetOp) == 16, "AssetOp alignment drift");

// =============================================================================
// Atomic export marker (DeviceHot, written by Export; verified by Import)
// =============================================================================

struct alignas(16) AtomicExportMarker {
    uint8_t  marker_id[32];        ///<  0  keccak(tx_id || target_chain || amount)
    uint8_t  asset_id[32];         ///< 32
    uint64_t amount_lo;            ///< 64
    uint64_t amount_hi;            ///< 72
    uint32_t source_chain;         ///< 80  XVM's chain id
    uint32_t target_chain;         ///< 84
    uint32_t status;               ///< 88  0=pending, 1=consumed
    uint32_t occupied;             ///< 92  0=free, 1=occupied
    uint8_t  recipient_root[32];   ///< 96  destination address commitment
    uint64_t _pad0;                ///< 128
    uint64_t _pad1;                ///< 136 -> 144
};
static_assert(sizeof(AtomicExportMarker) == 144, "AtomicExportMarker layout drift");
static_assert(alignof(AtomicExportMarker) == 16, "AtomicExportMarker alignment drift");

inline constexpr uint32_t kDefaultExportMarkerSlots = 1024u;

// =============================================================================
// Round descriptor (host -> GPU, written once per round)
// =============================================================================

enum class XVMTransitionMode : uint32_t {
    InputCheck       = 0,
    TransitionApply  = 1,
    AssetTransition  = 2,
    RootUpdate       = 3,
    FullRound        = 4,   ///< chain all four in canonical order
};

struct alignas(16) XVMRoundDescriptor {
    uint64_t chain_id;            ///<  0  X-Chain canonical id
    uint64_t round;               ///<  8  monotonic
    uint64_t timestamp_ns;        ///< 16
    uint64_t height;              ///< 24  block height for this round
    uint32_t mode;                ///< 32  XVMTransitionMode
    uint32_t tx_count;            ///< 36
    uint32_t input_count;         ///< 40
    uint32_t output_count;        ///< 44
    uint32_t asset_op_count;      ///< 48
    uint32_t input_batch_count;   ///< 52
    uint32_t output_batch_count;  ///< 56
    uint32_t closing_flag;        ///< 60  reserved (1 = finalize round)
    uint8_t  parent_execution_root[32]; ///< 64
    uint64_t _pad0;               ///< 96
    uint64_t _pad1;               ///< 104 -> 112
};
static_assert(sizeof(XVMRoundDescriptor) == 112, "XVMRoundDescriptor layout drift");
static_assert(alignof(XVMRoundDescriptor) == 16,  "XVMRoundDescriptor alignment drift");

// =============================================================================
// Round result (GPU -> host)
// =============================================================================

struct alignas(16) XVMTransitionResult {
    uint32_t status;               ///<  0  0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t tx_accepted;          ///<  4
    uint32_t tx_rejected;          ///<  8
    uint32_t inputs_consumed;      ///< 12
    uint32_t outputs_created;      ///< 16
    uint32_t asset_ops_applied;    ///< 20
    uint32_t export_markers;       ///< 24
    uint32_t import_verified;      ///< 28
    uint64_t total_burned_lo;      ///< 32
    uint64_t total_burned_hi;      ///< 40
    uint64_t total_minted_lo;      ///< 48
    uint64_t total_minted_hi;      ///< 56
    uint64_t height;               ///< 64
    uint64_t _pad0;                ///< 72  -> 80
    uint8_t  utxo_root[32];        ///< 80
    uint8_t  asset_root[32];       ///< 112
    uint8_t  tx_root[32];          ///< 144
    uint8_t  execution_root[32];   ///< 176 -> 208
};
static_assert(sizeof(XVMTransitionResult) == 208, "XVMTransitionResult layout drift");
static_assert(alignof(XVMTransitionResult) == 16, "XVMTransitionResult alignment drift");

// Tx status / reject reasons.
inline constexpr uint32_t kTxStatusPending   = 0u;
inline constexpr uint32_t kTxStatusAccepted  = 1u;
inline constexpr uint32_t kTxStatusRejected  = 2u;

inline constexpr uint32_t kRejectMissingInput      = 1u;
inline constexpr uint32_t kRejectDuplicateInput    = 2u;
inline constexpr uint32_t kRejectAlreadySpent      = 3u;
inline constexpr uint32_t kRejectLocktime          = 4u;
inline constexpr uint32_t kRejectAuth              = 5u;
inline constexpr uint32_t kRejectMintAuthority     = 6u;
inline constexpr uint32_t kRejectAssetMissing      = 7u;
inline constexpr uint32_t kRejectImportNoMarker    = 8u;
inline constexpr uint32_t kRejectArenaFull         = 9u;
inline constexpr uint32_t kRejectAmountOverflow    = 10u;

// UTXO status bits.
inline constexpr uint32_t kUtxoOccupied = 0x1u;
inline constexpr uint32_t kUtxoSpent    = 0x2u;

// Asset freeze bits.
inline constexpr uint32_t kAssetActive = 0x1u;
inline constexpr uint32_t kAssetFrozen = 0x2u;

// Export marker status.
inline constexpr uint32_t kExportPending  = 0u;
inline constexpr uint32_t kExportConsumed = 1u;

}  // namespace xvm::gpu
