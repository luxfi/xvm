// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file xvm_cpu_reference.hpp
/// CPU reference implementation of the XVM transition kernels — the
/// differential-fuzz oracle for cross-backend determinism (CPU vs Metal
/// vs CUDA must produce byte-identical roots on the same input).
///
/// The reference processes each tx in canonical order:
///   1. UTXOInputCheck      (reject on missing/duplicate/spent/locktime/auth)
///   2. UTXOTransitionApply (mark inputs spent, insert outputs, update membership)
///   3. XAssetTransition    (mint/burn/transfer/export/import per AssetOp)
///   4. XRootUpdate         (utxo_root, asset_root, tx_root, execution_root)
///
/// State carries forward across run_reference() calls only via the arenas
/// the caller threads through — the reference is pure on (state, ops).

#pragma once

#include "xvm_gpu_layout.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace xvm::gpu::ref {

struct XVMReferenceState {
    std::vector<UTXO>               utxos;             ///< sized to kDefaultUtxoSlots
    std::vector<uint64_t>           addresses;         ///< addresses arena (uint64 entries)
    std::vector<Asset>              assets;            ///< sized to kDefaultAssetSlots
    std::vector<uint8_t>            asset_names;       ///< asset_names arena (bytes)
    std::vector<AtomicExportMarker> export_markers;    ///< sized to kDefaultExportMarkerSlots
    std::vector<uint8_t>            bloom_bits;        ///< Bloom filter (kDefaultBloomBits / 8 bytes)
    std::vector<CuckooEntry>        cuckoo;            ///< 4 slots per bucket
    uint64_t                        height = 0;
    uint8_t                         execution_root[32]{};

    static XVMReferenceState empty();

    /// Pre-seed an asset record. Used by tests to register assets before the
    /// first round runs (so Mint/Burn/Transfer have an asset_id to reference).
    void seed_asset(const uint8_t asset_id[32],
                    uint64_t total_supply_lo,
                    uint64_t total_supply_hi,
                    const uint8_t mint_authority_root[32],
                    uint32_t denomination = 9);

    /// Pre-seed a UTXO. Used by tests so `inputs[]` of the first round can
    /// reference real prior-state outputs without a chain history.
    /// Returns the UTXO's slot index (or 0xFFFFFFFF on arena-full).
    uint32_t seed_utxo(const uint8_t utxo_id[32],
                       const uint8_t asset_id[32],
                       uint64_t amount_lo,
                       uint64_t amount_hi,
                       const uint8_t owner_root[32],
                       uint64_t locktime,
                       uint32_t threshold);
};

XVMTransitionResult run_reference(XVMReferenceState& state,
                                  const XVMRoundDescriptor& desc,
                                  std::span<const XvmTx>      txs,
                                  std::span<const InputBatch> input_batches,
                                  std::span<const OutputBatch> output_batches,
                                  std::span<const uint8_t>    inputs,         ///< 32-byte utxo_ids concatenated
                                  std::span<const UTXO>       outputs,        ///< per-round outputs to insert
                                  std::span<const AssetOp>    asset_ops);

// =============================================================================
// Membership table primitives — exposed so tests can validate FPR / FNR.
// =============================================================================

/// Insert a utxo_id into the Bloom + cuckoo membership table.
void membership_insert(XVMReferenceState& state, const uint8_t utxo_id[32],
                       uint32_t slot_index);

/// Remove a utxo_id from the cuckoo half (Bloom is monotonic; we don't
/// clear bits — a Bloom positive followed by cuckoo miss is the spent path).
void membership_remove(XVMReferenceState& state, const uint8_t utxo_id[32]);

/// Test membership. `out_slot` is set to the UTXO arena index when found.
/// Returns true on definite present, false on definite absent.
bool membership_query(const XVMReferenceState& state,
                      const uint8_t utxo_id[32], uint32_t* out_slot);

/// Bloom-only query — returns the conservative answer (true => maybe, false => definitely not).
bool bloom_query(const XVMReferenceState& state, const uint8_t utxo_id[32]);

}  // namespace xvm::gpu::ref
