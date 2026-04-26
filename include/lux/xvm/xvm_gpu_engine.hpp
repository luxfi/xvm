// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file xvm_gpu_engine.hpp
/// XVMGPUEngine — GPU-native X-Chain UTXO transition substrate.
///
/// Lifecycle (mirrors PVMGPUEngine / QuasarGPUEngine):
///   begin_round(XVMRoundDescriptor)
///   push_txs / push_inputs / push_outputs / push_asset_ops
///   run_until_done -> XVMTransitionResult (with execution_root)
///   end_round
///
/// One XVMTransitionResult.execution_root is what the X-Chain block header
/// commits to — this is the linkage that makes X-Chain GPU-native under
/// LP-137.

#pragma once

#include "xvm_gpu_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace xvm::gpu {

struct XVMRoundHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

class XVMGPUEngine {
public:
    virtual ~XVMGPUEngine() = default;

    static std::unique_ptr<XVMGPUEngine> create();

    virtual XVMRoundHandle begin_round(const XVMRoundDescriptor& desc) = 0;

    virtual void push_txs(XVMRoundHandle h, std::span<const XvmTx> txs) = 0;
    virtual void push_input_batches(XVMRoundHandle h,
                                    std::span<const InputBatch> batches) = 0;
    virtual void push_output_batches(XVMRoundHandle h,
                                     std::span<const OutputBatch> batches) = 0;
    virtual void push_inputs(XVMRoundHandle h,
                             std::span<const uint8_t> inputs) = 0;     ///< 32-byte utxo_ids concatenated
    virtual void push_outputs(XVMRoundHandle h,
                              std::span<const UTXO> outputs) = 0;
    virtual void push_asset_ops(XVMRoundHandle h,
                                std::span<const AssetOp> ops) = 0;

    /// Seed an asset record into the GPU asset arena. Required for any tx
    /// that references an existing asset (anything other than the first
    /// CreateAsset / Mint of a brand-new asset). Idempotent on the same key.
    virtual void seed_asset(XVMRoundHandle h,
                            const uint8_t asset_id[32],
                            uint64_t total_supply_lo,
                            uint64_t total_supply_hi,
                            const uint8_t mint_authority_root[32],
                            uint32_t denomination) = 0;

    /// Seed a UTXO record into the GPU UTXO arena. Used by tests to bootstrap
    /// inputs for a round.
    virtual uint32_t seed_utxo(XVMRoundHandle h,
                               const uint8_t utxo_id[32],
                               const uint8_t asset_id[32],
                               uint64_t amount_lo,
                               uint64_t amount_hi,
                               const uint8_t owner_root[32],
                               uint64_t locktime,
                               uint32_t threshold) = 0;

    virtual XVMTransitionResult run_until_done(XVMRoundHandle h,
                                               std::size_t max_epochs = 1) = 0;
    virtual XVMTransitionResult poll_round_result(XVMRoundHandle h) const = 0;

    virtual void end_round(XVMRoundHandle h) = 0;

    virtual bool round_active() const = 0;
    virtual const char* device_name() const = 0;

protected:
    XVMGPUEngine() = default;
    XVMGPUEngine(const XVMGPUEngine&) = delete;
    XVMGPUEngine& operator=(const XVMGPUEngine&) = delete;
};

}  // namespace xvm::gpu
