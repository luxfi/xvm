// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_gpu_engine_cuda.cpp — CUDA-backed driver for XVMGPUEngine.
//
// Mirrors the Metal driver: one round = four sequential kernel launches
// (membership rebuild -> utxo transition -> asset transition -> roots).

#include "lux/xvm/xvm_gpu_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace xvm::gpu {

extern void launch_xvm_membership_rebuild(
    UTXO* utxos, uint8_t* bloom_bits, CuckooEntry* cuckoo,
    uint32_t utxo_count, uint32_t bloom_bit_count, uint32_t cuckoo_bucket_count);

extern void launch_xvm_utxo_transition(
    const XVMRoundDescriptor*, XvmTx*,
    const InputBatch*, const OutputBatch*,
    const uint8_t*, const UTXO*, UTXO*,
    uint8_t*, CuckooEntry*,
    uint32_t*, uint32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

extern void launch_xvm_asset_transition(
    const XVMRoundDescriptor*, XvmTx*,
    const AssetOp*, Asset*, AtomicExportMarker*,
    uint32_t*, uint32_t*, uint32_t*, uint64_t*, uint64_t*,
    uint32_t, uint32_t, uint32_t);

extern void launch_xvm_root_update(
    const XVMRoundDescriptor*, const XvmTx*,
    const UTXO*, const Asset*, XVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);

namespace {

constexpr uint32_t kUtxoSlots         = kDefaultUtxoSlots;
constexpr uint32_t kAssetSlots        = kDefaultAssetSlots;
constexpr uint32_t kBloomBits         = kDefaultBloomBits;
constexpr uint32_t kBloomBytes        = kBloomBits / 8u;
constexpr uint32_t kCuckooBuckets     = kDefaultCuckooBuckets;
constexpr uint32_t kCuckooSlotsTotal  = kCuckooBuckets * kDefaultCuckooSlotsPerBucket;
constexpr uint32_t kExportMarkerSlots = kDefaultExportMarkerSlots;

constexpr uint32_t kMaxTxs            = 4096u;
constexpr uint32_t kMaxInputBatches   = 4096u;
constexpr uint32_t kMaxOutputBatches  = 4096u;
constexpr uint32_t kMaxInputs         = 1u << 16;
constexpr uint32_t kMaxOutputs        = 1u << 16;
constexpr uint32_t kMaxAssetOps       = 1u << 13;

#define CUDA_CHECK(expr) do {                                       \
        cudaError_t e = (expr);                                     \
        if (e != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",    \
                         #expr, __FILE__, __LINE__,                 \
                         cudaGetErrorString(e));                    \
            return;                                                 \
        }                                                           \
    } while (0)

struct Round {
    XVMRoundHandle handle{};
    XVMRoundDescriptor desc{};

    XVMRoundDescriptor* d_desc           = nullptr;
    XvmTx*              d_txs            = nullptr;
    InputBatch*         d_input_batches  = nullptr;
    OutputBatch*        d_output_batches = nullptr;
    uint8_t*            d_inputs         = nullptr;
    UTXO*               d_outputs        = nullptr;
    AssetOp*            d_asset_ops      = nullptr;
    UTXO*               d_utxos          = nullptr;
    Asset*              d_assets         = nullptr;
    AtomicExportMarker* d_markers        = nullptr;
    uint8_t*            d_bloom          = nullptr;
    CuckooEntry*        d_cuckoo         = nullptr;
    XVMTransitionResult* d_result        = nullptr;
    uint32_t*           d_inputs_consumed = nullptr;
    uint32_t*           d_outputs_created = nullptr;
    uint32_t*           d_asset_applied  = nullptr;
    uint32_t*           d_exports        = nullptr;
    uint32_t*           d_imports        = nullptr;
    uint64_t*           d_minted         = nullptr;
    uint64_t*           d_burned         = nullptr;

    // Host shadow copies for arenas the host pre-populates via seed_*.
    std::vector<UTXO>  h_utxos;
    std::vector<Asset> h_assets;

    uint32_t tx_count            = 0;
    uint32_t input_batch_count   = 0;
    uint32_t output_batch_count  = 0;
    uint32_t input_count         = 0;
    uint32_t output_count        = 0;
    uint32_t asset_op_count      = 0;
};

class XVMGPUEngineCuda final : public XVMGPUEngine {
public:
    XVMGPUEngineCuda() {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            device_name_str_ = prop.name;
        } else {
            device_name_str_ = "cuda";
        }
    }
    ~XVMGPUEngineCuda() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    XVMRoundHandle begin_round(const XVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return XVMRoundHandle{0};
        round_ = Round{};
        round_.desc = desc;

        auto alloc = [](void** p, size_t n) -> bool {
            if (cudaMalloc(p, n) != cudaSuccess) return false;
            return cudaMemset(*p, 0, n) == cudaSuccess;
        };

        if (!alloc((void**)&round_.d_desc,           sizeof(XVMRoundDescriptor))
            || !alloc((void**)&round_.d_txs,            sizeof(XvmTx) * kMaxTxs)
            || !alloc((void**)&round_.d_input_batches,  sizeof(InputBatch) * kMaxInputBatches)
            || !alloc((void**)&round_.d_output_batches, sizeof(OutputBatch) * kMaxOutputBatches)
            || !alloc((void**)&round_.d_inputs,         32u * kMaxInputs)
            || !alloc((void**)&round_.d_outputs,        sizeof(UTXO) * kMaxOutputs)
            || !alloc((void**)&round_.d_asset_ops,      sizeof(AssetOp) * kMaxAssetOps)
            || !alloc((void**)&round_.d_utxos,          sizeof(UTXO) * kUtxoSlots)
            || !alloc((void**)&round_.d_assets,         sizeof(Asset) * kAssetSlots)
            || !alloc((void**)&round_.d_markers,        sizeof(AtomicExportMarker) * kExportMarkerSlots)
            || !alloc((void**)&round_.d_bloom,          kBloomBytes)
            || !alloc((void**)&round_.d_cuckoo,         sizeof(CuckooEntry) * kCuckooSlotsTotal)
            || !alloc((void**)&round_.d_result,         sizeof(XVMTransitionResult))
            || !alloc((void**)&round_.d_inputs_consumed, sizeof(uint32_t))
            || !alloc((void**)&round_.d_outputs_created, sizeof(uint32_t))
            || !alloc((void**)&round_.d_asset_applied,  sizeof(uint32_t))
            || !alloc((void**)&round_.d_exports,        sizeof(uint32_t))
            || !alloc((void**)&round_.d_imports,        sizeof(uint32_t))
            || !alloc((void**)&round_.d_minted,         sizeof(uint64_t))
            || !alloc((void**)&round_.d_burned,         sizeof(uint64_t))) {
            return XVMRoundHandle{0};
        }

        round_.h_utxos.assign(kUtxoSlots, UTXO{});
        round_.h_assets.assign(kAssetSlots, Asset{});

        round_.handle = XVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_txs(XVMRoundHandle h, std::span<const XvmTx> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || txs.empty()) return;
        uint32_t cap_left = kMaxTxs - round_.tx_count;
        uint32_t take = std::min<uint32_t>(uint32_t(txs.size()), cap_left);
        cudaMemcpy(round_.d_txs + round_.tx_count, txs.data(),
                   take * sizeof(XvmTx), cudaMemcpyHostToDevice);
        round_.tx_count += take;
    }
    void push_input_batches(XVMRoundHandle h, std::span<const InputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxInputBatches - round_.input_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        cudaMemcpy(round_.d_input_batches + round_.input_batch_count, batches.data(),
                   take * sizeof(InputBatch), cudaMemcpyHostToDevice);
        round_.input_batch_count += take;
    }
    void push_output_batches(XVMRoundHandle h, std::span<const OutputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxOutputBatches - round_.output_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        cudaMemcpy(round_.d_output_batches + round_.output_batch_count, batches.data(),
                   take * sizeof(OutputBatch), cudaMemcpyHostToDevice);
        round_.output_batch_count += take;
    }
    void push_inputs(XVMRoundHandle h, std::span<const uint8_t> inputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || inputs.empty()) return;
        uint32_t bytes_cap = (kMaxInputs - round_.input_count) * 32u;
        uint32_t take = std::min<uint32_t>(uint32_t(inputs.size()), bytes_cap);
        cudaMemcpy(round_.d_inputs + round_.input_count * 32u, inputs.data(),
                   take, cudaMemcpyHostToDevice);
        round_.input_count += take / 32u;
    }
    void push_outputs(XVMRoundHandle h, std::span<const UTXO> outputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || outputs.empty()) return;
        uint32_t cap_left = kMaxOutputs - round_.output_count;
        uint32_t take = std::min<uint32_t>(uint32_t(outputs.size()), cap_left);
        cudaMemcpy(round_.d_outputs + round_.output_count, outputs.data(),
                   take * sizeof(UTXO), cudaMemcpyHostToDevice);
        round_.output_count += take;
    }
    void push_asset_ops(XVMRoundHandle h, std::span<const AssetOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxAssetOps - round_.asset_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        cudaMemcpy(round_.d_asset_ops + round_.asset_op_count, ops.data(),
                   take * sizeof(AssetOp), cudaMemcpyHostToDevice);
        round_.asset_op_count += take;
    }

    void seed_asset(XVMRoundHandle h,
                    const uint8_t asset_id[32],
                    uint64_t total_supply_lo,
                    uint64_t total_supply_hi,
                    const uint8_t mint_authority_root[32],
                    uint32_t denomination) override
    {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        uint32_t mask = kAssetSlots - 1u;
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (uint32_t i = 0; i < 32u; ++i) { hash ^= uint64_t(asset_id[i]); hash *= 0x100000001b3ULL; }
        uint32_t idx = uint32_t(hash) & mask;
        for (uint32_t probe = 0; probe < kAssetSlots; ++probe) {
            Asset& a = round_.h_assets[idx];
            bool match = (a.occupied != 0u);
            if (match) {
                for (uint32_t i = 0; i < 32u; ++i)
                    if (a.asset_id[i] != asset_id[i]) { match = false; break; }
            }
            if (a.occupied == 0u || match) {
                std::memset(&a, 0, sizeof(a));
                std::memcpy(a.asset_id, asset_id, 32);
                a.occupied = 1u;
                a.freeze_flag = kAssetActive;
                a.total_supply_lo = total_supply_lo;
                a.total_supply_hi = total_supply_hi;
                std::memcpy(a.mint_authority_root, mint_authority_root, 32);
                a.denomination = denomination;
                cudaMemcpy(round_.d_assets + idx, &a, sizeof(Asset),
                           cudaMemcpyHostToDevice);
                return;
            }
            idx = (idx + 1u) & mask;
        }
    }

    uint32_t seed_utxo(XVMRoundHandle h,
                       const uint8_t utxo_id[32],
                       const uint8_t asset_id[32],
                       uint64_t amount_lo,
                       uint64_t amount_hi,
                       const uint8_t owner_root[32],
                       uint64_t locktime,
                       uint32_t threshold) override
    {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return 0xFFFFFFFFu;
        for (uint32_t i = 0; i < kUtxoSlots; ++i) {
            UTXO& s = round_.h_utxos[i];
            if ((s.status & kUtxoOccupied) == 0u) {
                std::memset(&s, 0, sizeof(s));
                std::memcpy(s.utxo_id,  utxo_id,  32);
                std::memcpy(s.asset_id, asset_id, 32);
                s.amount_lo = amount_lo;
                s.amount_hi = amount_hi;
                std::memcpy(s.owner_root, owner_root, 32);
                s.locktime = locktime;
                s.threshold = threshold;
                s.status = kUtxoOccupied;
                cudaMemcpy(round_.d_utxos + i, &s, sizeof(UTXO),
                           cudaMemcpyHostToDevice);
                return i;
            }
        }
        return 0xFFFFFFFFu;
    }

    XVMTransitionResult run_until_done(XVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return XVMTransitionResult{};

        round_.desc.tx_count           = round_.tx_count;
        round_.desc.input_count        = round_.input_count;
        round_.desc.output_count       = round_.output_count;
        round_.desc.asset_op_count     = round_.asset_op_count;
        round_.desc.input_batch_count  = round_.input_batch_count;
        round_.desc.output_batch_count = round_.output_batch_count;
        cudaMemcpy(round_.d_desc, &round_.desc, sizeof(XVMRoundDescriptor),
                   cudaMemcpyHostToDevice);

        launch_xvm_membership_rebuild(round_.d_utxos, round_.d_bloom, round_.d_cuckoo,
                                      kUtxoSlots, kBloomBits, kCuckooBuckets);
        launch_xvm_utxo_transition(round_.d_desc, round_.d_txs,
                                   round_.d_input_batches, round_.d_output_batches,
                                   round_.d_inputs, round_.d_outputs, round_.d_utxos,
                                   round_.d_bloom, round_.d_cuckoo,
                                   round_.d_inputs_consumed, round_.d_outputs_created,
                                   kUtxoSlots, kBloomBits, kCuckooBuckets,
                                   round_.input_batch_count,
                                   round_.output_batch_count,
                                   round_.output_count);
        launch_xvm_asset_transition(round_.d_desc, round_.d_txs, round_.d_asset_ops,
                                    round_.d_assets, round_.d_markers,
                                    round_.d_asset_applied, round_.d_exports,
                                    round_.d_imports, round_.d_minted, round_.d_burned,
                                    kAssetSlots, round_.asset_op_count,
                                    kExportMarkerSlots);
        launch_xvm_root_update(round_.d_desc, round_.d_txs,
                               round_.d_utxos, round_.d_assets, round_.d_result,
                               round_.tx_count, kUtxoSlots, kAssetSlots);

        cudaDeviceSynchronize();

        XVMTransitionResult r{};
        cudaMemcpy(&r, round_.d_result, sizeof(XVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        uint32_t inputs_consumed = 0, outputs_created = 0, asset_applied = 0;
        uint32_t exports_n = 0, imports_n = 0;
        uint64_t minted_lo = 0, burned_lo = 0;
        cudaMemcpy(&inputs_consumed, round_.d_inputs_consumed, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&outputs_created, round_.d_outputs_created, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&asset_applied,   round_.d_asset_applied,   sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&exports_n,       round_.d_exports,         sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&imports_n,       round_.d_imports,         sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&minted_lo,       round_.d_minted,          sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&burned_lo,       round_.d_burned,          sizeof(uint64_t), cudaMemcpyDeviceToHost);
        r.inputs_consumed   = inputs_consumed;
        r.outputs_created   = outputs_created;
        r.asset_ops_applied = asset_applied;
        r.export_markers    = exports_n;
        r.import_verified   = imports_n;
        r.total_minted_lo   = minted_lo;
        r.total_minted_hi   = 0;
        r.total_burned_lo   = burned_lo;
        r.total_burned_hi   = 0;
        return r;
    }

    XVMTransitionResult poll_round_result(XVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return XVMTransitionResult{};
        XVMTransitionResult r{};
        cudaMemcpy(&r, round_.d_result, sizeof(XVMTransitionResult),
                   cudaMemcpyDeviceToHost);
        return r;
    }

    void end_round(XVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        cudaFree(round_.d_desc);
        cudaFree(round_.d_txs);
        cudaFree(round_.d_input_batches);
        cudaFree(round_.d_output_batches);
        cudaFree(round_.d_inputs);
        cudaFree(round_.d_outputs);
        cudaFree(round_.d_asset_ops);
        cudaFree(round_.d_utxos);
        cudaFree(round_.d_assets);
        cudaFree(round_.d_markers);
        cudaFree(round_.d_bloom);
        cudaFree(round_.d_cuckoo);
        cudaFree(round_.d_result);
        cudaFree(round_.d_inputs_consumed);
        cudaFree(round_.d_outputs_created);
        cudaFree(round_.d_asset_applied);
        cudaFree(round_.d_exports);
        cudaFree(round_.d_imports);
        cudaFree(round_.d_minted);
        cudaFree(round_.d_burned);
        round_ = Round{};
    }

private:
    bool check_handle(XVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(XVMRoundHandle h) const { return check_handle(h); }

    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

#if !defined(__APPLE__)
std::unique_ptr<XVMGPUEngine> XVMGPUEngine::create() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return nullptr;
    return std::unique_ptr<XVMGPUEngine>(new XVMGPUEngineCuda());
}
#endif

}  // namespace xvm::gpu
