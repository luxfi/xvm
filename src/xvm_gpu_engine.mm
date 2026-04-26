// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_gpu_engine.mm — Metal-backed driver for XVMGPUEngine.
//
// One round = four sequential kernel dispatches in canonical order:
//   1. xvm_membership_rebuild   (clears + re-seeds cuckoo from utxos)
//   2. xvm_utxo_transition      (input check + apply, output insert)
//   3. xvm_asset_transition     (mint/burn/transfer/export/import)
//   4. xvm_root_update          (utxo/asset/tx/execution roots)
//
// Each dispatch is a single thread (1x1x1) — the kernels do canonical
// in-order traversal of their op streams. Sharding lives in a future
// revision once the determinism contract is locked in.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "lux/xvm/xvm_gpu_engine.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace xvm::gpu {

namespace {

id<MTLLibrary> load_xvm_metallib(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates[] = {
        here / "xvm.metallib",
        std::filesystem::current_path() / "xvm.metallib",
        std::filesystem::current_path() / "src" / "xvm.metallib",
        std::filesystem::current_path() / "xvm" / "src" / "xvm.metallib",
        std::filesystem::current_path().parent_path() / "src" / "xvm.metallib",
    };
    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) continue;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
        if (error)
            std::fprintf(stderr, "metallib load error %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

id<MTLLibrary> compile_xvm_library(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "xvm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    auto load_file = [&](const std::filesystem::path& p) -> NSString* {
        if (!std::filesystem::exists(p)) return nil;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        return [NSString stringWithContentsOfFile:path
                                         encoding:NSUTF8StringEncoding
                                            error:&error];
    };
    NSString* common = nil;
    NSString* k_m = nil;
    NSString* k_u = nil;
    NSString* k_a = nil;
    NSString* k_r = nil;
    for (const auto& dir : candidates_dir) {
        common = load_file(dir / "xvm_kernels_common.h.metal");
        k_m    = load_file(dir / "xvm_membership.metal");
        k_u    = load_file(dir / "xvm_utxo.metal");
        k_a    = load_file(dir / "xvm_asset.metal");
        k_r    = load_file(dir / "xvm_roots.metal");
        if (common && k_m && k_u && k_a && k_r) break;
    }
    if (!common || !k_m || !k_u || !k_a || !k_r) {
        std::fprintf(stderr, "XVM Metal sources not found near %s\n",
                     here.c_str());
        return nil;
    }
    auto strip_include = [](NSString* src) -> NSString* {
        NSMutableString* out = [NSMutableString string];
        NSArray<NSString*>* lines = [src componentsSeparatedByString:@"\n"];
        for (NSString* line in lines) {
            if ([line containsString:@"xvm_kernels_common.h.metal"]) continue;
            [out appendString:line];
            [out appendString:@"\n"];
        }
        return out;
    };
    NSMutableString* combined = [NSMutableString string];
    [combined appendString:common];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_m)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_u)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_a)];
    [combined appendString:@"\n"];
    [combined appendString:strip_include(k_r)];

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion3_0;
    id<MTLLibrary> lib = [device newLibraryWithSource:combined
                                              options:opts
                                                error:&error];
    if (!lib && error)
        std::fprintf(stderr, "XVM Metal compile error: %s\n",
                     [[error localizedDescription] UTF8String]);
    return lib;
}

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
constexpr uint32_t kMaxInputs         = 1u << 16;       // 65536 inputs (utxo_id slots)
constexpr uint32_t kMaxOutputs        = 1u << 16;       // 65536 output UTXOs
constexpr uint32_t kMaxAssetOps       = 1u << 13;       // 8192 asset ops

struct Round {
    XVMRoundHandle handle{};
    XVMRoundDescriptor desc{};

    id<MTLBuffer> desc_buf            = nil;
    id<MTLBuffer> txs_buf             = nil;
    id<MTLBuffer> input_batches_buf   = nil;
    id<MTLBuffer> output_batches_buf  = nil;
    id<MTLBuffer> inputs_buf          = nil;       // raw bytes (32 per input)
    id<MTLBuffer> outputs_buf         = nil;       // UTXO records
    id<MTLBuffer> asset_ops_buf       = nil;
    id<MTLBuffer> utxos_buf           = nil;
    id<MTLBuffer> assets_buf          = nil;
    id<MTLBuffer> markers_buf         = nil;
    id<MTLBuffer> bloom_buf           = nil;
    id<MTLBuffer> cuckoo_buf          = nil;
    id<MTLBuffer> result_buf          = nil;

    id<MTLBuffer> inputs_consumed_buf = nil;
    id<MTLBuffer> outputs_created_buf = nil;
    id<MTLBuffer> asset_applied_buf   = nil;
    id<MTLBuffer> exports_buf         = nil;
    id<MTLBuffer> imports_buf         = nil;
    id<MTLBuffer> minted_lo_buf       = nil;
    id<MTLBuffer> minted_hi_buf       = nil;
    id<MTLBuffer> burned_lo_buf       = nil;
    id<MTLBuffer> burned_hi_buf       = nil;

    uint32_t tx_count            = 0;
    uint32_t input_batch_count   = 0;
    uint32_t output_batch_count  = 0;
    uint32_t input_count         = 0;     // count of 32-byte utxo_id slots in inputs[]
    uint32_t output_count        = 0;     // count of UTXO records in outputs[]
    uint32_t asset_op_count      = 0;
};

class XVMGPUEngineMetal final : public XVMGPUEngine {
public:
    XVMGPUEngineMetal(id<MTLDevice> device,
                      id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> membership_pso,
                      id<MTLComputePipelineState> utxo_pso,
                      id<MTLComputePipelineState> asset_pso,
                      id<MTLComputePipelineState> roots_pso,
                      NSString* device_name)
        : device_(device)
        , queue_(queue)
        , membership_pso_(membership_pso)
        , utxo_pso_(utxo_pso)
        , asset_pso_(asset_pso)
        , roots_pso_(roots_pso)
        , device_name_str_([device_name UTF8String]) {}

    ~XVMGPUEngineMetal() override {
        if (round_active()) end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    XVMRoundHandle begin_round(const XVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return XVMRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;
        round_.tx_count = 0;
        round_.input_batch_count = 0;
        round_.output_batch_count = 0;
        round_.input_count = 0;
        round_.output_count = 0;
        round_.asset_op_count = 0;

        auto mk = [&](size_t bytes) -> id<MTLBuffer> {
            return [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };

        round_.desc_buf            = mk(sizeof(XVMRoundDescriptor));
        round_.txs_buf             = mk(sizeof(XvmTx) * kMaxTxs);
        round_.input_batches_buf   = mk(sizeof(InputBatch)  * kMaxInputBatches);
        round_.output_batches_buf  = mk(sizeof(OutputBatch) * kMaxOutputBatches);
        round_.inputs_buf          = mk(32u * kMaxInputs);
        round_.outputs_buf         = mk(sizeof(UTXO) * kMaxOutputs);
        round_.asset_ops_buf       = mk(sizeof(AssetOp) * kMaxAssetOps);
        round_.utxos_buf           = mk(sizeof(UTXO) * kUtxoSlots);
        round_.assets_buf          = mk(sizeof(Asset) * kAssetSlots);
        round_.markers_buf         = mk(sizeof(AtomicExportMarker) * kExportMarkerSlots);
        round_.bloom_buf           = mk(kBloomBytes);
        round_.cuckoo_buf          = mk(sizeof(CuckooEntry) * kCuckooSlotsTotal);
        round_.result_buf          = mk(sizeof(XVMTransitionResult));
        round_.inputs_consumed_buf = mk(sizeof(uint32_t));
        round_.outputs_created_buf = mk(sizeof(uint32_t));
        round_.asset_applied_buf   = mk(sizeof(uint32_t));
        round_.exports_buf         = mk(sizeof(uint32_t));
        round_.imports_buf         = mk(sizeof(uint32_t));
        round_.minted_lo_buf       = mk(sizeof(uint32_t));
        round_.minted_hi_buf       = mk(sizeof(uint32_t));
        round_.burned_lo_buf       = mk(sizeof(uint32_t));
        round_.burned_hi_buf       = mk(sizeof(uint32_t));

        if (!round_.desc_buf || !round_.txs_buf || !round_.input_batches_buf
            || !round_.output_batches_buf || !round_.inputs_buf || !round_.outputs_buf
            || !round_.asset_ops_buf || !round_.utxos_buf || !round_.assets_buf
            || !round_.markers_buf || !round_.bloom_buf || !round_.cuckoo_buf
            || !round_.result_buf || !round_.inputs_consumed_buf
            || !round_.outputs_created_buf || !round_.asset_applied_buf
            || !round_.exports_buf || !round_.imports_buf
            || !round_.minted_lo_buf || !round_.minted_hi_buf
            || !round_.burned_lo_buf || !round_.burned_hi_buf)
            return XVMRoundHandle{0};

        std::memset([round_.txs_buf contents], 0, sizeof(XvmTx) * kMaxTxs);
        std::memset([round_.input_batches_buf contents], 0,
                    sizeof(InputBatch) * kMaxInputBatches);
        std::memset([round_.output_batches_buf contents], 0,
                    sizeof(OutputBatch) * kMaxOutputBatches);
        std::memset([round_.inputs_buf contents], 0, 32u * kMaxInputs);
        std::memset([round_.outputs_buf contents], 0, sizeof(UTXO) * kMaxOutputs);
        std::memset([round_.asset_ops_buf contents], 0, sizeof(AssetOp) * kMaxAssetOps);
        std::memset([round_.utxos_buf contents], 0, sizeof(UTXO) * kUtxoSlots);
        std::memset([round_.assets_buf contents], 0, sizeof(Asset) * kAssetSlots);
        std::memset([round_.markers_buf contents], 0,
                    sizeof(AtomicExportMarker) * kExportMarkerSlots);
        std::memset([round_.bloom_buf contents], 0, kBloomBytes);
        std::memset([round_.cuckoo_buf contents], 0,
                    sizeof(CuckooEntry) * kCuckooSlotsTotal);
        std::memset([round_.result_buf contents], 0, sizeof(XVMTransitionResult));
        zero_u32_buf(round_.inputs_consumed_buf);
        zero_u32_buf(round_.outputs_created_buf);
        zero_u32_buf(round_.asset_applied_buf);
        zero_u32_buf(round_.exports_buf);
        zero_u32_buf(round_.imports_buf);
        zero_u32_buf(round_.minted_lo_buf);
        zero_u32_buf(round_.minted_hi_buf);
        zero_u32_buf(round_.burned_lo_buf);
        zero_u32_buf(round_.burned_hi_buf);

        round_.handle = XVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_txs(XVMRoundHandle h, std::span<const XvmTx> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || txs.empty()) return;
        uint32_t cap_left = kMaxTxs - round_.tx_count;
        uint32_t take = std::min<uint32_t>(uint32_t(txs.size()), cap_left);
        auto* dst = static_cast<XvmTx*>([round_.txs_buf contents]);
        std::memcpy(dst + round_.tx_count, txs.data(), take * sizeof(XvmTx));
        round_.tx_count += take;
    }
    void push_input_batches(XVMRoundHandle h,
                            std::span<const InputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxInputBatches - round_.input_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        auto* dst = static_cast<InputBatch*>([round_.input_batches_buf contents]);
        std::memcpy(dst + round_.input_batch_count,
                    batches.data(), take * sizeof(InputBatch));
        round_.input_batch_count += take;
    }
    void push_output_batches(XVMRoundHandle h,
                             std::span<const OutputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxOutputBatches - round_.output_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        auto* dst = static_cast<OutputBatch*>([round_.output_batches_buf contents]);
        std::memcpy(dst + round_.output_batch_count,
                    batches.data(), take * sizeof(OutputBatch));
        round_.output_batch_count += take;
    }
    void push_inputs(XVMRoundHandle h, std::span<const uint8_t> inputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || inputs.empty()) return;
        uint32_t bytes_cap = (kMaxInputs - round_.input_count) * 32u;
        uint32_t take = std::min<uint32_t>(uint32_t(inputs.size()), bytes_cap);
        auto* dst = static_cast<uint8_t*>([round_.inputs_buf contents]);
        std::memcpy(dst + round_.input_count * 32u, inputs.data(), take);
        round_.input_count += take / 32u;
    }
    void push_outputs(XVMRoundHandle h, std::span<const UTXO> outputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || outputs.empty()) return;
        uint32_t cap_left = kMaxOutputs - round_.output_count;
        uint32_t take = std::min<uint32_t>(uint32_t(outputs.size()), cap_left);
        auto* dst = static_cast<UTXO*>([round_.outputs_buf contents]);
        std::memcpy(dst + round_.output_count, outputs.data(), take * sizeof(UTXO));
        round_.output_count += take;
    }
    void push_asset_ops(XVMRoundHandle h, std::span<const AssetOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxAssetOps - round_.asset_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        auto* dst = static_cast<AssetOp*>([round_.asset_ops_buf contents]);
        std::memcpy(dst + round_.asset_op_count, ops.data(), take * sizeof(AssetOp));
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
        // Open-addressing locate (mirrors Metal kernel + CPU reference).
        auto* tab = static_cast<Asset*>([round_.assets_buf contents]);
        uint32_t mask = kAssetSlots - 1u;
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (uint32_t i = 0; i < 32u; ++i) {
            hash ^= uint64_t(asset_id[i]);
            hash *= 0x100000001b3ULL;
        }
        uint32_t idx = uint32_t(hash) & mask;
        for (uint32_t probe = 0; probe < kAssetSlots; ++probe) {
            Asset& a = tab[idx];
            if (a.occupied == 0u) {
                std::memset(&a, 0, sizeof(a));
                std::memcpy(a.asset_id, asset_id, 32);
                a.occupied = 1u;
                a.freeze_flag = kAssetActive;
                a.total_supply_lo = total_supply_lo;
                a.total_supply_hi = total_supply_hi;
                std::memcpy(a.mint_authority_root, mint_authority_root, 32);
                a.denomination = denomination;
                return;
            }
            bool match = true;
            for (uint32_t i = 0; i < 32u; ++i)
                if (a.asset_id[i] != asset_id[i]) { match = false; break; }
            if (match) {
                a.total_supply_lo = total_supply_lo;
                a.total_supply_hi = total_supply_hi;
                std::memcpy(a.mint_authority_root, mint_authority_root, 32);
                a.denomination = denomination;
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
        auto* tab = static_cast<UTXO*>([round_.utxos_buf contents]);
        for (uint32_t i = 0; i < kUtxoSlots; ++i) {
            UTXO& s = tab[i];
            if ((s.status & kUtxoOccupied) == 0u) {
                std::memset(&s, 0, sizeof(s));
                std::memcpy(s.utxo_id,    utxo_id,    32);
                std::memcpy(s.asset_id,   asset_id,   32);
                s.amount_lo  = amount_lo;
                s.amount_hi  = amount_hi;
                std::memcpy(s.owner_root, owner_root, 32);
                s.locktime   = locktime;
                s.threshold  = threshold;
                s.status     = kUtxoOccupied;
                return i;
            }
        }
        return 0xFFFFFFFFu;
    }

    XVMTransitionResult run_until_done(XVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return XVMTransitionResult{};

        // Stamp current counts into desc and push to the device.
        round_.desc.tx_count          = round_.tx_count;
        round_.desc.input_count       = round_.input_count;
        round_.desc.output_count      = round_.output_count;
        round_.desc.asset_op_count    = round_.asset_op_count;
        round_.desc.input_batch_count  = round_.input_batch_count;
        round_.desc.output_batch_count = round_.output_batch_count;
        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(XVMRoundDescriptor));

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

        auto dispatch = [&](id<MTLComputePipelineState> pso,
                            void(^bind)(id<MTLComputeCommandEncoder>)) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso];
            bind(enc);
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        };

        uint32_t utxo_count_v       = kUtxoSlots;
        uint32_t bloom_bit_count_v  = kBloomBits;
        uint32_t cuckoo_bucket_v    = kCuckooBuckets;
        uint32_t input_batch_v      = round_.input_batch_count;
        uint32_t output_batch_v     = round_.output_batch_count;
        uint32_t outputs_count_v    = round_.output_count;
        uint32_t asset_count_v      = kAssetSlots;
        uint32_t asset_op_count_v   = round_.asset_op_count;
        uint32_t marker_count_v     = kExportMarkerSlots;
        uint32_t tx_count_v         = round_.tx_count;

        // 1) membership rebuild — re-seeds cuckoo from utxos.
        dispatch(membership_pso_, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.utxos_buf  offset:0 atIndex:0];
            [enc setBuffer:round_.bloom_buf  offset:0 atIndex:1];
            [enc setBuffer:round_.cuckoo_buf offset:0 atIndex:2];
            [enc setBytes:&utxo_count_v      length:sizeof(utxo_count_v)      atIndex:3];
            [enc setBytes:&bloom_bit_count_v length:sizeof(bloom_bit_count_v) atIndex:4];
            [enc setBytes:&cuckoo_bucket_v   length:sizeof(cuckoo_bucket_v)   atIndex:5];
        });

        // 2) utxo transition
        dispatch(utxo_pso_, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.desc_buf            offset:0 atIndex:0];
            [enc setBuffer:round_.txs_buf             offset:0 atIndex:1];
            [enc setBuffer:round_.input_batches_buf   offset:0 atIndex:2];
            [enc setBuffer:round_.output_batches_buf  offset:0 atIndex:3];
            [enc setBuffer:round_.inputs_buf          offset:0 atIndex:4];
            [enc setBuffer:round_.outputs_buf         offset:0 atIndex:5];
            [enc setBuffer:round_.utxos_buf           offset:0 atIndex:6];
            [enc setBuffer:round_.bloom_buf           offset:0 atIndex:7];
            [enc setBuffer:round_.cuckoo_buf          offset:0 atIndex:8];
            [enc setBuffer:round_.inputs_consumed_buf offset:0 atIndex:9];
            [enc setBuffer:round_.outputs_created_buf offset:0 atIndex:10];
            [enc setBytes:&utxo_count_v       length:sizeof(utxo_count_v)       atIndex:11];
            [enc setBytes:&bloom_bit_count_v  length:sizeof(bloom_bit_count_v)  atIndex:12];
            [enc setBytes:&cuckoo_bucket_v    length:sizeof(cuckoo_bucket_v)    atIndex:13];
            [enc setBytes:&input_batch_v      length:sizeof(input_batch_v)      atIndex:14];
            [enc setBytes:&output_batch_v     length:sizeof(output_batch_v)     atIndex:15];
            [enc setBytes:&outputs_count_v    length:sizeof(outputs_count_v)    atIndex:16];
        });

        // 3) asset transition
        dispatch(asset_pso_, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.desc_buf          offset:0 atIndex:0];
            [enc setBuffer:round_.txs_buf           offset:0 atIndex:1];
            [enc setBuffer:round_.asset_ops_buf     offset:0 atIndex:2];
            [enc setBuffer:round_.assets_buf        offset:0 atIndex:3];
            [enc setBuffer:round_.markers_buf       offset:0 atIndex:4];
            [enc setBuffer:round_.asset_applied_buf offset:0 atIndex:5];
            [enc setBuffer:round_.exports_buf       offset:0 atIndex:6];
            [enc setBuffer:round_.imports_buf       offset:0 atIndex:7];
            [enc setBuffer:round_.minted_lo_buf     offset:0 atIndex:8];
            [enc setBuffer:round_.minted_hi_buf     offset:0 atIndex:9];
            [enc setBuffer:round_.burned_lo_buf     offset:0 atIndex:10];
            [enc setBuffer:round_.burned_hi_buf     offset:0 atIndex:11];
            [enc setBytes:&asset_count_v    length:sizeof(asset_count_v)    atIndex:12];
            [enc setBytes:&asset_op_count_v length:sizeof(asset_op_count_v) atIndex:13];
            [enc setBytes:&marker_count_v   length:sizeof(marker_count_v)   atIndex:14];
        });

        // 4) roots
        dispatch(roots_pso_, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:round_.desc_buf   offset:0 atIndex:0];
            [enc setBuffer:round_.txs_buf    offset:0 atIndex:1];
            [enc setBuffer:round_.utxos_buf  offset:0 atIndex:2];
            [enc setBuffer:round_.assets_buf offset:0 atIndex:3];
            [enc setBuffer:round_.result_buf offset:0 atIndex:4];
            [enc setBytes:&tx_count_v    length:sizeof(tx_count_v)    atIndex:5];
            [enc setBytes:&utxo_count_v  length:sizeof(utxo_count_v)  atIndex:6];
            [enc setBytes:&asset_count_v length:sizeof(asset_count_v) atIndex:7];
        });

        [cmd commit];
        [cmd waitUntilCompleted];

        XVMTransitionResult* res = static_cast<XVMTransitionResult*>([round_.result_buf contents]);
        uint32_t inputs_consumed = read_u32(round_.inputs_consumed_buf);
        uint32_t outputs_created = read_u32(round_.outputs_created_buf);
        uint32_t asset_applied   = read_u32(round_.asset_applied_buf);
        uint32_t exports_n       = read_u32(round_.exports_buf);
        uint32_t imports_n       = read_u32(round_.imports_buf);
        uint32_t minted_lo32     = read_u32(round_.minted_lo_buf);
        uint32_t minted_hi32     = read_u32(round_.minted_hi_buf);
        uint32_t burned_lo32     = read_u32(round_.burned_lo_buf);
        uint32_t burned_hi32     = read_u32(round_.burned_hi_buf);
        res->inputs_consumed   = inputs_consumed;
        res->outputs_created   = outputs_created;
        res->asset_ops_applied = asset_applied;
        res->export_markers    = exports_n;
        res->import_verified   = imports_n;
        res->total_minted_lo   = (uint64_t(minted_hi32) << 32) | uint64_t(minted_lo32);
        res->total_minted_hi   = 0;
        res->total_burned_lo   = (uint64_t(burned_hi32) << 32) | uint64_t(burned_lo32);
        res->total_burned_hi   = 0;
        return *res;
    }

    XVMTransitionResult poll_round_result(XVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return XVMTransitionResult{};
        return *static_cast<const XVMTransitionResult*>([round_.result_buf contents]);
    }

    void end_round(XVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_ = Round{};
    }

private:
    bool check_handle(XVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(XVMRoundHandle h) const { return check_handle(h); }

    static void zero_u32_buf(id<MTLBuffer> b) {
        *static_cast<uint32_t*>([b contents]) = 0u;
    }
    static uint32_t read_u32(id<MTLBuffer> b) {
        return *static_cast<uint32_t*>([b contents]);
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> membership_pso_;
    id<MTLComputePipelineState> utxo_pso_;
    id<MTLComputePipelineState> asset_pso_;
    id<MTLComputePipelineState> roots_pso_;
    std::string device_name_str_;
    Round round_;
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<XVMGPUEngine> XVMGPUEngine::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_xvm_metallib(device);
        if (!lib) lib = compile_xvm_library(device);
        if (!lib) return nullptr;

        NSError* err = nil;
        auto fn = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> f = [lib newFunctionWithName:name];
            if (!f) {
                std::fprintf(stderr, "XVM kernel %s not found in library\n",
                             [name UTF8String]);
                return nil;
            }
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&err];
            if (!p && err)
                std::fprintf(stderr, "PSO compile error for %s: %s\n",
                             [name UTF8String], [[err localizedDescription] UTF8String]);
            return p;
        };
        id<MTLComputePipelineState> m_pso = fn(@"xvm_membership_rebuild");
        id<MTLComputePipelineState> u_pso = fn(@"xvm_utxo_transition");
        id<MTLComputePipelineState> a_pso = fn(@"xvm_asset_transition");
        id<MTLComputePipelineState> r_pso = fn(@"xvm_root_update");
        if (!m_pso || !u_pso || !a_pso || !r_pso) return nullptr;

        return std::unique_ptr<XVMGPUEngine>(
            new XVMGPUEngineMetal(device, queue, m_pso, u_pso, a_pso, r_pso,
                                  [device name]));
    }
}

}  // namespace xvm::gpu
