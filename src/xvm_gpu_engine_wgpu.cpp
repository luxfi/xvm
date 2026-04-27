// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_gpu_engine_wgpu.cpp — wgpu-native / Dawn driver for XVMGPUEngine.
//
// One round = four sequential compute dispatches in canonical order:
//   1. xvm_membership_rebuild
//   2. xvm_utxo_transition
//   3. xvm_asset_transition
//   4. xvm_root_update
//
// Each dispatch is a single thread (workgroup_size 1, dispatch 1x1x1) — the
// kernels do canonical in-order traversal of their op streams, identical
// to the Metal/CUDA layout.
//
// The shader source is compiled at engine creation time from the
// concatenated WGSL files (xvm_kernels_common.wgsl + xvm_membership.wgsl +
// xvm_utxo.wgsl + xvm_asset.wgsl + xvm_roots.wgsl). All buffers are mapped
// as storage buffers; uniform parameter blocks are tiny structs uploaded
// each round. Determinism is byte-for-byte against the CPU reference and
// the Metal/CUDA backends.

#include "lux/xvm/xvm_gpu_engine.hpp"

#if defined(LUX_XVM_HAS_WEBGPU)

#include <webgpu.h>

#include "xvm_wgsl_sources.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace xvm::gpu {

namespace {

// =============================================================================
// Async helpers — wgpu-native uses futures; we synchronise via WaitAny.
// =============================================================================

WGPUStringView mkstr(const char* s) {
    WGPUStringView v{};
    v.data = s;
    v.length = (s ? std::strlen(s) : 0);
    return v;
}

// Adapter request (synchronous wait).
WGPUAdapter request_adapter(WGPUInstance inst) {
    struct Slot {
        WGPUAdapter adapter = nullptr;
        bool done = false;
        std::string msg;
    } slot;
    WGPURequestAdapterOptions options{};
    WGPURequestAdapterCallbackInfo cb{};
    cb.nextInChain = nullptr;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
                     WGPUStringView message, void* ud1, void* /*ud2*/) {
        auto* s = static_cast<Slot*>(ud1);
        if (status == WGPURequestAdapterStatus_Success) {
            s->adapter = adapter;
        } else if (message.data) {
            s->msg.assign(message.data, message.length);
        }
        s->done = true;
    };
    cb.userdata1 = &slot;
    cb.userdata2 = nullptr;
    WGPUFuture fut = wgpuInstanceRequestAdapter(inst, &options, cb);
    WGPUFutureWaitInfo info{};
    info.future = fut;
    info.completed = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!slot.done && std::chrono::steady_clock::now() < deadline) {
        wgpuInstanceProcessEvents(inst);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!slot.adapter && !slot.msg.empty()) {
        std::fprintf(stderr, "wgpu adapter request: %s\n", slot.msg.c_str());
    }
    return slot.adapter;
}

WGPUDevice request_device(WGPUInstance inst, WGPUAdapter adapter) {
    struct Slot {
        WGPUDevice device = nullptr;
        bool done = false;
        std::string msg;
    } slot;

    // Query the adapter's natively-supported limits and request them all on
    // the device. Without this, wgpu-native defaults the device to the
    // minimum WebGPU spec (8 storage buffers per stage), and the
    // xvm_utxo_transition pipeline binds 10 storage buffers.
    WGPULimits adapter_limits{};
    wgpuAdapterGetLimits(adapter, &adapter_limits);

    WGPUDeviceDescriptor desc{};
    desc.label = mkstr("xvm-wgpu-device");
    desc.requiredLimits = &adapter_limits;
    desc.uncapturedErrorCallbackInfo.callback =
        [](WGPUDevice const*, WGPUErrorType type, WGPUStringView message,
           void* /*ud1*/, void* /*ud2*/) {
            std::fprintf(stderr,
                         "[wgpu uncaptured error type=%u] %.*s\n",
                         (unsigned)type,
                         (int)message.length,
                         message.data ? message.data : "");
        };
    WGPURequestDeviceCallbackInfo cb{};
    cb.nextInChain = nullptr;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device,
                     WGPUStringView message, void* ud1, void* /*ud2*/) {
        auto* s = static_cast<Slot*>(ud1);
        if (status == WGPURequestDeviceStatus_Success) {
            s->device = device;
        } else if (message.data) {
            s->msg.assign(message.data, message.length);
        }
        s->done = true;
    };
    cb.userdata1 = &slot;
    cb.userdata2 = nullptr;
    WGPUFuture fut = wgpuAdapterRequestDevice(adapter, &desc, cb);
    (void)fut;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!slot.done && std::chrono::steady_clock::now() < deadline) {
        wgpuInstanceProcessEvents(inst);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!slot.device && !slot.msg.empty()) {
        std::fprintf(stderr, "wgpu device request: %s\n", slot.msg.c_str());
    }
    return slot.device;
}

// Map a buffer for read and copy contents out (synchronous wait).
bool buffer_read(WGPUInstance inst, WGPUBuffer buf, size_t bytes, void* dst) {
    struct Slot { bool done = false; bool ok = false; std::string msg; } slot;
    WGPUBufferMapCallbackInfo cb{};
    cb.nextInChain = nullptr;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView message,
                     void* ud1, void* /*ud2*/) {
        auto* s = static_cast<Slot*>(ud1);
        s->ok = (status == WGPUMapAsyncStatus_Success);
        if (!s->ok && message.data) s->msg.assign(message.data, message.length);
        s->done = true;
    };
    cb.userdata1 = &slot;
    cb.userdata2 = nullptr;
    WGPUFuture fut = wgpuBufferMapAsync(buf, WGPUMapMode_Read, 0, bytes, cb);
    (void)fut;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!slot.done && std::chrono::steady_clock::now() < deadline) {
        wgpuInstanceProcessEvents(inst);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!slot.ok) {
        if (!slot.msg.empty())
            std::fprintf(stderr, "wgpu buffer map: %s\n", slot.msg.c_str());
        return false;
    }
    const void* src = wgpuBufferGetConstMappedRange(buf, 0, bytes);
    if (!src) { wgpuBufferUnmap(buf); return false; }
    std::memcpy(dst, src, bytes);
    wgpuBufferUnmap(buf);
    return true;
}

// =============================================================================
// Shader source loader — concatenates the WGSL files in dependency order.
// =============================================================================

std::string load_wgsl_source() {
    // Primary: use embedded raw-string literals (CMake-generated header).
    std::string combined;
    combined += kXVMWGSL_Common;     combined += "\n";
    combined += kXVMWGSL_Membership; combined += "\n";
    combined += kXVMWGSL_Utxo;       combined += "\n";
    combined += kXVMWGSL_Asset;      combined += "\n";
    combined += kXVMWGSL_Roots;      combined += "\n";
    if (!combined.empty()) return combined;

    // Fallback: load from filesystem (development path).
    std::filesystem::path here = std::filesystem::path(__FILE__).parent_path();
    std::filesystem::path candidates_dir[] = {
        here,
        std::filesystem::current_path(),
        std::filesystem::current_path() / "src",
        std::filesystem::current_path() / "xvm" / "src",
        std::filesystem::current_path().parent_path() / "src",
    };
    auto load_file = [](const std::filesystem::path& p) -> std::string {
        std::ifstream f(p);
        if (!f) return {};
        std::ostringstream ss; ss << f.rdbuf();
        return ss.str();
    };
    const char* files[] = {
        "xvm_kernels_common.wgsl",
        "xvm_membership.wgsl",
        "xvm_utxo.wgsl",
        "xvm_asset.wgsl",
        "xvm_roots.wgsl",
    };
    for (const auto& dir : candidates_dir) {
        std::string c;
        bool ok = true;
        for (const char* f : files) {
            auto p = dir / f;
            std::string s = load_file(p);
            if (s.empty()) { ok = false; break; }
            c += s;
            c += "\n";
        }
        if (ok) return c;
    }
    return {};
}

WGPUShaderModule create_shader_module(WGPUDevice device, const std::string& src) {
    WGPUShaderSourceWGSL wgsl{};
    wgsl.chain.next = nullptr;
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = mkstr(src.c_str());
    WGPUShaderModuleDescriptor desc{};
    desc.nextInChain = &wgsl.chain;
    desc.label = mkstr("xvm-wgsl");
    return wgpuDeviceCreateShaderModule(device, &desc);
}

WGPUComputePipeline create_pipeline(WGPUDevice device, WGPUShaderModule mod,
                                    const char* entry)
{
    WGPUComputePipelineDescriptor desc{};
    desc.label = mkstr(entry);
    desc.compute.module = mod;
    desc.compute.entryPoint = mkstr(entry);
    desc.compute.constantCount = 0;
    desc.compute.constants = nullptr;
    return wgpuDeviceCreateComputePipeline(device, &desc);
}

// =============================================================================
// Engine
// =============================================================================

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

// UBO/storage layout for params — must match the WGSL struct ordering.
// All uniform/storage atomic structs are 4-byte-aligned per WGSL rules; we
// hand-pack them because we need no_padding equivalence.
struct MembershipParamsUBO {
    uint32_t utxo_count;
    uint32_t bloom_bit_count;
    uint32_t cuckoo_bucket_count;
    uint32_t pad;
};

struct UtxoParamsSSBO {
    uint32_t utxo_count;
    uint32_t bloom_bit_count;
    uint32_t cuckoo_bucket_count;
    uint32_t input_batch_count;
    uint32_t output_batch_count;
    uint32_t outputs_count;
    uint32_t inputs_consumed;     // atomic<u32>
    uint32_t outputs_created;     // atomic<u32>
};

struct AssetParamsSSBO {
    uint32_t asset_count;
    uint32_t asset_op_count;
    uint32_t marker_count;
    uint32_t pad;
    uint32_t applied;             // atomic<u32>
    uint32_t exports_n;           // atomic<u32>
    uint32_t imports_n;           // atomic<u32>
    uint32_t minted_lo;           // atomic<u32>
    uint32_t minted_hi;           // atomic<u32>
    uint32_t burned_lo;           // atomic<u32>
    uint32_t burned_hi;           // atomic<u32>
    uint32_t _pad2;               // align to 16
};

struct RootsParamsUBO {
    uint32_t tx_count;
    uint32_t utxo_count;
    uint32_t asset_count;
    uint32_t pad;
};

struct Round {
    XVMRoundHandle handle{};
    XVMRoundDescriptor desc{};

    // Storage buffers (mirror Metal driver naming).
    WGPUBuffer desc_buf            = nullptr;
    WGPUBuffer txs_buf             = nullptr;
    WGPUBuffer input_batches_buf   = nullptr;
    WGPUBuffer output_batches_buf  = nullptr;
    WGPUBuffer inputs_buf          = nullptr;
    WGPUBuffer outputs_buf         = nullptr;
    WGPUBuffer asset_ops_buf       = nullptr;
    WGPUBuffer utxos_buf           = nullptr;
    WGPUBuffer assets_buf          = nullptr;
    WGPUBuffer markers_buf         = nullptr;
    WGPUBuffer bloom_buf           = nullptr;
    WGPUBuffer cuckoo_buf          = nullptr;
    WGPUBuffer result_buf          = nullptr;
    WGPUBuffer mb_params_buf       = nullptr;
    WGPUBuffer ut_params_buf       = nullptr;
    WGPUBuffer as_params_buf       = nullptr;
    WGPUBuffer rt_params_buf       = nullptr;

    // Readback staging buffers (MAP_READ | COPY_DST).
    WGPUBuffer result_readback     = nullptr;
    WGPUBuffer ut_params_readback  = nullptr;
    WGPUBuffer as_params_readback  = nullptr;

    // Bind groups — created lazily once the pipelines/layouts are known.
    WGPUBindGroup mb_bg = nullptr;
    WGPUBindGroup ut_bg = nullptr;
    WGPUBindGroup as_bg = nullptr;
    WGPUBindGroup rt_bg = nullptr;

    // Per-round host shadow buffers (we hold the entire arena in a vector
    // and queueWriteBuffer it once per round).
    std::vector<XvmTx>      shadow_txs;
    std::vector<InputBatch> shadow_ibs;
    std::vector<OutputBatch> shadow_obs;
    std::vector<uint8_t>    shadow_inputs;     // raw bytes, multiples of 32
    std::vector<UTXO>       shadow_outputs;
    std::vector<AssetOp>    shadow_ops;

    std::vector<UTXO>               shadow_utxos;
    std::vector<Asset>              shadow_assets;
    std::vector<AtomicExportMarker> shadow_markers;

    uint32_t tx_count            = 0;
    uint32_t input_batch_count   = 0;
    uint32_t output_batch_count  = 0;
    uint32_t input_count         = 0;
    uint32_t output_count        = 0;
    uint32_t asset_op_count      = 0;
};

class XVMGPUEngineWGPU final : public XVMGPUEngine {
public:
    XVMGPUEngineWGPU(WGPUInstance inst, WGPUAdapter adapter, WGPUDevice device,
                     WGPUQueue queue, WGPUShaderModule mod,
                     WGPUComputePipeline mb_pso, WGPUComputePipeline ut_pso,
                     WGPUComputePipeline as_pso, WGPUComputePipeline rt_pso,
                     std::string device_name)
        : inst_(inst)
        , adapter_(adapter)
        , device_(device)
        , queue_(queue)
        , mod_(mod)
        , mb_pso_(mb_pso)
        , ut_pso_(ut_pso)
        , as_pso_(as_pso)
        , rt_pso_(rt_pso)
        , device_name_(std::move(device_name))
    {}

    ~XVMGPUEngineWGPU() override {
        if (round_active()) end_round(round_.handle);
        if (rt_pso_) wgpuComputePipelineRelease(rt_pso_);
        if (as_pso_) wgpuComputePipelineRelease(as_pso_);
        if (ut_pso_) wgpuComputePipelineRelease(ut_pso_);
        if (mb_pso_) wgpuComputePipelineRelease(mb_pso_);
        if (mod_)    wgpuShaderModuleRelease(mod_);
        if (queue_)  wgpuQueueRelease(queue_);
        if (device_) wgpuDeviceRelease(device_);
        if (adapter_) wgpuAdapterRelease(adapter_);
        if (inst_)   wgpuInstanceRelease(inst_);
    }

    const char* device_name() const override { return device_name_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    XVMRoundHandle begin_round(const XVMRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return XVMRoundHandle{0};
        round_ = Round{};
        round_.desc = desc;

        round_.shadow_txs.assign(kMaxTxs, {});
        round_.shadow_ibs.assign(kMaxInputBatches, {});
        round_.shadow_obs.assign(kMaxOutputBatches, {});
        round_.shadow_inputs.assign(uint64_t(kMaxInputs) * 32u, 0);
        round_.shadow_outputs.assign(kMaxOutputs, {});
        round_.shadow_ops.assign(kMaxAssetOps, {});
        round_.shadow_utxos.assign(kUtxoSlots, {});
        round_.shadow_assets.assign(kAssetSlots, {});
        round_.shadow_markers.assign(kExportMarkerSlots, {});

        auto mk = [&](size_t bytes, WGPUBufferUsage usage,
                      const char* label) -> WGPUBuffer {
            WGPUBufferDescriptor d{};
            d.label = mkstr(label);
            d.size = bytes;
            d.usage = usage;
            d.mappedAtCreation = WGPU_FALSE;
            return wgpuDeviceCreateBuffer(device_, &d);
        };
        const auto STORAGE = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        const auto UNIFORM = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        const auto READBACK = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;

        round_.desc_buf            = mk(sizeof(XVMRoundDescriptor), STORAGE, "desc");
        round_.txs_buf             = mk(sizeof(XvmTx) * kMaxTxs, STORAGE, "txs");
        round_.input_batches_buf   = mk(sizeof(InputBatch)  * kMaxInputBatches, STORAGE, "ibs");
        round_.output_batches_buf  = mk(sizeof(OutputBatch) * kMaxOutputBatches, STORAGE, "obs");
        round_.inputs_buf          = mk(uint64_t(kMaxInputs) * 32u, STORAGE, "inputs");
        round_.outputs_buf         = mk(sizeof(UTXO) * kMaxOutputs, STORAGE, "outputs");
        round_.asset_ops_buf       = mk(sizeof(AssetOp) * kMaxAssetOps, STORAGE, "asset_ops");
        round_.utxos_buf           = mk(sizeof(UTXO) * kUtxoSlots, STORAGE, "utxos");
        round_.assets_buf          = mk(sizeof(Asset) * kAssetSlots, STORAGE, "assets");
        round_.markers_buf         = mk(sizeof(AtomicExportMarker) * kExportMarkerSlots, STORAGE, "markers");
        round_.bloom_buf           = mk(kBloomBytes, STORAGE, "bloom");
        round_.cuckoo_buf          = mk(sizeof(CuckooEntry) * kCuckooSlotsTotal, STORAGE, "cuckoo");
        round_.result_buf          = mk(sizeof(XVMTransitionResult), STORAGE, "result");
        round_.mb_params_buf       = mk(sizeof(MembershipParamsUBO), UNIFORM, "mb_params");
        round_.ut_params_buf       = mk(sizeof(UtxoParamsSSBO), STORAGE, "ut_params");
        round_.as_params_buf       = mk(sizeof(AssetParamsSSBO), STORAGE, "as_params");
        round_.rt_params_buf       = mk(sizeof(RootsParamsUBO), UNIFORM, "rt_params");
        round_.result_readback     = mk(sizeof(XVMTransitionResult), READBACK, "result_rb");
        round_.ut_params_readback  = mk(sizeof(UtxoParamsSSBO), READBACK, "ut_rb");
        round_.as_params_readback  = mk(sizeof(AssetParamsSSBO), READBACK, "as_rb");

        round_.handle = XVMRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_txs(XVMRoundHandle h, std::span<const XvmTx> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || txs.empty()) return;
        uint32_t cap_left = kMaxTxs - round_.tx_count;
        uint32_t take = std::min<uint32_t>(uint32_t(txs.size()), cap_left);
        std::memcpy(round_.shadow_txs.data() + round_.tx_count, txs.data(),
                    take * sizeof(XvmTx));
        round_.tx_count += take;
    }
    void push_input_batches(XVMRoundHandle h,
                            std::span<const InputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxInputBatches - round_.input_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        std::memcpy(round_.shadow_ibs.data() + round_.input_batch_count,
                    batches.data(), take * sizeof(InputBatch));
        round_.input_batch_count += take;
    }
    void push_output_batches(XVMRoundHandle h,
                             std::span<const OutputBatch> batches) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || batches.empty()) return;
        uint32_t cap_left = kMaxOutputBatches - round_.output_batch_count;
        uint32_t take = std::min<uint32_t>(uint32_t(batches.size()), cap_left);
        std::memcpy(round_.shadow_obs.data() + round_.output_batch_count,
                    batches.data(), take * sizeof(OutputBatch));
        round_.output_batch_count += take;
    }
    void push_inputs(XVMRoundHandle h, std::span<const uint8_t> inputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || inputs.empty()) return;
        uint32_t bytes_cap = (kMaxInputs - round_.input_count) * 32u;
        uint32_t take = std::min<uint32_t>(uint32_t(inputs.size()), bytes_cap);
        std::memcpy(round_.shadow_inputs.data() + round_.input_count * 32u,
                    inputs.data(), take);
        round_.input_count += take / 32u;
    }
    void push_outputs(XVMRoundHandle h, std::span<const UTXO> outputs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || outputs.empty()) return;
        uint32_t cap_left = kMaxOutputs - round_.output_count;
        uint32_t take = std::min<uint32_t>(uint32_t(outputs.size()), cap_left);
        std::memcpy(round_.shadow_outputs.data() + round_.output_count,
                    outputs.data(), take * sizeof(UTXO));
        round_.output_count += take;
    }
    void push_asset_ops(XVMRoundHandle h, std::span<const AssetOp> ops) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || ops.empty()) return;
        uint32_t cap_left = kMaxAssetOps - round_.asset_op_count;
        uint32_t take = std::min<uint32_t>(uint32_t(ops.size()), cap_left);
        std::memcpy(round_.shadow_ops.data() + round_.asset_op_count,
                    ops.data(), take * sizeof(AssetOp));
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
        for (uint32_t i = 0; i < 32u; ++i) {
            hash ^= uint64_t(asset_id[i]);
            hash *= 0x100000001b3ULL;
        }
        uint32_t idx = uint32_t(hash) & mask;
        for (uint32_t probe = 0; probe < kAssetSlots; ++probe) {
            Asset& a = round_.shadow_assets[idx];
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
            bool match = std::memcmp(a.asset_id, asset_id, 32) == 0;
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
        for (uint32_t i = 0; i < kUtxoSlots; ++i) {
            UTXO& s = round_.shadow_utxos[i];
            if ((s.status & kUtxoOccupied) == 0u) {
                std::memset(&s, 0, sizeof(s));
                std::memcpy(s.utxo_id, utxo_id, 32);
                std::memcpy(s.asset_id, asset_id, 32);
                s.amount_lo = amount_lo;
                s.amount_hi = amount_hi;
                std::memcpy(s.owner_root, owner_root, 32);
                s.locktime = locktime;
                s.threshold = threshold;
                s.status = kUtxoOccupied;
                return i;
            }
        }
        return 0xFFFFFFFFu;
    }

    XVMTransitionResult run_until_done(XVMRoundHandle h, std::size_t /*max*/) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return XVMTransitionResult{};

        // Stamp counts.
        round_.desc.tx_count           = round_.tx_count;
        round_.desc.input_count        = round_.input_count;
        round_.desc.output_count       = round_.output_count;
        round_.desc.asset_op_count     = round_.asset_op_count;
        round_.desc.input_batch_count  = round_.input_batch_count;
        round_.desc.output_batch_count = round_.output_batch_count;

        // Upload all shadow buffers to the GPU once.
        wgpuQueueWriteBuffer(queue_, round_.desc_buf, 0,
                             &round_.desc, sizeof(XVMRoundDescriptor));
        wgpuQueueWriteBuffer(queue_, round_.txs_buf, 0,
                             round_.shadow_txs.data(),
                             sizeof(XvmTx) * kMaxTxs);
        wgpuQueueWriteBuffer(queue_, round_.input_batches_buf, 0,
                             round_.shadow_ibs.data(),
                             sizeof(InputBatch) * kMaxInputBatches);
        wgpuQueueWriteBuffer(queue_, round_.output_batches_buf, 0,
                             round_.shadow_obs.data(),
                             sizeof(OutputBatch) * kMaxOutputBatches);
        wgpuQueueWriteBuffer(queue_, round_.inputs_buf, 0,
                             round_.shadow_inputs.data(),
                             round_.shadow_inputs.size());
        wgpuQueueWriteBuffer(queue_, round_.outputs_buf, 0,
                             round_.shadow_outputs.data(),
                             sizeof(UTXO) * kMaxOutputs);
        wgpuQueueWriteBuffer(queue_, round_.asset_ops_buf, 0,
                             round_.shadow_ops.data(),
                             sizeof(AssetOp) * kMaxAssetOps);
        wgpuQueueWriteBuffer(queue_, round_.utxos_buf, 0,
                             round_.shadow_utxos.data(),
                             sizeof(UTXO) * kUtxoSlots);
        wgpuQueueWriteBuffer(queue_, round_.assets_buf, 0,
                             round_.shadow_assets.data(),
                             sizeof(Asset) * kAssetSlots);
        wgpuQueueWriteBuffer(queue_, round_.markers_buf, 0,
                             round_.shadow_markers.data(),
                             sizeof(AtomicExportMarker) * kExportMarkerSlots);
        // Zero bloom + cuckoo.
        std::vector<uint8_t> zero_bloom(kBloomBytes, 0);
        wgpuQueueWriteBuffer(queue_, round_.bloom_buf, 0,
                             zero_bloom.data(), kBloomBytes);
        std::vector<CuckooEntry> zero_cuckoo(kCuckooSlotsTotal);
        wgpuQueueWriteBuffer(queue_, round_.cuckoo_buf, 0,
                             zero_cuckoo.data(),
                             sizeof(CuckooEntry) * kCuckooSlotsTotal);
        XVMTransitionResult zero_result{};
        wgpuQueueWriteBuffer(queue_, round_.result_buf, 0,
                             &zero_result, sizeof(zero_result));

        MembershipParamsUBO mb_p{
            kUtxoSlots, kBloomBits, kCuckooBuckets, 0
        };
        wgpuQueueWriteBuffer(queue_, round_.mb_params_buf, 0, &mb_p, sizeof(mb_p));

        UtxoParamsSSBO ut_p{
            kUtxoSlots, kBloomBits, kCuckooBuckets,
            round_.input_batch_count, round_.output_batch_count,
            round_.output_count,
            0, 0
        };
        wgpuQueueWriteBuffer(queue_, round_.ut_params_buf, 0, &ut_p, sizeof(ut_p));

        AssetParamsSSBO as_p{
            kAssetSlots, round_.asset_op_count, kExportMarkerSlots, 0,
            0, 0, 0, 0, 0, 0, 0, 0
        };
        wgpuQueueWriteBuffer(queue_, round_.as_params_buf, 0, &as_p, sizeof(as_p));

        RootsParamsUBO rt_p{
            round_.tx_count, kUtxoSlots, kAssetSlots, 0
        };
        wgpuQueueWriteBuffer(queue_, round_.rt_params_buf, 0, &rt_p, sizeof(rt_p));

        // Build bind groups. Layouts come from the pipelines.
        if (!round_.mb_bg) round_.mb_bg = make_membership_bg();
        if (!round_.ut_bg) round_.ut_bg = make_utxo_bg();
        if (!round_.as_bg) round_.as_bg = make_asset_bg();
        if (!round_.rt_bg) round_.rt_bg = make_roots_bg();
        if (!round_.mb_bg || !round_.ut_bg || !round_.as_bg || !round_.rt_bg) {
            std::fprintf(stderr, "WGPU bind group creation failed\n");
            return XVMTransitionResult{};
        }

        // Encode and submit.
        WGPUCommandEncoderDescriptor cd{};
        cd.label = mkstr("xvm-cmd");
        WGPUCommandEncoder cmd = wgpuDeviceCreateCommandEncoder(device_, &cd);

        auto dispatch = [&](WGPUComputePipeline pso, WGPUBindGroup bg,
                            const char* label) {
            WGPUComputePassDescriptor pd{};
            pd.label = mkstr(label);
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(cmd, &pd);
            wgpuComputePassEncoderSetPipeline(pass, pso);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        };

        dispatch(mb_pso_, round_.mb_bg, "membership");
        dispatch(ut_pso_, round_.ut_bg, "utxo");
        dispatch(as_pso_, round_.as_bg, "asset");
        dispatch(rt_pso_, round_.rt_bg, "roots");

        // Copy result + atomic-counter buffers to readback staging.
        wgpuCommandEncoderCopyBufferToBuffer(cmd, round_.result_buf, 0,
                                             round_.result_readback, 0,
                                             sizeof(XVMTransitionResult));
        wgpuCommandEncoderCopyBufferToBuffer(cmd, round_.ut_params_buf, 0,
                                             round_.ut_params_readback, 0,
                                             sizeof(UtxoParamsSSBO));
        wgpuCommandEncoderCopyBufferToBuffer(cmd, round_.as_params_buf, 0,
                                             round_.as_params_readback, 0,
                                             sizeof(AssetParamsSSBO));

        WGPUCommandBufferDescriptor bd{}; bd.label = mkstr("xvm-cb");
        WGPUCommandBuffer cb_obj = wgpuCommandEncoderFinish(cmd, &bd);
        wgpuQueueSubmit(queue_, 1, &cb_obj);
        wgpuCommandBufferRelease(cb_obj);
        wgpuCommandEncoderRelease(cmd);

        // Drain queue (synchronously) by waiting on a sentinel buffer map.
        // The result_readback map call below also flushes the queue.
        XVMTransitionResult result{};
        if (!buffer_read(inst_, round_.result_readback,
                         sizeof(XVMTransitionResult), &result)) {
            return XVMTransitionResult{};
        }
        UtxoParamsSSBO ut_back{};
        buffer_read(inst_, round_.ut_params_readback, sizeof(ut_back), &ut_back);
        AssetParamsSSBO as_back{};
        buffer_read(inst_, round_.as_params_readback, sizeof(as_back), &as_back);

        result.inputs_consumed   = ut_back.inputs_consumed;
        result.outputs_created   = ut_back.outputs_created;
        result.asset_ops_applied = as_back.applied;
        result.export_markers    = as_back.exports_n;
        result.import_verified   = as_back.imports_n;
        // Match Metal's split-low/high pattern: lo = (hi<<32) | lo.
        result.total_minted_lo   = (uint64_t(as_back.minted_hi) << 32)
                                 |  uint64_t(as_back.minted_lo);
        result.total_minted_hi   = 0;
        result.total_burned_lo   = (uint64_t(as_back.burned_hi) << 32)
                                 |  uint64_t(as_back.burned_lo);
        result.total_burned_hi   = 0;
        cached_result_ = result;
        return result;
    }

    XVMTransitionResult poll_round_result(XVMRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle_const(h)) return XVMTransitionResult{};
        return cached_result_;
    }

    void end_round(XVMRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        release_round();
    }

private:
    void release_round() {
        auto rel = [](WGPUBuffer& b) { if (b) { wgpuBufferRelease(b); b = nullptr; } };
        auto relbg = [](WGPUBindGroup& g) { if (g) { wgpuBindGroupRelease(g); g = nullptr; } };
        relbg(round_.mb_bg); relbg(round_.ut_bg); relbg(round_.as_bg); relbg(round_.rt_bg);
        rel(round_.desc_buf);
        rel(round_.txs_buf);
        rel(round_.input_batches_buf);
        rel(round_.output_batches_buf);
        rel(round_.inputs_buf);
        rel(round_.outputs_buf);
        rel(round_.asset_ops_buf);
        rel(round_.utxos_buf);
        rel(round_.assets_buf);
        rel(round_.markers_buf);
        rel(round_.bloom_buf);
        rel(round_.cuckoo_buf);
        rel(round_.result_buf);
        rel(round_.mb_params_buf);
        rel(round_.ut_params_buf);
        rel(round_.as_params_buf);
        rel(round_.rt_params_buf);
        rel(round_.result_readback);
        rel(round_.ut_params_readback);
        rel(round_.as_params_readback);
        round_ = Round{};
    }

    bool check_handle(XVMRoundHandle h) const {
        return h.valid() && h.opaque == round_.handle.opaque;
    }
    bool check_handle_const(XVMRoundHandle h) const { return check_handle(h); }

    WGPUBindGroup make_bg(WGPUBindGroupLayout layout,
                          std::vector<WGPUBindGroupEntry> entries,
                          const char* label) {
        WGPUBindGroupDescriptor d{};
        d.label = mkstr(label);
        d.layout = layout;
        d.entryCount = entries.size();
        d.entries = entries.data();
        return wgpuDeviceCreateBindGroup(device_, &d);
    }

    static WGPUBindGroupEntry buf_entry(uint32_t binding, WGPUBuffer b, size_t bytes) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = b;
        e.offset = 0;
        e.size = bytes;
        return e;
    }

    WGPUBindGroup make_membership_bg() {
        auto layout = wgpuComputePipelineGetBindGroupLayout(mb_pso_, 0);
        std::vector<WGPUBindGroupEntry> entries{
            buf_entry(0, round_.utxos_buf,    sizeof(UTXO) * kUtxoSlots),
            buf_entry(1, round_.bloom_buf,    kBloomBytes),
            buf_entry(2, round_.cuckoo_buf,   sizeof(CuckooEntry) * kCuckooSlotsTotal),
            buf_entry(3, round_.mb_params_buf, sizeof(MembershipParamsUBO)),
        };
        WGPUBindGroup bg = make_bg(layout, std::move(entries), "mb_bg");
        wgpuBindGroupLayoutRelease(layout);
        return bg;
    }

    WGPUBindGroup make_utxo_bg() {
        auto layout = wgpuComputePipelineGetBindGroupLayout(ut_pso_, 0);
        std::vector<WGPUBindGroupEntry> entries{
            buf_entry(0, round_.desc_buf,           sizeof(XVMRoundDescriptor)),
            buf_entry(1, round_.txs_buf,            sizeof(XvmTx) * kMaxTxs),
            buf_entry(2, round_.input_batches_buf,  sizeof(InputBatch) * kMaxInputBatches),
            buf_entry(3, round_.output_batches_buf, sizeof(OutputBatch) * kMaxOutputBatches),
            buf_entry(4, round_.inputs_buf,         uint64_t(kMaxInputs) * 32u),
            buf_entry(5, round_.outputs_buf,        sizeof(UTXO) * kMaxOutputs),
            buf_entry(6, round_.utxos_buf,          sizeof(UTXO) * kUtxoSlots),
            buf_entry(7, round_.bloom_buf,          kBloomBytes),
            buf_entry(8, round_.cuckoo_buf,         sizeof(CuckooEntry) * kCuckooSlotsTotal),
            buf_entry(9, round_.ut_params_buf,      sizeof(UtxoParamsSSBO)),
        };
        WGPUBindGroup bg = make_bg(layout, std::move(entries), "ut_bg");
        wgpuBindGroupLayoutRelease(layout);
        return bg;
    }

    WGPUBindGroup make_asset_bg() {
        auto layout = wgpuComputePipelineGetBindGroupLayout(as_pso_, 0);
        std::vector<WGPUBindGroupEntry> entries{
            buf_entry(0, round_.desc_buf,         sizeof(XVMRoundDescriptor)),
            buf_entry(1, round_.txs_buf,          sizeof(XvmTx) * kMaxTxs),
            buf_entry(2, round_.asset_ops_buf,    sizeof(AssetOp) * kMaxAssetOps),
            buf_entry(3, round_.assets_buf,       sizeof(Asset) * kAssetSlots),
            buf_entry(4, round_.markers_buf,      sizeof(AtomicExportMarker) * kExportMarkerSlots),
            buf_entry(5, round_.as_params_buf,    sizeof(AssetParamsSSBO)),
        };
        WGPUBindGroup bg = make_bg(layout, std::move(entries), "as_bg");
        wgpuBindGroupLayoutRelease(layout);
        return bg;
    }

    WGPUBindGroup make_roots_bg() {
        auto layout = wgpuComputePipelineGetBindGroupLayout(rt_pso_, 0);
        std::vector<WGPUBindGroupEntry> entries{
            buf_entry(0, round_.desc_buf,        sizeof(XVMRoundDescriptor)),
            buf_entry(1, round_.txs_buf,         sizeof(XvmTx) * kMaxTxs),
            buf_entry(2, round_.utxos_buf,       sizeof(UTXO) * kUtxoSlots),
            buf_entry(3, round_.assets_buf,      sizeof(Asset) * kAssetSlots),
            buf_entry(4, round_.result_buf,      sizeof(XVMTransitionResult)),
            buf_entry(5, round_.rt_params_buf,   sizeof(RootsParamsUBO)),
        };
        WGPUBindGroup bg = make_bg(layout, std::move(entries), "rt_bg");
        wgpuBindGroupLayoutRelease(layout);
        return bg;
    }

    WGPUInstance inst_;
    WGPUAdapter adapter_;
    WGPUDevice device_;
    WGPUQueue queue_;
    WGPUShaderModule mod_;
    WGPUComputePipeline mb_pso_;
    WGPUComputePipeline ut_pso_;
    WGPUComputePipeline as_pso_;
    WGPUComputePipeline rt_pso_;
    std::string device_name_;

    Round round_;
    XVMTransitionResult cached_result_{};
    uint64_t next_handle_ = 0;
    mutable std::mutex mu_;
};

}  // namespace

std::unique_ptr<XVMGPUEngine> create_xvm_gpu_engine_wgpu() {
    WGPUInstanceDescriptor inst_desc{};
    WGPUInstance inst = wgpuCreateInstance(&inst_desc);
    if (!inst) return nullptr;
    WGPUAdapter adapter = request_adapter(inst);
    if (!adapter) { wgpuInstanceRelease(inst); return nullptr; }
    WGPUDevice device = request_device(inst, adapter);
    if (!device) {
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(inst);
        return nullptr;
    }
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    if (!queue) {
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(inst);
        return nullptr;
    }
    std::string src = load_wgsl_source();
    if (src.empty()) {
        std::fprintf(stderr, "WGPU: WGSL sources not found\n");
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(inst);
        return nullptr;
    }
    WGPUShaderModule mod = create_shader_module(device, src);
    if (!mod) {
        std::fprintf(stderr, "WGPU: shader module compile failed\n");
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(inst);
        return nullptr;
    }
    WGPUComputePipeline mb_pso = create_pipeline(device, mod, "xvm_membership_rebuild");
    WGPUComputePipeline ut_pso = create_pipeline(device, mod, "xvm_utxo_transition");
    WGPUComputePipeline as_pso = create_pipeline(device, mod, "xvm_asset_transition");
    WGPUComputePipeline rt_pso = create_pipeline(device, mod, "xvm_root_update");
    if (!mb_pso || !ut_pso || !as_pso || !rt_pso) {
        std::fprintf(stderr, "WGPU: pipeline creation failed\n");
        if (rt_pso) wgpuComputePipelineRelease(rt_pso);
        if (as_pso) wgpuComputePipelineRelease(as_pso);
        if (ut_pso) wgpuComputePipelineRelease(ut_pso);
        if (mb_pso) wgpuComputePipelineRelease(mb_pso);
        wgpuShaderModuleRelease(mod);
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(inst);
        return nullptr;
    }
    WGPUAdapterInfo info{};
    wgpuAdapterGetInfo(adapter, &info);
    std::string name = "wgpu";
    if (info.device.data) {
        name.assign(info.device.data, info.device.length);
    }
    wgpuAdapterInfoFreeMembers(info);
    return std::unique_ptr<XVMGPUEngine>(
        new XVMGPUEngineWGPU(inst, adapter, device, queue, mod,
                             mb_pso, ut_pso, as_pso, rt_pso, std::move(name)));
}

}  // namespace xvm::gpu

#else  // !LUX_XVM_HAS_WEBGPU

namespace xvm::gpu {

std::unique_ptr<XVMGPUEngine> create_xvm_gpu_engine_wgpu()
{
    return nullptr;
}

}  // namespace xvm::gpu

#endif  // LUX_XVM_HAS_WEBGPU

// Default factory — wired by the platform (Metal first on Apple, CUDA on
// Linux). When neither is enabled and WGPU is, this is the lone backend.
#if !defined(__APPLE__) && !defined(LUX_HAS_CUDA) && defined(LUX_XVM_HAS_WEBGPU)
namespace xvm::gpu {
std::unique_ptr<XVMGPUEngine> XVMGPUEngine::create() {
    return create_xvm_gpu_engine_wgpu();
}
}
#endif
