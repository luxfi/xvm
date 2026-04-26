// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_cuda_launchers.cu — host-side <<<1,1>>> launchers for the four XVM
// CUDA kernel entry points. Kept in its own translation unit so the host
// driver (xvm_gpu_engine_cuda.cpp) stays pure C++ and can be compiled by
// the host C++ compiler without nvcc.

#include "xvm_kernels_common.cuh"

#include "lux/xvm/xvm_gpu_layout.hpp"

namespace xvm::gpu {

namespace cuda_decl {
extern "C" __global__ void xvm_membership_rebuild(
    ::xvm::cuda::UTXO*, uint8_t*, ::xvm::cuda::CuckooEntry*,
    uint32_t, uint32_t, uint32_t);

extern "C" __global__ void xvm_utxo_transition(
    const ::xvm::cuda::XVMRoundDescriptor*, ::xvm::cuda::XvmTx*,
    const ::xvm::cuda::InputBatch*, const ::xvm::cuda::OutputBatch*,
    const uint8_t*, const ::xvm::cuda::UTXO*, ::xvm::cuda::UTXO*,
    uint8_t*, ::xvm::cuda::CuckooEntry*,
    uint32_t*, uint32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

extern "C" __global__ void xvm_asset_transition(
    const ::xvm::cuda::XVMRoundDescriptor*, ::xvm::cuda::XvmTx*,
    const ::xvm::cuda::AssetOp*, ::xvm::cuda::Asset*,
    ::xvm::cuda::AtomicExportMarker*,
    uint32_t*, uint32_t*, uint32_t*, uint64_t*, uint64_t*,
    uint32_t, uint32_t, uint32_t);

extern "C" __global__ void xvm_root_update(
    const ::xvm::cuda::XVMRoundDescriptor*, const ::xvm::cuda::XvmTx*,
    const ::xvm::cuda::UTXO*, const ::xvm::cuda::Asset*,
    ::xvm::cuda::XVMTransitionResult*,
    uint32_t, uint32_t, uint32_t);
}  // namespace cuda_decl

void launch_xvm_membership_rebuild(
    UTXO* utxos, uint8_t* bloom_bits, CuckooEntry* cuckoo,
    uint32_t utxo_count, uint32_t bloom_bit_count, uint32_t cuckoo_bucket_count)
{
    cuda_decl::xvm_membership_rebuild<<<1, 1>>>(
        reinterpret_cast<::xvm::cuda::UTXO*>(utxos),
        bloom_bits,
        reinterpret_cast<::xvm::cuda::CuckooEntry*>(cuckoo),
        utxo_count, bloom_bit_count, cuckoo_bucket_count);
}

void launch_xvm_utxo_transition(
    const XVMRoundDescriptor* desc, XvmTx* txs,
    const InputBatch* input_batches, const OutputBatch* output_batches,
    const uint8_t* inputs, const UTXO* outputs, UTXO* utxos,
    uint8_t* bloom_bits, CuckooEntry* cuckoo,
    uint32_t* inputs_consumed_out, uint32_t* outputs_created_out,
    uint32_t utxo_count, uint32_t bloom_bit_count, uint32_t cuckoo_bucket_count,
    uint32_t input_batch_count, uint32_t output_batch_count, uint32_t outputs_count)
{
    cuda_decl::xvm_utxo_transition<<<1, 1>>>(
        reinterpret_cast<const ::xvm::cuda::XVMRoundDescriptor*>(desc),
        reinterpret_cast<::xvm::cuda::XvmTx*>(txs),
        reinterpret_cast<const ::xvm::cuda::InputBatch*>(input_batches),
        reinterpret_cast<const ::xvm::cuda::OutputBatch*>(output_batches),
        inputs,
        reinterpret_cast<const ::xvm::cuda::UTXO*>(outputs),
        reinterpret_cast<::xvm::cuda::UTXO*>(utxos),
        bloom_bits,
        reinterpret_cast<::xvm::cuda::CuckooEntry*>(cuckoo),
        inputs_consumed_out, outputs_created_out,
        utxo_count, bloom_bit_count, cuckoo_bucket_count,
        input_batch_count, output_batch_count, outputs_count);
}

void launch_xvm_asset_transition(
    const XVMRoundDescriptor* desc, XvmTx* txs,
    const AssetOp* asset_ops, Asset* assets, AtomicExportMarker* markers,
    uint32_t* applied_out, uint32_t* exports_out, uint32_t* imports_out,
    uint64_t* minted_out, uint64_t* burned_out,
    uint32_t asset_count, uint32_t asset_op_count, uint32_t marker_count)
{
    cuda_decl::xvm_asset_transition<<<1, 1>>>(
        reinterpret_cast<const ::xvm::cuda::XVMRoundDescriptor*>(desc),
        reinterpret_cast<::xvm::cuda::XvmTx*>(txs),
        reinterpret_cast<const ::xvm::cuda::AssetOp*>(asset_ops),
        reinterpret_cast<::xvm::cuda::Asset*>(assets),
        reinterpret_cast<::xvm::cuda::AtomicExportMarker*>(markers),
        applied_out, exports_out, imports_out, minted_out, burned_out,
        asset_count, asset_op_count, marker_count);
}

void launch_xvm_root_update(
    const XVMRoundDescriptor* desc, const XvmTx* txs,
    const UTXO* utxos, const Asset* assets, XVMTransitionResult* result,
    uint32_t tx_count, uint32_t utxo_count, uint32_t asset_count)
{
    cuda_decl::xvm_root_update<<<1, 1>>>(
        reinterpret_cast<const ::xvm::cuda::XVMRoundDescriptor*>(desc),
        reinterpret_cast<const ::xvm::cuda::XvmTx*>(txs),
        reinterpret_cast<const ::xvm::cuda::UTXO*>(utxos),
        reinterpret_cast<const ::xvm::cuda::Asset*>(assets),
        reinterpret_cast<::xvm::cuda::XVMTransitionResult*>(result),
        tx_count, utxo_count, asset_count);
}

}  // namespace xvm::gpu
