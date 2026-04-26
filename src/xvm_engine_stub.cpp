// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// xvm_engine_stub.cpp — CPU-only fallback for XVMGPUEngine::create().
// Linked when neither Metal nor CUDA is enabled, so tests still build.
// The stub returns nullptr, exercising the CPU-only test paths.

#include "lux/xvm/xvm_gpu_engine.hpp"

namespace xvm::gpu {

#if !defined(__APPLE__)
std::unique_ptr<XVMGPUEngine> XVMGPUEngine::create() {
    return nullptr;
}
#else
__attribute__((weak)) std::unique_ptr<XVMGPUEngine> XVMGPUEngine::create() {
    return nullptr;
}
#endif

}  // namespace xvm::gpu
