# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# XVM (X-Chain Asset VM) GPU substrate.
# v0.54: layout + CPU reference oracle.
# v0.55: CUDA + Metal transition kernels with cross-backend determinism.

cmake_minimum_required(VERSION 3.20)

project(lux-xvm
    VERSION 0.54.0
    DESCRIPTION "GPU-native X-Chain UTXO transition substrate"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_XVM_BUILD_TESTS "Build XVM tests" ON)

# ---- core library (host code: layout + CPU reference) ---------------------

add_library(lux_xvm_core STATIC
    src/xvm_cpu_reference.cpp
)

target_include_directories(lux_xvm_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(lux_xvm_core PUBLIC cxx_std_20)

set_target_properties(lux_xvm_core PROPERTIES
    OUTPUT_NAME "lux_xvm_core"
    POSITION_INDEPENDENT_CODE ON
)

# ---- tests ----------------------------------------------------------------

if(LUX_XVM_BUILD_TESTS)
    enable_testing()

    # Layout test — host-only.
    add_executable(xvm-layout-test test/xvm_layout_test.cpp)
    target_link_libraries(xvm-layout-test PRIVATE lux_xvm_core)
    target_compile_features(xvm-layout-test PRIVATE cxx_std_20)
    add_test(NAME xvm-layout-test COMMAND xvm-layout-test)
endif()

message(STATUS "")
message(STATUS "=== Lux XVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_XVM_BUILD_TESTS}")
message(STATUS "")
