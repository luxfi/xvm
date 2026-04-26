# Copyright (C) 2026, Lux Partners Limited. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# XVM (X-Chain Asset VM) GPU substrate.
# v0.54: layout + CPU reference oracle.
# v0.55: CUDA + Metal transition kernels with cross-backend determinism.

cmake_minimum_required(VERSION 3.20)

project(lux-xvm
    VERSION 0.55.0
    DESCRIPTION "GPU-native X-Chain UTXO transition substrate"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(LUX_XVM_BUILD_TESTS "Build XVM tests" ON)
option(LUX_XVM_ENABLE_METAL "Build Metal backend (Apple)" ${APPLE})
option(LUX_XVM_ENABLE_CUDA  "Build CUDA backend"          OFF)

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

# ---- Metal driver (Apple) -------------------------------------------------

if(LUX_XVM_ENABLE_METAL AND APPLE)
    add_library(lux_xvm_metal STATIC
        src/xvm_gpu_engine.mm
    )
    target_include_directories(lux_xvm_metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_compile_features(lux_xvm_metal PUBLIC cxx_std_20)
    target_link_libraries(lux_xvm_metal PUBLIC lux_xvm_core
        "-framework Metal" "-framework Foundation")
    set_target_properties(lux_xvm_metal PROPERTIES
        OUTPUT_NAME "lux_xvm_metal"
        POSITION_INDEPENDENT_CODE ON
    )

    foreach(src
        xvm_kernels_common.h.metal
        xvm_membership.metal
        xvm_utxo.metal
        xvm_asset.metal
        xvm_roots.metal)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/${src}
            ${CMAKE_CURRENT_BINARY_DIR}/src/${src}
            COPYONLY)
    endforeach()

    set(XVM_METAL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/src/xvm_combined.metal)
    add_custom_command(
        OUTPUT ${XVM_METAL_COMBINED}
        COMMAND bash -c "cat ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_kernels_common.h.metal > ${XVM_METAL_COMBINED} && for f in xvm_membership.metal xvm_utxo.metal xvm_asset.metal xvm_roots.metal; do grep -v 'xvm_kernels_common.h.metal' ${CMAKE_CURRENT_SOURCE_DIR}/src/$f >> ${XVM_METAL_COMBINED}; done"
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_kernels_common.h.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_membership.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_utxo.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_asset.metal
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_roots.metal
        VERBATIM)

    set(XVM_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/src/xvm_combined.air)
    set(XVM_METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/src/xvm.metallib)

    add_custom_command(
        OUTPUT ${XVM_METAL_AIR}
        COMMAND xcrun -sdk macosx metal -std=metal3.0
            -Wno-pragma-once-outside-header
            -Wno-unused-const-variable
            -c ${XVM_METAL_COMBINED} -o ${XVM_METAL_AIR}
        DEPENDS ${XVM_METAL_COMBINED}
        VERBATIM)

    add_custom_command(
        OUTPUT ${XVM_METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${XVM_METAL_AIR} -o ${XVM_METAL_LIB}
        DEPENDS ${XVM_METAL_AIR}
        VERBATIM)

    add_custom_target(xvm_metallib ALL DEPENDS ${XVM_METAL_LIB})
    add_dependencies(lux_xvm_metal xvm_metallib)
endif()

# ---- CUDA driver ----------------------------------------------------------

if(LUX_XVM_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(lux_xvm_cuda STATIC
        src/xvm_gpu_engine_cuda.cpp
        src/xvm_cuda_launchers.cu
        src/xvm_membership.cu
        src/xvm_utxo.cu
        src/xvm_asset.cu
        src/xvm_roots.cu
    )
    target_include_directories(lux_xvm_cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    )
    set_target_properties(lux_xvm_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME "lux_xvm_cuda"
    )
    target_link_libraries(lux_xvm_cuda PUBLIC lux_xvm_core)
    target_compile_definitions(lux_xvm_cuda PUBLIC LUX_HAS_CUDA=1)
endif()

# ---- tests ----------------------------------------------------------------

if(LUX_XVM_BUILD_TESTS)
    enable_testing()

    # Layout test — host-only.
    add_executable(xvm-layout-test test/xvm_layout_test.cpp)
    target_link_libraries(xvm-layout-test PRIVATE lux_xvm_core)
    target_compile_features(xvm-layout-test PRIVATE cxx_std_20)
    add_test(NAME xvm-layout-test COMMAND xvm-layout-test)

    # GPU engine test — Apple-only (.mm).
    if(LUX_XVM_ENABLE_METAL AND APPLE)
        add_executable(xvm-gpu-engine-test test/xvm_gpu_engine_test.mm)
        target_link_libraries(xvm-gpu-engine-test PRIVATE lux_xvm_metal)
        target_compile_features(xvm-gpu-engine-test PRIVATE cxx_std_20)
        add_test(NAME xvm-gpu-engine-test COMMAND xvm-gpu-engine-test)
    endif()

    # Cross-backend determinism — links to whichever GPU backend is enabled.
    add_executable(xvm-determinism-test test/xvm_determinism_test.cpp)
    target_compile_features(xvm-determinism-test PRIVATE cxx_std_20)
    if(LUX_XVM_ENABLE_METAL AND APPLE)
        target_link_libraries(xvm-determinism-test PRIVATE lux_xvm_metal)
    elseif(LUX_XVM_ENABLE_CUDA)
        target_link_libraries(xvm-determinism-test PRIVATE lux_xvm_cuda)
    else()
        target_sources(xvm-determinism-test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/xvm_engine_stub.cpp)
        target_link_libraries(xvm-determinism-test PRIVATE lux_xvm_core)
    endif()
    add_test(NAME xvm-determinism-test COMMAND xvm-determinism-test)
endif()

message(STATUS "")
message(STATUS "=== Lux XVM Configuration ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Tests: ${LUX_XVM_BUILD_TESTS}")
message(STATUS "  Metal:       ${LUX_XVM_ENABLE_METAL}")
message(STATUS "  CUDA:        ${LUX_XVM_ENABLE_CUDA}")
message(STATUS "")
