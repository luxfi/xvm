# XVM coverage

Tooling: clang `-fprofile-instr-generate -fcoverage-mapping`, `llvm-cov`.

```
rm -rf build-cov
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug \
    -DLUX_XVM_ENABLE_METAL=ON -DLUX_XVM_ENABLE_WGPU=ON \
    -DCMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping -O0 -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fprofile-instr-generate"
cmake --build build-cov -j
cd build-cov
LLVM_PROFILE_FILE="cov.%p.profraw" ctest
xcrun llvm-profdata merge -sparse cov.*.profraw -o cov.profdata
xcrun llvm-cov report -instr-profile=cov.profdata \
    ./xvm-layout-test ./xvm-determinism-test ./xvm-gpu-engine-test \
    -ignore-filename-regex='build|test/|/usr/'
```

## Summary

| Target                  | Lines    | Branches | Functions |
| ----------------------- | -------- | -------- | --------- |
| `src/xvm_cpu_reference.cpp` | 95.15 % | 90.08 % | 97.14 % |
| **Total**               | **95.15 %** | **90.08 %** | **97.14 %** |

GPU drivers (`xvm_gpu_engine.mm`, `xvm_gpu_engine_cuda.cpp`,
`xvm_gpu_engine_wgpu.cpp`) are excluded from line/branch coverage —
they are device-bound and validated end-to-end by the determinism harness.

## Tests

`xvm-layout-test` — 43 tests covering layout invariants, CPU reference
edge cases (locktime, threshold, double-spend, bloom-pass/cuckoo-miss,
already-spent, cuckoo→unoccupied, cuckoo→out-of-range, asset op offset
overrun, asset count overrun, unknown AssetOpKind, u128 carry/borrow,
mode-only invocations, burn-greater-than-supply, empty assets table).

`xvm-gpu-engine-test` — Metal driver smoke tests (round lifecycle,
single-tx transfer, mint, export marker).

`xvm-determinism-test` — 4-way byte-equality between CPU reference,
Metal, CUDA (when present) and WGSL (Dawn / wgpu-native) for:

* Brief workload (10 K UTXOs × 1 K transfers)
* Empty round (deterministic non-zero execution_root)
* Duplicate input rejection
* Mint authority enforcement
* Cross-chain export marker
* Two engines bytewise identical
* Bloom membership: zero false-negatives, FPR < 5 % at 4 K inserts × 10 K probes

Wallclock: ~35 s (M1 Max, single-thread WGSL kernels dominate).

## Uncovered residue

Remaining branches are dead-code defenses in arena-overflow and
arena-empty paths (`utxo_arena_insert` returning `0xFFFFFFFF`, asset and
export-marker tables exhausted, `bloom_bits.empty()`,
`asset.empty()` after `run_reference` auto-grew the arena). Triggering
these would require either fabricating a state with arenas
hand-truncated below `kDefault*Slots` or seeding > 16 384 distinct
UTXOs in a single test, both of which exercise no real codepath beyond
the host guard. The roots-only path (`mode == FullRound` already
covered) and `inputs_have_duplicates` returning false (covered by the
multi-input test) are the meaningful ones.
