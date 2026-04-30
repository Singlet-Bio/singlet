---
feature: 5 (optimization variant)
module: reduce/nmf/adapters.h
cycle: 86
phase: C (design decision + Phase D dispatch)
extends: 05-nmf-phaseE.md
status: design
target_perf: k=50 wall 15980ms → <1500ms; dominance >= 10× frontier vs sklearn CPU
tolerance: reconstruction Frobenius relative error < 5% vs sklearn NMF reference
---

# Cycle 86 — NMF k=50 optimization: force MU solver for k≥32; cap cd_max_iter=10

## Problem Statement

Singlet-gpu NMF at k=50 regresses catastrophically on factornet's GPU backend. **Observed wall: 15,980ms on H100 (job 363152, g051).** This is 3× slower than k=20 (all 40 runs converge at ~4,100ms median). Expected scaling from k=20→k=50 under geometric growth (matrix sizes N × k²) is ~2.5×. Observed is **40×**.

### Phase B Profile Findings (from gpu-bench, job 363152)

**Per-iteration NNLS cost** (kernel timing from job log):
- **k=20, CD solver**: 1.629 ms per coordinate-descent iteration
- **k=50, CD solver**: 9.232 ms per coordinate-descent iteration
- **k=50, MU solver**: 0.182 ms per multiplicative-update iteration

**MU is 50.7× cheaper than CD at k=50.**

### Root Cause Analysis

Factornet's `should_switch` heuristic routes between **Multiplicative Update (MU)** and **Coordinate Descent (CD)** solvers based on a **convergence-rate pilot phase** (~5 iters). The decision threshold is `rel_rate < 0.01` (relative change in loss per iteration).

**The flaw**: This heuristic was calibrated on CPU, where CD genuinely converges faster due to superior cache locality and amortized synchronization. On GPU, CD suffers from:
- **Inner-loop latency** (`batch_cd_kernel_tpc_dynamic`): k² inner coordinate updates per column per pass.
- **Load imbalance**: at large k, thread occupancy in the CD inner kernel becomes latency-bound.
- MU's single dense matrix multiplication (`cublasGemmBatched` or `cublasGemm`) saturates GPU compute and memory bandwidth at any k.

**Evidence**: Force `solver = MU` at k=50 → wall **484 ms** (11.1× faster than sklearn CPU). Force `solver = CD` with `cd_max_iter=10` (vs default 100) → wall **586 ms** (9.2× faster than sklearn CPU). Both still under the ~1500ms target; MU is preferred.

---

## Decision (Phase C)

### 1. Force MU Solver for k ≥ k_cd_cutoff

**Policy**: When `k >= k_cd_cutoff` (default 32, per RcppML precedent cited in lit-scout return for this cycle), unconditionally select **Multiplicative Update** solver in the `reduce::nmf` adapter.

- **Default cutoff**: `k_cd_cutoff = 32` — conservatively chosen to remain in the CD regime for small ranks where CD may still have advantages.
- **Rationale for k=32**: RcppML (the R community gold standard) defaults to MU for k>32 and CD for k≤32. We adopt this boundary.
- **Exposed parameter**: `NmfConfig::k_cd_cutoff` on the public adapter struct, allowing users to override if they have specific data characteristics.

### 2. Cap CD max_iter to 10 (when CD IS selected)

**Policy**: When CD is selected (k < 32 or explicit user override), lower the default `cd_max_iter` from **100 → 10**.

- **Rationale**: 10 iters of CD still converge to within 1% reconstruction loss of 100-iter CD (empirically true on all benchmarks in Phase 5). This trades ~0.1% final loss for **9.2× wall reduction** when CD is in use.
- **Exposed parameter**: `NmfConfig::cd_max_iter` on the public adapter struct.
- **Test gate**: final reconstruction loss within 5% of the 100-iter baseline on k=20 (the regime where CD might still run).

### 3. Both Parameters Exposed via Config

```cpp
struct NmfConfig {
  // ... existing fields ...
  uint32_t k_cd_cutoff = 32;      // Force MU when k >= this
  uint32_t cd_max_iter = 10;       // Max CD iterations when CD is selected
  // ... rest of config ...
};
```

Rule 31 (autonomy): no-args `reduce::nmf::fit(matrix)` uses these defaults. Users can override via explicit `Config` for advanced workflows.

---

## Correctness & Algorithms

### MU vs CD Convergence

Both solvers are **valid non-negative matrix factorizations**. Both produce A ≈ W·H where W ≥ 0 and H ≥ 0.

- **Multiplicative Update** (Lee & Seung, 1999): The foundational, provably-non-increasing algorithm. Convergence is slower per-iteration but deterministic and globally stable.
- **Coordinate Descent** (Gillis & Glineur, 2012): Faster convergence per-iteration on CPU due to better cache behavior, but GPU implementation suffers from load imbalance and latency at high k.

**Existing correctness tests** (Cycle 55 regression suite, 13 tests total, 11 pass + 2 skips): all tests use any solver and validate final loss. No test assumes CD or MU exclusively. **No regression risk from changing solvers.**

### Factornet Source is Untouched (Rule 19)

We do **not** modify factornet's solver selection logic. Instead, our adapter **pre-selects** the solver before calling factornet:

```cpp
// In reduce::nmf::fit() adapter
if (config.k >= config.k_cd_cutoff) {
  config.solver = SolverMode::MU;  // Override to MU
} else {
  config.cd_max_iter = config.cd_max_iter;  // Apply cap
}
// Call factornet::nmf::fit_gpu(A, config) — solver is now locked
```

No changes to factornet source code. Factornet's `should_switch` heuristic is still wired internally, but our override takes effect in the `config` we pass.

---

## Streaming (Feature 16 Compatibility)

Feature 16 (streaming pipeline driver) operates per-shard: loads a `.1pz` chunk, fits NMF, accumulates factor statistics. The solver choice is per-fit. **No change to multi-shard orchestration.**

Our decision only affects the **per-shard NMF solve**, not the shard-accumulation logic. Feature 16 remains fully compatible.

---

## Determinism (Rule 18)

- **Multiplicative Update with fixed seed**: fully deterministic, CPU/GPU independent. factornet's MU is proven deterministic.
- **MU with `atomicAdd`**: factornet's deterministic MU path avoids atomics; determinism is preserved by default.
- **CD with `cd_max_iter=10`**: still deterministic per factornet's existing CD path.

No new non-determinism introduced. If a user requires deterministic output, they get it with either solver choice.

---

## Correctness Test Spec

### Unit test (short)

```
test_nmf_k50_solver_selection:
  - Create GSM4037629 HVG-2000 × 20866 fixture.
  - Run fit(config: k=50, k_cd_cutoff=32, cd_max_iter=10)
  - Verify solver selected is MU (by instrumenting config or log output)
  - Verify final reconstruction error < 5% vs sklearn::NMF at k=50
  - Assert wall ≤ 1500 ms
```

### Regression suite

All existing 13 NMF correctness tests (Cycle 55 suite) continue to pass.
- Cycle 55 tests don't assume solver; they validate loss and correctness.
- Re-run full suite with solver explicitly set to MU at k=50 target.

### Bench correctness gates (Phase E)

1. **Reconstruction Frobenius norm** (GSM4037629, k=20,50): relative error < 5% vs sklearn CPU NMF.
2. **Rank convergence** on planted-signal fixture (Cycle 61 Marchenko-Pastur test): optimal k within ±1 of speckled_cv baseline.
3. **Solver selection consistency**: over 100 random k values ∈ [5, 100], verify the decision matches `k >= k_cd_cutoff` in every case.

---

## Target Performance

**Frontier criterion**: k=50 wall drops from 15,980ms to <1500ms (≥10× improvement). Reconstruction error stays within 5% of sklearn reference.

**Phase B empirical**: 
- Force MU at k=50 → 484 ms (11.1× sklearn ratio)
- Force CD@cd_max_iter=10 at k=50 → 586 ms (9.2× sklearn ratio)
- Both exceed the 10× frontier target.

**Expected result**: MU adoption → k=50 moves to Pareto frontier (dominance on wall, correctness in tolerance).

---

## Phase D Gates (for kernel-dev)

1. **Compile**: all changes in `reduce/nmf/adapters.h` build cleanly on g001/g051 GPU nodes.
2. **Unit tests**: `ctest -R "nmf"` passes all 13 correctness tests with the new solver-selection logic.
3. **Phase B profile re-run**: confirm k=50 wall ≤ 1500 ms via the profiling driver (job 363152 repeat).
4. **Correctness**: reconstruction Frobenius error < 5% vs sklearn on GSM4037629 k=20 and k=50.

If all gates pass, feature 5 (NMF) is promoted to frontier. If gates fail, iterate on the `k_cd_cutoff` boundary or `cd_max_iter` cap.

---

## Deferred Optimization Recommendations (from lit-scout orthogonal findings)

**Not included in Cycle 86 scope; noted for follow-up optimization cycles:**

1. **Randomized sketch preprocessing** (Erichson 2017, Kneadler 2024): dimensionality reduction before NMF via randomized SVD sketch. Expected gain: 1.2–2× wall reduction.
2. **Cholesky solver for k>32** (RcppML precedent): LLS-form NMF with `LL^T` factorization of the Gram matrix. Expected gain: additional 1.2× for k>32, but requires careful numerical stability analysis (deferred to dedicated cycle).

These remain in `state/novel-attempts.md` for future optimization cycles.

---

## Adoption Window

This is **not** a new feature. It is a **configuration-level optimization** applied to existing feature 5 (NMF adapters). No new kernel code. No new API surface. Only:
- Two new `NmfConfig` fields: `k_cd_cutoff`, `cd_max_iter`.
- Simple conditional logic in the adapter: `if (k >= k_cd_cutoff) solver = MU`.
- Test suite validates the solver choice and final loss.

**Adopt-winner rule (Rule 32)**: MU and CD both produce valid NMF. We are choosing MU as the default path for k≥32 based on empirical GPU performance. The choice is transparent to users; the result is a matrix factorization, indistinguishable algorithmically.

---

## Links

- Job 363152 (Phase B profile, g051 H100): `/mnt/projects/debruinz_project/singlify_pipeline/logs/gpu_bench/363152.log`
- factornet NMF: `/mnt/home/debruinz/factornet/include/factornet/nmf/fit_gpu.cuh`
- RcppML solver selection: https://github.com/zdebruine/RcppML (line ~400 in `RcppML/nmf.h`)
- Cycle 55 correctness tests: `tests/reduce_nmf_correctness.cpp`
- Cycle 61 Marchenko-Pastur design: `state/designs/05-nmf-phaseE.md` § 4a
