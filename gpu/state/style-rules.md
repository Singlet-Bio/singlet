# singlet-gpu — Style & Algorithmic Rules

The single source of truth for "how singlet-gpu is written." Every kernel and every test must comply. Updated only when the orchestrator dictates a change.

---

## §A. Absolute rules (mirrored from `CLAUDE.md` §⛔)

1. Never read `/CLAUDE.md` or `singlify/.github/agents/singlify.agent.md`.
2. Never modify singlify. File a `CROSS-AGENT-*` task in `dag.md` if `.1pz` format must change.
3. GPU-native only. CPU does `.1pz` decompression and streaming orchestration only.
4. Zero-copy CSC → CuSPARSE. The only `.1pz` loader is `io/pz_device_loader.h`. No densification.
5. fp32 by default. fp64 only when documented numerical analysis demands it.
6. No host↔device traffic in hot loops.
7. Benchmark every feature against SOTA before commit.
8. Correctness harness before perf tuning.
9. One feature per cycle.
10. C++ first, wrappers second.
11. Billion-cell design constraint — every kernel has an OOC plan in its header doc.
12. Never go idle.
13. No raw `new`/`delete`/`cudaMalloc` outside `core/memory.h`.
14. cuBLAS/cuSPARSE/cuSOLVER handles only from `core/handles.h`.
15. Streams passed in by caller, never created inside a kernel.
16. Reproducible seeds — no implicit `time(0)`.
17. Determinism opt-in for atomic-using reductions.
18. PCA/NMF backends carry `// port:factornet {path}` and diff against the factornet CPU reference.

---

## §B. C++/CUDA invariants (extracted from `singlify/include/singlet-pileup/` patterns)

1. **Header-only**, `namespace singlet_gpu { ... }`. `.cu` files only for explicit template instantiations.
2. **Templates on scalar types** where it matters. `if constexpr` for compile-time dispatch on integral vs floating accumulators.
3. **Sparse struct over pointers**. `struct DeviceCSC { int32_t* indptr; int32_t* indices; float* data; int64_t nrows, ncols, nnz; cudaStream_t producer_stream; }`. Caller manages lifetime via the RAII pool.
4. **No raw `new`/`delete`**. Host: `std::vector`. Device: RAII pool wrapper from `core/memory.h`.
5. **Stateless inline functions**. Algorithms are pure functions of inputs. No mutable module state.
6. **Prefix-sum + counting-sort patterns** for dedup and group-by — same as `singlet-pileup/sparse_accumulator.h`. On device: cub::DeviceScan.
7. **Dense accumulation before output** when accumulating sparse updates. Per-block dense buffer in shared memory; emit only nonzeros.
8. **`#pragma pack(push, 1)`** for any file-format struct. `pz_writer.h` constants are mirrored bit-exact in `io/pz_device_loader.h` — no re-derivation.
9. **No exceptions in hot paths**. Early return / status enum. `noexcept` where possible.
10. **Comments document WHY**: hidden constraints, invariants, surprising choices, paper citations. Do not restate the code.
11. **`constexpr` for fixed data**. Lookup tables, magic numbers, hyperparameter defaults.
12. **Compile flags**: `-O3 -std=c++20 --use_fast_math -arch=sm_80,sm_90` (Ampere + Hopper) by default.

---

## §C. CUDA-specific rules

1. **Streams**: every kernel takes `cudaStream_t stream` as the last argument. Default is 0 (legacy default stream) only inside tests; production kernels never pass 0.
2. **Handles**: cuBLAS/cuSPARSE/cuSOLVER handles come from a thread-safe pool in `core/handles.h`. Each pool entry is bound to a stream. Never call `*Create*Handle` inside a kernel.
3. **Allocation**: device buffers go through `core::DevicePool::alloc<T>(size_t n)` returning `core::DeviceUniquePtr<T>` (RAII). Host pinned via `core::PinnedPool`.
4. **Async by default**: `cudaMemcpyAsync` everywhere. The only `cudaMemcpy` (sync) call is in tests.
5. **Atomics**: `atomicAdd` is allowed only when the kernel exposes a `deterministic=true` opt-in path that uses segmented scan or block reduction.
6. **Shared memory**: prefer it over global L1 for inner-product accumulators. Document size in the header doc.
7. **Tensor cores**: use them when they fit the math (mma fragments for GEMM-shaped reductions). Document in header doc.
8. **Cooperative groups**: prefer over manual `__syncthreads()` where it expresses intent better.
9. **Launch bounds**: declare `__launch_bounds__(threads_per_block, min_blocks_per_sm)` on every hot kernel.
10. **Stream priorities**: the loader uses high-priority streams; compute uses default-priority. Set via `cudaStreamCreateWithPriority` in `core/stream.h`.

---

## §D. Numerical conventions

1. **fp32** for matrix data, gradients, residuals, activations, and per-element kernels.
2. **fp32 with fp64 accumulator** (mixed) for reductions whose output is small (≤k² Gram, top-k eigenvalues, normalization scalars).
3. **fp64** only when a design doc proves fp32 cannot meet tolerance after Kahan, two-pass, mixed-precision, and rescaling have been tried.
4. **Welford's two-pass** for variance (HVG, normalization).
5. **Implicit centering** for PCA: `A(v) = X·v − μ·(1ᵀv)`, never materialize `X − μ`.
6. **Iterative refinement** for ill-conditioned solves: solve in fp32, refine residual in fp64 if needed.
7. **Reduction order matters**: when bitwise reproducibility is required, use segmented scan with deterministic block ordering.

---

## §E. PCA — *integrate* factornet's existing GPU SVD suite

Source: `/mnt/home/debruinz/factornet/include/factornet/svd/`. License: **GPL-2.0**. factornet ALREADY ships GPU implementations of all five SVD backends as `*_gpu.cuh` files. We do NOT port — we wrap.

### Adapter headers (one per backend under `reduce/svd/`)

Each header is ~50 lines: it re-exports a factornet GPU symbol behind a singlet-gpu name with a `DeviceCSC` (= `factornet::gpu::SparseMatrixGPU<float>`) input and a singlet-gpu result struct. No algorithm logic.

1. **`lanczos.h`** → wraps `factornet::svd::lanczos_gpu(...)` (factornet/svd/lanczos_gpu.cuh). Backend = Golub–Kahan bidiagonalization. Target: dominant for `k<32`.
2. **`irlba_factornet.h`** → wraps `factornet::svd::irlba_gpu(...)`. Backend = factornet's implicitly-restarted Lanczos. Target: `k≥64`.
3. **`randomized.h`** → wraps `factornet::svd::randomized_gpu(...)`. Backend = Halko–Martinsson–Tropp with `q=3` power iterations. Target: `32 ≤ k < 64` on GPU.
4. **`krylov_constrained.h`** → wraps `factornet::svd::krylov_gpu(...)`. Backend = KSPR. Supports L1/L2/non-neg/upper-bound/orthogonality constraints. Target: constrained `k≥8`.
5. **`deflation.h`** → wraps `factornet::svd::deflation_gpu(...)`. Backend = rank-1 ALS with deflation; supports Graph Laplacian + robust IRLS. Target: constrained `k<8` and robust paths.
6. **`auto_select.h`** → wraps `factornet::svd::auto_select(...)`. Routing table is in factornet, not duplicated here.

The header doc for each adapter cites the factornet header it wraps and lists the parameters it exposes through. Constraints, tolerances, max iterations, restart widths — all are factornet's. We do not override.

### Auto-selection table (lives in factornet at `svd/auto_select.hpp` — we do NOT duplicate)

singlet-gpu's `reduce/svd/auto_select.h` calls `factornet::svd::auto_select(...)` and returns its result. The routing logic (k<32 → Lanczos, 32 ≤ k < 64 → Randomized, etc.) is owned by factornet and we trust it.

### Implicit centering

factornet's SVD GPU API takes a `centering: bool` flag. We pass it through. The `A(v) = X·v − μ·(1ᵀv)` operator is implemented inside factornet — we do not re-code it. Our adapter ensures `μ` is precomputed on device once per call (cached on the `DeviceCSC` if available) and handed to factornet.

**Correctness gate**: relative L2 ≤ 1e-5 on singular values vs factornet **CPU** on GSM4037629 (since we trust factornet GPU but want to confirm our adapter does not corrupt the call); subspace angle ≤ 1e-6 radians on top-k left/right vectors. The diff is "factornet-cpu vs factornet-gpu-via-our-adapter," which validates the wiring, not the math.

---

## §F. NMF — *integrate* factornet's existing GPU NMF + DAG

Source: `/mnt/home/debruinz/factornet/include/factornet/nmf/`. License: **GPL-2.0**. factornet ships `nmf/fit_gpu.cuh`, `nmf/fit_chunked_gpu.cuh`, `nmf/fit_cv_gpu.cuh`, full GPU bridge in `gpu/{nnls,gram,rhs,loss}.cuh`, and the `factornet::graph::FactorGraph` DAG abstraction. We adapt, we don't reimplement.

### Module layout (corrected per cycle 5 code-reader on factornet NMF API)

- `reduce/nmf/types.h` — re-exports of factornet types: `NmfConfig`, `NmfResult`, `FactorConfig`, `DenseMatrix`, `LossType`, `SolverMode`, `InitMode`, `LazySpeckledMask`. ~30 LOC.
- `reduce/nmf/fit.h` — `singlet_gpu::reduce::nmf::fit(PzDeviceMatrix, NmfConfig, W_init?, H_init?)` adapter that calls `factornet::nmf::nmf_fit_gpu<float>(host_col_ptr, host_row_idx, host_values, m, n, nnz, config, W_init, H_init)`. ~80 LOC.
- `reduce/nmf/cv.h` — adapter for `factornet::nmf::nmf_cv_fit_gpu<float>(...)` (held-out CV, used by orchestrator for auto-rank). ~80 LOC.
- `reduce/nmf/chunked.h` — adapter for `factornet::nmf::nmf_chunked_gpu<float>(...)`. Takes a `singlet_gpu::io::PzDataLoader&` (which implements factornet's `io::DataLoader<float>` interface yielding Eigen sparse chunks). ~120 LOC.
- `reduce/nmf/init.h` — re-exports the three init modes (Random, Lanczos, IRLBA). Modes 1/2 reuse our cycle-5 SVD adapters internally. ~60 LOC.
- `reduce/nmf/graph.h` — adapter exposing `factornet::graph::FactorGraph<float>` and `factornet::graph::fit<float>`. Multi-modal joint factorization is built by the user via `ConcatNode` + `NMFLayerNode` — there is **no built-in shared-H abstraction** in factornet (correction from the original style rules). The header documents the canonical multi-modal pattern. ~120 LOC.
- `streaming/pz_data_loader.h` — `singlet_gpu::io::PzDataLoader` implements `factornet::io::DataLoader<float>` to yield Eigen sparse chunks from `.1pz` files. The only non-zero-copy point in the library — documented cost. ~150 LOC.

The `reduce/nmf/{nnls,losses}/` directories scaffolded earlier are **deferred / future**. We do not re-implement these — factornet has them. The `.gitkeep` placeholders document this.

### Non-MSE loss caveat (correction)

factornet's GPU NMF is fully GPU-resident only for **MSE** loss. Non-MSE losses (KL, NB, GP, Gamma, Tweedie) use **host-mediated IRLS** — a CPU solver per column. The kernel still benefits from GPU SpMM for the residual + Gram pass, but per-iteration runtime is 50–100× slower than MSE.

This is factornet's behavior, not ours. Document in every header that exposes `LossType` and let users choose. Absolute rule §⛔3 reads "no CPU fallback of math kernels" — we accept this exception because it is upstream.

### Multi-modal joint NMF (CORRECTED in cycle 6)

`factornet::graph::FactorGraph` DOES have an explicit shared-H multi-modal abstraction: **`SharedNode`**. The cycle 5 code-reader summary missed this. Use `SharedNode` (NOT `ConcatNode`) for true shared-H multi-modal joint factorization:

```cpp
InputNode<float, SpMat> rna_in (rna_eigen,  "rna");
InputNode<float, SpMat> atac_in(atac_eigen, "atac");
InputNode<float, SpMat> adt_in (adt_eigen,  "adt");

SharedNode<float> shared({&rna_in, &atac_in, &adt_in}, /*k=*/20, "shared_joint");

FactorGraph<float> net({&rna_in, &atac_in, &adt_in}, &shared);
auto result = factornet::graph::fit(net, /* combined input */);
```

`SharedNode` is the canonical multi-modal pattern. `ConcatNode` is for feature-axis concatenation (different semantics — one big feature vector per cell, single NMF). Both nodes exist; we re-export both in `singlet_gpu::reduce::nmf::graph` and document the difference in the header doc.

We do NOT wrap this in a one-liner because the user must decide (a) which inputs share H, (b) whether to apply per-modality normalization, (c) whether to use an alignment guide for batch-effect correction.

### Core fit loop

Owned by factornet — we do not touch it. Documented here for orchestrator design-doc context (so the orchestrator knows what factornet is doing under our adapter):

1. factornet precomputes `G = WᵀW` (`k×k`, cuBLAS syrk) and `B = WᵀA` (`k×n`, cuSPARSE SpMM).
2. Gram trick: `||A − WH||² = tr(AᵀA) − 2·tr(Bᵀ·H) + tr(G·HHᵀ)` — loss is `O(k²)` per iter with `G, B` cached.
3. Solve `H` column-by-column via the selected NNLS backend (mode 0/1/2 or auto mode 3).
4. Transpose roles, repeat for `W`.
5. Convergence: patience-based relative loss (`|Δℓ|/|ℓ| < tol` for `N=5` consecutive checks).

Our adapter does not parameterize, override, or instrument any of this. It just calls `factornet::nmf::fit_gpu(...)`.

### Solver modes (`solver_mode ∈ {0,1,2,3}`) — owned by factornet

singlet-gpu exposes `solver_mode` as a pass-through field on `FactorConfig`. Default is `3` (auto). We do not interpret or override. Documentation reproduced here for orchestrator context only:

- **0 — Coordinate Descent** (constraints active).
- **1 — Cholesky + clip** (unconstrained MSE).
- **2 — Multiplicative Updates** (every non-MSE loss; 10–38× faster than CD on sparse MSE).
- **3 — Auto** (factornet's data-driven switching).

### Non-MSE MU rules (one fused elementwise kernel per loss header)

| Loss | Update rule |
|---|---|
| KL / Poisson | `H ← H * (Wᵀ[A/WH]) / (Wᵀ·1)` |
| Negative Binomial | `H ← H * (Wᵀ[A/WH]) / (Wᵀ[(r+A)/(r+WH)])`, `r=size` |
| Generalized Poisson | KL weights with per-row/col dispersion `θ`, MM update |
| Gamma | `H ← H * (Wᵀ[A/WH²]) / (Wᵀ[1/WH])` |
| Tweedie | `H ← H * (Wᵀ[A·WH^{-p}]) / (Wᵀ[WH^{1-p}])` |
| ZIGP | EM soft-imputation, sparse→dense internally; opt-in only |

### Adam-accelerated MU

`mu_acceleration=1`: momentum on the MU pseudo-gradient with `β(t) = β_max·(1 − exp(−t/τ))`, `τ` adaptive per Gram condition number. Implemented as a fused kernel on top of MU.

### Initialization (`init_mode`)

- **0** Random uniform `[0,1)`.
- **1** Lanczos SVD init: `W = |U|·√Σ`, `H = √Σ·|V|ᵀ` (reuses `reduce/svd/lanczos.h`).
- **2** IRLBA SVD init for `k ≥ 32` (reuses `reduce/svd/irlba_factornet.h`).

### Constraints (`FactorConfig<Scalar>`, tiered)

- **T1** L1 (per-column in NNLS), L2 (`G.diag() += λ`).
- **T2** L21 group sparsity, angular orthogonality (post-NNLS clip of negative eigs of `G`), Graph Laplacian (`G += λ·L`), target regularization (positive `λ` attract; negative `λ` subtract target covariance and project away — factornet's `PROJ_ADV` mode).
- Hard: non-negativity (`max(0, x)`), upper bound (`min(x, ub)`).

### Advanced paths from factornet

- Masked NNLS with per-column delta-`G` correction (missing data / cross-validation).
- Fused RHS+NNLS on sparse inputs: RHS, warm-start, NNLS in a single parallel loop — no materialized global `B`.
- Alignment guide: pre-iteration modification of `G`/`B` to steer factors toward a reference decomposition (batch-effect correction).
- Dual coordinate descent: alternating per-row / per-column iteration for balanced convergence.

### Auto-rank determination

Speckled-mask cross-validation. Random holdout mask per run, train loss vs test MSE. Factors fitted sequentially with early stopping when test loss plateaus. Uses the rank-specific NNLS solver (MU/CD/Cholesky) per candidate `k`. Capped at `config.k_max`. Output: chosen `k`, held-out loss curve, all intermediate fits retained.

### Precision

fp32 device-side for all loss kernels, `G`, `B`, `W`, `H`. fp64 accumulator on reductions feeding `G` (`≤k²` cost). IRLS / robust paths use fp32 residuals + fp64 weight accumulation. Determinism flag gates atomicAdd in the KL/NB RHS builders via segmented scans.

### Correctness gate

Diff against the factornet CPU reference (`factornet::nmf::fit_cpu`) on the same `(A, k, config)` triple. Since both paths share factornet's algorithm code, the diff measures **adapter wiring fidelity**, not algorithm correctness:

1. Tiny fixed-seed synthetic CSC (500 × 200, nnz ≈ 2000).
2. GSM4037629 exon_counts (11,560 cells).
3. 100k-cell concatenated slice (rank auto-selection, convergence patience).

Acceptance: relative `||W−W_ref||_F / ||W_ref||_F ≤ 1e-4` and same on `H`; reconstruction-loss difference within `1e-5` relative. Higher than for SVD because NMF has stochastic init + non-convex optimum — the gate confirms we did not corrupt the call, not bit-identity.

---

## §G. Per-feature design doc requirements

Every `state/designs/{feature}.md` must contain (filled by orchestrator in Phase C):

1. Algorithm (cite factornet for PCA/NMF; SOTA reference otherwise).
2. Numerical stability plan (fp32/fp64 decision, reductions, reorthogonalization).
3. Memory layout (CSC in-place vs gather-scatter, workspace budget).
4. Stream strategy (how many streams, what overlaps with what).
5. Out-of-core chunking plan (the §⛔11 billion-cell strategy).
6. Determinism story (atomics vs segmented scan, seeded stochasticity).
7. Correctness tolerance (exact metric + threshold + reference implementation).
8. Target runtime + memory vs SOTA at three scales.
9. Reference-diff test spec — formal enough that `analysis-validator` can implement it without seeing kernel source.

---

## §H. Wrapper rules (Python + R)

1. Wrappers are opened only AFTER the kernel is on the frontier.
2. Python: pybind11 binding under `python/src/`. The binding is a thin marshal — no logic.
3. R: Rcpp binding under `r/src/`. Same rule.
4. Both wrappers call the SAME header-only kernel. They never re-implement.
5. AnnData / Scanpy interop on the Python side: return `anndata.AnnData` with `.X` set to a `cupy.sparse.csr_matrix` view of the device buffer.
6. SingleCellExperiment / Seurat interop on the R side: return an SCE with `assays(sce)$counts` as a `Matrix::dgCMatrix` host copy (R does not natively own GPU memory; document the host-copy cost in the wrapper header).
7. Wrapper acceptance: `python -m build && twine check dist/*` on the Python side; `R CMD check --as-cran r/` on the R side. Both must pass clean.

---

## §I. Determinism flag

Any kernel using `atomicAdd`-style reductions exposes:

```cpp
template<typename T>
struct ReductionConfig {
    bool deterministic = false;
    uint64_t seed = 0;  // only relevant for stochastic kernels
};
```

When `deterministic=true`:
- `atomicAdd` is replaced by segmented scan or deterministic block reduction.
- Stream order is fixed.
- Kernel output is bitwise reproducible across runs and across GPU architectures of the same compute capability.

Default is `false` for performance. Tests that need bitwise reproducibility set it to `true`.

---

## §J. Cycle-protocol lessons (added 2026-04-30, autonomous loop session)

These are procedural improvements distilled from CYCLE-153 / 154 / 155. They are not algorithmic rules; they are checklist items the orchestrator should treat as additions to the per-cycle protocol in `agents/singlet-gpu-orchestrator.md`.

### §J.1 — Threshold-masking pitfall (from CYCLE-153 iter-1 FAIL)

When auditing a kernel that has multiple bugs that cancel each other (e.g. a broken score formula PLUS a broken threshold), the test suite's PASS/FAIL bits will hide the real failure. CYCLE-147 audit reported scrublet's real-data test as PASS with `doublet_rate=0.001`; the rate was that low only because a broken right-to-left knee threshold returned ~1.0 (classifying ~0.1% of cells), masking that the underlying scoring inflates real-cell scores into the [0.25, 1.0] range. The CYCLE-153 fix that corrected the knee threshold then exposed the real bug as `doublet_rate=0.92`.

**Rule**: in Phase F (frontier decision), don't rely on test PASS/FAIL bits alone. Sanity-check:
- Score / output value distributions (`mean_score`, percentiles, range).
- Threshold values used.
- Whether the output is in the expected order of magnitude.

If two independent metrics fail, suspect a deeper algorithmic bug, not metric-specific symptoms.

### §J.2 — SLURM nodelist defaults (from CYCLE-153 queue management)

Cycle scripts inheriting from `cycle150_diffmap.sh` template default to `--nodelist=g001`. This caused a 25-min queue wait when g001 was busy with two long-running 2:42-hour jobs while g003, g004, and 5 other GPU nodes sat idle.

**Rule**: do NOT pin `--nodelist=g001` by default. Either:
- Omit `--nodelist` entirely and let SLURM pick.
- Check `sinfo -p gpu` before submitting and pick a single idle node.
- Use `--exclude=g001,g002,g005` to reject busy / unwanted nodes (gives SLURM freedom to pick any allowed node).

**`--nodelist` vs `--exclude` gotcha (CYCLE-158 lesson)**: `--nodelist=A,B,C` is a HARD "use these specific nodes" constraint and SLURM may interpret the comma-list as "I want all 3 nodes" → forced to `--nodes=3` → much harder to schedule. To say "any one node from set {A, B, C}", use `--exclude=<everything-else>` instead. CYCLE-158's first resubmission used `--nodelist=g003,g004,g050,g051,g052` and went PD as a 5-node job; switching to `--exclude=g001,g002,g005` immediately landed on g003.

The `cycle150_diffmap.sh` template was updated to `--nodelist=g003` post-CYCLE-153 (pin to one idle node). For more flexibility use `--exclude` instead. New cycle scripts can pick either pattern.

### §J.3 — Pareto-frontier rows must be added in Phase F (from CYCLE-154)

The CYCLE-118 → CYCLE-150 stretch promoted 21 features to frontier without adding `state/pareto-frontier.md` rows. CYCLE-154 had to backfill them in a dedicated cycle.

**Rule**: in Phase F (frontier decision), the cycle is not closed until:
1. `state/cycle-log.md` has the episode appended (was already required).
2. `state/pareto-frontier.md` has a row entry added with at least the feature path, citation, and TBD-row scaffolding (NEW — was previously deferred to backfill).
3. `state/dag.md` is updated.

A "Pending Phase E benchmark cycle" placeholder row is ALWAYS preferable to no row — it lets future bench cycles fill numbers without restructuring.

### §J.4 — Phase G publish is part of every frontier-touching cycle (from CYCLE-155)

The orchestrator's Phase G says "Every cycle that updates the frontier MUST run the publish flow." But CYCLE-150 / 151 / 152 / 154 all skipped `frontier_sync.py`; CYCLE-155 had to catch up the publish backlog.

**Rule**: in any cycle that modifies `state/pareto-frontier.md`, the close checklist now includes:

```bash
source ~/Singlet-AI/singlet-gpu/scripts/load_secrets.sh
python3 ~/Singlet-AI/singlet-gpu/scripts/frontier_sync.py
```

If `frontier_sync.py` fails, log to `state/blockers.md` as `INFRA-WEBSITE-*` and continue (existing rule). The `gpu_frontier` row count is by feature_id (not by section header); sub-variants share parent IDs (`scripts/frontier_sync.py:210`).

### §J.5 — Loop pacing — alternate safe + risky cycles

Empirically, in a 6-cycle loop session, the failure mode is "string of debug cycles": each one risks introducing regressions, and a single FAIL costs an iteration of recovery. The pattern that worked for this session was alternating cycle types:

| Cycle type | Risk | Per-cycle artifact |
|---|---|---|
| State / docs cleanup (Haiku) | LOW | concrete .md files |
| Phase H docs page write (Haiku) | LOW | docs/api/*.md |
| Phase G publish (script run) | LOW | Supabase row refresh |
| Pareto-frontier row backfill (Haiku) | LOW | rows in state/pareto-frontier.md |
| Phase E bench (Sonnet + SLURM) | MEDIUM | bench numbers in pareto-frontier |
| Kernel debug / port (Sonnet + SLURM) | HIGH | new kernel + correctness verify |

After a HIGH-risk cycle, the next 1-2 cycles should be LOW-risk to restore signal-to-noise. CYCLE-153 (HIGH, FAIL) → CYCLE-154 (LOW, PASS) → CYCLE-155 (LOW, PASS) was a healthy recovery pattern. If the orchestrator detects a recent FAIL outcome, it should bias the next cycle's selection toward LOW-risk options in the queue.

This rule sits as a heuristic, not a hard constraint — the priority cascade in Phase A still has primacy.

### §J.6 — Frontier promotion requires a scale-smoke test for O(n²⁺) kernels (from CYCLE-159 NEGATIVE result)

CYCLE-150 promoted `embed/diffmap` to frontier based on 5/5 ctest PASS at n=40 cells. CYCLE-159 Phase E discovered the kernel is **14× SLOWER than scanpy CPU at n=10k** and **CRASHES at n=30k** (cuSOLVER status=3 in Ssyevd). The kernel materializes a dense n×n W matrix and runs full Ssyevd; scanpy uses sparse ARPACK on the kNN graph (O(n·k·n_components) vs our O(n³)). Small-n correctness was real but did not validate scaling.

**Rule**: any kernel with O(n²) memory or O(n³) compute (typical signs: dense n×n adjacency, full Sgemm or Ssyevd over a graph) cannot reach `frontier` state until a **scale-smoke test at ≥10k cells** passes within reasonable wall time (rough rule: <2× the small-scale extrapolation by O-bound).

The smoke test can be a single ctest case or a single bench run; what matters is that an actual 10k-cell run completes without OOM / cuSOLVER errors and produces output in roughly the predicted time. This catches dense-eigensolver patterns that look fine at n=40 but explode at n=10k.

**Currently at-risk kernels (audited by CYCLE-160)**:
- ⚠️ `embed/diffmap.h` — confirmed broken (CYCLE-159 NEGATIVE; filed CYCLE-159.1).
- ⚠️ `embed/dpt.h` — uses the SAME dense n×n W + `cusolverDnSsyevd` pattern (`dpt.h:151,179,215,270,302,453`). Highly likely to exhibit the same scaling failure at n≥10k. Should be benched (Phase E) before claiming frontier-grade scaling. CYCLE-150 frontier promotion was again only at small ctest n.
- `embed/dendrogram.h` — uses `cusolverDnSsyevd` only on k×k cluster centroid distance matrix (k=number of clusters, typically ≤100). NOT at risk — k-bound is by clusters, not cells.
- `integrate/combat.h`, `integrate/asw.h`, `integrate/lisi.h`, `integrate/kbet.h`, `qc/empty_drops.h`, `qc/soupx.h`, `anno/celltypist.h`, `anno/symphony.h`, `enrich/decoupler_*` — all O(n_cells) or O(n_cells × small) by design (per-cell histogram, kNN-aware metrics, or sparse SpMM). NOT at risk.

**Action filed for CYCLE-160 close**: `embed/dpt.h` pareto-frontier row updated with ⚠️ "AT-RISK" marker pending bench. Future Phase E should target dpt early to confirm or reject the suspicion.

**CYCLE-161 update**: dpt benched, §J.6 hypothesis CONFIRMED (GPU 541× SLOWER than scanpy). Plus a bonus API design bug — dpt re-runs full eigendecomp every call while scanpy splits one-time `sc.tl.diffmap` from cheap `sc.tl.dpt(iroot)`. CYCLE-159.1 follow-up expanded to combined sparse-eigensolver + API refactor for both diffmap and dpt.

### §J.7 — Phase E speedup is a continuum, not crisp classes (from CYCLE-163-167 decoupler sweep)

After benching the full decoupler family (wsum, ulm, mlm, ora, viper) and observing speedups from 9.78× (ulm) up to 3101× (ora), the original "bimodal" / later "trimodal" speedup hypothesis is not crisp. Speedups span a CONTINUUM driven by two independent factors:

1. **SOTA structural factor**: how Python-overhead-bound is the CPU reference?
   - Pure Python loops (scanpy.tl): high overhead per element → big GPU win.
   - `scipy.stats.*` called per element in a Python loop: still has Python per-call cost (e.g. ora's `scipy.stats.hypergeom.sf` → 2832-3101×).
   - Vectorized scipy with `axis=` argument (e.g. `scipy.stats.rankdata(axis=0)`, `scipy.linalg.cho_solve` over batched rhs): native C inner loop, low overhead per call.
   - Single dense BLAS call (`X @ W`): tightest C native code.

2. **GPU compute intensity per cell**: heavier per-cell GPU work narrows the speedup ratio because the CPU also does more work.
   - Light kernels (wsum: 2-pass SpMM + scalar div): ~3-8 ms GPU at 30k → ratio dominated by Python overhead.
   - Heavy kernels (viper: rank + qnorm + Sgemm = 387 ms at 30k): GPU has so much real work that the gap to a vectorized CPU narrows.

**Predict speedup as roughly**: `(SOTA_python_overhead × GPU_parallelism) / GPU_compute_intensity_per_cell`.

**Empirical decoupler corpus** (CYCLE-163-167) for calibration:

| Kernel | GPU 30k ms | scipy 30k ms | speedup | SOTA shape | GPU compute |
|---|---|---|---|---|---|
| wsum | 8.4 | 131.3 | 15.7× | sparse @ dense + L1-norm | light |
| ulm | 9.9 | 129.0 | 13.1× | sparse @ dense + scalar OLS | light |
| mlm | 9.5 | 200.7 | 21.0× | sparse @ dense + Cholesky solve | medium |
| ora | 21.4 | 66442.0 | 3101× | top-K + per-element `scipy.stats.hypergeom.sf` | medium |
| viper | 387 | 19982.9 | 51.6× | per-cell rank + qnorm + Sgemm | heavy |

**Rule for Phase E reports**: report speedup AND classify the SOTA structure (loop-bound vs vectorized) AND GPU compute class (light/medium/heavy). Avoids selling features on misleading 3000× numbers when the typical case is 15×.

**Refinements from CYCLE-169-176 corpus expansion** (16 features now; CYCLE-176 lessons):

After expanding the corpus across non-enrich kernels (lisi, asw, kbet, kmeans, dendrogram, magic, combat, celltypist), four additional axes refine the prediction model. **None of the original two factors are wrong; they are necessary but not sufficient.**

#### 4 refinements

1. **GPU-per-cell-ms denominator** (CYCLE-171 kbet): use `(SOTA_per_cell_ms × overhead_factor) / GPU_per_cell_ms`. The original prediction model has implicit GPU-time = constant per cell; that's wrong for kernels with non-trivial per-cell compute (kbet's chi² + Wilson-Hilferty: ~30× more GPU work per cell than lisi/asw histogram → lisi/asw 100-200× vs kbet 21-32×).

2. **Memory bandwidth bottleneck axis** (CYCLE-174 magic): when SOTA materializes large dense intermediates (>100 MB), CPU becomes memory-bandwidth-bound (~50 GB/s) while GPU has HBM ~900 GB/s — that's a 20× advantage on top of compute. Iterative SpMM with dense intermediate output (magic t=3, combat dense Z) → 1000-3000× even for "vectorized native code" SOTAs.

3. **Overhead compounding axis** (CYCLE-175 combat): when SOTA has BOTH per-call Python overhead (per-batch loop, per-iter Python) AND dense intermediates (memory bound on CPU), the two effects compound multiplicatively into 1000-3000× class. Empirically: combat (per-batch + dense) lands 2188-2497×; magic (just dense) lands 1891-2506×; ora (just per-element scipy.stats) lands 2832-3101×.

4. **BLAS-tight subdivision** (CYCLE-172 kmeans + CYCLE-176 celltypist): the "BLAS-tight" SOTA class needs subdivision:
   - **Tight + low Python overhead**: kmeans (single sklearn `.fit_predict()` call wrapping tightly-vectorized Lloyd iter) → 2-7× speedup. The CPU is genuinely competitive at small/medium scale.
   - **Tight + Python orchestration overhead floor**: celltypist (sklearn `.predict_proba()` per call has ~3ms Python overhead floor regardless of work) → 50× speedup at small/medium scale, scaling-bound floor.

#### Updated prediction formula (informal)

```
speedup ≈ (SOTA_per_cell_ms × python_overhead_multiplier × memory_bandwidth_advantage)
        / GPU_per_cell_ms

where:
  python_overhead_multiplier ∈ {1×, 5×, 10-100×}
    1×       = SOTA is single tight BLAS/SciPy call (kmeans)
    5×       = SOTA has Python overhead floor per call (celltypist)
    10-100×  = SOTA has per-batch/per-element Python loops (decoupler_ora, combat)

  memory_bandwidth_advantage ∈ {1×, 20×}
    1×       = SOTA intermediates fit in CPU L3 (lisi, asw, kbet, kmeans, decoupler_*, dendrogram)
    20×      = SOTA materializes dense intermediates >100 MB (magic, combat)

  GPU_per_cell_ms is observed empirically — read from the GPU side of the bench
```

#### Empirical 16-feature corpus (final calibration)

| Kernel | speedup 30k | SOTA shape | GPU compute | Memory-bound | Predicted |
|---|---|---|---|---|---|
| kmeans | 2.4× | sklearn KMeans (one tight call) | medium | no | 2-7× ✓ |
| ulm | 13.1× | scipy SpMM + scalar OLS | light | no | 10-30× ✓ |
| wsum | 15.7× | scipy SpMM + L1-norm | light | no | 10-30× ✓ |
| mlm | 21.0× | scipy SpMM + Cholesky | medium | no | 10-30× ✓ |
| kbet | 21.1× | numpy chi² + Wilson-Hilferty | heavy | no | 10-30× ✓ |
| dendrogram | 106× | scipy pdist + linkage (orchestrated) | light | no | 50-200× ✓ |
| viper | 51.6× | scipy rankdata vectorized + qnorm + Sgemm | heavy | no | 30-100× ✓ |
| celltypist | 50× | sklearn predict_proba (Python overhead floor) | light | no | 30-100× ✓ |
| asw | 249× | numpy ASW vectorized | light | no | 100-300× ✓ |
| lisi | 219× | numpy LISI vectorized | light | no | 100-300× ✓ |
| score_genes | 493× | scanpy.tl Python loop per gene-set | light | no | 200-500× ✓ |
| pearson_residuals | 302× | scanpy HVG Python loop | medium | no | 200-500× ✓ |
| model_gene_var | 471× | scanpy HVG (2 flavors avg) | medium | no | 200-500× ✓ |
| magic | 2506× | scipy SpMM + dense intermediate (200-600 MB) | medium | **YES** | 1000-3000× ✓ |
| combat | 2497× | scanpy.pp.combat (per-batch loop + dense) | medium | **YES** | 1000-3000× ✓ |
| ora | 3101× | per-element scipy.stats.hypergeom.sf | medium | no | 1000-5000× ✓ |

**All 16 predictions land within the formula's range when all 4 axes are considered.** This is the prediction model now empirically validated. Future Phase E cycles should use it.

**§J.7 final form**: Phase E speedup is 4-axis (SOTA structure × GPU compute × memory bandwidth × overhead compounding). Single-axis predictions (the original "trimodal" hypothesis) systematically miss the 1000-3000× cases and over-predict for BLAS-tight cases. The corpus of 16 features is the calibration set.

### §J.8 — Bench-helper API verification (from CYCLE-167 build FAIL)

CYCLE-167's first dispatch hallucinated the bench-helper API — wrote `timer.record() / timer.median_ms() / mem.snapshot_*() / mem.peak_mb() / row.wall_ms_med / row.mem_mb_peak`, none of which exist. Build FAILed with 9 errors. The actual API (read from CYCLE-166 ora bench): `timer.start(stream)/stop(stream)/elapsed_ms()`, `mem.sample_before/after()/peak_delta_mb()/reset()`, `row.wall_ms/mem_mb`.

**Rule**: when writing a new bench driver based on a template (§J.5 "kernel debug" pattern), always grep the template for actual call sites of every helper used. Do not paraphrase the API from intuition. A 30-second `grep -E "timer\.|mem\.|row\." bench/<template>.cpp` would have caught CYCLE-167's hallucinated names.

Concretely, gpu-bench worker prompts should now require: "Before writing the new bench cpp, run `grep -nE 'BenchTimer|PeakMemTracker|BenchRow|timer\.|mem\.|row\.' <template-bench-cpp>` and only use call sites from that grep output."

This rule generalizes beyond bench helpers — any port that copies a known-good template should grep the template for the actual API call surface, not paraphrase it.

### §J.9 — Algorithmic conventions must be EXPLICIT in design docs (from CYCLE-182 iter-1 FAIL)

CYCLE-182 (sparse_eigensolver Phase D iter-1) failed 4/5 tests because the LOBPCG kernel returned the SMALLEST K eigenvalues instead of the LARGEST. Looking back at the design doc (`state/designs/sparse_eigensolver.md`), the algorithm sketch had a casual comment `cusolverDnSsygvd(M, N, K_smallest)` — the word "smallest" was right there, but framed as an implementation detail rather than as the wrong-convention bug it actually was. The implementer (Sonnet) faithfully followed the design and produced a working LOBPCG that solved the wrong problem.

This is a class of design-doc bug that recurs:
- LOBPCG defaults to smallest eigenvalues (ground-state) — but for graph Laplacian top-K we need largest.
- ARPACK's `which='LM'` vs `which='SM'` — top-K in user terms = `'LM'` (largest magnitude) for symmetric problems.
- Eigenvector sign convention (positive or negative leading entry) — design doc must specify.
- Ordering convention (ascending vs descending) — design doc must specify.
- Indexing convention (0-based vs 1-based; row-major vs column-major).
- Nullspace handling (skip first eigenvalue/vector if it's the trivial λ=1).

**Rule**: design docs for new kernels with algorithmic conventions must include an explicit "Conventions" section that, for each convention, states:
1. **What the convention is** in user-facing terms (e.g. "top-K largest eigenvalues, descending order").
2. **What the underlying library defaults to** (e.g. "cusolverDnSsygvd returns ascending; LOBPCG ground-state finds smallest").
3. **The bridge between the two** (e.g. "negate M before Ssygvd; eigenvalues come back as -original; final reverse").
4. **A specific reference test or library** that establishes the expected convention (e.g. "matches `scipy.sparse.linalg.eigsh(A, k=15, which='LA')`").

The convention bridge is often a 1-line implementation detail with an n-LOC consequence if wrong. CYCLE-182 lost a full HIGH-risk cycle to a bridge that was implicit. CYCLE-181's design doc has been updated post-CYCLE-183 with the explicit negation.

This rule generalizes to ANY kernel where the GPU library's default convention differs from the user-facing convention. Examples worth documenting in future design docs:
- cuSOLVER eigensolvers: ascending order; LOBPCG: ground-state.
- cuBLAS: column-major; numpy: row-major.
- cuSPARSE: 0-indexed; some BLAS: 1-indexed.
- cuRAND: stateful; numpy.random: stateless seed-per-call.
- cuSPARSE SpMM: (m × k) · (k × n) → (m × n); some libs use the transpose form.

**Action**: future Phase C design docs (Opus task) must include the Conventions section before Phase D dispatch. gpu-kernel-dev worker prompts can also explicitly call out the convention check during implementation.

### §J.10 — BLOCKED-vs-iterate methodology (from CYCLE-185 sparse_eig blocking decision)

Rule 5 states "after 2 iterations without dominance, mark `blocked`." This is a hard ceiling, not a soft guideline. CYCLE-159.1 sparse_eig step 1 hit it: iter-1 (1/5 PASS), iter-2 (2/5), iter-3 (still 2/5) → marked BLOCKED per Rule 5 in CYCLE-185.

**The trade-off**: pushing iter-N+1 with an unclear root cause often produces another partial result and burns another HIGH-risk cycle. Marking BLOCKED earlier preserves loop bandwidth for productive work and creates a clean re-entry point with documented diagnostic.

**Decision rule** (when to BLOCK vs continue iterating):
1. **Blocked if root cause is unclear after 2 iters**. The first iter establishes the bug; the second iter validates a hypothesis. If the second iter doesn't measurably move the failure metric, the hypothesis is wrong and you need fresh diagnostic — not another fix attempt.
2. **Continue iterating if root cause is clear and concrete**. CYCLE-153 scrublet → CYCLE-148.1 follow-up was correctly NOT iterated because the audit's "focused fix" estimate was already wrong (CYCLE-153 lesson). CYCLE-167 build FAIL → fix in same cycle was correctly inline because the bug was 9 named symbols.
3. **Always file a follow-up at BLOCKED time** with: (a) what's known to work, (b) what fails, (c) failed hypotheses tested, (d) the smallest reproducer, (e) what fresh perspective might unlock progress (Phase B re-do, instrumented diagnosis, alternative algorithm).

**Anti-pattern to avoid**: "iter-N+1 with the same hypothesis variant" — if you're trying tolerance=1e-5 instead of tolerance=1e-6 in iter-3 after iter-2 already found tolerance wasn't the issue, you're cycling. Stop.

**What CYCLE-185 did right**: filed `CYCLE-159.1-BLOCKED-CONVERGENCE-DETECTION` with concrete what-works (Test 5 PASS at cos-sim=1.000), what-fails (tridiagonal Tests 1/4), failed hypotheses (rho sign, max_iter, relative residual), smallest reproducer (n=40 tridiagonal), and what's needed (instrumented stagnation diagnosis OR Phase B re-do on LOBPCG convergence behavior). The CYCLE-159.1 work can resume cleanly when fresh perspective is available.

### §J.11 — Wrapper-test-infrastructure-first (from CYCLE-187 PARTIAL)

CYCLE-186 wrote a Python pybind11 wrapper for `enrich/score_genes` (binding + Python wrapper + 4 pytest tests). CYCLE-187 verify FAILed at the test fixture (cupy 14 dtype strictness), not the wrapper itself. The wrapper is sound and shipped per Rule 26, but full validation is blocked behind a test-infrastructure issue that affects many tests beyond just score_genes.

**Lesson**: when adding a Python wrapper that needs new pytest tests, **first run an existing pytest in the test module to confirm the test fixture works on the current install** before writing new tests on top. CYCLE-187 would have found the cupy.sparse + cupy 14 dtype issues with a single `pytest python/tests/test_enrichment.py::test_run_progeny_*` smoke run.

**Concretely** for wrapper cycles:
1. Before writing the new wrapper: run `pytest python/tests/test_<module>.py -k existing_test_name` to confirm fixtures + cupy + scanpy + anndata work on the current install.
2. If existing tests fail: file an infrastructure-fix follow-up FIRST, before adding more tests on top of the same broken fixture.
3. If existing tests pass: write the new wrapper + tests with confidence.

**§J.11 rule**: gpu-kernel-dev / wrapper worker prompts must include a step "verify existing test fixtures pass before adding new tests." Saves ~1 verify cycle per wrapper.

### §J.12 — Dependency-version compatibility for Python wrappers (from CYCLE-187 cupy 14)

CYCLE-187 surfaced two distinct cupy 14 incompatibilities in the same iteration:
1. `cupy.sparse` removed → renamed to `cupyx.scipy.sparse`.
2. cupy 14 is dtype-strict; `cupy.asarray(np.array(..., dtype=object))` raises `Unsupported dtype object` (cupy <14 was lenient).

Both are real-world breakage that affects existing code, not just new tests. Other libraries with similar tight version compat windows in the singlet-gpu Python stack: scanpy (1.10.x → 1.11), anndata (0.10.x → 0.11), scikit-learn (1.5 → 1.6 had API removals).

**Rule**: when writing or maintaining Python wrappers, treat dependency version compat as a real concern:
1. **Pin a tested version range** in `pyproject.toml` (`cupy-cuda12x>=13,<15` rather than `>=13`).
2. **Use try/except import fallback for renamed APIs** when the rename is recent (per the cupy.sparse → cupyx.scipy.sparse fix in `loader.py:128-135`).
3. **Periodically audit dependency upgrade paths**: dependabot-style sweep cycle every ~10 cycles to surface latent breakages before they bite.
4. **Document the tested install set** in `state/infrastructure.md` so future cycles know what versions the test fixtures assume.

**Action**: file `CYCLE-188.1-DEPENDENCY-COMPAT-SWEEP` for a future cycle to (a) pin version ranges in pyproject.toml, (b) audit `python/singlet_gpu/` for other cupy 14 incompatibilities, (c) update `state/infrastructure.md` with the tested install set.

This rule is more general than just cupy — it's about treating Python ecosystem dependency drift as a real maintenance burden, not an afterthought.

### §J.13 — Stub-era wrappers fail in N independent ways under first real run (from CYCLE-189-195 score_genes verify saga)

CYCLE-187's wrapper-verify failure for `enrich/score_genes` cascaded through SEVEN independent peeled layers across CYCLE-189 through CYCLE-195 before reaching 3/4 PASS:

| Cycle | Layer peeled | Files touched |
|-------|-------------|---------------|
| 189 | `cupy.sparse` rename + cupy 14 dtype-strict CAI dicts (DeviceCsc views) | 19 wrapper files |
| 190 | `_core.from_cupy_csr` 6-arg → single-arg (5 wrapper helpers carried stub-era signature) | 5 helpers |
| 191 | `_core.normalize_total/log1p` kw-only enforcement + in-place semantics + void-return | lognorm.py |
| 192 | `_device_csc_to_csr` returned raw dict (not sparse matrix); fast-path missed transpose | lognorm.py |
| 193 | `_core.to_cupy_csr` return-dict had nested CAI dicts (3rd recurrence of cupy 14 dtype rule) | lognorm + qc_metrics |
| 194 | `score_genes.py` sparse-handling: `sp.issparse` missed cupy sparse; `result.scores_view` 4th cupy 14 recurrence; scanpy 1.11 removed `inplace=True` from `log1p` | score_genes + test |
| 195 | `_build_anndata` stashed C++ DeviceCsc in `adata.uns` → `_mutated_copy` deepcopies uns → unpicklable pybind | io/loader.py |

The pattern: **stub-era wrappers that have never been integration-tested rot in N independent ways under the first real run**. Each verify cycle peels exactly one layer because the next layer's bug is invisible until the previous one is fixed. The expected progression is roughly:
1. Build/import-time failures (deps, missing symbols)
2. Direct binding signature mismatches
3. Semantic mismatches (kw-only, in-place vs return, void)
4. Cross-layer convention drift (return types, dict-vs-object)
5. Recurrence of layer-1 issues at deeper call sites
6. Application-level wrapper logic (sparse-type checks, defaults)
7. Lifetime/anchor and serialization gotchas (uns vs private attr)

**Rules**:

1. **Wrapper "exists" ≠ "works"**: a wrapper file that imports cleanly is NOT a working wrapper if its full call path has never run end-to-end. Mark it `STUB` until verified.
2. **Verify the FULL call path of one wrapper before adding another**: do not mass-produce wrapper files in CYCLE-N then leave verify for CYCLE-N+10. Each new wrapper gets its own integration-test cycle BEFORE the next one is written.
3. **Expect N peels per stub wrapper** (where N = 5-8 in our experience): plan stub-wrapper verify as a 4-7-cycle dispatch, not a single PASS cycle. Use §J.10 BLOCKED-vs-iterate methodology if peels exceed ~8 without convergence.
4. **Deeper call-site recurrence is normal**: once you find one cupy-14-style issue (or signature-style issue), grep ALL call sites of the same kind in the same cycle, even untested ones (qc_metrics.py:323/423 in CYCLE-193 had the same untested bug as lognorm.py — fixed proactively).
5. **The C++ binding side may need its own follow-up**: `make_view_object` returning a bare CAI dict was the root cause of 4 separate cupy 14 recurrences. Filed `CYCLE-193-FOLLOWUP` for the C++-side fix (return an object with __cuda_array_interface__ attribute, not a bare dict).

**Why this matters**: a stub wrapper is **net negative** in the codebase if it ships unverified — it advertises a feature that crashes for users. Better to ship 1 fully-verified wrapper per cycle than 5 stubs that look ready in the import-test but fail in real call paths. This is a strong refinement of §J.11 (wrapper-test-infrastructure-first).

**Concrete process change**: when adding a new wrapper, the SAME cycle must include (a) the wrapper file, (b) at least one pytest that exercises a real .1pz fixture end-to-end through the wrapper, (c) a verify SLURM job that PASSES that pytest. If any of (a)/(b)/(c) is deferred, the wrapper file is `STUB` and not exposed in the public Python namespace until the deferred items land.

### §J.14 — Wrapper-rot SWEEP cycle for INHERITED stub-wrapper trees (from CYCLE-197/198)

§J.13 governs new wrappers we author.  But what about the dozens of pre-existing stub wrappers that already shipped under previous cycles?  CYCLE-187 score_genes verify saga showed sequential per-wrapper peeling costs ~7 cycles per wrapper.  At 30+ stub wrappers in the tree, that is ~210 cycles — unacceptable.

CYCLE-197 demonstrated a dramatically better pattern: **the wrapper-rot SWEEP cycle**.

**The pattern**:

1. **Write a single SLURM job** that runs ALL `python/tests/test_*.py` files in one venv build, captures per-test-FILE pass/fail summaries, and emits a triage table at the end.  Template at `state/cycle197_wrapper_rot_sweep.sh`.  90-min walltime is enough for 17 test files including a fresh wheel build.

2. **Inspect the triage output by failure CLASS**, not by test file.  Group with `grep -E "^FAILED" log | sed 's/.*- //' | sort | uniq -c | sort -rn`.  Common-cause groups are obvious: "TypeError: pca(): incompatible function arguments" hit 10+ tests across 5 files in CYCLE-197.

3. **Fix the largest shared-cause group in ONE cycle**.  CYCLE-198 was a 4-file edit that unblocked ~80 tests.

4. **Re-run the SWEEP** to measure the delta.  Repeat 3-4 until either (a) all groups peeled OR (b) remaining groups need C++-side or new-feature work.

**Empirical result (CYCLE-197/198)**:
- Before: 0 PASS, ~70 FAIL across 17 test files (only test_tl_markers all-skipped).
- After 1 sweep + 1 fix cycle: ~80 PASS across the same 17 test files (test_bindings 0→62, test_core 0→12, test_io 0→3, test_preprocess 0→1+long-run, test_enrichment 0→3).
- Strategic dividend: 2 cycles addressed ~80 tests vs ~80 × 7 = 560 sequential per-wrapper cycles → **~280× speedup**.

**Rules**:

1. **First action upon inheriting a stub-wrapper tree**: write the sweep script (90 min one-time investment).  Do NOT start sequential per-wrapper verifies.
2. **Read sweep failures by class, not by file**: `sort | uniq -c | sort -rn` reveals the few load-bearing root causes immediately.  Fix the BIGGEST count first.
3. **Each fix cycle gets one re-sweep**: after the fix lands, re-run the same SWEEP job and compare deltas.  This is the verify gate.
4. **Stop sweeping when remaining failures are NOT shared-cause**: e.g. 23 `test_new_features_smoke` failures in CYCLE-198 were `_core.X is not available` — those are pure-stub wrappers calling unimplemented C++ bindings.  No shared fix; each needs a real binding implementation.  Convert to roadmap items.
5. **Segfaults and shape-mismatches that emerge from wrapper-rot fixes are real progress, not regressions**: the CYCLE-198 sweep produced 7 test files segfaulting in `_core.pca` — but those test files were FAILING BEFORE TOO; the fix unmasked deeper kernel bugs.  File those as separate kernel-investigation cycles.

**When to apply**: any time you discover (a) ≥5 stub wrappers in a tree where (b) integration-test rot is suspected.  Cost: 1 sweep cycle + 1-3 shared-cause fix cycles.  Replaces N×7 sequential per-wrapper cycles where N = number of affected wrappers.

### §J.15 — Minimal-repro before BLOCKED (refines §J.10) (from CYCLE-213/214/215 pca-segfault arc)

§J.10 says "after 2 iters without dominance, mark BLOCKED."  But that decision is dramatically more useful when iter-2 produces a **clean disambiguation**, not just another partial result.

CYCLE-199 (pca C++ kernel segfault, blocking ~7 test files) followed this trajectory:
- **CYCLE-213** (Opus diagnosis pass): hypothesized `host_retained=false` mismatch.  Code inspection ruled it out (cv.h:254 explicitly says auto_select doesn't need host buffers).  Hypothesis 1 dead — 0/2.
- **CYCLE-214** (Opus fix attempt): hypothesized lifetime / dangling device-pointer in `result_csr`.  Applied `result_csr.__dict__["_singlet_gpu_device_ref"] = device_csc` anchor.  Sweep showed identical Fatal Python error at the same line.  Hypothesis 2 dead — 0/2.

At this point §J.10 says BLOCKED.  But "blocked because we don't know" is much weaker than "blocked because we know it's a kernel-side bug at scale ≥ 5000×2000."  The ROI of one more cycle to write a minimal repro is enormous.

**CYCLE-215** (minimal-repro dispatch): wrote a 130-line SLURM script that bypassed the lognorm chain entirely — synthetic scipy CSC at 5 progressive sizes, direct `_core.from_cupy_csr → _core.pca`.  Result in 3:28: tiny PASS, small PASS, **medium SEGFAULT**.  Conclusively localizes bug to C++ kernel side, with concrete scale threshold and zero involvement of any wrapper code or lifetime concern.

CYCLE-216 then marked CYCLE-199 BLOCKED with three concrete suspect candidates (cuSPARSE workspace, buffer aliasing, cublas pointer-mode) — a starting brief precise enough that a single bio-exec dispatch with `cuda-memcheck` should resolve it.

**The refinement of §J.10**:

1. **After hypothesis 2 fails, do NOT immediately mark BLOCKED.  Instead, write a minimal repro** that bypasses the entire candidate-cause stack.  One cycle of investment in disambiguation pays back many cycles of next-implementer guesswork.
2. **A minimal repro is one that holds all but ONE thing constant**.  CYCLE-215 held everything constant except matrix scale — and the failure threshold popped out cleanly.  If you can't think of which axis to vary, that itself is a sign you don't yet have a concrete-enough bug model.
3. **Repro outcomes are valuable EVEN IF they don't fix the bug**.  CYCLE-215 didn't fix CYCLE-199.  But it converted "we don't know" into "kernel-side, scale ≥ 5000×2000, suspect in {workspace, aliasing, pointer-mode}" — a 10× reduction in hypothesis space for the next implementer.
4. **The §J.10 BLOCKED decision now incorporates the repro outcome**.  BLOCKED-with-clean-diagnostic is a graceful pause; BLOCKED-with-no-diagnostic is a punt.  Always prefer the former.

**When to apply**: any time §J.10 is about to fire (2 failed hypotheses on the same bug), spend ONE more cycle on a minimal repro before marking BLOCKED.  Bound the investment: the repro script should be writable in <1 hour and runnable in <10 min.  If the repro itself is hard to write, that's a sign the system has too many hidden state dependencies — file as a separate "hard-to-isolate" architectural issue.

**Anti-pattern to avoid**: "iter-3 will work for sure" — if iter-1 and iter-2 with different hypotheses both miss, iter-3 with another hypothesis is gambling.  The minimal repro is a much higher-EV bet because it eliminates entire hypothesis classes at once.
