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
