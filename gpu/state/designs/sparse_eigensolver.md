# `core/sparse_eigensolver` — design doc (CYCLE-159.1, started in CYCLE-181 Phase B)

**Status**: Phase B research complete (CYCLE-181). Phase C design (this doc). Phase D implementation pending CYCLE-182+.

**Motivation** (CYCLE-159, CYCLE-161 NEGATIVE results): `embed/diffmap` and `embed/dpt` materialize a dense n×n W matrix and run `cusolverDnSsyevd`. At n=10k this is 14× SLOWER than scanpy's sparse ARPACK and at n=30k it crashes. Both kernels share the underlying graph-eigendecomposition pattern. A shared sparse symmetric eigensolver fixes both.

## Phase B research (CYCLE-181 lit-scout + code-reader returns)

**Recommended algorithm**: **LOBPCG** (Locally Optimal Block Preconditioned Conjugate Gradient) for top-K exterior eigenvalues.
- Block iteration handles clustered eigenvalues that power iteration fails on.
- Implementable as header-only C++20 wrappers over cuBLAS + cuSPARSE — no runtime linking beyond what singlet-gpu already uses.
- Convergence: O(n_iter × log(1/ε)) where SpMV dominates; ~5-10× speedup over scipy ARPACK demonstrated in literature.
- Deterministic with fixed seed for initial block (Rule 11).

**SOTA reference (CPU)**: `scipy.sparse.linalg.eigsh(A, k=15, which='LA', return_eigenvectors=True)` — what scanpy uses internally for `sc.tl.diffmap` / `sc.tl.dpt`. ARPACK backend.

**Alternatives considered**:
- Lanczos / IRAM — no GPU library; would need to write from scratch.
- cuVS LOBPCG — exists in RAFT but API location unconfirmed; would add runtime linking.
- Custom Filtered Lanczos — only needed for interior eigenvalues; we only need top-K.

## Algorithm sketch (LOBPCG header-only)

```
input:  A — n×n symmetric sparse (CSR), n_components K, max_iter, tolerance ε
output: eigenvalues λ ∈ ℝ^K, eigenvectors V ∈ ℝ^{n×K}

# Initialize: random orthonormal block X ∈ ℝ^{n×K}
X ← QR(rand(n, K), seed=cfg.seed)   # one-time cuRAND Philox + cuBLAS Sgeqrf
P ← zeros(n, K)                     # previous direction

repeat:
  AX ← A @ X                         # cuSPARSE SpMM
  ρ_diag ← diag(X^T @ AX)            # block Rayleigh quotient via cuBLAS Sgemm
  R ← AX − X @ diag(ρ_diag)          # block residual

  # Optional preconditioner (skip for v0; add Jacobi later if convergence slow):
  W ← R                              # = M^{-1} R if M is preconditioner

  # Construct subspace S = [X | W | P] (3K wide), orthogonalize:
  S ← block_gram_schmidt([X, W, P])  # cuBLAS Sorgqr or Cholesky-based BGS
  AS ← A @ S                         # cuSPARSE SpMM

  # Solve K×K Rayleigh-Ritz subproblem:
  M ← S^T @ AS                       # 3K × 3K small matrix via cuBLAS Sgemm
  N ← S^T @ S                        # for generalized; identity if S orthonormal
  (λ_new, U) ← cusolverDnSsygvd(M, N, K_smallest)  # solve 3K × 3K dense

  # Update X, P:
  X_new ← S @ U[:, :K]               # cuBLAS Sgemm
  P_new ← S @ U[:, K:K+P_cols]
  X ← X_new; P ← P_new
  λ ← λ_new

  if max(|R|) < ε: break
  if iter > max_iter: break

return λ, X
```

## Sparse matrix interface

```cpp
namespace singlet_gpu::core {

struct SparseSymmetricCSR {
    const float* values;     // device, nnz
    const int*   row_ptr;    // device, n+1
    const int*   col_idx;    // device, nnz
    int          n;
    int          nnz;
};

template<typename T = float>
struct SparseEigConfig {
    int      n_components = 15;
    int      max_iter     = 200;
    T        tolerance    = static_cast<T>(1e-6);
    uint64_t seed         = 42;
    bool     deterministic = true;  // §J.6 + Rule 11; fixed seed + deterministic SpMM
};

template<typename T = float>
struct SparseEigResult {
    core::DeviceMemory<T>    eigenvalues;   // [K]
    core::DeviceMemory<T>    eigenvectors;  // [n × K] col-major
    int                      iters_run;
    bool                     converged;
};

template<typename T = float>
inline SparseEigResult<T> top_k_eigsh_lobpcg(
    const SparseSymmetricCSR&    A,
    const SparseEigConfig<T>&    cfg = {},
    cudaStream_t                 stream = nullptr);

}  // namespace singlet_gpu::core
```

## Refactor of `embed/diffmap.h` and `embed/dpt.h`

**Diffmap** (CYCLE-150 currently broken):
- Old: build dense W (n×n) from kNN + cusolverDnSsyevd → top eigenvectors.
- New: build SPARSE W from kNN (no densification) → `top_k_eigsh_lobpcg` → scale eigenvectors by λ_k^t.

**DPT** (CYCLE-142 currently broken + has API design bug):
- Old: re-runs full eigendecomposition every call (CYCLE-161 finding).
- New: split into two functions:
  1. `compute_diffusion_eigenvectors(W_sparse, cfg) → DiffusionResult` — heavy, run once per dataset.
  2. `dpt(diffusion_result, root_cell) → pseudotime` — cheap, can be called multiple times with different roots.
- Mirrors scanpy's `sc.tl.diffmap` + `sc.tl.dpt(adata, iroot=...)` separation.

## Memory + complexity

For n=10k, k_nn=10, n_components=15:
- Sparse W: nnz = n × k_nn = 100k entries × (4+4+4) = 1.2 MB device (vs old 400 MB dense)
- Block X, AX, R, S, AS: each n × {15..45} × 4 = 0.6-1.8 MB device
- 3K × 3K dense Rayleigh-Ritz: 8 KB
- Total per iter: ~10 MB; per-iter cost: SpMM(O(nnz × K)) + few small Sgemm + small dense eigsolve

For n=1M, same settings:
- Sparse W: 120 MB (vs old 4 TB dense — completely infeasible)
- Block X etc: ~60 MB device
- Per iter: still bounded by O(nnz × K + n × K²) = manageable

## Correctness validation

- **Reference**: scipy.sparse.linalg.eigsh on the same sparse W.
- **Tolerance**: relative eigenvalue error ≤ 1e-4, eigenvector cosine similarity ≥ 0.9999 for non-degenerate eigenvalues.
- **Test scales**: n=40 (matches CYCLE-150 ctest), n=10k (where current diffmap fails), n=30k (where current diffmap crashes).
- **Determinism**: fixed seed → bit-identical output across two runs.

## Phase D implementation plan (CYCLE-182+)

1. **CYCLE-182**: implement `core/sparse_eigensolver.h` (LOBPCG core kernel) + correctness ctest at n=40, 10k. ~600 LOC.
2. **CYCLE-183**: refactor `embed/diffmap.h` to use sparse_eigensolver. Verify CYCLE-150 ctest still passes + add 10k scale-smoke test (per §J.6).
3. **CYCLE-184**: refactor `embed/dpt.h` API (split `compute_diffusion_eigenvectors` from `dpt`) + use sparse_eigensolver. Verify CYCLE-142 ctest + scale-smoke + (separately) the API-design fix from CYCLE-161.
4. **CYCLE-185**: re-run Phase E for diffmap + dpt. Should now beat scanpy at 10k+ scales.

Total: ~4 cycles for the full sparse-eigensolver + diffmap/dpt rewrite + revalidation. Cleanly testable at each step.

## Risks

- **LOBPCG convergence on highly clustered eigenvalues** (graph Laplacians often have such): mitigate with block-size = K + small over-sample (e.g. K+5).
- **No preconditioner in v0**: convergence may be slow on poorly-conditioned Laplacians. Add Jacobi preconditioner in v1 if needed.
- **Numerical stability of block Gram-Schmidt** at fp32: use modified GS or Cholesky-based BGS; fp64 only if needed (Rule 8).

## Citations

- Knyazev (2001) "Toward the Optimal Preconditioned Eigensolver: LOBPCG". SIAM J. Sci. Comput. 23:517-541.
- Kalantzi & Saad (2015) "GPU Implementation of the Filtered Lanczos Procedure".
- Coifman & Lafon (2006) "Diffusion maps". Applied and Computational Harmonic Analysis 21:5-30.
- Haghverdi et al. (2016) "Diffusion pseudotime robustly reconstructs lineage branching".
