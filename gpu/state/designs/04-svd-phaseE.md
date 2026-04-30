---
feature: 4
module: reduce/svd/*.h
cycle: 60 or later (pending 57, 58, 59)
phase: E (benchmark) + Rule 30 novel pursuit + Rule 31 autonomy pass
extends: 04-svd-adapters.md
status: draft
depends_on: Cycle 55b (all 10 SVD tests runtime-correct)
---

# Feature 4 — SVD adapters Phase E + novel pursuit + autonomy

Feature 4 (`reduce/svd/{randomized,lanczos,irlba_factornet,krylov_constrained,deflation,auto_select}.h`) comprises six thin adapters over factornet's GPU SVD backends. **All 10 SVD correctness tests pass as of Cycle 55b** (including the Cycle 55's legitimate non-determinism fix on `SVD_RoundTrip.Lanczos`). This document specs the frontier push — the hardest part is getting the `auto_select` heuristic right.

## 1. Success metrics (frontier promotion)

On GSM4037629 (310,797 × 20,866 → PCA with k ∈ {30, 50, 100}):

- **Wall p50**: ≤ best of {cuml.TruncatedSVD, cuml.RandomizedPCA, factornet CPU IRLBA, factornet CPU randomized}.
- **Peak dev mem**: ≤ cuml.TruncatedSVD.
- **Correctness**: top-k singular values within 1e-3 relative vs factornet CPU IRLBA (the Cycle 1 reference); U/V Frobenius reconstruction error within 1e-4; sign flips allowed.

Promotion gate: dominance on wall OR memory at k=50, correctness match, `auto_select` delta ≤ 5% vs the best manual backend.

## 2. SOTA baselines

| Baseline | Backend | Rationale |
|---|---|---|
| **cuml `TruncatedSVD`** | cuSOLVER Jacobi | Primary cuml GPU baseline |
| **cuml `RandomizedPCA`** | randomized SVD | rapids-singlecell's primary PCA path |
| **scanpy `tl.pca`** | scipy `svds` | CPU Lanczos, most common scRNA workflow |
| **Seurat `RunPCA`** | IRLBA (R CRAN) | Gold-standard R-side PCA |
| **factornet CPU IRLBA** | `svd::irlba(...)` | Cycle 1 correctness reference |
| **factornet GPU randomized** | `svd::randomized_gpu(...)` | The backend our adapter wraps — our wall time cannot be slower than this by more than adapter overhead (~1%) |

## 3. Bench configurations (Cycle 60 dispatch)

Scales: tiny synthetic (500×200 → k=10), small real (GSM4037629 → k=30/50/100), medium concat (~100k cells → k=50).

Backends × flavors:

1. `ours_randomized_auto` (default — random Gaussian projection + power iteration + QR)
2. `ours_lanczos`
3. `ours_irlba` (factornet IRLBA GPU adapter)
4. `ours_krylov_constrained` (with non-negativity constraint for NMF warm-start comparison)
5. `ours_deflation`
6. `ours_auto_select` (the Rule 31 autonomy path — should pick randomized for k ≤ 50, IRLBA for k > 50, Lanczos for high-accuracy mode)
7. `cuml_truncated_svd`
8. `cuml_randomized_pca`
9. `scanpy_pca`
10. `seurat_pca` (R subprocess, small scale only — R IRLBA on 310k×20k likely >10 min)
11. `factornet_cpu_irlba` (correctness reference AND wall-time floor for CPU comparison)

Metrics: wall p50, peak dev/host mem, top-k singular value relative error vs factornet CPU IRLBA, U Frobenius reconstruction error, throughput (cells/sec).

## 4. Novel pursuit (Rule 30) — beating IRLBA at small k

**Hypothesis**: factornet's GPU IRLBA is a good general-purpose Krylov SVD but it is over-engineered for the small-k scRNA PCA case (k ≤ 50, which is 90%+ of real use). A purpose-built **randomized SVD with 2-pass power iteration + Gram-Schmidt orthogonalization + implicit centering** can beat IRLBA on both wall and memory at k ≤ 50.

### The algorithm (hand-tuned for scRNA)

```
Input: A (genes × cells, sparse CSC, fp32), k (target rank), p (oversampling, default 10)
Output: U (genes × k), Σ (k×1), V (cells × k)

1. Y = A @ Ω  where Ω is (cells × (k+p)) Gaussian random (1 cusparse SpMM)
2. For iter in 0..n_power (default 2):
     Y = A @ (A^T @ Y)    (2 SpMMs per iter — this is the accuracy driver)
     Y = Gram-Schmidt orthogonalize (fp64 accumulators for the column norms)
3. Q, _ = QR(Y)            (cuSOLVER orgqr)
4. B = A^T @ Q              (1 SpMM)
5. U_B, Σ, V = SVD(B)       (cuSOLVER gesvdj on (k+p) × (k+p) — tiny matrix, microseconds)
6. U = Q @ U_B[:, :k]
7. V = V[:, :k]
```

**Total SpMM count**: 1 + 2*n_power + 1 = 5 for n_power=2. IRLBA typically needs 10–20 SpMMs for the same accuracy (Krylov restarts). **Claim**: 5 SpMMs beats 10+ SpMMs when SpMM is the dominant cost (which it is for scRNA CSC matrices).

**Implicit centering**: instead of materializing the dense `A - μ·1^T` matrix (which would densify 310k×20k = ~25 GB), we use the identity `(A - μ·1^T) @ x = A @ x - μ·(1^T @ x)`. Every SpMM becomes `cusparseSpMM(A, x) - outer(mu, sum(x))` where `sum(x)` is a single cub reduction. Same for `A^T @ y`. Zero materialization of the centered matrix. **This is the headline memory win.**

### Success gates

- Top-50 singular value relative error vs factornet CPU IRLBA: ≤ 1e-3.
- Frobenius reconstruction error vs ground truth: ≤ 1e-4.
- Wall time p50 on GSM4037629 at k=50: ≤ 50% of our own `ours_irlba` adapter AND ≤ 75% of `cuml_randomized_pca`.
- Peak device memory at k=50: ≤ 40% of cuml's TruncatedSVD (cuml densifies the result of `A @ A^T` which is huge).

If all four gates pass, the `randomized_smallk` variant becomes the primary `reduce::pca()` backend for k ≤ 50 and IRLBA becomes the fallback for high accuracy mode. If not, log in `state/novel-attempts.md` with the failed gate + residual trajectory.

### Additional novel: fp16 first, fp32 refinement

For k ≤ 30 (common scanpy default), the first n_power iterations can run in **fp16** (Tensor Core cuBLAS GEMM where applicable) with a final fp32 refinement step. Tensor Core SpMM doesn't exist but dense GEMM in fp16 on the small Q, B matrices would give a 4× arithmetic speedup on the `(k+p) × (k+p)` SVD tail. Ship as opt-in first; document the accuracy envelope.

### Additional novel: cached power-iteration warm start

For repeated PCA on the same matrix with different k values (common in tuning), cache Y from the previous run and restart the power iteration from there. Amortizes SpMM cost across multiple PCA calls. Rule-31 relevant: `auto_select` can detect a repeat-call pattern via matrix hash and reuse Y automatically.

## 5. Autonomy pass (Rule 31)

Current SVD backends likely require the user to pick: `backend`, `k`, `n_power_iter`, `oversample`, `tol`, `max_iter`, `center`. All seven auto-tune.

| Config field | Default | Auto-tune strategy |
|---|---|---|
| `backend` | user-pick | Auto: `k ≤ 50 → randomized_smallk`, `50 < k ≤ 200 → irlba`, `k > 200 → lanczos`, `k > 500 → deflation`. |
| `k` | 50 hardcoded | Variance-explained plateau detector: compute first 100 components, return the smallest k where the cumulative variance-explained curve's second derivative < `1e-3` (elbow detection on-device). |
| `n_power_iter` | 2 | Auto: start with 2, if the Rayleigh-quotient residual for the smallest retained singular value > `1e-2` after 2 iterations, escalate to 4. Bounded at 6. |
| `oversample` | 10 | Auto: `max(10, k × 0.2)`. |
| `tol` | 1e-6 | Auto: relaxed to 1e-4 for `randomized_smallk`, tight 1e-8 for `lanczos`. |
| `max_iter` | 300 | Auto: `50 × log2(k)`. |
| `center` | `true` | Always true for PCA; false opt-in for raw SVD. Auto stays true. |

No-args `reduce::pca(matrix)` returns the PCA result with all seven parameters auto-chosen. Decisions recorded in `PcaResult::metadata` under `_autotune_*`.

## 6. OOC streaming contract (Rule 14)

Billion-cell PCA:
- Implicit centering is already streaming-friendly (no materialization).
- Randomized SVD's power iteration is streaming-compatible: `A @ x` and `A^T @ y` both decompose across cell-wise chunks (`A` is genes × cells CSC, chunk along cell axis; `A^T` is cells × genes, chunk along row axis).
- The `(k+p) × (k+p)` tail SVD is tiny and runs once at the end, no streaming.
- Chunk size: `min(free_dev_mem / 4, 500k cells)`.
- Per-chunk: compute partial `A_chunk @ x`, accumulate into global Y. Determinism via Kahan summation across chunks (opt-in).

Document the ooc contract; implement after feature 16 lands.

## 7. Determinism contract

Randomized SVD is inherently non-deterministic via the Gaussian projection seed. Provide explicit seed parameter. Additional non-determinism: SpMM reductions (atomicAdd ordering). Opt-in determinism: cub::DeviceSegmentedReduce for SpMM, stable QR.

## 8. Phase E dispatch spec (Cycle 60)

Dispatch `gpu-bench` with:
- Read 04-svd-adapters.md + this doc.
- Run the 11-config bench table at three scales.
- Implement the `randomized_smallk` variant as a prototype kernel inside the bench driver (not in the header yet).
- Implement the implicit-centering SpMM wrapper as a real change in `reduce/svd/randomized.h` (it's non-invasive and useful for every backend).
- Implement the `auto_select` heuristic update as a real change.
- Write to benchmark-registry.md, pareto-frontier.md (if promoted), novel-attempts.md (for the randomized_smallk gates).

## 9. Open questions

- **cuml TruncatedSVD vs RandomizedPCA**: which is faster for scRNA? Anecdotally cuml.RandomizedPCA wins at k ≤ 100 but cuml.TruncatedSVD wins at higher accuracy requirements. Bench both.
- **fp16 tensor core utility for scRNA PCA**: unclear. The matmul-dominated portion is SpMM which has no fp16 tensor core path. Only the tail dense operations (Q, B, U_B) see fp16 benefit. Measure first before committing to the fp16 novel variant.
- **Power iteration with preconditioning**: could further reduce SpMM count. Next-level novel-pursuit if randomized_smallk lands successfully.

## 10. Links

- Original: `state/designs/04-svd-adapters.md`
- factornet IRLBA ref: `/mnt/home/debruinz/factornet/include/factornet/svd/irlba_gpu.cuh`
- factornet randomized ref: `/mnt/home/debruinz/factornet/include/factornet/svd/randomized_gpu.cuh`
- cuml RandomizedPCA: `https://github.com/rapidsai/cuml` (read via code-reader if needed)
- Halko 2011 randomized SVD paper: the reference theoretical basis for power iteration accuracy bounds
