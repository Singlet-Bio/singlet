# Cycle 72 Design — BUG-WILCOXON-POST-NORMALIZE-CRASH

**Opened**: 2026-04-16
**Feature**: #11 DE pipeline (wilcoxon, t-test, donor_pseudobulk)
**Classification**: priority cascade #1 — compile/test/runtime blocker
**Blocks**: BUG-WILCOXON-REALDATA-CRASH, OPTIM-WILCOXON-SCANPY-PARITY, all P1 DE correctness work
**Prior cycles referenced**: 70 (bug filed), 71 (partial async fix for TinyPlanted; RealData still crashes)

## Symptom

After `log_normalize(csc)` runs, calling `wilcoxon_de(csc, ...)` crashes with
`CUDA error: illegal memory access encountered` reported from `factornet/gpu/types.cuh:105`.
Test 83 (RealData, 20k cells × 30k genes) crashes every run; tests 80–82 (TinyPlanted)
were fixed by the Cycle 71 partial async fix but the same underlying race remains.

## Root cause (Phase B scout synthesis)

1. **lognorm.h**: `log_normalize(mat, cfg, stream)` launches `compute_size_factors_and_apply_kernel`
   at line 451 on the caller's stream. This kernel reads `d_size_factors` and `d_col_sums`
   (local DeviceMemory workspace) and mutates `mat.values` in place. The function then
   constructs `LogNormResult{std::move(d_size_factors), std::move(d_qc_mask), T}` and
   returns. **There is no `cudaStreamSynchronize(stream)` after the apply kernel launch.**
   The two earlier syncs (lines 431, 440) are before the apply kernel.

2. **wilcoxon.h**: kernel assumes fp32 values (enforced via `CUDA_R_32F` in the
   `cusparseCsr2cscEx2` call at line 553). Reads `mat.values`, `mat.col_ptr`,
   `mat.row_indices`, `mat.nnz`. No assumption violated by lognorm; lognorm keeps the
   matrix as fp32.

3. **factornet/gpu/types.cuh:105**: line is
   `CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));` inside `DeviceMemory<T>::DeviceMemory(size_t count)`.
   `cudaMalloc` is where a sticky prior CUDA error surfaces via `CUDA_CHECK`. The
   underlying illegal access happened earlier in the stream; `cudaMalloc` is just the
   next CUDA API call that forces error-query.

### The race

The workspace DeviceMemory for `d_size_factors`/`d_col_sums` has lifetime bounded by
`LogNormResult`. If the caller discards the return value (or the buffer destructs before
the apply kernel finishes), `cudaFree` runs on device pointers the async kernel is still
reading → illegal memory access. The 20k×30k path makes the apply kernel take longer,
extending the race window and making test 83 deterministic.

**Cycle 71's fix** (captured return value in test 83 and added a sync at test level)
reduces the window but does not eliminate it — any caller that doesn't follow the same
discipline will crash. The kernel contract must be self-contained.

## Fix

**Single-line defensive sync at `log_normalize` exit**:

```cpp
// Ensure the in-place apply kernel has completed before any workspace
// DeviceMemory (held by LogNormResult) can be released by the caller.
// Function-boundary sync — not in a hot loop; Rule 9 compliant.
CUDA_CHECK(cudaStreamSynchronize(stream));
return LogNormResult{std::move(d_size_factors), std::move(d_qc_mask), T};
```

Cost: one sync per `log_normalize` call (~μs). lognorm wall is ~0.11ms at small scale
and ~seconds at 1M scale — the sync is negligible either way.

## Same-pattern audit (mandatory this cycle)

Any kernel that:
- Launches an async kernel using a locally-owned DeviceMemory workspace AND
- Moves that workspace into a result struct AND
- Returns without syncing

...has the same latent bug. Explicitly audit:
- `preprocess/scale.h` — scaling / regress_out
- `de/ttest.h` — Welch's t-test
- `de/donor_pseudobulk.h` — NB GLM
- `preprocess/hvg.h` — Seurat v3 + Pearson residuals

For each, confirm presence of a terminal sync OR add one.

## Correctness gate

Regression test `tests/de_wilcoxon_correctness.cpp::Test83_RealData_PostNormalizeNoCrash`:
- 20k cells × 30k genes synthetic sparse (seeded, reproducible)
- Run `log_normalize → wilcoxon_de` with `LogNormResult` discarded explicitly
  (to force the race)
- Expected: no CUDA error, wilcoxon output finite, no NaNs
- Before fix: deterministic crash
- After fix: PASS

Correctness vs scanpy (Jaccard / Spearman) is **out of scope** for this cycle
(tracked separately as OPTIM-WILCOXON-SCANPY-PARITY).

## Streaming contract

No change. `log_normalize` already operates per-shard; the sync is per-call, not per-shard.
Rule 14 streaming doc unchanged.

## Determinism

No change. The sync does not introduce non-determinism.

## Expected Phase E/F outcome

1. build exit 0 on GPU node
2. Test 83 passes after fix (was crashing)
3. Tests 80–82 continue to pass (no regression)
4. Other DE tests unaffected
5. No measurable wall-time regression on lognorm small-scale bench (<1% delta on the 0.11ms baseline)

## Phase D dispatch plan

- `gpu-kernel-dev` (Sonnet): apply the sync line, run the audit of the 4 other kernels,
  submit a GPU-node `make -j8` build job, report build exit + any audit findings.
- `analysis-validator` (Sonnet): write the regression test (Test 83), run it before and
  after the fix (once kernel-dev's build job completes), report pass/fail deltas.

## Out of scope

- OPTIM-WILCOXON-SCANPY-PARITY (separate cycle)
- Donor pseudobulk NB GLM correctness (separate cycle)
- cuVS CAGRA install (environment issue, separate DAG task)
