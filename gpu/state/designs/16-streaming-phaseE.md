---
feature: 16
module: streaming/chunk_iter.h, streaming/streamed_pipeline.h, streaming/reduction_tree.h
cycle: 69 or later
phase: D + E (foundational implementation + benchmark)
extends: 16-streaming-pipeline.md
status: draft
unlocks: every feature's Rule 14 OOC contract (features 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, scVI)
---

# Feature 16 — Billion-cell streaming driver

Feature 16 is the **keystone OOC feature** — it unlocks the out-of-core contract for every other feature previously specced. Once this lands, lognorm / HVG / PCA / kNN / Leiden / UMAP / DE / GSEA / velocity / MT / scVI can all run on >1 billion cells streamed from `.1pz` shards on disk.

Per Rule 14 ("billion-cell design constraint"), every prior feature's design doc has an OOC section pointing to this one. Feature 16 is where the abstract chunk-iterator becomes real.

## 1. Success metrics

On a synthetic 1 billion cell concat (streamed from ~500 `.1pz` shards):

- **Streaming lognorm** end-to-end: ≤ 1 hour on one H100. Target throughput: ~300k cells/sec sustained.
- **Streaming HVG** end-to-end: ≤ 2 hours (two passes: variance estimation + residual computation).
- **Streaming PCA** (randomized_smallk from 04-svd-phaseE): ≤ 4 hours including the power-iteration convergence.
- **Streaming kNN** via blockwise brute-force: ≤ 8 hours (dominated by quadratic pass; approximate mode available for faster).
- **Peak device memory**: bounded at `residency_budget` (user-configurable, default `free_device_mem × 0.8`).
- **Correctness**: every streaming kernel must produce output bit-equal to the non-streaming variant on any dataset that fits in memory (verified at the small scale).

Frontier gate: all four streaming kernels hit their wall targets, correctness bit-exact on small inputs, Rule 31 auto-residency within 10% of manual optimum.

## 2. SOTA baselines

| Baseline | Path | Notes |
|---|---|---|
| **rapids-singlecell `concat`** | Python / cupy | Concatenates AnnData in host memory, fails at >50M cells on typical GPU |
| **Dask-AnnData** | Python | Distributed scRNA; closest direct competitor, but CPU-heavy |
| **scanpy `on_disk` / zarr** | Python | Zarr-backed chunked access; correctness reference only (not competitive on wall) |
| **velocyto.R `loom` streaming** | R | CPU loom streaming; baseline only for wall |

No GPU-native streaming library for scRNA exists at billion-cell scale. This is a greenfield.

## 3. Module layout

```
include/singlet-gpu/streaming/
├── chunk_iter.h              // ChunkIterator template: iterates .1pz shards, returns DeviceCSC chunks
├── shard_catalog.h           // reads a manifest.jsonl of shard paths + metadata
├── residency_budget.h        // tracks free device memory, chooses chunk size
├── streamed_pipeline.h       // composable streaming pipeline (lognorm → HVG → PCA as one pass)
├── reduction_tree.h          // tree-of-partial-results accumulator (sum, sum², quantile sketches)
└── detail/
    ├── prefetch.h            // double-buffer prefetch of the next chunk while kernel runs on current
    ├── checkpoint.h          // checkpoint intermediate accumulator state to disk for crash recovery
    └── procrustes.h          // align chunk-local embeddings (for UMAP-like chunked kernels)
```

All header-only. No new external dependencies beyond CUDA + factornet + Cycle 56's loader.

## 4. Core API

```cpp
namespace singlet_gpu::streaming {

struct ChunkIterConfig {
    size_t residency_budget_bytes = 0;      // 0 = auto (free_dev_mem × 0.8)
    size_t min_chunk_cells = 1024;          // smallest useful chunk
    size_t max_chunk_cells = 500000;        // cap on chunk size
    bool   prefetch = true;                  // double-buffer next chunk
    int    n_prefetch_streams = 2;           // streams for prefetch pipeline
    uint64_t random_seed = 0;                // for deterministic shard ordering
    cudaStream_t stream = 0;
};

struct Chunk {
    factornet::gpu::SparseMatrixGPU<float> matrix;   // cells in this chunk (rows = genes, cols = cells)
    size_t global_cell_offset;                       // index of the first cell in the dataset
    size_t n_cells_in_chunk;
    std::string source_path;                          // which .1pz this chunk came from
    std::map<std::string, std::string> metadata;     // GEO metadata for the source sample
};

class ChunkIterator {
public:
    ChunkIterator(const std::vector<std::string>& shard_paths, const ChunkIterConfig& cfg);
    bool has_next() const;
    Chunk next();             // yields a new chunk on the configured stream
    size_t total_cells() const;
    size_t chunks_remaining() const;
};

// Composable pipeline: run a sequence of kernels on every chunk, accumulate global state
template<typename... Stages>
class StreamedPipeline {
public:
    StreamedPipeline(ChunkIterator iter, Stages... stages);
    auto run();   // returns the tuple of final accumulators from each stage
};

} // namespace singlet_gpu::streaming
```

### Example usage

```cpp
auto iter = streaming::ChunkIterator(shard_paths, {});
auto pipeline = streaming::StreamedPipeline(
    iter,
    preprocess::LognormStage{},
    preprocess::HvgStage{n_top=2000},
    reduce::svd::RandomizedStage{k=50}
);
auto [lognorm_stats, hvg_set, pca_result] = pipeline.run();
```

One shard iteration = one pass over the data. All three kernels run per-chunk + accumulate globally.

## 5. Novel pursuit (Rule 30)

### 5a. Prefetch + overlap via dual streams

Standard pattern: while kernel runs on stream 0 with chunk N, prefetch stream 1 reads chunk N+1 from disk + decompresses + uploads. cudaStreamWaitEvent between stages overlaps I/O, decompression, and kernel wall.

Expected: for a kernel that is GPU-bound (e.g. HVG two-pass), prefetch is invisible. For I/O-bound kernels (e.g. a first pass that only counts cells), prefetch can halve total wall.

### 5b. Reduction tree for global accumulators

Many streaming kernels need global statistics (per-gene mean, variance, quantiles). Naive accumulation is `reduce_per_chunk → atomic_global_add` — non-deterministic and memory-unfriendly.

Better: **tree-of-partial-results**. Each chunk produces a small accumulator (e.g. per-gene Welford tuple). Accumulators merge pairwise via Chan's parallel algorithm — numerically stable and deterministic.

Expected: eliminate atomics + provide determinism opt-in.

### 5c. Quantile sketches for streaming HVG

HVG uses per-gene variance-explained ranking. For streaming, maintain a **TDigest** per gene across chunks instead of a full histogram. TDigest is O(k log k) storage (k = compression factor, default 100), supports merge, and gives ε-approximate quantiles with ε ≤ 0.01.

Published as standard in streaming analytics; new to scRNA HVG.

### 5d. Checkpoint-and-resume for multi-hour runs

A 1B-cell PCA may take 8 hours. If the job crashes, start over? No: checkpoint the accumulator state (factorized matrices, intermediate Q, current partial Y) to disk every N chunks. On restart, read the checkpoint and resume from the next unprocessed shard.

Checkpoint format: just another `.1pz` with `META_TAG_USER_KV` indicating it's a checkpoint + the shard index to resume from.

Expected: enables multi-day billion-cell jobs that would otherwise be infeasible due to hardware reliability.

## 6. Autonomy pass (Rule 31)

| Config field | Auto |
|---|---|
| `residency_budget_bytes` | Auto: `free_dev_mem × 0.8`, probed at pipeline start. |
| `min_chunk_cells` | Auto: 1024 (always). |
| `max_chunk_cells` | Auto: `residency_budget / (n_genes × fp32_size × 3)` — ensures the chunk CSC + workspace fits. |
| `prefetch` | Auto: true unless `free_dev_mem < 4 × chunk_size` (not enough slack for double-buffering). |
| `n_prefetch_streams` | Auto: 2. |
| `random_seed` | Auto: 0 (deterministic shard order unless user pins). |

`StreamedPipeline::run()` picks all of the above automatically — users never touch them.

## 7. Determinism contract

- Chunk order is deterministic (shard paths sorted lexicographically unless user pins a seed and shuffles).
- Reduction tree accumulation is deterministic via pairwise Chan merge.
- Prefetch does not change output — it only changes wall time.
- TDigest merge is deterministic.

Default: deterministic. No opt-in needed.

## 8. Integration with existing features

The previously-specced features' Rule-14 OOC sections all point to this driver. Each feature gets a `Stage` class:

| Feature | Stage class | Notes |
|---|---|---|
| 2 (lognorm) | `LognormStage` | Per-chunk fused kernel; no global accumulator needed (size factors are per-cell) |
| 3 (HVG) | `HvgStage` | Two-pass; stage A = per-gene mean/var accumulation, stage B = residual ranking |
| 4 (PCA) | `RandomizedSvdStage` | Power iteration across chunks; Y accumulator merged via Chan |
| 6 (kNN) | `BrutekNnStage` | Block-wise brute force; approximate mode uses IVF with on-disk inverted lists |
| 7 (Leiden) | — | Full graph must be resident; streaming not supported. Use METIS-partition-and-stitch as a workaround. |
| 8 (UMAP) | `ChunkedUmapStage` | Per-chunk UMAP + Procrustes alignment between adjacent chunks |
| 9 (DE) | `PseudobulkDeStage` | Per-chunk pseudobulk aggregation; GLM at the end on accumulated matrix |
| 11 (GSEA) | — | Needs full gene ranking; run after DE stage returns |
| 13 (velocity) | `VelocityKineticStage` | Per-gene closed-form fit; streaming-native |
| scVI | `ScviEpochStage` | Streaming minibatch sampler pulls cells from current chunk, trains, then advances |

Each feature owns its `Stage` class in its own header. Feature 16 only provides the pipeline infrastructure.

## 9. Phase D+E dispatch spec

Sub-cycle 1: ChunkIterator + residency budget auto-tune + prefetch + correctness test on a 5-shard small dataset matching full-residency output.

Sub-cycle 2: StreamedPipeline template + reduction tree. Correctness test: composed LognormStage + HvgStage pipeline on a 5-shard dataset.

Sub-cycle 3: Checkpoint-and-resume.

Sub-cycle 4: Bench vs Dask-AnnData on a 100-sample concat (~2M cells). This is as large as we can feasibly get from the catalog without synthetic data.

Sub-cycle 5: 1B-cell synthetic scaling test. If the infrastructure scales, promote to frontier.

## 10. Open questions

- **Shard catalog format**: jsonl with one line per shard containing `{path, n_cells, n_genes, vt_code, metadata}`. Suggest: generate this once from the singlify pipeline output tree.
- **Chunk boundaries across shards**: when a chunk is smaller than a full shard, do we split the shard or wait for multiple? Suggest: one chunk = one shard by default; split only if the shard exceeds `max_chunk_cells`.
- **Multi-GPU support**: out of scope for Phase E. Flag as Cycle N+1 follow-up.
- **Checkpoint resumption after code change**: checkpoints tie to the kernel version. A code change invalidates checkpoints. Document.

## 11. Links

- Original: `state/designs/16-streaming-pipeline.md`
- Dask-AnnData: https://github.com/dask/dask (scanpy has a `distributed` extension)
- Chan's parallel algorithm for variance: Chan, Golub & LeVeque 1979
- TDigest: Dunning 2019
- Cycle 56 pz_device_loader: state/designs/00-pz-device-loader.md
