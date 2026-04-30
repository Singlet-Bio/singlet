---
feature: 6
roadmap_id: 6
module: include/singlet-gpu/qc/metrics.h
status: design (Phase C)
tolerance: bit-exact vs scanpy sc.pp.calculate_qc_metrics for n_genes, n_umis; <=1e-6 relative error for pct_mt, pct_ribo
target_perf: 100k cells <= 5ms; 1M cells <= 50ms (pure sparse reduction, no GEMM)
ooc_plan: embarrassingly parallel per-shard; accumulate global gene stats via Welford merge
---

# Feature 6a — QC metrics + filtering

Simple sparse matrix reductions on CSC. No cuBLAS needed — this is cuSPARSE + custom reduction kernels.

## Per-cell metrics (one kernel, one pass over CSC)

Input: `DeviceCSC` (genes x cells) from `.1pz` loader.

For each column (cell) j, compute:
- `n_umis[j]` = sum of values in column j (= colsum)
- `n_genes[j]` = nnz in column j (= indptr[j+1] - indptr[j])
- `pct_mt[j]` = sum(values where row_index is a MT gene) / n_umis[j] * 100

MT gene detection: pass a `DeviceMemory<bool> is_mt[n_genes]` mask. Pre-compute from rownames (genes starting with "MT-" for human, "mt-" for mouse). The mask is constructed on host from the `.1pz` metadata and uploaded once.

Similarly for `pct_ribo` (ribosomal genes: "RPS", "RPL" prefixes) and `pct_hb` (hemoglobin: "HB[^P]").

### Kernel design

One CUDA kernel, one thread-block per cell (column):
```
__global__ void qc_per_cell(
    const int* indptr, const int* indices, const float* data,
    int n_cells, const bool* is_mt, const bool* is_ribo,
    float* n_umis, int* n_genes, float* pct_mt, float* pct_ribo)
{
    int j = blockIdx.x;  // cell
    if (j >= n_cells) return;
    
    int start = indptr[j], end = indptr[j+1];
    float sum = 0, mt_sum = 0, ribo_sum = 0;
    int genes = end - start;
    
    // Cooperative reduction within block for large columns
    for (int idx = start + threadIdx.x; idx < end; idx += blockDim.x) {
        float val = data[idx];
        int gene = indices[idx];
        sum += val;
        if (is_mt[gene]) mt_sum += val;
        if (is_ribo[gene]) ribo_sum += val;
    }
    
    // Warp/block reduce
    sum = block_reduce_sum(sum);
    mt_sum = block_reduce_sum(mt_sum);
    ribo_sum = block_reduce_sum(ribo_sum);
    
    if (threadIdx.x == 0) {
        n_umis[j] = sum;
        n_genes[j] = genes;
        pct_mt[j] = (sum > 0) ? (mt_sum / sum * 100.0f) : 0.0f;
        pct_ribo[j] = (sum > 0) ? (ribo_sum / sum * 100.0f) : 0.0f;
    }
}
```

Block size: 128 threads (most cells have 500-5000 nnz, well-matched to 128 threads).

## Per-gene metrics (one kernel, transpose-aware)

For each row (gene) i:
- `mean[i]` = sum of values in row i / n_cells
- `var[i]` = Welford variance
- `n_cells_expressing[i]` = nnz in row i
- `dropout_rate[i]` = 1 - n_cells_expressing[i] / n_cells

CSC is column-major, so row iteration requires a scatter. Two approaches:
1. **Atomic scatter**: One thread per nnz entry, `atomicAdd` to `gene_sums[row]` and `gene_counts[row]`
2. **CSR transpose**: Convert CSC→CSR via cuSPARSE `csr2csc`, then column-iterate

Prefer (1) for simplicity. The atomic contention is low because genes >> warps and the access pattern is random. If profiling shows contention, switch to (2).

## Cell filtering

```cpp
struct FilterConfig {
    float min_genes = 200;
    float max_genes = std::numeric_limits<float>::infinity();
    float min_umis = 0;
    float max_umis = std::numeric_limits<float>::infinity();
    float max_pct_mt = 100.0f;  // e.g., 20.0 to filter high-MT cells
};

DeviceCSC filter_cells(const DeviceCSC& mat, const QcResult& qc, 
                       const FilterConfig& cfg, cudaStream_t stream);
```

Boolean mask → stream compaction → CSC column gather. Output is a new DeviceCSC with filtered columns.

## Gene filtering

```cpp
DeviceCSC filter_genes(const DeviceCSC& mat, int min_cells = 3, 
                       float min_counts = 0, cudaStream_t stream = nullptr);
```

Per-gene nnz check → boolean mask → CSC row gather (requires index relabeling).

## Doublet detection

Already implemented in `qc/doublet_score.h` (421 LOC, Cycle 31). Scrublet-GPU equivalent. Needs runtime verification on GPU node — included in Cycle 55 DAG but not yet confirmed.

## Streaming (billion-cell)

QC metrics are embarrassingly parallel per shard:
- Per-cell metrics: compute per shard, concatenate results
- Per-gene metrics: accumulate `(sum, sum_sq, count)` across shards via Welford online update, then finalize `mean = sum/N`, `var = (sum_sq - sum²/N) / (N-1)`
- Filtering: apply per shard independently (threshold is global)

One pass per shard. No global communication needed for per-cell. Two-pass for per-gene (accumulate, then finalize).

## API

```cpp
namespace singlet_gpu::qc {
    struct QcResult {
        DeviceMemory<float> n_umis;        // [n_cells]
        DeviceMemory<int>   n_genes;       // [n_cells]
        DeviceMemory<float> pct_mt;        // [n_cells]
        DeviceMemory<float> pct_ribo;      // [n_cells]
        DeviceMemory<float> gene_mean;     // [n_genes]
        DeviceMemory<float> gene_var;      // [n_genes]
        DeviceMemory<int>   gene_n_cells;  // [n_genes]
        int n_cells;
        int n_genes;
    };
    
    QcResult calculate_qc_metrics(
        const DeviceCSC& mat,
        const DeviceMemory<bool>& is_mt,     // gene mask
        const DeviceMemory<bool>& is_ribo,   // gene mask
        cudaStream_t stream = nullptr);
    
    DeviceCSC filter_cells(const DeviceCSC& mat, const QcResult& qc,
                           const FilterConfig& cfg, cudaStream_t stream = nullptr);
    
    DeviceCSC filter_genes(const DeviceCSC& mat, int min_cells = 3,
                           cudaStream_t stream = nullptr);
}
```

## Correctness test spec

1. Tiny synthetic (500x200, fixed seed): bit-exact n_genes, n_umis vs numpy; pct_mt within 1e-6
2. GSM4037629: compare vs scanpy `sc.pp.calculate_qc_metrics` on same exon_counts.1pz
3. Filter round-trip: filter → check dimensions match expected
4. Gene filter: min_cells=3 → verify no gene with <3 expressing cells remains

## Target performance

| Scale | n_cells | n_genes | Target wall | Notes |
|---|---|---|---|---|
| tiny | 500 | 200 | <1ms | smoke |
| small | 11.5k | 20.8k | <2ms | GSM4037629 |
| 100k | 120k | 30k | <5ms | concat |
| 1M | 1M | 30k | <50ms | streaming |

QC is pure sparse reduction — should be memory-bandwidth-bound, not compute-bound.
