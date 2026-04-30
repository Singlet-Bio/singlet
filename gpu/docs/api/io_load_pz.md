# `io::load_pz`

Zero-copy `.1pz` → device CSC loader. Reads a singlify TP1Z v1 bundle, decompresses on CPU, stages to pinned host memory, copies to device asynchronously, and returns a `PzDeviceMatrix` whose `mat` member is a `factornet::gpu::SparseMatrixGPU<float>` ready for any cuSPARSE / cuBLAS kernel.

## C++ signature

```cpp
namespace singlet_gpu::io {

struct PzDeviceMatrix {
    core::DeviceCSC      mat;                   // device CSC (factornet SparseMatrixGPU<float>)
    core::Metadata       meta;                  // rownames, colnames, GEO KV
    cudaStream_t         producer_stream;       // stream the H→D copy was issued on
    core::PinnedBuffer   pinned_indptr;
    core::PinnedBuffer   pinned_indices;
    core::PinnedBuffer   pinned_values;
    // Optional retained host CSC for SVD adapters (keep_host_pinned=true)
    std::shared_ptr<int>   host_indptr;
    std::shared_ptr<int>   host_indices;
    std::shared_ptr<float> host_values;
    bool host_retained = false;
    int n_genes = 0;
    int n_cells = 0;
};

PzDeviceMatrix load_pz(const std::string& path,
                       cudaStream_t stream = nullptr,
                       bool keep_host_pinned = false);

}  // namespace singlet_gpu::io
```

If `stream == nullptr`, the loader creates a new high-priority stream and returns it in `result.producer_stream`. The caller is responsible for `cudaStreamDestroy` only when they passed `nullptr` — stream ownership is not transferred when the caller supplies their own.

The pinned host buffers are kept alive inside the returned struct until it goes out of scope; the caller must `cudaStreamSynchronize(result.producer_stream)` before letting the struct die if any kernel is still reading from device.

## Python signature

```python
import singlet_gpu as sg
mat = sg.io.load_pz(path: str, keep_host_pinned: bool = False) -> PzDeviceMatrix
```

## R signature

```r
mat <- singletGpu::load_pz(path, keep_host_pinned = FALSE)
```

## Inputs

- **path** — filesystem path to a `.1pz` file. Format: TP1Z v1 (header magic `0x5A315054` = `"TP1Z"`).
- **stream** — optional CUDA stream. Pass `nullptr` to have one created for you.
- **keep_host_pinned** — when `true`, retain pinned host CSC buffers in `host_indptr` / `host_indices` / `host_values`. Required by factornet SVD/NMF adapters that take host pointers; ignore otherwise. Costs 2× the matrix in pinned RAM while the device copy is also live.

## Outputs

`PzDeviceMatrix` containing:

- `mat` — `core::DeviceCSC` with `rows` (genes), `cols` (cells), `nnz`, `col_ptr`, `row_indices`, `values` all on device.
- `meta` — `core::Metadata` (typed sidecar reconstructed from the file's TLV block): `gsm_id`, `gse_id`, `organism`, `taxon_id`, `protocol` (e.g. `"10xv3"`), `modality` (e.g. `"scrna"`, `"cite"`, `"multiome"`), `srr_ids`, `read_count`, `geo_title`, `geo_source_name`, `singlify_version`, `pipeline_date`, `rownames` (gene names), `colnames` (cell barcodes). Empty / zero-initialized for fields not present in the file.
- `producer_stream` — the stream the H→D copies were issued on. Synchronize this before launching kernels that read `mat`.

## Complexity

| | Small (~10k cells, 30M nnz) | Medium (~100k cells, 300M nnz) | Large (1M+, streaming required) |
|---|---|---|---|
| Wall (V100S) | 268.8 ms | TBD | use `PzChunkIterator` |
| Pinned host | ~400 MB | ~4 GB | ≤400 MB (sliding window) |
| Device | ~400 MB | ~4 GB | one chunk at a time |

The full-matrix `load_pz` is bounded by the int32 nnz cap inside `factornet::gpu::SparseMatrixGPU<float>` (~2.1B). At typical scRNA density (1k–3k nnz/cell) that's 0.7M–2M cells per matrix. Beyond that, switch to `PzChunkIterator`.

**Three separate async copies** for `indptr`, `indices`, `values` — lets the runtime pipeline them over PCIe while the CPU does other work.

## Streaming behavior

`PzChunkIterator` (same header) yields fixed-column-width slices without loading the full matrix into pinned host at once. Streaming spec:

- Per-chunk memory: O(chunk_cols × avg_nnz_per_cell × 4B) pinned + same on device.
- Number of passes: 1 over the file. The chunk table at the file's head allows random access to any chunk if needed.
- Reduction tree: per-shard partials → host-side merge → broadcast (kernel-specific; the iterator does not impose one).
- Per-shard independence: yes — every chunk is a self-contained CSC slab. Global statistics (cell library size totals, gene means) require a two-pass driver in the consuming kernel.

See [`streaming.md`](streaming.md) (pending) for the full streaming kernel contract.

## Determinism

Fully deterministic. No atomics, no reductions in the loader itself. Two invocations on the same file produce bit-identical `mat`, `meta`, and `producer_stream` contents.

## Correctness contract

| Reference | Tolerance | Sample |
|---|---|---|
| anndata-gpu read_h5ad | bit-exact CSC values (allowing fp32 cast saturation at 2^24) | GSM4037629 |
| scanpy read_10x_h5 | bit-exact CSC values after dtype cast | GSM4037629 |

The fp32 cast saturates at 2^24 with a stderr warning — uint32 SNP/SJ counts beyond that point lose precision. fp64 is not provided here; promote upstream if needed (see Rule 8 in `agents/singlet-gpu-orchestrator.md`).

Unit tests: `tests/io_pz_device_loader_correctness.cpp` (8/8 PASS as of Cycle 55b).

## Citation

The TP1Z v1 format is defined by [singlify's `pz_writer.h`](../../singlify/include/singlet-pileup/pz_writer.h) and is a singlify-native single-cell exchange format. No external method paper.

singlet-gpu's loader contribution: zero-copy decompression to a CuSPARSE-compatible CSC layout that downstream kernels can consume with no further reshaping. Design doc: [`state/designs/00-pz-device-loader.md`](../../state/designs/00-pz-device-loader.md).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <iostream>

int main() {
    namespace sg = singlet_gpu;

    auto mat = sg::io::load_pz(
        "/mnt/projects/.../GSM4037629/exon_counts.1pz");

    cudaStreamSynchronize(mat.producer_stream);

    std::cout << "loaded: "
              << mat.n_genes << " genes × "
              << mat.n_cells << " cells, "
              << mat.mat.nnz << " nonzeros\n";

    std::cout << "sample: " << mat.meta.gsm_id
              << "  (organism=" << mat.meta.organism
              << ", protocol=" << mat.meta.protocol << ")\n";

    // mat.mat is a factornet::gpu::SparseMatrixGPU<float> ready for
    // any cuSPARSE / cuBLAS / cuSOLVER kernel.

    if (mat.producer_stream && /* we created it */ true)
        cudaStreamDestroy(mat.producer_stream);
}
```

## Pareto-frontier row

| scale | wall_ms | mem_mb | accuracy | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|---|
| small (GSM4037629) | 268.8 | 34 | bit-exact (8/8 gtest) | 1729 (anndata) / 1442 (scanpy) | anndata-gpu, scanpy/read_10x_h5 | wall (6.4×), memory (9.4×) |

100k and 1M scales pending feature 17 (streaming driver) completion.

## Links

- Design doc: [`state/designs/00-pz-device-loader.md`](../../state/designs/00-pz-device-loader.md)
- Format spec: `singlify/include/singlet-pileup/pz_writer.h` (TP1Z v1)
- Equivalence notebook: `docs/notebooks/pz_loader.ipynb` (pending)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § io/pz_device_loader
