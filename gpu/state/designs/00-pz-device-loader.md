---
feature: pz_device_loader
roadmap_id: 0
module: include/singlet-gpu/io/pz_device_loader.h
status: design
tolerance: scipy.sparse.csc_matrix equality (indptr, indices, data) bit-identical for fp32 cast
target_perf: 10k-cell sample loaded + on device in <50ms; 100k <250ms; 1M <2s (single GPU, PCIe gen4)
ooc_plan: chunked load via PzChunkIterator yielding fixed-size column slices; per-chunk pinned-host stage + cudaMemcpyAsync overlap with next chunk decompress
---

## Algorithm

singlet-gpu's `.1pz` device loader is the single ingestion point for every downstream kernel. It reads a `.1pz` file (TP1Z v1, header + zstd VOCSC chunks + footer) on the CPU, decompresses chunks into a pinned host CSC staging buffer, and `cudaMemcpyAsync`'s the result into a `factornet::gpu::SparseMatrixGPU<float>`. No densification, no dtype widening beyond uint8/uint16/uint32 → fp32.

Pipeline:
1. Open file + read 96-byte header (mirror `pz_writer.h::PZHeader` struct via `#pragma pack(push, 1)`).
2. Read footer (16 bytes) and validate `file_crc32` + magic.
3. Read metadata TLV block if `FLAG_HAS_METADATA` is set; parse rownames, colnames, user_kv (GEO context).
4. Read permutation block if `FLAG_HAS_PERM`; expanded to `uint32[m]`.
5. Read column-pointer block (varint stream); decompress to `int32[n+1]` via prefix sum.
6. Allocate pinned host buffers: `int32 indptr[n+1]`, `int32 indices[nnz]`, `float data[nnz]`.
7. Iterate over VOCSC chunks; for each chunk: decompress to local row-index + value buffers, scatter into `indices` + `data` at the correct column offset, advance.
8. Cast values uint8/uint16/uint32 → fp32 during scatter (saturating; values beyond 2^24 are flagged but not refused).
9. Allocate `factornet::gpu::SparseMatrixGPU<float>` on the GPU (m, n, nnz) via factornet's RAII `DeviceMemory<int32>` + `DeviceMemory<float>`.
10. Three async copies on a high-priority stream: `indptr`, `indices`, `data`.
11. Stream-attach the embedded GEO metadata as a `singlet_gpu::core::Metadata` struct on the host side (does not live on device).
12. Return a `singlet_gpu::io::PzDeviceMatrix` containing the device CSC + host metadata + the producer stream.

Streaming variant (`PzChunkIterator`) yields fixed-column-width slices for out-of-core kernels; same pipeline, but each `next()` returns one chunk's worth of CSC.

## Numerical stability

- uint8/uint16/uint32 → fp32 cast is exact for values up to 2^24 (16M). Beyond that we lose precision in the mantissa; warn but do not refuse. SNP DP / SJ counts can exceed this in rare deeply-sequenced samples — for those, a fp64 overload (`PzDeviceMatrixDouble`) is exposed but defaults off.
- No accumulation in this kernel; precision concerns are downstream.

## Memory layout

- Output: column-major CSC (`int32 indptr[n+1]`, `int32 indices[nnz]`, `float data[nnz]`), bit-compatible with `factornet::gpu::SparseMatrixGPU<float>`.
- Pinned host staging is allocated from `core::PinnedPool` (a thin wrapper over `cudaMallocHost`). Released after the async copy completes.
- Device buffers are owned by `factornet::gpu::DeviceMemory<T>` (RAII).
- Workspace budget: 3 × nnz × 4 bytes pinned host + same on device, plus indptr (4(n+1) bytes). For 1M cells × 1B nnz: ~12 GB pinned, ~12 GB device. For 100k × 100M nnz: ~1.2 GB each.

## Streams

- One high-priority stream owned by the loader for async copies.
- Caller can pass an existing `cudaStream_t`; default is a stream from `core::StreamPool::high_priority()`.
- Decompression and scatter happen on the host while the previous chunk's copy is in flight (CPU/GPU overlap).

## Out-of-core chunking

`PzChunkIterator` yields fixed-column-width slices (default 100k columns, configurable). Each slice is a complete `factornet::gpu::SparseMatrixGPU<float>` on its own; downstream kernels consume them sequentially. Total memory footprint is bounded by `chunk_cols × max_nnz_per_col × 4 bytes` plus headroom for ping-pong prefetch.

For the 1M-cell benchmark and beyond, the iterator is the only sane way to load. It also enables `factornet::nmf::fit_streaming_spz`-style adapters by implementing factornet's `io::loader<T>` interface (see `state/integration-notes.md` open question #2).

## Determinism

The loader is fully deterministic — no atomics, no reductions, no stochasticity. The output device CSC is bit-identical for the same `.1pz` input across runs and architectures (modulo PCIe transfer ordering, which does not affect bytes received).

## Correctness test spec

The test (in `tests/io_pz_device_loader_correctness.cpp`):

1. Tiny synthetic: write a `.1pz` from a fixed-seed scipy CSC matrix (500 × 200) using `singlify.io.write_pz` (or directly via singlify's `pz_writer.h`). Load via `pz_device_loader`. Copy device CSC back to host. Compare element-by-element to the original scipy matrix using `numpy.array_equal` semantics on indptr/indices and `numpy.allclose(rtol=0)` on data.
2. Real sample: GSM4037629 exon_counts.1pz (11,560 cells). Load via the C++ loader. Subprocess: `python -c "import singlify.io as io; m = io.read_pz('exon_counts.1pz'); print(m.shape, m.nnz, m.indptr.tolist()[:5], m.indices.tolist()[:5])"`. Compare.
3. Streaming: load the same sample via `PzChunkIterator` with chunk_cols=2000. Concatenate chunks. Equal to the in-one-go load.
4. GEO metadata: confirm `gsm_id, gse_id, organism, taxon_id, protocol, modality, srr_ids[0], read_count, geo_title, singlify_version, pipeline_date` round-trip from the .1pz metadata TLV block.

Tolerance: indptr, indices = bit-identical; data = bit-identical for inputs ≤ 2^24; metadata strings = bit-identical UTF-8.

Reference implementation: `singlify.io` (Python). Subprocess installs from `singlify/python/` — verify it's importable in the bench venv before running.

## Target performance

| Scale | Cells | nnz | Target wall | Target peak host | Target peak device |
|---|---|---|---|---|---|
| tiny | 200 | 2k | <1ms | <1MB | <1MB |
| 10k | 11,560 | ~30M | <50ms | <400MB | <400MB |
| 100k | ~120k | ~300M | <250ms | <4GB | <4GB |
| 1M | ~1M | ~3B | <2s (chunked) | <4GB (sliding) | <4GB (sliding) |

SOTA to beat: scanpy `read_10x_h5` + `cupy.sparse.csr_matrix` upload (typical: 200ms / 1.5s / 12s for the same scales). Win expected on the chunked path because singlify's `.1pz` is ~3× smaller than `.h5` and decompresses faster than `h5py`.

## Implementation notes (for gpu-kernel-dev)

- Mirror `pz_writer.h` constants exactly. Read `pz_reader.h` first if it has a CPU reader implementation worth copying; if so, port the byte-level parser, then layer the device staging on top.
- Use `#pragma pack(push, 1)` on `PZHeader` and `PZFooter`.
- Use `core::PinnedPool` (to be implemented in `core/memory.h` as a thin wrapper around `cudaMallocHost`).
- Output type: `singlet_gpu::io::PzDeviceMatrix` — a struct holding `factornet::gpu::SparseMatrixGPU<float>`, host-side `Metadata`, and the producer stream.
- API:
  ```cpp
  namespace singlet_gpu::io {
      struct Metadata { /* gsm_id, gse_id, ... rownames, colnames */ };
      struct PzDeviceMatrix {
          factornet::gpu::SparseMatrixGPU<float> mat;
          Metadata meta;
          cudaStream_t producer_stream;
      };
      PzDeviceMatrix load_pz(const std::string& path,
                             cudaStream_t stream = nullptr);
      class PzChunkIterator { /* ... */ };
  }
  ```

## Dependencies on other features

- `core/memory.h` (`PinnedPool`) — must be implemented BEFORE this feature.
- `core/handles.h` (`GPUContext` re-export from factornet) — must exist.
- `core/types.h` (`DeviceCSC` alias) — must exist.

So the actual ordering is **feature 1 (core) before feature 0 (loader)**. The roadmap says feature 0 first, but the dependency runs the other way — we'll need to flip them or implement a minimal `core/` slice as part of cycle 1. Update the roadmap and dag.md accordingly.
