# SinglePress SOTA Competitor Analysis

## Table 1: SOTA Competitors

| Tool / Format | Paper | DOI | Key Claim | Benchmark Dataset |
|---------------|-------|-----|-----------|-------------------|
| **H5AD (AnnData)** | Virshup et al. 2021; Wolf et al. 2018 | 10.1101/2021.12.16.473007; 10.1186/s13059-017-1382-0 | De facto Python scRNA-seq standard (Scanpy/AnnData ecosystem) | Various |
| **BPCells** | Parks & Greenleaf 2025 | 10.1101/2025.03.27.645853 | Bitpacking + LZ4 compression; disk-backed streaming reduces RAM ~70×; integrated into Seurat v5 | 44M cell atlas on laptop; 1.3M PBMC |
| **scipy npz (CSC/CSR)** | Part of SciPy | — | Standard sparse matrix interchange format | Various |
| **10x HDF5** | 10x Genomics | — (vendor docs) | Cell Ranger native output; HDF5-based feature-barcode matrices | 10x datasets |
| **RDS dgCMatrix** | R Matrix package | — | R/Bioconductor ecosystem standard sparse matrix | Various |
| **TileDB-SOMA** | TileDB Inc. + CZI | — (software) | Cloud-native array storage; SOMA API for CZ CELLxGENE Census | CZI Census (33M+ cells) |
| **Zarr** | Miles et al. 2024 | 10.1038/s41597-024-03174-3 | Chunked cloud-native multidimensional arrays; Python-first | Various |
| **Parquet** | Apache Foundation | — | Columnar analytics format; used by some atlas projects | Various |
| **Loom** | Linnarsson Lab | — (software spec) | HDF5-based scRNA format; largely superseded by H5AD | Various |
| **VCSC** | Wolfgang et al. 2024; Ruiter et al. 2024 | 10.1109/BigData62323.2024.10825091; 10.1109/DCC58796.2024.00065 | Value-compressed sparse column; predecessor to SinglePress | scRNA-seq benchmarks |
| **MEX (MatrixMarket)** | Boisvert et al. 1997 | 10.1007/978-1-5041-2940-4_9 | Text-based sparse matrix exchange format; Cell Ranger output | Various |

## Table 2: Proposed Benchmark Datasets

| Dataset | Cells | Source | Justification |
|---------|-------|--------|---------------|
| 3,253 scRNA-seq from GEO | ~500M total reads | GEO/SRA | Cross-species, cross-protocol generalization; real-world diversity |
| 1.3M PBMC (10x) | 1.3M | 10x Genomics | Standard large-scale benchmark; used by BPCells and Seurat |
| Mouse brain atlas | ~1M | 10x Genomics | Large non-human dataset; tests species generalization |
| CZI Census subset | variable | CZI CELLxGENE | Cloud-native comparison with TileDB-SOMA |
| 44M cell merged atlas | 44M | BPCells paper | Stress test: largest published single-cell dataset benchmark |

## Table 3: Acceptance Metrics

| Metric | Threshold | SOTA Baseline |
|--------|-----------|---------------|
| Compression ratio vs raw CSC | ≥8× median | scipy npz ~2×, H5AD ~3×, BPCells ~5× |
| Decode throughput | ≥500 MB/s | H5AD ~200 MB/s, BPCells ~400 MB/s |
| Write throughput | ≥200 MB/s | H5AD ~100 MB/s |
| Peak RSS (read 1M cells) | ≤2 GB | H5AD ~8 GB, BPCells ~1 GB (disk-backed) |
| Column-range read latency | ≤10 ms for 1000 columns | H5AD: full scan required |
| Gene-slice random access | O(1) seek | H5AD: O(n) scan; BPCells: O(1) with index |
| Round-trip fidelity | Bit-exact | All formats: lossless required |
| Language bindings | Python + R + C++ | H5AD: Python only; BPCells: R only; TileDB-SOMA: Python + R |

## Notes

- **BPCells** is the strongest direct competitor. Their March 2025 bioRxiv preprint introduces bitpacking compression for sparse matrices and ATAC fragments. Key differentiation for SinglePress: dual-language support (Python + R + C++), value compression (VCSC ancestry), and streaming decode without full decompression.
- **TileDB-SOMA** targets the cloud-native atlas use case (CZ CELLxGENE Census). Not optimized for local single-sample read/write throughput.
- **H5AD** remains the default interchange format but has well-known limitations: poor random column access, high memory footprint, and Python-only native support.
- **Zarr** is gaining traction for cloud storage but is not specifically optimized for sparse single-cell count matrices.
- No significant new competitors discovered in 2024–2026 literature search beyond those listed above.
