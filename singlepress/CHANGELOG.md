# Changelog

## [1.0.0] — 2026-04-07

### Format
- `.1pz` v1.0 binary format specification (`docs/format_spec.md`)
- VOCSC + byte-split + bit-plane(p0) + bitmap pre-filter compression pipeline
- Multi-codec support: zstd (default), LZ4 (fast mode), LZ4-HC
- Feature flags for forward-compatible extensions (`FEAT_ZSTD_CHECKSUMS`, `FEAT_BITPLANE_BITMAP`)
- CRC32 per-chunk and whole-file integrity verification
- Native columnar DataFrame encoding for obs/var metadata
- Key-value pair metadata for study-level annotations
- Column sums for fast library-size normalization without decompression
- Optional transpose (CSR) section for efficient row slicing

### API
- `write_1pz()` with `mode` parameter: `"fast"` (LZ4), `"default"` (zstd), `"small"` (zstd-16)
- `read_1pz()` / `read_1pz_int()` with on-the-fly log-normalization
- `read_1pz_columns()` for partial column-range reads
- `read_1pz_rows()` for partial row-range reads via transpose
- `colsums_1pz()` for reading column sums without matrix decode
- `validate_1pz()` for CRC32 integrity verification
- `open_1pz()` — lazy `OnePZFile` handle with `.obs`, `.var`, `.uns`, `.shape`, slicing
- `cbind_1pz()` / `rbind_1pz()` for dataset concatenation
- `subset_1pz()` / `sample_1pz()` for subsetting and downsampling
- `lognorm()` for log-normalization
- PyTorch zero-copy dataloaders (`OnePZDataset`, `OnePZCellDataset`, `OnePZShuffleDataset`)
- AnnData, Seurat, SingleCellExperiment, HDF5 interop
- Command-line interface (`singlepress info/validate/colsums/inspect`)

### Removed
- Legacy `.spz` format support
