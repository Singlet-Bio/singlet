# .1pz Format Specification (TP1Z v1)

## Overview

`.1pz` is singlet's native sparse matrix format — a **VOCSC** (Value-Offset Compressed Sparse Column) file with zstd block compression. It stores single-cell feature matrices (genes × cells) in a format optimised for:

1. **Fast random column (cell) access** — one column decompresses in ~10 µs
2. **Very high compression ratio** — UMI count matrices compress 40–80× vs MTX
3. **Direct read into R/Python sparse objects** — no intermediate copy

Output `.1pz` files are readable by singlet (`singlet.read_1pz()` in Python, `singlet::read_1pz()` in R).

---

## File layout

```
[ PZHeader (96 bytes) ]
[ optional: permutation block (zstd-compressed) ]
[ optional: metadata block (zstd-compressed JSON) ]
[ optional: column sums block (zstd-compressed) ]
[ chunk 0: ptr_chunk + val_chunk ]
[ chunk 1: ptr_chunk + val_chunk ]
...
[ chunk N-1 ]
[ optional: transpose (CSR) section ]
[ PZFooter (16 bytes) ]
```

---

## PZHeader (96 bytes, packed)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | `0x5A315054` = bytes `"TP1Z"` |
| 4 | 2 | `version` | Format version = 1 |
| 6 | 1 | `vt_code` | Value type code (see below) |
| 7 | 1 | `flags` | Feature flags byte (see below) |
| 8 | 4 | `m` | Number of rows (features / genes) |
| 12 | 4 | `n` | Number of columns (cells / barcodes) |
| 16 | 8 | `nnz` | Total non-zero entries |
| 24 | 1 | `ptr_width` | Bytes per column pointer (1, 2, or 4) |
| 25 | 1 | `codec_level` | zstd compression level used |
| 26 | 2 | `_pad0` | Reserved (must be 0) |
| 28 | 4 | `num_chunks` | Number of column chunks |
| 32 | 4 | `perm_z_sz` | Compressed size of permutation block (0 = absent) |
| 36 | 4 | `ptr_z_sz` | Compressed size of pointer array (0 = absent) |
| 40 | 4 | `chunk_cols` | Columns per chunk |
| 44 | 4 | `feature_flags` | Extended feature flags (see below) |
| 48 | 8 | `metadata_offset` | Byte offset of metadata block (0 = absent) |
| 56 | 4 | `metadata_z_sz` | Compressed size of metadata block |
| 60 | 4 | `colsums_z_sz` | Compressed size of column sums (0 = absent) |
| 64 | 8 | `transpose_offset` | Byte offset of transpose/CSR section (0 = absent) |
| 72 | 4 | `transpose_z_sz` | Compressed size of transpose block |
| 76 | 4 | `transpose_chunks` | Number of chunks in transpose section |
| 80 | 16 | `reserved` | Reserved (must be 0) |

Total: **96 bytes** (verified by `static_assert`).

### Value type codes (`vt_code`)

| Code | C type | Description |
|------|--------|-------------|
| 0 | `float` | 32-bit IEEE float |
| 1 | `double` | 64-bit IEEE float |
| 2 | `int32_t` | 32-bit signed integer |
| 3 | `uint32_t` | 32-bit unsigned integer |
| 4 | `int16_t` | 16-bit signed integer |
| 5 | `uint16_t` | 16-bit unsigned integer |
| 6 | `uint8_t` | 8-bit unsigned integer |

UMI count matrices use `uint16_t` (code 5). SNP AD/DP matrices use `uint32_t` (code 3).

### Header flags byte (`flags`)

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `FLAG_HAS_PERM` | Permutation block present (cells reordered for compression) |
| 1 | `FLAG_GAP16` | Row indices stored as 16-bit gap-encoded values |
| 2 | `FLAG_HAS_METADATA` | Metadata JSON block present |
| 4 | `FLAG_HAS_COLSUMS` | Column sums block present |

### Extended feature flags (`feature_flags`)

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `FEAT_ZSTD_CHECKSUMS` | Each chunk includes a zstd checksum |
| 1 | `FEAT_BITPLANE_BITMAP` | Values bitplane-encoded before compression |

---

## PZFooter (16 bytes, packed)

Immediately follows the last chunk (or transpose section if present).

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `file_crc32` | ISO 3309 CRC32 of entire file up to (not including) footer |
| 4 | 4 | `_reserved` | Reserved (must be 0) |
| 8 | 4 | `num_chunks` | Redundant chunk count (for validation) |
| 12 | 4 | `magic` | `0x5A315054` — same magic as header (identifies end of file) |

---

## VOCSC encoding

Each chunk stores a set of consecutive columns in **VOCSC** format:

```
[ compressed pointer block ]
[ compressed row-index + value block ]
```

The pointer block is an array of `ptr_width`-byte offsets: `ptr[i]` is the start offset of column `i`'s data within the row-index+value block (after decompression). The last entry is the total data size.

Row indices are stored gap-encoded (delta from previous row index within the column), then byte-split (all low bytes together, then high bytes) before zstd compression. This allows zstd to exploit the high correlation between adjacent row indices in biological data.

Values are byte-split (value[0]_lo, value[0]_hi, ...) and then optionally bitplane-encoded before compression.

---

## Permutation block

When `FLAG_HAS_PERM` is set, the permutation block immediately follows the header. It maps output column order → input barcode order: `perm[i]` is the index of the cell assigned to output column `i`. The cells are grouped by non-zero density for improved compression (dense columns together).

---

## Metadata block

When `FLAG_HAS_METADATA` is set, the metadata block is a zstd-compressed UTF-8 JSON string containing:

```json
{
  "rownames": ["ENSG00000001", "ENSG00000002", ...],
  "colnames": ["AACGTAGCTAGCT-1", "AACCTTGCTAGCT-1", ...],
  "singlet_version": "0.2.0",
  "pipeline_mode": "exon_counts",
  "created": "2026-04-10T14:30:00Z"
}
```

Rownames are feature identifiers (gene IDs or SNP positions). Colnames are cell barcodes.

---

## Outputs produced by singlet

| File | Value type | Rows | Columns | Description |
|------|-----------|------|---------|-------------|
| `exon_counts.1pz` | `uint16_t` | genes | cells | UMI-deduplicated exon counts |
| `intron_counts.1pz` | `uint16_t` | genes | cells | UMI-deduplicated intron counts |
| `sj_counts.1pz` | `uint32_t` | junctions | cells | Splice junction read counts |
| `snp_ad.1pz` | `uint32_t` | SNP sites | cells | Allele depth (alt reads) |
| `snp_dp.1pz` | `uint32_t` | SNP sites | cells | Total depth (all reads) |
| `mt_heteroplasmy.1pz` | `float` | mt sites | cells | Per-cell mitochondrial VAF |

---

## Reading .1pz files

### Python (singlet)

```python
import singlet

# Returns scipy.sparse.csc_matrix via AnnData
adata = singlet.read_1pz("exon_counts.1pz")

# Or use the low-level I/O
from singlet.io import read_1pz
mat = read_1pz("exon_counts.1pz")
```

### R (singlet)

```r
library(singlet)

# Returns dgCMatrix
mat <- read_1pz("exon_counts.1pz")

# Or directly into a SingleCellExperiment
sce <- as_sce("results/")
```

---

## Format versioning

The current format is **TP1Z v1**. The `reserved[16]` bytes in the header and the `_reserved` field in the footer are available for future extensions. A reader MUST skip unknown flags and unknown metadata fields. The `vt_code` field allows future value types to be added without breaking existing readers (an unknown `vt_code` should cause the reader to return an error rather than silently misinterpret data).
