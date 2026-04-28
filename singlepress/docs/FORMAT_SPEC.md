# .1pz Binary Format Specification — Version 1.0

> Singlepress native sparse-matrix format for single-cell genomics.
> VOCSC + byte-split + bit-plane + bitmap pre-filter + zstd/LZ4 compression,
> embedded metadata, CRC32 integrity.

---

## 1. Overview

`.1pz` stores a sparse matrix in CSC (Compressed Sparse Column) layout with:

- **Row-frequency permutation** — rows sorted by descending non-zero count
- **VOCSC encoding** — Value-Ordered Column-Sparse Coding with delta-encoded row gaps
- **Byte-split filter** — de-interleaves bytes for entropy reduction
- **Zstd-3 compression** — fast dictionary-based compression
- **Embedded metadata** — row/column names, obs/var DataFrames, key-value pairs

Typical compression: **13× over raw CSC**, decode throughput: **4000+ MB/s**.

---

## 2. File Layout

```
Offset          Section
───────────────────────────────────────────────
0               Header (96 bytes, fixed)
96              Permutation array (zstd)
+perm_z_sz      Column pointer counts (zstd)
+ptr_z_sz       Chunk table: uint32[num_chunks]
+num_chunks×4   Chunk 0 blob
+ctable[0]      Chunk 1 blob
...             ...
+ctable[N-1]    Column sums (zstd uint64[n])
+colsums_z_sz   Metadata section (zstd TLV)
+metadata_z_sz  Transpose section (optional)
+transpose_z_sz ...
EOF-16          Footer (16 bytes, fixed)
```

---

## 3. Header (96 bytes)

All multi-byte values are **little-endian**.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | `0x5A315054` ("TP1Z" in LE) |
| 4 | 2 | `version` | Format version (`1`) |
| 6 | 1 | `vt_code` | Value type: `1`=uint8, `2`=uint16, `3`=uint32 |
| 7 | 1 | `flags` | Bitfield (see §3.1) |
| 8 | 4 | `m` | Number of rows (genes/features) |
| 12 | 4 | `n` | Number of columns (cells) |
| 16 | 8 | `nnz` | Total non-zero entries |
| 24 | 1 | `ptr_width` | Column pointer width: `2` or `4` bytes |
| 25 | 1 | `codec_level` | Zstd compression level (typically `3`) |
| 26 | 2 | `_pad0` | Reserved (zero) |
| 28 | 4 | `num_chunks` | Number of column chunks |
| 32 | 4 | `perm_z_sz` | Compressed permutation array size |
| 36 | 4 | `ptr_z_sz` | Compressed column pointer counts size |
| 40 | 4 | `chunk_cols` | Columns per chunk (default `1024`) |
| 44 | 4 | `feature_flags` | Extended feature flags (see §3.2) |
| 48 | 8 | `metadata_offset` | Byte offset to metadata section, or `0` |
| 56 | 4 | `metadata_z_sz` | Compressed metadata size |
| 60 | 4 | `colsums_z_sz` | Compressed column sums size |
| 64 | 8 | `transpose_offset` | Byte offset to transpose section, or `0` |
| 72 | 4 | `transpose_z_sz` | Compressed transpose total size |
| 76 | 4 | `transpose_chunks` | Number of transpose chunks |
| 80 | 16 | `reserved` | Reserved for future use (zero) |

Total: **96 bytes**.

### 3.1 Flags Byte (offset 7)

| Bit | Mask | Name | Meaning |
|-----|------|------|---------|
| 0 | `0x01` | `FLAG_HAS_PERM` | Permutation array present |
| 1 | `0x02` | `FLAG_GAP16` | 16-bit gap encoding (else 32-bit) |
| 2 | `0x04` | `FLAG_HAS_METADATA` | Metadata section present |
| 3 | `0x08` | `FLAG_HAS_TRANSPOSE` | Transpose (CSR) section stored |
| 4 | `0x10` | `FLAG_HAS_COLSUMS` | Column sums array stored |
| 5 | `0x20` | `FLAG_HAS_OBS_VAR` | Obs/var DataFrames embedded |

### 3.2 Feature Flags (offset 44)

| Bit | Mask | Name | Meaning |
|-----|------|------|---------|
| 0 | `0x01` | `FEAT_ZSTD_CHECKSUMS` | Zstd checksum mode enabled |
| 1 | `0x02` | `FEAT_BITPLANE_BITMAP` | Bit-plane(p0) + bitmap pre-filter |

---

## 4. Compression Pipeline

### 4.1 Row-Frequency Permutation

Before encoding, rows are sorted by **descending non-zero count**. This clusters frequently-used row indices near zero, producing smaller gap values.

- Stored as `uint32_t[m]` array, zstd-compressed
- Size recorded in `perm_z_sz`
- Present when `FLAG_HAS_PERM` is set

### 4.2 Column Pointer Counts

Per-column non-zero counts, stored as:
- `uint16_t[n]` if `ptr_width == 2` (max 65,535 nnz per column)
- `uint32_t[n]` if `ptr_width == 4`

Zstd-compressed; size in `ptr_z_sz`. Reconstruct CSC `indptr` by cumulative sum:

```
indptr[0] = 0
indptr[j+1] = indptr[j] + counts[j]   for j in [0, n)
```

### 4.3 VOCSC Encoding

Data is split into **chunks** of `chunk_cols` columns (default 1024). Each chunk is independently encoded:

1. **Per-column grouping**: Within each column, non-zero entries are grouped by value
2. **LEB128 varint metadata**:
   - Per column: number of distinct values (groups)
   - Per group: value, then count of entries with that value
3. **Gap encoding**: Row indices within each group are sorted and stored as deltas:
   ```
   gap[0] = row_index[0]
   gap[i] = row_index[i] - row_index[i-1]    for i > 0
   ```

### 4.4 Byte-Split Filter

Gap arrays are split into separate byte planes for better entropy:

**16-bit mode** (`FLAG_GAP16` set, when `max_gap < 65536`):
```
Plane 0: low byte of each gap
Plane 1: high byte of each gap
```

**32-bit mode** (larger gaps):
```
Plane 0: byte 0 of each gap
Plane 1: byte 1
Plane 2: byte 2
Plane 3: byte 3
```

SSE2-accelerated unsplit on decode with scalar fallback.

### 4.5 Zstd Compression

Each chunk blob:

| Field | Type | Description |
|-------|------|-------------|
| `n_gaps` | uint32 | Number of gap values in chunk |
| `metadata_size` | uint32 | Size of VOCSC metadata |
| `compressed_size` | uint32 | Size of compressed payload |
| `chunk_crc32` | uint32 | CRC32 of pre-compression data |
| `packed_size` | uint32 | Pre-compression size (when `FEAT_BITPLANE_BITMAP` set) |
| `payload` | bytes | Compressed (metadata + byte-split gap planes) |

### 4.6 Chunk Table

Immediately after the column pointer section: `uint32_t[num_chunks]` array of compressed chunk sizes. Each entry is the total blob size for that chunk. Not compressed.

---

## 5. Column Sums

When `FLAG_HAS_COLSUMS` is set: zstd-compressed `uint64_t[n]` array of per-column sums. Enables fast library-size normalization without decompressing the matrix. Size in `colsums_z_sz`.

---

## 6. Metadata Section (TLV Format)

When `FLAG_HAS_METADATA` is set, a zstd-compressed byte sequence at `metadata_offset` containing **Tag-Length-Value** entries:

| Tag | Name | Format |
|-----|------|--------|
| 0 | `END` | Sentinel — no size or data |
| 1 | `ROWNAMES` | Null-delimited (`\0`) UTF-8 strings, `m` entries |
| 2 | `COLNAMES` | Null-delimited UTF-8 strings, `n` entries |
| 3 | `KV` | Alternating `key\0value\0` pairs |
| 4 | `OBS` | Native columnar DataFrame (see §6.1) |
| 5 | `VAR` | Native columnar DataFrame (see §6.1) |

Each entry (except END):
```
[uint8 tag] [uint32 size] [data of `size` bytes]
```

### 6.1 Native DataFrame Binary Format

Used for OBS (cell annotations) and VAR (feature annotations):

```
[uint32 nrows]
[uint32 ncols]
[uint32 index_bytes]
[index: null-delimited UTF-8 strings]
For each column:
  [uint16 col_name_length]
  [col_name bytes]
  [uint8 dtype]
  [uint32 col_data_size]
  [column data]
```

**DataFrame dtypes:**

| Code | Name | Encoding |
|------|------|----------|
| 0 | `DF_STRING` | Null-separated UTF-8 strings |
| 1 | `DF_INT32` | `nrows × 4` bytes, little-endian |
| 2 | `DF_INT64` | `nrows × 8` bytes, little-endian |
| 3 | `DF_FLOAT32` | `nrows × 4` bytes, little-endian |
| 4 | `DF_FLOAT64` | `nrows × 8` bytes, little-endian |
| 5 | `DF_UINT8` | `nrows` bytes (booleans: 0/1) |
| 6 | `DF_CATEGORICAL` | `[uint32 nlev][null-sep levels][nrows × int32 codes]` |

---

## 7. Transpose Section (Optional)

When `FLAG_HAS_TRANSPOSE` is set, a CSR-layout copy is appended for efficient row slicing. Starts at `transpose_offset`:

**TransposeHeader (16 bytes):**

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | `perm_z_sz` |
| 4 | 4 | `ptr_z_sz` |
| 8 | 4 | `chunk_cols` |
| 12 | 1 | `ptr_width` |
| 13 | 1 | `flags` (bit 0 = gap16) |
| 14 | 2 | `_pad` |

Followed by the same structure as the forward section: permutation, pointer counts, chunk table, chunk blobs.

---

## 8. Footer (16 bytes)

Last 16 bytes of the file:

| Offset from EOF | Size | Field | Description |
|-----------------|------|-------|-------------|
| -16 | 4 | `file_crc32` | CRC32 of bytes `[0, EOF-16)` |
| -12 | 4 | `_reserved` | Reserved (zero) |
| -8 | 4 | `num_chunks` | Redundant check against header |
| -4 | 4 | `magic` | `0x5A315054` (must match header) |

---

## 9. Value Types

| `vt_code` | C Type | Range | Typical Use |
|-----------|--------|-------|-------------|
| 1 | `uint8_t` | 0–255 | Low-depth UMI counts |
| 2 | `uint16_t` | 0–65,535 | Standard UMI counts |
| 3 | `uint32_t` | 0–2³²−1 | Full-length / Smart-seq counts |

The writer automatically selects the smallest type that fits all values.

---

## 10. Conventions

- **Byte order**: All multi-byte fields are little-endian
- **Alignment**: No alignment requirements beyond natural packing
- **String encoding**: UTF-8 throughout, null-terminated/delimited
- **Default chunk size**: 1024 columns
- **Default compression**: zstd level 3
- **File extension**: `.1pz`
- **MIME type**: `application/x-singlepress`

---

## 11. Version History

| Version | Changes |
|---------|---------|
| 1 | Initial VOCSC + zstd format |
| 2 | Added metadata, column sums, transpose |
| 3 | Native DataFrame obs/var, KV metadata, CRC32 footer, feature flags |
