# C++ API Reference

## Overview

The core .1pz codec is implemented in `singlepress/pz_codec.cpp` as a single-file
C++17 implementation with pybind11 bindings. For HPC and embedded use, the codec
can be used directly from C++ by including the relevant functions.

## Data Structures

### PZHeader (96 bytes)

```cpp
struct PZHeader {
    uint32_t magic;           // 0x5A315054 ("TP1Z")
    uint16_t version;         // 3
    uint8_t  vt_code;         // 1=uint8, 2=uint16, 3=uint32
    uint8_t  flags;           // Bitfield: HAS_PERM, GAP16, HAS_METADATA, etc.
    uint32_t m, n;            // Matrix dimensions (rows, columns)
    uint64_t nnz;             // Number of non-zeros
    uint8_t  ptr_width;       // Column pointer width: 2 or 4
    uint8_t  codec_level;     // Zstd compression level
    uint16_t _pad0;
    uint32_t num_chunks;      // Number of column chunks
    uint32_t perm_z_sz;       // Compressed perm array bytes
    uint32_t ptr_z_sz;        // Compressed pointer array bytes
    uint32_t chunk_cols;      // Columns per chunk (default 1024)
    uint32_t feature_flags;   // Extended feature flags
    uint64_t metadata_offset; // Byte offset to metadata section
    uint32_t metadata_z_sz;   // Compressed metadata bytes
    uint32_t colsums_z_sz;    // Compressed colsums bytes
    uint64_t transpose_offset;
    uint32_t transpose_z_sz;
    uint32_t transpose_chunks;
    uint8_t  reserved[16];
};
```

### PZFooter (16 bytes)

```cpp
struct PZFooter {
    uint32_t file_crc32;   // CRC32 of all preceding bytes
    uint32_t _reserved;
    uint32_t num_chunks;   // Redundant check
    uint32_t magic;        // 0x5A315054 (validation)
};
```

### CSCView

Non-owning view over a CSC sparse matrix:

```cpp
struct CSCView {
    uint32_t m, n;
    uint64_t nnz;
    const uint32_t* p;  // Column pointers, length n+1
    const uint32_t* i;  // Row indices, length nnz
    const double* x;    // Values, length nnz
};
```

## Core Functions

### CRC32

```cpp
struct CRC32 {
    void update(const uint8_t* data, size_t len);
    uint32_t finalize() const;
    static uint32_t compute(const uint8_t* data, size_t len);
};
```

ISO 3309 polynomial (0xEDB88320). Used for per-chunk and file-level integrity.

### Compression

```cpp
// Compress data with zstd
std::vector<uint8_t> zstd_compress(
    ZSTD_CCtx* ctx, const uint8_t* data, size_t len, int level);

// Decompress data with zstd
size_t zstd_decompress(
    ZSTD_DCtx* ctx, void* dst, size_t cap, const void* src, size_t len);
```

### Byte-Split / Unsplit

```cpp
// Split 16/32-bit arrays into byte planes for better compression
void byte_split_16(const uint32_t* src, size_t n, uint8_t* dst);
void byte_split_32(const uint32_t* src, size_t n, uint8_t* dst);

// Reconstruct from byte planes (SSE2-accelerated with scalar fallback)
void byte_unsplit_16(const uint8_t* src, size_t n, uint32_t* dst);
void byte_unsplit_32(const uint8_t* src, size_t n, uint32_t* dst);
```

### Varint (LEB128)

```cpp
void varint_push(std::vector<uint8_t>& out, uint32_t value);
uint32_t varint_read(const uint8_t*& ptr);  // Advances pointer
```

### VOCSC Encoding/Decoding

```cpp
// Encode a chunk of columns using Value-Ordered Column-Sparse Coding
VocscResult vocsc_encode_chunk(
    const CSCView& view,
    const uint32_t* perm,
    const uint32_t* inv_perm,
    int64_t col_start, int64_t col_end,
    std::vector<uint8_t>& meta_out);

// Decode a VOCSC chunk
template <typename ValT>
void vocsc_decode_chunk(
    const uint8_t* meta, const uint32_t* gaps,
    const uint32_t* perm, const uint32_t* inv_perm,
    int64_t col_start, int64_t col_end,
    const int32_t* indptr, int32_t* indices, ValT* values);
```

### Full Read/Write

```cpp
// Write CSC matrix to .1pz file
py::dict pz_write_core(
    const uint32_t* indptr, const uint32_t* indices, const double* data,
    uint32_t m, uint32_t n, uint64_t nnz,
    const std::string& path,
    int num_threads, int level, int chunk_cols,
    const std::vector<std::string>& rownames,
    const std::vector<std::string>& colnames,
    bool store_transpose,
    /* ... obs/var/uns metadata ... */);

// Full read: decompress all chunks into indptr/indices/values arrays
template <typename ValT>
py::dict pz_read_typed(const std::string& path, int num_threads);

// Column-range read: decompress only chunks covering [col_start, col_end)
template <typename ValT>
py::dict pz_read_columns_typed(
    const std::string& path, int64_t col_start, int64_t col_end, int num_threads);

// Row-range read (requires stored transpose)
template <typename ValT>
py::dict pz_read_rows_typed(
    const std::string& path, int64_t row_start, int64_t row_end, int num_threads);
```

## Threading Model

The codec uses OpenMP for parallel chunk encoding/decoding:

```cpp
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
for (int c = 0; c < num_chunks; ++c) {
    // Thread-local zstd contexts (RAII wrappers)
    ZstdCCtx cctx;
    ZstdDCtx dctx;
    // Decode chunk c independently
}
```

Each chunk is fully independent, enabling:
- **Parallel encode** during write (4–8× speedup with 8 threads)
- **Parallel decode** during read (near-linear scaling)
- **Partial reads** with zero wasted decompression

## pybind11 Module Bindings

The following functions are exposed to Python via pybind11:

| Function | Parameters | Returns |
|----------|-----------|---------|
| `pz_write` | indptr, indices, data (f64), nrows, path, kwargs | dict |
| `pz_write_int` | indptr, indices, data (i32), nrows, path, kwargs | dict |
| `pz_read` | path, num_threads | dict |
| `pz_read_int` | path, num_threads | dict |
| `pz_read_columns` | path, col_start, col_end, num_threads | dict |
| `pz_read_columns_int` | path, col_start, col_end, num_threads | dict |
| `pz_read_rows` | path, row_start, row_end, num_threads | dict |
| `pz_colsums` | path | numpy array (uint64) |
| `pz_info` | path | dict |
| `pz_validate` | path | dict |

## Building from Source

```bash
# Build the extension
pip install -e ".[dev]"

# Or build manually
cd singlepress
c++ -std=c++17 -O3 -fopenmp -march=native -shared -fPIC \
    -I$(python -c "import pybind11; print(pybind11.get_include())") \
    $(python3-config --includes) \
    pz_codec.cpp -o _pz_codec$(python3-config --extension-suffix) \
    -lzstd
```
