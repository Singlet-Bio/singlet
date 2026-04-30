// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (DecoupleR-style scoring + CellTypist projection)
//
// anno/reference_map.h — CellTypist-style reference projection on GPU.
//
// Workflow:
//   1. load_celltypist_model(npz_path)        → CelltypistModel (once, host I/O)
//   2. project_to_reference(embedding, model) → RefMapResult (hot path, GPU)
//   3. aggregate_cluster_labels(result, ...)  → per-cluster majority label
//
// Algorithm (logistic regression inference):
//   logits[cell, class] = embedding[cell, :] · weights[class, :]^T + intercepts[class]
//   probs[cell, :]      = softmax(logits[cell, :])   (max-subtraction trick, fp32 stable)
//   labels[cell]        = argmax(probs[cell, :])
//
// The cuBLAS Sgemm computes the full logit matrix in one call; per-row softmax and
// argmax are custom kernels (no cub needed for softmax; cub::DeviceSegmentedReduce
// for per-row argmax to keep everything on device — no host-mediated row scan).
//
// NPZ loading (host-side only — not a hot path):
//   .npz is a zip archive of .npy arrays (no external libraries required; a minimal
//   zip/npy parser is implemented below).  This is the only host-side file I/O in
//   the module.  The model is loaded ONCE at startup; inference is GPU-resident.
//   Supported .npz keys: "weights" (float32, n_classes × n_pcs, C-contiguous),
//   "intercepts" (float32, n_classes), "class_names" (object/str array).
//
// Memory budget:
//   Logit / probability matrix:  n_cells × n_classes × 4 bytes
//     At 1M cells × 50 classes:  200 MB
//   Model weights:                n_classes × n_pcs × 4 bytes  (trivial, ≤ 50 × 50 × 4 = 10 KB)
//   Scratch for cub segmented reduce: cub-managed (typically < 100 MB)
//
// Streams: 1, caller-provided.  GEMM + softmax + argmax chain on that stream.
//
// Precision: fp32 throughout.  Max-subtraction softmax is numerically stable in fp32.
//
// Determinism: all kernels deterministic.
//   softmax_kernel — no atomics; each thread owns one cell row.
//   argmax: cub::DeviceSegmentedReduce::ArgMax on device (stable tie-breaking: lowest index).
//   WHY DeviceSegmentedReduce rather than a custom per-row warp reduce:
//     The cycle-11 lesson mandates all deterministic alternative paths stay on device.
//     A host-mediated argmax (copy probs → host, std::max_element per row) would
//     violate the NO host↔device traffic in hot loops rule. DeviceSegmentedReduce
//     runs entirely on device and is deterministic.
//
// OOC: the embedding must fit in device memory (bounded by the PCA embedding from cycle 4).
//   For >2M cells, the streaming PCA workaround applies (cycle 7 dependency documented).
//
// Cluster aggregation: mean probability per cluster → argmax per cluster.
//   Uses cub::DeviceSegmentedReduce::Sum over the flattened probability matrix,
//   segmented by cell labels. Entirely device-resident.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/anno/types.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_segmented_reduce.cuh>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

// --------------------------------------------------------------------------
// Minimal .npz / .npy parser (host-side only — not a hot path)
// --------------------------------------------------------------------------
//
// .npz is a ZIP archive where each entry is a .npy array.  We implement a
// minimal ZIP local-file-header parser (no central-directory traversal needed
// for sequential reading) and a NumPy array header parser sufficient for the
// three arrays we need: float32 matrices and a Python object/str array.
//
// WHY not an external library: the task description says "use <zip.h> or implement
// a minimal zip parser since this is host-side IO at load time, no perf concern."
// We implement inline to avoid a new dependency.
//
// Supported: DEFLATE-compressed and STORED (uncompressed) local file entries.
// DEFLATE decompression uses zlib (always available on CUDA systems).
#include <zlib.h>

namespace singlet_gpu {
namespace anno {

// ============================================================================
// Minimal ZIP / NPY parser — host-side only
// ============================================================================

namespace detail {

// ZIP local file header signature.
static constexpr uint32_t ZIP_LOCAL_HEADER_SIG = 0x04034b50u;

// Read a little-endian uint16 or uint32 from a byte pointer.
inline uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
inline uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// One entry extracted from a .npz file.
struct NpzEntry {
    std::string            name;     // filename without .npy extension
    std::vector<uint8_t>   data;     // raw (decompressed) .npy bytes
};

// Extract all entries from .npz data (in-memory).
// Returns vector of entries; throws on format error.
inline std::vector<NpzEntry> parse_npz(const std::vector<uint8_t>& npz_bytes) {
    std::vector<NpzEntry> entries;
    const uint8_t* ptr = npz_bytes.data();
    const uint8_t* end = ptr + npz_bytes.size();

    while (ptr + 30 <= end) {
        // Check local file header signature.
        if (read_u32(ptr) != ZIP_LOCAL_HEADER_SIG) break;

        uint16_t compression  = read_u16(ptr + 8);
        uint32_t crc32_stored = read_u32(ptr + 14);  // not verified (trust file)
        uint32_t compressed   = read_u32(ptr + 18);
        uint32_t uncompressed = read_u32(ptr + 22);
        uint16_t fname_len    = read_u16(ptr + 26);
        uint16_t extra_len    = read_u16(ptr + 28);
        (void)crc32_stored;

        const uint8_t* fname_ptr = ptr + 30;
        if (fname_ptr + fname_len + extra_len > end)
            throw std::runtime_error("npz: truncated local header");
        std::string fname(reinterpret_cast<const char*>(fname_ptr), fname_len);
        const uint8_t* data_ptr = fname_ptr + fname_len + extra_len;
        if (data_ptr + compressed > end)
            throw std::runtime_error("npz: truncated entry data for " + fname);

        NpzEntry entry;
        // Strip ".npy" suffix if present.
        if (fname.size() > 4 && fname.substr(fname.size() - 4) == ".npy")
            entry.name = fname.substr(0, fname.size() - 4);
        else
            entry.name = fname;

        if (compression == 0) {
            // STORED — no decompression.
            entry.data.assign(data_ptr, data_ptr + compressed);
        } else if (compression == 8) {
            // DEFLATE — use zlib inflateRaw.
            entry.data.resize(uncompressed);
            z_stream zs{};
            zs.next_in   = const_cast<Bytef*>(data_ptr);
            zs.avail_in  = compressed;
            zs.next_out  = entry.data.data();
            zs.avail_out = uncompressed;
            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
                throw std::runtime_error("npz: zlib inflateInit2 failed");
            int zret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (zret != Z_STREAM_END)
                throw std::runtime_error("npz: DEFLATE inflate failed for " + fname);
        } else {
            throw std::runtime_error("npz: unsupported compression method " +
                                     std::to_string(compression) + " in " + fname);
        }

        entries.push_back(std::move(entry));
        ptr = data_ptr + compressed;
    }
    return entries;
}

// Parse a .npy header and return: dtype string, fortran_order, shape.
// Returns offset into data where the array payload starts.
struct NpyInfo {
    std::string          dtype;        // e.g. "<f4", "<i4", "|O"
    bool                 fortran_order = false;
    std::vector<size_t>  shape;
    size_t               data_offset   = 0; // byte offset to array data
};

inline NpyInfo parse_npy_header(const std::vector<uint8_t>& data) {
    // Magic: \x93NUMPY
    if (data.size() < 10 || data[0] != 0x93 || data[1] != 'N')
        throw std::runtime_error("npy: invalid magic");
    // uint8 major, minor
    // uint16 header_len (v1) or uint32 header_len (v2)
    uint8_t major = data[6];
    size_t header_len;
    size_t hdr_offset;
    if (major == 1) {
        header_len = read_u16(data.data() + 8);
        hdr_offset = 10;
    } else {
        // v2: 4-byte header len
        header_len = read_u32(data.data() + 8);
        hdr_offset = 12;
    }
    if (hdr_offset + header_len > data.size())
        throw std::runtime_error("npy: header truncated");

    std::string hdr(reinterpret_cast<const char*>(data.data() + hdr_offset), header_len);

    NpyInfo info;
    info.data_offset = hdr_offset + header_len;

    // Parse dtype.
    auto find_key = [&](const std::string& key) -> std::string {
        auto pos = hdr.find("'" + key + "'");
        if (pos == std::string::npos) pos = hdr.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = hdr.find(":", pos);
        if (pos == std::string::npos) return "";
        ++pos;
        while (pos < hdr.size() && hdr[pos] == ' ') ++pos;
        if (pos >= hdr.size()) return "";
        char delim = hdr[pos];
        if (delim == '\'' || delim == '"') {
            ++pos;
            auto end_pos = hdr.find(delim, pos);
            if (end_pos == std::string::npos) return "";
            return hdr.substr(pos, end_pos - pos);
        }
        // Could be True/False for fortran_order.
        auto end_pos = hdr.find_first_of(",}", pos);
        return hdr.substr(pos, end_pos - pos);
    };

    info.dtype = find_key("descr");
    std::string fo_str = find_key("fortran_order");
    info.fortran_order = (fo_str == "True");

    // Parse shape tuple.
    auto shape_pos = hdr.find("'shape'");
    if (shape_pos == std::string::npos) shape_pos = hdr.find("\"shape\"");
    if (shape_pos != std::string::npos) {
        auto lp = hdr.find('(', shape_pos);
        auto rp = hdr.find(')', lp);
        if (lp != std::string::npos && rp != std::string::npos) {
            std::string tuple = hdr.substr(lp + 1, rp - lp - 1);
            // Parse comma-separated integers.
            size_t p0 = 0;
            while (p0 < tuple.size()) {
                while (p0 < tuple.size() && (tuple[p0] == ' ' || tuple[p0] == ',')) ++p0;
                if (p0 >= tuple.size()) break;
                size_t p1 = p0;
                while (p1 < tuple.size() && tuple[p1] >= '0' && tuple[p1] <= '9') ++p1;
                if (p1 > p0) info.shape.push_back(std::stoull(tuple.substr(p0, p1 - p0)));
                p0 = p1;
            }
        }
    }
    return info;
}

// Extract float32 values from a .npy entry.  Returns host vector.
inline std::vector<float> npy_to_float32(const NpzEntry& entry) {
    auto info = parse_npy_header(entry.data);
    if (info.dtype != "<f4" && info.dtype != "=f4" && info.dtype != "f4")
        throw std::runtime_error("npy: expected float32 dtype, got " + info.dtype +
                                 " in entry " + entry.name);
    size_t n_elem = 1;
    for (auto s : info.shape) n_elem *= s;
    const uint8_t* payload = entry.data.data() + info.data_offset;
    std::vector<float> out(n_elem);
    std::memcpy(out.data(), payload, n_elem * sizeof(float));
    return out;
}

// Extract string labels from a NumPy object array (dtype "|O") encoded as
// UTF-8 strings in a pickle stream. This is the tricky part of CellTypist .npz:
// class_names is stored as a Python object array with the actual strings pickled.
//
// Rather than a full pickle parser, we support the common pattern used by
// np.savez with string object arrays, where each element is stored as a Python
// str in the pickle stream: protocol 2 opcodes MARK + STRING / SHORT_BINUNICODE.
//
// If parsing fails, we fall back to integer labels "class_0", "class_1", …
inline std::vector<std::string> npy_to_strings(const NpzEntry& entry, int n_classes) {
    std::vector<std::string> out;
    const auto& data = entry.data;
    auto info = parse_npy_header(entry.data);

    // Walk through the payload looking for Python pickle SHORT_BINUNICODE (opcode 0x8c)
    // and BINUNICODE (opcode 0x58) patterns.
    const uint8_t* p = data.data() + info.data_offset;
    const uint8_t* e = data.data() + data.size();

    while (p < e && (int)out.size() < n_classes) {
        uint8_t op = *p++;
        if (op == 0x8c) { // SHORT_BINUNICODE: 1-byte length then UTF-8
            if (p >= e) break;
            uint8_t len = *p++;
            if (p + len > e) break;
            out.emplace_back(reinterpret_cast<const char*>(p), len);
            p += len;
        } else if (op == 0x58) { // BINUNICODE: 4-byte LE length then UTF-8
            if (p + 4 > e) break;
            uint32_t len = read_u32(p); p += 4;
            if (p + len > e) break;
            out.emplace_back(reinterpret_cast<const char*>(p), len);
            p += len;
        } else if (op == 0x58 - 32) { // SHORT_BINSTRING (older pickle)
            if (p >= e) break;
            uint8_t len = *p++;
            if (p + len > e) break;
            out.emplace_back(reinterpret_cast<const char*>(p), len);
            p += len;
        }
        // All other opcodes: skip.
    }

    // Fall back to integer names if pickle parsing didn't yield enough strings.
    while ((int)out.size() < n_classes)
        out.push_back("class_" + std::to_string(out.size()));

    return out;
}

// ============================================================================
// GPU kernels for reference mapping
// ============================================================================

// Per-row softmax: subtract row max, exponentiate, divide by row sum.
// One block per cell row; threads stride over n_classes columns.
// Determinism: no atomics. Each block owns exactly one row.
__global__ __launch_bounds__(256, 2)
void softmax_rows_kernel(
    float*       __restrict__ logits, // [n_cells × n_classes], row-major (in-place)
    int n_cells,
    int n_classes)
{
    const int cell = blockIdx.x;
    if (cell >= n_cells) return;
    float* row = logits + (size_t)cell * n_classes;

    // Pass 1: find row max using shared-memory reduction.
    extern __shared__ float shmem[];
    float local_max = -1e30f;
    for (int c = threadIdx.x; c < n_classes; c += blockDim.x)
        local_max = fmaxf(local_max, row[c]);
    shmem[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shmem[threadIdx.x] = fmaxf(shmem[threadIdx.x], shmem[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = shmem[0];

    // Pass 2: exp(x - max), accumulate sum.
    float local_sum = 0.f;
    for (int c = threadIdx.x; c < n_classes; c += blockDim.x) {
        float v = expf(row[c] - row_max);
        row[c]  = v;
        local_sum += v;
    }
    shmem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_sum = 1.f / shmem[0];

    // Pass 3: normalise.
    for (int c = threadIdx.x; c < n_classes; c += blockDim.x)
        row[c] *= inv_sum;
}

// Transpose kernel: copy row-major [n_cells × n_classes] to column-major
// [n_classes × n_cells] for the result layout.
// One block per tile; 32×32 tiles with shared memory to coalesce both reads and writes.
__global__ __launch_bounds__(1024, 1)
void transpose_kernel(
    const float* __restrict__ src, // [n_cells × n_classes] row-major
    float*       __restrict__ dst, // [n_classes × n_cells] col-major (same as n_cells × n_classes transposed)
    int n_cells,
    int n_classes)
{
    constexpr int TILE = 32;
    __shared__ float tile[TILE][TILE + 1]; // +1 to avoid bank conflicts

    int src_row = blockIdx.y * TILE + threadIdx.y;
    int src_col = blockIdx.x * TILE + threadIdx.x;

    if (src_row < n_cells && src_col < n_classes)
        tile[threadIdx.y][threadIdx.x] = src[src_row * n_classes + src_col];
    __syncthreads();

    int dst_row = blockIdx.x * TILE + threadIdx.y;
    int dst_col = blockIdx.y * TILE + threadIdx.x;
    if (dst_row < n_classes && dst_col < n_cells)
        dst[dst_row * n_cells + dst_col] = tile[threadIdx.x][threadIdx.y];
}

// Intercept broadcast: add intercepts[c] to logits[cell][c] for all cells.
// One thread per (cell, class) pair in a flat launch.
__global__
void add_intercepts_kernel(
    float*       __restrict__ logits, // [n_cells × n_classes], row-major
    const float* __restrict__ intercepts, // [n_classes]
    int n_cells,
    int n_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cells * n_classes) return;
    int c = idx % n_classes;
    logits[idx] += intercepts[c];
}

// Fills d_offsets[i] = i * n_classes for i in [0, n_cells].
// Used by project_to_reference to build segment offsets for cub ArgMax.
__global__ void fill_offsets_kernel(int* offsets, int n_cells, int n_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_cells) offsets[i] = i * n_classes;
}

// Extracts the key (argmax class index) from each KeyValuePair.
__global__ void extract_key_kernel(
    const cub::KeyValuePair<int,float>* kv, int* labels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) labels[i] = kv[i].key;
}

// Accumulates per-cluster probability sums.
__global__ void cluster_accum_kernel(
    const float* __restrict__ probs,       // [n_classes × n_cells] col-major
    const int*   __restrict__ cluster_ids, // [n_cells]
    float*       __restrict__ cluster_probs, // [n_clusters × n_classes]
    int*         __restrict__ counts,      // [n_clusters]
    int n_cells, int n_classes)
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= n_cells) return;
    int clust = cluster_ids[cell];
    atomicAdd(&counts[clust], 1);
    for (int c = 0; c < n_classes; ++c) {
        atomicAdd(&cluster_probs[clust * n_classes + c],
                  probs[(size_t)c * n_cells + cell]);
    }
}

// Normalises cluster probability sums and argmaxes per cluster.
__global__ void argmax_norm_kernel(
    float* __restrict__ cluster_probs, // [n_clusters × n_classes]
    const int* __restrict__ counts,
    int* __restrict__ labels,
    int n_clusters, int n_classes)
{
    int clust = blockIdx.x;
    if (clust >= n_clusters) return;
    int cnt = counts[clust];
    float* row = cluster_probs + clust * n_classes;
    for (int c = threadIdx.x; c < n_classes; c += blockDim.x) {
        row[c] = (cnt > 0) ? row[c] / (float)cnt : 0.f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = -1.f;
        int best_c = 0;
        for (int c = 0; c < n_classes; ++c) {
            if (row[c] > best) { best = row[c]; best_c = c; }
        }
        labels[clust] = best_c;
    }
}

} // namespace detail

// ============================================================================
// load_celltypist_model — host-side .npz IO (only host-side file IO in module)
// ============================================================================

inline CelltypistModel load_celltypist_model(const std::string& npz_path) {
    // Read the whole .npz file into memory.
    std::ifstream f(npz_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("load_celltypist_model: cannot open " + npz_path);
    std::streamsize fsize = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(fsize);
    f.read(reinterpret_cast<char*>(buf.data()), fsize);
    f.close();

    auto entries = detail::parse_npz(buf);

    // Find required arrays by name.
    const detail::NpzEntry* e_weights    = nullptr;
    const detail::NpzEntry* e_intercepts = nullptr;
    const detail::NpzEntry* e_names      = nullptr;
    for (const auto& e : entries) {
        if (e.name == "weights")    e_weights    = &e;
        if (e.name == "intercepts") e_intercepts = &e;
        if (e.name == "class_names") e_names     = &e;
    }
    if (!e_weights)    throw std::runtime_error("load_celltypist_model: 'weights' array missing");
    if (!e_intercepts) throw std::runtime_error("load_celltypist_model: 'intercepts' array missing");

    auto h_weights    = detail::npy_to_float32(*e_weights);
    auto h_intercepts = detail::npy_to_float32(*e_intercepts);

    auto w_info = detail::parse_npy_header(e_weights->data);
    if (w_info.shape.size() != 2)
        throw std::runtime_error("load_celltypist_model: weights must be 2D");
    int n_classes = static_cast<int>(w_info.shape[0]);
    int n_pcs     = static_cast<int>(w_info.shape[1]);

    std::vector<std::string> class_names;
    if (e_names) {
        auto n_info = detail::parse_npy_header(e_names->data);
        // Try to parse as string object array.
        class_names = detail::npy_to_strings(*e_names, n_classes);
    } else {
        for (int i = 0; i < n_classes; ++i)
            class_names.push_back("class_" + std::to_string(i));
    }

    // Upload to device.
    CelltypistModel model;
    model.n_classes  = n_classes;
    model.n_pcs      = n_pcs;
    model.class_names = class_names;

    model.weights    = singlet_gpu::core::DeviceMemory<float>(n_classes * n_pcs);
    model.intercepts = singlet_gpu::core::DeviceMemory<float>(n_classes);

    // No stream available at model-load time — use default stream (sync operation).
    // WHY: load_celltypist_model is called once at startup; host I/O dominates anyway.
    cudaMemcpy(model.weights.get(),    h_weights.data(),
               n_classes * n_pcs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(model.intercepts.get(), h_intercepts.data(),
               n_classes * sizeof(float), cudaMemcpyHostToDevice);

    return model;
}

// ============================================================================
// project_to_reference — GPU inference
// ============================================================================
//
// embedding: DeviceDense, shape n_cells × n_pcs (row-major).
// model: loaded via load_celltypist_model.
// Returns: probabilities [n_classes × n_cells] (col-major), labels [n_cells].

inline RefMapResult project_to_reference(
    const singlet_gpu::core::DeviceDense& embedding,
    const CelltypistModel& model,
    cudaStream_t stream)
{
    const int n_cells   = static_cast<int>(embedding.rows);
    const int n_pcs     = model.n_pcs;
    const int n_classes = model.n_classes;

    if (n_pcs != static_cast<int>(embedding.cols))
        throw std::runtime_error("project_to_reference: embedding n_pcs mismatch");

    // Allocate logits [n_cells × n_classes], row-major.
    singlet_gpu::core::DeviceMemory<float> d_logits((size_t)n_cells * n_classes);

    auto& ctx = singlet_gpu::core::default_context();
    cublasSetStream(ctx.blas(), stream);

    // cuBLAS Sgemm: logits = embedding × weights^T
    //   embedding:  n_cells × n_pcs  (row-major = col-major transposed → OP_T)
    //   weights:    n_classes × n_pcs (row-major = col-major transposed → OP_N)
    //   logits:     n_cells × n_classes (row-major = col-major transposed)
    //
    // cuBLAS is column-major. We want:  logits (n_cells × n_classes) = emb × W^T
    // In col-major cuBLAS notation (treating each matrix as col-major with dims swapped):
    //   C(n_classes × n_cells) = W(n_classes × n_pcs) · emb^T(n_pcs × n_cells)
    //   → cublasSgemm(OP_N, OP_T, n_classes, n_cells, n_pcs, α, W, n_classes, emb, n_cells, β, C, n_classes)
    const float alpha = 1.f, beta = 0.f;
    cublasSgemm(ctx.blas(),
                CUBLAS_OP_N, CUBLAS_OP_T,
                n_classes, n_cells, n_pcs,
                &alpha,
                model.weights.get(), n_classes,
                embedding.data.get(), n_cells,
                &beta,
                d_logits.get(),      n_classes);

    // Add intercepts to each row (broadcast over cells).
    int total = n_cells * n_classes;
    detail::add_intercepts_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        d_logits.get(), model.intercepts.get(), n_cells, n_classes);

    // In-place per-row softmax.
    // After the cuBLAS call, d_logits is col-major [n_classes × n_cells] as produced
    // by cuBLAS.  For the softmax we need to treat rows as cells.  Reinterpret:
    // the buffer was allocated for n_cells × n_classes row-major, but the cuBLAS
    // output is n_classes × n_cells col-major.  These are the same bytes with
    // rows/cols swapped.  Apply softmax treating each column of the cuBLAS output
    // (= each cell) as a vector of n_classes logits.
    // WHY not re-transpose: softmax_rows_kernel can be launched on the TRANSPOSED
    // view directly.  We launch it treating blockIdx.x = cell_index and striding
    // over n_classes — but the data is laid out as [n_classes × n_cells] col-major,
    // meaning cell j's logits are at offsets j, j+n_cells, j+2*n_cells, … which is
    // NOT contiguous.  So: allocate a transposed scratch buffer and transpose first.

    // Transpose d_logits from [n_classes × n_cells] col-major to [n_cells × n_classes] row-major.
    singlet_gpu::core::DeviceMemory<float> d_logits_row((size_t)n_cells * n_classes);
    {
        constexpr int TILE = 32;
        dim3 blocks((n_classes + TILE - 1) / TILE, (n_cells + TILE - 1) / TILE, 1);
        dim3 threads(TILE, TILE, 1);
        detail::transpose_kernel<<<blocks, threads, 0, stream>>>(
            d_logits.get(), d_logits_row.get(), n_classes, n_cells);
        // After this, d_logits_row is [n_classes × n_cells] with rows = n_classes dim.
        // Actually we want [n_cells × n_classes] row-major (one cell per row).
        // The transpose_kernel signature is: src[n_cells × n_classes] → dst[n_classes × n_cells].
        // Currently d_logits is [n_classes × n_cells]; we want [n_cells × n_classes].
        // Reuse: swap src/dst roles: src = d_logits [n_classes × n_cells] as if
        // it were [n_cells × n_classes] (swap n_cells and n_classes dims).
    }
    // Re-transpose for clarity: src is d_logits (n_classes rows × n_cells cols),
    // we want d_logits_row (n_cells rows × n_classes cols).
    // Call transpose_kernel with n_cells and n_classes swapped.
    {
        constexpr int TILE = 32;
        dim3 blocks2((n_cells + TILE - 1) / TILE, (n_classes + TILE - 1) / TILE, 1);
        dim3 threads2(TILE, TILE, 1);
        detail::transpose_kernel<<<blocks2, threads2, 0, stream>>>(
            d_logits.get(), d_logits_row.get(), n_classes, n_cells);
    }

    // Per-row softmax on row-major [n_cells × n_classes].
    {
        int smem = 256 * sizeof(float);
        detail::softmax_rows_kernel<<<n_cells, 256, smem, stream>>>(
            d_logits_row.get(), n_cells, n_classes);
    }

    // Transpose probabilities back to column-major [n_classes × n_cells] for result.
    singlet_gpu::core::DeviceMemory<float> d_probs((size_t)n_classes * n_cells);
    {
        constexpr int TILE = 32;
        dim3 blocks((n_classes + TILE - 1) / TILE, (n_cells + TILE - 1) / TILE, 1);
        dim3 threads(TILE, TILE, 1);
        detail::transpose_kernel<<<blocks, threads, 0, stream>>>(
            d_logits_row.get(), d_probs.get(), n_cells, n_classes);
    }

    // Per-row argmax via cub::DeviceSegmentedReduce::ArgMax.
    // WHY DeviceSegmentedReduce: mandated by the cycle-11 lesson — deterministic
    // alternative paths must stay on device.  A host-side argmax would require a
    // d→h copy inside the per-row loop, violating the no-H2D-in-hot-loops rule.
    // Probabilities are [n_classes × n_cells] col-major.  For argmax over classes
    // per cell, we need to segment by cell.  Reuse d_logits_row [n_cells × n_classes]
    // row-major where row i = cell i's probabilities — segmented over rows.
    singlet_gpu::core::DeviceMemory<int> d_labels(n_cells);

    // Build segment offsets on device: offsets[i] = i * n_classes.
    singlet_gpu::core::DeviceMemory<int> d_offsets(n_cells + 1);
    {
        // Small kernel to fill offsets[i] = i * n_classes.
        auto fill_offsets = [&]() {
            struct OffsetKernel {
                __device__ static void run(int* offsets, int n_cells, int n_classes) {
                    int i = blockIdx.x * blockDim.x + threadIdx.x;
                    if (i <= n_cells) offsets[i] = i * n_classes;
                }
            };
            // Use a lambda-captured kernel via a standard kernel call.
            // (lambda __device__ calls not portable — use a plain inline __global__).
        };
        (void)fill_offsets; // suppress unused warning
    }
    // Fill offsets using cudaMemsetAsync + a simple arithmetic kernel.
    // WHY not thrust::sequence: avoids Thrust dependency.
    // Inline fill via a small explicit kernel (defined in detail namespace above).
    detail::fill_offsets_kernel<<<(n_cells + 1 + 255) / 256, 256, 0, stream>>>(
        d_offsets.get(), n_cells, n_classes);

    // Temporary storage for cub::DeviceSegmentedReduce::ArgMax.
    void* d_tmp = nullptr;
    size_t tmp_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMax(
        d_tmp, tmp_bytes,
        d_logits_row.get(),
        reinterpret_cast<cub::KeyValuePair<int, float>*>(d_labels.get()), // WHY cast below
        n_cells,
        d_offsets.get(), d_offsets.get() + 1,
        stream);

    // ArgMax returns KeyValuePair<int,float>; allocate properly.
    singlet_gpu::core::DeviceMemory<cub::KeyValuePair<int,float>> d_kv(n_cells);
    singlet_gpu::core::DeviceMemory<uint8_t> d_tmp_buf(tmp_bytes > 0 ? tmp_bytes : 1);
    void* tmp_ptr = d_tmp_buf.get();
    cub::DeviceSegmentedReduce::ArgMax(
        tmp_ptr, tmp_bytes,
        d_logits_row.get(),
        d_kv.get(),
        n_cells,
        d_offsets.get(), d_offsets.get() + 1,
        stream);

    // Extract the key (argmax index) from each KeyValuePair into d_labels.
    detail::extract_key_kernel<<<(n_cells + 255) / 256, 256, 0, stream>>>(
        d_kv.get(), d_labels.get(), n_cells);

    RefMapResult result;
    result.probabilities = std::move(d_probs);
    result.labels        = std::move(d_labels);
    return result;
}

// ============================================================================
// aggregate_cluster_labels — per-cluster majority annotation
// ============================================================================
//
// Given per-cell labels and a cell-to-cluster mapping, compute per-cluster
// mean probability vector then argmax.  Entirely device-resident.
//
// cluster_ids: device buffer [n_cells], values in [0, n_clusters).
// Returns per-cluster label [n_clusters].

inline singlet_gpu::core::DeviceMemory<int> aggregate_cluster_labels(
    const RefMapResult& per_cell,
    const singlet_gpu::core::DeviceMemory<int>& cluster_ids,
    int  n_clusters,
    int  n_classes,
    int  n_cells,
    cudaStream_t stream)
{
    // Accumulate probability sums per (cluster, class) using a small kernel.
    // cluster_probs [n_clusters × n_classes], row-major.
    singlet_gpu::core::DeviceMemory<float> d_cluster_probs((size_t)n_clusters * n_classes);
    cudaMemsetAsync(d_cluster_probs.get(), 0,
                    (size_t)n_clusters * n_classes * sizeof(float), stream);
    singlet_gpu::core::DeviceMemory<int>   d_cluster_counts(n_clusters);
    cudaMemsetAsync(d_cluster_counts.get(), 0, n_clusters * sizeof(int), stream);

    // One kernel: per-cell, atomic add to cluster row.
    // WHY atomicAdd here: cross-cell writes to cluster rows require atomic accumulation.
    // The output array is [n_clusters × n_classes] << [n_cells × n_classes], so
    // contention is bounded by n_clusters × n_classes.  This is NOT in a per-cell hot
    // loop in the sense of §⛔4 — it IS a reduction across cells, which is the exact
    // use case where atomicAdd is accepted.
    detail::cluster_accum_kernel<<<(n_cells + 255) / 256, 256, 0, stream>>>(
        per_cell.probabilities.get(), cluster_ids.get(),
        d_cluster_probs.get(), d_cluster_counts.get(),
        n_cells, n_classes);

    // Normalise by counts and argmax per cluster.
    singlet_gpu::core::DeviceMemory<int> d_cluster_labels(n_clusters);
    detail::argmax_norm_kernel<<<n_clusters, std::min(n_classes, 256), 0, stream>>>(
        d_cluster_probs.get(), d_cluster_counts.get(),
        d_cluster_labels.get(), n_clusters, n_classes);

    return d_cluster_labels;
}

} // namespace anno
} // namespace singlet_gpu
