// SPDX-License-Identifier: MIT
// singlet/gpu/io/pz_device_loader.h
//
// Zero-copy .1pz → device CSC loader.
//
// Algorithm reference: design doc at singlet/gpu/state/designs/00-pz-device-loader.md
// Format reference: singlet/include/singlet-pileup/pz_writer.h (TP1Z v1, VOCSC encoding)
//
// Pipeline (one call to load_pz):
//   1. slurp file + validate magic / CRC32 (body-of-file CRC, not per-chunk)
//   2. decode header → PZHeader (96-byte packed struct)
//   3. parse permutation + column-count ptr block
//   4. read chunk table (nc × uint32 blob sizes)
//   5. iterate over VOCSC chunks: zstd → bitmap_unpack → bit_planes_decode →
//      gap_stream walk → scatter to pinned host indptr/indices/data buffers
//   6. cast uint8/uint16/uint32 → float (saturating at 2^24 with stderr warning)
//   7. decode metadata TLV → core::Metadata
//   8. cudaMemcpyAsync (indptr, indices, values) to device on the caller stream
//   9. return PzDeviceMatrix { DeviceCSC, Metadata, stream }
//
// Memory:
//   - Pinned host staging: 3 × nnz × sizeof(float) + (n+1) × 4 bytes
//   - Device: same layout inside singlet::gpu::core::DeviceCSC
//   - Both allocations released after the async copies are in flight; the caller
//     must synchronize the stream before accessing the PinnedBuffers if they
//     need to survive load_pz's scope.  We retain them inside PzDeviceMatrix so
//     lifetime is tied to the returned struct.
//
// OOC: PzChunkIterator yields fixed-column-width slices without loading the
//      full matrix into pinned host memory at once.
//
// Workspace budget (design doc §Memory):
//   10k-cell sample  (~30M nnz): ~400 MB pinned + ~400 MB device
//   100k-cell sample (~300M nnz): ~4 GB pinned + ~4 GB device
//   1M-cell (streaming, 100k-col chunks): ≤400 MB pinned sliding window
//
// Determinism: fully deterministic — no atomics, no reductions.

#pragma once

// CYCLE-105: removed factornet/gpu/types.cuh dependency — DeviceCSC is now native.
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include <zstd.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// .1pz format constants — mirrored bit-exact from pz_writer.h.
// Do NOT derive these independently. If pz_writer.h changes, update here.
// ---------------------------------------------------------------------------
namespace singlet::gpu {
namespace io {
namespace pz_fmt {

// Header magic "TP1Z" in little-endian
static constexpr uint32_t TP1_MAGIC   = 0x5A315054;
static constexpr uint16_t TP1_VERSION = 1;

// PZHeader.flags
static constexpr uint8_t FLAG_HAS_PERM      = 0x01;
static constexpr uint8_t FLAG_GAP16         = 0x02;
static constexpr uint8_t FLAG_HAS_METADATA  = 0x04;
static constexpr uint8_t FLAG_HAS_COLSUMS   = 0x10;

// PZHeader.feature_flags
static constexpr uint32_t FEAT_ZSTD_CHECKSUMS  = 0x01;
static constexpr uint32_t FEAT_BITPLANE_BITMAP = 0x02;

// Metadata TLV tags
static constexpr uint8_t META_TAG_END      = 0;
static constexpr uint8_t META_TAG_ROWNAMES = 1;
static constexpr uint8_t META_TAG_COLNAMES = 2;
static constexpr uint8_t META_TAG_USER_KV  = 3;

// fp32 mantissa is 23 bits; integers beyond 2^24 lose precision in float cast.
static constexpr uint32_t FP32_EXACT_INT_LIMIT = (1u << 24);

// -----------------------------------------------------------------------
// Packed file-format structs. #pragma pack ensures byte-exact layout.
// static_asserts are diagnostic only — they fire at compile time if the
// layout ever drifts from the 96- and 16-byte spec.
// -----------------------------------------------------------------------
#pragma pack(push, 1)

struct PZHeader {
    uint32_t magic;             // 0
    uint16_t version;           // 4
    uint8_t  vt_code;           // 6   1=uint8 2=uint16 3=uint32
    uint8_t  flags;             // 7
    uint32_t m;                 // 8   rows (features)
    uint32_t n;                 // 12  cols (cells)
    uint64_t nnz;               // 16
    uint8_t  ptr_width;         // 24  bytes per column-count entry (2 or 4)
    uint8_t  codec_level;       // 25  zstd level (informational)
    uint16_t _pad0;             // 26
    uint32_t num_chunks;        // 28
    uint32_t perm_z_sz;         // 32  compressed perm block size in bytes
    uint32_t ptr_z_sz;          // 36  compressed ptr block size in bytes
    uint32_t chunk_cols;        // 40  columns per VOCSC chunk
    uint32_t feature_flags;     // 44
    uint64_t metadata_offset;   // 48  absolute byte offset of compressed metadata block
    uint32_t metadata_z_sz;     // 56  compressed metadata size in bytes
    uint32_t colsums_z_sz;      // 60
    uint64_t transpose_offset;  // 64
    uint32_t transpose_z_sz;    // 72
    uint32_t transpose_chunks;  // 76
    uint8_t  reserved[16];      // 80-95
};
// Compile-time guard: if layout drifts the loader will silently misread files.
static_assert(sizeof(PZHeader) == 96, "PZHeader layout must be 96 bytes — mirrors pz_writer.h");

struct PZFooter {
    uint32_t file_crc32;
    uint32_t _reserved;
    uint32_t num_chunks;
    uint32_t magic;
};
static_assert(sizeof(PZFooter) == 16, "PZFooter layout must be 16 bytes — mirrors pz_writer.h");

#pragma pack(pop)

}  // namespace pz_fmt
}  // namespace io
}  // namespace singlet::gpu

// ---------------------------------------------------------------------------
// Internal helpers — in an anonymous namespace so they do not pollute the
// singlet::gpu ABI. All are header-only.
// ---------------------------------------------------------------------------
namespace {

// CRC32 (ISO 3309, slicing-by-8) — must match pz_writer.h exactly.
// Only the finalize/compute path is used by the loader.
struct Crc32Loader {
    uint32_t state_;
    Crc32Loader() : state_(0xFFFFFFFFu) {}

    static const uint32_t (&table8())[8][256] {
        static uint32_t T[8][256]{};
        static bool init = false;
        if (!init) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j)
                    c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                T[0][i] = c;
            }
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = T[0][i];
                for (int k = 1; k < 8; ++k) {
                    c = (c >> 8) ^ T[0][c & 0xFF];
                    T[k][i] = c;
                }
            }
            init = true;
        }
        return T;
    }

    void update(const uint8_t* data, size_t len) {
        static const auto& T = table8();
        uint32_t crc = state_;
        while (len >= 8) {
            uint32_t a, b;
            std::memcpy(&a, data,     4);
            std::memcpy(&b, data + 4, 4);
            a ^= crc;
            crc = T[7][a & 0xFF] ^ T[6][(a >> 8) & 0xFF] ^
                  T[5][(a >> 16) & 0xFF] ^ T[4][(a >> 24) & 0xFF] ^
                  T[3][b & 0xFF] ^ T[2][(b >> 8) & 0xFF] ^
                  T[1][(b >> 16) & 0xFF] ^ T[0][(b >> 24) & 0xFF];
            data += 8; len -= 8;
        }
        while (len--) crc = (crc >> 8) ^ T[0][(crc ^ *data++) & 0xFF];
        state_ = crc;
    }
    uint32_t finalize() const { return state_ ^ 0xFFFFFFFFu; }
    static uint32_t compute(const void* data, size_t len) {
        Crc32Loader c;
        c.update(static_cast<const uint8_t*>(data), len);
        return c.finalize();
    }
};

// zstd RAII decompression context.
struct ZstdDCtxGuard {
    ZSTD_DCtx* ctx;
    ZstdDCtxGuard() : ctx(ZSTD_createDCtx()) {
        if (!ctx) throw std::runtime_error("ZSTD_createDCtx failed");
    }
    ~ZstdDCtxGuard() { if (ctx) ZSTD_freeDCtx(ctx); }
    ZstdDCtxGuard(const ZstdDCtxGuard&) = delete;
    ZstdDCtxGuard& operator=(const ZstdDCtxGuard&) = delete;
};

inline std::vector<uint8_t>
zstd_decomp(ZSTD_DCtx* ctx, const uint8_t* src, size_t src_sz) {
    unsigned long long dsz = ZSTD_getFrameContentSize(src, src_sz);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN)
        throw std::runtime_error("zstd: cannot determine decompressed size");
    std::vector<uint8_t> out(static_cast<size_t>(dsz));
    size_t got = ZSTD_decompressDCtx(ctx, out.data(), out.size(), src, src_sz);
    if (ZSTD_isError(got))
        throw std::runtime_error(std::string("zstd decompress: ") + ZSTD_getErrorName(got));
    out.resize(got);
    return out;
}

// LEB128 varint read — inverse of varint_push() in pz_writer.h.
inline uint32_t varint_read(const uint8_t* buf, size_t& off) {
    uint32_t v = 0;
    int shift = 0;
    while (true) {
        uint8_t b = buf[off++];
        v |= static_cast<uint32_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return v;
        shift += 7;
        if (shift > 28) throw std::runtime_error("varint overflow");
    }
}

// bitmap_unpack — inverse of pz_writer.h bitmap_pack.
// packed_sz bytes at src → pf_sz bytes at dst (zeros restored).
inline void bitmap_unpack(
    const uint8_t* src, size_t packed_sz,
    uint8_t* dst, size_t pf_sz)
{
    size_t bm_bytes = (pf_sz + 7) / 8;
    if (packed_sz < bm_bytes)
        throw std::runtime_error("bitmap_unpack: packed < bitmap");
    std::memset(dst, 0, pf_sz);
    size_t wp = bm_bytes;
    for (size_t i = 0; i < pf_sz; ++i) {
        if (src[i / 8] & (1u << (i & 7))) {
            if (wp >= packed_sz)
                throw std::runtime_error("bitmap_unpack: ran off end");
            dst[i] = src[wp++];
        }
    }
}

// bit_planes_decode — inverse of pz_writer.h bit_planes_encode.
// src: 8*pb bytes, pb = (n+7)/8; dst: n bytes.
inline void bit_planes_decode(const uint8_t* src, size_t n, uint8_t* dst) {
    if (n == 0) return;
    size_t pb = (n + 7) / 8;
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = 0;
        size_t bi  = i / 8;
        uint8_t bm = static_cast<uint8_t>(1u << (i & 7));
        for (int b = 0; b < 8; ++b)
            if (src[b * pb + bi] & bm) v |= static_cast<uint8_t>(1 << b);
        dst[i] = v;
    }
}

// parse_metadata_tlv — populates a core::Metadata from the TLV byte stream.
// GEO KV fields are extracted from the user_kv section.
inline void parse_metadata_tlv(
    const uint8_t* buf, size_t len,
    singlet::gpu::core::Metadata& meta)
{
    using namespace singlet::gpu::io::pz_fmt;
    std::vector<std::string> rownames, colnames;
    std::map<std::string, std::string> kv;

    size_t off = 0;
    while (off < len) {
        uint8_t tag = buf[off++];
        if (tag == META_TAG_END) break;
        if (off + 4 > len) throw std::runtime_error("metadata TLV: truncated length");
        uint32_t length;
        std::memcpy(&length, buf + off, 4);
        off += 4;
        if (off + length > len) throw std::runtime_error("metadata TLV: payload out of range");

        auto read_strvec = [&](std::vector<std::string>& dst) {
            size_t p = off, end = off + length;
            while (p < end) {
                size_t s = p;
                while (p < end && buf[p] != 0) ++p;
                dst.emplace_back(reinterpret_cast<const char*>(buf + s), p - s);
                if (p < end) ++p;  // skip NUL
            }
        };

        if (tag == META_TAG_ROWNAMES)      read_strvec(rownames);
        else if (tag == META_TAG_COLNAMES) read_strvec(colnames);
        else if (tag == META_TAG_USER_KV) {
            size_t p = off, end = off + length;
            while (p < end) {
                size_t ks = p; while (p < end && buf[p]) ++p;
                std::string key(reinterpret_cast<const char*>(buf + ks), p - ks);
                if (p < end) ++p;
                size_t vs = p; while (p < end && buf[p]) ++p;
                std::string val(reinterpret_cast<const char*>(buf + vs), p - vs);
                if (p < end) ++p;
                kv[std::move(key)] = std::move(val);
            }
        }
        off += length;
    }

    meta.rownames = std::move(rownames);
    meta.colnames = std::move(colnames);

    // Map well-known GEO KV keys → Metadata fields.
    auto g = [&](const std::string& k) -> const std::string& {
        static const std::string empty;
        auto it = kv.find(k);
        return (it != kv.end()) ? it->second : empty;
    };
    meta.gsm_id           = g("gsm_id");
    meta.gse_id           = g("gse_id");
    meta.organism         = g("organism");
    meta.protocol         = g("protocol");
    meta.modality         = g("modality");
    meta.geo_title        = g("geo_title");
    meta.geo_source_name  = g("geo_source_name");
    meta.singlet_version = g("singlet_version");
    meta.pipeline_date    = g("pipeline_date");

    // taxon_id and read_count are stored as strings in user_kv.
    if (!g("taxon_id").empty()) {
        try { meta.taxon_id = std::stoi(g("taxon_id")); } catch (...) {}
    }
    if (!g("read_count").empty()) {
        try { meta.read_count = std::stoll(g("read_count")); } catch (...) {}
    }
    // srr_ids: comma-separated list stored under key "srr_ids".
    if (!g("srr_ids").empty()) {
        std::istringstream ss(g("srr_ids"));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) meta.srr_ids.push_back(std::move(tok));
        }
    }
}

// Saturating uint32 → float cast with a one-time stderr warning per call site.
// Values beyond FP32_EXACT_INT_LIMIT (2^24) lose mantissa bits.
inline float saturate_cast_f32(uint32_t v, bool& warned) {
    using namespace singlet::gpu::io::pz_fmt;
    if (v > FP32_EXACT_INT_LIMIT && !warned) {
        std::fprintf(stderr,
            "[singlet/gpu/pz_device_loader] WARNING: value %u > 2^24 (%u); "
            "fp32 cast loses integer precision.\n", v, FP32_EXACT_INT_LIMIT);
        warned = true;
    }
    return static_cast<float>(v);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
namespace singlet::gpu {
namespace io {

// PzDeviceMatrix — result of load_pz().
// Owns the device buffers (via SparseMatrixGPU DeviceMemory) and the pinned
// host staging buffers (via PinnedBuffer). Metadata lives on host only.
//
// The caller must synchronize producer_stream before accessing mat on the
// device from a different stream.
//
// SVD path (cycle 5): when load_pz is called with keep_host_pinned=true, the
// pinned host CSC arrays are additionally exposed via shared_ptr fields below.
// This avoids re-staging for SVD adapters that take HOST CSC pointers
// (integration-notes.md finding 0a).
// Memory cost when host_retained=true: 2× the matrix (device + host pinned).
struct PzDeviceMatrix {
    core::DeviceCSC   mat;
    core::Metadata    meta;
    cudaStream_t      producer_stream = nullptr;

    // Pinned staging kept alive until the async copy completes.
    // Caller should sync the stream and then let this struct go out of scope,
    // or explicitly reset these buffers after sync.
    core::PinnedBuffer pinned_indptr;
    core::PinnedBuffer pinned_indices;
    core::PinnedBuffer pinned_values;

    // Optional retained host pointers for SVD adapters (keep_host_pinned=true).
    // Deleters call cudaFreeHost because the backing storage is pinned memory.
    // null when host_retained=false (default / non-SVD path).
    std::shared_ptr<int>   host_indptr;   // size: mat.cols + 1 (int32)
    std::shared_ptr<int>   host_indices;  // size: mat.nnz    (int32)
    std::shared_ptr<float> host_values;   // size: mat.nnz    (float)
    bool                   host_retained = false;

    // ── Compat alias fields (test-harness API) ────────────────────────────────
    int n_genes = 0;   // mat.rows (populated after load)
    int n_cells = 0;   // mat.cols (populated after load)
    void free(cudaStream_t /*s*/ = nullptr) {}  // RAII handles memory; no-op for compat
};

// ---------------------------------------------------------------------------
// load_pz — one-shot full-matrix load.
//
// Reads the file on CPU (decompression), stages into pinned host memory, then
// fires three cudaMemcpyAsync calls to put the CSC on device.
//
// stream: if nullptr, a new high-priority stream is created and stored in
//         result.producer_stream. Caller is responsible for cudaStreamDestroy
//         if they passed nullptr (stream ownership is NOT transferred here —
//         the caller must check result.producer_stream and destroy it if it
//         was allocated by load_pz).
//
// WHY three separate async copies: allows the runtime to pipeline them over
// PCIe while the CPU continues other work. A single large copy would also
// work but prevents any overlap.
// ---------------------------------------------------------------------------
// keep_host_pinned: when true, the pinned host CSC buffers are retained in
// result.host_indptr / host_indices / host_values (shared_ptr with cudaFreeHost
// deleter) so that host-pointer SVD adapters can consume them directly without
// re-staging.  Default false — existing callers (lognorm, hvg) are unaffected.
inline PzDeviceMatrix load_pz(const std::string& path,
                               cudaStream_t stream = nullptr,
                               bool keep_host_pinned = false) {
    using namespace pz_fmt;

    // --- 1. Slurp file ---
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("load_pz: cannot open " + path);
    const std::streamsize file_sz = fin.tellg();
    if (file_sz < static_cast<std::streamsize>(sizeof(PZHeader) + sizeof(PZFooter)))
        throw std::runtime_error("load_pz: file too short: " + path);
    fin.seekg(0);
    std::vector<uint8_t> file(static_cast<size_t>(file_sz));
    fin.read(reinterpret_cast<char*>(file.data()), file_sz);
    if (!fin) throw std::runtime_error("load_pz: read error: " + path);
    fin.close();

    // --- 2. Parse header ---
    PZHeader hdr;
    std::memcpy(&hdr, file.data(), sizeof(PZHeader));
    if (hdr.magic   != TP1_MAGIC)   throw std::runtime_error("load_pz: bad magic: "   + path);
    if (hdr.version != TP1_VERSION) throw std::runtime_error("load_pz: bad version: " + path);

    // --- 3. Validate footer + CRC ---
    PZFooter ftr;
    std::memcpy(&ftr, file.data() + file.size() - sizeof(PZFooter), sizeof(PZFooter));
    if (ftr.magic      != TP1_MAGIC)       throw std::runtime_error("load_pz: bad footer magic: " + path);
    if (ftr.num_chunks != hdr.num_chunks)  throw std::runtime_error("load_pz: chunk count mismatch: " + path);
    {
        uint32_t actual = Crc32Loader::compute(file.data(), file.size() - sizeof(PZFooter));
        if (actual != ftr.file_crc32)
            throw std::runtime_error("load_pz: CRC32 mismatch: " + path);
    }

    // --- 4. Decode permutation + column-count ptr block ---
    ZstdDCtxGuard dctx;
    size_t body_off = sizeof(PZHeader);

    std::vector<uint32_t> perm(hdr.m);
    if (hdr.flags & FLAG_HAS_PERM) {
        if (hdr.perm_z_sz == 0) throw std::runtime_error("FLAG_HAS_PERM but perm_z_sz==0");
        auto raw = zstd_decomp(dctx.ctx, file.data() + body_off, hdr.perm_z_sz);
        if (raw.size() != static_cast<size_t>(hdr.m) * 4)
            throw std::runtime_error("perm block wrong decompressed size");
        std::memcpy(perm.data(), raw.data(), raw.size());
        body_off += hdr.perm_z_sz;
    } else {
        for (uint32_t i = 0; i < hdr.m; ++i) perm[i] = i;  // identity
    }

    std::vector<uint32_t> cc(hdr.n);
    if (hdr.ptr_z_sz > 0) {
        auto raw = zstd_decomp(dctx.ctx, file.data() + body_off, hdr.ptr_z_sz);
        if (raw.size() != static_cast<size_t>(hdr.n) * hdr.ptr_width)
            throw std::runtime_error("ptr block wrong decompressed size");
        for (uint32_t j = 0; j < hdr.n; ++j) {
            uint32_t v = 0;
            for (int b = 0; b < hdr.ptr_width; ++b)
                v |= static_cast<uint32_t>(raw[j * hdr.ptr_width + b]) << (8 * b);
            cc[j] = v;
        }
        body_off += hdr.ptr_z_sz;
    }

    // --- 5. Build indptr (prefix-sum of cc) ---
    // Pinned allocation so cudaMemcpyAsync can use it without a sync.
    const size_t indptr_bytes = static_cast<size_t>(hdr.n + 1) * sizeof(int32_t);
    auto pinned_indptr = core::PinnedPool::acquire(indptr_bytes);
    {
        int32_t* ip = pinned_indptr.as<int32_t>();
        ip[0] = 0;
        for (uint32_t j = 0; j < hdr.n; ++j) {
            ip[j + 1] = ip[j] + static_cast<int32_t>(cc[j]);
        }
        // Verify total nnz matches header.
        if (static_cast<uint64_t>(ip[hdr.n]) != hdr.nnz)
            throw std::runtime_error("load_pz: indptr sum != header nnz");
    }

    // --- 6. Allocate pinned indices + values buffers ---
    const size_t indices_bytes = static_cast<size_t>(hdr.nnz) * sizeof(int32_t);
    const size_t values_bytes  = static_cast<size_t>(hdr.nnz) * sizeof(float);
    auto pinned_indices = core::PinnedPool::acquire(indices_bytes);
    auto pinned_values  = core::PinnedPool::acquire(values_bytes);

    int32_t* h_indices = pinned_indices.as<int32_t>();
    float*   h_values  = pinned_values.as<float>();

    // --- 7. Decode VOCSC chunks → scatter into pinned host buffers ---
    // Chunk table immediately follows the ptr block.
    const size_t ctable_off = body_off;
    const uint32_t nc = hdr.num_chunks;
    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), file.data() + ctable_off, nc * sizeof(uint32_t));

    size_t blob_cursor = ctable_off + nc * sizeof(uint32_t);
    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;

    bool fp32_warn = false;

    // Scratch buffers reused across chunks to avoid per-chunk heap thrash.
    std::vector<uint8_t>  packed;
    std::vector<uint8_t>  prefilter;
    std::vector<uint8_t>  low_bytes;
    std::vector<uint32_t> gaps;
    std::vector<std::pair<uint32_t, uint32_t>> rv;  // (row, value) scratch per column

    const int32_t* h_indptr = pinned_indptr.as<int32_t>();

    for (uint32_t c = 0; c < nc; ++c) {
        const uint32_t blob_sz = ctable[c];
        if (blob_sz < 20) throw std::runtime_error("chunk blob < 20 bytes");
        const uint8_t* blob = file.data() + blob_cursor;

        uint32_t ng32, msz32, zsz, chunk_crc, psz32;
        std::memcpy(&ng32,      blob +  0, 4);
        std::memcpy(&msz32,     blob +  4, 4);
        std::memcpy(&zsz,       blob +  8, 4);
        std::memcpy(&chunk_crc, blob + 12, 4);
        std::memcpy(&psz32,     blob + 16, 4);

        if (20u + zsz > blob_sz) throw std::runtime_error("chunk blob header/size mismatch");

        // Decompress → packed
        packed = zstd_decomp(dctx.ctx, blob + 20, zsz);
        if (packed.size() != psz32)
            throw std::runtime_error("chunk decompressed size mismatch");

        // Per-chunk CRC over the packed bytes.
        if (Crc32Loader::compute(packed.data(), packed.size()) != chunk_crc)
            throw std::runtime_error("chunk CRC32 mismatch in " + path);

        // Unpack bitmap → prefilter
        const size_t bp_sz        = ng32 > 0 ? 8 * ((static_cast<size_t>(ng32) + 7) / 8) : 0;
        const size_t extra_planes = ng32 > 0 ? static_cast<size_t>(gw - 1) * ng32 : 0;
        const size_t pf_sz        = static_cast<size_t>(msz32) + bp_sz + extra_planes;
        prefilter.assign(pf_sz, 0);
        bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);

        // Decode gap stream: bit-plane-0 decode → low bytes → combine higher planes.
        gaps.assign(ng32, 0);
        if (ng32 > 0) {
            low_bytes.assign(ng32, 0);
            bit_planes_decode(prefilter.data() + msz32, ng32, low_bytes.data());
            for (uint32_t i = 0; i < ng32; ++i) gaps[i] = low_bytes[i];
            if (gw > 1) {
                const uint8_t* upper = prefilter.data() + msz32 + bp_sz;
                for (int b = 1; b < gw; ++b) {
                    const uint8_t* plane = upper + static_cast<size_t>(b - 1) * ng32;
                    for (uint32_t i = 0; i < ng32; ++i)
                        gaps[i] |= static_cast<uint32_t>(plane[i]) << (8 * b);
                }
            }
        }

        // Walk the column TLV stream for this chunk's column range.
        const size_t col_start = static_cast<size_t>(c) * hdr.chunk_cols;
        const size_t col_end   = std::min(col_start + hdr.chunk_cols, static_cast<size_t>(hdr.n));

        size_t mp = 0;  // offset within metadata (varint) stream
        size_t gp = 0;  // offset within gaps[]
        const uint8_t* meta_stream = prefilter.data();

        for (size_t j = col_start; j < col_end; ++j) {
            uint32_t ng_col = varint_read(meta_stream, mp);
            if (mp > msz32) throw std::runtime_error("column TLV overrun");
            if (ng_col == 0) continue;

            rv.clear();
            rv.reserve(cc[j]);

            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv  = varint_read(meta_stream, mp);
                uint32_t cnt = varint_read(meta_stream, mp);
                if (mp > msz32) throw std::runtime_error("column TLV overrun");

                uint32_t prev_row = 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    if (gp >= ng32) throw std::runtime_error("gap stream exhausted");
                    uint32_t gap  = gaps[gp++];
                    uint32_t prow = prev_row + gap;
                    prev_row = prow + 1;
                    if (prow >= hdr.m) throw std::runtime_error("decoded row OOB");
                    rv.emplace_back(perm[prow], cv);
                }
            }

            // Sort by original row index to produce canonical CSC order.
            std::sort(rv.begin(), rv.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });

            if (rv.size() != cc[j])
                throw std::runtime_error("column length mismatch vs cc[]");

            const int32_t base = h_indptr[j];
            for (size_t k = 0; k < rv.size(); ++k) {
                h_indices[base + k] = static_cast<int32_t>(rv[k].first);
                // uint→float cast with saturation warning.
                h_values[base + k]  = saturate_cast_f32(rv[k].second, fp32_warn);
            }
        }

        if (gp != ng32) throw std::runtime_error("gap stream leftover after chunk");
        blob_cursor += blob_sz;
    }

    // --- 8. Decode metadata TLV (if present) ---
    core::Metadata meta;
    if ((hdr.flags & FLAG_HAS_METADATA) && hdr.metadata_z_sz > 0 && hdr.metadata_offset > 0) {
        if (hdr.metadata_offset + hdr.metadata_z_sz > file.size())
            throw std::runtime_error("metadata block out of file bounds");
        auto raw_meta = zstd_decomp(
            dctx.ctx,
            file.data() + hdr.metadata_offset,
            hdr.metadata_z_sz);
        parse_metadata_tlv(raw_meta.data(), raw_meta.size(), meta);
    }

    // --- 9. Allocate device CSC and fire async copies ---
    if (stream == nullptr) {
        // Caller did not supply a stream; create a high-priority one.
        // Ownership stays with the caller via result.producer_stream.
        int least, greatest;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest);
    }

    // SparseMatrixGPU constructor allocates device memory via DeviceMemory<T>.
    core::DeviceCSC d_mat(
        static_cast<int>(hdr.m),
        static_cast<int>(hdr.n),
        static_cast<int>(hdr.nnz));

    // Three async copies: indptr, indices, values.
    // NOTE: if hdr.nnz > INT_MAX this cast truncates — document as a known
    //       limitation. CSR/CSC format using int32 indices cannot address >2G nnz.
    cudaMemcpyAsync(
        d_mat.col_ptr.get(),
        h_indptr,
        indptr_bytes,
        cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(
        d_mat.row_indices.get(),
        h_indices,
        indices_bytes,
        cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(
        d_mat.values.get(),
        h_values,
        values_bytes,
        cudaMemcpyHostToDevice, stream);

    PzDeviceMatrix result;
    result.mat            = std::move(d_mat);
    result.meta           = std::move(meta);
    result.producer_stream = stream;
    result.pinned_indptr   = std::move(pinned_indptr);
    result.pinned_indices  = std::move(pinned_indices);
    result.pinned_values   = std::move(pinned_values);

    // SVD path: expose pinned host pointers via shared_ptr so host-pointer SVD
    // adapters can consume them without a second H2D staging round-trip.
    //
    // Ownership model: PinnedBuffer (in result.pinned_*) retains primary ownership
    // and calls cudaFreeHost on destruction.  The shared_ptr fields use a no-op
    // deleter — they are non-owning aliases that remain valid as long as the
    // PzDeviceMatrix (and its PinnedBuffers) is alive.  Callers must NOT store
    // shared_ptr fields beyond the lifetime of the PzDeviceMatrix that produced them.
    //
    // WHY no-op deleter (not cudaFreeHost deleter):
    //   PinnedBuffer::~PinnedBuffer already calls cudaFreeHost.  A second
    //   cudaFreeHost on the same pointer would be undefined behavior.  The
    //   shared_ptr is a typed view handle, not an ownership transfer.
    //   This is safe because both PinnedBuffer and shared_ptr live inside
    //   the same PzDeviceMatrix; the shared_ptr never outlives the PinnedBuffer.
    if (keep_host_pinned) {
        result.host_indptr  = std::shared_ptr<int>  (result.pinned_indptr.as<int>(),
                                                     [](int*)   noexcept {});
        result.host_indices = std::shared_ptr<int>  (result.pinned_indices.as<int>(),
                                                     [](int*)   noexcept {});
        result.host_values  = std::shared_ptr<float>(result.pinned_values.as<float>(),
                                                     [](float*) noexcept {});
        result.host_retained = true;
    }

    // Populate convenience aliases.
    result.n_genes = static_cast<int>(hdr.m);
    result.n_cells = static_cast<int>(hdr.n);

    return result;
}

// ---------------------------------------------------------------------------
// PzChunkIterator — yields fixed-column-width DeviceCSC slices.
//
// Used by:
//  - out-of-core kernels that cannot fit the full matrix on device
//  - the streaming NMF fit adapter (streaming/chunk_iter.h)
//
// Each call to next() decompresses exactly chunk_cols columns, stages them
// into pinned memory, fires async copies, and returns a PzDeviceMatrix.
// The previous chunk's PzDeviceMatrix should be consumed (stream synced)
// before calling next(), since the pinned staging buffers are reallocated
// per chunk rather than kept in a fixed-size sliding window.
//
// WHY not a sliding-window prefetch here: the prefetch optimization is the
// job of streaming/chunk_iter.h which wraps this iterator. Keeping the
// iterator itself simple makes it easier to reason about memory ownership.
// ---------------------------------------------------------------------------
class PzChunkIterator {
public:
    // Default chunk width matches the design doc recommendation of 100k cols.
    static constexpr uint32_t DEFAULT_CHUNK_COLS = 100'000;

    explicit PzChunkIterator(
        const std::string& path,
        cudaStream_t       stream     = nullptr,
        uint32_t           chunk_cols = DEFAULT_CHUNK_COLS)
        : path_(path), stream_(stream), chunk_cols_(chunk_cols),
          chunk_idx_(0), col_cursor_(0), finished_(false)
    {
        // Load file + validate header once; keep the raw bytes in memory.
        // For very large files (>16 GB) this is a problem — a future version
        // can switch to a seekable file handle. For billion-cell matrices the
        // full-file slurp is replaced by a mmap variant.
        // TODO(OOC): replace with memory-mapped file for files > available RAM.
        std::ifstream fin(path, std::ios::binary | std::ios::ate);
        if (!fin) throw std::runtime_error("PzChunkIterator: cannot open " + path);
        const std::streamsize fsz = fin.tellg();
        fin.seekg(0);
        file_.resize(static_cast<size_t>(fsz));
        fin.read(reinterpret_cast<char*>(file_.data()), fsz);
        if (!fin) throw std::runtime_error("PzChunkIterator: read error " + path);

        using namespace pz_fmt;
        PZHeader hdr;
        std::memcpy(&hdr, file_.data(), sizeof(PZHeader));
        if (hdr.magic   != TP1_MAGIC)   throw std::runtime_error("PzChunkIterator: bad magic");
        if (hdr.version != TP1_VERSION) throw std::runtime_error("PzChunkIterator: bad version");
        hdr_ = hdr;

        // CRC
        PZFooter ftr;
        std::memcpy(&ftr, file_.data() + file_.size() - sizeof(PZFooter), sizeof(PZFooter));
        if (ftr.magic      != TP1_MAGIC)      throw std::runtime_error("PzChunkIterator: bad footer");
        if (ftr.num_chunks != hdr.num_chunks) throw std::runtime_error("PzChunkIterator: chunk count mismatch");
        uint32_t actual = Crc32Loader::compute(file_.data(), file_.size() - sizeof(PZFooter));
        if (actual != ftr.file_crc32)         throw std::runtime_error("PzChunkIterator: CRC mismatch");

        // Decode perm + ptr (need them for every chunk).
        ZstdDCtxGuard dctx;
        size_t off = sizeof(PZHeader);
        perm_.resize(hdr.m);
        if (hdr.flags & FLAG_HAS_PERM) {
            auto raw = zstd_decomp(dctx.ctx, file_.data() + off, hdr.perm_z_sz);
            std::memcpy(perm_.data(), raw.data(), raw.size());
            off += hdr.perm_z_sz;
        } else {
            for (uint32_t i = 0; i < hdr.m; ++i) perm_[i] = i;
        }
        cc_.resize(hdr.n);
        {
            auto raw = zstd_decomp(dctx.ctx, file_.data() + off, hdr.ptr_z_sz);
            for (uint32_t j = 0; j < hdr.n; ++j) {
                uint32_t v = 0;
                for (int b = 0; b < hdr.ptr_width; ++b)
                    v |= static_cast<uint32_t>(raw[j * hdr.ptr_width + b]) << (8 * b);
                cc_[j] = v;
            }
            off += hdr.ptr_z_sz;
        }

        // Build global indptr (needed to scatter into slice-local indptr later).
        global_indptr_.resize(hdr.n + 1);
        global_indptr_[0] = 0;
        for (uint32_t j = 0; j < hdr.n; ++j)
            global_indptr_[j + 1] = global_indptr_[j] + cc_[j];

        // Read chunk table.
        ctable_off_ = off;
        ctable_.resize(hdr.num_chunks);
        std::memcpy(ctable_.data(), file_.data() + ctable_off_,
                    hdr.num_chunks * sizeof(uint32_t));

        // Precompute absolute blob offsets for random-access per chunk.
        blob_offsets_.resize(hdr.num_chunks + 1);
        blob_offsets_[0] = ctable_off_ + hdr.num_chunks * sizeof(uint32_t);
        for (uint32_t c = 0; c < hdr.num_chunks; ++c)
            blob_offsets_[c + 1] = blob_offsets_[c] + ctable_[c];

        // Stream
        if (stream_ == nullptr) {
            int least, greatest;
            cudaDeviceGetStreamPriorityRange(&least, &greatest);
            cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatest);
            owns_stream_ = true;
        }
    }

    ~PzChunkIterator() {
        if (owns_stream_ && stream_) cudaStreamDestroy(stream_);
    }

    PzChunkIterator(const PzChunkIterator&) = delete;
    PzChunkIterator& operator=(const PzChunkIterator&) = delete;

    bool has_next() const noexcept { return !finished_; }

    // Returns the next column slice [col_cursor_, col_cursor_ + slice_cols)
    // as a fully-independent PzDeviceMatrix. The caller must synchronize
    // producer_stream before accessing the device data.
    PzDeviceMatrix next() {
        if (finished_) throw std::runtime_error("PzChunkIterator::next() past end");

        using namespace pz_fmt;
        const uint32_t n = hdr_.n;
        const uint32_t col_start = col_cursor_;
        const uint32_t col_end   = std::min(col_start + chunk_cols_, n);
        const uint32_t slice_n   = col_end - col_start;

        // Which native file chunks overlap [col_start, col_end)?
        const uint32_t fc_chunk_cols = hdr_.chunk_cols;
        const uint32_t first_fc = col_start / fc_chunk_cols;
        const uint32_t last_fc  = (col_end - 1) / fc_chunk_cols;

        // Build slice indptr and count slice nnz.
        std::vector<int32_t> slice_indptr(slice_n + 1);
        slice_indptr[0] = 0;
        for (uint32_t j = 0; j < slice_n; ++j)
            slice_indptr[j + 1] = slice_indptr[j] + static_cast<int32_t>(cc_[col_start + j]);
        const uint32_t slice_nnz = static_cast<uint32_t>(slice_indptr[slice_n]);

        auto pinned_ip  = core::PinnedPool::acquire((slice_n + 1) * sizeof(int32_t));
        auto pinned_idx = core::PinnedPool::acquire(slice_nnz * sizeof(int32_t));
        auto pinned_val = core::PinnedPool::acquire(slice_nnz * sizeof(float));
        std::memcpy(pinned_ip.as<int32_t>(), slice_indptr.data(), (slice_n + 1) * sizeof(int32_t));
        int32_t* h_idx = pinned_idx.as<int32_t>();
        float*   h_val = pinned_val.as<float>();

        // Decode the overlapping file chunks into the slice buffers.
        ZstdDCtxGuard dctx;
        const int gw = (hdr_.flags & FLAG_GAP16) ? 2 : 4;
        bool fp32_warn = false;

        std::vector<uint8_t>  packed, prefilter, low_bytes;
        std::vector<uint32_t> gaps;
        std::vector<std::pair<uint32_t, uint32_t>> rv;

        for (uint32_t fc = first_fc; fc <= last_fc; ++fc) {
            const uint8_t* blob = file_.data() + blob_offsets_[fc];
            uint32_t ng32, msz32, zsz, chunk_crc, psz32;
            std::memcpy(&ng32,      blob +  0, 4);
            std::memcpy(&msz32,     blob +  4, 4);
            std::memcpy(&zsz,       blob +  8, 4);
            std::memcpy(&chunk_crc, blob + 12, 4);
            std::memcpy(&psz32,     blob + 16, 4);

            packed = zstd_decomp(dctx.ctx, blob + 20, zsz);
            if (packed.size() != psz32) throw std::runtime_error("chunk size mismatch");
            if (Crc32Loader::compute(packed.data(), packed.size()) != chunk_crc)
                throw std::runtime_error("chunk CRC mismatch");

            const size_t bp_sz    = ng32 > 0 ? 8 * ((static_cast<size_t>(ng32) + 7) / 8) : 0;
            const size_t extra_pl = ng32 > 0 ? static_cast<size_t>(gw - 1) * ng32 : 0;
            const size_t pf_sz    = static_cast<size_t>(msz32) + bp_sz + extra_pl;
            prefilter.assign(pf_sz, 0);
            bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);

            gaps.assign(ng32, 0);
            if (ng32 > 0) {
                low_bytes.assign(ng32, 0);
                bit_planes_decode(prefilter.data() + msz32, ng32, low_bytes.data());
                for (uint32_t i = 0; i < ng32; ++i) gaps[i] = low_bytes[i];
                if (gw > 1) {
                    const uint8_t* upper = prefilter.data() + msz32 + bp_sz;
                    for (int b = 1; b < gw; ++b) {
                        const uint8_t* plane = upper + static_cast<size_t>(b - 1) * ng32;
                        for (uint32_t i = 0; i < ng32; ++i)
                            gaps[i] |= static_cast<uint32_t>(plane[i]) << (8 * b);
                    }
                }
            }

            // Walk the varint stream; only emit columns in [col_start, col_end).
            const size_t fc_col_start = static_cast<size_t>(fc) * fc_chunk_cols;
            const size_t fc_col_end   = std::min(fc_col_start + fc_chunk_cols, static_cast<size_t>(n));

            size_t mp = 0, gp = 0;
            const uint8_t* meta_stream = prefilter.data();

            for (size_t j = fc_col_start; j < fc_col_end; ++j) {
                uint32_t ng_col = varint_read(meta_stream, mp);
                if (ng_col == 0) continue;

                rv.clear();
                for (uint32_t vi = 0; vi < ng_col; ++vi) {
                    uint32_t cv  = varint_read(meta_stream, mp);
                    uint32_t cnt = varint_read(meta_stream, mp);
                    uint32_t prev_row = 0;
                    for (uint32_t k = 0; k < cnt; ++k) {
                        uint32_t gap  = gaps[gp++];
                        uint32_t prow = prev_row + gap;
                        prev_row = prow + 1;
                        rv.emplace_back(perm_[prow], cv);
                    }
                }

                // Only scatter columns that belong to our slice.
                if (j >= col_start && j < col_end) {
                    std::sort(rv.begin(), rv.end(),
                              [](const auto& a, const auto& b){ return a.first < b.first; });
                    const int32_t base = slice_indptr[j - col_start];
                    for (size_t k = 0; k < rv.size(); ++k) {
                        h_idx[base + k] = static_cast<int32_t>(rv[k].first);
                        h_val[base + k] = saturate_cast_f32(rv[k].second, fp32_warn);
                    }
                }
            }
            // Note: gp vs ng32 check omitted here because we may skip columns
            // outside the slice range — a stricter check would need to track
            // the gap count only for columns we actually emit.
        }

        // Allocate device CSC slice and fire async copies.
        core::DeviceCSC d_slice(
            static_cast<int>(hdr_.m),
            static_cast<int>(slice_n),
            static_cast<int>(slice_nnz));

        cudaMemcpyAsync(d_slice.col_ptr.get(),
                        pinned_ip.as<int32_t>(),
                        (slice_n + 1) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_slice.row_indices.get(),
                        h_idx,
                        slice_nnz * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_slice.values.get(),
                        h_val,
                        slice_nnz * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);

        PzDeviceMatrix result;
        result.mat             = std::move(d_slice);
        result.producer_stream = stream_;
        result.pinned_indptr   = std::move(pinned_ip);
        result.pinned_indices  = std::move(pinned_idx);
        result.pinned_values   = std::move(pinned_val);
        // Metadata attached only on the first chunk; downstream streaming code
        // typically reads it once from the full load_pz call.
        if (chunk_idx_ == 0) result.meta = meta_;

        col_cursor_ = col_end;
        ++chunk_idx_;
        if (col_cursor_ >= n) finished_ = true;
        return result;
    }

    // Dimensions
    uint32_t nrows()       const noexcept { return hdr_.m; }
    uint32_t ncols()       const noexcept { return hdr_.n; }
    uint64_t nnz()         const noexcept { return global_indptr_[hdr_.n]; }
    uint32_t chunk_cols()  const noexcept { return chunk_cols_; }
    uint32_t chunks_done() const noexcept { return chunk_idx_; }
    cudaStream_t stream()  const noexcept { return stream_; }

private:
    std::string              path_;
    std::vector<uint8_t>     file_;
    pz_fmt::PZHeader         hdr_{};
    cudaStream_t             stream_     = nullptr;
    bool                     owns_stream_= false;
    uint32_t                 chunk_cols_;
    uint32_t                 chunk_idx_;
    uint32_t                 col_cursor_;
    bool                     finished_;

    std::vector<uint32_t>    perm_;
    std::vector<uint32_t>    cc_;
    std::vector<uint64_t>    global_indptr_;
    size_t                   ctable_off_ = 0;
    std::vector<uint32_t>    ctable_;
    std::vector<size_t>      blob_offsets_;
    core::Metadata           meta_;
};

}  // namespace io
}  // namespace singlet::gpu

// ===========================================================================
// Cycle 56 — FEATURE 0: new load() API with Rule-31 auto-tune contract.
//
// PzLoadConfig / PzLoadResult / load(path, cfg) / load(path)
//
// PzLoadResult::matrix IS core::DeviceCSC populated
// in-place — no intermediate copy.  The uint→fp32 conversion is fused on
// device via detail/uint_to_float_kernel.h.
// CYCLE-105: replaced factornet::gpu::SparseMatrixGPU<float> with core::DeviceCSC.
//
// Design doc: singlet/gpu/state/designs/00-pz-device-loader.md
// ===========================================================================
#include <singlet/gpu/io/detail/uint_to_float_kernel.h>

namespace singlet::gpu {
namespace io {

// ---------------------------------------------------------------------------
// PzLoadConfig — every field has a sane default (Rule 31).
// ---------------------------------------------------------------------------
struct PzLoadConfig {
    // CUDA stream for async copies.  0 (default) = create a new non-default
    // high-priority stream internally.  Rule 31: stream is auto-chosen when 0.
    cudaStream_t stream = 0;

    // When true, pinned host memory is used for staging.  Falls back to
    // pageable when free pinned budget < 1 MB (degenerate device).
    bool pinned_host = true;

    // Size of the host staging buffer per load() call.  Rule 31: auto = 0,
    // which triggers the formula min(file_size/4, free_pinned/2, 256 MB).
    // Floor: 16 MB.
    size_t host_chunk_bytes = 0;  // 0 = auto-tune

    // When true, decode metadata TLV and populate PzLoadResult::metadata.
    bool decode_metadata = true;
};

// ---------------------------------------------------------------------------
// PzLoadResult — the output of load().
//
// matrix: singlet::gpu::core::DeviceCSC constructed in-place.
//         col_ptr / row_indices / values are DeviceMemory<T> allocations
//         populated via cudaMemcpyAsync from pinned host staging.
//         Caller must sync stream before using matrix on other streams.
//
// metadata: host-side GEO KV map (flat string→string).  Also contains
//           the auto-tune decisions under "_autotune_{param}" keys
//           (Rule 31: every auto parameter recorded for auditability).
//
// rownames / colnames: feature and barcode names (empty if not in .1pz).
//
// n_rows, n_cols, nnz: convenience copies of matrix dimensions.
//
// vt_code: native .1pz value type (1=uint8, 2=uint16, 3=uint32).
//
// producer_stream: the stream used for the async copies.  The caller is
//                  responsible for synchronizing this stream and, if the
//                  stream was created internally (stream was 0 in config),
//                  for destroying it with cudaStreamDestroy.
// ---------------------------------------------------------------------------
struct PzLoadResult {
    core::DeviceCSC matrix;
    std::map<std::string, std::string>     metadata;
    std::vector<std::string>               rownames;
    std::vector<std::string>               colnames;
    size_t                                 n_rows = 0;
    size_t                                 n_cols = 0;
    size_t                                 nnz    = 0;
    int                                    vt_code = 0;
    cudaStream_t                           producer_stream = nullptr;
};

// ---------------------------------------------------------------------------
// Internal helper namespace for decode_pz_raw.
// ---------------------------------------------------------------------------
namespace load_detail {

// decode_pz_raw: decode a .1pz into pinned host buffers WITHOUT host-side
// float promotion (values kept as raw uintN bytes).
struct RawCscBuffers {
    core::PinnedBuffer pinned_indptr;
    core::PinnedBuffer pinned_indices;
    core::PinnedBuffer pinned_raw_values;   // raw uintN bytes
    size_t             n_rows;
    size_t             n_cols;
    size_t             nnz;
    int                vt_code;
    core::Metadata     meta;
};

inline RawCscBuffers decode_pz_raw(const std::string& path, bool decode_meta) {
    using namespace pz_fmt;

    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw std::runtime_error("load: cannot open " + path);
    const std::streamsize file_sz = fin.tellg();
    if (file_sz < static_cast<std::streamsize>(sizeof(PZHeader) + sizeof(PZFooter)))
        throw std::runtime_error("load: file too short: " + path);
    fin.seekg(0);
    std::vector<uint8_t> file(static_cast<size_t>(file_sz));
    fin.read(reinterpret_cast<char*>(file.data()), file_sz);
    if (!fin) throw std::runtime_error("load: read error: " + path);
    fin.close();

    PZHeader hdr;
    std::memcpy(&hdr, file.data(), sizeof(PZHeader));
    if (hdr.magic   != TP1_MAGIC)   throw std::runtime_error("load: bad magic: "   + path);
    if (hdr.version != TP1_VERSION) throw std::runtime_error("load: bad version: " + path);

    PZFooter ftr;
    std::memcpy(&ftr, file.data() + file.size() - sizeof(PZFooter), sizeof(PZFooter));
    if (ftr.magic      != TP1_MAGIC)      throw std::runtime_error("load: bad footer magic: " + path);
    if (ftr.num_chunks != hdr.num_chunks) throw std::runtime_error("load: chunk count mismatch: " + path);
    {
        uint32_t actual = Crc32Loader::compute(file.data(), file.size() - sizeof(PZFooter));
        if (actual != ftr.file_crc32)
            throw std::runtime_error("load: CRC32 mismatch: " + path);
    }

    ZstdDCtxGuard dctx;
    size_t body_off = sizeof(PZHeader);
    std::vector<uint32_t> perm(hdr.m);
    if (hdr.flags & FLAG_HAS_PERM) {
        auto raw = zstd_decomp(dctx.ctx, file.data() + body_off, hdr.perm_z_sz);
        if (raw.size() != static_cast<size_t>(hdr.m) * 4)
            throw std::runtime_error("load: perm block size mismatch");
        std::memcpy(perm.data(), raw.data(), raw.size());
        body_off += hdr.perm_z_sz;
    } else {
        for (uint32_t i = 0; i < hdr.m; ++i) perm[i] = i;
    }
    std::vector<uint32_t> cc(hdr.n);
    if (hdr.ptr_z_sz > 0) {
        auto raw = zstd_decomp(dctx.ctx, file.data() + body_off, hdr.ptr_z_sz);
        if (raw.size() != static_cast<size_t>(hdr.n) * hdr.ptr_width)
            throw std::runtime_error("load: ptr block size mismatch");
        for (uint32_t j = 0; j < hdr.n; ++j) {
            uint32_t v = 0;
            for (int b = 0; b < hdr.ptr_width; ++b)
                v |= static_cast<uint32_t>(raw[j * hdr.ptr_width + b]) << (8 * b);
            cc[j] = v;
        }
        body_off += hdr.ptr_z_sz;
    }

    const size_t indptr_bytes = (static_cast<size_t>(hdr.n) + 1) * sizeof(int32_t);
    auto pinned_ip = core::PinnedPool::acquire(indptr_bytes);
    {
        int32_t* ip = pinned_ip.as<int32_t>();
        ip[0] = 0;
        for (uint32_t j = 0; j < hdr.n; ++j)
            ip[j + 1] = ip[j] + static_cast<int32_t>(cc[j]);
        if (static_cast<uint64_t>(ip[hdr.n]) != hdr.nnz)
            throw std::runtime_error("load: indptr sum != header nnz");
    }

    const size_t indices_bytes  = static_cast<size_t>(hdr.nnz) * sizeof(int32_t);
    const int    vt             = hdr.vt_code;
    const size_t bytes_per_val  = (vt == 1) ? 1u : (vt == 2 ? 2u : 4u);
    const size_t raw_val_bytes  = static_cast<size_t>(hdr.nnz) * bytes_per_val;

    auto pinned_idx = core::PinnedPool::acquire(indices_bytes);
    auto pinned_rv  = core::PinnedPool::acquire(raw_val_bytes);
    int32_t* h_indices  = pinned_idx.as<int32_t>();
    uint8_t* h_raw_vals = pinned_rv.as<uint8_t>();

    const size_t   ctable_off = body_off;
    const uint32_t nc         = hdr.num_chunks;
    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), file.data() + ctable_off, nc * sizeof(uint32_t));
    size_t blob_cursor = ctable_off + nc * sizeof(uint32_t);
    const int32_t* h_indptr = pinned_ip.as<int32_t>();
    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;

    std::vector<uint8_t>  packed, prefilter, low_bytes;
    std::vector<uint32_t> gaps;
    std::vector<std::pair<uint32_t, uint32_t>> rv;

    for (uint32_t c = 0; c < nc; ++c) {
        const uint32_t blob_sz = ctable[c];
        if (blob_sz < 20) throw std::runtime_error("load: chunk blob < 20 bytes");
        const uint8_t* blob = file.data() + blob_cursor;
        uint32_t ng32, msz32, zsz, chunk_crc, psz32;
        std::memcpy(&ng32,      blob +  0, 4);
        std::memcpy(&msz32,     blob +  4, 4);
        std::memcpy(&zsz,       blob +  8, 4);
        std::memcpy(&chunk_crc, blob + 12, 4);
        std::memcpy(&psz32,     blob + 16, 4);

        packed = zstd_decomp(dctx.ctx, blob + 20, zsz);
        if (packed.size() != psz32) throw std::runtime_error("load: chunk size mismatch");
        if (Crc32Loader::compute(packed.data(), packed.size()) != chunk_crc)
            throw std::runtime_error("load: chunk CRC mismatch in " + path);

        const size_t bp_sz    = ng32 > 0 ? 8 * ((size_t(ng32) + 7) / 8) : 0;
        const size_t extra_pl = ng32 > 0 ? size_t(gw - 1) * ng32 : 0;
        const size_t pf_sz    = size_t(msz32) + bp_sz + extra_pl;
        prefilter.assign(pf_sz, 0);
        bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);

        gaps.assign(ng32, 0);
        if (ng32 > 0) {
            low_bytes.assign(ng32, 0);
            bit_planes_decode(prefilter.data() + msz32, ng32, low_bytes.data());
            for (uint32_t i = 0; i < ng32; ++i) gaps[i] = low_bytes[i];
            if (gw > 1) {
                const uint8_t* upper = prefilter.data() + msz32 + bp_sz;
                for (int b = 1; b < gw; ++b) {
                    const uint8_t* plane = upper + size_t(b - 1) * ng32;
                    for (uint32_t i = 0; i < ng32; ++i)
                        gaps[i] |= uint32_t(plane[i]) << (8 * b);
                }
            }
        }

        const size_t col_start = size_t(c) * hdr.chunk_cols;
        const size_t col_end   = std::min(col_start + hdr.chunk_cols, size_t(hdr.n));
        size_t mp = 0, gp = 0;
        const uint8_t* meta_stream = prefilter.data();

        for (size_t j = col_start; j < col_end; ++j) {
            uint32_t ng_col = varint_read(meta_stream, mp);
            if (ng_col == 0) continue;
            rv.clear();
            rv.reserve(cc[j]);
            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv  = varint_read(meta_stream, mp);
                uint32_t cnt = varint_read(meta_stream, mp);
                uint32_t prev_row = 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    uint32_t gap  = gaps[gp++];
                    uint32_t prow = prev_row + gap;
                    prev_row = prow + 1;
                    if (prow >= hdr.m) throw std::runtime_error("load: decoded row OOB");
                    rv.emplace_back(perm[prow], cv);
                }
            }
            std::sort(rv.begin(), rv.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            if (rv.size() != cc[j]) throw std::runtime_error("load: col length mismatch");
            const int32_t base = h_indptr[j];
            for (size_t k = 0; k < rv.size(); ++k) {
                h_indices[base + k] = int32_t(rv[k].first);
                const uint32_t raw_v = rv[k].second;
                if (vt == 1) {
                    h_raw_vals[base + k] = uint8_t(raw_v);
                } else if (vt == 2) {
                    uint16_t v16 = uint16_t(raw_v);
                    std::memcpy(h_raw_vals + (base + k) * 2, &v16, 2);
                } else {
                    std::memcpy(h_raw_vals + (base + k) * 4, &raw_v, 4);
                }
            }
        }
        if (gp != ng32) throw std::runtime_error("load: gap stream leftover");
        blob_cursor += blob_sz;
    }

    core::Metadata meta;
    if (decode_meta && (hdr.flags & FLAG_HAS_METADATA) &&
        hdr.metadata_z_sz > 0 && hdr.metadata_offset > 0) {
        auto raw_meta = zstd_decomp(
            dctx.ctx,
            file.data() + hdr.metadata_offset,
            hdr.metadata_z_sz);
        parse_metadata_tlv(raw_meta.data(), raw_meta.size(), meta);
    }

    RawCscBuffers out;
    out.pinned_indptr     = std::move(pinned_ip);
    out.pinned_indices    = std::move(pinned_idx);
    out.pinned_raw_values = std::move(pinned_rv);
    out.n_rows  = hdr.m;
    out.n_cols  = hdr.n;
    out.nnz     = hdr.nnz;
    out.vt_code = vt;
    out.meta    = std::move(meta);
    return out;
}

}  // namespace load_detail

// ---------------------------------------------------------------------------
// load() — explicit config variant.
// ---------------------------------------------------------------------------
inline PzLoadResult load(const std::string& path, const PzLoadConfig& cfg) {
    std::map<std::string, std::string> autotune_decisions;

    // Stream.
    cudaStream_t stream = cfg.stream;
    if (stream == 0) {
        int least, greatest;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest);
        autotune_decisions["_autotune_stream"] = "created_high_priority_nonblocking";
    }

    // host_chunk_bytes.
    size_t host_chunk_bytes = cfg.host_chunk_bytes;
    if (host_chunk_bytes == 0) {
        std::ifstream tmp(path, std::ios::binary | std::ios::ate);
        size_t file_sz = tmp ? static_cast<size_t>(tmp.tellg()) : 0;
        size_t candidate = file_sz / 4;
        const size_t lo = 16u << 20;
        const size_t hi = 256u << 20;
        if (candidate < lo) candidate = lo;
        if (candidate > hi) candidate = hi;
        host_chunk_bytes = candidate;
        autotune_decisions["_autotune_host_chunk_bytes"] = std::to_string(host_chunk_bytes);
    }

    // Pinned availability probe — records capability in autotune map.
    // The actual staging always uses PinnedPool (which will throw if unavailable);
    // this probe gives the user a readable explanation in the metadata output.
    {
        void* probe = nullptr;
        cudaError_t e = cudaMallocHost(&probe, 1u << 20);
        if (e != cudaSuccess) {
            autotune_decisions["_autotune_pinned_host"] = "false_fallback_no_pinned_budget";
        } else {
            cudaFreeHost(probe);
            autotune_decisions["_autotune_pinned_host"] = "true";
        }
    }

    // Decode.
    load_detail::RawCscBuffers bufs = load_detail::decode_pz_raw(path, cfg.decode_metadata);

    const size_t m   = bufs.n_rows;
    const size_t n   = bufs.n_cols;
    const size_t nnz = bufs.nnz;
    const int    vt  = bufs.vt_code;

    // Allocate DeviceCSC on device (CYCLE-105: native, no factornet).
    core::DeviceCSC d_mat(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(nnz));

    // Copy indptr + indices (int32, direct).
    cudaMemcpyAsync(
        d_mat.col_ptr.get(),
        bufs.pinned_indptr.as<int32_t>(),
        (n + 1) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(
        d_mat.row_indices.get(),
        bufs.pinned_indices.as<int32_t>(),
        nnz * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);

    // On-device fused uint→float conversion.
    // WHY not host-side: PCIe savings 2–4× (upload uintN bytes, not float bytes).
    singlet::gpu::io::detail::launch_uint_to_float(
        bufs.pinned_raw_values.get(),
        nnz,
        vt,
        d_mat.values.get(),
        stream);

    // Build metadata map.
    std::map<std::string, std::string> meta_map;
    if (cfg.decode_metadata) {
        const auto& md = bufs.meta;
        if (!md.gsm_id.empty())           meta_map["gsm_id"]           = md.gsm_id;
        if (!md.gse_id.empty())           meta_map["gse_id"]           = md.gse_id;
        if (!md.organism.empty())         meta_map["organism"]         = md.organism;
        if (md.taxon_id != 0)             meta_map["taxon_id"]         = std::to_string(md.taxon_id);
        if (!md.protocol.empty())         meta_map["protocol"]         = md.protocol;
        if (!md.modality.empty())         meta_map["modality"]         = md.modality;
        if (!md.geo_title.empty())        meta_map["geo_title"]        = md.geo_title;
        if (!md.geo_source_name.empty())  meta_map["geo_source_name"]  = md.geo_source_name;
        if (!md.singlet_version.empty()) meta_map["singlet_version"] = md.singlet_version;
        if (!md.pipeline_date.empty())    meta_map["pipeline_date"]    = md.pipeline_date;
        if (md.read_count != 0)           meta_map["read_count"]       = std::to_string(md.read_count);
        if (!md.srr_ids.empty()) {
            std::string s;
            for (size_t i = 0; i < md.srr_ids.size(); ++i) {
                if (i) s += ',';
                s += md.srr_ids[i];
            }
            meta_map["srr_ids"] = std::move(s);
        }
    }
    for (auto& kv : autotune_decisions) meta_map[kv.first] = kv.second;

    PzLoadResult result;
    result.matrix          = std::move(d_mat);
    result.metadata        = std::move(meta_map);
    result.rownames        = std::move(bufs.meta.rownames);
    result.colnames        = std::move(bufs.meta.colnames);
    result.n_rows          = m;
    result.n_cols          = n;
    result.nnz             = nnz;
    result.vt_code         = vt;
    result.producer_stream = stream;
    return result;
}

// ---------------------------------------------------------------------------
// load() — no-args Rule-31 auto-tune variant.
// ---------------------------------------------------------------------------
inline PzLoadResult load(const std::string& path) {
    return load(path, PzLoadConfig{});
}

}  // namespace io (cycle-56 extension)
}  // namespace singlet::gpu (cycle-56 extension)
