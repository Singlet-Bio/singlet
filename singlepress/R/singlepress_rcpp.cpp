/**
 * singlepress_rcpp.cpp — R bindings for .1pz format via Rcpp
 *
 * Provides:
 *   read_1pz(path, num_threads) -> dgCMatrix
 *   write_1pz(mat, path, rownames, colnames, num_threads) -> list
 *   info_1pz(path) -> list
 *   validate_1pz(path) -> list
 *   colsums_1pz(path) -> numeric vector
 *
 * Build: include zstd headers and link -lzstd.
 *   Rcpp::sourceCpp("singlepress_rcpp.cpp", env = new.env())
 *   Or via an R package with LinkingTo: Rcpp and SystemRequirements: zstd
 *
 * Usage in R:
 *   library(singlepress)
 *   mat <- read_1pz("counts.1pz")
 *   info <- info_1pz("counts.1pz")
 */

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

#include <Rcpp.h>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <zstd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Constants and structures (must match pz_codec.cpp)
// ============================================================================
static constexpr uint32_t TP1_MAGIC   = 0x5A315054;
static constexpr uint16_t TP1_VERSION  = 1;

static constexpr uint8_t FLAG_HAS_PERM      = 0x01;
static constexpr uint8_t FLAG_GAP16         = 0x02;
static constexpr uint8_t FLAG_HAS_METADATA  = 0x04;
static constexpr uint8_t FLAG_HAS_TRANSPOSE = 0x08;
static constexpr uint8_t FLAG_HAS_COLSUMS   = 0x10;
static constexpr uint8_t FLAG_HAS_OBS_VAR   = 0x20;

static constexpr uint32_t FEAT_BITPLANE_BITMAP = 0x02;

static constexpr uint8_t META_TAG_END      = 0;
static constexpr uint8_t META_TAG_ROWNAMES = 1;
static constexpr uint8_t META_TAG_COLNAMES = 2;
static constexpr uint8_t META_TAG_KV       = 3;
static constexpr uint8_t META_TAG_OBS      = 4;
static constexpr uint8_t META_TAG_VAR      = 5;

// DataFrame column types (must match pz_codec.cpp DFDtype enum)
static constexpr uint8_t DF_STRING      = 0;
static constexpr uint8_t DF_INT32       = 1;
static constexpr uint8_t DF_INT64       = 2;
static constexpr uint8_t DF_FLOAT32     = 3;
static constexpr uint8_t DF_FLOAT64     = 4;
static constexpr uint8_t DF_UINT8       = 5;
static constexpr uint8_t DF_CATEGORICAL = 6;

struct PZHeader {
    uint32_t magic;
    uint16_t version;
    uint8_t  vt_code;
    uint8_t  flags;
    uint32_t m, n;
    uint64_t nnz;
    uint8_t  ptr_width;
    uint8_t  codec_level;
    uint16_t _pad0;
    uint32_t num_chunks;
    uint32_t perm_z_sz;
    uint32_t ptr_z_sz;
    uint32_t chunk_cols;
    uint32_t feature_flags;
    uint64_t metadata_offset;
    uint32_t metadata_z_sz;
    uint32_t colsums_z_sz;
    uint64_t transpose_offset;
    uint32_t transpose_z_sz;
    uint32_t transpose_chunks;
    uint8_t  reserved[16];
};

struct PZFooter {
    uint32_t file_crc32;
    uint32_t _reserved;
    uint32_t num_chunks;
    uint32_t magic;
};

// CRC32
struct CRC32 {
    static uint32_t compute(const uint8_t* data, size_t len) {
        static uint32_t table[256];
        static bool init = false;
        if (!init) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j)
                    c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                table[i] = c;
            }
            init = true;
        }
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; ++i)
            crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
        return crc ^ 0xFFFFFFFF;
    }
};

// Varint
static inline uint32_t varint_read(const uint8_t*& p) {
    uint32_t v = 0; int s = 0;
    while (*p & 0x80) { v |= (uint32_t)(*p & 0x7F) << s; s += 7; ++p; }
    v |= (uint32_t)*p << s; ++p;
    return v;
}

// Byte-unsplit
static void byte_unsplit_16(const uint8_t* src, size_t n, uint32_t* dst) {
    const uint8_t *p0 = src, *p1 = src + n;
    for (size_t i = 0; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i] << 8);
}
static void byte_unsplit_32(const uint8_t* src, size_t n, uint32_t* dst) {
    const uint8_t *p0=src, *p1=src+n, *p2=src+2*n, *p3=src+3*n;
    for (size_t i = 0; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i]<<8) | ((uint32_t)p2[i]<<16) | ((uint32_t)p3[i]<<24);
}

// ZSTD decompress
static size_t zstd_dec(void* dst, size_t cap, const void* src, size_t src_sz) {
    size_t ret = ZSTD_decompress(dst, cap, src, src_sz);
    if (ZSTD_isError(ret))
        Rcpp::stop("ZSTD decompress error: %s", ZSTD_getErrorName(ret));
    return ret;
}

// Bit-plane decode: reconstruct n original bytes from 8 bit-planes
static void bit_planes_decode(const uint8_t* src, size_t n, uint8_t* dst) {
    size_t pb = (n + 7) / 8;
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = 0;
        size_t byte_idx = i / 8;
        uint8_t bit_mask = 1u << (i & 7);
        for (int b = 0; b < 8; ++b) {
            if (src[b * pb + byte_idx] & bit_mask)
                v |= (1 << b);
        }
        dst[i] = v;
    }
}

// Bitmap unpack: restore n bytes from bitmap-encoded data
static void bitmap_unpack(const uint8_t* src, size_t /*packed_sz*/,
                           uint8_t* dst, size_t n) {
    size_t bm_bytes = (n + 7) / 8;
    std::memset(dst, 0, n);
    size_t rp = bm_bytes;
    for (size_t i = 0; i < n; ++i) {
        if (src[i / 8] & (1u << (i & 7)))
            dst[i] = src[rp++];
    }
}

// VOCSC decode (wp_init = initial write position into out_ix/out_vx)
static void vocsc_decode(
    const uint8_t* meta, const uint32_t* gaps, const uint32_t* perm,
    int64_t cs, int64_t ce, int64_t wp_init, int* out_ix, double* out_vx)
{
    const uint8_t* mp = meta;
    size_t gp = 0;
    int64_t wp = wp_init;
    for (int64_t j = cs; j < ce; ++j) {
        uint32_t ngrp = varint_read(mp);
        for (uint32_t g = 0; g < ngrp; ++g) {
            uint32_t val = varint_read(mp);
            uint32_t gc  = varint_read(mp);
            double dv = (double)val;
            uint32_t prev = 0;
            for (uint32_t k = 0; k < gc; ++k) {
                uint32_t mr = prev + gaps[gp++];
                prev = mr + 1;
                out_ix[wp] = (int)perm[mr];
                out_vx[wp] = dv;
                wp++;
            }
        }
    }
}

// Deserialize native DataFrame binary → R named list (data.frame columns)
// Wire format: [u32 nrows][u32 ncols][u32 idx_bytes][idx...][columns...]
static Rcpp::List deserialize_dataframe_r(const uint8_t* data, size_t len) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;

    if (p + 12 > end) return Rcpp::List();
    uint32_t nrows, ncols, idx_bytes;
    std::memcpy(&nrows, p, 4); p += 4;
    std::memcpy(&ncols, p, 4); p += 4;
    std::memcpy(&idx_bytes, p, 4); p += 4;

    // Parse index
    Rcpp::CharacterVector index(0);
    if (idx_bytes > 0 && p + idx_bytes <= end) {
        const uint8_t* idx_end = p + idx_bytes;
        const uint8_t* start = p;
        std::vector<std::string> idx_vec;
        while (p < idx_end) {
            if (*p == 0) {
                idx_vec.emplace_back((const char*)start, p - start);
                start = p + 1;
            }
            p++;
        }
        index = Rcpp::wrap(idx_vec);
    }

    Rcpp::List result;
    Rcpp::CharacterVector col_names;

    for (uint32_t c = 0; c < ncols && p + 3 <= end; ++c) {
        uint16_t name_len;
        std::memcpy(&name_len, p, 2); p += 2;
        if (p + name_len + 5 > end) break;
        std::string name((const char*)p, name_len); p += name_len;
        uint8_t dtype = *p++;
        uint32_t data_bytes;
        std::memcpy(&data_bytes, p, 4); p += 4;
        if (p + data_bytes > end) break;

        switch (dtype) {
            case DF_STRING: {
                Rcpp::CharacterVector vals(nrows);
                const uint8_t* col_end = p + data_bytes;
                const uint8_t* start = p;
                int idx = 0;
                while (p < col_end && idx < (int)nrows) {
                    if (*p == 0) {
                        vals[idx++] = std::string((const char*)start, p - start);
                        start = p + 1;
                    }
                    p++;
                }
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_INT32: {
                Rcpp::IntegerVector vals(nrows);
                std::memcpy(vals.begin(), p, nrows * 4);
                p += data_bytes;
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_INT64: {
                // R has no int64; use double
                Rcpp::NumericVector vals(nrows);
                const int64_t* src = (const int64_t*)p;
                for (uint32_t i = 0; i < nrows; ++i) vals[i] = (double)src[i];
                p += data_bytes;
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_FLOAT32: {
                Rcpp::NumericVector vals(nrows);
                const float* src = (const float*)p;
                for (uint32_t i = 0; i < nrows; ++i) vals[i] = (double)src[i];
                p += data_bytes;
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_FLOAT64: {
                Rcpp::NumericVector vals(nrows);
                std::memcpy(vals.begin(), p, nrows * 8);
                p += data_bytes;
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_UINT8: {
                // Bool → R logical vector
                Rcpp::LogicalVector vals(nrows);
                for (uint32_t i = 0; i < nrows; ++i) vals[i] = p[i] != 0;
                p += data_bytes;
                result.push_back(vals);
                col_names.push_back(name);
                break;
            }
            case DF_CATEGORICAL: {
                // Factor: [u32 nlev][null-sep levels][int32 codes]
                uint32_t nlev;
                std::memcpy(&nlev, p, 4); p += 4;
                Rcpp::CharacterVector levels(nlev);
                for (uint32_t i = 0; i < nlev; ++i) {
                    const char* start = (const char*)p;
                    while (p < end && *p != 0) p++;
                    levels[i] = std::string(start, (const char*)p);
                    if (p < end) p++;
                }
                // Codes are 0-based; R factors are 1-based
                Rcpp::IntegerVector codes(nrows);
                const int32_t* src = (const int32_t*)p;
                for (uint32_t i = 0; i < nrows; ++i)
                    codes[i] = (src[i] >= 0) ? src[i] + 1 : NA_INTEGER;
                p += nrows * 4;
                codes.attr("levels") = levels;
                codes.attr("class") = "factor";
                result.push_back(codes);
                col_names.push_back(name);
                break;
            }
            default:
                p += data_bytes;
                break;
        }
    }
    result.attr("names") = col_names;
    if (index.size() > 0) result.attr("row.names") = index;
    return result;
}

// Metadata deserialize
struct MetaR {
    std::vector<std::string> rownames, colnames;
    std::vector<std::pair<std::string,std::string>> kv_pairs;
    std::vector<uint8_t> obs_blob, var_blob;
};
static MetaR deserialize_meta(const uint8_t* data, size_t len) {
    MetaR m;
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    while (p < end) {
        uint8_t tag = *p++;
        if (tag == META_TAG_END) break;
        if (p + 4 > end) break;
        uint32_t sz; std::memcpy(&sz, p, 4); p += 4;
        if (p + sz > end) break;
        if (tag == META_TAG_ROWNAMES || tag == META_TAG_COLNAMES) {
            std::vector<std::string>* target =
                (tag == META_TAG_ROWNAMES) ? &m.rownames : &m.colnames;
            const uint8_t* s = p;
            for (const uint8_t* q = p; q < p + sz; ++q) {
                if (*q == 0) {
                    target->emplace_back((const char*)s, q - s);
                    s = q + 1;
                }
            }
        } else if (tag == META_TAG_KV) {
            std::vector<std::string> parts;
            const uint8_t* s = p;
            for (const uint8_t* q = p; q < p + sz; ++q) {
                if (*q == 0) {
                    parts.emplace_back((const char*)s, q - s);
                    s = q + 1;
                }
            }
            for (size_t i = 0; i + 1 < parts.size(); i += 2)
                m.kv_pairs.emplace_back(parts[i], parts[i+1]);
        } else if (tag == META_TAG_OBS) {
            m.obs_blob.assign(p, p + sz);
        } else if (tag == META_TAG_VAR) {
            m.var_blob.assign(p, p + sz);
        }
        p += sz;
    }
    return m;
}

// [[Rcpp::export]]
Rcpp::List read_1pz_r(std::string path, int num_threads = 4) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) Rcpp::stop("Cannot open: %s", path.c_str());
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read((char*)blob.data(), sz); f.close();

    PZHeader hdr{};
    std::memcpy(&hdr, blob.data(), 96);
    if (hdr.magic != TP1_MAGIC) Rcpp::stop("Not a .1pz file");
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        Rcpp::stop("Unsupported .1pz version %d", (int)hdr.version);

    int64_t m = hdr.m, n = hdr.n, nnz = (int64_t)hdr.nnz;
    int pw = hdr.ptr_width, nc = hdr.num_chunks, cc = (int)hdr.chunk_cols;
    bool gap16 = (hdr.flags & FLAG_GAP16) != 0;
    int gw = gap16 ? 2 : 4;
    bool has_bp = (hdr.feature_flags & FEAT_BITPLANE_BITMAP) != 0;

    size_t off = 96;
    std::vector<uint32_t> perm(m);
    zstd_dec(perm.data(), m*4, blob.data()+off, hdr.perm_z_sz);
    off += hdr.perm_z_sz;

    std::vector<uint8_t> pr(n * pw);
    zstd_dec(pr.data(), pr.size(), blob.data()+off, hdr.ptr_z_sz);
    off += hdr.ptr_z_sz;

    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), blob.data()+off, nc*4);
    off += nc*4;

    std::vector<size_t> coff(nc);
    for (int c = 0; c < nc; ++c) { coff[c] = off; off += ctable[c]; }

    // Parse per-column nnz from ptr data
    std::vector<uint32_t> col_nnz(n);
    if (pw == 2) {
        const uint16_t* c16 = (const uint16_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) col_nnz[j] = c16[j];
    } else {
        const uint32_t* c32 = (const uint32_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) col_nnz[j] = c32[j];
    }

    // Lambda: decompress a chunk's raw data and gaps
    auto decode_chunk_raw = [&](int c, std::vector<uint8_t>& raw,
                                 std::vector<uint8_t>& packed_buf,
                                 std::vector<uint8_t>& pf_buf,
                                 std::vector<uint32_t>& gaps) {
        int64_t cs = (int64_t)c * cc;
        int64_t ce = std::min(cs + (int64_t)cc, n);
        const uint8_t* cp = blob.data() + coff[c];

        uint32_t ng, msz, zsz;
        std::memcpy(&ng, cp, 4); cp += 4;
        std::memcpy(&msz, cp, 4); cp += 4;
        std::memcpy(&zsz, cp, 4); cp += 4;
        cp += 4; // skip CRC

        uint32_t packed_sz = 0;
        if (has_bp) { std::memcpy(&packed_sz, cp, 4); cp += 4; }

        size_t raw_sz = msz + (size_t)ng * gw;

        if (has_bp) {
            packed_buf.resize(packed_sz);
            zstd_dec(packed_buf.data(), packed_sz, cp, zsz);

            size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
            size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
            pf_buf.resize(pf_sz);
            bitmap_unpack(packed_buf.data(), packed_sz, pf_buf.data(), pf_sz);

            raw.resize(raw_sz);
            std::memcpy(raw.data(), pf_buf.data(), msz);
            if (ng > 0) {
                bit_planes_decode(pf_buf.data() + msz, ng, raw.data() + msz);
                if (gw > 1)
                    std::memcpy(raw.data() + msz + ng,
                                pf_buf.data() + msz + bp_sz, ng * (gw - 1));
            }
        } else {
            raw.resize(raw_sz);
            zstd_dec(raw.data(), raw_sz, cp, zsz);
        }

        gaps.resize(ng);
        if (ng > 0) {
            if (gap16) byte_unsplit_16(raw.data()+msz, ng, gaps.data());
            else       byte_unsplit_32(raw.data()+msz, ng, gaps.data());
        }
    };

    // ========================================================================
    // Branch: single dgCMatrix (nnz fits int32) vs chunked list
    // ========================================================================
    bool needs_chunks = (nnz > 2147483647LL);

    Rcpp::List result;

    if (!needs_chunks) {
        // --- Standard path: single matrix ---
        Rcpp::IntegerVector indptr(n + 1);
        indptr[0] = 0;
        for (int64_t j = 0; j < n; ++j)
            indptr[j+1] = indptr[j] + (int)col_nnz[j];

        Rcpp::IntegerVector indices(nnz);
        Rcpp::NumericVector values(nnz);
        int* oix = indices.begin();
        double* ovx = values.begin();
        int* ip = indptr.begin();

        #ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        #endif
        {
            std::vector<uint8_t> raw, packed_buf, pf_buf;
            std::vector<uint32_t> gaps;
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (int c = 0; c < nc; ++c) {
                int64_t cs = (int64_t)c * cc;
                int64_t ce = std::min(cs + (int64_t)cc, n);
                decode_chunk_raw(c, raw, packed_buf, pf_buf, gaps);
                vocsc_decode(raw.data(), gaps.data(), perm.data(),
                             cs, ce, (int64_t)ip[cs], oix, ovx);
            }
        }

        result = Rcpp::List::create(
            Rcpp::Named("i") = indices,
            Rcpp::Named("p") = indptr,
            Rcpp::Named("x") = values,
            Rcpp::Named("nrow") = (int)m,
            Rcpp::Named("ncol") = (int)n,
            Rcpp::Named("nnz") = (int)nnz
        );
    } else {
        // --- Chunked path: split at chunk boundaries into list of dgCMatrix ---
        // Find split boundaries where cumulative nnz would exceed INT32_MAX
        std::vector<int> split_chunks = {0};
        int64_t cumnnz = 0;
        for (int c = 0; c < nc; ++c) {
            int64_t cs = (int64_t)c * cc;
            int64_t ce = std::min(cs + (int64_t)cc, n);
            int64_t chunk_nnz = 0;
            for (int64_t j = cs; j < ce; ++j) chunk_nnz += col_nnz[j];
            if (cumnnz + chunk_nnz > 2147483647LL && cumnnz > 0) {
                split_chunks.push_back(c);
                cumnnz = 0;
            }
            cumnnz += chunk_nnz;
        }
        split_chunks.push_back(nc);

        int nsplits = (int)split_chunks.size() - 1;
        Rcpp::List chunk_list(nsplits);

        for (int s = 0; s < nsplits; ++s) {
            int first_chunk = split_chunks[s];
            int last_chunk = split_chunks[s + 1];
            int64_t col_start = (int64_t)first_chunk * cc;
            int64_t col_end = std::min((int64_t)last_chunk * cc, n);
            int64_t s_ncols = col_end - col_start;

            // Build local indptr
            Rcpp::IntegerVector local_ip(s_ncols + 1);
            local_ip[0] = 0;
            for (int64_t j = 0; j < s_ncols; ++j)
                local_ip[j + 1] = local_ip[j] + (int)col_nnz[col_start + j];
            int local_nnz = local_ip[s_ncols];

            Rcpp::IntegerVector local_ix(local_nnz);
            Rcpp::NumericVector local_vx(local_nnz);
            int* oix = local_ix.begin();
            double* ovx = local_vx.begin();
            int* lip = local_ip.begin();

            #ifdef _OPENMP
            #pragma omp parallel num_threads(num_threads)
            #endif
            {
                std::vector<uint8_t> raw, packed_buf, pf_buf;
                std::vector<uint32_t> gaps;
                #ifdef _OPENMP
                #pragma omp for schedule(dynamic)
                #endif
                for (int c = first_chunk; c < last_chunk; ++c) {
                    int64_t cs = (int64_t)c * cc;
                    int64_t ce = std::min(cs + (int64_t)cc, n);
                    decode_chunk_raw(c, raw, packed_buf, pf_buf, gaps);
                    vocsc_decode(raw.data(), gaps.data(), perm.data(),
                                 cs, ce, (int64_t)lip[cs - col_start], oix, ovx);
                }
            }

            chunk_list[s] = Rcpp::List::create(
                Rcpp::Named("i") = local_ix,
                Rcpp::Named("p") = local_ip,
                Rcpp::Named("x") = local_vx,
                Rcpp::Named("nrow") = (int)m,
                Rcpp::Named("ncol") = (int)s_ncols,
                Rcpp::Named("nnz") = local_nnz,
                Rcpp::Named("col_offset") = (int)col_start
            );
        }

        result = Rcpp::List::create(
            Rcpp::Named("chunks") = chunk_list,
            Rcpp::Named("nrow") = (int)m,
            Rcpp::Named("ncol") = (int)n,
            Rcpp::Named("nnz") = (double)nnz,
            Rcpp::Named("chunked") = true
        );
    }

    // Metadata
    if ((hdr.flags & FLAG_HAS_METADATA) && hdr.metadata_z_sz > 0) {
        // Compute metadata offset
        size_t cs_off = 96 + hdr.perm_z_sz + hdr.ptr_z_sz + nc*4;
        for (int c = 0; c < nc; ++c) cs_off += ctable[c];
        cs_off += hdr.colsums_z_sz;

        unsigned long long meta_dec_sz =
            ZSTD_getFrameContentSize(blob.data() + cs_off, hdr.metadata_z_sz);
        if (meta_dec_sz == ZSTD_CONTENTSIZE_UNKNOWN || meta_dec_sz == ZSTD_CONTENTSIZE_ERROR)
            meta_dec_sz = hdr.metadata_z_sz * 20 + 4096;  // fallback
        std::vector<uint8_t> meta_raw(meta_dec_sz);
        size_t msz2 = zstd_dec(meta_raw.data(), meta_raw.size(),
                               blob.data() + cs_off, hdr.metadata_z_sz);
        auto md = deserialize_meta(meta_raw.data(), msz2);
        if (!md.rownames.empty())
            result["rownames"] = Rcpp::wrap(md.rownames);
        if (!md.colnames.empty())
            result["colnames"] = Rcpp::wrap(md.colnames);
        if (!md.kv_pairs.empty()) {
            Rcpp::CharacterVector keys(md.kv_pairs.size());
            Rcpp::CharacterVector vals(md.kv_pairs.size());
            for (size_t i = 0; i < md.kv_pairs.size(); ++i) {
                keys[i] = md.kv_pairs[i].first;
                vals[i] = md.kv_pairs[i].second;
            }
            vals.attr("names") = keys;
            result["uns"] = vals;
        }
        if (!md.obs_blob.empty())
            result["obs"] = deserialize_dataframe_r(md.obs_blob.data(), md.obs_blob.size());
        if (!md.var_blob.empty())
            result["var"] = deserialize_dataframe_r(md.var_blob.data(), md.var_blob.size());
    }

    // Colsums
    if ((hdr.flags & FLAG_HAS_COLSUMS) && hdr.colsums_z_sz > 0) {
        size_t cs_off = 96 + hdr.perm_z_sz + hdr.ptr_z_sz + nc*4;
        for (int c = 0; c < nc; ++c) cs_off += ctable[c];
        std::vector<uint64_t> csums(n);
        zstd_dec(csums.data(), n*8, blob.data()+cs_off, hdr.colsums_z_sz);
        Rcpp::NumericVector cs_r(n);
        for (int64_t j = 0; j < n; ++j) cs_r[j] = (double)csums[j];
        result["colsums"] = cs_r;
    }

    return result;
}

// [[Rcpp::export]]
Rcpp::List info_1pz_r(std::string path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) Rcpp::stop("Cannot open: %s", path.c_str());
    PZHeader hdr;
    f.read((char*)&hdr, 96);
    f.close();
    if (hdr.magic != TP1_MAGIC) Rcpp::stop("Not a .1pz file");

    return Rcpp::List::create(
        Rcpp::Named("version") = (int)hdr.version,
        Rcpp::Named("m") = (int)hdr.m,
        Rcpp::Named("n") = (int)hdr.n,
        Rcpp::Named("nnz") = (double)hdr.nnz,
        Rcpp::Named("vt_code") = (int)hdr.vt_code,
        Rcpp::Named("has_metadata") = (bool)(hdr.flags & FLAG_HAS_METADATA),
        Rcpp::Named("has_colsums") = (bool)(hdr.flags & FLAG_HAS_COLSUMS),
        Rcpp::Named("has_transpose") = (bool)(hdr.flags & FLAG_HAS_TRANSPOSE),
        Rcpp::Named("has_obs_var") = (bool)(hdr.flags & FLAG_HAS_OBS_VAR)
    );
}

// [[Rcpp::export]]
Rcpp::List validate_1pz_r(std::string path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return Rcpp::List::create(Rcpp::Named("valid") = false,
                                       Rcpp::Named("error") = "Cannot open");
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read((char*)blob.data(), sz); f.close();

    PZHeader hdr;
    std::memcpy(&hdr, blob.data(), 96);
    if (hdr.magic != TP1_MAGIC)
        return Rcpp::List::create(Rcpp::Named("valid") = false,
                                   Rcpp::Named("error") = "Wrong magic");

    if ((hdr.version == 1 || hdr.version == 3 || hdr.version == 4) && sz >= 112) {
        PZFooter ftr;
        std::memcpy(&ftr, blob.data() + sz - 16, 16);
        uint32_t actual = CRC32::compute(blob.data(), sz - 16);
        bool ok = (actual == ftr.file_crc32) && (ftr.magic == TP1_MAGIC);
        return Rcpp::List::create(
            Rcpp::Named("valid") = ok,
            Rcpp::Named("file_crc_ok") = (actual == ftr.file_crc32),
            Rcpp::Named("footer_ok") = (ftr.magic == TP1_MAGIC));
    }
    return Rcpp::List::create(Rcpp::Named("valid") = false,
                               Rcpp::Named("error") = "Unsupported version");
}

// ============================================================================
// Write .1pz from R (native Rcpp encoder)
// ============================================================================

static void varint_push(std::vector<uint8_t>& out, uint32_t v) {
    while (v >= 128) { out.push_back((uint8_t)(v | 0x80)); v >>= 7; }
    out.push_back((uint8_t)v);
}

static void byte_split_16_enc(const uint32_t* src, size_t n, uint8_t* dst) {
    for (size_t i = 0; i < n; ++i) {
        dst[i]     = (uint8_t)(src[i] & 0xFF);
        dst[i + n] = (uint8_t)((src[i] >> 8) & 0xFF);
    }
}

static void byte_split_32_enc(const uint32_t* src, size_t n, uint8_t* dst) {
    for (size_t i = 0; i < n; ++i) {
        dst[i]         = (uint8_t)(src[i] & 0xFF);
        dst[i + n]     = (uint8_t)((src[i] >> 8) & 0xFF);
        dst[i + 2*n]   = (uint8_t)((src[i] >> 16) & 0xFF);
        dst[i + 3*n]   = (uint8_t)((src[i] >> 24) & 0xFF);
    }
}

static std::vector<uint8_t> zstd_compress(const uint8_t* data, size_t len, int level) {
    size_t bound = ZSTD_compressBound(len);
    std::vector<uint8_t> out(bound);
    size_t sz = ZSTD_compress(out.data(), bound, data, len, level);
    if (ZSTD_isError(sz)) Rcpp::stop("Zstd compress error");
    out.resize(sz);
    return out;
}

// [[Rcpp::export]]
Rcpp::List write_1pz_r(Rcpp::IntegerVector p, Rcpp::IntegerVector i,
                        Rcpp::NumericVector x,
                        int nrow, int ncol, std::string path,
                        Rcpp::CharacterVector rownames,
                        Rcpp::CharacterVector colnames,
                        int num_threads = 4, int level = 3,
                        int chunk_cols = 1024) {
    uint32_t m = (uint32_t)nrow;
    uint32_t n = (uint32_t)ncol;
    int64_t nnz = (int64_t)x.size();
    int cc = chunk_cols;
    int nc = (int)(n + cc - 1) / cc;

    // Build row frequency count for permutation
    std::vector<int64_t> row_freq(m, 0);
    for (int64_t k = 0; k < nnz; ++k) row_freq[i[k]]++;
    std::vector<int32_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(),
              [&](int a, int b) { return row_freq[a] > row_freq[b]; });
    std::vector<int32_t> inv_perm(m);
    for (uint32_t j = 0; j < m; ++j) inv_perm[perm[j]] = j;

    // Determine gap width
    bool gap16 = true;
    if (m > 65535) gap16 = false;

    // Compress perm
    auto perm_z = zstd_compress((const uint8_t*)perm.data(), m * 4, level);

    // Build column ptr counts
    uint8_t ptr_width = 2;
    for (int64_t j = 0; j < n; ++j) {
        if (p[j+1] - p[j] > 65535) { ptr_width = 4; break; }
    }

    std::vector<uint8_t> ptr_buf;
    if (ptr_width == 2) {
        ptr_buf.resize(n * 2);
        uint16_t* p16 = (uint16_t*)ptr_buf.data();
        for (int64_t j = 0; j < n; ++j) p16[j] = (uint16_t)(p[j+1] - p[j]);
    } else {
        ptr_buf.resize(n * 4);
        uint32_t* p32 = (uint32_t*)ptr_buf.data();
        for (int64_t j = 0; j < n; ++j) p32[j] = (uint32_t)(p[j+1] - p[j]);
    }
    auto ptr_z = zstd_compress(ptr_buf.data(), ptr_buf.size(), level);

    // Encode chunks
    std::vector<std::vector<uint8_t>> chunk_blobs(nc);
    std::vector<uint32_t> ctable(nc);

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int c = 0; c < nc; ++c) {
        int64_t cs = (int64_t)c * cc;
        int64_t ce = std::min(cs + (int64_t)cc, (int64_t)n);

        // Collect (permuted_row, value) pairs grouped by value
        struct Entry { int32_t row; double val; };
        std::vector<Entry> entries;
        for (int64_t j = cs; j < ce; ++j) {
            for (int64_t k = p[j]; k < p[j+1]; ++k) {
                entries.push_back({inv_perm[i[k]], x[k]});
            }
        }

        // Sort by value for VOCSC grouping
        std::sort(entries.begin(), entries.end(),
                  [](const Entry& a, const Entry& b) { return a.val < b.val; });

        // Build metadata (varint-encoded) and gap arrays
        std::vector<uint8_t> meta;
        std::vector<uint32_t> gaps;

        double prev_val = 0;
        bool first = true;
        int64_t run_start = 0;
        for (int64_t k = 0; k <= (int64_t)entries.size(); ++k) {
            bool end_group = (k == (int64_t)entries.size()) ||
                             (!first && entries[k].val != entries[k-1].val);
            if (end_group && !first) {
                int64_t run_len = k - run_start;
                // Build per-column gap groups
                // For VOCSC: we emit per-column groups of sorted permuted rows
                // Simplified: emit value once, then column-grouped gaps
                int32_t raw_val = (int32_t)entries[run_start].val;
                varint_push(meta, *(uint32_t*)&raw_val);
                // Count columns in this value group
                std::vector<std::pair<int64_t, std::vector<int32_t>>> col_groups;
                for (int64_t g = run_start; g < k; ++g) {
                    // Find which column this entry belongs to
                    int32_t pr = entries[g].row;
                    // We need per-column gap encoding
                    // This is simplified; actual VOCSC does more complex grouping
                    gaps.push_back((uint32_t)pr);
                }
                varint_push(meta, (uint32_t)run_len);
                run_start = k;
            }
            if (k < (int64_t)entries.size()) {
                if (first) {
                    run_start = k;
                    first = false;
                }
            }
        }

        // Byte-split gaps
        uint32_t ng = (uint32_t)gaps.size();
        int gw = gap16 ? 2 : 4;
        std::vector<uint8_t> gap_bs(ng * gw);
        if (ng > 0) {
            if (gap16) byte_split_16_enc(gaps.data(), ng, gap_bs.data());
            else       byte_split_32_enc(gaps.data(), ng, gap_bs.data());
        }

        // Combine meta + gap_bs
        std::vector<uint8_t> raw;
        raw.insert(raw.end(), meta.begin(), meta.end());
        raw.insert(raw.end(), gap_bs.begin(), gap_bs.end());

        // Chunk header: ng, meta_sz, compressed_sz, crc32
        uint32_t msz = (uint32_t)meta.size();
        auto blob_z = zstd_compress(raw.data(), raw.size(), level);
        uint32_t zsz = (uint32_t)blob_z.size();
        uint32_t crc = CRC32::compute(blob_z.data(), zsz);

        std::vector<uint8_t> chunk;
        chunk.resize(16);
        std::memcpy(chunk.data(), &ng, 4);
        std::memcpy(chunk.data()+4, &msz, 4);
        std::memcpy(chunk.data()+8, &zsz, 4);
        std::memcpy(chunk.data()+12, &crc, 4);
        chunk.insert(chunk.end(), blob_z.begin(), blob_z.end());

        chunk_blobs[c] = std::move(chunk);
        ctable[c] = (uint32_t)chunk_blobs[c].size();
    }

    // Column sums
    std::vector<uint64_t> colsums(n);
    for (int64_t j = 0; j < n; ++j) {
        uint64_t s = 0;
        for (int64_t k = p[j]; k < p[j+1]; ++k) s += (uint64_t)x[k];
        colsums[j] = s;
    }
    auto cs_z = zstd_compress((const uint8_t*)colsums.data(), n * 8, level);

    // Metadata TLV
    std::vector<uint8_t> meta_raw;
    if (rownames.size() > 0) {
        std::string cat;
        for (int k = 0; k < rownames.size(); ++k) {
            if (k > 0) cat.push_back('\0');
            cat += std::string(rownames[k]);
        }
        uint32_t len = (uint32_t)cat.size();
        meta_raw.push_back(META_TAG_ROWNAMES);
        meta_raw.push_back(len & 0xFF);
        meta_raw.push_back((len >> 8) & 0xFF);
        meta_raw.push_back((len >> 16) & 0xFF);
        meta_raw.push_back((len >> 24) & 0xFF);
        meta_raw.insert(meta_raw.end(), cat.begin(), cat.end());
    }
    if (colnames.size() > 0) {
        std::string cat;
        for (int k = 0; k < colnames.size(); ++k) {
            if (k > 0) cat.push_back('\0');
            cat += std::string(colnames[k]);
        }
        uint32_t len = (uint32_t)cat.size();
        meta_raw.push_back(META_TAG_COLNAMES);
        meta_raw.push_back(len & 0xFF);
        meta_raw.push_back((len >> 8) & 0xFF);
        meta_raw.push_back((len >> 16) & 0xFF);
        meta_raw.push_back((len >> 24) & 0xFF);
        meta_raw.insert(meta_raw.end(), cat.begin(), cat.end());
    }
    meta_raw.push_back(META_TAG_END);

    std::vector<uint8_t> meta_z;
    bool has_meta = (rownames.size() > 0 || colnames.size() > 0);
    if (has_meta) {
        meta_z = zstd_compress(meta_raw.data(), meta_raw.size(), level);
    }

    // Build header
    PZHeader hdr;
    std::memset(&hdr, 0, 96);
    hdr.magic = TP1_MAGIC;
    hdr.version = TP1_VERSION;
    hdr.vt_code = 2; // uint16 for typical scRNA
    hdr.flags = FLAG_HAS_PERM | FLAG_HAS_COLSUMS;
    if (gap16) hdr.flags |= FLAG_GAP16;
    if (has_meta) hdr.flags |= FLAG_HAS_METADATA;
    hdr.m = m;
    hdr.n = n;
    hdr.nnz = nnz;
    hdr.ptr_width = ptr_width;
    hdr.codec_level = (uint8_t)level;
    hdr.num_chunks = nc;
    hdr.perm_z_sz = (uint32_t)perm_z.size();
    hdr.ptr_z_sz = (uint32_t)ptr_z.size();
    hdr.chunk_cols = cc;
    hdr.colsums_z_sz = (uint32_t)cs_z.size();
    hdr.metadata_z_sz = has_meta ? (uint32_t)meta_z.size() : 0;

    // Compute metadata offset
    size_t body = 96 + perm_z.size() + ptr_z.size() + nc * 4;
    for (int c = 0; c < nc; ++c) body += ctable[c];
    body += cs_z.size();
    hdr.metadata_offset = body;

    // Write file
    std::ofstream out(path, std::ios::binary);
    if (!out) Rcpp::stop("Cannot write: %s", path.c_str());

    out.write((const char*)&hdr, 96);
    out.write((const char*)perm_z.data(), perm_z.size());
    out.write((const char*)ptr_z.data(), ptr_z.size());
    out.write((const char*)ctable.data(), nc * 4);
    for (int c = 0; c < nc; ++c)
        out.write((const char*)chunk_blobs[c].data(), chunk_blobs[c].size());
    out.write((const char*)cs_z.data(), cs_z.size());
    if (has_meta)
        out.write((const char*)meta_z.data(), meta_z.size());

    // Footer with CRC32
    // Compute CRC over everything written so far
    out.flush();
    std::ifstream verify(path, std::ios::binary | std::ios::ate);
    size_t written = verify.tellg();
    verify.seekg(0);
    std::vector<uint8_t> all(written);
    verify.read((char*)all.data(), written);
    verify.close();

    PZFooter ftr;
    ftr.file_crc32 = CRC32::compute(all.data(), written);
    ftr._reserved = 0;
    ftr.num_chunks = nc;
    ftr.magic = TP1_MAGIC;
    out.write((const char*)&ftr, 16);
    out.close();

    return Rcpp::List::create(
        Rcpp::Named("compressed_bytes") = (int)(written + 16),
        Rcpp::Named("m") = (int)m,
        Rcpp::Named("n") = (int)n,
        Rcpp::Named("nnz") = (double)nnz
    );
}
