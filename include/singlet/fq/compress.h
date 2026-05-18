// SPDX-License-Identifier: MIT
// lib1fq/compress.h — Codec-agnostic compression for .1fq blocks
//
// Wraps zstd (required) and lz4/lz4hc (optional, via HAS_LZ4 define).
// Each block is independently compressed for random access.

#pragma once

#include "types.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <zstd.h>

#ifdef HAS_LZ4
#include <lz4.h>
#include <lz4hc.h>
#endif

namespace singlet::fq {
namespace compress {

// Compress `src` into `dst` using the specified codec.
// Returns compressed size. Throws on error.
// dst must be pre-allocated to at least compress_bound(codec, src_size).

inline size_t compress_bound(Codec codec, size_t src_size) {
    switch (codec) {
        case Codec::ZSTD:
            return ZSTD_compressBound(src_size);
#ifdef HAS_LZ4
        case Codec::LZ4:
        case Codec::LZ4HC:
            return static_cast<size_t>(LZ4_compressBound(static_cast<int>(src_size)));
#endif
        case Codec::NONE:
            return src_size;
        default:
            throw std::runtime_error("Unsupported codec: " +
                std::to_string(static_cast<int>(codec)));
    }
}

inline size_t compress_block(Codec codec, int level,
                             const void* src, size_t src_size,
                             void* dst, size_t dst_capacity) {
    size_t result;
    switch (codec) {
        case Codec::ZSTD:
            result = ZSTD_compress(dst, dst_capacity, src, src_size, level);
            if (ZSTD_isError(result)) {
                throw std::runtime_error(std::string("ZSTD compress: ") +
                    ZSTD_getErrorName(result));
            }
            return result;

#ifdef HAS_LZ4
        case Codec::LZ4: {
            int r = LZ4_compress_default(
                static_cast<const char*>(src), static_cast<char*>(dst),
                static_cast<int>(src_size), static_cast<int>(dst_capacity));
            if (r <= 0) throw std::runtime_error("LZ4 compress failed");
            return static_cast<size_t>(r);
        }

        case Codec::LZ4HC: {
            int r = LZ4_compress_HC(
                static_cast<const char*>(src), static_cast<char*>(dst),
                static_cast<int>(src_size), static_cast<int>(dst_capacity), level);
            if (r <= 0) throw std::runtime_error("LZ4HC compress failed");
            return static_cast<size_t>(r);
        }
#endif

        case Codec::NONE:
            std::memcpy(dst, src, src_size);
            return src_size;

        default:
            throw std::runtime_error("Unsupported codec for compression");
    }
}

// Compress using a persistent ZSTD_CCtx (carries LZ77 hash table warm across blocks).
// For ZSTD, uses the provided context with ZSTD_compress2 (respects CCtx params
// like nbWorkers for multi-threaded compression). For other codecs, falls back.
// n_workers: if > 0, enable multi-threaded compression (set on each call to
// ensure the parameter persists across ZSTD_compress2 state resets).
inline size_t compress_block_ctx(Codec codec, int level,
                                 ZSTD_CCtx* cctx,
                                 const void* src, size_t src_size,
                                 void* dst, size_t dst_capacity,
                                 int n_workers = 0) {
    if (codec == Codec::ZSTD && cctx) {
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
        if (n_workers > 0) {
            ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, n_workers);
        }
        size_t result = ZSTD_compress2(cctx, dst, dst_capacity, src, src_size);
        if (ZSTD_isError(result)) {
            throw std::runtime_error(std::string("ZSTD compress2: ") +
                ZSTD_getErrorName(result));
        }
        return result;
    }
    return compress_block(codec, level, src, src_size, dst, dst_capacity);
}

// Decompress `src` into `dst`.
// Returns decompressed size. dst must hold the full uncompressed payload.

inline size_t decompress_block(Codec codec,
                               const void* src, size_t src_size,
                               void* dst, size_t dst_capacity) {
    size_t result;
    switch (codec) {
        case Codec::ZSTD:
            result = ZSTD_decompress(dst, dst_capacity, src, src_size);
            if (ZSTD_isError(result)) {
                throw std::runtime_error(std::string("ZSTD decompress: ") +
                    ZSTD_getErrorName(result));
            }
            return result;

#ifdef HAS_LZ4
        case Codec::LZ4: 
        case Codec::LZ4HC: {
            int r = LZ4_decompress_safe(
                static_cast<const char*>(src), static_cast<char*>(dst),
                static_cast<int>(src_size), static_cast<int>(dst_capacity));
            if (r < 0) throw std::runtime_error("LZ4 decompress failed");
            return static_cast<size_t>(r);
        }
#endif

        case Codec::NONE:
            std::memcpy(dst, src, src_size);
            return src_size;

        default:
            throw std::runtime_error("Unsupported codec for decompression");
    }
}

} // namespace compress
} // namespace singlet::fq
