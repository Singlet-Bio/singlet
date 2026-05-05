// minimizer_index.h — MinimizerSAIndex loader + query interface
//
// Binary format: minimizerIndex.bin
//   • 32-byte MinimizerIndexHeader
//   • nRecords × 16-byte MinimizerRecord, sorted ascending by hash
//
// Built by: singlify genome build-minimizers --genome-dir <path>
//
// Used in Phase 2 by STAR's seed search to narrow SA binary search range
// from log₂(3.2B) ≈ 32 probes to ~10 probes per seed.
//
// Thread-safety: MinimizerSAIndex is read-only after load — safe to query
// concurrently from multiple STAR worker threads.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ─── Binary structures (must match builder in singlify.cpp) ─────────────────

struct MinimizerRecord {        // 16 bytes/entry
    uint64_t hash;              // canonical minimizer hash (2-bit packed k-mer)
    uint32_t SA_lo;             // lower SA bound (inclusive)
    uint32_t SA_hi;             // upper SA bound (inclusive)
};
static_assert(sizeof(MinimizerRecord) == 16, "MinimizerRecord must be 16 bytes");

struct MinimizerIndexHeader {   // 32 bytes
    char     magic[8];          // "MINIIDX\0"
    uint32_t k;                 // k-mer length (21)
    uint32_t w;                 // window size (10)
    uint32_t T;                 // max SA range filter (100)
    uint32_t reserved;
    uint64_t nRecords;
};
static_assert(sizeof(MinimizerIndexHeader) == 32, "Header must be 32 bytes");

// ─── Minimizer computation (matches builder) ─────────────────────────────────

// Compute canonical minimizer of seq[0..k+w-2] (genome 2-bit bytes: A=0,C=1,G=2,T=3).
// Returns 0 if any N (value > 3) is encountered — treated as invalid (same as builder).
inline uint64_t minimizer_compute(const uint8_t* seq, int k, int w) {
    const uint64_t MASK = (1ULL << (2 * k)) - 1;

    uint64_t fwd = 0;
    for (int i = 0; i < k; i++) {
        if (seq[i] > 3) return 0;
        fwd = (fwd << 2) | seq[i];
    }
    uint64_t rc = 0;
    for (int i = k - 1; i >= 0; i--)
        rc = (rc << 2) | (3u - seq[i]);

    uint64_t best = std::min(fwd, rc);

    for (int j = 1; j < w; j++) {
        uint8_t b = seq[j + k - 1];
        if (b > 3) return 0;
        fwd = ((fwd << 2) & MASK) | b;
        rc  = (rc >> 2) | ((uint64_t)(3u - b) << (2 * (k - 1)));
        uint64_t canon = std::min(fwd, rc);
        if (canon < best) best = canon;
    }
    return best;
}

// Hash a single k-mer (canonical min of fwd/revcomp).
// Same 2-bit encoding as genome bytes (A=0,C=1,G=2,T=3).
// Returns 0 if any N base — invalid.
inline uint64_t hash_kmer(const uint8_t* seq, int k) {
    uint64_t fwd = 0, rc = 0;
    for (int i = 0; i < k; i++) {
        if (seq[i] > 3) return 0;
        fwd = (fwd << 2) | seq[i];
        rc  = (rc >> 2) | ((uint64_t)(3u - seq[i]) << (2 * (k - 1)));
    }
    return std::min(fwd, rc);
}

// ─── MinimizerSAIndex ────────────────────────────────────────────────────────

class MinimizerSAIndex {
public:
    struct QueryResult {
        uint32_t sa_lo;
        uint32_t sa_hi;
        bool     found;
    };

    MinimizerSAIndex() = default;

    // Move-only (owns mmap region)
    MinimizerSAIndex(const MinimizerSAIndex&)            = delete;
    MinimizerSAIndex& operator=(const MinimizerSAIndex&) = delete;
    MinimizerSAIndex(MinimizerSAIndex&& o) noexcept
        : mmap_ptr_(o.mmap_ptr_), mmap_size_(o.mmap_size_),
          hdr_(o.hdr_), records_(o.records_), nRecords_(o.nRecords_)
    { o.mmap_ptr_ = nullptr; }

    ~MinimizerSAIndex() {
        if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED)
            munmap(mmap_ptr_, mmap_size_);
    }

    // Load index from minimizerIndex.bin via mmap.
    // Throws std::runtime_error on failure.
    void mmap_load(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("MinimizerSAIndex: cannot open " + path);

        struct stat sb{};
        fstat(fd, &sb);
        mmap_size_ = static_cast<size_t>(sb.st_size);

        mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmap_ptr_ == MAP_FAILED) {
            mmap_ptr_ = nullptr;
            throw std::runtime_error("MinimizerSAIndex: mmap failed for " + path);
        }
        madvise(mmap_ptr_, mmap_size_, MADV_RANDOM);

        if (mmap_size_ < sizeof(MinimizerIndexHeader))
            throw std::runtime_error("MinimizerSAIndex: file too small");

        hdr_ = static_cast<const MinimizerIndexHeader*>(mmap_ptr_);
        if (memcmp(hdr_->magic, "MINIIDX", 7) != 0)
            throw std::runtime_error("MinimizerSAIndex: bad magic in " + path);

        nRecords_ = hdr_->nRecords;
        records_  = reinterpret_cast<const MinimizerRecord*>(
            static_cast<const char*>(mmap_ptr_) + sizeof(MinimizerIndexHeader));

        size_t expected = sizeof(MinimizerIndexHeader) + nRecords_ * sizeof(MinimizerRecord);
        if (mmap_size_ < expected)
            throw std::runtime_error("MinimizerSAIndex: truncated index file");
    }

    bool loaded() const { return records_ != nullptr; }
    int  k()       const { return hdr_ ? (int)hdr_->k : 21; }
    int  w()       const { return hdr_ ? (int)hdr_->w : 10; }
    int  T()       const { return hdr_ ? (int)hdr_->T : 100; }
    uint64_t size() const { return nRecords_; }

    // Binary search for minimizer_hash. O(log n).
    QueryResult query(uint64_t minimizer_hash) const {
        if (!records_ || minimizer_hash == 0)
            return {0, 0, false};

        uint64_t lo = 0, hi = nRecords_;
        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            if (records_[mid].hash < minimizer_hash)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < nRecords_ && records_[lo].hash == minimizer_hash)
            return {records_[lo].SA_lo, records_[lo].SA_hi, true};
        return {0, 0, false};
    }

private:
    void*                         mmap_ptr_ = nullptr;
    size_t                        mmap_size_ = 0;
    const MinimizerIndexHeader*   hdr_       = nullptr;
    const MinimizerRecord*        records_   = nullptr;
    uint64_t                      nRecords_  = 0;
};
