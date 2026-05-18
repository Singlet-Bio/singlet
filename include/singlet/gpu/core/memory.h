// SPDX-License-Identifier: MIT
// singlet/gpu/core/memory.h
//
// PinnedPool  — RAII pinned host memory via cudaMallocHost / cudaFreeHost.
// Metadata    — POD struct mirroring the GEO key-value set embedded in every
//               .1pz header (plus optional rownames / colnames).
//
// WHY pinned memory for staging: cudaMemcpyAsync requires the source buffer to
// remain valid and page-locked until the stream reaches the copy operation.
// Pageable host allocations silently force a synchronous copy inside the driver.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace core {

// ---------------------------------------------------------------------------
// PinnedBuffer — a non-copyable, movable RAII handle to a cudaMallocHost block.
//
// Returned by PinnedPool::acquire(). Releases on destruction.
// ---------------------------------------------------------------------------
class PinnedBuffer {
public:
    PinnedBuffer() noexcept : ptr_(nullptr), bytes_(0) {}

    PinnedBuffer(void* ptr, std::size_t bytes) noexcept
        : ptr_(ptr), bytes_(bytes) {}

    ~PinnedBuffer() {
        if (ptr_) {
            // Ignore CUDA errors during teardown — nothing useful we can do.
            cudaFreeHost(ptr_);
        }
    }

    // Movable only — pinned buffers are not copyable.
    PinnedBuffer(PinnedBuffer&& o) noexcept
        : ptr_(o.ptr_), bytes_(o.bytes_)
    { o.ptr_ = nullptr; o.bytes_ = 0; }

    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_   = o.ptr_;
            bytes_ = o.bytes_;
            o.ptr_ = nullptr;
            o.bytes_ = 0;
        }
        return *this;
    }

    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    void*       get()   const noexcept { return ptr_; }
    std::size_t bytes() const noexcept { return bytes_; }

    // Typed view — caller is responsible for correct T and size.
    template<typename T>
    T* as() const noexcept { return static_cast<T*>(ptr_); }

private:
    void*       ptr_;
    std::size_t bytes_;
};

// ---------------------------------------------------------------------------
// PinnedPool — stateless factory. "Pool" in name only: we simply wrap
// cudaMallocHost. A real pool (suballocator with a freelist) can be swapped
// in later without changing the call sites.
// ---------------------------------------------------------------------------
struct PinnedPool {
    // Allocate bytes of page-locked host memory.
    // Throws std::runtime_error on CUDA allocation failure.
    static PinnedBuffer acquire(std::size_t bytes) {
        if (bytes == 0) return PinnedBuffer{};
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("PinnedPool::acquire cudaMallocHost failed: ")
                + cudaGetErrorString(err));
        }
        return PinnedBuffer{ptr, bytes};
    }
};

// ---------------------------------------------------------------------------
// Metadata — GEO key-value fields embedded in every .1pz file plus optional
// row / column name vectors.
//
// Fields mirror the JSON schema from CLAUDE.md §GEO Metadata Embedding and
// style-rules.md §B. All strings are UTF-8. Fields not present in the .1pz
// metadata TLV remain empty / zero-initialized.
// ---------------------------------------------------------------------------
struct Metadata {
    // GEO identifiers
    std::string gsm_id;
    std::string gse_id;

    // Organism / taxonomy
    std::string organism;
    int32_t     taxon_id = 0;

    // Assay description
    std::string protocol;   // e.g. "10xv3", "Drop-seq"
    std::string modality;   // e.g. "scrna", "cite", "multiome"

    // Run accessions (serialized as comma-joined string from user_kv)
    std::vector<std::string> srr_ids;

    // Approximate read count from the catalog
    int64_t read_count = 0;

    // GEO textual fields
    std::string geo_title;
    std::string geo_source_name;

    // Pipeline provenance
    std::string singlet_version;
    std::string pipeline_date;

    // Row / column names from the .1pz metadata TLV (optional).
    // rows == feature names (genes), cols == barcode strings.
    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
};

}  // namespace core
}  // namespace singlet::gpu
