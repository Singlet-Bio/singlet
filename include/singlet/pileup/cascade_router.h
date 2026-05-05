// cascade_router.h — Track B cascade: single-pass per-read router
// Part of droplet-hardening Track B prototype (T-L2-1).
//
// Determinism seed: 0xC0FFEE (3,405,691,582).
// Zero heap allocation in hot path (decision loop only touches stack + passed-in refs).
//
// Decision enum:
//   L1_resolve     — unique transcriptome hit; routed to L1 gene/exon counters
//   L2_resolve     — TE family k-mer hit; routed to te_counts.1pz
//   L3_passthrough — must go through STAR whole-genome alignment
//   L4_passthrough — STAR unmapped; goes to non-host cascade (L4)
//
// Usage:
//   CascadeRouter router(txome_aligner, &te_classifier);
//   for each block:
//     for (uint32_t i = 0; i < blk.n_reads; ++i) {
//       auto d = router.route(blk, i);
//       if (d == CascadeDecision::L1_resolve) { ... }
//       ...
//     }

#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace singlet {

// ── Decision enum ──────────────────────────────────────────────────────────────
enum class CascadeDecision : uint8_t {
    L1_resolve    = 0,  // unique transcriptome hit resolved at L1
    L2_resolve    = 1,  // TE/repeat family resolved at L2
    L3_passthrough = 2, // send to STAR whole-genome alignment
    L4_passthrough = 3, // STAR-unmapped; send to non-host cascade
};

// ── Forward declarations of L1 and L2 classifiers (header-only stubs) ──────────
class TxomeAligner;   // include/singlet-pileup/txome_aligner.h
class TeClassifier;   // include/singlet-pileup/te_classifier.h (bio-exec, T-L2-3)

// ── CascadeRouterStats — per-run layer accounting ─────────────────────────────
struct CascadeRouterStats {
    uint64_t n_total      = 0;
    uint64_t n_l1_resolve = 0;
    uint64_t n_l2_resolve = 0;
    uint64_t n_l3_pass    = 0;
    uint64_t n_l4_pass    = 0;

    double l1_fraction() const {
        return n_total ? static_cast<double>(n_l1_resolve) / n_total : 0.0;
    }
    double l2_fraction() const {
        return n_total ? static_cast<double>(n_l2_resolve) / n_total : 0.0;
    }
    double l3_fraction() const {
        return n_total ? static_cast<double>(n_l3_pass) / n_total : 0.0;
    }
};

// ── CascadeReadView — lightweight non-owning view of a single decoded read ─────
// Filled from a DecodedBlock slot by the caller; no allocation.
struct CascadeReadView {
    uint32_t    barcode_id;   // bc_indices[i] from DecodedBlock (sentinel=dict_size if unknown)
    const uint8_t* r1_seq;   // pointer into r1_data (BC+UMI bytes, 2-bit ACGTN)
    uint16_t    r1_len;
    const uint8_t* r2_seq;   // pointer into r2_data (cDNA bytes, 2-bit ACGTN)
    uint16_t    r2_len;
    const uint8_t* r2_qual;  // pointer into r2_qual (phred), may be nullptr
    uint16_t    protocol_id; // from .1fq FileHeader::protocol_id
};

// ── CascadeRouter ──────────────────────────────────────────────────────────────
//
// Hot path: route() touches no heap. All state is in passed-in references.
// The TeClassifier pointer is nullable; when nullptr, L2 layer is disabled.
//
// Thread safety: route() is const — multiple threads may share one CascadeRouter
// instance provided stats are thread-local or externally serialized.
class CascadeRouter {
public:
    static constexpr uint64_t DETERMINISM_SEED = 0xC0FFEEULL; // 3,405,691,582

    // Construct with L1 aligner (required) and optional L2 classifier.
    // te_classifier may be nullptr to disable L2.
    explicit CascadeRouter(const TxomeAligner& txome_aligner,
                           const TeClassifier* te_classifier = nullptr) noexcept
        : txome_aligner_(txome_aligner)
        , te_classifier_(te_classifier)
    {}

    // Main hot-path router.  No allocation; decision is deterministic.
    // stats is updated in-place; pass a thread-local struct for parallel use.
    CascadeDecision route(const CascadeReadView& rv,
                          CascadeRouterStats&    stats) const noexcept;

    // Convenience: route and return decision without updating stats.
    CascadeDecision route(const CascadeReadView& rv) const noexcept {
        CascadeRouterStats dummy;
        return route(rv, dummy);
    }

    bool te_enabled() const noexcept { return te_classifier_ != nullptr; }

private:
    const TxomeAligner&   txome_aligner_;
    const TeClassifier*   te_classifier_;  // nullable
};

} // namespace singlet

// ── Implementation (header-only) ──────────────────────────────────────────────
// Placed after the class declaration to keep declarations readable.
// The TxomeAligner and TeClassifier implementations are in their own headers;
// we include them here so route() can call their methods inline.

#include "txome_aligner.h"

// te_classifier.h is bio-exec (T-L2-3); include only if present, else stub.
#if __has_include("te_classifier.h")
#  include "te_classifier.h"
#else
// Minimal stub so this header compiles standalone before T-L2-3 ships.
namespace singlet {
class TeClassifier {
public:
    struct TeFamilyHit { uint32_t family_id; uint32_t n_kmers_matched; uint32_t n_kmers_total; };
    // Stub: always returns nullopt (no TE DB loaded).
    std::optional<TeFamilyHit> classify(std::string_view /*read*/) const noexcept {
        return std::nullopt;
    }
};
} // namespace singlet
#endif

namespace singlet {

inline CascadeDecision
CascadeRouter::route(const CascadeReadView& rv,
                     CascadeRouterStats&    stats) const noexcept
{
    ++stats.n_total;

    // ── L1: transcriptome-first unique resolve ─────────────────────────────
    // Convert 2-bit encoded R2 to string_view for aligner.
    // The aligner works directly on the 2-bit buffer; the string_view
    // carries (ptr, len) without copying.
    {
        std::string_view r2_sv(reinterpret_cast<const char*>(rv.r2_seq),
                               rv.r2_len);
        auto hit = txome_aligner_.resolve_unique(r2_sv);
        if (hit.has_value()) {
            ++stats.n_l1_resolve;
            return CascadeDecision::L1_resolve;
        }
    }

    // ── L2: TE / repeat pre-classifier ────────────────────────────────────
    if (te_classifier_ != nullptr) {
        // bio-exec API: classify(string_view) → optional<TeFamilyHit>
        std::string_view r2_sv2(reinterpret_cast<const char*>(rv.r2_seq), rv.r2_len);
        if (te_classifier_->classify(r2_sv2).has_value()) {
            ++stats.n_l2_resolve;
            return CascadeDecision::L2_resolve;
        }
    }

    // ── L3: passthrough to STAR whole-genome alignment ────────────────────
    ++stats.n_l3_pass;
    return CascadeDecision::L3_passthrough;

    // L4 is assigned post-STAR by the pileup thread when STAR returns unmapped;
    // it is not a decision this router makes at read-dispatch time.
    (void)CascadeDecision::L4_passthrough; // silence unused-enum warning
}

} // namespace singlet
