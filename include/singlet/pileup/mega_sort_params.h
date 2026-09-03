// SPDX-License-Identifier: MIT
// mega_sort_params.h — STAR BAM-sort parameter tiers for large samples
//
// Pure, header-only helpers that decide:
//   (1) --outBAMcompression level (0 / 1 / 6)
//   (2) --limitBAMsortRAM cap as a fraction of SLURM_MEM
//
// Three tiers, selected by a (n_reads, use_complex) pair:
//
//                           compression   ram_cap_frac          sort_bins
//   normal  (<=200M)              0          3/4 (75%)           default (50)
//   mega    (>200M, <500M)        1          1/2 (50%)           default (50)      ← GSM3743501 408M SUCCESS
//   ultra   (>=500M)              0          15% + hard cap 64 GiB   100           ← AUTOFIX-MEGA-SORT-RSS-OVERAGE fix
//   complex (CB_UMI_Complex)      1          15% + hard cap 64 GiB   default       ← BD Rhapsody, any n_reads
//
// Rationale:
//   At ~500M+ reads, STAR's limitBAMsortRAM is a *soft hint* — the 50%/comp=1
//   regime observed RSS hitting the full 384G cgroup cap and SIGKILL during
//   sort (GSM7102845 666M, GSM5239644 817M). Cutting the cap to 15% WITH a
//   hard ceiling of 64 GiB and raising sort bins to 100 (singlet.cpp)
//   shrinks per-bin allocations during the merge phase. At 384G:
//   limitBAMsortRAM ≤ 57 GiB (15%) leaving ≥327 GiB for genome + thread
//   buffers. At any allocation with 15% > 64 GiB the hard cap takes over,
//   trading minimal sort throughput for OOM safety.
//
//   Iteration #2 (after GSM7102845 666M validator run, 2026-04-16):
//     Memory cap worked (MaxRSS well under 384G). However STAR crashed at the
//     very start of bin-merge with:
//       "number of bytes expected from the BAM bin does not agree with the
//        actual size on disk; bin number=494"
//     This is a known STAR pathology at very high bin counts (500 was
//     aggressive; default is 50). Dropping ULTRA_SORT_BINS from 500 → 100 is
//     still 2× default (smaller per-bin buffers) but well inside STAR's
//     tested range. Compression stays at the actually-passed value (0) — the
//     old log line printed the intended "6" from compression_level() while
//     the arg cascade wrote "0"; log now reflects the arg that STAR receives.
//
//   The 200-500M band is NOT regressed: GSM3743501 (408M) ran SUCCESS at 50%
//   / comp=1 so that tier stays intact.
//
//   If use_complex (BD Rhapsody CB_UMI_Complex) is set, we force comp=1 + 15%
//   + 64 GiB hard cap regardless of read count; the 97^3 combinatorial match
//   tables plus BAM sort OOM at 50% even on 30M samples.

#pragma once

#include <cstdint>

namespace singlet::pileup {
namespace mega_sort {

// Thresholds (exposed so tests can sanity-check them without hard-coding).
constexpr uint64_t MEGA_READ_THRESHOLD  = 200'000'000ULL;  // >  this  => comp=1, 50% cap
constexpr uint64_t ULTRA_READ_THRESHOLD = 500'000'000ULL;  // >= this  => 15% + 64 GiB hard cap + bins=100

// ── B-G3-3/4: SLURM allocation tier → XL tier read threshold ──────────────
// Clipper-class HPC memory tiers (SLURM --mem):
//
//   Standard  64 GB   n_reads ≤  50M   limitBAMsortRAM = 12 GB  (~19%)
//   Barnyard  96 GB   barnyard jobs    limitBAMsortRAM = 20 GB  (~21%)
//   Large    128 GB   n_reads ≤ 200M   limitBAMsortRAM = 50 GB  (~39%)
//   XL-192   192 GB   n_reads > 200M   limitBAMsortRAM = 75 GB  (~39%)
//   XL-384   384 GB   n_reads > 200M   limitBAMsortRAM = 150 GB (~39%)
//
// These are *floor-pinned* values: the MEGA/ULTRA fraction caps above may
// reduce limitBAMsortRAM further for large samples where RSS would otherwise
// exceed the cgroup cap.  The tier table provides the deterministic starting
// point; mega_sort::ram_cap_bytes() applies the final safety ceiling.
constexpr uint64_t XL_READ_THRESHOLD   = 200'000'000ULL;  // > this   => XL tier required

// Deterministic limitBAMsortRAM given a known SLURM tier allocation (bytes).
// Returns 0 if slurm_mem_bytes does not match a recognised tier (caller should
// fall back to the fraction-based ram_cap_bytes() heuristic).
inline uint64_t slurm_tier_bamsort_ram(uint64_t slurm_mem_bytes) {
    // Tolerance: ±1 GiB around each known tier boundary
    constexpr uint64_t GiB = 1024ULL * 1024 * 1024;
    auto near = [&](uint64_t tier_gib) {
        return slurm_mem_bytes >= (tier_gib - 1) * GiB &&
               slurm_mem_bytes <= (tier_gib + 1) * GiB;
    };
    if (near(64))  return 25ULL * GiB;   // Standard  64 GB → 25 GB
    if (near(128)) return 50ULL * GiB;   // Large    128 GB → 50 GB
    if (near(192)) return 75ULL * GiB;   // XL-192   192 GB → 75 GB
    if (near(384)) return 150ULL * GiB;  // XL-384   384 GB → 150 GB
    return 0;  // not a recognised tier
}

// outBAMsortingBinsN for the ultra tier. STAR's default is 50; we use 100
// here to halve per-bin memory (useful with the tight 15%/64-GiB cap) while
// staying well inside STAR's tested range. Previous value of 500 triggered
// STAR's "bin number=N: size on disk != expected" pathology on GSM7102845.
constexpr int ULTRA_SORT_BINS = 100;

// Pick the --outBAMcompression level for the SortedByCoordinate STAR invocation.
//
// Returns:
//   0  if n_reads >= ULTRA_READ_THRESHOLD (ultra tier; uncompressed sort is
//      fastest, and the tight 15%/64-GiB limitBAMsortRAM cap is what keeps
//      RSS in check — compression is NOT the memory lever here)
//   1  if use_complex is true, OR n_reads > MEGA_READ_THRESHOLD
//   0  otherwise (normal small/mid sample; uncompressed sort is fastest)
inline int compression_level(uint64_t n_reads, bool use_complex) {
    if (n_reads >= ULTRA_READ_THRESHOLD) return 0;
    if (use_complex || n_reads > MEGA_READ_THRESHOLD) return 1;
    return 0;
}

// Pick the --limitBAMsortRAM cap in bytes, given the SLURM node allocation.
//
// slurm_mem_bytes: the total memory the job was allocated (i.e. the cgroup
//                  cap). Caller is responsible for passing the right number;
//                  `mega_sort_ram_cap_bytes` does not read the environment.
//
// Returns:
//   min(slurm_mem_bytes * 15/100, 64 GiB)   if use_complex, OR n_reads >= ULTRA_READ_THRESHOLD
//   slurm_mem_bytes * 1/2                   if n_reads > MEGA_READ_THRESHOLD
//   slurm_mem_bytes * 3/4                   otherwise
//
// The ultra/complex hard cap of 64 GiB prevents OOM regardless of how large
// the SLURM allocation is. Above that threshold, more sort bins (singlet.cpp
// sets --outBAMsortingBinsN=ULTRA_SORT_BINS for ultra) is the correct
// scaling strategy.
//
// Returns 0 if slurm_mem_bytes is 0 (meaning "no SLURM allocation known";
// caller must fall back to its auto-computed value with no cap).
inline uint64_t ram_cap_bytes(uint64_t n_reads,
                              uint64_t slurm_mem_bytes,
                              bool use_complex) {
    if (slurm_mem_bytes == 0) return 0;
    if (use_complex || n_reads >= ULTRA_READ_THRESHOLD) {
        uint64_t cap = (slurm_mem_bytes * 15) / 100;  // 15% of allocation
        // Hard cap: ≥500M reads + 500 sort bins is more memory-efficient above 64 GiB
        constexpr uint64_t HARD_CAP_BYTES = 64ULL * 1024 * 1024 * 1024;
        if (cap > HARD_CAP_BYTES) cap = HARD_CAP_BYTES;
        return cap;
    }
    if (n_reads > MEGA_READ_THRESHOLD) {
        return slurm_mem_bytes / 2;           // 50%
    }
    return (slurm_mem_bytes * 3) / 4;         // 75%
}

// Classifier for diagnostic logging only: which tier did we pick?
enum class Tier { Normal, Mega, Ultra, Complex };

inline Tier tier_for(uint64_t n_reads, bool use_complex) {
    if (use_complex) return Tier::Complex;
    if (n_reads >= ULTRA_READ_THRESHOLD) return Tier::Ultra;
    if (n_reads > MEGA_READ_THRESHOLD) return Tier::Mega;
    return Tier::Normal;
}

inline const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Normal:  return "normal";
        case Tier::Mega:    return "mega";
        case Tier::Ultra:   return "ultra";
        case Tier::Complex: return "complex";
    }
    return "unknown";
}

}  // namespace mega_sort
}  // namespace singlet::pileup
