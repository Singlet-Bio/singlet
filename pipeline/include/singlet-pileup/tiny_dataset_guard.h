#pragma once
// tiny_dataset_guard.h — Guards against STAR crashes on very small datasets.
// Provides threshold checks and conservative memory recommendations for tiny inputs.
#include <cstdint>
#include <string>

namespace singlet_pileup {

struct TinyDatasetGuard {
    static constexpr uint64_t MIN_READS_FOR_PIPELINE  = 10000; ///< Absolute floor: <10K reads is never a real single-cell run
    static constexpr uint64_t MIN_READS_FOR_ALIGNMENT = 1000;
    static constexpr uint64_t MIN_READS_FOR_PILEUP = 100;
    static constexpr uint64_t MIN_BARCODES_FOR_CALLING = 10;

    // True if dataset is too small for reliable alignment
    static bool is_too_small(uint64_t total_reads) {
        return total_reads < MIN_READS_FOR_ALIGNMENT;
    }

    // Recommended --limitBAMsortRAM value for STAR based on read count.
    // Tiny datasets don't need the default 60 GB; overly large allocations can
    // trigger STAR's internal allocator on systems with low memory limits.
    // For large datasets (≥10M reads) return UINT64_MAX so the G-TINY guard
    // never overrides the MEGA formula, which already computes the correct value.
    static uint64_t recommended_bam_sort_ram(uint64_t total_reads) {
        if (total_reads < 1'000'000ULL) return 1ULL << 30;   // 1 GB  for < 1M reads
        if (total_reads < 10'000'000ULL) return 4ULL << 30;  // 4 GB  for < 10M reads
        return UINT64_MAX;  // No cap — MEGA formula handles large datasets
    }

    // True if there are enough unique barcodes for meaningful cell calling
    static bool enough_for_cell_calling(uint64_t unique_barcodes) {
        return unique_barcodes >= MIN_BARCODES_FOR_CALLING;
    }

    // Human-readable warning for a dataset too small to process reliably.
    // Returns empty string if the dataset size is acceptable.
    static std::string warning_message(uint64_t total_reads) {
        if (total_reads < MIN_READS_FOR_PILEUP) {
            return "Dataset has fewer than " +
                   std::to_string(MIN_READS_FOR_PILEUP) +
                   " reads — too small for reliable quantification";
        }
        if (total_reads < MIN_READS_FOR_ALIGNMENT) {
            return "Dataset has fewer than " +
                   std::to_string(MIN_READS_FOR_ALIGNMENT) +
                   " reads — results may be unreliable";
        }
        return "";
    }
};

}  // namespace singlet_pileup
