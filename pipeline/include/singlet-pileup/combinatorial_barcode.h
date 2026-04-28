#pragma once
// combinatorial_barcode.h — Multi-level combinatorial barcode decoder
// for split-pool protocols: SHARE-seq, PAIRED-seq, scifi-RNA-seq
//
// SHARE-seq (Ma et al. 2020, Cell):
//   R1: [BC1:8bp][Linker1:30bp][BC2:8bp][Linker2:30bp][BC3:8bp][UMI:10bp]
//   Composite cell barcode = BC1 + BC2 + BC3 (24bp)

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace singlet {

// ── Data structures ──────────────────────────────────────────────────────────

struct BarcodeLevel {
    uint32_t offset;         // position of barcode in R1
    uint32_t length;         // barcode segment length (bp)
    std::string linker;      // expected linker sequence after this barcode (empty for last level)
    uint32_t linker_offset;  // position of linker in R1
};

// ── Decoder ──────────────────────────────────────────────────────────────────

struct CombinatorialBarcodeDecoder {
    std::vector<BarcodeLevel> levels;
    uint32_t umi_offset = 0;
    uint32_t umi_length = 0;
    uint32_t max_linker_mismatches = 1;  // Hamming tolerance for linker matching

    struct DecodeResult {
        std::string barcode;  // concatenated composite barcode (all levels)
        std::string umi;
        bool valid = false;         // all linkers matched within tolerance
        int linker_mismatches = 0;  // total mismatches across all linkers
    };

    // Decode composite barcode from an R1 read sequence.
    // Returns valid=false if any linker mismatches exceed tolerance.
    DecodeResult decode(const std::string& r1_seq) const {
        DecodeResult result;
        result.barcode.reserve(total_barcode_length());

        int total_mm = 0;

        for (const auto& lvl : levels) {
            // Extract barcode segment
            if (lvl.offset + lvl.length > r1_seq.size()) return result;
            result.barcode += r1_seq.substr(lvl.offset, lvl.length);

            // Check linker (if present)
            if (!lvl.linker.empty()) {
                uint32_t llen = static_cast<uint32_t>(lvl.linker.size());
                if (lvl.linker_offset + llen > r1_seq.size()) return result;
                int mm = hamming(r1_seq, lvl.linker_offset, lvl.linker);
                if (mm > static_cast<int>(max_linker_mismatches)) return result;
                total_mm += mm;
            }
        }

        // Extract UMI
        if (umi_length > 0) {
            if (umi_offset + umi_length > r1_seq.size()) return result;
            result.umi = r1_seq.substr(umi_offset, umi_length);
        }

        result.linker_mismatches = total_mm;
        result.valid = true;
        return result;
    }

    // ── Factory methods for known protocols ──────────────────────────────────

    // SHARE-seq (Ma et al. 2020, Cell 183:3428)
    // R1 layout:
    //   [0..7]   BC1 (8bp)
    //   [8..37]  Linker1 (30bp) — first 22bp used for matching
    //   [38..45] BC2 (8bp)
    //   [46..75] Linker2 (30bp) — first 22bp used for matching
    //   [76..83] BC3 (8bp)
    //   [84..93] UMI (10bp)
    static CombinatorialBarcodeDecoder share_seq() {
        CombinatorialBarcodeDecoder dec;
        dec.umi_offset = 84;
        dec.umi_length = 10;
        dec.max_linker_mismatches = 1;

        // Level 1: BC1 at [0], Linker1 at [8]
        BarcodeLevel l1;
        l1.offset = 0;
        l1.length = 8;
        l1.linker = "ATCCACGTGCTTGAGACTGTGG";  // 22bp of Linker1
        l1.linker_offset = 8;
        dec.levels.push_back(l1);

        // Level 2: BC2 at [38], Linker2 at [46]
        BarcodeLevel l2;
        l2.offset = 38;
        l2.length = 8;
        l2.linker = "GTGGCCGATGTTTCGCATCGGC";  // 22bp of Linker2
        l2.linker_offset = 46;
        dec.levels.push_back(l2);

        // Level 3: BC3 at [76], no linker
        BarcodeLevel l3;
        l3.offset = 76;
        l3.length = 8;
        l3.linker = "";
        l3.linker_offset = 0;
        dec.levels.push_back(l3);

        return dec;
    }

    // PAIRED-seq (Zhu et al. 2019, Nature Methods)
    // Similar 3-level structure with slightly different layout:
    //   [0..7]   BC1 (8bp)
    //   [8..29]  Linker1 (22bp)
    //   [30..37] BC2 (8bp)
    //   [38..59] Linker2 (22bp)
    //   [60..67] BC3 (8bp)
    //   [68..77] UMI (10bp)
    static CombinatorialBarcodeDecoder paired_seq() {
        CombinatorialBarcodeDecoder dec;
        dec.umi_offset = 68;
        dec.umi_length = 10;
        dec.max_linker_mismatches = 1;

        BarcodeLevel l1;
        l1.offset = 0;
        l1.length = 8;
        l1.linker = "CATGAATCGGTACAGTTCGCTC";  // PAIRED-seq Linker1 (22bp)
        l1.linker_offset = 8;
        dec.levels.push_back(l1);

        BarcodeLevel l2;
        l2.offset = 30;
        l2.length = 8;
        l2.linker = "GGTCATCAGCATCGTCAAGCTA";  // PAIRED-seq Linker2 (22bp)
        l2.linker_offset = 38;
        dec.levels.push_back(l2);

        BarcodeLevel l3;
        l3.offset = 60;
        l3.length = 8;
        l3.linker = "";
        l3.linker_offset = 0;
        dec.levels.push_back(l3);

        return dec;
    }

    // scifi-RNA-seq (Datlinger et al. 2021, Nature Methods)
    // 2-level combinatorial indexing:
    //   [0..7]   BC1 (8bp)
    //   [8..27]  Linker (20bp)
    //   [28..35] BC2 (8bp)
    //   [36..43] UMI (8bp)
    static CombinatorialBarcodeDecoder scifi_seq() {
        CombinatorialBarcodeDecoder dec;
        dec.umi_offset = 36;
        dec.umi_length = 8;
        dec.max_linker_mismatches = 1;

        BarcodeLevel l1;
        l1.offset = 0;
        l1.length = 8;
        l1.linker = "TCTTCAGCGTTCCCGAGA";  // scifi-RNA-seq linker (18bp)
        l1.linker_offset = 8;
        dec.levels.push_back(l1);

        BarcodeLevel l2;
        l2.offset = 28;
        l2.length = 8;
        l2.linker = "";
        l2.linker_offset = 0;
        dec.levels.push_back(l2);

        return dec;
    }

   private:
    // Compute Hamming distance between r1_seq[start..] and ref_seq
    static int hamming(const std::string& r1_seq, uint32_t start,
                       const std::string& ref_seq) {
        int mm = 0;
        for (uint32_t i = 0; i < ref_seq.size(); ++i) {
            if (r1_seq[start + i] != ref_seq[i]) ++mm;
        }
        return mm;
    }

    uint32_t total_barcode_length() const {
        uint32_t n = 0;
        for (const auto& lvl : levels) n += lvl.length;
        return n;
    }
};

}  // namespace singlet
