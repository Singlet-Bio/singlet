// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: crispr_guide.h
// N18: CRISPR guide capture counting
//
// Counts guide sequences per cell barcode with UMI deduplication.
// Guide library reads are expected to contain the guide sequence in the
// query sequence of BAM alignments (aligned or unaligned reads whose
// sequence matches a provided guide reference).
//
// CSV format: name,sequence (with or without header row)
//   MyGuide_1,AGTCCGAATTAGGCGTACGA
//   MyGuide_2,TACGGCAATCGGTACGCAAT
//
// API:
//   GuideRef ref;
//   ref.load_csv("guides.csv");           // load guide library
//   int id = ref.match(seq, len);         // -1 if no match
//   GuideCounter ctr(ref.size(), bcs);    // per-barcode × guide counter
//   ctr.count(bc_idx, guide_id, umi, ul); // count with UMI dedup
//   auto& acc = ctr.accumulator();        // SparseAccumulator for CSC export

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sparse_accumulator.h"
#include "umi_dedup.h"

namespace singlet {

// ── GuideEntry ────────────────────────────────────────────────────────────

struct GuideEntry {
    std::string name;     ///< Human-readable guide name
    std::string sequence; ///< Guide sequence (uppercase, no whitespace)
    uint32_t    id;       ///< Zero-based index in the guide library
};

// ── GuideRef ──────────────────────────────────────────────────────────────

/// Loaded guide library with fast exact-match lookup.
/// match() searches for the guide sequence as a substring of the query read,
/// allowing the guide to appear at any offset (handles variable cell-bc + UMI
/// offsets and sequencing primer artefacts).
class GuideRef {
public:
    GuideRef() = default;

    /// Load guide library from a CSV file.
    /// Accepts lines of the form:  name,sequence
    /// A header line containing "name" or "sequence" (case-insensitive) is
    /// skipped automatically.
    /// @return true on success, false if the file could not be opened.
    bool load_csv(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            // Strip Windows CR
            if (!line.empty() && line.back() == '\r') line.pop_back();

            // Skip header: if the first non-empty line looks like a header
            if (first) {
                first = false;
                std::string lo = line;
                for (auto& c : lo) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (lo.find("name") != std::string::npos ||
                    lo.find("sequence") != std::string::npos ||
                    lo.find("guide") != std::string::npos) {
                    continue;
                }
            }

            // Split on first comma
            auto comma = line.find(',');
            if (comma == std::string::npos) continue;

            std::string name = line.substr(0, comma);
            std::string seq  = line.substr(comma + 1);

            // Trim trailing whitespace / quotes
            while (!seq.empty() && (std::isspace((unsigned char)seq.back()) || seq.back() == '"'))
                seq.pop_back();
            while (!name.empty() && (std::isspace((unsigned char)name.back()) || name.back() == '"'))
                name.pop_back();
            while (!name.empty() && (std::isspace((unsigned char)name.front()) || name.front() == '"'))
                name.erase(name.begin());
            while (!seq.empty() && (std::isspace((unsigned char)seq.front()) || seq.front() == '"'))
                seq.erase(seq.begin());

            if (seq.empty()) continue;

            // Uppercase sequence
            for (auto& c : seq)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

            uint32_t id = static_cast<uint32_t>(guides_.size());
            seq_to_id_[seq] = id;
            guides_.push_back({std::move(name), seq, id});
        }
        return !guides_.empty();
    }

    /// Try to match the read sequence (raw pointer, length) against all
    /// loaded guides.  Uses exact substring matching of each guide sequence
    /// within the query sequence.
    /// @return guide id (0-based) on success, -1 if no guide matches.
    int match(const char* read_seq, int read_len) const {
        if (seq_to_id_.empty() || read_seq == nullptr || read_len <= 0) return -1;
        // Build a temporary std::string for substring search.
        // For typical guide lengths (≤30 bp) this is fast.
        std::string rstr(read_seq, static_cast<size_t>(read_len));
        for (auto& c : rstr)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        for (const auto& [seq, id] : seq_to_id_) {
            if (rstr.find(seq) != std::string::npos) return static_cast<int>(id);
        }
        return -1;
    }

    /// Overload accepting std::string.
    int match(const std::string& read_seq) const {
        return match(read_seq.c_str(), static_cast<int>(read_seq.size()));
    }

    const std::vector<GuideEntry>& guides() const { return guides_; }
    size_t size()  const { return guides_.size(); }
    bool   empty() const { return guides_.empty(); }

    /// Names in guide-id order (for matrix feature list).
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(guides_.size());
        for (const auto& g : guides_) out.push_back(g.name);
        return out;
    }

private:
    std::vector<GuideEntry>                  guides_;
    std::unordered_map<std::string, uint32_t> seq_to_id_;
};

// ── GuideCounter ──────────────────────────────────────────────────────────

/// Per-barcode × guide count accumulator with UMI deduplication.
/// Wraps SparseAccumulator<uint16_t> (same type used for exon/intron counts)
/// with an inline UmiDedup for (barcode, guide, UMI) triple deduplication.
class GuideCounter {
public:
    GuideCounter() = default;

    /// Initialise accumulator dimensions.
    void init(size_t n_guides, const std::vector<std::string>& barcodes) {
        acc_.set_barcodes(barcodes);
        acc_.set_n_features(static_cast<uint32_t>(n_guides));
        umi_dedup_.reserve(n_guides * barcodes.size() / 10 + 10000);
    }

    /// Record one observed guide-barcode-UMI triple.
    /// @param bc_idx   Barcode index (from barcode lookup)
    /// @param guide_id Guide id returned by GuideRef::match()
    /// @param umi      Pointer to UMI bases
    /// @param umi_len  Length of UMI (0 = no UMI, always count)
    void count(uint32_t bc_idx, uint32_t guide_id, const char* umi, int umi_len) {
        bool should_count = true;
        if (umi_len > 0 && umi != nullptr) {
            should_count = umi_dedup_.insert(bc_idx, guide_id, umi, umi_len);
        }
        if (should_count) {
            acc_.increment(guide_id, bc_idx);
        }
    }

    SparseAccumulator<uint16_t>&       accumulator()       { return acc_; }
    const SparseAccumulator<uint16_t>& accumulator() const { return acc_; }

    /// Merge another GuideCounter's accumulator into this one (for multi-threaded merge).
    void merge(GuideCounter& other) {
        acc_.merge_from(other.acc_);
    }

private:
    SparseAccumulator<uint16_t> acc_;
    UmiDedup                    umi_dedup_;
};

} // namespace singlet
