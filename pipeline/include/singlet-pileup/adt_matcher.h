#pragma once
// singlet-pileup: adt_matcher.h
// T1: CITE-seq / ADT antibody barcode tag matching
//
// Loads a tag reference (CSV/TSV: name,sequence) and builds a Hamming-1
// neighbourhood hash for O(1) per-read matching.
//
// Tag sequences are typically 15 bp (TotalSeq-A/B/C) but any length 10–30 bp
// is accepted.  The index matches the first `tag_len` bases of the read at
// offsets 0..max_offset (default 5) to tolerate soft-clip and adapter jitter.
//
// Thread-safety: after build_index() the object is read-only; multiple
// threads may call match() simultaneously without synchronisation.
//
// CSV/TSV format (header line optional):
//   name,sequence      or     name\tsequence
//   CD4,AACAAGACCCCTCCT
//   CD8a,TGCATGTCATCGGTGG
//   ...
//
// Return codes from match():
//   ≥ 0   tag index (0-based)
//   -1    no match (sequence not within Hamming distance 1 of any tag)
//   -2    ambiguous (sequence is Hamming-1 neighbour of two different tags)

#include <atomic>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

// ── AdtTag ────────────────────────────────────────────────────────────────

struct AdtTag {
    std::string name;      ///< Human-readable antibody name
    std::string sequence;  ///< Tag sequence (uppercase A/C/G/T/N)
    int id;                ///< Zero-based index in the loaded reference
};

// ── AdtMatcher ────────────────────────────────────────────────────────────

class AdtMatcher {
   public:
    /// Maximum offset from the start of R2 to try when scanning for a tag.
    static constexpr int kDefaultMaxOffset = 5;

    AdtMatcher() = default;

    // ── Reference loading ─────────────────────────────────────────────────

    /// Load a tag reference from a CSV or TSV file.
    /// Both comma and tab delimiters are accepted; the delimiter is
    /// auto-detected from the first non-header data line.
    /// Returns true on success, false if the file could not be opened or
    /// contained no valid tag entries.
    bool load_reference(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string line;
        bool first = true;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            // Auto-skip header line
            if (first) {
                first = false;
                std::string lo = line;
                for (auto& c : lo)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                if (lo.find("name") != std::string::npos ||
                    lo.find("sequence") != std::string::npos ||
                    lo.find("tag") != std::string::npos) {
                    continue;
                }
            }

            // Detect delimiter: prefer tab, fall back to comma
            char delim = (line.find('\t') != std::string::npos) ? '\t' : ',';
            auto sep = line.find(delim);
            if (sep == std::string::npos) continue;

            std::string name = trim_(line.substr(0, sep));
            std::string seq = trim_(line.substr(sep + 1));
            if (name.empty() || seq.empty()) continue;

            // Uppercase
            for (auto& c : seq)
                c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

            add_tag(std::move(name), std::move(seq));
        }
        return !tags_.empty();
    }

    /// Add a single tag programmatically (before build_index()).
    void add_tag(const std::string& name, const std::string& sequence) {
        std::string seq = sequence;
        for (auto& c : seq)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        int id = static_cast<int>(tags_.size());
        tags_.push_back({name, seq, id});
    }

    /// Overload accepting rvalue strings.
    void add_tag(std::string&& name, std::string&& seq) {
        int id = static_cast<int>(tags_.size());
        tags_.push_back({std::move(name), std::move(seq), id});
    }

    // ── Index construction ────────────────────────────────────────────────

    /// Build the Hamming-1 lookup hash.  Must be called after all tags are
    /// loaded and before any call to match().
    ///
    /// The exact-match hash maps each tag sequence → its id.
    /// The Hamming-1 hash maps every 1-substitution neighbour → tag id, or
    /// kAmbiguous if two different tags share that neighbour (collision).
    ///
    /// Only neighbours that are NOT exact matches of another tag are stored
    /// in the Hamming-1 bucket; exact matches always win and are resolved
    /// first in match().
    void build_index() {
        exact_.clear();
        hamming1_.clear();
        tag_len_ = 0;

        if (tags_.empty()) return;

        // Use the tag length of the first entry; warn if lengths vary.
        tag_len_ = static_cast<int>(tags_[0].sequence.size());

        // Build exact hash
        exact_.reserve(tags_.size() * 2);
        for (const auto& t : tags_) {
            exact_[t.sequence] = t.id;
        }

        // Build Hamming-1 hash
        // For each tag, generate all L*3 neighbours (L = tag_len, 3 substitutions per position)
        const char BASES[] = {'A', 'C', 'G', 'T'};
        hamming1_.reserve(tags_.size() * static_cast<size_t>(tag_len_) * 3 * 2);

        for (const auto& t : tags_) {
            std::string nbr = t.sequence;
            for (int pos = 0; pos < static_cast<int>(nbr.size()); ++pos) {
                char orig = nbr[pos];
                for (char b : BASES) {
                    if (b == orig) continue;
                    nbr[pos] = b;
                    // Skip if this neighbour is an exact match of any tag
                    // (exact_ lookup will handle it; let exact take priority)
                    if (exact_.count(nbr)) {
                        nbr[pos] = orig;
                        continue;
                    }
                    auto it = hamming1_.find(nbr);
                    if (it == hamming1_.end()) {
                        hamming1_[nbr] = t.id;
                    } else if (it->second != t.id) {
                        // Collision: two different tags share this neighbour
                        it->second = kAmbiguous;
                    }
                    // else: same tag → already stored, no-op
                    nbr[pos] = orig;
                }
            }
        }
    }

    // ── Matching ──────────────────────────────────────────────────────────

    /// Match an ADT read sequence against the loaded tag reference.
    ///
    /// Scans offsets 0..max_offset (default kDefaultMaxOffset) from the
    /// start of the read, extracting a window of tag_len() bases at each
    /// offset and looking it up via exact match first, then Hamming-1.
    ///
    /// @param seq        Pointer to raw read sequence (A/C/G/T/N, any case)
    /// @param len        Length of the read sequence
    /// @param max_offset Maximum start offset to try (default 5)
    /// @return tag index ≥ 0, -1 (no match), or -2 (ambiguous)
    int match(const char* seq, int len,
              int max_offset = kDefaultMaxOffset) const {
        if (tag_len_ <= 0 || seq == nullptr || len < tag_len_) {
            ++stats_unmatched_;
            return -1;
        }

        int limit = std::min(max_offset, len - tag_len_);
        if (limit < 0) {
            ++stats_unmatched_;
            return -1;
        }

        // Build uppercase window once per position to avoid repeated alloc.
        std::string window(static_cast<size_t>(tag_len_), ' ');

        for (int off = 0; off <= limit; ++off) {
            for (int i = 0; i < tag_len_; ++i)
                window[i] = static_cast<char>(
                    std::toupper(static_cast<unsigned char>(seq[off + i])));

            // 1. Exact match
            auto it = exact_.find(window);
            if (it != exact_.end()) {
                ++stats_matched_;
                return it->second;
            }

            // 2. Hamming-1 match
            auto it2 = hamming1_.find(window);
            if (it2 != hamming1_.end()) {
                if (it2->second == kAmbiguous) {
                    ++stats_ambiguous_;
                    return -2;
                }
                ++stats_matched_;
                return it2->second;
            }
        }

        ++stats_unmatched_;
        return -1;
    }

    /// Overload accepting std::string.
    int match(const std::string& seq) const {
        return match(seq.c_str(), static_cast<int>(seq.size()));
    }

    // ── Tag accessors ─────────────────────────────────────────────────────

    size_t num_tags() const { return tags_.size(); }
    bool empty() const { return tags_.empty(); }
    int tag_len() const { return tag_len_; }

    const std::string& tag_name(int idx) const { return tags_[idx].name; }
    const std::string& tag_sequence(int idx) const { return tags_[idx].sequence; }

    const std::vector<AdtTag>& tags() const { return tags_; }

    /// Tag names in index order (for feature list output).
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(tags_.size());
        for (const auto& t : tags_) out.push_back(t.name);
        return out;
    }

    // ── Statistics ────────────────────────────────────────────────────────
    // Counters are atomic so they can be safely incremented from multiple
    // threads (read is relaxed; exact value under race is diagnostics-only).

    size_t matched() const { return stats_matched_.load(std::memory_order_relaxed); }
    size_t ambiguous() const { return stats_ambiguous_.load(std::memory_order_relaxed); }
    size_t unmatched() const { return stats_unmatched_.load(std::memory_order_relaxed); }

    void reset_stats() {
        stats_matched_.store(0);
        stats_ambiguous_.store(0);
        stats_unmatched_.store(0);
    }

   private:
    // Sentinel stored in hamming1_ map to indicate an ambiguous neighbour.
    static constexpr int kAmbiguous = -2;

    std::vector<AdtTag> tags_;
    std::unordered_map<std::string, int> exact_;     ///< seq → tag id
    std::unordered_map<std::string, int> hamming1_;  ///< Hamm-1 nbr → tag id or kAmbiguous
    int tag_len_ = 0;

    mutable std::atomic<size_t> stats_matched_{0};
    mutable std::atomic<size_t> stats_ambiguous_{0};
    mutable std::atomic<size_t> stats_unmatched_{0};

    // ── Internal helpers ──────────────────────────────────────────────────

    static std::string trim_(std::string s) {
        // Strip leading/trailing whitespace and quotes
        const auto bad = [](unsigned char c) {
            return std::isspace(c) || c == '"' || c == '\'';
        };
        while (!s.empty() && bad(static_cast<unsigned char>(s.back())))
            s.pop_back();
        size_t i = 0;
        while (i < s.size() && bad(static_cast<unsigned char>(s[i]))) ++i;
        return s.substr(i);
    }
};

}  // namespace singlet
