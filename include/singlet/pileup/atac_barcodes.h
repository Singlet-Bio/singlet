// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_barcodes.h
// ATAC cell-barcode layout detection, whitelist correction and Multiome ATAC -> GEX translation.
//
// The ATAC barcode read is stored raw (QNAME prefix "BC:<raw>_"). Its layout depends on the kit and the sequencer:
//   10x scATAC     16 bp i5 read; forward or reverse complement of 737K-cratac-v1 depending on the i5 chemistry
//   10x Multiome   24 bp = 8 bp spacer + 16 bp barcode; forward or reverse complement of the ARC ATAC list
// Cell Ranger ATAC / ARC pick the orientation from the whitelist hit rate. We do the same: observe() a sample of raw
// barcodes, choose the (offset, strand) window with the most exact whitelist hits, then count exact hits per whitelist
// entry over all reads (the prior). resolve() maps a raw barcode to a whitelist index: exact match, else the unique most
// supported Hamming-1 neighbour (posterior >= 0.975 under the count prior, the Cell Ranger rule without base qualities).
//
// Multiome pairing: the ATAC and GEX barcodes of one gel bead are different sequences, paired by line in Cell Ranger
// ARC's two 737K-arc-v1 lists. load_pairs() reads "atac<TAB>gex" pairs; output_barcode() then reports the GEX barcode so
// ATAC columns join the GEX sample's cells table.

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace singlet {

class AtacBarcodeCorrector {
   public:
    static constexpr int kLen = 16;

    struct Layout {
        int offset = 0;        // start of the 16 bp window in the raw read (after reverse complement if revcomp)
        bool revcomp = false;  // reverse-complement the raw read first
        double hit_rate = 0;   // exact whitelist hits / observed reads in the detection sample
    };

    struct Stats {
        uint64_t reads = 0, exact = 0, corrected = 0, ambiguous = 0, unmatched = 0, invalid = 0;
    };

    /// Whitelist, one barcode per line (plain or gzip). If pairs are loaded instead, the whitelist is their ATAC side.
    bool load_whitelist(const std::string& path) {
        std::vector<std::string> lines;
        if (!read_lines(path, lines)) return false;
        for (auto& l : lines) {
            auto tab = l.find('\t');
            add_entry(tab == std::string::npos ? l : l.substr(0, tab), tab == std::string::npos ? "" : l.substr(tab + 1));
        }
        return !wl_.empty();
    }

    /// "atac_barcode<TAB>gex_barcode" pairs (plain or gzip); same as load_whitelist with a second column.
    bool load_pairs(const std::string& path) { return load_whitelist(path); }

    void add_entry(const std::string& atac, const std::string& gex = "") {
        uint32_t code;
        if (atac.size() < kLen || !encode(std::string_view(atac).substr(0, kLen), code)) return;
        if (index_.count(code)) return;
        index_.emplace(code, static_cast<uint32_t>(wl_.size()));
        wl_.push_back(atac.substr(0, kLen));
        gex_.push_back(gex);
        if (!gex.empty()) has_pairs_ = true;
    }

    size_t size() const { return wl_.size(); }
    bool has_pairs() const { return has_pairs_; }

    /// Phase 1: feed raw barcodes from the first reads (layout detection); call before count().
    void observe(std::string_view raw) {
        if (sample_.size() < kSample) sample_.emplace_back(raw);
    }

    /// Choose the window with the highest exact hit rate over the observed sample.
    Layout detect_layout() {
        Layout best;
        for (int rc = 0; rc < 2; ++rc) {
            size_t maxlen = 0;
            for (auto& s : sample_) maxlen = std::max(maxlen, s.size());
            for (int off = 0; off + kLen <= static_cast<int>(maxlen); ++off) {
                uint64_t hits = 0, n = 0;
                for (auto& s : sample_) {
                    if (static_cast<int>(s.size()) < off + kLen) continue;
                    ++n;
                    uint32_t code;
                    if (window(s, off, rc, code) && index_.count(code)) ++hits;
                }
                double r = n ? static_cast<double>(hits) / n : 0;
                if (r > best.hit_rate) best = {off, rc == 1, r};
            }
        }
        layout_ = best;
        counts_.assign(wl_.size(), 0);
        sample_.clear();
        sample_.shrink_to_fit();
        return best;
    }

    /// Phase 2: count exact hits per whitelist entry over all reads (the correction prior).
    void count(std::string_view raw) {
        uint32_t code;
        if (!window(raw, layout_.offset, layout_.revcomp, code)) return;
        auto it = index_.find(code);
        if (it != index_.end()) ++counts_[it->second];
    }

    /// Phase 3: raw barcode -> whitelist index, or -1.
    int64_t resolve(std::string_view raw) {
        ++stats_.reads;
        uint32_t code;
        if (!window(raw, layout_.offset, layout_.revcomp, code)) {
            // N in the window: try each base at the (single) N position
            int npos = -1;
            std::string w = extract(raw);
            if (w.empty()) { ++stats_.invalid; return -1; }
            for (int i = 0; i < kLen; ++i)
                if (base2(w[i]) < 0) { if (npos >= 0) { ++stats_.invalid; return -1; } npos = i; }
            return best_neighbour(w, npos);
        }
        auto it = index_.find(code);
        if (it != index_.end()) { ++stats_.exact; return it->second; }
        return best_neighbour(extract(raw), -1);
    }

    const std::string& atac_barcode(uint32_t idx) const { return wl_[idx]; }
    /// GEX barcode when pairs are loaded (for joining with the GEX sample), else the ATAC barcode.
    const std::string& output_barcode(uint32_t idx) const {
        return (has_pairs_ && !gex_[idx].empty()) ? gex_[idx] : wl_[idx];
    }
    uint64_t prior(uint32_t idx) const { return idx < counts_.size() ? counts_[idx] : 0; }
    const Layout& layout() const { return layout_; }
    void set_layout(const Layout& l) { layout_ = l; if (counts_.size() != wl_.size()) counts_.assign(wl_.size(), 0); }
    const Stats& stats() const { return stats_; }

   private:
    static constexpr size_t kSample = 200000;

    static int base2(char c) {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'G': case 'g': return 2;
            case 'T': case 't': return 3;
            default: return -1;
        }
    }
    static char comp(char c) {
        switch (c) {
            case 'A': return 'T'; case 'C': return 'G'; case 'G': return 'C'; case 'T': return 'A';
            case 'a': return 't'; case 'c': return 'g'; case 'g': return 'c'; case 't': return 'a';
            default: return 'N';
        }
    }
    static bool encode(std::string_view s, uint32_t& code) {
        code = 0;
        for (int i = 0; i < kLen; ++i) {
            int b = base2(s[i]);
            if (b < 0) return false;
            code = (code << 2) | static_cast<uint32_t>(b);
        }
        return true;
    }
    // 16 bp window of the raw read in whitelist orientation.
    static bool window(std::string_view raw, int off, bool rc, uint32_t& code) {
        if (static_cast<int>(raw.size()) < off + kLen) return false;
        if (!rc) return encode(raw.substr(off, kLen), code);
        char buf[kLen];
        // reverse complement of the whole read, then take [off, off+16)
        const int n = static_cast<int>(raw.size());
        for (int i = 0; i < kLen; ++i) buf[i] = comp(raw[n - 1 - (off + i)]);
        return encode(std::string_view(buf, kLen), code);
    }
    std::string extract(std::string_view raw) const {
        const int off = layout_.offset;
        if (static_cast<int>(raw.size()) < off + kLen) return {};
        std::string w(kLen, 'N');
        const int n = static_cast<int>(raw.size());
        for (int i = 0; i < kLen; ++i) w[i] = layout_.revcomp ? comp(raw[n - 1 - (off + i)]) : raw[off + i];
        return w;
    }
    int64_t best_neighbour(const std::string& w, int only_pos) {
        static const char kB[4] = {'A', 'C', 'G', 'T'};
        std::string t = w;
        int64_t best = -1;
        double best_c = 0, total = 0;
        for (int i = 0; i < kLen; ++i) {
            if (only_pos >= 0 && i != only_pos) continue;
            const char orig = t[i];
            for (char b : kB) {
                if (b == orig) continue;
                t[i] = b;
                uint32_t code;
                if (encode(t, code)) {
                    auto it = index_.find(code);
                    if (it != index_.end()) {
                        // +0.5 pseudocount, as Cell Ranger's prior on never-seen whitelist barcodes
                        double c = static_cast<double>(counts_[it->second]) + 0.5;
                        total += c;
                        if (c > best_c) { best_c = c; best = it->second; }
                    }
                }
            }
            t[i] = orig;
        }
        if (best < 0) { ++stats_.unmatched; return -1; }
        if (best_c / total < 0.975) { ++stats_.ambiguous; return -1; }
        ++stats_.corrected;
        return best;
    }
    static bool read_lines(const std::string& path, std::vector<std::string>& out) {
        gzFile f = gzopen(path.c_str(), "rb");  // reads plain text too
        if (!f) return false;
        char buf[512];
        while (gzgets(f, buf, sizeof(buf))) {
            std::string s(buf);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            if (!s.empty() && s[0] != '#') out.push_back(std::move(s));
        }
        gzclose(f);
        return true;
    }

    std::unordered_map<uint32_t, uint32_t> index_;
    std::vector<std::string> wl_, gex_;
    std::vector<uint64_t> counts_;
    std::vector<std::string> sample_;
    Layout layout_;
    Stats stats_;
    bool has_pairs_ = false;
};

}  // namespace singlet
