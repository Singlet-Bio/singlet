// SPDX-License-Identifier: MIT
// lib1fq/run_screen.h — per-run read screen before the runs of a sample are pooled
//
// Atlas re-run plan §6 item 3 (docs/atlas-rerun-plan.md on plan/atlas-rerun; signatures in
// docs/library-type-screen.md). A GEO sample (GSM) can hold several sequencing runs, and some of
// them are not the gene-expression library: hashtag/antibody/CMO/guide (feature-barcode) runs, or a
// library of another chemistry. Pooling them blindly mixes libraries. This screen reads the head of
// each run (default 1M read pairs) and reports
//
//   * read lengths (mode of R1 and R2),
//   * the barcode match rate against each 10x whitelist (exact on every sampled read; exact or one
//     mismatch on the first kOneMismatchReads reads), the best whitelist and the chemistry it implies,
//   * 3' vs 5' evidence (protocol.h five_prime_evidence: 5' TSO tail in R1, 3' TSO / poly(A) / poly(T)
//     in R2), which splits 10x 3' from 5' kits that share a whitelist,
//   * feature-barcode evidence on R2: the share of reads whose R2[0:15] or R2[10:25] is one of the 100
//     most common 15-mers (a gene-expression R2 is a random cDNA fragment, a feature-barcode R2 is one
//     of a few dozen tags), poly(A) right after a 15-base tag (TotalSeq-A / HTO / MULTI-seq), and the
//     10x Feature Barcode capture sequences and the Cas9 sgRNA scaffold anywhere in R2,
//   * V(D)J: NOT assessable from the reads alone. A 10x V(D)J library has the same R1 (BC+UMI+TSO) as
//     a 5' GEX library and a fragmented cDNA R2; telling them apart needs the share of reads on TR/IG
//     genes after alignment (pileup vdj counts). The report says so (vdj_assessable = false); the
//     campaign must rely on the metadata screen of docs/library-type-screen.md or a post-alignment check.
//
// decide_runs() then keeps the runs whose best whitelist matches >= 30% of reads with the sample's
// majority chemistry and drops feature-barcode-like runs. Majority rule: among runs that are not
// feature-barcode-like and pass the whitelist threshold, the chemistry with the most reads (total
// reads of the run when every candidate run knows it, e.g. from the .1fq header; otherwise sampled
// reads), ties broken by number of runs, then by name. For non-10x protocols (no whitelist reaches
// 30% in any run) the chemistry is protocol.h detect_protocol()'s tag and the whitelist rule is not
// applied.

#pragma once

#include "protocol.h"
#include "reader.h"
#include "types.h"

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <climits>
#include <unistd.h>
#endif

namespace singlet::fq {

// ── thresholds ──
constexpr uint64_t kScreenDefaultReads = 1000000;
constexpr uint64_t kOneMismatchReads = 100000;   // 1-mismatch whitelist lookups on this many reads
constexpr uint64_t kComplexityReads = 200000;    // R2 15-mer complexity on this many reads
constexpr double kScreenMinWhitelistRate = 0.30; // plan §6.3
constexpr double kFeatureTopShare = 0.50;        // top-100 15-mers hold >= half of R2 -> feature barcodes
constexpr double kFeatureSeqShare = 0.20;        // capture sequence / scaffold / tag+poly(A) in >= 20% of R2
constexpr uint64_t kScreenMinReads = 1000;       // fewer sampled reads: not screened, kept

struct ScreenWhitelist {
    std::string file;      // e.g. "3M-february-2018.txt"
    std::string family;    // chemistry family it implies: "v3", "v2", "arc", "v1"
    uint16_t bc_len = 16;
    std::vector<uint64_t> sorted;  // 2-bit packed barcodes, sorted
    bool loaded() const { return !sorted.empty(); }
};

/// The 10x whitelists the screen tests (3' v4 / 5' v3 / v4 share 3M-february-2018: v4 is reported as v3).
inline std::vector<ScreenWhitelist> screen_whitelist_specs() {
    return {{"3M-february-2018.txt", "v3", 16, {}},
            {"737K-august-2016.txt", "v2", 16, {}},
            {"gex_737K-arc-v1.txt", "arc", 16, {}},
            {"737K-april-2014.txt", "v1", 14, {}}};
}

/// 2-bit pack of numeric bases (A0 C1 G2 T3); false when the barcode holds an N.
inline bool screen_pack(const uint8_t* s, uint16_t len, uint64_t& out) {
    uint64_t h = 0;
    for (uint16_t i = 0; i < len; ++i) {
        if (s[i] > 3) return false;
        h = (h << 2) | s[i];
    }
    out = h;
    return true;
}

inline bool load_screen_whitelist(ScreenWhitelist& wl, const std::vector<std::string>& dirs) {
    for (const auto& d : dirs) {
        std::string path = d + "/" + wl.file;
        FILE* fp = std::fopen(path.c_str(), "r");
        if (!fp) continue;
        char buf[256];
        std::vector<uint8_t> num(wl.bc_len);
        while (std::fgets(buf, sizeof(buf), fp)) {
            size_t len = std::strlen(buf);
            while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) --len;
            if (len < wl.bc_len) continue;
            for (uint16_t i = 0; i < wl.bc_len; ++i) num[i] = nuc::ascii_to_num(buf[i]);
            uint64_t h;
            if (screen_pack(num.data(), wl.bc_len, h)) wl.sorted.push_back(h);
        }
        std::fclose(fp);
        std::sort(wl.sorted.begin(), wl.sorted.end());
        wl.sorted.erase(std::unique(wl.sorted.begin(), wl.sorted.end()), wl.sorted.end());
        return wl.loaded();
    }
    return false;
}

inline bool screen_wl_has(const ScreenWhitelist& wl, uint64_t h) {
    return std::binary_search(wl.sorted.begin(), wl.sorted.end(), h);
}

inline bool screen_wl_match1(const ScreenWhitelist& wl, uint64_t h) {
    if (screen_wl_has(wl, h)) return true;
    for (uint16_t i = 0; i < wl.bc_len; ++i) {
        const unsigned sh = 2u * (wl.bc_len - 1 - i);
        const uint64_t orig = (h >> sh) & 3u;
        for (uint64_t b = 0; b < 4; ++b) {
            if (b == orig) continue;
            if (screen_wl_has(wl, (h & ~(uint64_t(3) << sh)) | (b << sh))) return true;
        }
    }
    return false;
}

// ── one run's reads ──

struct ScreenSpot {
    std::vector<uint8_t> r1_seq, r2_seq;  // numeric bases (A0 C1 G2 T3 N4)
    uint16_t r1_len = 0, r2_len = 0;
};

struct WhitelistRate {
    std::string file;
    std::string family;
    bool loaded = false;
    double exact = 0.0;      // share of all sampled reads
    double one_mm = 0.0;     // exact or 1 mismatch, share of the first kOneMismatchReads reads
};

struct RunScreenReport {
    std::string run_id;
    std::string input;
    std::string error;            // non-empty: the run could not be read
    uint64_t total_reads = 0;     // reads in the run when known (.1fq header), else 0
    uint64_t reads_sampled = 0;
    uint16_t r1_len_mode = 0, r2_len_mode = 0;
    bool reads_swapped = false;   // barcode read found in R2 (short R2, long R1): roles swapped
    std::vector<WhitelistRate> whitelists;
    std::string best_whitelist;   // empty when none loaded
    double best_rate = 0.0;       // exact-or-1mm rate of the best whitelist
    std::string chemistry;        // e.g. "10x-3p-v3", "10x-5p-v2", "10x-arc-gex", or a protocol.h tag
    std::string detected_protocol;  // protocol.h detect_protocol tag (only when no whitelist reaches 30%)
    FivePrimeEvidence five_prime;
    bool is_5prime = false;
    // feature-barcode evidence (R2)
    double r2_top100_share_off0 = 0.0;
    double r2_top100_share_off10 = 0.0;
    double r2_tag_polya_share = 0.0;   // poly(A)>=8 starting at R2 positions 15..18
    double r2_capture_share = 0.0;     // 10x capture sequence 1/2 anywhere in R2
    double r2_scaffold_share = 0.0;    // Cas9 sgRNA scaffold anywhere in R2
    bool feature_barcode_like = false;
    std::string feature_reason;
    bool vdj_assessable = false;       // see header comment
};

namespace detail {

inline uint16_t length_mode(const std::vector<ScreenSpot>& spots, bool r1) {
    std::map<uint16_t, uint64_t> h;
    for (const auto& s : spots) h[r1 ? s.r1_len : s.r2_len]++;
    uint16_t best = 0;
    uint64_t n = 0;
    for (const auto& kv : h)
        if (kv.second > n) { n = kv.second; best = kv.first; }
    return best;
}

inline std::vector<uint8_t> to_num(const char* s) {
    std::vector<uint8_t> v;
    for (; *s; ++s) v.push_back(nuc::ascii_to_num(*s));
    return v;
}

inline bool contains(const std::vector<uint8_t>& seq, uint16_t len, const std::vector<uint8_t>& pat,
                     int max_mm) {
    if (len < pat.size()) return false;
    for (size_t j = 0; j + pat.size() <= len; ++j) {
        int mm = 0;
        for (size_t k = 0; k < pat.size(); ++k)
            if (seq[j + k] != pat[k] && ++mm > max_mm) break;
        if (mm <= max_mm) return true;
    }
    return false;
}

inline double top_share(const std::vector<ScreenSpot>& spots, uint16_t off, uint64_t max_reads) {
    std::unordered_map<uint64_t, uint64_t> cnt;
    uint64_t n = 0;
    for (const auto& s : spots) {
        if (n >= max_reads) break;
        if (s.r2_len < off + 15) continue;
        uint64_t h;
        if (!screen_pack(s.r2_seq.data() + off, 15, h)) continue;
        cnt[h]++;
        ++n;
    }
    if (n == 0) return 0.0;
    std::vector<uint64_t> v;
    v.reserve(cnt.size());
    for (const auto& kv : cnt) v.push_back(kv.second);
    const size_t k = std::min<size_t>(100, v.size());
    std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<uint64_t>());
    uint64_t top = 0;
    for (size_t i = 0; i < k; ++i) top += v[i];
    return static_cast<double>(top) / static_cast<double>(n);
}

}  // namespace detail

/// Screen the spots of one run (already read; see read_run_spots). whitelists must be loaded by the
/// caller (load_screen_whitelist); missing ones are reported as not loaded.
inline RunScreenReport screen_spots(std::vector<ScreenSpot> spots, const std::vector<ScreenWhitelist>& whitelists,
                                    const std::vector<std::string>& whitelist_dirs = {}) {
    RunScreenReport r;
    r.reads_sampled = spots.size();
    r.r1_len_mode = detail::length_mode(spots, true);
    r.r2_len_mode = detail::length_mode(spots, false);
    // Barcode read in R2 (short R2, long R1): swap so r1 is always the barcode read.
    if (r.r2_len_mode > 0 && r.r2_len_mode <= 34 && r.r1_len_mode > 40 && r.r1_len_mode > 2 * r.r2_len_mode) {
        r.reads_swapped = true;
        for (auto& s : spots) {
            std::swap(s.r1_seq, s.r2_seq);
            std::swap(s.r1_len, s.r2_len);
        }
        std::swap(r.r1_len_mode, r.r2_len_mode);
    }

    // ── whitelist match rates ──
    for (const auto& wl : whitelists) {
        WhitelistRate wr;
        wr.file = wl.file;
        wr.family = wl.family;
        wr.loaded = wl.loaded();
        if (wr.loaded && !spots.empty()) {
            uint64_t ex = 0, n1 = 0, m1 = 0;
            for (size_t i = 0; i < spots.size(); ++i) {
                const auto& s = spots[i];
                uint64_t h;
                const bool ok = s.r1_len >= wl.bc_len && screen_pack(s.r1_seq.data(), wl.bc_len, h);
                const bool hit = ok && screen_wl_has(wl, h);
                ex += hit;
                if (i < kOneMismatchReads) {
                    ++n1;
                    m1 += hit || (ok && screen_wl_match1(wl, h));
                }
            }
            wr.exact = static_cast<double>(ex) / spots.size();
            wr.one_mm = n1 ? static_cast<double>(m1) / n1 : 0.0;
        }
        if (wr.loaded && wr.one_mm > r.best_rate) {
            r.best_rate = wr.one_mm;
            r.best_whitelist = wr.file;
        }
        r.whitelists.push_back(wr);
    }

    // ── 3' vs 5' ──
    uint16_t bc_umi = 28;
    std::string family;
    for (const auto& wr : r.whitelists)
        if (wr.file == r.best_whitelist) family = wr.family;
    if (family == "v2") bc_umi = 26;
    if (family == "v1") bc_umi = 24;
    r.five_prime = five_prime_evidence(spots, bc_umi);
    r.is_5prime = r.five_prime.is_5prime();

    if (r.best_rate >= kScreenMinWhitelistRate) {
        if (family == "v3") r.chemistry = r.is_5prime ? "10x-5p-v3" : "10x-3p-v3";
        else if (family == "v2") r.chemistry = r.is_5prime ? "10x-5p-v2" : "10x-3p-v2";
        else if (family == "arc") r.chemistry = "10x-arc-gex";
        else if (family == "v1") r.chemistry = "10x-3p-v1";
    } else if (!spots.empty()) {
        // not a 10x whitelist: protocol.h detection on the head of the run
        std::vector<ScreenSpot> probe(spots.begin(), spots.begin() + std::min<size_t>(spots.size(), 20000));
        auto cand = detect_protocol(probe, r.r1_len_mode, r.r2_len_mode, whitelist_dirs);
        r.detected_protocol = cand.tag;
        r.chemistry = (cand.confidence == Confidence::NONE || cand.tag.empty()) ? "unknown" : cand.tag;
    } else {
        r.chemistry = "unknown";
    }

    // ── feature-barcode evidence ──
    r.r2_top100_share_off0 = detail::top_share(spots, 0, kComplexityReads);
    r.r2_top100_share_off10 = detail::top_share(spots, 10, kComplexityReads);
    static const std::vector<uint8_t> cap1 = detail::to_num("GCTTTAAGGCCGGTCCTAGCAA");   // 10x capture seq 1
    static const std::vector<uint8_t> cap2 = detail::to_num("GCTCACCTATTAGCGGCTAAGG");   // 10x capture seq 2
    static const std::vector<uint8_t> scaf = detail::to_num("GTTTTAGAGCTAGAAATAGCAAGTT"); // Cas9 sgRNA scaffold
    uint64_t n = 0, tag_pa = 0, cap = 0, sc = 0;
    for (const auto& s : spots) {
        if (n >= kComplexityReads) break;
        ++n;
        bool pa = false;
        for (uint16_t st = 15; st <= 18 && !pa; ++st) {
            if (s.r2_len < st + 8) break;
            int a = 0;
            for (int k = 0; k < 8; ++k) a += (s.r2_seq[st + k] == 0);
            pa = (a >= 7);
        }
        tag_pa += pa;
        cap += detail::contains(s.r2_seq, s.r2_len, cap1, 2) || detail::contains(s.r2_seq, s.r2_len, cap2, 2);
        sc += detail::contains(s.r2_seq, s.r2_len, scaf, 2);
    }
    if (n) {
        r.r2_tag_polya_share = static_cast<double>(tag_pa) / n;
        r.r2_capture_share = static_cast<double>(cap) / n;
        r.r2_scaffold_share = static_cast<double>(sc) / n;
    }
    const double top = std::max(r.r2_top100_share_off0, r.r2_top100_share_off10);
    std::ostringstream why;
    if (top >= kFeatureTopShare) why << "R2 top-100 15-mers hold " << static_cast<int>(top * 100) << "% of reads; ";
    if (r.r2_capture_share >= kFeatureSeqShare)
        why << "10x capture sequence in " << static_cast<int>(r.r2_capture_share * 100) << "% of R2; ";
    if (r.r2_scaffold_share >= kFeatureSeqShare)
        why << "sgRNA scaffold in " << static_cast<int>(r.r2_scaffold_share * 100) << "% of R2; ";
    // a tag followed by poly(A) only counts together with low complexity (3' GEX R2 rarely starts so)
    if (r.r2_tag_polya_share >= kFeatureSeqShare && top >= 0.25)
        why << "15-base tag + poly(A) in " << static_cast<int>(r.r2_tag_polya_share * 100) << "% of R2; ";
    r.feature_reason = why.str();
    if (r.feature_reason.size() >= 2) r.feature_reason.resize(r.feature_reason.size() - 2);
    r.feature_barcode_like = !r.feature_reason.empty();
    return r;
}

/// The 10x chemistry `screen-runs` calls for these reads (same whitelists, match rates and 3'/5' rule), or "" when
/// no 10x whitelist reaches kScreenMinWhitelistRate, the reads look like a feature-barcode library, or the barcode
/// read would have to be swapped. `download` / `encode` use it so that the .1fq header and screen-runs agree.
/// SpotT needs r1_seq, r2_seq (numeric bases), r1_len, r2_len. Whitelists are loaded once per directory list.
template <typename SpotT>
inline std::string screen_10x_chemistry(const std::vector<SpotT>& spots, const std::vector<std::string>& dirs,
                                        double* best_rate = nullptr) {
    static std::vector<ScreenWhitelist> wls;
    static std::vector<std::string> wl_dirs;
    static bool wl_init = false;
    if (!wl_init || wl_dirs != dirs) {
        wls = screen_whitelist_specs();
        for (auto& wl : wls) load_screen_whitelist(wl, dirs);
        wl_dirs = dirs;
        wl_init = true;
    }
    if (best_rate) *best_rate = 0.0;
    bool any = false;
    for (const auto& wl : wls) any = any || wl.loaded();
    if (!any || spots.empty()) return "";
    std::vector<ScreenSpot> ss;
    ss.reserve(spots.size());
    for (const auto& s : spots) {
        ScreenSpot t;
        t.r1_len = static_cast<uint16_t>(std::min<size_t>(s.r1_len, s.r1_seq.size()));
        t.r2_len = static_cast<uint16_t>(std::min<size_t>(s.r2_len, s.r2_seq.size()));
        t.r1_seq.assign(s.r1_seq.begin(), s.r1_seq.begin() + t.r1_len);
        t.r2_seq.assign(s.r2_seq.begin(), s.r2_seq.begin() + t.r2_len);
        ss.push_back(std::move(t));
    }
    RunScreenReport r = screen_spots(std::move(ss), wls, {});
    if (best_rate) *best_rate = r.best_rate;
    if (r.best_rate < kScreenMinWhitelistRate || r.reads_swapped || r.feature_barcode_like) return "";
    if (r.chemistry.rfind("10x-", 0) != 0) return "";
    return r.chemistry;
}

// ── input ──

namespace detail {

inline bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

inline std::string basename_noext(const std::string& p) {
    std::string b = p.substr(p.find_last_of('/') == std::string::npos ? 0 : p.find_last_of('/') + 1);
    for (const char* ext : {".gz", ".fastq", ".fq", ".1fq"})
        if (ends_with(b, ext)) b.resize(b.size() - std::strlen(ext));
    for (const char* suf : {"_R1_001", "_R1", "_1"})
        if (ends_with(b, suf)) { b.resize(b.size() - std::strlen(suf)); break; }
    return b;
}

/// Next FASTQ record's sequence; false at EOF.
inline bool gz_next_seq(gzFile f, std::string& seq) {
    char buf[65536];
    if (!gzgets(f, buf, sizeof(buf))) return false;           // header
    if (!gzgets(f, buf, sizeof(buf))) return false;           // sequence
    seq.assign(buf);
    while (!seq.empty() && (seq.back() == '\n' || seq.back() == '\r')) seq.pop_back();
    if (!gzgets(f, buf, sizeof(buf))) return false;           // +
    if (!gzgets(f, buf, sizeof(buf))) return false;           // quality
    return true;
}

}  // namespace detail

/// Read the first max_reads pairs of a run. `input` is a .1fq path or "R1.fastq[.gz],R2.fastq[.gz]".
inline std::vector<ScreenSpot> read_run_spots(const std::string& input, uint64_t max_reads,
                                              uint64_t* total_reads, std::string* error) {
    std::vector<ScreenSpot> spots;
    if (total_reads) *total_reads = 0;
    const auto comma = input.find(',');
    try {
        if (comma == std::string::npos) {
            Reader rd;
            rd.open(input.c_str());
            if (total_reads) *total_reads = rd.total_reads();
            DecodedBlock blk;
            while (spots.size() < max_reads && rd.read_block(blk)) {
                for (uint32_t i = 0; i < blk.n_reads && spots.size() < max_reads; ++i) {
                    ScreenSpot s;
                    s.r1_len = blk.r1_len(i);
                    s.r2_len = blk.r2_len(i);
                    s.r1_seq.assign(blk.r1_seq(i), blk.r1_seq(i) + s.r1_len);
                    s.r2_seq.assign(blk.r2_seq(i), blk.r2_seq(i) + s.r2_len);
                    spots.push_back(std::move(s));
                }
            }
        } else {
            const std::string p1 = input.substr(0, comma), p2 = input.substr(comma + 1);
            gzFile f1 = gzopen(p1.c_str(), "rb");
            gzFile f2 = gzopen(p2.c_str(), "rb");
            if (!f1 || !f2) {
                if (f1) gzclose(f1);
                if (f2) gzclose(f2);
                if (error) *error = "cannot open " + (f1 ? p2 : p1);
                return spots;
            }
            std::string a, b;
            while (spots.size() < max_reads && detail::gz_next_seq(f1, a) && detail::gz_next_seq(f2, b)) {
                ScreenSpot s;
                for (char c : a) s.r1_seq.push_back(nuc::ascii_to_num(c));
                for (char c : b) s.r2_seq.push_back(nuc::ascii_to_num(c));
                s.r1_len = static_cast<uint16_t>(std::min<size_t>(s.r1_seq.size(), 65535));
                s.r2_len = static_cast<uint16_t>(std::min<size_t>(s.r2_seq.size(), 65535));
                spots.push_back(std::move(s));
            }
            gzclose(f1);
            gzclose(f2);
        }
    } catch (const std::exception& e) {
        if (error) *error = e.what();
    }
    return spots;
}

inline RunScreenReport screen_run(const std::string& input, const std::vector<ScreenWhitelist>& whitelists,
                                  const std::vector<std::string>& whitelist_dirs,
                                  uint64_t max_reads = kScreenDefaultReads) {
    uint64_t total = 0;
    std::string err;
    auto spots = read_run_spots(input, max_reads, &total, &err);
    RunScreenReport r = screen_spots(std::move(spots), whitelists, whitelist_dirs);
    r.input = input;
    r.run_id = detail::basename_noext(input.substr(0, input.find(',')));
    r.total_reads = total;
    r.error = err;
    return r;
}

// ── decision ──

struct RunVerdict {
    std::string run_id;
    bool keep = false;
    std::string reason;
};

struct RunDecision {
    std::string majority_chemistry;   // empty when no run qualifies
    std::string majority_basis;       // "total_reads" | "sampled_reads"
    bool whitelist_rule = false;      // the >= 30% whitelist rule applied (10x majority)
    std::vector<RunVerdict> runs;     // in input order
    std::vector<std::string> kept() const {
        std::vector<std::string> v;
        for (const auto& r : runs) if (r.keep) v.push_back(r.run_id);
        return v;
    }
    std::vector<std::string> dropped() const {
        std::vector<std::string> v;
        for (const auto& r : runs) if (!r.keep) v.push_back(r.run_id);
        return v;
    }
};

inline bool is_10x_chemistry(const std::string& c) { return c.rfind("10x-", 0) == 0; }

inline RunDecision decide_runs(const std::vector<RunScreenReport>& reps) {
    RunDecision d;
    auto screened = [](const RunScreenReport& r) { return r.error.empty() && r.reads_sampled >= kScreenMinReads; };
    // candidates for the majority: screened, not feature-barcode-like
    bool any10x = false;
    for (const auto& r : reps)
        if (screened(r) && !r.feature_barcode_like && r.best_rate >= kScreenMinWhitelistRate && is_10x_chemistry(r.chemistry))
            any10x = true;
    d.whitelist_rule = any10x;
    auto candidate = [&](const RunScreenReport& r) {
        if (!screened(r) || r.feature_barcode_like) return false;
        if (any10x) return r.best_rate >= kScreenMinWhitelistRate && is_10x_chemistry(r.chemistry);
        return r.chemistry != "unknown";
    };
    bool all_total = true;
    for (const auto& r : reps)
        if (candidate(r) && r.total_reads == 0) all_total = false;
    d.majority_basis = all_total ? "total_reads" : "sampled_reads";
    std::map<std::string, std::pair<uint64_t, uint64_t>> votes;  // chemistry -> (reads, runs)
    for (const auto& r : reps)
        if (candidate(r)) {
            auto& v = votes[r.chemistry];
            v.first += all_total ? r.total_reads : r.reads_sampled;
            v.second += 1;
        }
    for (const auto& kv : votes) {  // std::map: ties end on the smallest name
        if (d.majority_chemistry.empty()) { d.majority_chemistry = kv.first; continue; }
        const auto& b = votes[d.majority_chemistry];
        if (kv.second.first > b.first || (kv.second.first == b.first && kv.second.second > b.second))
            d.majority_chemistry = kv.first;
    }
    for (const auto& r : reps) {
        RunVerdict v;
        v.run_id = r.run_id;
        std::ostringstream why;
        if (!r.error.empty()) {
            v.keep = false;
            why << "unreadable: " << r.error;
        } else if (r.reads_sampled < kScreenMinReads) {
            v.keep = true;
            why << "kept unscreened: only " << r.reads_sampled << " reads";
        } else if (r.feature_barcode_like) {
            why << "feature-barcode-like R2 (" << r.feature_reason << ")";
        } else if (any10x && r.best_rate < kScreenMinWhitelistRate) {
            why << "best whitelist match " << static_cast<int>(r.best_rate * 100) << "% < "
                << static_cast<int>(kScreenMinWhitelistRate * 100) << "%";
        } else if (r.chemistry != d.majority_chemistry) {
            why << "chemistry " << r.chemistry << " differs from the sample majority " << d.majority_chemistry;
        } else {
            v.keep = true;
            why << "chemistry " << r.chemistry;
            if (any10x) why << ", whitelist match " << static_cast<int>(r.best_rate * 100) << "%";
        }
        v.reason = why.str();
        d.runs.push_back(v);
    }
    return d;
}

// ── JSON ──

namespace detail {
inline std::string jesc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o;
}
inline std::string jnum(double x) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.6g", x);
    return b;
}
inline std::string jlist(const std::vector<std::string>& v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", \"" : "\"") + jesc(v[i]) + "\"";
    return o + "]";
}
}  // namespace detail

inline std::string run_screen_json(const std::vector<RunScreenReport>& reps, const RunDecision& d, uint64_t max_reads) {
    using namespace detail;
    std::ostringstream o;
    o << "{\n  \"screen\": \"singlet screen-runs\",\n  \"screen_version\": 1,\n"
      << "  \"reads_per_run\": " << max_reads << ",\n"
      << "  \"min_whitelist_rate\": " << jnum(kScreenMinWhitelistRate) << ",\n"
      << "  \"majority_chemistry\": \"" << jesc(d.majority_chemistry) << "\",\n"
      << "  \"majority_basis\": \"" << d.majority_basis << "\",\n"
      << "  \"whitelist_rule\": " << (d.whitelist_rule ? "true" : "false") << ",\n"
      << "  \"vdj_assessable\": false,\n"
      << "  \"runs_kept\": " << jlist(d.kept()) << ",\n"
      << "  \"runs_dropped\": " << jlist(d.dropped()) << ",\n  \"runs\": [\n";
    for (size_t i = 0; i < reps.size(); ++i) {
        const auto& r = reps[i];
        const auto& v = d.runs[i];
        o << "    {\"run_id\": \"" << jesc(r.run_id) << "\", \"input\": \"" << jesc(r.input) << "\""
          << ", \"keep\": " << (v.keep ? "true" : "false") << ", \"reason\": \"" << jesc(v.reason) << "\""
          << ", \"error\": \"" << jesc(r.error) << "\""
          << ", \"total_reads\": " << r.total_reads << ", \"reads_sampled\": " << r.reads_sampled
          << ", \"r1_len_mode\": " << r.r1_len_mode << ", \"r2_len_mode\": " << r.r2_len_mode
          << ", \"reads_swapped\": " << (r.reads_swapped ? "true" : "false")
          << ", \"chemistry\": \"" << jesc(r.chemistry) << "\""
          << ", \"detected_protocol\": \"" << jesc(r.detected_protocol) << "\""
          << ", \"best_whitelist\": \"" << jesc(r.best_whitelist) << "\", \"best_rate\": " << jnum(r.best_rate)
          << ", \"whitelists\": {";
        for (size_t j = 0; j < r.whitelists.size(); ++j) {
            const auto& w = r.whitelists[j];
            o << (j ? ", " : "") << "\"" << jesc(w.file) << "\": ";
            if (w.loaded) o << "{\"exact\": " << jnum(w.exact) << ", \"one_mm\": " << jnum(w.one_mm) << "}";
            else o << "null";
        }
        o << "}, \"five_prime\": {\"is_5prime\": " << (r.is_5prime ? "true" : "false")
          << ", \"n\": " << r.five_prime.n << ", \"r1_tso\": " << r.five_prime.r1_tso
          << ", \"r1_polyt\": " << r.five_prime.r1_polyt << ", \"r2_smart\": " << r.five_prime.r2_smart
          << ", \"r2_polya\": " << r.five_prime.r2_polya << ", \"r2_polyt\": " << r.five_prime.r2_polyt << "}"
          << ", \"feature_barcode\": {\"like\": " << (r.feature_barcode_like ? "true" : "false")
          << ", \"reason\": \"" << jesc(r.feature_reason) << "\""
          << ", \"r2_top100_share_off0\": " << jnum(r.r2_top100_share_off0)
          << ", \"r2_top100_share_off10\": " << jnum(r.r2_top100_share_off10)
          << ", \"r2_tag_polya_share\": " << jnum(r.r2_tag_polya_share)
          << ", \"r2_capture_share\": " << jnum(r.r2_capture_share)
          << ", \"r2_scaffold_share\": " << jnum(r.r2_scaffold_share) << "}"
          << ", \"vdj_assessable\": false}" << (i + 1 < reps.size() ? "," : "") << "\n";
    }
    o << "  ]\n}\n";
    return o.str();
}

// ── CLI ──

inline std::vector<std::string> default_whitelist_dirs() {
    std::vector<std::string> dirs;
    if (const char* e = std::getenv("SINGLET_WHITELIST_DIR")) dirs.push_back(e);
#if defined(__linux__)
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        std::string exe(buf, static_cast<size_t>(len));
        auto pos = exe.rfind('/');
        if (pos != std::string::npos) {
            std::string wl = exe.substr(0, pos) + "/../whitelists";
            struct stat st;
            if (stat(wl.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) dirs.push_back(wl);
        }
    }
#endif
    return dirs;
}

inline int screen_runs_main(int argc, char** argv) {
    uint64_t max_reads = kScreenDefaultReads;
    std::vector<std::string> dirs, runs;
    std::string json_out;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--reads" || a == "-n") && i + 1 < argc) max_reads = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--whitelist-dir" && i + 1 < argc) dirs.push_back(argv[++i]);
        else if ((a == "--json" || a == "-o") && i + 1 < argc) json_out = argv[++i];
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: singlet screen-runs [--reads N] [--whitelist-dir DIR ...] [--json OUT] RUN1 [RUN2 ...]\n"
                         "  RUN = run.1fq or R1.fastq[.gz],R2.fastq[.gz]\n"
                         "  Screens the first N read pairs (default 1000000) of each run of one sample: 10x whitelist\n"
                         "  match, 3'/5' evidence, feature-barcode signatures. Keeps runs with >= 30% whitelist match\n"
                         "  and the majority chemistry; drops feature-barcode-like runs. JSON to --json (or stdout).\n"
                         "  Exit 0 when at least one run is kept, 6 when none is.\n";
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::cerr << "screen-runs: unknown option " << a << "\n";
            return 1;
        } else runs.push_back(a);
    }
    if (runs.empty()) {
        std::cerr << "screen-runs: no runs given (see --help)\n";
        return 1;
    }
    if (dirs.empty()) dirs = default_whitelist_dirs();
    auto wls = screen_whitelist_specs();
    for (auto& wl : wls)
        if (!load_screen_whitelist(wl, dirs))
            std::cerr << "[screen-runs] whitelist not found: " << wl.file << "\n";
    std::vector<RunScreenReport> reps;
    for (const auto& run : runs) {
        reps.push_back(screen_run(run, wls, dirs, max_reads));
        const auto& r = reps.back();
        std::cerr << "[screen-runs] " << r.run_id << ": " << r.reads_sampled << " reads, chemistry=" << r.chemistry
                  << " best_wl=" << r.best_whitelist << " (" << static_cast<int>(r.best_rate * 100) << "%)"
                  << (r.feature_barcode_like ? " feature-barcode-like" : "") << (r.error.empty() ? "" : " ERROR " + r.error)
                  << "\n";
    }
    RunDecision d = decide_runs(reps);
    const std::string js = run_screen_json(reps, d, max_reads);
    if (json_out.empty()) std::cout << js;
    else {
        std::ofstream f(json_out);
        f << js;
        if (!f) {
            std::cerr << "screen-runs: cannot write " << json_out << "\n";
            return 1;
        }
    }
    for (const auto& v : d.runs)
        std::cerr << "[screen-runs] " << (v.keep ? "KEEP " : "DROP ") << v.run_id << ": " << v.reason << "\n";
    return d.kept().empty() ? 6 : 0;
}

}  // namespace singlet::fq
