// SPDX-License-Identifier: MIT
// fq/identify.h — read-based library identification (`singlet identify`)
//
// From the first reads of a dataset, work out what the library is (assay family, protocol tag, modality) and say what to do with it.
// The classifier composes the existing detectors (protocol.h detect_protocol, five_prime_evidence, AssaySignals; run_screen.h
// whitelist rates and feature-barcode evidence) and adds recognizers for the families they miss. It changes nothing in the pipeline.
//
// Evidence layers
//   A  layout          segments per spot, length mode and spread per segment, index-like segments
//   B  barcode         whitelist hit rates per segment (forward and reverse complement), well-barcode windows, multiplicity
//   C  insert content  poly(T)/poly(A), Tn5 ends, feature-barcode constants, Smart-seq3 tag, N and C content
//   D  mapping         optional: a bundled-STAR alignment of the head's cDNA reads (mapped, exonic, sense fraction, chrM share)
//   E  usefulness      share of reads expected to give a countable molecule, with its components
//
// The classifier (identify_spots) returns ranked interpretations; the policy table (policy_for) is separate and maps a family to an
// action. Every threshold sits in kThr below with the measurement that set it.
//
// Header only; the head readers (SRA, URL, FASTQ, BAM) live in identify_cli.h so this header has no VDB dependency.

#pragma once

#include "protocol.h"
#include "run_screen.h"
#include "types.h"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet::fq::identify {

constexpr int kSchemaVersion = 1;

// ── thresholds (one table; each names the measurement that set it) ──
struct Thresholds {
    // A whitelist hit (one mismatch) of at least this share of reads calls the list. The plan's screen floor; real 10x libraries hit
    // 50-95%, random 16-mers hit a 3.6 M list about once in a million (protocol.h kProtocolWlRateFloor note).
    double wl_hit = 0.30;
    // v2 (737K) is a subset of v3 (3M): a v3 library hits the 737K list in about 20% of reads, a v2 library hits both alike. Above this
    // ratio of v2 to v3 hits the library is v2 when R1 is not 28 or longer.
    double v2_over_v3 = 0.70;
    // Strand: 3' kits give sense cDNA reads (sense fraction above 0.9 in STARsolo-style runs), 5' kits antisense (below 0.2); between
    // these bounds the strand does not decide and the motif evidence (five_prime_evidence) stays in charge.
    double sense_3p = 0.70, sense_5p = 0.30;
    // Well barcodes: the top K sequences of a window hold at least this share while a random window of the same length would hold K/4^len.
    double plate_top_share = 0.60;
    double plate_random_share_max = 0.05;
    // Tn5 mosaic end in this share of reads: Tn5 tagmentation. 10x ATAC and Nextera RNA libraries both carry it (AssaySignals).
    double tn5 = 0.15;
    // Smart-seq3 5' tag (ATTGCGCAATG) at the read start in this share of reads (published design: every 5' read carries the tag).
    double smart3_tag = 0.05;
    // Non-sequence data: more than 20% N (protocol.h classify_assay); methylome: C below 8% (same).
    double n_share = 0.20, c_share_methyl = 0.08;
    // Mapped evidence: RNA libraries put at least this share of uniquely mapped reads on annotated exons; genomic DNA and ATAC put few
    // there. Chosen at the gap between RNA (30-90% in 10x GEX, nuclei 20%) and genomic (2-8%).
    double exonic_rna_min = 0.15, exonic_dna_max = 0.10;
    // Uniquely mapped share below this on the right species: the head is not a transcriptome of the given genome.
    double mapped_min = 0.50;
    // Verdict margin over the best interpretation of another family.
    double margin = 0.15;
    // Verdict floor on the top support.
    double support_min = 0.30;
    // Full-length RNA without a cell barcode: runs of at most this many spots are single-cell-sized (Smart-seq2 plate wells hold
    // 0.2-5 M reads; bulk RNA runs hold 20 M or more).
    uint64_t smartseq_max_spots = 8000000;
    // Feature-barcode evidence (run_screen.h): top-100 R2 15-mers hold at least this share, capture or scaffold constants this share.
    double feature_top100 = kFeatureTopShare, feature_const = kFeatureSeqShare;
};
inline const Thresholds kThr{};

// ── input ──

struct HeadSpot {
    std::vector<std::string> seg;   // segments in spot order, ASCII bases
};

struct SourceInfo {
    std::string source;      // what the caller asked for
    std::string route;       // sra | url | fastq | bam | 1fq
    std::string endpoint;    // host or file that answered
    int64_t bytes = -1;      // compressed bytes fetched, -1 when not measurable
    std::string bytes_note;
    double seconds = 0;      // wall time of the head read
    uint64_t spots = 0;      // spots returned
    int64_t run_spots = -1;  // spots in the whole run when the source says
    std::string error;
};

// ── results ──

struct SegStat {
    uint32_t len_mode = 0, len_min = 0, len_max = 0;
    double mode_share = 0, mean_len = 0, entropy_first32 = 0;   // mean per-position Shannon entropy in bits (max 2)
    bool index_like = false;                                    // short and low diversity
};

struct ListHit {
    std::string file, family;
    int seg = -1;
    bool revcomp = false;
    bool loaded = false;
    double exact = 0, one_mm = 0;
};

struct WindowHit {
    bool found = false;
    int seg = -1;
    uint16_t off = 0, len = 0, k = 0;
    uint32_t distinct = 0;
    double top_share = 0, random_share = 0;
};

struct MapEvidence {
    bool measured = false;
    std::string note;
    uint64_t reads = 0;
    double unique = 0, multi = 0, unmapped = 0;   // shares of input reads
    double exonic = 0;                            // share of uniquely mapped reads assigned to a gene in either orientation
    double sense = -1;                            // sense / (sense + antisense) over assigned reads, -1 when unmeasured
    double chrm = 0;                              // share of mapped reads on chrM or MT
    bool exonic_measured = false;
    double seconds = 0;
};

struct Interp {
    std::string family, protocol, modality;
    double support = 0;
    double usefulness = 0;
    std::vector<std::pair<std::string, double>> components;   // name -> value, usefulness is their product
    std::vector<std::string> unmeasured;                      // components left out of the product
    std::vector<std::string> reasons;
};

struct Evidence {
    // A
    std::vector<SegStat> segs;
    int bc_seg = -1, cdna_seg = -1, cdna2_seg = -1;
    // B
    std::vector<ListHit> lists;      // best (segment, orientation) per list
    WindowHit window;
    ProtocolCandidate detected;      // protocol.h detect_protocol on the head
    double repeat_share_bc = 0;      // share of reads whose first 16 bases at the barcode segment repeat another read's
    // C
    FivePrimeEvidence five_prime;
    bool five_prime_called = false;
    double tn5 = 0, polyt = 0, adt_cap = 0, gc = 0, c_share = 0, n_share = 0;
    double smart3_tag = 0;
    double top100_off0 = 0, top100_off10 = 0, tag_polya = 0, capture = 0, scaffold = 0;
    std::string feature_reason;
    bool feature_like = false;
    // D
    MapEvidence map;
};

struct Report {
    int schema = kSchemaVersion;
    SourceInfo src;
    uint64_t spots = 0;
    Evidence ev;
    std::vector<Interp> ranked;
    std::string verdict;          // family of the top interpretation, or "ambiguous"
    std::string verdict_protocol; // tag of the top interpretation
    std::string ambiguous_with;   // when ambiguous: the two families
    std::string action;           // process | classify_only | defer | refuse
    std::string action_note;
    std::string hint;             // caller-supplied metadata, shown and never counted as evidence
};

// ── policy table (separate from the classifier) ──

struct Policy {
    const char* family;
    const char* action;
    const char* note;
};

inline const std::vector<Policy>& policy_table() {
    static const std::vector<Policy> t = {
        {"droplet_3p", "process", "10x 3' and GEM-X"},
        {"droplet_5p", "process", "10x 5'"},
        {"multiome_gex", "process", "10x multiome GEX half"},
        {"visium", "process", "Visium poly(A)"},
        {"droplet_other", "process", "droplet scRNA with a known protocol spec"},
        {"combinatorial", "process", "combinatorial scRNA with a known protocol spec"},
        {"atac", "classify_only", "ATAC is recognized and not processed"},
        {"multiome_atac", "classify_only", "ATAC half of multiome is recognized and not processed"},
        {"feature_barcode", "classify_only", "ADT, HTO, CMO or guide library: recognized and not processed"},
        {"vdj", "classify_only", "V(D)J enrichment: recognized and not processed"},
        {"plate_barcoded", "defer", "single-cell plate protocol with well barcodes: recognized, processing awaits a settled spec"},
        {"smartseq", "defer", "single-cell full-length protocol: recognized, not yet supported"},
        {"smartseq3", "defer", "single-cell full-length protocol with UMI: recognized, not yet supported"},
        {"rna_no_barcode", "refuse", "full-length RNA without a cell barcode: bulk unless metadata says single cell"},
        {"bulk_rna", "refuse", "bulk RNA-seq is not single-cell"},
        {"genomic_dna", "refuse", "genomic DNA"},
        {"methylome", "refuse", "bisulfite reads"},
        {"not_sequence", "refuse", "not sequence data"},
        {"unknown", "refuse", "no known library structure"},
        {"ambiguous", "refuse", "two families remain; pass a metadata hint or run the mapping evidence"},
    };
    return t;
}

inline const Policy& policy_for(const std::string& family) {
    for (const auto& p : policy_table())
        if (family == p.family) return p;
    return policy_table().back();
}

// ── helpers ──

namespace detail_id {

inline char comp(char c) {
    switch (c) {
        case 'A': return 'T'; case 'C': return 'G'; case 'G': return 'C'; case 'T': return 'A';
        case 'a': return 't'; case 'c': return 'g'; case 'g': return 'c'; case 't': return 'a';
        default: return 'N';
    }
}
inline std::string revcomp(const std::string& s) {
    std::string r(s.rbegin(), s.rend());
    for (auto& c : r) c = comp(c);
    return r;
}

inline bool pack(const char* s, uint16_t len, uint64_t& out) {
    uint64_t h = 0;
    for (uint16_t i = 0; i < len; ++i) {
        const uint8_t b = nuc::ascii_to_num(s[i]);
        if (b > 3) return false;
        h = (h << 2) | b;
    }
    out = h;
    return true;
}

inline std::vector<uint8_t> num(const std::string& s) {
    std::vector<uint8_t> v(s.size());
    for (size_t i = 0; i < s.size(); ++i) v[i] = nuc::ascii_to_num(s[i]);
    return v;
}

// Load a barcode list from the first directory holding file or file.gz.
inline bool load_list(ScreenWhitelist& wl, const std::vector<std::string>& dirs) {
    for (const auto& d : dirs) {
        for (const std::string suffix : {"", ".gz"}) {
            const std::string path = d + "/" + wl.file + suffix;
            gzFile f = gzopen(path.c_str(), "rb");
            if (!f) continue;
            char buf[512];
            while (gzgets(f, buf, sizeof(buf))) {
                size_t len = std::strlen(buf);
                while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) --len;
                uint64_t h;
                if (len >= wl.bc_len && pack(buf, wl.bc_len, h)) wl.sorted.push_back(h);
            }
            gzclose(f);
            std::sort(wl.sorted.begin(), wl.sorted.end());
            wl.sorted.erase(std::unique(wl.sorted.begin(), wl.sorted.end()), wl.sorted.end());
            if (wl.loaded()) return true;
        }
    }
    return false;
}

inline double entropy_bits(const uint32_t c[4]) {
    const double n = double(c[0]) + c[1] + c[2] + c[3];
    if (n <= 0) return 0;
    double h = 0;
    for (int i = 0; i < 4; ++i)
        if (c[i]) { const double p = c[i] / n; h -= p * std::log2(p); }
    return h;
}

inline bool has_motif(const std::string& s, const std::string& m, int max_mm, size_t from = 0, size_t to = std::string::npos) {
    if (s.size() < m.size()) return false;
    const size_t last = std::min(to, s.size() - m.size());
    for (size_t p = from; p <= last; ++p) {
        int mm = 0;
        for (size_t k = 0; k < m.size() && mm <= max_mm; ++k) mm += (s[p + k] != m[k]);
        if (mm <= max_mm) return true;
    }
    return false;
}

}  // namespace detail_id

/// The lists identify tests. Order matters only for reporting.
inline std::vector<ScreenWhitelist> identify_lists() {
    return {{"3M-february-2018.txt", "v3", 16, {}},     {"737K-august-2016.txt", "v2", 16, {}},
            {"gex_737K-arc-v1.txt", "arc", 16, {}},     {"737K-april-2014.txt", "v1", 14, {}},
            {"visium-v1.txt", "visium", 16, {}},        {"737K-cratac-v1.txt", "atac", 16, {}},
            {"3M-3pgex-may-2023.txt", "v4_3p", 16, {}}, {"3M-5pgex-jan-2023.txt", "v4_5p", 16, {}},
            {"737K-arc-v1-atac-gex.tsv.gz", "arc_atac", 16, {}}};   // first column: the ATAC half of multiome barcodes
}

// ── layer A ──

inline SegStat seg_stat(const std::vector<HeadSpot>& spots, size_t s) {
    SegStat st;
    std::map<uint32_t, uint64_t> lens;
    uint64_t n = 0, sum = 0;
    uint32_t cnt[32][4] = {};
    st.len_min = UINT32_MAX;
    for (const auto& sp : spots) {
        if (s >= sp.seg.size()) continue;
        const auto& q = sp.seg[s];
        ++n;
        ++lens[static_cast<uint32_t>(q.size())];
        sum += q.size();
        st.len_min = std::min<uint32_t>(st.len_min, q.size());
        st.len_max = std::max<uint32_t>(st.len_max, q.size());
        for (size_t i = 0; i < q.size() && i < 32; ++i) {
            const uint8_t b = nuc::ascii_to_num(q[i]);
            if (b < 4) ++cnt[i][b];
        }
    }
    if (!n) { st.len_min = 0; return st; }
    uint64_t best = 0;
    for (const auto& [l, c] : lens)
        if (c > best) { best = c; st.len_mode = l; }
    st.mode_share = double(best) / n;
    st.mean_len = double(sum) / n;
    const uint32_t npos = std::min<uint32_t>(32, st.len_mode);
    double h = 0;
    for (uint32_t i = 0; i < npos; ++i) h += detail_id::entropy_bits(cnt[i]);
    st.entropy_first32 = npos ? h / npos : 0;
    st.index_like = st.len_mode <= 12 || (st.len_mode <= 24 && st.entropy_first32 < 1.0);
    return st;
}

// ── layer B: well-barcode windows ──

/// Scan windows of one segment for a small set of barcodes that holds most reads (plates have tens to hundreds of wells).
inline WindowHit scan_windows(const std::vector<HeadSpot>& spots, int s, size_t max_reads = 50000) {
    WindowHit best;
    const uint32_t L = [&] { std::map<size_t, int> m; for (size_t i = 0; i < std::min<size_t>(spots.size(), 2000); ++i) if ((size_t)s < spots[i].seg.size()) ++m[spots[i].seg[s].size()]; size_t b = 0; int c = 0; for (auto& [l, k] : m) if (k > c) { c = k; b = l; } return (uint32_t)b; }();
    if (L < 6) return best;
    const size_t n = std::min(spots.size(), max_reads);
    static const uint16_t Ks[3] = {96, 384, 1536};
    std::vector<WindowHit> cands;
    for (uint16_t len = 12; len >= 6; --len) {
        for (uint16_t off = 0; off + len <= std::min<uint32_t>(L, 36); ++off) {
            std::unordered_map<uint64_t, uint32_t> cnt;
            uint64_t tot = 0;
            for (size_t i = 0; i < n; ++i) {
                if ((size_t)s >= spots[i].seg.size() || spots[i].seg[s].size() < off + len) continue;
                uint64_t h;
                if (!detail_id::pack(spots[i].seg[s].data() + off, len, h)) continue;
                ++cnt[h];
                ++tot;
            }
            if (tot < 1000 || cnt.size() > tot / 4) continue;
            std::vector<uint32_t> v;
            v.reserve(cnt.size());
            for (auto& kv : cnt) v.push_back(kv.second);
            std::sort(v.begin(), v.end(), std::greater<uint32_t>());
            for (uint16_t K : Ks) {
                const double rnd = std::min(1.0, double(K) / std::pow(4.0, len));
                if (rnd > kThr.plate_random_share_max) continue;
                uint64_t top = 0;
                const size_t keff = std::min<size_t>(v.size(), K);
                for (size_t i = 0; i < keff; ++i) top += v[i];
                const double share = double(top) / tot;
                if (share < kThr.plate_top_share) continue;
                // barcodes proper: sequences seen at a tenth of the average top count (errors and noise stay out)
                const double thr = std::max(2.0, 0.1 * double(top) / keff);
                uint32_t distinct = 0;
                for (uint32_t c : v) distinct += (c >= thr);
                if (distinct < 12) break;   // a linker or poly(T) run is one or a few sequences, not a barcode set
                WindowHit w;
                w.found = true; w.seg = s; w.off = off; w.len = len; w.k = K; w.distinct = distinct; w.top_share = share; w.random_share = rnd;
                cands.push_back(w);
                break;
            }
        }
    }
    // A window that reaches into the UMI multiplies the distinct count by four per base; the barcode is the longest window whose
    // distinct count stays within a third of the smallest one seen.
    if (!cands.empty()) {
        uint32_t mind = UINT32_MAX;
        for (const auto& c : cands) mind = std::min(mind, c.distinct);
        for (const auto& c : cands) {
            if (c.distinct > 1.3 * mind) continue;
            if (!best.found || c.len > best.len || (c.len == best.len && c.off < best.off)) best = c;
        }
        // fixed bases (a poly(T) or linker run) next to the barcode keep the distinct count unchanged: trim edges that carry no variation
        uint32_t cnt[40][4] = {};
        for (size_t i = 0; i < n; ++i) {
            if ((size_t)s >= spots[i].seg.size() || spots[i].seg[s].size() < best.off + best.len) continue;
            for (uint16_t k = 0; k < best.len && k < 40; ++k) {
                const uint8_t b = nuc::ascii_to_num(spots[i].seg[s][best.off + k]);
                if (b < 4) ++cnt[k][b];
            }
        }
        while (best.len > 6 && detail_id::entropy_bits(cnt[best.len - 1]) < 0.5) --best.len;
        uint16_t shift = 0;
        while (best.len - shift > 6 && detail_id::entropy_bits(cnt[shift]) < 0.5) ++shift;
        best.off += shift; best.len -= shift;
    }
    return best;
}

// ── layer D: bundled-STAR alignment of the head ──

/// Parse the outputs of a STAR run on the cDNA reads: Log.final.out, ReadsPerGene.out.tab and the SAM's reference names.
inline MapEvidence parse_star_outputs(const std::string& prefix, uint64_t n_reads) {
    MapEvidence m;
    m.reads = n_reads;
    std::ifstream lg(prefix + "Log.final.out");
    if (!lg) { m.note = "no STAR log"; return m; }
    std::string line;
    double input = 0, uniq = 0, multi = 0;
    auto val = [](const std::string& l) { auto p = l.find('|'); return p == std::string::npos ? std::string() : l.substr(p + 1); };
    while (std::getline(lg, line)) {
        if (line.find("Number of input reads") != std::string::npos) input = std::atof(val(line).c_str());
        else if (line.find("Uniquely mapped reads number") != std::string::npos) uniq = std::atof(val(line).c_str());
        else if (line.find("Number of reads mapped to multiple loci") != std::string::npos) multi = std::atof(val(line).c_str());
    }
    if (input <= 0) { m.note = "STAR log has no input reads"; return m; }
    m.unique = uniq / input;
    m.multi = multi / input;
    m.unmapped = std::max(0.0, 1.0 - m.unique - m.multi);
    m.measured = true;
    std::ifstream rg(prefix + "ReadsPerGene.out.tab");
    if (rg) {
        double sense = 0, anti = 0, nofeat = 0, amb = 0;
        while (std::getline(rg, line)) {
            std::istringstream is(line);
            std::string id;
            double c1, c2, c3;
            if (!(is >> id >> c1 >> c2 >> c3)) continue;
            if (id == "N_noFeature") nofeat = c1;
            else if (id == "N_ambiguous") amb = c1;
            else if (id.rfind("N_", 0) == 0) continue;
            else { sense += c2; anti += c3; }
        }
        const double assigned = std::max(sense, anti);   // unstranded column would double count; use the larger strand total as the floor
        const double assigned_any = sense + anti;
        if (uniq > 0) { m.exonic = std::min(1.0, assigned_any / uniq); m.exonic_measured = true; }
        (void)assigned; (void)nofeat; (void)amb;
        if (sense + anti >= 200) m.sense = sense / (sense + anti);
    } else {
        m.note = "no ReadsPerGene (genome index lacks annotation): exonic and sense not measured";
    }
    std::ifstream sam(prefix + "Aligned.out.sam");
    if (sam) {
        uint64_t n = 0, mt = 0;
        while (std::getline(sam, line)) {
            if (line.empty() || line[0] == '@') continue;
            auto t1 = line.find('\t');
            for (int k = 0; k < 1 && t1 != std::string::npos; ++k) t1 = line.find('\t', t1 + 1);   // after FLAG
            auto t2 = line.find('\t', t1 + 1);
            const std::string rname = line.substr(t1 + 1, t2 - t1 - 1);
            ++n;
            if (rname == "chrM" || rname == "MT" || rname == "M" || rname == "chrMT") ++mt;
        }
        if (n) m.chrm = double(mt) / n;
    }
    return m;
}

// ── the classifier ──

struct Options {
    std::vector<std::string> whitelist_dirs;
    std::string hint;               // e.g. "assay=smartseq2"; reported, never counted
    int64_t run_spots = -1;         // spots in the whole run when known
};

inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

inline const char* conf_name(Confidence c) {
    switch (c) { case Confidence::NONE: return "none"; case Confidence::LOW: return "low"; case Confidence::MEDIUM: return "medium";
                 case Confidence::HIGH: return "high"; default: return "manual"; }
}

/// Stage 1 of identify: layers A, B, C on the head. Layer D is attached afterwards (attach_mapping) and the interpretations ranked by rank().
inline void measure(const std::vector<HeadSpot>& spots, const Options& opt, std::vector<ScreenWhitelist>& lists, Evidence& ev) {
    using namespace detail_id;
    size_t nseg = 0;
    for (size_t i = 0; i < std::min<size_t>(spots.size(), 1000); ++i) nseg = std::max(nseg, spots[i].seg.size());
    for (size_t s = 0; s < nseg; ++s) ev.segs.push_back(seg_stat(spots, s));
    for (auto& wl : lists)
        if (!wl.loaded()) load_list(wl, opt.whitelist_dirs);

    // B: every list against every segment of useful length, forward and reverse complement; keep the best place per list.
    const size_t probe_n = std::min<size_t>(spots.size(), 50000);
    for (const auto& wl : lists) {
        ListHit best;
        best.file = wl.file; best.family = wl.family; best.loaded = wl.loaded();
        if (wl.loaded()) {
            for (size_t s = 0; s < nseg; ++s) {
                if (ev.segs[s].len_mode < wl.bc_len) continue;
                for (int rc = 0; rc < ((wl.family == "atac" || wl.family == "arc_atac") ? 2 : 1); ++rc) {
                    uint64_t ex = 0, n1 = 0, m1 = 0, n = 0;
                    for (size_t i = 0; i < probe_n; ++i) {
                        if (s >= spots[i].seg.size() || spots[i].seg[s].size() < wl.bc_len) continue;
                        ++n;
                        std::string bc = rc ? revcomp(spots[i].seg[s]).substr(0, wl.bc_len) : spots[i].seg[s].substr(0, wl.bc_len);
                        uint64_t h;
                        const bool ok = pack(bc.data(), wl.bc_len, h);
                        const bool hit = ok && screen_wl_has(wl, h);
                        ex += hit;
                        if (i < kOneMismatchReads) { ++n1; m1 += hit || (ok && screen_wl_match1(wl, h)); }
                    }
                    const double e = n ? double(ex) / n : 0, o = n1 ? double(m1) / n1 : 0;
                    if (o > best.one_mm) { best.seg = (int)s; best.revcomp = rc; best.exact = e; best.one_mm = o; }
                }
            }
        }
        ev.lists.push_back(best);
    }

    // barcode segment = best whitelist segment when a list is called, else the shortest segment of 8 to 40 bases
    double best_rate = 0;
    for (const auto& h : ev.lists)
        if (h.one_mm > best_rate) { best_rate = h.one_mm; ev.bc_seg = h.seg; }
    if (best_rate < kThr.wl_hit) {
        ev.bc_seg = -1;
        uint32_t best_len = UINT32_MAX;
        for (size_t s = 0; s < nseg; ++s)
            if (nseg >= 2 && ev.segs[s].len_mode >= 8 && ev.segs[s].len_mode <= 60 && ev.segs[s].len_mode < best_len &&
                ev.segs[s].len_mode * 1.0 < 0.85 * [&] { uint32_t m = 0; for (auto& t : ev.segs) m = std::max(m, t.len_mode); return (double)m; }()) {
                best_len = ev.segs[s].len_mode; ev.bc_seg = (int)s;
            }
    }
    // cDNA segments = longest segments that are not the barcode segment
    std::vector<int> order;
    for (size_t s = 0; s < nseg; ++s)
        if ((int)s != ev.bc_seg && !ev.segs[s].index_like) order.push_back((int)s);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return ev.segs[a].len_mode > ev.segs[b].len_mode; });
    if (!order.empty()) ev.cdna_seg = order[0];
    if (order.size() > 1) ev.cdna2_seg = order[1];

    // windows on the barcode candidate (or the first segment of a single-read run)
    const int wseg = ev.bc_seg >= 0 ? ev.bc_seg : (nseg == 1 ? 0 : -1);
    if (wseg >= 0 && best_rate < kThr.wl_hit) ev.window = scan_windows(spots, wseg);

    // ScreenSpot view: r1 = barcode segment, r2 = cDNA segment (screen conventions)
    std::vector<ScreenSpot> sv;
    sv.reserve(std::min<size_t>(spots.size(), 200000));
    const int r1s = ev.bc_seg >= 0 ? ev.bc_seg : (ev.cdna_seg >= 0 ? ev.cdna_seg : 0);
    const int r2s = ev.bc_seg >= 0 ? ev.cdna_seg : ev.cdna2_seg;
    for (size_t i = 0; i < std::min<size_t>(spots.size(), 200000); ++i) {
        ScreenSpot s;
        if ((size_t)r1s < spots[i].seg.size()) { s.r1_seq = num(spots[i].seg[r1s]); s.r1_len = (uint16_t)std::min<size_t>(s.r1_seq.size(), 65535); }
        if (r2s >= 0 && (size_t)r2s < spots[i].seg.size()) { s.r2_seq = num(spots[i].seg[r2s]); s.r2_len = (uint16_t)std::min<size_t>(s.r2_seq.size(), 65535); }
        sv.push_back(std::move(s));
    }
    const uint16_t r1m = sv.empty() ? 0 : (uint16_t)ev.segs[r1s].len_mode, r2m = (r2s >= 0) ? (uint16_t)ev.segs[r2s].len_mode : 0;

    // C: composition and constants on the cDNA read(s)
    AssaySignals<ScreenSpot> sig = AssaySignals<ScreenSpot>::compute(sv, r1m, r2m);
    ev.tn5 = sig.tn5_fraction; ev.polyt = sig.polyt_fraction; ev.adt_cap = sig.adt_fraction;
    ev.gc = sig.gc_fraction; ev.c_share = sig.c_fraction; ev.n_share = sig.n_fraction;

    // 10x family: five-prime motifs and feature-barcode constants through the screen, on the called list only
    const ListHit* top = nullptr;
    for (const auto& h : ev.lists)
        if (h.one_mm >= kThr.wl_hit && h.family != "atac" && h.family != "arc_atac" && (!top || h.one_mm > top->one_mm)) top = &h;
    if (top && r2s >= 0 && !sv.empty()) {
        std::vector<ScreenSpot> v2 = sv;
        std::vector<ScreenWhitelist> one;
        for (const auto& wl : lists) if (wl.file == top->file) one.push_back(wl);
        auto rep = ::singlet::fq::screen_spots(std::move(v2), one, opt.whitelist_dirs);
        ev.five_prime = rep.five_prime;
        ev.five_prime_called = rep.is_5prime;
        ev.top100_off0 = rep.r2_top100_share_off0; ev.top100_off10 = rep.r2_top100_share_off10;
        ev.tag_polya = rep.r2_tag_polya_share; ev.capture = rep.r2_capture_share; ev.scaffold = rep.r2_scaffold_share;
        ev.feature_reason = rep.feature_reason; ev.feature_like = rep.feature_barcode_like;
    } else if (!sv.empty()) {
        // feature-barcode constants without a whitelist hit are reported but do not call a family on their own
        std::vector<ScreenSpot> v2 = sv;
        auto rep = ::singlet::fq::screen_spots(std::move(v2), {}, opt.whitelist_dirs);
        ev.top100_off0 = rep.r2_top100_share_off0; ev.top100_off10 = rep.r2_top100_share_off10;
        ev.tag_polya = rep.r2_tag_polya_share; ev.capture = rep.r2_capture_share; ev.scaffold = rep.r2_scaffold_share;
        ev.feature_reason = rep.feature_reason; ev.feature_like = rep.feature_barcode_like;
        ev.detected = detect_protocol(std::vector<ScreenSpot>(sv.begin(), sv.begin() + std::min<size_t>(sv.size(), 20000)), r1m, r2m,
                                      opt.whitelist_dirs);
    }

    // Smart-seq3 tag at the read start of any segment
    {
        uint64_t n = 0, hit = 0;
        for (size_t i = 0; i < std::min<size_t>(spots.size(), 50000); ++i)
            for (const auto& q : spots[i].seg) {
                if (q.size() < 30) continue;
                ++n;
                hit += has_motif(q, "ATTGCGCAATG", 1, 0, 8);
            }
        ev.smart3_tag = n ? double(hit) / n : 0;
    }
    // repeat share of the first 16 bases at the barcode candidate: barcoded libraries repeat, random inserts rarely do
    if (ev.bc_seg >= 0) {
        std::unordered_map<uint64_t, uint32_t> c;
        uint64_t n = 0, rep = 0;
        for (size_t i = 0; i < probe_n; ++i) {
            if ((size_t)ev.bc_seg >= spots[i].seg.size() || spots[i].seg[ev.bc_seg].size() < 16) continue;
            uint64_t h;
            if (!pack(spots[i].seg[ev.bc_seg].data(), 16, h)) continue;
            ++n; ++c[h];
        }
        for (auto& kv : c) if (kv.second > 1) rep += kv.second;
        ev.repeat_share_bc = n ? double(rep) / n : 0;
    }
}

// ── interpretations ──

namespace detail_id {

inline const ListHit* find_hit(const Evidence& ev, const char* fam) {
    for (const auto& h : ev.lists)
        if (h.family == fam) return &h;
    return nullptr;
}

inline double conf_support(Confidence c) {
    switch (c) {
        case Confidence::HIGH: case Confidence::MANUAL: case Confidence::FORCE: return 0.85;
        case Confidence::MEDIUM: return 0.65;
        case Confidence::LOW: return 0.35;
        default: return 0.0;
    }
}

// Plate protocol tag from the window geometry (measured on Mereu 2020 libraries, see protocol.h forced-only specs).
inline std::string plate_tag(const WindowHit& w, uint32_t seg_len) {
    if (w.off == 6 && w.len == 8) return "celseq2-u6b8";
    if (w.off == 0 && w.len == 6 && seg_len >= 15 && seg_len <= 17) return "scrbseq";
    if (w.off == 0 && w.len == 11 && seg_len >= 24 && seg_len <= 27) return "icell8";
    if (w.off == 0 && w.len == 6 && seg_len <= 13) return "celseq2";
    if (w.off == 0 && w.len == 7 && seg_len >= 14 && seg_len <= 16) return "marsseq2";
    return "plate-unknown";
}

inline std::string family_of_tag(const std::string& tag) {
    if (tag == "dropseq" || tag == "seqwell" || tag == "indrop" || tag == "dnbelab-c4" || tag == "ddseq" || tag == "microwell-seq" ||
        tag == "surecell" || tag == "bd-rhapsody")
        return "droplet_other";
    if (tag == "sci-rna-seq3" || tag == "splitseq") return "combinatorial";
    if (tag == "celseq2" || tag == "marsseq2" || tag == "quartzseq2" || tag == "strtseq" || tag == "scrbseq" || tag == "icell8" ||
        tag == "celseq2-u6b8")
        return "plate_barcoded";
    return "";
}

inline void add_usefulness(Interp& it, const Evidence& ev, double valid_bc) {
    it.components.push_back({"valid_barcode", valid_bc});
    double u = valid_bc;
    if (ev.map.measured) {
        it.components.push_back({"uniquely_mapped", ev.map.unique});
        u *= ev.map.unique;
        if (ev.map.exonic_measured) { it.components.push_back({"gene_assigned", ev.map.exonic}); u *= ev.map.exonic; }
        else it.unmeasured.push_back("gene_assigned");
    } else {
        it.unmeasured.push_back("uniquely_mapped");
        it.unmeasured.push_back("gene_assigned");
    }
    it.unmeasured.push_back("non_duplicate");   // needs a whole run; a head has too few repeats to estimate it
    it.usefulness = u;
}

}  // namespace detail_id

inline void rank(Report& R, const Options& opt) {
    using namespace detail_id;
    const Evidence& ev = R.ev;
    std::vector<Interp> H;
    R.hint = opt.hint;
    auto hit = [&](const char* f) { return find_hit(ev, f); };
    const ListHit *v3 = hit("v3"), *v2 = hit("v2"), *arc = hit("arc"), *v1 = hit("v1"), *vis = hit("visium"), *atac = hit("atac"),
                  *a_arc = hit("arc_atac"), *v43 = hit("v4_3p"), *v45 = hit("v4_5p");
    auto rate = [](const ListHit* h) { return h ? h->one_mm : 0.0; };

    // ── 10x barcode lists ──
    // resolve the GEX list: the 737K lists (v2, arc) sit inside the 3M list, so a v3 library hits them in about a fifth of reads
    std::string gkey;
    double grate = 0;
    const ListHit* ghit = nullptr;
    auto consider = [&](const char* key, const ListHit* h) {
        if (h && h->one_mm >= kThr.wl_hit && h->one_mm > grate) { gkey = key; grate = h->one_mm; ghit = h; }
    };
    consider("v3", v3); consider("v1", v1); consider("v4_3p", v43); consider("v4_5p", v45);
    const double big = rate(v3);
    for (const auto& [key, h] : {std::pair<const char*, const ListHit*>{"v2", v2}, {"arc", arc}}) {
        const double r = rate(h);
        if (r >= kThr.wl_hit && (big < kThr.wl_hit || r >= kThr.v2_over_v3 * big) && (gkey.empty() || gkey == "v3" || r > grate)) {
            gkey = key; grate = r; ghit = h;
        }
    }
    if (rate(vis) >= kThr.wl_hit && rate(vis) > grate) { gkey = "visium"; grate = rate(vis); ghit = vis; }

    bool atac_hit = rate(atac) >= kThr.wl_hit, arc_atac_hit = rate(a_arc) >= kThr.wl_hit;
    if (arc_atac_hit && rate(a_arc) >= rate(atac)) atac_hit = false;   // the ARC ATAC list is a different set from the scATAC list
    else arc_atac_hit = false;

    if (!gkey.empty()) {
        // 3' or 5': strand evidence first when it decides (sense reads: 3'; antisense: 5'), then the motif evidence
        bool five = ev.five_prime_called;
        std::string basis = "5' motifs (TSO tail after the barcode read, poly(A)/poly(T) in the cDNA read)";
        if (ev.map.measured && ev.map.sense >= 0) {
            if (ev.map.sense >= kThr.sense_3p) { five = false; basis = "sense fraction " + detail::jnum(ev.map.sense) + " of mapped reads"; }
            else if (ev.map.sense <= kThr.sense_5p) { five = true; basis = "sense fraction " + detail::jnum(ev.map.sense) + " of mapped reads"; }
        }
        Interp g;
        g.modality = "gex";
        std::string tag;
        if (gkey == "visium") { g.family = "visium"; tag = "10x-visium"; g.modality = "spatial_gex"; }
        else if (gkey == "arc") { g.family = "multiome_gex"; tag = "10x-arc-gex"; }
        else if (gkey == "v1") { g.family = "droplet_3p"; tag = "10x-3p-v1"; }
        else if (gkey == "v4_3p") { g.family = "droplet_3p"; tag = "10x-3p-v4"; }
        else if (gkey == "v4_5p") { g.family = "droplet_5p"; tag = "10x-5p-v4"; }
        else { g.family = five ? "droplet_5p" : "droplet_3p"; tag = std::string("10x-") + (five ? "5p-" : "3p-") + gkey; }
        g.protocol = tag;
        g.reasons.push_back("barcode list " + gkey + " hit by " + detail::jnum(grate) + " of reads at segment " + std::to_string(ghit ? ghit->seg : -1));
        if (gkey == "v3" || gkey == "v2") g.reasons.push_back("3'/5' call from " + basis);
        g.support = clamp01(grate * (ev.feature_like && gkey != "visium" ? 0.3 : 1.0));
        add_usefulness(g, ev, grate);
        H.push_back(g);
        if (ev.feature_like && gkey != "visium") {
            Interp f;
            f.family = "feature_barcode";
            f.protocol = tag;
            if (ev.scaffold >= kThr.feature_const) f.modality = "crispr_guide";
            else if (ev.capture >= kThr.feature_const) f.modality = "feature_totalseq_bc";
            else f.modality = "feature_totalseq_a_or_hto";
            f.reasons.push_back("feature-barcode evidence: " + ev.feature_reason);
            f.support = clamp01(grate * 0.95);
            add_usefulness(f, ev, grate);
            H.push_back(f);
        }
    }
    if (atac_hit || arc_atac_hit) {
        Interp a;
        a.family = arc_atac_hit ? "multiome_atac" : "atac";
        a.protocol = arc_atac_hit ? "10x-arc-atac" : "10x-atac";
        a.modality = "chromatin_accessibility";
        const ListHit* h = arc_atac_hit ? a_arc : atac;
        a.reasons.push_back("ATAC barcode list hit by " + detail::jnum(h->one_mm) + " of reads (segment " + std::to_string(h->seg) +
                            (h->revcomp ? ", reverse complement)" : ")"));
        a.support = clamp01(h->one_mm);
        a.usefulness = 0;
        a.unmeasured.push_back("not processed");
        H.push_back(a);
    }

    // ── non-whitelist barcoded libraries ──
    const bool any_list = !gkey.empty() || atac_hit || arc_atac_hit;
    if (!any_list) {
        const std::string tag = ev.detected.tag;
        const std::string fam = family_of_tag(tag);
        if (!fam.empty() && ev.detected.confidence >= Confidence::LOW) {
            Interp d;
            d.family = fam; d.protocol = tag; d.modality = "gex";
            d.support = conf_support(ev.detected.confidence);
            d.reasons.push_back(std::string("protocol detector: ") + tag + " (" + conf_name(ev.detected.confidence) + " confidence)");
            add_usefulness(d, ev, ev.detected.wl_match_rate > 0 ? ev.detected.wl_match_rate : 1.0);
            H.push_back(d);
        } else if (tag == "agnostic" && ev.detected.confidence >= Confidence::LOW) {
            Interp d;
            d.family = "unknown"; d.protocol = "barcoded-unspecified"; d.modality = "gex";
            d.support = 0.7 * conf_support(ev.detected.confidence);
            d.reasons.push_back("layout detector found barcode-like structure without a known protocol");
            H.push_back(d);
        }
        if (ev.window.found) {
            Interp w;
            const uint32_t sl = ev.segs[ev.window.seg].len_mode;
            w.protocol = plate_tag(ev.window, sl);
            w.family = "plate_barcoded"; w.modality = "gex";
            w.support = clamp01(ev.window.top_share * (1.0 - ev.window.random_share));
            w.reasons.push_back("top " + std::to_string(ev.window.k) + " " + std::to_string(ev.window.len) + "-mers at offset " +
                                std::to_string(ev.window.off) + " hold " + detail::jnum(ev.window.top_share) + " of reads (random expectation " +
                                detail::jnum(ev.window.random_share) + "): well barcodes");
            add_usefulness(w, ev, ev.window.top_share);
            H.push_back(w);
        }
        if (ev.smart3_tag >= kThr.smart3_tag) {
            Interp s;
            s.family = "smartseq3"; s.protocol = "smartseq3"; s.modality = "gex";
            s.support = clamp01(ev.smart3_tag * 5.0);
            s.reasons.push_back("Smart-seq3 5' tag at the read start in " + detail::jnum(ev.smart3_tag) + " of reads");
            H.push_back(s);
        }
        // ── no cell barcode ──
        const bool sequence_like = ev.n_share <= kThr.n_share;
        if (ev.n_share > kThr.n_share) {
            Interp n; n.family = "not_sequence"; n.protocol = "none"; n.modality = "none"; n.support = 0.95;
            n.reasons.push_back("N share " + detail::jnum(ev.n_share));
            H.push_back(n);
        } else if (ev.c_share > 0.001 && ev.c_share < kThr.c_share_methyl) {
            Interp m; m.family = "methylome"; m.protocol = "none"; m.modality = "methylation"; m.support = 0.8;
            m.reasons.push_back("C share " + detail::jnum(ev.c_share) + " (bisulfite conversion)");
            H.push_back(m);
        }
        const bool no_struct = H.empty() || std::all_of(H.begin(), H.end(), [](const Interp& i) { return i.support < 0.5; });
        uint32_t longest = 0;
        for (const auto& s : ev.segs) longest = std::max(longest, s.len_mode);
        if (sequence_like && no_struct && longest >= 36) {
            const bool small = opt.run_spots >= 0 && (uint64_t)opt.run_spots <= kThr.smartseq_max_spots;
            const bool large = opt.run_spots >= 0 && (uint64_t)opt.run_spots > kThr.smartseq_max_spots;
            Interp s, b;
            s.family = "smartseq"; s.protocol = "smartseq2-like"; s.modality = "gex";
            b.family = "bulk_rna"; b.protocol = "bulk"; b.modality = "gex";
            s.support = small ? 0.62 : (large ? 0.38 : 0.5);
            b.support = small ? 0.38 : (large ? 0.62 : 0.5);
            const std::string sz = opt.run_spots >= 0 ? ("run holds " + std::to_string(opt.run_spots) + " spots") : "run size unknown";
            s.reasons.push_back("no cell barcode, long reads; " + sz + " (full-length single-cell and bulk libraries look alike in reads)");
            b.reasons = s.reasons;
            if (ev.tn5 >= kThr.tn5) {
                s.reasons.push_back("Tn5 end in " + detail::jnum(ev.tn5) + " of reads: tagmented, as Smart-seq2 and many bulk kits are");
                Interp a;
                a.family = "atac"; a.protocol = "tn5-genomic"; a.modality = "chromatin_accessibility";
                a.support = 0.45;
                a.reasons.push_back("Tn5 end in " + detail::jnum(ev.tn5) + " of reads and no barcode: ATAC or tagmented RNA");
                a.unmeasured.push_back("not processed");
                H.push_back(a);
            }
            if (!ev.map.measured) {
                Interp g;
                g.family = "genomic_dna"; g.protocol = "none"; g.modality = "none"; g.support = 0.15;
                g.reasons.push_back("not excluded without mapping evidence");
                H.push_back(g);
            }
            add_usefulness(s, ev, 0.0);
            s.usefulness = 0;
            s.unmeasured.push_back("no cell barcode: single-cell usefulness depends on the plate layout");
            H.push_back(s);
            H.push_back(b);
        }
    }

    // ── layer D adjusts support ──
    if (ev.map.measured) {
        for (auto& it : H) {
            const bool rna = it.family == "droplet_3p" || it.family == "droplet_5p" || it.family == "multiome_gex" || it.family == "visium" ||
                             it.family == "droplet_other" || it.family == "combinatorial" || it.family == "plate_barcoded" ||
                             it.family == "smartseq" || it.family == "smartseq3" || it.family == "bulk_rna";
            const bool dna = it.family == "atac" || it.family == "multiome_atac" || it.family == "genomic_dna";
            if (ev.map.unique < kThr.mapped_min && (rna || dna)) {
                it.support *= 0.5;
                it.reasons.push_back("uniquely mapped share " + detail::jnum(ev.map.unique) + " is below " + detail::jnum(kThr.mapped_min));
            }
            if (ev.map.exonic_measured) {
                if (rna && ev.map.exonic < kThr.exonic_rna_min) {
                    it.support *= 0.3;
                    it.reasons.push_back("gene-assigned share " + detail::jnum(ev.map.exonic) + " is low for RNA");
                } else if (dna && ev.map.exonic > kThr.exonic_dna_max) {
                    it.support *= 0.2;
                    it.reasons.push_back("gene-assigned share " + detail::jnum(ev.map.exonic) + " is high for genomic reads");
                }
            }
        }
        if (ev.map.exonic_measured && ev.map.exonic <= kThr.exonic_dna_max && ev.map.unique >= kThr.mapped_min && !any_list &&
            std::none_of(H.begin(), H.end(), [](const Interp& i) { return i.family == "genomic_dna" || i.family == "atac"; })) {
            Interp g;
            g.family = "genomic_dna"; g.protocol = "none"; g.modality = "none";
            g.support = 0.7;
            g.reasons.push_back("reads map but few land in genes (" + detail::jnum(ev.map.exonic) + ")");
            H.push_back(g);
        }
    }

    if (H.empty()) {
        Interp u; u.family = "unknown"; u.protocol = "none"; u.modality = "none"; u.support = 0.1;
        u.reasons.push_back("no list, barcode structure or insert signature matched");
        H.push_back(u);
    }
    std::stable_sort(H.begin(), H.end(), [](const Interp& a, const Interp& b) { return a.support > b.support; });
    R.ranked = H;

    // verdict: the top interpretation, unless the best one of another family is within the margin
    const Interp& top = H[0];
    const Interp* rival = nullptr;
    for (size_t i = 1; i < H.size(); ++i)
        if (H[i].family != top.family) { rival = &H[i]; break; }
    if (top.support < kThr.support_min) {
        R.verdict = "unknown"; R.verdict_protocol = "none";
    } else if (rival && top.support - rival->support < kThr.margin && rival->support >= kThr.support_min) {
        R.verdict = "ambiguous"; R.verdict_protocol = "none";
        R.ambiguous_with = top.family + " | " + rival->family;
    } else {
        R.verdict = top.family; R.verdict_protocol = top.protocol;
    }
    const Policy& p = policy_for(R.verdict);
    R.action = p.action;
    R.action_note = p.note;
    // a metadata hint is shown and never changes the verdict; it only selects the action inside an ambiguous full-length RNA call
    if (R.verdict == "ambiguous" && R.ambiguous_with.find("smartseq") != std::string::npos && R.ambiguous_with.find("bulk_rna") != std::string::npos &&
        opt.hint.find("assay=smartseq") != std::string::npos) {
        R.action = "defer";
        R.action_note = "hint assay=smartseq2 (metadata, not read evidence) selects defer; reads alone cannot separate Smart-seq from bulk";
    }
}

// ── JSON ──

inline std::string report_json(const Report& R) {
    using namespace detail;
    std::ostringstream o;
    auto q = [](const std::string& s) { return "\"" + jesc(s) + "\""; };
    const auto& ev = R.ev;
    o << "{\n  \"schema_version\": " << R.schema << ",\n";
    o << "  \"source\": {\"source\": " << q(R.src.source) << ", \"route\": " << q(R.src.route) << ", \"endpoint\": " << q(R.src.endpoint)
      << ", \"bytes\": " << (R.src.bytes >= 0 ? std::to_string(R.src.bytes) : std::string("null")) << ", \"bytes_note\": " << q(R.src.bytes_note)
      << ", \"seconds\": " << jnum(R.src.seconds) << ", \"spots\": " << R.src.spots << ", \"run_spots\": "
      << (R.src.run_spots >= 0 ? std::to_string(R.src.run_spots) : std::string("null")) << ", \"error\": " << q(R.src.error) << "},\n";
    o << "  \"verdict\": " << q(R.verdict) << ",\n  \"protocol\": " << q(R.verdict_protocol) << ",\n  \"ambiguous_with\": " << q(R.ambiguous_with)
      << ",\n  \"action\": " << q(R.action) << ",\n  \"action_note\": " << q(R.action_note) << ",\n  \"hint\": " << q(R.hint) << ",\n";
    o << "  \"layout\": [";
    for (size_t i = 0; i < ev.segs.size(); ++i) {
        const auto& s = ev.segs[i];
        o << (i ? ", " : "") << "{\"segment\": " << i << ", \"len_mode\": " << s.len_mode << ", \"len_min\": " << s.len_min << ", \"len_max\": " << s.len_max
          << ", \"mode_share\": " << jnum(s.mode_share) << ", \"entropy_bits\": " << jnum(s.entropy_first32) << ", \"index_like\": " << (s.index_like ? "true" : "false") << "}";
    }
    o << "],\n  \"roles\": {\"barcode_segment\": " << ev.bc_seg << ", \"cdna_segment\": " << ev.cdna_seg << ", \"cdna2_segment\": " << ev.cdna2_seg << "},\n";
    o << "  \"lists\": [";
    bool first = true;
    for (const auto& h : ev.lists) {
        o << (first ? "" : ", ") << "{\"file\": " << q(h.file) << ", \"family\": " << q(h.family) << ", \"loaded\": " << (h.loaded ? "true" : "false")
          << ", \"segment\": " << h.seg << ", \"revcomp\": " << (h.revcomp ? "true" : "false") << ", \"exact\": " << jnum(h.exact) << ", \"one_mismatch\": " << jnum(h.one_mm) << "}";
        first = false;
    }
    o << "],\n  \"window\": {\"found\": " << (ev.window.found ? "true" : "false") << ", \"segment\": " << ev.window.seg << ", \"offset\": " << ev.window.off
      << ", \"length\": " << ev.window.len << ", \"k\": " << ev.window.k << ", \"top_share\": " << jnum(ev.window.top_share) << ", \"random_share\": " << jnum(ev.window.random_share) << "},\n";
    o << "  \"detector\": {\"tag\": " << q(ev.detected.tag) << ", \"confidence\": " << q(conf_name(ev.detected.confidence)) << ", \"wl_rate\": " << jnum(ev.detected.wl_match_rate) << "},\n";
    o << "  \"insert\": {\"five_prime_called\": " << (ev.five_prime_called ? "true" : "false") << ", \"r1_tso\": " << ev.five_prime.r1_tso << ", \"r1_polyt\": " << ev.five_prime.r1_polyt
      << ", \"r2_smart\": " << ev.five_prime.r2_smart << ", \"r2_polya\": " << ev.five_prime.r2_polya << ", \"r2_polyt\": " << ev.five_prime.r2_polyt << ", \"motif_spots\": " << ev.five_prime.n
      << ", \"tn5\": " << jnum(ev.tn5) << ", \"polyt_head\": " << jnum(ev.polyt) << ", \"adt_capture_a\": " << jnum(ev.adt_cap) << ", \"gc\": " << jnum(ev.gc) << ", \"c_share\": " << jnum(ev.c_share)
      << ", \"n_share\": " << jnum(ev.n_share) << ", \"smart3_tag\": " << jnum(ev.smart3_tag) << ", \"top100_off0\": " << jnum(ev.top100_off0) << ", \"top100_off10\": " << jnum(ev.top100_off10)
      << ", \"tag_polya\": " << jnum(ev.tag_polya) << ", \"capture_bc\": " << jnum(ev.capture) << ", \"scaffold\": " << jnum(ev.scaffold) << ", \"feature_reason\": " << q(ev.feature_reason)
      << ", \"bc_repeat_share\": " << jnum(ev.repeat_share_bc) << "},\n";
    o << "  \"mapping\": {\"measured\": " << (ev.map.measured ? "true" : "false") << ", \"note\": " << q(ev.map.note) << ", \"reads\": " << ev.map.reads << ", \"unique\": " << jnum(ev.map.unique)
      << ", \"multi\": " << jnum(ev.map.multi) << ", \"unmapped\": " << jnum(ev.map.unmapped) << ", \"gene_assigned\": " << jnum(ev.map.exonic) << ", \"sense\": " << jnum(ev.map.sense)
      << ", \"chrM\": " << jnum(ev.map.chrm) << ", \"seconds\": " << jnum(ev.map.seconds) << "},\n";
    o << "  \"interpretations\": [";
    for (size_t i = 0; i < R.ranked.size(); ++i) {
        const auto& it = R.ranked[i];
        o << (i ? ",\n    " : "\n    ") << "{\"rank\": " << i + 1 << ", \"family\": " << q(it.family) << ", \"protocol\": " << q(it.protocol) << ", \"modality\": " << q(it.modality)
          << ", \"support\": " << jnum(it.support) << ", \"usefulness\": " << jnum(it.usefulness) << ", \"components\": {";
        for (size_t k = 0; k < it.components.size(); ++k) o << (k ? ", " : "") << q(it.components[k].first) << ": " << jnum(it.components[k].second);
        o << "}, \"unmeasured\": " << jlist(it.unmeasured) << ", \"reasons\": " << jlist(it.reasons) << "}";
    }
    o << "\n  ]\n}\n";
    return o.str();
}

/// One call for tests and callers that already hold the head: layers A to C, the interpretations and the verdict.
inline Report identify_spots(const std::vector<HeadSpot>& spots, const Options& opt, std::vector<ScreenWhitelist>& lists, const MapEvidence* map = nullptr) {
    Report R;
    R.spots = spots.size();
    R.src.spots = spots.size();
    measure(spots, opt, lists, R.ev);
    if (map) R.ev.map = *map;
    rank(R, opt);
    return R;
}
}  // namespace singlet::fq::identify
