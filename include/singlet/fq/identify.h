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
#include "singlet/pileup/bloom_filter.h"
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
// Record version of the JSON report (the validation engine reads this); 2 adds protocol_tag, sub_call, species and policy_source.
constexpr int kRecordVersion = 2;

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
    // Barcode sets larger than a plate (Drop-seq, Seq-Well): the first 12 bases of the barcode read repeat at least five times in this share of
    // reads (200000-read heads of 5,000 to 10,000 cells); random cDNA starts repeat in a small share.
    double prefix_mult = 0.60;
    // cDNA reads starting at repeated 25-mers (probe sites) in at least this share, on reads of 60 bases or fewer: probe-based library.
    double probe_start = 0.50;
    // V(D)J enrichment: at least this share of gene-assigned reads on IG or TR genes (5' GEX libraries hold a few percent).
    double vdj_share = 0.40;
    // Tag-based single-cell cDNA without its barcode read: at least this share of reads on transcripts of 4000 nt or more lie within 1000 nt of
    // one transcript end, at least this many times what even coverage gives, with the strand that goes with that end (3': sense, 5': antisense).
    // Even coverage (bulk, Smart-seq): under even_end_share at either end. 0.25 was tried on the development panel and rejected: bulk fell to 2 of 3
    // and four more 10x runs without a barcode read were called bulk (7 instead of 3 of 27).
    double tag_end_share = 0.35, tag_end_enrich = 3.0, even_end_share = 0.15;
    // Species block (validation/references/SPECIES_BARS.md, rule v0): a filter calls when its hit rate (share of canonical 21-mers of cDNA reads
    // present in it) is at least species_min_rate and species_ratio times the runner-up. A mixture needs the minor species to hold
    // species_mix_share of the species-specific reads and species_mix_reads reads (a read is specific to A with species_mix_excess more hits).
    double species_min_rate = 0.02, species_ratio = 1.25, species_mix_share = 0.15;
    int species_mix_reads = 50, species_mix_excess = 10;
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
    // from the alignments and the index's transcript table (TxAnnot); unique alignments on annotated transcripts
    bool tx_measured = false;
    uint64_t tx_reads = 0;        // unique reads that overlap a transcript on one orientation
    double sense_span = -1;       // sense share over those reads (introns included)
    double end3 = 0, end5 = 0;    // share within 1000 nt of the transcript 3' end, and of the 5' end (transcripts of 4000 nt or more)
    double end_expected = 0;      // share expected from even coverage on the same transcripts
    uint64_t end_reads = 0;
    double profile[10] = {};      // read midpoints along transcripts of 5000 nt or more, 5' to 3', shares per tenth
    double insert_median = 0;     // median proper-pair insert, 0 when single-end
    uint64_t insert_n = 0;
    double vdj = 0;               // share of gene-assigned reads on IG or TR genes
    double rrna = 0;              // share of gene-assigned reads on annotated rRNA genes
    bool paired = false;
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
    int mate_seg = -1;               // second insert read of a pair with no barcode read (mapped as a pair), else -1
    // B
    std::vector<ListHit> lists;      // best (segment, orientation) per list
    WindowHit window;
    WindowHit window2;               // a second, disjoint barcode window: split-pool bead barcodes (inDrop, ddSEQ), not a plate
    bool window_on_index = false;    // the window sits in a separate index read: well indices of a plate with full-length reads
    double prefix_mult = 0;          // share of reads whose first 12 bases repeat (at least 5 times) at the best segment
    int polyt_seg = -1, polyt_off = 0;   // a poly(T) run that starts at one fixed offset after a varied prefix: bead-barcode read of Drop-seq geometry
    int prefix_seg = -1;
    double cdna_start_repeat = 0;    // share of cDNA reads whose first 25 bases occur at least three times: probe panels repeat, random fragments do not
    uint32_t cdna_len = 0;
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

struct EndpointRec {
    std::string name, note, verdict, protocol;
    bool reachable = false, barcode_read = false;
    std::vector<uint32_t> seg_lens;
    double list_rate = 0;
    int64_t bytes = -1;
    double seconds = 0;
};

struct SpeciesHit {
    std::string tag, species, group;
    double rate = 0;      // share of canonical 21-mers of the scored reads present in the filter
};
struct SpeciesBlock {
    bool measured = false;
    std::vector<SpeciesHit> hits;       // the top two by rate
    bool mixture = false;
    std::string minor_species;          // minor species of the best split outside the top hit's close group
    double minor_share = 0;             // its share of the species-specific reads
    int minor_reads = 0;
    std::string call;                   // CALL | CLOSE_PAIR | MIXTURE | AMBIGUOUS | NONE
    int filters_scored = 0, filters_available = 0;
    uint64_t reads_scored = 0;
    bool short_reads = false, screened_by_hint = false;
    std::string note;
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
    std::vector<EndpointRec> endpoints;   // --endpoints: every endpoint of the run, with whether it carries a barcode read
    std::string protocol_tag;                      // registry id (validation/routes.yaml), empty when the verdict has none (JSON null)
    std::vector<std::string> protocol_tag_alternatives;   // ids the reads cannot separate from protocol_tag
    std::string policy_source = "builtin";         // "builtin" | "route-status"
    struct Hint { std::string key, value, effect; };       // hints[]: caller metadata, with what it changed (empty: nothing)
    std::vector<Hint> hints;
    // sub_call: version-level calls; an empty value is JSON null
    std::string sub_strand, sub_visium, sub_vdj, sub_indrop_version, sub_bd_bead, sub_parse_kit;
    std::string sub_call_reason;
    SpeciesBlock species;   // measured only with --species-filters
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
        {"rna_no_barcode", "defer", "RNA reads without a barcode read and without mapping evidence that settles tag, Smart-seq or bulk"},
        {"barcode_read_missing", "defer", "tag-based single-cell cDNA whose barcode read is absent from this endpoint; look for it in another endpoint"},
        {"flex", "defer", "probe-based single-cell RNA (10x Flex): recognized, not processed"},
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

/// Reads a flat JSON map of registry id to status ({"10x-3p-v3": "validated", ...}). Returns false when no pair is found.
inline bool parse_route_status(const std::string& text, std::unordered_map<std::string, std::string>& out) {
    std::vector<std::string> strs;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '"') continue;
        std::string v;
        for (++i; i < text.size() && text[i] != '"'; ++i) { if (text[i] == '\\' &&i + 1 < text.size()) ++i; v += text[i]; }
        strs.push_back(v);
    }
    for (size_t i = 0; i + 1 < strs.size(); i += 2) out[strs[i]] = strs[i + 1];
    return !out.empty();
}

// ── registry ids: one table from the internal protocol tags to the family ids of validation/routes.yaml ──

struct TagCall {
    std::string id;                      // empty: the verdict has no registry id
    std::vector<std::string> alternatives;
};

inline TagCall registry_tag(const std::string& family, const std::string& tag, const std::string& modality) {
    static const std::unordered_map<std::string, std::string> m = {
        {"10x-3p-v1", "10x-3p-v1"}, {"10x-3p-v2", "10x-3p-v2"}, {"10x-3p-v3", "10x-3p-v3"}, {"10x-3p-v4", "10x-3p-v4"},
        {"10x-5p-v1", "10x-5p-v1"}, {"10x-5p-v2", "10x-5p-v2"}, {"10x-5p-v3", "10x-5p-v3"},
        {"10x-5p-v4", "10x-5p-v3"},   // GEM-X 5' is listed under v3 in the registry
        {"10x-arc-gex", "10x-arc-gex"}, {"10x-visium", "10x-visium"}, {"10x-atac", "scatac"}, {"10x-arc-atac", "10x-arc-atac"},
        {"dropseq", "dropseq"}, {"dropseq-like", "dropseq"}, {"seqwell", "seqwell"}, {"indrop", "indrop"}, {"bd-rhapsody", "bd-rhapsody"},
        {"microwell-seq", "microwell-seq"}, {"dnbelab-c4", "dnbelab-c4"}, {"singleron", "singleron"}, {"pipseq", "pipseq"},
        {"ddseq", "ddseq"}, {"surecell", "ddseq"}, {"splitseq", "splitseq-parse"}, {"sci-rna-seq3", "sci-rna-seq3"},
        {"celseq2", "celseq2"}, {"celseq2-u6b8", "celseq2"}, {"marsseq2", "marsseq2"}, {"quartzseq2", "quartzseq2"}, {"strtseq", "strtseq"},
        {"scrbseq", "scrbseq"}, {"icell8", "icell8"}, {"smartseq2", "smartseq2"}, {"smartseq2-like", "smartseq2"}, {"smartseq3", "smartseq3"},
    };
    TagCall t;
    if (family == "bulk_rna" || family == "genomic_dna" || family == "methylome" || family == "not_sequence") { t.id = "generic-negative"; return t; }
    if (family == "flex") { t.id = "10x-flex"; return t; }
    if (family == "vdj") { t.id = "tcr"; t.alternatives = {"bcr"}; return t; }   // TR and IG genes are pooled in the mapping evidence
    if (family == "feature_barcode") {
        if (modality == "crispr_guide") t.id = "crispr-guide";
        else if (modality == "feature_totalseq_bc") t.id = "cite-adt";
        else { t.id = "cite-adt"; t.alternatives = {"cite-hashtag"}; }   // TotalSeq-A capture: ADT and hashtag libraries share it
        return t;
    }
    if (family == "barcode_read_missing" || family == "rna_no_barcode" || family == "unknown" || family == "ambiguous") return t;
    const auto it = m.find(tag);
    if (it == m.end()) return t;
    t.id = it->second;
    if (tag == "dropseq" || tag == "dropseq-like") t.alternatives = {"seqwell"};   // the two differ only by text
    if (tag == "seqwell") t.alternatives = {"dropseq"};
    return t;
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
inline WindowHit scan_windows(const std::vector<HeadSpot>& spots, int s, size_t max_reads = 50000, WindowHit* second = nullptr, bool on_index = false) {
    WindowHit best;
    const uint32_t L = [&] { std::map<size_t, int> m; for (size_t i = 0; i < std::min<size_t>(spots.size(), 2000); ++i) if ((size_t)s < spots[i].seg.size()) ++m[spots[i].seg[s].size()]; size_t b = 0; int c = 0; for (auto& [l, k] : m) if (k > c) { c = k; b = l; } return (uint32_t)b; }();
    if (L < 6) return best;
    const size_t n = std::min(spots.size(), max_reads);
    static const uint16_t Ks[3] = {96, 384, 1536};
    // per-position base entropy: a window must start and end on a varying position (a linker or poly(T) run is not part of a barcode)
    double pe[40] = {};
    {
        uint32_t cnt[40][4] = {};
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < 40 && (size_t)s < spots[i].seg.size() && k < spots[i].seg[s].size(); ++k) {
                const uint8_t b = nuc::ascii_to_num(spots[i].seg[s][k]);
                if (b < 4) ++cnt[k][b];
            }
        for (int k = 0; k < 40; ++k) pe[k] = detail_id::entropy_bits(cnt[k]);
    }
    std::vector<WindowHit> cands;
    for (uint16_t len = 12; len >= 6; --len) {
        for (uint16_t off = 0; off + len <= std::min<uint32_t>(L, 36); ++off) {
            if (pe[off] < 0.5 || pe[off + len - 1] < 0.5) continue;
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
                // chance that K random sequences over the window's varying positions hold this much (constant bases add nothing)
                int nvar = 0;
                for (uint16_t k = 0; k < len; ++k) nvar += (pe[off + k] >= 0.5);
                const double rnd = std::min(1.0, double(K) / std::pow(4.0, nvar));
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
                if (distinct > 1.5 * K) continue;   // a set larger than the K sequences counted is not a plate of K wells (Drop-seq, Seq-Well bead sets)
                if (distinct < (on_index ? 48u : 12u)) break;   // a linker or poly(T) run is one or a few sequences; sample indices are few, wells many
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
        if (second) {
            // a second window that does not touch the first, on the same rule
            std::vector<WindowHit> rest;
            for (const auto& c : cands)
                if (c.off >= best.off + best.len + 0u || c.off + c.len <= best.off) rest.push_back(c);
            uint32_t mind2 = UINT32_MAX;
            for (const auto& c : rest) mind2 = std::min(mind2, c.distinct);
            for (const auto& c : rest) {
                if (c.distinct > 1.3 * mind2) continue;
                if (!second->found || c.len > second->len || (c.len == second->len && c.off < second->off)) *second = c;
            }
            // Split-pool bead barcodes combine independently (96 x 96 sequences); the second window of a plate read is fixed by the first.
            // The joint distinct count at the sampled depth tells them apart.
            if (second->found && best.len <= 12 && second->len <= 12) {
                std::unordered_map<uint64_t, uint32_t> joint;
                for (size_t i = 0; i < n; ++i) {
                    if ((size_t)s >= spots[i].seg.size() || spots[i].seg[s].size() < static_cast<size_t>(std::max(best.off + best.len, second->off + second->len))) continue;
                    uint64_t h1, h2;
                    if (!detail_id::pack(spots[i].seg[s].data() + best.off, best.len, h1) || !detail_id::pack(spots[i].seg[s].data() + second->off, second->len, h2)) continue;
                    ++joint[(h1 << 24) | h2];
                }
                uint32_t jd = 0;
                uint64_t jn = 0;
                for (auto& kv : joint) { jd += (kv.second >= 2); jn += kv.second; }
                const double prod = double(best.distinct) * double(second->distinct);
                // under independence the pairs seen twice or more number prod * (1 - e^-l (1 + l)) with l = reads / prod
                const double lam = prod > 0 ? double(jn) / prod : 0;
                const double expect_jd = prod * (1.0 - std::exp(-lam) * (1.0 + lam));
                if (prod <= 0 || expect_jd < 5 || double(jd) < 0.5 * expect_jd) *second = WindowHit();
            }
        }
    }
    return best;
}

// ── layer D: bundled-STAR alignment of the head ──

/// Parse the outputs of a STAR run on the cDNA reads: Log.final.out, ReadsPerGene.out.tab and the SAM's reference names.
/// Transcript table of a STAR index (transcriptInfo.tab, geneInfo.tab, chrStart.txt, chrName.txt), for the strand and end-distance evidence.
struct TxAnnot {
    struct Tx { uint64_t s, e, emax; uint8_t str; uint32_t gene; };
    std::vector<Tx> tx;
    std::vector<uint64_t> chr_start;
    std::unordered_map<std::string, size_t> chr_idx;
    std::vector<std::string> biotype;
    bool loaded = false;
    bool load(const std::string& dir) {
        std::ifstream cn(dir + "/chrName.txt"), cs(dir + "/chrStart.txt"), ti(dir + "/transcriptInfo.tab"), gi(dir + "/geneInfo.tab");
        if (!cn || !cs || !ti) return false;
        std::string l;
        size_t k = 0;
        while (std::getline(cn, l)) chr_idx[l] = k++;
        while (std::getline(cs, l)) if (!l.empty()) chr_start.push_back(std::strtoull(l.c_str(), nullptr, 10));
        std::getline(ti, l);
        uint64_t mx = 0;
        while (std::getline(ti, l)) {
            std::istringstream is(l);
            std::string id;
            uint64_t s, e, em;
            int str;
            uint32_t exn, exi, gene;
            if (!(is >> id >> s >> e >> em >> str >> exn >> exi >> gene)) continue;
            mx = std::max(mx, e);
            tx.push_back({s, e, mx, static_cast<uint8_t>(str), gene});
        }
        if (gi) {
            std::getline(gi, l);
            while (std::getline(gi, l)) {
                std::istringstream is(l);
                std::string id, name, bt;
                is >> id >> name >> bt;
                biotype.push_back(bt);
            }
        }
        loaded = !tx.empty() && !chr_start.empty();
        return loaded;
    }
};

/// Strand, end-distance, coverage, insert, V(D)J and rRNA evidence from the SAM of the head's unique alignments.
inline void sam_features(const std::string& sam_path, const TxAnnot* ann, MapEvidence& m) {
    std::ifstream sam(sam_path);
    if (!sam) return;
    std::string line;
    uint64_t n = 0, mt = 0, assigned = 0, sense = 0, vdj = 0, rrna = 0, n3 = 0, n5 = 0, nend = 0;
    double expect = 0;
    double prof[10] = {};
    uint64_t nprof = 0;
    std::vector<int64_t> ins;
    std::vector<std::string> f;
    while (std::getline(sam, line)) {
        if (line.empty() || line[0] == '@') continue;
        f.clear();
        size_t a = 0;
        for (int k = 0; k < 9; ++k) {
            size_t b = line.find('\t', a);
            if (b == std::string::npos) break;
            f.push_back(line.substr(a, b - a));
            a = b + 1;
        }
        if (f.size() < 9) continue;
        const int flag = std::atoi(f[1].c_str());
        const std::string& rname = f[2];
        ++n;
        if (rname == "chrM" || rname == "MT" || rname == "M" || rname == "chrMT") ++mt;
        if (!ann || !ann->loaded) continue;
        if (line.find("NH:i:1\t") == std::string::npos && (line.size() < 6 || line.compare(line.size() - 6, 6, "NH:i:1") != 0)) continue;   // unique alignments only
        if ((flag & 0x2) && std::atoll(f[8].c_str()) > 0) ins.push_back(std::atoll(f[8].c_str()));
        if (flag & 0x80) continue;   // one mate per pair for the transcript statistics
        auto ci = ann->chr_idx.find(rname);
        if (ci == ann->chr_idx.end() || ci->second >= ann->chr_start.size()) continue;
        const uint64_t g0 = ann->chr_start[ci->second] + std::strtoull(f[3].c_str(), nullptr, 10) - 1;
        uint64_t rl = 0, num = 0;
        for (char c : f[5]) {
            if (c >= '0' && c <= '9') num = num * 10 + static_cast<uint64_t>(c - '0');
            else { if (c == 'M' || c == 'D' || c == 'N' || c == '=' || c == 'X') rl += num; num = 0; }
        }
        if (!rl) continue;
        const uint64_t g1 = g0 + rl - 1;
        const size_t hi = std::upper_bound(ann->tx.begin(), ann->tx.end(), g1, [](uint64_t v, const TxAnnot::Tx& t) { return v < t.s; }) - ann->tx.begin();
        const TxAnnot::Tx* best = nullptr;
        int nsense = 0, nanti = 0;
        for (size_t i = hi; i-- > 0;) {
            if (ann->tx[i].emax < g0) break;
            const auto& t = ann->tx[i];
            if (t.e < g0 || t.str == 0) continue;
            const bool sn = (flag & 16) ? t.str == 2 : t.str == 1;
            (sn ? nsense : nanti)++;
            if (!best || (t.e - t.s) > (best->e - best->s)) best = &t;
        }
        if (!best || (nsense && nanti)) continue;
        ++assigned;
        sense += nsense > 0;
        const std::string bt = best->gene < ann->biotype.size() ? ann->biotype[best->gene] : std::string();
        if ((bt.rfind("IG_", 0) == 0 || bt.rfind("TR_", 0) == 0) && bt.find("pseudogene") == std::string::npos) ++vdj;
        if (bt == "rRNA" || bt == "Mt_rRNA") ++rrna;
        const uint64_t span = best->e - best->s + 1;
        const bool plus = best->str == 1;
        if (span >= 4000) {
            const uint64_t d3 = plus ? (best->e > g1 ? best->e - g1 : 0) : (g0 > best->s ? g0 - best->s : 0);
            const uint64_t d5 = plus ? (g0 > best->s ? g0 - best->s : 0) : (best->e > g1 ? best->e - g1 : 0);
            ++nend;
            n3 += d3 <= 1000;
            n5 += d5 <= 1000;
            expect += 1000.0 / static_cast<double>(span);
        }
        if (span >= 5000) {
            const double mid = double((g0 + g1) / 2 - best->s) / static_cast<double>(span);
            const double pos = plus ? mid : 1.0 - mid;
            prof[std::min(9, std::max(0, static_cast<int>(pos * 10)))] += 1;
            ++nprof;
        }
    }
    if (n) m.chrm = double(mt) / n;
    if (assigned >= 50) {
        m.tx_measured = true;
        m.tx_reads = assigned;
        m.sense_span = double(sense) / assigned;
        m.vdj = double(vdj) / assigned;
        m.rrna = double(rrna) / assigned;
    }
    if (nend >= 50) { m.end3 = double(n3) / nend; m.end5 = double(n5) / nend; m.end_expected = expect / nend; m.end_reads = nend; }
    if (nprof >= 50) for (int k = 0; k < 10; ++k) m.profile[k] = prof[k] / nprof;
    if (ins.size() >= 50) {
        std::sort(ins.begin(), ins.end());
        m.insert_median = double(ins[ins.size() / 2]);
        m.insert_n = ins.size();
        m.paired = true;
    }
}

inline MapEvidence parse_star_outputs(const std::string& prefix, uint64_t n_reads, const TxAnnot* ann = nullptr) {
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
    sam_features(prefix + "Aligned.out.sam", ann, m);
    return m;
}

// ── the classifier ──

struct Options {
    std::vector<std::string> whitelist_dirs;
    std::string hint;               // e.g. "assay=smartseq2"; reported, never counted
    int64_t run_spots = -1;         // spots in the whole run when known
    // --route-status: registry id -> route status. When use_route_status is set, `process` needs status "validated".
    bool use_route_status = false;
    std::unordered_map<std::string, std::string> route_status;
    // --species-filters DIR: directory of <tag>_21mer.bloom files with species_filters.tsv in or beside it; --species-hint: stated organism
    std::string species_filter_dir, species_hint;
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
    auto assign_cdna = [&] {
        std::vector<int> order;
        ev.cdna_seg = ev.cdna2_seg = -1;
        for (size_t s = 0; s < nseg; ++s)
            if ((int)s != ev.bc_seg && !ev.segs[s].index_like) order.push_back((int)s);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return ev.segs[a].len_mode > ev.segs[b].len_mode; });
        if (!order.empty()) ev.cdna_seg = order[0];
        if (order.size() > 1) ev.cdna2_seg = order[1];
    };
    assign_cdna();

    if (ev.bc_seg < 0 && ev.cdna_seg >= 0 && ev.cdna2_seg >= 0) {
        const double a = ev.segs[ev.cdna_seg].len_mode, b = ev.segs[ev.cdna2_seg].len_mode;
        if (std::min(a, b) >= 30 && std::min(a, b) >= 0.8 * std::max(a, b)) ev.mate_seg = ev.cdna2_seg;
    }

    // windows on the barcode candidate (or the first segment of a single-read run)
    int wseg = ev.bc_seg >= 0 ? ev.bc_seg : (nseg == 1 ? 0 : -1);
    if (best_rate < kThr.wl_hit) {
        // well barcodes sit on a short read: a long single read has repeated starts (adapters, primers) and is not scanned
        const bool ok_len = wseg >= 0 && ev.segs[wseg].len_mode <= (nseg == 1 ? 48u : 64u);
        if (ok_len) ev.window = scan_windows(spots, wseg, 50000, &ev.window2, ev.segs[wseg].index_like);
        // plate indices in a separate index read (Smart-seq with Nextera well indices)
        if (!ev.window.found) {
            for (size_t t = 0; t < nseg; ++t)
                if (ev.segs[t].index_like && ev.segs[t].len_mode >= 8) {
                    WindowHit w = scan_windows(spots, (int)t, 50000, nullptr, true);
                    if (w.found) { ev.window = w; ev.window_on_index = true; break; }
                }
        } else ev.window_on_index = ev.segs[ev.window.seg].index_like;
        // a barcode set larger than a plate (Drop-seq, Seq-Well, microwell bead sets): the first 12 bases repeat across reads
        if (!ev.window.found) {
            for (size_t t = 0; t < nseg; ++t) {
                if (ev.segs[t].index_like || ev.segs[t].len_mode < 12) continue;
                std::unordered_map<uint64_t, uint32_t> c;
                uint64_t n = 0;
                for (size_t i = 0; i < probe_n; ++i) {
                    if (t >= spots[i].seg.size() || spots[i].seg[t].size() < 12) continue;
                    uint64_t h;
                    if (!pack(spots[i].seg[t].data(), 12, h)) continue;
                    ++c[h]; ++n;
                }
                uint64_t rep = 0;
                for (auto& kv : c) if (kv.second >= 5) rep += kv.second;
                double share = n ? double(rep) / n : 0;
                {   // the repeating bases must vary across reads (a barcode set); a fixed linker or adapter repeats without being one
                    uint32_t pc[12][4] = {};
                    for (size_t i = 0; i < probe_n; ++i) {
                        if (t >= spots[i].seg.size() || spots[i].seg[t].size() < 12) continue;
                        for (size_t k = 0; k < 12; ++k) { const uint8_t b = nuc::ascii_to_num(spots[i].seg[t][k]); if (b < 4) ++pc[k][b]; }
                    }
                    int varied = 0;
                    for (int k = 0; k < 12; ++k) varied += (detail_id::entropy_bits(pc[k]) >= 1.5);
                    if (varied < 10) share = 0;
                }
                if (share > ev.prefix_mult) { ev.prefix_mult = share; ev.prefix_seg = (int)t; }
            }
            // the repeating read is the barcode read, whatever its length (Seq-Well runs with two equal-length reads)
            if (nseg >= 2 && ev.prefix_mult >= kThr.prefix_mult && ev.prefix_seg >= 0 && ev.prefix_seg != ev.bc_seg) { ev.bc_seg = ev.prefix_seg; ev.mate_seg = -1; assign_cdna(); }
        }
    }

    // bead barcode read with a poly(T) tail: barcode and UMI (a fixed number of varied bases) then poly(T)
    if (best_rate < kThr.wl_hit && ev.window_on_index == false && !ev.window.found) {
        for (size_t t = 0; t < nseg; ++t) {
            if (ev.segs[t].index_like || ev.segs[t].len_mode < 28) continue;
            const size_t np = std::min<size_t>(spots.size(), 20000);
            uint32_t cnt[40][4] = {};
            std::vector<uint32_t> tcount(48, 0);
            uint64_t nn = 0;
            for (size_t i = 0; i < np; ++i) {
                if (t >= spots[i].seg.size() || spots[i].seg[t].size() < 30) continue;
                const std::string& q = spots[i].seg[t];
                ++nn;
                const size_t ql = std::min<size_t>(q.size(), 40);
                for (size_t k = 0; k < ql; ++k) { const uint8_t b = nuc::ascii_to_num(q[k]); if (b < 4) ++cnt[k][b]; }
                for (size_t k = 8; k <= 32 && k + 8 <= q.size(); ++k) { int tt = 0; for (size_t j = 0; j < 8; ++j) tt += (q[k + j] == 'T'); if (tt >= 7) ++tcount[k]; }
            }
            if (nn < 1000) continue;
            int k0 = -1;
            for (int k = 8; k <= 32; ++k) if (double(tcount[k]) / nn >= 0.5) { k0 = k; break; }
            if (k0 < 12) continue;
            double h = 0;
            for (int k = 0; k < k0; ++k) h += detail_id::entropy_bits(cnt[k]);
            if (h / k0 >= 1.7) { ev.polyt_seg = (int)t; ev.polyt_off = k0; if (nseg >= 2 && ev.bc_seg != (int)t) { ev.bc_seg = (int)t; ev.mate_seg = -1; assign_cdna(); } break; }
        }
    }

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

    // probe panels (Visium FFPE and CytAssist, Flex) start every cDNA read at one of a few thousand probe sites
    if (ev.cdna_seg >= 0) {
        std::unordered_map<uint64_t, uint32_t> c;
        uint64_t n = 0, rep = 0;
        ev.cdna_len = ev.segs[ev.cdna_seg].len_mode;
        for (size_t i = 0; i < probe_n; ++i) {
            if ((size_t)ev.cdna_seg >= spots[i].seg.size() || spots[i].seg[ev.cdna_seg].size() < 25) continue;
            uint64_t h;
            if (!pack(spots[i].seg[ev.cdna_seg].data(), 25, h)) continue;
            ++c[h]; ++n;
        }
        for (auto& kv : c) if (kv.second >= 3) rep += kv.second;
        ev.cdna_start_repeat = n ? double(rep) / n : 0;
    }
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
        const bool probe_like = ev.cdna_start_repeat >= kThr.probe_start && ev.cdna_len <= 60;
        if (gkey == "visium") {
            g.modality = probe_like ? "spatial_gex_probe" : "spatial_gex_polya";
            g.reasons.push_back(std::string("Visium sub-call ") + (probe_like ? "probe-based" : "poly(A)") + ": read 2 is " + std::to_string(ev.cdna_len) +
                                " bases and " + detail::jnum(ev.cdna_start_repeat) + " of reads start at a repeated 25-mer");
        }
        const bool vdj_like = ev.map.tx_measured && ev.map.vdj >= kThr.vdj_share && gkey != "visium";
        g.support = clamp01(grate * ((ev.feature_like && gkey != "visium") || probe_like && gkey != "visium" || vdj_like ? 0.3 : 1.0));
        add_usefulness(g, ev, grate);
        H.push_back(g);
        if (probe_like && gkey != "visium" && !ev.feature_like) {
            Interp fx;
            fx.family = "flex"; fx.protocol = tag; fx.modality = "gex_probe";
            fx.support = clamp01(grate * 0.9);
            fx.reasons.push_back("barcode list hit and probe-like cDNA reads: " + detail::jnum(ev.cdna_start_repeat) + " of reads start at a repeated 25-mer, read length " +
                                 std::to_string(ev.cdna_len) + " (probe panel, as in 10x Flex)");
            H.push_back(fx);
        }
        if (vdj_like) {
            Interp vd;
            vd.family = "vdj"; vd.protocol = tag; vd.modality = "vdj";
            vd.support = clamp01(grate * 0.95);
            vd.reasons.push_back(detail::jnum(ev.map.vdj) + " of gene-assigned reads fall on immunoglobulin or T-cell receptor genes");
            H.push_back(vd);
        }
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
        int nonindex = 0;
        for (const auto& sg : ev.segs) nonindex += !sg.index_like;
        if (!fam.empty() && ev.detected.confidence >= Confidence::LOW && nonindex >= 2) {
            Interp d;
            d.family = fam; d.protocol = tag; d.modality = "gex";
            d.support = conf_support(ev.detected.confidence);
            d.reasons.push_back(std::string("protocol detector: ") + tag + " (" + conf_name(ev.detected.confidence) + " confidence)");
            add_usefulness(d, ev, ev.detected.wl_match_rate > 0 ? ev.detected.wl_match_rate : 1.0);
            H.push_back(d);
        } else if (tag == "agnostic" && ev.detected.confidence >= Confidence::LOW && nonindex >= 2) {
            Interp d;
            d.family = "unknown"; d.protocol = "barcoded-unspecified"; d.modality = "gex";
            d.support = 0.7 * conf_support(ev.detected.confidence);
            d.reasons.push_back("layout detector found barcode-like structure without a known protocol");
            H.push_back(d);
        }
        if (ev.window.found && ev.window_on_index) {
            Interp sm;
            sm.family = "smartseq"; sm.protocol = "smartseq2-like"; sm.modality = "gex";
            sm.support = clamp01(0.85 * ev.window.top_share);
            sm.reasons.push_back("index read carries " + std::to_string(ev.window.distinct) + " well indices (top " + std::to_string(ev.window.k) + " hold " +
                                 detail::jnum(ev.window.top_share) + " of reads) and the insert reads have no barcode: plate-based full-length RNA");
            H.push_back(sm);
        } else if (ev.window.found && ev.window2.found) {
            Interp w;
            w.family = "droplet_other";
            w.protocol = family_of_tag(tag) == "droplet_other" ? tag : "barcoded-unspecified";
            w.modality = "gex";
            w.support = clamp01(ev.window.top_share * ev.window2.top_share);
            w.reasons.push_back("two disjoint barcode windows (offsets " + std::to_string(ev.window.off) + " and " + std::to_string(ev.window2.off) + ", " +
                                std::to_string(ev.window.len) + " and " + std::to_string(ev.window2.len) + " bases) each hold few sequences: split-pool bead barcodes");
            add_usefulness(w, ev, ev.window.top_share * ev.window2.top_share);
            H.push_back(w);
        } else if (ev.window.found) {
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
        } else if (ev.polyt_seg >= 0 && nonindex >= 2 && ev.polyt_off >= 18 && ev.polyt_off <= 22) {
            Interp w;
            w.family = "droplet_other";
            w.protocol = "dropseq-like";
            w.modality = "gex";
            w.support = 0.8;
            w.reasons.push_back("segment " + std::to_string(ev.polyt_seg) + " holds " + std::to_string(ev.polyt_off) + " varied bases (barcode 12 and UMI 8) and then poly(T): "
                                "Drop-seq geometry, shared by Seq-Well, DroNc-seq and Drop-seq");
            add_usefulness(w, ev, 1.0);
            H.push_back(w);
        } else if (ev.prefix_mult >= kThr.prefix_mult && ev.smart3_tag < kThr.smart3_tag && nonindex >= 2 &&
                   !(ev.detected.confidence >= Confidence::MEDIUM && !fam.empty() && fam != "droplet_other")) {
            Interp w;
            w.family = "droplet_other";
            w.protocol = family_of_tag(tag) == "droplet_other" ? tag : "barcoded-unspecified";
            w.modality = "gex";
            w.support = clamp01(0.9 * ev.prefix_mult);
            w.reasons.push_back("the first 12 bases of segment " + std::to_string(ev.prefix_seg) + " repeat across reads (" + detail::jnum(ev.prefix_mult) +
                                " of reads): a barcode set larger than a plate");
            add_usefulness(w, ev, ev.prefix_mult);
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
            // A run without a barcode-like read is not bulk by default. Bulk and Smart-seq need positive evidence from mapping (cDNA reads spread
            // along the transcript); tag-based single-cell cDNA piles up at one transcript end on one strand; anything else stays rna_no_barcode.
            const bool mapped = ev.map.measured && ev.map.tx_measured && ev.map.end_reads >= 200;
            const double endmax = std::max(ev.map.end3, ev.map.end5);
            const double enrich = ev.map.end_expected > 0 ? endmax / ev.map.end_expected : 0;
            const bool tag3 = mapped && ev.map.end3 >= kThr.tag_end_share && ev.map.end3 >= ev.map.end5 && enrich >= kThr.tag_end_enrich && ev.map.sense_span >= kThr.sense_3p;
            const bool tag5 = mapped && ev.map.end5 >= kThr.tag_end_share && ev.map.end5 > ev.map.end3 && enrich >= kThr.tag_end_enrich && ev.map.sense_span <= kThr.sense_5p;
            const bool even = mapped && endmax < kThr.even_end_share;
            const std::string ends = "reads within 1000 nt of the transcript 3' end " + detail::jnum(ev.map.end3) + ", of the 5' end " + detail::jnum(ev.map.end5) +
                                     " (even coverage would give " + detail::jnum(ev.map.end_expected) + "), sense share " + detail::jnum(ev.map.sense_span);
            if (tag3 || tag5) {
                Interp t;
                t.family = "barcode_read_missing"; t.protocol = tag3 ? "tag-3p" : "tag-5p"; t.modality = "gex";
                t.support = 0.85;
                t.reasons.push_back("single-cell style cDNA without its barcode read in this endpoint: " + ends);
                t.reasons.push_back("try_endpoints: the barcode read may sit in another endpoint of the run (ENA submitted files, SRA original-format objects); run with --endpoints");
                t.unmeasured.push_back("valid_barcode: no barcode read");
                H.push_back(t);
            } else if (even) {
                const bool small = opt.run_spots >= 0 && (uint64_t)opt.run_spots <= kThr.smartseq_max_spots;
                const bool large = opt.run_spots >= 0 && (uint64_t)opt.run_spots > kThr.smartseq_max_spots;
                Interp s, b;
                s.family = "smartseq"; s.protocol = "smartseq2-like"; s.modality = "gex";
                b.family = "bulk_rna"; b.protocol = "bulk"; b.modality = "gex";
                s.support = small ? 0.62 : (large ? 0.38 : 0.5);
                b.support = small ? 0.38 : (large ? 0.62 : 0.5);
                const std::string sz = opt.run_spots >= 0 ? ("run holds " + std::to_string(opt.run_spots) + " spots") : "run size unknown";
                s.reasons.push_back("no cell barcode and even coverage along transcripts (" + ends + "); " + sz);
                b.reasons = s.reasons;
                add_usefulness(s, ev, 0.0);
                s.usefulness = 0;
                s.unmeasured.push_back("no cell barcode: single-cell usefulness depends on the plate layout");
                H.push_back(s);
                H.push_back(b);
            } else {
                Interp r;
                r.family = "rna_no_barcode"; r.protocol = "undetermined"; r.modality = "gex";
                r.support = 0.6;
                r.reasons.push_back(mapped ? "no cell barcode; coverage is neither even nor a tag end pile-up (" + ends + ")"
                                           : "no cell barcode and no mapping evidence: bulk and Smart-seq need positive evidence, so the run is held (pass --genome-dir)");
                H.push_back(r);
            }
            if (ev.tn5 >= kThr.tn5) {
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
        }
    }

    // a known plate tag from the protocol detector (medium or high confidence) wins over an unspecified bead layout
    {
        Interp* plate = nullptr;
        for (auto& it : H) if (it.family == "plate_barcoded" && it.reasons.size() && it.reasons[0].rfind("protocol detector", 0) == 0) plate = &it;
        if (plate && ev.detected.confidence >= Confidence::MEDIUM)
            for (auto& it : H)
                if (it.family == "droplet_other" && it.protocol == "barcoded-unspecified" && it.support >= plate->support - 0.001) {
                    plate->support = std::min(0.99, it.support + 0.05);
                    plate->reasons.push_back("known plate protocol tag outranks an unspecified bead layout");
                }
    }

    // ── layer D adjusts support ──
    if (ev.map.measured && !any_list) {   // a barcode-list hit is stronger evidence than where the cDNA lands; the list families keep their support
        for (auto& it : H) {
            const bool rna = it.family == "droplet_3p" || it.family == "droplet_5p" || it.family == "multiome_gex" || it.family == "visium" ||
                             it.family == "droplet_other" || it.family == "combinatorial" || it.family == "plate_barcoded" ||
                             it.family == "smartseq" || it.family == "smartseq3" || it.family == "bulk_rna" || it.family == "barcode_read_missing" || it.family == "rna_no_barcode";
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
        if (ev.map.exonic_measured && ev.map.exonic <= kThr.exonic_dna_max && ev.map.unique >= kThr.mapped_min &&
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
    if (R.action == "process" && R.verdict_protocol == "dropseq-like") {
        R.action = "defer";
        R.action_note = "Drop-seq geometry in a longer first read than the dropseq spec: recognized, deferred until the layout is trimmed to the spec";
    }
    if (R.action == "process" && R.verdict_protocol == "barcoded-unspecified") {
        R.action = "defer";
        R.action_note = "barcoded single-cell library whose layout has no protocol spec: recognized, deferred";
    }
    // registry id and sub-calls
    {
        std::string modality;
        for (const auto& it : R.ranked)
            if (it.family == R.verdict) { modality = it.modality; break; }
        const TagCall tc = registry_tag(R.verdict, R.verdict_protocol, modality);
        R.protocol_tag = tc.id;
        R.protocol_tag_alternatives = tc.alternatives;
        if (R.verdict == "droplet_3p") R.sub_strand = "3p";
        else if (R.verdict == "droplet_5p") R.sub_strand = "5p";
        if (R.verdict == "visium") R.sub_visium = modality == "spatial_gex_probe" ? "probe" : "polya";
        if (R.verdict == "vdj") R.sub_vdj = "enriched";
        R.sub_call_reason = "indrop_version, bd_bead and parse_kit: not implemented";
    }
    // a metadata hint is shown and never changes the verdict; it only selects the action inside an ambiguous full-length RNA call
    if (R.verdict == "ambiguous" && R.ambiguous_with.find("smartseq") != std::string::npos && R.ambiguous_with.find("bulk_rna") != std::string::npos &&
        opt.hint.find("assay=smartseq") != std::string::npos) {
        R.action = "defer";
        R.action_note = "hint assay=smartseq2 (metadata, not read evidence) selects defer; reads alone cannot separate Smart-seq from bulk";
    }
    // single_cell=1 (archive metadata, not read evidence): a library without barcode structure that the archive calls single-cell is deferred
    // instead of refused, since reads cannot confirm the metadata. The verdict is unchanged.
    {
        std::string h = opt.hint;
        for (auto& c : h) if (c == ';' || c == ' ') c = ',';
        size_t i = 0;
        while (i <= h.size()) {
            size_t j = h.find(',', i);
            if (j == std::string::npos) j = h.size();
            const std::string kv = h.substr(i, j - i);
            i = j + 1;
            if (kv.empty()) continue;
            const size_t eq = kv.find('=');
            Report::Hint ht;
            ht.key = kv.substr(0, eq);
            ht.value = eq == std::string::npos ? "" : kv.substr(eq + 1);
            if (ht.key == "single_cell" && ht.value == "1" &&
                (R.verdict == "bulk_rna" || R.verdict == "genomic_dna" || R.verdict == "unknown" || R.verdict == "rna_no_barcode")) {
                if (R.action != "defer") ht.effect = "action " + R.action + " -> defer";
                else ht.effect = "action stays defer";
                R.action = "defer";
                R.action_note = "hint single_cell=1 (archive metadata, not read evidence): the reads show no barcode structure and cannot confirm it, so the run is deferred, not refused";
            }
            R.hints.push_back(ht);
        }
    }
    // route status (--route-status): process only for registry ids whose route is validated; refuse, classify_only and defer rows stay as they are
    if (opt.use_route_status) {
        R.policy_source = "route-status";
        if (R.action == "process") {
            const auto it = opt.route_status.find(R.protocol_tag);
            const std::string st = R.protocol_tag.empty() ? "no registry id" : (it == opt.route_status.end() ? "unrouted" : it->second);
            if (st != "validated") {
                R.action = "defer";
                R.action_note = "route " + (R.protocol_tag.empty() ? std::string("none") : R.protocol_tag) + " has status " + st + ": recognized, deferred until the route is validated";
            }
        }
    }
}

// ── JSON ──

inline std::string report_json(const Report& R) {
    using namespace detail;
    std::ostringstream o;
    auto q = [](const std::string& s) { return "\"" + jesc(s) + "\""; };
    const auto& ev = R.ev;
    o << "{\n  \"record_version\": " << kRecordVersion << ",\n  \"schema_version\": " << R.schema << ",\n";
    o << "  \"source\": {\"source\": " << q(R.src.source) << ", \"route\": " << q(R.src.route) << ", \"endpoint\": " << q(R.src.endpoint)
      << ", \"bytes\": " << (R.src.bytes >= 0 ? std::to_string(R.src.bytes) : std::string("null")) << ", \"bytes_note\": " << q(R.src.bytes_note)
      << ", \"seconds\": " << jnum(R.src.seconds) << ", \"spots\": " << R.src.spots << ", \"run_spots\": "
      << (R.src.run_spots >= 0 ? std::to_string(R.src.run_spots) : std::string("null")) << ", \"error\": " << q(R.src.error) << "},\n";
    o << "  \"verdict\": " << q(R.verdict) << ",\n  \"protocol\": " << q(R.verdict_protocol) << ",\n  \"ambiguous_with\": " << q(R.ambiguous_with)
      << ",\n  \"action\": " << q(R.action) << ",\n  \"action_note\": " << q(R.action_note) << ",\n  \"policy_source\": " << q(R.policy_source) << ",\n  \"hint\": " << q(R.hint) << ",\n";
    auto qn = [&](const std::string& v) { return v.empty() ? std::string("null") : q(v); };
    o << "  \"hints\": [";
    for (size_t i = 0; i < R.hints.size(); ++i)
        o << (i ? ", " : "") << "{\"key\": " << q(R.hints[i].key) << ", \"value\": " << q(R.hints[i].value) << ", \"effect\": " << qn(R.hints[i].effect) << "}";
    o << "],\n";
    o << "  \"protocol_tag\": " << qn(R.protocol_tag) << ",\n  \"protocol_tag_alternatives\": " << jlist(R.protocol_tag_alternatives) << ",\n";
    o << "  \"sub_call\": {\"strand\": " << qn(R.sub_strand) << ", \"visium\": " << qn(R.sub_visium) << ", \"vdj\": " << qn(R.sub_vdj) << ", \"indrop_version\": "
      << qn(R.sub_indrop_version) << ", \"bd_bead\": " << qn(R.sub_bd_bead) << ", \"parse_kit\": " << qn(R.sub_parse_kit) << ", \"sub_call_reason\": " << qn(R.sub_call_reason) << "},\n";
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
    o << "  \"probe\": {\"cdna_start_repeat\": " << jnum(ev.cdna_start_repeat) << ", \"cdna_len\": " << ev.cdna_len << "},\n";
    o << "  \"window2\": {\"found\": " << (ev.window2.found ? "true" : "false") << ", \"offset\": " << ev.window2.off << ", \"length\": " << ev.window2.len << ", \"top_share\": " << jnum(ev.window2.top_share)
      << "},\n  \"window_on_index\": " << (ev.window_on_index ? "true" : "false") << ",\n  \"prefix\": {\"share\": " << jnum(ev.prefix_mult) << ", \"segment\": " << ev.prefix_seg << "},\n";
    o << "  \"detector\": {\"tag\": " << q(ev.detected.tag) << ", \"confidence\": " << q(conf_name(ev.detected.confidence)) << ", \"wl_rate\": " << jnum(ev.detected.wl_match_rate) << "},\n";
    o << "  \"insert\": {\"five_prime_called\": " << (ev.five_prime_called ? "true" : "false") << ", \"r1_tso\": " << ev.five_prime.r1_tso << ", \"r1_polyt\": " << ev.five_prime.r1_polyt
      << ", \"r2_smart\": " << ev.five_prime.r2_smart << ", \"r2_polya\": " << ev.five_prime.r2_polya << ", \"r2_polyt\": " << ev.five_prime.r2_polyt << ", \"motif_spots\": " << ev.five_prime.n
      << ", \"tn5\": " << jnum(ev.tn5) << ", \"polyt_head\": " << jnum(ev.polyt) << ", \"adt_capture_a\": " << jnum(ev.adt_cap) << ", \"gc\": " << jnum(ev.gc) << ", \"c_share\": " << jnum(ev.c_share)
      << ", \"n_share\": " << jnum(ev.n_share) << ", \"smart3_tag\": " << jnum(ev.smart3_tag) << ", \"top100_off0\": " << jnum(ev.top100_off0) << ", \"top100_off10\": " << jnum(ev.top100_off10)
      << ", \"tag_polya\": " << jnum(ev.tag_polya) << ", \"capture_bc\": " << jnum(ev.capture) << ", \"scaffold\": " << jnum(ev.scaffold) << ", \"feature_reason\": " << q(ev.feature_reason)
      << ", \"bc_repeat_share\": " << jnum(ev.repeat_share_bc) << "},\n";
    o << "  \"mapping\": {\"measured\": " << (ev.map.measured ? "true" : "false") << ", \"note\": " << q(ev.map.note) << ", \"reads\": " << ev.map.reads << ", \"unique\": " << jnum(ev.map.unique)
      << ", \"multi\": " << jnum(ev.map.multi) << ", \"unmapped\": " << jnum(ev.map.unmapped) << ", \"gene_assigned\": " << jnum(ev.map.exonic) << ", \"sense\": " << jnum(ev.map.sense)
      << ", \"chrM\": " << jnum(ev.map.chrm) << ", \"seconds\": " << jnum(ev.map.seconds)
      << ", \"tx_measured\": " << (ev.map.tx_measured ? "true" : "false") << ", \"tx_reads\": " << ev.map.tx_reads << ", \"sense_span\": " << jnum(ev.map.sense_span)
      << ", \"end3\": " << jnum(ev.map.end3) << ", \"end5\": " << jnum(ev.map.end5) << ", \"end_expected\": " << jnum(ev.map.end_expected) << ", \"end_reads\": " << ev.map.end_reads
      << ", \"insert_median\": " << jnum(ev.map.insert_median) << ", \"insert_n\": " << ev.map.insert_n << ", \"vdj\": " << jnum(ev.map.vdj) << ", \"rrna\": " << jnum(ev.map.rrna)
      << ", \"rrna_note\": \"annotated rRNA genes only\", \"profile\": [";
    for (int k = 0; k < 10; ++k) o << (k ? ", " : "") << jnum(ev.map.profile[k]);
    o << "]},\n";
    o << "  \"species\": ";
    if (!R.species.measured) o << "null";
    else {
        const auto& sp = R.species;
        o << "{\"call\": " << q(sp.call) << ", \"hits\": [";
        for (size_t i = 0; i < sp.hits.size(); ++i)
            o << (i ? ", " : "") << "{\"filter\": " << q(sp.hits[i].tag) << ", \"species\": " << q(sp.hits[i].species) << ", \"close_group\": " << q(sp.hits[i].group)
              << ", \"rate\": " << jnum(sp.hits[i].rate) << "}";
        o << "], \"mixture\": " << (sp.mixture ? "true" : "false") << ", \"minor_species\": " << qn(sp.minor_species) << ", \"minor_read_share\": " << jnum(sp.minor_share)
          << ", \"minor_reads\": " << sp.minor_reads << ", \"filters_scored\": " << sp.filters_scored << ", \"filters_available\": " << sp.filters_available
          << ", \"reads_scored\": " << sp.reads_scored << ", \"short_reads\": " << (sp.short_reads ? "true" : "false") << ", \"screened_by_hint\": "
          << (sp.screened_by_hint ? "true" : "false") << ", \"note\": " << qn(sp.note) << "}";
    }
    o << ",\n";
    o << "  \"endpoints\": [";
    for (size_t i = 0; i < R.endpoints.size(); ++i) {
        const auto& e = R.endpoints[i];
        o << (i ? ", " : "") << "{\"endpoint\": " << q(e.name) << ", \"reachable\": " << (e.reachable ? "true" : "false") << ", \"segments\": [";
        for (size_t k = 0; k < e.seg_lens.size(); ++k) o << (k ? ", " : "") << e.seg_lens[k];
        o << "], \"barcode_read_present\": " << (e.barcode_read ? "true" : "false") << ", \"best_list_rate\": " << jnum(e.list_rate) << ", \"verdict\": " << q(e.verdict)
          << ", \"protocol\": " << q(e.protocol) << ", \"bytes\": " << (e.bytes >= 0 ? std::to_string(e.bytes) : std::string("null")) << ", \"seconds\": " << jnum(e.seconds)
          << ", \"note\": " << q(e.note) << "}";
    }
    o << "],\n";
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

// ── species (validation reference pack filters) ──

namespace detail_sp {
inline std::string lower(std::string v) { for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return v; }
inline std::vector<std::string> cols(const std::string& line) {
    std::vector<std::string> o;
    std::string cur;
    for (char c : line) { if (c == '\t') { o.push_back(cur); cur.clear(); } else if (c != '\r') cur += c; }
    o.push_back(cur);
    return o;
}
struct Row { std::string tag, species, group; };
inline bool exists(const std::string& f) { std::ifstream in(f, std::ios::binary); return static_cast<bool>(in); }
}  // namespace detail_sp

/// Scores the head against the species filters of `opt.species_filter_dir` and fills R.species. Without an index it falls back to
/// human_21mer.bloom and mouse_21mer.bloom. With a hint the hinted close group plus human and mouse are scored first; the rest load
/// only when no filter of that set clears the call rule.
inline void attach_species(Report& R, const std::vector<HeadSpot>& spots, const Options& opt) {
    using namespace detail_sp;
    if (opt.species_filter_dir.empty()) return;
    SpeciesBlock& B = R.species;
    const std::string dir = opt.species_filter_dir;
    std::vector<Row> rows;
    for (const std::string& idx : {dir + "/species_filters.tsv", dir + "/../species_filters.tsv"}) {
        std::ifstream in(idx);
        if (!in) continue;
        std::string line;
        std::getline(in, line);
        const auto h = cols(line);
        int ct = -1, cs = -1, cg = -1;
        for (size_t i = 0; i < h.size(); ++i) { if (h[i] == "filter_tag") ct = (int)i; else if (h[i] == "species") cs = (int)i; else if (h[i] == "close_group") cg = (int)i; }
        if (ct < 0) continue;
        std::unordered_map<std::string, bool> active;
        bool have_active = false;
        for (const std::string& af : {dir + "/active_filters.tsv", dir + "/../active_filters.tsv"}) {
            std::ifstream ain(af);
            if (!ain) continue;
            std::string al;
            std::getline(ain, al);
            while (std::getline(ain, al)) { const auto c = cols(al); if (!c.empty() && !c[0].empty()) { active[c[0]] = true; have_active = true; } }
            break;
        }
        while (std::getline(in, line)) {
            const auto c = cols(line);
            if (c.size() <= (size_t)ct || c[ct].empty()) continue;
            if (have_active && !active.count(c[ct])) continue;
            if (!exists(dir + "/" + c[ct] + "_21mer.bloom")) continue;
            rows.push_back({c[ct], cs >= 0 && c.size() > (size_t)cs ? c[cs] : "", cg >= 0 && c.size() > (size_t)cg ? c[cg] : ""});
        }
        if (!rows.empty()) break;
    }
    if (rows.empty()) {
        if (exists(dir + "/human_21mer.bloom")) rows.push_back({"human", "Homo sapiens", "Hominidae"});
        if (exists(dir + "/mouse_21mer.bloom")) rows.push_back({"mouse", "Mus musculus", "Mus"});
        B.note = "no species_filters.tsv index; human and mouse filters only";
    }
    B.filters_available = static_cast<int>(rows.size());
    if (rows.empty()) { B.note = "no filter files found in " + dir; return; }

    // reads: every segment of 50 bases or more, up to 20,000; below 2,000 reads the limit drops to 30 bases
    std::vector<std::vector<uint8_t>> reads;
    for (const int minlen : {50, 30}) {
        reads.clear();
        for (const auto& sp : spots)
            for (const auto& sg : sp.seg)
                if (sg.size() >= (size_t)minlen && reads.size() < 20000) reads.push_back(detail_id::num(sg));
        B.short_reads = minlen == 30;
        if (reads.size() >= 2000) break;
    }
    B.reads_scored = reads.size();
    if (reads.empty()) { B.note = "no read of 30 bases or more"; return; }

    struct Scored { size_t row; double rate; std::vector<uint16_t> hits; };
    std::vector<Scored> sc;
    std::vector<char> done(rows.size(), 0);
    auto score = [&](size_t ri) {
        pileup::BloomFilter bf;
        if (!bf.load(dir + "/" + rows[ri].tag + "_21mer.bloom")) { done[ri] = 1; return; }
        Scored x{ri, 0, std::vector<uint16_t>(reads.size(), 0)};
        uint64_t tot = 0, hit = 0;
        for (size_t i = 0; i < reads.size(); ++i) {
            const int h = bf.count_hits_numeric(reads[i].data(), static_cast<int>(reads[i].size()), 21);
            x.hits[i] = static_cast<uint16_t>(std::min(h, 65535));
            hit += h;
            tot += reads[i].size() >= 21 ? reads[i].size() - 20 : 0;
        }
        x.rate = tot ? double(hit) / double(tot) : 0;
        sc.push_back(std::move(x));
        done[ri] = 1;
    };
    auto order = [&]() { std::stable_sort(sc.begin(), sc.end(), [](const Scored& a, const Scored& b) { return a.rate > b.rate; }); };
    auto clears = [&]() {
        order();
        return !sc.empty() && sc[0].rate >= kThr.species_min_rate && sc.size() >= 2 && sc[0].rate >= kThr.species_ratio * sc[1].rate;
    };
    bool all = opt.species_hint.empty();
    if (!all) {
        const std::string hint = lower(opt.species_hint);
        std::string group;
        for (const auto& r : rows)
            if (!r.species.empty() && (lower(r.species) == hint || hint.find(lower(r.species)) == 0)) { group = r.group; break; }
        for (size_t i = 0; i < rows.size(); ++i) {
            const bool same = !group.empty() && rows[i].group == group;
            const bool named = !rows[i].species.empty() && lower(rows[i].species) == hint;
            if (same || named || rows[i].tag == "human" || rows[i].tag == "mouse") score(i);
        }
        B.screened_by_hint = true;
        if (!clears()) all = true;
    }
    if (all) {
        for (size_t i = 0; i < rows.size(); ++i) if (!done[i]) score(i);
        B.screened_by_hint = false;
        order();
    }
    B.filters_scored = static_cast<int>(sc.size());
    if (sc.empty()) { B.note = "no filter could be loaded"; return; }
    B.measured = true;
    for (size_t i = 0; i < sc.size() && i < 2; ++i) B.hits.push_back({rows[sc[i].row].tag, rows[sc[i].row].species, rows[sc[i].row].group, sc[i].rate});
    const Scored& A = sc[0];
    // mixture: the highest-rate filter outside A's close group, split per read between the two
    const Scored* Bm = nullptr;
    for (size_t i = 1; i < sc.size(); ++i)
        if (rows[sc[i].row].group.empty() || rows[sc[i].row].group != rows[A.row].group) { Bm = &sc[i]; break; }
    if (Bm) {
        int na = 0, nb = 0;
        for (size_t i = 0; i < reads.size(); ++i) {
            const int a = A.hits[i], b = Bm->hits[i];
            if (a - b >= kThr.species_mix_excess && 2 * b <= a) ++na;
            else if (b - a >= kThr.species_mix_excess && 2 * a <= b) ++nb;
        }
        const int minor = std::min(na, nb), n = na + nb;
        B.minor_reads = minor;
        B.minor_share = n ? double(minor) / n : 0;
        B.minor_species = rows[(nb < na ? Bm : &A)->row].species;
        B.mixture = n > 0 && minor >= kThr.species_mix_reads && B.minor_share >= kThr.species_mix_share;
    }
    const double ra = A.rate, rb = sc.size() > 1 ? sc[1].rate : 0;
    const bool same_group = sc.size() > 1 && !rows[A.row].group.empty() && rows[A.row].group == rows[sc[1].row].group;
    if (B.mixture) B.call = "MIXTURE";
    else if (ra < kThr.species_min_rate) B.call = "NONE";
    else if (ra >= kThr.species_ratio * rb && !(same_group && ra < 1.5 * rb)) B.call = "CALL";
    else if (same_group) B.call = "CLOSE_PAIR";
    else B.call = "AMBIGUOUS";
}

/// One call for tests and callers that already hold the head: layers A to C, the interpretations and the verdict.
inline Report identify_spots(const std::vector<HeadSpot>& spots, const Options& opt, std::vector<ScreenWhitelist>& lists, const MapEvidence* map = nullptr) {
    Report R;
    R.spots = spots.size();
    R.src.spots = spots.size();
    measure(spots, opt, lists, R.ev);
    if (map) R.ev.map = *map;
    rank(R, opt);
    attach_species(R, spots, opt);
    return R;
}
}  // namespace singlet::fq::identify
