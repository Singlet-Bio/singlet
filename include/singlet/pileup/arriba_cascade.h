// SPDX-License-Identifier: MIT
// arriba_cascade.h - Arriba's fusion filter and scoring cascade on the engine's chimeric reads.
//
// Derived from Arriba 2.5.1 (https://github.com/suhrig/arriba), Copyright (C) 2016-2025 Sebastian Uhrig, MIT license (NOTICE,
// data/fusion/arriba/LICENSE). Uhrig S, Ellermann J, Walther T, et al. Accurate and efficient detection of gene fusions from RNA sequencing
// data. Genome Research 31:448-460 (2021). Every ported function keeps the name of its Arriba counterpart and cites the Arriba source file.
// The logic is transliterated; the comments in the function bodies are Arriba's own unless they say "singlet:".
//
// Input is STAR's Chimeric.out.junction (format 1, single-end, raw cell barcode and UMI as the last two columns), the GTF, the genome FASTA
// (with its .fai) and, for the read-sequence filters, the reads of the chimeric pass (FASTQ, names r<n> as the engine writes them).
// Arriba reads a BAM file instead. What this data lacks, and what stands in for it (singlet: substitutions):
//   - reads are single-end cDNA, so there are no discordant mates; each junction line becomes a split read exactly as Arriba's
//     remove_malformed_alignments() reformats single-end reads (the longer segment is the split read, mate 1 a copy of it);
//   - duplicates: reads carry a cell barcode and UMI, so Arriba's external duplicate marking (-u) is used; a read is a duplicate when an
//     earlier read (in read-name order) has the same cell, UMI and breakpoints;
//   - multi-mapping chimeric reads are the lines of one read name (num_chim_aln > 1), numbered in file order as Arriba numbers HI tags;
//   - coverage (Arriba counts every alignment of the BAM in 20-bp windows): no BAM is kept, so a breakpoint's coverage is the mean read
//     depth of the most-counted annotated gene overlapping it, count * (read length + window) / effective length from gene_expression.tsv
//     (read length 90, window 20), and "a fragment starts/ends here" holds when a counted gene overlaps the window; with no gene counts,
//     coverage is unknown (-1, as Arriba returns for contigs without coverage);
//   - mapped reads (the e-value's library size): STAR's uniquely and multi-mapped reads of the main pass, given by the caller;
//   - read-through and tandem-duplication candidates that Arriba extracts from ordinary (non-chimeric) alignments are not available;
//   - with no read sequences, the homopolymer, mismatches and low_entropy filters are skipped and multi-mapper scores fall back to STAR's.
// Not ported (singlet: reasons): viral-contig filters (GRCh38-2024-A has no viral contigs), known_fusions and tags (no catalogue term may
// enter the ranking), internal_tandem_duplication (no ITD extraction without ordinary alignments), homologs and mismappers (they need
// Arriba's k-mer index of gene sequences, a separate port), isoforms (adds breakpoints to pairs that already pass, so pair ranks do not
// change), genomic_support (no WGS), and the output's transcript, peptide and protein-domain columns.
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace arriba {

typedef bool strand_t;
const strand_t FORWARD = true;
const strand_t REVERSE = false;
typedef bool direction_t;
const direction_t UPSTREAM = true;
const direction_t DOWNSTREAM = false;
typedef int position_t;
typedef int contig_t;

// common.hpp: filter IDs (same order as Arriba's FILTERS)
enum filter_t : unsigned char {
    FILTER_none = 0, FILTER_duplicates, FILTER_inconsistently_clipped, FILTER_homopolymer, FILTER_read_through, FILTER_same_gene,
    FILTER_small_insert_size, FILTER_long_gap, FILTER_hairpin, FILTER_multimappers, FILTER_mismatches, FILTER_mismappers,
    FILTER_relative_support, FILTER_intronic, FILTER_non_coding_neighbors, FILTER_intragenic_exonic, FILTER_internal_tandem_duplication,
    FILTER_min_support, FILTER_known_fusions, FILTER_spliced, FILTER_blacklist, FILTER_end_to_end, FILTER_in_vitro, FILTER_merge_adjacent,
    FILTER_select_best, FILTER_marginal_read_through, FILTER_short_anchor, FILTER_no_coverage, FILTER_many_spliced,
    FILTER_no_genomic_support, FILTER_uninteresting_contigs, FILTER_viral_contigs, FILTER_top_expressed_viral_contigs,
    FILTER_low_coverage_viral_contigs, FILTER_genomic_support, FILTER_isoforms, FILTER_low_entropy, FILTER_homologs
};
inline const char* filter_name(filter_t f) {
    static const char* n[] = {"", "duplicates", "inconsistently_clipped", "homopolymer", "read_through", "same_gene", "small_insert_size",
                              "long_gap", "hairpin", "multimappers", "mismatches", "mismappers", "relative_support", "intronic",
                              "non_coding_neighbors", "intragenic_exonic", "internal_tandem_duplication", "min_support", "known_fusions",
                              "spliced", "blacklist", "end_to_end", "in_vitro", "merge_adjacent", "select_best", "marginal_read_through",
                              "short_anchor", "no_coverage", "many_spliced", "no_genomic_support", "uninteresting_contigs", "viral_contigs",
                              "top_expressed_viral_contigs", "low_coverage_viral_contigs", "genomic_support", "isoforms", "low_entropy",
                              "homologs"};
    return n[f];
}

typedef unsigned char confidence_t;
const confidence_t CONFIDENCE_LOW = 0;
const confidence_t CONFIDENCE_MEDIUM = 1;
const confidence_t CONFIDENCE_HIGH = 2;

typedef char strandedness_t;
const strandedness_t STRANDEDNESS_NO = 0;
const strandedness_t STRANDEDNESS_YES = 1;
const strandedness_t STRANDEDNESS_REVERSE = 2;
const strandedness_t STRANDEDNESS_AUTO = 3;

const unsigned int MATE1 = 0;
const unsigned int MATE2 = 1;
const unsigned int SPLIT_READ = 1;
const unsigned int SUPPLEMENTARY = 2;
const unsigned int MAX_SPLICE_SITE_DISTANCE = 2;  // annotation.hpp

// annotation.hpp
inline strand_t complement_strand(const strand_t strand) { return (strand == FORWARD) ? REVERSE : FORWARD; }
inline strand_t complement_strand_if(const strand_t strand, const bool condition) { return condition ? complement_strand(strand) : strand; }
inline std::string strip_ensembl_version_number(const std::string& id) {
    std::string::size_type trim_position = std::string::npos;
    if (id.substr(0, 3) == "ENS" && (trim_position = id.find_last_of('.')) < id.size()) return id.substr(0, trim_position);
    return id;
}
// common.hpp
inline std::string removeChr(std::string contig) {
    if (contig.substr(0, 3) == "chr") contig = contig.substr(3);
    if (contig == "M") contig = "MT";
    return contig;
}
// common.hpp: is_interesting_contig
inline bool is_interesting_contig(std::string contig, const std::string& interesting_contigs) {
    contig = removeChr(contig);
    std::istringstream iss(interesting_contigs);
    while (iss) {
        std::string pattern;
        iss >> pattern;
        pattern = removeChr(pattern);
        if (!pattern.empty()) {
            bool is_prefix = pattern[pattern.size() - 1] == '*';
            bool is_suffix = pattern[0] == '*';
            std::replace(pattern.begin(), pattern.end(), '*', ' ');
            std::istringstream iss2(pattern);
            size_t pos = 0;
            while (iss2) {
                std::string segment;
                iss2 >> segment;
                if (pos == 0 && !is_suffix && contig.substr(0, segment.size()) != segment) break;
                if (segment.empty() && (pos == contig.size() || is_prefix)) return true;
                pos = contig.find(segment, pos);
                if (pos == std::string::npos) break;
                pos += segment.size();
            }
        }
    }
    return false;
}

// ---- CIGAR (htslib op codes) ----
enum : uint32_t { BAM_CMATCH = 0, BAM_CINS = 1, BAM_CDEL = 2, BAM_CREF_SKIP = 3, BAM_CSOFT_CLIP = 4, BAM_CHARD_CLIP = 5, BAM_CPAD = 6,
                  BAM_CEQUAL = 7, BAM_CDIFF = 8 };
struct cigar_t : public std::vector<std::pair<uint32_t, uint32_t>> {  // (op, length)
    uint32_t operation(unsigned int i) const { return at(i).first; }
    uint32_t op_length(unsigned int i) const { return at(i).second; }
};
inline bool parse_cigar(const std::string& s, cigar_t& c) {
    c.clear();
    uint64_t n = 0;
    bool have = false;
    for (char ch : s) {
        if (ch >= '0' && ch <= '9') { n = n * 10 + (ch - '0'); have = true; continue; }
        if (!have) return false;
        uint32_t op;
        switch (ch) {
            case 'M': op = BAM_CMATCH; break;  case 'I': op = BAM_CINS; break;      case 'D': op = BAM_CDEL; break;
            case 'N': op = BAM_CREF_SKIP; break; case 'S': op = BAM_CSOFT_CLIP; break; case 'H': op = BAM_CHARD_CLIP; break;
            case 'P': op = BAM_CPAD; break;    case '=': op = BAM_CEQUAL; break;    case 'X': op = BAM_CDIFF; break;
            default: return false;  // 'p' (paired-end gap) and negative lengths do not occur in single-end junction lines
        }
        c.emplace_back(op, (uint32_t)n);
        n = 0; have = false;
    }
    return !c.empty() && !have;
}
inline bool cigar_consumes_ref(uint32_t op) { return op == BAM_CMATCH || op == BAM_CDEL || op == BAM_CREF_SKIP || op == BAM_CEQUAL || op == BAM_CDIFF; }
inline bool cigar_consumes_query(uint32_t op) { return op == BAM_CMATCH || op == BAM_CINS || op == BAM_CSOFT_CLIP || op == BAM_CEQUAL || op == BAM_CDIFF; }
inline int cigar_ref_len(const cigar_t& c) { int n = 0; for (auto& o : c) if (cigar_consumes_ref(o.first)) n += o.second; return n; }
inline int cigar_query_len(const cigar_t& c) { int n = 0; for (auto& o : c) if (cigar_consumes_query(o.first) || o.first == BAM_CHARD_CLIP) n += o.second; return n; }

inline std::string dna_to_reverse_complement(const std::string& dna) {  // assembly.cpp
    std::string r(dna.rbegin(), dna.rend());
    for (char& c : r) {
        switch (c) {
            case 'A': c = 'T'; break; case 'T': c = 'A'; break; case 'C': c = 'G'; break; case 'G': c = 'C'; break;
            case 'a': c = 't'; break; case 't': c = 'a'; break; case 'c': c = 'g'; break; case 'g': c = 'c'; break;
            default: c = 'N';
        }
    }
    return r;
}

// ---- annotation (common.hpp) ----
struct gene_annotation_record_t {
    contig_t contig = 0;
    position_t start = 0, end = 0;  // zero-based, inclusive
    strand_t strand = FORWARD;
    unsigned int id = 0;  // ID used internally; singlet: also the order of gene sets (Arriba orders them by pointer)
    std::string gene_id, name;
    int exonic_length = 0;
    bool is_dummy = false, is_protein_coding = false;
    unsigned int length() const { return end - start; }
};
typedef gene_annotation_record_t* gene_t;
struct exon_annotation_record_t;
struct transcript_annotation_record_t {
    unsigned int id = 0;
    std::string name;
    exon_annotation_record_t* first_exon = nullptr;
    exon_annotation_record_t* last_exon = nullptr;
};
typedef transcript_annotation_record_t* transcript_t;
struct exon_annotation_record_t {
    contig_t contig = 0;
    position_t start = 0, end = 0;
    strand_t strand = FORWARD;
    unsigned int id = 0;  // singlet: GTF order, the order of exon sets
    gene_t gene = nullptr;
    transcript_t transcript = nullptr;
    exon_annotation_record_t* previous_exon = nullptr;
    exon_annotation_record_t* next_exon = nullptr;
    position_t coding_region_start = -1, coding_region_end = -1;
    bool operator<(const exon_annotation_record_t& x) const {
        if (contig != x.contig) return contig < x.contig;
        if (end != x.end) return end < x.end;
        return start < x.start;
    }
};
typedef exon_annotation_record_t* exon_t;

template <class T> struct by_id { bool operator()(const T* a, const T* b) const { return a->id < b->id; } };
// annotation_set_t: sorted, unique
template <class T> struct annotation_set_t : public std::vector<T*> {
    void insert_one(T* v) {
        auto it = std::lower_bound(this->begin(), this->end(), v, by_id<T>());
        if (it == this->end() || *it != v) std::vector<T*>::insert(it, v);
    }
    void insert_all(const annotation_set_t<T>& o) { for (T* v : o) insert_one(v); }
};
typedef annotation_set_t<gene_annotation_record_t> gene_set_t;
typedef annotation_set_t<exon_annotation_record_t> exon_set_t;

// contig_annotation_index_t: Arriba's map<position, set>, here two parallel sorted vectors
template <class T> struct contig_annotation_index_t {
    std::vector<position_t> keys;
    std::vector<annotation_set_t<T>> sets;
    size_t lower_bound(position_t p) const { return std::lower_bound(keys.begin(), keys.end(), p) - keys.begin(); }
    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }
};
template <class T> using annotation_index_t = std::vector<contig_annotation_index_t<T>>;
typedef annotation_index_t<gene_annotation_record_t> gene_annotation_index_t;
typedef annotation_index_t<exon_annotation_record_t> exon_annotation_index_t;

// annotation.t.hpp: make_annotation_index. A key k holds the features with start <= k <= end; keys are every start-1 and end.
template <class T> void make_annotation_index(std::list<T>& annotation, annotation_index_t<T>& index, size_t n_contigs) {
    index.clear();
    index.resize(n_contigs);
    std::vector<std::vector<position_t>> keys(n_contigs);
    for (auto& f : annotation) { keys[f.contig].push_back(f.start - 1); keys[f.contig].push_back(f.end); }
    for (size_t c = 0; c < n_contigs; ++c) {
        std::sort(keys[c].begin(), keys[c].end());
        keys[c].erase(std::unique(keys[c].begin(), keys[c].end()), keys[c].end());
        index[c].keys = keys[c];
        index[c].sets.resize(keys[c].size());
    }
    std::vector<T*> sorted;
    for (auto& f : annotation) sorted.push_back(&f);
    std::sort(sorted.begin(), sorted.end(), by_id<T>());
    for (T* f : sorted) {
        auto& ci = index[f->contig];
        for (size_t i = ci.lower_bound(f->start); i < ci.keys.size() && ci.keys[i] <= f->end; ++i) ci.sets[i].push_back(f);
    }
}

// annotation.t.hpp: combine_annotations
template <class T> void combine_annotations(const annotation_set_t<T>& a, const annotation_set_t<T>& b, annotation_set_t<T>& combined, bool make_union = true) {
    combined.clear();
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(combined), by_id<T>());
    if (combined.empty() && make_union) std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(combined), by_id<T>());
}

// annotation.t.hpp: get_annotation_by_coordinate
template <class T> void get_annotation_by_coordinate(contig_t contig, position_t start, position_t end, annotation_set_t<T>& out, const annotation_index_t<T>& index) {
    out.clear();
    if ((unsigned int)contig >= index.size()) return;
    const auto& ci = index[contig];
    if (start == end) {
        size_t p = ci.lower_bound(start);
        if (p < ci.size()) out = ci.sets[p];
        return;
    }
    if (start > end) std::swap(start, end);
    annotation_set_t<T> result_start;
    size_t ps = ci.lower_bound(start);
    if (ps < ci.size()) {
        result_start = ci.sets[ps];
        if (ci.keys[ps] - start <= 2) {
            ++ps;
            if (ps < ci.size()) result_start.insert_all(ci.sets[ps]);
        }
    }
    annotation_set_t<T> result_end;
    size_t pe = ci.lower_bound(end);
    if (pe < ci.size()) result_end = ci.sets[pe];
    if (pe != 0 && ci.size() > 0) {
        --pe;
        if (end - ci.keys[pe] <= 2) result_end.insert_all(ci.sets[pe]);
    }
    combine_annotations(result_start, result_end, out);
}

// ---- contigs and assembly ----
struct contigs_t {
    std::map<std::string, contig_t> by_name;  // removeChr names
    std::vector<std::string> original;
    contig_t get_or_add(const std::string& name) {
        auto ins = by_name.emplace(removeChr(name), (contig_t)by_name.size());
        if ((size_t)ins.first->second >= original.size()) original.resize(ins.first->second + 1);
        if (ins.second) original[ins.first->second] = name;
        return ins.first->second;
    }
    contig_t find(const std::string& name) const { auto it = by_name.find(removeChr(name)); return it == by_name.end() ? -1 : it->second; }
    size_t size() const { return by_name.size(); }
};

// The genome by memory map (Arriba loads it into memory: assembly.cpp load_assembly). Indexed FASTA (.fai) required.
struct assembly_t {
    struct seq_t { int64_t length = 0, offset = 0, line_bases = 0, line_width = 0; };
    std::vector<seq_t> seqs;  // by contig ID; length 0 = not in the FASTA
    const char* data = nullptr;
    size_t mapped = 0;
    int fd = -1;
    assembly_t() = default;
    assembly_t(const assembly_t&) = delete;
    assembly_t& operator=(const assembly_t&) = delete;
    ~assembly_t() { if (data) munmap((void*)data, mapped); if (fd >= 0) close(fd); }
    bool load(const std::string& fasta, contigs_t& contigs) {
        FILE* f = std::fopen((fasta + ".fai").c_str(), "r");
        if (!f) return false;
        char name[4096];
        long long len, off, lb, lw;
        while (std::fscanf(f, "%4095s %lld %lld %lld %lld", name, &len, &off, &lb, &lw) == 5) {
            contig_t c = contigs.get_or_add(name);
            if ((size_t)c >= seqs.size()) seqs.resize(c + 1);
            seqs[c] = seq_t{len, off, lb, lw};
        }
        std::fclose(f);
        fd = open(fasta.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        if (fstat(fd, &st) != 0) return false;
        mapped = (size_t)st.st_size;
        void* p = mmap(nullptr, mapped, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { data = nullptr; return false; }
        data = (const char*)p;
        return true;
    }
    bool has(contig_t c) const { return data && (size_t)c < seqs.size() && seqs[c].length > 0; }
    int64_t size_of(contig_t c) const { return has(c) ? seqs[c].length : 0; }
    char at(contig_t c, int64_t pos) const {  // upper case; 'N' outside the contig
        if (!has(c) || pos < 0 || pos >= seqs[c].length) return 'N';
        const seq_t& s = seqs[c];
        char b = data[s.offset + (pos / s.line_bases) * s.line_width + pos % s.line_bases];
        return (b >= 'a' && b <= 'z') ? (char)(b - 32) : b;
    }
};

// ---- coverage (singlet: substitution for read_stats.cpp coverage_t; see the header comment) ----
struct coverage_t {
    std::unordered_map<std::string, unsigned int> count_by_gene_id;  // short gene IDs
    const gene_annotation_index_t* gene_index = nullptr;
    bool known = false;
    unsigned int gene_count(const gene_t g) const {
        if (g->is_dummy) return 0;
        auto it = count_by_gene_id.find(strip_ensembl_version_number(g->gene_id));
        return it == count_by_gene_id.end() ? 0 : it->second;
    }
    int get_coverage(contig_t contig, position_t position, direction_t /*direction*/) const {
        if (!known || !gene_index) return -1;
        gene_set_t genes;
        get_annotation_by_coordinate(contig, position, position, genes, *gene_index);
        unsigned int best = 0;
        for (gene_t g : genes) best = std::max(best, gene_count(g));
        return (int)std::min<unsigned int>(best, 65535);  // Arriba stores coverage as unsigned short
    }
    bool any_counted_gene(contig_t contig, position_t start, position_t end) const {
        if (!known || !gene_index || (unsigned int)contig >= gene_index->size()) return false;
        const auto& ci = (*gene_index)[contig];
        if (start > end) std::swap(start, end);
        for (size_t i = ci.lower_bound(start); i < ci.size(); ++i) {
            for (gene_t g : ci.sets[i]) if (gene_count(g) > 0) return true;
            if (ci.keys[i] >= end) break;
        }
        return false;
    }
    bool fragment_starts_here(contig_t contig, position_t start, position_t end) const { return any_counted_gene(contig, start, end); }
    bool fragment_ends_here(contig_t contig, position_t start, position_t end) const { return any_counted_gene(contig, start, end); }
};

// ---- alignments (common.hpp) ----
struct alignment_t {
    bool supplementary = false, first_in_pair = false, exonic = false;
    strand_t strand = FORWARD, predicted_strand = FORWARD;
    bool predicted_strand_ambiguous = true;
    contig_t contig = 0;
    position_t start = 0, end = 0;
    cigar_t cigar;
    gene_set_t genes;
    unsigned int preclipping() const {
        return (cigar.operation(0) == BAM_CSOFT_CLIP || cigar.operation(0) == BAM_CHARD_CLIP) ? cigar.op_length(0) : 0;
    }
    unsigned int postclipping() const {
        const unsigned int l = cigar.size() - 1;
        return (cigar.operation(l) == BAM_CSOFT_CLIP || cigar.operation(l) == BAM_CHARD_CLIP) ? cigar.op_length(l) : 0;
    }
};
struct mates_t : public std::vector<alignment_t> {
    bool single_end = true, multimapper = false, duplicate = false;
    filter_t filter = FILTER_none;
    std::string sequence;     // singlet: the read in the split read's reference orientation (Arriba stores it on MATE1 and SPLIT_READ)
    bool has_sequence = false;
    uint32_t cell = 0;        // singlet: index of the matched barcode
    std::string umi;
    int star_score = 0;       // singlet: STAR's this_chim_aln_score
};
typedef std::map<std::string, mates_t> chimeric_alignments_t;  // ordered by read name (with ",HI")
inline std::string strip_hi_tag_from_read_name(const std::string& n) { return n.substr(0, n.find_last_of(',')); }

// ---- fusions (common.hpp) ----
typedef bool transcript_start_t;
const transcript_start_t TRANSCRIPT_START_GENE1 = true;
const transcript_start_t TRANSCRIPT_START_GENE2 = false;
struct fusion_t {
    bool transcript_start_ambiguous = true;
    unsigned int split_reads1 = 0, split_reads2 = 0, discordant_mates = 0;
    transcript_start_t transcript_start = TRANSCRIPT_START_GENE1;
    bool spliced1 = false, spliced2 = false, exonic1 = false, exonic2 = false;
    strand_t predicted_strand1 = FORWARD, predicted_strand2 = FORWARD;
    direction_t direction1 = DOWNSTREAM, direction2 = DOWNSTREAM;
    confidence_t confidence = CONFIDENCE_LOW;
    filter_t filter = FILTER_none;
    bool predicted_strands_ambiguous = true;
    contig_t contig1 = -1, contig2 = -1;
    float evalue = 0;
    position_t breakpoint1 = -1, breakpoint2 = -1, anchor_start1 = 0, anchor_start2 = 0;
    position_t closest_genomic_breakpoint1 = -1, closest_genomic_breakpoint2 = -1;
    gene_t gene1 = nullptr, gene2 = nullptr;
    std::vector<chimeric_alignments_t::iterator> split_read1_list, split_read2_list, discordant_mate_list;
    unsigned int supporting_reads() const { return split_reads1 + split_reads2 + discordant_mates; }
    bool breakpoint_overlaps_both_genes(const unsigned int which = 0) const {
        if (which == 1) return breakpoint1 >= gene2->start && breakpoint1 <= gene2->end;
        if (which == 2) return breakpoint2 >= gene1->start && breakpoint2 <= gene1->end;
        return breakpoint_overlaps_both_genes(1) || breakpoint_overlaps_both_genes(2);
    }
    bool is_read_through() const { return contig1 == contig2 && breakpoint2 - breakpoint1 < 400000 && direction1 == DOWNSTREAM && direction2 == UPSTREAM; }
    bool is_internal_tandem_duplication(const unsigned int max_itd_length) const {
        return gene1 == gene2 && ((unsigned int)breakpoint2 - breakpoint1) < max_itd_length && direction1 == UPSTREAM && direction2 == DOWNSTREAM;
    }
    bool is_intragenic() const {
        return gene1 == gene2 || (breakpoint1 >= gene2->start - 10000 && breakpoint1 <= gene2->end + 10000 &&
                                  breakpoint2 >= gene1->start - 10000 && breakpoint2 <= gene1->end + 10000);
    }
    bool both_breakpoints_spliced() const {
        return spliced1 && spliced2 && ((gene1->strand == gene2->strand && direction1 != direction2) || (gene1->strand != gene2->strand && direction1 == direction2));
    }
};
typedef std::tuple<unsigned int, unsigned int, contig_t, contig_t, position_t, position_t, direction_t, direction_t> fusion_key_t;
typedef std::map<fusion_key_t, fusion_t> fusions_t;  // singlet: ordered (Arriba: unordered_map) for deterministic iteration

// ---- the annotation bundle ----
struct annotation_t {
    contigs_t contigs;
    std::list<gene_annotation_record_t> genes;
    std::list<transcript_annotation_record_t> transcripts;
    std::list<exon_annotation_record_t> exons;
    std::unordered_map<std::string, gene_t> gene_names;
    gene_annotation_index_t gene_index;
    exon_annotation_index_t exon_index;
};

namespace detail {
template <class Fn> inline bool for_each_line(const std::string& path, Fn&& fn) {
    gzFile f = gzopen(path.c_str(), "rb");
    if (!f) return false;
    std::string line;
    std::vector<char> buf(1 << 16);
    for (;;) {
        char* r = gzgets(f, buf.data(), (int)buf.size());
        if (!r) break;
        line += r;
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
            if (!line.empty() && line.back() == '\r') line.pop_back();
            fn(line);
            line.clear();
        }
    }
    if (!line.empty()) fn(line);
    gzclose(f);
    return true;
}
inline std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t a = 0;
    for (;;) {
        size_t b = s.find(sep, a);
        if (b == std::string::npos) { out.push_back(s.substr(a)); break; }
        out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}
}  // namespace detail

// annotation.cpp: get_gtf_attribute (one or several attribute names, the first one found wins)
inline bool get_gtf_attribute(const std::string& attributes, const std::vector<std::string>& names, std::string& value) {
    size_t start = std::string::npos;
    for (auto n = names.begin(); n != names.end() && start >= attributes.size(); ++n) start = attributes.find(*n + " \"");
    if (start < attributes.size()) start = attributes.find('"', start);
    if (start >= attributes.size()) return false;
    start++;
    size_t end = attributes.find('"', start);
    if (end >= attributes.size()) return false;
    value = attributes.substr(start, end - start);
    return true;
}

// annotation.cpp: remove_gene, remove_transcript
inline void remove_gene(const gene_t gene_to_remove, annotation_t& a) {
    for (auto e = a.exons.begin(); e != a.exons.end();) { if (e->gene == gene_to_remove) e = a.exons.erase(e); else ++e; }
    for (auto g = a.genes.begin(); g != a.genes.end(); ++g) if (&(*g) == gene_to_remove) { a.genes.erase(g); break; }
}
inline void remove_transcript(const transcript_t t, annotation_t& a) {
    gene_t gene = nullptr;
    for (auto e = a.exons.begin(); e != a.exons.end();) { if (e->transcript == t) { gene = e->gene; e = a.exons.erase(e); } else ++e; }
    if (!gene) return;
    position_t ns = -1, ne = -1;
    for (auto& e : a.exons) if (e.gene == gene) { if (ns == -1 || ns > e.start) ns = e.start; if (ne == -1 || ne < e.end) ne = e.end; }
    if (ns == -1) remove_gene(gene, a); else { gene->start = ns; gene->end = ne; }
}

// annotation.cpp: read_annotation_gtf (default GTF features: gene_name|gene_id, gene_id, transcript_id, exon, CDS)
inline bool read_annotation_gtf(const std::string& filename, const assembly_t& assembly, annotation_t& a) {
    const std::vector<std::string> f_gene_name = {"gene_name", "gene_id"}, f_gene_id = {"gene_id"}, f_transcript_id = {"transcript_id"};
    std::map<std::tuple<std::string, contig_t, strand_t>, transcript_t> transcripts;
    std::map<std::tuple<std::string, contig_t, strand_t>, gene_t> gene_by_id;
    std::map<std::tuple<std::string, contig_t, strand_t>, std::vector<exon_t>> exons_by_transcript_id;
    struct coding_region_t { strand_t strand; contig_t contig; position_t start, end; std::string transcript_id; };
    std::vector<coding_region_t> coding_regions;
    const int max_gene_size = 3000000;
    std::vector<gene_t> malformed_genes;
    std::vector<std::tuple<std::string, contig_t, strand_t>> malformed_transcripts;
    unsigned int new_id = 0, exon_id = 0;
    const bool ok = detail::for_each_line(filename, [&](const std::string& line) {
        if (line.empty() || line[0] == '#') return;
        auto t = detail::split(line, '\t');
        if (t.size() < 9 || t[0].empty() || t[2].empty() || t[6].empty()) return;
        const std::string& feature = t[2];
        if (feature != "exon" && feature != "CDS") return;
        std::string gene_name, gene_id;
        if (!get_gtf_attribute(t[8], f_gene_name, gene_name) || !get_gtf_attribute(t[8], f_gene_id, gene_id)) return;
        const std::string short_gene_id = strip_ensembl_version_number(gene_id);
        const contig_t contig = a.contigs.get_or_add(t[0]);
        const position_t start = std::atoi(t[3].c_str()) - 1, end = std::atoi(t[4].c_str()) - 1;  // GTF files are one-based
        const strand_t strand = t[6][0] == '+' ? FORWARD : REVERSE;
        if (feature == "exon") {
            std::string transcript_id;
            if (!get_gtf_attribute(t[8], f_transcript_id, transcript_id)) return;
            const std::string short_transcript_id = strip_ensembl_version_number(transcript_id);
            transcript_t& transcript = transcripts[std::make_tuple(short_transcript_id, contig, strand)];
            if (!transcript) {
                a.transcripts.emplace_back();
                transcript = &a.transcripts.back();
                transcript->id = new_id++;
                transcript->name = transcript_id;
            }
            gene_t& gene = gene_by_id[std::make_tuple(short_gene_id, contig, strand)];
            if (!gene) {
                a.genes.emplace_back();
                gene = &a.genes.back();
                gene->contig = contig; gene->start = start; gene->end = end; gene->strand = strand;
                gene->id = new_id++;
                gene->gene_id = gene_id; gene->name = gene_name;
            } else {
                if (gene->start > start) gene->start = start;
                if (gene->end < end) gene->end = end;
                if (gene->contig != contig || gene->end - gene->start > max_gene_size) malformed_genes.push_back(gene);
            }
            if (assembly.has(gene->contig) && gene->end >= assembly.size_of(gene->contig)) malformed_genes.push_back(gene);
            a.exons.emplace_back();
            exon_t e = &a.exons.back();
            e->contig = contig; e->start = start; e->end = end; e->strand = strand; e->id = exon_id++;
            e->transcript = transcript; e->gene = gene;
            exons_by_transcript_id[std::make_tuple(transcript_id, contig, strand)].push_back(e);
        } else {
            coding_region_t cr{strand, contig, start, end, ""};
            if (!get_gtf_attribute(t[8], f_transcript_id, cr.transcript_id)) return;
            coding_regions.push_back(cr);
        }
    });
    if (!ok || a.genes.empty()) return false;
    for (auto& cr : coding_regions) {
        auto tr = exons_by_transcript_id.find(std::make_tuple(cr.transcript_id, cr.contig, cr.strand));
        if (tr == exons_by_transcript_id.end()) continue;
        for (exon_t e : tr->second)
            if ((e->start <= cr.start && e->end >= cr.start) || (e->start <= cr.end && e->end >= cr.end) || (e->start >= cr.start && e->end <= cr.end)) {
                e->coding_region_start = std::max(cr.start, e->start);
                e->coding_region_end = std::min(cr.end, e->end);
                e->gene->is_protein_coding = true;
            }
    }
    for (auto& tr : exons_by_transcript_id) {
        std::sort(tr.second.begin(), tr.second.end(), [](const exon_t x, const exon_t y) { return *x < *y; });
        for (size_t i = 0; i < tr.second.size(); ++i) {
            tr.second[i]->previous_exon = i > 0 ? tr.second[i - 1] : nullptr;
            tr.second[i]->next_exon = i + 1 < tr.second.size() ? tr.second[i + 1] : nullptr;
        }
    }
    for (auto& e : a.exons) {
        if (!e.transcript->first_exon || e.start < e.transcript->first_exon->start) e.transcript->first_exon = &e;
        if (!e.transcript->last_exon || e.end > e.transcript->last_exon->end) e.transcript->last_exon = &e;
    }
    // fix some errors in the Gencode annotation
    auto fix = [&](const char* id, const char* c, strand_t s) { contig_t k = a.contigs.find(c); if (k >= 0) malformed_transcripts.emplace_back(id, k, s); };
    fix("ENST00000507166", "4", FORWARD); fix("ENST00000467125", "6", REVERSE);
    fix("ENST00000404796", "9", FORWARD); fix("ENST00000577563", "9", FORWARD); fix("ENST00000580900", "9", FORWARD);
    fix("ENSMUST00000124096", "7", REVERSE);
    for (auto& tr : transcripts)
        if (tr.second->last_exon && tr.second->first_exon && tr.second->last_exon->end - tr.second->first_exon->start > max_gene_size)
            malformed_transcripts.push_back(tr.first);
    for (auto& mt : malformed_transcripts) { auto it = transcripts.find(mt); if (it != transcripts.end()) remove_transcript(it->second, a); }
    std::sort(malformed_genes.begin(), malformed_genes.end());
    malformed_genes.erase(std::unique(malformed_genes.begin(), malformed_genes.end()), malformed_genes.end());
    for (gene_t g : malformed_genes) remove_gene(g, a);
    for (auto& g : a.genes) a.gene_names[g.name] = &g;
    return true;
}

// annotation.cpp: filter_exons_near_splice_site, is_breakpoint_spliced
inline bool filter_exons_near_splice_site(const gene_t gene, const direction_t direction, const position_t breakpoint, const exon_set_t& exons) {
    for (exon_t e : exons)
        if (e->gene == gene)
            if ((direction == UPSTREAM && std::abs(e->start - breakpoint) <= (int)MAX_SPLICE_SITE_DISTANCE &&
                 (e->previous_exon != nullptr || (e->previous_exon == nullptr && e->next_exon == nullptr && e->coding_region_start != -1) ||
                  e->start == e->coding_region_start)) ||
                (direction == DOWNSTREAM && std::abs(e->end - breakpoint) <= (int)MAX_SPLICE_SITE_DISTANCE &&
                 (e->next_exon != nullptr || (e->previous_exon == nullptr && e->next_exon == nullptr && e->coding_region_start != -1) ||
                  e->end == e->coding_region_end)))
                return true;
    return false;
}
inline bool is_breakpoint_spliced(const gene_t gene, const direction_t direction, const position_t breakpoint, const exon_annotation_index_t& index) {
    if ((unsigned int)gene->contig >= index.size() || index[gene->contig].empty()) return false;
    const auto& ci = index[gene->contig];
    size_t at = ci.lower_bound(breakpoint);
    if (at < ci.size()) {
        if (filter_exons_near_splice_site(gene, direction, breakpoint, ci.sets[at])) return true;
        if (at + 1 < ci.size() && filter_exons_near_splice_site(gene, direction, breakpoint, ci.sets[at + 1])) return true;
    }
    if (at != 0 && filter_exons_near_splice_site(gene, direction, breakpoint, ci.sets[at - 1])) return true;
    return false;
}

// annotation.cpp: annotate_alignment, annotate_alignments
inline void annotate_alignment(alignment_t& alignment, gene_set_t& gene_set, const exon_annotation_index_t& exon_index) {
    exon_set_t exon_set;
    get_annotation_by_coordinate(alignment.contig, alignment.start, alignment.end, exon_set, exon_index);
    for (exon_t e : exon_set) gene_set.insert_one(e->gene);
    if (alignment.cigar.size() > 1 && (gene_set.size() > 1 || alignment.predicted_strand_ambiguous)) {
        gene_set_t supported;
        position_t reference_position = alignment.start;
        for (unsigned int i = 0; i < alignment.cigar.size() && supported.empty(); ++i) {
            const uint32_t op = alignment.cigar.operation(i);
            if (op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP || op == BAM_CREF_SKIP) {
                supported = gene_set;
                for (auto g = supported.begin(); g != supported.end();) {
                    if (((op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP) &&
                         ((i == 0 && !is_breakpoint_spliced(*g, UPSTREAM, reference_position, exon_index)) ||
                          (i != 0 && !is_breakpoint_spliced(*g, DOWNSTREAM, reference_position, exon_index)))) ||
                        (op == BAM_CREF_SKIP && !is_breakpoint_spliced(*g, DOWNSTREAM, reference_position, exon_index) &&
                         !is_breakpoint_spliced(*g, UPSTREAM, reference_position + alignment.cigar.op_length(i), exon_index)))
                        g = supported.erase(g);
                    else
                        ++g;
                }
            }
            if (op == BAM_CREF_SKIP || op == BAM_CMATCH || op == BAM_CDIFF || op == BAM_CEQUAL || op == BAM_CDEL) reference_position += alignment.cigar.op_length(i);
        }
        if (!supported.empty()) {
            if (supported.size() < gene_set.size()) gene_set = supported;
            if (alignment.predicted_strand_ambiguous) {
                strand_t predicted_strand = (*supported.begin())->strand;
                alignment.predicted_strand_ambiguous = false;
                for (auto g = supported.begin(); g != supported.end() && !alignment.predicted_strand_ambiguous; ++g)
                    if ((*g)->strand != predicted_strand) alignment.predicted_strand_ambiguous = true;
                if (!alignment.predicted_strand_ambiguous) alignment.predicted_strand = predicted_strand;
            }
        }
    }
}
inline void annotate_alignments(mates_t& mates, const exon_annotation_index_t& exon_index) {
    for (auto& mate : mates) { annotate_alignment(mate, mate.genes, exon_index); mate.exonic = !mate.genes.empty(); }
    if (mates[MATE1].predicted_strand_ambiguous && !mates[MATE2].predicted_strand_ambiguous) {
        mates[MATE1].predicted_strand = complement_strand_if(mates[MATE2].predicted_strand, mates[MATE1].strand == mates[MATE2].strand);
        mates[MATE1].predicted_strand_ambiguous = false;
    } else if (!mates[MATE1].predicted_strand_ambiguous && mates[MATE2].predicted_strand_ambiguous) {
        mates[MATE2].predicted_strand = complement_strand_if(mates[MATE1].predicted_strand, mates[MATE1].strand == mates[MATE2].strand);
        mates[MATE2].predicted_strand_ambiguous = false;
    } else if (!mates[MATE1].predicted_strand_ambiguous && !mates[MATE2].predicted_strand_ambiguous) {
        if ((mates[MATE1].predicted_strand != mates[MATE2].predicted_strand) != (mates[MATE1].strand == mates[MATE2].strand)) {
            mates[MATE1].predicted_strand_ambiguous = true;
            mates[MATE2].predicted_strand_ambiguous = true;
        }
    }
    if (mates.size() == 3) {
        gene_set_t combined;
        combine_annotations(mates[SPLIT_READ].genes, mates[MATE1].genes, combined);
        if (mates[MATE1].genes.empty() || combined.size() < mates[MATE1].genes.size()) mates[MATE1].genes = combined;
        if (mates[SPLIT_READ].genes.empty() || combined.size() < mates[SPLIT_READ].genes.size()) mates[SPLIT_READ].genes = combined;
        if (mates[SPLIT_READ].predicted_strand_ambiguous && !mates[SUPPLEMENTARY].predicted_strand_ambiguous) {
            mates[MATE1].predicted_strand = complement_strand_if(mates[SUPPLEMENTARY].predicted_strand, mates[SUPPLEMENTARY].strand != mates[SPLIT_READ].strand);
            mates[MATE1].predicted_strand_ambiguous = false;
            mates[SPLIT_READ].predicted_strand = mates[MATE1].predicted_strand;
            mates[SPLIT_READ].predicted_strand_ambiguous = false;
        } else if (!mates[SPLIT_READ].predicted_strand_ambiguous && mates[SUPPLEMENTARY].predicted_strand_ambiguous) {
            mates[SUPPLEMENTARY].predicted_strand = complement_strand_if(mates[SPLIT_READ].predicted_strand, mates[SUPPLEMENTARY].strand != mates[SPLIT_READ].strand);
            mates[SUPPLEMENTARY].predicted_strand_ambiguous = false;
        } else if (!mates[SPLIT_READ].predicted_strand_ambiguous && !mates[SUPPLEMENTARY].predicted_strand_ambiguous) {
            if ((mates[SPLIT_READ].predicted_strand != mates[SUPPLEMENTARY].predicted_strand) != (mates[SPLIT_READ].strand != mates[SUPPLEMENTARY].strand)) {
                mates[MATE1].predicted_strand_ambiguous = true;
                mates[SPLIT_READ].predicted_strand_ambiguous = true;
                mates[SUPPLEMENTARY].predicted_strand_ambiguous = true;
            }
        }
    }
}

// annotation.cpp: get_boundaries_of_biggest_gene, get_spliced_distance
inline void get_boundaries_of_biggest_gene(const gene_set_t& genes, position_t& start, position_t& end) {
    start = -1; end = -1;
    for (gene_t g : genes) { if (start == -1 || start > g->start) start = g->start; if (end == -1 || end < g->end) end = g->end; }
}
inline int get_spliced_distance(const contig_t contig, position_t position1, position_t position2, const gene_t gene, const exon_annotation_index_t& index) {
    if (position1 > position2) std::swap(position1, position2);
    if ((unsigned int)contig >= index.size() || index[contig].empty()) return position2 - position1;
    const auto& ci = index[contig];
    size_t k = ci.lower_bound(position1);
    int distance = 0;
    if (k < ci.size() && ci.keys[k] < position2) { distance += ci.keys[k] - position1; position1 = ci.keys[k]; }
    for (; k < ci.size() && ci.keys[k] < position2; ++k) {
        if (ci.keys[k] >= position1) {
            position_t fs = -1, fe = -1, fk = -1;
            for (exon_t e : ci.sets[k]) {
                if (e->gene == gene && e->next_exon != nullptr && e->next_exon->start <= position2) {
                    position_t es = std::max(position1, e->start), ee = std::min(position2, e->end), sk = e->next_exon->start - es + 1;
                    if (fs == -1 || 1.0 * (ee - es) / sk < 1.0 * (fe - fs) / fk) { fs = es; fe = ee; fk = sk; }
                }
            }
            if (fs != -1) { distance += fe - fs + 1; position1 = fs + fk - 1; }
        }
    }
    distance += position2 - position1;
    return distance;
}

// ---- reading the engine's chimeric reads (singlet: replaces read_chimeric_alignments.cpp, which reads a BAM) ----
struct input_t {
    std::string junction_path;                 // Chimeric.out.junction (format 1, CB and UMI last)
    std::vector<std::string> barcodes;         // allowed cell barcodes (exact or unique one-mismatch match)
    std::vector<std::string> fastq_paths;      // reads of the chimeric pass (names r<n>); empty = no read sequences
};

// read_chimeric_alignments.cpp: disjoin_split_read_segments
inline bool disjoin_split_read_segments(alignment_t& split_read, alignment_t& supplementary, size_t sequence_length) {
    const int min_remaining_supplementary_segment = 10;
    unsigned int clipped_split = (split_read.strand == FORWARD) ? split_read.preclipping() : split_read.postclipping();
    unsigned int clipped_supp = (supplementary.strand == FORWARD) ? supplementary.postclipping() : supplementary.preclipping();
    int overlap = (int)sequence_length - clipped_split - clipped_supp;
    if (overlap <= 0) return true;
    unsigned int clipped_op = (supplementary.strand == FORWARD) ? supplementary.cigar.size() - 1 : 0;
    unsigned int matching_op = (supplementary.strand == FORWARD) ? clipped_op - 1 : 1;
    if (supplementary.cigar.size() < 2 || supplementary.cigar.operation(matching_op) != BAM_CMATCH ||
        (int)supplementary.cigar.op_length(matching_op) < overlap + min_remaining_supplementary_segment)
        return false;
    supplementary.cigar[clipped_op].second += overlap;
    supplementary.cigar[matching_op].second -= overlap;
    if (supplementary.strand == FORWARD) supplementary.end -= overlap; else supplementary.start += overlap;
    return true;
}

// Builds Arriba's single-end split-read record (read_chimeric_alignments.cpp: add_chimeric_alignment and the single-end branch of
// remove_malformed_alignments) from the two segments of one junction line. Returns false for a malformed record.
inline bool make_split_read(alignment_t a, alignment_t b, const std::string* read_sequence, mates_t& mates) {
    a.supplementary = false; b.supplementary = true;  // the line's first segment stands for the BAM's primary record
    mates.clear();
    mates.push_back(a);
    mates.push_back(b);
    if (mates[MATE1].end - mates[MATE1].start > mates[MATE2].end - mates[MATE2].start) {
        mates.push_back(mates[MATE2]);
        mates[MATE2] = mates[MATE1];
    } else {
        mates.push_back(mates[MATE1]);
        mates[MATE1] = mates[MATE2];
    }
    // singlet: the sequence is the read in the split read's own reference orientation, as BAM SEQ stores it
    const int L = cigar_query_len(mates[SPLIT_READ].cigar);
    mates.has_sequence = false;
    if (read_sequence && (int)read_sequence->size() == L) {
        mates.sequence = mates[SPLIT_READ].strand == REVERSE ? dna_to_reverse_complement(*read_sequence) : *read_sequence;
        mates.has_sequence = true;
    } else {
        mates.sequence.assign(L, 'N');
    }
    for (unsigned int k : {MATE1, SPLIT_READ}) {
        auto& c = mates[k].cigar;
        if (c[0].first == BAM_CHARD_CLIP) c[0].first = BAM_CSOFT_CLIP;
        if (c[c.size() - 1].first == BAM_CHARD_CLIP) c[c.size() - 1].first = BAM_CSOFT_CLIP;
    }
    mates[SUPPLEMENTARY].supplementary = true;
    mates[MATE1].supplementary = false;
    mates[SPLIT_READ].supplementary = false;
    const unsigned int len = mates.sequence.length();
    const bool same = mates[SPLIT_READ].strand == mates[SUPPLEMENTARY].strand;
    bool flip_mate1_strand;
    if (len - mates[SPLIT_READ].preclipping() - (same ? mates[SUPPLEMENTARY].postclipping() : mates[SUPPLEMENTARY].preclipping()) <
        len - mates[SPLIT_READ].postclipping() - (same ? mates[SUPPLEMENTARY].preclipping() : mates[SUPPLEMENTARY].postclipping()))
        flip_mate1_strand = mates[SPLIT_READ].strand == FORWARD;
    else
        flip_mate1_strand = mates[SPLIT_READ].strand == REVERSE;
    mates[MATE1].strand = complement_strand_if(mates[MATE1].strand, flip_mate1_strand);
    mates[SPLIT_READ].strand = complement_strand_if(mates[SPLIT_READ].strand, !flip_mate1_strand);
    mates[SUPPLEMENTARY].strand = complement_strand_if(mates[SUPPLEMENTARY].strand, !flip_mate1_strand);
    mates[MATE1].first_in_pair = !flip_mate1_strand;
    mates[SPLIT_READ].first_in_pair = flip_mate1_strand;
    mates[SUPPLEMENTARY].first_in_pair = flip_mate1_strand;
    return disjoin_split_read_segments(mates[SPLIT_READ], mates[SUPPLEMENTARY], mates.sequence.length());
}

struct load_stats_t {
    uint64_t lines = 0, no_barcode = 0, unmatched = 0, ambiguous = 0, malformed = 0, loaded = 0, with_sequence = 0;
};

// Loads the junction lines of matched cells. Read sequences come from the FASTQ files when given (only the names in the junction file are kept).
inline bool read_chimeric_junctions(const input_t& in, contigs_t& contigs, chimeric_alignments_t& out, load_stats_t& st) {
    std::unordered_map<std::string, uint32_t> bc_index;
    for (uint32_t i = 0; i < in.barcodes.size(); ++i) bc_index.emplace(in.barcodes[i], i);
    auto resolve = [&](const std::string& raw, bool& amb) -> int64_t {
        auto it = bc_index.find(raw);
        if (it != bc_index.end()) return it->second;
        static const char kB[4] = {'A', 'C', 'G', 'T'};
        int64_t hit = -1; int n = 0;
        std::string t = raw;
        for (size_t i = 0; i < t.size(); ++i) {
            const char o = t[i];
            for (char b : kB) { if (b == o) continue; t[i] = b; auto j = bc_index.find(t); if (j != bc_index.end()) { hit = j->second; ++n; } }
            t[i] = o;
        }
        if (n == 1) return hit;
        amb = n > 1;
        return -1;
    };
    struct line_t { std::string name; alignment_t a, b; uint32_t cell; std::string umi; int score; };
    std::vector<line_t> lines;
    std::unordered_map<std::string, int> hits;  // HI numbering per read name
    const bool ok = detail::for_each_line(in.junction_path, [&](const std::string& ln) {
        if (ln.empty() || ln[0] == '#' || ln.compare(0, 10, "chr_donorA") == 0) return;
        auto f = detail::split(ln, '\t');
        st.lines++;
        if (f.size() < 22) { st.no_barcode++; return; }
        const std::string& cb = f[f.size() - 2];
        const std::string& umi = f[f.size() - 1];
        auto acgtn = [](const std::string& s) { return !s.empty() && s.find_first_not_of("ACGTN") == std::string::npos; };
        if (!acgtn(cb) || !acgtn(umi)) { st.no_barcode++; return; }
        bool amb = false;
        const int64_t c = resolve(cb, amb);
        if (c < 0) { (amb ? st.ambiguous : st.unmatched)++; return; }
        line_t l;
        l.a.contig = contigs.get_or_add(f[0]);
        l.a.strand = f[2] == "-" ? REVERSE : FORWARD;
        l.b.contig = contigs.get_or_add(f[3]);
        l.b.strand = f[5] == "-" ? REVERSE : FORWARD;
        if (!parse_cigar(f[11], l.a.cigar) || !parse_cigar(f[13], l.b.cigar)) { st.malformed++; return; }
        l.a.start = std::atoi(f[10].c_str()) - 1; l.a.end = l.a.start + cigar_ref_len(l.a.cigar) - 1;
        l.b.start = std::atoi(f[12].c_str()) - 1; l.b.end = l.b.start + cigar_ref_len(l.b.cigar) - 1;
        l.cell = (uint32_t)c; l.umi = umi; l.score = std::atoi(f[17].c_str());
        l.name = f[9] + "," + std::to_string(++hits[f[9]]);
        lines.push_back(std::move(l));
    });
    if (!ok) return false;
    hits.clear();
    // read sequences for the names in the junction file
    std::unordered_map<std::string, std::string> seqs;
    if (!in.fastq_paths.empty()) {
        std::unordered_set<std::string> want;
        for (const auto& l : lines) want.insert(strip_hi_tag_from_read_name(l.name));
        for (const auto& fq : in.fastq_paths) {
            uint64_t k = 0;
            std::string name;
            detail::for_each_line(fq, [&](const std::string& ln) {
                if (k % 4 == 0) { name = ln.substr(1, ln.find_first_of(" \t") == std::string::npos ? std::string::npos : ln.find_first_of(" \t") - 1); }
                else if (k % 4 == 1 && want.count(name)) seqs[name] = ln;
                ++k;
            });
        }
    }
    for (auto& l : lines) {
        auto s = seqs.find(strip_hi_tag_from_read_name(l.name));
        mates_t m;
        if (!make_split_read(l.a, l.b, s == seqs.end() ? nullptr : &s->second, m)) { st.malformed++; continue; }
        m.cell = l.cell; m.umi = l.umi; m.star_score = l.score;
        if (m.has_sequence) st.with_sequence++;
        st.loaded++;
        out[l.name] = std::move(m);
    }
    return true;
}

// read_chimeric_alignments.cpp: mark_multimappers
inline unsigned int mark_multimappers(chimeric_alignments_t& ca) {
    unsigned int count = 0;
    if (ca.empty()) return 0;
    for (auto it = ca.begin(); std::next(it) != ca.end(); ++it)
        if (strip_hi_tag_from_read_name(it->first) == strip_hi_tag_from_read_name(std::next(it)->first)) {
            it->second.multimapper = true;
            std::next(it)->second.multimapper = true;
            count++;
        }
    return count;
}

// singlet: duplicate marking by cell barcode, UMI and breakpoints (the external marking Arriba relies on with -u)
inline void mark_umi_duplicates(chimeric_alignments_t& ca) {
    std::set<std::tuple<uint32_t, std::string, contig_t, position_t, contig_t, position_t>> seen;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        const position_t b1 = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
        const position_t b2 = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start;
        m.duplicate = !seen.emplace(m.cell, m.umi, m[SPLIT_READ].contig, b1, m[SUPPLEMENTARY].contig, b2).second;
    }
}

// read_stats.cpp: detect_strandedness
inline strandedness_t detect_strandedness(const chimeric_alignments_t& ca, const gene_annotation_index_t& gene_index, const exon_annotation_index_t& exon_index) {
    const unsigned int sample_size = 100;
    const float threshold = 0.95;
    unsigned int count = 0, matching_strand = 0;
    for (const auto& kv : ca) {
        const mates_t& m = kv.second;
        if (m.size() != 3) continue;
        if (m[SPLIT_READ].contig == m[SUPPLEMENTARY].contig && m[SPLIT_READ].strand == m[SUPPLEMENTARY].strand &&
            std::abs(m[SPLIT_READ].start - m[SUPPLEMENTARY].start) < 400000) {
            gene_set_t genes;
            get_annotation_by_coordinate(m[SPLIT_READ].contig, m[SPLIT_READ].start, m[SPLIT_READ].end, genes, gene_index);
            if (genes.size() == 1) {
                direction_t direction = (m[SPLIT_READ].strand == FORWARD) ? UPSTREAM : DOWNSTREAM;
                position_t position = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
                if (is_breakpoint_spliced(*genes.begin(), direction, position, exon_index)) {
                    if ((m[SPLIT_READ].first_in_pair && m[SPLIT_READ].strand == (*genes.begin())->strand) ||
                        (m[MATE1].first_in_pair && m[MATE1].strand == (*genes.begin())->strand))
                        matching_strand++;
                    count++;
                    if (count >= sample_size) break;
                }
            }
        }
    }
    if (count < sample_size) return STRANDEDNESS_NO;
    if (matching_strand < (1 - threshold) * count) return STRANDEDNESS_REVERSE;
    if (matching_strand > threshold * count) return STRANDEDNESS_YES;
    return STRANDEDNESS_NO;
}

// read_chimeric_alignments.cpp: assign_strands_from_strandedness
inline void assign_strands_from_strandedness(chimeric_alignments_t& ca, const strandedness_t strandedness) {
    if (strandedness == STRANDEDNESS_NO) return;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        unsigned int fip = m[MATE1].first_in_pair ? MATE1 : MATE2, sip = m[MATE1].first_in_pair ? MATE2 : MATE1;
        m[fip].predicted_strand = complement_strand_if(m[fip].strand, strandedness == STRANDEDNESS_REVERSE);
        m[fip].predicted_strand_ambiguous = false;
        m[sip].predicted_strand = complement_strand_if(m[fip].predicted_strand, m[fip].strand == m[sip].strand);
        m[sip].predicted_strand_ambiguous = false;
        if (m.size() == 3) {
            m[SUPPLEMENTARY].predicted_strand = complement_strand_if(m[SPLIT_READ].predicted_strand, m[SUPPLEMENTARY].strand != m[SPLIT_READ].strand);
            m[SUPPLEMENTARY].predicted_strand_ambiguous = false;
        }
    }
}

// arriba.cpp (lines ~180-320): exonic lengths, gene annotation of alignments, dummy genes for intergenic breakpoints, gene IDs
inline void annotate_all(chimeric_alignments_t& ca, annotation_t& a) {
    for (auto& ci : a.exon_index) {
        position_t region_start = 0;
        for (size_t r = 0; r < ci.size(); ++r) {
            gene_t previous_gene = nullptr;
            for (exon_t e : ci.sets[r]) {
                gene_t current = e->gene;
                if (previous_gene != current) { current->exonic_length += ci.keys[r] - region_start; previous_gene = current; }
            }
            region_start = ci.keys[r];
        }
    }
    for (auto& g : a.genes) if (g.exonic_length == 0) g.exonic_length = g.end - g.start;
    for (auto& kv : ca) annotate_alignments(kv.second, a.exon_index);
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        for (auto& mate : m) if (mate.genes.empty()) get_annotation_by_coordinate(mate.contig, mate.start, mate.end, mate.genes, a.gene_index);
        if (m.size() == 3) {
            gene_set_t combined;
            combine_annotations(m[SPLIT_READ].genes, m[MATE1].genes, combined);
            if (m[MATE1].genes.empty() || combined.size() < m[MATE1].genes.size()) m[MATE1].genes = combined;
            if (m[SPLIT_READ].genes.empty() || combined.size() < m[SPLIT_READ].genes.size()) m[SPLIT_READ].genes = combined;
        }
    }
    // dummy genes subsume unannotated breakpoints within 10 kb
    struct rec_t { contig_t contig; position_t start, end; bool operator<(const rec_t& x) const { if (contig != x.contig) return contig < x.contig; if (end != x.end) return end < x.end; return start < x.start; } };
    std::vector<rec_t> unmapped;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m[SPLIT_READ].genes.empty()) { position_t p = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end; unmapped.push_back({m[SPLIT_READ].contig, p, p}); }
        if (m[SUPPLEMENTARY].genes.empty()) { position_t p = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start; unmapped.push_back({m[SUPPLEMENTARY].contig, p, p}); }
    }
    unsigned int next_id = 0;
    for (auto& g : a.genes) next_id = std::max(next_id, g.id + 1);
    if (!unmapped.empty()) {
        std::stable_sort(unmapped.begin(), unmapped.end());
        gene_annotation_record_t rec;
        rec.contig = unmapped[0].contig; rec.start = unmapped[0].start; rec.end = unmapped[0].end;
        rec.strand = FORWARD; rec.exonic_length = 10000; rec.is_dummy = true; rec.is_protein_coding = false;
        auto next_known = [&](contig_t c, position_t p) -> position_t {  // key of gene_annotation_index[c].lower_bound(p), INT_MAX at the end
            if ((size_t)c >= a.gene_index.size()) return INT_MAX;
            size_t k = a.gene_index[c].lower_bound(p);
            return k < a.gene_index[c].size() ? a.gene_index[c].keys[k] : INT_MAX;
        };
        position_t next_known_gene = next_known(unmapped[0].contig, unmapped[0].end);
        for (size_t u = 1;; ++u) {
            if (u == unmapped.size() || rec.end + 10000 < unmapped[u].start || (next_known_gene != INT_MAX && next_known_gene <= unmapped[u].start) ||
                unmapped[u].contig != rec.contig) {
                rec.id = next_id++;
                a.genes.push_back(rec);
                if (u == unmapped.size()) break;
                rec.contig = unmapped[u].contig; rec.start = unmapped[u].start;
                next_known_gene = next_known(unmapped[u].contig, unmapped[u].end);
            }
            rec.end = unmapped[u].end;
        }
    }
    make_annotation_index(a.genes, a.gene_index, a.contigs.size());
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m[MATE1].genes.empty() || m[SPLIT_READ].genes.empty()) {
            position_t p = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
            get_annotation_by_coordinate(m[SPLIT_READ].contig, p, p, m[SPLIT_READ].genes, a.gene_index);
            m[MATE1].genes = m[SPLIT_READ].genes;
        }
        if (m[SUPPLEMENTARY].genes.empty()) {
            position_t p = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start;
            get_annotation_by_coordinate(m[SUPPLEMENTARY].contig, p, p, m[SUPPLEMENTARY].genes, a.gene_index);
        }
    }
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        for (auto& mate : m) {
            if (mate.genes.size() > 1 && mate.genes[0]->is_dummy) {
                position_t bp = (mate.strand == FORWARD) ? mate.start : mate.end;
                gene_t enc = m[MATE1].genes.empty() ? mate.genes[0] : m[MATE1].genes[0];
                for (gene_t d : mate.genes) if (d->start <= bp && d->end >= bp) enc = d;
                mate.genes.clear();
                mate.genes.push_back(enc);
            }
        }
        if (m.size() == 3 && !m[MATE1].genes.empty() && !m[SPLIT_READ].genes.empty() && m[MATE1].genes[0] != m[SPLIT_READ].genes[0] &&
            m[MATE1].genes[0]->is_dummy && m[SPLIT_READ].genes[0]->is_dummy) {
            const position_t bp = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
            gene_t enc = m[MATE1].genes[0];
            for (gene_t d : m[MATE1].genes) if (d->start <= bp && d->end >= bp) enc = d;
            for (gene_t d : m[SPLIT_READ].genes) if (d->start <= bp && d->end >= bp) enc = d;
            m[MATE1].genes.clear(); m[SPLIT_READ].genes.clear();
            m[MATE1].genes.push_back(enc); m[SPLIT_READ].genes.push_back(enc);
        }
    }
    // assign IDs to genes (deterministic hashing of fusions); the gene sets keep their order because IDs follow list order
    unsigned int gene_id = 0;
    for (auto& g : a.genes) g.id = gene_id++;
}

// ================================================================= read-level filters

// filter_duplicates.cpp: filter_duplicates (external duplicate marking)
inline unsigned int filter_duplicates(chimeric_alignments_t& ca) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        if (kv.second.filter != FILTER_none) continue;
        if (kv.second.duplicate) kv.second.filter = FILTER_duplicates; else remaining++;
    }
    return remaining;
}

// filter_uninteresting_contigs.cpp: filter_uninteresting_contigs
inline unsigned int filter_uninteresting_contigs(chimeric_alignments_t& ca, const std::vector<bool>& interesting) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        if (kv.second.filter != FILTER_none) continue;
        bool ok = true;
        for (auto& mate : kv.second) if (!interesting[mate.contig]) { ok = false; break; }
        if (ok) ++remaining; else kv.second.filter = FILTER_uninteresting_contigs;
    }
    return remaining;
}

// filter_proximal_read_through.cpp: filter_proximal_read_through
inline unsigned int filter_proximal_read_through(chimeric_alignments_t& ca, const int min_distance) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        alignment_t *fwd, *rev;
        if (m.size() == 2) { fwd = &(m[MATE1].strand == FORWARD ? m[MATE1] : m[MATE2]); rev = &(m[MATE1].strand == FORWARD ? m[MATE2] : m[MATE1]); }
        else { fwd = &(m[SPLIT_READ].strand == FORWARD ? m[SUPPLEMENTARY] : m[SPLIT_READ]); rev = &(m[SPLIT_READ].strand == FORWARD ? m[SPLIT_READ] : m[SUPPLEMENTARY]); }
        if ((m.size() == 2 && fwd->strand != rev->strand && fwd->contig == rev->contig && fwd->end < rev->start) ||
            (m.size() == 3 && fwd->strand == rev->strand && fwd->contig == rev->contig && fwd->end < rev->start)) {
            position_t fs, fe, rs, re;
            get_boundaries_of_biggest_gene(fwd->genes, fs, fe);
            get_boundaries_of_biggest_gene(rev->genes, rs, re);
            if (fwd->end >= rs - min_distance || rev->start <= fe + min_distance) { m.filter = FILTER_read_through; continue; }
        }
        ++remaining;
    }
    return remaining;
}

// filter_inconsistently_clipped.cpp: filter_inconsistently_clipped_mates
inline unsigned int filter_inconsistently_clipped_mates(chimeric_alignments_t& ca) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        if (m.size() == 3 && ((m[MATE1].strand == FORWARD && m[MATE1].end > m[SPLIT_READ].end + 3) ||
                              (m[MATE1].strand == REVERSE && m[MATE1].start < m[SPLIT_READ].start - 3))) {
            m.filter = FILTER_inconsistently_clipped;
            continue;
        }
        ++remaining;
    }
    return remaining;
}

// filter_homopolymer.cpp: is_split_read_spliced, filter_homopolymer
inline bool is_split_read_spliced(const alignment_t& split_read, const exon_annotation_index_t& exon_index) {
    direction_t direction = (split_read.strand == FORWARD) ? UPSTREAM : DOWNSTREAM;
    position_t breakpoint = (split_read.strand == FORWARD) ? split_read.start : split_read.end;
    for (gene_t g : split_read.genes) if (is_breakpoint_spliced(g, direction, breakpoint, exon_index)) return true;
    return false;
}
inline unsigned int filter_homopolymer(chimeric_alignments_t& ca, const unsigned int homopolymer_length, const exon_annotation_index_t& exon_index) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        if (m.size() == 3 && m.has_sequence) {
            const alignment_t& sr = m[SPLIT_READ];
            const std::string& s = m.sequence;
            std::string sequence;
            if (sr.strand == FORWARD) {
                if (sr.preclipping() >= homopolymer_length) sequence += s.substr(sr.preclipping() - homopolymer_length, homopolymer_length) + " ";
                if (s.length() - sr.preclipping() >= homopolymer_length) sequence += s.substr(sr.preclipping(), homopolymer_length) + " ";
            } else {
                if (sr.postclipping() >= homopolymer_length) sequence += s.substr(s.length() - sr.postclipping(), homopolymer_length) + " ";
                if (s.length() - sr.postclipping() >= homopolymer_length) sequence += s.substr(s.length() - sr.postclipping() - homopolymer_length, homopolymer_length) + " ";
            }
            unsigned int run = 1;
            bool hit = false;
            for (unsigned int c = 1; c < sequence.length() && !hit; c++) {
                if (sequence[c - 1] == sequence[c]) {
                    run++;
                    if (run == homopolymer_length && !is_split_read_spliced(sr, exon_index)) hit = true;
                } else {
                    run = 1;
                }
            }
            if (hit) { m.filter = FILTER_homopolymer; continue; }
        }
        ++remaining;
    }
    return remaining;
}

// filter_small_insert_size.cpp: filter_small_insert_size (discordant mates only: a no-op on single-end reads)
inline unsigned int filter_small_insert_size(chimeric_alignments_t& ca, const unsigned int max_overhang) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        if (m.size() == 2 && m[MATE1].strand != m[MATE2].strand && m[MATE1].contig == m[MATE2].contig &&
            ((unsigned int)std::abs(m[MATE1].start - m[MATE2].start) <= max_overhang || (unsigned int)std::abs(m[MATE1].end - m[MATE2].end) <= max_overhang)) {
            m.filter = FILTER_small_insert_size;
            continue;
        }
        ++remaining;
    }
    return remaining;
}

// filter_long_gap.cpp: filter_long_gap
inline unsigned int filter_long_gap(chimeric_alignments_t& ca) {
    const int min_long_gap = 700000, max_long_gap = 1500000;
    const unsigned int short_segment = 15;
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        int size_of_deletion = 0;
        if (m.size() == 3 && m[SPLIT_READ].contig == m[SUPPLEMENTARY].contig) {
            if (m[SPLIT_READ].strand == REVERSE && m[SUPPLEMENTARY].strand == REVERSE) size_of_deletion = m[SUPPLEMENTARY].start - m[SPLIT_READ].end;
            else if (m[SPLIT_READ].strand == FORWARD && m[SUPPLEMENTARY].strand == FORWARD) size_of_deletion = m[SPLIT_READ].start - m[SUPPLEMENTARY].end;
        }
        bool filtered = false;
        for (auto& mate : m) {
            for (unsigned int i = 1; i + 1 < mate.cigar.size() && !filtered; ++i) {
                if (mate.cigar.operation(i) == BAM_CREF_SKIP && ((int)mate.cigar.op_length(i) >= min_long_gap || (size_of_deletion >= min_long_gap && size_of_deletion <= max_long_gap))) {
                    unsigned int left = 0, right = 0;
                    for (int j = (int)i - 1; j >= 0; --j) {
                        uint32_t op = mate.cigar.operation(j);
                        if (op == BAM_CMATCH || op == BAM_CDIFF || op == BAM_CEQUAL) left += mate.cigar.op_length(j);
                        else if (op == BAM_CDEL || op == BAM_CINS || op == BAM_CPAD) continue;
                        else break;
                    }
                    for (unsigned int j = i + 1; j < mate.cigar.size(); ++j) {
                        uint32_t op = mate.cigar.operation(j);
                        if (op == BAM_CMATCH || op == BAM_CDIFF || op == BAM_CEQUAL) right += mate.cigar.op_length(j);
                        else if (op == BAM_CDEL || op == BAM_CINS || op == BAM_CPAD) continue;
                        else break;
                    }
                    if (left <= short_segment && right <= short_segment) filtered = true;
                }
            }
            if (filtered) break;
        }
        if (filtered) m.filter = FILTER_long_gap; else remaining++;
    }
    return remaining;
}

// filter_same_gene.cpp: filter_same_gene
inline unsigned int filter_same_gene(chimeric_alignments_t& ca) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        gene_set_t common;
        if (m.size() == 2) combine_annotations(m[MATE1].genes, m[MATE2].genes, common, false);
        else combine_annotations(m[MATE2].genes, m[SUPPLEMENTARY].genes, common, false);
        if (common.empty()) { remaining++; continue; }
        if (m.size() == 2) {
            if ((m[MATE1].strand == FORWARD && m[MATE2].strand == REVERSE && m[MATE1].start <= m[MATE2].end) ||
                (m[MATE1].strand == REVERSE && m[MATE2].strand == FORWARD && m[MATE1].end >= m[MATE2].start)) { m.filter = FILTER_same_gene; continue; }
        } else {
            if ((m[SPLIT_READ].strand == FORWARD && m[SUPPLEMENTARY].strand == FORWARD && m[SPLIT_READ].start >= m[SUPPLEMENTARY].end) ||
                (m[SPLIT_READ].strand == REVERSE && m[SUPPLEMENTARY].strand == REVERSE && m[SPLIT_READ].end <= m[SUPPLEMENTARY].start)) { m.filter = FILTER_same_gene; continue; }
        }
        remaining++;
    }
    return remaining;
}

// filter_hairpin.cpp: is_breakpoint_within_aligned_segment, filter_hairpin
inline bool is_breakpoint_within_aligned_segment(const position_t breakpoint, const alignment_t& alignment) {
    position_t ref = alignment.start;
    for (unsigned int i = 0; i < alignment.cigar.size(); ++i) {
        switch (alignment.cigar.operation(i)) {
            case BAM_CREF_SKIP: case BAM_CDEL: ref += alignment.cigar.op_length(i); break;
            case BAM_CMATCH: case BAM_CDIFF: case BAM_CEQUAL:
                if (breakpoint >= ref && breakpoint <= ref + (int)alignment.cigar.op_length(i)) return true;
                ref += alignment.cigar.op_length(i);
                break;
        }
    }
    return false;
}
inline unsigned int filter_hairpin(chimeric_alignments_t& ca, const int /*max_mate_gap*/) {
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        gene_set_t common;
        if (m.size() == 2) {
            combine_annotations(m[MATE1].genes, m[MATE2].genes, common, false);
            if (common.empty() && m[MATE1].contig != m[MATE2].contig) { remaining++; continue; }
        } else {
            combine_annotations(m[SPLIT_READ].genes, m[SUPPLEMENTARY].genes, common, false);
            if (common.empty() && m[SPLIT_READ].contig != m[SUPPLEMENTARY].contig) { remaining++; continue; }
        }
        if (m.size() == 2) {
            position_t b1 = (m[MATE1].strand == FORWARD) ? m[MATE1].end : m[MATE1].start;
            position_t b2 = (m[MATE2].strand == FORWARD) ? m[MATE2].end : m[MATE2].start;
            if (is_breakpoint_within_aligned_segment(b1, m[MATE2]) || is_breakpoint_within_aligned_segment(b2, m[MATE1])) { m.filter = FILTER_hairpin; continue; }
        } else {
            position_t bs = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
            position_t bu = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start;
            if (is_breakpoint_within_aligned_segment(bs, m[SUPPLEMENTARY]) || is_breakpoint_within_aligned_segment(bu, m[SPLIT_READ]) ||
                is_breakpoint_within_aligned_segment(bu, m[MATE1])) { m.filter = FILTER_hairpin; continue; }
        }
        remaining++;
    }
    return remaining;
}

// filter_mismatches.cpp: count_mismatches, calculate_binomial_coefficient, calculate_binomial_distribution, test_mismatch_probability,
// filter_mismatches
inline void count_mismatches(const alignment_t& alignment, const std::string& sequence, const assembly_t& assembly, unsigned int& mismatches, unsigned int& alignment_length) {
    mismatches = 0; alignment_length = 0;
    position_t ref = alignment.start, rp = 0;
    for (unsigned int i = 0; i < alignment.cigar.size(); ++i) {
        const uint32_t op = alignment.cigar.operation(i), n = alignment.cigar.op_length(i);
        switch (op) {
            case BAM_CSOFT_CLIP: case BAM_CHARD_CLIP:
                rp += n;
                if (!((i == 0 && alignment.strand == REVERSE) || (i == alignment.cigar.size() - 1 && alignment.strand == FORWARD))) mismatches++;
                break;
            case BAM_CDEL: mismatches++; ref += n; break;
            case BAM_CREF_SKIP: ref += n; break;
            case BAM_CINS: mismatches++; rp += n; break;
            case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF:
                for (unsigned int k = 1; k <= n; ++k) {
                    if (rp < (position_t)sequence.size() && sequence[rp] != 'N') { if (sequence[rp] != assembly.at(alignment.contig, ref)) mismatches++; alignment_length++; }
                    ref++; rp++;
                }
                break;
        }
    }
}
inline double calculate_binomial_coefficient(const unsigned int k, const unsigned int n) {
    double r = 1;
    for (unsigned int i = n - k + 1; i <= n; ++i) r *= i;
    for (unsigned int i = 1; i <= k; ++i) r /= i;
    return r;
}
inline float calculate_binomial_distribution(const unsigned int k, const unsigned int n, const float p) {
    return calculate_binomial_coefficient(k, n) * std::pow(p, k) * std::pow(1 - p, n - k);
}
inline bool test_mismatch_probability(const alignment_t& alignment, const std::string& sequence, const assembly_t& assembly, const float mismatch_probability,
                                      unsigned long genome_size, const float pvalue_cutoff, const bool is_multimapper) {
    unsigned int mismatches, alignment_length;
    count_mismatches(alignment, sequence, assembly, mismatches, alignment_length);
    if (is_multimapper) mismatches += 2;
    if (calculate_binomial_distribution(mismatches, alignment_length, mismatch_probability) < pvalue_cutoff) return true;
    if (mismatches > 0) {
        long double perms = std::pow((long double)4, (long double)alignment_length - mismatches);
        if (genome_size >= perms) return true;
        return (1 - std::pow(1 - genome_size / perms, calculate_binomial_coefficient(mismatches, alignment_length))) > 0.01;
    }
    return false;
}
inline unsigned int filter_mismatches(chimeric_alignments_t& ca, const assembly_t& assembly, const std::vector<bool>& interesting, const std::vector<bool>& viral,
                                      const float mismatch_probability, const float pvalue_cutoff) {
    unsigned long genome_size = 0;
    for (contig_t c = 0; c < (contig_t)interesting.size(); ++c) if (interesting[c]) genome_size += assembly.size_of(c);
    unsigned int remaining = 0;
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        if (m.filter != FILTER_none) continue;
        if (m.size() == 3 && m.has_sequence) {
            const std::string supp_seq = (m[SUPPLEMENTARY].strand == m[SPLIT_READ].strand) ? m.sequence : dna_to_reverse_complement(m.sequence);
            if ((!viral[m[MATE1].contig] && test_mismatch_probability(m[MATE1], m.sequence, assembly, mismatch_probability, genome_size, pvalue_cutoff, m.multimapper && !viral[m[SUPPLEMENTARY].contig])) ||
                (!viral[m[SUPPLEMENTARY].contig] && test_mismatch_probability(m[SUPPLEMENTARY], supp_seq, assembly, mismatch_probability, genome_size, pvalue_cutoff, m.multimapper && !viral[m[MATE1].contig]))) {
                m.filter = FILTER_mismatches;
                continue;
            }
        }
        ++remaining;
    }
    return remaining;
}

// filter_mismappers.cpp: kmer_to_int; filter_low_entropy.cpp: filter_low_entropy
inline uint32_t kmer_to_int(const std::string& kmer, const std::string::size_type position, const char kmer_length) {
    uint32_t result = 0;
    for (char base = 0; base < kmer_length; ++base) {
        result = result << 2;
        switch (kmer.c_str()[position + base]) { case 'T': result += 0; break; case 'G': result += 1; break; case 'C': result += 2; break; default: result += 3; break; }
    }
    return result;
}
inline unsigned int filter_low_entropy(chimeric_alignments_t& ca, const unsigned int kmer_length, const float kmer_content, const unsigned int max_itd_length) {
    for (auto& kv : ca) {
        mates_t& m = kv.second;
        bool is_itd = m.size() == 3 && m[SPLIT_READ].strand == m[SUPPLEMENTARY].strand && m[SPLIT_READ].contig == m[SUPPLEMENTARY].contig &&
                      ((m[SPLIT_READ].strand == FORWARD && m[SPLIT_READ].start < m[SUPPLEMENTARY].end && m[SPLIT_READ].start + (int)max_itd_length >= m[SUPPLEMENTARY].end) ||
                       (m[SPLIT_READ].strand == REVERSE && m[SPLIT_READ].end > m[SUPPLEMENTARY].start && m[SPLIT_READ].end <= m[SUPPLEMENTARY].start + (int)max_itd_length));
        if (!is_itd || m.filter == FILTER_duplicates)
            if (m.filter != FILTER_none) continue;
        if (!m.has_sequence) continue;
        bool hit = false;
        for (unsigned int mate = MATE1; mate <= MATE2 && !hit; ++mate) {
            const std::string& seq = m.sequence;  // MATE1 and SPLIT_READ carry the same sequence
            if (seq.length() < kmer_length) continue;
            unsigned int as1, ae1, as2, ae2;
            as1 = (m[mate].cigar.operation(0) == BAM_CSOFT_CLIP) ? m[mate].cigar.op_length(0) : 0;
            ae1 = seq.length();
            if (m[mate].cigar.operation(m[mate].cigar.size() - 1) == BAM_CSOFT_CLIP) ae1 -= m[mate].cigar.op_length(m[mate].cigar.size() - 1);
            if (m.size() == 3 && mate == SPLIT_READ) {
                as2 = (m[SUPPLEMENTARY].cigar.operation(0) == BAM_CSOFT_CLIP) ? m[SUPPLEMENTARY].cigar.op_length(0) : 0;
                ae2 = seq.length();
                if (m[SUPPLEMENTARY].cigar.operation(m[SUPPLEMENTARY].cigar.size() - 1) == BAM_CSOFT_CLIP) ae2 -= m[SUPPLEMENTARY].cigar.op_length(m[SUPPLEMENTARY].cigar.size() - 1);
                if (m[SUPPLEMENTARY].strand != m[SPLIT_READ].strand) { as2 = seq.length() - as2; ae2 = seq.length() - ae2; std::swap(as2, ae2); }
            } else { as2 = as1; ae2 = ae1; }
            const size_t nk = (size_t)1 << (2 * kmer_length);
            std::vector<unsigned int> kc(nk), kc1(nk), kc2(nk);
            const unsigned int maxk = seq.length() * kmer_content / kmer_length + 0.5;
            const unsigned int maxk1 = (ae1 - as1) * kmer_content / kmer_length + 0.5;
            const unsigned int maxk2 = (ae2 - as2) * kmer_content / kmer_length + 0.5;
            std::vector<std::string::size_type> prev(nk);
            for (std::string::size_type p = 0; p < seq.length() - kmer_length; p++) {
                uint32_t k = kmer_to_int(seq, p, kmer_length);
                if (prev[k] <= p) {
                    prev[k] = p + kmer_length;
                    ++kc[k];
                    if (p + 1 >= as1 && p < ae1) ++kc1[k];
                    if (p + 1 >= as2 && p < ae2) ++kc2[k];
                    if (kc[k] >= maxk || kc1[k] >= maxk1 || kc2[k] >= maxk2) { hit = true; break; }
                }
            }
        }
        if (hit) m.filter = FILTER_low_entropy;
    }
    unsigned int remaining = 0;
    for (auto& kv : ca) if (kv.second.filter == FILTER_none) ++remaining;
    return remaining;
}

// ================================================================= fusions

// fusions.cpp: predict_fusion_strands
inline void predict_fusion_strands(fusion_t& fusion) {
    unsigned int f = 0, r = 0;
    for (auto it : fusion.split_read1_list) if (!it->second[SPLIT_READ].predicted_strand_ambiguous) { if (it->second[SPLIT_READ].predicted_strand == FORWARD) ++f; else ++r; }
    for (auto it : fusion.split_read2_list) if (!it->second[SUPPLEMENTARY].predicted_strand_ambiguous) { if (it->second[SUPPLEMENTARY].predicted_strand == FORWARD) ++f; else ++r; }
    // (discordant mates: none in single-end data)
    if (f == r) { fusion.predicted_strands_ambiguous = true; }
    else {
        fusion.predicted_strands_ambiguous = false;
        fusion.predicted_strand1 = (f > r) ? FORWARD : REVERSE;
        fusion.predicted_strand2 = complement_strand_if(fusion.predicted_strand1, fusion.direction1 == fusion.direction2);
    }
}

// fusions.cpp: predict_transcript_start
inline void predict_transcript_start(fusion_t& fu) {
    fu.transcript_start_ambiguous = false;
    if (fu.spliced1 || (!fu.predicted_strands_ambiguous && !fu.gene1->is_dummy && fu.predicted_strand1 == fu.gene1->strand)) {
        if (fu.gene1->strand == FORWARD && fu.direction1 == DOWNSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if (fu.gene1->strand == FORWARD && fu.direction1 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE2;
        else if (fu.gene1->strand == REVERSE && fu.direction1 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE1;
        else fu.transcript_start = TRANSCRIPT_START_GENE2;
    } else if (fu.spliced2 || (!fu.predicted_strands_ambiguous && !fu.gene2->is_dummy && fu.predicted_strand2 == fu.gene2->strand)) {
        if (fu.gene2->strand == FORWARD && fu.direction2 == DOWNSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE2;
        else if (fu.gene2->strand == FORWARD && fu.direction2 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if (fu.gene2->strand == REVERSE && fu.direction2 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE2;
        else fu.transcript_start = TRANSCRIPT_START_GENE1;
    } else if (!fu.predicted_strands_ambiguous) {
        if (((fu.predicted_strand1 == FORWARD && fu.direction1 == DOWNSTREAM) || (fu.predicted_strand1 == REVERSE && fu.direction1 == UPSTREAM)) &&
            ((fu.predicted_strand2 == REVERSE && fu.direction2 == DOWNSTREAM) || (fu.predicted_strand2 == FORWARD && fu.direction2 == UPSTREAM)))
            fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if (((fu.predicted_strand2 == FORWARD && fu.direction2 == DOWNSTREAM) || (fu.predicted_strand2 == REVERSE && fu.direction2 == UPSTREAM)) &&
                 ((fu.predicted_strand1 == REVERSE && fu.direction1 == DOWNSTREAM) || (fu.predicted_strand1 == FORWARD && fu.direction1 == UPSTREAM)))
            fu.transcript_start = TRANSCRIPT_START_GENE2;
        else fu.transcript_start_ambiguous = true;
    } else if (!fu.exonic1 && !fu.exonic2) {
        fu.transcript_start_ambiguous = true;
    } else if (!fu.exonic1 && fu.exonic2) {
        if (fu.gene2->strand == FORWARD && fu.direction2 == DOWNSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE2;
        else if (fu.gene2->strand == REVERSE && fu.direction2 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE2;
        else if (fu.split_reads1 + fu.split_reads2 == 0 && fu.is_read_through() &&
                 ((fu.gene2->strand == FORWARD && fu.direction2 == UPSTREAM) || (fu.gene2->strand == REVERSE && fu.direction2 == DOWNSTREAM)))
            fu.transcript_start = TRANSCRIPT_START_GENE1;
        else fu.transcript_start_ambiguous = true;
    } else if (!fu.exonic2 && fu.exonic1) {
        if (fu.gene1->strand == FORWARD && fu.direction1 == DOWNSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if (fu.gene1->strand == REVERSE && fu.direction1 == UPSTREAM) fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if (fu.split_reads1 + fu.split_reads2 == 0 && fu.is_read_through() &&
                 ((fu.gene1->strand == FORWARD && fu.direction1 == UPSTREAM) || (fu.gene1->strand == REVERSE && fu.direction1 == DOWNSTREAM)))
            fu.transcript_start = TRANSCRIPT_START_GENE1;
        else fu.transcript_start_ambiguous = true;
    } else {
        if ((!fu.gene1->is_dummy && fu.gene1->strand == FORWARD && fu.direction1 == DOWNSTREAM) || (fu.gene1->strand == REVERSE && fu.direction1 == UPSTREAM))
            fu.transcript_start = TRANSCRIPT_START_GENE1;
        else if ((!fu.gene2->is_dummy && fu.gene2->strand == FORWARD && fu.direction2 == DOWNSTREAM) || (fu.gene2->strand == REVERSE && fu.direction2 == UPSTREAM))
            fu.transcript_start = TRANSCRIPT_START_GENE2;
        else fu.transcript_start_ambiguous = true;
    }
    if (fu.transcript_start_ambiguous) fu.transcript_start = TRANSCRIPT_START_GENE1;
    if (!fu.transcript_start_ambiguous && fu.predicted_strands_ambiguous) {
        fu.predicted_strands_ambiguous = false;
        if (fu.transcript_start == TRANSCRIPT_START_GENE1) {
            fu.predicted_strand1 = fu.gene1->strand;
            fu.predicted_strand2 = complement_strand_if(fu.predicted_strand1, fu.direction1 == fu.direction2);
        } else {
            fu.predicted_strand2 = fu.gene2->strand;
            fu.predicted_strand1 = complement_strand_if(fu.predicted_strand2, fu.direction1 == fu.direction2);
        }
    }
}

// fusions.cpp: find_fusions (split reads; single-end data have no discordant mates)
inline unsigned int find_fusions(chimeric_alignments_t& ca, fusions_t& fusions, const exon_annotation_index_t& exon_index, const unsigned int subsampling_threshold) {
    for (auto it = ca.begin(); it != ca.end(); ++it) {
        mates_t& m = it->second;
        if (m.size() != 3) continue;
        contig_t contig1 = m[SPLIT_READ].contig, contig2 = m[SUPPLEMENTARY].contig;
        position_t bp1 = (m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end;
        position_t bp2 = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start;
        gene_set_t genes1 = m[SPLIT_READ].genes, genes2 = m[SUPPLEMENTARY].genes;
        direction_t d1 = (m[SPLIT_READ].strand == FORWARD) ? UPSTREAM : DOWNSTREAM;
        direction_t d2 = (m[SUPPLEMENTARY].strand == FORWARD) ? DOWNSTREAM : UPSTREAM;
        bool e1 = m[SPLIT_READ].exonic, e2 = m[SUPPLEMENTARY].exonic;
        position_t a1 = (m[MATE1].strand == FORWARD) ? m[MATE1].start : m[MATE1].end;
        position_t a2 = (m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].start : m[SUPPLEMENTARY].end;
        bool swapped = false;
        if (contig1 > contig2 || (contig1 == contig2 && bp1 > bp2)) {
            std::swap(contig1, contig2); std::swap(bp1, bp2); std::swap(genes1, genes2); std::swap(d1, d2); std::swap(e1, e2); std::swap(a1, a2);
            swapped = true;
        }
        for (gene_t g1 : genes1) {
            for (gene_t g2 : genes2) {
                auto ins = fusions.emplace(fusion_key_t(g1->id, g2->id, contig1, contig2, bp1, bp2, d1, d2), fusion_t());
                fusion_t& fu = ins.first->second;
                if (ins.second) {
                    fu.gene1 = g1; fu.gene2 = g2; fu.direction1 = d1; fu.direction2 = d2;
                    fu.contig1 = contig1; fu.contig2 = contig2; fu.breakpoint1 = bp1; fu.breakpoint2 = bp2;
                }
                fu.exonic1 = e1 || fu.exonic1; fu.exonic2 = e2 || fu.exonic2;
                if (ins.second || m.filter == FILTER_none || fu.filter == FILTER_duplicates) fu.filter = m.filter;
                if ((fu.split_reads1 >= subsampling_threshold && !swapped) || (fu.split_reads2 >= subsampling_threshold && swapped) ||
                    (m.filter != FILTER_none && !swapped && fu.split_read1_list.size() >= subsampling_threshold) ||
                    (m.filter != FILTER_none && swapped && fu.split_read2_list.size() >= subsampling_threshold)) {
                    // subsampling
                } else {
                    if (fu.direction1 == DOWNSTREAM && (a1 < fu.anchor_start1 || fu.anchor_start1 == 0)) fu.anchor_start1 = a1;
                    else if (fu.direction1 == UPSTREAM && (a1 > fu.anchor_start1 || fu.anchor_start1 == 0)) fu.anchor_start1 = a1;
                    if (fu.direction2 == DOWNSTREAM && (a2 < fu.anchor_start2 || fu.anchor_start2 == 0)) fu.anchor_start2 = a2;
                    else if (fu.direction2 == UPSTREAM && (a2 > fu.anchor_start2 || fu.anchor_start2 == 0)) fu.anchor_start2 = a2;
                    if (swapped) { fu.split_read2_list.push_back(it); if (m.filter == FILTER_none) fu.split_reads2++; }
                    else { fu.split_read1_list.push_back(it); if (m.filter == FILTER_none) fu.split_reads1++; }
                }
            }
        }
    }
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& fu = kv.second;
        predict_fusion_strands(fu);
        if (fu.split_read1_list.size() + fu.split_read2_list.size() == 0 || fu.predicted_strands_ambiguous) { fu.spliced1 = false; fu.spliced2 = false; }
        else {
            fu.spliced1 = fu.exonic1 && fu.gene1->strand == fu.predicted_strand1 && is_breakpoint_spliced(fu.gene1, fu.direction1, fu.breakpoint1, exon_index);
            fu.spliced2 = fu.exonic2 && fu.gene2->strand == fu.predicted_strand2 && is_breakpoint_spliced(fu.gene2, fu.direction2, fu.breakpoint2, exon_index);
        }
        predict_transcript_start(fu);
        if (fu.filter == FILTER_none) remaining++;
    }
    return remaining;
}

inline unsigned int count_remaining(const fusions_t& fusions) { unsigned int n = 0; for (auto& kv : fusions) if (kv.second.filter == FILTER_none) n++; return n; }

// merge_adjacent_fusions.cpp: merge_adjacent_fusions
inline unsigned int merge_adjacent_fusions(fusions_t& fusions, const int max_distance, const unsigned int max_itd_length) {
    std::vector<fusion_t*> sorted;
    for (auto& kv : fusions) if (kv.second.filter == FILTER_none || kv.second.is_internal_tandem_duplication(max_itd_length)) sorted.push_back(&kv.second);
    std::stable_sort(sorted.begin(), sorted.end(), [](const fusion_t* x, const fusion_t* y) {
        if (x->contig1 != y->contig1) return x->contig1 < y->contig1;
        if (x->breakpoint1 != y->breakpoint1) return x->breakpoint1 < y->breakpoint1;
        if (x->contig2 != y->contig2) return x->contig2 < y->contig2;
        return x->breakpoint2 < y->breakpoint2;
    });
    for (size_t i = 0; i < sorted.size(); ++i) {
        fusion_t& f = *sorted[i];
        const bool itd = f.is_internal_tandem_duplication(max_itd_length);
        if ((!itd && f.split_reads1 + f.split_reads2 == 0) || (itd && f.split_read1_list.size() + f.split_read2_list.size() == 0)) continue;
        std::vector<fusion_t*> adjacent;
        auto same = [&](const fusion_t& o) {
            return o.gene1 == f.gene1 && o.gene2 == f.gene2 && o.direction1 == f.direction1 && o.direction2 == f.direction2 && o.contig2 == f.contig2;
        };
        for (size_t j = i; j-- > 0;) {  // look upstream
            const fusion_t& o = *sorted[j];
            if (o.contig1 != f.contig1 || o.breakpoint1 < f.breakpoint1 - max_distance) break;
            if (same(o) && (o.breakpoint2 == f.breakpoint2 + (f.breakpoint1 - o.breakpoint1) * ((f.direction1 == f.direction2) ? +1 : -1) ||
                            (itd && std::abs(f.breakpoint2 - o.breakpoint2) <= max_distance)))
                if (o.split_reads1 + o.split_reads2 > 0 || (itd && o.split_read1_list.size() + o.split_read2_list.size() > 0)) adjacent.push_back(sorted[j]);
        }
        for (size_t j = i; j < sorted.size(); ++j) {  // look downstream
            const fusion_t& o = *sorted[j];
            if (o.contig1 != f.contig1 || o.breakpoint1 > f.breakpoint1 + max_distance) break;
            if (j != i && same(o) && (o.breakpoint2 == f.breakpoint2 + (o.breakpoint1 - f.breakpoint1) * ((f.direction1 == f.direction2) ? -1 : +1) ||
                                      (itd && std::abs(f.breakpoint2 - o.breakpoint2) <= max_distance)))
                if (o.split_reads1 + o.split_reads2 > 0 || (itd && o.split_read1_list.size() + o.split_read2_list.size() > 0)) adjacent.push_back(sorted[j]);
        }
        unsigned int s1 = 0, s2 = 0;
        bool most = true;
        for (fusion_t* o : adjacent) {
            if (f.supporting_reads() < o->supporting_reads() ||
                (f.supporting_reads() == o->supporting_reads() && f.split_read1_list.size() + f.split_read2_list.size() < o->split_read1_list.size() + o->split_read2_list.size())) {
                most = false; break;
            }
            s1 += o->split_reads1; s2 += o->split_reads2;
        }
        if (most) {
            f.split_reads1 += s1; f.split_reads2 += s2;
            for (fusion_t* o : adjacent) {
                o->filter = FILTER_merge_adjacent;
                if (itd) {
                    f.split_read1_list.insert(f.split_read1_list.end(), o->split_read1_list.begin(), o->split_read1_list.end());
                    f.split_read2_list.insert(f.split_read2_list.end(), o->split_read2_list.begin(), o->split_read2_list.end());
                }
            }
        }
    }
    return count_remaining(fusions);
}

// filter_multimappers.cpp: is_gap_at_splice_site, calculate_segment_score, calculate_alignment_score, fusion_has_more_support,
// filter_multimappers
inline bool is_gap_at_splice_site(const position_t position, const direction_t direction, const gene_set_t& genes, const exon_annotation_index_t& exon_index) {
    for (gene_t g : genes) if (is_breakpoint_spliced(g, direction, position, exon_index)) return true;
    return false;
}
inline int calculate_segment_score(const alignment_t& alignment, const std::string& sequence, const exon_annotation_index_t& exon_index, const assembly_t& assembly) {
    if (!assembly.has(alignment.contig)) return 0;
    int score = 0;
    position_t ref = alignment.start, rp = 0;
    for (unsigned int i = 0; i < alignment.cigar.size(); ++i) {
        const uint32_t n = alignment.cigar.op_length(i);
        switch (alignment.cigar.operation(i)) {
            case BAM_CSOFT_CLIP: case BAM_CHARD_CLIP: rp += n; break;
            case BAM_CDEL: score--; ref += n; break;
            case BAM_CREF_SKIP:
                if (!is_gap_at_splice_site(ref, DOWNSTREAM, alignment.genes, exon_index) || !is_gap_at_splice_site(ref + n, UPSTREAM, alignment.genes, exon_index)) score--;
                ref += n;
                break;
            case BAM_CINS: score--; rp += n; break;
            case BAM_CEQUAL: score += n; ref += n; rp += n; break;
            case BAM_CDIFF: ref += n; rp += n; break;
            case BAM_CMATCH:
                for (unsigned int k = 1; k <= n; ++k) {
                    if (rp < (position_t)sequence.size() && sequence[rp] == assembly.at(alignment.contig, ref)) score++;
                    ref++; rp++;
                }
                break;
        }
    }
    return score;
}
inline int calculate_alignment_score(const mates_t& m, const exon_annotation_index_t& exon_index, const assembly_t& assembly) {
    if (!m.has_sequence || !assembly.data) return m.star_score;  // singlet: STAR's chimeric score when the read or genome is missing
    const std::string& s = m.sequence;
    int score = calculate_segment_score(m[MATE1], s, exon_index, assembly) + calculate_segment_score(m[MATE2], s, exon_index, assembly);
    if (m.size() == 3) {
        score += calculate_segment_score(m[SUPPLEMENTARY], (m[SUPPLEMENTARY].strand == m[SPLIT_READ].strand) ? s : dna_to_reverse_complement(s), exon_index, assembly);
        if (!is_gap_at_splice_site((m[SUPPLEMENTARY].strand == FORWARD) ? m[SUPPLEMENTARY].end : m[SUPPLEMENTARY].start, (m[SUPPLEMENTARY].strand == FORWARD) ? DOWNSTREAM : UPSTREAM, m[SUPPLEMENTARY].genes, exon_index) ||
            !is_gap_at_splice_site((m[SPLIT_READ].strand == FORWARD) ? m[SPLIT_READ].start : m[SPLIT_READ].end, (m[SPLIT_READ].strand == FORWARD) ? UPSTREAM : DOWNSTREAM, m[SPLIT_READ].genes, exon_index))
            score--;
    }
    return score;
}
inline bool fusion_has_more_support(const fusion_t* f, const fusion_t* b) {
    if (f == nullptr) return false;
    if (b == nullptr) return true;
    if (b->supporting_reads() != f->supporting_reads()) return b->supporting_reads() < f->supporting_reads();
    if (f->gene1->is_protein_coding != b->gene1->is_protein_coding) return f->gene1->is_protein_coding;
    if (f->gene2->is_protein_coding != b->gene2->is_protein_coding) return f->gene2->is_protein_coding;
    if (f->contig1 != b->contig1) return f->contig1 < b->contig1;
    if (f->contig2 != b->contig2) return f->contig2 < b->contig2;
    if (f->breakpoint1 != b->breakpoint1) return f->breakpoint1 < b->breakpoint1;
    if (f->breakpoint2 != b->breakpoint2) return f->breakpoint2 < b->breakpoint2;
    if (f->direction1 != b->direction1) return f->direction1 < b->direction1;
    if (f->direction2 != b->direction2) return f->direction2 < b->direction2;
    if (f->gene1->id != b->gene1->id) return f->gene1->id < b->gene1->id;
    return f->gene2->id < b->gene2->id;
}
inline unsigned int filter_multimappers(chimeric_alignments_t& ca, fusions_t& fusions, const exon_annotation_index_t& exon_index, const assembly_t& assembly) {
    std::unordered_map<const mates_t*, const fusion_t*> best_fusion;
    for (auto& kv : fusions) {
        for (auto* lst : {&kv.second.split_read1_list, &kv.second.split_read2_list, &kv.second.discordant_mate_list})
            for (auto it : *lst) {
                const fusion_t*& cur = best_fusion[&it->second];
                if (fusion_has_more_support(&kv.second, cur)) cur = &kv.second;
            }
    }
    auto get_best = [&](const mates_t* m) -> const fusion_t* { auto f = best_fusion.find(m); return f == best_fusion.end() ? nullptr : f->second; };
    auto start_of_cluster = ca.begin();
    std::string read_name = ca.empty() ? "" : strip_hi_tag_from_read_name(start_of_cluster->first);
    std::string next_read_name = read_name, cluster_name = read_name;
    mates_t* best_alignment = nullptr;
    int best_score = INT_MIN;
    for (auto it = ca.begin();; ++it) {
        read_name = next_read_name;
        if (cluster_name != read_name) {
            if (best_alignment != nullptr)
                for (auto it2 = start_of_cluster; it2 != it; ++it2)
                    if (&it2->second != best_alignment && it2->second.filter == FILTER_none) it2->second.filter = FILTER_multimappers;
            cluster_name = read_name;
            start_of_cluster = it;
            best_alignment = nullptr;
            best_score = INT_MIN;
        }
        if (it == ca.end()) break;
        next_read_name = (std::next(it) != ca.end()) ? strip_hi_tag_from_read_name(std::next(it)->first) : "";
        if (start_of_cluster == it && next_read_name != read_name) continue;
        int score = calculate_alignment_score(it->second, exon_index, assembly);
        if (best_score < score) { best_alignment = &it->second; best_score = score; }
        else if (best_score == score && fusion_has_more_support(get_best(&it->second), get_best(best_alignment))) best_alignment = &it->second;
    }
    for (auto& kv : fusions) {
        fusion_t& fu = kv.second;
        if (fu.filter != FILTER_none || fu.supporting_reads() == 0) continue;
        for (auto it : fu.split_read1_list) if (it->second.filter == FILTER_multimappers && fu.split_reads1 > 0) fu.split_reads1--;
        for (auto it : fu.split_read2_list) if (it->second.filter == FILTER_multimappers && fu.split_reads2 > 0) fu.split_reads2--;
        for (auto it : fu.discordant_mate_list) if (it->second.filter == FILTER_multimappers && fu.discordant_mates > 0) fu.discordant_mates--;
        if (fu.supporting_reads() == 0) fu.filter = FILTER_multimappers;
    }
    return count_remaining(fusions);
}

// filter_relative_support.cpp: estimate_expected_fusions, filter_relative_support
inline void estimate_expected_fusions(fusions_t& fusions, const unsigned long mapped_reads, const exon_annotation_index_t& exon_index) {
    std::map<gene_t, std::set<gene_t>> fusion_partners;
    {
        std::set<std::tuple<gene_t, position_t, position_t>> seen;
        for (auto& kv : fusions) {
            fusion_t& f = kv.second;
            if (f.filter == FILTER_none && f.gene1 != f.gene2) {
                if (seen.emplace(f.gene2, f.breakpoint1, f.breakpoint2).second) fusion_partners[f.gene2].insert(f.gene1);
                if (seen.emplace(f.gene1, f.breakpoint1, f.breakpoint2).second) fusion_partners[f.gene1].insert(f.gene2);
            }
        }
    }
    std::unordered_map<gene_t, int> partner_count;
    for (auto& p1 : fusion_partners)
        for (gene_t p2 : p1.second)
            if (p1.second.size() >= fusion_partners[p2].size()) partner_count[p1.first]++;
    unsigned int spliced = 0, exonic = 0, intronic = 0, exonic_intronic = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter == FILTER_none && (f.contig1 != f.contig2 || f.breakpoint2 - f.breakpoint1 > 500000) && f.supporting_reads() >= 2 &&
            f.split_reads1 + f.split_reads2 > 0 && !f.gene1->is_dummy && !f.gene2->is_dummy) {
            if (f.spliced1 || f.spliced2) spliced++;
            else if (f.exonic1 && f.exonic2) exonic++;
            else if (!f.exonic1 && !f.exonic2) intronic++;
            else exonic_intronic++;
        }
    }
    if (spliced + exonic + intronic + exonic_intronic < 100 || spliced == 0 || exonic == 0 || intronic == 0 || exonic_intronic == 0) {
        spliced = 10; exonic = 65; intronic = 10; exonic_intronic = 15;
    }
    unsigned int dups = 0, invs = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter == FILTER_none && f.gene1 == f.gene2 && f.split_reads1 + f.split_reads2 >= 2) {
            if (f.direction1 == UPSTREAM && f.direction2 == DOWNSTREAM) dups++;
            else if (f.direction1 == f.direction2) invs++;
        }
    }
    if (invs + dups < 100) { invs = 1; dups = 1; }
    unsigned int same_gene = 0, diff_genes = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.spliced1 && f.spliced2) { if (f.gene1 == f.gene2) same_gene++; else diff_genes++; }
    }
    if (same_gene + diff_genes < 100) { same_gene = 0; diff_genes = 100; }
    std::set<gene_t> with_fusions, with_rt;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (!f.gene1->is_dummy && !f.gene2->is_dummy && f.split_reads1 + f.split_reads2 > 0) {
            with_fusions.insert(f.gene1); with_fusions.insert(f.gene2);
            if (f.is_read_through()) { with_rt.insert(f.gene1); with_rt.insert(f.gene2); }
        }
    }
    float frac_rt = with_fusions.empty() ? 0 : 1.0 * with_rt.size() / with_fusions.size();
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        float maxp = std::max(10000.0 / f.gene1->exonic_length * std::max(partner_count[f.gene1] - 1, 1),
                              10000.0 / f.gene2->exonic_length * std::max(partner_count[f.gene2] - 1, 1));
        f.evalue = maxp * std::max(1.0, mapped_reads / 20000000.0 * std::pow(0.02, (double)f.supporting_reads() - 2));
        if (f.is_intragenic()) {
            f.evalue *= 2.0 / (dups + invs);
            if (f.direction1 == UPSTREAM && f.direction2 == DOWNSTREAM) f.evalue *= dups;
            else if (f.direction1 == f.direction2) f.evalue *= invs;
            if (f.supporting_reads() >= 1) {
                f.evalue *= std::pow(f.supporting_reads() - 0.42, -2.11) * std::pow(10, -1.11);
                int sd = get_spliced_distance(f.contig1, f.breakpoint1, f.breakpoint2, f.gene1, exon_index);
                if (sd < 1000) {
                    f.evalue *= std::pow(std::max(400, sd) / 1000.0, -2);
                    if (sd < 400) f.evalue *= std::pow(std::max(1, sd) / 400.0, -4.58);
                }
            }
            f.evalue *= std::max(1.0, same_gene / 0.25 / diff_genes);
        } else {
            if (f.supporting_reads() >= 1) {
                f.evalue *= std::pow(f.supporting_reads() - 0.73, -2.28) * std::pow(10, -1.75);
                if (f.is_read_through()) f.evalue *= std::pow(std::max(1, f.breakpoint2 - f.breakpoint1) / 400000.0, -0.63);
                else if (f.contig1 == f.contig2 && f.breakpoint2 - f.breakpoint1 < 400000) f.evalue *= std::pow(std::max(1, f.breakpoint2 - f.breakpoint1) / 400000.0, -1.53);
            }
        }
        f.evalue *= 4.0 / (spliced + exonic + intronic + exonic_intronic);
        if (f.spliced1 || f.spliced2) f.evalue *= spliced;
        else if (f.exonic1 && f.exonic2) f.evalue *= std::max(spliced, exonic);
        else if (!f.exonic1 && !f.exonic2) f.evalue *= std::max(spliced, intronic);
        else f.evalue *= std::max(spliced, exonic_intronic);
        if (frac_rt > 0.25 && f.is_read_through()) f.evalue *= 1 + std::pow((frac_rt - 0.25) * 20, 2);
    }
}
inline unsigned int filter_relative_support(fusions_t& fusions, const float evalue_cutoff) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (f.evalue < evalue_cutoff && !(f.is_intragenic() && f.split_reads1 + f.split_reads2 == 0)) remaining++;
        else f.filter = FILTER_relative_support;
    }
    return remaining;
}

// filter_non_coding_neighbors.cpp
inline unsigned int filter_non_coding_neighbors(fusions_t& fusions) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (!f.gene1->is_protein_coding && !f.gene2->is_protein_coding && f.is_read_through()) f.filter = FILTER_non_coding_neighbors; else ++remaining;
    }
    return remaining;
}

// filter_intragenic_both_exonic.cpp
inline unsigned int filter_intragenic_both_exonic(fusions_t& fusions, const exon_annotation_index_t& exon_index, const float exonic_fraction) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if ((f.breakpoint_overlaps_both_genes() || f.gene1 == f.gene2) && f.exonic1 && f.exonic2 && !(f.spliced1 && f.spliced2)) {
            int sd = get_spliced_distance(f.contig1, f.breakpoint1, f.breakpoint2, f.gene1, exon_index);
            int d = f.breakpoint2 - f.breakpoint1;
            if (sd == d || 1.0 * sd / d < exonic_fraction) { f.filter = FILTER_intragenic_exonic; continue; }
        }
        ++remaining;
    }
    return remaining;
}

// filter_min_support.cpp
inline unsigned int filter_min_support(fusions_t& fusions, const unsigned int min_support) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (f.split_reads1 + f.split_reads2 + f.discordant_mates < min_support || (f.breakpoint_overlaps_both_genes() && f.split_reads1 + f.split_reads2 < min_support))
            f.filter = FILTER_min_support;
        else remaining++;
    }
    return remaining;
}

// filter_both_intronic.cpp: list_contains_exonic_reads, filter_both_intronic
inline bool list_contains_exonic_reads(const std::vector<chimeric_alignments_t::iterator>& lst) {
    for (auto it : lst) if (it->second.filter == FILTER_none) for (auto& mate : it->second) if (mate.exonic) return true;
    return false;
}
inline unsigned int filter_both_intronic(fusions_t& fusions, const std::vector<bool>& viral) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (viral[f.contig1] || viral[f.contig2]) continue;
        if (!list_contains_exonic_reads(f.split_read1_list) && !list_contains_exonic_reads(f.split_read2_list) && !list_contains_exonic_reads(f.discordant_mate_list))
            f.filter = FILTER_intronic;
        else ++remaining;
    }
    return remaining;
}

// filter_in_vitro.cpp: find_higher_expressed_gene, find_top_expressed_genes, filter_in_vitro
inline gene_t find_higher_expressed_gene(contig_t contig, position_t bp, const gene_annotation_index_t& gene_index, const std::unordered_map<gene_t, unsigned int>& expr, gene_t best) {
    auto get = [&](gene_t g) { auto it = expr.find(g); return it == expr.end() ? 0u : it->second; };
    unsigned int hi = get(best);
    gene_set_t genes;
    get_annotation_by_coordinate(contig, bp, bp, genes, gene_index);
    for (gene_t g : genes) { unsigned int e = get(g); if (e > hi) { hi = e; best = g; } }
    return best;
}
inline void find_top_expressed_genes(const chimeric_alignments_t& ca, const float quantile_f, std::unordered_map<gene_t, unsigned int>& read_count, unsigned int& threshold) {
    for (auto& kv : ca) {
        for (gene_t g : kv.second[MATE1].genes) read_count[g]++;
        unsigned int mate2 = (kv.second.size() == 2) ? MATE2 : SUPPLEMENTARY;
        for (gene_t g : kv.second[mate2].genes) read_count[g]++;
    }
    threshold = 0;
    if (!read_count.empty()) {
        std::vector<gene_t> v;
        for (auto& kv : read_count) v.push_back(kv.first);
        unsigned int q = (unsigned int)std::floor(quantile_f * v.size());
        if (q >= v.size()) q = v.size() - 1;
        std::nth_element(v.begin(), v.begin() + q, v.end(), [&](gene_t x, gene_t y) {
            unsigned int rx = read_count.at(x), ry = read_count.at(y);
            return rx != ry ? rx < ry : x->id < y->id;
        });
        threshold = read_count[v[q]];
    }
}
inline unsigned int filter_in_vitro(fusions_t& fusions, const chimeric_alignments_t& ca, const float high_expression_quantile, const gene_annotation_index_t& gene_index, const coverage_t& coverage) {
    const unsigned int min_clipped_length = 3;
    const unsigned int max_exonic_breakpoints_by_gene_pair = 8;
    std::map<std::pair<gene_t, gene_t>, unsigned int> exonic_bps;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.gene1 != f.gene2 && !f.spliced1 && !f.spliced2 && f.exonic1 && f.exonic2 && f.split_read1_list.size() + f.split_read2_list.size() > 0 &&
            f.filter != FILTER_merge_adjacent && f.filter != FILTER_uninteresting_contigs) {
            exonic_bps[{f.gene1, f.gene2}]++;
            exonic_bps[{f.gene2, f.gene1}]++;
        }
    }
    std::unordered_map<gene_t, unsigned int> read_count;
    unsigned int thr;
    find_top_expressed_genes(ca, high_expression_quantile, read_count, thr);
    auto rc = [&](gene_t g) { auto it = read_count.find(g); return it == read_count.end() ? 0u : it->second; };
    auto eb = [&](gene_t a, gene_t b) { auto it = exonic_bps.find({a, b}); return it == exonic_bps.end() ? 0u : it->second; };
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none &&
            !((f.spliced1 || f.spliced2) && (f.filter == FILTER_relative_support || f.filter == FILTER_min_support || f.filter == FILTER_homopolymer)))
            continue;
        float rt = 0;
        if (!f.exonic1) rt += 0.5; else if (!f.spliced1) rt += 1;
        if (!f.exonic2) rt += 0.5; else if (!f.spliced2) rt += 1;
        unsigned int cdm1 = 0, cdm2 = 0;
        for (auto it : f.discordant_mate_list) {
            if (it->second.filter != FILTER_none) continue;
            for (auto& mate : it->second) {
                if (mate.strand == FORWARD && mate.postclipping() >= min_clipped_length) {
                    if (mate.contig == f.contig1 && mate.end == f.breakpoint1) cdm1++; else if (mate.contig == f.contig2 && mate.end == f.breakpoint2) cdm2++;
                } else if (mate.strand == REVERSE && mate.preclipping() >= min_clipped_length) {
                    if (mate.contig == f.contig1 && mate.start == f.breakpoint1) cdm1++; else if (mate.contig == f.contig2 && mate.start == f.breakpoint2) cdm2++;
                }
            }
        }
        unsigned int tsr = std::min(cdm1, cdm2) + f.split_reads1 + f.split_reads2;
        gene_t g1 = find_higher_expressed_gene(f.contig1, f.breakpoint1, gene_index, read_count, f.gene1);
        gene_t g2 = find_higher_expressed_gene(f.contig2, f.breakpoint2, gene_index, read_count, f.gene2);
        unsigned int e1 = rc(g1), e2 = rc(g2);
        unsigned int exb = std::max(eb(g1, g2), eb(f.gene1, f.gene2));
        int cov1 = coverage.get_coverage(f.contig1, f.breakpoint1, (f.direction1 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        int cov2 = coverage.get_coverage(f.contig2, f.breakpoint2, (f.direction2 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        if (tsr <= 2 + 0.0001 * (e1 + e2) && (tsr * 2 <= f.discordant_mates || tsr <= 2) && e1 + e2 > thr &&
            !(f.supporting_reads() >= 10 && ((int)f.supporting_reads() * 4) >= std::max(cov1, cov2) && cov1 > (int)f.supporting_reads() && cov2 > (int)f.supporting_reads() &&
              (f.spliced1 || f.spliced2) && ((f.spliced1 || !f.exonic1) && (f.spliced2 || !f.exonic2))) &&
            (rt > 1 || (rt > 0 && (e1 > thr || e2 > thr)) || e1 > 2 * thr || e2 > 2 * thr || (e1 > thr && e2 > thr) || exb > max_exonic_breakpoints_by_gene_pair ||
             f.supporting_reads() <= 1))
            f.filter = FILTER_in_vitro;
    }
    return count_remaining(fusions);
}

// recover_both_spliced.cpp: opposite_direction, count_supporting_reads, recover_both_spliced
inline direction_t opposite_direction(const direction_t d) { return d == DOWNSTREAM ? UPSTREAM : DOWNSTREAM; }
inline unsigned int count_supporting_reads(const fusion_t& f, std::unordered_map<gene_t, unsigned int>& read_count, const exon_annotation_index_t& exon_index, const coverage_t& coverage,
                                           const unsigned int thr, const int max_exon_size, const unsigned int max_coverage) {
    if (read_count[f.gene1] > thr || read_count[f.gene2] > thr) {
        if (f.both_breakpoints_spliced() && f.discordant_mates <= f.split_reads1 + f.split_reads2) return 1;
        return 0;
    }
    if (!f.both_breakpoints_spliced()) {
        unsigned int c1 = coverage.get_coverage(f.contig1, f.breakpoint1, (f.direction1 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        unsigned int c2 = coverage.get_coverage(f.contig2, f.breakpoint2, (f.direction2 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        if (c1 + c2 > f.supporting_reads() * max_coverage) return 0;
        exon_set_t exons;
        get_annotation_by_coordinate(f.contig1, f.breakpoint1, f.breakpoint1, exons, exon_index);
        for (exon_t e : exons) if (e->end + 1 - e->start > max_exon_size) return 0;
        get_annotation_by_coordinate(f.contig2, f.breakpoint2, f.breakpoint2, exons, exon_index);
        for (exon_t e : exons) if (e->end + 1 - e->start > max_exon_size) return 0;
    }
    unsigned int multi = 0, uniq = 0;
    for (auto* lst : {&f.split_read1_list, &f.split_read2_list, &f.discordant_mate_list})
        for (auto it : *lst) { if (it->second.multimapper) multi++; else if (it->second.filter == FILTER_none) uniq++; }
    if (multi >= 0.5 * (f.split_read1_list.size() + f.split_read2_list.size() + f.discordant_mate_list.size())) return 0;
    if (uniq == 0) return 1;
    return uniq;
}
inline unsigned int recover_both_spliced(fusions_t& fusions, const chimeric_alignments_t& ca, const exon_annotation_index_t& exon_index, const coverage_t& coverage,
                                         const unsigned int max_fusions_to_recover, const float high_expression_quantile, const int max_exon_size, const unsigned int max_coverage) {
    std::unordered_map<gene_t, unsigned int> read_count;
    unsigned int thr;
    find_top_expressed_genes(ca, high_expression_quantile, read_count, thr);
    typedef std::tuple<unsigned int, unsigned int, direction_t, direction_t> gp_t;
    std::map<gp_t, std::vector<fusion_t*>> by_pair;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_merge_adjacent)
            if (f.filter == FILTER_none || f.filter == FILTER_in_vitro || f.filter == FILTER_intronic || f.filter == FILTER_relative_support || f.filter == FILTER_min_support ||
                (f.filter == FILTER_inconsistently_clipped && f.both_breakpoints_spliced()))
                if (count_supporting_reads(f, read_count, exon_index, coverage, thr, max_exon_size, max_coverage) > 0)
                    by_pair[gp_t(f.gene1->id, f.gene2->id, f.direction1, f.direction2)].push_back(&f);
    }
    unsigned int remaining = 0;
    std::map<unsigned int, unsigned int> recovered_by_reads;
    unsigned int min_reads = 1;
    for (int mode = 0; mode <= 1; ++mode) {  // 0 = counting, 1 = recover
        for (auto& kv : fusions) {
            fusion_t& f = kv.second;
            if (f.filter == FILTER_none) { if (mode == 1) remaining++; continue; }
            if (!f.both_breakpoints_spliced()) continue;
            if (f.gene1 == f.gene2 || f.breakpoint_overlaps_both_genes()) continue;
            if (f.is_read_through()) continue;
            if (f.filter != FILTER_relative_support && f.filter != FILTER_min_support && f.filter != FILTER_in_vitro) continue;
            unsigned int sum = 0;
            auto same = by_pair.find(gp_t(f.gene1->id, f.gene2->id, f.direction1, f.direction2));
            if (same != by_pair.end()) for (fusion_t* o : same->second) sum += count_supporting_reads(*o, read_count, exon_index, coverage, thr, max_exon_size, max_coverage);
            auto recip = by_pair.find(gp_t(f.gene1->id, f.gene2->id, opposite_direction(f.direction1), opposite_direction(f.direction2)));
            if (recip != by_pair.end())
                for (fusion_t* o : recip->second)
                    if (!o->is_read_through())
                        if (o->both_breakpoints_spliced() || (((f.direction1 == DOWNSTREAM) != (f.breakpoint1 > o->breakpoint1)) && ((f.direction2 == DOWNSTREAM) != (f.breakpoint2 > o->breakpoint2))))
                            sum += count_supporting_reads(*o, read_count, exon_index, coverage, thr, max_exon_size, max_coverage);
            if (sum >= 2) {
                if (mode == 1) {
                    unsigned int add = (f.contig1 == f.contig2 && std::abs(f.breakpoint1 - f.breakpoint2) < 1000000) ? 1 : 0;
                    if (f.supporting_reads() >= min_reads + add) { f.filter = FILTER_none; remaining++; }
                } else {
                    recovered_by_reads[f.supporting_reads()]++;
                }
            }
        }
        if (mode == 0) {
            unsigned int would = 0;
            for (auto r = recovered_by_reads.rbegin(); r != recovered_by_reads.rend(); ++r) {
                would += r->second;
                if (would >= max_fusions_to_recover) { min_reads = r->first + 1; break; }
            }
        }
    }
    return remaining;
}

// select_best.cpp: rank_fusion, select_most_supported_breakpoints
inline unsigned int rank_fusion(const fusion_t& f) {
    if (f.split_reads1 != 0 && f.split_reads2 != 0) return 3;
    if ((f.split_reads1 != 0 || f.split_reads2 != 0) && f.discordant_mates != 0) return 2;
    if (f.split_reads1 != 0 || f.split_reads2 != 0) return 1;
    return 0;
}
inline unsigned int select_most_supported_breakpoints(fusions_t& fusions) {
    typedef std::tuple<gene_t, gene_t, direction_t, direction_t> gp_t;
    std::map<gp_t, fusion_t*> best;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        gp_t gp(f.gene1, f.gene2, f.direction1, f.direction2);
        auto it = best.find(gp);
        if (it == best.end()) { best[gp] = &f; continue; }
        fusion_t& c = *it->second;
        if (rank_fusion(f) > rank_fusion(c)) it->second = &f;
        else if (rank_fusion(f) == rank_fusion(c)) {
            if (f.supporting_reads() > c.supporting_reads()) it->second = &f;
            else if (f.supporting_reads() == c.supporting_reads()) {
                if ((f.exonic1 && !c.exonic1) || (f.exonic2 && !c.exonic2)) it->second = &f;
                else if ((!c.exonic1 || f.exonic1 == c.exonic1) && (!c.exonic2 || f.exonic2 == c.exonic2)) {
                    if ((f.direction1 == DOWNSTREAM && f.breakpoint1 > c.breakpoint1) || (f.direction1 == UPSTREAM && f.breakpoint1 < c.breakpoint1)) it->second = &f;
                    else if (f.breakpoint1 == c.breakpoint1) {
                        if ((f.direction2 == DOWNSTREAM && f.breakpoint2 > c.breakpoint2) || (f.direction2 == UPSTREAM && f.breakpoint2 < c.breakpoint2)) it->second = &f;
                    }
                }
            }
        }
    }
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (&f == best[gp_t(f.gene1, f.gene2, f.direction1, f.direction2)]) remaining++; else f.filter = FILTER_select_best;
    }
    return remaining;
}

// filter_marginal_read_through.cpp
inline unsigned int filter_marginal_read_through(fusions_t& fusions, const coverage_t& coverage) {
    const float margin = 0.01, min_vaf = 0.07;
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter == FILTER_none && f.is_read_through()) {
            double pd = 1, pa = 1;
            if (!f.gene1->is_dummy && f.gene1->strand == FORWARD && f.direction1 == DOWNSTREAM) pd = 1.0 * (f.breakpoint1 - f.gene1->start) / (f.gene1->end - f.gene1->start);
            else if (!f.gene2->is_dummy && f.gene2->strand == REVERSE && f.direction2 == UPSTREAM) pd = 1.0 * (f.gene2->end - f.breakpoint2) / (f.gene2->end - f.gene2->start);
            else if (!f.gene1->is_dummy && f.gene1->strand == REVERSE && f.direction1 == DOWNSTREAM) pa = 1.0 * (f.breakpoint1 - f.gene1->start) / (f.gene1->end - f.gene1->start);
            else if (!f.gene2->is_dummy && f.gene2->strand == FORWARD && f.direction2 == UPSTREAM) pa = 1.0 * (f.gene2->end - f.breakpoint2) / (f.gene2->end - f.gene2->start);
            else continue;
            int c1 = coverage.get_coverage(f.contig1, f.breakpoint1, (f.direction1 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
            int c2 = coverage.get_coverage(f.contig2, f.breakpoint2, (f.direction2 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
            if (pd > 1 - margin && pa > 1 - margin && f.supporting_reads() < min_vaf * std::max(c1, c2)) f.filter = FILTER_marginal_read_through;
        }
        if (f.filter == FILTER_none) ++remaining;
    }
    return remaining;
}

// recover_many_spliced.cpp
inline unsigned int recover_many_spliced(fusions_t& fusions, const unsigned int min_spliced_events) {
    std::map<std::pair<gene_t, gene_t>, std::set<std::pair<position_t, position_t>>> spliced;
    auto eligible = [](filter_t x) { return x == FILTER_none || x == FILTER_inconsistently_clipped || x == FILTER_relative_support || x == FILTER_min_support || x == FILTER_select_best; };
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (!f.is_read_through() && (f.spliced1 || f.spliced2) && f.gene1 != f.gene2 && !f.breakpoint_overlaps_both_genes() && eligible(f.filter))
            spliced[{f.gene1, f.gene2}].insert({f.breakpoint1 / 10, f.breakpoint2 / 10});
    }
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter == FILTER_none) { remaining++; continue; }
        if (f.is_read_through() || f.gene1 == f.gene2 || f.breakpoint_overlaps_both_genes()) continue;
        if (f.filter == FILTER_inconsistently_clipped || f.filter == FILTER_relative_support || f.filter == FILTER_min_support || f.filter == FILTER_select_best)
            if ((f.spliced1 || f.spliced2) && spliced[{f.gene1, f.gene2}].size() >= min_spliced_events) { f.filter = FILTER_none; remaining++; }
    }
    return remaining;
}

// filter_blacklisted_ranges.cpp: parse_range, parse_blacklist_item, overlapping_fraction, matches_blacklist_item,
// get_genome_bins_from_range, filter_blacklisted_ranges
enum blacklist_type_t { BLACKLIST_RANGE, BLACKLIST_POSITION, BLACKLIST_GENE, BLACKLIST_ANY, BLACKLIST_SPLIT_READ_DONOR, BLACKLIST_SPLIT_READ_ACCEPTOR,
                        BLACKLIST_SPLIT_READ_ANY, BLACKLIST_DISCORDANT_MATES, BLACKLIST_READ_THROUGH, BLACKLIST_LOW_SUPPORT, BLACKLIST_FILTER_SPLICED,
                        BLACKLIST_NOT_BOTH_SPLICED };
struct blacklist_item_t {
    blacklist_type_t type = BLACKLIST_ANY;
    contig_t contig = -1;
    position_t start = 0, end = 0;
    bool strand_defined = false;
    strand_t strand = FORWARD;
    gene_t gene = nullptr;
};
inline bool parse_range(std::string range, const contigs_t& contigs, blacklist_item_t& item) {
    size_t sep = range.find_last_of(':');
    if (sep >= range.size()) return false;
    std::string contig_name = range.substr(0, sep), se = range.substr(sep + 1);
    if (contig_name.empty() || se.empty()) return false;
    if (contig_name[0] == '+') { item.strand_defined = true; item.strand = FORWARD; contig_name = contig_name.substr(1); }
    else if (contig_name[0] == '-') { item.strand_defined = true; item.strand = REVERSE; contig_name = contig_name.substr(1); }
    else item.strand_defined = false;
    contig_name = removeChr(contig_name);
    auto contig = contigs.by_name.end();
    if (contig_name.size() >= 2 && contig_name.back() == '*') {
        contig_name.pop_back();
        contig = contigs.by_name.lower_bound(contig_name);
        if (contig != contigs.by_name.end() && contig_name != contig->first.substr(0, contig_name.size())) contig = contigs.by_name.end();
    } else {
        contig = contigs.by_name.find(contig_name);
    }
    if (contig == contigs.by_name.end()) return false;
    item.contig = contig->second;
    char* endp = nullptr;
    size_t dash = se.find('-');
    if (dash < se.size()) {
        item.start = std::strtol(se.substr(0, dash).c_str(), &endp, 10) - 1;
        item.end = std::strtol(se.substr(dash + 1).c_str(), &endp, 10) - 1;
    } else {
        item.start = std::strtol(se.c_str(), &endp, 10) - 1;
        item.end = item.start;
    }
    return true;
}
inline bool parse_blacklist_item(const std::string& text, blacklist_item_t& item, const contigs_t& contigs, const std::unordered_map<std::string, gene_t>& genes, const bool allow_keyword) {
    if (text.empty()) return false;
    if (allow_keyword) {
        if (text == "any") { item.type = BLACKLIST_ANY; return true; }
        if (text == "split_read_donor") { item.type = BLACKLIST_SPLIT_READ_DONOR; return true; }
        if (text == "split_read_acceptor") { item.type = BLACKLIST_SPLIT_READ_ACCEPTOR; return true; }
        if (text == "split_read_any") { item.type = BLACKLIST_SPLIT_READ_ANY; return true; }
        if (text == "discordant_mates") { item.type = BLACKLIST_DISCORDANT_MATES; return true; }
        if (text == "read_through") { item.type = BLACKLIST_READ_THROUGH; return true; }
        if (text == "low_support") { item.type = BLACKLIST_LOW_SUPPORT; return true; }
        if (text == "filter_spliced") { item.type = BLACKLIST_FILTER_SPLICED; return true; }
        if (text == "not_both_spliced") { item.type = BLACKLIST_NOT_BOTH_SPLICED; return true; }
    }
    auto g = genes.find(text);
    if (g != genes.end()) {
        item.type = BLACKLIST_GENE; item.gene = g->second; item.contig = g->second->contig; item.start = g->second->start; item.end = g->second->end;
    } else if (parse_range(text, contigs, item)) {
        item.type = item.start == item.end ? BLACKLIST_POSITION : BLACKLIST_RANGE;
    } else return false;
    return true;
}
inline float overlapping_fraction(position_t s1, position_t e1, position_t s2, position_t e2) {
    if (s1 >= s2 && e1 <= e2) return 1;
    if (s1 < s2 && e1 > e2) return 1.0 * (e2 - s2) / (e1 - s1 + 1);
    if (s1 >= s2 && s1 <= e2) return 1.0 * (e2 - s1) / (e1 - s1 + 1);
    if (e1 >= s2 && e1 <= e2) return 1.0 * (e1 - s2) / (e1 - s1 + 1);
    return 0;
}
inline bool matches_blacklist_item(const blacklist_item_t& it, const fusion_t& f, const unsigned char which, const int max_mate_gap, const float evalue_cutoff) {
    switch (it.type) {
        case BLACKLIST_ANY: return true;
        case BLACKLIST_SPLIT_READ_DONOR: return (which == 1 && f.discordant_mates + f.split_reads1 == 0) || (which == 2 && f.discordant_mates + f.split_reads2 == 0);
        case BLACKLIST_SPLIT_READ_ACCEPTOR: return (which == 1 && f.discordant_mates + f.split_reads2 == 0) || (which == 2 && f.discordant_mates + f.split_reads1 == 0);
        case BLACKLIST_SPLIT_READ_ANY: return f.discordant_mates == 0;
        case BLACKLIST_DISCORDANT_MATES: return f.split_reads1 + f.split_reads2 == 0;
        case BLACKLIST_READ_THROUGH: return f.is_read_through();
        case BLACKLIST_LOW_SUPPORT: return f.evalue > evalue_cutoff;
        case BLACKLIST_FILTER_SPLICED: return f.evalue > evalue_cutoff && f.spliced1 && f.spliced2;
        case BLACKLIST_NOT_BOTH_SPLICED: return !f.spliced1 || !f.spliced2;
        case BLACKLIST_GENE: return (which == 1 && f.gene1 == it.gene) || (which == 2 && f.gene2 == it.gene);
        case BLACKLIST_POSITION: {
            contig_t c = (which == 1) ? f.contig1 : f.contig2;
            if (c != it.contig) return false;
            if (it.strand_defined && !f.predicted_strands_ambiguous) { strand_t s = (which == 1) ? f.predicted_strand1 : f.predicted_strand2; if (s != it.strand) return false; }
            position_t bp = (which == 1) ? f.breakpoint1 : f.breakpoint2;
            if (bp == it.start) return true;
            if (f.split_reads1 + f.split_reads2 == 0) {
                direction_t d = (which == 1) ? f.direction1 : f.direction2;
                if ((d == DOWNSTREAM && bp <= it.start && bp >= it.start - max_mate_gap) || (d == UPSTREAM && bp >= it.start && bp <= it.start + max_mate_gap)) return true;
            }
            return false;
        }
        case BLACKLIST_RANGE: {
            contig_t c = (which == 1) ? f.contig1 : f.contig2;
            if (c != it.contig) return false;
            if (it.strand_defined && !f.predicted_strands_ambiguous) { strand_t s = (which == 1) ? f.predicted_strand1 : f.predicted_strand2; if (s != it.strand) return false; }
            gene_t g = (which == 1) ? f.gene1 : f.gene2;
            return overlapping_fraction(g->start, g->end, it.start, it.end) > 0.5;
        }
    }
    return false;
}
inline void get_genome_bins_from_range(contig_t contig, position_t start, position_t end, std::vector<std::pair<contig_t, position_t>>& bins) {
    const int bucket = 100000;
    for (position_t p = start / bucket; p <= (end + bucket - 1) / bucket; ++p) bins.emplace_back(contig, p * bucket);
}
inline unsigned int filter_blacklisted_ranges(fusions_t& fusions, const std::string& path, const contigs_t& contigs, const std::unordered_map<std::string, gene_t>& genes,
                                              const float evalue_cutoff, const int max_mate_gap) {
    std::map<std::pair<contig_t, position_t>, std::set<fusion_t*>> by_bin;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none && f.closest_genomic_breakpoint1 < 0) continue;
        std::vector<std::pair<contig_t, position_t>> bins;
        get_genome_bins_from_range(f.contig1, f.breakpoint1, f.breakpoint1, bins);
        get_genome_bins_from_range(f.contig2, f.breakpoint2, f.breakpoint2, bins);
        get_genome_bins_from_range(f.contig1, f.gene1->start, f.gene1->end, bins);
        get_genome_bins_from_range(f.contig2, f.gene2->start, f.gene2->end, bins);
        for (auto& b : bins) by_bin[b].insert(&f);
    }
    detail::for_each_line(path, [&](const std::string& line) {
        if (line.empty() || line[0] == '#') return;
        std::istringstream tsv(line);
        std::string r1, r2;
        tsv >> r1 >> r2;
        blacklist_item_t i1, i2;
        if (!parse_blacklist_item(r1, i1, contigs, genes, false) || !parse_blacklist_item(r2, i2, contigs, genes, true)) return;
        std::vector<std::pair<contig_t, position_t>> bins;
        if (i1.type == BLACKLIST_POSITION || i1.type == BLACKLIST_RANGE || i1.type == BLACKLIST_GENE) get_genome_bins_from_range(i1.contig, i1.start - max_mate_gap, i1.end + max_mate_gap, bins);
        if (i2.type == BLACKLIST_POSITION || i2.type == BLACKLIST_RANGE || i2.type == BLACKLIST_GENE) get_genome_bins_from_range(i2.contig, i2.start - max_mate_gap, i2.end + max_mate_gap, bins);
        for (auto& b : bins) {
            auto near = by_bin.find(b);
            if (near == by_bin.end()) continue;
            for (auto f = near->second.begin(); f != near->second.end();) {
                if ((matches_blacklist_item(i1, **f, 1, max_mate_gap, evalue_cutoff) && matches_blacklist_item(i2, **f, 2, max_mate_gap, evalue_cutoff)) ||
                    (matches_blacklist_item(i1, **f, 2, max_mate_gap, evalue_cutoff) && matches_blacklist_item(i2, **f, 1, max_mate_gap, evalue_cutoff))) {
                    (*f)->filter = FILTER_blacklist;
                    f = near->second.erase(f);
                } else ++f;
            }
        }
    });
    return count_remaining(fusions);
}

// filter_short_anchor.cpp
inline unsigned int filter_short_anchor(fusions_t& fusions, const unsigned int min_length) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (!(f.spliced1 && f.spliced2) && ((unsigned int)std::abs(f.anchor_start1 - f.breakpoint1) < min_length || (unsigned int)std::abs(f.anchor_start2 - f.breakpoint2) < min_length))
            f.filter = FILTER_short_anchor;
        else remaining++;
    }
    return remaining;
}

// filter_end_to_end.cpp: calculate_intronic_fraction, filter_end_to_end_fusions
inline float calculate_intronic_fraction(const gene_t gene, const exon_annotation_index_t& exon_index) {
    unsigned int intronic = 0;
    position_t prev = gene->start;
    if ((size_t)gene->contig < exon_index.size()) {
        const auto& ci = exon_index[gene->contig];
        for (size_t k = ci.lower_bound(gene->start); k < ci.size() && ci.keys[k] <= gene->end; ++k)
            for (exon_t e : ci.sets[k])
                if (e->gene == gene) {
                    if (prev < e->start) intronic += e->start - prev;
                    if (prev < e->end) prev = e->end + 1;
                    break;
                }
    }
    return ((float)intronic) / (gene->end - gene->start + 1);
}
inline unsigned int filter_end_to_end_fusions(fusions_t& fusions, const exon_annotation_index_t& exon_index, const std::vector<bool>& viral) {
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (viral[f.contig1] || viral[f.contig2]) continue;
        if (!f.is_read_through() && f.gene1 != f.gene2 && (f.spliced1 || f.spliced2)) { ++remaining; continue; }
        if (f.discordant_mates + f.split_reads1 == 0 || f.discordant_mates + f.split_reads2 == 0 || f.split_reads1 + f.split_reads2 == 0 ||
            (f.breakpoint_overlaps_both_genes() && (f.split_reads1 == 0 || f.split_reads2 == 0))) {
            if ((f.gene1->is_dummy || (f.gene1->strand == FORWARD && f.direction1 == UPSTREAM) || (f.gene1->strand == REVERSE && f.direction1 == DOWNSTREAM)) &&
                (f.gene2->is_dummy || (f.gene2->strand == FORWARD && f.direction2 == UPSTREAM) || (f.gene2->strand == REVERSE && f.direction2 == DOWNSTREAM))) {
                if (f.discordant_mates < 10 || (f.contig1 == f.contig2 && std::abs(f.breakpoint1 - f.breakpoint2) < 1000000) ||
                    (f.exonic1 && f.exonic2 && calculate_intronic_fraction(f.gene1, exon_index) > 0.66 && calculate_intronic_fraction(f.gene2, exon_index) > 0.66)) {
                    f.filter = FILTER_end_to_end;
                    continue;
                }
            }
        }
        ++remaining;
    }
    return remaining;
}

// filter_no_coverage.cpp
inline unsigned int filter_no_coverage(fusions_t& fusions, const coverage_t& coverage, const exon_annotation_index_t& exon_index) {
    const int scan_range = 200;
    unsigned int remaining = 0;
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        if (f.filter != FILTER_none) continue;
        if (!f.is_read_through()) {
            if (f.split_reads1 + f.split_reads2 != 0 && f.split_reads1 + f.discordant_mates != 0 && f.split_reads2 + f.discordant_mates != 0) { ++remaining; continue; }
            if (f.spliced1 || f.spliced2) { ++remaining; continue; }
        } else if (f.spliced1 && f.spliced2) { ++remaining; continue; }
        bool filtered = false;
        for (int side = 1; side <= 2 && !filtered; ++side) {
            const contig_t c = side == 1 ? f.contig1 : f.contig2;
            const position_t bp = side == 1 ? f.breakpoint1 : f.breakpoint2, anchor = side == 1 ? f.anchor_start1 : f.anchor_start2;
            const direction_t d = side == 1 ? f.direction1 : f.direction2;
            const gene_t g = side == 1 ? f.gene1 : f.gene2;
            exon_set_t exons;
            get_annotation_by_coordinate(c, bp, bp, exons, exon_index);
            bool terminal = false;
            for (auto e = exons.begin(); e != exons.end() && !terminal; ++e) if ((*e)->gene == g && ((*e)->previous_exon == nullptr || (*e)->next_exon == nullptr)) terminal = true;
            if (terminal) continue;
            position_t start, end;
            if (d == UPSTREAM) { start = bp; if (f.split_reads1 + f.split_reads2 == 0) start -= scan_range; end = std::max(bp + scan_range, anchor); }
            else { start = std::min(bp - scan_range, anchor); end = bp; if (f.split_reads1 + f.split_reads2 == 0) end += scan_range; }
            if ((d == UPSTREAM && !coverage.fragment_starts_here(c, start, end)) || (d == DOWNSTREAM && !coverage.fragment_ends_here(c, start, end))) filtered = true;
        }
        if (filtered) f.filter = FILTER_no_coverage; else remaining++;
    }
    return remaining;
}

// filter_genomic_support.cpp: assign_confidence
inline void assign_confidence(fusions_t& fusions, const coverage_t& coverage) {
    std::map<gene_t, std::vector<fusion_t*>> by_gene;
    for (auto& kv : fusions) { by_gene[kv.second.gene1].push_back(&kv.second); by_gene[kv.second.gene2].push_back(&kv.second); }
    for (auto& kv : fusions) {
        fusion_t& f = kv.second;
        int c1 = coverage.get_coverage(f.contig1, f.breakpoint1, (f.direction1 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        int c2 = coverage.get_coverage(f.contig2, f.breakpoint2, (f.direction2 == UPSTREAM) ? DOWNSTREAM : UPSTREAM);
        float cf = ((float)(f.split_read1_list.size() + f.split_read2_list.size() + f.discordant_mate_list.size())) / std::max(1, std::max(c1, c2));
        if (f.filter != FILTER_none) { f.confidence = CONFIDENCE_LOW; continue; }
        f.confidence = CONFIDENCE_HIGH;
        if (f.evalue > 0.3 || f.supporting_reads() < 2) {
            f.confidence = CONFIDENCE_LOW;
        } else if (f.is_read_through()) {
            f.confidence = CONFIDENCE_LOW;
            if (((f.split_reads1 > 0 && f.split_reads2 > 0) || (f.split_reads1 > 0 && f.discordant_mates > 0) || (f.split_reads2 > 0 && f.discordant_mates > 0)) && f.supporting_reads() >= 10) {
                f.confidence = (f.split_reads1 + f.split_reads2 >= 10 && cf > 0.07) ? CONFIDENCE_HIGH : CONFIDENCE_MEDIUM;
            } else {
                unsigned int dels = 0;
                for (gene_t g : {f.gene1, f.gene2})
                    for (fusion_t* o : by_gene[g])
                        if (o->filter == FILTER_none && o->split_reads1 + o->split_reads2 > 0 && o->direction1 == DOWNSTREAM && o->direction2 == UPSTREAM &&
                            ((o->gene1 == f.gene1 && o->gene2 != f.gene2) || (o->gene1 != f.gene1 && o->gene2 == f.gene2)) &&
                            (o->breakpoint1 != f.breakpoint1 || o->breakpoint2 != f.breakpoint2) && o->breakpoint2 > f.breakpoint1 && o->breakpoint1 < f.breakpoint2)
                            ++dels;
                if (dels >= 1) f.confidence = CONFIDENCE_MEDIUM;
            }
        } else if (f.breakpoint_overlaps_both_genes() || f.gene1 == f.gene2) {
            f.confidence = CONFIDENCE_LOW;
            if (f.split_reads1 + f.split_reads2 > 0) {
                if (!f.exonic1 && !f.exonic2) f.confidence = (f.split_reads1 > 0 && f.split_reads2 > 0) ? CONFIDENCE_HIGH : CONFIDENCE_MEDIUM;
                else if (!f.exonic1 || !f.exonic2) f.confidence = (f.split_reads1 > 3 && f.split_reads2 > 3) ? CONFIDENCE_HIGH : CONFIDENCE_MEDIUM;
            }
        }
        if (f.confidence == CONFIDENCE_LOW && f.gene1 == f.gene2 && f.exonic1 && f.exonic2 && !f.spliced1 && !f.spliced2 && f.breakpoint2 - f.breakpoint1 < 100 &&
            f.split_reads1 > 0 && f.split_reads2 > 0 && f.split_reads1 + f.split_reads2 >= 10 && cf > 0.15 && f.direction1 == UPSTREAM && f.direction2 == DOWNSTREAM)
            f.confidence = CONFIDENCE_MEDIUM;
        if (f.confidence < CONFIDENCE_HIGH && f.spliced1 && f.spliced2 && !f.is_read_through() && f.gene1 != f.gene2) {
            unsigned int n = 0;
            for (gene_t g : {f.gene1, f.gene2})
                for (fusion_t* o : by_gene[g])
                    if (o->gene1 == f.gene1 && o->gene2 == f.gene2 && o->spliced1 && o->spliced2 &&
                        (std::abs(o->breakpoint1 - f.breakpoint1) > 2 || std::abs(o->breakpoint2 - f.breakpoint2) > 2))
                        ++n;
            if (n > 0) f.confidence++;
        }
        if (f.gene1 != f.gene2 && f.confidence > CONFIDENCE_LOW && !f.spliced1 && !f.spliced2) f.confidence--;
        if (f.split_reads1 > 20 && f.split_reads2 > 20 && f.supporting_reads() > 60) f.confidence = CONFIDENCE_HIGH;
        if (f.confidence > CONFIDENCE_LOW) {
            if (f.split_reads1 + f.split_reads2 == 0 || f.split_reads1 + f.discordant_mates == 0 || f.split_reads2 + f.discordant_mates == 0) f.confidence--;
            else if ((f.split_reads1 + f.split_reads2) * 20 < f.discordant_mates) f.confidence--;
            else if (f.evalue > 0.2 || cf < 0.01) f.confidence = CONFIDENCE_MEDIUM;
        }
        // (increase for WGS support: not applicable, no genomic breakpoints)
    }
}

// output_fusions.cpp: gene_to_name, sort_fusions_by_support
inline std::string gene_to_name(const gene_t gene, contig_t contig, position_t bp, const gene_annotation_index_t& gene_index) {
    if (!gene->is_dummy) return gene->name;
    std::string result;
    if ((size_t)contig >= gene_index.size()) return ".";
    const auto& ci = gene_index[contig];
    size_t hit2 = ci.lower_bound(bp);
    // go upstream until we find a non-dummy gene
    size_t hit1 = hit2;
    while (hit1 > 0 && (ci.sets[hit1 - 1].empty() || ci.sets[hit1 - 1][0]->is_dummy)) --hit1;
    if (hit1 > 0)
        for (gene_t g : ci.sets[hit1 - 1]) if (!g->is_dummy) { if (!result.empty()) result += ","; result += g->name + "(" + std::to_string(bp - g->end) + ")"; }
    while (hit2 < ci.size() && (ci.sets[hit2].empty() || ci.sets[hit2][0]->is_dummy)) ++hit2;
    if (hit2 < ci.size())
        for (gene_t g : ci.sets[hit2]) if (!g->is_dummy) { if (!result.empty()) result += ","; result += g->name + "(" + std::to_string(g->start - bp) + ")"; }
    return result.empty() ? "." : result;
}
inline bool sort_fusions_by_support(const fusion_t* x, const fusion_t* y) {
    if (x->confidence != y->confidence) return x->confidence > y->confidence;
    if (x->supporting_reads() != y->supporting_reads()) return x->supporting_reads() > y->supporting_reads();
    if (x->evalue != y->evalue) return x->evalue < y->evalue;
    if (x->gene1->id != y->gene1->id) return x->gene1->id < y->gene1->id;
    if (x->gene2->id != y->gene2->id) return x->gene2->id < y->gene2->id;
    if (x->breakpoint1 != y->breakpoint1) return x->breakpoint1 < y->breakpoint1;
    return x->breakpoint2 < y->breakpoint2;
}

// ================================================================= driver (arriba.cpp: main, the cascade order of lines ~320-560)

struct options_t {  // options.cpp defaults
    std::string interesting_contigs = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y AC_* NC_*";
    std::string viral_contigs = "AC_* NC_*";
    float evalue_cutoff = 0.3;
    unsigned int min_support = 2;
    unsigned int min_anchor_length = 23;
    unsigned int homopolymer_length = 6;
    int min_read_through_distance = 10000;
    float max_kmer_content = 0.6;
    int fragment_length = 200;
    strandedness_t strandedness = STRANDEDNESS_AUTO;
    unsigned int min_spliced_events = 4;
    float mismatch_pvalue_cutoff = 0.01;
    unsigned int subsampling_threshold = 300;
    float high_expression_quantile = 0.998;
    float exonic_fraction = 0.33;
    unsigned int max_itd_length = 100;
    std::string blacklist_path;  // empty = no blacklist
};

struct config_t {
    input_t input;
    std::string gtf_path, assembly_path, gene_counts_path;  // gene_counts_path: gene_expression.tsv (gene_id, gene_name, effective_length, count, ...)
    unsigned long mapped_reads = 0;
    options_t options;
};

struct result_t {
    bool ok = false;
    std::string error;
    load_stats_t load;
    strandedness_t strandedness = STRANDEDNESS_NO;
    std::vector<std::pair<std::string, unsigned int>> log;  // (step, remaining), as Arriba logs them
    annotation_t annotation;
    std::unique_ptr<assembly_t> assembly;
    chimeric_alignments_t alignments;
    fusions_t fusions;
    coverage_t coverage;
    std::vector<std::string> barcodes;
};

inline bool load_gene_counts(const std::string& path, coverage_t& cov) {
    if (path.empty()) return false;
    bool header = true;
    const bool ok = detail::for_each_line(path, [&](const std::string& ln) {
        if (header) { header = false; if (ln.compare(0, 7, "gene_id") == 0) return; }
        auto f = detail::split(ln, '\t');
        if (f.size() < 4) return;
        const double c = std::atof(f[3].c_str()), eff = std::max(110.0, std::atof(f[2].c_str()));
        const double depth = c > 0 ? std::max(1.0, std::round(c * 110.0 / eff)) : 0.0;  // singlet: mean depth over a 20-bp window of 90-bp reads
        cov.count_by_gene_id[strip_ensembl_version_number(f[0])] = (unsigned int)std::min(depth, 4e9);
    });
    cov.known = ok && !cov.count_by_gene_id.empty();
    return cov.known;
}

// The cascade, step by step in Arriba's order. Steps that need data this input lacks are logged as skipped.
inline void run(const config_t& cfg, result_t& r) {
    const options_t& o = cfg.options;
    r.barcodes = cfg.input.barcodes;
    r.assembly.reset(new assembly_t());
    if (!cfg.assembly_path.empty() && !r.assembly->load(cfg.assembly_path, r.annotation.contigs)) { r.error = "cannot load assembly " + cfg.assembly_path; return; }
    if (!read_annotation_gtf(cfg.gtf_path, *r.assembly, r.annotation)) { r.error = "cannot read GTF " + cfg.gtf_path; return; }
    annotation_t& a = r.annotation;
    make_annotation_index(a.exons, a.exon_index, a.contigs.size());
    make_annotation_index(a.genes, a.gene_index, a.contigs.size());
    chimeric_alignments_t& ca = r.alignments;
    if (!read_chimeric_junctions(cfg.input, a.contigs, ca, r.load)) { r.error = "cannot read " + cfg.input.junction_path; return; }
    r.log.emplace_back("chimeric_reads", (unsigned int)ca.size());
    std::vector<bool> viral(a.contigs.size()), interesting(a.contigs.size());
    for (auto& c : a.contigs.by_name) { viral[c.second] = is_interesting_contig(c.first, o.viral_contigs); interesting[c.second] = is_interesting_contig(c.first, o.interesting_contigs); }
    r.log.emplace_back("multimappers_marked", mark_multimappers(ca));
    a.gene_index.resize(a.contigs.size());
    a.exon_index.resize(a.contigs.size());
    r.strandedness = o.strandedness == STRANDEDNESS_AUTO ? detect_strandedness(ca, a.gene_index, a.exon_index) : o.strandedness;
    if (r.strandedness != STRANDEDNESS_NO) assign_strands_from_strandedness(ca, r.strandedness);
    annotate_all(ca, a);
    load_gene_counts(cfg.gene_counts_path, r.coverage);
    r.coverage.gene_index = &a.gene_index;
    mark_umi_duplicates(ca);
    const bool seq = r.load.with_sequence > 0 && r.assembly->data != nullptr;
    r.log.emplace_back("duplicates", filter_duplicates(ca));
    r.log.emplace_back("uninteresting_contigs", filter_uninteresting_contigs(ca, interesting));
    // estimate_fragment_length needs paired-end reads; Arriba then falls back to the fragment length option
    const int max_mate_gap = o.fragment_length;
    r.log.emplace_back("read_through", filter_proximal_read_through(ca, o.min_read_through_distance));
    r.log.emplace_back("inconsistently_clipped", filter_inconsistently_clipped_mates(ca));
    r.log.emplace_back(seq ? "homopolymer" : "homopolymer (skipped: no read sequences)", seq ? filter_homopolymer(ca, o.homopolymer_length, a.exon_index) : 0u);
    r.log.emplace_back("small_insert_size", filter_small_insert_size(ca, 5));
    r.log.emplace_back("long_gap", filter_long_gap(ca));
    r.log.emplace_back("same_gene", filter_same_gene(ca));
    r.log.emplace_back("hairpin", filter_hairpin(ca, max_mate_gap));
    r.log.emplace_back(seq ? "mismatches" : "mismatches (skipped: no read sequences)", seq ? filter_mismatches(ca, *r.assembly, interesting, viral, 0.01, o.mismatch_pvalue_cutoff) : 0u);
    r.log.emplace_back(seq ? "low_entropy" : "low_entropy (skipped: no read sequences)", seq ? filter_low_entropy(ca, 3, o.max_kmer_content, o.max_itd_length) : 0u);
    fusions_t& fu = r.fusions;
    r.log.emplace_back("find_fusions", find_fusions(ca, fu, a.exon_index, o.subsampling_threshold));
    r.log.emplace_back("merge_adjacent", merge_adjacent_fusions(fu, 5, o.max_itd_length));
    r.log.emplace_back("multimappers", filter_multimappers(ca, fu, a.exon_index, *r.assembly));
    estimate_expected_fusions(fu, cfg.mapped_reads, a.exon_index);
    r.log.emplace_back("non_coding_neighbors", filter_non_coding_neighbors(fu));
    r.log.emplace_back("intragenic_exonic", filter_intragenic_both_exonic(fu, a.exon_index, o.exonic_fraction));
    r.log.emplace_back("min_support", filter_min_support(fu, o.min_support));
    r.log.emplace_back("relative_support", filter_relative_support(fu, o.evalue_cutoff));
    r.log.emplace_back("intronic", filter_both_intronic(fu, viral));
    r.log.emplace_back("in_vitro", filter_in_vitro(fu, ca, o.high_expression_quantile, a.gene_index, r.coverage));
    r.log.emplace_back("spliced", recover_both_spliced(fu, ca, a.exon_index, r.coverage, 200, 0.998, 1000, 1000));
    r.log.emplace_back("select_best", select_most_supported_breakpoints(fu));
    r.log.emplace_back("marginal_read_through", filter_marginal_read_through(fu, r.coverage));
    r.log.emplace_back("many_spliced", recover_many_spliced(fu, o.min_spliced_events));
    if (!o.blacklist_path.empty()) r.log.emplace_back("blacklist", filter_blacklisted_ranges(fu, o.blacklist_path, a.contigs, a.gene_names, o.evalue_cutoff, max_mate_gap));
    r.log.emplace_back("short_anchor", filter_short_anchor(fu, o.min_anchor_length));
    r.log.emplace_back("end_to_end", filter_end_to_end_fusions(fu, a.exon_index, viral));
    r.log.emplace_back("no_coverage", filter_no_coverage(fu, r.coverage, a.exon_index));
    // the 'select_best' filter runs again after 'many_spliced'
    r.log.emplace_back("select_best", select_most_supported_breakpoints(fu));
    assign_confidence(fu, r.coverage);
    r.ok = true;
}

inline const char* confidence_name(confidence_t c) { return c == CONFIDENCE_HIGH ? "high" : c == CONFIDENCE_MEDIUM ? "medium" : "low"; }

inline std::string site_name(const fusion_t& f, int which) {
    const bool sp = which == 1 ? f.spliced1 : f.spliced2, ex = which == 1 ? f.exonic1 : f.exonic2;
    const gene_t g = which == 1 ? f.gene1 : f.gene2;
    if (g->is_dummy) return "intergenic";
    return sp ? "splice-site" : ex ? "exon" : "intron";
}

// Writes the calls (passing fusions, Arriba's output order) or the discarded fusions. Columns follow Arriba's fusions.tsv where they exist;
// cells and molecules are singlet's: distinct cells and the molecules (cell + UMI) behind the non-filtered supporting reads.
inline bool write_fusions(const result_t& r, const std::string& path, bool discarded) {
    std::vector<const fusion_t*> v;
    for (auto& kv : r.fusions) if (discarded != (kv.second.filter == FILTER_none)) v.push_back(&kv.second);
    if (!discarded) {
        std::map<std::pair<gene_t, gene_t>, const fusion_t*> best;
        for (const fusion_t* f : v) { const fusion_t*& b = best[{f->gene1, f->gene2}]; if (!b || sort_fusions_by_support(f, b)) b = f; }
        std::stable_sort(v.begin(), v.end(), [&](const fusion_t* x, const fusion_t* y) {
            const fusion_t* bx = best.at({x->gene1, x->gene2});
            const fusion_t* by = best.at({y->gene1, y->gene2});
            return bx != by ? sort_fusions_by_support(bx, by) : sort_fusions_by_support(x, y);
        });
    }
    gzFile out = gzopen(path.c_str(), "wb6");
    if (!out) return false;
    std::string h = "#gene1\tgene2\tstrand1(gene/fusion)\tstrand2(gene/fusion)\tbreakpoint1\tbreakpoint2\tsite1\tsite2\tsplit_reads1\tsplit_reads2\tdiscordant_mates\t"
                    "coverage1\tcoverage2\tconfidence\tevalue\tgene_id1\tgene_id2\tdirection1\tdirection2\tfilters\tcells\tmolecules\tbarcodes\tread_identifiers\n";
    gzwrite(out, h.data(), (unsigned)h.size());
    const auto& a = r.annotation;
    for (const fusion_t* f : v) {
        std::set<uint32_t> cells;
        std::set<std::pair<uint32_t, std::string>> mols;
        std::map<std::string, unsigned int> filters;
        std::vector<std::string> names;
        for (auto* lst : {&f->split_read1_list, &f->split_read2_list})
            for (auto it : *lst) {
                if (it->second.filter == FILTER_none) {
                    cells.insert(it->second.cell);
                    mols.emplace(it->second.cell, it->second.umi);
                    names.push_back(strip_hi_tag_from_read_name(it->first));
                } else filters[filter_name(it->second.filter)]++;
            }
        std::string fl;
        if (f->filter != FILTER_none) fl = filter_name(f->filter);
        for (auto& kv : filters) fl += (fl.empty() ? "" : ",") + kv.first + "(" + std::to_string(kv.second) + ")";
        std::string bcs;
        for (uint32_t c : cells) bcs += (bcs.empty() ? "" : ",") + r.barcodes[c];
        std::string nm;
        for (auto& n : names) nm += (nm.empty() ? "" : ",") + n;
        auto sc = [](strand_t s) { return s == FORWARD ? "+" : "-"; };
        std::ostringstream os;
        os << gene_to_name(f->gene1, f->contig1, f->breakpoint1, a.gene_index) << '\t' << gene_to_name(f->gene2, f->contig2, f->breakpoint2, a.gene_index) << '\t'
           << (f->gene1->is_dummy ? "." : sc(f->gene1->strand)) << '/' << (f->predicted_strands_ambiguous ? "." : sc(f->predicted_strand1)) << '\t'
           << (f->gene2->is_dummy ? "." : sc(f->gene2->strand)) << '/' << (f->predicted_strands_ambiguous ? "." : sc(f->predicted_strand2)) << '\t'
           << a.contigs.original[f->contig1] << ':' << f->breakpoint1 + 1 << '\t' << a.contigs.original[f->contig2] << ':' << f->breakpoint2 + 1 << '\t'
           << site_name(*f, 1) << '\t' << site_name(*f, 2) << '\t' << f->split_reads1 << '\t' << f->split_reads2 << '\t' << f->discordant_mates << '\t'
           << r.coverage.get_coverage(f->contig1, f->breakpoint1, f->direction1) << '\t' << r.coverage.get_coverage(f->contig2, f->breakpoint2, f->direction2) << '\t'
           << confidence_name(f->confidence) << '\t' << f->evalue << '\t' << (f->gene1->is_dummy ? "." : f->gene1->gene_id) << '\t'
           << (f->gene2->is_dummy ? "." : f->gene2->gene_id) << '\t' << (f->direction1 == UPSTREAM ? "upstream" : "downstream") << '\t'
           << (f->direction2 == UPSTREAM ? "upstream" : "downstream") << '\t' << (fl.empty() ? "." : fl) << '\t' << cells.size() << '\t' << mols.size() << '\t'
           << (bcs.empty() ? "." : bcs) << '\t' << (nm.empty() ? "." : nm) << '\n';
        const std::string s = os.str();
        gzwrite(out, s.data(), (unsigned)s.size());
    }
    return gzclose(out) == Z_OK;
}

}  // namespace arriba
}  // namespace singlet
