// SPDX-License-Identifier: MIT
#pragma once
// polya_sites: per-cell poly(A) site usage from read 3' ends that carry a soft-clipped poly(A) tail.
//
// In 3' libraries (10x 3', Drop-seq, ...) the cDNA read runs 5'->3' along the transcript; reads that reach the
// poly(A) tail end in A's that STAR soft-clips. The last aligned base is then the cleavage / poly(A) site. This is lost
// once the BAM is deleted, so the engine records (site, gene, cell, UMI) for every unique, barcoded read with such a
// tail, and export clusters the sites and writes a sparse member like sj_counts:
//
//   polya_counts        sites x barcodes, molecules (distinct UMI per site cluster and cell), 1pz or mtx.gz
//   polya_sites.tsv.gz  one row per site (matrix row order): site, gene_id, gene_name, chrom, strand, position (1-based,
//                       the cluster's mode), start, end (cluster span), n_molecules, n_reads, n_cells,
//                       internal_priming (0/1), downstream_a (A count in the 20 nt downstream on the transcript strand),
//                       pas_motif (best canonical hexamer AATAAA/ATTAAA/... found 10-40 nt upstream, or "-")
//
// Rules: read uniquely mapped (NH 1), MAPQ filter as for counting, gene assigned (exon or intron); soft clip at the
// transcript 3' end >= 6 nt with >= 80% A (T on reverse-mapped reads). 5' protocols (reads run towards the 5' end) are
// skipped. Sites on the same gene and strand within 12 nt of each other (single linkage) form one cluster; clusters
// with < 2 molecules over the sample are dropped. internal_priming = >= 12 A in the 20 nt downstream, or a run of
// >= 8 A in it (oligo-dT primed on genomic A-rich sequence, not a true poly(A) site; flagged, not removed). The
// genome sequence is read from the STAR Genome binary (one byte per base, as mt_reference.h reads chrM).

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <zlib.h>

#include <htslib/sam.h>

namespace singlet {

struct PolyARecord {
    uint64_t site;     ///< tid << 33 | strand << 32 | 0-based position
    uint32_t gene;     ///< gene index (UINT32_MAX = none)
    uint32_t bc;       ///< barcode column
    uint64_t umi;      ///< UMI hash (0 = no UMI)
};

inline uint64_t polya_site_key(int32_t tid, bool minus, uint32_t pos0) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(tid)) << 33) | (static_cast<uint64_t>(minus ? 1 : 0) << 32) | pos0;
}
inline int32_t polya_site_tid(uint64_t k) { return static_cast<int32_t>(k >> 33); }
inline bool polya_site_minus(uint64_t k) { return ((k >> 32) & 1u) != 0; }
inline uint32_t polya_site_pos(uint64_t k) { return static_cast<uint32_t>(k & 0xffffffffu); }

inline uint64_t polya_umi_hash(const char* umi, int len) {
    if (!umi || len <= 0) return 0;
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < len; ++i) { h ^= static_cast<unsigned char>(umi[i]); h *= 1099511628211ull; }
    return h | 1u;
}

/// Site of a read with a soft-clipped poly(A) tail at its transcript 3' end; false when the read has none.
inline bool polya_site_from_read(const bam1_t* b, int min_clip, double min_frac, uint64_t& key) {
    const uint32_t nc = b->core.n_cigar;
    if (nc < 2) return false;
    const uint32_t* cig = bam_get_cigar(b);
    const bool rev = (b->core.flag & BAM_FREVERSE) != 0;
    const uint8_t* seq = bam_get_seq(b);
    const int qlen = b->core.l_qseq;
    if (!rev) {
        // + strand transcript: tail = trailing soft clip of A's; site = last aligned reference base
        if (bam_cigar_op(cig[nc - 1]) != BAM_CSOFT_CLIP) return false;
        const int cl = static_cast<int>(bam_cigar_oplen(cig[nc - 1]));
        if (cl < min_clip || cl > qlen) return false;
        int na = 0;
        for (int i = qlen - cl; i < qlen; ++i) na += (bam_seqi(seq, i) == 1);  // nt16 A = 1
        if (na < min_frac * cl) return false;
        if (bam_seqi(seq, qlen - cl) != 1) return false;  // tail starts right at the clip
        const hts_pos_t end = bam_endpos(b);  // exclusive
        if (end <= 0) return false;
        key = polya_site_key(b->core.tid, false, static_cast<uint32_t>(end - 1));
        return true;
    }
    // - strand transcript: read stored reverse-complemented; tail = leading soft clip of T's; site = first aligned base
    if (bam_cigar_op(cig[0]) != BAM_CSOFT_CLIP) return false;
    const int cl = static_cast<int>(bam_cigar_oplen(cig[0]));
    if (cl < min_clip || cl > qlen) return false;
    int nt = 0;
    for (int i = 0; i < cl; ++i) nt += (bam_seqi(seq, i) == 8);  // nt16 T = 8
    if (nt < min_frac * cl) return false;
    if (bam_seqi(seq, cl - 1) != 8) return false;
    key = polya_site_key(b->core.tid, true, static_cast<uint32_t>(b->core.pos));
    return true;
}

/// Random access to chromosome sequence in a STAR genome directory (Genome binary: one byte per base, 0-3 = ACGT).
class StarGenomeReader {
public:
    bool open(const std::string& genome_dir) {
        std::ifstream fn(genome_dir + "/chrName.txt"), fs(genome_dir + "/chrStart.txt");
        if (!fn || !fs) return false;
        std::string line;
        while (std::getline(fn, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) names_.push_back(line);
        }
        uint64_t v;
        while (fs >> v) starts_.push_back(v);
        if (starts_.size() < names_.size() + 1) return false;
        for (size_t i = 0; i < names_.size(); ++i) idx_[names_[i]] = i;
        g_.open(genome_dir + "/Genome", std::ios::binary);
        return static_cast<bool>(g_);
    }
    /// Bases [pos0, pos0 + len) of chrom as ACGTN (clipped to the chromosome; empty if unknown).
    std::string fetch(const std::string& chrom, int64_t pos0, int64_t len) {
        auto it = idx_.find(chrom);
        if (it == idx_.end() || !g_) return {};
        const uint64_t cs = starts_[it->second], ce = starts_[it->second + 1];
        if (pos0 < 0) { len += pos0; pos0 = 0; }
        if (len <= 0) return {};
        uint64_t a = cs + static_cast<uint64_t>(pos0), e = std::min<uint64_t>(a + static_cast<uint64_t>(len), ce);
        if (a >= e) return {};
        std::string raw(e - a, '\0');
        g_.clear();
        g_.seekg(static_cast<std::streamoff>(a));
        g_.read(&raw[0], static_cast<std::streamsize>(raw.size()));
        static const char M[] = "ACGTN";
        for (auto& c : raw) c = (static_cast<uint8_t>(c) < 4) ? M[static_cast<uint8_t>(c)] : 'N';
        return raw;
    }
private:
    std::vector<std::string> names_;
    std::vector<uint64_t> starts_;
    std::unordered_map<std::string, size_t> idx_;
    std::ifstream g_;
};

inline std::string polya_revcomp(const std::string& s) {
    std::string r(s.rbegin(), s.rend());
    for (auto& c : r) c = c == 'A' ? 'T' : c == 'T' ? 'A' : c == 'C' ? 'G' : c == 'G' ? 'C' : 'N';
    return r;
}

struct PolyASiteTable {
    uint32_t nrows = 0, ncols = 0;
    std::vector<int32_t> indptr, indices;   ///< CSC sites x barcodes
    std::vector<uint16_t> data;
    std::vector<std::string> names;         ///< gene_name:chrom:strand:pos (1-based)
    std::string table_tsv;                  ///< polya_sites.tsv content
    uint64_t n_records = 0, n_internal_priming = 0;
};

/// Cluster the recorded sites and count molecules per (cluster, barcode).
inline PolyASiteTable build_polya_sites(std::vector<PolyARecord>& recs, uint32_t n_barcodes,
                                        const std::vector<std::string>& tid_to_chrom,
                                        const std::vector<std::string>& gene_ids,
                                        const std::vector<std::string>& gene_names, const std::string& genome_dir,
                                        int link_nt = 12, uint32_t min_molecules = 2) {
    PolyASiteTable t;
    t.ncols = n_barcodes;
    t.n_records = recs.size();
    // order by gene, strand/tid, position
    std::sort(recs.begin(), recs.end(), [](const PolyARecord& a, const PolyARecord& b) {
        if (a.gene != b.gene) return a.gene < b.gene;
        if ((a.site >> 32) != (b.site >> 32)) return (a.site >> 32) < (b.site >> 32);
        if (a.site != b.site) return a.site < b.site;
        if (a.bc != b.bc) return a.bc < b.bc;
        return a.umi < b.umi;
    });
    struct Cluster { size_t begin, end; };
    std::vector<Cluster> cl;
    for (size_t i = 0; i < recs.size();) {
        size_t j = i + 1;
        while (j < recs.size() && recs[j].gene == recs[i].gene && (recs[j].site >> 32) == (recs[i].site >> 32) &&
               polya_site_pos(recs[j].site) - polya_site_pos(recs[j - 1].site) <= static_cast<uint32_t>(link_nt))
            ++j;
        cl.push_back({i, j});
        i = j;
    }
    StarGenomeReader genome;
    const bool have_genome = !genome_dir.empty() && genome.open(genome_dir);
    // triplets (row, col, count)
    struct E { uint32_t col, row; uint16_t v; };
    std::vector<E> ent;
    std::string tsv =
        "site\tgene_id\tgene_name\tchrom\tstrand\tposition\tstart\tend\tn_molecules\tn_reads\tn_cells"
        "\tinternal_priming\tdownstream_a\tpas_motif\n";
    static const char* kPAS[] = {"AATAAA", "ATTAAA", "AGTAAA", "TATAAA", "CATAAA", "GATAAA", "AATATA", "AATACA",
                                 "AATAGA", "AAAAAG", "ACTAAA", "AAGAAA", "AATGAA", "TTTAAA", "AAAACA", "GGGGCT"};
    for (const auto& c : cl) {
        // molecules: distinct (bc, umi) (reads without UMI count as reads)
        std::unordered_map<uint32_t, std::unordered_set<uint64_t>> mol;
        std::unordered_map<uint32_t, uint32_t> nonumi;
        std::unordered_map<uint32_t, uint32_t> pos_count;
        for (size_t k = c.begin; k < c.end; ++k) {
            if (recs[k].umi) mol[recs[k].bc].insert(recs[k].umi); else nonumi[recs[k].bc]++;
            pos_count[polya_site_pos(recs[k].site)]++;
        }
        uint64_t n_mol = 0;
        std::unordered_map<uint32_t, uint32_t> per_bc;
        for (auto& kv : mol) { per_bc[kv.first] += static_cast<uint32_t>(kv.second.size()); }
        for (auto& kv : nonumi) per_bc[kv.first] += kv.second;
        for (auto& kv : per_bc) n_mol += kv.second;
        if (n_mol < min_molecules) continue;
        uint32_t mode = 0, best = 0;
        for (auto& kv : pos_count)
            if (kv.second > best || (kv.second == best && kv.first < mode)) { best = kv.second; mode = kv.first; }
        const uint64_t s0 = recs[c.begin].site;
        const int32_t tid = polya_site_tid(s0);
        const bool minus = polya_site_minus(s0);
        const std::string chrom = (tid >= 0 && static_cast<size_t>(tid) < tid_to_chrom.size()) ? tid_to_chrom[tid] : "?";
        const uint32_t g = recs[c.begin].gene;
        const std::string gid = g < gene_ids.size() ? gene_ids[g] : "NA";
        const std::string gname = g < gene_names.size() ? gene_names[g] : "NA";
        int down_a = -1;
        bool ip = false;
        std::string pas = "-";
        if (have_genome) {
            // downstream 20 nt and upstream 10-40 nt, on the transcript strand
            std::string down, up;
            if (!minus) {
                down = genome.fetch(chrom, static_cast<int64_t>(mode) + 1, 20);
                up = genome.fetch(chrom, static_cast<int64_t>(mode) - 40, 41);
            } else {
                down = polya_revcomp(genome.fetch(chrom, static_cast<int64_t>(mode) - 20, 20));
                up = polya_revcomp(genome.fetch(chrom, static_cast<int64_t>(mode), 41));
            }
            down_a = static_cast<int>(std::count(down.begin(), down.end(), 'A'));
            int run = 0, best_run = 0;
            for (char ch : down) { run = ch == 'A' ? run + 1 : 0; best_run = std::max(best_run, run); }
            ip = down_a >= 12 || best_run >= 8;
            // up holds positions mode-40 .. mode (transcript orientation); PAS 10-40 nt upstream of the site
            if (up.size() >= 41) {
                const std::string win = up.substr(0, 31 + 5);  // starts 40 nt upstream, ends ~10 nt upstream
                for (const char* m : kPAS)
                    if (win.find(m) != std::string::npos) { pas = m; break; }
            }
        }
        if (ip) ++t.n_internal_priming;
        uint32_t lo = polya_site_pos(recs[c.begin].site), hi = polya_site_pos(recs[c.end - 1].site);
        const uint32_t row = t.nrows++;
        const std::string name = gname + ":" + chrom + ":" + (minus ? "-" : "+") + ":" + std::to_string(mode + 1);
        t.names.push_back(name);
        tsv += name + '\t' + gid + '\t' + gname + '\t' + chrom + '\t' + (minus ? "-" : "+") + '\t' +
               std::to_string(mode + 1) + '\t' + std::to_string(lo + 1) + '\t' + std::to_string(hi + 1) + '\t' +
               std::to_string(n_mol) + '\t' + std::to_string(c.end - c.begin) + '\t' + std::to_string(per_bc.size()) +
               '\t' + (have_genome ? std::string(ip ? "1" : "0") : std::string("NA")) + '\t' +
               (down_a >= 0 ? std::to_string(down_a) : std::string("NA")) + '\t' + pas + '\n';
        for (auto& kv : per_bc)
            ent.push_back({kv.first, row, static_cast<uint16_t>(std::min<uint32_t>(kv.second, 65535))});
    }
    std::sort(ent.begin(), ent.end(), [](const E& a, const E& b) { return a.col != b.col ? a.col < b.col : a.row < b.row; });
    t.indptr.assign(static_cast<size_t>(n_barcodes) + 1, 0);
    t.indices.reserve(ent.size());
    t.data.reserve(ent.size());
    for (const auto& e : ent) {
        if (e.col >= n_barcodes) continue;
        t.indptr[e.col + 1]++;
        t.indices.push_back(static_cast<int32_t>(e.row));
        t.data.push_back(e.v);
    }
    for (uint32_t j = 0; j < n_barcodes; ++j) t.indptr[j + 1] += t.indptr[j];
    t.table_tsv = std::move(tsv);
    return t;
}

inline bool write_gz_text(const std::string& path, const std::string& text) {
    gzFile gz = gzopen(path.c_str(), "wb6");
    if (!gz) return false;
    size_t off = 0;
    while (off < text.size()) {
        const unsigned n = static_cast<unsigned>(std::min<size_t>(text.size() - off, 1u << 24));
        if (gzwrite(gz, text.data() + off, n) != static_cast<int>(n)) { gzclose(gz); return false; }
        off += n;
    }
    return gzclose(gz) == Z_OK;
}

}  // namespace singlet
