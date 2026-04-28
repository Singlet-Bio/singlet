#pragma once
// singlet-pileup: vdj_assembly.h — G-VDJ
// De novo CDR3 clonotype assembly from V(D)J-enriched reads.
//
// Strategy (TRUST4-inspired, simplified):
//   1. Index V/J gene segments with 15-mer k-mer hash
//   2. Scan each read for V-gene and J-gene anchors
//   3. Extract CDR3 between Cys104 (V) and Phe/Trp118 (J) anchor codons
//   4. Translate to amino acid; filter non-productive rearrangements
//   5. Group identical CDR3-aa sequences into clonotypes
//
// Output: clonotypes.tsv with columns:
//   barcode, chain, v_gene, d_gene, j_gene, cdr3_nt, cdr3_aa,
//   umi_count, productive, clonotype_id

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Chain type
// ---------------------------------------------------------------------------

enum class ChainType : uint8_t {
    TRA = 0,
    TRB,
    TRG,
    TRD,
    IGH,
    IGK,
    IGL,
    UNKNOWN
};

static inline const char* chain_name(ChainType c) {
    switch (c) {
        case ChainType::TRA:
            return "TRA";
        case ChainType::TRB:
            return "TRB";
        case ChainType::TRG:
            return "TRG";
        case ChainType::TRD:
            return "TRD";
        case ChainType::IGH:
            return "IGH";
        case ChainType::IGK:
            return "IGK";
        case ChainType::IGL:
            return "IGL";
        default:
            return "UNKNOWN";
    }
}

static inline ChainType chain_from_gene_name(const std::string& name) {
    if (name.size() < 3) return ChainType::UNKNOWN;
    // Prefix: TRAV, TRBV, TRGV, TRDV, IGHV, IGKV, IGLV (and J/D variants)
    if (name[0] == 'T') {
        if (name[1] == 'R') {
            if (name[2] == 'A') return ChainType::TRA;
            if (name[2] == 'B') return ChainType::TRB;
            if (name[2] == 'G') return ChainType::TRG;
            if (name[2] == 'D') return ChainType::TRD;
        }
    } else if (name[0] == 'I' && name[1] == 'G') {
        if (name[2] == 'H') return ChainType::IGH;
        if (name[2] == 'K') return ChainType::IGK;
        if (name[2] == 'L') return ChainType::IGL;
    }
    return ChainType::UNKNOWN;
}

// ---------------------------------------------------------------------------
// Codon translation — standard genetic code
// ---------------------------------------------------------------------------

// Encode a single base A=0, C=1, G=2, T=3 (other → 4)
static inline int base_idx(char c) {
    switch (c) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            return 4;
    }
}

// Standard genetic code lookup: codon index = 16*b0 + 4*b1 + b2
// Returns amino acid single-letter code, '*' for stop, 'X' for ambiguous.
static inline char translate_codon(char b0, char b1, char b2) {
    int i0 = base_idx(b0), i1 = base_idx(b1), i2 = base_idx(b2);
    if (i0 > 3 || i1 > 3 || i2 > 3) return 'X';

    // Build from NCBI standard genetic code (codon table 1)
    // Rows: first base T/C/A/G; columns: second base T/C/A/G
    // We use A=0 C=1 G=2 T=3 ordering internally, so remap
    // Standard table indexed [first][second][third] in TCAG order:
    static const char TABLE[4][4][4] = {
        // first = T(3)
        {
            // second = T(3)
            {'F', 'F', 'L', 'L'},  // third = A,C,G,T → T: TTT=F, TTC=F... wait
            // Need to reconsider: use i=ACGT=0123 ordering
            // Actually let me just build the full 64-entry flat table
            {' ', ' ', ' ', ' '},
            {' ', ' ', ' ', ' '},
            {' ', ' ', ' ', ' '}},
        {{' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}},
        {{' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}},
        {{' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}, {' ', ' ', ' ', ' '}},
    };
    (void)TABLE;  // use flat array below instead

    // idx = 16*i0 + 4*i1 + i2  (A=0,C=1,G=2,T=3)
    static const char CODON[64] = {
        //      _A   _C   _G   _T
        /* AA */ 'K',
        'N',
        'K',
        'N',  // AA*
        /* AC */ 'T',
        'T',
        'T',
        'T',  // AC*
        /* AG */ 'R',
        'S',
        'R',
        'S',  // AG*
        /* AT */ 'I',
        'I',
        'M',
        'I',  // AT*
        /* CA */ 'Q',
        'H',
        'Q',
        'H',  // CA*
        /* CC */ 'P',
        'P',
        'P',
        'P',  // CC*
        /* CG */ 'R',
        'R',
        'R',
        'R',  // CG*
        /* CT */ 'L',
        'L',
        'L',
        'L',  // CT*
        /* GA */ 'E',
        'D',
        'E',
        'D',  // GA*
        /* GC */ 'A',
        'A',
        'A',
        'A',  // GC*
        /* GG */ 'G',
        'G',
        'G',
        'G',  // GG*
        /* GT */ 'V',
        'V',
        'V',
        'V',  // GT*
        /* TA */ '*',
        'Y',
        '*',
        'Y',  // TA*
        /* TC */ 'S',
        'S',
        'S',
        'S',  // TC*
        /* TG */ '*',
        'C',
        'W',
        'C',  // TG*
        /* TT */ 'L',
        'F',
        'L',
        'F',  // TT*
    };
    return CODON[16 * i0 + 4 * i1 + i2];
}

/// Translate a nucleotide sequence to amino acids.
/// Trailing incomplete codon is ignored. Stop codons become '*'.
inline std::string translate(const std::string& nt) {
    std::string aa;
    aa.reserve(nt.size() / 3 + 1);
    for (size_t i = 0; i + 2 < nt.size(); i += 3)
        aa += translate_codon(nt[i], nt[i + 1], nt[i + 2]);
    return aa;
}

/// Return true if CDR3 aa is productive:
///   starts with C, ends with F or W, no internal stop codon, len >= 5
inline bool is_productive(const std::string& cdr3_aa) {
    if (cdr3_aa.size() < 5) return false;
    if (cdr3_aa.front() != 'C') return false;
    char last = cdr3_aa.back();
    if (last != 'F' && last != 'W') return false;
    // No stop codons in the body (allow stop at end is already excluded above)
    for (char c : cdr3_aa)
        if (c == '*') return false;
    return true;
}

// ---------------------------------------------------------------------------
// V/J reference database
// ---------------------------------------------------------------------------

struct VJReference {
    struct Gene {
        std::string name;
        std::string sequence;
        ChainType chain;
        bool is_v;            // true = V gene, false = J gene
        int32_t cdr3_anchor;  // offset of anchor codon start within sequence
                              // Cys for V, Phe/Trp for J; -1 = inferred
    };

    std::vector<Gene> genes;

    // k-mer → list of gene indices
    std::unordered_map<uint64_t, std::vector<uint32_t>> kmer_index;
    uint32_t k = 15;

    // -----------------------------------------------------------------------
    // Add a gene directly (used by tests and synthetic references)
    // -----------------------------------------------------------------------
    void add_gene(const std::string& name, const std::string& seq,
                  bool is_v, int32_t anchor = -1) {
        Gene g;
        g.name = name;
        g.sequence = seq;
        g.chain = chain_from_gene_name(name);
        g.is_v = is_v;
        g.cdr3_anchor = anchor;
        genes.push_back(std::move(g));
    }

    // -----------------------------------------------------------------------
    // Load from simple two-column TSV: name<TAB>sequence (no FASTA needed)
    // -----------------------------------------------------------------------
    bool load_from_tsv(const std::string& path, bool is_v) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string name = line.substr(0, tab);
            std::string seq = line.substr(tab + 1);
            add_gene(name, seq, is_v);
        }
        return !genes.empty();
    }

    // -----------------------------------------------------------------------
    // Load from FASTA file (>name\nsequence\n...)
    // -----------------------------------------------------------------------
    bool load_from_fasta(const std::string& v_fasta, const std::string& j_fasta) {
        auto parse = [&](const std::string& path, bool is_v) {
            std::ifstream f(path);
            if (!f) return false;
            std::string line, name, seq;
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                if (line[0] == '>') {
                    if (!name.empty()) add_gene(name, seq, is_v);
                    name = line.substr(1);
                    // Strip whitespace/description after space
                    auto sp = name.find(' ');
                    if (sp != std::string::npos) name = name.substr(0, sp);
                    seq.clear();
                } else {
                    seq += line;
                }
            }
            if (!name.empty()) add_gene(name, seq, is_v);
            return true;
        };
        bool ok = parse(v_fasta, true);
        ok &= parse(j_fasta, false);
        return ok;
    }

    // -----------------------------------------------------------------------
    // Build k-mer index over all gene sequences
    // -----------------------------------------------------------------------
    void build_index(uint32_t kk = 15) {
        k = kk;
        kmer_index.clear();
        for (uint32_t gi = 0; gi < (uint32_t)genes.size(); ++gi) {
            const std::string& seq = genes[gi].sequence;
            if (seq.size() < k) continue;
            for (size_t i = 0; i + k <= seq.size(); ++i) {
                uint64_t h = hash_kmer(seq.data() + i, k);
                if (h == (uint64_t)-1) continue;  // ambiguous bases
                kmer_index[h].push_back(gi);
            }
        }
    }

    // FNV-like 2-bit k-mer hash; returns -1 if ambiguous base found
    static uint64_t hash_kmer(const char* s, uint32_t kk) {
        uint64_t h = 0;
        for (uint32_t i = 0; i < kk; ++i) {
            int b = base_idx(s[i]);
            if (b > 3) return (uint64_t)-1;
            h = (h << 2) | (uint64_t)b;
        }
        return h;
    }
};

// ---------------------------------------------------------------------------
// CDR3 extraction from a single read
// ---------------------------------------------------------------------------

struct CDR3Extraction {
    std::string cdr3_nt;
    uint32_t v_gene_idx = (uint32_t)-1;
    uint32_t j_gene_idx = (uint32_t)-1;
    int score = 0;  // k-mer hit count (quality proxy)
};

/// Scan `read_seq` for best V and J k-mer matches, then extract CDR3.
/// Returns a CDR3Extraction; cdr3_nt is empty if no CDR3 found.
inline CDR3Extraction extract_cdr3(const std::string& read_seq,
                                   const VJReference& ref) {
    CDR3Extraction result;
    const size_t rlen = read_seq.size();
    if (rlen < ref.k) return result;

    // Count k-mer hits per gene
    std::unordered_map<uint32_t, int> v_hits, j_hits;

    for (size_t i = 0; i + ref.k <= rlen; ++i) {
        uint64_t h = VJReference::hash_kmer(read_seq.data() + i, ref.k);
        if (h == (uint64_t)-1) continue;
        auto it = ref.kmer_index.find(h);
        if (it == ref.kmer_index.end()) continue;
        for (uint32_t gi : it->second) {
            if (ref.genes[gi].is_v)
                v_hits[gi]++;
            else
                j_hits[gi]++;
        }
    }

    // Best V gene
    int best_v_score = 0;
    for (auto& [gi, cnt] : v_hits)
        if (cnt > best_v_score) {
            best_v_score = cnt;
            result.v_gene_idx = gi;
        }

    // Best J gene
    int best_j_score = 0;
    for (auto& [gi, cnt] : j_hits)
        if (cnt > best_j_score) {
            best_j_score = cnt;
            result.j_gene_idx = gi;
        }

    if (result.v_gene_idx == (uint32_t)-1 || result.j_gene_idx == (uint32_t)-1)
        return result;

    result.score = best_v_score + best_j_score;

    // --------------------------------------------------------------------
    // Locate CDR3 boundaries in the read
    //
    // V anchor: find the Cys codon (TGT or TGC) in the read that best
    //   corresponds to the anchor position in the V gene. We do this by
    //   finding the last TGT/TGC in the V-gene-matching region of the read.
    // J anchor: find the first Phe (TTT/TTC) or Trp (TGG) codon in the
    //   J-gene-matching region → CDR3 end (inclusive).
    // --------------------------------------------------------------------

    // Find approximate V-match region start in read (last hit position)
    // Simple heuristic: scan all V k-mers in read, find rightmost hit
    size_t v_region_end = 0;
    {
        const std::string& vseq = ref.genes[result.v_gene_idx].sequence;
        for (size_t i = 0; i + ref.k <= rlen; ++i) {
            uint64_t rh = VJReference::hash_kmer(read_seq.data() + i, ref.k);
            if (rh == (uint64_t)-1) continue;
            // Check if this k-mer appears in the V gene
            for (size_t j = 0; j + ref.k <= vseq.size(); ++j) {
                if (VJReference::hash_kmer(vseq.data() + j, ref.k) == rh) {
                    if (i + ref.k > v_region_end) v_region_end = i + ref.k;
                    break;
                }
            }
        }
    }
    if (v_region_end == 0) return result;  // no clear V match

    // Find CDR3 start: last TGT or TGC at or before v_region_end (codon-aligned scan)
    // We scan all positions (not just codon-aligned) since frame may shift
    int32_t cdr3_start = -1;
    // Scan backward from v_region_end
    int32_t scan_end = (int32_t)std::min(v_region_end, rlen - 2);
    for (int32_t p = scan_end; p >= 0; --p) {
        char b0 = read_seq[p], b1 = read_seq[p + 1], b2 = read_seq[p + 2];
        if (((b0 == 'T' || b0 == 't') && (b1 == 'G' || b1 == 'g') &&
             ((b2 == 'T' || b2 == 't') || (b2 == 'C' || b2 == 'c')))) {
            cdr3_start = p;
            break;
        }
    }
    if (cdr3_start < 0) return result;

    // Find approximate J-match region in read (first hit position)
    size_t j_region_start = rlen;
    {
        const std::string& jseq = ref.genes[result.j_gene_idx].sequence;
        for (size_t i = 0; i + ref.k <= rlen; ++i) {
            uint64_t rh = VJReference::hash_kmer(read_seq.data() + i, ref.k);
            if (rh == (uint64_t)-1) continue;
            for (size_t j = 0; j + ref.k <= jseq.size(); ++j) {
                if (VJReference::hash_kmer(jseq.data() + j, ref.k) == rh) {
                    if (i < j_region_start) j_region_start = i;
                    break;
                }
            }
        }
    }
    if (j_region_start == rlen) return result;
    if (j_region_start <= (size_t)cdr3_start) return result;

    // Find CDR3 end: first Phe (TTT/TTC) or Trp (TGG) codon at or after j_region_start
    int32_t cdr3_end = -1;  // inclusive end of last base of anchor codon
    for (size_t p = j_region_start; p + 2 < rlen; ++p) {
        char b0 = read_seq[p], b1 = read_seq[p + 1], b2 = read_seq[p + 2];
        bool is_phe = ((b0 == 'T' || b0 == 't') && (b1 == 'T' || b1 == 't') &&
                       ((b2 == 'T' || b2 == 't') || (b2 == 'C' || b2 == 'c')));
        bool is_trp = ((b0 == 'T' || b0 == 't') && (b1 == 'G' || b1 == 'g') && (b2 == 'G' || b2 == 'g'));
        if (is_phe || is_trp) {
            cdr3_end = (int32_t)(p + 3);  // exclusive end
            break;
        }
    }
    if (cdr3_end < 0) return result;
    if (cdr3_end <= cdr3_start + 3) return result;  // too short

    // Extract CDR3 nt (from Cys codon to end of Phe/Trp codon)
    result.cdr3_nt = read_seq.substr(cdr3_start, cdr3_end - cdr3_start);
    return result;
}

// ---------------------------------------------------------------------------
// Result structures
// ---------------------------------------------------------------------------

struct CDR3Result {
    std::string cdr3_nt;
    std::string cdr3_aa;
    std::string v_gene;
    std::string d_gene;  // empty for TRA/IGK/IGL
    std::string j_gene;
    ChainType chain = ChainType::UNKNOWN;
    uint32_t umi_count = 1;
    bool productive = false;
};

struct CellVDJ {
    uint32_t bc_idx;
    std::vector<CDR3Result> chains;
};

struct VDJAssemblyResult {
    std::vector<CellVDJ> cells;
    uint32_t n_cells_with_cdr3 = 0;
    uint32_t n_productive = 0;
    uint32_t n_chains[8] = {};
};

// ---------------------------------------------------------------------------
// Clonotype assembly
// ---------------------------------------------------------------------------

/// Assemble clonotypes from per-cell CDR3 extractions.
/// extractions: vector of (bc_idx, CDR3Extraction)
/// Cells sharing identical CDR3 nucleotide sequence get the same clonotype_id.
/// (Amino acid grouping via a secondary map; nt used for exact grouping.)
inline VDJAssemblyResult assemble_clonotypes(
    const std::vector<std::pair<uint32_t, CDR3Extraction>>& extractions,
    const VJReference& ref) {
    VDJAssemblyResult result;

    // Group by barcode to deduplicate & accumulate umi counts
    // For simplicity: one CDR3 per extraction (no multi-UMI merging in this layer)
    std::unordered_map<uint32_t, CellVDJ> cell_map;

    for (auto& [bc, ext] : extractions) {
        if (ext.cdr3_nt.empty()) continue;

        CDR3Result cr;
        cr.cdr3_nt = ext.cdr3_nt;
        cr.cdr3_aa = translate(ext.cdr3_nt);
        cr.productive = is_productive(cr.cdr3_aa);
        cr.umi_count = 1;

        if (ext.v_gene_idx < (uint32_t)ref.genes.size()) {
            cr.v_gene = ref.genes[ext.v_gene_idx].name;
            cr.chain = ref.genes[ext.v_gene_idx].chain;
        }
        if (ext.j_gene_idx < (uint32_t)ref.genes.size())
            cr.j_gene = ref.genes[ext.j_gene_idx].name;

        auto& cell = cell_map[bc];
        cell.bc_idx = bc;

        // Check if this CDR3 nt already seen for this cell (merge UMI counts)
        bool found = false;
        for (auto& existing : cell.chains) {
            if (existing.cdr3_nt == cr.cdr3_nt) {
                existing.umi_count++;
                found = true;
                break;
            }
        }
        if (!found) cell.chains.push_back(std::move(cr));
    }

    // Build result
    for (auto& [bc, cell] : cell_map) {
        if (cell.chains.empty()) continue;
        result.cells.push_back(std::move(cell));
    }

    // Sort cells by bc_idx for determinism
    std::sort(result.cells.begin(), result.cells.end(),
              [](const CellVDJ& a, const CellVDJ& b) { return a.bc_idx < b.bc_idx; });

    // Count stats
    for (auto& cell : result.cells) {
        result.n_cells_with_cdr3++;
        for (auto& cr : cell.chains) {
            if (cr.productive) result.n_productive++;
            int ci = (int)cr.chain;
            if (ci >= 0 && ci < 8) result.n_chains[ci]++;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// TSV output
// ---------------------------------------------------------------------------

/// Write clonotype TSV.
/// clonotype_id assigned by grouping cells with identical CDR3 aa sequences.
inline bool write_clonotype_tsv(const std::string& path,
                                const VDJAssemblyResult& result,
                                const std::vector<std::string>& barcodes) {
    std::ofstream f(path);
    if (!f) return false;

    f << "barcode\tchain\tv_gene\td_gene\tj_gene\tcdr3_nt\tcdr3_aa"
      << "\tumi_count\tproductive\tclonotype_id\n";

    // Build clonotype_id map: CDR3 aa → id
    std::unordered_map<std::string, uint32_t> clono_map;
    uint32_t next_id = 1;

    for (auto& cell : result.cells) {
        const std::string& bc_str =
            (cell.bc_idx < barcodes.size()) ? barcodes[cell.bc_idx]
                                            : std::to_string(cell.bc_idx);
        for (auto& cr : cell.chains) {
            auto it = clono_map.find(cr.cdr3_aa);
            if (it == clono_map.end()) {
                clono_map[cr.cdr3_aa] = next_id++;
                it = clono_map.find(cr.cdr3_aa);
            }
            f << bc_str << '\t'
              << chain_name(cr.chain) << '\t'
              << cr.v_gene << '\t'
              << cr.d_gene << '\t'
              << cr.j_gene << '\t'
              << cr.cdr3_nt << '\t'
              << cr.cdr3_aa << '\t'
              << cr.umi_count << '\t'
              << (cr.productive ? "true" : "false") << '\t'
              << it->second << '\n';
        }
    }
    return true;
}

}  // namespace singlet
