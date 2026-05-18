// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: transcript_compat.h
// G-TXLEVEL: Transcript-Level Equivalence Class Counting (TCC)
//
// Computes a TCC matrix (equivalence_classes × cells) after pileup completes.
// Does NOT modify the hot path — uses the already-computed exon-level CSC.
//
// Pipeline:
//   1. TranscriptModel: re-parse GTF to map gene_model exon_idx → transcript set
//   2. TCCBuilder: accumulate (bc, exon, count) from exon_csc → EC counts
//   3. TCCResult: sparse EC × cell matrix in CSC format
//   4. write_tcc_*: bustools-compatible output files
//
// Bustools EC format: tcc_ec.tsv (one row per EC, tab-separated tx indices)
// Matrix format:      tcc_matrix.mtx (MatrixMarket, EC × cells)
// Transcript table:   tcc_transcripts.tsv (transcript_id, gene_id)

#include <zlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "gene_model.h"

namespace singlet {

// ── Data types ────────────────────────────────────────────────────────────────

struct EquivClassDef {
    std::vector<uint32_t> transcript_ids;  // sorted transcript indices
};

struct TCCResult {
    // EC definitions — indexed by ec_idx
    std::vector<EquivClassDef> equiv_classes;

    // TCC matrix in CSC format (n_ec × n_cells)
    std::vector<int32_t> indptr;   // size: n_cells + 1
    std::vector<int32_t> indices;  // row (ec) indices
    std::vector<uint32_t> data;    // UMI counts

    uint32_t n_ec = 0;
    uint32_t n_cells = 0;

    // Transcript metadata (indexed by tx_idx)
    std::vector<std::string> tx_ids;    // transcript_id strings
    std::vector<std::string> tx_genes;  // gene_id strings
};

// ── TranscriptModel ───────────────────────────────────────────────────────────
// Parses the same GTF file used for the pileup and builds a mapping:
//   gene_model exon_idx  →  sorted list of transcript indices
//
// A transcript is "compatible" with a gene_model exon if the transcript has
// at least one raw GTF exon overlapping that merged exon interval.

class TranscriptModel {
   public:
    TranscriptModel() = default;

    // Load transcript annotations from GTF (plain or gzipped).
    // gm must be the same GeneModel used to build the pileup.
    // Returns false on error.
    bool load_gtf(const std::string& path, const GeneModel& gm) {
        bool is_gz = path.size() >= 3 && path.substr(path.size() - 3) == ".gz";
        gzFile gz = nullptr;
        std::ifstream plain;

        if (is_gz) {
            gz = gzopen(path.c_str(), "r");
            if (!gz) {
                std::cerr << "[transcript_model] Cannot open: " << path << "\n";
                return false;
            }
            gzbuffer(gz, 256 * 1024);
        } else {
            plain.open(path);
            if (!plain.is_open()) {
                std::cerr << "[transcript_model] Cannot open: " << path << "\n";
                return false;
            }
        }

        // Build gene_id → gene_idx lookup
        std::unordered_map<std::string, uint32_t> gene_id_to_idx;
        const auto& gids = gm.gene_ids();
        for (uint32_t i = 0; i < static_cast<uint32_t>(gids.size()); ++i) {
            gene_id_to_idx[gids[i]] = i;
        }

        // Build exon bounds per gene_model exon_idx:
        //   exon_gene_[i] = gene_idx, exon_s_[i] = start, exon_e_[i] = end
        uint32_t ne = gm.n_exons();
        std::vector<uint32_t> exon_gene(ne);
        std::vector<int32_t> exon_s(ne), exon_e(ne);
        for (uint32_t i = 0; i < ne; ++i) {
            exon_gene[i] = gm.exon_to_gene(i);
            exon_s[i] = gm.exon_start(i);
            exon_e[i] = gm.exon_end(i);
        }

        // Build per-gene list of exon indices for fast lookup
        uint32_t ng = gm.n_genes();
        std::vector<std::vector<uint32_t>> gene_exon_list(ng);
        for (uint32_t i = 0; i < ne; ++i) {
            if (exon_gene[i] < ng) {
                gene_exon_list[exon_gene[i]].push_back(i);
            }
        }

        // ── Parse GTF exon records ─────────────────────────────────────────────
        // We collect: gene_id × transcript_id → list of (exon intervals that
        // overlap a gene_model exon).  For each gene_model exon hit, we record
        // the transcript index that should be compatible.

        // First pass: register all transcripts encountered in GTF exon lines.
        // tx_key_to_idx_: (gene_id + "|" + transcript_id) → tx_idx
        // tx_ids_, tx_genes_: metadata

        // We use a two-pass approach to avoid repeated string hashing:
        //   Pass 1: collect raw exon rows (chrom unused; we match by gene_id only)
        //   Then: for each gm exon, find overlapping raw exons per gene → tx set

        struct RawTxExon {
            std::string gene_id;
            std::string tx_id;
            int32_t start;
            int32_t end;
        };
        std::vector<RawTxExon> raw;
        raw.reserve(256 * 1024);

        auto read_line = [&](std::string& line) -> bool {
            if (gz) {
                char buf[8192];
                line.clear();
                while (gzgets(gz, buf, sizeof(buf))) {
                    line += buf;
                    if (!line.empty() && line.back() == '\n') {
                        line.pop_back();
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        return true;
                    }
                }
                return !line.empty();
            } else {
                return static_cast<bool>(std::getline(plain, line));
            }
        };

        std::string line;
        while (read_line(line)) {
            if (line.empty() || line[0] == '#') continue;

            // Extract field 2 (feature type)
            const char* p = line.c_str();
            const char* f[9];
            int fi = 0;
            f[0] = p;
            while (*p && fi < 8) {
                if (*p == '\t') {
                    fi++;
                    f[fi] = p + 1;
                }
                p++;
            }
            if (fi < 8) continue;

            const char* feat = f[2];
            const char* feat_end = f[3] - 1;
            if (feat_end - feat != 4 || strncmp(feat, "exon", 4) != 0) continue;

            int32_t start = atoi(f[3]) - 1;  // GTF 1-based → 0-based
            int32_t end = atoi(f[4]);        // half-open

            const char* attrs = f[8];
            std::string gene_id = extract_attr(attrs, "gene_id");
            std::string tx_id = extract_attr(attrs, "transcript_id");
            if (gene_id.empty() || tx_id.empty()) continue;

            raw.push_back({std::move(gene_id), std::move(tx_id), start, end});
        }

        if (gz) gzclose(gz);

        // ── Register unique transcripts ───────────────────────────────────────
        // Use "gene_id|tx_id" key so same tx_id in different genes doesn't collide
        for (auto& r : raw) {
            std::string key = r.gene_id + "|" + r.tx_id;
            if (tx_key_to_idx_.find(key) == tx_key_to_idx_.end()) {
                uint32_t idx = static_cast<uint32_t>(tx_ids_.size());
                tx_key_to_idx_[key] = idx;
                tx_ids_.push_back(r.tx_id);
                tx_genes_.push_back(r.gene_id);
            }
        }

        std::cerr << "[transcript_model] Registered " << tx_ids_.size()
                  << " transcripts from " << raw.size() << " exon records\n";

        // ── Build exon_idx → compatible transcript set ────────────────────────
        // Group raw exon records by gene_id
        std::unordered_map<std::string, std::vector<size_t>> gene_raw_idx;
        gene_raw_idx.reserve(ng * 2);
        for (size_t i = 0; i < raw.size(); ++i) {
            gene_raw_idx[raw[i].gene_id].push_back(i);
        }

        exon_tx_.assign(ne, {});

        for (uint32_t gi = 0; gi < ng; ++gi) {
            const std::string& gid = gids[gi];
            auto it = gene_raw_idx.find(gid);
            if (it == gene_raw_idx.end()) continue;

            const auto& ridx_list = it->second;
            const auto& gm_exons = gene_exon_list[gi];

            // For each gene_model exon interval, find all transcripts with
            // an overlapping raw exon in this gene.
            for (uint32_t eidx : gm_exons) {
                int32_t es = exon_s[eidx];
                int32_t ee = exon_e[eidx];

                std::vector<uint32_t> compat;
                for (size_t ri : ridx_list) {
                    const auto& r = raw[ri];
                    // Overlap test: r.start < ee && r.end > es
                    if (r.start < ee && r.end > es) {
                        std::string key = r.gene_id + "|" + r.tx_id;
                        auto tit = tx_key_to_idx_.find(key);
                        if (tit != tx_key_to_idx_.end()) {
                            compat.push_back(tit->second);
                        }
                    }
                }
                // Deduplicate and sort
                std::sort(compat.begin(), compat.end());
                compat.erase(std::unique(compat.begin(), compat.end()), compat.end());
                exon_tx_[eidx] = std::move(compat);
            }
        }

        std::cerr << "[transcript_model] Built exon→transcript map for "
                  << ne << " exons\n";
        return true;
    }

    // Build the model programmatically (for unit tests).
    // transcripts: list of (tx_id, gene_id)
    // exon_assignments: for each gene_model exon_idx, the sorted list of tx_indices
    void build_from_data(
        const std::vector<std::pair<std::string, std::string>>& transcripts,
        const std::vector<std::vector<uint32_t>>& exon_assignments) {
        tx_ids_.clear();
        tx_genes_.clear();
        tx_key_to_idx_.clear();
        exon_tx_.clear();
        for (const auto& [tid, gid] : transcripts) {
            std::string key = gid + "|" + tid;
            uint32_t idx = static_cast<uint32_t>(tx_ids_.size());
            tx_key_to_idx_[key] = idx;
            tx_ids_.push_back(tid);
            tx_genes_.push_back(gid);
        }
        exon_tx_ = exon_assignments;
    }

    uint32_t n_transcripts() const { return static_cast<uint32_t>(tx_ids_.size()); }

    const std::string& tx_id(uint32_t i) const {
        static const std::string empty;
        return i < tx_ids_.size() ? tx_ids_[i] : empty;
    }
    const std::string& tx_gene(uint32_t i) const {
        static const std::string empty;
        return i < tx_genes_.size() ? tx_genes_[i] : empty;
    }

    // Returns the sorted list of transcript indices for a gene_model exon_idx.
    // Returns an empty vector if the exon has no transcript annotations.
    const std::vector<uint32_t>& exon_to_transcripts(uint32_t exon_idx) const {
        static const std::vector<uint32_t> empty;
        if (exon_idx >= exon_tx_.size()) return empty;
        return exon_tx_[exon_idx];
    }

   private:
    std::vector<std::string> tx_ids_;
    std::vector<std::string> tx_genes_;
    std::unordered_map<std::string, uint32_t> tx_key_to_idx_;
    // exon_tx_[exon_idx] = sorted list of transcript indices compatible
    std::vector<std::vector<uint32_t>> exon_tx_;

    static std::string extract_attr(const char* attrs, const char* key) {
        std::string search = std::string(key) + " \"";
        const char* p = strstr(attrs, search.c_str());
        if (!p) return {};
        p += search.size();
        const char* q = strchr(p, '"');
        if (!q) return {};
        return std::string(p, q);
    }
};

// ── TCCBuilder ────────────────────────────────────────────────────────────────
// Accumulates (bc, exon_idx, count) triples and produces a TCCResult.
//
// Input is the exon-level CSC (already UMI-deduplicated counts from pileup).
// Each (bc, exon) entry with count > 0 contributes `count` reads to the EC
// defined by exon_to_transcripts(exon_idx).

class TCCBuilder {
   public:
    // Add a single (bc, exon, count) observation.
    // tm provides the exon → transcript set mapping.
    void add_exon_count(uint32_t bc_idx, uint32_t exon_idx, uint32_t count,
                        const TranscriptModel& tm) {
        const auto& tx_ids = tm.exon_to_transcripts(exon_idx);
        if (tx_ids.empty()) return;  // no transcript annotation for this exon
        uint32_t ec_idx = get_or_create_ec(tx_ids);
        // Accumulate into per-cell sparse map
        if (bc_idx >= per_cell_.size()) per_cell_.resize(bc_idx + 1);
        per_cell_[bc_idx][ec_idx] += count;
    }

    // Convenience: add all non-zero entries from an exon-level CSC.
    // exon_indptr/indices/data: CSC format (n_exons × n_cells)
    void add_from_exon_csc(
        const std::vector<int32_t>& exon_indptr,
        const std::vector<int32_t>& exon_indices,
        const std::vector<uint32_t>& exon_data,
        uint32_t n_cells,
        const TranscriptModel& tm) {
        for (uint32_t bc = 0; bc < n_cells; ++bc) {
            for (int32_t k = exon_indptr[bc]; k < exon_indptr[bc + 1]; ++k) {
                add_exon_count(bc, static_cast<uint32_t>(exon_indices[k]),
                               exon_data[k], tm);
            }
        }
    }

    // Build the final TCC result.
    TCCResult build(uint32_t n_cells, const TranscriptModel& tm) const {
        TCCResult result;
        result.n_ec = static_cast<uint32_t>(ec_registry_.size());
        result.n_cells = n_cells;

        // Populate EC definitions
        result.equiv_classes.resize(result.n_ec);
        for (const auto& [tx_vec, eidx] : ec_registry_) {
            result.equiv_classes[eidx].transcript_ids = tx_vec;
        }

        // Build CSC matrix
        result.indptr.assign(n_cells + 1, 0);
        result.indices.clear();
        result.data.clear();

        for (uint32_t bc = 0; bc < n_cells; ++bc) {
            if (bc < per_cell_.size() && !per_cell_[bc].empty()) {
                // Emit in EC-index order (map is sorted)
                for (const auto& [eidx, cnt] : per_cell_[bc]) {
                    result.indices.push_back(static_cast<int32_t>(eidx));
                    result.data.push_back(cnt);
                }
            }
            result.indptr[bc + 1] = static_cast<int32_t>(result.data.size());
        }

        // Populate transcript metadata
        uint32_t nt = tm.n_transcripts();
        result.tx_ids.resize(nt);
        result.tx_genes.resize(nt);
        for (uint32_t i = 0; i < nt; ++i) {
            result.tx_ids[i] = tm.tx_id(i);
            result.tx_genes[i] = tm.tx_gene(i);
        }

        return result;
    }

    void reset() {
        per_cell_.clear();
        ec_registry_.clear();
    }

   private:
    // EC registry: sorted tx_id vector → ec_index (insertion order)
    std::map<std::vector<uint32_t>, uint32_t> ec_registry_;
    // Sparse per-cell accumulator: per_cell_[bc][ec_idx] = count
    std::vector<std::map<uint32_t, uint32_t>> per_cell_;

    uint32_t get_or_create_ec(const std::vector<uint32_t>& tx_ids) {
        auto it = ec_registry_.find(tx_ids);
        if (it != ec_registry_.end()) return it->second;
        uint32_t idx = static_cast<uint32_t>(ec_registry_.size());
        ec_registry_.emplace(tx_ids, idx);
        return idx;
    }
};

// ── Output writers ────────────────────────────────────────────────────────────

// Write tcc_matrix.mtx (MatrixMarket, 1-based, EC × cells)
inline bool write_tcc_matrix(const std::string& path, const TCCResult& tcc) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[tcc] Cannot open for write: " << path << "\n";
        return false;
    }
    // Count non-zeros
    size_t nnz = tcc.data.size();
    f << "%%MatrixMarket matrix coordinate integer general\n";
    f << "%\n";
    f << tcc.n_ec << " " << tcc.n_cells << " " << nnz << "\n";
    for (uint32_t bc = 0; bc < tcc.n_cells; ++bc) {
        for (int32_t k = tcc.indptr[bc]; k < tcc.indptr[bc + 1]; ++k) {
            // MatrixMarket: 1-based row (ec) and col (cell)
            f << (tcc.indices[k] + 1) << "\t" << (bc + 1) << "\t" << tcc.data[k] << "\n";
        }
    }
    return true;
}

// Write tcc_ec.tsv: one line per EC, tab-separated 0-based transcript indices
// (bustools-compatible format)
inline bool write_ec_table(const std::string& path, const TCCResult& tcc) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[tcc] Cannot open for write: " << path << "\n";
        return false;
    }
    for (uint32_t ec = 0; ec < tcc.n_ec; ++ec) {
        const auto& tx_ids = tcc.equiv_classes[ec].transcript_ids;
        for (size_t i = 0; i < tx_ids.size(); ++i) {
            if (i > 0) f << '\t';
            f << tx_ids[i];
        }
        f << '\n';
    }
    return true;
}

// Write tcc_transcripts.tsv: one line per transcript
// Columns: transcript_id, gene_id
inline bool write_transcript_table(const std::string& path, const TCCResult& tcc) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[tcc] Cannot open for write: " << path << "\n";
        return false;
    }
    for (size_t i = 0; i < tcc.tx_ids.size(); ++i) {
        f << tcc.tx_ids[i] << "\t" << tcc.tx_genes[i] << "\n";
    }
    return true;
}

}  // namespace singlet
