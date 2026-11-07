// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: multimap_record.h
// The multi-mapping record beside the gene counts (D-101, D-104; docs/pipeline-outputs.md "Multi-mapping record").
//
// The gene count matrices hold molecules whose reads are assigned to one gene. A read compatible with two genes (a multimapper whose
// alignments touch two genes, a uniquely mapped read on exons of two overlapping genes, or with intron_rule::kGeneFullSet a read whose
// gene-body set, exon genes plus intron genes, has two genes) is left out of them, as STARsolo's Unique counts leave it out. This record keeps those reads as molecules so that retrieval can add them back (EM or Uniform, as STARsolo
// --soloMultiMappers does) or leave them out. The default is to leave them out (D-104).
//
// Input: the engine's ambiguous reads (em_rescue::AmbigRead: barcode column, UMI, the two genes). The CPU pileup collects them in its
// multimapper buffer (resolve_mm, the cross-worker merge) and on unique multi-gene primaries; the GPU pileup returns the same reads
// (PileupOutput::ambig). This function alone turns them into the record, so the CPU and GPU paths write the same bytes whenever they
// hand it the same multiset of reads (the order is irrelevant: everything is sorted).
//
// Per (cell, UMI) the record keeps the genes present in every read of that UMI (STARsolo's umiGenes in collapseUMIperCB: exact UMIs,
// no 1-mismatch collapse, the intersection of the reads' gene sets). A UMI left with two genes is a two-gene molecule, one left with a
// single gene is a molecule of that gene, and an empty intersection is dropped. Molecules are counted per class (the gene set) and cell.
// A (cell, UMI) that also has a read in the gene counts (`counted`, sorted cell << 32 | packed UMI) is left out: STARsolo pools all of a
// UMI's reads, so one uniquely assigned read makes the whole UMI a unique molecule, which the gene counts already hold.
//
// Output: a classes x barcodes integer matrix (rows sorted by gene index pair; the columns are the sample's barcode columns, as every
// other count matrix). Row names are the class's genes as "<gene_id>_<gene_name>", two genes joined by "|". Reads whose gene set has
// more than two genes, or whose genes were not all noted (a multi-gene secondary alignment), are not in the input (MultiHitEntry::partial,
// body_more; the GPU's AA::pmore).
#include "em_rescue.h"
#include "umi_dedup.h"
#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace singlet {
namespace multimap {

struct Record {
    std::vector<int32_t> indptr;    // column pointers (ncols + 1)
    std::vector<int32_t> indices;   // class rows
    std::vector<uint32_t> data;     // molecules
    uint32_t nrows = 0;             // classes
    uint32_t ncols = 0;             // barcodes
    std::vector<std::string> names; // one per class
    uint64_t reads = 0;             // input reads used
    uint64_t molecules = 0;         // (cell, UMI) with a non-empty gene set
    uint64_t dropped_umis = 0;      // (cell, UMI) whose reads share no gene
    uint64_t counted_umis = 0;      // (cell, UMI) left out because the gene counts hold a read of it
};

inline Record build(const std::vector<em_rescue::AmbigRead>& in, uint32_t n_barcodes, const std::vector<std::string>& gene_ids,
                    const std::vector<std::string>& gene_names, const std::vector<uint64_t>* counted = nullptr) {
    constexpr uint32_t NONE = UINT32_MAX;
    const uint32_t ng = static_cast<uint32_t>(gene_ids.size());
    struct R { uint32_t bc; uint64_t umi; uint32_t ga, gb; };
    std::vector<R> rs;
    rs.reserve(in.size());
    for (const auto& a : in) {
        if (a.bc_idx < 0 || static_cast<uint32_t>(a.bc_idx) >= n_barcodes || a.umi_len == 0) continue;
        uint32_t ga = a.gene_a, gb = a.gene_b;
        if (ga > gb) std::swap(ga, gb);
        if (gb >= ng || ga == gb) continue;
        rs.push_back(R{static_cast<uint32_t>(a.bc_idx), umi_pack_2bit(a.umi, a.umi_len), ga, gb});
    }
    std::sort(rs.begin(), rs.end(), [](const R& x, const R& y) { return std::tie(x.bc, x.umi, x.ga, x.gb) < std::tie(y.bc, y.umi, y.ga, y.gb); });

    Record out;
    out.ncols = n_barcodes;
    out.reads = rs.size();
    // (class genes, cell) per molecule
    struct M { uint32_t g1, g2, bc; };
    std::vector<M> ms;
    for (size_t i = 0; i < rs.size();) {
        size_t j = i;
        uint32_t g1 = rs[i].ga, g2 = rs[i].gb;   // the intersection so far (NONE = gone)
        while (j < rs.size() && rs[j].bc == rs[i].bc && rs[j].umi == rs[i].umi) {
            const uint32_t a = rs[j].ga, b = rs[j].gb;
            if (g1 != NONE && g1 != a && g1 != b) g1 = NONE;
            if (g2 != NONE && g2 != a && g2 != b) g2 = NONE;
            ++j;
        }
        if (g1 == NONE) { g1 = g2; g2 = NONE; }
        const uint64_t key = (static_cast<uint64_t>(rs[i].bc) << 32) | (rs[i].umi & 0xffffffffULL);
        if (counted && std::binary_search(counted->begin(), counted->end(), key)) ++out.counted_umis;
        else if (g1 == NONE) ++out.dropped_umis;
        else ms.push_back(M{g1, g2, rs[i].bc});
        i = j;
    }
    out.molecules = ms.size();
    std::sort(ms.begin(), ms.end(), [](const M& x, const M& y) { return std::tie(x.g1, x.g2, x.bc) < std::tie(y.g1, y.g2, y.bc); });
    // classes in (g1, g2) order (a single gene sorts after its pairs: g2 = NONE), then the triplets per column
    struct T { uint32_t bc, row, n; };
    std::vector<T> ts;
    for (size_t i = 0; i < ms.size();) {
        if (i == 0 || ms[i].g1 != ms[i - 1].g1 || ms[i].g2 != ms[i - 1].g2) {
            std::string nm = gene_ids[ms[i].g1] + "_" + gene_names[ms[i].g1];
            if (ms[i].g2 != NONE) nm += "|" + gene_ids[ms[i].g2] + "_" + gene_names[ms[i].g2];
            out.names.push_back(std::move(nm));
            ++out.nrows;
        }
        const uint32_t row = out.nrows - 1;
        size_t j = i;
        while (j < ms.size() && ms[j].g1 == ms[i].g1 && ms[j].g2 == ms[i].g2 && ms[j].bc == ms[i].bc) ++j;
        ts.push_back(T{ms[i].bc, row, static_cast<uint32_t>(j - i)});
        i = j;
    }
    std::sort(ts.begin(), ts.end(), [](const T& x, const T& y) { return std::tie(x.bc, x.row) < std::tie(y.bc, y.row); });
    out.indptr.assign(static_cast<size_t>(n_barcodes) + 1, 0);
    out.indices.reserve(ts.size());
    out.data.reserve(ts.size());
    for (const auto& t : ts) {
        ++out.indptr[t.bc + 1];
        out.indices.push_back(static_cast<int32_t>(t.row));
        out.data.push_back(t.n);
    }
    for (uint32_t c = 0; c < n_barcodes; ++c) out.indptr[c + 1] += out.indptr[c];
    return out;
}

}  // namespace multimap
}  // namespace singlet
