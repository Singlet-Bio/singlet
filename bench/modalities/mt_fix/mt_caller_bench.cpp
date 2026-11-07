// SPDX-License-Identifier: MIT
// Harness: feed the chrM caller (mt_heteroplasmy.h + mt_event_caller.h) per-cell allele counts built from a tagged BAM
// by mt_pileup_strand.py, and write the outputs export.h would (Phase 2 heteroplasmy, G6 donor outputs).
//
//   mt_caller_bench COUNTS.bin ASSIGN.tsv GENOME_DIR OUTDIR
//
// Built twice: the fixed headers (default) and the pre-fix headers (-DMTFIX_OLD, -I old headers first) so before and
// after run through identical harness code on identical counts.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "singlet/pileup/mt_event_caller.h"
#include "singlet/pileup/mt_reference.h"

using namespace singlet;

struct Counts {
    uint32_t n_cells = 0;
    std::vector<int32_t> ip_t, ix_t, ip_r, ix_r;
    std::vector<uint16_t> d_t, d_r;
};

static bool read_counts(const std::string& path, Counts& c) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic, 4);
    uint32_t n; uint64_t nt, nr;
    f.read(reinterpret_cast<char*>(&n), 4); f.read(reinterpret_cast<char*>(&nt), 8); f.read(reinterpret_cast<char*>(&nr), 8);
    c.n_cells = n;
    c.ip_t.resize(n + 1); c.ix_t.resize(nt); c.d_t.resize(nt);
    c.ip_r.resize(n + 1); c.ix_r.resize(nr); c.d_r.resize(nr);
    f.read(reinterpret_cast<char*>(c.ip_t.data()), 4 * (n + 1));
    f.read(reinterpret_cast<char*>(c.ix_t.data()), 4 * nt);
    f.read(reinterpret_cast<char*>(c.d_t.data()), 2 * nt);
    f.read(reinterpret_cast<char*>(c.ip_r.data()), 4 * (n + 1));
    f.read(reinterpret_cast<char*>(c.ix_r.data()), 4 * nr);
    f.read(reinterpret_cast<char*>(c.d_r.data()), 2 * nr);
    return static_cast<bool>(f);
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: mt_caller_bench COUNTS.bin ASSIGN.tsv GENOME_DIR OUTDIR\n"; return 2; }
    const std::string counts_f = argv[1], assign_f = argv[2], genome_dir = argv[3], out = argv[4];
    std::filesystem::create_directories(out);
    Counts c;
    if (!read_counts(counts_f, c)) { std::cerr << "cannot read " << counts_f << "\n"; return 1; }

    SparseAccumulator<uint16_t>::CSCMatrix mt_csc, mt_rev;
    mt_csc.indptr = c.ip_t; mt_csc.indices = c.ix_t; mt_csc.data = c.d_t; mt_csc.ncols = c.n_cells; mt_csc.nrows = mt::MT_N_FEATURES;
    mt_rev.indptr = c.ip_r; mt_rev.indices = c.ix_r; mt_rev.data = c.d_r; mt_rev.ncols = c.n_cells; mt_rev.nrows = mt::MT_N_FEATURES;

    // donor assignments (same convention as DonorAssignment: 0..K-1 donors, K doublet, K+1 unassigned)
    DemuxResult dm;
    int K = 0;
    {
        std::ifstream f(assign_f);
        std::string line; std::getline(f, line);
        std::vector<std::string> lab, st;
        while (std::getline(f, line)) {
            std::stringstream ss(line); std::string i, d, s;
            std::getline(ss, i, '\t'); std::getline(ss, d, '\t'); std::getline(ss, s, '\t');
            lab.push_back(d); st.push_back(s);
            if (d.rfind("donor", 0) == 0) K = std::max(K, std::stoi(d.substr(5)) + 1);
        }
        dm.n_donors_k = K;
        for (size_t i = 0; i < lab.size(); ++i) {
            DonorAssignment a;
            a.prob_max = 1.f; a.prob_doublet = 0.f; a.label = lab[i]; a.cell_status = st[i];
            a.donor_id = lab[i].rfind("donor", 0) == 0 ? std::stoi(lab[i].substr(5)) : (lab[i] == "doublet" ? K : K + 1);
            dm.assignments.push_back(a);
        }
    }
    std::vector<std::string> barcodes(c.n_cells);
    for (uint32_t i = 0; i < c.n_cells; ++i) barcodes[i] = "cell" + std::to_string(i);

    auto ref = extract_mt_reference(genome_dir, "GRCh38-2024-A");
    std::cerr << "[bench] reference found=" << ref.found << " " << ref.contig_name << " len=" << ref.length << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    mt::MtHetResult het;
#ifdef MTFIX_OLD
    het = mt::compute_heteroplasmy(mt_csc.indptr.data(), mt_csc.indices.data(), mt_csc.data.data(),
                                   mt_csc.nrows, mt_csc.ncols);
#else
    mt::MtCounts tot, rev;
    tot.indptr = mt_csc.indptr.data(); tot.indices = mt_csc.indices.data(); tot.data = mt_csc.data.data(); tot.n_cells = c.n_cells;
    rev.indptr = mt_rev.indptr.data(); rev.indices = mt_rev.indices.data(); rev.data = mt_rev.data.data(); rev.n_cells = c.n_cells;
    // evidence from called cells only (ambient droplets are back-assigned donors but carry cross-donor chrM reads)
    std::vector<uint8_t> cell_mask(c.n_cells, 0);
    size_t n_called = 0;
    for (size_t i = 0; i < dm.assignments.size() && i < cell_mask.size(); ++i)
        if (dm.assignments[i].cell_status == "cell") { cell_mask[i] = 1; ++n_called; }
    std::cerr << "[bench] called cells in mask: " << n_called << " of " << c.n_cells << "\n";
    het = mt::call_mt_variants(tot, &rev, ref.found ? ref.sequence : std::string(), mt::MtCallerConfig(), &cell_mask);
    std::cerr << "[bench] reference=" << het.ref_source << " pool!=ref positions=" << het.n_ref_mismatch_pool
              << " sites evaluated=" << het.sites.size() << " heteroplasmic=" << het.n_variants << "\n";
    std::filesystem::create_directories(out + "/private");
    mt::write_mt_sites_private(out + "/private/mt_sites.tsv", het);
    {   // near-miss table of the donor haplotype caller (why AF>=0.5 donor alleles were or were not called)
        std::vector<int> grp(c.n_cells, -1);
        for (size_t i = 0; i < dm.assignments.size() && i < grp.size(); ++i)
            if (dm.assignments[i].cell_status == "cell" && dm.assignments[i].donor_id >= 0 && dm.assignments[i].donor_id < K)
                grp[i] = dm.assignments[i].donor_id;
        std::vector<std::string> dbg;
        mt::call_group_haplotypes(tot, &rev, grp, K, ref.found ? ref.sequence : std::string(), mt::MtCallerConfig(), &dbg);
        std::ofstream d(out + "/private/hap_debug.tsv");
        d << "donor\tpos\tref\talt\tdp\talt_n\talt_fwd\talt_rev\tprof_fwd\tprof_rev\tn_cells\taf\tmasked\tresult\n";
        for (const auto& l : dbg) d << l << "\n";
    }
#endif
    auto t1 = std::chrono::high_resolution_clock::now();
    mt::write_mt_variants(out + "/mt_variants.tsv", het);
    {   // heteroplasmy matrix as COO text: variant_idx, cell_idx, vaf
        std::ofstream m(out + "/mt_heteroplasmy_entries.tsv");
        m << "variant\tcell\tvaf\n";
        for (uint32_t cc = 0; cc < het.n_cells; ++cc)
            for (int32_t k = het.indptr[cc]; k < het.indptr[cc + 1]; ++k)
                m << het.indices[k] << "\t" << cc << "\t" << het.data[k] << "\n";
        std::ofstream n(out + "/mt_heteroplasmy_features.tsv");
        for (const auto& s : mt::het_feature_names(het)) n << s << "\n";
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::map<std::string, std::string> meta;
#ifdef MTFIX_OLD
    mt::write_mt_donor_outputs(out, mt_csc, dm, barcodes, meta, {}, "");
#else
    mt::MtCallerConfig gcfg;
    if (const char* e = std::getenv("MTFIX_HAP_AF")) gcfg.hap_af = static_cast<float>(std::atof(e));
    mt::write_mt_donor_outputs(out, mt_csc, dm, barcodes, meta, {}, "", ref.found ? ref.sequence : std::string(), &mt_rev, &gcfg, &het);
#endif
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cerr << "[bench] call time " << std::chrono::duration<double>(t1 - t0).count() << "s, G6 "
              << std::chrono::duration<double>(t3 - t2).count() << "s\n";
    return 0;
}
