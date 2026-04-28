// test/test_cell_qc.cpp
// Unit tests for cell_qc_metrics.h (G-QC feature).
// Constructs a minimal in-memory GTF → GeneModel, builds synthetic CSC matrices,
// and verifies n_genes / total_umis / pct_mt / pct_ribo / pct_intronic.

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "singlet-pileup/cell_qc_metrics.h"
#include "singlet-pileup/gene_model.h"
#include "singlet-pileup/sparse_accumulator.h"

// ── Helpers ───────────────────────────────────────────────────────────────

static std::string write_test_gtf(const std::string& path,
                                  const std::vector<std::tuple<std::string, std::string, std::string, int, int>>& genes) {
    // Each tuple: (chrom, gene_id, gene_name, start, end)
    std::ofstream f(path);
    for (auto& [chrom, gid, gname, gstart, gend] : genes) {
        f << chrom << "\tHAVANA\tgene\t" << gstart << "\t" << gend
          << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \"" << gname << "\";\n";
        // Single exon occupying first half, intron in second half
        int emid = gstart + (gend - gstart) / 2;
        f << chrom << "\tHAVANA\texon\t" << gstart << "\t" << emid
          << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \"" << gname << "\";\n";
    }
    return path;
}

using CSCMatrix = singlet::SparseAccumulator<uint16_t>::CSCMatrix;

// Build CSC from (row, col, value) entry list
static CSCMatrix make_csc(uint32_t nrows, uint32_t ncols,
                          const std::vector<std::tuple<uint32_t, uint32_t, uint16_t>>& entries) {
    auto sorted = entries;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
        return std::get<0>(a) < std::get<0>(b);
    });

    CSCMatrix m;
    m.nrows = nrows;
    m.ncols = ncols;
    m.indptr.resize(ncols + 1, 0);

    std::vector<int32_t> col_count(ncols, 0);
    for (auto& [r, c, v] : sorted) {
        m.indices.push_back(static_cast<int32_t>(r));
        m.data.push_back(v);
        col_count[c]++;
    }
    for (uint32_t j = 0; j < ncols; ++j)
        m.indptr[j + 1] = m.indptr[j] + col_count[j];

    return m;
}

static CSCMatrix make_empty_csc(uint32_t nrows, uint32_t ncols) {
    CSCMatrix m;
    m.nrows = nrows;
    m.ncols = ncols;
    m.indptr.resize(ncols + 1, 0);
    return m;
}

static float approx(float a, float b, float tol = 0.05f) {
    return std::fabs(a - b) <= tol;
}

// ── Tests ─────────────────────────────────────────────────────────────────

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* n) { ++passed; std::cerr << "PASS: " << n << "\n"; };
    auto FAIL = [&](const char* n, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << n << " — " << msg << "\n";
    };

    const std::string gtf_path = "/tmp/test_cell_qc_genes.gtf";

    // ── Build gene model with 4 genes ──
    // Gene 0: GAPDH    chr1  (housekeeping — neither MT nor ribo)
    // Gene 1: MT-CO1   chrM  (mitochondrial by chrom + name)
    // Gene 2: RPS3     chr2  (ribosomal small subunit)
    // Gene 3: Rpl5     chr3  (mouse-style ribosomal large subunit)
    // GTF exon occupies first half of each gene body; second half becomes intron.
    // gene body 1000–2000 → exon 1000–1500, intron 1500–2000

    write_test_gtf(gtf_path, {
                                 {"chr1", "ENSG_GAPDH", "GAPDH", 1000, 2000},
                                 {"chrM", "ENSG_MTCO1", "MT-CO1", 1000, 2000},
                                 {"chr2", "ENSG_RPS3", "RPS3", 1000, 2000},
                                 {"chr3", "ENSG_RPL5", "Rpl5", 1000, 2000},
                             });

    singlet::GeneModel gm;
    if (!gm.load_gtf(gtf_path)) {
        std::cerr << "FATAL: could not load GTF\n";
        return 1;
    }

    // ── Test 1: Gene classification ──────────────────────────────────────
    {
        const auto& genes = gm.genes();
        // Sorted alphabetically by gene_id: ENSG_GAPDH, ENSG_MTCO1, ENSG_RPL5, ENSG_RPS3
        // Indices: 0=GAPDH, 1=MT-CO1, 2=Rpl5, 3=RPS3
        bool mt_co1_is_mt = false;
        bool gapdh_is_mt = false;
        bool rps3_is_ribo = false;
        bool rpl5_is_ribo = false;
        bool gapdh_is_ribo = false;

        for (const auto& g : genes) {
            if (g.gene_name == "MT-CO1") mt_co1_is_mt = singlet::is_mt_gene(g);
            if (g.gene_name == "GAPDH") gapdh_is_mt = singlet::is_mt_gene(g);
            if (g.gene_name == "RPS3") rps3_is_ribo = singlet::is_ribo_gene(g);
            if (g.gene_name == "Rpl5") rpl5_is_ribo = singlet::is_ribo_gene(g);
            if (g.gene_name == "GAPDH") gapdh_is_ribo = singlet::is_ribo_gene(g);
        }

        if (mt_co1_is_mt)
            PASS("MT-CO1 classified as MT");
        else
            FAIL("MT-CO1 classified as MT", "is_mt_gene returned false");
        if (!gapdh_is_mt)
            PASS("GAPDH not classified as MT");
        else
            FAIL("GAPDH not classified as MT", "is_mt_gene returned true");
        if (rps3_is_ribo)
            PASS("RPS3 classified as ribo");
        else
            FAIL("RPS3 classified as ribo", "is_ribo_gene returned false");
        if (rpl5_is_ribo)
            PASS("Rpl5 classified as ribo (mouse case)");
        else
            FAIL("Rpl5 classified as ribo (mouse case)", "is_ribo_gene returned false");
        if (!gapdh_is_ribo)
            PASS("GAPDH not classified as ribo");
        else
            FAIL("GAPDH not classified as ribo", "is_ribo_gene returned true");
    }

    // ── Test 2: n_genes and total_umis. ──────────────────────────────────
    // 4 genes sorted by gene_id: GAPDH(0), MT-CO1(1), Rpl5(2), RPS3(3)
    // Each gene has 1 exon (exon idx == gene idx since 1 exon per gene)
    // and 1 intron (intron idx == gene idx since 1 intron per gene).
    //
    // Cell 0: GAPDH exon=10, MT-CO1 exon=20  → 2 genes, total=30, mt%=66.7
    // Cell 1: RPS3 exon=5, Rpl5 exon=15      → 2 genes, total=20, ribo%=100
    {
        uint32_t n_exons = gm.n_exons();
        uint32_t n_introns = gm.n_introns();
        uint32_t n_cells = 2;

        // Find per-gene exon indices
        std::vector<uint32_t> gene_exon0(4, UINT32_MAX);
        for (uint32_t gi = 0; gi < gm.n_genes(); ++gi) {
            const auto& g = gm.genes()[gi];
            if (!g.exon_indices.empty())
                gene_exon0[gi] = g.exon_indices[0];
        }

        // Cell 0: GAPDH exon=10, MT-CO1 exon=20
        // Cell 1: RPS3 exon=5, Rpl5 exon=15
        CSCMatrix exon_csc = make_csc(n_exons, n_cells, {
                                                            {gene_exon0[0], 0, 10},  // GAPDH in cell 0
                                                            {gene_exon0[1], 0, 20},  // MT-CO1 in cell 0
                                                            {gene_exon0[2], 1, 15},  // Rpl5 in cell 1
                                                            {gene_exon0[3], 1, 5},   // RPS3 in cell 1
                                                        });
        CSCMatrix intron_csc = make_empty_csc(n_introns, n_cells);

        auto qc = singlet::compute_cell_qc(exon_csc, intron_csc, gm);

        // n_genes
        if (qc.total_genes[0] == 2)
            PASS("cell0 n_genes=2");
        else
            FAIL("cell0 n_genes=2", ("got " + std::to_string(qc.total_genes[0])).c_str());
        if (qc.total_genes[1] == 2)
            PASS("cell1 n_genes=2");
        else
            FAIL("cell1 n_genes=2", ("got " + std::to_string(qc.total_genes[1])).c_str());

        // total_umis
        if (qc.total_umis[0] == 30)
            PASS("cell0 total_umis=30");
        else
            FAIL("cell0 total_umis=30", ("got " + std::to_string(qc.total_umis[0])).c_str());
        if (qc.total_umis[1] == 20)
            PASS("cell1 total_umis=20");
        else
            FAIL("cell1 total_umis=20", ("got " + std::to_string(qc.total_umis[1])).c_str());

        // pct_mt: cell0 = 20/30 * 100 = 66.67, cell1 = 0
        if (approx(qc.mt_pct[0], 66.67f, 0.1f))
            PASS("cell0 pct_mt≈66.67");
        else
            FAIL("cell0 pct_mt≈66.67", ("got " + std::to_string(qc.mt_pct[0])).c_str());
        if (approx(qc.mt_pct[1], 0.f, 0.01f))
            PASS("cell1 pct_mt=0");
        else
            FAIL("cell1 pct_mt=0", ("got " + std::to_string(qc.mt_pct[1])).c_str());

        // pct_ribo: cell0 = 0, cell1 = 20/20 = 100
        if (approx(qc.ribo_pct[0], 0.f, 0.01f))
            PASS("cell0 pct_ribo=0");
        else
            FAIL("cell0 pct_ribo=0", ("got " + std::to_string(qc.ribo_pct[0])).c_str());
        if (approx(qc.ribo_pct[1], 100.f, 0.1f))
            PASS("cell1 pct_ribo=100");
        else
            FAIL("cell1 pct_ribo=100", ("got " + std::to_string(qc.ribo_pct[1])).c_str());
    }

    // ── Test 3: pct_intronic ─────────────────────────────────────────────
    // Cell 0: GAPDH exon=10, GAPDH intron=10 → intronic = 10/20 = 50%
    // Cell 1: MT-CO1 exon=20, MT-CO1 intron=0 → intronic = 0%
    {
        uint32_t n_exons = gm.n_exons();
        uint32_t n_introns = gm.n_introns();
        uint32_t n_cells = 2;

        const auto& g_gapdh = gm.genes()[0];  // GAPDH sorted first
        const auto& g_mtco1 = gm.genes()[1];  // MT-CO1 sorted second

        uint32_t gapdh_exon = g_gapdh.exon_indices[0];
        uint32_t gapdh_intron = g_gapdh.intron_indices.empty() ? UINT32_MAX : g_gapdh.intron_indices[0];
        uint32_t mtco1_exon = g_mtco1.exon_indices[0];

        if (gapdh_intron == UINT32_MAX) {
            std::cerr << "[skip] GAPDH has no introns — intronic_pct test skipped\n";
        } else {
            CSCMatrix exon_csc = make_csc(n_exons, n_cells, {
                                                                {gapdh_exon, 0, 10},
                                                                {mtco1_exon, 1, 20},
                                                            });
            CSCMatrix intron_csc = make_csc(n_introns, n_cells, {
                                                                    {gapdh_intron, 0, 10},  // 50% intronic
                                                                });

            auto qc = singlet::compute_cell_qc(exon_csc, intron_csc, gm);

            if (approx(qc.intronic_pct[0], 50.f, 0.1f))
                PASS("cell0 pct_intronic=50");
            else
                FAIL("cell0 pct_intronic=50", ("got " + std::to_string(qc.intronic_pct[0])).c_str());
            if (approx(qc.intronic_pct[1], 0.f, 0.01f))
                PASS("cell1 pct_intronic=0");
            else
                FAIL("cell1 pct_intronic=0", ("got " + std::to_string(qc.intronic_pct[1])).c_str());
        }
    }

    // ── Test 4: pct values in [0, 100] range ─────────────────────────────
    {
        uint32_t n_exons = gm.n_exons();
        uint32_t n_introns = gm.n_introns();
        uint32_t n_cells = 1;

        // Give all UMIs to MT gene
        CSCMatrix exon_csc = make_csc(n_exons, n_cells, {
                                                            {gm.genes()[1].exon_indices[0], 0, 1000},  // MT-CO1
                                                        });
        CSCMatrix intron_csc = make_empty_csc(n_introns, n_cells);

        auto qc = singlet::compute_cell_qc(exon_csc, intron_csc, gm);

        bool mt_bounded = (qc.mt_pct[0] >= 0.f && qc.mt_pct[0] <= 100.f);
        bool ribo_bounded = (qc.ribo_pct[0] >= 0.f && qc.ribo_pct[0] <= 100.f);
        bool intr_bounded = (qc.intronic_pct[0] >= 0.f && qc.intronic_pct[0] <= 100.f);

        if (mt_bounded)
            PASS("pct_mt in [0,100]");
        else
            FAIL("pct_mt in [0,100]", ("got " + std::to_string(qc.mt_pct[0])).c_str());
        if (ribo_bounded)
            PASS("pct_ribo in [0,100]");
        else
            FAIL("pct_ribo in [0,100]", ("got " + std::to_string(qc.ribo_pct[0])).c_str());
        if (intr_bounded)
            PASS("pct_intronic in [0,100]");
        else
            FAIL("pct_intronic in [0,100]", ("got " + std::to_string(qc.intronic_pct[0])).c_str());

        // mt% should be 100 when all UMIs are MT
        if (approx(qc.mt_pct[0], 100.f, 0.01f))
            PASS("all-MT cell → pct_mt=100");
        else
            FAIL("all-MT cell → pct_mt=100", ("got " + std::to_string(qc.mt_pct[0])).c_str());
    }

    // ── Test 5: TSV writer roundtrip ─────────────────────────────────────
    {
        singlet::CellQCMetrics qc;
        qc.total_umis = {100, 200};
        qc.total_genes = {50, 80};
        qc.mt_pct = {5.5f, 3.2f};
        qc.ribo_pct = {12.0f, 8.1f};
        qc.intronic_pct = {30.0f, 25.5f};

        std::vector<std::string> barcodes = {"AAACCTGA", "AAACGGAT"};
        const std::string tsv_path = "/tmp/test_cell_qc_out.tsv";

        singlet::write_cell_qc_tsv(tsv_path, qc, barcodes);

        std::ifstream f(tsv_path);
        std::string line;
        std::getline(f, line);  // header
        bool header_ok = (line.find("barcode") != std::string::npos &&
                          line.find("total_umis") != std::string::npos &&
                          line.find("mt_pct") != std::string::npos);
        if (header_ok)
            PASS("TSV header present");
        else
            FAIL("TSV header present", line.c_str());

        std::getline(f, line);  // cell 0
        bool row0_ok = (line.find("AAACCTGA") != std::string::npos &&
                        line.find("100") != std::string::npos);
        if (row0_ok)
            PASS("TSV row 0 barcode+umis correct");
        else
            FAIL("TSV row 0 barcode+umis correct", line.c_str());

        std::getline(f, line);  // cell 1
        bool row1_ok = (line.find("AAACGGAT") != std::string::npos &&
                        line.find("200") != std::string::npos);
        if (row1_ok)
            PASS("TSV row 1 barcode+umis correct");
        else
            FAIL("TSV row 1 barcode+umis correct", line.c_str());
    }

    // ── Test 6: empty matrix → zero metrics ──────────────────────────────
    {
        uint32_t n_exons = gm.n_exons();
        uint32_t n_introns = gm.n_introns();
        CSCMatrix exon_csc = make_empty_csc(n_exons, 3);
        CSCMatrix intron_csc = make_empty_csc(n_introns, 3);

        auto qc = singlet::compute_cell_qc(exon_csc, intron_csc, gm);

        bool all_zero = (qc.total_umis[0] == 0 && qc.total_genes[0] == 0 &&
                         qc.mt_pct[0] == 0.f && qc.ribo_pct[0] == 0.f);
        if (all_zero)
            PASS("empty matrix → zero metrics");
        else
            FAIL("empty matrix → zero metrics", "non-zero found");
    }

    // ── Summary ──────────────────────────────────────────────────────────
    std::cerr << "\nResults: " << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
