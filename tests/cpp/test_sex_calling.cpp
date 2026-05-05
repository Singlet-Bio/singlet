// N14: Unit tests for sex_calling.h
// Tests sex inference from XIST (chrX) and chrY marker expression.
//
// Approach: write a minimal in-memory GTF to a tmp file, load it into GeneModel,
// construct a synthetic sparse exon matrix, and verify call_sex() output.

#include "singlet/pileup/sex_calling.h"
#include "singlet/pileup/gene_model.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>

// Helper: write a minimal GTF with specified genes
static std::string write_test_gtf(const std::string& path,
    const std::vector<std::tuple<std::string/*chrom*/,
                                  std::string/*gene_id*/,
                                  std::string/*gene_name*/,
                                  int/*start*/,int/*end*/>>& genes)
{
    std::ofstream f(path);
    for (auto& [chrom, gid, gname, gstart, gend] : genes) {
        // gene line
        f << chrom << "\tHAVANA\tgene\t" << gstart << "\t" << gend
          << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \"" << gname << "\";\n";
        // exon line (same coords as gene for simplicity)
        f << chrom << "\tHAVANA\texon\t" << gstart << "\t" << gend
          << "\t.\t+\t.\tgene_id \"" << gid << "\"; gene_name \"" << gname << "\";\n";
    }
    return path;
}

// Minimal CSC matrix struct compatible with call_sex() template
struct TestCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
};

// Build a CSC matrix: values[row][col] = count
static TestCSC make_csc(uint32_t nrows, uint32_t ncols,
                         const std::vector<std::tuple<uint32_t,uint32_t,uint16_t>>& entries)
{
    // Sort entries by column then row
    auto sorted = entries;
    std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){
        return std::get<1>(a) < std::get<1>(b) ||
               (std::get<1>(a) == std::get<1>(b) && std::get<0>(a) < std::get<0>(b));
    });

    TestCSC m;
    m.nrows = nrows;
    m.ncols = ncols;
    m.indptr.resize(ncols + 1, 0);

    for (auto& [r, c, v] : sorted) {
        m.indices.push_back(static_cast<int32_t>(r));
        m.data.push_back(v);
    }
    // Build indptr
    std::vector<int32_t> col_count(ncols, 0);
    for (auto& [r, c, v] : sorted) col_count[c]++;
    for (uint32_t j = 0; j < ncols; ++j)
        m.indptr[j + 1] = m.indptr[j] + col_count[j];

    return m;
}

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* n) { ++passed; std::cerr << "PASS: " << n << "\n"; };
    auto FAIL = [&](const char* n, const char* msg) {
        ++failed; std::cerr << "FAIL: " << n << " — " << msg << "\n";
    };

    // Temporary GTF file path
    const std::string gtf_path = "/tmp/test_sex_n14.gtf";

    // ── Gene set: XIST on chrX, DDX3Y+UTY on chrY, GAPDH on chr12 (autosome) ──
    // GTF gene/exon indices (0-based after building):
    //   GAPDH   exon index 0  (chr12)
    //   XIST    exon index 1  (chrX)
    //   DDX3Y   exon index 2  (chrY)
    //   UTY     exon index 3  (chrY)
    write_test_gtf(gtf_path, {
        {"chr12",  "ENSG00000111640", "GAPDH",  6534512, 6538264},
        {"chrX",   "ENSG00000229807", "XIST",  73820657, 73838787},
        {"chrY",   "ENSG00000067048", "DDX3Y", 15016484, 15028547},
        {"chrY",   "ENSG00000183878", "UTY",   13291498, 13629962},
    });

    singlet::GeneModel model;
    bool loaded = model.load_gtf(gtf_path);
    if (!loaded) {
        std::cerr << "FATAL: could not load test GTF " << gtf_path << "\n";
        return 1;
    }

    std::cerr << "[test] GeneModel: " << model.n_exons() << " exons from " << model.n_genes() << " genes\n";

    // Expected exon row mapping after load_gtf()
    // The GeneModel builds exon_names in parse order from GTF (gene order: GAPDH, XIST, DDX3Y, UTY)
    // Exon 0 = GAPDH (chr12), exon 1 = XIST (chrX), exon 2 = DDX3Y (chrY), exon 3 = UTY (chrY)
    // But models may reorder by chromosome; verify by checking exon_names
    uint32_t n_exons = model.n_exons();
    if (n_exons != 4) {
        std::cerr << "Expected 4 exons, got " << n_exons << "\n";
        return 1;
    }

    // Find actual row indices for each gene by checking exon names
    const auto& genes = model.genes();
    uint32_t xist_exon = UINT32_MAX, ddx3y_exon = UINT32_MAX, uty_exon = UINT32_MAX, gapdh_exon = UINT32_MAX;
    for (auto& g : genes) {
        if (!g.exon_indices.empty()) {
            if (g.gene_name == "XIST")  xist_exon  = g.exon_indices[0];
            if (g.gene_name == "DDX3Y") ddx3y_exon = g.exon_indices[0];
            if (g.gene_name == "UTY")   uty_exon   = g.exon_indices[0];
            if (g.gene_name == "GAPDH") gapdh_exon = g.exon_indices[0];
        }
    }
    std::cerr << "[test] xist_exon=" << xist_exon << " ddx3y_exon=" << ddx3y_exon
              << " uty_exon=" << uty_exon << " gapdh_exon=" << gapdh_exon << "\n";

    // ── Test 1: Female sample — high XIST, no chrY markers ──
    {
        // 3 cells, 4 features
        // XIST: 2000 UMIs total across cells → 2000/~2500 * 1e6 = ~800 CPM >> 1.0
        // Y markers: 0 UMIs
        // GAPDH: 500 UMIs (background)
        auto csc = make_csc(4, 3, {
            {gapdh_exon, 0, 150}, {gapdh_exon, 1, 200}, {gapdh_exon, 2, 150},
            {xist_exon,  0, 700}, {xist_exon,  1, 800}, {xist_exon,  2, 500},
        });
        auto res = singlet::call_sex(csc, model);
        std::cerr << "[test1] sex=" << res.sex << " conf=" << res.confidence
                  << " xist_cpm=" << res.xist_cpm << " y_cpm=" << res.y_cpm
                  << " n_y=" << res.n_y_genes_detected << "\n";
        if (res.sex == "female") PASS("female_call");
        else FAIL("female_call", ("expected female got " + res.sex).c_str());
        if (res.confidence > 0) PASS("female_confidence_nonzero");
        else FAIL("female_confidence_nonzero", "confidence=0");
    }

    // ── Test 2: Male sample — no XIST, moderate chrY ──
    {
        // DDX3Y+UTY: ~500 UMIs total → high Y_cpm (>2.0)
        // XIST: 0 UMIs
        auto csc = make_csc(4, 3, {
            {gapdh_exon, 0, 150}, {gapdh_exon, 1, 200}, {gapdh_exon, 2, 150},
            {ddx3y_exon, 0, 80},  {ddx3y_exon, 1, 100}, {ddx3y_exon, 2, 70},
            {uty_exon,   0, 50},  {uty_exon,   1, 60},  {uty_exon,   2, 40},
        });
        auto res = singlet::call_sex(csc, model);
        std::cerr << "[test2] sex=" << res.sex << " conf=" << res.confidence
                  << " xist_cpm=" << res.xist_cpm << " y_cpm=" << res.y_cpm
                  << " n_y=" << res.n_y_genes_detected << "\n";
        if (res.sex == "male") PASS("male_call");
        else FAIL("male_call", ("expected male got " + res.sex).c_str());
        if (res.n_y_genes_detected >= 2) PASS("male_n_y_genes_ge2");
        else FAIL("male_n_y_genes_ge2", "expected >=2 Y genes detected");
    }

    // ── Test 3: Unknown — neither XIST nor chrY markers ──
    {
        auto csc = make_csc(4, 3, {
            {gapdh_exon, 0, 150}, {gapdh_exon, 1, 200}, {gapdh_exon, 2, 150},
        });
        auto res = singlet::call_sex(csc, model);
        std::cerr << "[test3] sex=" << res.sex << " conf=" << res.confidence << "\n";
        if (res.sex == "unknown") PASS("unknown_call");
        else FAIL("unknown_call", ("expected unknown got " + res.sex).c_str());
    }

    // ── Test 4: Empty matrix → no crash ──
    {
        TestCSC empty;
        empty.nrows = 0; empty.ncols = 0;
        empty.indptr = {0};
        auto res = singlet::call_sex(empty, model);
        if (res.sex == "unknown") PASS("empty_matrix_safe");
        else FAIL("empty_matrix_safe", "expected unknown on empty");
    }

    // ── Test 5: write_sex_call_json / read back ──
    {
        singlet::SexCallResult r;
        r.sex = "female"; r.confidence = 0.87;
        r.xist_cpm = 145.3; r.y_cpm = 0.1;
        r.n_y_genes_detected = 0; r.xist_umis = 12345; r.y_umis = 8; r.total_umis = 85000;
        singlet::write_sex_call_json("/tmp/test_sex_n14.json", r);
        std::ifstream f("/tmp/test_sex_n14.json");
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        if (content.find("\"female\"") != std::string::npos) PASS("json_female_written");
        else FAIL("json_female_written", "female not in JSON");
        if (content.find("145.3") != std::string::npos) PASS("json_xist_cpm");
        else FAIL("json_xist_cpm", "xist_cpm not in JSON");
    }

    std::cerr << "\n=== N14 sex_calling: " << passed << " passed, " << failed << " failed ===\n";
    return failed > 0 ? 1 : 0;
}
