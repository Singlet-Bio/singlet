// amb_harness: rerun only the ambient fit (ambient_correction.h) on a finished run's stored counts, for fast variants
// of order-free-rules R10b/R10c (ws-w3b update i). Gene counts are the exon and intron features summed per gene id
// (the text before the first '_' of the feature name), genes in ambient_profile.tsv order; cells are the barcodes of
// ambient_contamination.tsv; pool, ties, keys and cell order as export.h sets them (R10, R10b).
//   amb_harness <out dir> <ensemble> <restarts> <out.tsv>    -> barcode, rho, cluster; summary on stderr
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/ambient_correction.h"

static std::string gene_id(const std::string& f) { const auto p = f.find('_'); return p == std::string::npos ? f : f.substr(0, p); }

int main(int argc, char** argv) {
    if (argc != 5) { std::cerr << "usage: amb_harness outdir ensemble restarts out.tsv\n"; return 2; }
    const std::string od = argv[1];
    singlet::pz::ReadLimits lim;
    lim.max_rows = 1ull << 26; lim.max_nnz = 1ull << 31; lim.max_metadata_bytes = 2ull << 30; lim.max_kv_entries = 1ull << 26;
    auto ex = singlet::pz::read_1pz(od + "/exon_counts.1pz", lim);
    auto in = singlet::pz::read_1pz(od + "/intron_counts.1pz", lim);
    std::vector<std::string> genes;
    std::unordered_map<std::string, uint32_t> gidx;
    {
        std::ifstream f(od + "/ambient_profile.tsv"); std::string line; std::getline(f, line);
        while (std::getline(f, line)) { const std::string id = gene_id(line.substr(0, line.find('\t'))); gidx.emplace(id, genes.size()); genes.push_back(id); }
    }
    const uint32_t G = static_cast<uint32_t>(genes.size()), B = ex.n;
    auto collapse = [&](const singlet::pz::ReadResult& r, const char* what) {
        std::vector<int64_t> row_gene(r.m, -1);
        size_t mapped = 0;
        for (uint32_t i = 0; i < r.m && i < r.rownames.size(); ++i) {
            auto it = gidx.find(gene_id(r.rownames[i]));
            if (it != gidx.end()) { row_gene[i] = it->second; ++mapped; }
        }
        std::cerr << "[harness] " << what << ": " << mapped << " of " << r.m << " rows mapped to " << G << " genes (first row "
                  << (r.rownames.empty() ? "" : r.rownames[0]) << ")\n";
        singlet::CSCu32 c; c.nrows = G; c.ncols = B; c.indptr.assign(B + 1, 0);
        std::vector<uint32_t> acc(G, 0); std::vector<uint32_t> touched;
        for (uint32_t j = 0; j < B; ++j) {
            touched.clear();
            for (uint32_t k = r.indptr[j]; k < r.indptr[j + 1]; ++k) {
                const int64_t g = row_gene[r.indices[k]];
                if (g < 0) continue;
                if (acc[g] == 0) touched.push_back(static_cast<uint32_t>(g));
                acc[g] += r.data[k];
            }
            std::sort(touched.begin(), touched.end());
            for (uint32_t g : touched) { c.indices.push_back(static_cast<int32_t>(g)); c.data.push_back(acc[g]); acc[g] = 0; }
            c.indptr[j + 1] = static_cast<int32_t>(c.indices.size());
        }
        return c;
    };
    auto eg = collapse(ex, "exon"), ig = collapse(in, "intron");
    std::unordered_map<std::string, uint32_t> col;
    for (uint32_t j = 0; j < B; ++j) col.emplace(ex.colnames[j], j);
    std::vector<uint32_t> cells;
    {
        std::ifstream f(od + "/ambient_contamination.tsv"); std::string line; std::getline(f, line);
        while (std::getline(f, line)) { auto it = col.find(line.substr(0, line.find('\t'))); if (it != col.end()) cells.push_back(it->second); }
    }
    std::vector<uint32_t> rank(B), by(B);
    for (uint32_t b = 0; b < B; ++b) by[b] = b;
    std::sort(by.begin(), by.end(), [&](uint32_t a, uint32_t b2) { return ex.colnames[a] < ex.colnames[b2]; });
    for (uint32_t r = 0; r < B; ++r) rank[by[r]] = r;
    std::sort(cells.begin(), cells.end(), [&](uint32_t a, uint32_t b2) { return rank[a] < rank[b2]; });
    std::vector<uint64_t> keys(B, 0);
    for (uint32_t b = 0; b < B; ++b) {
        uint64_t k = 0;
        for (size_t i = 0; i < ex.colnames[b].size() && i < 32; ++i) {
            const char c = static_cast<char>(ex.colnames[b][i] | 32);
            k = (k << 2) | (c == 'a' ? 0u : c == 'c' ? 1u : c == 'g' ? 2u : 3u);
        }
        keys[b] = k;
    }
    singlet::AmbientDecontConfig cfg;
    cfg.lower_umi = 100; cfg.pool_min_umi = 5; cfg.pool_tie_rank = &rank; cfg.barcode_keys = &keys;
    cfg.ensemble = static_cast<uint32_t>(std::atoi(argv[2])); cfg.kmeans_restarts = static_cast<uint32_t>(std::atoi(argv[3]));
    auto R = singlet::correct_ambient_gene_level_split(eg, ig, cells, cfg);
    std::ofstream o(argv[4]);
    o << "barcode\trho\tcluster\n";
    double mean = 0;
    for (size_t i = 0; i < cells.size(); ++i) { o << ex.colnames[cells[i]] << '\t' << R.contam[i].rho << '\t' << R.contam[i].cluster << '\n'; mean += R.contam[i].rho; }
    std::cerr << "[harness] " << od << " ensemble=" << cfg.ensemble << " restarts=" << cfg.kmeans_restarts << " cells=" << cells.size()
              << " clusters=" << R.n_clusters << " mean_rho=" << mean / std::max<size_t>(cells.size(), 1)
              << " pool_umis=" << static_cast<uint64_t>(R.profile.total_ambient_umi) << " seconds=" << R.seconds << "\n";
    return 0;
}
