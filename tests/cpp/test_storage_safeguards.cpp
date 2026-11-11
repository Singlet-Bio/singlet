// Private storage safeguards (storage_safeguards.h): the pooled chrM counts of the uncalled barcodes, column selection, the per-donor
// callable masks, and the called-cell columns of cells_allsite.vcf.gz when the store holds further (uncalled) columns.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <zlib.h>

#include "singlet/pileup/storage_safeguards.h"

namespace SG = singlet::safeguards;
namespace A = singlet::allsite;
static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

static std::string slurp_gz(const std::string& path) {
    gzFile f = gzopen(path.c_str(), "rb");
    std::string out;
    if (!f) return out;
    char buf[4096]; int n;
    while ((n = gzread(f, buf, sizeof(buf))) > 0) out.append(buf, static_cast<size_t>(n));
    gzclose(f);
    return out;
}

using CSC = singlet::SparseAccumulator<uint16_t>::CSCMatrix;

static CSC make_csc(uint32_t nrows, const std::vector<std::vector<std::pair<int32_t, uint16_t>>>& cols) {
    CSC m; m.nrows = nrows; m.ncols = static_cast<uint32_t>(cols.size()); m.indptr.assign(cols.size() + 1, 0);
    for (size_t j = 0; j < cols.size(); ++j) {
        for (const auto& e : cols[j]) { m.indices.push_back(e.first); m.data.push_back(e.second); }
        m.indptr[j + 1] = static_cast<int32_t>(m.indices.size());
    }
    return m;
}

int main() {
    std::filesystem::create_directories("tmp_sg");
    // ---- select_columns: the chosen columns keep their entries, the others stay as empty columns
    CSC a = make_csc(8, {{{0, 3}, {5, 1}}, {{1, 2}}, {{5, 7}, {7, 1}}});
    const std::vector<uint8_t> called = {1, 0, 1};
    CSC c1 = SG::select_columns(a, called, 1), c0 = SG::select_columns(a, called, 0);
    CHECK(c1.ncols == 3 && c1.indptr == std::vector<int32_t>({0, 2, 2, 4}) && c1.data.size() == 4, "called columns kept, the middle one empty");
    CHECK(c0.ncols == 3 && c0.indptr == std::vector<int32_t>({0, 0, 1, 1}) && c0.indices == std::vector<int32_t>({1}), "uncalled columns kept, the others empty");

    // ---- pooled chrM counts of the uncalled barcodes: 2 positions (rows 0-3, 4-7), barcodes 1 and 3 uncalled
    CSC tot = make_csc(8, {{{0, 3}}, {{1, 2}, {5, 4}}, {{5, 7}}, {{1, 1}, {7, 2}}});
    CSC rev = make_csc(8, {{}, {{1, 1}}, {}, {{7, 2}}});
    const std::vector<uint8_t> is_cell = {1, 0, 1, 0};
    const std::string p = "tmp_mt_uncalled.tsv.gz";
    const long long n = SG::write_mt_uncalled_pooled(p, tot, &rev, is_cell, 2);
    const std::string txt = slurp_gz(p);
    std::remove(p.c_str());
    CHECK(n == 2, "two uncalled barcodes pooled");
    CHECK(txt.find("#barcodes_pooled=2 called_cells=2 entries_pooled=4\n") != std::string::npos, "header counts");
    // position 1: C = 2 + 1 = 3 (rev 1); position 2: C = 4 (rev 0), T = 2 (rev 2)
    CHECK(txt.find("\npos\tA\tC\tG\tT\trevA\trevC\trevG\trevT\n1\t0\t3\t0\t0\t0\t1\t0\t0\n2\t0\t4\t0\t2\t0\t0\t0\t2\n") != std::string::npos, "pooled rows");
    const long long n2 = SG::write_mt_uncalled_pooled(p, tot, static_cast<const CSC*>(nullptr), is_cell, 2);
    CHECK(n2 == 2 && slurp_gz(p).find("1\t0\t3\t0\t0\t0\t0\t0\t0\n") != std::string::npos, "no reverse-strand matrix: zero reverse counts");
    std::remove(p.c_str());

    // ---- callable masks: 2 donors, panel sites chr1:100, chr1:101, chr2:50 and one all-site site chr1:102
    singlet::DemuxResult demux; demux.n_donors_k = 2; demux.covered_to_original = {0, 1, 2};
    const std::vector<std::string> snps = {"chr1:100:A>G", "chr1:101:A>G", "chr2:50:C>T"};
    singlet::DonorDepths dd; dd.n_donors = 2; dd.n_covered = 3;
    dd.dp = {{20, 0}, {20, 2}, {20, 30}};
    dd.ad = {{0, 0}, {0, 1}, {20, 0}};
    A::Result r; r.sites.push_back({0, 101, 'A', 'G', 5, 3});   // contig 0 = chr1, 0-based position 101 = chr1:102
    A::DonorCalls calls; calls.K = 2; calls.sites = 1; calls.sad = {0, 0}; calls.sdp = {25, 0};
    const std::vector<std::string> contigs = {"chr1", "chr2"};
    std::vector<uint64_t> iv;
    const long long m = SG::write_callable_masks("tmp_sg/callable", demux, dd, snps, contigs, &r, &calls, &iv);
    const std::string d0 = slurp_gz("tmp_sg/callable/donor0_callable.rle.gz"), d1 = slurp_gz("tmp_sg/callable/donor1_callable.rle.gz");
    std::remove("tmp_sg/callable/donor0_callable.rle.gz"); std::remove("tmp_sg/callable/donor1_callable.rle.gz"); 
    // donor0: chr1:100, 101, 102 are adjacent (one interval 99-102), chr2:50 has 20 alt reads out of 20 (confident 1/1)
    CHECK(d0.find(">chr1\n99\t3\n>chr2\n49\t1\n") != std::string::npos, "donor0: adjacent confident positions merged, run-length coded");
    // donor1: depth 2 at chr1:101 is not confident, depth 30 all reference at chr2:50 is
    CHECK(d1.find(">chr1") == std::string::npos && d1.find(">chr2\n49\t1\n") != std::string::npos, "donor1 keeps only the deep position");
    CHECK(m == 3 && iv.size() == 2 && iv[0] == 2 && iv[1] == 1, "interval counts");

    // ---- cells_allsite.vcf.gz: a store with an uncalled column after the called ones writes the same bytes as the called-only store
    A::DeltaStore s3, s2;
    s3.n_sites = s2.n_sites = 1; s3.K = s2.K = 1;
    s3.off = {0, 1, 2, 3}; s3.row = {0, 0, 0}; s3.ad = {1, 0, 1}; s3.dp = {4, 3, 1};
    s2.off = {0, 1, 2};    s2.row = {0, 0};    s2.ad = {1, 0};    s2.dp = {4, 3};
    A::DonorCalls vc; vc.K = 1; vc.sites = 1; vc.gt = {1}; vc.differs = {0}; vc.variant = {1}; vc.sad = {5}; vc.sdp = {9};
    A::Result vr; vr.sites.push_back({0, 9, 'A', 'G', 2, 2});
    const std::vector<std::string> barcodes = {"AAAA", "CCCC", "GGGG"};
    const std::vector<uint32_t> cells2 = {0, 1};
    const std::vector<int32_t> donor_of = {0, 0, -1};
    const long long v3 = A::write_cells_vcf("tmp_sg/cells3.vcf.gz", s3, vr, {"chr1"}, barcodes, cells2, donor_of, vc);
    const long long v2 = A::write_cells_vcf("tmp_sg/cells2.vcf.gz", s2, vr, {"chr1"}, barcodes, cells2, donor_of, vc);
    const std::string t3 = slurp_gz("tmp_sg/cells3.vcf.gz"), t2 = slurp_gz("tmp_sg/cells2.vcf.gz");
    std::remove("tmp_sg/cells3.vcf.gz"); std::remove("tmp_sg/cells2.vcf.gz");
    CHECK(v3 == 1 && v2 == 1 && !t2.empty() && t3 == t2, "uncalled store columns do not reach the cells VCF");
    CHECK(t3.find("GGGG") == std::string::npos, "no uncalled barcode column in the VCF");

    std::filesystem::remove_all("tmp_sg");
    std::cout << (n_fail ? "FAILED" : "all passed") << "\n";
    return n_fail ? 1 : 0;
}
