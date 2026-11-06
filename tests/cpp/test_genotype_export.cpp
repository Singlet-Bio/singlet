// D-52 genotype outputs (genotype_export.h): the donor GT agrees with R12's call_donor_genotype, and the cellsnp-lite layout and the
// multi-sample donor VCF hold the expected sites, cells and counts on a small hand-made matrix.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <zlib.h>

#include "singlet/pileup/genotype_export.h"

namespace G = singlet::genotype;

static int n_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL " << msg << "\n"; ++n_fail; } } while (0)

struct CSC {   // SNPs x barcodes, the shape of SparseAccumulator<uint16_t>::CSCMatrix
    uint32_t nrows = 0, ncols = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<uint16_t> data;
};

static CSC from_dense(const std::vector<std::vector<int>>& m) {   // m[snp][cell]
    CSC c; c.nrows = m.size(); c.ncols = m[0].size(); c.indptr.push_back(0);
    for (uint32_t j = 0; j < c.ncols; ++j) {
        for (uint32_t i = 0; i < c.nrows; ++i) if (m[i][j]) { c.indices.push_back(i); c.data.push_back(m[i][j]); }
        c.indptr.push_back(c.indices.size());
    }
    return c;
}

static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
static std::string slurp_gz(const std::string& p) {
    gzFile f = gzopen(p.c_str(), "rb"); std::string s; char b[4096]; int n;
    while (f && (n = gzread(f, b, sizeof b)) > 0) s.append(b, n);
    if (f) gzclose(f);
    return s;
}
static std::vector<std::string> body_lines(const std::string& s) {
    std::vector<std::string> v; std::stringstream ss(s); std::string l;
    while (std::getline(ss, l)) if (!l.empty() && l[0] != '#') v.push_back(l);
    return v;
}

int main() {
    // 1. GT from donor_fields == call_donor_genotype on every (ad, dp) up to 200; PL of the best genotype is 0; GQ in [0, 99]
    for (uint32_t dp = 0; dp <= 200; ++dp)
        for (uint32_t ad = 0; ad <= dp; ++ad) {
            const std::string f = G::donor_fields(ad, dp);
            const std::string gt = f.substr(0, 3);
            if (gt != singlet::call_donor_genotype(ad, dp)) { std::cout << "FAIL gt ad=" << ad << " dp=" << dp << " " << f << "\n"; ++n_fail; }
        }
    CHECK(G::donor_fields(0, 0) == "./.", "no depth");
    CHECK(G::donor_fields(0, 30).== "0/0:89:30,0:30:0,89,599", "hom-ref 30 " + G::donor_fields(0, 30));
    CHECK(G::donor_fields(1, 10).== "./.:10:9,1:10:0,10,160", "1 of 10 is no-call with GQ 10: " + G::donor_fields(1, 10));

    // 2. 4 SNPs x 5 barcodes; barcodes 0, 2, 3 are called cells; barcode 4 is ambient back-assigned to donor 0
    const std::vector<std::string> snps = {"chr1:100:A>G", "chr1:200:C>T", "chr2:50:G>A", "chr2:60:T>C"};
    const std::vector<std::string> bcs = {"AAA-1", "CCC-1", "GGG-1", "TTT-1", "NNN-1"};
    const CSC dp = from_dense({{10, 50, 10, 0, 40}, {0, 0, 0, 25, 0}, {5, 0, 5, 5, 0}, {0, 0, 1, 0, 0}});
    const CSC ad = from_dense({{0, 50, 10, 0, 40}, {0, 0, 0, 12, 0}, {5, 0, 0, 5, 0}, {0, 0, 1, 0, 0}});
    const std::vector<uint32_t> called = {3, 0, 2};
    const std::string dir = (std::filesystem::temp_directory_path() / "singlet_test_genotype_export").string();
    std::filesystem::remove_all(dir);

    const long long nc = G::write_cellsnp_dir(dir + "/cellsnp", ad, dp, called, bcs, snps);
    // callable over called cells (DP >= 20): chr1:100 (20) and chr2:60? no (1); chr1:200 (25); chr2:50 (15) no
    CHECK(nc == 2, "callable sites " << nc);
    CHECK(slurp(dir + "/cellsnp/cellSNP.samples.tsv") == "AAA-1\nGGG-1\nTTT-1\n", "samples");
    const auto base = body_lines(slurp_gz(dir + "/cellsnp/cellSNP.base.vcf.gz"));
    CHECK(base.size() == 2 && base[0] == "chr1\t100\t.\tA\tG\t.\tPASS\tAD=10;DP=20;OTH=0" &&
          base[1] == "chr1\t200\t.\tC\tT\t.\tPASS\tAD=12;DP=25;OTH=0", "base vcf");
    CHECK(slurp(dir + "/cellsnp/cellSNP.tag.DP.mtx") ==
          "%%MatrixMarket matrix coordinate integer general\n%\n2\t3\t3\n1\t1\t10\n1\t2\t10\n2\t3\t25\n", "DP mtx");
    CHECK(slurp(dir + "/cellsnp/cellSNP.tag.AD.mtx") ==
          "%%MatrixMarket matrix coordinate integer general\n%\n2\t3\t2\n1\t2\t10\n2\t3\t12\n", "AD mtx");
    CHECK(slurp(dir + "/cellsnp/cellSNP.tag.OTH.mtx") == "%%MatrixMarket matrix coordinate integer general\n%\n2\t3\t0\n", "OTH mtx");

    // donors: barcode 0 -> donor0, 2 -> donor1, 3 -> donor0 at prob 0.5 (left out), 4 ambient donor0 (left out), 1 uncalled donor1
    singlet::DemuxResult dm;
    dm.n_donors_k = 2;
    dm.assignments = {{0, 0.99f, 0.f, "donor0", "cell"}, {1, 0.99f, 0.f, "donor1", "cell"}, {1, 0.99f, 0.f, "donor1", "cell"},
                      {0, 0.5f, 0.f, "unassigned", "cell"}, {0, 0.99f, 0.f, "donor0", "ambient"}};
    const long long nv = G::write_donors_vcf(dir + "/donors.vcf.gz", dm, ad, dp, called, snps);
    const std::string vcf = slurp_gz(dir + "/donors.vcf.gz");
    const auto rows = body_lines(vcf);
    CHECK(nv == 3 && rows.size() == 3, "donor sites " << nv);
    CHECK(vcf.find("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdonor0\tdonor1\n") != std::string::npos, "header");
    CHECK(vcf.find("##contig=<ID=chr1>\n##contig=<ID=chr2>\n") != std::string::npos, "contigs");
    if (rows.size() == 3) {
        CHECK(rows[0] == "chr1\t100\tchr1:100:A>G\tA\tG\t.\tPASS\tNS=2\tGT:GQ:AD:DP:PL\t" + G::donor_fields(0, 10) + "\t" + G::donor_fields(10, 10),
              "row0 " << rows[0]);
        CHECK(rows[1] == "chr2\t50\tchr2:50:G>A\tG\tA\t.\tPASS\tNS=2\tGT:GQ:AD:DP:PL\t" + G::donor_fields(5, 5) + "\t" + G::donor_fields(0, 5),
              "row1 " << rows[1]);
        CHECK(rows[2] == "chr2\t60\tchr2:60:T>C\tT\tC\t.\tPASS\tNS=1\tGT:GQ:AD:DP:PL\t./.\t" + G::donor_fields(1, 1), "row2 " << rows[2]);
    }
    singlet::DemuxResult none;
    CHECK(G::write_donors_vcf(dir + "/none.vcf.gz", none, ad, dp, called, snps) == 0, "no donors");

    std::filesystem::remove_all(dir);
    if (n_fail == 0) std::cout << "all passed\n";
    return n_fail ? 1 : 0;
}
