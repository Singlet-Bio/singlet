// Poly(A) site detection and clustering (polya_sites.h), STAR CellReads.stats parsing (star_cell_read_stats.h) and
// junction annotation (sj_features.h).
#include "singlet/pileup/polya_sites.h"
#include "singlet/pileup/sj_features.h"
#include "singlet/pileup/star_cell_read_stats.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(c, msg)                                                           \
    do {                                                                        \
        if (c) { ++g_pass; } else { ++g_fail; std::printf("FAIL: %s\n", msg); } \
    } while (0)

static bam1_t* make_read(int32_t tid, int64_t pos, bool rev, const std::vector<uint32_t>& cigar, const std::string& seq) {
    bam1_t* b = bam_init1();
    std::string qual(seq.size(), 30);
    bam_set1(b, 2, "r", rev ? BAM_FREVERSE : 0, tid, pos, 60, cigar.size(), cigar.data(), -1, -1, 0, seq.size(),
             seq.c_str(), qual.c_str(), 0);
    return b;
}

int main() {
    using namespace singlet;
    // + strand read: 40M then 12 A soft clip -> site at pos + 39
    {
        std::string s = std::string(40, 'C') + std::string(12, 'A');
        bam1_t* b = make_read(3, 1000, false, {bam_cigar_gen(40, BAM_CMATCH), bam_cigar_gen(12, BAM_CSOFT_CLIP)}, s);
        uint64_t k = 0;
        CHECK(polya_site_from_read(b, 6, 0.8, k), "+ strand poly(A) tail detected");
        CHECK(polya_site_tid(k) == 3 && !polya_site_minus(k) && polya_site_pos(k) == 1039, "+ strand site position");
        bam_destroy1(b);
    }
    // - strand read: 10 T soft clip then 40M at pos 5000 -> site at 5000, minus
    {
        std::string s = std::string(10, 'T') + std::string(40, 'G');
        bam1_t* b = make_read(1, 5000, true, {bam_cigar_gen(10, BAM_CSOFT_CLIP), bam_cigar_gen(40, BAM_CMATCH)}, s);
        uint64_t k = 0;
        CHECK(polya_site_from_read(b, 6, 0.8, k), "- strand poly(T) clip detected");
        CHECK(polya_site_minus(k) && polya_site_pos(k) == 5000, "- strand site position");
        bam_destroy1(b);
    }
    // non-A clip and short clip: no site
    {
        std::string s = std::string(40, 'C') + "ACGTACGTAC";
        bam1_t* b = make_read(0, 10, false, {bam_cigar_gen(40, BAM_CMATCH), bam_cigar_gen(10, BAM_CSOFT_CLIP)}, s);
        uint64_t k = 0;
        CHECK(!polya_site_from_read(b, 6, 0.8, k), "mixed clip rejected");
        bam_destroy1(b);
        s = std::string(40, 'C') + "AAAA";
        b = make_read(0, 10, false, {bam_cigar_gen(40, BAM_CMATCH), bam_cigar_gen(4, BAM_CSOFT_CLIP)}, s);
        CHECK(!polya_site_from_read(b, 6, 0.8, k), "4 nt clip rejected");
        bam_destroy1(b);
    }
    // clustering: two nearby positions -> one site; distinct UMIs -> molecules; a lone read dropped
    {
        std::vector<PolyARecord> r;
        r.push_back({polya_site_key(0, false, 100), 0, 0, 11});
        r.push_back({polya_site_key(0, false, 100), 0, 0, 11});  // same molecule
        r.push_back({polya_site_key(0, false, 105), 0, 0, 12});
        r.push_back({polya_site_key(0, false, 100), 0, 1, 13});
        r.push_back({polya_site_key(0, false, 900), 0, 1, 14});  // lone: < 2 molecules
        auto t = build_polya_sites(r, 2, {"chr1"}, {"G1"}, {"Gene1"}, "");
        CHECK(t.nrows == 1, "one cluster kept");
        CHECK(t.names.size() == 1 && t.names[0] == "Gene1:chr1:+:101", "cluster named by mode (1-based)");
        CHECK(t.indptr.size() == 3 && t.data.size() == 2 && t.data[0] == 2 && t.data[1] == 1, "molecules per cell");
        CHECK(t.table_tsv.find("NA\tNA\t-") != std::string::npos, "no genome -> NA priming fields");
    }
    // CellReads.stats parsing
    {
        const char* p = "test_cellreads.stats";
        std::ofstream f(p);
        f << "CB\tcbMatch\tcbPerfect\tcbMMunique\tcbMMmultiple\tgenomeU\tgenomeM\tfeatureU\tfeatureM\texonic\tintronic"
             "\texonicAS\tintronicAS\tmito\tcountedU\tcountedM\tnUMIunique\tnGenesUnique\tnUMImulti\tnGenesMulti\n";
        f << "CBnotInPasslist\t500\t0\t0\t0\t300\t50\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\n";
        f << "AAAC\t100\t90\t10\t0\t70\t10\t50\t5\t40\t15\t3\t2\t5\t40\t5\t30\t20\t0\t0\n";
        f.close();
        auto s = read_star_cell_read_stats(p, {"GGGG", "AAAC-1"});
        CHECK(s.found && s.no_passlist.cb_match == 500, "no-passlist row");
        CHECK(!s.per_barcode[0].present && s.per_barcode[1].present, "barcode matched with -1 suffix");
        CHECK(s.per_barcode[1].unmapped() == 20 && s.per_barcode[1].antisense() == 5 &&
                  s.per_barcode[1].intergenic() == 5,
              "derived unmapped / antisense / intergenic");
        std::remove(p);
    }
    // sj_features
    {
        const char* tab = "test_sj.tab";
        std::ofstream f(tab);
        f << "chr1\t101\t200\t1\t1\t1\t7\t0\t30\n";
        f.close();
        struct C { uint32_t nrows, ncols; std::vector<int32_t> indptr, indices; std::vector<uint16_t> data; } sj;
        sj.nrows = 2; sj.ncols = 2; sj.indptr = {0, 2, 3}; sj.indices = {0, 1, 0}; sj.data = {3, 1, 2};
        uint64_t n = write_sj_features("test_sj.tsv.gz", {"chr1:100-200", "chr1:500-600"}, sj, {1, 0}, tab);
        CHECK(n == 1, "one junction in SJ.out.tab (donor+1 coordinate shift)");
        std::remove(tab);
        std::remove("test_sj.tsv.gz");
    }
    std::printf("test_polya_sites: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
