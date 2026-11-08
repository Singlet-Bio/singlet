// SPDX-License-Identifier: MIT
// test_te_counts.cpp - te_counts.h: TE GTF loading, the read assignment rule (largest subfamily overlap, exons win, ties dropped,
// locus only for unique reads) and the order-free molecule tables.
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet/pileup/te_counts.h"

using namespace singlet;

static int g_fail = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++g_fail; } } while (0)

static std::string write_gtf() {
    const std::string p = "/tmp/test_te_counts_" + std::to_string(getpid()) + ".gtf.gz";
    gzFile f = gzopen(p.c_str(), "wb");
    auto line = [&](const char* chr, int s, int e, char st, const char* sub, const char* id, const char* fam, const char* cls) {
        char b[512];
        std::snprintf(b, sizeof b, "%s\thg38_rmsk\texon\t%d\t%d\t1\t%c\t.\tgene_id \"%s\"; transcript_id \"%s\"; family_id \"%s\"; class_id \"%s\";\n",
                      chr, s, e, st, sub, id, fam, cls);
        gzputs(f, b);
    };
    gzputs(f, "# header\n");
    line("chr1", 101, 400, '+', "AluY", "AluY_dup1", "Alu", "SINE");      // [100, 400)
    line("chr1", 1001, 1300, '-', "L1HS", "L1HS_dup1", "L1", "LINE");     // [1000, 1300)
    line("chr1", 1251, 1500, '+', "AluY", "AluY_dup2", "Alu", "SINE");    // [1250, 1500)
    line("chr1", 3001, 3100, '+', "MER4", "MER4_dup1", "ERV1", "LTR?");   // [3000, 3100)
    line("chr1", 3101, 3200, '+', "MER5", "MER5_dup1", "hAT", "DNA");     // [3100, 3200)
    line("2", 501, 600, '+', "AluY", "AluY_dup3", "Alu", "SINE");         // contig named without "chr"
    gzclose(f);
    return p;
}

int main() {
    const std::string gtf = write_gtf();
    TeModel m;
    CHECK(m.load_gtf(gtf), "load");
    CHECK(m.n_loci() == 6 && m.n_families() == 4, "6 loci in 4 subfamilies");
    CHECK(m.family_names()[2] == "MER4:ERV1:LTR", "names: subfamily:family:class, trailing ? dropped");
    std::unordered_map<std::string, int32_t> c2t{{"chr1", 0}, {"chr2", 1}};
    m.resolve_chroms(c2t);
    CHECK(m.n_resolved_contigs() == 2, "chr prefix added for contig 2");

    std::vector<uint32_t> scr;
    auto no_exon = [](int32_t, int32_t) { return false; };
    uint32_t fam = 9, locus = 9;
    using B = std::vector<std::pair<int32_t, int32_t>>;
    CHECK(m.assign(0, B{{150, 240}}, true, no_exon, fam, locus, scr) && fam == 0 && locus == 0, "unique read inside AluY_dup1");
    CHECK(m.assign(0, B{{150, 240}}, false, no_exon, fam, locus, scr) && fam == 0 && locus == UINT32_MAX, "multi-mapped: subfamily only");
    CHECK(!m.assign(0, B{{500, 590}}, true, no_exon, fam, locus, scr), "no TE");
    CHECK(!m.assign(0, B{{150, 240}}, true, [](int32_t s, int32_t e) { return s < 200 && e > 180; }, fam, locus, scr), "exonic read stays with the gene");
    // L1HS covers [1200,1300) = 100 bases, AluY_dup2 covers [1250,1290) = 40 of a [1200,1290) read: L1HS wins
    CHECK(m.assign(0, B{{1200, 1290}}, true, no_exon, fam, locus, scr) && fam == 1 && locus == 1, "largest overlap wins");
    // [3050,3150): 50 bases MER4, 50 bases MER5 -> tie, dropped
    CHECK(!m.assign(0, B{{3050, 3150}}, true, no_exon, fam, locus, scr), "tie dropped");
    // spliced read: two blocks summed per subfamily
    CHECK(m.assign(0, B{{350, 400}, {1260, 1280}}, true, no_exon, fam, locus, scr) && fam == 0 && locus == 0, "blocks summed; locus by its own bases");
    CHECK(m.assign(1, B{{520, 560}}, true, no_exon, fam, locus, scr) && fam == 0 && locus == 5, "second contig");

    // tables: molecules = distinct (cell, feature, UMI); records in any order give the same matrices
    std::vector<TeRecord> recs = {
        {0, 0, 11, 0}, {0, 0, 11, 0}, {0, 0, 12, UINT32_MAX}, {0, 0, 0, UINT32_MAX}, {0, 0, 0, UINT32_MAX},  // cell 0: AluY 2 UMI + 2 no-UMI
        {1, 2, 13, 1}, {1, 2, 14, 1}, {0, 2, 13, 5},                                                    // cell 2: L1HS 2, AluY 1
    };
    std::vector<std::string> t2c{"chr1", "chr2"};
    auto a = build_te_tables(recs, m, 3, t2c);
    CHECK(a.fam.nrows == 4 && a.fam.ncols == 3, "fam shape");
    CHECK(a.fam.indptr == (std::vector<int32_t>{0, 1, 1, 3}), "fam indptr");
    CHECK(a.fam.indices == (std::vector<int32_t>{0, 0, 1}) && a.fam.data == (std::vector<uint16_t>{4, 1, 2}), "fam molecules");
    CHECK(a.fam_molecules == 7, "fam total");
    CHECK(a.locus.nrows == 3 && a.locus.names == (std::vector<std::string>{"AluY_dup1", "L1HS_dup1", "AluY_dup3"}), "loci with a unique record");
    CHECK(a.locus.indptr == (std::vector<int32_t>{0, 1, 1, 3}) && a.locus.data == (std::vector<uint16_t>{1, 2, 1}), "locus molecules");
    CHECK(a.loci_tsv.find("L1HS_dup1\tL1HS\tchr1\t1001\t1300\t-\t2\t1\n") != std::string::npos, "locus table row");
    std::mt19937 rng(7);
    for (int k = 0; k < 20; ++k) {
        std::shuffle(recs.begin(), recs.end(), rng);
        auto b = build_te_tables(recs, m, 3, t2c);
        CHECK(b.fam.indptr == a.fam.indptr && b.fam.indices == a.fam.indices && b.fam.data == a.fam.data && b.locus.indices == a.locus.indices &&
                  b.locus.data == a.locus.data && b.loci_tsv == a.loci_tsv,
              "order-free");
    }
    std::remove(gtf.c_str());
    if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
    std::printf("ALL PASS: test_te_counts\n");
    return 0;
}
