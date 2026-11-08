// SPDX-License-Identifier: MIT
// test_te_counts.cpp - te_counts.h: TE GTF loading, the read assignment rule (largest subfamily overlap, exons win, ties dropped,
// locus only for unique reads) and the order-free molecule tables. Then the scte rule (port of scTE's exclusive index, every-overlap
// assignment and leftmost-read molecules) and the solote rule (SoloTE method: gene-assigned reads stay with genes).
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

// TE GTF for the scTE index tests: one locus per case
static std::string write_te_gtf2() {
    const std::string p = "/tmp/test_te_counts2_" + std::to_string(getpid()) + ".gtf.gz";
    gzFile f = gzopen(p.c_str(), "wb");
    auto line = [&](const char* chr, int s, int e, const char* sub, const char* id) {
        char b[512];
        std::snprintf(b, sizeof b, "%s\trmsk\texon\t%d\t%d\t1\t+\t.\tgene_id \"%s\"; transcript_id \"%s\"; family_id \"F\"; class_id \"C\";\n",
                      chr, s, e, sub, id);
        gzputs(f, b);
    };
    line("chr1", 101, 400, "T0", "l0");      // [100, 400): on a protein_coding exon -> masked
    line("chr1", 1001, 1300, "T1", "l1");    // [1000, 1300): on an lncRNA exon only -> kept (scTE: 'lncRNA' is not 'lincRNA')
    line("chr1", 3001, 3100, "T2", "l2");    // [3000, 3100): exon 2950-3001 shares base 3001, scTE's test misses it -> kept
    line("chr1", 4001, 4100, "T3", "l3");    // [4000, 4100): exon 4099-4200 overlaps by bases 4099-4100 -> masked
    line("chr1", 5001, 5100, "T4", "l4");    // [5000, 5100): on a lincRNA exon -> masked
    line("chrUn_KI270302v1", 1, 50, "T5", "l5");  // contig scTE drops -> masked
    line("KI270728.1", 1, 50, "T6", "l6");   // contig without "chr" -> kept
    line("chr2", 101, 200, "T7", "l7");      // gene line without gene_name, and a gene on a non-main contig -> kept
    line("chr3", 101, 200, "T8", "l8");      // gene GTF names the contig "3" -> masked (scTE matches without "chr")
    gzclose(f);
    return p;
}

static std::string write_gene_gtf() {
    const std::string p = "/tmp/test_te_genes_" + std::to_string(getpid()) + ".gtf.gz";
    gzFile f = gzopen(p.c_str(), "wb");
    auto line = [&](const char* chr, const char* feat, int s, int e, const char* attrs) {
        char b[512];
        std::snprintf(b, sizeof b, "%s\tsrc\t%s\t%d\t%d\t.\t+\t.\t%s\n", chr, feat, s, e, attrs);
        gzputs(f, b);
    };
    const char* pc = "gene_id \"G1\"; gene_name \"PC1\"; gene_type \"protein_coding\";";
    line("chr1", "gene", 150, 4300, pc);   // gene lines are not exons
    line("chr1", "exon", 150, 160, pc);
    line("chr1", "exon", 2950, 3001, pc);
    line("chr1", "exon", 4099, 4200, pc);
    line("chr1", "exon", 1100, 1150, "gene_id \"G2\"; gene_name \"NC1\"; gene_type \"lncRNA\";");
    line("chr1", "exon", 1250, 1260, "gene_id \"G2\"; gene_name \"NC1\"; gene_type \"lncRNA\";");
    line("chr1", "exon", 5050, 5060, "gene_id \"G3\"; gene_name \"LI1\"; gene_type \"lincRNA\";");
    line("chr2", "exon", 120, 180, "gene_id \"G4\"; gene_type \"protein_coding\";");
    line("3", "exon", 150, 160, "gene_id \"G6\"; gene_name \"PC3\"; gene_type \"protein_coding\";");   // Ensembl contig name
    line("chr2_alt", "exon", 120, 180, "gene_id \"G5\"; gene_name \"ALT\"; gene_type \"protein_coding\";");
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

    // ---- scTE port: contig filter and exclusive index ----
    CHECK(TeModel::scte_main_contig("chr1") && TeModel::scte_main_contig("19") && TeModel::scte_main_contig("chrX") &&
              TeModel::scte_main_contig("chrM") && !TeModel::scte_main_contig("chrUn_KI270302v1") && !TeModel::scte_main_contig("KI270728.1") &&
              !TeModel::scte_main_contig("chr50"),
          "scTE contig filter");
    const std::string tg = write_te_gtf2(), gg = write_gene_gtf();
    {
        TeModel x;
        CHECK(x.load_gtf(tg) && x.n_loci() == 9, "load index-test GTF");
        CHECK(x.mask_scte_exclusive("/nonexistent.gtf") == -1, "unreadable gene GTF");
        const long n = x.mask_scte_exclusive(gg);
        CHECK(x.scte_masked(0), "TE on a protein_coding exon masked");
        CHECK(!x.scte_masked(1), "TE on an lncRNA exon kept");
        CHECK(!x.scte_masked(2), "scTE's one-base boundary kept");
        CHECK(x.scte_masked(3), "two-base overlap masked");
        CHECK(x.scte_masked(4), "lincRNA exon masks");
        CHECK(x.scte_masked(5), "chrUn contig masked");
        CHECK(!x.scte_masked(6), "contig without chr kept");
        CHECK(!x.scte_masked(7), "no gene_name and non-main gene contigs ignored");
        CHECK(x.scte_masked(8), "Ensembl-named gene contig masks the chr-named TE");
        CHECK(n == 5, "5 loci masked");
        // nointron: NC1's span 1100-1260 now covers T1
        TeModel y;
        y.load_gtf(tg);
        y.mask_scte_exclusive(gg, true);
        CHECK(y.scte_masked(1) && y.scte_masked(0), "nointron: gene spans mask TEs in introns");

        // scte records: every unmasked overlap, exons ignored
        x.resolve_chroms({{"chr1", 0}, {"chr2", 1}, {"KI270728.1", 2}});
        using B = std::vector<std::pair<int32_t, int32_t>>;
        std::vector<uint32_t> s1, s2;
        std::vector<TeRecord> out;
        auto exonic = [](int32_t, int32_t) { return true; };
        auto no_genes = [](int32_t, int32_t, std::vector<uint32_t>& v) { v.clear(); };
        x.records(TeRule::kScte, 0, B{{150, 240}}, true, 3, 77, exonic, no_genes, s1, s2, out);
        CHECK(out.empty(), "scte: masked locus gives no record");
        x.records(TeRule::kScte, 0, B{{1050, 1150}}, true, 3, 77, exonic, no_genes, s1, s2, out);
        CHECK(out.size() == 1 && out[0].fam == 1 && out[0].locus == 1 && out[0].rule == TeRule::kScte && out[0].pos == 1050,
              "scte: exonic read still counts for the kept TE");
        x.records(TeRule::kSinglet, 0, B{{1050, 1150}}, true, 3, 77, exonic, no_genes, s1, s2, out);
        CHECK(out.size() == 1, "singlet: exonic read stays with the gene");
    }
    {
        // every overlapping locus: [1200,1290) touches L1HS (locus 1) and AluY_dup2 (locus 2); the singlet rule keeps only L1HS
        using B = std::vector<std::pair<int32_t, int32_t>>;
        std::vector<uint32_t> s1, s2;
        std::vector<TeRecord> out;
        auto no_exon2 = [](int32_t, int32_t) { return false; };
        auto no_genes = [](int32_t, int32_t, std::vector<uint32_t>& v) { v.clear(); };
        m.records(TeRule::kScte, 0, B{{1200, 1290}}, false, 1, 5, no_exon2, no_genes, s1, s2, out);
        CHECK(out.size() == 2 && out[0].fam == 1 && out[0].locus == UINT32_MAX && out[1].fam == 0, "scte: every overlap, multi -> no locus");
        out.clear();
        m.records(TeRule::kScte, 0, B{{3050, 3150}}, true, 1, 5, no_exon2, no_genes, s1, s2, out);
        CHECK(out.size() == 2 && out[0].locus == 3 && out[1].locus == 4, "scte: a subfamily tie counts for both");

        // solote: a unique read inside one gene's exon on the sense strand stays with the gene
        auto one_gene = [](int32_t, int32_t, std::vector<uint32_t>& v) { v.assign(1, 7); };
        auto two_genes = [](int32_t, int32_t, std::vector<uint32_t>& v) { v.assign({7, 8}); };
        out.clear();
        m.records(TeRule::kSoloTE, 0, B{{150, 240}}, true, 1, 5, no_exon2, one_gene, s1, s2, out);
        CHECK(out.empty(), "solote: gene-assigned unique read stays with the gene");
        m.records(TeRule::kSoloTE, 0, B{{150, 240}}, false, 1, 5, no_exon2, one_gene, s1, s2, out);
        CHECK(out.size() == 1 && out[0].fam == 0 && out[0].locus == UINT32_MAX, "solote: multi-mapped read counts at the subfamily");
        out.clear();
        m.records(TeRule::kSoloTE, 0, B{{150, 240}}, true, 1, 5, no_exon2, two_genes, s1, s2, out);
        CHECK(out.size() == 1 && out[0].locus == 0, "solote: two-gene read is not gene-assigned; unique -> locus");
        out.clear();
        m.records(TeRule::kSoloTE, 0, B{{1200, 1290}}, true, 1, 5, no_exon2, no_genes, s1, s2, out);
        CHECK(out.size() == 2 && out[0].locus == 1 && out[1].locus == 2, "solote: every overlapping locus");
        // gene assignment over a spliced read: the genes common to all blocks
        auto per_block = [](int32_t s, int32_t, std::vector<uint32_t>& v) {
            if (s < 1000) v.assign({7, 8}); else v.assign({8, 9});
        };
        CHECK(te_gene_assigned(B{{100, 150}, {1100, 1150}}, true, per_block, s1, s2), "spliced: one common gene");
        CHECK(!te_gene_assigned(B{{100, 150}}, true, per_block, s1, s2), "two genes: not assigned");
        CHECK(!te_gene_assigned(B{{100, 150}, {1100, 1150}}, false, per_block, s1, s2), "multi-mapped: never assigned");
    }
    {
        // scTE molecule rule: one (cell, UMI) counts at its leftmost read; records of other rules are ignored
        std::vector<TeRecord> r = {
            {0, 0, 21, 0, TeRule::kScte, 0, 150},          // leftmost read of UMI 21 in cell 0
            {1, 0, 21, 1, TeRule::kScte, 0, 1200},         // same molecule, later read: dropped
            {0, 0, 21, 2, TeRule::kScte, 0, 150},          // same leftmost read, second locus: kept (AluY again: one molecule)
            {1, 0, 22, 1, TeRule::kScte, 0, 1200},         // another UMI
            {1, 1, 21, 1, TeRule::kScte, 0, 1200},         // UMI 21 in another cell: its own molecule
            {2, 0, 0, 3, TeRule::kScte, 0, 3050},          // no UMI: kept
            {2, 0, 0, 3, TeRule::kScte, 0, 3050},          // no UMI: its own molecule
            {3, 0, 30, UINT32_MAX, TeRule::kSinglet, 0, 1},  // other rule
        };
        std::vector<std::string> t2c2{"chr1", "chr2"};
        auto a2 = build_te_tables(r, m, 2, t2c2, TeRule::kScte);
        CHECK(a2.n_records == 6, "scte: 6 records after the leftmost-read rule");
        CHECK(a2.fam.indptr == (std::vector<int32_t>{0, 3, 4}), "scte fam indptr");
        CHECK(a2.fam.indices == (std::vector<int32_t>{0, 1, 2, 1}) && a2.fam.data == (std::vector<uint16_t>{1, 1, 2, 1}), "scte fam molecules");
        auto a3 = build_te_tables(r, m, 2, t2c2, TeRule::kSinglet);
        CHECK(a3.n_records == 1 && a3.fam_molecules == 1 && a3.fam.indices == (std::vector<int32_t>{3}), "singlet table sees its own records");
        std::mt19937 rng2(11);
        for (int k = 0; k < 20; ++k) {
            std::shuffle(r.begin(), r.end(), rng2);
            auto b = build_te_tables(r, m, 2, t2c2, TeRule::kScte);
            CHECK(b.fam.indptr == a2.fam.indptr && b.fam.indices == a2.fam.indices && b.fam.data == a2.fam.data && b.loci_tsv == a2.loci_tsv,
                  "scte order-free");
        }
        TeRule rr;
        CHECK(te_rule_parse("scte", rr) && rr == TeRule::kScte && te_rule_parse("solote", rr) && rr == TeRule::kSoloTE &&
                  !te_rule_parse("x", rr) && std::string(te_rule_name(TeRule::kSinglet)) == "singlet",
              "rule names");
    }
    std::remove(tg.c_str());
    std::remove(gg.c_str());

    std::remove(gtf.c_str());
    if (g_fail) { std::printf("%d failures\n", g_fail); return 1; }
    std::printf("ALL PASS: test_te_counts\n");
    return 0;
}
