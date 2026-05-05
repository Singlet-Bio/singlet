// test/test_pileup_integration.cpp
// Integration test: create a synthetic BAM via SAM text + samtools conversion,
// then run the pileup engine end-to-end.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <iostream>
#include <unistd.h>

#include "singlet/pileup/pileup_engine.h"
#include "singlet/pileup/mtx_writer.h"

// Create SAM text and convert to BAM via samtools (reliable)
static bool create_test_bam(const std::string& bam_path, const std::string& tmp_dir) {
    std::string sam_path = tmp_dir + "/test.sam";
    {
        std::ofstream f(sam_path);
        if (!f) return false;

        // SAM header
        f << "@HD\tVN:1.6\tSO:unsorted\n";
        f << "@SQ\tSN:chr1\tLN:1000\n";
        f << "@SQ\tSN:chrM\tLN:500\n";

        // Read 1: chr1:101-110 (1-based), barcode AAACCTG…-1, MAPQ=30
        // Sequence pos 6 (0-indexed from read start) = genome pos 106 (1-based) = G (REF)
        f << "read1\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAGGAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";

        // Read 2: same region, different barcode, T at pos 106 = ALT
        f << "read2\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAGTAAAA\t*\tCB:Z:AAACCTGCAGGTTCAC-1\n";

        // Read 3: chrM, barcode 1, tests chrM extraction
        f << "read3\t0\tchrM\t51\t30\t10M\t*\t0\t0\tCCCCCCCCCC\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";

        // Read 4: low MAPQ=5, should be filtered
        f << "read4\t0\tchr1\t101\t5\t10M\t*\t0\t0\tAAAAGTAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";

        // Read 5: unknown barcode, should not contribute to matrices
        f << "read5\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAAAAAAA\t*\tCB:Z:ZZZZZZZZZZZZZZZZ-1\n";

        // Read 6: barcode 1 with ALT at pos 106, tests accumulation
        f << "read6\t0\tchr1\t101\t30\t10M\t*\t0\t0\tAAAAGTAAAA\t*\tCB:Z:AAACCTGAGAAACGAG-1\n";
    }

    // Convert SAM → BAM using samtools
    std::string cmd = "samtools view -bS " + sam_path + " -o " + bam_path + " 2>/dev/null";
    return system(cmd.c_str()) == 0;
}

static void create_test_barcodes(const std::string& path) {
    std::ofstream f(path);
    f << "AAACCTGAGAAACGAG-1\n";
    f << "AAACCTGCAGGTTCAC-1\n";
}

static void create_test_snps(const std::string& path) {
    std::ofstream f(path);
    // SNP at chr1:106 (1-based) → engine converts to 0-based pos 105
    // REF=G, ALT=T
    f << "chr1\t106\tG\tT\n";
    // SNP at chr1:200 — no reads cover this
    f << "chr1\t200\tA\tC\n";
}

int main() {
    std::string tmp = "/tmp/test_pileup_" + std::to_string(getpid());
    std::string bam_path = tmp + "/test.bam";
    std::string bc_path = tmp + "/barcodes.tsv";
    std::string snp_path = tmp + "/snps.tsv";
    std::string out_prefix = tmp + "/output";
    std::string chrm_path = tmp + "/chrm.bam";

    std::string mkdir_cmd = "mkdir -p " + tmp + "/output";
    system(mkdir_cmd.c_str());

    std::cerr << "Creating test data in " << tmp << "\n";

    if (!create_test_bam(bam_path, tmp)) {
        std::cerr << "ERROR: Failed to create test BAM (is samtools available?)\n";
        return 1;
    }
    create_test_barcodes(bc_path);
    create_test_snps(snp_path);

    // Configure engine
    singlet::PileupConfig config;
    config.barcode_path = bc_path;
    config.snp_path = snp_path;
    config.out_chrm_bam = chrm_path;
    config.threads = 1;
    config.min_mapq = 20;
    config.min_baseq = 10;

    singlet::PileupEngine engine(config);
    if (!engine.load_references()) {
        std::cerr << "ERROR: Failed to load references\n";
        return 1;
    }

    auto stats = engine.run(bam_path);

    // ── Verify statistics ──
    int failures = 0;
    auto check = [&](const char* name, uint64_t actual, uint64_t expected) {
        if (actual != expected) {
            std::cerr << "FAIL: " << name << " = " << actual << ", expected " << expected << "\n";
            failures++;
        } else {
            std::cerr << "  OK: " << name << " = " << actual << "\n";
        }
    };

    std::cerr << "\n=== Statistics Checks ===\n";
    check("total_reads", stats.total_reads, 6);
    check("mapped_reads", stats.mapped_reads, 6);
    // read1, read2, read6 pass MAPQ+barcode, read3 passes (chrM+barcode), read4 passes barcode (fails MAPQ later), read5 unknown bc
    check("barcoded_reads", stats.barcoded_reads, 5);
    check("low_mapq", stats.low_mapq, 1);       // read4
    check("no_barcode", stats.no_barcode, 0);    // read5 has CB tag, just not in whitelist
    check("chrm_reads", stats.chrm_reads, 1);    // read3
    check("snp_hits", stats.snp_hits, 3);        // reads 1, 2, 6

    // ── Verify SNP matrices ──
    std::cerr << "\n=== SNP Matrix Checks ===\n";
    auto snp_dp_csc = engine.snp_dp().to_csc();
    auto snp_ad_csc = engine.snp_ad().to_csc();

    uint64_t dp_total = 0;
    for (auto v : snp_dp_csc.data) dp_total += v;
    check("DP total", dp_total, 3);  // reads 1, 2, 6

    uint64_t ad_total = 0;
    for (auto v : snp_ad_csc.data) ad_total += v;
    check("AD total", ad_total, 2);  // reads 2, 6 (ALT=T)

    // ── Export MTX and verify ──
    std::cerr << "\n=== MTX Export ===\n";

    std::vector<std::string> barcodes = {"AAACCTGAGAAACGAG-1", "AAACCTGCAGGTTCAC-1"};
    singlet::export_accumulator(out_prefix, "snp_dp", engine.snp_dp(),
                                engine.snp_names(), barcodes);
    singlet::export_accumulator(out_prefix, "snp_ad", engine.snp_ad(),
                                engine.snp_names(), barcodes);

    auto file_exists = [](const std::string& p) {
        std::ifstream f(p);
        return f.good();
    };
    if (!file_exists(out_prefix + "/snp_dp.mtx.gz") ||
        !file_exists(out_prefix + "/snp_ad.mtx.gz")) {
        std::cerr << "FAIL: MTX files not created\n";
        failures++;
    } else {
        std::cerr << "  OK: All MTX files created\n";
    }

    // Cleanup
    std::string rm_cmd = "rm -rf " + tmp;
    system(rm_cmd.c_str());

    if (failures == 0) {
        std::cerr << "\n=== ALL CHECKS PASSED ===\n";
    } else {
        std::cerr << "\n=== " << failures << " CHECKS FAILED ===\n";
        return 1;
    }

    // ────────────────────────────────────────────────────────────────────────
    // N16: Splice-junction multi-exon read counting test
    // Verify a spliced read (CIGAR 5M10N5M) spanning 2 exons of the same gene
    // is counted exactly once, not twice.
    // ────────────────────────────────────────────────────────────────────────
    std::cerr << "\n=== N16: Splice-Junction Single-Count Test ===\n";
    std::string tmp2 = "/tmp/test_sj_" + std::to_string(getpid());
    system(("mkdir -p " + tmp2).c_str());

    // Mini GTF: gene ENSG1 with two exons on chr1 '+'-strand
    // Exon1: 1-based [101,110] → 0-based [100,110)
    // Intron:               → 0-based [110,120)
    // Exon2: 1-based [121,130] → 0-based [120,130)
    {
        std::ofstream f(tmp2 + "/mini.gtf");
        f << "chr1\t.\tgene\t101\t130\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        f << "chr1\t.\texon\t101\t110\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
        f << "chr1\t.\texon\t121\t130\t.\t+\t.\tgene_id \"ENSG1\"; gene_name \"GENE1\";\n";
    }

    // SAM: three reads on chr1
    //  read_sj: spliced across exon1-intron-exon2, 0-based start=105, CIGAR 5M10N5M
    //           block1=[105,110) in exon1, block2=[120,125) in exon2 → same gene → 1 count
    //  read_fwd: unspliced, entirely within exon1 → 1 count
    //  read_chimeric: unspliced, spans gene boundary (before exon1) → 0 counts
    {
        std::string sam_path2 = tmp2 + "/sj.sam";
        std::ofstream f(sam_path2);
        f << "@HD\tVN:1.6\tSO:unsorted\n";
        f << "@SQ\tSN:chr1\tLN:2000\n";
        // read_sj: 5M10N5M starting at 0-based pos 105 → blocks [105,110) and [120,125)
        // Strand forward (+), so read_reverse=false, gene '+' → expect_reverse=false → OK
        f << "read_sj\t0\tchr1\t106\t30\t5M10N5M\t*\t0\t0\tAAAAAAAAAA\t*\tCB:Z:BARCODE1\tUB:Z:UMI0000001\n";
        // read_fwd: 5M at pos 103 → [103,108) fully in exon1 [100,110) → 1 count
        f << "read_fwd\t0\tchr1\t104\t30\t5M\t*\t0\t0\tAAAAA\t*\tCB:Z:BARCODE1\tUB:Z:UMI0000002\n";
        // read_nogene: pos 10 → [10,15) no exon → 0 counts
        f << "read_nogene\t0\tchr1\t11\t30\t5M\t*\t0\t0\tAAAAA\t*\tCB:Z:BARCODE1\tUB:Z:UMI0000003\n";
    }

    std::string bam_sj = tmp2 + "/sj.bam";
    if (system(("samtools view -bS " + tmp2 + "/sj.sam -o " + bam_sj + " 2>/dev/null").c_str()) != 0) {
        std::cerr << "SKIP N16: samtools not available\n";
        return failures;
    }

    {
        std::ofstream f(tmp2 + "/bc.txt");
        f << "BARCODE1\n";
    }

    singlet::PileupConfig cfg2;
    cfg2.barcode_path = tmp2 + "/bc.txt";
    cfg2.exon_gtf_path = tmp2 + "/mini.gtf";
    cfg2.threads = 1;
    cfg2.min_mapq = 20;
    cfg2.count_introns = true;
    cfg2.count_sj = true;
    cfg2.umi_dedup = true;
    cfg2.stranded = true;

    singlet::PileupEngine eng2(cfg2);
    if (!eng2.load_references()) {
        std::cerr << "FAIL N16: load_references\n";
        return 1;
    }
    auto st2 = eng2.run(bam_sj);

    auto check2 = [&](const char* name, uint64_t actual, uint64_t expected) {
        if (actual != expected) {
            std::cerr << "FAIL: " << name << " = " << actual << ", expected " << expected << "\n";
            failures++;
        } else {
            std::cerr << "  OK: " << name << " = " << actual << "\n";
        }
    };

    // read_sj + read_fwd both contribute 1 exon hit each (different UMIs)
    check2("N16 exon_hits", st2.exon_hits, 2);
    // read_sj spans 2 blocks — still only 1 multigene incident = 0
    check2("N16 multigene_reads", st2.multigene_reads, 0);
    // read_sj should contribute 1 SJ (donor=110, acceptor=120)
    check2("N16 sj_hits", st2.sj_hits, 1);

    system(("rm -rf " + tmp2).c_str());

    if (failures == 0) {
        std::cerr << "\n=== ALL N16 CHECKS PASSED ===\n";
        return 0;
    } else {
        std::cerr << "\n=== " << failures << " N16 CHECKS FAILED ===\n";
        return 1;
    }
}
