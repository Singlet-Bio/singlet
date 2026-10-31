// SPDX-License-Identifier: MIT
// tests/cpp/test_molecule_labels.cpp
//
// Every (barcode, gene, UMI) is counted exactly once and labelled spliced,
// unspliced or ambiguous (velocyto / STARsolo Velocyto rules on the collapsed
// gene model). Before this was fixed the engine deduplicated exonic and
// intronic UMIs in separate sets, so a molecule whose reads touched both an
// exon and an intron was counted in exon_counts AND intron_counts.
//
// Part 1 checks DirectionalUmiStore::finalize_molecules() directly.
// Part 2 builds a synthetic BAM (samtools) and runs PileupEngine::run() and
// run_parallel() (2 workers, so a multi-mapper split across chromosomes goes
// through the cross-worker merge) with directional and exact UMI collapse.

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "singlet/pileup/pileup_engine.h"
#include "test_harness.h"

using singlet::DirectionalUmiStore;

namespace {

struct CountingAcc {
    std::map<std::pair<uint32_t, uint32_t>, int> n;  // (feature, bc) -> count
    void increment(uint32_t f, uint32_t bc) { n[{f, bc}]++; }
    int total() const { int t = 0; for (auto& [k, v] : n) t += v; return t; }
};

uint64_t pk(const char* s) { return singlet::umi_pack_2bit(s, 10); }

constexpr uint8_t kE = DirectionalUmiStore::kReadExonOnly | DirectionalUmiStore::kReadTouchesExon;
constexpr uint8_t kSpan = DirectionalUmiStore::kReadIntron | DirectionalUmiStore::kReadTouchesExon;
constexpr uint8_t kI = DirectionalUmiStore::kReadIntron;

void test_store() {
    std::cout << "\n=== Part 1: DirectionalUmiStore::finalize_molecules ===\n";
    DirectionalUmiStore s;
    // gene 0, bc 0
    s.record_read(0, 0, pk("AAAAAAAAAA"), 10, UINT32_MAX, kE, DirectionalUmiStore::kDiagSpliced);   // spliced (junction read)
    s.record_read(0, 0, pk("CACACACACA"), 9, UINT32_MAX, kE);     // exon-only, no junction: ambiguous
    s.record_read(0, 0, pk("CCCCCCCCCC"), 10, 20, kSpan);        // boundary read -> unspliced
    s.record_read(0, 0, pk("GGGGGGGGGG"), 11, UINT32_MAX, kE);   // exon read ...
    s.record_read(0, 0, pk("GGGGGGGGGG"), UINT32_MAX, 21, kI);   // ... + intron read: unspliced (any intron-touching read)
    s.record_read(0, 0, pk("TTTTTTTTTT"), UINT32_MAX, 22, kI);   // unspliced
    // directional: 5 exonic reads of ACGTACGTAC + 1 intronic read one mismatch away
    for (int i = 0; i < 5; ++i) s.record_read(0, 0, pk("ACGTACGTAC"), 12, UINT32_MAX, kE);
    s.record_read(0, 0, pk("ACGTACGTAA"), UINT32_MAX, 23, kI);

    CountingAcc ex, in;
    std::map<uint32_t, int> amb;
    auto st = s.finalize_molecules(ex, in, [&](uint32_t, uint32_t g) { amb[g]++; }, 10, true);
    CHECK(st.spliced == 1, "directional: 1 spliced (junction read, no intron)");
    CHECK(st.unspliced == 4, "directional: 4 unspliced (boundary read, mixed GGG, intronic, merged ACGT)");
    CHECK(st.ambiguous == 1, "directional: 1 ambiguous (exon-only, no junction)");
    CHECK(st.total() == 6, "directional: 6 molecules, each once");
    CHECK(ex.total() == 2, "exon matrix = spliced + ambiguous = 2");
    CHECK(in.total() == 4, "intron matrix = unspliced = 4");
    CHECK(ex.total() + in.total() == static_cast<int>(st.total()), "exon + intron = molecules");
    CHECK(amb[0] == 1, "ambiguous per gene = 1");
    CHECK(st.legacy_double == 3, "legacy double count: boundary, GGG, merged ACGT = 3");

    CountingAcc ex2, in2;
    auto st2 = s.finalize_molecules(ex2, in2, [](uint32_t, uint32_t) {}, 10, false);
    CHECK(st2.total() == 7, "exact: 1-Hamming UMIs stay separate (7 molecules)");
    CHECK(st2.ambiguous == 2 && st2.spliced == 1 && st2.unspliced == 4, "exact: 1 S / 4 U / 2 A");
    CHECK(ex2.total() + in2.total() == 7, "exact: exon + intron = molecules");

    // reads per molecule: the sink sees every emitted molecule once, with the reads merged into it, and the counts do not move
    {
        CountingAcc exr, inr;
        uint64_t reads = 0, mols = 0, hist1 = 0;
        CountingAcc ex3, in3;
        auto st3 = s.finalize_molecules(ex3, in3, [](uint32_t, uint32_t) {}, 10, true,
            [&](bool intronic, uint32_t f, uint32_t bc, uint32_t n) {
                (intronic ? inr : exr).n[{f, bc}] += static_cast<int>(n); reads += n; ++mols; hist1 += (n == 1);
            });
        CHECK(mols == st3.total(), "reads sink: one call per emitted molecule");
        CHECK(reads == 12, "reads sink: 12 reads over the 6 molecules");
        CHECK(exr.total() + inr.total() == 12, "reads sink: exon + intron reads = 12");
        CHECK(hist1 == 4, "reads sink: four single-read molecules (AAAA, CACA, CCCC, TTTT)");
        CHECK(ex3.total() == ex.total() && in3.total() == in.total(), "reads sink: counts unchanged");
    }

    // per_cell_molecule_stats: 12 reads, 6 molecules after directional collapse
    auto pc = s.per_cell_molecule_stats(10);
    CHECK(pc[0].total_reads == 12, "saturation: 12 reads");
    CHECK(pc[0].unique_umis == 6, "saturation: 6 molecules");

    // Read order must not change the output: STAR emits reads in a thread-count-dependent
    // order, and a molecule's exon/intron feature used to be the first one seen.
    struct Rec { uint64_t umi; uint32_t ex, in; uint8_t f; };
    std::vector<Rec> recs = {
        {pk("AAAAAAAAAA"), 12, UINT32_MAX, kE}, {pk("AAAAAAAAAA"), 10, UINT32_MAX, kE},
        {pk("AAAAAAAAAC"), 11, UINT32_MAX, kE}, {pk("CCCCCCCCCC"), UINT32_MAX, 25, kI},
        {pk("CCCCCCCCCC"), UINT32_MAX, 21, kI}, {pk("GGGGGGGGGG"), 13, 22, kSpan},
        {pk("GGGGGGGGTG"), 14, UINT32_MAX, kE}, {pk("TTTTTTTTTT"), 15, UINT32_MAX, kE},
        {pk("TTTTTTTTTA"), 16, UINT32_MAX, kE},
    };
    auto run = [&](bool reversed) {
        DirectionalUmiStore st;
        for (size_t i = 0; i < recs.size(); ++i) {
            const auto& r = recs[reversed ? recs.size() - 1 - i : i];
            st.record_read(0, 0, r.umi, r.ex, r.in, r.f);
        }
        CountingAcc e, n;
        st.finalize_molecules(e, n, [](uint32_t, uint32_t) {}, 10, true);
        return std::make_pair(e.n, n.n);
    };
    CHECK(run(false) == run(true), "read order does not change exon/intron features");
}

// ── Synthetic BAM ─────────────────────────────────────────────────────────
// chr1, '+' strand:
//   G1  gene 101-400, exon1 101-200, exon2 301-400, intron 201-300 (1-based)
//   G2  gene 221-260, one exon, nested in G1's intron on the same strand
// chr2, '+' strand:
//   G3  gene 501-600, one exon
const char* kBC = "AAACCTGAGAAACGAG-1";

std::string sam_line(const std::string& q, int flag, const std::string& chr, int pos, int mapq,
                     const std::string& cigar, int read_len, const std::string& umi, int nh = 1) {
    std::string seq(static_cast<size_t>(read_len), 'A');
    return q + "\t" + std::to_string(flag) + "\t" + chr + "\t" + std::to_string(pos) + "\t" +
           std::to_string(mapq) + "\t" + cigar + "\t*\t0\t0\t" + seq + "\t*\tCB:Z:" + kBC +
           "\tUB:Z:" + umi + "\tNH:i:" + std::to_string(nh) + "\n";
}

bool write_inputs(const std::string& dir) {
    std::ofstream g(dir + "/mini.gtf");
    auto gl = [&](const char* chr, const char* feat, int s, int e, const char* id, const char* nm) {
        g << chr << "\t.\t" << feat << "\t" << s << "\t" << e << "\t.\t+\t.\tgene_id \"" << id
          << "\"; gene_name \"" << nm << "\";\n";
    };
    gl("chr1", "gene", 101, 400, "G1", "GENE1");
    gl("chr1", "exon", 101, 200, "G1", "GENE1");
    gl("chr1", "exon", 301, 400, "G1", "GENE1");
    gl("chr1", "gene", 221, 260, "G2", "GENE2");
    gl("chr1", "exon", 221, 260, "G2", "GENE2");
    gl("chr2", "gene", 501, 600, "G3", "GENE3");
    gl("chr2", "exon", 501, 600, "G3", "GENE3");
    g.close();

    std::ofstream b(dir + "/barcodes.tsv");
    b << kBC << "\n";
    b.close();

    std::ofstream f(dir + "/t.sam");
    f << "@HD\tVN:1.6\tSO:unsorted\n@SQ\tSN:chr1\tLN:1000\n@SQ\tSN:chr2\tLN:1000\n";
    // M1 exonic only, no junction -> ambiguous (alevin-fry USA)
    f << sam_line("m1", 0, "chr1", 120, 255, "20M", 20, "AAAAAAAAAA");
    // M2 one read across the exon1/intron boundary -> unspliced
    f << sam_line("m2", 0, "chr1", 191, 255, "20M", 20, "CCCCCCCCCC");
    // M3 exonic read + intronic read of the same UMI -> ambiguous
    f << sam_line("m3a", 0, "chr1", 320, 255, "20M", 20, "GGGGGGGGGG");
    f << sam_line("m3b", 0, "chr1", 270, 255, "20M", 20, "GGGGGGGGGG");
    // M4 intronic only -> unspliced
    f << sam_line("m4", 0, "chr1", 205, 255, "10M", 10, "TTTTTTTTTT");
    // M5 spliced exon1 -> exon2 -> spliced
    f << sam_line("m5", 0, "chr1", 196, 255, "5M100N5M", 10, "ACACACACAC");
    // M6 inside G2's exon (also inside G1's intron) -> G2 spliced only
    f << sam_line("m6", 0, "chr1", 230, 255, "10M", 10, "AGAGAGAGAG");
    // M7 5 exonic reads of UMI X + 1 intronic read of X' (1 mismatch):
    //    directional -> one ambiguous molecule; exact -> spliced + unspliced
    for (int i = 0; i < 5; ++i)
        f << sam_line("m7_" + std::to_string(i), 0, "chr1", 140, 255, "20M", 20, "ACGTACGTAC");
    f << sam_line("m7x", 0, "chr1", 280, 255, "10M", 10, "ACGTACGTAA");
    // M8 unique exonic read + a gene-unique multi-mapper (NH=2) whose primary
    //    is intronic on chr1 and secondary intergenic on chr2 -> ambiguous
    f << sam_line("m8a", 0, "chr1", 150, 255, "20M", 20, "CTCTCTCTCT");
    f << sam_line("m8b", 0, "chr1", 210, 3, "10M", 10, "CTCTCTCTCT", 2);
    f << sam_line("m8b", 256, "chr2", 100, 3, "10M", 10, "CTCTCTCTCT", 2);
    // M9 multi-mapper G1 exon1 / G3 exon -> gene-ambiguous, not counted
    f << sam_line("m9", 0, "chr1", 160, 3, "10M", 10, "GAGAGAGAGA", 2);
    f << sam_line("m9", 256, "chr2", 550, 3, "10M", 10, "GAGAGAGAGA", 2);
    f.close();

    std::string cmd = "samtools sort -o " + dir + "/t.bam " + dir + "/t.sam 2>/dev/null && samtools index " +
                      dir + "/t.bam";
    return std::system(cmd.c_str()) == 0;
}

struct GeneCounts { int exon = 0, intron = 0, amb = 0; };

std::map<std::string, GeneCounts> by_gene(const singlet::PileupEngine& e) {
    std::map<std::string, GeneCounts> out;
    const auto& gm = e.gene_model();
    auto ex = e.exons().to_csc();
    for (size_t k = 0; k < ex.data.size(); ++k)
        out[gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(ex.indices[k]))]].exon += ex.data[k];
    auto in = e.introns().to_csc();
    for (size_t k = 0; k < in.data.size(); ++k)
        out[gm.gene_ids()[gm.intron_to_gene(static_cast<uint32_t>(in.indices[k]))]].intron += in.data[k];
    auto am = e.ambiguous().to_csc();
    for (size_t k = 0; k < am.data.size(); ++k)
        out[gm.gene_ids()[static_cast<uint32_t>(am.indices[k])]].amb += am.data[k];
    return out;
}

void run_case(const std::string& dir, bool parallel, bool directional) {
    std::string tag = std::string(parallel ? "run_parallel" : "run") +
                      (directional ? " / directional" : " / exact");
    std::cout << "\n=== Part 2: " << tag << " ===\n";
    singlet::PileupConfig cfg;
    cfg.barcode_path = dir + "/barcodes.tsv";
    cfg.exon_gtf_path = dir + "/mini.gtf";
    cfg.threads = 2;
    cfg.umi_len = 10;
    cfg.umi_dedup_directional = directional;
    cfg.count_sj = false;
    cfg.aim_ancestry = false;
    cfg.count_vdj = false;
    cfg.intron_rules = 0;  // these cases test the molecule labels of the unrestricted pileup; rules have their own tests
    singlet::PileupEngine e(cfg);
    if (!e.load_references()) { CHECK(false, tag + ": load_references"); return; }
    auto st = parallel ? e.run_parallel(dir + "/t.bam", 2) : e.run(dir + "/t.bam");
    auto g = by_gene(e);

    // G1 molecules (alevin-fry USA semantics): M5 S (junction read); M1 A (exon-only, no junction); M2, M4, M3, M8 U (any
    // intron-touching read, M3 and M8 mix exon and intron reads); M7 U merged (directional) or A (exon UMI) + U (exact)
    const int s = 1, u = 5, a = directional ? 1 : 2;
    CHECK(g["G1"].exon == s + a, tag + ": G1 exon_counts = spliced + ambiguous");
    CHECK(g["G1"].intron == u, tag + ": G1 intron_counts = unspliced");
    CHECK(g["G1"].amb == a, tag + ": G1 ambiguous_counts");
    CHECK(g["G2"].exon == 1 && g["G2"].intron == 0, tag + ": nested G2 read counted once, for G2");
    CHECK(g["G3"].exon == 0, tag + ": gene-ambiguous multi-mapper not counted");
    CHECK(st.molecules_spliced == static_cast<uint64_t>(s), tag + ": spliced molecules");
    CHECK(st.molecules_unspliced == static_cast<uint64_t>(u), tag + ": unspliced molecules");
    CHECK(st.molecules_ambiguous == static_cast<uint64_t>(a + 1), tag + ": ambiguous molecules (incl. G2: exon-only, no junction)");
    CHECK(st.umi_unique == st.molecules_spliced + st.molecules_unspliced + st.molecules_ambiguous,
          tag + ": umi_unique = molecules");
    // pre-fix double count: M2 (boundary), M3, M8, and M7 when merged
    CHECK(st.molecules_legacy_double == static_cast<uint64_t>(directional ? 4 : 3),
          tag + ": legacy double-counted molecules");
    CHECK(st.cross_gene_intron_reads == 1, tag + ": M6 lies in another gene's intron");
    {   // unspliced molecules that also have an exon-only read: M3, M8 and the merged M7 (directional) / M3, M8 (exact)
        const auto extra = e.legacy_exon_extra();
        uint64_t tot = 0;
        for (auto v : extra) tot += v;
        CHECK(tot == static_cast<uint64_t>(directional ? 3 : 2), tag + ": legacy exonic extra (mixed unspliced molecules)");
        // the same molecules per feature (export.h's pre-USA split: legacy exon = exon + mixed_exon, legacy intron = intron -
        // mixed_intron): all in G1, the column sums are legacy_exon_extra, and the legacy split is the old labels' (exon-only
        // read and intron read = ambiguous, in the exon matrix): G1 exon M5, M1 + M3, M8, M7 / M5, M1, M7-exon + M3, M8 = 5
        const auto& gm = e.gene_model();
        const auto mx = e.mixed_exon().to_csc(), mi = e.mixed_intron().to_csc();
        std::map<std::string, int> gx, gi;
        for (size_t k = 0; k < mx.data.size(); ++k) gx[gm.gene_ids()[gm.exon_to_gene(static_cast<uint32_t>(mx.indices[k]))]] += mx.data[k];
        for (size_t k = 0; k < mi.data.size(); ++k) gi[gm.gene_ids()[gm.intron_to_gene(static_cast<uint32_t>(mi.indices[k]))]] += mi.data[k];
        const int nm = directional ? 3 : 2;
        CHECK(gx.size() == 1 && gx["G1"] == nm && gi.size() == 1 && gi["G1"] == nm, tag + ": mixed_exon / mixed_intron hold the mixed molecules, in G1");
        bool cols = mx.ncols == extra.size();
        for (uint32_t j = 0; cols && j < mx.ncols; ++j) {
            uint64_t c = 0; for (int32_t k = mx.indptr[j]; k < mx.indptr[j + 1]; ++k) c += mx.data[k];
            cols = c == extra[j];
        }
        CHECK(cols, tag + ": mixed_exon column sums = legacy_exon_extra");
        CHECK(g["G1"].exon + gx["G1"] == 5 && g["G1"].intron - gi["G1"] == u - nm, tag + ": legacy split = the pre-USA labels' (G1 exon 5)");
    }
    // reads-per-UMI matrices: same molecules, uint32 reads; the histogram holds each emitted molecule once
    uint64_t hist_mols = 0;
    for (const auto& [bc, h] : e.reads_per_umi_hist()) for (uint32_t v : h) hist_mols += v;
    CHECK(hist_mols == st.umi_unique, tag + ": reads_per_umi histogram holds every molecule once");
    const auto er = e.exon_reads().to_csc(), ir = e.intron_reads().to_csc();
    uint64_t rd = 0; for (auto v : er.data) rd += v; for (auto v : ir.data) rd += v;
    CHECK(rd >= st.umi_unique, tag + ": reads >= molecules");
    const auto ec = e.exons().to_csc();
    CHECK(er.indices == ec.indices && er.indptr == ec.indptr, tag + ": exon_reads has the exon_counts sparsity");
    // the same run with the matrix off gives identical counts
    singlet::PileupConfig off = cfg; off.reads_matrix = false;
    singlet::PileupEngine e2(off);
    if (e2.load_references()) {
        if (parallel) e2.run_parallel(dir + "/t.bam", 2); else e2.run(dir + "/t.bam");
        const auto ec2 = e2.exons().to_csc();
        CHECK(ec2.indices == ec.indices && ec2.data == ec.data && ec2.indptr == ec.indptr, tag + ": counts identical with the reads matrix off");
        CHECK(e2.exon_reads().nnz_raw() == 0 && e2.reads_per_umi_hist().empty(), tag + ": matrix off writes nothing");
    }
}

}  // namespace

int main() {
    test_store();

    std::string dir = "/tmp/test_molecule_labels_" + std::to_string(getpid());
    std::system(("mkdir -p " + dir).c_str());
    if (!write_inputs(dir)) {
        std::cout << "SKIP Part 2: samtools not available\n";
    } else {
        for (bool par : {false, true})
            for (bool dir_dedup : {true, false}) run_case(dir, par, dir_dedup);
    }
    std::system(("rm -rf " + dir).c_str());

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
