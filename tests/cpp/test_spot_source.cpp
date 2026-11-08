// SPDX-License-Identifier: MIT
// tests/cpp/test_spot_source.cpp
//
// fq/spot_source.h builds SRA-style spots from FASTQ files for `singlet download --r1/--r2/--fastq-dir`. Checked here on
// synthetic files: spot assembly across lanes (plain and gzipped), the spot count, seeking back and forward, the bcl2fastq
// folder layout with two lanes, and the Cell Ranger 1.x GemCode layout (interleaved read-RA, 14-nt barcode index read) giving
// the 24-nt barcode read of the 10x-3p-v1 spec.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/spot_source.h"
#include "test_harness.h"

using namespace singlet::fq;
namespace fs = std::filesystem;

namespace {

std::string rnd(std::mt19937_64& rng, size_t n) {
    static const char b[] = "ACGT";
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() % 4];
    return s;
}

struct Rec { std::string seq, qual; };

void write_fq(const std::string& path, const std::vector<Rec>& recs, bool gz) {
    const std::string plain = gz ? path.substr(0, path.size() - 3) : path;
    {
        std::ofstream f(plain);
        for (size_t i = 0; i < recs.size(); ++i) f << "@r" << i << " x\n" << recs[i].seq << "\n+\n" << recs[i].qual << "\n";
    }
    if (gz) std::system(("gzip -f '" + plain + "'").c_str());
}

std::vector<Rec> make(std::mt19937_64& rng, size_t n, size_t len) {
    std::vector<Rec> v;
    for (size_t i = 0; i < n; ++i) {
        std::string q(len, 'F');
        for (auto& c : q) c = static_cast<char>('#' + rng() % 40);
        v.push_back({rnd(rng, len), q});
    }
    return v;
}

std::string spot_seq(const SpotSource& s) { return std::string(s.seq(), s.total_bases()); }

}  // namespace

int main() {
    std::mt19937_64 rng(11);
    const fs::path root = fs::temp_directory_path() / ("spot_source_test_" + std::to_string(rng() % 1000000));
    fs::create_directories(root / "lib");

    // ── bcl2fastq names, two lanes, the second lane gzipped ──
    const auto r1a = make(rng, 5, 28), r2a = make(rng, 5, 90), r1b = make(rng, 3, 28), r2b = make(rng, 3, 90);
    write_fq((root / "lib/pbmc_S1_L001_R1_001.fastq").string(), r1a, false);
    write_fq((root / "lib/pbmc_S1_L001_R2_001.fastq").string(), r2a, false);
    write_fq((root / "lib/pbmc_S1_L002_R1_001.fastq.gz").string(), r1b, true);
    write_fq((root / "lib/pbmc_S1_L002_R2_001.fastq.gz").string(), r2b, true);
    write_fq((root / "lib/pbmc_S1_L001_I1_001.fastq").string(), make(rng, 5, 8), false);

    const auto L = discover_fastq_layout(root.string());
    CHECK(L.kind == "illumina" && L.library == "pbmc", "bcl2fastq folder: library found");
    CHECK(L.spec.find("I1") == std::string::npos, "index reads stay out of the spot by default");

    SpotSource s;
    s.open(L.spec);
    CHECK(s.n_spots() == 8, "spot count is the R1 records of both lanes");
    std::vector<std::string> seen;
    bool layout_ok = true, qual_ok = true;
    for (size_t i = 0; s.read(); ++i) {
        const auto& a = i < 5 ? r1a[i] : r1b[i - 5];
        const auto& b = i < 5 ? r2a[i] : r2b[i - 5];
        layout_ok = layout_ok && s.n_segments() == 2 && s.seg_start()[0] == 0 && s.seg_len()[0] == 28 &&
                    s.seg_start()[1] == 28 && s.seg_len()[1] == 90 && spot_seq(s) == a.seq + b.seq;
        qual_ok = qual_ok && s.qual()[0] == static_cast<uint8_t>(a.qual[0] - 33) && s.qual()[28] == static_cast<uint8_t>(b.qual[0] - 33);
        seen.push_back(spot_seq(s));
    }
    CHECK(seen.size() == 8, "every spot read, lane 1 then lane 2");
    CHECK(layout_ok, "spot = R1 + R2 with SRA segment starts and lengths");
    CHECK(qual_ok, "qualities are raw phred (FASTQ minus 33)");
    s.seek(6);
    CHECK(s.read() && spot_seq(s) == seen[6], "seek back to spot 6 reopens the files");
    s.seek(2);
    CHECK(s.read() && spot_seq(s) == seen[2], "seek back to spot 2");
    s.seek(7);
    CHECK(s.read() && spot_seq(s) == seen[7] && !s.read(), "seek forward to the last spot, then the end");

    // ── explicit spec with an index segment ──
    SpotSource t;
    t.open("fastq:" + (root / "lib/pbmc_S1_L001_I1_001.fastq").string() + "|" + (root / "lib/pbmc_S1_L001_R1_001.fastq").string() +
           "|" + (root / "lib/pbmc_S1_L001_R2_001.fastq").string());
    CHECK(t.read() && t.n_segments() == 3 && t.seg_len()[0] == 8 && t.seg_start()[2] == 36, "three-segment spot (I1, R1, R2)");

    // ── GemCode (Cell Ranger 1.x): RA = cDNA, UMI interleaved; I1 = 8-nt sample index; I2 = 14-nt cell barcode ──
    fs::create_directories(root / "gem/293t");
    const auto cdna = make(rng, 4, 98), umi = make(rng, 4, 10), bc = make(rng, 4, 14);
    std::vector<Rec> ra;
    for (size_t i = 0; i < 4; ++i) { ra.push_back(cdna[i]); ra.push_back(umi[i]); }
    write_fq((root / "gem/293t/read-RA_si-ACGTACGT_lane-001-chunk-001.fastq.gz").string(), ra, true);
    write_fq((root / "gem/293t/read-I1_si-ACGTACGT_lane-001-chunk-001.fastq.gz").string(), make(rng, 4, 8), true);
    write_fq((root / "gem/293t/read-I2_si-ACGTACGT_lane-001-chunk-001.fastq.gz").string(), bc, true);
    const auto G = discover_fastq_layout((root / "gem").string());
    CHECK(G.kind == "gemcode" && G.library == "293t", "GemCode folder: library found");
    SpotSource g;
    g.open(G.spec);
    CHECK(g.n_spots() == 4, "GemCode spot count is half the RA records");
    bool gem_ok = true;
    for (size_t i = 0; g.read(); ++i)
        gem_ok = gem_ok && g.n_segments() == 2 && g.seg_len()[0] == 24 && g.seg_len()[1] == 98 &&
                 spot_seq(g) == bc[i].seq + umi[i].seq + cdna[i].seq;
    CHECK(gem_ok, "GemCode spot = barcode (14) + UMI (10) | cDNA (98)");

    // ── two libraries in one folder need a filter ──
    write_fq((root / "lib/other_S2_L001_R1_001.fastq").string(), make(rng, 2, 28), false);
    write_fq((root / "lib/other_S2_L001_R2_001.fastq").string(), make(rng, 2, 90), false);
    bool threw = false;
    try { discover_fastq_layout((root / "lib").string()); } catch (const std::exception&) { threw = true; }
    CHECK(threw, "two libraries without --fastq-sample is an error");
    CHECK(discover_fastq_layout((root / "lib").string(), "other").library == "other", "--fastq-sample picks one library");

    fs::remove_all(root);
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
