// SPDX-License-Identifier: MIT
// tests/cpp/test_run_screen.cpp
//
// Per-run read screen (include/singlet/fq/run_screen.h): synthetic runs of one sample — 10x 3' v3
// gene expression, 10x 5' v3, a hashtag (HTO) run, a 3' v2 run, a run whose R1 matches no whitelist —
// and the keep/drop decision. Also the FASTQ reader path and the JSON.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/run_screen.h"
#include "test_harness.h"

using namespace singlet::fq;

namespace {

std::mt19937 rng(20261024);

std::string rand_seq(size_t n) {
    static const char b[4] = {'A', 'C', 'G', 'T'};
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() & 3];
    return s;
}

std::vector<uint8_t> num(const std::string& s) {
    std::vector<uint8_t> v;
    for (char c : s) v.push_back(singlet::fq::nuc::ascii_to_num(c));
    return v;
}

ScreenSpot spot(const std::string& r1, const std::string& r2) {
    ScreenSpot s;
    s.r1_seq = num(r1);
    s.r2_seq = num(r2);
    s.r1_len = static_cast<uint16_t>(r1.size());
    s.r2_len = static_cast<uint16_t>(r2.size());
    return s;
}

std::vector<std::string> wl_v3, wl_v2;
std::vector<std::string> hto_tags;

// 3' GEX: R1 = BC + UMI12 (28 bp), R2 = random cDNA (90 bp). mm1 = share of barcodes with one mismatch.
std::vector<ScreenSpot> gex3(size_t n, const std::vector<std::string>& wl, size_t umi, double mm1 = 0.0) {
    std::vector<ScreenSpot> v;
    for (size_t i = 0; i < n; ++i) {
        std::string bc = wl[rng() % wl.size()];
        if (std::uniform_real_distribution<double>(0, 1)(rng) < mm1) bc[rng() % bc.size()] = bc[0] == 'A' ? 'C' : 'A';
        v.push_back(spot(bc + rand_seq(umi), rand_seq(90)));
    }
    return v;
}

// 5' GEX: R1 = BC + UMI12 + TSO tail + cDNA (50 bp)
std::vector<ScreenSpot> gex5(size_t n) {
    std::vector<ScreenSpot> v;
    for (size_t i = 0; i < n; ++i)
        v.push_back(spot(wl_v3[rng() % wl_v3.size()] + rand_seq(12) + "TTTCTTATATGGG" + rand_seq(9), rand_seq(90)));
    return v;
}

// HTO: R1 = BC + UMI, R2 = 15-base tag + poly(A) + random
std::vector<ScreenSpot> hto(size_t n) {
    std::vector<ScreenSpot> v;
    for (size_t i = 0; i < n; ++i)
        v.push_back(spot(wl_v3[rng() % wl_v3.size()] + rand_seq(12),
                         hto_tags[rng() % hto_tags.size()] + "AAAAAAAAAAAA" + rand_seq(63)));
    return v;
}

std::vector<ScreenSpot> junk(size_t n) {
    std::vector<ScreenSpot> v;
    for (size_t i = 0; i < n; ++i) v.push_back(spot(rand_seq(28), rand_seq(90)));
    return v;
}

void write_wl(const std::string& path, const std::vector<std::string>& v) {
    std::ofstream f(path);
    for (const auto& s : v) f << s << "\n";
}

RunScreenReport rep(const std::string& id, std::vector<ScreenSpot> s, const std::vector<ScreenWhitelist>& wls,
                    uint64_t total = 0) {
    auto r = screen_spots(std::move(s), wls);
    r.run_id = id;
    r.total_reads = total;
    return r;
}

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const std::string dir = (fs::temp_directory_path() / ("test_run_screen_" + std::to_string(rng()))).generic_string();
    fs::create_directories(dir);
    for (int i = 0; i < 3000; ++i) wl_v3.push_back(rand_seq(16));
    for (int i = 0; i < 3000; ++i) wl_v2.push_back(rand_seq(16));
    for (int i = 0; i < 12; ++i) hto_tags.push_back(rand_seq(15));
    write_wl(dir + "/3M-february-2018.txt", wl_v3);
    write_wl(dir + "/737K-august-2016.txt", wl_v2);

    auto wls = screen_whitelist_specs();
    for (auto& wl : wls) load_screen_whitelist(wl, {dir});
    CHECK(wls[0].loaded() && wls[1].loaded(), "synthetic v3/v2 whitelists load");
    CHECK(!wls[2].loaded() && !wls[3].loaded(), "absent whitelists are reported as not loaded");

    std::cout << "\n=== single-run reports ===\n";
    auto g1 = rep("SRR_gex_a", gex3(5000, wl_v3, 12, 0.1), wls);
    CHECK(g1.chemistry == "10x-3p-v3", "3' v3 GEX -> 10x-3p-v3");
    CHECK(g1.best_whitelist == "3M-february-2018.txt", "best whitelist 3M");
    CHECK(g1.best_rate > 0.98, "exact + 1-mismatch rate counts the mutated barcodes");
    CHECK(g1.whitelists[0].exact < 0.95 && g1.whitelists[0].exact > 0.85, "exact rate misses the 10% mutated barcodes");
    CHECK(!g1.feature_barcode_like, "GEX R2 is not feature-barcode-like");
    CHECK(!g1.is_5prime, "3' run is not 5'");
    CHECK(g1.r1_len_mode == 28 && g1.r2_len_mode == 90, "length modes");

    auto f5 = rep("SRR_5p", gex5(3000), wls);
    CHECK(f5.is_5prime && f5.five_prime.r1_tso > 2900, "TSO tail in R1 -> 5'");
    CHECK(f5.chemistry == "10x-5p-v3", "5' v3 run -> 10x-5p-v3");

    auto h = rep("SRR_hto", hto(2000), wls);
    CHECK(h.feature_barcode_like, "HTO run is feature-barcode-like");
    CHECK(h.r2_top100_share_off0 > 0.99, "HTO: 12 tags hold all R2 prefixes");
    CHECK(h.r2_tag_polya_share > 0.9, "HTO: tag followed by poly(A)");

    auto v2 = rep("SRR_v2", gex3(1500, wl_v2, 10), wls);
    CHECK(v2.chemistry == "10x-3p-v2", "737K barcodes + 26-bp R1 -> 10x-3p-v2");

    auto j = rep("SRR_junk", junk(1500), wls);
    CHECK(j.best_rate < 0.05, "random R1 matches no whitelist");

    std::cout << "\n=== decide_runs: GEX majority, mixed runs ===\n";
    auto g2 = rep("SRR_gex_b", gex3(3000, wl_v3, 12), wls);
    std::vector<RunScreenReport> sample1 = {g1, h, g2, v2, j};
    auto d1 = decide_runs(sample1);
    CHECK(d1.majority_chemistry == "10x-3p-v3", "majority chemistry 10x-3p-v3");
    CHECK(d1.whitelist_rule, "10x majority: whitelist rule applies");
    CHECK(d1.majority_basis == "sampled_reads", "no total reads known -> sampled reads");
    CHECK((d1.kept() == std::vector<std::string>{"SRR_gex_a", "SRR_gex_b"}), "keeps the two 3' v3 GEX runs");
    CHECK(!d1.runs[1].keep && d1.runs[1].reason.find("feature-barcode") != std::string::npos, "drops HTO run");
    CHECK(!d1.runs[3].keep && d1.runs[3].reason.find("differs") != std::string::npos, "drops the v2 run (other chemistry)");
    CHECK(!d1.runs[4].keep && d1.runs[4].reason.find("whitelist") != std::string::npos, "drops the run below 30%");

    std::cout << "\n=== decide_runs: 5' majority by total reads ===\n";
    auto a = rep("SRR_5p_big", gex5(1200), wls, 90000000);
    auto b = rep("SRR_3p_small", gex3(3000, wl_v3, 12), wls, 20000000);
    auto d2 = decide_runs({b, a});
    CHECK(d2.majority_basis == "total_reads", "total reads known -> majority by total reads");
    CHECK(d2.majority_chemistry == "10x-5p-v3", "5' run is the majority by reads although it has fewer sampled");
    CHECK((d2.kept() == std::vector<std::string>{"SRR_5p_big"}), "3' run of a 5' sample is dropped");

    std::cout << "\n=== decide_runs: tie and tiny run ===\n";
    auto t1 = rep("SRR_t1", gex3(2000, wl_v3, 12), wls, 100);
    auto t2 = rep("SRR_t2", gex3(2000, wl_v2, 10), wls, 100);
    auto tiny = rep("SRR_tiny", gex3(50, wl_v2, 10), wls, 50);
    auto d3 = decide_runs({t2, t1, tiny});
    CHECK(d3.majority_chemistry == "10x-3p-v2", "tie in reads and runs -> smallest chemistry name (deterministic)");
    CHECK(d3.runs[2].keep && d3.runs[2].reason.find("unscreened") != std::string::npos, "too few reads: kept unscreened");
    auto d3b = decide_runs({t1, t2, tiny});
    CHECK(d3b.majority_chemistry == d3.majority_chemistry, "decision independent of run order");

    std::cout << "\n=== FASTQ input + JSON ===\n";
    {
        std::ofstream r1(dir + "/S1_R1.fastq"), r2(dir + "/S1_R2.fastq");
        for (int i = 0; i < 2000; ++i) {
            std::string s1 = wl_v3[rng() % wl_v3.size()] + rand_seq(12), s2 = rand_seq(90);
            r1 << "@r" << i << "\n" << s1 << "\n+\n" << std::string(s1.size(), 'F') << "\n";
            r2 << "@r" << i << "\n" << s2 << "\n+\n" << std::string(s2.size(), 'F') << "\n";
        }
    }
    auto fq = screen_run(dir + "/S1_R1.fastq," + dir + "/S1_R2.fastq", wls, {dir}, 1500);
    CHECK(fq.error.empty(), "FASTQ pair reads");
    CHECK(fq.reads_sampled == 1500, "--reads caps the sample");
    CHECK(fq.run_id == "S1", "run id from the R1 file name");
    CHECK(fq.chemistry == "10x-3p-v3", "FASTQ run chemistry");
    auto missing = screen_run(dir + "/nope_R1.fastq," + dir + "/nope_R2.fastq", wls, {dir}, 1500);
    CHECK(!missing.error.empty(), "missing file -> error");
    auto d4 = decide_runs({fq, missing});
    CHECK(d4.runs[0].keep && !d4.runs[1].keep, "unreadable run dropped");
    std::string js = run_screen_json({fq, missing}, d4, 1500);
    CHECK(js.find("\"runs_kept\": [\"S1\"]") != std::string::npos, "JSON lists runs_kept");
    CHECK(js.find("\"runs_dropped\": [\"nope\"]") != std::string::npos, "JSON lists runs_dropped");
    CHECK(js.find("\"vdj_assessable\": false") != std::string::npos, "JSON says V(D)J is not assessable");

    fs::remove_all(dir);
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
