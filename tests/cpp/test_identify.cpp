// SPDX-License-Identifier: MIT
// tests/cpp/test_identify.cpp
//
// `singlet identify` classifier (include/singlet/fq/identify.h) on synthetic spots, one decisive signal per family: 10x 3' and 5'
// (barcode list, TSO tail), v2 inside v3 (nested lists), Visium (spot list), scATAC (barcode segment, reverse complement), HTO
// (feature-barcode constants), well-barcode plates (CEL-Seq2, SCRB-seq), Smart-seq3 (UMI tag), full-length RNA without a barcode
// (run size decides Smart-seq against bulk, both refuse or defer, never process), not-sequence reads. Also the policy table and the
// mapping evidence parser.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "singlet/fq/identify.h"
#include "test_harness.h"

using namespace singlet::fq;
using namespace singlet::fq::identify;

namespace {

std::mt19937 rng(20261112);

std::string rand_seq(size_t n) {
    static const char b[4] = {'A', 'C', 'G', 'T'};
    std::string s(n, 'A');
    for (auto& c : s) c = b[rng() & 3];
    return s;
}

void write_list(const std::string& path, const std::vector<std::string>& v) {
    std::ofstream f(path);
    for (const auto& s : v) f << s << "\n";
}

std::vector<std::string> make_list(size_t n, size_t len = 16) {
    std::vector<std::string> v;
    for (size_t i = 0; i < n; ++i) v.push_back(rand_seq(len));
    return v;
}

HeadSpot spot(std::vector<std::string> segs) {
    HeadSpot s;
    s.seg = std::move(segs);
    return s;
}

std::string pick(const std::vector<std::string>& v) { return v[rng() % v.size()]; }

Report run(const std::vector<HeadSpot>& spots, std::vector<ScreenWhitelist>& lists, const std::string& dir, int64_t run_spots = -1,
           const MapEvidence* map = nullptr) {
    Options o;
    o.whitelist_dirs = {dir};
    o.run_spots = run_spots;
    return identify_spots(spots, o, lists, map);
}

std::vector<std::string> wl_v3, wl_v2, wl_vis, wl_atac;

}  // namespace

int main() {
    namespace fs = std::filesystem;
    const std::string dir = (fs::temp_directory_path() / ("test_identify_" + std::to_string(rng()))).generic_string();
    fs::create_directories(dir);
    wl_v3 = make_list(6000);
    wl_v2.assign(wl_v3.begin(), wl_v3.begin() + 1200);   // the 737K list sits inside the 3M list
    wl_vis = make_list(1500);
    wl_atac = make_list(6000);
    write_list(dir + "/3M-february-2018.txt", wl_v3);
    write_list(dir + "/737K-august-2016.txt", wl_v2);
    write_list(dir + "/visium-v1.txt", wl_vis);
    write_list(dir + "/737K-cratac-v1.txt", wl_atac);
    auto lists = identify_lists();
    const size_t N = 8000;

    std::cout << "\n=== 10x droplet ===\n";
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12), rand_seq(90)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "droplet_3p" && r.verdict_protocol == "10x-3p-v3", "3' v3: family and tag");
        CHECK(r.action == "process", "3' v3 is processed");
        CHECK(r.ev.bc_seg == 0 && r.ev.cdna_seg == 1, "roles: barcode in segment 0, cDNA in segment 1");
        CHECK(!r.ranked.empty() && r.ranked[0].usefulness > 0.9 && !r.ranked[0].unmeasured.empty(), "usefulness reported with its unmeasured components");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12) + "TTTCTTATATGGG" + rand_seq(9), rand_seq(90)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "droplet_5p" && r.verdict_protocol == "10x-5p-v3", "5' v3 from the TSO tail");
    }
    {   // the same 5' reads with strand evidence that says sense: the strand decides the 3'/5' question when it is clear
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12) + "TTTCTTATATGGG" + rand_seq(9), rand_seq(90)}));
        MapEvidence m; m.measured = true; m.unique = 0.9; m.exonic = 0.5; m.exonic_measured = true; m.sense = 0.05;
        auto r = run(v, lists, dir, -1, &m);
        CHECK(r.verdict_protocol == "10x-5p-v3", "antisense reads keep 5'");
        MapEvidence m2 = m; m2.sense = 0.95;
        std::vector<HeadSpot> w;
        for (size_t i = 0; i < N; ++i) w.push_back(spot({pick(wl_v3) + rand_seq(12), rand_seq(90)}));
        auto r2 = run(w, lists, dir, -1, &m2);
        CHECK(r2.verdict_protocol == "10x-3p-v3", "sense reads give 3'");
        MapEvidence m3 = m; m3.sense = 0.97;   // motifs say 5', strand says sense: the strand wins
        auto r3 = run(v, lists, dir, -1, &m3);
        CHECK(r3.verdict_protocol == "10x-3p-v3", "strand evidence overrules the motifs");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v2) + rand_seq(10), rand_seq(98)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict_protocol == "10x-3p-v2", "nested lists: a library on the 737K list is v2");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_vis) + rand_seq(12), rand_seq(90)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "visium" && r.action == "process", "Visium spot list -> visium, processed");
    }
    {   // a read-1-only head: no cDNA read, barcode list still calls the chemistry
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12)}));
        auto r = run(v, lists, dir);
        CHECK(r.ranked[0].family.rfind("droplet", 0) == 0 || r.verdict == "ambiguous" || r.verdict == "droplet_3p", "single barcode read does not crash");
    }

    std::cout << "\n=== ATAC and feature barcodes ===\n";
    {
        std::vector<HeadSpot> v, w;
        for (size_t i = 0; i < N; ++i) {
            const std::string bc = pick(wl_atac);
            v.push_back(spot({rand_seq(50), bc, rand_seq(50)}));
            std::string rc(bc.rbegin(), bc.rend());
            for (auto& c : rc) c = singlet::fq::identify::detail_id::comp(c);
            w.push_back(spot({rand_seq(50), rand_seq(50), rc}));
        }
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "atac" && r.action == "classify_only", "ATAC barcode segment -> atac, classify only");
        auto r2 = run(w, lists, dir);
        CHECK(r2.verdict == "atac" && r2.action == "classify_only", "reverse-complement ATAC barcode segment");
    }
    {
        std::vector<std::string> tags = make_list(12, 15);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12), pick(tags) + "AAAAAAAAAAAA" + rand_seq(63)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "feature_barcode" && r.action == "classify_only", "hashtag R2 -> feature_barcode, classify only");
    }

    std::cout << "\n=== plates, Smart-seq3, no barcode ===\n";
    {
        const auto wells = make_list(96, 6);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(wells) + rand_seq(10), rand_seq(70)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "plate_barcoded" && r.verdict_protocol == "scrbseq", "96 well barcodes in a 16-base read -> SCRB-seq layout");
        CHECK(r.action == "defer", "plate protocols are deferred");
    }
    {
        const auto wells = make_list(300, 8);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({rand_seq(6) + pick(wells) + std::string(10, 'T') + rand_seq(8), rand_seq(70)}));
        auto r = run(v, lists, dir);
        std::cout << "  [celseq2 case] verdict " << r.verdict << " protocol " << r.verdict_protocol << " window off " << r.ev.window.off << " len " << r.ev.window.len << " k " << r.ev.window.k << " found " << r.ev.window.found << " bc_seg " << r.ev.bc_seg << " detector " << r.ev.detected.tag << std::endl;
        CHECK(r.verdict == "plate_barcoded" && r.verdict_protocol == "celseq2-u6b8", "UMI6 + 8-base well barcode -> CEL-Seq2 layout");
    }
    {   // Drop-seq style: a 12-base barcode from thousands of beads is more than a plate holds; the barcode read repeats at its start
        const auto beads = make_list(2500, 12);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < 40000; ++i) v.push_back(spot({pick(beads) + rand_seq(8), rand_seq(60)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "droplet_other", "large barcode set (12 bases + 8 base UMI) -> droplet_other");
        std::vector<HeadSpot> w;
        for (size_t i = 0; i < 40000; ++i) w.push_back(spot({pick(beads) + rand_seq(10) + rand_seq(33), rand_seq(60)}));
        auto r2 = run(w, lists, dir);
        CHECK(r2.verdict == "droplet_other", "large barcode set in a long first read -> droplet_other");
        CHECK(r2.verdict_protocol != "barcoded-unspecified" || r2.action == "defer", "an unspecified layout is deferred, not processed");
    }
    {   // split-pool beads: two disjoint 96-barcode windows separated by a linker
        const auto b1 = make_list(96, 6), b2 = make_list(96, 6);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({pick(b1) + "ACG" + pick(b2) + "GACTCA" + rand_seq(8), rand_seq(70)}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "droplet_other", "two barcode windows -> split-pool droplet beads, not a plate");
    }
    {   // plate-based Smart-seq: 96 well indices in index reads, two long insert reads without a barcode
        const auto idx = make_list(96, 8);
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) { const std::string w = pick(idx); v.push_back(spot({rand_seq(76), rand_seq(76), w, w})); }
        auto r = run(v, lists, dir, 500000);
        CHECK(r.verdict == "smartseq" && r.action == "defer", "well indices in the index reads -> smartseq, deferred");
    }
    {   // a long single read with a repeated start is not a barcoded plate
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({"GATCGGAAGAGC" + rand_seq(70)}));
        auto r = run(v, lists, dir, 20000000);
        CHECK(r.verdict != "plate_barcoded" && r.action != "process", "long single read: no plate call, never processed");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) {
            const bool tag = (i % 3) != 0;
            v.push_back(spot({(tag ? std::string("ATTGCGCAATG") + rand_seq(8) + rand_seq(80) : rand_seq(99)), rand_seq(100)}));
        }
        auto r = run(v, lists, dir, 400000);
        CHECK(r.verdict == "smartseq3" && r.action == "defer", "Smart-seq3 tag -> smartseq3, deferred");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({rand_seq(100), rand_seq(100)}));
        auto big = run(v, lists, dir, 60000000);
        CHECK(big.verdict == "bulk_rna" && big.action == "refuse", "large run, no barcode -> bulk, refused");
        auto small = run(v, lists, dir, 2000000);
        CHECK(small.verdict == "smartseq" && small.action == "defer", "plate-sized run, no barcode -> smartseq, deferred");
        auto unk = run(v, lists, dir, -1);
        CHECK(unk.verdict == "ambiguous" && unk.action == "refuse", "run size unknown -> ambiguous, refused");
        CHECK(unk.ambiguous_with.find("smartseq") != std::string::npos && unk.ambiguous_with.find("bulk_rna") != std::string::npos, "ambiguity names both");
        Options o; o.whitelist_dirs = {dir}; o.hint = "assay=smartseq2";
        auto hinted = identify_spots(v, o, lists);
        CHECK(hinted.verdict == "ambiguous" && hinted.action == "defer" && hinted.hint == "assay=smartseq2", "a hint selects the action and leaves the verdict");
        std::vector<HeadSpot> se;
        for (size_t i = 0; i < N; ++i) se.push_back(spot({rand_seq(75)}));
        auto r = run(se, lists, dir, 30000000);
        CHECK(r.verdict == "bulk_rna" && r.action == "refuse", "single-end long reads, no barcode, large run -> bulk refused");
    }
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < 3000; ++i) v.push_back(spot({std::string(100, 'N'), std::string(100, 'N')}));
        auto r = run(v, lists, dir);
        CHECK(r.verdict == "not_sequence" && r.action == "refuse", "all-N reads -> not_sequence");
    }

    std::cout << "\n=== mapping evidence ===\n";
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < N; ++i) v.push_back(spot({rand_seq(100), rand_seq(100)}));
        MapEvidence g; g.measured = true; g.unique = 0.85; g.exonic = 0.02; g.exonic_measured = true; g.sense = 0.5;
        auto r = run(v, lists, dir, 2000000, &g);
        CHECK(r.verdict == "genomic_dna" && r.action == "refuse", "reads that map outside genes -> genomic DNA, refused");
        MapEvidence rna; rna.measured = true; rna.unique = 0.9; rna.exonic = 0.6; rna.exonic_measured = true; rna.sense = 0.5;
        auto r2 = run(v, lists, dir, 2000000, &rna);
        CHECK(r2.verdict == "smartseq", "reads that map to genes keep the RNA calls");
        const std::string p = dir + "/star.";
        { std::ofstream l(p + "Log.final.out"); l << "                          Number of input reads |\t1000\n                   Uniquely mapped reads number |\t900\n"
                                                     "        Number of reads mapped to multiple loci |\t50\n"; }
        { std::ofstream l(p + "ReadsPerGene.out.tab"); l << "N_unmapped\t100\t100\t100\nN_noFeature\t10\t300\t10\nG1\t800\t780\t10\nG2\t50\t45\t5\n"; }
        auto m = parse_star_outputs(p, 1000);
        CHECK(m.measured && std::abs(m.unique - 0.9) < 1e-9 && std::abs(m.multi - 0.05) < 1e-9, "STAR log parsed");
        CHECK(m.exonic_measured && std::abs(m.sense - 825.0 / 840.0) < 1e-9, "sense fraction from the strand columns");
    }

    std::cout << "\n=== policy and JSON ===\n";
    for (const char* f : {"atac", "multiome_atac", "feature_barcode", "vdj", "bulk_rna", "genomic_dna", "methylome", "not_sequence", "unknown", "ambiguous", "rna_no_barcode"})
        CHECK(std::string(policy_for(f).action) != "process", std::string("family never processed: ") + f);
    CHECK(std::string(policy_for("droplet_3p").action) == "process" && std::string(policy_for("visium").action) == "process", "10x and Visium are processed");
    CHECK(std::string(policy_for("not a family").action) == "refuse", "an unlisted family is refused");
    {
        std::vector<HeadSpot> v;
        for (size_t i = 0; i < 2000; ++i) v.push_back(spot({pick(wl_v3) + rand_seq(12), rand_seq(90)}));
        auto r = run(v, lists, dir);
        const std::string js = report_json(r);
        CHECK(js.find("\"schema_version\": 1") != std::string::npos, "JSON carries the schema version");
        CHECK(js.find("\"interpretations\"") != std::string::npos && js.find("\"components\"") != std::string::npos, "JSON lists interpretations with components");
    }

    fs::remove_all(dir);
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
